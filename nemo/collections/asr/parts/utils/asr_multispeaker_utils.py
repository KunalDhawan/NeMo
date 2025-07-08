# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import math
import json
import random
import logging
from copy import deepcopy
from cytoolz import groupby
import time
from collections import defaultdict

import numpy as np
import soundfile
from tqdm import tqdm
from scipy.stats import norm

import torch.utils.data
from lhotse.cut.set import mix
from lhotse.cut import Cut, CutSet, MixedCut, MonoCut, MixTrack
from lhotse import SupervisionSet, SupervisionSegment, dill_enabled, AudioSource, Recording
from lhotse.utils import uuid4, compute_num_samples, ifnone
from lhotse.lazy import LazyIteratorChain, LazyJsonlIterator
from nemo.collections.asr.data.data_simulation import MultiSpeakerSimulator
from nemo.collections.asr.parts.utils.data_simulation_utils import read_rir_manifest
from typing import Optional, Union, List, Tuple, Dict, Any

from omegaconf import OmegaConf
from dataclasses import dataclass, field
from typing import List, Optional

import soundfile as sf
import os
@dataclass
class SessionConfig:
    num_speakers: int = 1
    num_sessions: int = 1
    session_length: int = 15

@dataclass
class SessionParams:
    max_audio_read_sec: float = 20.0
    sentence_length_params: List[float] = field(default_factory=lambda: [0.4, 0.05])
    dominance_var: float = 0.11
    min_dominance: float = 0.05
    turn_prob: float = 0.875
    min_turn_prob: float = 0.5
    mean_silence: float = 0.15
    mean_silence_var: float = 0.01
    per_silence_var: int = 900
    per_silence_min: float = 0.0
    per_silence_max: float = -1.0
    mean_overlap: float = 0.1
    mean_overlap_var: float = 0.01
    per_overlap_var: int = 900
    per_overlap_min: float = 0.0
    per_overlap_max: float = -1.0
    start_window: bool = True
    window_type: str = "hamming"
    window_size: float = 0.02
    start_buffer: float = 0.0
    split_buffer: float = 0.01
    release_buffer: float = 0.0
    normalize: bool = True
    normalization_type: str = "equal"
    normalization_var: float = 0.1
    min_volume: float = 0.75
    max_volume: float = 1.25
    end_buffer: float = 0.5
    random_offset: bool = True

@dataclass
class OutputConfig:
    output_dir: str = ""
    output_filename: str = "multispeaker_session"
    overwrite_output: bool = True
    output_precision: int = 3

@dataclass
class BackgroundNoise:
    add_bg: bool = True
    background_manifest: Optional[str] = None
    rir_manifest: Optional[str] = None
    num_noise_files: int = 10
    snr: int = 60
    snr_min: Optional[float] = None
    snr_max: Optional[float] = None

@dataclass
class SegmentAugmentor:
    add_seg_aug: bool = False
    gain_prob: float = 0.5
    min_gain_dbfs: float = -10.0
    max_gain_dbfs: float = 10.0

@dataclass
class SessionAugmentor:
    add_sess_aug: bool = False
    white_noise_prob: float = 1.0
    min_white_noise_level: int = -90
    max_white_noise_level: int = -46

@dataclass
class SpeakerEnforcement:
    enforce_num_speakers: bool = True
    enforce_time: List[float] = field(default_factory=lambda: [0.25, 0.75])

@dataclass
class SegmentManifest:
    window: float = 0.5
    shift: float = 0.25
    step_count: int = 50
    deci: int = 3

@dataclass
class RIRGeneration:
    use_rir: bool = False
    toolkit: str = "pyroomacoustics"
    room_sz: List[List[int]] = field(default_factory=lambda: [[2, 3], [2, 3], [2, 3]])
    pos_src: List[List[List[float]]] = field(default_factory=lambda: [[[0.5, 1.5]] * 3] * 4)
    noise_src_pos: List[float] = field(default_factory=lambda: [1.5, 1.5, 2])
    num_channels: int = 2
    pos_rcv: List[List[List[float]]] = field(default_factory=lambda: [[[0.5, 1.5]] * 3] * 2)
    orV_rcv: Optional[List[List[float]]] = None
    mic_pattern: str = "omni"
    abs_weights: List[float] = field(default_factory=lambda: [0.9] * 6)
    T60: float = 0.1
    att_diff: float = 15.0
    att_max: float = 60.0

@dataclass
class DataSimConfig:
    """Configuration for data simulation."""
    manifest_filepath: str = ""
    sr: int = 16000
    random_seed: int = 42
    multiprocessing_chunksize: int = 10000
    session_config: SessionConfig = field(default_factory=SessionConfig)
    session_params: SessionParams = field(default_factory=SessionParams)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    background_noise: BackgroundNoise = field(default_factory=BackgroundNoise)
    segment_augmentor: SegmentAugmentor = field(default_factory=SegmentAugmentor)
    session_augmentor: SessionAugmentor = field(default_factory=SessionAugmentor)
    speaker_enforcement: SpeakerEnforcement = field(default_factory=SpeakerEnforcement)
    segment_manifest: SegmentManifest = field(default_factory=SegmentManifest)
    rir_generation: RIRGeneration = field(default_factory=RIRGeneration)

@dataclass
class MultiSpeakerSimulatorConfig:
    data_simulator: DataSimConfig = field(default_factory=DataSimConfig)

class Segment:
    def __init__(self, start, end, speaker_id, text):
        self.start = start
        self.end = end
        self.speaker_id = speaker_id
        self.text = text

class SegList:
    def __init__(self, segments: List[Segment] = None, seglst_filepath: str = None):
        if segments is not None:
            self.segments = segments
        elif seglst_filepath is not None:
            self._load_seglst(seglst_filepath)
        else:
            raise ValueError("Either segments or seglst_filepath must be provided")
    
    def _load_seglst(self, seglst_filepath: str|list[str]):
        if isinstance(seglst_filepath, str):
            with open(seglst_filepath, 'r', encoding='utf-8') as f:
                seglst = json.load(f)
                self.segments = [
                    Segment(seg['start_time'], seg['end_time'], seg['speaker'], seg['words']) for seg in seglst
                ]
        elif isinstance(seglst_filepath, list):
            for seglst_file in seglst_filepath:
                with open(seglst_file, 'r', encoding='utf-8') as f:
                    seglst = json.load(f)
                    segments = [
                        Segment(seg['start_time'], seg['end_time'], seg['speaker'], seg['words']) for seg in seglst
                    ]
                self.segments.extend(segments)
        else:
            raise ValueError("seglst_filepath must be a string or a list of strings")
        self.sort()
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        return self.segments[idx]
    
    def __iter__(self):
        return iter(self.segments)
    
    def sort(self):
        self.segments.sort(key=lambda x: x.start)

    def get_segments(self, min_duration: float, max_duration: float):
        
        duration = random.uniform(min_duration, max_duration)

        first_segment_idx = random.randint(0, len(self) - 1)
        segments = [self[first_segment_idx]]

        offset = self[first_segment_idx].start
        for i in range(first_segment_idx + 1, len(self)):
            if self[i].end - offset <= duration:
                segments.append(self[i])
            else:
                break
        
        return segments
    
    def get_text_from_segments(self, segments: list[Segment], speaker_token_style='<|spltoken*|>', speaker_token_position='sot'):
        text = ''
        speakers = set([segment.speaker_id for segment in segments])
        speaker2start = {spk_id: min(segment.start for segment in segments if segment.speaker_id == spk_id) for spk_id in speakers}
        sorted_speakers = sorted(speakers, key=lambda x: speaker2start[x])
        speaker2token = {spk: speaker_token_style.replace('*', str(i)) for i, spk in enumerate(sorted_speakers)}
        for segment in segments:
            text += f'{speaker2token[segment.speaker_id]} '
            text += segment.text
        return text.strip()

def find_first_nonzero(mat: torch.Tensor, max_cap_val=-1, thres: float = 0.5) -> torch.Tensor:
    """
    Finds the first nonzero value in the matrix, discretizing it to the specified maximum capacity.

    Args:
        mat (Tensor): A torch tensor representing the matrix.
        max_cap_val (int): The maximum capacity to which the matrix values will be discretized.
        thres (float): The threshold value for discretizing the matrix values.

    Returns:
        mask_max_indices (Tensor): A torch tensor representing the discretized matrix with the first
        nonzero value in each row.
    """
    # Discretize the matrix to the specified maximum capacity
    labels_discrete = mat.clone()
    labels_discrete[labels_discrete < thres] = 0
    labels_discrete[labels_discrete >= thres] = 1

    # non zero values mask
    non_zero_mask = labels_discrete != 0
    # operations on the mask to find first nonzero values in the rows
    mask_max_values, mask_max_indices = torch.max(non_zero_mask, dim=1)
    # if the max-mask is zero, there is no nonzero value in the row
    mask_max_indices[mask_max_values == 0] = max_cap_val
    return mask_max_indices


def find_best_permutation(match_score: torch.Tensor, speaker_permutations: torch.Tensor) -> torch.Tensor:
    """
    Finds the best permutation indices based on the match score.

    Args:
        match_score (torch.Tensor): A tensor containing the match scores for each permutation.
            Shape: (batch_size, num_permutations)
        speaker_permutations (torch.Tensor): A tensor containing all possible speaker permutations.
            Shape: (num_permutations, num_speakers)

    Returns:
        torch.Tensor: A tensor containing the best permutation indices for each batch.
            Shape: (batch_size, num_speakers)
    """
    batch_best_perm = torch.argmax(match_score, axis=1)
    rep_speaker_permutations = speaker_permutations.repeat(batch_best_perm.shape[0], 1).to(match_score.device)
    perm_size = speaker_permutations.shape[0]
    global_inds_vec = (
        torch.arange(0, perm_size * batch_best_perm.shape[0], perm_size).to(batch_best_perm.device) + batch_best_perm
    )
    return rep_speaker_permutations[global_inds_vec.to(rep_speaker_permutations.device), :]


def reconstruct_labels(labels: torch.Tensor, batch_perm_inds: torch.Tensor) -> torch.Tensor:
    """
    Reconstructs the labels using the best permutation indices with matrix operations.

    Args:
        labels (torch.Tensor): A tensor containing the original labels.
            Shape: (batch_size, num_frames, num_speakers)
        batch_perm_inds (torch.Tensor): A tensor containing the best permutation indices for each batch.
            Shape: (batch_size, num_speakers)

    Returns:
        torch.Tensor: A tensor containing the reconstructed labels using the best permutation indices.
            Shape: (batch_size, num_frames, num_speakers)
    """
    # Expanding batch_perm_inds to align with labels dimensions
    batch_size, num_frames, num_speakers = labels.shape
    batch_perm_inds_exp = batch_perm_inds.unsqueeze(1).expand(-1, num_frames, -1)

    # Reconstructing the labels using advanced indexing
    reconstructed_labels = torch.gather(labels, 2, batch_perm_inds_exp)
    return reconstructed_labels


def get_ats_targets(
    labels: torch.Tensor,
    preds: torch.Tensor,
    speaker_permutations: torch.Tensor,
    thres: float = 0.5,
    tolerance: float = 0,
) -> torch.Tensor:
    """
    Sorts labels and predictions to get the optimal of all arrival-time ordered permutations.

    Args:
        labels (torch.Tensor): A tensor containing the original labels.
            Shape: (batch_size, num_frames, num_speakers)
        preds (torch.Tensor): A tensor containing the predictions.
            Shape: (batch_size, num_frames, num_speakers)
        speaker_permutations (torch.Tensor): A tensor containing all possible speaker permutations.
            Shape: (num_permutations, num_speakers)
        thres (float): The threshold value for discretizing the matrix values. Default is 0.5.
        tolerance (float): The tolerance for comparing the first speech frame indices. Default is 0.

    Returns:
        torch.Tensor: A tensor containing the reconstructed labels using the best permutation indices.
            Shape: (batch_size, num_frames, num_speakers)
    """
    # Find the first nonzero frame index for each speaker in each batch
    nonzero_ind = find_first_nonzero(
        mat=labels, max_cap_val=labels.shape[1], thres=thres
    )  # (batch_size, num_speakers)

    # Sort the first nonzero frame indices for arrival-time ordering
    sorted_values = torch.sort(nonzero_ind)[0]  # (batch_size, num_speakers)
    perm_size = speaker_permutations.shape[0]  # Scalar value (num_permutations)
    permed_labels = labels[:, :, speaker_permutations]  # (batch_size, num_frames, num_permutations, num_speakers)
    permed_nonzero_ind = find_first_nonzero(
        mat=permed_labels, max_cap_val=labels.shape[1]
    )  # (batch_size, num_permutations, num_speakers)

    # Compare the first frame indices of sorted labels with those of the permuted labels using tolerance
    perm_compare = (
        torch.abs(sorted_values.unsqueeze(1) - permed_nonzero_ind) <= tolerance
    )  # (batch_size, num_permutations, num_speakers)
    perm_mask = torch.all(perm_compare, dim=2).float()  # (batch_size, num_permutations)
    preds_rep = torch.unsqueeze(preds, 2).repeat(
        1, 1, perm_size, 1
    )  # Exapnd the preds: (batch_size, num_frames, num_permutations, num_speakers)

    # Compute the match score for each permutation by comparing permuted labels with preds
    match_score = (
        torch.sum(permed_labels * preds_rep, axis=1).sum(axis=2) * perm_mask
    )  # (batch_size, num_permutations)
    batch_perm_inds = find_best_permutation(match_score, speaker_permutations)  # (batch_size, num_speakers)
    max_score_permed_labels = reconstruct_labels(labels, batch_perm_inds)  # (batch_size, num_frames, num_speakers)
    return max_score_permed_labels  # (batch_size, num_frames, num_speakers)


def get_pil_targets(labels: torch.Tensor, preds: torch.Tensor, speaker_permutations: torch.Tensor) -> torch.Tensor:
    """
    Sorts labels and predictions to get the optimal permutation based on the match score.

    Args:
        labels (torch.Tensor): A tensor containing the ground truth labels.
            Shape: (batch_size, num_speakers, num_classes)
        preds (torch.Tensor): A tensor containing the predicted values.
            Shape: (batch_size, num_speakers, num_classes)
        speaker_permutations (torch.Tensor): A tensor containing all possible speaker permutations.
            Shape: (num_permutations, num_speakers)

    Returns:
        torch.Tensor: A tensor of permuted labels that best match the predictions.
            Shape: (batch_size, num_speakers, num_classes)
    """
    permed_labels = labels[:, :, speaker_permutations]  # (batch_size, num_classes, num_permutations, num_speakers)
    # Repeat preds to match permutations for comparison
    preds_rep = torch.unsqueeze(preds, 2).repeat(
        1, 1, speaker_permutations.shape[0], 1
    )  # (batch_size, num_speakers, num_permutations, num_classes)
    match_score = torch.sum(permed_labels * preds_rep, axis=1).sum(axis=2)  # (batch_size, num_permutations)
    batch_perm_inds = find_best_permutation(match_score, speaker_permutations)  # (batch_size, num_speakers)
    # Reconstruct labels based on the best permutation for each batch
    max_score_permed_labels = reconstruct_labels(labels, batch_perm_inds)  # (batch_size, num_speakers, num_classes)
    return max_score_permed_labels  # (batch_size, num_speakers, num_classes)


def find_segments_from_rttm(
    recording_id: str,
    rttms: SupervisionSet,
    start_after: float,
    end_before: float,
    adjust_offset: bool = True,
    tolerance: float = 0.001,
):
    """
    Finds segments from the given rttm file.
    This function is designed to replace rttm

    Args:
        recording_id (str): The recording ID in string format.
        rttms (SupervisionSet): The SupervisionSet instance.
        start_after (float): The start time after which segments are selected.
        end_before (float): The end time before which segments are selected.
        adjust_offset (bool): Whether to adjust the offset of the segments.
        tolerance (float): The tolerance for time matching. 0.001 by default.
    Returns:
        segments (List[SupervisionSegment]): A list of SupervisionSegment instances.
    """
    segment_by_recording_id = rttms._segments_by_recording_id
    if segment_by_recording_id is None:
        from cytoolz import groupby

        segment_by_recording_id = groupby(lambda seg: seg.recording_id, rttms)

    return [
        # We only modify the offset - the duration remains the same, as we're only shifting the segment
        # relative to the Cut's start, and not truncating anything.
        segment.with_offset(-start_after) if adjust_offset else segment
        for segment in segment_by_recording_id.get(recording_id, [])
        if segment.start < end_before + tolerance and segment.end > start_after + tolerance
    ]


def get_mask_from_segments(
    segments: list,
    a_cut: Optional[Union[MonoCut, MixedCut]],
    speaker_to_idx_map: torch.Tensor,
    num_speakers: int = 4,
    feat_per_sec: int = 100,
    ignore_num_spk_mismatch: bool = False,
):
    """
    Generate mask matrix from segments list.
    This function is needed for speaker diarization with ASR model trainings.

    Args:
        segments: A list of Lhotse Supervision segments iterator.
        cut (MonoCut, MixedCut): Lhotse MonoCut or MixedCut instance.
        speaker_to_idx_map (dict): A dictionary mapping speaker names to indices.
        num_speakers (int): max number of speakers for all cuts ("mask" dim0), 4 by default
        feat_per_sec (int): number of frames per second, 100 by default, 0.01s frame rate
        ignore_num_spk_mismatch (bool): This is a temporary solution to handle speaker mismatch.
                                        Will be removed in the future.

    Returns:
        mask (Tensor): A numpy array of shape (num_speakers, encoder_hidden_len).
            Dimension: (num_speakers, num_frames)
    """
    # get targets with 0.01s frame rate
    num_samples = round(a_cut.duration * feat_per_sec)
    mask = torch.zeros((num_samples, num_speakers))
    for rttm_sup in segments:
        speaker_idx = speaker_to_idx_map[rttm_sup.speaker]
        if speaker_idx >= num_speakers:
            if ignore_num_spk_mismatch:
                continue
            else:
                raise ValueError(f"Speaker Index {speaker_idx} exceeds the max index: {num_speakers-1}")
        stt = max(rttm_sup.start, 0)
        ent = min(rttm_sup.end, a_cut.duration)
        stf = int(stt * feat_per_sec)
        enf = int(ent * feat_per_sec)
        mask[stf:enf, speaker_idx] = 1.0
    return mask


def get_soft_mask(feat_level_target, num_frames, stride):
    """
    Get soft mask from feat_level_target with stride.
    This function is needed for speaker diarization with ASR model trainings.

    Args:
        feat_level_target (Tensor): A numpy array of shape (num_frames, num_speakers).
            Dimension: (num_frames, num_speakers)
        num_sample (int): The total number of samples.
        stride (int): The stride for the mask.

    Returns:
        mask: The soft mask of shape (num_frames, num_speakers).
            Dimension: (num_frames, num_speakers)
    """

    num_speakers = feat_level_target.shape[1]
    mask = torch.zeros(num_frames, num_speakers)

    for index in range(num_frames):
        if index == 0:
            seg_stt_feat = 0
        else:
            seg_stt_feat = stride * index - 1 - int(stride / 2)
        if index == num_frames - 1:
            seg_end_feat = feat_level_target.shape[0]
        else:
            seg_end_feat = stride * index - 1 + int(stride / 2)
        mask[index] = torch.mean(feat_level_target[seg_stt_feat : seg_end_feat + 1, :], axis=0)
    return mask


def get_hidden_length_from_sample_length(
    num_samples: int, num_sample_per_mel_frame: int = 160, num_mel_frame_per_asr_frame: int = 8
) -> int:
    """
    Calculate the hidden length from the given number of samples.
    This function is needed for speaker diarization with ASR model trainings.

    This function computes the number of frames required for a given number of audio samples,
    considering the number of samples per mel frame and the number of mel frames per ASR frame.

    Parameters:
        num_samples (int): The total number of audio samples.
        num_sample_per_mel_frame (int, optional): The number of samples per mel frame. Default is 160.
        num_mel_frame_per_asr_frame (int, optional): The number of mel frames per ASR frame. Default is 8.

    Returns:
        hidden_length (int): The calculated hidden length in terms of the number of frames.
    """
    mel_frame_count = math.ceil((num_samples + 1) / num_sample_per_mel_frame)
    hidden_length = math.ceil(mel_frame_count / num_mel_frame_per_asr_frame)
    return int(hidden_length)

class ConcatenationMeetingSimulator():
    """
    This simulator concatenates the segments from different/same sessions to create a
    multi-speaker meeting. 
    """

    def __init__(
        self,
        intra_session_concat_prob: Union[float, List[float]] = [0, 1.0, 0.5, 0.2],
        data_type: str = "msasr",
        min_duration: float = 30.0,
        max_duration: float = 40.0,
        max_num_speakers: int = 4,
        speaker_count_distribution: List[float] = [0, 2, 3, 4],
        skip_long_segments: bool = True,
        valid_dataset_ids: List[str] = [],
    ):
        """
        :param intra_session_concat_prob: the probability of concatenating segments from the same
            session. [Default: 1]
        :param data_type: the type of data to simulate. Either 'msasr' or 'diar'. If 'msasr',
            the transcripts are included in the simulation,and the boundary segments are 
            not included. [Default: 'msasr']
        :param max_duration: the maximum duration of the simulated meeting. [Default: 40.0]
        """
        super().__init__()
        if isinstance(intra_session_concat_prob, float):
            self.intra_session_concat_prob = [intra_session_concat_prob] * (max_num_speakers)
        elif len(intra_session_concat_prob) == max_num_speakers:
            self.intra_session_concat_prob = intra_session_concat_prob
        else:
            raise ValueError(f"intra_session_concat_prob must be either a float or a list of floats, but got {intra_session_concat_prob}")
        if data_type not in ["msasr", "diar"]:
            raise ValueError("data_type must be either 'msasr' or 'diar', but got {data_type}")
        self.data_type = data_type
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_num_speakers = max_num_speakers
        self.speaker_count_distribution = speaker_count_distribution
        assert len(speaker_count_distribution) == max_num_speakers, f"Length of speaker_count_distribution {len(speaker_count_distribution)} must be equal to max_num_speakers {max_num_speakers}"

        if skip_long_segments:
            self.skip_duration = max_duration / 2
        else:
            self.skip_duration = max_duration

        self.valid_dataset_ids = valid_dataset_ids

    def fit(self, cuts) -> CutSet:
        """
        Read the manifest file and return a CutSet object. 
        Each line in the manifest file should be a JSON object representing a segment.
        """
        self.id2cut = {}
        self.sess2cut_ids = defaultdict(list)
        self.sess2spks = defaultdict(set)
        self.data2sess_ids = defaultdict(list)
        self.spk2cut_ids = defaultdict(list)
        self.data2num_spk2cut_ids = {}
        self.sess2num_spk2cut_ids = {}
        self.num_spk2cut_ids = {i+1:[] for i in range(self.max_num_speakers)}
        for i, cut in tqdm(enumerate(cuts), desc="Reading segments", ncols=100, total=len(cuts)):
            if cut.duration > self.skip_duration:
                continue
            if not hasattr(cut, 'dataset_id') or cut.dataset_id is None:
                continue
            if self.valid_dataset_ids and cut.dataset_id not in self.valid_dataset_ids:
                continue
            if cut.dataset_id not in self.data2num_spk2cut_ids:
                self.data2num_spk2cut_ids[cut.dataset_id] = defaultdict(list)
            if cut.recording_id not in self.sess2num_spk2cut_ids:
                self.sess2num_spk2cut_ids[cut.recording_id] = defaultdict(list)
            
            speakers = cut.global_speaker_ids
            if self.data_type == "msasr":
                speaker_tokens = set(re.findall(r'<\|spltoken\d+\|>', cut.text))
                if len(speakers) != len(speaker_tokens): 
                    # Lhotse automatically fixes the max duration of the cut, 
                    # resulting in the mismatch of the number of speakers 
                    # and speaker tokens for the last segment
                    # TODO: need to fix the issue in Lhotse that automatically fixes the max duration
                    continue
            for spk in speakers:
                self.spk2cut_ids[spk].append(cut.id)
            self.sess2spks[cut.recording_id] = self.sess2spks[cut.recording_id].union(speakers)
            
            self.id2cut[cut.id] = cut
            self.sess2cut_ids[cut.recording_id].append(cut.id)
            self.data2num_spk2cut_ids[cut.dataset_id][len(speakers)].append(cut.id)
            self.sess2num_spk2cut_ids[cut.recording_id][len(speakers)].append(cut.id)
            self.num_spk2cut_ids[len(speakers)].append(cut.id)
            if cut.recording_id not in self.data2sess_ids[cut.dataset_id]:
                self.data2sess_ids[cut.dataset_id].append(cut.recording_id)
                
        self.cut_ids = list(self.id2cut.keys())
        self.num_spk2sess_ids = groupby(lambda x: len(self.sess2spks[x]), self.sess2spks.keys())
        
        self.data2global_speaker = {
            dataset_id: True for dataset_id in self.data2sess_ids.keys()
        }        
            
    def _create_mixture(self, n_speakers: int, is_intra_session_concat=False) -> MixedCut:

        db_norm = norm.rvs(-32.05957708631966, 5.66648411405886) # mean and std from Fisher data
        
        if is_intra_session_concat:
            # intra-dataset and intra-session concatenation
            tracks, num_speakers = self.get_intra_session_tracks(n_speakers, db_norm=db_norm)

        else: 
            # intra-dataset but inter-session concatenation
            tracks, num_speakers = self.get_inter_session_tracks(n_speakers, db_norm=db_norm)

        cut = MixedCut(id='concat_' + '_'.join([track.cut.id for track in tracks]), tracks=tracks)
        if self.data_type == "msasr":
            cut = self.reorder_spk_mapping(cut)

        assert self.min_duration <= cut.duration <= self.max_duration, f"Total duration {cut.duration} is not within the range of min {self.min_duration} and max {self.max_duration}"
        assert n_speakers == num_speakers, f"Total number of speakers {cut.num_speakers} is not equal to the number of speakers {n_speakers}"

        return cut
    
    def get_intra_session_tracks(self, n_speakers: int=4, db_norm: float=-25) -> List[MixTrack]:
        """
        Get the tracks for the MixedCut object.
        """
        session_id = random.choice(self.num_spk2sess_ids[n_speakers])
        
        total_duration = 0.0
        total_spk_set = set()
        tracks = []
        while True:
            cut = self.id2cut[random.choice(self.sess2cut_ids[session_id])]
            tracks.append(MixTrack(cut=deepcopy(cut.normalize_loudness(target=db_norm, mix_first=False)), type=type(cut), offset=total_duration))
            total_spk_set = total_spk_set.union(cut.global_speaker_ids)
            total_duration += cut.duration

            # break condition
            if total_duration >= self.min_duration:
                if total_duration > self.max_duration: # exceed the maximum duration, starting over
                    total_duration = 0.0
                    total_spk_set = set()
                    tracks = []
                    session_id = random.choice(self.num_spk2sess_ids[n_speakers])
                if len(total_spk_set) == n_speakers: # meet the number of speakers and duration, break
                    break
                else:
                    total_duration = 0.0
                    total_spk_set = set()
                    tracks = []
                    session_id = random.choice(self.num_spk2sess_ids[n_speakers])
            
        return tracks, len(total_spk_set)

    def get_inter_session_tracks(self, n_speakers: int=4, db_norm: float=-25) -> List[MixTrack]:
        """
        Get the tracks for the MixedCut object.
        """
        sample_cut = self.id2cut[random.choice(self.cut_ids)]
        dataset_id = sample_cut.dataset_id
        n_spk_list = [n_spk for n_spk, cut_ids in self.data2num_spk2cut_ids[dataset_id].items() if len(cut_ids) > 0]
        sum_spk_list = set([i + j for i in n_spk_list for j in n_spk_list])

        if min(sum_spk_list) > n_speakers:
            raise ValueError(f"Cannot generate {n_speakers}-speaker inter session samples by concatenating two samples since the dataset {dataset_id} only have {','.join([str(i) for i in n_spk_list])} speakers.")

        n_spk_left = n_speakers
        total_duration = 0.0
        total_spk_set = set()
        tracks = []
        num_spk2cut_ids = self.data2num_spk2cut_ids[dataset_id]
        while True:
            #if n_spk_left == n_speakers: # for more speakers cases
            #    n_spk = random.choice([n_spk for n_spk in n_spk_list if n_spk < n_spk_left])
            if n_spk_left >= 2:
                n_spk = 2
            else:
                # n_spk = random.choice([n_spk for n_spk in n_spk_list if n_spk <= n_spk_left])
                n_spk = 1

            while True:
                cut = self.id2cut[random.choice(num_spk2cut_ids[n_spk])]
                spks = set(cut.global_speaker_ids)
                if not spks.intersection(total_spk_set):
                    break

            tracks.append(MixTrack(cut=deepcopy(cut.normalize_loudness(target=db_norm, mix_first=False)), type=type(cut), offset=total_duration))
            total_duration += cut.duration
            n_spk_left -= n_spk
            total_spk_set = total_spk_set.union(spks)

            # break condition
            
            if total_duration >= self.min_duration:
                if total_duration > self.max_duration or len(total_spk_set) < n_speakers: # exceed the maximum duration, starting over
                    total_duration = 0.0
                    n_spk_left = n_speakers
                    total_spk_set = set()
                    tracks = []
                if len(total_spk_set) == n_speakers: # meet the number of speakers and duration, break
                    break
            else:
                if len(total_spk_set) == n_speakers: # meet the number of speakers, but not the duration, starting over --- TODO: will try to find the segments that only contains those speakers
                    total_duration = 0.0
                    n_spk_left = n_speakers
                    total_spk_set = set()
                    tracks = []
                    
        return tracks, len(total_spk_set)
    
    def reorder_spk_mapping(self, cut: MixedCut, pattern=r'<\|spltoken\d+\|>') -> str:
        """
        Concatenate the texts of the input cuts.
        
        """
        global_spk_mapping = {}
        str_pattern = pattern.replace("\\", '')
        left_str, right_str = str_pattern.split('d+')
        for i, track in enumerate(cut.tracks):
            local_inverse_spk_mapping = {}
            local_spk_mapping = {}
            for speaker in track.cut.global_speaker_ids:
                if speaker not in global_spk_mapping:
                    global_spk_mapping[speaker] = len(global_spk_mapping)
                if speaker not in local_spk_mapping:
                    local_spk_mapping[speaker] = len(local_spk_mapping)
                    local_inverse_spk_mapping[len(local_inverse_spk_mapping)] = speaker
                    
            if i != 0:
                text = ''
                for word in track.cut.text.split(): 
                    if len(re.findall(pattern, word)) > 0:
                        local_spk_idx = int(word.replace(left_str,'').replace(right_str, ''))
                        spk = local_inverse_spk_mapping[local_spk_idx]
                        global_spk_idx = global_spk_mapping[spk]
                        text += f'{left_str}{global_spk_idx}{right_str}'
                    else:
                        text += ' ' + word
                track.cut.supervisions[0].text = text
                cut.supervisions[i].text = text
            else:
                cut.supervisions[0].text = track.cut.text
                # TODO: need to check the last speaker of last track and the first speaker of the current track 
                # if they are the same, we need to remove the the speaker token from the current track for segment-level
                # Do not need to remove the speaker token for word-level
            
        return cut
    
    def apply_speaker_distribution(self, num_meetings: int, speaker_count_distribution) -> Dict[int, int]:
        """
        Balance the speaker distribution for the simulated meetings.
        Args:
            num_meetings: The total number of simulated meetings.
            speaker_count_distribution: The speaker count distribution for the simulated meetings.
        For each number of speakers, calculate the number of meetings needed to balance the distribution.
        """

        total_spk = sum(speaker_count_distribution)
        num_speakers2num_meetings = {}
        for i_spk in range(self.max_num_speakers):
            num_speakers2num_meetings[i_spk+1] = round(num_meetings * speaker_count_distribution[i_spk] / total_spk)

        return num_speakers2num_meetings
        
    
    @dill_enabled(True)
    def simulate(self, 
        cuts: CutSet,
        num_meetings: int = 10000,
        seed: int = 0,
        num_jobs: int = 1,
    ) -> CutSet:
        random.seed(seed)

        self.fit(cuts)
        

        num_speakers2num_meetings = self.apply_speaker_distribution(num_meetings, self.speaker_count_distribution)
        logging.warn(f"Will be generating {(','.join([str(i) for i in num_speakers2num_meetings.values()]))} samples for {(','.join([str(i) for i in num_speakers2num_meetings.keys()]))} speakers given speaker count distribution of {str(self.speaker_count_distribution)}.")
        num_speakers2num_meetings[1] = 0 # skip 1-speaker samples
        logging.warn(f'But 1-speaker samples will be skipped. Will be generating {sum(num_speakers2num_meetings.values()) - num_speakers2num_meetings[1]} samples in total.')

        # Step 0: Calculate the number of intra-session and inter-session concatentation samples
        n_spks = [k for k, v in self.num_spk2cut_ids.items() if len(v) > 0]
        valid_sim_n_spks = set([i+j for i in n_spks for j in n_spks]) # valid number of speakers for inter-session samples
        n_spk2n_intra_mt, n_spk2n_inter_mt = {i+1:0 for i in range(self.max_num_speakers)}, {i+1:0 for i in range(self.max_num_speakers)}
        for n_spk, n_mt in num_speakers2num_meetings.items():
            logging.warn(f"=="*16 + f"{n_spk}-speaker" + "=="*16)
            if n_mt <= 0:
                logging.warning(f"No concatentation samples for {n_spk} speakers. Will skip simulation for {n_spk} speakers.")
                continue
            n_intra_mt = int(n_mt * self.intra_session_concat_prob[n_spk-1])
            n_inter_mt = n_mt - n_intra_mt
            if n_spk in self.num_spk2sess_ids:
                logging.warn(f"Will be genrating {n_intra_mt} {n_spk}-speaker intra-session concatentation samples.")
                n_spk2n_intra_mt[n_spk] = n_intra_mt
            else:
                logging.warning(f"Cannot generate {n_intra_mt} {n_spk}-speaker intra-session samples by concatenating two samples from the same session since we only have samples for {','.join([str(i) for i in n_spks])} speakers.")
                n_spk2n_intra_mt[n_spk] = 0
                n_inter_mt = n_mt
            if n_spk in valid_sim_n_spks:
                logging.warn(f"Will be genrating {n_inter_mt} {n_spk}-speaker inter-session concatentation samples.")
                n_spk2n_inter_mt[n_spk] = n_inter_mt
            else:
                logging.warning(f"Cannot generate {n_inter_mt} {n_spk}-speaker inter-session samples by concatenating two samples from different sessions since we only have samples for {','.join([str(i) for i in n_spks])} speakers.")
                if n_spk2n_intra_mt[n_spk] != 0:
                    n_spk2n_intra_mt[n_spk] = n_mt
                    logging.warn(f"Will be genrating {n_spk2n_intra_mt[n_spk]} {n_spk}-speaker intra-session concatentation samples instead.")
                else:
                    logging.warning(f"No samples for {n_spk} speakers. Will skip simulation for {n_spk} speakers.")
        logging.warn(f"""Will be generating {','.join([str(i) for i in n_spk2n_intra_mt.values()])} intra-session concatentation samples and {','.join([str(i) for i in n_spk2n_inter_mt.values()])} inter-session concatentation samples for {','.join([str(i+1) for i in range(self.max_num_speakers)])} speakers.""")
        # Step 1: intra-session
        num_intra_meetings = 0
        intra_mixtures = []
        logging.info(f"Simulating intra-session concatentation samples.")
        for n_spk, n_mt in n_spk2n_intra_mt.items():
            if n_mt <= 0:
                continue

            for i in tqdm(range(n_mt), desc=f"Simulating {n_spk}-speaker intra-session mixtures", ncols=128):
                intra_mixtures.append(self._create_mixture(n_speakers=n_spk, is_intra_session_concat=True))
            num_intra_meetings += n_mt
        logging.info(f"Finished simulating intra-session concatentation samples. Total number of intra-session concatentation samples: {num_intra_meetings}")
    
        # Steo 2: inter-session
        logging.info(f"Simulating inter-session concatentation samples.")
        
        num_inter_meetings = 0
        inter_mixtures = []
        for n_spk, n_mt in n_spk2n_inter_mt.items():
            if n_mt <= 0:
                continue
            
            for i in tqdm(range(n_mt), desc=f"Simulating {n_spk}-speaker inter-session mixtures", ncols=128):
                inter_mixtures.append(self._create_mixture(n_speakers=n_spk, is_intra_session_concat=False))
            num_inter_meetings += n_mt
        logging.info(f"Finished simulating inter-session concatentation samples. Total number of inter-session concatentation samples: {num_inter_meetings}")

        if num_inter_meetings + num_intra_meetings == 0:
            logging.warning(f"No samples are generated. Probably the duration of the segments is not within the range of min {self.min_duration//2} and max {self.max_duration//2}, or the speaker count distribution is not correctly set.")


        # Multi-processing gets slower, TODO
        # else:
        #     futures = []
        #     for n_spk, n_mt in num_speakers2num_meetings.items():
        #         tp = concurrent.futures.ProcessPoolExecutor(max_workers=num_jobs)
        #         futures.extend([tp.submit(self._create_mixture, n_spk) for _ in range(n_mt)])
        #     pbar = tqdm(total=num_meetings, desc=f"Simulating mixtures", unit="line", ncols=128) 
        #     count = 0
        #     for f in concurrent.futures.as_completed(futures):
        #         count += 1
        #         pbar.update()
        #         mixtures.append(f.result())
        #     tp.shutdown()
        #     pbar.close()

        return CutSet.from_cuts(intra_mixtures + inter_mixtures)
    

class MixMeetingSimulator():
    """
    This simulator Mix the segments from different/same sessions to create a
    multi-speaker meeting. 
    """

    def __init__(
        self,
        intra_session_mix_prob: Union[float, List[float]] = [0, 0, 0, 0],
        data_type: str = "msasr",
        min_duration: float = 80.0,
        max_duration: float = 100.0,
        max_num_speakers: int = 4,
        speaker_count_distribution: List[float] = [0, 0, 0.1, 4],
        valid_dataset_ids: List[str] = [],
    ):
        """
        :param intra_session_mix_prob: the probability of concatenating segments from the same
            session. [Default: 1]
        :param data_type: the type of data to simulate. Either 'msasr' or 'diar'. If 'msasr',
            the transcripts are included in the simulation,and the boundary segments are 
            not included. [Default: 'msasr']
        :param max_duration: the maximum duration of the simulated meeting. [Default: 40.0]
        """
        super().__init__()
        if isinstance(intra_session_mix_prob, float):
            self.intra_session_mix_prob = [intra_session_mix_prob] * (max_num_speakers)
        elif len(intra_session_mix_prob) == max_num_speakers:
            self.intra_session_mix_prob = intra_session_mix_prob
        else:
            raise ValueError(f"intra_session_mix_prob must be either a float or a list of floats, but got {intra_session_mix_prob}")
        if data_type not in ["msasr", "diar"]:
            raise ValueError("data_type must be either 'msasr' or 'diar', but got {data_type}")
        self.data_type = data_type
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_num_speakers = max_num_speakers
        self.speaker_count_distribution = speaker_count_distribution
        self.valid_dataset_ids = valid_dataset_ids
        assert len(speaker_count_distribution) == max_num_speakers, f"Length of speaker_count_distribution {len(speaker_count_distribution)} must be equal to max_num_speakers {max_num_speakers}"

    def fit(self, cuts) -> CutSet:
        """
        Read the manifest file and return a CutSet object. 
        Each line in the manifest file should be a JSON object representing a segment.
        """

        self.id2cut = {}
        self.sess2cut_ids = defaultdict(list)
        self.sess2spks = defaultdict(set)
        self.data2sess_ids = defaultdict(list)
        self.spk2cut_ids = defaultdict(list)
        self.data2num_spk2cut_ids = {}
        self.sess2num_spk2cut_ids = {}
        self.num_spk2cut_ids = {i+1:[] for i in range(self.max_num_speakers)}
        for i, cut in tqdm(enumerate(cuts), desc="Reading segments", ncols=100, total=len(cuts)):
            if not self.min_duration <= cut.duration <= self.max_duration:
                continue
            if not hasattr(cut, 'dataset_id') or cut.dataset_id is None:
                continue
            if self.valid_dataset_ids and cut.dataset_id not in self.valid_dataset_ids:
                continue
            if cut.dataset_id not in self.data2num_spk2cut_ids:
                self.data2num_spk2cut_ids[cut.dataset_id] = defaultdict(list)
            if cut.recording_id not in self.sess2num_spk2cut_ids:
                self.sess2num_spk2cut_ids[cut.recording_id] = defaultdict(list)
            
            speakers = cut.global_speaker_ids
            if self.data_type == "msasr":
                speaker_tokens = set(re.findall(r'<\|spltoken\d+\|>', cut.text))
                if len(speakers) != len(speaker_tokens): 
                    # Lhotse automatically fixes the max duration of the cut, 
                    # resulting in the mismatch of the number of speakers 
                    # and speaker tokens for the last segment
                    # TODO: need to fix the issue in Lhotse that automatically fixes the max duration
                    continue
            for spk in speakers:
                self.spk2cut_ids[spk].append(cut.id)
            self.sess2spks[cut.recording_id] = self.sess2spks[cut.recording_id].union(speakers)
            
            self.id2cut[cut.id] = cut
            self.sess2cut_ids[cut.recording_id].append(cut.id)
            self.data2num_spk2cut_ids[cut.dataset_id][len(speakers)].append(cut.id)
            self.sess2num_spk2cut_ids[cut.recording_id][len(speakers)].append(cut.id)
            self.num_spk2cut_ids[len(speakers)].append(cut.id)
            if cut.recording_id not in self.data2sess_ids[cut.dataset_id]:
                self.data2sess_ids[cut.dataset_id].append(cut.recording_id)
                
        self.cut_ids = list(self.id2cut.keys())
        self.num_spk2sess_ids = groupby(lambda x: len(self.sess2spks[x]), self.sess2spks.keys())
        
        self.data2global_speaker = {
            dataset_id: True for dataset_id in self.data2sess_ids.keys()
        }        
            
    def _create_mixture(self, n_speakers: int, is_intra_session_concat=False) -> MixedCut:

        db_norm = norm.rvs(-32.05957708631966, 5.66648411405886) # mean and std from Fisher data
        
        if is_intra_session_concat:
            # intra-dataset and intra-session concatenation
            tracks, num_speakers = self.get_intra_session_tracks(n_speakers, db_norm=db_norm)

        else: 
            # intra-dataset but inter-session concatenation
            tracks, num_speakers = self.get_inter_session_tracks(n_speakers, db_norm=db_norm)

        cut = MixedCut(id='mix_' + '_'.join([track.cut.id for track in tracks]), tracks=tracks)
        if self.data_type == "msasr":
            cut = self.reorder_spk_mapping(cut)

        assert self.min_duration <= cut.duration <= self.max_duration, f"Total duration {cut.duration} is not within the range of min {self.min_duration} and max {self.max_duration}"
        assert n_speakers == num_speakers, f"Total number of speakers {cut.num_speakers} is not equal to the number of speakers {n_speakers}"

        return cut
    
    def get_intra_session_tracks(self, n_speakers: int=4, db_norm: float=-25) -> List[MixTrack]:
        """
        Get the tracks for the MixedCut object.
        """
        session_id = random.choice(self.num_spk2sess_ids[n_speakers])
        
        total_spk_set = set()
        tracks = []
        while True:
            cut = self.id2cut[random.choice(self.sess2cut_ids[session_id])]
            tracks.append(MixTrack(cut=deepcopy(cut.normalize_loudness(target=db_norm, mix_first=False)), type=type(cut), offset=0))
            total_spk_set = total_spk_set.union(cut.global_speaker_ids)
            total_duration = max(total_duration, cut.duration)

            # break condition
            if total_duration >= self.min_duration:
                if total_duration > self.max_duration: # exceed the maximum duration, starting over
                    total_duration = 0.0
                    total_spk_set = set()
                    tracks = []
                    session_id = random.choice(self.num_spk2sess_ids[n_speakers])
                if len(total_spk_set) == n_speakers: # meet the number of speakers and duration, break
                    break
                else:
                    total_duration = 0.0
                    total_spk_set = set()
                    tracks = []
                    session_id = random.choice(self.num_spk2sess_ids[n_speakers])
            
        return tracks, len(total_spk_set)

    def get_inter_session_tracks(self, n_speakers: int=4, db_norm: float=-25) -> List[MixTrack]:
        """
        Get the tracks for the MixedCut object.
        """
        sample_cut = self.id2cut[random.choice(self.cut_ids)]
        dataset_id = sample_cut.dataset_id
        n_spk_list = [n_spk for n_spk, cut_ids in self.data2num_spk2cut_ids[dataset_id].items() if len(cut_ids) > 0]
        sum_spk_list = set([i + j for i in n_spk_list for j in n_spk_list])

        if min(sum_spk_list) > n_speakers:
            raise ValueError(f"Cannot generate {n_speakers}-speaker inter session samples by concatenating two samples since the dataset {dataset_id} only have {','.join([str(i) for i in n_spk_list])} speakers.")

        n_spk_left = n_speakers
        total_duration = 0.0
        total_spk_set = set()
        tracks = []
        num_spk2cut_ids = self.data2num_spk2cut_ids[dataset_id]
        while True:
            if n_spk_left >= 2:
                n_spk = 2
            else:
                # n_spk = random.choice([n_spk for n_spk in n_spk_list if n_spk <= n_spk_left])
                n_spk = 1

            while True:
                cut = self.id2cut[random.choice(num_spk2cut_ids[n_spk])]
                spks = set(cut.global_speaker_ids)
                if not spks.intersection(total_spk_set):
                    break

            tracks.append(MixTrack(cut=deepcopy(cut.normalize_loudness(target=db_norm, mix_first=False)), type=type(cut), offset=0))
            total_duration = max(total_duration, cut.duration)
            n_spk_left -= n_spk
            total_spk_set = total_spk_set.union(spks)

            # break condition
            
            if total_duration >= self.min_duration:
                if total_duration > self.max_duration or len(tracks) > 2: # exceed the maximum duration, starting over
                    total_duration = 0.0
                    n_spk_left = n_speakers
                    total_spk_set = set()
                    tracks = []
                if len(total_spk_set) == n_speakers: # meet the number of speakers and duration, break
                    break
            else:
                if len(total_spk_set) == n_speakers: # meet the number of speakers, but not the duration, starting over --- TODO: will try to find the segments that only contains those speakers
                    total_duration = 0.0
                    n_spk_left = n_speakers
                    total_spk_set = set()
                    tracks = []
                    
        return tracks, len(total_spk_set)
    
    def reorder_spk_mapping(self, cut: MixedCut, pattern=r'<\|spltoken\d+\|>') -> str:
        """
        Concatenate the texts of the input cuts.
        
        """
        global_spk_mapping = {}
        str_pattern = pattern.replace("\\", '')
        left_str, right_str = str_pattern.split('d+')
        for i, track in enumerate(cut.tracks):
            local_inverse_spk_mapping = {}
            local_spk_mapping = {}
            for speaker in track.cut.global_speaker_ids:
                if speaker not in global_spk_mapping:
                    global_spk_mapping[speaker] = len(global_spk_mapping)
                if speaker not in local_spk_mapping:
                    local_spk_mapping[speaker] = len(local_spk_mapping)
                    local_inverse_spk_mapping[len(local_inverse_spk_mapping)] = speaker
                    
            if i != 0:
                text = ''
                for word in track.cut.text.split(): 
                    if len(re.findall(pattern, word)) > 0:
                        local_spk_idx = int(word.replace(left_str,'').replace(right_str, ''))
                        spk = local_inverse_spk_mapping[local_spk_idx]
                        global_spk_idx = global_spk_mapping[spk]
                        text += f'{left_str}{global_spk_idx}{right_str}'
                    else:
                        text += ' ' + word
                track.cut.supervisions[0].text = text
                cut.supervisions[i].text = text
            else:
                cut.supervisions[0].text = track.cut.text
                # TODO: need to check the last speaker of last track and the first speaker of the current track 
                # if they are the same, we need to remove the the speaker token from the current track for segment-level
                # Do not need to remove the speaker token for word-level
            
        return cut
    
    def apply_speaker_distribution(self, num_meetings: int, speaker_count_distribution) -> Dict[int, int]:
        """
        Balance the speaker distribution for the simulated meetings.
        Args:
            num_meetings: The total number of simulated meetings.
            speaker_count_distribution: The speaker count distribution for the simulated meetings.
        For each number of speakers, calculate the number of meetings needed to balance the distribution.
        """

        total_spk = sum(speaker_count_distribution)
        num_speakers2num_meetings = {}
        for i_spk in range(self.max_num_speakers):
            num_speakers2num_meetings[i_spk+1] = round(num_meetings * speaker_count_distribution[i_spk] / total_spk)

        return num_speakers2num_meetings
        
    
    @dill_enabled(True)
    def simulate(self, 
        cuts: CutSet,
        num_meetings: int = 10000,
        seed: int = 0,
        num_jobs: int = 1,
    ) -> CutSet:
        random.seed(seed)

        self.fit(cuts)

        num_speakers2num_meetings = self.apply_speaker_distribution(num_meetings, self.speaker_count_distribution)
        logging.warn(f"Will be generating {(','.join([str(i) for i in num_speakers2num_meetings.values()]))} samples for {(','.join([str(i) for i in num_speakers2num_meetings.keys()]))} speakers given speaker count distribution of {str(self.speaker_count_distribution)}.")
        num_speakers2num_meetings[1] = 0 # skip 1-speaker samples
        logging.warn(f'But 1-speaker samples will be skipped. Will be generating {sum(num_speakers2num_meetings.values()) - num_speakers2num_meetings[1]} samples in total.')

        # Step 0: Calculate the number of intra-session and inter-session concatentation samples
        n_spks = [k for k, v in self.num_spk2cut_ids.items() if len(v) > 0]
        valid_sim_n_spks = set([i+j for i in n_spks for j in n_spks]) # valid number of speakers for inter-session samples
        n_spk2n_intra_mt, n_spk2n_inter_mt = {i+1:0 for i in range(self.max_num_speakers)}, {i+1:0 for i in range(self.max_num_speakers)}
        for n_spk, n_mt in num_speakers2num_meetings.items():
            logging.warn(f"=="*16 + f"{n_spk}-speaker" + "=="*16)
            if n_mt <= 0:
                logging.warning(f"No intra-session concatentation samples for {n_spk} speakers. Will skip simulation for {n_spk} speakers.")
                continue
            n_intra_mt = int(n_mt * self.intra_session_mix_prob[n_spk-1])
            n_inter_mt = n_mt - n_intra_mt
            if n_spk in self.num_spk2sess_ids:
                logging.warn(f"Will be genrating {n_intra_mt} {n_spk}-speaker intra-session concatentation samples.")
                n_spk2n_intra_mt[n_spk] = n_intra_mt
            else:
                logging.warning(f"Cannot generate {n_intra_mt} {n_spk}-speaker intra-session samples by concatenating two samples from the same session since we only have samples for {','.join([str(i) for i in n_spks])} speakers.")
                n_spk2n_intra_mt[n_spk] = 0
                n_inter_mt = n_mt
            if n_spk in valid_sim_n_spks:
                logging.warn(f"Will be genrating {n_inter_mt} {n_spk}-speaker inter-session concatentation samples.")
                n_spk2n_inter_mt[n_spk] = n_inter_mt
            else:
                logging.warning(f"Cannot generate {n_inter_mt} {n_spk}-speaker inter-session samples by concatenating two samples from different sessions since we only have samples for {','.join([str(i) for i in n_spks])} speakers.")
                if n_spk2n_intra_mt[n_spk] != 0:
                    n_spk2n_intra_mt[n_spk] = n_mt
                    logging.warn(f"Will be genrating {n_spk2n_intra_mt[n_spk]} {n_spk}-speaker intra-session concatentation samples instead.")
                else:
                    logging.warning(f"No samples for {n_spk} speakers. Will skip simulation for {n_spk} speakers.")
        logging.warn(f"""Will be generating {','.join([str(i) for i in n_spk2n_intra_mt.values()])} intra-session concatentation samples and {','.join([str(i) for i in n_spk2n_inter_mt.values()])} inter-session concatentation samples for {','.join([str(i+1) for i in range(self.max_num_speakers)])} speakers.""")
        # Step 1: intra-session
        num_intra_meetings = 0
        intra_mixtures = []
        logging.info(f"Simulating intra-session concatentation samples.")
        for n_spk, n_mt in n_spk2n_intra_mt.items():
            if n_mt <= 0:
                continue

            for i in tqdm(range(n_mt), desc=f"Simulating {n_spk}-speaker intra-session mixtures", ncols=128):
                intra_mixtures.append(self._create_mixture(n_speakers=n_spk, is_intra_session_concat=True))
            num_intra_meetings += n_mt
        logging.info(f"Finished simulating intra-session concatentation samples. Total number of intra-session concatentation samples: {num_intra_meetings}")
    
        # Steo 2: inter-session
        logging.info(f"Simulating inter-session concatentation samples.")
        
        num_inter_meetings = 0
        inter_mixtures = []
        for n_spk, n_mt in n_spk2n_inter_mt.items():
            if n_mt <= 0:
                continue
            
            for i in tqdm(range(n_mt), desc=f"Simulating {n_spk}-speaker inter-session mixtures", ncols=128):
                inter_mixtures.append(self._create_mixture(n_speakers=n_spk, is_intra_session_concat=False))
            num_inter_meetings += n_mt
        logging.info(f"Finished simulating inter-session concatentation samples. Total number of inter-session concatentation samples: {num_inter_meetings}")

        if num_inter_meetings + num_intra_meetings == 0:
            logging.warning(f"No samples are generated. Probably the duration of the segments is not within the range of min {self.min_duration} and max {self.max_duration}, or the speaker count distribution is not correctly set.")

        return CutSet.from_cuts(intra_mixtures + inter_mixtures)

class LibriSpeechMixSimulator():

    def __init__(
        self,
        data_type: str = "msasr",
        min_delay: float = 0.5,
        max_num_speakers: int = 4,
        speaker_token_position: str = 'sot',
        speaker_count_distribution: List[float] = [0, 2, 3, 4],
        delay_factor: int = 1
    ):
        """
        Args:
        data_type: the type of data to simulate. Either 'msasr', 'tsasr' or 'diar'. [Default: 'msasr']
        min_delay: the minimum delay between the segments. [Default: 0.5]
        max_num_speakers: the maximum number of speakers in the meeting. [Default: 4]
        speaker_token_position: the position of the speaker token in the text. Either 'sot', 'word', or 'segments'. [Default: 'sot']
        speaker_count_distribution: the speaker count distribution for the simulated meetings. [Default: [0, 2, 3, 4]]
        delay_factor: the number of times to repeat the meeting with the same speakers. [Default: 1]
        """
        super().__init__()
        self.data_type = data_type
        self.min_delay = min_delay
        self.delay_factor = delay_factor
        self.max_num_speakers = max_num_speakers
        self.speaker_token_position = speaker_token_position
        self.speaker_count_distribution = speaker_count_distribution
        assert len(speaker_count_distribution) == max_num_speakers, f"Length of speaker_count_distribution {len(speaker_count_distribution)} must be equal to max_num_speakers {max_num_speakers}"

    def fit(self, cuts) -> CutSet:
        self.speaker_id2cut_ids = defaultdict(list)
        self.id2cuts = defaultdict(list)
        for cut in tqdm(cuts, desc="Reading segments", ncols=100):
            # if not hasattr(cut, 'dataset_id') or cut.dataset_id != 'librispeech':
            #     continue
            if hasattr(cuts[0], 'speaker_id'):
                speaker_id = cut.speaker_id
            else: #LibriSpeech
                speaker_id = cut.recording_id.split('-')[0]
                cut.speaker_id = speaker_id
            self.speaker_id2cut_ids[speaker_id].append(cut.id)
            self.id2cuts[cut.id] = cut
        
        self.speaker_ids = list(self.speaker_id2cut_ids.keys())

    def _create_mixture(self, n_speakers: int) -> MixedCut:
        sampled_speaker_ids = random.sample(self.speaker_ids, n_speakers)
        
        mono_cuts = []
        for speaker_id in sampled_speaker_ids:
            cut_id = random.choice(self.speaker_id2cut_ids[speaker_id])
            cut = self.id2cuts[cut_id]
            mono_cuts.append(cut)

        mixed_cuts = []
        for i in range(self.delay_factor):
            tracks = []
            offset = 0.0
            for mono_cut in mono_cuts:
                custom = {
                        'pnc': 'no',
                        'source_lang': 'en',
                        'target_lang': 'en',
                        'task': 'asr'
                    }
                mono_cut.custom.update(custom)
                tracks.append(MixTrack(cut=deepcopy(mono_cut), type=type(mono_cut), offset=offset))
                offset += random.uniform(self.min_delay, mono_cut.duration)
        
            mixed_cut = MixedCut(id='lsmix_' + '_'.join([track.cut.id for track in tracks]) + '_' + str(uuid4()), tracks=tracks)
            
            if self.data_type == "msasr":
                text = self.get_text(mixed_cut, speaker_token_position=self.speaker_token_position)
                sup = SupervisionSegment(id=mixed_cut.id, recording_id=mixed_cut.id, start=0, duration=mixed_cut.duration, text=text)
                mixed_cut.tracks[0].cut.supervisions = [sup]

            if self.data_type == "tsasr":
                query_speaker_id = random.choice(sampled_speaker_ids)
                query_audio_path = random.choice(self.speaker_id2cut_ids[query_speaker_id])
                pass # TODO: need to implement the query audio path

            if self.data_type == "diar":
                pass # TODO: need to implement the diar data type

            mixed_cuts.append(mixed_cut)

        return mixed_cuts
    
    # TODO: text is necessary for msasr and tsasr, but not for diar
    def get_text(self, cut: MixedCut, speaker_token_style='<|spltoken*|>', speaker_token_position='sot') -> str:
        text = ''
        stt_words_spks = []
        if speaker_token_position == 'word' or speaker_token_position == 'segment':
            SSP = SyllableTokenizer()
            for i, track in enumerate(cut.tracks):
                stt_time, end_time = track.offset, track.offset + track.cut.duration
                word_seq = track.cut.text
                word_list = word_seq.split()
                syllab_word_list = [[] for _ in range(len(word_list))]
                syllab_list = []
                for idx,word in enumerate(word_list):
                    syllables = SSP.tokenize(word)
                    syllab_word_list[idx].extend(syllables)
                    syllab_list.extend(syllables)
                avg_sylllable_dur = (end_time - stt_time) / len(syllab_list)
                offset = stt_time
                stt_times, end_times = [], []
                for word, syllabs in zip(word_list, syllab_word_list):
                    stt_times.append(round(offset, 3))
                    end_time = round(offset + avg_sylllable_dur * len(syllabs), 3)
                    end_times.append(end_time)
                    offset = end_time
                    stt_words_spks.append([offset, word, speaker_token_style.replace('*', str(i))])
            stt_words_spks = sorted(stt_words_spks, key=lambda x: x[0])
            if speaker_token_position == 'word':
                text += ' '.join([f'{spk} {word}' for stt_time, word, spk in stt_words_spks])   
            elif speaker_token_position == 'segment':
                pre_spk = ''
                for stt_time, word, spk in stt_words_spks:
                    if pre_spk != spk:
                        text += spk + ' '
                        pre_spk = spk
                    text += word + ' '
        elif speaker_token_position == 'sot':
            for i, track in enumerate(cut.tracks):
                cut = track.cut
                text += speaker_token_style.replace('*', str(i)) + ' ' + cut.text + ' '
        else:
            raise ValueError(f"speaker_token_position must be either 'sot', 'word', or 'segments', but got {speaker_token_position}")
        return text
    
    def apply_speaker_distribution(self, num_meetings: int, speaker_count_distribution) -> Dict[int, int]:
        """
        Balance the speaker distribution for the simulated meetings.
        Args:
            num_meetings: The total number of simulated meetings.
            speaker_count_distribution: The speaker count distribution for the simulated meetings.
        For each number of speakers, calculate the number of meetings needed to balance the distribution.
        """

        total_spk = sum(speaker_count_distribution)
        num_speakers2num_meetings = {}
        for i_spk in range(self.max_num_speakers):
            num_speakers2num_meetings[i_spk+1] = round(num_meetings * speaker_count_distribution[i_spk] / total_spk)

        return num_speakers2num_meetings
            
    def simulate(self, 
        cuts: CutSet,
        num_meetings: int = 10000,
        seed: int = 0,
        num_jobs: int = 1,
    ) -> CutSet:
        random.seed(seed)

        self.fit(cuts)

        self.num_speakers2num_meetings = self.apply_speaker_distribution(num_meetings, self.speaker_count_distribution)

        cut_set = []
        for n_speakers, n_mt in self.num_speakers2num_meetings.items():
            if n_mt <= 0:
                continue
            for i in tqdm(range(n_mt), desc=f"Simulating {n_speakers}-speaker mixtures", ncols=128):
                cut_set.extend(self._create_mixture(n_speakers=n_speakers))

        return CutSet.from_cuts(cut_set).shuffle()

class LibriSpeechMixGenerator():
    def __init__(self):
        pass

    def generate(self, cuts):
        cut_set = []
        for cut in tqdm(cuts, desc="Generating LibriSpeechMix", total=len(cuts), disable=True):
            offsets = cut.delays
            durations = cut.durations
            wavs = cut.wavs
            texts = cut.texts
            speakers = cut.speakers

            tracks = []
            for i, (offset, duration, wav, text, speaker) in enumerate(zip(offsets, durations, wavs, texts, speakers)):
                wav_dur = soundfile.info(wav).duration
                wav_samples = soundfile.info(wav).frames
                custom = {
                    'speaker': speaker,
                    'text': text,
                }
                cut_1spk = MonoCut(
                    id=wav.split('/')[-1].replace('.wav', ''),
                    start=0,
                    duration=duration,
                    channel=0,
                    supervisions=[],
                    recording=Recording(
                        id=wav.split('/')[-1].replace('.wav', ''),
                        sources=[
                            AudioSource(
                                type='file',
                                channels=[0],
                                source=wav
                            )
                        ],
                        sampling_rate=16000, 
                        num_samples=wav_samples,
                        duration=wav_dur
                    ),
                    custom=custom
                )

                tracks.append(MixTrack(cut=cut_1spk, type=type(cut_1spk), offset=offset))
            sup = SupervisionSegment(
                id=cut.id,
                recording_id=cut.recording_id,
                start=0,
                duration=offset+wav_dur,
                text=cut.text,
            )
            tracks[0].cut.supervisions.append(sup)
            cut_multi_spk = MixedCut(id=cut.id, tracks=tracks)
            
            cut_set.append(cut_multi_spk)
        
        return CutSet.from_cuts(cut_set)

import numpy as np
import matplotlib.pyplot as plt

def save_numpy_array_as_png(np_array_source: np.ndarray, output_filepath: str):
    """
    Save a NumPy array as a PNG image.

    Args:
        np_array_source (np.ndarray): The image data (2D or 3D).
        output_filepath (str): File path where the image will be saved (e.g., "image.png").
    """
    expanded_array = np.repeat(np_array_source, 100, axis=1)
    expanded_array = expanded_array.T  # Transpose the array
    plt.imsave(output_filepath, expanded_array)

def speaker_to_target(
    a_cut,
    num_speakers: int = 4, 
    num_sample_per_mel_frame: int = 160, 
    num_mel_frame_per_asr_frame: int = 8, 
    spk_tar_all_zero: bool = False,
    boundary_segments: bool = False,
    soft_label: bool = False,
    ignore_num_spk_mismatch: bool = True,
    soft_thres: float = 0.5,
    is_audio_mix_sim: bool = False,
    ):
    '''
    Get rttm samples corresponding to one cut, generate speaker mask numpy.ndarray with shape (num_speaker, hidden_length)
    This function is needed for speaker diarization with ASR model trainings.

    Args:
        a_cut (MonoCut, MixedCut): Lhotse Cut instance which is MonoCut or MixedCut instance.
        num_speakers (int): max number of speakers for all cuts ("mask" dim0), 4 by default
        num_sample_per_mel_frame (int): number of sample per mel frame, sample_rate / 1000 * window_stride, 160 by default (10ms window stride)
        num_mel_frame_per_asr_frame (int): encoder subsampling_factor, 8 by default
        spk_tar_all_zero (Tensor): set to True gives all zero "mask"
        boundary_segments (bool): set to True to include segments containing the boundary of the cut, False by default for multi-speaker ASR training
        soft_label (bool): set to True to use soft label that enables values in [0, 1] range, False by default and leads to binary labels.
        ignore_num_spk_mismatch (bool): This is a temporary solution to handle speaker mismatch. Will be removed in the future.
    
    Returns:
        mask (Tensor): speaker mask with shape (num_speaker, hidden_lenght)
    '''
    # get cut-related segments from rttms
    # basename = os.path.basename(a_cut.rttm_filepath).replace('.rttm', '')
    if isinstance(a_cut, MixedCut):
        cut_list = [track.cut for track in a_cut.tracks if isinstance(track.cut, MonoCut)]
        offsets = [track.offset for track in a_cut.tracks if isinstance(track.cut, MonoCut)]
    elif isinstance(a_cut, MonoCut):
        cut_list = [a_cut]
        offsets = [0]
    else:
        raise ValueError(f"Unsupported cut type type{a_cut}: only MixedCut and MonoCut are supported")
    
    segments_total = []

    for i, cut in enumerate(cut_list):
        if hasattr(cut, 'rttm_filepath') and cut.rttm_filepath is not None:
            rttms = SupervisionSet.from_rttm(cut.rttm_filepath)
        elif hasattr(cut, 'speaker_id') and cut.speaker_id is not None:

            rttms = SupervisionSet.from_segments([SupervisionSegment(
                id=uuid4(),
                recording_id=cut.recording_id,
                start=cut.start,
                duration=cut.duration,
                channel=1,
                speaker=cut.speaker_id,
                language=None
            )])
        else:
            rttms = SupervisionSet.from_segments([SupervisionSegment(
                id=uuid4(),
                recording_id=cut.recording_id,
                start=cut.start,
                duration=cut.duration,
                channel=1,
                speaker='0',
                language=None
            )])
        if boundary_segments: # segments with seg_start < total_end and seg_end > total_start are included
            segments_iterator = find_segments_from_rttm(recording_id=cut.recording_id, rttms=rttms, start_after=cut.offset, end_before=cut.end, tolerance=0.0)
        else: # segments with seg_start > total_start and seg_end < total_end are included
            if hasattr(cut, 'offset'): # tarred dataset
                segments_iterator = rttms.find(recording_id=rttms[0].recording_id, start_after=cut.offset, end_before=cut.offset+cut.duration, adjust_offset=True) #, tolerance=0.0)
            else:
                segments_iterator = rttms.find(recording_id=cut.recording_id, start_after=cut.start, end_before=cut.end, adjust_offset=True) #, tolerance=0.0)
            
        for seg in segments_iterator:
            if seg.start < 0:
                seg.duration += seg.start
                seg.start = 0
            if seg.end > cut.duration:
                seg.duration -= seg.end - cut.duration
            seg.start += offsets[i]
            segments_total.append(seg)
    # apply arrival time sorting to the existing segments
    segments_total.sort(key = lambda rttm_sup: rttm_sup.start)

    seen = set()
    seen_add = seen.add
    speaker_ats = [s.speaker for s in segments_total if not (s.speaker in seen or seen_add(s.speaker))]
     
    speaker_to_idx_map = {
            spk: idx
            for idx, spk in enumerate(speaker_ats)
    }
    if len(speaker_to_idx_map) > num_speakers and not ignore_num_spk_mismatch:  # raise error if number of speakers
        raise ValueError(f"Number of speakers {len(speaker_to_idx_map)} is larger than the maximum number of speakers {num_speakers}")
        
    # initialize mask matrices (num_speaker, encoder_hidden_len)
    feat_per_sec = int(a_cut.sampling_rate / num_sample_per_mel_frame) # 100 by default
    num_samples = get_hidden_length_from_sample_length(a_cut.num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
    if spk_tar_all_zero: 
        frame_mask = torch.zeros((num_samples, num_speakers))
    else:
        frame_mask = get_mask_from_segments(segments_total, a_cut, speaker_to_idx_map, num_speakers, feat_per_sec, ignore_num_spk_mismatch)
    soft_mask = get_soft_mask(frame_mask, num_samples, num_mel_frame_per_asr_frame)

    if soft_label:
        mask = soft_mask
    else:
        mask = (soft_mask > soft_thres).float()
    return mask

def get_vad_mask(timestamps, offset, duration, sampling_rate=16000):
    num_samples = get_hidden_length_from_sample_length(duration * sampling_rate)
    mask = torch.zeros((num_samples, ))
    for timestamp in timestamps:
        stt = max(timestamp[0] - offset, 0)
        ent = min(timestamp[1] - offset, duration)
        stf = int(stt / 0.08)
        enf = int(ent / 0.08)
        mask[stf:enf] = 1.0 
    return mask

class MultiSpeakerMixtureGenerator():
    """
    This class is used to simulate multi-speaker audio data,
    which can be used for multi-speaker ASR and speaker diarization training.
    """
    def __init__(
        self, 
        manifest_filepath,
        rir_manifest,
        background_manifest,
        simulator_type,
        min_delay=0.5,
        outputs=None,
        random_seed=42,
        session_config=None, 
        num_speakers=2,
    ):
        """
        Args:
            cuts (CutSet): The cutset that contains single-speaker audio cuts.
                Please make sure that the cuts have the 'speaker_id' attribute.                    
            num_speakers (int): The number of speakers in the simulated audio.
                We only simulate the samples with the fixed number of speakers.
                The variation of the number of speakers is controlled by the weights in Lhotse dataloader config.
            simulator_type (str): The type of simulator to use.
                - 'lsmix': LibriSpeechMix-style training sample.
                - 'meeting': Meeting-style training sample.
                - 'conversation': Conversation-style training sample.
            speaker_distribution (list): The distribution of speakers in the simulated audio.
                The length of the list is the maximum number of speakers.
                The list elements are the weights for each speaker.
            min_delay (float): The minimum delay between speakers
                to avoid the same starting time for multiple speakers.
        """
        # cfg = OmegaConf.structured(MultiSpeakerSimulatorConfig())
        # cfg.data_simulator.manifest_filepath = manifest_filepath
        # # cfg.data_simulator.outputs.output_dir = self.outputs.output_dir
        # # cfg.data_simulator.outputs.output_filename = self.outputs.output_filename
        # cfg.data_simulator.session_config.num_sessions = session_config.num_sessions
        # cfg.data_simulator.session_config.session_length = session_config.session_length
        # cfg.data_simulator.background_noise.background_manifest = background_manifest
        # cfg.data_simulator.background_noise.rir_manifest = rir_manifest

        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)

        self.manifest_filepath = manifest_filepath 
        self.manifests = list(LazyJsonlIterator(manifest_filepath))
        self.background_manifest = background_manifest
        self.rir_manifest = read_rir_manifest(rir_manifest=rir_manifest) if rir_manifest else None

        self.min_delay = min_delay
        self.simulator_type = simulator_type
        self.outputs = outputs
        self.session_config = session_config
        self.max_speakers = num_speakers

        
        print("======  simulator_type", simulator_type)

        type2simulator = {
            'lsmix': self.LibriSpeechMixSimulator,
            'audiomix': self.AudioMixtureSimulator,
            'channel_separated_audio_mixer': self.ChannelSeparatedAudioMixer,
            'meeting': self.MeetingSimulator,
            'conversation': self.ConversationSimulator
        }
    
        self.simulator = type2simulator[simulator_type]

        if simulator_type == 'lsmix':
            self.spk2manifests = groupby(lambda x: x["speaker_id"], self.manifests)
            self.speaker_ids = list(self.spk2manifests.keys())

        self.count = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        self.count += 1
        return self.simulator()

    def _get_custom_dict(self, additional_info=None):
        custom = {
            'pnc': 'no',
            'source_lang': 'en',
            'target_lang': 'en',
            'task': 'asr',
        }
        if additional_info:
            custom.update(additional_info)
        return custom

    def AudioMixtureSimulator(self):
        """
        NeMo MultiSpeaker Speech Simulator
        """
        cfg = OmegaConf.structured(MultiSpeakerSimulatorConfig())
        cfg.data_simulator.manifest_filepath = self.manifest_filepath
        # cfg.data_simulator.outputs.output_dir = self.outputs.output_dir
        # cfg.data_simulator.outputs.output_filename = self.outputs.output_filename
        cfg.data_simulator.session_config.num_sessions = self.session_config.num_sessions
        cfg.data_simulator.session_config.num_speakers = self.session_config.num_speakers
        cfg.data_simulator.session_config.session_length = self.session_config.session_length
        cfg.data_simulator.background_noise.background_manifest = self.background_manifest
        simulator = MultiSpeakerSimulator(cfg=cfg)
        # set_start_method('spawn', force=True)
        # simulator.generate_sessions()
        # mixed_cut 
        manifest_json_list, uniq_id, mix_array, speaker_ids = simulator.generate_single_session()
        spk_mapping = {spk_id: spk_idx for spk_idx, spk_id in enumerate(speaker_ids)}
        uniq_id = f"audiomix_{self.count}_nspk{len(spk_mapping.keys())}"
        noise_json_list = simulator.annotator.annote_lists['noise']

        # load RIR
        num_speakers = self.session_config.num_speakers
        selected_rir_paths = random.sample(self.rir_manifest_list, num_speakers)
        reverb_rirs = [Recording.from_file(rir_path['audio_filepath']) for rir_path in selected_rir_paths]

        start_time_1 = time.time()

        # Speech Cuts added
        mono_cuts = []
        for manifest in manifest_json_list:
            mono_cuts.append(self.json_to_cut(json_dict=manifest, 
                                              speaker_index=spk_mapping[manifest['speaker_id']]))

        speaker_tracks, speaker_rir_mapping = {}, {}
        for mono_cut in mono_cuts:
            speaker_id=mono_cut.custom['speaker_id']
            if speaker_id not in speaker_rir_mapping:
                speaker_rir_mapping[speaker_id] = reverb_rirs[len(speaker_rir_mapping.keys())]

            speaker_rir_recording = speaker_rir_mapping[speaker_id]
            mono_cut.custom.update(self._get_custom_dict(speaker_id=speaker_id))
            rir_mono_cut = mono_cut.reverb_rir(speaker_rir_recording).perturb_volume(0.9)
            cln_mono_cut = mono_cut.perturb_volume(0.1)
            cln_mix_track = MixTrack(cut=deepcopy(cln_mono_cut), 
                                 type=type(cln_mono_cut), 
                                 offset=cln_mono_cut.custom['mixed_cut_offset']
                                )
            rir_mix_track = MixTrack(cut=deepcopy(rir_mono_cut), 
                                 type=type(rir_mono_cut), 
                                 offset=rir_mono_cut.custom['mixed_cut_offset']
                                )
            if speaker_id not in speaker_tracks:
                speaker_tracks[speaker_id] = [cln_mix_track, rir_mix_track]
            else:
                speaker_tracks[speaker_id].extend([cln_mix_track, rir_mix_track])

        # Noise Cuts added
        noise_cuts = []
        for noise_manifest in noise_json_list:
            noise_cuts.append(self.json_to_cut(noise_manifest))
        
        noise_tracks = []
        for noise_cut in noise_cuts:
            noise_cut.custom.update(self._get_custom_dict())
            noise_cut = noise_cut.perturb_volume(1.0)
            noise_tracks.append(MixTrack(cut=deepcopy(noise_cut), type=type(noise_cut), offset=noise_cut.custom['mixed_cut_offset']))

        # Merge all speaker tracks into a list 
        all_spk_tracks = []
        for spk_id, tracks in speaker_tracks.items():
            all_spk_tracks.extend(tracks)

        end_time_1 = time.time()
        elapsed_time_1 = end_time_1 - start_time_1

        start_time = time.time()
        mixed_cut = MixedCut(id=uniq_id, tracks=all_spk_tracks) 
        # + noise_tracks)
        end_time = time.time()
        elapsed_time = end_time - start_time
        mixed_noisy_cut = MixedCut(id=uniq_id, tracks=all_spk_tracks + noise_tracks)


        # uniq_folder_name = f"{uniq_id}_test_an_rir_v6_ro{str(simulator._params.data_simulator.session_params.random_offset)}_{self.session_config.num_speakers}spk"
        # bs_true = speaker_to_target(mixed_cut, boundary_segments=True, is_audio_mix_sim=True)
        # bs_false = speaker_to_target(mixed_cut, boundary_segments=False, is_audio_mix_sim=True)
        # save_numpy_array_as_png(bs_true, f"/home/taejinp/Downloads/lhotse_sup_boundary/{uniq_folder_name}/bs_true.png")
        # save_numpy_array_as_png(bs_false, f"/home/taejinp/Downloads/lhotse_sup_boundary/{uniq_folder_name}/bs_false.png")
        # basepath = "/home/taejinp/Downloads/lhotse_sup_boundary"
        # ### loaded_audio = mixed_cut.load_audio().squeeze(0)
        # loaded_audio = mixed_noisy_cut.load_audio().squeeze(0)
        # mix_array = mix_array.cpu().numpy()
        # ### Normalize the audio array
        # array = mix_array / (np.max(np.abs(mix_array)) + 1e-8)  # Avoid division by zero

        # package_path = f"{basepath}/{uniq_folder_name}"
        # os.makedirs(package_path, exist_ok=True)
        # sf.write(os.path.join(package_path, uniq_id + '.wav'), array, simulator._params.data_simulator.sr)
        # sf.write(os.path.join(package_path, uniq_id + '_lhotse.wav'), loaded_audio, simulator._params.data_simulator.sr)

        # package_path = None
        # simulator.annotator.write_annotation_rttm_and_ctm(
        #     basepath=package_path,
        #     filename=uniq_id,
        # )
        # print(f"====[  MixedCut Generated]===== {mixed_cut.id} {uniq_id}")
        # print(f"====[  MixedCut Generated]===== {self.count} mixed c
        return mixed_cut


    def LibriSpeechMixSimulator(self):
        """
        This function simulates a LibriSpeechMix-style training sample.
        Ref:
            Paper: https://arxiv.org/abs/2003.12687
            Github: https://github.com/NaoyukiKanda/LibriSpeechMix
        """
        # Sample the speakers
        sampled_speaker_ids = random.sample(self.speaker_ids, self.max_speakers)
        # Sample the cuts for each speaker
        mono_cuts = []
        for speaker_id in sampled_speaker_ids:
            manifest = random.choice(self.spk2manifests[speaker_id])
            mono_cuts.append(self.json_to_cut(manifest))

        tracks = []
        offset = 0.0
        for speaker_id, mono_cut in zip(sampled_speaker_ids, mono_cuts):
            tracks.append(MixTrack(cut=deepcopy(mono_cut), type=type(mono_cut), offset=offset))
            offset += random.uniform(self.min_delay, mono_cut.duration)
    
        mixed_cut = MixedCut(id='lsmix_' + '_'.join([track.cut.id for track in tracks]) + '_' + str(uuid4()), tracks=tracks)
        
        # update custom for vad_target
        offset, duration = 0, mixed_cut.duration
        for i in range(self.max_speakers):
            additional_info = {
                'vad_target': get_vad_mask(
                    timestamps=[[tracks[i].offset, tracks[i].offset + tracks[i].cut.duration]], 
                    offset=offset, 
                    duration=duration
                ),
                'speaker_id': sampled_speaker_ids[i]
            }
            custom = self._get_custom_dict(additional_info=additional_info)
            mixed_cut.tracks[i].cut.custom.update(custom)

        return mixed_cut

    def ChannelSeparatedAudioMixer(self):
        """
        This function simulates a channel-separated audio mixer.
        """
        manifest = random.choice(self.manifests)
        audio_filepath_list = manifest['audio_filepath']
        seglst_filepath_list = manifest['seglst_filepath']
        speaker_ids = manifest['speaker_ids']
        channel_list = manifest['channels']

        selected_channel = random.choice(channel_list)
        selected_seglst_filepath = seglst_filepath_list[selected_channel]
        seglst = SegList(seglst_filepath=selected_seglst_filepath)
        segments = seglst.get_segments(
            min_duration=10,
            max_duration=50
        )
        offset = min(segment.start for segment in segments)
        duration = max(segment.end for segment in segments) - offset
        text = ' '.join([segment.text for segment in segments])

        tracks = []
        for i_ch in channel_list:
            audio_filepath = audio_filepath_list[i_ch]
            json_dict = {
                'audio_filepath': audio_filepath,
                'duration': duration,
                'offset': offset,
                'speaker_id': speaker_ids[i_ch],
            }
            cut = self.json_to_cut(json_dict)
            if i_ch == selected_channel:
                additional_info={
                    'speaker_id': speaker_ids[i_ch],
                    'text': text,
                    'vad_target': get_vad_mask(
                        timestamps=[(s.start, s.end) for s in segments], 
                        offset=offset, 
                        duration=duration
                    )
                }
                cut.custom = self._get_custom_dict(
                    additional_info=additional_info
                )
            else:
                cut.custom = self._get_custom_dict()
            tracks.append(MixTrack(cut=deepcopy(cut), type=type(cut), offset=0))

        mixed_cut = MixedCut(id='channel_separated_audio_mixer_' + '_'.join([track.cut.id for track in tracks]) + '_' + str(uuid4()), tracks=tracks)

        return mixed_cut

    def MultiSpeakerMixtureLoader(self):

        manifest = random.choice(self.manifests)
        audio_filepath = manifest['audio_filepath']
        seglst_filepath = manifest['seglst_filepath']
        speaker_ids = manifest['speaker_ids']

        seglst = SegList(seglst_filepath=seglst_filepath)
        segments = seglst.get_segments(
            min_duration=10,
            max_duration=50
        )
        offset = min(segment.start for segment in segments)
        duration = max(segment.end for segment in segments) - offset
        text = ' '.join([segment.text for segment in segments])

        json_dict = {
            'audio_filepath': audio_filepath,
            'duration': duration,
            'offset': offset,
            'speaker_id': speaker_ids,
        }
        cut = self.json_to_cut(json_dict)
        additional_info={
            'text': text,
            'vad_target': get_vad_mask(
                timestamps=[(s.start, s.end) for s in segments], 
                offset=offset, 
                duration=duration
            )
        }
        cut.custom = self._get_custom_dict(
            additional_info=additional_info
        )

        return cut

    def MeetingSimulator(self):
        raise NotImplementedError("MeetingSimulator is not implemented yet.")   

    def ConversationSimulator(self):
        raise NotImplementedError("ConversationSimulator is not implemented yet.")
    
    def json_to_cut(self, json_dict, speaker_index: Union[int, None] = None):
        """
        Convert a json dictionary to a Cut instance.
        """
        audio_path = json_dict["audio_filepath"]
        duration = json_dict["duration"]
        offset = json_dict.get("offset", None)
        cut = self._create_cut(
            audio_path=audio_path, offset=offset, duration=duration, sampling_rate=json_dict.get("sampling_rate", None),
        )
        # Note that start=0 and not start=offset because supervision's start if relative to the
        # start of the cut; and cut.start is already set to offset

        if speaker_index is not None:
            cut_speaker_id = speaker_index
        else:
            cut_speaker_id = json_dict.get("speaker_id")

        if json_dict.get("text") is not None and json_dict.get("text") != "":
            cut_text = json_dict.get("text")
        else:
            cut_text = " ".join(json_dict.get("words", []))
            if cut_text == " ":
                cut_text = ""

        cut.supervisions.append(
            SupervisionSegment(
                id=cut.id,
                recording_id=cut.recording_id,
                start=0,
                duration=cut.duration,
                text=cut_text,
                language=json_dict.get("language", "en"),
                speaker=cut_speaker_id,
            )
        )
        cut.custom = json_dict
        cut.duration = duration
        return cut

    def _create_cut(
        self,
        audio_path: str,
        offset: float,
        duration: float,
        sampling_rate: int | None = None,
        channel: int = 0,
    ) -> Cut:
        
        recording = self._create_recording(audio_path, duration, sampling_rate)
        cut = recording.to_cut()
        if isinstance(cut.channel, list) and len(cut.channel) > 1:
            cut.channel = [channel]
        if offset is not None:
            cut = cut.truncate(offset=offset, duration=duration, preserve_id=True)
            cut.id = f"{cut.id}-{round(offset * 1e2):06d}-{round(duration * 1e2):06d}"
        return cut
    
    def _create_recording(
        self,
        audio_path: str,
        duration: float,
        sampling_rate: int | None = None,
    ) -> Recording:
        if sampling_rate is not None:
            # TODO(pzelasko): It will only work with single-channel audio in the current shape.
            return Recording(
                id=audio_path,
                sources=[AudioSource(type="file", channels=[0], source=audio_path)],
                sampling_rate=sampling_rate,
                num_samples=compute_num_samples(duration, sampling_rate),
                duration=duration,
                channel_ids=[0],
            )
        else:
            return Recording.from_file(audio_path)
            return Recording.from_file(audio_path)