# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import itertools
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
    session_length_range: List[int] = field(default_factory=lambda: [10, 40])

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
    background_manifest: str = ""
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
    
    def __str__(self):
        return f"Segment(start={self.start}, end={self.end}, speaker_id={self.speaker_id}, text=\"{self.text}\")"

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
    return_text: bool = False,
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
        if cut.custom.get('rttm_filepath', None):
            rttms = SupervisionSet.from_rttm(cut.rttm_filepath)
        elif cut.supervisions:
            rttms = SupervisionSet(cut.supervisions)
        else:
            # logging.warning(f"No rttm or supervisions found for cut {cut.id}")
            continue
            
        start = cut.offset if hasattr(cut, 'offset') else cut.start
        end = start + cut.duration
        recording_id = rttms[0].recording_id if len(rttms) > 0 else cut.recording_id
        if boundary_segments: # segments with seg_start < total_end and seg_end > total_start are included
            segments_iterator = find_segments_from_rttm(recording_id=recording_id, rttms=rttms, start_after=start, end_before=end, tolerance=0.0)
        else: # segments with seg_start > total_start and seg_end < total_end are included
            segments_iterator = rttms.find(recording_id=recording_id, start_after=start, end_before=end, adjust_offset=True) #, tolerance=0.0)
        
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
        
    if return_text:
        speaker2text = defaultdict(list)
        for seg in segments_total:
            speaker2text[seg.speaker].append(seg.text)
        texts = [' '.join(text_list) for text_list in speaker2text.values()]
        return mask, texts
    else:
        return mask



def read_seglst(seglst_filepath: str, session_id: Optional[str] = None):
    """
    Read the seglst file and return a list of segments.
    """
    with open(seglst_filepath, 'r', encoding='utf-8') as f:
        seglst = json.load(f)
        return [
            SupervisionSegment(
                id=f'{seg["session_id"]}-sup{i:05d}',
                recording_id=seg['session_id'] if session_id is None else session_id,
                start=float(seg['start_time']),
                duration=float(seg['end_time']) - float(seg['start_time']),
                text=seg['words'],
                speaker=seg['speaker']
            ) for i, seg in enumerate(seglst)
        ]
        

class MultiSpeakerMixtureGenerator():
    """
    This class is used to simulate multi-speaker audio data,
    which can be used for multi-speaker ASR and speaker diarization training.
    """
    def __init__(
        self, 
        manifest_filepath,
        rir_manifest,
        sample_rate,
        background_manifest,
        simulator_type,
        min_duration=0.1,
        max_duration=50.0,
        min_delay=0.5,
        output_dir=None,
        rir_ratio_range=None,
        rir_perturb_prob_range=None,
        rir_variance=None,
        noise_ratio_range=None,
        random_seed=42,
        session_config=None, 
        num_speakers=2,
        global_rank=0,
        world_size=1,
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
        self.random_seed = random_seed
        self.global_rank = global_rank
        self.world_size = world_size

        self.manifest_filepath = manifest_filepath 
        self.manifests = list(LazyJsonlIterator(manifest_filepath))
        self.background_manifest = background_manifest
        self.rir_manifest = rir_manifest
        self.sample_rate = sample_rate
        self.rir_manifest_list = read_rir_manifest(rir_manifest=self.rir_manifest) if self.rir_manifest else None
        self.rir_ratio_range = rir_ratio_range
        self.rir_perturb_prob_range = rir_perturb_prob_range
        self.rir_variance = rir_variance
        self.noise_ratio_range = noise_ratio_range
        
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_delay = min_delay
        self.simulator_type = simulator_type
        self.output_dir = output_dir
        self.session_config = session_config
        self.max_speakers = num_speakers

        self.global_rank_offset = 1000
        
        print("======  simulator_type", simulator_type)

        type2simulator = {
            'lsmix': self.LibriSpeechMixSimulator,
            'audiomix': self.AudioMixtureSimulator,
            'channel_separated_audio_mixer': self.ChannelSeparatedAudioMixer,
            'mixture_loader': self.MultiSpeakerMixtureLoader,
            'meeting': self.MeetingSimulator,
            'conversation': self.ConversationSimulator
        }

        self.simulator = type2simulator[simulator_type]

        if simulator_type == 'audiomix':
            self._init_audiomix_simulator()

        if simulator_type == 'lsmix':
            self.spk2manifests = groupby(lambda x: x["speaker_id"], self.manifests)
            self.speaker_ids = list(self.spk2manifests.keys())

        self.count = 0

    def _init_audiomix_simulator(self):
        """ 
        Initialize the NeMo MultiSpeakerSimulator. An instance of MultiSpeakerSimulator is created 
        under self.nemo_ms_simulator.
        """
        cfg = OmegaConf.structured(MultiSpeakerSimulatorConfig())
        cfg.data_simulator.manifest_filepath = self.manifest_filepath
        cfg.data_simulator.outputs.output_dir = self.output_dir
        cfg.data_simulator.session_config.num_sessions = self.session_config.get("num_sessions", 1)
        cfg.data_simulator.session_config.num_speakers = self.session_config.get("num_speakers", 2)
        cfg.data_simulator.session_config.session_length = self.session_config.get("session_length", 45)
        cfg.data_simulator.session_config.session_length_range = self.session_config.get("session_length_range", [10, 40])
        cfg.data_simulator.background_noise.background_manifest = self.background_manifest
        self.nemo_ms_simulator = MultiSpeakerSimulator(cfg=cfg)

    def __iter__(self):
        return self
    
    def __next__(self):
        self.count += 1
        return self.simulator()

    def _get_custom_dict(self, additional_info=None, speaker_id=None):
        custom = {
            'pnc': 'no',
            'source_lang': 'en',
            'target_lang': 'en',
            'task': 'asr',
            'speaker_id': speaker_id,
        }
        if additional_info:
            custom.update(additional_info)
        return custom

    def _save_audio_and_annotation(self, mixed_cut, uniq_id):
        # self._init_audiomix_simulator()
        # uniq_folder_name = f"{uniq_id}_test_an_rir_v6_ro{str(self.nemo_ms_simulator._params.data_simulator.session_params.random_offset)}_{self.session_config.num_speakers}spk"
        uniq_folder_name = uniq_id
        bs_true = speaker_to_target(mixed_cut, boundary_segments=True, is_audio_mix_sim=True)
        bs_false = speaker_to_target(mixed_cut, boundary_segments=False, is_audio_mix_sim=True)
        os.makedirs(f"{self.output_dir}/{uniq_folder_name}", exist_ok=True)
        save_numpy_array_as_png(bs_true, f"{self.output_dir}/{uniq_folder_name}/bs_true.png")
        save_numpy_array_as_png(bs_false, f"{self.output_dir}/{uniq_folder_name}/bs_false.png")
        ### loaded_audio = mixed_cut.load_audio().squeeze(0)
        loaded_audio = mixed_cut.load_audio().squeeze(0)
        ### Normalize the audio array

        package_path = f"{self.output_dir}/{uniq_folder_name}"
        os.makedirs(package_path, exist_ok=True)
        # sf.write(os.path.join(package_path, uniq_id + '.wav'), array, self.nemo_ms_simulator._params.data_simulator.sr)
        sf.write(os.path.join(package_path, uniq_id + '_lhotse.wav'), loaded_audio, self.sample_rate)

        self.nemo_ms_simulator.annotator.write_annotation_rttm_and_ctm(
            basepath=package_path,
            filename=uniq_id,
        )

    def _random_seed(self, session_seed):
        random.seed(session_seed)
        np.random.seed(session_seed)
        torch.manual_seed(session_seed)
        torch.cuda.manual_seed(session_seed)
        torch.cuda.manual_seed_all(session_seed)

    def _get_sample_seed(self):
        """ 
        Change the random seed for the sample.
        To avoid duplicated seeds, we multiply the global_rank_offset by the global_rank and add the count.
        """
        return self.global_rank_offset * self.global_rank + self.count
    
    def _get_noise_variations(self):
        """
        Get the perturb variations for the audio.
        Sample the noise ratio and rir ratio from the ranges.
        """
        noise_json_list = self.nemo_ms_simulator.annotator.annote_lists['noise']
        noise_ratio = random.uniform(self.noise_ratio_range[0], self.noise_ratio_range[1])
        return noise_ratio, noise_json_list

    def _get_rir_variations(self, num_speakers):
        """
        Get the perturb variations for the audio.
        """
        # Sample the noise ratio and rir ratio from the ranges
        rir_ratio = random.uniform(self.rir_ratio_range[0], self.rir_ratio_range[1])
        if self.rir_perturb_prob_range is not None:
            try:
                rir_perturb_prob = random.uniform(self.rir_perturb_prob_range[0], self.rir_perturb_prob_range[1])
            except:
                import ipdb; ipdb.set_trace()
        else:
            rir_perturb_prob = 0.0

        # load RIR
        selected_rir_paths = random.sample(self.rir_manifest_list, num_speakers)
        reverb_rirs = [Recording.from_file(rir_path['audio_filepath']) for rir_path in selected_rir_paths]
        return rir_ratio, rir_perturb_prob, reverb_rirs
    
    def _get_rir_ratio_varied(self, rir_ratio):
        """
        Get the perturbed RIR ratio to simulate the head-movement of the speakers in the room.
        """
        if self.rir_variance is not None: 
            if isinstance(self.rir_variance, float):
                rir_ratio_varied = random.uniform(rir_ratio - self.rir_variance, rir_ratio + self.rir_variance)
            else:
                raise ValueError(f"RIR variance must be a float, but got {type(self.rir_variance)}")
        else:
            rir_ratio_varied = rir_ratio
        return rir_ratio_varied

    def _set_session_length(self):
        """
        If session_length_range is set, set the session length to a random value within the range.
        """
        if self.session_config.session_length_range is not None:
            self.nemo_ms_simulator._params.data_simulator.session_config.session_length = random.randint(self.session_config.session_length_range[0], self.session_config.session_length_range[1])
        else:
            self.nemo_ms_simulator._params.data_simulator.session_config.session_length = self.session_config.session_length
        print(f"-[ Session Length Set! ]----->>> session_length: {self.nemo_ms_simulator._params.data_simulator.session_config.session_length}")

    def AudioMixtureSimulator(self):
        """
        NeMo MultiSpeaker Speech Simulator
        """
        sample_seed = self._get_sample_seed()
        self.nemo_ms_simulator.update_random_seed(sample_seed)

        self._set_session_length()
        manifest_json_list, uniq_id, mix_array, speaker_ids = self.nemo_ms_simulator.generate_single_session(random_seed=sample_seed, sess_idx=self.count)
        spk_mapping = {spk_id: spk_idx for spk_idx, spk_id in enumerate(speaker_ids)}
        uniq_id = f"audiomix_{self.count}_nspk{len(spk_mapping.keys())}"

        noise_ratio, noise_json_list = self._get_noise_variations()
        rir_ratio, rir_perturb_prob, reverb_rirs = self._get_rir_variations(self.session_config.num_speakers)

        # Speech Cuts added
        mono_cuts = []
        for manifest in manifest_json_list:
            mono_cuts.append(self.json_to_cut(json_dict=manifest))
            mono_cuts[-1].supervisions.append(
                SupervisionSegment(
                    id=uuid4(),
                    recording_id=uuid4(),
                    start=manifest['offset'],
                    duration=manifest['duration'],
                    text=' '.join(manifest['words']), # TODO: fix it for text
                    speaker=manifest['speaker_id']
                )
            )

        speaker_tracks, speaker_rir_mapping = {}, {}
        # print(f"-[ RIR IF STATEMENT ]----->>> self.rir_manifest: {self.rir_manifest}, rir_ratio: {rir_ratio}, rir_perturb_prob: {rir_perturb_prob}")
        for mono_cut in mono_cuts:
            speaker_id=mono_cut.supervisions[0].speaker
            if self.rir_manifest is None or random.random() > rir_perturb_prob:
                cln_mix_track = MixTrack(cut=deepcopy(mono_cut), 
                                    type=type(mono_cut), 
                                    offset=mono_cut.custom['mixed_cut_offset']
                                    )
                speaker_tracks.setdefault(speaker_id, []).append(cln_mix_track)
            else:
                if speaker_id not in speaker_rir_mapping:
                    speaker_rir_mapping[speaker_id] = reverb_rirs[len(speaker_rir_mapping.keys())]
                speaker_rir_recording = speaker_rir_mapping[speaker_id]
                mono_cut.custom.update(self._get_custom_dict(speaker_id=speaker_id))
                rir_ratio_varied = self._get_rir_ratio_varied(rir_ratio)
                # print(f"-[ RIR Perturbed ]----->>> rir_ratio: {rir_ratio}, rir_ratio_varied: {rir_ratio_varied}, rir_perturb_prob: {rir_perturb_prob}")
                rir_mono_cut = mono_cut.reverb_rir(speaker_rir_recording).perturb_volume(rir_ratio_varied)
                cln_mono_cut = mono_cut.perturb_volume(1-rir_ratio_varied)
                cln_mix_track = MixTrack(cut=deepcopy(cln_mono_cut), 
                                    type=type(cln_mono_cut), 
                                    offset=cln_mono_cut.custom['mixed_cut_offset']
                                    )
                rir_mix_track = MixTrack(cut=deepcopy(rir_mono_cut), 
                                    type=type(rir_mono_cut), 
                                    offset=rir_mono_cut.custom['mixed_cut_offset']
                                    )
                # clear the supervisions of the rir_mix_track
                # to avoid the rir_mix_track has the same supervisions as the cln_mix_track
                # to avoid repeated supervisions in the mixed_cut
                rir_mix_track.cut.supervisions = []
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
            noise_cut = noise_cut.perturb_volume(noise_ratio)
            noise_tracks.append(MixTrack(cut=deepcopy(noise_cut), type=type(noise_cut), offset=noise_cut.custom['mixed_cut_offset']))

        # Merge all speaker tracks into a list 
        all_spk_tracks = []
        for spk_id, tracks in speaker_tracks.items():
            all_spk_tracks.extend(tracks)

        # mixed_cut = MixedCut(id=uniq_id, tracks=all_spk_tracks) 
        mixed_cut = MixedCut(id=uniq_id, tracks=all_spk_tracks + noise_tracks)
        mixed_cut.id = uniq_id

        import ipdb; ipdb.set_trace()   
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
            mono_cuts[-1].supervisions.append(
                SupervisionSegment(
                    id=uuid4(),
                    recording_id=uuid4(),
                    start=0.0,
                    duration=mono_cuts[-1].duration,
                    text=mono_cuts[-1].custom['text'],
                    speaker=speaker_id
                )
            )
            
        tracks = []
        offset = 0.0
        for speaker_id, mono_cut in zip(sampled_speaker_ids, mono_cuts):
            tracks.append(MixTrack(cut=deepcopy(mono_cut), type=type(mono_cut), offset=offset))
            offset += random.uniform(self.min_delay, mono_cut.duration)
    
        mixed_cut = MixedCut(id='lsmix_' + '_'.join([track.cut.id for track in tracks]) + '_' + str(uuid4()), tracks=tracks)

        return mixed_cut

    def ChannelSeparatedAudioMixer(self):
        """
        This function simulates a channel-separated audio mixer.
        """

        # basic info for both channels
        manifest = random.choice(self.manifests)
        audio_filepath_list = manifest['audio_filepath']
        seglst_filepath_list = manifest['seglst_filepath']
        channel_list = manifest['channels']

        all_supervisions = []
        for i_ch in channel_list:
            seglst_filepath = seglst_filepath_list[i_ch]
            supervisions = read_seglst(seglst_filepath, session_id=manifest['session_id'])
            all_supervisions.append(supervisions)
            
        sorted_supervisions = sorted(list(itertools.chain.from_iterable(all_supervisions)), key=lambda x: x.start)
        # find the start and the end of the segment
        start_idx = random.randint(0, len(sorted_supervisions) - 1)
        end_idx = start_idx 
        offset = sorted_supervisions[start_idx].start
        for i in range(start_idx + 1, len(sorted_supervisions)):
            if sorted_supervisions[i].end - offset <= self.min_duration:
                end_idx = i
            else:
                break
        segment_offset = offset
        segment_duration = sorted_supervisions[end_idx].end - offset
        
        tracks = []
        if self.rir_manifest is not None:
            rir_ratio, rir_perturb_prob, reverb_rirs = self._get_rir_variations(num_speakers=len(channel_list))
            speaker_rir_mapping = { x: reverb_rirs[i] for i, x in enumerate(channel_list) }

        for i_ch in channel_list:
            audio_filepath = audio_filepath_list[i_ch]
            seglst_filepath = seglst_filepath_list[i_ch]
            supervisions = all_supervisions[i_ch]
            
            json_dict = {
                'audio_filepath': audio_filepath,
                'duration': segment_duration,
                'offset': segment_offset,
                'speaker_id': supervisions[0].speaker,
                'supervisions': find_segments_from_rttm(recording_id=supervisions[0].recording_id, rttms=SupervisionSet(supervisions), start_after=segment_offset, end_before=segment_offset + segment_duration, adjust_offset=False)
            }
            cut = self.json_to_cut(json_dict)
            cut.custom = self._get_custom_dict()

            # print(f"-[ RIR IF STATEMENT ]----->>> self.rir_manifest: {self.rir_manifest}, rir_ratio: {rir_ratio}, rir_perturb_prob: {rir_perturb_prob}")
            if (self.rir_manifest is None or self.rir_manifest == "") or random.random() > rir_perturb_prob:
                cln_mix_track = MixTrack(cut=deepcopy(cut), type=type(cut), offset=0)
                # speaker_tracks.setdefault(speaker_id, []).append(cln_mix_track)
                tracks.append(cln_mix_track)
            else:
                speaker_rir_recording = speaker_rir_mapping[i_ch]
                cut.custom.update(self._get_custom_dict(speaker_id=i_ch))
                rir_ratio_varied = self._get_rir_ratio_varied(rir_ratio)
                print(f"-[ RIR Perturbed ]----->>> rir_ratio: {rir_ratio}, rir_ratio_varied: {rir_ratio_varied}, rir_perturb_prob: {rir_perturb_prob}")

                rir_mono_cut = cut.reverb_rir(speaker_rir_recording).perturb_volume(rir_ratio_varied)
                cln_mono_cut = cut.perturb_volume(1-rir_ratio_varied)
                cln_mix_track = MixTrack(cut=deepcopy(cln_mono_cut), 
                                    type=type(cln_mono_cut), 
                                    offset=0,
                                    )
                rir_mix_track = MixTrack(cut=deepcopy(rir_mono_cut), 
                                    type=type(rir_mono_cut), 
                                    offset=0,
                                    )
                tracks.extend([cln_mix_track, rir_mix_track])
            
        uniq_id = 'channel_separated_audio_mixer_' + '_'.join([track.cut.id for track in tracks]) + '_' + str(uuid4())
        mixed_cut = MixedCut(
            id=uniq_id, 
            tracks=tracks, 
            # custom={
            #     'session_id': manifest['session_id'],
            # }
        )
        # self._save_audio_and_annotation(mixed_cut, uniq_id)
        return mixed_cut

    def MultiSpeakerMixtureLoader(self):

        manifest = random.choice(self.manifests)
        audio_filepath = manifest['audio_filepath']
        seglst_filepath = manifest['seglst_filepath']

        supervisions = read_seglst(seglst_filepath, session_id=manifest['session_id'])
        supervisions = sorted(supervisions, key=lambda x: x.start)

        segment_offset, segment_duration = self._get_offset_and_duration(supervisions)
        
        json_dict = {
            'audio_filepath': audio_filepath,
            'duration': segment_duration,
            'offset': segment_offset,
            'supervisions': find_segments_from_rttm(recording_id=supervisions[0].recording_id, rttms=SupervisionSet(supervisions), start_after=segment_offset, end_before=segment_offset + segment_duration, adjust_offset=False)
        }
        cut = self.json_to_cut(json_dict)

        return cut

    def MeetingSimulator(self):
        raise NotImplementedError("MeetingSimulator is not implemented yet.")   

    def ConversationSimulator(self):
        raise NotImplementedError("ConversationSimulator is not implemented yet.")

    # def _get_offset_and_duration(self, supervisions):
    #     """
    #     Get the offset and duration of the segment.
    #     supervisions should be sorted by start time
    #     """
    #     import ipdb; ipdb.set_trace()
    #     offsets, durations = self.stg_v4(supervisions)
    #     idx = random.randint(0, len(offsets) - 1)
    #     segment_offset = offsets[idx]
    #     segment_duration = durations[idx]
    #     import ipdb; ipdb.set_trace()
    #     return segment_offset, segment_duration
    def _get_offset_and_duration(self, supervisions):
        """
        Get the offset and duration of the segment.
        supervisions should be sorted by start time
        """
        non_overlap_supervisions_indices = self._get_non_overlap_supervisions_indices(supervisions)
        # find the start and the end of the segment
        start_idx = random.choice(non_overlap_supervisions_indices)
        end_idx = start_idx 
        offset = supervisions[start_idx].start
        for i in range(start_idx + 1, len(supervisions)):
            end_idx = i
            if supervisions[i].end - offset <= self.min_duration:
                pass
            else:
                if i in non_overlap_supervisions_indices:
                    break
        segment_offset = offset
        segment_duration = supervisions[end_idx].end - offset

        return segment_offset, segment_duration
    
    def _get_non_overlap_supervisions_indices(self, supervisions):
        """
        Get the indices of the non-overlapping supervisions.
        supervisions should be sorted by start time
        """
        non_overlap_supervisions_indices = []
        max_end = -1
        for i in range(len(supervisions)):
            if supervisions[i].start >= max_end:
                non_overlap_supervisions_indices.append(i)
                max_end = max(max_end, supervisions[i].end)
        return non_overlap_supervisions_indices
    
    def json_to_cut(self, json_dict, speaker_index: Union[int, None] = None):
        """
        Convert a json dictionary to a Cut instance.
        """
        audio_path = json_dict["audio_filepath"]
        duration = json_dict["duration"]
        offset = json_dict.get("offset", 0.0)
        supervisions = json_dict.get("supervisions", [])
        cut = self._create_cut(
            audio_path=audio_path, offset=offset, duration=duration, sampling_rate=json_dict.get("sampling_rate", None),
        )
        # Note that start=0 and not start=offset because supervision's start if relative to the
        # start of the cut; and cut.start is already set to offset

        if json_dict.get("text") is not None and json_dict.get("text") != "":
            cut_text = json_dict.get("text")
        else:
            cut_text = " ".join(json_dict.get("words", []))
            if cut_text == " ":
                cut_text = ""

        cut.supervisions.extend(supervisions)
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

    def stg_v4(
        self, 
        supervisions: list[SupervisionSegment], 
        round_digits: int = 3, 
        edge_ovl_thres: float = -0.5, 
        leading_word_thres: int = 5, 
        seg_hop: int = 0,
        min_gap: int = 100
    ):
        """
        Get the start and end time of the segment.
        """
        offsets, durations = [], []
        sid = 0
        cuts = []
        while sid < len(supervisions) - 1:
            use_seg_hop = True
            cut_sups = []
            text = supervisions[sid].text
            
            # skip case 1: the overlap duration at the beginning is larger than the threshold
            if sid and supervisions[sid-1].end - supervisions[sid].start > edge_ovl_thres: # checking overlap duration at the beginning
                sid += 1
                continue
            # skip case 2: the first segment is too short, will also remove that later in _process function
            words = text.split()
            if len(words) <= leading_word_thres and supervisions[sid].speaker != supervisions[sid+1].speaker:
                sid += 1
                continue

            # skip case 3: the start time of the first segment is exactly the same as that of previous segment
            if sid and supervisions[sid-1].start == supervisions[sid].start:
                sid += 1
                continue
            
            starts, ends = [], []
            for eid in range(sid, len(supervisions)):
                if text: 
                    starts.append(supervisions[eid].start)
                    ends.append(supervisions[eid].end)
                    cut_sups.append(supervisions[eid])
                else: # if words is empty, this means that there is a mismatch between rttm and ctm, and we will skip this segment
                    break

                # large gap between two segments
                if eid < len(supervisions) - 1 and supervisions[eid+1].start - max(ends) > min_gap:                
                    break

                if max(ends) - min(starts) > self.min_duration: 

                    # check if all segments are included, this is the rare case but will cause the disagreement between rttm and lhotse
                    for eid_sub in range(eid+1, len(supervisions)):
                        if supervisions[eid_sub].end <= max(ends):
                            starts.append(supervisions[eid_sub].start)
                            ends.append(supervisions[eid_sub].end)
                            cut_sups.append(supervisions[eid_sub])
                            eid = eid_sub
                            use_seg_hop = False
                        elif supervisions[eid_sub].start >= max(ends):
                            break

                        if max(ends) - min(starts) > self.max_duration:
                            break

                    if not use_seg_hop:
                        sid = eid_sub
                    break
                    
            # update the start index of the next segment
            if use_seg_hop:
                if seg_hop == 0: 
                    sid = eid + 1
                elif seg_hop > 0:
                    sid = sid + seg_hop
                else:
                    raise ValueError(f"args.seg_hop should be greater than or equal to 0 but got {seg_hop}")
                
            # skip case 4: the overlap duration at the end is larger than the threshold
            if eid < len(supervisions) - 1 and max(ends) - supervisions[eid+1].start > edge_ovl_thres: # checking overlap duration at the end
                continue

            # skip case 5: two segments start at the same time
            if len(starts) > 1 and starts[0] == starts[1]:
                continue

            # skip case 6: too short or too long segment
            start, end = min(starts), max(ends)
            dur=round(end - start, round_digits)
            if dur < self.min_duration or dur > self.max_duration:
                continue

            offsets.append(start)
            durations.append(dur)

            # cut_sups = SupervisionSet.from_segments(cut_sups)
            # cut = MonoCut(
            #     id=f'{recording_id}-{int(start*10**round_digits):09d}-{int(end*10**round_digits):09d}',
            #     start=start,
            #     duration=dur,
            #     channel=0,
            #     supervisions=cut_sups
            # )
            # cuts.append(cut)

        return offsets, durations