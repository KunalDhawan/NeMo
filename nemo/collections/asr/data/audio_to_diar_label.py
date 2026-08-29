# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import io
import importlib
import os
import random
from collections import OrderedDict
from statistics import mode
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from nemo.collections.asr.parts.utils.offline_clustering import get_argmin_mat
from nemo.collections.asr.parts.utils.speaker_utils import convert_rttm_line, get_subsegments, prepare_split_data
from nemo.collections.common.parts.preprocessing.collections import (
    DiarizationSpeechLabel,
    EndtoEndDiarizationSpeechLabel,
)
from nemo.core.classes import Dataset
from nemo.core.neural_types import (
    AudioSignal,
    EncodedRepresentation,
    LengthsType,
    NeuralType,
    ProbsType,
    SpectrogramType,
)
from nemo.utils import logging


def get_scale_mapping_list(uniq_timestamps):
    """
    Call get_argmin_mat function to find the index of the non-base-scale segment that is closest to the
    given base-scale segment. For each scale and each segment, a base-scale segment is assigned.

    Args:
        uniq_timestamps: (dict)
            The dictionary containing embeddings, timestamps and multiscale weights.
            If uniq_timestamps contains only one scale, single scale diarization is performed.

    Returns:
        scale_mapping_argmat (torch.tensor):

            The element at the m-th row and the n-th column of the scale mapping matrix indicates the (m+1)-th scale
            segment index which has the closest center distance with (n+1)-th segment in the base scale.

            - Example:
                `scale_mapping_argmat[2][101] = 85`

            In the above example, the code snippet means that 86-th segment in the 3rd scale (python index is 2) is
            mapped to the 102-th segment in the base scale. Thus, the longer segments bound to have more repeating
            numbers since multiple base scale segments (since the base scale has the shortest length) fall into the
            range of the longer segments. At the same time, each row contains N numbers of indices where N is number
            of segments in the base-scale (i.e., the finest scale).
    """
    timestamps_in_scales = []
    for key, val in uniq_timestamps['scale_dict'].items():
        timestamps_in_scales.append(torch.tensor(val['time_stamps']))
    session_scale_mapping_list = get_argmin_mat(timestamps_in_scales)
    scale_mapping_argmat = [[] for _ in range(len(uniq_timestamps['scale_dict'].keys()))]
    for scale_idx in range(len(session_scale_mapping_list)):
        scale_mapping_argmat[scale_idx] = session_scale_mapping_list[scale_idx]
    scale_mapping_argmat = torch.stack(scale_mapping_argmat)
    return scale_mapping_argmat


def extract_seg_info_from_rttm(rttm_lines, mapping_dict=None, target_spks=None):
    """
    Get RTTM lines containing speaker labels, start time and end time. target_spks contains two targeted
    speaker indices for creating groundtruth label files. Only speakers in target_spks variable will be
    included in the output lists.

    Args:
        uniq_id (str):
            Unique file ID that refers to an input audio file and corresponding RTTM (Annotation) file.
        rttm_lines (list):
            List containing RTTM lines in str format.
        mapping_dict (dict):
            Mapping between the estimated speakers and the speakers in the ground-truth annotation.
            `mapping_dict` variable is only provided when the inference mode is running in sequence-eval mode.
            Sequence eval mode uses the mapping between the estimated speakers and the speakers
            in ground-truth annotation.
    Returns:
        rttm_tup (tuple):
            Tuple containing lists of start time, end time and speaker labels.

    """
    stt_list, end_list, speaker_list, pairwise_infer_spks = [], [], [], []
    if target_spks:
        inv_map = {v: k for k, v in mapping_dict.items()}
        for spk_idx in target_spks:
            spk_str = f'speaker_{spk_idx}'
            if spk_str in inv_map:
                pairwise_infer_spks.append(inv_map[spk_str])
    for rttm_line in rttm_lines:
        start, end, speaker = convert_rttm_line(rttm_line)
        if target_spks is None or speaker in pairwise_infer_spks:
            end_list.append(end)
            stt_list.append(start)
            speaker_list.append(speaker)
    rttm_tup = (stt_list, end_list, speaker_list)
    return rttm_tup


def assign_frame_level_spk_vector(rttm_timestamps, round_digits, frame_per_sec, target_spks, min_spks=2):
    """
    Create a multi-dimensional vector sequence containing speaker timestamp information in RTTM.
    The unit-length is the frame shift length of the acoustic feature. The feature-level annotations
    `fr_level_target` will later be converted to base-segment level diarization label.

    Args:
        rttm_timestamps (list):
            List containing start and end time for each speaker segment label.
            `stt_list`, `end_list` and `speaker_list` are contained.
        frame_per_sec (int):
            Number of feature frames per second. This quantity is determined by
            `window_stride` variable in preprocessing module.
        target_spks (tuple):
            Speaker indices that are generated from combinations.
            If there are only one or two speakers,
            only a single `target_spks` variable is generated.

    Returns:
        fr_level_target (torch.tensor):
            Tensor containing label for each feature level frame.
    """
    stt_list, end_list, speaker_list = rttm_timestamps
    if len(speaker_list) == 0:
        return None
    else:
        sorted_speakers = sorted(list(set(speaker_list)))
        total_fr_len = int(max(end_list) * (10**round_digits))
        spk_num = max(len(sorted_speakers), min_spks)
        speaker_mapping_dict = {rttm_key: x_int for x_int, rttm_key in enumerate(sorted_speakers)}
        fr_level_target = torch.zeros(total_fr_len, spk_num)

        # If RTTM is not provided, then there is no speaker mapping dict in target_spks.
        # Thus, return a zero-filled tensor as a placeholder.
        for count, (stt, end, spk_rttm_key) in enumerate(zip(stt_list, end_list, speaker_list)):
            stt, end = round(stt, round_digits), round(end, round_digits)
            spk = speaker_mapping_dict[spk_rttm_key]
            stt_fr, end_fr = int(round(stt, 2) * frame_per_sec), int(round(end, round_digits) * frame_per_sec)
            fr_level_target[stt_fr:end_fr, spk] = 1
        return fr_level_target


def get_subsegments_to_timestamps(
    subsegments: List[Tuple[float, float]], feat_per_sec: int = 100, max_end_ts: float = None, decimals=2
):
    """
    Convert subsegment timestamps to scale timestamps by multiplying with the feature rate (`feat_per_sec`)
    and rounding. Segment is consisted of many subsegments and sugsegments are equivalent to `frames`
    in end-to-end speaker diarization models.

    Args:
        subsegments (List[Tuple[float, float]]):
            A list of tuples where each tuple contains the start and end times of a subsegment
            (frames in end-to-end models).
            >>> subsegments = [[t0_start, t0_duration], [t1_start, t1_duration],..., [tN_start, tN_duration]]
        feat_per_sec (int, optional):
            The number of feature frames per second. Defaults to 100.
        max_end_ts (float, optional):
            The maximum end timestamp to clip the results. If None, no clipping is applied. Defaults to None.
        decimals (int, optional):
            The number of decimal places to round the timestamps. Defaults to 2.

    Example:
        Segments starting from 0.0 and ending at 69.2 seconds.
        If hop-length is 0.08 and the subsegment (frame) length is 0.16 seconds,
        there are 864 = (69.2 - 0.16)/0.08 + 1 subsegments (frames in end-to-end models) in this segment.
        >>> subsegments = [[[0.0, 0.16], [0.08, 0.16], ..., [69.04, 0.16], [69.12, 0.08]]

    Returns:
        ts (torch.tensor):
            A tensor containing the scaled and rounded timestamps for each subsegment.
    """
    if len(subsegments) == 0:
        return torch.zeros((0, 2), dtype=torch.long)
    seg_ts = (torch.tensor(subsegments) * feat_per_sec).float()
    ts_round = torch.round(seg_ts, decimals=decimals)
    ts = ts_round.long()
    ts[:, 1] = ts[:, 0] + ts[:, 1]
    if max_end_ts is not None:
        ts = np.clip(ts, 0, int(max_end_ts * feat_per_sec))
    return ts


def extract_frame_info_from_rttm(offset, duration, rttm_lines, round_digits=3):
    """
    Extracts RTTM lines containing speaker labels, start time, and end time for a given audio segment.

    Args:
        uniq_id (str): Unique identifier for the audio file and corresponding RTTM file.
        offset (float): The starting time offset for the segment of interest.
        duration (float): The duration of the segment of interest.
        rttm_lines (list): List of RTTM lines in string format.
        round_digits (int, optional): Number of decimal places to round the start and end times. Defaults to 3.

    Returns:
        rttm_mat (tuple): A tuple containing lists of start times, end times, and speaker labels.
        sess_to_global_spkids (dict): A mapping from session-specific speaker indices to global speaker identifiers.
    """
    rttm_stt, rttm_end = offset, offset + duration
    stt_list, end_list, speaker_list, speaker_set = [], [], [], []
    sess_to_global_spkids = dict()

    for rttm_line in rttm_lines:
        start, end, speaker = convert_rttm_line(rttm_line)

        # Skip invalid RTTM lines where the start time is greater than the end time.
        if start > end:
            continue

        # Check if the RTTM segment overlaps with the specified segment of interest.
        if (end > rttm_stt and start < rttm_end) or (start < rttm_end and end > rttm_stt):
            # Adjust the start and end times to fit within the segment of interest.
            start, end = max(start, rttm_stt), min(end, rttm_end)
        else:
            continue

        # Round the start and end times to the specified number of decimal places.
        end_list.append(round(end, round_digits))
        stt_list.append(round(start, round_digits))

        # Assign a unique index to each speaker and maintain a mapping.
        if speaker not in speaker_set:
            speaker_set.append(speaker)
        speaker_list.append(speaker_set.index(speaker))
        sess_to_global_spkids.update({speaker_set.index(speaker): speaker})

    rttm_mat = (stt_list, end_list, speaker_list)
    return rttm_mat, sess_to_global_spkids


def get_frame_targets_from_rttm(
    rttm_timestamps: list,
    offset: float,
    duration: float,
    round_digits: int,
    feat_per_sec: int,
    max_spks: int,
):
    """
    Create a multi-dimensional vector sequence containing speaker timestamp information in RTTM.
    The unit-length is the frame shift length of the acoustic feature. The feature-level annotations
    `feat_level_target` will later be converted to base-segment level diarization label.

    Args:
        rttm_timestamps (list):
            List containing start and end time for each speaker segment label.
            stt_list, end_list and speaker_list are contained.
        feat_per_sec (int):
            Number of feature frames per second.
            This quantity is determined by window_stride variable in preprocessing module.
        target_spks (tuple):
            Speaker indices that are generated from combinations. If there are only one or two speakers,
            only a single target_spks variable is generated.

    Returns:
        feat_level_target (torch.tensor):
            Tensor containing label for each feature level frame.
    """
    stt_list, end_list, speaker_list = rttm_timestamps
    sorted_speakers = sorted(list(set(speaker_list)))
    total_fr_len = int(duration * feat_per_sec)
    if len(sorted_speakers) > max_spks:
        logging.warning(
            f"Number of speakers in RTTM file {len(sorted_speakers)} exceeds the maximum number of speakers: "
            f"{max_spks}! Only {max_spks} first speakers remain, and this will affect frame metrics!"
        )
    feat_level_target = torch.zeros(total_fr_len, max_spks)
    for count, (stt, end, spk_rttm_key) in enumerate(zip(stt_list, end_list, speaker_list)):
        if end < offset or stt > offset + duration:
            continue
        stt, end = max(offset, stt), min(offset + duration, end)
        spk = spk_rttm_key
        if spk < max_spks:
            stt_fr, end_fr = int((stt - offset) * feat_per_sec), int((end - offset) * feat_per_sec)
            feat_level_target[stt_fr:end_fr, spk] = 1
    return feat_level_target


class _AudioMSDDTrainDataset(Dataset):
    """
    Dataset class that loads a json file containing paths to audio files,
    RTTM files and number of speakers. This Dataset class is designed for
    training or fine-tuning speaker embedding extractor and diarization decoder
    at the same time.

    Example:
    {"audio_filepath": "/path/to/audio_0.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_0.rttm}
    ...
    {"audio_filepath": "/path/to/audio_n.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_n.rttm}

    Args:
        manifest_filepath (str):
            Path to input manifest json files.
        multiscale_args_dict (dict):
            Dictionary containing the parameters for multiscale segmentation and clustering.
        emb_dir (str):
            Path to a temporary folder where segmentation information for embedding extraction is saved.
        soft_label_thres (float):
            Threshold that determines the label of each segment based on RTTM file information.
        featurizer:
            Featurizer instance for generating features from the raw waveform.
        window_stride (float):
            Window stride for acoustic feature. This value is used for calculating the numbers of feature-level frames.
        validate_manifest_paths (bool):
            If True, verify that each unique audio and RTTM path in the manifest exists.
        emb_batch_size (int):
            Number of embedding vectors that are trained with attached computational graphs.
        pairwise_infer (bool):
            This variable should be True if dataloader is created for an inference task.
        random_flip (bool):
            If True, the two labels and input signals are randomly flipped per every epoch while training.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        output_types = {
            "features": NeuralType(('B', 'T'), AudioSignal()),
            "feature_length": NeuralType(('B'), LengthsType()),
            "ms_seg_timestamps": NeuralType(('B', 'C', 'T', 'D'), LengthsType()),
            "ms_seg_counts": NeuralType(('B', 'C'), LengthsType()),
            "clus_label_index": NeuralType(('B', 'T'), LengthsType()),
            "scale_mapping": NeuralType(('B', 'C', 'T'), LengthsType()),
            "targets": NeuralType(('B', 'T', 'C'), ProbsType()),
        }

        return output_types

    def __init__(
        self,
        *,
        manifest_filepath: str,
        multiscale_args_dict: str,
        emb_dir: str,
        soft_label_thres: float,
        featurizer,
        window_stride,
        emb_batch_size,
        pairwise_infer: bool,
        random_flip: bool = True,
        global_rank: int = 0,
    ):
        super().__init__()
        self.collection = DiarizationSpeechLabel(
            manifests_files=manifest_filepath.split(','),
            emb_dict=None,
            clus_label_dict=None,
            pairwise_infer=pairwise_infer,
        )
        self.featurizer = featurizer
        self.multiscale_args_dict = multiscale_args_dict
        self.emb_dir = emb_dir
        self.round_digits = 2
        self.decim = 10**self.round_digits
        self.soft_label_thres = soft_label_thres
        self.pairwise_infer = pairwise_infer
        self.max_spks = 2
        self.frame_per_sec = int(1 / window_stride)
        self.emb_batch_size = emb_batch_size
        self.random_flip = random_flip
        self.global_rank = global_rank
        self.manifest_filepath = manifest_filepath
        self.multiscale_timestamp_dict = prepare_split_data(
            self.manifest_filepath,
            self.emb_dir,
            self.multiscale_args_dict,
            self.global_rank,
        )

    def __len__(self):
        return len(self.collection)

    def assign_labels_to_longer_segs(self, uniq_id, base_scale_clus_label):
        """
        Assign the generated speaker labels from the base scale (the finest scale) to the longer scales.
        This process is needed to get the cluster labels for each scale. The cluster labels are needed to
        calculate the cluster-average speaker embedding for each scale.

        Args:
            uniq_id (str):
                Unique sample ID for training.
            base_scale_clus_label (torch.tensor):
                Tensor variable containing the speaker labels for the base-scale segments.

        Returns:
            per_scale_clus_label (torch.tensor):
                Tensor variable containing the speaker labels for each segment in each scale.
                Note that the total length of the speaker label sequence differs over scale since
                each scale has a different number of segments for the same session.

            scale_mapping (torch.tensor):
                Matrix containing the segment indices of each scale. scale_mapping is necessary for reshaping the
                multiscale embeddings to form an input matrix for the MSDD model.
        """
        per_scale_clus_label = []
        self.scale_n = len(self.multiscale_timestamp_dict[uniq_id]['scale_dict'])
        uniq_scale_mapping = get_scale_mapping_list(self.multiscale_timestamp_dict[uniq_id])
        for scale_index in range(self.scale_n):
            new_clus_label = []
            scale_seq_len = len(self.multiscale_timestamp_dict[uniq_id]["scale_dict"][scale_index]["time_stamps"])
            for seg_idx in range(scale_seq_len):
                if seg_idx in uniq_scale_mapping[scale_index]:
                    seg_clus_label = mode(base_scale_clus_label[uniq_scale_mapping[scale_index] == seg_idx])
                else:
                    seg_clus_label = 0 if len(new_clus_label) == 0 else new_clus_label[-1]
                new_clus_label.append(seg_clus_label)
            per_scale_clus_label.extend(new_clus_label)
        per_scale_clus_label = torch.tensor(per_scale_clus_label)
        return per_scale_clus_label, uniq_scale_mapping

    def get_diar_target_labels(self, uniq_id, sample, fr_level_target):
        """
        Convert frame-level diarization target variable into segment-level target variable.
        Since the granularity is reduced from frame level (10ms) to segment level (100ms~500ms),
        we need a threshold value, `soft_label_thres`, which determines the label of each segment
        based on the overlap between a segment range (start and end time) and the frame-level target variable.

        Args:
            uniq_id (str):
                Unique file ID that refers to an input audio file and corresponding RTTM (Annotation) file.
            sample:
                `DiarizationSpeechLabel` instance containing sample information such as
                audio filepath and RTTM filepath.
            fr_level_target (torch.tensor):
                Tensor containing label for each feature-level frame.

        Returns:
            seg_target (torch.tensor):
                Tensor containing binary speaker labels for base-scale segments.
            base_clus_label (torch.tensor):
                Representative speaker label for each segment. This variable only has one speaker label
                for each base-scale segment.
                -1 means that there is no corresponding speaker in the target_spks tuple.
        """
        seg_target_list, base_clus_label = [], []
        self.scale_n = len(self.multiscale_timestamp_dict[uniq_id]['scale_dict'])
        subseg_time_stamp_list = self.multiscale_timestamp_dict[uniq_id]["scale_dict"][self.scale_n - 1]["time_stamps"]
        for seg_stt, seg_end in subseg_time_stamp_list:
            seg_stt_fr, seg_end_fr = int(seg_stt * self.frame_per_sec), int(seg_end * self.frame_per_sec)
            soft_label_vec_sess = torch.sum(fr_level_target[seg_stt_fr:seg_end_fr, :], axis=0) / (
                seg_end_fr - seg_stt_fr
            )
            label_int_sess = torch.argmax(soft_label_vec_sess)
            soft_label_vec = soft_label_vec_sess.unsqueeze(0)[:, sample.target_spks].squeeze()
            if label_int_sess in sample.target_spks and torch.sum(soft_label_vec_sess) > 0:
                label_int = sample.target_spks.index(label_int_sess)
            else:
                label_int = -1
            label_vec = (soft_label_vec > self.soft_label_thres).float()
            seg_target_list.append(label_vec.detach())
            base_clus_label.append(label_int)
        seg_target = torch.stack(seg_target_list)
        base_clus_label = torch.tensor(base_clus_label)
        return seg_target, base_clus_label

    def parse_rttm_for_ms_targets(self, sample):
        """
        Generate target tensor variable by extracting groundtruth diarization labels from an RTTM file.
        This function converts (start, end, speaker_id) format into base-scale (the finest scale) segment level
        diarization label in a matrix form.

        Example of seg_target:
            [[0., 1.], [0., 1.], [1., 1.], [1., 0.], [1., 0.], ..., [0., 1.]]

        Args:
            sample:
                `DiarizationSpeechLabel` instance containing sample information such as
                audio filepath and RTTM filepath.
            target_spks (tuple):
                Speaker indices that are generated from combinations. If there are only one or two speakers,
                only a single target_spks tuple is generated.

        Returns:
            clus_label_index (torch.tensor):
                Groundtruth clustering label (cluster index for each segment) from RTTM files for training purpose.
            seg_target  (torch.tensor):
                Tensor variable containing hard-labels of speaker activity in each base-scale segment.
            scale_mapping (torch.tensor):
                Matrix containing the segment indices of each scale. scale_mapping is necessary for reshaping the
                multiscale embeddings to form an input matrix for the MSDD model.

        """
        with open(sample.rttm_file, 'r') as file:
            rttm_lines = file.readlines()
        uniq_id = self.get_uniq_id_with_range(sample)
        rttm_timestamps = extract_seg_info_from_rttm(rttm_lines)
        fr_level_target = assign_frame_level_spk_vector(
            rttm_timestamps, self.round_digits, self.frame_per_sec, target_spks=sample.target_spks
        )
        seg_target, base_clus_label = self.get_diar_target_labels(uniq_id, sample, fr_level_target)
        clus_label_index, scale_mapping = self.assign_labels_to_longer_segs(uniq_id, base_clus_label)
        return clus_label_index, seg_target, scale_mapping

    def get_uniq_id_with_range(self, sample, deci=3):
        """
        Generate unique training sample ID from unique file ID, offset and duration. The start-end time added
        unique ID is required for identifying the sample since multiple short audio samples are generated from a single
        audio file. The start time and end time of the audio stream uses millisecond units if `deci=3`.

        Args:
            sample:
                `DiarizationSpeechLabel` instance from collections.

        Returns:
            uniq_id (str):
                Unique sample ID which includes start and end time of the audio stream.
                Example: abc1001_3122_6458

        """
        bare_uniq_id = os.path.splitext(os.path.basename(sample.rttm_file))[0]
        offset = str(int(round(sample.offset, deci) * pow(10, deci)))
        endtime = str(int(round(sample.offset + sample.duration, deci) * pow(10, deci)))
        uniq_id = f"{bare_uniq_id}_{offset}_{endtime}"
        return uniq_id

    def get_ms_seg_timestamps(self, sample):
        """
        Get start and end time of each diarization frame.

        Args:
            sample:
                `DiarizationSpeechLabel` instance from preprocessing.collections
        Returns:
            ms_seg_timestamps (torch.tensor):
                Tensor containing timestamps for each frame.
            ms_seg_counts (torch.tensor):
                Number of segments for each scale. This information is used for reshaping embedding batch
                during forward propagation.
        """
        uniq_id = self.get_uniq_id_with_range(sample)
        ms_seg_timestamps_list = []
        max_seq_len = len(self.multiscale_timestamp_dict[uniq_id]["scale_dict"][self.scale_n - 1]["time_stamps"])
        ms_seg_counts = [0 for _ in range(self.scale_n)]
        for scale_idx in range(self.scale_n):
            scale_ts_list = []
            for k, (seg_stt, seg_end) in enumerate(
                self.multiscale_timestamp_dict[uniq_id]["scale_dict"][scale_idx]["time_stamps"]
            ):
                stt, end = (
                    int((seg_stt - sample.offset) * self.frame_per_sec),
                    int((seg_end - sample.offset) * self.frame_per_sec),
                )
                scale_ts_list.append(torch.tensor([stt, end]).detach())
            ms_seg_counts[scale_idx] = len(
                self.multiscale_timestamp_dict[uniq_id]["scale_dict"][scale_idx]["time_stamps"]
            )
            scale_ts = torch.stack(scale_ts_list)
            scale_ts_padded = torch.cat([scale_ts, torch.zeros(max_seq_len - len(scale_ts_list), 2)], dim=0)
            ms_seg_timestamps_list.append(scale_ts_padded.detach())
        ms_seg_timestamps = torch.stack(ms_seg_timestamps_list)
        ms_seg_counts = torch.tensor(ms_seg_counts)
        return ms_seg_timestamps, ms_seg_counts

    def __getitem__(self, index):
        sample = self.collection[index]
        if sample.offset is None:
            sample.offset = 0
        clus_label_index, targets, scale_mapping = self.parse_rttm_for_ms_targets(sample)
        features = self.featurizer.process(sample.audio_file, offset=sample.offset, duration=sample.duration)
        feature_length = torch.tensor(features.shape[0]).long()
        ms_seg_timestamps, ms_seg_counts = self.get_ms_seg_timestamps(sample)
        if self.random_flip:
            torch.manual_seed(index)
            flip = torch.cat([torch.randperm(self.max_spks), torch.tensor(-1).unsqueeze(0)])
            clus_label_index, targets = flip[clus_label_index], targets[:, flip[: self.max_spks]]
        return features, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets


class _AudioMSDDInferDataset(Dataset):
    """
    Dataset class that loads a json file containing paths to audio files,
    RTTM files and number of speakers. This Dataset class is built for diarization inference and
    evaluation. Speaker embedding sequences, segment timestamps, cluster-average speaker embeddings
    are loaded from memory and fed into the dataloader.

    Example:
    {"audio_filepath": "/path/to/audio_0.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_0.rttm}
    ...
    {"audio_filepath": "/path/to/audio_n.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_n.rttm}

    Args:
        manifest_filepath (str):
             Path to input manifest json files.
        emb_dict (dict):
            Dictionary containing cluster-average embeddings and speaker mapping information.
        emb_seq (dict):
            Dictionary containing multiscale speaker embedding sequence,
            scale mapping and corresponding segment timestamps.
        clus_label_dict (dict):
            Subsegment-level (from base-scale) speaker labels from clustering results.
        soft_label_thres (float):
            A threshold that determines the label of each segment based on RTTM file information.
        featurizer:
            Featurizer instance for generating features from raw waveform.
        seq_eval_mode (bool):
            If True, F1 score will be calculated for each speaker pair during inference mode.
        window_stride (float):
            Window stride for acoustic feature. This value is used for calculating the numbers of feature-level frames.
        use_single_scale_clus (bool):
            Use only one scale for clustering instead of using multiple scales of embeddings for clustering.
        pairwise_infer (bool):
            This variable should be True if dataloader is created for an inference task.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        output_types = OrderedDict(
            {
                "ms_emb_seq": NeuralType(('B', 'T', 'C', 'D'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
                "ms_avg_embs": NeuralType(('B', 'C', 'D', 'C'), EncodedRepresentation()),
                "targets": NeuralType(('B', 'T', 'C'), ProbsType()),
            }
        )
        return output_types

    def __init__(
        self,
        *,
        manifest_filepath: str,
        emb_dict: Dict,
        emb_seq: Dict,
        clus_label_dict: Dict,
        soft_label_thres: float,
        seq_eval_mode: bool,
        window_stride: float,
        use_single_scale_clus: bool,
        pairwise_infer: bool,
    ):
        super().__init__()
        self.collection = DiarizationSpeechLabel(
            manifests_files=manifest_filepath.split(','),
            emb_dict=emb_dict,
            clus_label_dict=clus_label_dict,
            seq_eval_mode=seq_eval_mode,
            pairwise_infer=pairwise_infer,
        )
        self.emb_dict = emb_dict
        self.emb_seq = emb_seq
        self.clus_label_dict = clus_label_dict
        self.round_digits = 2
        self.decim = 10**self.round_digits
        self.frame_per_sec = int(1 / window_stride)
        self.soft_label_thres = soft_label_thres
        self.pairwise_infer = pairwise_infer
        self.max_spks = 2
        self.use_single_scale_clus = use_single_scale_clus
        self.seq_eval_mode = seq_eval_mode

    def __len__(self):
        return len(self.collection)

    def parse_rttm_multiscale(self, sample):
        """
        Generate target tensor variable by extracting groundtruth diarization labels from an RTTM file.
        This function is only used when ``self.seq_eval_mode=True`` and RTTM files are provided. This function converts
        (start, end, speaker_id) format into base-scale (the finest scale) segment level diarization label in a matrix
        form to create target matrix.

        Args:
            sample:
                DiarizationSpeechLabel instance containing sample information such as audio filepath and RTTM filepath.
            target_spks (tuple):
                Two Indices of targeted speakers for evaluation.
                Example of target_spks: (2, 3)
        Returns:
            seg_target (torch.tensor):
                Tensor variable containing hard-labels of speaker activity in each base-scale segment.
        """
        if sample.rttm_file is None:
            raise ValueError(f"RTTM file is not provided for this sample {sample}")
        rttm_lines = open(sample.rttm_file).readlines()
        uniq_id = os.path.splitext(os.path.basename(sample.rttm_file))[0]
        mapping_dict = self.emb_dict[max(self.emb_dict.keys())][uniq_id]['mapping']
        rttm_timestamps = extract_seg_info_from_rttm(rttm_lines, mapping_dict, sample.target_spks)
        fr_level_target = assign_frame_level_spk_vector(
            rttm_timestamps, self.round_digits, self.frame_per_sec, sample.target_spks
        )
        seg_target = self.get_diar_target_labels_from_fr_target(uniq_id, fr_level_target)
        return seg_target

    def get_diar_target_labels_from_fr_target(self, uniq_id: str, fr_level_target: torch.Tensor) -> torch.Tensor:
        """
        Generate base-scale level binary diarization label from frame-level target matrix. For the given frame-level
        speaker target matrix fr_level_target, we count the number of frames that belong to each speaker and calculate
        ratios for each speaker into the `soft_label_vec` variable. Finally, `soft_label_vec` variable is compared
        with `soft_label_thres` to determine whether a label vector should contain 0 or 1 for each speaker bin.
        Note that seg_target variable has dimension of (number of base-scale segments x 2) dimension.

        Example of seg_target:
            [[0., 1.], [0., 1.], [1., 1.], [1., 0.], [1., 0.], ..., [0., 1.]]

        Args:
            uniq_id (str):
                Unique file ID that refers to an input audio file and corresponding RTTM (Annotation) file.
            fr_level_target (torch.tensor):
                frame-level binary speaker annotation (1: exist 0: non-exist) generated from RTTM file.

        Returns:
            seg_target (torch.tensor):
                Tensor variable containing binary hard-labels of speaker activity in each base-scale segment.

        """
        if fr_level_target is None:
            return None
        else:
            seg_target_list = []
            for seg_stt, seg_end, label_int in self.clus_label_dict[uniq_id]:
                seg_stt_fr, seg_end_fr = int(seg_stt * self.frame_per_sec), int(seg_end * self.frame_per_sec)
                soft_label_vec = torch.sum(fr_level_target[seg_stt_fr:seg_end_fr, :], axis=0) / (
                    seg_end_fr - seg_stt_fr
                )
                label_vec = (soft_label_vec > self.soft_label_thres).int()
                seg_target_list.append(label_vec)
            seg_target = torch.stack(seg_target_list)
            return seg_target

    def __getitem__(self, index):
        sample = self.collection[index]
        if sample.offset is None:
            sample.offset = 0

        uniq_id = os.path.splitext(os.path.basename(sample.audio_file))[0]
        scale_n = len(self.emb_dict.keys())
        _avg_embs = torch.stack([self.emb_dict[scale_index][uniq_id]['avg_embs'] for scale_index in range(scale_n)])

        if self.pairwise_infer:
            avg_embs = _avg_embs[:, :, self.collection[index].target_spks]
        else:
            avg_embs = _avg_embs

        if avg_embs.shape[2] > self.max_spks:
            raise ValueError(
                f" avg_embs.shape[2] {avg_embs.shape[2]} should be less than or equal to "
                f"self.max_num_speakers {self.max_spks}"
            )

        feats = []
        for scale_index in range(scale_n):
            repeat_mat = self.emb_seq["session_scale_mapping"][uniq_id][scale_index]
            feats.append(self.emb_seq[scale_index][uniq_id][repeat_mat, :])
        feats_out = torch.stack(feats).permute(1, 0, 2)
        feats_len = feats_out.shape[0]

        if self.seq_eval_mode:
            targets = self.parse_rttm_multiscale(sample)
        else:
            targets = torch.zeros(feats_len, 2).float()

        return feats_out, feats_len, targets, avg_embs


def _msdd_train_collate_fn(self, batch):
    """
    Collate batch of variables that are needed for raw waveform to diarization label training.
    The following variables are included in training/validation batch:

    Args:
        batch (tuple):
            Batch tuple containing the variables for the diarization training.
    Returns:
        features (torch.tensor):
            Raw waveform samples (time series) loaded from the audio_filepath in the input manifest file.
        feature lengths (time series sample length):
            A list of lengths of the raw waveform samples.
        ms_seg_timestamps (torch.tensor):
            Matrix containing the start time and end time (timestamps) for each segment and each scale.
            ms_seg_timestamps is needed for extracting acoustic features from raw waveforms.
        ms_seg_counts (torch.tensor):
            Matrix containing The number of segments for each scale. ms_seg_counts is necessary for reshaping
            the input matrix for the MSDD model.
        clus_label_index (torch.tensor):
            Groundtruth Clustering label (cluster index for each segment) from RTTM files for training purpose.
            clus_label_index is necessary for calculating cluster-average embedding.
        scale_mapping (torch.tensor):
            Matrix containing the segment indices of each scale. scale_mapping is necessary for reshaping the
            multiscale embeddings to form an input matrix for the MSDD model.
        targets (torch.tensor):
            Groundtruth Speaker label for the given input embedding sequence.
    """
    packed_batch = list(zip(*batch))
    features, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets = packed_batch
    features_list, feature_length_list = [], []
    ms_seg_timestamps_list, ms_seg_counts_list, scale_clus_label_list, scale_mapping_list, targets_list = (
        [],
        [],
        [],
        [],
        [],
    )

    max_raw_feat_len = max([x.shape[0] for x in features])
    max_target_len = max([x.shape[0] for x in targets])
    max_total_seg_len = max([x.shape[0] for x in clus_label_index])

    for feat, feat_len, ms_seg_ts, ms_seg_ct, scale_clus, scl_map, tgt in batch:
        seq_len = tgt.shape[0]
        pad_feat = (0, max_raw_feat_len - feat_len)
        pad_tgt = (0, 0, 0, max_target_len - seq_len)
        pad_sm = (0, max_target_len - seq_len)
        pad_ts = (0, 0, 0, max_target_len - seq_len)
        pad_sc = (0, max_total_seg_len - scale_clus.shape[0])
        padded_feat = torch.nn.functional.pad(feat, pad_feat)
        padded_tgt = torch.nn.functional.pad(tgt, pad_tgt)
        padded_sm = torch.nn.functional.pad(scl_map, pad_sm)
        padded_ms_seg_ts = torch.nn.functional.pad(ms_seg_ts, pad_ts)
        padded_scale_clus = torch.nn.functional.pad(scale_clus, pad_sc)

        features_list.append(padded_feat)
        feature_length_list.append(feat_len.clone().detach())
        ms_seg_timestamps_list.append(padded_ms_seg_ts)
        ms_seg_counts_list.append(ms_seg_ct.clone().detach())
        scale_clus_label_list.append(padded_scale_clus)
        scale_mapping_list.append(padded_sm)
        targets_list.append(padded_tgt)

    features = torch.stack(features_list)
    feature_length = torch.stack(feature_length_list)
    ms_seg_timestamps = torch.stack(ms_seg_timestamps_list)
    clus_label_index = torch.stack(scale_clus_label_list)
    ms_seg_counts = torch.stack(ms_seg_counts_list)
    scale_mapping = torch.stack(scale_mapping_list)
    targets = torch.stack(targets_list)
    return features, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets


def _msdd_infer_collate_fn(self, batch):
    """
    Collate batch of feats (speaker embeddings), feature lengths, target label sequences
    and cluster-average embeddings.

    Args:
        batch (tuple):
            Batch tuple containing feats, feats_len, targets and ms_avg_embs.
    Returns:
        feats (torch.tensor):
            Collated speaker embedding with unified length.
        feats_len (torch.tensor):
            The actual length of each embedding sequence without zero padding.
        targets (torch.tensor):
            Groundtruth Speaker label for the given input embedding sequence.
        ms_avg_embs (torch.tensor):
            Cluster-average speaker embedding vectors.
    """

    packed_batch = list(zip(*batch))
    feats, feats_len, targets, ms_avg_embs = packed_batch
    feats_list, flen_list, targets_list, ms_avg_embs_list = [], [], [], []
    max_audio_len = max(feats_len)
    max_target_len = max([x.shape[0] for x in targets])

    for feature, feat_len, target, ivector in batch:
        flen_list.append(feat_len)
        ms_avg_embs_list.append(ivector)
        if feat_len < max_audio_len:
            pad_a = (0, 0, 0, 0, 0, max_audio_len - feat_len)
            pad_t = (0, 0, 0, max_target_len - target.shape[0])
            padded_feature = torch.nn.functional.pad(feature, pad_a)
            padded_target = torch.nn.functional.pad(target, pad_t)
            feats_list.append(padded_feature)
            targets_list.append(padded_target)
        else:
            targets_list.append(target.clone().detach())
            feats_list.append(feature.clone().detach())

    feats = torch.stack(feats_list)
    feats_len = torch.tensor(flen_list)
    targets = torch.stack(targets_list)
    ms_avg_embs = torch.stack(ms_avg_embs_list)
    return feats, feats_len, targets, ms_avg_embs


class AudioToSpeechMSDDTrainDataset(_AudioMSDDTrainDataset):
    """
    Dataset class that loads a json file containing paths to audio files,
    rttm files and number of speakers. This Dataset class is designed for
    training or fine-tuning speaker embedding extractor and diarization decoder
    at the same time.

    Example:
    {"audio_filepath": "/path/to/audio_0.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_0.rttm}
    ...
    {"audio_filepath": "/path/to/audio_n.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_n.rttm}

    Args:
        manifest_filepath (str):
            Path to input manifest json files.
        multiscale_args_dict (dict):
            Dictionary containing the parameters for multiscale segmentation and clustering.
        emb_dir (str):
            Path to a temporary folder where segmentation information for embedding extraction is saved.
        soft_label_thres (float):
            A threshold that determines the label of each segment based on RTTM file information.
        featurizer:
            Featurizer instance for generating features from the raw waveform.
        window_stride (float):
            Window stride for acoustic feature. This value is used for calculating the numbers of feature-level frames.
        emb_batch_size (int):
            Number of embedding vectors that are trained with attached computational graphs.
        pairwise_infer (bool):
            This variable should be True if dataloader is created for an inference task.
    """

    def __init__(
        self,
        *,
        manifest_filepath: str,
        multiscale_args_dict: Dict,
        emb_dir: str,
        soft_label_thres: float,
        featurizer,
        window_stride,
        emb_batch_size,
        pairwise_infer: bool,
        global_rank: int,
    ):
        super().__init__(
            manifest_filepath=manifest_filepath,
            multiscale_args_dict=multiscale_args_dict,
            emb_dir=emb_dir,
            soft_label_thres=soft_label_thres,
            featurizer=featurizer,
            window_stride=window_stride,
            emb_batch_size=emb_batch_size,
            pairwise_infer=pairwise_infer,
            global_rank=global_rank,
        )

    def msdd_train_collate_fn(self, batch):
        """Collate batch of audio features, feature lengths, target label sequences for training."""
        return _msdd_train_collate_fn(self, batch)


class AudioToSpeechMSDDInferDataset(_AudioMSDDInferDataset):
    """
    Dataset class that loads a json file containing paths to audio files,
    rttm files and number of speakers. The created labels are used for diarization inference.

    Example:
    {"audio_filepath": "/path/to/audio_0.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_0.rttm}
    ...
    {"audio_filepath": "/path/to/audio_n.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_n.rttm}

    Args:
        manifest_filepath (str):
            Path to input manifest json files.
        emb_dict (dict):
            Dictionary containing cluster-average embeddings and speaker mapping information.
        emb_seq (dict):
            Dictionary containing multiscale speaker embedding sequence, scale mapping
            and corresponding segment timestamps.
        clus_label_dict (dict):
            Subsegment-level (from base-scale) speaker labels from clustering results.
        soft_label_thres (float):
            Threshold that determines speaker labels of segments depending on the overlap
            with groundtruth speaker timestamps.
        featurizer:
            Featurizer instance for generating features from raw waveform.
        use_single_scale_clus (bool):
            Use only one scale for clustering instead of using multiple scales of embeddings for clustering.
        seq_eval_mode (bool):
            If True, F1 score will be calculated for each speaker pair during inference mode.
        window_stride (float):
            Window stride for acoustic feature. This value is used for calculating the numbers of
            feature-level frames.
        pairwise_infer (bool):
            If True, this Dataset class operates in inference mode. In inference mode, a set of speakers
            in the input audio is split into multiple pairs of speakers and speaker tuples
            (e.g. 3 speakers: [(0,1), (1,2), (0,2)]) and then fed into the MSDD to merge the individual results.
    """

    def __init__(
        self,
        *,
        manifest_filepath: str,
        emb_dict: Dict,
        emb_seq: Dict,
        clus_label_dict: Dict,
        soft_label_thres: float,
        use_single_scale_clus: bool,
        seq_eval_mode: bool,
        window_stride: float,
        pairwise_infer: bool,
    ):
        super().__init__(
            manifest_filepath=manifest_filepath,
            emb_dict=emb_dict,
            emb_seq=emb_seq,
            clus_label_dict=clus_label_dict,
            soft_label_thres=soft_label_thres,
            use_single_scale_clus=use_single_scale_clus,
            window_stride=window_stride,
            seq_eval_mode=seq_eval_mode,
            pairwise_infer=pairwise_infer,
        )

    def msdd_infer_collate_fn(self, batch):
        """Collate batch of audio features, feature lengths, target label sequences for inference."""
        return _msdd_infer_collate_fn(self, batch)


def extract_global_speaker_ids(
    collection,
    min_speaker_duration_sec: float = 0.0,
) -> dict:
    """
    Scan all RTTM files in the collection to build a mapping from
    speaker name strings (RTTM column 8) to globally unique integer IDs.

    The collection is first deduplicated by ``(rttm_file, offset, duration)``
    so that multi-microphone recordings sharing the same RTTM and duplicate
    manifest lines do not inflate per-speaker duration counts.

    Args:
        collection: An EndtoEndDiarizationSpeechLabel collection whose
            items have ``rttm_file``, ``offset`` and ``duration`` attributes.
        min_speaker_duration_sec: If > 0, only speakers whose total speech
            duration (summed across all unique segments) meets this threshold
            are included. Set to 0 to include all speakers (default).

    Returns:
        speaker_to_id (dict): Sorted mapping ``{speaker_name: int_id}``.
    """
    from collections import defaultdict

    seen_segments = set()
    speaker_duration = defaultdict(float)

    for sample in collection:
        if sample.rttm_file in (None, ''):
            continue
        offset = sample.offset if sample.offset is not None else 0
        seg_key = (sample.rttm_file, offset, sample.duration)
        if seg_key in seen_segments:
            continue
        seen_segments.add(seg_key)

        with open(sample.rttm_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                speaker = parts[7]
                seg_start = float(parts[3])
                seg_dur = float(parts[4])
                seg_end = seg_start + seg_dur

                clipped_start = max(seg_start, offset)
                clipped_end = min(seg_end, offset + sample.duration)
                if clipped_end > clipped_start:
                    speaker_duration[speaker] += clipped_end - clipped_start

    if min_speaker_duration_sec > 0:
        all_speakers = sorted(speaker_duration.keys())
        speakers = [
            s for s in all_speakers
            if speaker_duration[s] >= min_speaker_duration_sec
        ]
        n_filtered = len(all_speakers) - len(speakers)
        logging.info(
            f"Built global speaker vocabulary: {len(speakers)} speakers "
            f"({n_filtered} filtered out with < {min_speaker_duration_sec}s speech)"
        )
    else:
        speakers = sorted(speaker_duration.keys())
        logging.info(
            f"Built global speaker vocabulary: {len(speakers)} unique speakers"
        )

    speaker_to_id = {s: i for i, s in enumerate(speakers)}
    return speaker_to_id


class _AudioToSpeechE2ESpkDiarDataset(Dataset):
    """
    Dataset class that loads a json file containing paths to audio files,
    RTTM files and number of speakers. This Dataset class is designed for
    training or fine-tuning speaker embedding extractor and diarization decoder
    at the same time.

    Example:
    {"audio_filepath": "/path/to/audio_0.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_0.rttm}
    ...
    {"audio_filepath": "/path/to/audio_n.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_n.rttm}

    Args:
        manifest_filepath (str):
            Path to input manifest json files.
        multiargs_dict (dict):
            Dictionary containing the parameters for multiscale segmentation and clustering.
        soft_label_thres (float):
            Threshold that determines the label of each segment based on RTTM file information.
        featurizer:
            Featurizer instance for generating audio_signal from the raw waveform.
        window_stride (float):
            Window stride for acoustic feature. This value is used for calculating the numbers of feature-level frames.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        output_types = {
            "audio_signal": NeuralType(('B', 'T'), AudioSignal()),
            "audio_length": NeuralType(('B'), LengthsType()),
            "targets": NeuralType(('B', 'T', 'C'), ProbsType()),
            "target_len": NeuralType(('B'), LengthsType()),
        }

        return output_types

    def __init__(
        self,
        *,
        manifest_filepath: str,
        soft_label_thres: float,
        session_len_sec: float,
        num_spks: int,
        featurizer,
        fb_featurizer,
        window_stride: float,
        min_subsegment_duration: float = 0.03,
        global_rank: int = 0,
        dtype=torch.float16,
        round_digits: int = 2,
        soft_targets: bool = False,
        subsampling_factor: int = 8,
        device: str = 'cpu',
        subsegment_mode: bool = False,
        subsegment_min_len_sec: float = 15.0,
        subsegment_two_chunks_rate: float = 0.0,
        subsegment_min_chunk_len_sec: float = 10.0,
        subsegment_margin_frames: int = 0,
        subsegment_nspk_bias: float = 1.0,
        subsegment_min_first_spk_frames: int = 50,
        subsegment_boundary_silence_frames: int = 10,
        subsegment_preload_sec: float = 0.0,
        opus_roundtrip_prob: float = 0.0,
        opus_roundtrip_compression_level: Optional[float] = None,
        validate_manifest_paths: bool = True,
    ):
        super().__init__()
        self.collection = EndtoEndDiarizationSpeechLabel(
            manifests_files=manifest_filepath.split(','),
            round_digits=round_digits,
            validate_manifest_paths=validate_manifest_paths,
        )

        self.featurizer = featurizer
        self.fb_featurizer = fb_featurizer
        # STFT and subsampling factor parameters
        self.n_fft = self.fb_featurizer.n_fft
        self.hop_length = self.fb_featurizer.hop_length
        self.stft_pad_amount = self.fb_featurizer.stft_pad_amount
        self.subsampling_factor = subsampling_factor
        # Annotation and target length parameters
        self.round_digits = round_digits
        self.feat_per_sec = int(1 / window_stride)
        self.diar_frame_length = round(subsampling_factor * window_stride, round_digits)
        self.session_len_sec = session_len_sec
        self.soft_label_thres = soft_label_thres
        self.max_spks = num_spks
        self.min_subsegment_duration = min_subsegment_duration
        self.dtype = dtype
        self.use_asr_style_frame_count = True
        self.soft_targets = soft_targets
        self.round_digits = 2
        self.floor_decimal = 10**self.round_digits
        self.device = device
        self.global_rank = global_rank
        self.subsegment_mode = subsegment_mode
        self.subsegment_min_len_sec = subsegment_min_len_sec
        self.subsegment_two_chunks_rate = subsegment_two_chunks_rate
        self.subsegment_min_chunk_len_sec = subsegment_min_chunk_len_sec
        self.subsegment_margin_frames = subsegment_margin_frames
        self.subsegment_nspk_bias = subsegment_nspk_bias
        self.subsegment_min_first_spk_frames = subsegment_min_first_spk_frames
        self.subsegment_boundary_silence_frames = subsegment_boundary_silence_frames
        self.subsegment_preload_sec = float(subsegment_preload_sec)
        for name, value in (
            ("subsegment_min_first_spk_frames", self.subsegment_min_first_spk_frames),
            ("subsegment_boundary_silence_frames", self.subsegment_boundary_silence_frames),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.subsegment_preload_sec < 0:
            raise ValueError(
                f"subsegment_preload_sec must be non-negative, got {self.subsegment_preload_sec}"
            )
        if (
            self.subsegment_preload_sec > 0
            and self.session_len_sec > 0
            and self.subsegment_preload_sec < self.session_len_sec
        ):
            raise ValueError(
                "subsegment_preload_sec cannot be shorter than session_len_sec"
            )
        if not 0.0 <= opus_roundtrip_prob <= 1.0:
            raise ValueError(f"opus_roundtrip_prob must be between 0 and 1, got {opus_roundtrip_prob}")
        if opus_roundtrip_compression_level is not None and not 0.0 <= opus_roundtrip_compression_level <= 1.0:
            raise ValueError(
                "opus_roundtrip_compression_level must be between 0 and 1, "
                f"got {opus_roundtrip_compression_level}"
            )
        self.opus_roundtrip_prob = opus_roundtrip_prob
        self.opus_roundtrip_compression_level = opus_roundtrip_compression_level
        if self.session_len_sec > 0:
            assert self.subsegment_min_len_sec <= self.session_len_sec, (
                f"subsegment_min_len_sec ({self.subsegment_min_len_sec}) cannot be greater than "
                f"session_len_sec ({self.session_len_sec})"
            )
            assert self.subsegment_min_chunk_len_sec * 2 <= self.session_len_sec, (
                "twice subsegment_min_chunk_len_sec cannot be greater than session_len_sec"
            )

    def __len__(self):
        return len(self.collection)

    def _maybe_apply_opus_roundtrip(self, audio_signal):
        if self.opus_roundtrip_prob <= 0.0 or random.random() >= self.opus_roundtrip_prob:
            return audio_signal
        return self._opus_roundtrip(audio_signal)

    def _opus_roundtrip(self, audio_signal):
        try:
            sf = importlib.import_module('soundfile')
        except ModuleNotFoundError as exc:
            raise RuntimeError("opus_roundtrip_prob > 0 requires the soundfile package with OGG/OPUS support") from exc

        original_dtype = audio_signal.dtype
        original_device = audio_signal.device
        original_num_samples = audio_signal.shape[0]

        audio_np = audio_signal.detach().cpu().float().numpy()
        if audio_np.ndim not in (1, 2):
            raise ValueError(f"Opus roundtrip supports 1D or 2D audio tensors, got shape {audio_signal.shape}")

        opus_buffer = io.BytesIO()
        write_kwargs = {}
        if self.opus_roundtrip_compression_level is not None:
            write_kwargs['compression_level'] = self.opus_roundtrip_compression_level

        try:
            sf.write(
                opus_buffer,
                audio_np,
                self.featurizer.sample_rate,
                format='OGG',
                subtype='OPUS',
                **write_kwargs,
            )
            opus_buffer.seek(0)
            decoded, _ = sf.read(opus_buffer, dtype='float32', always_2d=audio_np.ndim == 2)
        except Exception as exc:
            raise RuntimeError("soundfile OGG/OPUS roundtrip failed") from exc

        if decoded.shape[0] > original_num_samples:
            decoded = decoded[:original_num_samples]
        elif decoded.shape[0] < original_num_samples:
            pad_width = original_num_samples - decoded.shape[0]
            if audio_np.ndim == 1:
                decoded = np.pad(decoded, (0, pad_width))
            else:
                decoded = np.pad(decoded, ((0, pad_width), (0, 0)))

        return torch.as_tensor(decoded.copy(), dtype=original_dtype, device=original_device)

    def get_frame_count_from_time_series_length(self, seq_len):
        """
        This function is used to get the sequence length of the audio signal. This is required to match
        the feature frame length with ASR (STT) models. This function is copied from
        NeMo/nemo/collections/asr/parts/preprocessing/features.py::FilterbankFeatures::get_seq_len.

        Args:
            seq_len (int):
                The sequence length of the time-series data.

        Returns:
            seq_len (int):
                The sequence length of the feature frames.
        """
        pad_amount = self.stft_pad_amount * 2 if self.stft_pad_amount is not None else self.n_fft // 2 * 2
        seq_len = torch.floor_divide((seq_len + pad_amount - self.n_fft), self.hop_length).to(dtype=torch.long)
        frame_count = int(np.ceil(seq_len / self.subsampling_factor))
        return frame_count

    def get_uniq_id_with_range(self, sample, deci=3):
        """
        Generate unique training sample ID from unique file ID, offset and duration. The start-end time added
        unique ID is required for identifying the sample since multiple short audio samples are generated from a single
        audio file. The start time and end time of the audio stream uses millisecond units if `deci=3`.

        Args:
            sample:
                `DiarizationSpeechLabel` instance from collections.

        Returns:
            uniq_id (str):
                Unique sample ID which includes start and end time of the audio stream.
                Example: abc1001_3122_6458
        """
        bare_uniq_id = os.path.splitext(os.path.basename(sample.rttm_file))[0]
        offset = str(int(round(sample.offset, deci) * pow(10, deci)))
        endtime = str(int(round(sample.offset + sample.duration, deci) * pow(10, deci)))
        uniq_id = f"{bare_uniq_id}_{offset}_{endtime}"
        return uniq_id

    def _build_speaker_names(self, sess_to_global_spkids, columns=None):
        """
        Build a list of RTTM speaker name strings for each target-matrix column.

        Args:
            sess_to_global_spkids (dict): ``{column_index: speaker_name}`` from
                ``extract_frame_info_from_rttm``.
            columns (list, optional): Subset of column indices that were kept
                (after speaker dropping in subsegment mode).  If *None*, all
                columns ``0 .. max_spks-1`` are used in order.

        Returns:
            speaker_names (list): Length ``max_spks``. Each entry is the RTTM
                speaker name string for that column, or None if unused.
        """
        speaker_names = [None] * self.max_spks
        if columns is None:
            for col_idx, spk_name in sess_to_global_spkids.items():
                if col_idx < self.max_spks:
                    speaker_names[col_idx] = spk_name
        else:
            for new_col, old_col in enumerate(columns):
                spk_name = sess_to_global_spkids.get(old_col)
                if spk_name is not None and new_col < self.max_spks:
                    speaker_names[new_col] = spk_name
        return speaker_names

    def parse_rttm_for_targets_and_lens(self, rttm_file, offset, duration, target_len):
        """
        Generate target tensor variable by extracting groundtruth diarization labels from an RTTM file.
        This function converts (start, end, speaker_id) format into base-scale (the finest scale) segment level
        diarization label in a matrix form.

        Example of seg_target:
            [[0., 1.], [0., 1.], [1., 1.], [1., 0.], [1., 0.], ..., [0., 1.]]

        Returns:
            step_target (torch.Tensor): Diarization targets, shape (num_seg, max_spks).
            speaker_names (list): RTTM speaker name per target column, length max_spks.
        """
        if rttm_file in [None, '']:
            num_seg = torch.max(target_len)
            targets = torch.zeros(num_seg, self.max_spks)
            speaker_names = [None] * self.max_spks
            return targets, speaker_names

        with open(rttm_file, 'r') as f:
            rttm_lines = f.readlines()

        rttm_timestamps, sess_to_global_spkids = extract_frame_info_from_rttm(offset, duration, rttm_lines)

        fr_level_target = get_frame_targets_from_rttm(
            rttm_timestamps=rttm_timestamps,
            offset=offset,
            duration=duration,
            round_digits=self.round_digits,
            feat_per_sec=self.feat_per_sec,
            max_spks=self.max_spks,
        )

        soft_target_seg = self.get_soft_targets_seg(feat_level_target=fr_level_target, target_len=target_len)
        if self.soft_targets:
            step_target = soft_target_seg
        else:
            step_target = (soft_target_seg >= self.soft_label_thres).float()

        speaker_names = self._build_speaker_names(sess_to_global_spkids)
        return step_target, speaker_names

    def get_soft_targets_seg(self, feat_level_target, target_len):
        """
        Generate the final targets for the actual diarization step.
        Here, frame level means step level which is also referred to as segments.
        We follow the original paper and refer to the step level as "frames".

        Args:
            feat_level_target (torch.tensor):
                Tensor variable containing hard-labels of speaker activity in each feature-level segment.
            target_len (torch.tensor):
                Numbers of ms segments

        Returns:
            soft_target_seg (torch.tensor):
                Tensor variable containing soft-labels of speaker activity in each step-level segment.
        """
        num_seg = torch.max(target_len)
        stride = int(self.feat_per_sec * self.diar_frame_length)

        # When stride=1 (no subsampling), there is a 1:1 mapping between feature
        # frames and diarization steps. Skip the averaging loop entirely to avoid
        # an indexing bug (empty slice at index=0) and the performance cost of a
        # Python loop over potentially thousands of frames.
        if stride <= 1:
            return feat_level_target[:num_seg, :].clone()

        targets = torch.zeros(num_seg, self.max_spks)
        for index in range(num_seg):
            if index == 0:
                seg_stt_feat = 0
            else:
                seg_stt_feat = stride * index - 1 - int(stride / 2)
            if index == num_seg - 1:
                seg_end_feat = feat_level_target.shape[0]
            else:
                seg_end_feat = stride * index - 1 + int(stride / 2)
            targets[index] = torch.mean(feat_level_target[seg_stt_feat : seg_end_feat + 1, :], axis=0)
        return targets

    def get_segment_timestamps(
        self,
        duration: float,
        offset: float = 0,
        sample_rate: int = 16000,
    ):
        """
        Get start and end time of segments in each scale.

        Args:
            sample:
                `DiarizationSpeechLabel` instance from preprocessing.collections
        Returns:
            segment_timestamps (torch.tensor):
                Tensor containing Multiscale segment timestamps.
            target_len (torch.tensor):
                Number of segments for each scale. This information is used for reshaping embedding batch
                during forward propagation.
        """
        stride = int(self.feat_per_sec * self.diar_frame_length)

        # When stride<=1 (no subsampling), there is a 1:1 mapping between feature
        # frames and diarization steps. Compute target_len directly from the audio
        # duration rather than going through get_subsegments, which would fail because
        # the subsegment window is shorter than min_subsegment_duration.
        if stride <= 1:
            num_frames = int(
                np.ceil((1 + duration * sample_rate) / int(sample_rate / self.feat_per_sec))
            )
            return torch.tensor([num_frames])

        subsegments = get_subsegments(
            offset=offset,
            window=round(self.diar_frame_length * 2, self.round_digits),
            shift=self.diar_frame_length,
            duration=duration,
            min_subsegment_duration=self.min_subsegment_duration,
            use_asr_style_frame_count=self.use_asr_style_frame_count,
            sample_rate=sample_rate,
            feat_per_sec=self.feat_per_sec,
        )
        if self.use_asr_style_frame_count:
            effective_dur = (
                np.ceil((1 + duration * sample_rate) / int(sample_rate / self.feat_per_sec)).astype(int)
                / self.feat_per_sec
            )
        else:
            effective_dur = duration
        ts_tensor = get_subsegments_to_timestamps(
            subsegments, self.feat_per_sec, decimals=2, max_end_ts=(offset + effective_dur)
        )
        target_len = torch.tensor([ts_tensor.shape[0]])
        return target_len

    def _compute_spk_bias_weights(
        self, frame_level_target, candidate_indices, window_len, included_speakers=None
    ):
        """Compute sampling weights for candidate start positions biased toward higher unique speaker counts.

        For each candidate window, counts how many distinct speakers have at least one
        active frame in that window or in ``included_speakers``, then returns
        ``subsegment_nspk_bias ** n_unique_speakers``.

        Uses per-speaker prefix sums so cost is O(T*S + N*S) where S is small (max_spks).

        Args:
            frame_level_target: (T, S) binary/soft target tensor.
            candidate_indices: 1-D tensor of eligible start-frame indices.
            window_len: number of frames in the evaluation window.
            included_speakers: optional (S,) mask of speakers already present.

        Returns:
            1-D float tensor of sampling weights aligned with candidate_indices.
        """
        active = (frame_level_target > self.soft_label_thres).float()  # (T, S)
        T, S = active.shape
        cumsum = torch.zeros(T + 1, S)
        cumsum[1:] = torch.cumsum(active, dim=0)
        ends = torch.clamp(candidate_indices + window_len, max=T)
        window_activity = cumsum[ends] - cumsum[candidate_indices]  # (N, S)
        speaker_presence = window_activity > 0
        if included_speakers is not None:
            speaker_presence |= included_speakers.to(
                device=speaker_presence.device, dtype=torch.bool
            ).unsqueeze(0)
        unique_spk_counts = speaker_presence.float().sum(dim=1)  # (N,)
        weights = self.subsegment_nspk_bias ** unique_spk_counts
        return weights

    @staticmethod
    def _next_true_indices(mask):
        """Return the first true index at or after every position, or ``len(mask)``."""
        length = mask.shape[0]
        indices = torch.arange(length, device=mask.device)
        marked = torch.where(mask, indices, length)
        return torch.flip(torch.cummin(torch.flip(marked, dims=(0,)), dim=0).values, dims=(0,))

    def _eligible_chunk_starts(self, frame_level_target):
        """Find starts whose first speaker is established before another speaker appears.

        Silence before the first speaker is unrestricted. Once the first active frame
        is reached, it must contain exactly one speaker. That speaker must accumulate
        ``subsegment_min_first_spk_frames`` active frames before any other speaker
        becomes active; intervening silence does not reset the count.
        """
        active = frame_level_target > self.soft_label_thres
        num_frames, num_speakers = active.shape
        if num_frames == 0:
            return torch.zeros(0, dtype=torch.bool, device=active.device)

        frame_indices = torch.arange(num_frames, device=active.device)
        active_count = active.sum(dim=1)
        next_active = self._next_true_indices(active_count > 0)
        activity_prefix = torch.zeros(
            num_frames + 1, num_speakers, dtype=torch.long, device=active.device
        )
        activity_prefix[1:] = active.long().cumsum(dim=0)

        qualifying_first_frame = torch.zeros(num_frames, dtype=torch.bool, device=active.device)
        for speaker in range(num_speakers):
            other_active = active_count - active[:, speaker].long() > 0
            next_other = self._next_true_indices(other_active)
            speaker_frames = (
                activity_prefix[next_other, speaker] - activity_prefix[frame_indices, speaker]
            )
            qualifying_first_frame |= (
                (active_count == 1)
                & active[:, speaker]
                & (speaker_frames >= self.subsegment_min_first_spk_frames)
            )

        eligible = torch.zeros(num_frames, dtype=torch.bool, device=active.device)
        has_future_speech = next_active < num_frames
        eligible[has_future_speech] = qualifying_first_frame[next_active[has_future_speech]]
        return eligible

    def _silence_boundary_mask(self, frame_level_target):
        """Mark boundaries surrounded by the configured number of silent frames."""
        num_frames = frame_level_target.shape[0]
        margin = self.subsegment_boundary_silence_frames
        boundaries = torch.arange(num_frames + 1, device=frame_level_target.device)
        valid = (boundaries >= margin) & (boundaries + margin <= num_frames)
        silence = (frame_level_target <= self.soft_label_thres).all(dim=1)
        silence_prefix = torch.zeros(num_frames + 1, dtype=torch.long, device=silence.device)
        silence_prefix[1:] = silence.long().cumsum(dim=0)

        safe = torch.zeros(num_frames + 1, dtype=torch.bool, device=silence.device)
        valid_boundaries = boundaries[valid]
        silent_count = (
            silence_prefix[valid_boundaries + margin] - silence_prefix[valid_boundaries - margin]
        )
        safe[valid_boundaries] = silent_count == 2 * margin
        return safe

    def _sample_start(
        self, frame_level_target, candidates, window_len, included_speakers=None
    ):
        """Sample one candidate while preserving the existing speaker-count bias."""
        if candidates.numel() == 0:
            return None
        if self.subsegment_nspk_bias > 1.0:
            weights = self._compute_spk_bias_weights(
                frame_level_target,
                candidates,
                window_len,
                included_speakers=included_speakers,
            )
            return candidates[torch.multinomial(weights, 1).item()].item()
        return candidates[random.randrange(candidates.numel())].item()

    def _sample_single_chunk_bounds(self, frame_level_target, max_len, min_len):
        """Sample one eligible chunk and return ``[(start, end)]``."""
        num_frames = frame_level_target.shape[0]
        if num_frames < min_len:
            return None

        eligible = self._eligible_chunk_starts(frame_level_target)
        candidates = torch.where(
            eligible & (torch.arange(num_frames, device=eligible.device) <= num_frames - min_len)
        )[0]
        start = self._sample_start(frame_level_target, candidates, max_len)
        if start is None:
            return None
        return [(start, min(start + max_len, num_frames))]

    def _sample_two_chunk_bounds(self, frame_level_target, total_len, min_chunk_len):
        """Sample two non-overlapping chunks with a silence-protected splice.

        Chunk one is always returned first, irrespective of source chronology. Its
        end and chunk two's start each lie at the center of a ``2 * S``-frame
        silence region, yielding at least ``2 * S`` silent frames at the splice.
        """
        num_frames = frame_level_target.shape[0]
        if total_len < 2 * min_chunk_len or num_frames < total_len:
            return None

        eligible = self._eligible_chunk_starts(frame_level_target)
        safe_boundary = self._silence_boundary_mask(frame_level_target)
        frame_indices = torch.arange(num_frames, device=eligible.device)
        first_candidates = torch.where(
            eligible
            & (frame_indices + min_chunk_len + self.subsegment_boundary_silence_frames <= num_frames)
        )[0]
        if first_candidates.numel() == 0:
            return None

        if self.subsegment_nspk_bias > 1.0:
            weights = self._compute_spk_bias_weights(
                frame_level_target, first_candidates, total_len - min_chunk_len
            )
            first_order = torch.multinomial(weights, first_candidates.numel(), replacement=False)
        else:
            first_order = torch.randperm(first_candidates.numel(), device=first_candidates.device)

        second_start_mask = eligible & safe_boundary[:-1]
        second_start_prefix = torch.zeros(
            num_frames + 1, dtype=torch.long, device=eligible.device
        )
        second_start_prefix[1:] = second_start_mask.long().cumsum(dim=0)
        safe_ends = torch.where(safe_boundary)[0]

        for first_index in first_order.tolist():
            start1 = first_candidates[first_index].item()
            min_end1 = start1 + min_chunk_len
            max_end1 = min(
                start1 + total_len - min_chunk_len,
                num_frames - self.subsegment_boundary_silence_frames,
            )
            end_candidates = safe_ends[
                (safe_ends >= min_end1) & (safe_ends <= max_end1)
            ]
            if end_candidates.numel() == 0:
                continue

            len2_by_end = total_len - (end_candidates - start1)
            before_last_start = start1 - len2_by_end
            before_exists = torch.zeros_like(before_last_start, dtype=torch.bool)
            has_before_range = before_last_start >= 0
            before_exists[has_before_range] = (
                second_start_prefix[before_last_start[has_before_range] + 1] > 0
            )

            after_last_start = num_frames - len2_by_end
            has_after_range = after_last_start >= end_candidates
            after_exists = torch.zeros_like(has_after_range)
            after_exists[has_after_range] = (
                second_start_prefix[after_last_start[has_after_range] + 1]
                - second_start_prefix[end_candidates[has_after_range]]
                > 0
            )
            end_candidates = end_candidates[before_exists | after_exists]
            if end_candidates.numel() == 0:
                continue

            end1 = end_candidates[random.randrange(end_candidates.numel())].item()
            len1 = end1 - start1
            len2 = total_len - len1
            valid_second = second_start_mask & (frame_indices + len2 <= num_frames)
            valid_second &= ((frame_indices + len2 <= start1) | (frame_indices >= end1))
            first_speakers = (
                frame_level_target[start1:end1] > self.soft_label_thres
            ).any(dim=0)
            start2 = self._sample_start(
                frame_level_target,
                torch.where(valid_second)[0],
                len2,
                included_speakers=first_speakers,
            )
            if start2 is not None:
                return [(start1, end1), (start2, start2 + len2)]

        return None

    def _slice_chunks(self, audio_signal, frame_level_target, bounds):
        """Slice and concatenate matching waveform and target chunks."""
        samples_per_frame = self.featurizer.sample_rate / self.feat_per_sec
        frame_level_target = torch.cat(
            [frame_level_target[start:end] for start, end in bounds]
        )
        audio_signal = torch.cat(
            [
                audio_signal[
                    int(start * samples_per_frame) : int(end * samples_per_frame)
                ]
                for start, end in bounds
            ]
        )
        return audio_signal, frame_level_target

    def _create_subsegment(self, sample, offset):
        duration = sample.duration

        # Pre-crop very long files before loading the waveform.
        if self.subsegment_preload_sec > 0 and duration > self.subsegment_preload_sec:
            preload_start = random.uniform(0, duration - self.subsegment_preload_sec)
            offset = offset + preload_start
            duration = self.subsegment_preload_sec

        audio_signal = self.featurizer.process(sample.audio_file, offset=offset, duration=duration)
        
        with open(sample.rttm_file, 'r') as f:
            rttm_lines = f.readlines()
        
        rttm_timestamps, sess_to_global_spkids = extract_frame_info_from_rttm(offset, duration, rttm_lines)
        num_speakers = len(sess_to_global_spkids)
        all_spks = list(sess_to_global_spkids.keys())

        frame_level_target = get_frame_targets_from_rttm(
            rttm_timestamps=rttm_timestamps,
            offset=offset,
            duration=duration,
            round_digits=self.round_digits,
            feat_per_sec=self.feat_per_sec,
            max_spks=num_speakers,
        )

        if num_speakers > self.max_spks:
            active_frames_per_spk = frame_level_target.sum(dim=0)
            weights = active_frames_per_spk.float()
            if weights.sum() == 0:
                weights = torch.ones(num_speakers)
            spk_indices = torch.multinomial(weights, self.max_spks, replacement=False)
            spks_tokeep = sorted(spk_indices.tolist())
            #logging.info(
            #    f"uniq_id: {sample.uniq_id}, active_frames_per_spk: {active_frames_per_spk.tolist()}, spks_tokeep: {spks_tokeep}"
            #)
        else:
            spks_tokeep = all_spks

        spks_todrop = [spk for spk in all_spks if spk not in spks_tokeep]

        if spks_todrop:
            samples_per_frame = int(self.featurizer.sample_rate / self.feat_per_sec)
            activity_todrop = frame_level_target[:, spks_todrop].sum(dim=1) > 0

            # Option B boundary margin: replace frames adjacent to excision boundaries
            # with silence (audio) and zeros (targets) to remove residual energy from
            # dropped speakers and provide an acoustic buffer at concatenation points.
            if self.subsegment_margin_frames > 0:
                margin = self.subsegment_margin_frames
                # Dilate the excision mask to find boundary margin frames
                activity_float = activity_todrop.float().unsqueeze(0).unsqueeze(0)
                dilated = (
                    torch.nn.functional.max_pool1d(
                        activity_float,
                        kernel_size=2 * margin + 1,
                        stride=1,
                        padding=margin,
                    ).squeeze(0).squeeze(0)
                    > 0
                )
                # Margin = frames in the dilated region but NOT in the original excision region
                margin_mask = dilated & ~activity_todrop

                # Zero out frame-level targets at margin frames
                frame_level_target[margin_mask] = 0

                # Replace audio at margin frames with low-level noise (avoids
                # log(0) artifacts in log-mel feature extraction that pure zeros would cause)
                audio_margin_mask = margin_mask.repeat_interleave(samples_per_frame)
                margin_len = min(audio_signal.shape[0], audio_margin_mask.shape[0])
                full_audio_margin = torch.zeros(audio_signal.shape[0], dtype=torch.bool)
                full_audio_margin[:margin_len] = audio_margin_mask[:margin_len]
                n_margin_samples = full_audio_margin.sum().item()
                audio_signal[full_audio_margin] = torch.randn(n_margin_samples) * 1e-3

            # Excise frames where dropped speakers are active
            frames_to_keep_mask = ~activity_todrop
            frame_level_target = frame_level_target[frames_to_keep_mask]

            # Create a corresponding mask for the audio signal by expanding the frame mask
            audio_mask = frames_to_keep_mask.repeat_interleave(samples_per_frame)

            # Ensure audio_signal and audio_mask have the same length before applying mask
            min_audio_len = min(audio_signal.shape[0], audio_mask.shape[0])
            audio_signal = audio_signal[:min_audio_len]
            audio_mask = audio_mask[:min_audio_len]

            # Apply the mask to the audio signal
            audio_signal = audio_signal[audio_mask]

        frame_level_target = frame_level_target[:, spks_tokeep]
        speaker_names = self._build_speaker_names(sess_to_global_spkids, columns=spks_tokeep)

        if frame_level_target.shape[1] < self.max_spks:
            pad_width = self.max_spks - frame_level_target.shape[1]
            frame_level_target = torch.nn.functional.pad(frame_level_target, (0, pad_width), 'constant', 0)

        # Select either one chunk or a silence-protected pair of chunks.
        max_len_frames = int(self.session_len_sec * self.feat_per_sec)
        min_len_frames = int(self.subsegment_min_len_sec * self.feat_per_sec)

        if self.session_len_sec > 0:
            bounds = None
            min_chunk_len_frames = int(self.subsegment_min_chunk_len_sec * self.feat_per_sec)
            if random.random() < self.subsegment_two_chunks_rate:
                bounds = self._sample_two_chunk_bounds(
                    frame_level_target,
                    total_len=max_len_frames,
                    min_chunk_len=min_chunk_len_frames,
                )
            if bounds is None:
                bounds = self._sample_single_chunk_bounds(
                    frame_level_target,
                    max_len=max_len_frames,
                    min_len=min_len_frames,
                )
            if bounds is None:
                return (
                    torch.tensor([], dtype=audio_signal.dtype),
                    torch.tensor(0).long(),
                    torch.zeros((0, self.max_spks), dtype=frame_level_target.dtype),
                    torch.tensor([0]).long(),
                    [None] * self.max_spks,
                )
            audio_signal, frame_level_target = self._slice_chunks(
                audio_signal, frame_level_target, bounds
            )

        min_viable_samples = int(self.min_subsegment_duration * self.featurizer.sample_rate)
        if audio_signal.shape[0] < min_viable_samples:
            return (
                torch.tensor([], dtype=audio_signal.dtype),
                torch.tensor(0).long(),
                torch.zeros((0, self.max_spks), dtype=frame_level_target.dtype),
                torch.tensor([0]).long(),
                [None] * self.max_spks,
            )

        audio_signal = self._maybe_apply_opus_roundtrip(audio_signal)
        audio_signal_length = torch.tensor(audio_signal.shape[0]).long()
        #logging.info(f"uniq_id: {sample.uniq_id}, audio_signal_length: {audio_signal_length}")
        session_len_sec = audio_signal.shape[0] / self.featurizer.sample_rate

        target_len = self.get_segment_timestamps(duration=session_len_sec, sample_rate=self.featurizer.sample_rate)
        target_len = torch.clamp(target_len, max=self.get_frame_count_from_time_series_length(audio_signal.shape[0]))
        targets = self.get_soft_targets_seg(feat_level_target=frame_level_target, target_len=target_len)
        targets = targets[:target_len, :]
        # TODO: support self.soft_targets parameter - now targets are always soft

        return audio_signal, audio_signal_length, targets, target_len, speaker_names

    def __getitem__(self, index):
        sample = self.collection[index]
        if sample.offset is None:
            sample.offset = 0
        offset = sample.offset

        if self.subsegment_mode:
            return self._create_subsegment(sample, offset)

        if self.session_len_sec < 0:
            session_len_sec = sample.duration
        else:
            session_len_sec = min(sample.duration, self.session_len_sec)

        audio_signal = self.featurizer.process(sample.audio_file, offset=offset, duration=session_len_sec)

        # We should resolve the length mis-match from the round-off errors between these two variables:
        # `session_len_sec` and `audio_signal.shape[0]`
        session_len_sec = (
            np.floor(audio_signal.shape[0] / self.featurizer.sample_rate * self.floor_decimal) / self.floor_decimal
        )
        audio_signal = audio_signal[: round(self.featurizer.sample_rate * session_len_sec)]
        audio_signal = self._maybe_apply_opus_roundtrip(audio_signal)
        audio_signal_length = torch.tensor(audio_signal.shape[0]).long()

        # Target length should be following the ASR feature extraction convention: Use self.get_frame_count_from_time_series_length.
        target_len = self.get_segment_timestamps(duration=session_len_sec, sample_rate=self.featurizer.sample_rate)
        target_len = torch.clamp(target_len, max=self.get_frame_count_from_time_series_length(audio_signal.shape[0]))

        targets, speaker_names = self.parse_rttm_for_targets_and_lens(
            rttm_file=sample.rttm_file, offset=offset, duration=session_len_sec, target_len=target_len
        )
        targets = targets[:target_len, :]
        return audio_signal, audio_signal_length, targets, target_len, speaker_names


def _eesd_train_collate_fn(self, batch):
    """
    Collate a batch of variables needed for training the end-to-end speaker diarization (EESD) model
    from raw waveforms to diarization labels.

    Args:
        batch (tuple):
            A tuple containing the variables for diarization training.

    Returns:
        audio_signal (torch.Tensor): Raw waveform samples.
        feature_length (torch.Tensor): Lengths of raw waveform samples.
        targets (torch.Tensor): Groundtruth speaker labels.
        target_lens (torch.Tensor): Number of segments per sample.
        speaker_names (list[list[str|None]]): RTTM speaker name per target
            column for each sample in the batch.
    """
    batch = [item for item in batch if item[0].numel() > 0]
    if len(batch) == 0:
        return (
            torch.zeros(0),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0),
            torch.zeros(0, dtype=torch.long),
            [],
        )

    packed_batch = list(zip(*batch))
    audio_signal, feature_length, targets, target_len, speaker_names = packed_batch
    audio_signal_list, feature_length_list = [], []
    target_len_list, targets_list = [], []
    speaker_names_list = []

    max_raw_feat_len = max([x.shape[0] for x in audio_signal])
    max_target_len = max([x.shape[0] for x in targets])
    if max([len(feat.shape) for feat in audio_signal]) > 1:
        max_ch = max([feat.shape[1] for feat in audio_signal])
    else:
        max_ch = 1
    for feat, feat_len, tgt, segment_ct, spk_names in batch:
        seq_len = tgt.shape[0]
        if len(feat.shape) > 1:
            pad_feat = (0, 0, 0, max_raw_feat_len - feat.shape[0])
        else:
            pad_feat = (0, max_raw_feat_len - feat.shape[0])
        if feat.shape[0] < feat_len:
            feat_len_pad = feat_len - feat.shape[0]
            feat = torch.nn.functional.pad(feat, (0, feat_len_pad))
        pad_tgt = (0, 0, 0, max_target_len - seq_len)
        padded_feat = torch.nn.functional.pad(feat, pad_feat)
        padded_tgt = torch.nn.functional.pad(tgt, pad_tgt)
        if max_ch > 1 and padded_feat.shape[1] < max_ch:
            feat_ch_pad = max_ch - padded_feat.shape[1]
            padded_feat = torch.nn.functional.pad(padded_feat, (0, feat_ch_pad))
        audio_signal_list.append(padded_feat)
        feature_length_list.append(feat_len.clone().detach())
        target_len_list.append(segment_ct.clone().detach())
        targets_list.append(padded_tgt)
        speaker_names_list.append(spk_names)
    audio_signal = torch.stack(audio_signal_list)
    feature_length = torch.stack(feature_length_list)
    target_lens = torch.stack(target_len_list).squeeze(1)
    targets = torch.stack(targets_list)
    return audio_signal, feature_length, targets, target_lens, speaker_names_list


class AudioToSpeechE2ESpkDiarDataset(_AudioToSpeechE2ESpkDiarDataset):
    """
    Dataset class for loading a JSON file containing paths to audio files,
    RTTM (Rich Transcription Time Marked) files, and the number of speakers.
    This class is designed for training or fine-tuning a speaker embedding
    extractor and diarization decoder simultaneously.

    The JSON manifest file should have entries in the following format:

    Example:
    {
        "audio_filepath": "/path/to/audio_0.wav",
        "num_speakers": 2,
        "rttm_filepath": "/path/to/diar_label_0.rttm"
    }
    ...
    {
        "audio_filepath": "/path/to/audio_n.wav",
        "num_speakers": 2,
        "rttm_filepath": "/path/to/diar_label_n.rttm"
    }

    Args:
        manifest_filepath (str):
            Path to the input manifest JSON file containing paths to audio and RTTM files.
        soft_label_thres (float):
            Threshold for assigning soft labels to segments based on RTTM file information.
        session_len_sec (float):
            Duration of each session (in seconds) for training or fine-tuning.
        num_spks (int):
            Number of speakers in the audio files.
        featurizer:
            Instance of a featurizer for generating features from the raw waveform.
        window_stride (float):
            Window stride (in seconds) for extracting acoustic features, used to calculate
            the number of feature frames.
        global_rank (int):
            Global rank of the current process (used for distributed training).
        soft_targets (bool):
            Whether or not to use soft targets during training.

    Methods:
        eesd_train_collate_fn(batch):
            Collates a batch of data for end-to-end speaker diarization training.
    """

    def __init__(
        self,
        *,
        manifest_filepath: str,
        soft_label_thres: float,
        session_len_sec: float,
        num_spks: int,
        featurizer,
        fb_featurizer,
        window_stride,
        global_rank: int,
        soft_targets: bool,
        device: str,
        subsampling_factor: int = 8,
        subsegment_mode: bool = False,
        subsegment_min_len_sec: float = 15.0,
        subsegment_two_chunks_rate: float = 0.0,
        subsegment_min_chunk_len_sec: float = 10.0,
        subsegment_margin_frames: int = 0,
        subsegment_nspk_bias: float = 1.0,
        subsegment_min_first_spk_frames: int = 50,
        subsegment_boundary_silence_frames: int = 10,
        subsegment_preload_sec: float = 0.0,
        opus_roundtrip_prob: float = 0.0,
        opus_roundtrip_compression_level: Optional[float] = None,
        validate_manifest_paths: bool = True,
    ):
        super().__init__(
            manifest_filepath=manifest_filepath,
            soft_label_thres=soft_label_thres,
            session_len_sec=session_len_sec,
            num_spks=num_spks,
            featurizer=featurizer,
            fb_featurizer=fb_featurizer,
            window_stride=window_stride,
            subsampling_factor=subsampling_factor,
            global_rank=global_rank,
            soft_targets=soft_targets,
            device=device,
            subsegment_mode=subsegment_mode,
            subsegment_min_len_sec=subsegment_min_len_sec,
            subsegment_two_chunks_rate=subsegment_two_chunks_rate,
            subsegment_min_chunk_len_sec=subsegment_min_chunk_len_sec,
            subsegment_margin_frames=subsegment_margin_frames,
            subsegment_nspk_bias=subsegment_nspk_bias,
            subsegment_min_first_spk_frames=subsegment_min_first_spk_frames,
            subsegment_boundary_silence_frames=subsegment_boundary_silence_frames,
            subsegment_preload_sec=subsegment_preload_sec,
            opus_roundtrip_prob=opus_roundtrip_prob,
            opus_roundtrip_compression_level=opus_roundtrip_compression_level,
            validate_manifest_paths=validate_manifest_paths,
        )

    def eesd_train_collate_fn(self, batch):
        """Collate a batch of data for end-to-end speaker diarization training."""
        return _eesd_train_collate_fn(self, batch)
