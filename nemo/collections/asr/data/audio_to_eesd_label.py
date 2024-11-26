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

import os
from typing import Dict, List, Tuple, Optional
import torch

from nemo.collections.asr.parts.utils.speaker_utils import convert_rttm_line, get_subsegments
from nemo.collections.common.parts.preprocessing.collections_eesd import DiarizationSpeechLabel
from nemo.core.classes import Dataset
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType, ProbsType
import numpy as np

def get_subsegments_to_scale_timestamps(subsegments: List[Tuple[float, float]], feat_per_sec: int = 100, max_end_ts: float=None, decimals=2):
    """
    Convert subsegment timestamps to scale timestamps.

    Args:
        subsegments (List[Tuple[float, float]]):
            List of subsegment timestamps.

    Returns:
        scale_ts (torch.tensor):
            Tensor containing scale timestamps.
    """
    # scale_ts = (torch.tensor(subsegments) * feat_per_sec).long()
    seg_ts = (torch.tensor(subsegments) * feat_per_sec).float()
    scale_ts_round = torch.round(seg_ts, decimals=decimals)
    scale_ts = scale_ts_round.long()
    scale_ts[:, 1] = scale_ts[:, 0] + scale_ts[:, 1]
    if max_end_ts is not None:
        scale_ts = np.clip(scale_ts, 0, int(max_end_ts*feat_per_sec))
    return scale_ts 

def extract_seg_info_from_rttm(uniq_id, offset, duration, rttm_lines, mapping_dict=None, target_spks=None, round_digits=3):
    """
    Get RTTM lines containing speaker labels, start time and end time. target_spks contains two targeted
    speaker indices for creating groundtruth label files. Only speakers in target_spks variable will be
    included in the output lists.
    """
    rttm_stt, rttm_end = offset, offset + duration
    stt_list, end_list, speaker_list = [], [], []

    speaker_set = []
    sess_to_global_spkids = dict()
    for rttm_line in rttm_lines:
        start, end, speaker = convert_rttm_line(rttm_line)
        if start > end:
            continue
        if (end > rttm_stt and start < rttm_end) or (start < rttm_end and end > rttm_stt):
            start, end = max(start, rttm_stt), min(end, rttm_end)
        else:
            continue
        # if target_spks is None or speaker in pairwise_infer_spks:
        end_list.append(round(end, round_digits))
        stt_list.append(round(start, round_digits))
        if speaker not in speaker_set:
            speaker_set.append(speaker)
        speaker_list.append(speaker_set.index(speaker))
        sess_to_global_spkids.update({speaker_set.index(speaker):speaker})
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
            Number of feature frames per second. This quantity is determined by window_stride variable in preprocessing module.
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
        raise ValueError(f"Number of speakers in RTTM file {len(sorted_speakers)} exceeds the maximum number of speakers: {max_spks}")
    feat_level_target = torch.zeros(total_fr_len, max_spks) 
    for count, (stt, end, spk_rttm_key) in enumerate(zip(stt_list, end_list, speaker_list)):
        if end < offset or stt > offset + duration:
            continue
        stt, end = max(offset, stt), min(offset + duration, end)
        spk = spk_rttm_key
        stt_fr, end_fr = int((stt - offset) * feat_per_sec), int((end - offset)* feat_per_sec)
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
            "segment_counts": NeuralType(('B', 'C'), LengthsType()),
        }

        return output_types

    def __init__(
        self,
        *,
        manifest_filepath: str,
        preprocessor,
        soft_label_thres: float,
        session_len_sec: float,
        num_spks: int,
        featurizer,
        window_stride: float,
        min_subsegment_duration: float = 0.03,
        global_rank: int = 0,
        dtype=torch.float16,
        round_digits: int = 2,
        soft_targets: bool = False,
        subsampling_factor: int = 8,
    ):
        super().__init__()
        self.collection = DiarizationSpeechLabel(
            manifests_files=manifest_filepath.split(','),
            emb_dict=None,
            clus_label_dict=None,
            pairwise_infer=True,
        )
        self.preprocessor = preprocessor
        self.featurizer = featurizer
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
        self.floor_decimal = 10 ** self.round_digits
    
    def __len__(self):
        return len(self.collection)

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

    def parse_rttm_for_ms_targets(self, uniq_id, rttm_file, offset, duration, segment_counts):
        """
        Generate target tensor variable by extracting groundtruth diarization labels from an RTTM file.
        This function converts (start, end, speaker_id) format into base-scale (the finest scale) segment level
        diarization label in a matrix form.

        Example of seg_target:
            [[0., 1.], [0., 1.], [1., 1.], [1., 0.], [1., 0.], ..., [0., 1.]]
        """
        rttm_lines = open(rttm_file).readlines()
        rttm_timestamps, sess_to_global_spkids = extract_seg_info_from_rttm(uniq_id, offset, duration, rttm_lines)

        fr_level_target = get_frame_targets_from_rttm(rttm_timestamps=rttm_timestamps,
                                                      offset=offset,
                                                      duration=duration,
                                                      round_digits=self.round_digits,
                                                      feat_per_sec=self.feat_per_sec,
                                                      max_spks=self.max_spks)

        soft_target_seg = self.get_soft_targets_seg(feat_level_target=fr_level_target,
                                                    segment_counts=segment_counts)
        if self.soft_targets:
            step_target = soft_target_seg
        else:
            step_target = (soft_target_seg >= self.soft_label_thres).float()
        return step_target

    def get_soft_targets_seg(self, feat_level_target, segment_counts):
        """
        Generate the final targets for the actual diarization step.
        Here, frame level means step level which is also referred to as segments.
        We follow the original paper and refer to the step level as "frames".

        Args:
            feat_level_target (torch.tensor):
                Tensor variable containing hard-labels of speaker activity in each feature-level segment.
            segment_counts (torch.tensor):
                Numbers of ms segments

        Returns:
            soft_target_seg (torch.tensor):
                Tensor variable containing soft-labels of speaker activity in each step-level segment.
        """
        num_seg = torch.max(segment_counts)
        targets = torch.zeros(num_seg, self.max_spks)
        stride = int(self.feat_per_sec * self.diar_frame_length)
        for index in range(num_seg):
            if index == 0:
                seg_stt_feat = 0
            else:
                seg_stt_feat = stride * index - 1 - int(stride / 2)
            if index == num_seg - 1:
                seg_end_feat = feat_level_target.shape[0]
            else:
                seg_end_feat = stride * index - 1 + int(stride / 2)
            targets[index] = torch.mean(feat_level_target[seg_stt_feat:seg_end_feat+1, :], axis=0)
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
            segment_counts (torch.tensor):
                Number of segments for each scale. This information is used for reshaping embedding batch
                during forward propagation.
        """
        segment_timestamps_list = []
        subsegments = get_subsegments(offset=offset, 
                                      window=round(self.diar_frame_length * 2, self.round_digits),
                                      shift=self.diar_frame_length,
                                      duration=duration, 
                                      min_subsegment_duration=self.min_subsegment_duration,
                                      use_asr_style_frame_count=self.use_asr_style_frame_count,
                                      sample_rate=sample_rate,
                                      feat_per_sec=self.feat_per_sec,
        )
        if self.use_asr_style_frame_count:
            effective_dur =  np.ceil((1+duration*sample_rate)/int(sample_rate/self.feat_per_sec)).astype(int)/self.feat_per_sec
        else:
            effective_dur = duration 
        scale_ts_tensor = get_subsegments_to_scale_timestamps(subsegments, self.feat_per_sec, decimals=2, max_end_ts=(offset+effective_dur))
        total_steps = scale_ts_tensor.shape[0]
        scale_ts_padded = torch.cat([scale_ts_tensor, torch.zeros(total_steps - scale_ts_tensor.shape[0], 2, dtype=scale_ts_tensor.dtype)], dim=0)
        segment_timestamps_list.append(scale_ts_padded.detach())
        segment_timestamps_list = segment_timestamps_list[::-1]
        segment_timestamps = torch.stack(segment_timestamps_list).type(self.dtype)
        segment_counts = torch.tensor([scale_ts_tensor.shape[0]])
        return segment_timestamps, segment_counts    

    def __getitem__(self, index):
        sample = self.collection[index]
        if sample.offset is None:
            sample.offset = 0
        offset = sample.offset
        if self.session_len_sec < 0:
            session_len_sec = sample.duration
        else:
            session_len_sec = min(sample.duration, self.session_len_sec)

        uniq_id = self.get_uniq_id_with_range(sample)
        audio_signal = self.featurizer.process(sample.audio_file, offset=offset, duration=session_len_sec)
        
        # We should resolve the length mis-match from the round-off errors: `session_len_sec` and `audio_signal.shape[0]`
        session_len_sec = np.floor(audio_signal.shape[0] / self.featurizer.sample_rate * self.floor_decimal)/self.floor_decimal
        audio_signal = audio_signal[:round(self.featurizer.sample_rate*session_len_sec)]
        
        audio_signal_length = torch.tensor(audio_signal.shape[0]).long()
        audio_signal, audio_signal_length = audio_signal.to('cpu'), audio_signal_length.to('cpu')
        segment_timestamps, segment_counts = self.get_segment_timestamps(duration=session_len_sec, sample_rate=self.featurizer.sample_rate)
        targets = self.parse_rttm_for_ms_targets(uniq_id=uniq_id,
                                                 rttm_file=sample.rttm_file,
                                                 offset=offset,
                                                 duration=session_len_sec,
                                                 segment_counts=segment_counts)
        return audio_signal, audio_signal_length, targets, segment_counts

def _msdd_train_collate_fn(self, batch):
    """
    Collate batch of variables that are needed for raw waveform to diarization label training.
    The following variables are included in training/validation batch:

    Args:
        batch (tuple:
            Batch tuple containing the variables for the diarization training.
    Returns:
        audio_signal (torch.tensor):
            Raw waveform samples (time series) loaded from the audio_filepath in the input manifest file.
        feature lengths (time series sample length):
            A list of lengths of the raw waveform samples.
        segment_timestamps (torch.tensor):
            Matrix containing the start time and end time (timestamps) for each segment and each scale.
            segment_timestamps is needed for extracting acoustic features from raw waveforms.
        segment_counts (torch.tensor):
            Matrix containing The number of segments for each scale. segment_counts is necessary for reshaping
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
    audio_signal, feature_length, targets, segment_counts = packed_batch
    audio_signal_list, feature_length_list = [], []
    segment_counts_list, targets_list = [], []

    max_raw_feat_len = max([x.shape[0] for x in audio_signal])
    max_target_len = max([x.shape[0] for x in targets])
    if max([len(feat.shape) for feat in audio_signal]) > 1:
        max_ch = max([feat.shape[1] for feat in audio_signal])
    else:
        max_ch = 1
    for feat, feat_len, tgt, segment_ct in batch:
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
        segment_counts_list.append(segment_ct.clone().detach())
        targets_list.append(padded_tgt)
        audio_signal = torch.stack(audio_signal_list)
    feature_length = torch.stack(feature_length_list)
    segment_counts = torch.stack(segment_counts_list)
    targets = torch.stack(targets_list)
    return audio_signal, feature_length, targets, segment_counts

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
        soft_label_thres (float):
            A threshold that determines the label of each segment based on RTTM file information.
        featurizer:
            Featurizer instance for generating features from the raw waveform.
        window_stride (float):
            Window stride for acoustic feature. This value is used for calculating the numbers of feature-level frames.
    """

    def __init__(
        self,
        *,
        manifest_filepath: str,
        preprocessor,
        soft_label_thres: float,
        session_len_sec: float,
        num_spks: int,
        featurizer,
        window_stride,
        global_rank: int,
        soft_targets: bool,
    ):
        super().__init__(
            manifest_filepath=manifest_filepath,
            preprocessor=preprocessor,
            soft_label_thres=soft_label_thres,
            session_len_sec=session_len_sec,
            num_spks=num_spks,
            featurizer=featurizer,
            window_stride=window_stride,
            global_rank=global_rank,
            soft_targets=soft_targets,
        )

    def msdd_train_collate_fn(self, batch):
        return _msdd_train_collate_fn(self, batch)