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

import os
import re
import math
from copy import deepcopy
from typing import Dict, Optional, Tuple, List

import torch.utils.data
from lhotse.cut import MixedCut, MonoCut
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors, collate_matrices
from lhotse.utils import compute_num_samples
from lhotse import SupervisionSet, SupervisionSegment, MonoCut, Recording, CutSet

import numpy as np

from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper
from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType

from nemo.collections.asr.parts.utils.asr_multispeaker_utils import (
    get_hidden_length_from_sample_length, 
    find_segments_from_rttm,
    shuffle_spk_mapping,
)

from nemo.collections.asr.parts.utils.asr_tgtspeaker_utils import (
    get_separator_audio,
    get_query_cut
)

class LhotseSpeechToTextTgtSpkBpeDataset(torch.utils.data.Dataset):
    """
    This dataset is based on BPE datasets from audio_to_text.py. It has the same functionality of LhotseSpeechToTextBpeDataset but also yield speaker target tensor.
    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'spk_tar_id': NeuralType(('B','T'), LabelsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.cfg = cfg
        self.spk_tar_all_zero = self.cfg.get('spk_tar_all_zero',False)
        self.num_speakers = self.cfg.get('num_speakers', 4)
        self.num_sample_per_mel_frame = self.cfg.get('num_sample_per_mel_frame', 160)
        self.num_mel_frame_per_asr_frame = self.cfg.get('num_mel_frame_per_asr_frame', 8)
        self.shuffle_spk_mapping = self.cfg.get('shuffle_spk_mapping', True)
        self.spk_token_pattern= r'<\|spltoken\d+\|>'
        self.add_separater_audio = self.cfg.get('add_separater_audio', True)
        self.separater_freq = self.cfg.get('separater_freq', 500)
        self.separater_duration = self.cfg.get('separater_duration',1)
        self.separater_unvoice_ratio = self.cfg.get('separater_unvoice_ratio', 0.3)
        if self.add_separater_audio:
            self.separater_audio = get_separator_audio(self.separater_freq, self.cfg.sample_rate, self.separater_duration, self.separater_unvoice_ratio)
        self.add_special_token = self.cfg.get('add_special_token',True)
        if self.add_special_token:
            self.special_token=self.cfg.get('special_token','<|beep|>')
        self.fix_query_audio_end_time = self.cfg.get('fix_query_audio_end_time',False)
        if self.fix_query_audio_end_time:
            self.query_audio_end_time = 10
        self.inference_mode = self.cfg.get('inference_mode', False)
    

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:
        cuts, spk_mappings = shuffle_spk_mapping(cuts=cuts, num_speakers=self.num_speakers, shuffle_spk_mapping=self.shuffle_spk_mapping, pattern=self.spk_token_pattern)
        

        if self.inference_mode:
            spk_targets = [torch.transpose(torch.zeros(self.num_speakers, get_hidden_length_from_sample_length(cut.num_samples, self.num_sample_per_mel_frame, self.num_mel_frame_per_asr_frame)), 0, 1) for cut in cuts]
            audio, audio_lens, cuts = self.load_audio(cuts)
        else:
            query_cuts = CutSet.from_cuts(get_query_cut(c) for c in cuts)
            spk_targets = [torch.transpose(torch.as_tensor(self.speaker_to_target_tgt_speaker_0(c, q, self.num_speakers, self.num_sample_per_mel_frame, self.num_mel_frame_per_asr_frame, self.spk_tar_all_zero), dtype=torch.float32), 0, 1) for c, q in zip(cuts,query_cuts)]
            audio, audio_lens, cuts = self.load_audio(cuts)
            query_audio, query_audio_lens, query_cuts = self.load_audio(query_cuts)
            if self.add_separater_audio:
                if self.fix_query_audio_end_time:
                    pad_right = self.query_audio_end_time * self.cfg.sample_rate - query_audio.shape[1]
                    padded_query_audio = torch.nn.functional.pad(query_audio, (0,pad_right,0, 0))
                    audio = torch.cat([padded_query_audio, torch.tensor(self.separater_audio).repeat(len(cuts),1).to(audio.dtype), audio], axis = 1)
                    audio_lens = audio_lens + self.query_audio_end_time * self.cfg.sample_rate + self.separater_duration * self.cfg.sample_rate                

                else:
                    concat_list = []
                    for i in range(len(audio)):
                        concat_list.append(torch.cat([query_audio[i,:query_audio_lens[i]],torch.tensor(self.separater_audio).to(audio.dtype),audio[i,:audio_lens[i]]]))
                    audio = collate_vectors(concat_list, padding_value = 0)
                    audio_lens = audio_lens + query_audio_lens + self.separater_duration * self.cfg.sample_rate
            else:
                audio = torch.cat([query_audio, audio], axis = 1)
                audio_lens = audio_lens + query_audio_lens
        if self.add_special_token:
            tokens = [torch.as_tensor(self.tokenizer(self.special_token + ' ' + c.supervisions[0].text, c.supervisions[0].language)) for c in cuts]
        else:
            tokens = [torch.as_tensor(self.tokenizer(c.supervisions[0].text, c.supervisions[0].language)) for c in cuts]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=0)
        spk_targets = collate_matrices(spk_targets)
        return audio, audio_lens, tokens, token_lens, spk_targets, spk_mappings
    

    def speaker_to_target_tgt_speaker_0(
        self,
        a_cut,
        query,
        num_speakers: int = 4, 
        num_sample_per_mel_frame: int = 160, 
        num_mel_frame_per_asr_frame: int = 8, 
        spk_tar_all_zero: bool = False,
        boundary_segments: bool = False
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
        
        Returns:
            mask (Tensor): speaker mask with shape (num_speaker, hidden_lenght)
        '''
        # get cut-related segments from rttms
        rttms = SupervisionSet.from_rttm(a_cut.rttm_filepath)
        basename = os.path.basename(a_cut.rttm_filepath).replace('.rttm', '')
        if isinstance(a_cut, MixedCut):
            cut_list = [track.cut for track in a_cut.tracks] 
        elif isinstance(a_cut, MonoCut):
            cut_list = [a_cut]
        else:
            raise ValueError(f"Unsupported cut type type{cut}: only MixedCut and MonoCut are supported")
        
        segments_total = []
        for cut in cut_list:
            if boundary_segments: # segments with seg_start < total_end and seg_end > total_start are included
                segments_iterator = find_segments_from_rttm(recording_id=cut.recording_id, rttms=rttms, start_after=cut.start, end_before=cut.end)
            else: # segments with seg_start > total_start and seg_end < total_end are included
                segments_iterator = rttms.find(recording_id=cut.recording_id, start_after=cut.start, end_before=cut.end, adjust_offset=True)
            segments = [s for s in segments_iterator]
            segments_total.extend(segments)
        # apply arrival time sorting to the existing segments
        segments_total.sort(key = lambda rttm_sup: rttm_sup.start)

        seen = set()
        seen_add = seen.add
        if 'query_speaker_id' in cut.custom:
            speaker_lst = [cut.query_speaker_id] + [s.speaker for s in segments_total] #add query speaker as the first speaker
        else:
            speaker_lst = [s.speaker for s in segments_total]

        speaker_ats = [s for s in speaker_lst if not (s in seen or seen_add(s))]
        
        speaker_to_idx_map = {
                spk: idx
                for idx, spk in enumerate(speaker_ats)
        }
        #initialize mask matrices (num_speaker, encoder_hidden_len)
        if self.add_separater_audio:
            if self.fix_query_audio_end_time:
                encoder_hidden_len = get_hidden_length_from_sample_length(cut.num_samples +  self.query_audio_end_time * self.cfg.sample_rate + self.separater_duration * self.cfg.sample_rate, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
                query_hidden_len = get_hidden_length_from_sample_length(query.num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame) if 'query_speaker_id' in cut.custom else 0
                query_separater_hidden_len = get_hidden_length_from_sample_length(self.query_audio_end_time * self.cfg.sample_rate + self.separater_duration * self.cfg.sample_rate, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
                mask = np.zeros((num_speakers, encoder_hidden_len))
                mask[0,:query_hidden_len] = 1
                if not spk_tar_all_zero:
                    for rttm_sup in segments:
                        speaker_idx = speaker_to_idx_map[rttm_sup.speaker]
                        #only consider the first <num_speakers> speakers
                        if speaker_idx < 4:
                            st = (
                                        compute_num_samples(rttm_sup.start, cut.sampling_rate)
                                        if rttm_sup.start > 0
                                        else 0
                                    )
                            et = (
                                        compute_num_samples(rttm_sup.end, cut.sampling_rate)
                                        if rttm_sup.end < cut.duration
                                        else compute_num_samples(cut.duration, cut.sampling_rate)
                                    )                   
                            
                            #map start time (st) and end time (et) to encoded hidden location
                            st_encoder_loc = get_hidden_length_from_sample_length(st, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
                            et_encoder_loc = get_hidden_length_from_sample_length(et, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)

                            mask[speaker_idx, query_separater_hidden_len + st_encoder_loc: query_separater_hidden_len + et_encoder_loc] = 1

            else:
                encoder_hidden_len = get_hidden_length_from_sample_length(cut.num_samples +  query.num_samples + self.separater_duration * self.cfg.sample_rate, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
                separater_hidden_len = get_hidden_length_from_sample_length(self.separater_duration * self.cfg.sample_rate, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
                query_hidden_len = get_hidden_length_from_sample_length(query.num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame) if 'query_speaker_id' in cut.custom else 0
                mask = np.zeros((num_speakers, encoder_hidden_len))
                mask[0,:query_hidden_len] = 1
                if not spk_tar_all_zero:
                    for rttm_sup in segments:
                        speaker_idx = speaker_to_idx_map[rttm_sup.speaker]
                        #only consider the first <num_speakers> speakers
                        if speaker_idx < 4:
                            st = (
                                        compute_num_samples(rttm_sup.start, cut.sampling_rate)
                                        if rttm_sup.start > 0
                                        else 0
                                    )
                            et = (
                                        compute_num_samples(rttm_sup.end, cut.sampling_rate)
                                        if rttm_sup.end < cut.duration
                                        else compute_num_samples(cut.duration, cut.sampling_rate)
                                    )                   
                            
                            #map start time (st) and end time (et) to encoded hidden location
                            st_encoder_loc = get_hidden_length_from_sample_length(st, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
                            et_encoder_loc = get_hidden_length_from_sample_length(et, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)

                            mask[speaker_idx, query_hidden_len + separater_hidden_len + st_encoder_loc: query_hidden_len + separater_hidden_len + et_encoder_loc] = 1

        else:
            encoder_hidden_len = get_hidden_length_from_sample_length(cut.num_samples +  query.num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
            query_hidden_len = get_hidden_length_from_sample_length(query.num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame) if 'query_speaker_id' in cut.custom else 0
            mask = np.zeros((num_speakers, encoder_hidden_len))
            mask[0,:query_hidden_len] = 1

            if not spk_tar_all_zero:
                for rttm_sup in segments:
                    speaker_idx = speaker_to_idx_map[rttm_sup.speaker]
                    #only consider the first <num_speakers> speakers
                    if speaker_idx < 4:
                        st = (
                                    compute_num_samples(rttm_sup.start, cut.sampling_rate)
                                    if rttm_sup.start > 0
                                    else 0
                                )
                        et = (
                                    compute_num_samples(rttm_sup.end, cut.sampling_rate)
                                    if rttm_sup.end < cut.duration
                                    else compute_num_samples(cut.duration, cut.sampling_rate)
                                )                   
                        
                        #map start time (st) and end time (et) to encoded hidden location
                        st_encoder_loc = get_hidden_length_from_sample_length(st, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
                        et_encoder_loc = get_hidden_length_from_sample_length(et, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)

                        mask[speaker_idx, query_hidden_len + st_encoder_loc:query_hidden_len + et_encoder_loc] = 1

        return mask

