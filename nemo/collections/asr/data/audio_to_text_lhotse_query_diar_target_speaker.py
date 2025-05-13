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
from lhotse.utils import compute_num_samples, uuid4
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
    get_query_cut,
    speaker_to_target_w_query
)

class LhotseSpeechToTextQueryDiarTgtSpkBpeDataset(torch.utils.data.Dataset):
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
            spk_targets = [torch.transpose(torch.as_tensor(speaker_to_target_w_query(
                c, q, 
                self.add_separater_audio,
                self.separater_duration,
                self.num_speakers, 
                self.num_sample_per_mel_frame, 
                self.num_mel_frame_per_asr_frame, 
                self.spk_tar_all_zero), 
                dtype=torch.float32), 0, 1) for c, q in zip(cuts,query_cuts)]
            audio, audio_lens, cuts = self.load_audio(cuts)
            query_audio, query_audio_lens, query_cuts = self.load_audio(query_cuts)

            if self.add_separater_audio:
                concat_list = []
                for i in range(len(audio)):
                    concat_list.append(torch.cat([query_audio[i,:query_audio_lens[i]],torch.tensor(self.separater_audio).to(audio.dtype),audio[i,:audio_lens[i]]]))
                audio_w_query = collate_vectors(concat_list, padding_value = 0)
                audio_w_query_lens = audio_lens + query_audio_lens + self.separater_duration * self.cfg.sample_rate
            else:
                concat_list = []
                for i in range(len(audio)):
                    concat_list.append(torch.cat([query_audio[i,:query_audio_lens[i]],audio[i,:audio_lens[i]]]))
                audio_w_query = collate_vectors(concat_list, padding_value = 0)
                audio_w_query_lens = audio_lens + query_audio_lens
        if self.add_special_token:
            tokens = [torch.as_tensor(self.tokenizer(self.special_token + ' ' + c.supervisions[0].text, c.supervisions[0].language)) for c in cuts]
        else:
            tokens = [torch.as_tensor(self.tokenizer(c.supervisions[0].text, c.supervisions[0].language)) for c in cuts]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=0)
        spk_targets = collate_matrices(spk_targets)
        return audio, audio_lens, audio_w_query, audio_w_query_lens, tokens, token_lens, spk_targets, spk_mappings
    

