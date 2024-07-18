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

from typing import Dict, Optional, Tuple

import torch.utils.data
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors, collate_matrices
from lhotse.utils import compute_num_samples
from lhotse import SupervisionSet

import numpy as np

from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper
from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType


class LhotseSpeechToTextSpkBpeDataset(torch.utils.data.Dataset):
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
        

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:
        spk_targets = [torch.transpose(torch.as_tensor(self.speaker_to_target(c, self.num_speakers, self.num_sample_per_mel_frame, self.num_mel_frame_per_asr_frame, self.spk_tar_all_zero), dtype=torch.float32), 0, 1) for c in cuts]
        audio, audio_lens, cuts = self.load_audio(cuts)
        tokens = [torch.as_tensor(self.tokenizer(c.supervisions[0].text, c.supervisions[0].language)) for c in cuts]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=0)
        spk_targets = collate_matrices(spk_targets)


        return audio, audio_lens, tokens, token_lens, spk_targets
    

    def speaker_to_target(self, cut, num_speakers: int = 4, num_sample_per_mel_frame: int = 160, num_mel_frame_per_asr_frame: int = 8, spk_tar_all_zero: bool = False):
        '''
        get rttm samples corresponding to one cut, generate speaker mask numpy.ndarray with shape (num_speaker, hidden_length)

        Args:
            cut: An audio cut
            num_speakers: max number of speakers for all cuts ("mask" dim0), 4 by default
            num_sample_per_mel_frame: number of sample per mel frame, sample_rate / 1000 * window_stride, 160 by default (10ms window stride)
            num_mel_frame_per_asr_frame: encoder subsampling_factor, 8 by default
            spk_tar_all_zero: set to True gives all zero "mask"
        
        Returns:
            mask: speaker mask with shape (num_speaker, hidden_length)

        '''

        #get cut-related segments from rttm of this cut's original recording
        rttms = SupervisionSet.from_rttm(cut.rttm_filepath)
        segments_iterator = rttms.find(recording_id=cut.recording_id, start_after=cut.start, end_before=cut.end, adjust_offset=True)

        segments = [s for s in segments_iterator]

        #generate speaker id according to arrival time
        segments.sort(key = lambda rttm_sup: rttm_sup.start)

        seen = set()
        seen_add = seen.add
        speaker_ats = [s.speaker for s in segments if not (s.speaker in seen or seen_add(s.speaker))]
        
        speaker_to_idx_map = {
                spk: idx
                for idx, spk in enumerate(speaker_ats)
        }

        #initialize mask matrices (num_speaker, encoder_hidden_len)
        encoder_hidden_len = self.get_hidden_length_from_sample_length(cut.num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
        mask = np.zeros((num_speakers, encoder_hidden_len))

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
                                else compute_num_samples(rttm_sup.duration, cut.sampling_rate)
                            )                   
                    
                    #map start time (st) and end time (et) to encoded hidden location
                    st_encoder_loc = self.get_hidden_length_from_sample_length(st, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)
                    et_encoder_loc = self.get_hidden_length_from_sample_length(et, num_sample_per_mel_frame, num_mel_frame_per_asr_frame)


                    mask[speaker_idx, st_encoder_loc:et_encoder_loc] = 1

        return mask

    @staticmethod
    def get_hidden_length_from_sample_length(num_samples: int, num_sample_per_mel_frame: int = 160, num_mel_frame_per_asr_frame: int = 8):
        return int(np.ceil(np.ceil((num_samples + 1) / num_sample_per_mel_frame) / num_mel_frame_per_asr_frame))
