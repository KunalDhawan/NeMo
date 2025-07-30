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
import random
from typing import Dict, Optional, Tuple
import soundfile

import torch.utils.data
from lhotse.cut import MixedCut, MonoCut, MixTrack, PaddingCut
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors, collate_matrices
from lhotse.utils import compute_num_samples
from lhotse import SupervisionSet, SupervisionSegment, MonoCut, Recording, CutSet, AudioSource

import numpy as np

from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper
from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType

from nemo.collections.asr.parts.utils.asr_multispeaker_utils import (
    speaker_to_target, 
    get_hidden_length_from_sample_length, 
)

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
        self.load_audio = AudioSamples(fault_tolerant=True, num_workers=8)
        self.cfg = cfg
        self.spk_tar_all_zero = self.cfg.get('spk_tar_all_zero',False)
        self.num_speakers = self.cfg.get('num_speakers', 4)
        self.num_sample_per_mel_frame = self.cfg.get('num_sample_per_mel_frame', 160)
        self.num_mel_frame_per_asr_frame = self.cfg.get('num_mel_frame_per_asr_frame', 8)
        self.fixed_spk_id = self.cfg.get('fixed_spk_id', None)
        self.inference_mode = self.cfg.get('inference_mode', False)

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:

        # import time
        # start_time = time.time()
        audio, audio_lens, cuts = self.load_audio(cuts)
        # end_time = time.time()
        # print(f"====[  Audio Loading Time ] ==== time taken: {end_time - start_time:.3f} seconds")

        tokens = []
        spk_targets = []

        if self.inference_mode:
            
            speaker_targets = [speaker_to_target(cut, self.num_speakers, self.num_sample_per_mel_frame, self.num_mel_frame_per_asr_frame, self.spk_tar_all_zero) for cut in cuts]
            spk_targets = collate_matrices(speaker_targets, padding_value=0)
            return audio, audio_lens, None, None, spk_targets

        for idx, cut in enumerate(cuts):
            non_padding_cuts = []
            if isinstance(cut, MonoCut):
                non_padding_cuts.append(cut)
            elif isinstance(cut, MixedCut):
                if len(cut.tracks) == 2 and isinstance(cut.tracks[1].cut, PaddingCut):
                    non_padding_cuts.append(cut.tracks[0].cut)
                else:
                    for track in cut.tracks:
                        if isinstance(track.cut, MonoCut):
                            non_padding_cuts.append(track.cut)

            if "audiomix" in cut.id and isinstance(cut, MixedCut):
                num_speakers_in_cut = int(cut.id.split("nspk")[-1])
                texts = ['' for _ in range(num_speakers_in_cut)]
                for track in cut.tracks:
                    if len(track.cut.supervisions) > 0 and track.cut.supervisions[0].speaker is not None:
                        texts[track.cut.supervisions[0].speaker] += f"{track.cut.supervisions[0].text} "
                texts = [text.strip() for text in texts]
                speaker_targets = speaker_to_target(cut, self.num_speakers, self.num_sample_per_mel_frame, self.num_mel_frame_per_asr_frame, self.spk_tar_all_zero)
                speaker_targets = speaker_targets.transpose(0, 1)[:len(texts)]
            else:
                if hasattr(non_padding_cuts[0], 'text') and '<|spltoken0|>' in non_padding_cuts[0].text:
                    # the previous data style with speaker tokens
                    texts = self.split_text(non_padding_cuts[0].custom['text'])
                    speaker_targets = speaker_to_target(cut, self.num_speakers, self.num_sample_per_mel_frame, self.num_mel_frame_per_asr_frame, self.spk_tar_all_zero)
                    speaker_targets = speaker_targets.transpose(0, 1)[:len(texts)]
                else:
                    # new channel
                    speaker_targets = [non_padding_cut.vad_target for non_padding_cut in non_padding_cuts if hasattr(non_padding_cut, 'vad_target')]
                    speaker_targets = torch.stack(speaker_targets)
                    texts = [non_padding_cut.custom['text'] for non_padding_cut in non_padding_cuts if hasattr(non_padding_cut, 'text')]

            if speaker_targets.shape[0] > 0:
                # multi-speaker
                target_speaker_id = random.choice(range(speaker_targets.shape[0]))
                text = texts[target_speaker_id]
                speaker_target = speaker_targets[target_speaker_id]
            else:
                # single speaker 
                text = texts[0]
                speaker_target = torch.ones((get_hidden_length_from_sample_length(cut.num_samples) ))
            
            tokens.append(torch.as_tensor(self.tokenizer(text, cut.supervisions[0].language)))
            spk_targets.append(speaker_target)
        
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=0)
        spk_targets = collate_vectors(spk_targets, padding_value=0)
        
        return audio, audio_lens, tokens, token_lens, spk_targets

    def split_text(self, text, speaker_token='<|spltoken*|>'):
        """
        Split text by speaker tokens and group text from the same speaker.
        
        Args:
            text (str): Input text with speaker tokens
            speaker_token (str): Base speaker token pattern, where * will be replaced with numbers
        
        Returns:
            list[str]: List of concatenated text for each speaker. Returns [text] if no speaker tokens found.
        """
        # Replace * with a digit in the pattern
        pattern = speaker_token.replace('*', r'\d+').replace('|', '\\|')
        # pattern = '(<\|spltoken\d+\|>)'
        
        # Split text by speaker tokens
        segments = re.split(rf'({pattern})', text.strip())

        spks = []
        spk2text = {}
        
        for i in range(1, len(segments), 2):  # Step by 2 to skip over text between speaker tags
            speaker_tag = segments[i]
            words = segments[i + 1]
            if speaker_tag not in spks:
                spk2text[speaker_tag] = words.strip()
                spks.append(speaker_tag)
            else:
                spk2text[speaker_tag] += ' ' + words.strip()
            
        return [spk2text[spk] for spk in spks]