# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import os
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from nemo.collections.asr.data.audio_to_text_lhotse_prompted import canary_prompt
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from nemo.collections.asr.parts.utils.audio_utils import get_samples
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import LengthsType, MelSpectrogramType, NeuralType
from nemo.collections.asr.parts.utils.asr_tgtspeaker_utils import get_separator_audio

# Minimum number of tokens required to assign a LCS merge step, otherwise ignore and
# select all i-1 and ith buffer tokens to merge.
MIN_MERGE_SUBSEQUENCE_LEN = 1


class CacheAwareStreamingAudioSpkBuffer:
    """
    A buffer to be used for cache-aware streaming. It can load a single or multiple audio files/processed signals, split them in chunks and return one on one.
    It can be used to simulate streaming audio or audios.
    """

    def __init__(self, model, online_normalization=None, pad_and_drop_preencoded=False):
        '''
        Args:
            model: An ASR model.
            online_normalization (bool): whether to perform online normalization per chunk or normalize the whole audio before chunking
            pad_and_drop_preencoded (bool): if true pad first audio chunk and always drop preencoded
        '''
        self.model = model
        self.buffer = None
        self.buffer_idx = 0
        self.buffer_audio_idx = 0
        self.streams_length = None
        self.step = 0
        self.pad_and_drop_preencoded = pad_and_drop_preencoded

        self.online_normalization = online_normalization
        if not isinstance(model.encoder, StreamingEncoder):
            raise ValueError(
                "The model's encoder is not inherited from StreamingEncoder, and likely not to support streaming!"
            )
        if model.encoder.streaming_cfg is None:
            model.encoder.setup_streaming_params()
        self.streaming_cfg = model.encoder.streaming_cfg

        self.input_features = model.encoder._feat_in

        self.preprocessor = self.extract_preprocessor()

        if hasattr(model.encoder, "pre_encode") and hasattr(model.encoder.pre_encode, "get_sampling_frames"):
            self.sampling_frames = model.encoder.pre_encode.get_sampling_frames()
        else:
            self.sampling_frames = None

    @staticmethod
    def get_audio_sample_len_from_frame(chunk_size):
        return int(chunk_size * 160)

    def __iter__(self):
        while True:
            if self.buffer_idx >= self.buffer.size(-1):
                return
            if self.buffer_idx == 0 and isinstance(self.streaming_cfg.chunk_size, list):
                if self.pad_and_drop_preencoded:
                    chunk_size = self.streaming_cfg.chunk_size[1]
                else:
                    chunk_size = self.streaming_cfg.chunk_size[0]
            else:
                chunk_size = (
                    self.streaming_cfg.chunk_size[1]
                    if isinstance(self.streaming_cfg.chunk_size, list)
                    else self.streaming_cfg.chunk_size
                )

            if self.buffer_idx == 0 and isinstance(self.streaming_cfg.shift_size, list):
                if self.pad_and_drop_preencoded:
                    shift_size = self.streaming_cfg.shift_size[1]
                else:
                    shift_size = self.streaming_cfg.shift_size[0]
            else:
                shift_size = (
                    self.streaming_cfg.shift_size[1]
                    if isinstance(self.streaming_cfg.shift_size, list)
                    else self.streaming_cfg.shift_size
                )

            audio_chunk = self.buffer[:, :, self.buffer_idx : self.buffer_idx + chunk_size]

            chunk_size_audio = self.get_audio_sample_len_from_frame(chunk_size)

            raw_audio_chunk = self.buffer_audio[:,self.buffer_audio_idx: self.buffer_audio_idx + chunk_size_audio]

            if self.sampling_frames is not None:
                # checking to make sure the audio chunk has enough frames to produce at least one output after downsampling
                if self.buffer_idx == 0 and isinstance(self.sampling_frames, list):
                    cur_sampling_frames = self.sampling_frames[0]
                else:
                    cur_sampling_frames = (
                        self.sampling_frames[1] if isinstance(self.sampling_frames, list) else self.sampling_frames
                    )
                if audio_chunk.size(-1) < cur_sampling_frames:
                    return

            # Adding the cache needed for the pre-encoder part of the model to the chunk
            # if there is not enough frames to be used as the pre-encoding cache, zeros would be added
            zeros_pads = None
            zeros_pads_audio = None
            if self.buffer_idx == 0 and isinstance(self.streaming_cfg.pre_encode_cache_size, list):
                if self.pad_and_drop_preencoded:
                    cache_pre_encode_num_frames = self.streaming_cfg.pre_encode_cache_size[1]
                else:
                    cache_pre_encode_num_frames = self.streaming_cfg.pre_encode_cache_size[0]
                cache_pre_encode = torch.zeros(
                    (audio_chunk.size(0), self.input_features, cache_pre_encode_num_frames),
                    device=audio_chunk.device,
                    dtype=audio_chunk.dtype,
                )
                cache_pre_encode_audio = torch.zeros(
                    (raw_audio_chunk.size(0), self.get_audio_sample_len_from_frame(cache_pre_encode_num_frames)),
                    device=audio_chunk.device,
                    dtype=audio_chunk.dtype,
                )
            else:
                if isinstance(self.streaming_cfg.pre_encode_cache_size, list):
                    pre_encode_cache_size = self.streaming_cfg.pre_encode_cache_size[1]
                    pre_encode_cache_size_audio = self.get_audio_sample_len_from_frame(self.streaming_cfg.pre_encode_cache_size[1])
                else:
                    pre_encode_cache_size = self.streaming_cfg.pre_encode_cache_size

                start_pre_encode_cache = self.buffer_idx - pre_encode_cache_size
                start_pre_encode_cache_audio = self.buffer_audio_idx - pre_encode_cache_size_audio
                if start_pre_encode_cache < 0:
                    start_pre_encode_cache = 0
                if start_pre_encode_cache_audio < 0:
                    start_pre_encode_cache_audio = 0
                cache_pre_encode = self.buffer[:, :, start_pre_encode_cache : self.buffer_idx]
                cache_pre_encode_audio = self.buffer_audio[:, start_pre_encode_cache_audio: self.buffer_audio_idx]
                if cache_pre_encode.size(-1) < pre_encode_cache_size:
                    zeros_pads = torch.zeros(
                        (
                            audio_chunk.size(0),
                            audio_chunk.size(-2),
                            pre_encode_cache_size - cache_pre_encode.size(-1),
                        ),
                        device=audio_chunk.device,
                        dtype=audio_chunk.dtype,
                    )
                if cache_pre_encode_audio.size(-1) < pre_encode_cache_size_audio:
                    zeros_pads_audio = torch.zeros(
                        (
                            raw_audio_chunk.size(0),
                            pre_encode_cache_size - cache_pre_encode_audio.size(-1),
                        ),
                        device=raw_audio_chunk.device,
                        dtype=raw_audio_chunk.dtype,
                    )

            added_len = cache_pre_encode.size(-1)
            added_len_audio = cache_pre_encode_audio.size(-1)
            audio_chunk = torch.cat((cache_pre_encode, audio_chunk), dim=-1)
            raw_audio_chunk = torch.cat((cache_pre_encode_audio, raw_audio_chunk), dim = -1)

            if self.online_normalization:
                audio_chunk, x_mean, x_std = normalize_batch(
                    x=audio_chunk,
                    seq_len=torch.tensor([audio_chunk.size(-1)] * audio_chunk.size(0)),
                    normalize_type=self.model_normalize_type,
                )

            if zeros_pads is not None:
                # TODO: check here when zero_pads is not None and added_len is already non-zero
                audio_chunk = torch.cat((zeros_pads, audio_chunk), dim=-1)
                added_len += zeros_pads.size(-1)
            if zeros_pads_audio is not None:
                raw_audio_chunk = torch.cat((zeros_pads_audio, raw_audio_chunk), dim=-1)
                added_len_audio += zeros_pads_audio.size(-1)

            max_chunk_lengths = self.streams_length - self.buffer_idx
            max_chunk_lengths = max_chunk_lengths + added_len
            chunk_lengths = torch.clamp(max_chunk_lengths, min=0, max=audio_chunk.size(-1))
            max_chunk_lengths_audio = self.streams_length_audio - self.buffer_audio_idx
            max_chunk_lengths_audio = max_chunk_lengths_audio + added_len_audio
            chunk_lengths_audio = torch.clamp(max_chunk_lengths_audio, min=0, max=raw_audio_chunk.size(-1))

            self.buffer_idx += shift_size
            self.buffer_audio_idx += self.get_audio_sample_len_from_frame(shift_size)
            self.step += 1
            yield audio_chunk, chunk_lengths, raw_audio_chunk, chunk_lengths_audio, self.buffer_audio_idx - self.get_audio_sample_len_from_frame(shift_size), self.buffer_audio_idx

    def is_buffer_empty(self):
        if self.buffer_idx >= self.buffer.size(-1):
            return True
        else:
            return False

    def __len__(self):
        return len(self.buffer)

    def reset_buffer(self):
        self.buffer = None
        self.buffer_idx = 0
        self.streams_length = None
        self.step = 0

    def reset_buffer_pointer(self):
        self.buffer_idx = 0
        self.step = 0

    def extract_preprocessor(self):
        cfg = copy.deepcopy(self.model._cfg)
        self.model_normalize_type = cfg.preprocessor.normalize
        OmegaConf.set_struct(cfg.preprocessor, False)
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        if self.online_normalization:
            cfg.preprocessor.normalize = "None"

        preprocessor = self.model.from_config_dict(cfg.preprocessor)
        return preprocessor.to(self.get_model_device())

    def append_audio_file(self, audio_filepath, stream_id=-1):
        audio = get_samples(audio_filepath)
        processed_signal, processed_signal_length, stream_id = self.append_audio(audio, stream_id)
        return processed_signal, processed_signal_length, stream_id

    def append_audio(self, audio, stream_id=-1):
        processed_signal, processed_signal_length, audio_signal, audio_signal_len = self.preprocess_audio(audio)
        processed_signal, processed_signal_length, stream_id = self.append_processed_signal(
            processed_signal, audio_signal, stream_id
        )
        return processed_signal, processed_signal_length, stream_id

    def append_processed_signal(self, processed_signal, audio_signal, stream_id=-1):
        #processed signal + raw audio signal
        processed_signal_length = torch.tensor(processed_signal.size(-1), device=processed_signal.device)
        audio_signal_length = torch.tensor(audio_signal.size(-1), device=processed_signal.device)
        if stream_id >= 0 and (self.streams_length is not None and stream_id >= len(self.streams_length)):
            raise ValueError("Not valid stream_id!")
        if self.buffer is None:
            if stream_id >= 0:
                raise ValueError("stream_id can not be specified when there is no stream.")
            self.buffer = processed_signal
            self.buffer_audio = audio_signal
            self.streams_length = torch.tensor([processed_signal_length], device=processed_signal.device)
            self.streams_length_audio = torch.tensor([audio_signal_length], device=processed_signal.device)
        else:
            if self.buffer.size(1) != processed_signal.size(1):
                raise ValueError("Buffer and the processed signal have different dimensions for processed signal!")
            if stream_id < 0:
                self.buffer = torch.nn.functional.pad(self.buffer, pad=(0, 0, 0, 0, 0, 1))
                self.streams_length = torch.cat(
                    (self.streams_length, torch.tensor([0], device=self.streams_length.device)), dim=-1
                )
                self.buffer_audio = torch.nn.functional.pad(self.buffer_audio, pad=(0, 0, 0, 1))
                self.streams_length_audio = torch.cat(
                    (self.streams_length_audio, torch.tensor([0], device=self.streams_length_audio.device)), dim=-1
                )
                stream_id = len(self.streams_length) - 1
            needed_len = self.streams_length[stream_id] + processed_signal_length
            needed_len_audio = self.streams_length_audio[stream_id] + audio_signal_length
            if needed_len > self.buffer.size(-1):
                self.buffer = torch.nn.functional.pad(self.buffer, pad=(0, needed_len - self.buffer.size(-1)))
            if needed_len_audio > self.buffer_audio.size(-1):
                self.buffer_audio = torch.nn.functional.pad(self.buffer_audio, pad=(0, needed_len_audio - self.buffer_audio.size(-1)))

            self.buffer[
                stream_id, :, self.streams_length[stream_id] : self.streams_length[stream_id] + processed_signal_length
            ] = processed_signal
            self.buffer_audio[
                stream_id, :, self.streams_length_audio[stream_id] : self.streams_length_audio[stream_id] + audio_signal_length
            ] = audio_signal
            self.streams_length[stream_id] = self.streams_length[stream_id] + processed_signal.size(-1)
            self.streams_length_audio[stream_id] = self.streams_length_audio[stream_id] + audio_signal.size(-1)

        if self.online_normalization:
            processed_signal, x_mean, x_std = normalize_batch(
                x=processed_signal,
                seq_len=torch.tensor([processed_signal_length]),
                normalize_type=self.model_normalize_type,
            )

        return processed_signal, processed_signal_length, stream_id

    def get_model_device(self):
        return self.model.device

    def preprocess_audio(self, audio, device=None):
        if device is None:
            device = self.get_model_device()
        audio_signal = torch.from_numpy(audio).unsqueeze_(0).to(device)
        audio_signal_len = torch.Tensor([audio.shape[0]]).to(device)
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=audio_signal, length=audio_signal_len
        )
        return processed_signal, processed_signal_length, audio_signal, audio_signal_len

    def get_all_audios(self):
        processed_signal = self.buffer
        if self.online_normalization:
            processed_signal, x_mean, x_std = normalize_batch(
                x=processed_signal,
                seq_len=torch.tensor(self.streams_length),
                normalize_type=self.model_normalize_type,
            )
        return processed_signal, self.streams_length
    

class CacheAwareStreamingAudioTgtSpkBuffer(CacheAwareStreamingAudioSpkBuffer):
    def __init__(self, model, online_normalization=None, pad_and_drop_preencoded=False):
        super().__init__(model = model, online_normalization = online_normalization, pad_and_drop_preencoded = pad_and_drop_preencoded)
    
    def append_audio_file(self, sample, stream_id=-1):
        audio_filepath = sample['audio_filepath']
        audio = get_samples(audio_filepath) #get sub clip
        audio = audio[int(sample['offset'] * 16000): int((sample['offset'] + sample['duration']) * 16000)]
        #prepend query audio
        query_audio_filepath = sample['query_audio_filepath']
        query_audio = get_samples(query_audio_filepath)
        query_audio = query_audio[int(sample['query_offset'] * 16000) : int((sample['query_offset'] + sample['query_duration']) * 16000)]
        #separater audio
        separater = get_separator_audio(freq=500, sr=16000, duration=1, ratio=0.3)
        separater = np.float32(separater)
        audio = np.concatenate([query_audio, separater, audio])
        processed_signal, processed_signal_length, stream_id = self.append_audio(audio, stream_id)
        return processed_signal, processed_signal_length, stream_id

    def append_audio(self, audio, stream_id=-1):
        processed_signal, processed_signal_length, audio_signal, audio_signal_len = self.preprocess_audio(audio)
        processed_signal, processed_signal_length, stream_id = self.append_processed_signal(
            processed_signal, audio_signal, stream_id
        )
        return processed_signal, processed_signal_length, stream_id


    def __iter__(self):
        while True:
            if self.buffer_idx >= self.buffer.size(-1):
                return
            if self.buffer_idx == 0 and isinstance(self.streaming_cfg.chunk_size, list):
                if self.pad_and_drop_preencoded:
                    chunk_size = self.streaming_cfg.chunk_size[1]
                else:
                    chunk_size = self.streaming_cfg.chunk_size[0]
            else:
                chunk_size = (
                    self.streaming_cfg.chunk_size[1]
                    if isinstance(self.streaming_cfg.chunk_size, list)
                    else self.streaming_cfg.chunk_size
                )

            if self.buffer_idx == 0 and isinstance(self.streaming_cfg.shift_size, list):
                if self.pad_and_drop_preencoded:
                    shift_size = self.streaming_cfg.shift_size[1]
                else:
                    shift_size = self.streaming_cfg.shift_size[0]
            else:
                shift_size = (
                    self.streaming_cfg.shift_size[1]
                    if isinstance(self.streaming_cfg.shift_size, list)
                    else self.streaming_cfg.shift_size
                )

            audio_chunk = self.buffer[:, :, self.buffer_idx : self.buffer_idx + chunk_size]

            chunk_size_audio = self.get_audio_sample_len_from_frame(chunk_size)

            raw_audio_chunk = self.buffer_audio[:,self.buffer_audio_idx: self.buffer_audio_idx + chunk_size_audio]

            if self.sampling_frames is not None:
                # checking to make sure the audio chunk has enough frames to produce at least one output after downsampling
                if self.buffer_idx == 0 and isinstance(self.sampling_frames, list):
                    cur_sampling_frames = self.sampling_frames[0]
                else:
                    cur_sampling_frames = (
                        self.sampling_frames[1] if isinstance(self.sampling_frames, list) else self.sampling_frames
                    )
                if audio_chunk.size(-1) < cur_sampling_frames:
                    return

            # Adding the cache needed for the pre-encoder part of the model to the chunk
            # if there is not enough frames to be used as the pre-encoding cache, zeros would be added
            zeros_pads = None
            zeros_pads_audio = None
            if self.buffer_idx == 0 and isinstance(self.streaming_cfg.pre_encode_cache_size, list):
                if self.pad_and_drop_preencoded:
                    cache_pre_encode_num_frames = self.streaming_cfg.pre_encode_cache_size[1]
                else:
                    cache_pre_encode_num_frames = self.streaming_cfg.pre_encode_cache_size[0]
                cache_pre_encode = torch.zeros(
                    (audio_chunk.size(0), self.input_features, cache_pre_encode_num_frames),
                    device=audio_chunk.device,
                    dtype=audio_chunk.dtype,
                )
                cache_pre_encode_audio = torch.zeros(
                    (raw_audio_chunk.size(0), self.get_audio_sample_len_from_frame(cache_pre_encode_num_frames)),
                    device=audio_chunk.device,
                    dtype=audio_chunk.dtype,
                )
            else:
                if isinstance(self.streaming_cfg.pre_encode_cache_size, list):
                    pre_encode_cache_size = self.streaming_cfg.pre_encode_cache_size[1]
                    pre_encode_cache_size_audio = self.get_audio_sample_len_from_frame(self.streaming_cfg.pre_encode_cache_size[1])
                else:
                    pre_encode_cache_size = self.streaming_cfg.pre_encode_cache_size

                start_pre_encode_cache = self.buffer_idx - pre_encode_cache_size
                start_pre_encode_cache_audio = self.buffer_audio_idx - pre_encode_cache_size_audio
                if start_pre_encode_cache < 0:
                    start_pre_encode_cache = 0
                if start_pre_encode_cache_audio < 0:
                    start_pre_encode_cache_audio = 0
                cache_pre_encode = self.buffer[:, :, start_pre_encode_cache : self.buffer_idx]
                cache_pre_encode_audio = self.buffer_audio[:, start_pre_encode_cache_audio: self.buffer_audio_idx]
                if cache_pre_encode.size(-1) < pre_encode_cache_size:
                    zeros_pads = torch.zeros(
                        (
                            audio_chunk.size(0),
                            audio_chunk.size(-2),
                            pre_encode_cache_size - cache_pre_encode.size(-1),
                        ),
                        device=audio_chunk.device,
                        dtype=audio_chunk.dtype,
                    )
                if cache_pre_encode_audio.size(-1) < pre_encode_cache_size_audio:
                    zeros_pads_audio = torch.zeros(
                        (
                            raw_audio_chunk.size(0),
                            pre_encode_cache_size - cache_pre_encode_audio.size(-1),
                        ),
                        device=raw_audio_chunk.device,
                        dtype=raw_audio_chunk.dtype,
                    )

            added_len = cache_pre_encode.size(-1)
            added_len_audio = cache_pre_encode_audio.size(-1)
            audio_chunk = torch.cat((cache_pre_encode, audio_chunk), dim=-1)
            raw_audio_chunk = torch.cat((cache_pre_encode_audio, raw_audio_chunk), dim = -1)

            if self.online_normalization:
                audio_chunk, x_mean, x_std = normalize_batch(
                    x=audio_chunk,
                    seq_len=torch.tensor([audio_chunk.size(-1)] * audio_chunk.size(0)),
                    normalize_type=self.model_normalize_type,
                )

            if zeros_pads is not None:
                # TODO: check here when zero_pads is not None and added_len is already non-zero
                audio_chunk = torch.cat((zeros_pads, audio_chunk), dim=-1)
                added_len += zeros_pads.size(-1)
            if zeros_pads_audio is not None:
                raw_audio_chunk = torch.cat((zeros_pads_audio, raw_audio_chunk), dim=-1)
                added_len_audio += zeros_pads_audio.size(-1)

            max_chunk_lengths = self.streams_length - self.buffer_idx
            max_chunk_lengths = max_chunk_lengths + added_len
            chunk_lengths = torch.clamp(max_chunk_lengths, min=0, max=audio_chunk.size(-1))
            max_chunk_lengths_audio = self.streams_length_audio - self.buffer_audio_idx
            max_chunk_lengths_audio = max_chunk_lengths_audio + added_len_audio
            chunk_lengths_audio = torch.clamp(max_chunk_lengths_audio, min=0, max=raw_audio_chunk.size(-1))

            self.buffer_idx += shift_size
            self.buffer_audio_idx += self.get_audio_sample_len_from_frame(shift_size)
            self.step += 1
            yield audio_chunk, chunk_lengths, raw_audio_chunk, chunk_lengths_audio, self.buffer_audio_idx - self.get_audio_sample_len_from_frame(shift_size), self.buffer_audio_idx