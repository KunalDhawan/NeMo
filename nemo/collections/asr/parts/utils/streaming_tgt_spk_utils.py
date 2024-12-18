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
import soundfile as sf
import librosa

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from nemo.collections.asr.data.audio_to_text_lhotse_prompted import PromptedAudioToTextMiniBatch
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from nemo.collections.asr.parts.preprocessing.segment import get_samples, AudioSegment
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import LengthsType, MelSpectrogramType, NeuralType
from nemo.collections.asr.parts.utils.streaming_utils import *
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_hidden_length_from_sample_length

from nemo.collections.asr.parts.utils.asr_tgtspeaker_utils import (
    get_separator_audio,
)


# class for streaming frame-based ASR
# 1) use reset() method to reset FrameASR's state
# 2) call transcribe(frame) to do ASR on
#    contiguous signal's frames
class FrameBatchASR_tgt_spk:
    """
    class for streaming frame-based ASR use reset() method to reset FrameASR's
    state call transcribe(frame) to do ASR on contiguous signal's frames
    """

    def __init__(
        self,
        asr_model,
        frame_len=1.6,
        total_buffer=4.0,
        batch_size=4,
        pad_to_buffer_len=True,
    ):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        self.frame_bufferer = FeatureFrameBufferer_tgt_spk(
            asr_model=asr_model,
            frame_len=frame_len,
            batch_size=batch_size,
            total_buffer=total_buffer,
            pad_to_buffer_len=pad_to_buffer_len,
        )

        self.asr_model = asr_model
        self.decoder = getattr(asr_model, "decoder", None)

        self.batch_size = batch_size
        self.all_logits = []
        self.all_preds = []

        self.unmerged = []

        if self.decoder is None:
            self.blank_id = len(asr_model.tokenizer.vocabulary)
        elif hasattr(asr_model.decoder, "vocabulary"):
            self.blank_id = len(asr_model.decoder.vocabulary)
        else:
            self.blank_id = len(asr_model.joint.vocabulary)
        self.tokenizer = asr_model.tokenizer
        self.toks_unmerged = []
        self.frame_buffers = []
        self.reset()
        cfg = copy.deepcopy(asr_model._cfg)
        self.cfg = cfg
        self.frame_len = frame_len
        OmegaConf.set_struct(cfg.preprocessor, False)

        # some changes for streaming scenario
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"
        self.raw_preprocessor = ASRModel.from_config_dict(cfg.preprocessor)
        self.raw_preprocessor.to(asr_model.device)
        self.preprocessor = self.raw_preprocessor

    def reset(self):
        """
        Reset frame_history and decoder's state
        """
        self.prev_char = ''
        self.unmerged = []
        self.data_layer = AudioBuffersDataLayer()
        self.data_loader = DataLoader(self.data_layer, batch_size=self.batch_size, collate_fn=speech_collate_fn)
        self.all_logits = []
        self.all_preds = []
        self.toks_unmerged = []
        self.frame_buffers = []
        self.frame_bufferer.reset()

    def get_partial_samples(self, audio_file: str, offset: float, duration: float, target_sr: int = 16000, dtype: str = 'float32'):
        try:
            with sf.SoundFile(audio_file, 'r') as f:
                start = int(offset * target_sr)
                f.seek(start)
                end = int((offset + duration) * target_sr)
                samples = f.read(dtype=dtype, frames = end - start)
                if f.samplerate != target_sr:
                    samples = librosa.core.resample(samples, orig_sr=f.samplerate, target_sr=target_sr)
                samples = samples.transpose()
        except:
            raise ValueError('Frame exceed audio')
        return samples

    def read_audio_file(self, audio_filepath: str, offset, duration, query_audio_file, query_offset, query_duration, separater_freq, separater_duration, separater_unvoice_ratio,delay, model_stride_in_secs):
        # samples = get_samples(audio_filepath)
        # rewrite loading audio function to support partial audio
        samples = self.get_partial_samples(audio_filepath, offset, duration)
        samples = np.pad(samples, (0, int(delay * model_stride_in_secs * self.asr_model._cfg.sample_rate)))
        # query related variables
        query_samples = self.get_partial_samples(query_audio_file, query_offset, query_duration)
        separater_audio = get_separator_audio(separater_freq, self.asr_model._cfg.sample_rate, separater_duration, separater_unvoice_ratio)
        query_samples = np.concatenate([query_samples, separater_audio])
        frame_reader = AudioFeatureIterator_tgt_spk(samples, query_samples, self.frame_len, self.raw_preprocessor, self.asr_model.device)
        self.query_pred_len = get_hidden_length_from_sample_length(len(query_samples), 160, 8)
        self.set_frame_reader(frame_reader)

    def set_frame_reader(self, frame_reader):
        self.frame_bufferer.set_frame_reader(frame_reader)

    @torch.no_grad()
    def infer_logits(self, keep_logits=False):
        frame_buffers = self.frame_bufferer.get_buffers_batch()

        while len(frame_buffers) > 0:
            self.frame_buffers += frame_buffers[:]
            self.data_layer.set_signal(frame_buffers[:])
            self._get_batch_preds(keep_logits)
            frame_buffers = self.frame_bufferer.get_buffers_batch()

    @torch.no_grad()
    def _get_batch_preds(self, keep_logits=False):
        device = self.asr_model.device
        for batch in iter(self.data_loader):
            # import ipdb; ipdb.set_trace()
            feat_signal, feat_signal_len = batch
            feat_signal, feat_signal_len = feat_signal.to(device), feat_signal_len.to(device)
            # forward_outs = self.asr_model(processed_signal=feat_signal, processed_signal_length=feat_signal_len)
            encoded, encoded_len, _, _ = self.asr_model.train_val_forward([feat_signal, feat_signal_len, None, None, None, None], 0)
            forward_outs = (encoded, encoded_len)
            if len(forward_outs) == 2:  # hybrid ctc rnnt model
                encoded, encoded_len = forward_outs
                log_probs = self.asr_model.ctc_decoder(encoder_output=encoded)
                predictions = log_probs.argmax(dim=-1, keepdim=False)
            else:
                log_probs, encoded_len, predictions = forward_outs

            #remove pred from query
            log_probs = log_probs[:,self.query_pred_len:,:]
            predictions = predictions[:,self.query_pred_len:]

            preds = torch.unbind(predictions)
            for pred in preds:
                self.all_preds.append(pred.cpu().numpy())
            if keep_logits:
                log_probs = torch.unbind(log_probs)
                for log_prob in log_probs:
                    self.all_logits.append(log_prob.cpu())
            else:
                del log_probs
            del encoded_len
            del predictions

    def transcribe(self, tokens_per_chunk: int, delay: int, keep_logits: bool = False):
        self.infer_logits(keep_logits)
        self.unmerged = []
        for pred in self.all_preds:
            decoded = pred.tolist()
            self.unmerged += decoded[max(0,len(decoded) - 1 - delay) : len(decoded) - 1 - delay + tokens_per_chunk]
        hypothesis = self.greedy_merge(self.unmerged)
        if not keep_logits:
            return hypothesis

        all_logits = []
        for log_prob in self.all_logits:
            T = log_prob.shape[0]
            log_prob = log_prob[T - 1 - delay : T - 1 - delay + tokens_per_chunk, :]
            all_logits.append(log_prob)
        all_logits = torch.concat(all_logits, 0)
        return hypothesis, all_logits

    def greedy_merge(self, preds):
        decoded_prediction = []
        previous = self.blank_id
        for p in preds:
            if (p != previous or previous == self.blank_id) and p != self.blank_id:
                decoded_prediction.append(p)
            previous = p
        hypothesis = self.tokenizer.ids_to_text(decoded_prediction)
        return hypothesis


class AudioFeatureIterator_tgt_spk(AudioFeatureIterator):
    def __init__(self, samples, query_samples, frame_len, preprocessor, device, pad_to_frame_len=True):
        super().__init__(samples, frame_len, preprocessor, device, pad_to_frame_len)
        #init query signal and len
        self._query_samples = query_samples
        query_audio_signal = torch.from_numpy(self._query_samples).unsqueeze_(0).to(device)
        query_audio_signal_len = torch.Tensor([self._query_samples.shape[0]]).to(device)
        self._query_features, self._query_features_len = preprocessor(
            input_signal=query_audio_signal,
            length=query_audio_signal_len,
        )

    
class FeatureFrameBufferer_tgt_spk(FeatureFrameBufferer):
    """
    Class to append each feature frame to a buffer and return
    an array of buffers.
    """

    def __init__(self, asr_model, frame_len=1.6, batch_size=4, total_buffer=4.0, pad_to_buffer_len=True):
        super().__init__(asr_model, frame_len, batch_size,  total_buffer, pad_to_buffer_len)
    

    def get_buffers_batch(self):
        batch_frames = self.get_batch_frames()
        query_features = np.copy(self.frame_reader._query_features.squeeze(0).cpu())
        while len(batch_frames) > 0:

            frame_buffers = self.get_frame_buffers(batch_frames)
            for i, frame_buffer in enumerate(frame_buffers):
                frame_buffers[i] = np.concatenate([query_features, frame_buffer], axis = 1)
            norm_consts = self.get_norm_consts_per_frame(batch_frames, query_features)
            if len(frame_buffers) == 0:
                continue
            self.normalize_frame_buffers(frame_buffers, norm_consts)
            return frame_buffers
        return []
    
    def get_norm_consts_per_frame(self, batch_frames, query_features):
        norm_consts = []
        for i, frame in enumerate(batch_frames):
            self._update_feature_buffer(frame)
            mean_from_buffer = np.mean(np.concatenate([query_features, self.feature_buffer], axis =1), axis=1)
            stdev_from_buffer = np.std(np.concatenate([query_features, self.feature_buffer], axis =1), axis=1)
            norm_consts.append((mean_from_buffer.reshape(self.n_feat, 1), stdev_from_buffer.reshape(self.n_feat, 1)))
        return norm_consts