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
from nemo.collections.asr.parts.utils.streaming_utils import BatchedFeatureFrameBufferer
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

import torch.nn.functional as F


# class for streaming frame-based ASR
# 1) use reset() method to reset FrameASR's state
# 2) call transcribe(frame) to do ASR on
#    contiguous signal's frames

# audio buffer
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
        dynamic_query=False,
        pad_to_buffer_len=True,
        activation_ratio=0.85,
        new_query_max_len=10,
        new_query_min_len=13, #4 for strategy 2, 13 for strategy 3 buffer 4, 38 for strategy 3 buffer 8
        query_refresh_rate=1, # 1 for strategy 3 high throughput, 10 for strategy 3 low latency
        query_change_once=True,#change query once and then not update query
        non_target_spk_offset_threshold=0.3, #0.85
        target_spk_onset_threshold=0.3, #0.3
        start_replace_step=3, #wait for x steps before start replacing query, 3 for high throughput, 7 for low latency, overall 3s 
        diar_model_streaming_mode=False,
        sortformer_loader_level='emb',
        initial_final_buffer=False,
        sortformer_left_context_in_sec=0.0,

    ):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        self.frame_bufferer = AudioBufferer_tgt_spk(
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
        self.dynamic_query = dynamic_query
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
        cfg = copy.deepcopy(asr_model._cfg)
        self.cfg = cfg
        self.frame_len = frame_len
        OmegaConf.set_struct(cfg.preprocessor, False)

        # some changes for streaming scenario
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"
        # import ipdb; ipdb.set_trace()
        self.raw_preprocessor = ASRModel.from_config_dict(cfg.preprocessor)
        self.raw_preprocessor.to(asr_model.device)
        self.preprocessor = self.raw_preprocessor
        #dynamic query related variables
        self.all_diar_preds = None
        self.all_audio = None
        self.new_query_min_len = new_query_min_len
        self.activation_ratio = activation_ratio
        self.new_query_max_len = new_query_max_len
        self.target_spk_onset_threshold = target_spk_onset_threshold
        self.non_target_spk_offset_threshold = non_target_spk_offset_threshold
        self.query_refresh_rate = query_refresh_rate
        self.query_change_once = query_change_once
        self.start_replace_step = start_replace_step
        self.diar_model_streaming_mode = diar_model_streaming_mode
        self.sortformer_loader_level = sortformer_loader_level
        self.sortformer_left_context_in_sec = sortformer_left_context_in_sec
        #disable initial_final_buffer for sortformer offline mode
        self.initial_final_buffer = initial_final_buffer
        self.reset()

    def reset(self):
        """
        Reset frame_history and decoder's state
        """
        self.prev_char = ''
        self.unmerged = []
        self.data_layer = AudioBuffersDatalayer_tgt_spk()
        self.data_loader = DataLoader(self.data_layer, batch_size=self.batch_size, collate_fn=speech_collate_fn)
        self.all_logits = []
        self.all_preds = []
        self.toks_unmerged = []
        self.frame_buffers = []
        self.frame_bufferer.reset()
        self.query_refresh_count = 0
        self.query_refresh_rate = 1
        if self.diar_model_streaming_mode:
            assert self.batch_size == 1, "batch_size must be 1 for diar_model_streaming_mode when using batchview sample"
            self.asr_model._reset_streaming_state(batch_size = self.batch_size, async_streaming = False)
            self.left_context = int(self.sortformer_left_context_in_sec * self.asr_model._cfg.sample_rate)

    def get_partial_samples(self, audio_file: str, offset: float, duration: float, target_sr: int = 16000, dtype: str = 'float32'):
        try:
            with sf.SoundFile(audio_file, 'r') as f:
                start = int(offset * f.samplerate)
                f.seek(start)
                end = int((offset + duration) * f.samplerate)
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
        # pad on the right side
        samples = np.pad(samples, (0, int(delay * model_stride_in_secs * self.asr_model._cfg.sample_rate)))
        self.pad_len = int(delay * model_stride_in_secs * self.asr_model._cfg.sample_rate)
        # query related variables
        separater_audio = get_separator_audio(separater_freq, self.asr_model._cfg.sample_rate, separater_duration, separater_unvoice_ratio)
        self.separater_audio = separater_audio
        if query_duration > 0:
            query_samples = self.get_partial_samples(query_audio_file, query_offset, query_duration)
            query_samples = np.concatenate([query_samples, separater_audio])
        else:
            query_samples = separater_audio
        # import ipdb; ipdb.set_trace()
        frame_reader = AudioIterator_tgt_spk(samples, query_samples, self.frame_len, self.asr_model.device)
        self.query_pred_len = get_hidden_length_from_sample_length(len(query_samples), 160, 8)
        self.set_frame_reader(frame_reader)
        #reset all_audio all_diar_preds
        self.all_audio = None
        self.all_diar_preds = None
        self.selected_regions = set()
        self.change_query_action = []
                #estimate number of buffers
        self.num_buffer = np.ceil((samples.shape[0] - self.frame_bufferer.feature_frame_len) / 16000 / self.frame_len) / self.batch_size
        self.temp_buffer = 0
        # self.num_tailing_silence = 
        if self.diar_model_streaming_mode:
            #cache contains query + separater --> query
            self.asr_model.diarization_model.sortformer_modules.spkcache_len = self.query_pred_len
            self.asr_model.diarization_model.sortformer_modules.spkcache_len_audio = int(self.asr_model.diarization_model.sortformer_modules.spkcache_len / 12.5 * 16000)
            self.asr_model.diarization_model.sortformer_modules.spkcache_refresh_rate = self.query_pred_len
        if self.initial_final_buffer:
            self.num_buffer_w_leading_silence = np.ceil((self.frame_bufferer.feature_buffer_len - self.frame_bufferer.feature_frame_len)/self.frame_bufferer.feature_frame_len)

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
            feat_signal, feat_signal_len = batch

            # # padding silence after each buffer
            # import numpy as np; import torch
            # # Add padding silence to each sample
            # batch_size = len(feat_signal)
            # padding_len = 1 # in s
            # batched_padding_silence =torch.tensor(np.array([np.zeros([16000*padding_len]) for _ in range(batch_size)]))
            # feat_signal = torch.cat([feat_signal, batched_padding_silence], axis=-1)
            # padding_len = padding_len*16000
            # feat_signal_len += padding_len

            if self.initial_final_buffer:
                if self.temp_buffer == self.num_buffer:
                    #final buffer
                    final_buffer =True
                else:
                    final_buffer = False
                if self.temp_buffer < self.num_buffer_w_leading_silence:
                    initial_buffer = True
                else:
                    initial_buffer = False
            else:
                initial_buffer, final_buffer = False, False


            if initial_buffer:
                orig_feat_len = feat_signal_len[0].clone()
                valid_feat_len = int(self.frame_bufferer.feature_frame_len * (self.temp_buffer + 1))
                # temp_pad_len = int(self.frame_bufferer.feature_buffer_len - self.frame_bufferer.feature_frame_len - self.temp_buffer * self.frame_bufferer.feature_frame_len)
                temp_pad_len = int(orig_feat_len - valid_feat_len - self.frame_bufferer.frame_reader.query_audio_signal_len[0])
                feat_signal[0] = torch.nn.functional.pad(
                    torch.cat([feat_signal[0][:int(self.frame_bufferer.frame_reader.query_audio_signal_len[0])], feat_signal[0][-valid_feat_len:]]), 
                    (0, temp_pad_len)
                )
                feat_signal_len[0] = feat_signal_len[0] - temp_pad_len
            if final_buffer:
                feat_signal_len[-1] = feat_signal_len[-1] - self.pad_len


            feat_signal, feat_signal_len = feat_signal.to(device), feat_signal_len.to(device)
            # forward_outs = self.asr_model(processed_signal=feat_signal, processed_signal_length=feat_signal_len)
            if self.diar_model_streaming_mode:
                encoded, encoded_len = self.asr_model.forward_sortformer_streaming(
                    signal = feat_signal,
                    signal_len = feat_signal_len,
                    query_len = [self.frame_bufferer.frame_reader.query_audio_signal_len[0]],
                    chunk_len = self.frame_bufferer.feature_frame_len,
                    buffer_len = self.frame_bufferer.feature_buffer_len,
                    initial_buffer = initial_buffer,
                    temp_buffer_index = self.temp_buffer,
                    sortformer_loader_level = self.sortformer_loader_level,
                    left_context = self.left_context,
                    tokens_per_chunk = self.tokens_per_chunk,
                )
            else:
                encoded, encoded_len, _, _ = self.asr_model.train_val_forward([feat_signal, feat_signal_len, None, None, None, None], 0)
            forward_outs = (encoded, encoded_len)
            if len(forward_outs) == 2:  # hybrid ctc rnnt model
                encoded, encoded_len = forward_outs
                log_probs = self.asr_model.ctc_decoder(encoder_output=encoded)
                predictions = log_probs.argmax(dim=-1, keepdim=False)
            else:
                log_probs, encoded_len, predictions = forward_outs
            # #remove pred from query
            log_probs = log_probs[:,self.query_pred_len:,:]
            predictions = predictions[:,self.query_pred_len:]
            if initial_buffer:
                valid_pred_len = get_hidden_length_from_sample_length(valid_feat_len, 160, 8)-1
                temp_pad_len_pred = len(predictions[0])- valid_pred_len
                predictions = torch.nn.functional.pad(predictions[:,:valid_pred_len], (temp_pad_len_pred, 0), mode='constant', value=self.blank_id)


            # hidden_padding_len = get_hidden_length_from_sample_length(padding_len, 160, 8)
            # log_probs = log_probs[:,self.query_pred_len-1:-hidden_padding_len+1,:]
            # predictions = predictions[:,self.query_pred_len-1:-hidden_padding_len+1]

            #dynamic query
            # import ipdb; ipdb.set_trace()
            if self.all_diar_preds is None:

                self.all_diar_preds = self.asr_model.diar_preds[:,self.query_pred_len-1:]

                self.all_audio = feat_signal[:,int(self.frame_bufferer.frame_reader.query_audio_signal_len[0]):]

            # else:
                # import numpy as np
                # self.all_diar_preds = F.pad(self.all_diar_preds, (0, 0, 0, int(np.ceil(self.frame_bufferer.feature_frame_len/16000 * 12.5)), 0, 0))
                # self.all_diar_preds[:, -get_hidden_length_from_sample_length(self.frame_bufferer.feature_buffer_len, 160, 8)+2:,:] = self.asr_model.diar_preds[:, -get_hidden_length_from_sample_length(self.frame_bufferer.feature_buffer_len, 160, 8)+2:,:]
                # self.all_audio = F.pad(self.all_audio, (0, self.frame_bufferer.feature_frame_len, 0, 0))
                # self.all_audio[:, -self.frame_bufferer.feature_buffer_len:] = feat_signal[:,int(self.frame_bufferer.frame_reader.query_audio_signal_len[0]):]
                #select new query from history
                # Find subrange where first speaker is active and last 3 are inactive
            if self.dynamic_query:
                self.query_refresh_count += 1
                diar_preds = self.all_diar_preds.squeeze(0) # Shape [len, 4]
                first_spk_active = diar_preds[:, 0] > self.target_spk_onset_threshold # High threshold for first speaker
                other_spks_inactive = torch.all(diar_preds[:, 1:] < self.non_target_spk_offset_threshold, dim=1) # Low threshold for other speakers
                valid_frames = torch.logical_and(first_spk_active, other_spks_inactive)

                # Helper function to check if a region overlaps with previously selected ones
                def add_region(start, end):
                    self.selected_regions.add((start, end))
                # Helper function to check if a region starts from previous region
                def no_look_back(start):
                    if not self.selected_regions:
                        return True
                    prev_start, prev_end = max(self.selected_regions) if self.selected_regions else (0, 0)
                    return start >= prev_end

                strategy = 3
                search_direction = 'forward' # 'backward
                # Keep track of previously selected regions to avoid duplicates

                """
                Strategy 1:
                    Search for valid from with min_length 3s, starting from the middle of all diar_preds and moving forward / backward, replace the old query with the new query
                """
                if strategy == 1:
                    min_length = self.new_query_min_len
                    # Start from middle and search forward
                    start_idx = len(valid_frames) // 2
                    end_idx = start_idx + min_length - 1

                    # Search forward until we find a valid sequence or reach en
                    if search_direction == 'forward':
                        while end_idx < len(valid_frames): 
                            if torch.sum(valid_frames[start_idx:end_idx+1]) / (end_idx - start_idx + 1) > self.activation_ratio and no_look_back(start_idx):
                                # and  torch.all(valid_frames[end_idx-6:end_idx+1])):
                                add_region(start_idx, end_idx)
                                break
                            start_idx += 1
                            end_idx = start_idx + min_length - 1
                        replace_query = (end_idx < len(valid_frames))
                    elif search_direction == 'backward':
                        while start_idx > 0:
                            if torch.sum(valid_frames[start_idx:end_idx+1]) / (end_idx - start_idx + 1) > self.activation_ratio:
                                # and  torch.all(valid_frames[end_idx-6:end_idx+1])):
                                break
                            start_idx -= 1
                            end_idx = start_idx + min_length - 1
                        replace_query = (start_idx > 0)
                
                    if replace_query:
                        shifted_start_idx = start_idx
                        shifted_end_idx = end_idx
                        import numpy as np
                        # Get candidate new query audio
                        candidate_query = self.all_audio[0,int(shifted_start_idx/12.5*16000):int(shifted_end_idx/12.5*16000)]
                        print('Change of query!')
                        print('Start idx: ', start_idx)
                        print('End idx: ', end_idx)

                        #replace old query with new query
                        new_query = np.concatenate([candidate_query.cpu().numpy(), self.separater_audio])

                elif strategy == 2:
                    """
                    Strategy 2:
                        Search for valid from with min_length min_length, starting from the middle of all diar_preds and moving forward / backward, add new query to the start / end of old query and truncate to 5s
                    """
                    min_length = self.new_query_min_len
                    # Start from middle and search forward
                    start_idx = len(valid_frames) // 2
                    end_idx = start_idx + min_length - 1

                    # Search forward until we find a valid sequence or reach en

                    if search_direction == 'forward':
                        while end_idx < len(valid_frames): 
                            if torch.sum(valid_frames[start_idx:end_idx+1]) / (end_idx - start_idx + 1) > self.activation_ratio:# and no_look_back(start_idx):
                                # and  torch.all(valid_frames[end_idx-6:end_idx+1])):
                                add_region(start_idx, end_idx)
                                break
                            start_idx += 1
                            end_idx = start_idx + min_length - 1
                        replace_query = (end_idx < len(valid_frames))
                    elif search_direction == 'backward':
                        while start_idx > 0:
                            if torch.sum(valid_frames[start_idx:end_idx+1]) / (end_idx - start_idx + 1) > self.activation_ratio:
                                # and  torch.all(valid_frames[end_idx-6:end_idx+1])):
                                break
                            start_idx -= 1
                            end_idx = start_idx + min_length - 1
                        replace_query = (start_idx > 0)
                    if replace_query:
                        shifted_start_idx = start_idx
                        shifted_end_idx = end_idx
                        import numpy as np
                        # Get candidate new query audio
                        candidate_query = self.all_audio[0,int(shifted_start_idx/12.5*16000):int(shifted_end_idx/12.5*16000)]
                        print('Change of query!')
                        print('Start idx: ', start_idx)
                        print('End idx: ', end_idx)

                        #replace old query with new query
                        # new_query = np.concatenate([candidate_query.cpu().numpy(), self.separater_audio])
                        #concatenate to the end of old query
                        new_query = np.concatenate([self.frame_bufferer.frame_reader._query_samples[:-len(self.separater_audio)], candidate_query.cpu().numpy(), self.separater_audio])
                        new_query = new_query[-int(self.new_query_max_len*16000):]
                        self.change_query_action.append(0)
                    else:
                        self.change_query_action.append(0)
                        #concatenate to the start of old query
                        # new_query = np.concatenate([candidate_query.cpu().numpy(), self.frame_bufferer.frame_reader._query_samples[:-len(self.separater_audio)]])
                        # new_query = new_query[:int(5*16000)]
                        # new_query = np.concatenate([new_query, self.separater_audio])
                elif strategy == 3:
                    """
                    Strategy 3:
                        Search for valid from with min_length to max_length, starting from the middle of all diar_preds and moving forward / backward, replace old query with the new query, if valid_sequence is short, repeat the valid_sequence to max_length
                    """
                    min_length = self.new_query_min_len
                    max_length = 25
                    # Start from middle and search forward
                    start_idx = 0 #len(valid_frames) // 2
                    end_idx = start_idx + min_length - 1

                    # Search forward until we find a valid sequence or reach en
                    if search_direction == 'forward':
                        while end_idx < len(valid_frames): 
                            if torch.sum(valid_frames[start_idx:end_idx+1]) / (end_idx - start_idx + 1) > self.activation_ratio:# and torch.all(valid_frames[end_idx-6:end_idx+1]):# and no_look_back(start_idx):
                                # Try extending end_idx up to max_length while maintaining activation ratio
                                anchored_end_index = end_idx
                                temp_end_idx = end_idx
                                while temp_end_idx < min(len(valid_frames), start_idx + max_length):
                                    if torch.sum(valid_frames[anchored_end_index:temp_end_idx+1]) / (temp_end_idx - anchored_end_index + 1) > self.activation_ratio:
                                        end_idx = temp_end_idx
                                        temp_end_idx += 1
                                    else:
                                        break
                                # and  torch.all(valid_frames[end_idx-6:end_idx+1])):
                                # add_region(start_idx, end_idx)
                                break
                            start_idx += 1
                            end_idx = start_idx + min_length - 1
                        replace_query = (end_idx < len(valid_frames)) if len(self.change_query_action) >= self.start_replace_step else False
                    elif search_direction == 'backward':
                        while start_idx > 0:
                            if torch.sum(valid_frames[start_idx:end_idx+1]) / (end_idx - start_idx + 1) > self.activation_ratio:
                                # and  torch.all(valid_frames[end_idx-6:end_idx+1])):
                                break
                            start_idx -= 1
                            end_idx = start_idx + min_length - 1
                        replace_query = (start_idx > 0)
                
                    if replace_query:
                        shifted_start_idx = start_idx # 6 frames after the valid sequence
                        shifted_end_idx = end_idx # 6 frames before the valid sequence
                        import numpy as np
                        # Get candidate new query audio
                        candidate_query = self.all_audio[0,int(shifted_start_idx/12.5*16000):int(shifted_end_idx/12.5*16000)]
                        # If candidate query is shorter than 2s (32000 samples), repeat to make it 3s (48000 samples)
                        if len(candidate_query) < 32000:
                            repeats = int(np.ceil(48000 / len(candidate_query)))
                            candidate_query = candidate_query.repeat(repeats)[:48000]
                        self.change_query_action.append(1) if self.query_refresh_count >= self.query_refresh_rate else self.change_query_action.append(0)

                        #replace old query with new query
                        new_query = np.concatenate([candidate_query.cpu().numpy(), self.separater_audio])
                    else:
                        self.change_query_action.append(0)
                    
            #manipulate audio
            # query_audio = feat_signal[0][-16000:].repeat(2)
            # changed_feat_signal = torch.cat([query_audio.unsqueeze(0), feat_signal[0][64000:].unsqueeze(0)], dim=1)
            # feat_signal = changed_feat_signal
            # feat_signal_len = torch.Tensor([changed_feat_signal.shape[1]]).to(device)
            
                if replace_query and self.query_refresh_count >= self.query_refresh_rate:
                    print('Change of query!')
                    print('Start idx: ', start_idx)
                    print('End idx: ', end_idx)
                    self.frame_bufferer.frame_reader._query_samples = new_query
                    self.frame_bufferer.frame_reader.query_audio_signal = torch.from_numpy(self.frame_bufferer.frame_reader._query_samples).unsqueeze_(0).to(device)
                    self.frame_bufferer.frame_reader.query_audio_signal_len = torch.Tensor([self.frame_bufferer.frame_reader.query_audio_signal.shape[1]]).to(device)
                    self.query_pred_len = get_hidden_length_from_sample_length(self.frame_bufferer.frame_reader.query_audio_signal_len, 160, 8)
                    self.query_refresh_count = 0
                    if self.query_change_once:
                        self.query_refresh_rate = np.inf



            # import ipdb; ipdb.set_trace()
            preds = torch.unbind(predictions)
            for pred in preds:
                self.all_preds.append(pred.cpu().numpy())
            
            save_intermediate_var = False

            if save_intermediate_var:
                delay = 26
                tokens_per_chunk = 2
                buffer_logits = []
                buffer_preds = []
                for i, pred in enumerate(self.all_preds):
                    decoded = pred.tolist()
                    self.unmerged += decoded[max(0,len(decoded) - 1 - delay) : len(decoded) - 1 - delay + tokens_per_chunk]
                    hypothesis = self.greedy_merge(self.unmerged)
                    # print(hypothesis)
                    buffer_preds.append(self.greedy_merge(decoded))
                    buffer_logits.append(decoded)
                parent_dir = '/home/jinhanw/workdir/workdir_nemo_speaker_asr/dataloader/pipeline/decode_scripts/saved/temp'
                os.makedirs(parent_dir, exist_ok=True)
                import pickle; import numpy as np;
                with open(os.path.join(parent_dir, 'buffer_preds.pickle'), 'wb') as f:
                    pickle.dump(buffer_preds, f)
                with open(os.path.join(parent_dir, 'buffer_logits.pickle'), 'wb') as f:
                    pickle.dump(buffer_logits, f)
                with open(os.path.join(parent_dir, 'feat_signal.pickle'), 'wb') as f:
                    pickle.dump(feat_signal, f)
                with open(os.path.join(parent_dir, 'feat_signal_len.pickle'), 'wb') as f:
                    pickle.dump(feat_signal_len, f)
                with open(os.path.join(parent_dir,'diar_model.cfg'), 'w') as f:
                    f.write(OmegaConf.to_yaml(self.asr_model.diarization_model._cfg))
                with open(os.path.join(parent_dir, 'diar_preds.pickle'), 'wb') as f:
                    pickle.dump(self.asr_model.diar_preds, f)
                # with open(os.path.join(parent_dir, 'total_diar_preds.pickle'), 'wb') as f:
                    # pickle.dump(self.asr_model.total_preds, f)
                with open(os.path.join(parent_dir, 'all_diar_preds.pickle'), 'wb') as f:
                    pickle.dump(self.all_diar_preds, f)
                with open(os.path.join(parent_dir, 'all_audio.pickle'), 'wb') as f:
                    pickle.dump(self.all_audio, f)
                if self.diar_model_streaming_mode:  
                    with open(os.path.join(parent_dir, 'spkcache_fifo_chunk_preds.pickle'), 'wb') as f:
                        pickle.dump(self.asr_model.diarization_model.spkcache_fifo_chunk_preds, f)
                        print('\n')
                    if self.sortformer_loader_level == 'emb':
                        print(self.asr_model.streaming_state.fifo.shape)
                        print(self.asr_model.streaming_state.spkcache.shape)
                    elif self.sortformer_loader_level == 'audio':
                        print(self.asr_model.streaming_state.fifo_audio.shape)
                        print(self.asr_model.streaming_state.spkcache_audio.shape)
                if self.dynamic_query:

                    with open(os.path.join(parent_dir, 'new_query.pickle'), 'wb') as f:
                        pickle.dump(self.frame_bufferer.frame_reader.query_audio_signal, f)
                if self.temp_buffer % 1 == 0:
                    import ipdb; ipdb.set_trace()

            
            if keep_logits:
                log_probs = torch.unbind(log_probs)
                for log_prob in log_probs:
                    self.all_logits.append(log_prob.cpu())
            else:
                del log_probs
            del encoded_len
            del predictions
                        
            self.temp_buffer += 1


    def transcribe(self, tokens_per_chunk: int, delay: int, keep_logits: bool = False):
        self.tokens_per_chunk = tokens_per_chunk
        self.delay = delay
        self.infer_logits(keep_logits)
        self.unmerged = []

        for i, pred in enumerate(self.all_preds):
            decoded = pred.tolist()

            if len(self.change_query_action) == 0 or (i > 0 and self.change_query_action[i-1] == 0):
                self.unmerged += decoded[max(0,len(decoded) - 1 - delay) : len(decoded) - 1 - delay + tokens_per_chunk]
            else:
                if (self.all_preds[i-1] == 1024).all():
                    # there is a change of query action in previous buffer, and previous buffer is blank
                    self.unmerged += decoded[: len(decoded) - 1 - delay + tokens_per_chunk]
                else:
                    # there is a change of query action in previous buffer, and previous buffer is not blank
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


class AudioIterator_tgt_spk(IterableDataset):
    def __init__(self, samples, query_samples, frame_len, device, pad_to_frame_len=True):
        self._samples = samples
        self._frame_len = frame_len
        self._start = 0
        self.output = True
        self.count = 0
        self.pad_to_frame_len = pad_to_frame_len
        # timestep_duration = preprocessor._cfg['window_stride']
        # self._feature_frame_len = frame_len / timestep_duration
        self._feature_frame_len = frame_len * 16000
        self.audio_signal = torch.from_numpy(self._samples).unsqueeze_(0).to(device)
        self.audio_signal_len = torch.Tensor([self._samples.shape[0]]).to(device)
        # self._features, self._features_len = preprocessor(
        #     input_signal=audio_signal,
        #     length=audio_signal_len,
        # )
        # self._features = self._features.squeeze()
        # super().__init__(samples, frame_len, preprocessor, device, pad_to_frame_len)
        #init query signal and len
        self._query_samples = query_samples
        self.query_audio_signal = torch.from_numpy(self._query_samples).unsqueeze_(0).to(device)
        self.query_audio_signal_len = torch.Tensor([self._query_samples.shape[0]]).to(device)
        # import ipdb; ipdb.set_trace()
        # self._query_features, self._query_features_len = preprocessor(
        #     input_signal=query_audio_signal,
        #     length=query_audio_signal_len,
        # )

    def __iter__(self):
        return self

    def __next__(self):
        if not self.output:
            raise StopIteration
        # import ipdb; ipdb.set_trace()
        last = int(self._start + self._feature_frame_len)
        if last <= self.audio_signal_len[0]:
            frame = self.audio_signal[:, self._start : last].cpu()
            self._start = last
        else:
            if not self.pad_to_frame_len:
                frame = self.audio_signal[:, self._start : self.audio_signal_len[0]].cpu()
            else:
                frame = np.zeros([self.audio_signal.shape[0], int(self._feature_frame_len)], dtype='float32')
                segment = self.audio_signal[:, self._start : int(self.audio_signal_len[0])].cpu()
                frame[:, : segment.shape[1]] = segment
            self.output = False
        self.count += 1
        return frame

class AudioBufferer_tgt_spk:
    """
    Class to append each feature frame to a buffer and return
    an array of buffers.
    """

    def __init__(self, asr_model, frame_len=1.6, batch_size=4, total_buffer=4.0, pad_to_buffer_len=True):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        if hasattr(asr_model.preprocessor, 'log') and asr_model.preprocessor.log:
            self.ZERO_LEVEL_SPEC_DB_VAL = -16.635  # Log-Melspectrogram value for zero signal
        else:
            self.ZERO_LEVEL_SPEC_DB_VAL = 0.0
        self.asr_model = asr_model
        self.sr = asr_model._cfg.sample_rate
        self.frame_len = frame_len
        self.feature_frame_len = int(frame_len * self.sr)
        # timestep_duration = asr_model._cfg.preprocessor.window_stride
        # self.n_frame_len = int(frame_len / timestep_duration)

        # total_buffer_len = int(total_buffer / timestep_duration)
        total_buffer_len = int(total_buffer * self.sr)
        # self.n_feat = asr_model._cfg.preprocessor.features
        
        # self.buffer = np.ones([self.n_feat, total_buffer_len], dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        self.buffer = np.ones([1, total_buffer_len], dtype = np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        self.pad_to_buffer_len = pad_to_buffer_len
        self.batch_size = batch_size

        self.signal_end = False
        self.frame_reader = None
        self.feature_buffer_len = total_buffer_len

        # self.feature_buffer = (
        #     np.ones([self.n_feat, self.feature_buffer_len], dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        # )
        self.feature_buffer = (
            np.ones([1, self.feature_buffer_len], dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        )
        self.frame_buffers = []
        self.buffered_features_size = 0
        self.reset()
        self.buffered_len = 0

    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer = np.ones(shape=self.buffer.shape, dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        self.prev_char = ''
        self.unmerged = []
        self.frame_buffers = []
        self.buffered_len = 0
        # self.feature_buffer = (
        #     np.ones([self.n_feat, self.feature_buffer_len], dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        # )
        self.feature_buffer = (
            np.ones([1, self.feature_buffer_len], dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        )

    def get_batch_frames(self):
        if self.signal_end:
            return []
        batch_frames = []
        for frame in self.frame_reader:
            batch_frames.append(np.copy(frame))
            if len(batch_frames) == self.batch_size:
                return batch_frames
        self.signal_end = True

        return batch_frames

    def get_frame_buffers(self, frames):
        # Build buffers for each frame
        self.frame_buffers = []
        for frame in frames:
            curr_frame_len = frame.shape[1]
            self.buffered_len += curr_frame_len
            if curr_frame_len < self.feature_buffer_len and not self.pad_to_buffer_len:
                self.frame_buffers.append(np.copy(frame))
                continue
            self.buffer[:, :-curr_frame_len] = self.buffer[:, curr_frame_len:]
            self.buffer[:, -self.feature_frame_len :] = frame
            self.frame_buffers.append(np.copy(self.buffer))
        return self.frame_buffers

    def set_frame_reader(self, frame_reader):
        self.frame_reader = frame_reader
        self.signal_end = False

    def _update_feature_buffer(self, feat_frame):
        curr_frame_len = feat_frame.shape[1]
        if curr_frame_len < self.feature_buffer_len and not self.pad_to_buffer_len:
            self.feature_buffer = np.copy(feat_frame)  # assume that only the last frame is less than the buffer length
        else:
            self.feature_buffer[:, : -feat_frame.shape[1]] = self.feature_buffer[:, feat_frame.shape[1] :]
            self.feature_buffer[:, -feat_frame.shape[1] :] = feat_frame
        self.buffered_features_size += feat_frame.shape[1]

    def get_norm_consts_per_frame(self, batch_frames):
        norm_consts = []
        for i, frame in enumerate(batch_frames):
            self._update_feature_buffer(frame)
            mean_from_buffer = np.mean(self.feature_buffer, axis=1)
            stdev_from_buffer = np.std(self.feature_buffer, axis=1)
            norm_consts.append((mean_from_buffer.reshape(self.n_feat, 1), stdev_from_buffer.reshape(self.n_feat, 1)))
        return norm_consts

    def normalize_frame_buffers(self, frame_buffers, norm_consts):
        CONSTANT = 1e-5
        for i, frame_buffer in enumerate(frame_buffers):
            frame_buffers[i] = (frame_buffer - norm_consts[i][0]) / (norm_consts[i][1] + CONSTANT)

    def get_buffers_batch(self):
        batch_frames = self.get_batch_frames()
        query_features = np.copy(self.frame_reader._query_samples)
        while len(batch_frames) > 0:

            frame_buffers = self.get_frame_buffers(batch_frames)
            for i, frame_buffer in enumerate(frame_buffers):
                frame_buffers[i] = np.concatenate([query_features, frame_buffer[0,:]], axis = 0)
            # norm_consts = self.get_norm_consts_per_frame(batch_frames, query_features)
            if len(frame_buffers) == 0:
                continue
            # self.normalize_frame_buffers(frame_buffers, norm_consts)
            return frame_buffers
        return []
    
class AudioBuffersDatalayer_tgt_spk(AudioBuffersDataLayer):
    def __init__(self):
        super().__init__()

    def __next__(self):
        if self._buf_count == len(self.signal):
            raise StopIteration
        self._buf_count += 1
        return (
            torch.as_tensor(self.signal[self._buf_count - 1], dtype=torch.float32),
            torch.as_tensor(self.signal[self._buf_count - 1].shape[0], dtype=torch.int64),
        )
    

    

