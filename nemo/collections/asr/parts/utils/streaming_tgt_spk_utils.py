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
        #disable initial_final_buffer for sortformer offline mode
        self.initial_final_buffer = initial_final_buffer if diar_model_streaming_mode else False
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
            self.asr_model._reset_streaming_state()

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
            self.asr_model.diarization_model.spkcache_len = self.query_pred_len
            self.asr_model.diarization_model.spkcache_len_audio = int(self.asr_model.diarization_model.spkcache_len / 12.5 * 16000)
            self.asr_model.diarization_model.spkcache_refresh_rate = self.query_pred_len
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
                # import ipdb; ipdb.set_trace()
                # LR_len = int((self.frame_bufferer.feature_buffer_len - self.frame_bufferer.feature_frame_len)/2)
                # LR_len_hidden = get_hidden_length_from_sample_length(LR_len, 160, 8)
                temp_pad_len = int(self.frame_bufferer.feature_buffer_len - self.frame_bufferer.feature_frame_len - self.temp_buffer * self.frame_bufferer.feature_frame_len)
                feat_signal[0] = torch.nn.functional.pad(
                    torch.cat([feat_signal[0][:int(self.frame_bufferer.frame_reader.query_audio_signal_len[0])], feat_signal[0][-int(self.frame_bufferer.feature_frame_len * (self.temp_buffer + 1)):]]), 
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
                    query_len = self.frame_bufferer.frame_reader.query_audio_signal_len[0],
                    chunk_len = self.frame_bufferer.feature_frame_len,
                    buffer_len = self.frame_bufferer.feature_buffer_len,
                    initial_buffer = initial_buffer,
                    temp_buffer_index = self.temp_buffer,
                    sortformer_loader_level = self.sortformer_loader_level
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
                predictions = torch.nn.functional.pad(predictions[:,:get_hidden_length_from_sample_length(self.frame_bufferer.feature_frame_len * (self.temp_buffer + 1), 160, 8)], (get_hidden_length_from_sample_length(self.frame_bufferer.feature_frame_len * (self.num_buffer_w_leading_silence - self.temp_buffer), 160, 8), 0), mode='constant', value=self.blank_id)


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
    

# feature buffer 

# ctc

class FeatureFrameBatchASR_tgt_spk:
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
    
#batched rnnt

class BatchedFrameASRRNNT_tgt_spk(FeatureFrameBatchASR_tgt_spk):
    def __init__(
        self,
        asr_model,
        frame_len=1.6,
        total_buffer=4.0,
        batch_size=4,
        max_steps_per_timestep: int = 5,
        stateful_decoding: bool = False,
    ):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        super().__init__(asr_model, frame_len, total_buffer=total_buffer, batch_size=batch_size)
        
        # OVERRIDES OF THE BASE CLASS
        self.max_steps_per_timestep = max_steps_per_timestep
        self.stateful_decoding = stateful_decoding

        self.all_alignments = [[] for _ in range(self.batch_size)]
        self.all_preds = [[] for _ in range(self.batch_size)]
        self.all_timestamps = [[] for _ in range(self.batch_size)]
        self.previous_hypotheses = None
        self.batch_index_map = {
            idx: idx for idx in range(self.batch_size)
        }  # pointer from global batch id : local sub-batch id

        try:
            self.eos_id = self.asr_model.tokenizer.eos_id
        except Exception:
            self.eos_id = -1

        print("Performing Stateful decoding :", self.stateful_decoding)

        # OVERRIDES
        self.frame_bufferer = BatchedFeatureFrameBufferer_tgt_spk(
            asr_model=asr_model, frame_len=frame_len, batch_size=batch_size, total_buffer=total_buffer
        )

        self.reset()

    def reset(self):
        """
        Reset frame_history and decoder's state
        """
        super().reset()

        self.all_alignments = [[] for _ in range(self.batch_size)]
        self.all_preds = [[] for _ in range(self.batch_size)]
        self.all_timestamps = [[] for _ in range(self.batch_size)]
        self.previous_hypotheses = None
        self.batch_index_map = {idx: idx for idx in range(self.batch_size)}

        self.data_layer = [AudioBuffersDataLayer() for _ in range(self.batch_size)]
        self.data_loader = [
            DataLoader(self.data_layer[idx], batch_size=1, collate_fn=speech_collate_fn)
            for idx in range(self.batch_size)
        ]

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

    def read_audio_file(self, audio_filepaths: list, offsets, durations, query_audio_files, query_offsets, query_durations, separater_freq, separater_duration, separater_unvoice_ratio,delay, model_stride_in_secs):
        # samples = get_samples(audio_filepath)
        # rewrite loading audio function to support partial audio
        for idx in range(self.batch_size):
            samples = self.get_partial_samples(audio_filepaths[idx], offsets[idx], durations[idx])
            samples = np.pad(samples, (0, int(delay * model_stride_in_secs * self.asr_model._cfg.sample_rate)))
            # query related variables
            query_samples = self.get_partial_samples(query_audio_files[idx], query_offsets[idx], query_durations[idx])
            separater_audio = get_separator_audio(separater_freq, self.asr_model._cfg.sample_rate, separater_duration, separater_unvoice_ratio)
            query_samples = np.concatenate([query_samples, separater_audio])
            frame_reader = AudioFeatureIterator_tgt_spk(samples, query_samples, self.frame_len, self.raw_preprocessor, self.asr_model.device)
            self.query_pred_len = get_hidden_length_from_sample_length(len(query_samples), 160, 8)
            self.set_frame_reader(frame_reader, idx)

    def set_frame_reader(self, frame_reader, idx):
        self.frame_bufferer.set_frame_reader(frame_reader, idx)

    @torch.no_grad()
    def infer_logits(self):
        frame_buffers = self.frame_bufferer.get_buffers_batch()
        while len(frame_buffers) > 0:
            self.frame_buffers += frame_buffers[:]
            for idx, buffer in enumerate(frame_buffers):
                self.data_layer[idx].set_signal([buffer[:]])

            self._get_batch_preds()
            frame_buffers = self.frame_bufferer.get_buffers_batch()

    @torch.no_grad()
    def _get_batch_preds(self):
        device = self.asr_model.device
        data_iters = [iter(data_loader) for data_loader in self.data_loader]
        feat_signals = []
        feat_signal_lens = []
        new_batch_keys = []
        # while not all(self.frame_bufferer.signal_end):
        # for batch in iter(self.data_loader):
        for idx in range(self.batch_size):
            if self.frame_bufferer.signal_end[idx]:
                continue
            batch = next(data_iters[idx])
            # import ipdb; ipdb.set_trace()
            feat_signal, feat_signal_len = batch
            feat_signal, feat_signal_len = feat_signal.to(device), feat_signal_len.to(device)
            
            feat_signals.append(feat_signal)
            feat_signal_lens.append(feat_signal_len)

            #preserve batch indices
            new_batch_keys.append(idx)
        
        if len(feat_signals) == 0:
            return
        
        feat_signal = torch.cat(feat_signals, 0)
        feat_signal_len = torch.cat(feat_signal_lens, 0)

        del feat_signals, feat_signal_lens
        encoded, encoded_len, _, _ = self.asr_model.train_val_forward([feat_signal, feat_signal_len, None, None, None, None], 0)

        # filter out partial hypotheses from older batch subset
        if self.stateful_decoding and self.previous_hypotheses is not None:
            new_prev_hypothesis = []
            for new_batch_idx, global_index_key in enumerate(new_batch_keys):
                old_pos = self.batch_index_map[global_index_key]
                new_prev_hypothesis.append(self.previous_hypotheses[old_pos])
            self.previous_hypotheses = new_prev_hypothesis

        best_hyp, _ = self.asr_model.decoding.rnnt_decoder_predictions_tensor(
            encoded, encoded_len, return_hypotheses=True, partial_hypotheses=self.previous_hypotheses
        )
        # import ipdb; ipdb.set_trace()
        if self.stateful_decoding:
            # preserve last state from hypothesis of new batch indices
            self.previous_hypotheses = best_hyp

        for idx, hyp in enumerate(best_hyp):
            global_index_key = new_batch_keys[idx]  # get index of this sample in the global batch

            has_signal_ended = self.frame_bufferer.signal_end[global_index_key]
            if not has_signal_ended:
                self.all_alignments[global_index_key].append(hyp.alignments)

        preds = [hyp.y_sequence for hyp in best_hyp]
        for idx, pred in enumerate(preds):
            global_index_key = new_batch_keys[idx]  # get index of this sample in the global batch

            has_signal_ended = self.frame_bufferer.signal_end[global_index_key]
            if not has_signal_ended:
                self.all_preds[global_index_key].append(pred.cpu().numpy())

        timestamps = [hyp.timestep for hyp in best_hyp]
        for idx, timestep in enumerate(timestamps):
            global_index_key = new_batch_keys[idx]  # get index of this sample in the global batch

            has_signal_ended = self.frame_bufferer.signal_end[global_index_key]
            if not has_signal_ended:
                self.all_timestamps[global_index_key].append(timestep)

        if self.stateful_decoding:
            # State resetting is being done on sub-batch only, global index information is not being updated
            reset_states = self.asr_model.decoder.initialize_state(encoded)

            for idx, pred in enumerate(preds):
                if len(pred) > 0 and pred[-1] == self.eos_id:
                    # reset states :
                    self.previous_hypotheses[idx].y_sequence = self.previous_hypotheses[idx].y_sequence[:-1]
                    self.previous_hypotheses[idx].dec_state = self.asr_model.decoder.batch_select_state(
                        reset_states, idx
                    )

        # Position map update
        if len(new_batch_keys) != len(self.batch_index_map):
            for new_batch_idx, global_index_key in enumerate(new_batch_keys):
                self.batch_index_map[global_index_key] = new_batch_idx  # let index point from global pos -> local pos

        del encoded, encoded_len
        del best_hyp, pred

    def transcribe(
            self, 
            tokens_per_chunk: int, 
            delay: int,
    ):
        """
        Performs "middle token" alignment prediction using the buffered audio chunk.
        """
        self.infer_logits()

        self.unmerged = [[] for _ in range(self.batch_size)]
        for idx, alignments in enumerate(self.all_alignments):

            signal_end_idx = self.frame_bufferer.signal_end_index[idx]
            if signal_end_idx is None:
                raise ValueError("Signal did not end")

            for a_idx, alignment in enumerate(alignments):
                # import ipdb; ipdb.set_trace()
                if delay == len(alignment):  # chunk size = buffer size
                    offset = 0
                else:  # all other cases
                    offset = 1
                # import ipdb; ipdb.set_trace()
                alignment = alignment[
                    len(alignment) - offset - delay : len(alignment) - offset - delay + tokens_per_chunk
                ]

                ids, toks = self._alignment_decoder(alignment, self.asr_model.tokenizer, self.blank_id)

                if len(ids) > 0 and a_idx < signal_end_idx:
                    self.unmerged[idx] = inplace_buffer_merge(
                        self.unmerged[idx],
                        ids,
                        delay,
                        model=self.asr_model,
                    )

        output = []
        for idx in range(self.batch_size):
            output.append(self.greedy_merge(self.unmerged[idx]))
        return output

    def _alignment_decoder(self, alignments, tokenizer, blank_id):
        s = []
        ids = []

        for t in range(len(alignments)):
            for u in range(len(alignments[t])):
                _, token_id = alignments[t][u]  # (logprob, token_id)
                token_id = int(token_id)
                if token_id != blank_id:
                    token = tokenizer.ids_to_tokens([token_id])[0]
                    s.append(token)
                    ids.append(token_id)

                else:
                    # blank token
                    pass

        return ids, s

    def greedy_merge(self, preds):
        decoded_prediction = [p for p in preds]
        hypothesis = self.asr_model.tokenizer.ids_to_text(decoded_prediction)
        return hypothesis
    
class BatchedFeatureFrameBufferer_tgt_spk(FeatureFrameBufferer_tgt_spk):
    """
    Batched variant of FeatureFrameBufferer where batch dimension is the independent audio samples.
    """

    def __init__(self, asr_model, frame_len=1.6, batch_size=4, total_buffer=4.0):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        super().__init__(asr_model, frame_len=frame_len, batch_size=batch_size, total_buffer=total_buffer)

        # OVERRIDES OF BASE CLASS
        timestep_duration = asr_model._cfg.preprocessor.window_stride
        total_buffer_len = int(total_buffer / timestep_duration)
        self.buffer = (
            np.ones([batch_size, self.n_feat, total_buffer_len], dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        )

        # Preserve list of buffers and indices, one for every sample
        self.all_frame_reader = [None for _ in range(self.batch_size)]
        self.signal_end = [False for _ in range(self.batch_size)]
        self.signal_end_index = [None for _ in range(self.batch_size)]
        self.buffer_number = 0  # preserve number of buffers returned since reset.

        self.reset()
        del self.buffered_len
        del self.buffered_features_size

    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        super().reset()
        self.feature_buffer = (
            np.ones([self.batch_size, self.n_feat, self.feature_buffer_len], dtype=np.float32)
            * self.ZERO_LEVEL_SPEC_DB_VAL
        )
        self.all_frame_reader = [None for _ in range(self.batch_size)]
        self.signal_end = [False for _ in range(self.batch_size)]
        self.signal_end_index = [None for _ in range(self.batch_size)]
        self.buffer_number = 0

    def get_batch_frames(self):
        # Exit if all buffers of all samples have been processed
        if all(self.signal_end):
            return []

        # Otherwise sequentially process frames of each sample one by one.
        batch_frames = []
        for idx, frame_reader in enumerate(self.all_frame_reader):
            try:
                frame = next(frame_reader)
                frame = np.copy(frame)
                query_features = np.copy(frame_reader._query_features.squeeze(0).cpu())
                batch_frames.append((query_features, frame))
            except StopIteration:
                # If this sample has finished all of its buffers
                # Set its signal_end flag, and assign it the id of which buffer index
                # did it finish the sample (if not previously set)
                # This will let the alignment module know which sample in the batch finished
                # at which index.
                batch_frames.append((None, None))
                self.signal_end[idx] = True

                if self.signal_end_index[idx] is None:
                    self.signal_end_index[idx] = self.buffer_number

        self.buffer_number += 1
        return batch_frames

    def get_frame_buffers(self, frames):
        # Build buffers for each frame
        self.frame_buffers = []
        # Loop over all buffers of all samples
        for idx in range(self.batch_size):
            frame = frames[idx]
            # If the sample has a buffer, then process it as usual
            if frame is not None:
                self.buffer[idx, :, : -self.n_frame_len] = self.buffer[idx, :, self.n_frame_len :]
                self.buffer[idx, :, -self.n_frame_len :] = frame
                # self.buffered_len += frame.shape[1]
                # WRAP the buffer at index idx into a outer list
                self.frame_buffers.append([np.copy(self.buffer[idx])])
            else:
                # If the buffer does not exist, the sample has finished processing
                # set the entire buffer for that sample to 0
                self.buffer[idx, :, :] *= 0.0
                self.frame_buffers.append([np.copy(self.buffer[idx])])

        return self.frame_buffers

    def set_frame_reader(self, frame_reader, idx):
        self.all_frame_reader[idx] = frame_reader
        self.signal_end[idx] = False
        self.signal_end_index[idx] = None

    def _update_feature_buffer(self, feat_frame, idx):
        # Update the feature buffer for given sample, or reset if the sample has finished processing
        if feat_frame is not None:
            self.feature_buffer[idx, :, : -feat_frame.shape[1]] = self.feature_buffer[idx, :, feat_frame.shape[1] :]
            self.feature_buffer[idx, :, -feat_frame.shape[1] :] = feat_frame
            # self.buffered_features_size += feat_frame.shape[1]
        else:
            self.feature_buffer[idx, :, :] *= 0.0

    def get_norm_consts_per_frame(self, batch_frames, query_features):
        norm_consts = []
        mean_consts = []
        std_consts = []
        for i, frame in enumerate(batch_frames):
            self._update_feature_buffer(frame, i)
            if frame is None:
                query_features[i] = np.zeros((80, 851))
            mean_from_buffer = np.mean(np.concatenate([query_features[i], self.feature_buffer[i]], axis =1), axis=1)
            stdev_from_buffer = np.std(np.concatenate([query_features[i], self.feature_buffer[i]], axis =1), axis=1)
            mean_consts.append(mean_from_buffer[np.newaxis,:,np.newaxis])
            std_consts.append(stdev_from_buffer[np.newaxis,:,np.newaxis])
        norm_consts = (np.concatenate(mean_consts, axis = 0), np.concatenate(std_consts, axis = 0))
        return norm_consts

    def normalize_frame_buffers(self, frame_buffers, norm_consts):
        CONSTANT = 1e-8
        for i in range(len(frame_buffers)):
            frame_buffers[i] = (frame_buffers[i] - norm_consts[0][i]) / (norm_consts[1][i] + CONSTANT)
    
    def get_buffers_batch(self):
        batch_frames = self.get_batch_frames()
        while len(batch_frames) > 0:
            #batch_frames = list of (query_Features, batch_frames)
            frame_buffers = self.get_frame_buffers([x[1] for x in batch_frames])
            for i, frame_buffer in enumerate(frame_buffers):
                try:
                    frame_buffers[i] = np.concatenate([batch_frames[i][0], frame_buffer[0]], axis = 1)
                except:
                    frame_buffers[i] = np.concatenate([np.zeros((80, 851)), np.zeros(self.feature_buffer[0].shape)], axis = 1)
            norm_consts = self.get_norm_consts_per_frame([x[1] for x in batch_frames],[x[0] for x in batch_frames])
            if len(frame_buffers) == 0:
                continue
            try:
                self.normalize_frame_buffers(frame_buffers, norm_consts)
            except:
                import ipdb; ipdb.set_trace()
            return frame_buffers
        return []
    

