
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
from nemo.collections.asr.parts.utils.streaming_tgt_spk_feature_buffer_ctc_batchview_sample_utils import FeatureFrameBatchASR_tgt_spk

from nemo.collections.asr.parts.utils.asr_tgtspeaker_utils import (
    get_separator_audio,
)

import torch.nn.functional as F



# class for streaming batched feature-based ASR with rnnt


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