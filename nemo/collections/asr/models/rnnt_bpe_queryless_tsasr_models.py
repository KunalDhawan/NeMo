# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
#
import copy
import os
from typing import Dict, List, Optional, Union
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader

from nemo.collections.asr.data.audio_to_text_lhotse_speaker import LhotseSpeechToTextSpkBpeDataset

from nemo.core.classes.mixins import AccessMixin
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType
from nemo.collections.asr.parts.mixins import (
    TranscribeConfig,
    TranscriptionReturnType,
)

from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.asr.modules.speaker_kernels import SpeakerMask, SpeakerConcat
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_pil_targets

from nemo.core.classes.common import typecheck
from nemo.utils import logging

class EncDecRNNTBPEQLTSASRModel(EncDecRNNTBPEModel):
    """Base class for encoder decoder RNNT-based models with subword tokenization."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        # config for speaker kernel
        self.spk_kernel_type = cfg.get('spk_kernel_type', None)
        self.spk_kernel_layers = cfg.get('spk_kernel_layers', [])
        self.spk_kernel_mask_original = cfg.get('spk_kernel_mask_original', True)
        self.spk_kernel_residual = cfg.get('spk_kernel_residual', True)

        self.spk_targets = None

        self._init_spk_kernel()
        
    def _init_spk_kernel(self):
        """Initialize speaker kernel modules and register them to encoder layers."""
        if not isinstance(self.spk_kernel_layers, ListConfig):
            if self.spk_kernel_type is not None:
                raise ValueError(f"spk_kernel_layers must be a list, got {type(self.spk_kernel_layers)}")
            return

        # Initialize speaker kernels for each specified layer
        hidden_size = self.cfg.model_defaults.enc_hidden
        self.spk_kernels = torch.nn.ModuleDict()
        
        kernel_types = {
            'mask': SpeakerMask,
            'concat': SpeakerConcat,
            'sinusoidal': None
        }

        # Create kernel for each layer index
        kernel_class = kernel_types[self.spk_kernel_type]
        for layer_idx in self.spk_kernel_layers:
            if kernel_class is not None:
                self.spk_kernels[str(layer_idx)] = kernel_class(hidden_size, hidden_size, mask_original=self.spk_kernel_mask_original, residual=self.spk_kernel_residual)

        if self.spk_kernels:
            logging.info(f"Initialized speaker kernels for layers: {list(self.spk_kernels.keys())}")
            self._attach_spk_kernel_hooks()
        else:
            logging.info("No speaker kernels initialized")

    def _attach_spk_kernel_hooks(self):
        """
        Attach speaker kernel hooks to encoder layers.
        Following NeMo pattern of separating hook attachment logic.
        """
        # Only attach hooks if not already attached
        if hasattr(self, 'encoder_hooks'):
            return

        self.encoder_hooks = []
        for layer_idx, kernel in self.spk_kernels.items():
            idx = int(layer_idx)

            if idx == 0:
                hook = self.encoder.layers[idx].register_forward_pre_hook(
                    self._get_spk_kernel_hook_pre_layer(layer_idx), with_kwargs=True
                )

            if idx > 0:
                # Attach a post-hook after each layer from 0 to 16.
                # Since idx > 0, we attach to layer idx-1.
                hook = self.encoder.layers[idx - 1].register_forward_hook(
                    self._get_spk_kernel_hook_post_layer(layer_idx)
                )
            self.encoder_hooks.append(hook)

    def _get_spk_kernel_hook_pre_layer(self, layer_idx: str):
        """
        Returns a hook function for applying speaker kernel transformation.
        
        Args:
            layer_idx (str): Index of the layer to apply the kernel
            
        Returns:
            callable: Hook function that applies speaker kernel
        """

        def hook_fn(module, args, kwargs):
            # Pre-hooks with with_kwargs=True must return a (new_args, new_kwargs) tuple.
            # The input tensor is passed as a keyword argument, so we find it in 'kwargs'.
            if 'x' in kwargs:
                x = kwargs['x']
                x = self.spk_kernels[layer_idx](x, self.spk_targets)
                kwargs['x'] = x
            elif args:
                # Fallback in case the call signature ever changes
                x, *rest = args
                x = self.spk_kernels[layer_idx](x, self.spk_targets)
                args = (x, *rest)

            return args, kwargs

        return hook_fn

    def _get_spk_kernel_hook_post_layer(self, layer_idx: str):
        """
        Returns a hook function for applying speaker kernel transformation.
        
        Args:
            layer_idx (str): Index of the layer to apply the kernel
            
        Returns:
            callable: Hook function that applies speaker kernel
        """
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                x, *cache = output
                x = self.spk_kernels[layer_idx](x, self.spk_targets)
                return (x, *cache)
            return self.spk_kernels[layer_idx](output, self.spk_targets)
        
        return hook_fn

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if config.get("use_lhotse"):
            # Use open_dict to allow dynamic key addition
            with open_dict(config):
                config.global_rank = self.global_rank
                config.world_size = self.world_size
            
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseSpeechToTextSpkBpeDataset(cfg = config, tokenizer=self.tokenizer,),
            )
        
    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        """
        Forward pass of the model. Note that for RNNT Models, the forward pass of the model is a 3 step process,
        and this method only performs the first step - forward of the acoustic model.

        Please refer to the `training_step` in order to see the full `forward` step for training - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the loss and possibly compute the detokenized text via the `decoding` step.

        Please refer to the `validation_step` in order to see the full `forward` step for inference - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the decoded tokens via the `decoding` step and possibly compute the batch metrics.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 2 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        return encoded, encoded_len

    def training_step(self, batch, batch_nb):
        
        signal, signal_len, transcript, transcript_len, self.spk_targets = batch

        batch = (signal, signal_len, transcript, transcript_len)

        return super().training_step(batch, batch_nb)


    def validation_pass(self, batch, batch_idx, dataloader_idx=0):

        signal, signal_len, transcript, transcript_len, self.spk_targets = batch

        batch = (signal, signal_len, transcript, transcript_len)

        return super().validation_pass(batch, batch_idx, dataloader_idx)

    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):

        signal, signal_len, transcript, transcript_len, spk_targets = batch

        # Multi-instance inference
        n_mix = trcfg.mix
        self.spk_targets = spk_targets[:, :, :n_mix].transpose(1, 2).reshape(-1, spk_targets.size(1)) # B x T x N_mix -> BN_mix x T
        signal = signal.unsqueeze(1).repeat(1, n_mix, 1).reshape(-1, signal.size(1)) # B x T -> BN_mix x T
        signal_len = signal_len.unsqueeze(1).repeat(1, n_mix).reshape(-1) # B -> BN_mix

        batch = (signal, signal_len, transcript, transcript_len)

        return super()._transcribe_forward(batch, trcfg)
    
    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        if 'dataset_manifest' in config:
            manifest_filepath = config['dataset_manifest']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'use_lhotse': True,
            'use_bucketing': False,
            'channel_selector': config.get('channel_selector', None),
            'inference_mode': self.cfg.test_ds.get('inference_mode', True),
            'fixed_spk_id': config.get('fixed_spk_id', None)
        }

        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))

        return temporary_datalayer

    def conformer_stream_step(
        self,
        processed_signal: torch.Tensor,
        processed_signal_length: torch.Tensor = None,
        cache_last_channel: torch.Tensor = None,
        cache_last_time: torch.Tensor = None,
        cache_last_channel_len: torch.Tensor = None,
        keep_all_outputs: bool = True,
        previous_hypotheses: List['Hypothesis'] = None,
        previous_pred_out: torch.Tensor = None,
        drop_extra_pre_encoded: int = None,
        return_transcription: bool = True,
        return_log_probs: bool = False,
        spk_targets: torch.Tensor = None,
        n_mix = 1,
        binary_diar_preds=False,
        cache_gating=False,
        cache_gating_buffer_size=2,
        valid_speakers_last_time=None
    ):
        """
        It simulates a forward step with caching for streaming purposes.
        It supports the ASR models where their encoder supports streaming like Conformer.
        Args:
            processed_signal: the input audio signals
            processed_signal_length: the length of the audios
            cache_last_channel: the cache tensor for last channel layers like MHA
            cache_last_channel_len: engths for cache_last_channel
            cache_last_time: the cache tensor for last time layers like convolutions
            keep_all_outputs: if set to True, would not drop the extra outputs specified by encoder.streaming_cfg.valid_out_len
            previous_hypotheses: the hypotheses from the previous step for RNNT models
            previous_pred_out: the predicted outputs from the previous step for CTC models
            drop_extra_pre_encoded: number of steps to drop from the beginning of the outputs after the downsampling module. This can be used if extra paddings are added on the left side of the input.
            return_transcription: whether to decode and return the transcriptions. It can not get disabled for Transducer models.
            return_log_probs: whether to return the log probs, only valid for ctc model

        Returns:
            greedy_predictions: the greedy predictions from the decoder
            all_hyp_or_transcribed_texts: the decoder hypotheses for Transducer models and the transcriptions for CTC models
            cache_last_channel_next: the updated tensor cache for last channel layers to be used for next streaming step
            cache_last_time_next: the updated tensor cache for last time layers to be used for next streaming step
            cache_last_channel_next_len: the updated lengths for cache_last_channel
            best_hyp: the best hypotheses for the Transducer models
            log_probs: the logits tensor of current streaming chunk, only returned when return_log_probs=True
            encoded_len: the length of the output log_probs + history chunk log_probs, only returned when return_log_probs=True
        """
        # Multi-instance inference
        # N: # speakers
        # B x T x N
        spk_targets = spk_targets[:, :, :n_mix] 
        if cache_gating:
            max_probs = torch.max(spk_targets, dim=1).values # B x N
            valid_speakers = max_probs > 0.5 # B x N, e.g., [True, False, True, False]

            if valid_speakers_last_time is None:
                valid_speakers_last_time = [torch.zeros_like(valid_speakers).bool()] 
            valid_speakers_last_time.append(valid_speakers)
            valid_speakers_last_time = valid_speakers_last_time[-cache_gating_buffer_size:]
            valid_speakers = torch.any(torch.stack(valid_speakers_last_time), dim=0)

        else:
            valid_speakers = torch.ones((spk_targets.size(0), spk_targets.size(2))).bool() # B x N

        if valid_speakers.sum() == 0: # early stop when all speakers are absent
            return previous_pred_out, previous_hypotheses, cache_last_channel, cache_last_time, cache_last_channel_len, previous_hypotheses, valid_speakers_last_time
        valid_speaker_ids = torch.where(valid_speakers)[1] # B 
        
        # spk_targets: (B, T, N) -> (BN, T)
        spk_targets = spk_targets.transpose(1, 2).reshape(-1, spk_targets.size(1))
        if binary_diar_preds:
            spk_targets = (spk_targets > 0.5).float()

        # 
        self.spk_targets = spk_targets[valid_speakers.reshape(-1)]
    
        mi_processed_signal = []
        mi_processed_signal_length = []
        for i in range(processed_signal.size(0)):
            mi_processed_signal.append(processed_signal[[i]].repeat(valid_speakers[i].sum(), 1, 1))
            mi_processed_signal_length.append(processed_signal_length[[i]].repeat(valid_speakers[i].sum()))
        
        processed_signal = torch.cat(mi_processed_signal, dim=0)
        processed_signal_length = torch.cat(mi_processed_signal_length, dim=0)
        
        # Initialization for the first step for all speakers
        # Just duplicating cache for all speakers (n_mix = # speakers)
        if cache_last_channel_len.shape[0] != n_mix: 
            cache_last_channel = cache_last_channel.unsqueeze(2).repeat(1, 1, n_mix, 1, 1).reshape(cache_last_channel.size(0), -1, cache_last_channel.size(2), cache_last_channel.size(3))
            cache_last_time = cache_last_time.unsqueeze(2).repeat(1, 1, n_mix, 1, 1).reshape(cache_last_time.size(0), -1, cache_last_time.size(2), cache_last_time.size(3))
            cache_last_channel_len = cache_last_channel_len.unsqueeze(1).repeat(1, n_mix).reshape(-1)

        # Selecting cache for valid speakers
        cache_last_channel_spk = cache_last_channel[:, valid_speakers.reshape(-1)]
        cache_last_time_spk = cache_last_time[:, valid_speakers.reshape(-1)]
        cache_last_channel_len_spk = cache_last_channel_len[valid_speakers.reshape(-1)]
        
        # Selecting previous hypotheses and predictions for valid speakers
        previous_pred_out_spk = [previous_pred_out[i] for i in valid_speaker_ids]
        previous_hypotheses_spk = [previous_hypotheses[i] for i in valid_speaker_ids]

        # Forward pass for valid speakers
        (
            asr_pred_out_stream_spk,
            transcribed_texts_spk,
            cache_last_channel_spk,
            cache_last_time_spk,
            cache_last_channel_len_spk,
            previous_hypotheses_spk 
        ) = super().conformer_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel_spk,
            cache_last_time=cache_last_time_spk,
            cache_last_channel_len=cache_last_channel_len_spk,
            keep_all_outputs=keep_all_outputs,
            previous_hypotheses=previous_hypotheses_spk,
            previous_pred_out=previous_pred_out_spk,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
            return_transcription=return_transcription,
            return_log_probs=return_log_probs
        )
        
        # update cache for valid speakers, and keep the rest unchanged 
        cache_last_channel[:, valid_speakers.reshape(-1)] = cache_last_channel_spk
        cache_last_time[:, valid_speakers.reshape(-1)] = cache_last_time_spk
        cache_last_channel_len[valid_speakers.reshape(-1)] = cache_last_channel_len_spk

        for i, spk_idx in enumerate(valid_speaker_ids):
            previous_hypotheses[spk_idx] = previous_hypotheses_spk[i]
            previous_pred_out[spk_idx] = asr_pred_out_stream_spk[i]

        # import ipdb; ipdb.set_trace()
        return previous_pred_out, transcribed_texts_spk, cache_last_channel, cache_last_time, cache_last_channel_len, previous_hypotheses, valid_speakers_last_time
    
    def conformer_stream_step_masked(
        self,
        processed_signal: torch.Tensor,
        processed_signal_length: torch.Tensor = None,
        cache_last_channel: torch.Tensor = None,
        cache_last_time: torch.Tensor = None,
        cache_last_channel_len: torch.Tensor = None,
        keep_all_outputs: bool = True,
        previous_hypotheses: List['Hypothesis'] = None,
        previous_pred_out: torch.Tensor = None,
        drop_extra_pre_encoded: int = None,
        return_transcription: bool = True,
        return_log_probs: bool = False,
        spk_targets: torch.Tensor = None,
        n_mix = 1,
        binary_diar_preds=False,
        cache_gating=False,
        cache_gating_buffer_size=2,
        valid_speakers_last_time=None
    ):
        """
        It simulates a forward step with caching for streaming purposes.
        It supports the ASR models where their encoder supports streaming like Conformer.
        Args:
            processed_signal: the input audio signals
            processed_signal_length: the length of the audios
            cache_last_channel: the cache tensor for last channel layers like MHA
            cache_last_channel_len: engths for cache_last_channel
            cache_last_time: the cache tensor for last time layers like convolutions
            keep_all_outputs: if set to True, would not drop the extra outputs specified by encoder.streaming_cfg.valid_out_len
            previous_hypotheses: the hypotheses from the previous step for RNNT models
            previous_pred_out: the predicted outputs from the previous step for CTC models
            drop_extra_pre_encoded: number of steps to drop from the beginning of the outputs after the downsampling module. This can be used if extra paddings are added on the left side of the input.
            return_transcription: whether to decode and return the transcriptions. It can not get disabled for Transducer models.
            return_log_probs: whether to return the log probs, only valid for ctc model

        Returns:
            greedy_predictions: the greedy predictions from the decoder
            all_hyp_or_transcribed_texts: the decoder hypotheses for Transducer models and the transcriptions for CTC models
            cache_last_channel_next: the updated tensor cache for last channel layers to be used for next streaming step
            cache_last_time_next: the updated tensor cache for last time layers to be used for next streaming step
            cache_last_channel_next_len: the updated lengths for cache_last_channel
            best_hyp: the best hypotheses for the Transducer models
            log_probs: the logits tensor of current streaming chunk, only returned when return_log_probs=True
            encoded_len: the length of the output log_probs + history chunk log_probs, only returned when return_log_probs=True
        """
        # Multi-instance inference
        # N: # speakers
        # B x T x N
        spk_targets = spk_targets[:, :, :n_mix] 
        if cache_gating:
            max_probs = torch.max(spk_targets, dim=1).values # B x N
            valid_speakers = max_probs > 0.5 # B x N, e.g., [True, False, True, False]

            if valid_speakers_last_time is None:
                valid_speakers_last_time = [torch.zeros_like(valid_speakers).bool()] 
            valid_speakers_last_time.append(valid_speakers)
            valid_speakers_last_time = valid_speakers_last_time[-cache_gating_buffer_size:]
            valid_speakers = torch.any(torch.stack(valid_speakers_last_time), dim=0)

        else:
            valid_speakers = torch.ones((spk_targets.size(0), spk_targets.size(2))).bool() # B x N

        if valid_speakers.sum() == 0: # early stop when all speakers are absent
            return previous_pred_out, previous_hypotheses, cache_last_channel, cache_last_time, cache_last_channel_len, previous_hypotheses, valid_speakers_last_time
        valid_speaker_ids = torch.where(valid_speakers)[1] # B 
        
        # spk_targets: (B, T, N) -> (BN, T)
        # mask0 = spk_targets[:, :, 0] > spk_targets[:, :, 1]
        # mask1 = spk_targets[:, :, 1] > spk_targets[:, :, 0]
        # mask = torch.cat([mask0.unsqueeze(-1), mask1.unsqueeze(-1)], dim=-1)

        # spk_targets = spk_targets * mask
        spk_targets = spk_targets.transpose(1, 2).reshape(-1, spk_targets.size(1))
        spk_targets = (spk_targets > 0.5).float()

        # 
        spk_targets = spk_targets[valid_speakers.reshape(-1)]
    
        mi_processed_signal = []
        mi_processed_signal_length = []
        for i in range(processed_signal.size(0)):
            mi_processed_signal.append(processed_signal[[i]].repeat(valid_speakers[i].sum(), 1, 1))
            mi_processed_signal_length.append(processed_signal_length[[i]].repeat(valid_speakers[i].sum()))
        
        processed_signal = torch.cat(mi_processed_signal, dim=0)
        processed_signal_length = torch.cat(mi_processed_signal_length, dim=0)
        
        # Initialization for the first step for all speakers
        # Just duplicating cache for all speakers (n_mix = # speakers)
        if cache_last_channel_len.shape[0] != n_mix: 
            cache_last_channel = cache_last_channel.unsqueeze(2).repeat(1, 1, n_mix, 1, 1).reshape(cache_last_channel.size(0), -1, cache_last_channel.size(2), cache_last_channel.size(3))
            cache_last_time = cache_last_time.unsqueeze(2).repeat(1, 1, n_mix, 1, 1).reshape(cache_last_time.size(0), -1, cache_last_time.size(2), cache_last_time.size(3))
            cache_last_channel_len = cache_last_channel_len.unsqueeze(1).repeat(1, n_mix).reshape(-1)

        # Selecting cache for valid speakers
        cache_last_channel_spk = cache_last_channel[:, valid_speakers.reshape(-1)]
        cache_last_time_spk = cache_last_time[:, valid_speakers.reshape(-1)]
        cache_last_channel_len_spk = cache_last_channel_len[valid_speakers.reshape(-1)]
        
        # Selecting previous hypotheses and predictions for valid speakers
        previous_pred_out_spk = [previous_pred_out[i] for i in valid_speaker_ids]
        previous_hypotheses_spk = [previous_hypotheses[i] for i in valid_speaker_ids]
        
        # mask the processed signal
        mask = spk_targets.unsqueeze(-1).repeat(1, 1, 8).flatten(1, 2)
        if mask.size(1) > processed_signal.size(2):
            mask = mask[:, -processed_signal.size(2):]
        else:
            mask = torch.nn.functional.pad(mask, (processed_signal.size(2) - mask.size(1), 0), mode='constant', value=0)
        
        mask = mask.unsqueeze(1)
        processed_signal = processed_signal * mask
        processed_signal[torch.where(processed_signal == 0)] = -16.6355

        # Forward pass for valid speakers
        (
            asr_pred_out_stream_spk,
            transcribed_texts_spk,
            cache_last_channel_spk,
            cache_last_time_spk,
            cache_last_channel_len_spk,
            previous_hypotheses_spk 
        ) = self.conformer_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel_spk,
            cache_last_time=cache_last_time_spk,
            cache_last_channel_len=cache_last_channel_len_spk,
            keep_all_outputs=keep_all_outputs,
            previous_hypotheses=previous_hypotheses_spk,
            previous_pred_out=previous_pred_out_spk,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
            return_transcription=return_transcription,
            return_log_probs=return_log_probs
        )
        
        # update cache for valid speakers, and keep the rest unchanged 
        cache_last_channel[:, valid_speakers.reshape(-1)] = cache_last_channel_spk
        cache_last_time[:, valid_speakers.reshape(-1)] = cache_last_time_spk
        cache_last_channel_len[valid_speakers.reshape(-1)] = cache_last_channel_len_spk

        for i, spk_idx in enumerate(valid_speaker_ids):
            previous_hypotheses[spk_idx] = previous_hypotheses_spk[i]
            previous_pred_out[spk_idx] = asr_pred_out_stream_spk[i]

        return previous_pred_out, transcribed_texts_spk, cache_last_channel, cache_last_time, cache_last_channel_len, previous_hypotheses, valid_speakers_last_time