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

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from omegaconf import ListConfig

from nemo.collections.asr.modules.speaker_kernels import SpeakerMask, SpeakerConcat
from nemo.utils import logging

__all__ = ['SpeakerKernelMixin']

def get_spk_kernel_class(
    spk_kernel_type,
    input_size,
    d_model,
    dropout=0.5
):
    if spk_kernel_type == 'ff':
        return nn.Sequential(nn.Linear(input_size, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, input_size))
    elif spk_kernel_type == 'conv2d':
        return 
    elif spk_kernel_type == 'mha':
        return

class SpeakerKernelMixin:
    """
    Mixin class for models that need speaker kernel functionality.
    
    This mixin provides:
    - Speaker kernel initialization
    - Hook attachment for applying speaker kernels at specific encoder layers
    - Support for both main and background speaker kernels
    
    Models using this mixin should have the following config parameters:
    - spk_kernel_type: Type of speaker kernel ('mask', 'concat', 'sinusoidal')
    - spk_kernel_layers: List of layer indices where to apply speaker kernels
    - spk_kernel_mask_original: Whether to mask original features
    - spk_kernel_residual: Whether to use residual connections
    - add_bg_spk_kernel: Whether to add background speaker kernels
    """
    
    def _init_speaker_kernel_config(self, cfg):
        """
        Initialize speaker kernel configuration from model config.
        
        Args:
            cfg: Model configuration containing speaker kernel parameters
        """
        # Speaker kernel config
        self.spk_kernel_type = cfg.get('spk_kernel_type', None)
        self.spk_kernel_layers = cfg.get('spk_kernel_layers', [])
        self.spk_kernel_mask_original = cfg.get('spk_kernel_mask_original', True)
        self.spk_kernel_residual = cfg.get('spk_kernel_residual', True)
        self.add_bg_spk_kernel = cfg.get('add_bg_spk_kernel', False)
        
        # Initialize speaker target containers
        self.spk_targets = None
        self.bg_spk_targets = None
        
        # Initialize speaker kernels
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
        self.bg_spk_kernels = torch.nn.ModuleDict()
    
        # Create kernel for each layer index
        for layer_idx in self.spk_kernel_layers:
            self.spk_kernels[str(layer_idx)] = get_spk_kernel_class(
                spk_kernel_type=self.spk_kernel_type,
                input_size=hidden_size,
                d_model=self.cfg.encoder.d_model,
                dropout=0.5
            )
            if self.add_bg_spk_kernel:
                self.bg_spk_kernels[str(layer_idx)] = get_spk_kernel_class(
                    spk_kernel_type=self.spk_kernel_type,
                    input_size=hidden_size,
                    d_model=self.cfg.encoder.d_model,
                    dropout=0.5
                )

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
                x_spk = self.spk_kernels[layer_idx](self.mask_with_speaker_targets(x, self.spk_targets))
                x_bg_spk = self.bg_spk_kernels[layer_idx](self.mask_with_speaker_targets(x, self.bg_spk_targets))
                # residual connection
                x = x + x_spk + x_bg_spk
                kwargs['x'] = x
            elif args:
                # Fallback in case the call signature ever changes
                x, *rest = args
                x_spk = self.spk_kernels[layer_idx](self.mask_with_speaker_targets(x, self.spk_targets))
                x_bg_spk = self.bg_spk_kernels[layer_idx](self.mask_with_speaker_targets(x, self.bg_spk_targets))
                # residual connection
                x = x + x_spk + x_bg_spk
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
            if self.spk_targets is None:
                return output
                
            if isinstance(output, tuple):
                x, *cache = output
            else:
                x = output

            x_spk = self.spk_kernels[layer_idx](self.mask_with_speaker_targets(x, self.spk_targets))
            x_bg_spk = self.bg_spk_kernels[layer_idx](self.mask_with_speaker_targets(x, self.bg_spk_targets))
            # residual connection
            x = x + x_spk + x_bg_spk

            if isinstance(output, tuple):
                return (x, *cache)
            return x
        
        return hook_fn

    def _cleanup_speaker_kernel_hooks(self):
        """
        Clean up speaker kernel hooks to prevent memory leaks.
        Can be called during model cleanup or when switching between modes.
        """
        if hasattr(self, 'encoder_hooks'):
            for hook in self.encoder_hooks:
                try:
                    hook.remove()
                except Exception as e:
                    logging.warning(f"Failed to remove speaker kernel hook: {e}")
            delattr(self, 'encoder_hooks')
            logging.info("Speaker kernel hooks cleaned up")

    def set_speaker_targets(self, spk_targets: Optional[torch.Tensor] = None, 
                           bg_spk_targets: Optional[torch.Tensor] = None):
        """
        Set speaker targets for the model.
        
        Args:
            spk_targets: Main speaker targets tensor
            bg_spk_targets: Background speaker targets tensor
        """
        self.spk_targets = spk_targets
        self.bg_spk_targets = bg_spk_targets

    def clear_speaker_targets(self):
        """Clear speaker targets."""
        self.spk_targets = None
        self.bg_spk_targets = None
    
    def solve_length_mismatch(self, x, mask):
        """
        Solve length mismatch between x and mask.
        """
        if mask is None:
            mask = torch.ones_like(x[:, :, 0])

        if mask.shape[1] < x.shape[1]:
            # pad zero to the left
            mask = torch.nn.functional.pad(mask, (x.shape[1] - mask.shape[1], 0), mode='constant', value=1)

        if mask.shape[1] > x.shape[1]:
            mask = mask[:, -x.shape[1]:]

        return mask

    def mask_with_speaker_targets(self, x, spk_targets):
        """
        Mask the input with speaker targets.
        """
        mask = self.solve_length_mismatch(x, spk_targets)
        x_spk = x * mask.unsqueeze(2)
        return x_spk

    def concat_with_speaker_targets(self, x, spk_targets):
        """
        Concatenate the input with speaker targets.
        """
        mask = self.solve_length_mismatch(x, spk_targets)
        x_spk = x * mask.unsqueeze(2)
        return x_spk

# Inference mixins
class MultiTalkerASRMixin(ABC):
    """
    Mixin class for models that need multi-talker ASR functionality.
    """

    def __init__(self):
        super().__init__()

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

        # Set processed speaker targets for valid speakers using mixin method
        self.set_speaker_targets(spk_targets[valid_speakers.reshape(-1)])
    
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

        # Process speaker targets for valid speakers only
        valid_spk_targets = spk_targets[valid_speakers.reshape(-1)]
        
        # Set speaker targets for hooks using mixin method
        self.set_speaker_targets(valid_spk_targets)
    
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
        
        # mask the processed signal using valid speaker targets
        mask = valid_spk_targets.unsqueeze(-1).repeat(1, 1, 8).flatten(1, 2)
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