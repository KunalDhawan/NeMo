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

from nemo.core.classes.common import typecheck
from nemo.utils import logging

class EncDecRNNTBPEQLTSASRModel(EncDecRNNTBPEModel):
    """Base class for encoder decoder RNNT-based models with subword tokenization."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

    def _init_diar_model(self, 
                         model_path = None,
                         num_speakers=None,
                         freeze_diar=None,
                         soft_decision=None,
                         gt_decision=None,
                         streaming_mode=None,
                         streaming_cfg=None):
        """
        Initialize the speaker model.
        """
        logging.info(f"Initializing diarization model from pretrained checkpoint {model_path}")
        self.num_speakers = num_speakers
        self.freeze_diar = freeze_diar
        self.soft_decision = soft_decision
        self.gt_decision = gt_decision
        self.streaming_mode = streaming_mode

        try:
            if model_path.endswith('.nemo'):
                pretrained_diar_model = SortformerEncLabelModel.restore_from(model_path, map_location="cpu")
                logging.info("Diarization Model restored locally from {}".format(model_path))
            elif model_path.endswith('.ckpt'):
                pretrained_diar_model = SortformerEncLabelModel.load_from_checkpoint(model_path, map_location="cpu")
                logging.info("Diarization Model restored locally from {}".format(model_path))
            else:
                pretrained_diar_model = SortformerEncLabelModel.from_pretrained(model_path)
                logging.info("Diarization Model restored from {} at NGC".format(model_path))
        except:
            pretrained_diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
            logging.info("Diarization Model restored from nvidia/diar_sortformer_4spk-v1 at NGC")
        
        if freeze_diar:
            pretrained_diar_model.eval()

        # Register diarization model 
        self.register_nemo_submodule(
            name="diarization_model",
            model=pretrained_diar_model,
            config_field="diarization_model" 
        )

        # Change the diarization model cfg for streaming inference
        if self.streaming_mode:
            self.diarization_model.streaming_mode = self.streaming_mode
            self.diarization_model.sortformer_modules.step_len = streaming_cfg.step_len
            self.diarization_model.sortformer_modules.mem_len = streaming_cfg.mem_len
            self.diarization_model.sortformer_modules.step_left_context = streaming_cfg.step_left_context
            self.diarization_model.sortformer_modules.step_right_context = streaming_cfg.step_right_context
            self.diarization_model.sortformer_modules.fifo_len = streaming_cfg.fifo_len
            self.diarization_model.sortformer_modules.mem_refresh_rate = streaming_cfg.mem_refresh_rate

    def forward_diar(
        self,
        audio_signal=None,
        audio_signal_length=None,
        spk_targets=None,
    ):
        
        if self.gt_decision:
            pass
        else:
            # Get diarization predictions from model
            with torch.set_grad_enabled(not self.freeze_diar):
                spk_targets = self.diarization_model(audio_signal=audio_signal, audio_signal_length=audio_signal_length)

        if self.soft_decision:
            pass
        else:
            spk_targets = (spk_targets > 0.5).to(spk_targets.dtype)

        return spk_targets

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if config.get("use_lhotse"):
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

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length, vad_mask=self.spk_targets)
        return encoded, encoded_len

    def training_step(self, batch, batch_nb):
        
        signal, signal_len, transcript, transcript_len, spk_targets, spk_ids = batch

        spk_targets = self.forward_diar(audio_signal=signal, audio_signal_length=signal_len, spk_targets=spk_targets)
        # Extract speaker-specific targets based on speaker IDs
        self.spk_targets = torch.stack([spk_targets[i, :, spk_ids[i]] for i in range(len(spk_ids))])

        batch = (signal, signal_len, transcript, transcript_len)

        return super().training_step(batch, batch_nb)


    def validation_pass(self, batch, batch_idx, dataloader_idx=0):

        signal, signal_len, transcript, transcript_len, spk_targets, spk_ids = batch

        spk_targets = self.forward_diar(audio_signal=signal, audio_signal_length=signal_len, spk_targets=spk_targets)
        # Extract speaker-specific targets based on speaker IDs
        self.spk_targets = torch.stack([spk_targets[i, :, spk_ids[i]] for i in range(len(spk_ids))])

        batch = (signal, signal_len, transcript, transcript_len)

        return super().validation_pass(batch, batch_idx, dataloader_idx)

    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):

        signal, signal_len, transcript, transcript_len, spk_targets, spk_ids = batch

        spk_targets = self.forward_diar(audio_signal=signal, audio_signal_length=signal_len, spk_targets=spk_targets)
        
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
        if len(spk_targets.size()) == 3:
            # N: # speakers
            # spk_targets: (B, T, N) -> (BN, T)
            self.spk_targets = spk_targets[:, :, :n_mix].transpose(1, 2).reshape(-1, spk_targets.size(1))

            # processed_signal: (B, T, D) -> (BN, T, D)
            processed_signal = processed_signal.unsqueeze(1).repeat(1, n_mix, 1, 1).reshape(-1, processed_signal.size(1), processed_signal.size(2))
            processed_signal_length = processed_signal_length.unsqueeze(1).repeat(1, n_mix).reshape(-1)

        if cache_last_channel_len.shape[0] != processed_signal.shape[0]:
            cache_last_channel = cache_last_channel.unsqueeze(2).repeat(1, 1, n_mix, 1, 1).reshape(cache_last_channel.size(0), -1, cache_last_channel.size(2), cache_last_channel.size(3))
            cache_last_time = cache_last_time.unsqueeze(2).repeat(1, 1, n_mix, 1, 1).reshape(cache_last_time.size(0), -1, cache_last_time.size(2), cache_last_time.size(3))
            cache_last_channel_len = cache_last_channel_len.unsqueeze(1).repeat(1, n_mix).reshape(-1)

        return super().conformer_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=keep_all_outputs,
            previous_hypotheses=previous_hypotheses,
            previous_pred_out=previous_pred_out,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
            return_transcription=return_transcription,
            return_log_probs=return_log_probs
        )