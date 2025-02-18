# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text import _AudioTextDataset
from nemo.collections.asr.data.audio_to_text_dali import AudioToBPEDALIDataset
from nemo.collections.asr.data.audio_to_text_lhotse_query_diar_target_speaker import LhotseSpeechToTextQueryDiarTgtSpkBpeDataset
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.metrics.wer import WER
# from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_tgt_spk_models import EncDecHybridRNNTCTCTgtSpkBPEModel
from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel
from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecoding, CTCBPEDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding, RNNTBPEDecodingConfig
from nemo.collections.asr.parts.utils.asr_batching import get_semi_sorted_batch_sampler
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging, model_utils
from nemo.core.classes.mixins import AccessMixin
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs

from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_hidden_length_from_sample_length

from lhotse.dataset.collation import collate_matrices


class MetaCatResidual(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(input_size, output_size*2),
            torch.nn.ReLU(),
            torch.nn.Linear(output_size*2, output_size)
        )

    def forward(self, x, mask):
        """
        x: (B, T, D)
        mask: (B, T, D)
        """
        if mask.shape[1] < x.shape[1]:
            mask = F.pad(mask, (0,0,0, x.shape[1] - mask.shape[1]), mode='replicate')

        if mask.shape[1] > x.shape[1]:
            mask = mask[:, :x.shape[1],:]
        #only pick speaker 0
        concat_enc_states = x.unsqueeze(2) * mask[:,:,:1].unsqueeze(3)
        concat_enc_states = concat_enc_states.flatten(2,3)
        x = x + self.proj(concat_enc_states)   

        return x


class EncDecHybridRNNTCTCQueryDiarTgtSpkBPEModel(EncDecHybridRNNTCTCBPEModel):
    """Base class for encoder decoder RNNT-based models with auxiliary CTC decoder/loss and subword tokenization."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.num_speakers = cfg.model_defaults.get('num_speakers', 4)

        if 'diar_model_path' in self.cfg:
            self.diar = True
            self._init_diar_model()

            self.pre_diar_kernel = cfg.get('pre_diar_kernel', None)
            self.fc_diar_kernel = cfg.get('fc_diar_kernel', None)
            self.diar_kernel_layer = cfg.get('diar_kernel_layer', [])
            self.binarize_diar_preds_threshold = cfg.get('binary_diar_preds', False)
            self.spk_supervision_strategy = cfg.get('spk_supervision_strategy', 'diar')
            self.rttm_mix_prob = cfg.get('rttm_mix_prob', 0)
            
            if isinstance(self.diar_kernel_layer, int):
                self.diar_kernel_layer = list(range(1, 1 + self.diar_kernel_layer))
            elif isinstance(self.diar_kernel_layer, ListConfig):
                pass
            elif self.fc_diar_kernel is not None:
                raise ValueError(f"Invalid diar_kernel_layer {self.diar_kernel_layer}, should be int or list")

            in_size = cfg.model_defaults.enc_hidden
            out_size = cfg.model_defaults.enc_hidden
            self.spk_kernels = torch.nn.ModuleDict()

            # diar kernel for pre-encoder
            if 0 in self.diar_kernel_layer:
                if self.pre_diar_kernel == 'metacat_residule':
                    # projection layer
                    self.spk_kernels['0'] = MetaCatResidual(in_size, out_size)

                elif self.pre_diar_kernel == 'sinusoidal':
                    # self.pre_proj = self.get_sinusoid_position_encoding(self.num_speakers, cfg.model_defaults.enc_hidden)
                    pass

            # diar kernel for fast conformer encoder layers
            if self.fc_diar_kernel == 'metacat_residule':
                for l_i in self.diar_kernel_layer:
                    if l_i != 0:
                        self.spk_kernels[str(l_i)] = MetaCatResidual(in_size, out_size)
            elif self.fc_diar_kernel == 'sinusoidal':
                # self.post_proj = self.get_sinusoid_position_encoding(self.num_speakers, cfg.model_defaults.enc_hidden)
                pass
            
            logging.info(f"Registered speaker kernels to layers {list(self.spk_kernels.keys())}")
            # register the speaker injection model to each layer
            for k, spk_kernel in self.spk_kernels.items():
                hook_func = self.get_hook_function(k)
                idx = int(k)
                if idx == 0:
                    self.encoder.pre_encode.register_forward_hook(hook_func)
                else:
                    self.encoder.layers[idx-1].register_forward_hook(hook_func)
    
    def _init_diar_model(self):
        """
        Initialize the speaker model.
        """
        logging.info(f"Initializing diarization model from pretrained checkpoint {self.cfg.diar_model_path}")

        model_path = self.cfg.diar_model_path
        # model_path = '/home/jinhanw/workdir/workdir_nemo_diarization/checkpoints/sortformer_rebase/im303a-ft7_epoch6-19_sortformer_rebase.nemo'

        try:
            if model_path.endswith('.nemo'):
                pretrained_diar_model = SortformerEncLabelModel.restore_from(model_path, map_location="cpu")
                logging.info("Diarization Model restored locally from {}".format(model_path))
            elif model_path.endswith('.ckpt'):
                pretrained_diar_model = SortformerEncLabelModel.load_from_checkpoint(model_path, map_location="cpu")
                logging.info("Diarization Model restored locally from {}".format(model_path))
            else:
                pretrained_diar_model = SortformerEncLabelModel.from_pretrained(model_path)
                logging.info("Diarization Model restored from NGC")
        except:
            pretrained_diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
            logging.info("Diarization Model restored from NGC")
        
        self.diarization_model = pretrained_diar_model
        if self.cfg.freeze_diar:
            self.diarization_model.eval()


    def _get_probablistic_mix(self, diar_preds, spk_targets, rttm_mix_prob:float=0.0):
        """ 
        Sample a probablistic mixture of speaker labels for each time step then apply it to the diarization predictions and the speaker targets.
        
        Args:
            diar_preds (Tensor): Tensor of shape [B, T, D] representing the diarization predictions.
            spk_targets (Tensor): Tensor of shape [B, T, D] representing the speaker targets.
            
        Returns:
            mix_prob (float): Tensor of shape [B, T, D] representing the probablistic mixture of speaker labels for each time step.
        """
        batch_probs_raw = torch.distributions.categorical.Categorical(probs=torch.tensor([(1-rttm_mix_prob), rttm_mix_prob]).repeat(diar_preds.shape[0],1)).sample()
        batch_probs = (batch_probs_raw.view(diar_preds.shape[0], 1, 1).repeat(1, diar_preds.shape[1], diar_preds.shape[2])).to(diar_preds.device)
        batch_diar_preds = (1 - batch_probs) * diar_preds + batch_probs * spk_targets
        return batch_diar_preds 

    def get_hook_function(self, index):
        """Returns a forward hook function that applies the metacat modules"""
        def hook(module, input, output):

            if isinstance(output, tuple): # pre-encode, B x T x D
                x, *cache = output
                x = self.spk_kernels[index](x, self.spk_targets)
                
                return (x, *cache)
            
            else:
                return self.spk_kernels[index](output, self.spk_targets)
        
        return hook
    
    def forward_diar(
        self,
        input_signal=None,
        input_signal_length=None,
        is_raw_waveform_input=True,
    ):
        # preds, _preds, attn_score_stack, total_memory_list, encoder_states_list = self.diarization_model.forward(audio_signal=input_signal, audio_signal_length=input_signal_length, is_raw_waveform_input=is_raw_waveform_input)
        preds = self.diarization_model.forward(audio_signal=input_signal, audio_signal_length=input_signal_length, is_raw_waveform_input=is_raw_waveform_input)


        # return preds, _preds, attn_score_stack, total_memory_list, encoder_states_list
        return preds

    def train_val_forward(self, batch, batch_nb):
        signal, signal_len, signal_w_query, signal_w_query_len, transcript, transcript_len, spk_targets, spk_mappings = batch
        query_len = signal_w_query_len - signal_len
        query_frame_len = [get_hidden_length_from_sample_length(x, 160, 8) for x in query_len]

        # speaker targetes manipulation
        if signal.shape[1] == 80:
            is_raw_waveform_input=False
        else:
            is_raw_waveform_input=True
        if self.diar == True:
            if self.cfg.spk_supervision_strategy == 'rttm':
                if spk_targets is not None:
                    diar_preds = spk_targets 
                else:
                    raise ValueError("`spk_targets` is required for speaker supervision strategy 'rttm'")
            elif self.cfg.spk_supervision_strategy == 'diar':
                with torch.set_grad_enabled(not self.cfg.freeze_diar):
                    # diar_preds, _preds, attn_score_stack, total_memory_list, encoder_states_list = self.forward_diar(signal, signal_len)
                    diar_preds = self.forward_diar(signal_w_query, signal_w_query_len, is_raw_waveform_input)
                    if self.binarize_diar_preds_threshold:
                        diar_preds = torch.where(diar_preds > self.binarize_diar_preds_threshold, torch.tensor(1), torch.tensor(0)).to(encoded.device).half()
                if diar_preds is None:
                    raise ValueError("`diar_pred`is required for speaker supervision strategy 'diar'")
            elif self.cfg.spk_supervision_strategy == 'mix':
                with torch.set_grad_enabled(not self.cfg.freeze_diar):
                    # diar_preds, _preds, attn_score_stack, total_memory_list, encoder_states_list = self.forward_diar(signal, signal_len)
                    diar_preds = self.forward_diar(signal_w_query, signal_w_query_len, is_raw_waveform_input)
                    if self.binarize_diar_preds_threshold:
                        diar_preds = torch.where(diar_preds > self.binarize_diar_preds_threshold, torch.tensor(1), torch.tensor(0)).to(encoded.device)
                diar_preds = self._get_probablistic_mix(diar_preds=diar_preds, spk_targets=spk_targets, rttm_mix_prob=float(self.cfg.rttm_mix_prob))
            else:
                raise ValueError(f"Invalid RTTM strategy {self.cfg.spk_supervision_strategy} is not supported.")
            
        #remove query part from diar_preds and re-collate
        diar_preds_no_query = []
        for i in range(signal.shape[0]):
            diar_preds_no_query.append(diar_preds[i,query_frame_len[i]:,:])
        self.spk_targets = collate_matrices(diar_preds_no_query)
        #truncate diar_preds to be the same length as encoded
        if (isinstance(batch, DALIOutputs) and batch.has_processed_signal) or signal.shape[1] == 80:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)

        #encoded (B, D, T)

        return encoded, encoded_len, transcript, transcript_len
    

    def _setup_dataloader_from_config(self, config: Optional[Dict]):

        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseSpeechToTextQueryDiarTgtSpkBpeDataset(cfg = config,
                    tokenizer=self.tokenizer,
                ),
                tokenizer=self.tokenizer,
            )

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
        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
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
            'use_start_end_token': self.cfg.validation_ds.get('use_start_end_token', False),
            'num_speakers': self.cfg.test_ds.get('num_speakers',4),
            'spk_tar_all_zero': self.cfg.test_ds.get('spk_tar_all_zero',False),
            'num_sample_per_mel_frame': self.cfg.test_ds.get('num_sample_per_mel_frame',160),
            'num_mel_frame_per_asr_frame': self.cfg.test_ds.get('num_mel_frame_per_asr_frame',8),
            'add_separater_audio': self.cfg.test_ds.get('add_separater_audio',True),
            'separater_freq': self.cfg.test_ds.get('separater_freq',500),
            'separater_duration': self.cfg.test_ds.get('separater_duration',1),
            'separater_unvoice_ratio': self.cfg.test_ds.get('separater_unvoice_ratio', 0.3),
            'fix_query_audio_end_time': self.cfg.test_ds.get('fix_query_audio_end_time', False),
            'add_special_token': self.cfg.test_ds.get('add_special_token', True),
            'shuffle_spk_mapping': self.cfg.test_ds.get('shuffle_spk_mapping',False),
            'inference_mode': self.cfg.test_ds.get('inference_mode', True)
        }

        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer


    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        return results
    
        # training_step include speaker information
    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        encoded, encoded_len, transcript, transcript_len = self.train_val_forward(batch, batch_nb)

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        if (sample_id + 1) % log_every_n_steps == 0:
            compute_wer = True
        else:
            compute_wer = False

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:  # If fused Joint-Loss-WER is used
            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            tensorboard_logs = {
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        if self.ctc_loss_weight > 0:
            log_probs = self.ctc_decoder(encoder_output=encoded)
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            tensorboard_logs['train_rnnt_loss'] = loss_value
            tensorboard_logs['train_ctc_loss'] = ctc_loss
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            if compute_wer:
                self.ctc_wer.update(
                    predictions=log_probs,
                    targets=transcript,
                    targets_lengths=transcript_len,
                    predictions_lengths=encoded_len,
                )
                ctc_wer, _, _ = self.ctc_wer.compute()
                self.ctc_wer.reset()
                tensorboard_logs.update({'training_batch_wer_ctc': ctc_wer})

        # note that we want to apply interctc independent of whether main ctc
        # loss is used or not (to allow rnnt + interctc training).
        # assuming ``ctc_loss_weight=0.3`` and interctc is applied to a single
        # layer with weight of ``0.1``, the total loss will be
        # ``loss = 0.9 * (0.3 * ctc_loss + 0.7 * rnnt_loss) + 0.1 * interctc_loss``
        loss_value, additional_logs = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=compute_wer
        )
        tensorboard_logs.update(additional_logs)
        tensorboard_logs['train_loss'] = loss_value
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}
    

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)
        encoded, encoded_len, transcript, transcript_len = self.train_val_forward(batch, batch_idx)

        tensorboard_logs = {}
        loss_value = None

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:
                decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
                joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)

                loss_value = self.loss(
                    log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
                )

                tensorboard_logs['val_loss'] = loss_value

            self.wer.update(
                predictions=encoded,
                predictions_lengths=encoded_len,
                targets=transcript,
                targets_lengths=transcript_len,
            )
            wer, wer_num, wer_denom = self.wer.compute()
            self.wer.reset()

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        else:
            # If experimental fused Joint-Loss-WER is used
            compute_wer = True

            if self.compute_eval_loss:
                decoded, target_len, states = self.decoder(targets=transcript, target_length=transcript_len)
            else:
                decoded = None
                target_len = transcript_len

            # Fused joint step
            loss_value, wer, wer_num, wer_denom = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoded,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=target_len,
                compute_wer=compute_wer,
            )

            if loss_value is not None:
                tensorboard_logs['val_loss'] = loss_value

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        log_probs = self.ctc_decoder(encoder_output=encoded)
        if self.compute_eval_loss:
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            tensorboard_logs['val_ctc_loss'] = ctc_loss
            tensorboard_logs['val_rnnt_loss'] = loss_value
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            tensorboard_logs['val_loss'] = loss_value
        self.ctc_wer.update(
            predictions=log_probs,
            targets=transcript,
            targets_lengths=transcript_len,
            predictions_lengths=encoded_len,
        )
        ctc_wer, ctc_wer_num, ctc_wer_denom = self.ctc_wer.compute()
        self.ctc_wer.reset()
        tensorboard_logs['val_wer_num_ctc'] = ctc_wer_num
        tensorboard_logs['val_wer_denom_ctc'] = ctc_wer_denom
        tensorboard_logs['val_wer_ctc'] = ctc_wer

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        loss_value, additional_logs = self.add_interctc_losses(
            loss_value,
            transcript,
            transcript_len,
            compute_wer=True,
            compute_loss=self.compute_eval_loss,
            log_wer_num_denom=True,
            log_prefix="val_",
        )
        if self.compute_eval_loss:
            # overriding total loss value. Note that the previous
            # rnnt + ctc loss is available in metrics as "val_final_loss" now
            tensorboard_logs['val_loss'] = loss_value
        tensorboard_logs.update(additional_logs)
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        return tensorboard_logs



    
