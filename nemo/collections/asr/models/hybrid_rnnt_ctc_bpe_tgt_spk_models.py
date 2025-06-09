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
from lightning.pytorch import Trainer
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text import _AudioTextDataset
from nemo.collections.asr.data.audio_to_text_dali import AudioToBPEDALIDataset
from nemo.collections.asr.data.audio_to_text_lhotse_target_speaker import LhotseSpeechToTextTgtSpkBpeDataset
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.metrics.wer import WER
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
import math
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_hidden_length_from_sample_length
from lhotse.dataset.collation import collate_vectors, collate_matrices
from tqdm import tqdm


class EncDecHybridRNNTCTCTgtSpkBPEModel(EncDecHybridRNNTCTCBPEModel):
    """Base class for encoder decoder RNNT-based models with auxiliary CTC decoder/loss and subword tokenization."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        if 'diar_model_path' in self.cfg:
            self.diar = True
            # Initialize the speaker branch
            self._init_diar_model()

            self.num_speakers = cfg.model_defaults.get('num_speakers', 4)
            
            # layer normalization, ln, l2, or None
            self.norm = cfg.get('norm', None)

            if cfg.norm == 'ln':
                self.asr_norm = torch.nn.LayerNorm(cfg.model_defaults.enc_hidden)
                self.diar_norm = torch.nn.LayerNorm(4)

            self.kernel_norm = cfg.get('kernel_norm',None)

            # projection layer
            self.diar_kernel_type = cfg.get('diar_kernel_type', None)

            proj_in_size = self.num_speakers + cfg.model_defaults.enc_hidden
            proj_out_size = cfg.model_defaults.enc_hidden
            self.joint_proj = torch.nn.Sequential(
                torch.nn.Linear(proj_in_size, proj_out_size*2),
                torch.nn.ReLU(),
                torch.nn.Linear(proj_out_size*2, proj_out_size)
            )
            self.diar_kernal = self.joint_proj

            if self.diar_kernel_type == 'sinusoidal':
                self.diar_kernel = self.get_sinusoid_position_encoding(self.num_speakers, cfg.model_defaults.enc_hidden)
            elif self.diar_kernel_type == 'metacat':
                # projection layer
                proj_in_size = self.num_speakers * cfg.model_defaults.enc_hidden
                proj_out_size = cfg.model_defaults.enc_hidden
                self.joint_proj = torch.nn.Sequential(
                    torch.nn.Linear(proj_in_size, proj_out_size*2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(proj_out_size*2, proj_out_size)
                )
                self.diar_kernel = self.joint_proj
            elif self.diar_kernel_type == 'metacat_residule' or self.diar_kernel_type == 'metacat_residule_early':
                # projection layer
                proj_in_size = cfg.model_defaults.enc_hidden
                proj_out_size = cfg.model_defaults.enc_hidden
                self.joint_proj = torch.nn.Sequential(
                    torch.nn.Linear(proj_in_size, proj_out_size*2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(proj_out_size*2, proj_out_size)
                )
                self.diar_kernel = self.joint_proj
            elif self.diar_kernel_type == 'metacat_residule_projection':
                proj_in_size = cfg.model_defaults.enc_hidden + self.num_speakers
                proj_out_size = cfg.model_defaults.enc_hidden
                self.joint_proj = torch.nn.Sequential(
                    torch.nn.Linear(proj_in_size, proj_out_size*2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(proj_out_size*2, proj_out_size)
                )
                self.diar_kernel = self.joint_proj                

            #binarize diar_pred
            self.binarize_diar_preds_threshold = cfg.get('binarize_diar_preds_threshold', None)
        
        else:
            self.diar = False

    def _init_diar_model(self):
        """
        Initialize the speaker model.
        """

        model_path = self.cfg.diar_model_path
        # model_path = '/home/jinhanw/workdir/workdir_nemo_diarization/checkpoints/sortformer_rebase/diar_sortformer_4spk-v1.nov20_2024.nemo'

        if model_path.endswith('.nemo'):
            pretrained_diar_model = SortformerEncLabelModel.restore_from(model_path, map_location="cpu")
            logging.info("Diarization Model restored locally from {}".format(model_path))
        elif model_path.endswith('.ckpt'):
            pretrained_diar_model = SortformerEncLabelModel.load_from_checkpoint(model_path, map_location="cpu")
            logging.info("Diarization Model restored locally from {}".format(model_path))
        else:
            pretrained_diar_model = None
            logging.info("Model path incorrect")

        self.diarization_model = pretrained_diar_model

        if self.cfg.freeze_diar:
           self.diarization_model.eval()
        #disable streaming mode for now, activate using _reset_streaming_state if necessary
        self.diarization_model.streaming_mode = False
    def forward_diar(
        self,
        input_signal=None,
        input_signal_length=None,
        is_raw_waveform_input=True,
    ):
        # preds, _preds, attn_score_stack, total_memory_list, encoder_states_list = self.diarization_model.forward(audio_signal=input_signal, audio_signal_length=input_signal_length, is_raw_waveform_input=is_raw_waveform_input)
        preds = self.diarization_model.forward(audio_signal=input_signal, audio_signal_length=input_signal_length)#, 
        self.diar_preds = preds

        # return preds, _preds, attn_score_stack, total_memory_list, encoder_states_list
        return preds

    def fix_diar_output(
        self,
        diar_pred,
        asr_frame_count
    ):
        """
        Duct-tapeß solution for extending the speaker predictions 
        """
        # Extract the first and last embeddings along the second dimension
        # first_emb = diar_pred[:, 0, :].unsqueeze(1)
        if diar_pred.shape[1] < asr_frame_count:
            last_emb = diar_pred[:, -1, :].unsqueeze(1)

            #number of repeatitions needed
            additional_frames = asr_frame_count - diar_pred.shape[1]

            # Create tensors of repeated first and last embeddings
            # first_repeats = first_emb.repeat(1, additional_frames // 2, 1)
            # last_repeats = last_emb.repeat(1, (additional_frames + 1) // 2, 1)
            last_repeats = last_emb.repeat(1, additional_frames, 1)

            # Concatenate the repeated tensors with the original embeddings
            # extended_diar_preds = torch.cat((first_repeats, diar_pred, last_repeats), dim=1)
            extended_diar_preds = torch.cat((diar_pred, last_repeats), dim=1)

            return extended_diar_preds
        else:
            # temporary solution if diar_pred longer than encoded
            return diar_pred[:, :asr_frame_count, :]

    
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

    def get_sinusoid_position_encoding(self, max_position, embedding_dim):
        """
        Generates a sinusoid position encoding matrix.
        
        Args:
        - max_position (int): The maximum position to generate encodings for.
        - embedding_dim (int): The dimension of the embeddings.
        
        Returns:
        - torch.Tensor: A tensor of shape (max_position, embedding_dim) containing the sinusoid position encodings.
        """
        position = np.arange(max_position)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))
        
        position_encoding = np.zeros((max_position, embedding_dim))
        position_encoding[:, 0::2] = np.sin(position * div_term)
        position_encoding[:, 1::2] = np.cos(position * div_term)
        
        # Convert the numpy array to a PyTorch tensor
        position_encoding_tensor = torch.tensor(position_encoding, dtype=torch.float32)
        
        return position_encoding_tensor


    def train_val_forward(self, batch, batch_nb):

        signal, signal_len, transcript, transcript_len, spk_targets, spk_mappings = batch

        # speaker targetes
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
                    diar_preds = self.forward_diar(signal, signal_len, is_raw_waveform_input)
                    if self.binarize_diar_preds_threshold:
                        diar_preds = torch.where(diar_preds > self.binarize_diar_preds_threshold, torch.tensor(1), torch.tensor(0)).to(signal.device).float()
                if diar_preds is None:
                    raise ValueError("`diar_pred`is required for speaker supervision strategy 'diar'")
            elif self.cfg.spk_supervision_strategy == 'mix':
                with torch.set_grad_enabled(not self.cfg.freeze_diar):
                    # diar_preds, _preds, attn_score_stack, total_memory_list, encoder_states_list = self.forward_diar(signal, signal_len)
                    diar_preds = self.forward_diar(signal, signal_len, is_raw_waveform_input)
                    if self.binarize_diar_preds_threshold:
                        diar_preds = torch.where(diar_preds > self.binarize_diar_preds_threshold, torch.tensor(1), torch.tensor(0)).to(encoded.device)
                diar_preds = self._get_probablistic_mix(diar_preds=diar_preds, spk_targets=spk_targets, rttm_mix_prob=float(self.cfg.rttm_mix_prob))
            else:
                raise ValueError(f"Invalid RTTM strategy {self.cfg.spk_supervision_strategy} is not supported.")

        if (isinstance(batch, DALIOutputs) and batch.has_processed_signal) or signal.shape[1] == 80:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
    
        encoded = torch.transpose(encoded, 1, 2) # B * D * T -> B * T * D
        if self.diar == True:        
            # Speaker mapping shuffling to equalize the speaker label's distributions
            if self.cfg.shuffle_spk_mapping:
                diar_preds = apply_spk_mapping(diar_preds, spk_mappings)

            if(diar_preds.shape[1]!=encoded.shape[1]):
            # KD duct-tape solution for extending the speaker predictions 
                asr_frame_count = encoded.shape[1]
                diar_preds = self.fix_diar_output(diar_preds, asr_frame_count)

            # Normalize the features
            if self.norm == 'ln':
                diar_preds = self.diar_norm(diar_preds)
                encoded = self.asr_norm(encoded)
            elif self.norm == 'l2':
                diar_preds = torch.nn.functional.normalize(diar_preds, p=2, dim=-1)
                encoded = torch.nn.functional.normalize(encoded, p=2, dim=-1)
            
            if diar_preds.shape[1] > encoded.shape[1]:
                diar_preds = diar_preds[:, :encoded.shape[1], :]

            if self.diar_kernel_type == 'sinusoidal':
                speaker_infusion_asr = torch.matmul(diar_preds, self.diar_kernel.to(diar_preds.device))
                if self.kernel_norm == 'l2':
                    speaker_infusion_asr = torch.nn.functional.normalize(speaker_infusion_asr, p=2, dim=-1)
                encoded = speaker_infusion_asr + encoded
            elif self.diar_kernel_type == 'metacat':
                concat_enc_states = encoded.unsqueeze(2) * diar_preds.unsqueeze(3)
                concat_enc_states = concat_enc_states.flatten(2,3)
                encoded = self.joint_proj(concat_enc_states)
            elif self.diar_kernel_type == 'metacat_residule':
                #only pick speaker 0
                concat_enc_states = encoded.unsqueeze(2) * diar_preds[:,:,:1].unsqueeze(3)
                concat_enc_states = concat_enc_states.flatten(2,3)
                encoded = encoded + self.joint_proj(concat_enc_states)    
            elif self.diar_kernel_type == 'metacat_residule_early':
                #only pick speaker 0
                concat_enc_states = encoded.unsqueeze(2) * diar_preds[:,:,:1].unsqueeze(3)
                concat_enc_states = concat_enc_states.flatten(2,3)
                encoded = self.joint_proj(encoded + concat_enc_states)
            elif self.diar_kernel_type == 'metacat_residule_projection':
                #only pick speaker 0 and add diar_preds
                concat_enc_states = encoded.unsqueeze(2) * diar_preds[:,:,:1].unsqueeze(3)
                concat_enc_states = concat_enc_states.flatten(2,3)
                encoded = encoded + concat_enc_states
                concat_enc_states = torch.cat([encoded, diar_preds], dim = -1)
                encoded = self.joint_proj(concat_enc_states)
            else: #projection
                concat_enc_states = torch.cat([encoded, diar_preds], dim=-1)
                encoded = self.joint_proj(concat_enc_states)
        else:
            encoded = encoded
        
        encoded = torch.transpose(encoded, 1, 2) # B * T * D -> B * D * T
        return encoded, encoded_len, transcript, transcript_len


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

    def _setup_dataloader_from_config(self, config: Optional[Dict]):

        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseSpeechToTextTgtSpkBpeDataset(cfg = config,
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
    
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None, diar_preds=None
    ):
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

    def _reset_streaming_state(self, batch_size, async_streaming):
        #async streaming is for batchview dataset configuration
        #otherwise use sync streaming with batch_size = 1--> mandatory for batchview sample (history matters)
        self.diarization_model.streaming_mode = True
        self.diarization_model.sortformer_modules.chunk_len = 376
        self.diarization_model.sortformer_modules.fifo_len = 376
        self.query_pred = None
        self.total_preds = torch.zeros((batch_size, 0, 4), device = self.diarization_model.device)
        self.diarization_model.sortformer_modules.chunk_len_audio = int(self.diarization_model.sortformer_modules.chunk_len / 12.5 * 16000)
        self.diarization_model.sortformer_modules.fifo_len_audio = int(self.diarization_model.sortformer_modules.fifo_len / 12.5 * 16000)
        self.streaming_state = self.diarization_model.sortformer_modules.init_streaming_state(
            batch_size = batch_size,
            device = self.diarization_model.device,
            async_streaming = async_streaming
        )
        self.streaming_state.query_pred_len = torch.zeros((batch_size), device = self.diarization_model.device, dtype = torch.int32)
        self.streaming_state.query_len_audio = torch.zeros((batch_size), device = self.diarization_model.device, dtype = torch.int32)
        self.diarization_model.async_streaming = async_streaming
        self.batch_index = list(np.arange(batch_size)) #record global batch index for each sample
        

    def forward_sortformer_streaming(self, signal, signal_len, query_len, chunk_len, buffer_len, initial_buffer=False, temp_buffer_index=0, sortformer_loader_level='emb', new_batch_keys=None, left_context=0, tokens_per_chunk=0):

        if new_batch_keys is not None and len(new_batch_keys) != self.streaming_state.spkcache.shape[0]:
            curr_batch_key = []
            for i, global_index in enumerate(self.batch_index):
                if global_index in new_batch_keys:
                    curr_batch_key.append(i)
            # Update streaming state tensors to only include valid rows
            # emb level
            self.streaming_state.spkcache = self.streaming_state.spkcache[curr_batch_key]
            self.streaming_state.spkcache_lengths = self.streaming_state.spkcache_lengths[curr_batch_key] 
            self.streaming_state.fifo = self.streaming_state.fifo[curr_batch_key]
            self.streaming_state.fifo_lengths = self.streaming_state.fifo_lengths[curr_batch_key]
            self.streaming_state.query_pred_len = self.streaming_state.query_pred_len[curr_batch_key]
            # audio level
            self.streaming_state.spkcache_audio = self.streaming_state.spkcache_audio[curr_batch_key]
            self.streaming_state.spkcache_audio_lengths = self.streaming_state.spkcache_audio_lengths[curr_batch_key]
            self.streaming_state.fifo_audio = self.streaming_state.fifo_audio[curr_batch_key]
            self.streaming_state.fifo_audio_lengths = self.streaming_state.fifo_audio_lengths[curr_batch_key]
            self.streaming_state.query_len_audio = self.streaming_state.query_len_audio[curr_batch_key]

            new_query_pred = []
            new_query_pred = [self.query_pred[i] for i in curr_batch_key]
            self.query_pred = new_query_pred
            self.batch_index = new_batch_keys
            
            

        if signal.shape[1] == 80:
            is_raw_waveform_input=False
        else:
            is_raw_waveform_input=True
        if self.query_pred is None:
            diar_input_signal_len = signal_len
            # diar_input_signal = signal[:,:signal_len]
            diar_input_signal = signal
        else:
            chunk_len += left_context
            # import ipdb; ipdb.set_trace()
            diar_input_signal = torch.empty((signal.size(0), chunk_len), 
                dtype=signal.dtype,
                device=signal.device)
            for i in range(signal.size(0)):
                diar_input_signal[i,:] = signal[i,signal_len[i] - int(chunk_len):signal_len[i]]
            # diar_input_signal = signal[:,-int(chunk_len):]
            diar_input_signal_len = torch.tensor([int(chunk_len)], device = signal.device).expand(signal.size(0))
        if self.cfg.spk_supervision_strategy == 'diar':
            with torch.set_grad_enabled(not self.cfg.freeze_diar):
                # diar_preds = self.forward_diar(signal, signal_len, is_raw_waveform_input)
                processed_signal, processed_signal_len = self.diarization_model.process_signal(
                    audio_signal = diar_input_signal,
                    audio_signal_length = diar_input_signal_len,
                )
                feat_len = processed_signal.shape[2]
                num_chunks = math.ceil(
                    feat_len / (self.diarization_model.sortformer_modules.chunk_len * self.diarization_model.sortformer_modules.subsampling_factor)
                )
                assert num_chunks == 1, "Only one chunk should be used for streaming mode"
                streaming_level = sortformer_loader_level
                if streaming_level in ['emb', 'feat']:
                    streaming_loader = self.diarization_model.sortformer_modules.streaming_feat_loader(
                        feat_seq = processed_signal,
                        feat_seq_length = processed_signal_len,
                        feat_seq_offset = 0
                    )
                    for _, chunk_feat_seq_t, feat_lengths, left_offset, right_offset in streaming_loader:
                        # import ipdb; ipdb.set_trace()
                        self.streaming_state, self.total_preds = self.diarization_model.forward_streaming_step(
                        processed_signal=chunk_feat_seq_t,
                        processed_signal_length=feat_lengths,
                        streaming_state=self.streaming_state,
                        total_preds=self.total_preds,
                        left_offset=left_offset,
                        right_offset=right_offset,
                        streaming_level = streaming_level,
                        left_context = left_context,
                        tokens_per_chunk = tokens_per_chunk
                        )
                elif streaming_level == 'audio':
                    # print('Inference on audio level')
                    self.streaming_state, self.total_preds = self.diarization_model.forward_streaming_step(
                    processed_signal=diar_input_signal,
                    processed_signal_length=diar_input_signal_len,
                    streaming_state=self.streaming_state,
                    total_preds=self.total_preds,
                    left_offset=0,
                    right_offset=0,
                    streaming_level = streaming_level,
                    )
        else:
            raise ValueError(f"Invalid RTTM strategy {self.cfg.spk_supervision_strategy} is not supported.")
        if self.query_pred is None:
            self.query_pred = []
            for i in range(signal.size(0)):
                query_pred_len = get_hidden_length_from_sample_length(query_len[i], 160, 8)

                # set all the non-query speaker predictions in query session to 0
                self.diarization_model.spkcache_fifo_chunk_preds[i, :query_pred_len, 1:] = 0
                
                self.query_pred.append(self.diarization_model.spkcache_fifo_chunk_preds[i, :query_pred_len])
            re_aranged_diar_preds = self.diarization_model.spkcache_fifo_chunk_preds
        else:
            if initial_buffer:
                re_aranged_diar_preds = torch.cat([self.query_pred, self.diarization_model.spkcache_fifo_chunk_preds[:,-get_hidden_length_from_sample_length(chunk_len * (temp_buffer_index + 1), 160, 8):]], dim = 1)
            else:
                re_aranged_diar_preds = []
                for i in range(signal.size(0)):
                    # as the last frame is predicted to be all silence somehow, shift backward by 1 frame while rearange diarization prediction
                    re_aranged_diar_preds.append(torch.cat([self.query_pred[i], self.diarization_model.spkcache_fifo_chunk_preds[i, self.diarization_model.spkcache_fifo_chunk_preds_lengths[i] - get_hidden_length_from_sample_length(buffer_len)-1:self.diarization_model.spkcache_fifo_chunk_preds_lengths[i]-1]], dim = 0))
                re_aranged_diar_preds = collate_matrices(re_aranged_diar_preds)

        diar_preds = re_aranged_diar_preds
        if self.binarize_diar_preds_threshold:
            diar_preds = torch.where(diar_preds > self.binarize_diar_preds_threshold, torch.tensor(1), torch.tensor(0)).to(signal.device).float()
        self.diar_preds = diar_preds

        # if (isinstance(batch, DALIOutputs) and batch.has_processed_signal) or signal.shape[1] == 80:
        #     encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        # else:
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
    
        encoded = torch.transpose(encoded, 1, 2) # B * D * T -> B * T * D
        if self.diar == True:        
            # Speaker mapping shuffling to equalize the speaker label's distributions
            if self.cfg.shuffle_spk_mapping:
                diar_preds = apply_spk_mapping(diar_preds, spk_mappings)

            if(diar_preds.shape[1]!=encoded.shape[1]):
            # KD duct-tape solution for extending the speaker predictions 
                asr_frame_count = encoded.shape[1]
                diar_preds = self.fix_diar_output(diar_preds, asr_frame_count)

            # Normalize the features
            if self.norm == 'ln':
                diar_preds = self.diar_norm(diar_preds)
                encoded = self.asr_norm(encoded)
            elif self.norm == 'l2':
                diar_preds = torch.nn.functional.normalize(diar_preds, p=2, dim=-1)
                encoded = torch.nn.functional.normalize(encoded, p=2, dim=-1)
            
            if diar_preds.shape[1] > encoded.shape[1]:
                diar_preds = diar_preds[:, :encoded.shape[1], :]

            if self.diar_kernel_type == 'sinusoidal':
                speaker_infusion_asr = torch.matmul(diar_preds, self.diar_kernel.to(diar_preds.device))
                if self.kernel_norm == 'l2':
                    speaker_infusion_asr = torch.nn.functional.normalize(speaker_infusion_asr, p=2, dim=-1)
                encoded = speaker_infusion_asr + encoded
            elif self.diar_kernel_type == 'metacat':
                concat_enc_states = encoded.unsqueeze(2) * diar_preds.unsqueeze(3)
                concat_enc_states = concat_enc_states.flatten(2,3)
                encoded = self.joint_proj(concat_enc_states)
            elif self.diar_kernel_type == 'metacat_residule':
                #only pick speaker 0
                concat_enc_states = encoded.unsqueeze(2) * diar_preds[:,:,:1].unsqueeze(3)
                concat_enc_states = concat_enc_states.flatten(2,3)
                encoded = encoded + self.joint_proj(concat_enc_states)    
            elif self.diar_kernel_type == 'metacat_residule_early':
                #only pick speaker 0
                concat_enc_states = encoded.unsqueeze(2) * diar_preds[:,:,:1].unsqueeze(3)
                concat_enc_states = concat_enc_states.flatten(2,3)
                encoded = self.joint_proj(encoded + concat_enc_states)
            elif self.diar_kernel_type == 'metacat_residule_projection':
                #only pick speaker 0 and add diar_preds
                concat_enc_states = encoded.unsqueeze(2) * diar_preds[:,:,:1].unsqueeze(3)
                concat_enc_states = concat_enc_states.flatten(2,3)
                encoded = encoded + concat_enc_states
                concat_enc_states = torch.cat([encoded, diar_preds], dim = -1)
                encoded = self.joint_proj(concat_enc_states)
            else: #projection
                concat_enc_states = torch.cat([encoded, diar_preds], dim=-1)
                encoded = self.joint_proj(concat_enc_states)
        else:
            encoded = encoded
        
        encoded = torch.transpose(encoded, 1, 2) # B * T * D -> B * D * T
        return encoded, encoded_len