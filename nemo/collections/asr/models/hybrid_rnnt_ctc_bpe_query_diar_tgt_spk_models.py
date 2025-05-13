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
from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_tgt_spk_models import EncDecHybridRNNTCTCTgtSpkBPEModel
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



class EncDecHybridRNNTCTCQueryDiarTgtSpkBPEModel(EncDecHybridRNNTCTCTgtSpkBPEModel):
    """Base class for encoder decoder RNNT-based models with auxiliary CTC decoder/loss and subword tokenization."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

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

        if (isinstance(batch, DALIOutputs) and batch.has_processed_signal) or signal.shape[1] == 80:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        encoded = torch.transpose(encoded, 1, 2) # B * D * T -> B * T * D

        #remove query part from diar_preds and re-collate
        diar_preds_no_query = []
        for i in range(signal.shape[0]):
            diar_preds_no_query.append(diar_preds[i,query_frame_len[i]:,:])
        self.spk_targets = collate_matrices(diar_preds_no_query)
        # import pickle; import numpy as np;
        # with open('audio.pickle', 'wb') as f:
        #     pickle.dump(signal, f)
        # with open('audio_w_query.pickle', 'wb') as f:
        #     pickle.dump(signal_w_query, f)
        # with open('spk_targets.pickle', 'wb') as f:
        #     pickle.dump(self.spk_targets, f)
        # with open('spk_targets_w_query.pickle', 'wb') as f:
        #     pickle.dump(diar_preds, f)  
        # import ipdb; ipdb.set_trace()
        
        diar_preds = self.spk_targets


        # truncate diar_preds to be the same length as encoded
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
    



    
