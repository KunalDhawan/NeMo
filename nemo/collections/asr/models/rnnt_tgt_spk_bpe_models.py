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

import copy
import os
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer

import nemo.collections.asr.models as asr_models
from nemo.core.classes.mixins import AccessMixin
from nemo.collections.asr.parts.mixins.asr_adapter_mixins import ASRAdapterModelMixin
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.utils import asr_module_utils
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.data.audio_to_text_lhotse_target_speaker import LhotseSpeechToTextTgtSpkBpeDataset
from nemo.core.classes.mixins import AccessMixin
from nemo.collections.asr.data.audio_to_text_dali import AudioToCharDALIDataset, DALIOutputs

from nemo.collections.asr.parts.mixins import (
    ASRModuleMixin,
    ASRTranscriptionMixin,
    TranscribeConfig,
    TranscriptionReturnType,
)

from nemo.collections.asr.models.rnnt_spk_bpe_models import EncDecRNNTSpkBPEModel
from nemo.collections.asr.models.eesd_models import SortformerEncLabelModel
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes.common import PretrainedModelInfo

from nemo.utils import logging

class EncDecRNNTTgtSpkBPEModel(EncDecRNNTSpkBPEModel):
    """Base class for encoder decoder RNNT-based models with subword tokenization."""

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        return results

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        super().__init__(cfg=cfg, trainer=trainer)


    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseSpeechToTextTgtSpkBpeDataset(cfg = config, tokenizer=self.tokenizer,),
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
            'shuffle_spk_mapping': self.cfg.test_ds.get('shuffle_spk_mapping',False),
            'add_separater_audio': self.cfg.test_ds.get('add_separater_audio', False),
            'add_special_token': self.cfg.test_ds.get('add_special_token', False),
            'fix_query_audio_end_time' : self.cfg.test_ds.get('fix_query_audio_end_time', False),
            'inference_mode': self.cfg.test_ds.get('inference_mode',True)
        }

        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer
    

    def conformer_stream_step(
        self,
        audio_signal: torch.Tensor,
        processed_signal: torch.Tensor,
        spk_targets: torch.Tensor,
        global_mapping: dict,
        text_idx: int,
        token_idx: int,
        raw_texts: str,
        prev_spk: str,
        audio_signal_lengths: torch.Tensor = None,
        processed_signal_length: torch.Tensor = None,
        cache_last_channel: torch.Tensor = None,
        cache_last_time: torch.Tensor = None,
        cache_last_channel_len: torch.Tensor = None,
        keep_all_outputs: bool = True,
        previous_hypotheses: List[Hypothesis] = None,
        previous_pred_out: torch.Tensor = None,
        drop_extra_pre_encoded: int = None,
        return_transcription: bool = True,
        return_log_probs: bool = False,
    ):
        """
        For detailed usage, please refer to ASRModuleMixin.conformer_stream_step
        """
        if not isinstance(self, asr_models.EncDecRNNTModel) and not isinstance(self, asr_models.EncDecCTCModel):
            raise NotImplementedError(f"stream_step does not support {type(self)}!")

        if not isinstance(self.encoder, StreamingEncoder):
            raise NotImplementedError(f"Encoder of this model does not support streaming!")

        if isinstance(self, asr_models.EncDecRNNTModel) and return_transcription is False:
            logging.info(
                "return_transcription can not be False for Transducer models as decoder returns the transcriptions too."
            )

        if not isinstance(self, asr_models.EncDecCTCModel) and return_log_probs is True:
            logging.info("return_log_probs can only be True for CTC models.")
        (
            encoded,
            encoded_len,
            cache_last_channel_next,
            cache_last_time_next,
            cache_last_channel_next_len,
        ) = self.encoder.cache_aware_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=keep_all_outputs,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
        )
        encoded = torch.transpose(encoded, 1, 2) # B * D * T -> B * T * D

        if self.diar == True:
            if self.cfg.spk_supervision_strategy == 'rttm':
                if spk_targets is not None:
                    diar_preds = spk_targets 
                else:
                    raise ValueError("`spk_targets` is required for speaker supervision strategy 'rttm'")
            elif self.cfg.spk_supervision_strategy == 'diar':
                with torch.set_grad_enabled(not self.cfg.freeze_diar):
                    diar_preds, _preds, attn_score_stack, total_memory_list, encoder_states_list = self.forward_diar(audio_signal, audio_signal_lengths)
                    if self.binarize_diar_preds_threshold:
                        diar_preds = torch.where(diar_preds > self.binarize_diar_preds_threshold, torch.tensor(1), torch.tensor(0)).to(encoded.device)
                    if diar_preds is None:
                        raise ValueError("`diar_pred`is required for speaker supervision strategy 'diar'")
            elif self.cfg.spk_supervision_strategy == 'mix':
                with torch.set_grad_enabled(not self.cfg.freeze_diar):
                    diar_preds, _preds, attn_score_stack, total_memory_list, encoder_states_list = self.forward_diar(audio_signal, audio_signal_lengths)
                    if self.binarize_diar_preds_threshold:
                        diar_preds = torch.where(diar_preds > self.binarize_diar_preds_threshold, torch.tensor(1), torch.tensor(0)).to(encoded.device)
                    diar_preds = self.fix_diar_output(diar_preds, spk_targets.shape[1])
                diar_preds = self._get_probablistic_mix(diar_preds=diar_preds, spk_targets=spk_targets, rttm_mix_prob=float(self.cfg.rttm_mix_prob))
            else:
                raise ValueError(f"Invalid RTTM strategy {self.cfg.spk_supervision_strategy} is not supported.")
        
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
                # import pdb; pdb.set_trace()
                concat_enc_states = encoded.unsqueeze(2) * diar_preds.unsqueeze(3)
                concat_enc_states = concat_enc_states.flatten(2,3)
                encoded = self.joint_proj(concat_enc_states)
            else:
                concat_enc_states = torch.cat([encoded, diar_preds], dim=-1)
                encoded = self.joint_proj(concat_enc_states)
        else:
            encoded = encoded
        
        encoded = torch.transpose(encoded, 1, 2) # B * T * 
        if isinstance(self, asr_models.EncDecCTCModel) or (
            isinstance(self, asr_models.EncDecHybridRNNTCTCModel) and self.cur_decoder == "ctc"
        ):
            if hasattr(self, "ctc_decoder"):
                decoding = self.ctc_decoding
                decoder = self.ctc_decoder
            else:
                decoding = self.decoding
                decoder = self.decoder
            log_probs = decoder(encoder_output=encoded)
            predictions_tensor = log_probs.argmax(dim=-1, keepdim=False)

            # Concatenate the previous predictions with the current one to have the full predictions.
            # We drop the extra predictions for each sample by using the lengths returned by the encoder (encoded_len)
            # Then create a list of the predictions for the batch. The predictions can have different lengths because of the paddings.
            greedy_predictions = []
            if return_transcription:
                all_hyp_or_transcribed_texts = []
            else:
                all_hyp_or_transcribed_texts = None
            for preds_idx, preds in enumerate(predictions_tensor):
                if encoded_len is None:
                    preds_cur = predictions_tensor[preds_idx]
                else:
                    preds_cur = predictions_tensor[preds_idx, : encoded_len[preds_idx]]
                if previous_pred_out is not None:
                    greedy_predictions_concat = torch.cat((previous_pred_out[preds_idx], preds_cur), dim=-1)
                    encoded_len[preds_idx] += len(previous_pred_out[preds_idx])
                else:
                    greedy_predictions_concat = preds_cur
                greedy_predictions.append(greedy_predictions_concat)

                # TODO: make decoding more efficient by avoiding the decoding process from the beginning
                if return_transcription:
                    decoded_out = decoding.ctc_decoder_predictions_tensor(
                        decoder_outputs=greedy_predictions_concat.unsqueeze(0),
                        decoder_lengths=encoded_len[preds_idx : preds_idx + 1],
                        return_hypotheses=False,
                    )
                    all_hyp_or_transcribed_texts.append(decoded_out[0][0])
            best_hyp = None
        else:
            best_hyp, all_hyp_or_transcribed_texts = self.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded,
                encoded_lengths=encoded_len,
                return_hypotheses=True,
                partial_hypotheses=previous_hypotheses,
                # partial_hypotheses=None,
            )
            greedy_predictions = [hyp.y_sequence for hyp in best_hyp]
            if all_hyp_or_transcribed_texts is None:
                all_hyp_or_transcribed_texts = best_hyp

        result = [
            greedy_predictions,
            all_hyp_or_transcribed_texts,
            raw_texts,
            cache_last_channel_next,
            cache_last_time_next,
            cache_last_channel_next_len,
            best_hyp,
            len(all_hyp_or_transcribed_texts[0].text),
            len(all_hyp_or_transcribed_texts[0].y_sequence),
            prev_spk,
        ]
        if return_log_probs:
            result.append(log_probs)
            result.append(encoded_len)

        return tuple(result)