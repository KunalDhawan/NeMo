# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# pylint: disable=E1101
import itertools
import math
import os
import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm

from nemo.collections.asr.data.audio_to_diar_label import AudioToSpeechE2ESpkDiarDataset
from nemo.collections.asr.data.audio_to_diar_label_lhotse import LhotseAudioToSpeechE2ESpkDiarDataset
from nemo.collections.asr.metrics.multi_binary_acc import MultiBinaryAccuracy
from nemo.collections.asr.models.asr_model import ExportableEncDecModel
from nemo.collections.asr.parts.mixins.diarization import DiarizeConfig, SpkDiarizationMixin
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures, WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.asr.modules.sortformer_modules import StreamingChunkTargetInfo
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import (
    get_ats_targets,
    get_ats_targets_streaming,
    get_pil_targets,
)
from nemo.collections.asr.parts.utils.speaker_utils import generate_diarization_output_lines
from nemo.collections.asr.parts.utils.vad_utils import ts_vad_post_processing
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType
from nemo.core.neural_types.elements import ProbsType
from nemo.utils import logging

__all__ = ['SortformerEncLabelModel']


class SortformerEncLabelModel(ModelPT, ExportableEncDecModel, SpkDiarizationMixin):
    """
    Encoder class for Sortformer diarization model.
    Model class creates training, validation methods for setting up data performing model forward pass.

    This model class expects config dict for:
        * preprocessor
        * Transformer Encoder
        * FastConformer Encoder
        * Sortformer Modules
    """

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly
        from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        return result

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """
        Initialize an Sortformer Diarizer model and a pretrained NEST encoder.
        In this init function, training and validation datasets are prepared.
        """
        random.seed(42)
        self._trainer = trainer if trainer else None
        self._cfg = cfg

        if self._trainer:
            self.world_size = trainer.num_nodes * trainer.num_devices
        else:
            self.world_size = 1

        if self._trainer is not None and self._cfg.get('augmentor', None) is not None:
            self.augmentor = process_augmentations(self._cfg.augmentor)
        else:
            self.augmentor = None

        # Output upsampling: when output_subsampling_factor < encoder.subsampling_factor,
        # sigmoid predictions are upsampled to produce finer-resolution output (e.g. 10ms).
        # Must be set before super().__init__() because it triggers setup_training_data()
        # which needs self.output_subsampling_factor.
        encoder_subsample = self._cfg.encoder.get("subsampling_factor", 8)
        self.output_subsampling_factor = self._cfg.get("output_subsampling_factor", encoder_subsample)
        valid_factors = [d for d in range(1, encoder_subsample + 1) if encoder_subsample % d == 0]
        if self.output_subsampling_factor not in valid_factors:
            raise ValueError(
                f"output_subsampling_factor ({self.output_subsampling_factor}) is invalid. "
                f"Must be a positive divisor of encoder.subsampling_factor ({encoder_subsample}). "
                f"Valid values: {valid_factors}"
            )
        self.upsample_factor = encoder_subsample // self.output_subsampling_factor
        self.upsample_smooth_kernel = self._cfg.get("upsample_smooth_kernel", self.upsample_factor + 1)

        super().__init__(cfg=self._cfg, trainer=trainer)
        self.preprocessor = SortformerEncLabelModel.from_config_dict(self._cfg.preprocessor)

        if (
            hasattr(self._cfg, 'spec_augment')
            and self._cfg.spec_augment is not None
            and self._cfg.spec_augment.get('freq_masks', 0) + self._cfg.spec_augment.get('time_masks', 0) > 0
        ):
            self.spec_augmentation = SortformerEncLabelModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

        self.encoder = SortformerEncLabelModel.from_config_dict(self._cfg.encoder).to(self.device)
        self.sortformer_modules = SortformerEncLabelModel.from_config_dict(self._cfg.sortformer_modules).to(
            self.device
        )
        self.transformer_encoder = SortformerEncLabelModel.from_config_dict(self._cfg.transformer_encoder).to(
            self.device
        )
        if self._cfg.encoder.d_model != self._cfg.model_defaults.tf_d_model:
            self.sortformer_modules.encoder_proj = self.sortformer_modules.encoder_proj.to(self.device)
        else:
            self.sortformer_modules.encoder_proj = None
        self._init_loss_weights()

        self.eps = 1e-3
        self.negative_init_val = -99
        self.loss = instantiate(self._cfg.loss)

        self.async_streaming = self._cfg.get("async_streaming", False)
        self.streaming_mode = self._cfg.get("streaming_mode", False)
        if self.streaming_mode:
            # Validate streaming parameters once at initialization for streaming models
            self.sortformer_modules._check_streaming_parameters()
        self.save_hyperparameters("cfg")
        self._init_eval_metrics()
        speaker_inds = list(range(self._cfg.max_num_of_spks))
        self.speaker_permutations = torch.tensor(list(itertools.permutations(speaker_inds)))  # Get all permutations

        self.max_batch_dur = self._cfg.get("max_batch_dur", 20000)
        self.concat_and_pad_script = torch.jit.script(self.sortformer_modules.concat_and_pad)

    def _init_loss_weights(self):
        pil_weight = self._cfg.get("pil_weight", 0.0)
        ats_weight = self._cfg.get("ats_weight", 1.0)
        local_pil_weight = self._cfg.get("local_pil_weight", 0.0)
        local_ats_weight = self._cfg.get("local_ats_weight", 0.0)
        total = pil_weight + ats_weight + local_pil_weight + local_ats_weight
        if total == 0:
            raise ValueError(
                f"Loss weights cannot all be zero: pil={pil_weight}, ats={ats_weight}, "
                f"local_pil={local_pil_weight}, local_ats={local_ats_weight}"
            )
        self.pil_weight = pil_weight / total
        self.ats_weight = ats_weight / total
        self.local_pil_weight = local_pil_weight / total
        self.local_ats_weight = local_ats_weight / total

    def _init_eval_metrics(self):
        """
        If there is no label, then the evaluation metrics will be based on Permutation Invariant Loss (PIL).
        """
        self._accuracy_test = MultiBinaryAccuracy()
        self._accuracy_train = MultiBinaryAccuracy()
        self._accuracy_valid = MultiBinaryAccuracy()

        self._accuracy_test_ats = MultiBinaryAccuracy()
        self._accuracy_train_ats = MultiBinaryAccuracy()
        self._accuracy_valid_ats = MultiBinaryAccuracy()

    def _reset_train_metrics(self):
        self._accuracy_train.reset()
        self._accuracy_train_ats.reset()

    def _reset_valid_metrics(self):
        self._accuracy_valid.reset()
        self._accuracy_valid_ats.reset()

    def __setup_dataloader_from_config(self, config):
        # Switch to lhotse dataloader if specified in the config
        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseAudioToSpeechE2ESpkDiarDataset(cfg=config),
            )

        featurizer = WaveformFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=self.augmentor
        )
        fb_featurizer = FilterbankFeatures(
            sample_rate=self._cfg.preprocessor.sample_rate,
            normalize=self._cfg.preprocessor.normalize,
            n_window_size=int(self._cfg.preprocessor.window_size * config['sample_rate']),
            n_window_stride=int(self._cfg.preprocessor.window_stride * config['sample_rate']),
            window=self._cfg.preprocessor.window,
            nfilt=self._cfg.preprocessor.features,
            n_fft=self._cfg.preprocessor.n_fft,
            frame_splicing=self._cfg.preprocessor.frame_splicing,
            dither=self._cfg.preprocessor.dither,
        )

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None

        logging.info(f"Loading dataset from {config.manifest_filepath}")

        if self._trainer is not None:
            global_rank = self._trainer.global_rank
        else:
            global_rank = 0

        dataset = AudioToSpeechE2ESpkDiarDataset(
            manifest_filepath=config.manifest_filepath,
            soft_label_thres=config.soft_label_thres,
            session_len_sec=config.session_len_sec,
            num_spks=config.num_spks,
            featurizer=featurizer,
            fb_featurizer=fb_featurizer,
            window_stride=self._cfg.preprocessor.window_stride,
            subsampling_factor=self.output_subsampling_factor,
            global_rank=global_rank,
            soft_targets=config.soft_targets if 'soft_targets' in config else False,
            device=self.device,
            subsegment_mode=config.get('subsegment_mode', False),
            subsegment_min_len_sec=config.get('subsegment_min_len_sec', 15.0),
            subsegment_two_chunks_rate=config.get('subsegment_two_chunks_rate', 0.0),
            subsegment_min_chunk_len_sec=config.get('subsegment_min_chunk_len_sec', 10.0),
            subsegment_margin_frames=config.get('subsegment_margin_frames', 0),
        )

        self.data_collection = dataset.collection
        self.collate_ds = dataset

        dataloader_instance = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            collate_fn=self.collate_ds.eesd_train_collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=False,
            num_workers=config.get('num_workers', 1),
            pin_memory=config.get('pin_memory', False),
        )
        return dataloader_instance

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        self._train_dl = self.__setup_dataloader_from_config(
            config=train_data_config,
        )

    def setup_validation_data(self, val_data_layer_config: Optional[Union[DictConfig, Dict]]):
        self._validation_dl = self.__setup_dataloader_from_config(
            config=val_data_layer_config,
        )

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        self._test_dl = self.__setup_dataloader_from_config(
            config=test_data_config,
        )

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl
        return None

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            audio_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            audio_eltype = AudioSignal()
        return {
            "audio_signal": NeuralType(('B', 'T'), audio_eltype),
            "audio_signal_length": NeuralType(('B',), LengthsType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        return OrderedDict(
            {
                "preds": NeuralType(('B', 'T', 'C'), ProbsType()),
            }
        )

    def frontend_encoder(self, processed_signal, processed_signal_length, bypass_pre_encode: bool = False):
        """
        Generate encoder outputs from frontend encoder.

        Args:
            processed_signal (torch.Tensor):
                tensor containing audio-feature (mel spectrogram, mfcc, etc.).
            processed_signal_length (torch.Tensor):
                tensor containing lengths of audio signal in integers.

        Returns:
            emb_seq (torch.Tensor):
                tensor containing encoder outputs.
            emb_seq_length (torch.Tensor):
                tensor containing lengths of encoder outputs.
        """
        # Spec augment is not applied during evaluation/testing.
        # In streaming mode (bypass_pre_encode=True), spec augment is applied per-chunk
        # in forward_streaming_step before pre_encode, so skip it here.
        if self.spec_augmentation is not None and self.training and not bypass_pre_encode:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)
        emb_seq, emb_seq_length = self.encoder(
            audio_signal=processed_signal,
            length=processed_signal_length,
            bypass_pre_encode=bypass_pre_encode,
        )
        emb_seq = emb_seq.transpose(1, 2)
        if self.sortformer_modules.encoder_proj is not None:
            emb_seq = self.sortformer_modules.encoder_proj(emb_seq)
        return emb_seq, emb_seq_length

    def forward_infer(self, emb_seq, emb_seq_length):
        """
        The main forward pass for diarization for offline diarization inference.

        Args:
            emb_seq (torch.Tensor): Tensor containing FastConformer encoder states (embedding vectors).
                Shape: (batch_size, diar_frame_count, emb_dim)
            emb_seq_length (torch.Tensor): Tensor containing lengths of FastConformer encoder states.
                Shape: (batch_size,)

        Returns:
            preds (torch.Tensor): Sorted tensor containing Sigmoid values for predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
        """
        encoder_mask = self.sortformer_modules.length_to_mask(emb_seq_length, emb_seq.shape[1])
        trans_emb_seq = self.transformer_encoder(encoder_states=emb_seq, encoder_mask=encoder_mask)
        preds = self.sortformer_modules.forward_speaker_sigmoids(trans_emb_seq)
        return preds

    def _diarize_forward(self, batch: Any):
        """
        A counterpart of `_transcribe_forward` function in ASR.
        This function is a wrapper for forward pass functions for compataibility
        with the existing classes.

        Args:
            batch (Any): The input batch containing audio signal and audio signal length.

        Returns:
            preds (torch.Tensor): Sorted tensor containing Sigmoid values for predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
        """
        with torch.no_grad():
            preds = self.forward(audio_signal=batch[0], audio_signal_length=batch[1])
            preds = preds.to('cpu')
            torch.cuda.empty_cache()
        return preds

    def _diarize_output_processing(
        self, outputs, uniq_ids, diarcfg: DiarizeConfig
    ) -> Union[List[List[str]], Tuple[List[List[str]], List[torch.Tensor]]]:
        """
        Processes the diarization outputs and generates RTTM (Real-time Text Markup) files.
        TODO: Currently, this function is not included in mixin test because of
              `ts_vad_post_processing` function.
              (1) Implement a test-compatible function
              (2) `vad_utils.py` has `predlist_to_timestamps` function that is close to this function.
                  Needs to consolute differences and implement the test-compatible function.

        Args:
            outputs (torch.Tensor): Sorted tensor containing Sigmoid values for predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            uniq_ids (List[str]): List of unique identifiers for each audio file.
            diarcfg (DiarizeConfig): Configuration object for diarization.

        Returns:
            diar_output_lines_list (List[List[str]]): A list of lists, where each inner list contains
                                                      the RTTM lines for a single audio file.
            preds_list (List[torch.Tensor]): A list of tensors containing the diarization outputs
                                             for each audio file.
        """
        preds_list, diar_output_lines_list = [], []
        if outputs.shape[0] == 1:  # batch size = 1
            preds_list.append(outputs)
        else:
            preds_list.extend(torch.split(outputs, [1] * outputs.shape[0]))

        for sample_idx, uniq_id in enumerate(uniq_ids):
            offset = self._diarize_audio_rttm_map[uniq_id]['offset']
            speaker_assign_mat = preds_list[sample_idx].squeeze(dim=0)
            speaker_timestamps = [[] for _ in range(speaker_assign_mat.shape[-1])]
            for spk_id in range(speaker_assign_mat.shape[-1]):
                ts_mat = ts_vad_post_processing(
                    speaker_assign_mat[:, spk_id],
                    cfg_vad_params=diarcfg.postprocessing_params,
                    unit_10ms_frame_count=int(self.output_subsampling_factor),
                    bypass_postprocessing=False,
                )
                ts_mat = ts_mat + offset
                ts_seg_raw_list = ts_mat.tolist()
                ts_seg_list = [[round(stt, 2), round(end, 2)] for (stt, end) in ts_seg_raw_list]
                speaker_timestamps[spk_id].extend(ts_seg_list)

            diar_output_lines = generate_diarization_output_lines(
                speaker_timestamps=speaker_timestamps, model_spk_num=len(speaker_timestamps)
            )
            diar_output_lines_list.append(diar_output_lines)
        if diarcfg.include_tensor_outputs:
            return (diar_output_lines_list, preds_list)
        else:
            return diar_output_lines_list

    def _setup_diarize_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            - manifest_filepath: Path to the manifest file containing audio file paths
              and corresponding speaker labels.

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
            'num_spks': config.get('num_spks', self._cfg.max_num_of_spks),
            'batch_size': batch_size,
            'shuffle': False,
            'soft_label_thres': 0.5,
            'session_len_sec': config['session_len_sec'],
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'subsegment_mode': config.get('subsegment_mode', False),
            'subsegment_min_len_sec': config.get('subsegment_min_len_sec', 15.0),
            'subsegment_two_chunks_rate': config.get('subsegment_two_chunks_rate', 0.0),
            'subsegment_min_chunk_len_sec': config.get('subsegment_min_chunk_len_sec', 10.0),
            'subsegment_margin_frames': config.get('subsegment_margin_frames', 0),
        }
        temporary_datalayer = self.__setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    def oom_safe_feature_extraction(self, input_signal, input_signal_length):
        """
        This function divides the input signal into smaller sub-batches and processes them sequentially
        to prevent out-of-memory errors during feature extraction.

        Args:
            input_signal (torch.Tensor): The input audio signal.
            input_signal_length (torch.Tensor): The lengths of the input audio signals.

        Returns:
            processed_signal (torch.Tensor): The aggregated audio signal.
                                             The length of this tensor should match the original batch size.
            processed_signal_length (torch.Tensor): The lengths of the processed audio signals.
        """
        input_signal = input_signal.cpu()
        processed_signal_list, processed_signal_length_list = [], []
        max_batch_sec = input_signal.shape[1] / self.preprocessor._cfg.sample_rate
        org_batch_size = input_signal.shape[0]
        div_batch_count = min(int(max_batch_sec * org_batch_size // self.max_batch_dur + 1), org_batch_size)
        div_size = math.ceil(org_batch_size / div_batch_count)

        for div_count in range(div_batch_count):
            start_idx = int(div_count * div_size)
            end_idx = int((div_count + 1) * div_size)
            if start_idx >= org_batch_size:
                break
            input_signal_div = input_signal[start_idx:end_idx, :].to(self.device)
            input_signal_length_div = input_signal_length[start_idx:end_idx]
            processed_signal_div, processed_signal_length_div = self.preprocessor(
                input_signal=input_signal_div, length=input_signal_length_div
            )
            processed_signal_div = processed_signal_div.detach().cpu()
            processed_signal_length_div = processed_signal_length_div.detach().cpu()
            processed_signal_list.append(processed_signal_div)
            processed_signal_length_list.append(processed_signal_length_div)

        processed_signal = torch.cat(processed_signal_list, 0)
        processed_signal_length = torch.cat(processed_signal_length_list, 0)
        assert processed_signal.shape[0] == org_batch_size, (
            f"The resulting batch size of processed signal - {processed_signal.shape[0]} "
            f"is not equal to original batch size: {org_batch_size}"
        )
        processed_signal = processed_signal.to(self.device)
        processed_signal_length = processed_signal_length.to(self.device)
        return processed_signal, processed_signal_length

    def process_signal(self, audio_signal, audio_signal_length):
        """
        Extract audio features from time-series signal for further processing in the model.

        This function performs the following steps:
        1. Moves the audio signal to the correct device.
        2. Normalizes the time-series audio signal.
        3. Extrac audio feature from from the time-series audio signal using the model's preprocessor.

        Args:
            audio_signal (torch.Tensor): The input audio signal.
                Shape: (batch_size, num_samples)
            audio_signal_length (torch.Tensor): The length of each audio signal in the batch.
                Shape: (batch_size,)

        Returns:
            processed_signal (torch.Tensor): The preprocessed audio signal.
                Shape: (batch_size, num_features, num_frames)
            processed_signal_length (torch.Tensor): The length of each processed signal.
                Shape: (batch_size,)
        """
        audio_signal, audio_signal_length = audio_signal.to(self.device), audio_signal_length.to(self.device)
        if not self.streaming_mode:
            audio_signal = (1 / (audio_signal.max() + self.eps)) * audio_signal

        batch_total_dur = audio_signal.shape[0] * audio_signal.shape[1] / self.preprocessor._cfg.sample_rate
        if self.max_batch_dur > 0 and self.max_batch_dur < batch_total_dur:
            processed_signal, processed_signal_length = self.oom_safe_feature_extraction(
                input_signal=audio_signal, input_signal_length=audio_signal_length
            )
        else:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=audio_signal, length=audio_signal_length
            )
        # This cache clearning can significantly slow down the training speed.
        # Only perform `empty_cache()` when the input file is extremely large for streaming mode.
        if not self.training and self.streaming_mode:
            del audio_signal, audio_signal_length
            torch.cuda.empty_cache()
        return processed_signal, processed_signal_length

    def forward(
        self,
        audio_signal,
        audio_signal_length,
    ):
        """
        Forward pass for training and inference.

        Args:
            audio_signal (torch.Tensor): Tensor containing audio waveform
                Shape: (batch_size, num_samples)
            audio_signal_length (torch.Tensor): Tensor containing lengths of audio waveforms
                Shape: (batch_size,)

        Returns:
            preds (torch.Tensor): Sorted tensor containing predicted speaker labels
                Shape: (batch_size, max. diar frame count, num_speakers)
        """
        processed_signal, processed_signal_length = self.process_signal(
            audio_signal=audio_signal, audio_signal_length=audio_signal_length
        )
        processed_signal = processed_signal[:, :, : processed_signal_length.max()]
        if self.streaming_mode:
            preds, chunk_info_list = self.forward_streaming(processed_signal, processed_signal_length)
            self._streaming_chunk_info = chunk_info_list
        else:
            emb_seq, emb_seq_length = self.frontend_encoder(
                processed_signal=processed_signal, processed_signal_length=processed_signal_length
            )
            preds = self.forward_infer(emb_seq, emb_seq_length)
            self._streaming_chunk_info = None
        if self.upsample_factor > 1:
            preds = self.sortformer_modules.upsample_preds(
                preds, upsample_factor=self.upsample_factor, smooth_kernel=self.upsample_smooth_kernel
            )
        return preds

    @property
    def input_names(self):
        return ["chunk", "chunk_lengths", "spkcache", "spkcache_lengths", "fifo", "fifo_lengths"]

    @property
    def output_names(self):
        return ["spkcache_fifo_chunk_preds", "chunk_pre_encode_embs", "chunk_pre_encode_lengths"]

    def streaming_input_examples(self):
        """Input tensor examples for exporting streaming version of model"""
        batch_size = 4
        chunk = torch.rand([batch_size, 120, 80]).to(self.device)
        chunk_lengths = torch.tensor([120] * batch_size).to(self.device)
        spkcache = torch.randn([batch_size, 188, 512]).to(self.device)
        spkcache_lengths = torch.tensor([40, 188, 0, 68]).to(self.device)
        fifo = torch.randn([batch_size, 188, 512]).to(self.device)
        fifo_lengths = torch.tensor([50, 88, 0, 90]).to(self.device)
        return chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths

    def streaming_export(self, output: str):
        """Exports the model for streaming inference."""
        input_example = self.streaming_input_examples()
        export_out = self.export(output, input_example=input_example)
        return export_out

    def forward_for_export(self, chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths):
        """
        This forward pass is for ONNX model export.

        Args:
            chunk (torch.Tensor): Tensor containing audio waveform.
                The term "chunk" refers to the "input buffer" in the speech processing pipeline.
                The size of chunk (input buffer) determines the latency introduced by buffering.
                Shape: (batch_size, feature frame count, dimension)
            chunk_lengths (torch.Tensor): Tensor containing lengths of audio waveforms
                Shape: (batch_size,)
            spkcache (torch.Tensor): Tensor containing speaker cache embeddings from start
                Shape: (batch_size, spkcache_len, emb_dim)
            spkcache_lengths (torch.Tensor): Tensor containing lengths of speaker cache
                Shape: (batch_size,)
            fifo (torch.Tensor): Tensor containing embeddings from latest chunks
                Shape: (batch_size, fifo_len, emb_dim)
            fifo_lengths (torch.Tensor): Tensor containing lengths of FIFO queue embeddings
                Shape: (batch_size,)

        Returns:
            spkcache_fifo_chunk_preds (torch.Tensor): Sorted tensor containing predicted speaker labels
                Shape: (batch_size, max. diar frame count, num_speakers)
            chunk_pre_encode_embs (torch.Tensor): Tensor containing pre-encoded embeddings from the chunk
                Shape: (batch_size, num_frames, emb_dim)
            chunk_pre_encode_lengths (torch.Tensor): Tensor containing lengths of pre-encoded embeddings
                from the chunk (=input buffer).
                Shape: (batch_size,)
        """
        # pre-encode the chunk
        chunk_pre_encode_embs, chunk_pre_encode_lengths = self.encoder.pre_encode(x=chunk, lengths=chunk_lengths)
        chunk_pre_encode_lengths = chunk_pre_encode_lengths.to(torch.int64)

        # concat the embeddings from speaker cache, FIFO queue and the chunk
        spkcache_fifo_chunk_pre_encode_embs, spkcache_fifo_chunk_pre_encode_lengths = self.concat_and_pad_script(
            [spkcache, fifo, chunk_pre_encode_embs], [spkcache_lengths, fifo_lengths, chunk_pre_encode_lengths]
        )

        # encode the concatenated embeddings
        spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths = self.frontend_encoder(
            processed_signal=spkcache_fifo_chunk_pre_encode_embs,
            processed_signal_length=spkcache_fifo_chunk_pre_encode_lengths,
            bypass_pre_encode=True,
        )

        # forward pass for inference
        spkcache_fifo_chunk_preds = self.forward_infer(
            spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths
        )
        return spkcache_fifo_chunk_preds, chunk_pre_encode_embs, chunk_pre_encode_lengths

    def forward_streaming(
        self,
        processed_signal,
        processed_signal_length,
    ):
        """
        The main forward pass for diarization inference in streaming mode.

        Args:
            processed_signal (torch.Tensor): Tensor containing audio waveform
                Shape: (batch_size, num_samples)
            processed_signal_length (torch.Tensor): Tensor containing lengths of audio waveforms
                Shape: (batch_size,)

        Returns:
            total_preds (torch.Tensor): Tensor containing predicted speaker labels for the current chunk
                and all previous chunks
                Shape: (batch_size, pred_len, num_speakers)
            chunk_info_list (list[StreamingChunkTargetInfo] or None): Per-chunk supplementary info
                for per-chunk target alignment during training. None when not training.
        """
        streaming_state = self.sortformer_modules.init_streaming_state(
            batch_size=processed_signal.shape[0], async_streaming=self.async_streaming, device=self.device
        )

        batch_size, ch, sig_length = processed_signal.shape
        processed_signal_offset = torch.zeros((batch_size,), dtype=torch.long, device=self.device)

        if dist.is_available() and dist.is_initialized():
            local_tensor = torch.tensor([sig_length], device=processed_signal.device)
            dist.all_reduce(
                local_tensor, op=dist.ReduceOp.MAX, async_op=False
            )  # get max feature length across all GPUs
            max_n_frames = local_tensor.item()
            if dist.get_rank() == 0:
                logging.info(f"Maximum feature length across all GPUs: {max_n_frames}")
        else:
            max_n_frames = sig_length

        if sig_length < max_n_frames:  # need padding to have the same feature length for all GPUs
            pad_tensor = torch.full(
                (batch_size, ch, max_n_frames - sig_length),
                self.negative_init_val,
                dtype=processed_signal.dtype,
                device=processed_signal.device,
            )
            processed_signal = torch.cat([processed_signal, pad_tensor], dim=2)

        att_mod = False
        if self.training:
            rand_num = random.random()
            if rand_num < self.sortformer_modules.causal_attn_rate:
                self.encoder.att_context_size = [-1, self.sortformer_modules.causal_attn_rc]
                self.transformer_encoder.diag = self.sortformer_modules.causal_attn_rc
                att_mod = True

        total_preds = torch.zeros((batch_size, 0, self.sortformer_modules.n_spk), device=self.device)

        feat_len = processed_signal.shape[2]
        num_chunks = math.ceil(
            feat_len / (self.sortformer_modules.chunk_len * self.sortformer_modules.subsampling_factor)
        )
        streaming_loader = self.sortformer_modules.streaming_feat_loader(
            feat_seq=processed_signal,
            feat_seq_length=processed_signal_length,
            feat_seq_offset=processed_signal_offset,
        )

        chunk_info_list = []
        frame_offset = 0

        for i, chunk_feat_seq_t, feat_lengths, left_offset, right_offset in tqdm(
            streaming_loader,
            total=num_chunks,
            desc="Streaming Steps",
            disable=self.training,
        ):
            streaming_state, total_preds, chunk_target_info = self.forward_streaming_step(
                processed_signal=chunk_feat_seq_t,
                processed_signal_length=feat_lengths,
                streaming_state=streaming_state,
                total_preds=total_preds,
                left_offset=left_offset,
                right_offset=right_offset,
                frame_offset=frame_offset,
                collect_chunk_info=True,
            )
            #if chunk_target_info is not None:
            #    logging.info(f"i={i}: chunk_target_info: {chunk_target_info.frame_indices.shape}, {chunk_target_info.preds.shape}, {chunk_target_info.chunk_start}, {chunk_target_info.chunk_end}")
            #    logging.info(f"i={i}: chunk_target_info.frame_indices: {chunk_target_info.frame_indices[0, :500]}")
            chunk_len = chunk_target_info.chunk_end - chunk_target_info.chunk_start
            chunk_info_list.append(chunk_target_info)
            frame_offset += chunk_len

        if att_mod:
            self.encoder.att_context_size = [-1, -1]
            self.transformer_encoder.diag = None

        del processed_signal, processed_signal_length

        if sig_length < max_n_frames:  # Discard preds corresponding to padding
            n_frames = math.ceil(sig_length / self.encoder.subsampling_factor)
            total_preds = total_preds[:, :n_frames, :]
        return total_preds, chunk_info_list

    def forward_streaming_step(
        self,
        processed_signal,
        processed_signal_length,
        streaming_state,
        total_preds,
        left_offset=0,
        right_offset=0,
        frame_offset=0,
        collect_chunk_info=False,
    ):
        """
        One-step forward pass for diarization inference in streaming mode.

        Args:
            processed_signal (torch.Tensor): Tensor containing audio waveform
                Shape: (batch_size, num_samples)
            processed_signal_length (torch.Tensor): Tensor containing lengths of audio waveforms
                Shape: (batch_size,)
            streaming_state (SortformerStreamingState):
                    Tensor variables that contain the streaming state of the model.
                    Find more details in the `SortformerStreamingState` class in `sortformer_modules.py`.

                Attributes:
                    spkcache (torch.Tensor): Speaker cache to store embeddings from start
                    spkcache_lengths (torch.Tensor): Lengths of the speaker cache
                    spkcache_preds (torch.Tensor): The speaker predictions for the speaker cache parts
                    fifo (torch.Tensor): FIFO queue to save the embedding from the latest chunks
                    fifo_lengths (torch.Tensor): Lengths of the FIFO queue
                    fifo_preds (torch.Tensor): The speaker predictions for the FIFO queue parts
                    spk_perm (torch.Tensor): Speaker permutation information for the speaker cache

            total_preds (torch.Tensor): Tensor containing total predicted speaker activity probabilities
                Shape: (batch_size, cumulative pred length, num_speakers)
            left_offset (int): left offset for the current chunk
            right_offset (int): right offset for the current chunk
            frame_offset (int): cumulative target frame offset for the current chunk core
            collect_chunk_info (bool): If True, build and return StreamingChunkTargetInfo

        Returns:
            streaming_state (SortformerStreamingState):
                    Tensor variables that contain the updated streaming state of the model from
                    this function call.
            total_preds (torch.Tensor):
                Tensor containing the updated total predicted speaker activity probabilities.
                Shape: (batch_size, cumulative pred length, num_speakers)
            chunk_target_info (StreamingChunkTargetInfo or None):
                Per-chunk supplementary info for target alignment. None if collect_chunk_info is False.
        """
        # Per-chunk spec augment: each chunk gets independently sampled masks,
        # simulating acoustic condition mismatch between cached and current embeddings.
        # Note: processed_signal arrives as (B, T, D) from streaming_feat_loader (transposed),
        # but SpecAugment expects (B, D, T), so we transpose before and after.
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(
                input_spec=processed_signal.transpose(1, 2), length=processed_signal_length
            ).transpose(1, 2)

        chunk_pre_encode_embs, chunk_pre_encode_lengths = self.encoder.pre_encode(
            x=processed_signal, lengths=processed_signal_length
        )

        if self.async_streaming:
            spkcache_fifo_chunk_pre_encode_embs, spkcache_fifo_chunk_pre_encode_lengths = (
                self.sortformer_modules.concat_and_pad(
                    [streaming_state.spkcache, streaming_state.fifo, chunk_pre_encode_embs],
                    [streaming_state.spkcache_lengths, streaming_state.fifo_lengths, chunk_pre_encode_lengths],
                )
            )
        else:
            spkcache_fifo_chunk_pre_encode_embs = self.sortformer_modules.concat_embs(
                [streaming_state.spkcache, streaming_state.fifo, chunk_pre_encode_embs], dim=1, device=self.device
            )
            spkcache_fifo_chunk_pre_encode_lengths = (
                streaming_state.spkcache.shape[1] + streaming_state.fifo.shape[1] + chunk_pre_encode_lengths
            )
        spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths = self.frontend_encoder(
            processed_signal=spkcache_fifo_chunk_pre_encode_embs,
            processed_signal_length=spkcache_fifo_chunk_pre_encode_lengths,
            bypass_pre_encode=True,
        )
        spkcache_fifo_chunk_preds = self.forward_infer(
            emb_seq=spkcache_fifo_chunk_fc_encoder_embs, emb_seq_length=spkcache_fifo_chunk_fc_encoder_lengths
        )

        spkcache_fifo_chunk_preds = self.sortformer_modules.apply_mask_to_preds(
            spkcache_fifo_chunk_preds, spkcache_fifo_chunk_fc_encoder_lengths
        )

        # Compute chunk geometry in pre-encode units
        lc = round(left_offset / self.encoder.subsampling_factor)
        rc = math.ceil(right_offset / self.encoder.subsampling_factor)
        spkcache_len = streaming_state.spkcache.shape[1]
        fifo_len = streaming_state.fifo.shape[1]
        chunk_len = chunk_pre_encode_embs.shape[1] - lc - rc
        batch_size = processed_signal.shape[0]

        # Build per-chunk target info BEFORE streaming_update.
        # Store canonical preds (after inv_spk_perm) and spkcache_len.
        # ATS uses spkcache-based content matching (invariant to cache block ordering).
        chunk_target_info = None
        chunk_frame_indices = None
        if collect_chunk_info:
            # Build context_frame_indices: [spkcache_indices, fifo_indices, lc_indices, chunk_indices, rc_indices]
            lc_start = max(frame_offset - lc, 0)
            lc_actual = frame_offset - lc_start
            lc_indices = torch.arange(lc_start, frame_offset, device=self.device).unsqueeze(0).expand(batch_size, -1)
            chunk_indices = torch.arange(frame_offset, frame_offset + chunk_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            rc_indices = torch.arange(frame_offset + chunk_len, frame_offset + chunk_len + rc, device=self.device).unsqueeze(0).expand(batch_size, -1)

            context_frame_indices = torch.cat([
                streaming_state.spkcache_frame_indices,
                streaming_state.fifo_frame_indices,
                lc_indices,
                chunk_indices,
                rc_indices,
            ], dim=1)

            chunk_start = spkcache_len + fifo_len + lc_actual
            chunk_end = chunk_start + chunk_len

            # Canonical preds: apply inv_spk_perm (same as streaming_update) so preds
            # match total_preds ordering.
            preds_for_info = spkcache_fifo_chunk_preds.detach()
            if streaming_state.spk_perm is not None:
                inv_spk_perm = torch.stack(
                    [torch.argsort(streaming_state.spk_perm[b]) for b in range(batch_size)]
                )
                preds_for_info = torch.stack(
                    [preds_for_info[b, :, inv_spk_perm[b]] for b in range(batch_size)]
                )

            chunk_target_info = StreamingChunkTargetInfo(
                preds=preds_for_info,
                frame_indices=context_frame_indices,
                spkcache_len=spkcache_len,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
            )

            # Core chunk frame indices for streaming_update frame tracking
            chunk_frame_indices = chunk_indices

        if self.async_streaming:
            streaming_state, chunk_preds = self.sortformer_modules.streaming_update_async(
                streaming_state=streaming_state,
                chunk=chunk_pre_encode_embs,
                chunk_lengths=chunk_pre_encode_lengths,
                preds=spkcache_fifo_chunk_preds,
                lc=lc,
                rc=rc,
            )
        else:
            streaming_state, chunk_preds = self.sortformer_modules.streaming_update(
                streaming_state=streaming_state,
                chunk=chunk_pre_encode_embs,
                preds=spkcache_fifo_chunk_preds,
                lc=lc,
                rc=rc,
                chunk_frame_indices=chunk_frame_indices,
            )
        total_preds = torch.cat([total_preds, chunk_preds], dim=1)

        return streaming_state, total_preds, chunk_target_info

    def _get_per_chunk_aligned_targets(self, targets, chunk_info_list):
        """
        Compute ATS and PIL targets using per-chunk permutation alignment.

        For each chunk, reconstructs the full-context targets from original target frames,
        computes the optimal ATS/PIL permutation for that chunk's context, and extracts
        only the chunk core portion. The chunk-level targets are then concatenated.

        ATS uses spkcache-based content matching (get_ats_targets_streaming):
        binarized preds and targets on the spkcache portion identify which pred column
        corresponds to which speaker, invariant to cache block ordering. When not all
        speakers are in the spkcache, full-context match score breaks ties.

        PIL uses standard permutation-invariant matching on canonical preds.

        Args:
            targets (torch.Tensor): Full-sequence ground truth speaker labels.
                Shape: (batch_size, total_frames, num_speakers)
            chunk_info_list (list[StreamingChunkTargetInfo]): Per-chunk supplementary info.

        Returns:
            total_targets_ats (torch.Tensor): Per-chunk aligned ATS targets in canonical order.
                Shape: (batch_size, total_pred_frames, num_speakers)
            total_targets_pil (torch.Tensor): Per-chunk aligned PIL targets in canonical order.
                Shape: (batch_size, total_pred_frames, num_speakers)
        """
        chunk_targets_ats_list = []
        chunk_targets_pil_list = []
        max_target_idx = targets.shape[1] - 1

        for chunk_info in chunk_info_list:
            frame_indices = chunk_info.frame_indices  # (B, context_len), long, -1 for invalid

            # Build targets_i by gathering from original targets; clamp invalid indices for safe gather
            invalid_mask = frame_indices < 0
            safe_indices = frame_indices.clone()
            safe_indices[invalid_mask] = 0
            safe_indices = safe_indices.clamp(0, max_target_idx)

            # Gather: (B, context_len, n_spk)
            indices_expanded = safe_indices.unsqueeze(-1).expand(-1, -1, targets.shape[2])
            targets_i = torch.gather(targets, 1, indices_expanded)
            # Zero out invalid positions (silence frames in spkcache)
            targets_i[invalid_mask.unsqueeze(-1).expand_as(targets_i)] = 0.0

            preds_i = chunk_info.preds  # canonical ordering (after inv_spk_perm)

            # ATS: spkcache-based content matching (invariant to cache block ordering)
            targets_ats_i = get_ats_targets_streaming(
                targets_i.clone(), preds_i,
                speaker_permutations=self.speaker_permutations,
                spkcache_len=chunk_info.spkcache_len,
            )

            # PIL: standard permutation-invariant matching on canonical preds
            targets_pil_i = get_pil_targets(
                targets_i.clone(), preds_i, speaker_permutations=self.speaker_permutations
            )

            # Extract chunk core portion
            cs, ce = chunk_info.chunk_start, chunk_info.chunk_end
            chunk_targets_ats_list.append(targets_ats_i[:, cs:ce, :])
            chunk_targets_pil_list.append(targets_pil_i[:, cs:ce, :])

        total_targets_ats = torch.cat(chunk_targets_ats_list, dim=1)
        total_targets_pil = torch.cat(chunk_targets_pil_list, dim=1)
        return total_targets_ats, total_targets_pil

    def _get_aux_train_evaluations(self, preds, targets, target_lens) -> dict:
        """
        Compute auxiliary training evaluations including losses and metrics.

        This function calculates various losses and metrics for the training process,
        including Arrival Time Sort (ATS) Loss and Permutation Invariant Loss (PIL)
        based evaluations.

        When streaming chunk info is available (self._streaming_chunk_info), uses per-chunk
        permutation alignment for correct gradient signals. Otherwise falls back to the
        global permutation approach.

        Args:
            preds (torch.Tensor): Predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            targets (torch.Tensor): Ground truth speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            target_lens (torch.Tensor): Lengths of target sequences.
                Shape: (batch_size,)

        Returns:
            (dict): A dictionary containing the following training metrics.
        """
        targets = targets.to(preds.dtype)
        if preds.shape[1] < targets.shape[1]:
            logging.info(
                f"WARNING! preds has less frames than targets ({preds.shape[1]} < {targets.shape[1]}). "
                "Truncating targets and clamping target_lens."
            )
            targets = targets[:, : preds.shape[1], :]
            target_lens = target_lens.clamp(max=preds.shape[1])
        elif preds.shape[1] > targets.shape[1]:
            preds = preds[:, : targets.shape[1], :]

        # Global targets (standard ATS/PIL on full concatenated predictions)
        targets_ats = get_ats_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        targets_pil = get_pil_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        ats_loss = self.loss(probs=preds, labels=targets_ats, target_lens=target_lens)
        pil_loss = self.loss(probs=preds, labels=targets_pil, target_lens=target_lens)

        # Local targets (per-chunk aligned, streaming only)
        chunk_info_list = getattr(self, '_streaming_chunk_info', None)
        if chunk_info_list:
            local_targets_ats, local_targets_pil = self._get_per_chunk_aligned_targets(targets.clone(), chunk_info_list)
            min_len = min(preds.shape[1], local_targets_ats.shape[1])
            local_targets_ats = local_targets_ats[:, :min_len, :]
            local_targets_pil = local_targets_pil[:, :min_len, :]
            preds_local = preds[:, :min_len, :]
            target_lens_local = target_lens.clamp(max=min_len)
            local_ats_loss = self.loss(probs=preds_local, labels=local_targets_ats, target_lens=target_lens_local)
            local_pil_loss = self.loss(probs=preds_local, labels=local_targets_pil, target_lens=target_lens_local)
        else:
            local_ats_loss = torch.tensor(0.0, device=preds.device)
            local_pil_loss = torch.tensor(0.0, device=preds.device)

        loss = (
            self.ats_weight * ats_loss
            + self.pil_weight * pil_loss
            + self.local_ats_weight * local_ats_loss
            + self.local_pil_weight * local_pil_loss
        )

        self._accuracy_train(preds, targets_pil, target_lens)
        train_f1_acc, train_precision, train_recall = self._accuracy_train.compute()

        self._accuracy_train_ats(preds, targets_ats, target_lens)
        train_f1_acc_ats, _, _ = self._accuracy_train_ats.compute()

        train_metrics = {
            'loss': loss,
            'ats_loss': ats_loss,
            'pil_loss': pil_loss,
            'local_ats_loss': local_ats_loss,
            'local_pil_loss': local_pil_loss,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'train_f1_acc': train_f1_acc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1_acc_ats': train_f1_acc_ats,
        }
        return train_metrics

    def training_step(self, batch: list, batch_idx: int) -> dict:
        """
        Performs a single training step.

        Args:
            batch (list): A list containing the following elements:
                - audio_signal (torch.Tensor): The input audio signal in time-series format.
                - audio_signal_length (torch.Tensor): The length of each audio signal in the batch.
                - targets (torch.Tensor): The target labels for the batch.
                - target_lens (torch.Tensor): The length of each target sequence in the batch.
            batch_idx (int): The index of the current batch.

        Returns:
            (dict): A dictionary containing the 'loss' key with the calculated loss value.
        """
        audio_signal, audio_signal_length, targets, target_lens = batch
        preds = self.forward(audio_signal=audio_signal, audio_signal_length=audio_signal_length)
        train_metrics = self._get_aux_train_evaluations(preds, targets, target_lens)
        self._reset_train_metrics()
        self.log_dict(train_metrics, sync_dist=True, on_step=True, on_epoch=False, logger=True)
        return {'loss': train_metrics['loss']}

    def _get_aux_validation_evaluations(self, preds, targets, target_lens) -> dict:
        """
        Compute auxiliary validation evaluations including losses and metrics.

        This function calculates various losses and metrics for the training process,
        including Arrival Time Sort (ATS) Loss and Permutation Invariant Loss (PIL)
        based evaluations.

        When streaming chunk info is available (self._streaming_chunk_info), uses per-chunk
        permutation alignment for correct evaluation. Otherwise falls back to the
        global permutation approach.

        Args:
            preds (torch.Tensor): Predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            targets (torch.Tensor): Ground truth speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            target_lens (torch.Tensor): Lengths of target sequences.
                Shape: (batch_size,)

        Returns:
            val_metrics (dict): A dictionary containing the following validation metrics
        """
        targets = targets.to(preds.dtype)
        if preds.shape[1] < targets.shape[1]:
            logging.info(
                f"WARNING! preds has less frames than targets ({preds.shape[1]} < {targets.shape[1]}). "
                "Truncating targets and clamping target_lens."
            )
            targets = targets[:, : preds.shape[1], :]
            target_lens = target_lens.clamp(max=preds.shape[1])
        elif preds.shape[1] > targets.shape[1]:
            preds = preds[:, : targets.shape[1], :]

        # Global targets
        targets_ats = get_ats_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        targets_pil = get_pil_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        val_ats_loss = self.loss(probs=preds, labels=targets_ats, target_lens=target_lens)
        val_pil_loss = self.loss(probs=preds, labels=targets_pil, target_lens=target_lens)

        # Local targets (per-chunk aligned, streaming only)
        chunk_info_list = getattr(self, '_streaming_chunk_info', None)
        if chunk_info_list:
            local_targets_ats, local_targets_pil = self._get_per_chunk_aligned_targets(targets.clone(), chunk_info_list)
            min_len = min(preds.shape[1], local_targets_ats.shape[1])
            local_targets_ats = local_targets_ats[:, :min_len, :]
            local_targets_pil = local_targets_pil[:, :min_len, :]
            preds_local = preds[:, :min_len, :]
            target_lens_local = target_lens.clamp(max=min_len)
            val_local_ats_loss = self.loss(probs=preds_local, labels=local_targets_ats, target_lens=target_lens_local)
            val_local_pil_loss = self.loss(probs=preds_local, labels=local_targets_pil, target_lens=target_lens_local)
        else:
            val_local_ats_loss = torch.tensor(0.0, device=preds.device)
            val_local_pil_loss = torch.tensor(0.0, device=preds.device)

        val_loss = (
            self.ats_weight * val_ats_loss
            + self.pil_weight * val_pil_loss
            + self.local_ats_weight * val_local_ats_loss
            + self.local_pil_weight * val_local_pil_loss
        )

        self._accuracy_valid(preds, targets_pil, target_lens)
        val_f1_acc, val_precision, val_recall = self._accuracy_valid.compute()

        self._accuracy_valid_ats(preds, targets_ats, target_lens)
        valid_f1_acc_ats, _, _ = self._accuracy_valid_ats.compute()

        self._accuracy_valid.reset()
        self._accuracy_valid_ats.reset()

        val_metrics = {
            'val_loss': val_loss,
            'val_ats_loss': val_ats_loss,
            'val_pil_loss': val_pil_loss,
            'val_local_ats_loss': val_local_ats_loss,
            'val_local_pil_loss': val_local_pil_loss,
            'val_f1_acc': val_f1_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1_acc_ats': valid_f1_acc_ats,
        }
        return val_metrics

    def validation_step(self, batch: list, batch_idx: int, dataloader_idx: int = 0):
        """
        Performs a single validation step.

        This method processes a batch of data during the validation phase. It forward passes
        the audio signal through the model, computes various validation metrics, and stores
        these metrics for later aggregation.

        Args:
            batch (list): A list containing the following elements:
                - audio_signal (torch.Tensor): The input audio signal.
                - audio_signal_length (torch.Tensor): The length of each audio signal in the batch.
                - targets (torch.Tensor): The target labels for the batch.
                - target_lens (torch.Tensor): The length of each target sequence in the batch.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int, optional): The index of the dataloader in case of multiple
                                            validation dataloaders. Defaults to 0.

        Returns:
            dict: A dictionary containing various validation metrics for this batch.
        """
        audio_signal, audio_signal_length, targets, target_lens = batch
        preds = self.forward(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
        )
        val_metrics = self._get_aux_validation_evaluations(preds, targets, target_lens)
        if isinstance(self.trainer.val_dataloaders, list) and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(val_metrics)
        else:
            self.validation_step_outputs.append(val_metrics)
        return val_metrics

    def test_step(self, batch: list, batch_idx: int, dataloader_idx: int = 0):
        """
        Performs a single validation step.

        This method processes a batch of data during the validation phase. It forward passes
        the audio signal through the model, computes various validation metrics, and stores
        these metrics for later aggregation.

        Args:
            batch (list): A list containing the following elements:
                - audio_signal (torch.Tensor): The input audio signal.
                - audio_signal_length (torch.Tensor): The length of each audio signal in the batch.
                - targets (torch.Tensor): The target labels for the batch.
                - target_lens (torch.Tensor): The length of each target sequence in the batch.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int, optional): The index of the dataloader in case of multiple
                                            validation dataloaders. Defaults to 0.

        Returns:
            dict: A dictionary containing various validation metrics for this batch.
        """
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def multi_validation_epoch_end(self, outputs: list, dataloader_idx: int = 0):
        if not outputs:
            logging.warning(f"`outputs` is None; empty outputs for dataloader={dataloader_idx}")
            return None
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_ats_loss_mean = torch.stack([x['val_ats_loss'] for x in outputs]).mean()
        val_pil_loss_mean = torch.stack([x['val_pil_loss'] for x in outputs]).mean()
        val_local_ats_loss_mean = torch.stack([x['val_local_ats_loss'] for x in outputs]).mean()
        val_local_pil_loss_mean = torch.stack([x['val_local_pil_loss'] for x in outputs]).mean()
        val_f1_acc_mean = torch.stack([x['val_f1_acc'] for x in outputs]).mean()
        val_precision_mean = torch.stack([x['val_precision'] for x in outputs]).mean()
        val_recall_mean = torch.stack([x['val_recall'] for x in outputs]).mean()
        val_f1_acc_ats_mean = torch.stack([x['val_f1_acc_ats'] for x in outputs]).mean()

        self._reset_valid_metrics()

        multi_val_metrics = {
            'val_loss': val_loss_mean,
            'val_ats_loss': val_ats_loss_mean,
            'val_pil_loss': val_pil_loss_mean,
            'val_local_ats_loss': val_local_ats_loss_mean,
            'val_local_pil_loss': val_local_pil_loss_mean,
            'val_f1_acc': val_f1_acc_mean,
            'val_precision': val_precision_mean,
            'val_recall': val_recall_mean,
            'val_f1_acc_ats': val_f1_acc_ats_mean,
        }
        return {'log': multi_val_metrics}

    def _get_aux_test_batch_evaluations(self, batch_idx: int, preds, targets, target_lens):
        """
        Compute auxiliary test batch evaluations using global (standard) accuracy metrics.

        Args:
            batch_idx (int): The index of the current batch.
            preds (torch.Tensor): Predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            targets (torch.Tensor): Ground truth speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            target_lens (torch.Tensor): Lengths of target sequences.
                Shape: (batch_size,)
        """
        targets = targets.to(preds.dtype)
        if preds.shape[1] < targets.shape[1]:
            logging.info(
                f"WARNING! preds has less frames than targets ({preds.shape[1]} < {targets.shape[1]}). "
                "Truncating targets and clamping target_lens."
            )
            targets = targets[:, : preds.shape[1], :]
            target_lens = target_lens.clamp(max=preds.shape[1])
        elif preds.shape[1] > targets.shape[1]:
            preds = preds[:, : targets.shape[1], :]

        # Global targets
        targets_ats = get_ats_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)
        targets_pil = get_pil_targets(targets.clone(), preds, speaker_permutations=self.speaker_permutations)

        self._accuracy_test(preds, targets_pil, target_lens)
        f1_acc, precision, recall = self._accuracy_test.compute()
        self.batch_f1_accs_list.append(f1_acc)
        self.batch_precision_list.append(precision)
        self.batch_recall_list.append(recall)
        logging.info(f"batch {batch_idx}: f1_acc={f1_acc}, precision={precision}, recall={recall}")

        self._accuracy_test_ats(preds, targets_ats, target_lens)
        f1_acc_ats, precision_ats, recall_ats = self._accuracy_test_ats.compute()
        self.batch_f1_accs_ats_list.append(f1_acc_ats)
        logging.info(
            f"batch {batch_idx}: f1_acc_ats={f1_acc_ats}, precision_ats={precision_ats}, recall_ats={recall_ats}"
        )

        self._accuracy_test.reset()
        self._accuracy_test_ats.reset()

    def test_batch(
        self,
    ):
        """
        Perform batch testing on the model.

        This method iterates through the test data loader, making predictions for each batch,
        and calculates various evaluation metrics. It handles both single and multi-sample batches.
        """
        (
            self.preds_total_list,
            self.batch_f1_accs_list,
            self.batch_precision_list,
            self.batch_recall_list,
            self.batch_f1_accs_ats_list,
        ) = ([], [], [], [], [])

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self._test_dl)):
                audio_signal, audio_signal_length, targets, target_lens = batch
                audio_signal = audio_signal.to(self.device)
                audio_signal_length = audio_signal_length.to(self.device)
                targets = targets.to(self.device)
                preds = self.forward(
                    audio_signal=audio_signal,
                    audio_signal_length=audio_signal_length,
                )
                self._get_aux_test_batch_evaluations(batch_idx, preds, targets, target_lens)
                preds = preds.detach().to('cpu')
                if preds.shape[0] == 1:  # batch size = 1
                    self.preds_total_list.append(preds)
                else:
                    self.preds_total_list.extend(torch.split(preds, [1] * preds.shape[0]))
                torch.cuda.empty_cache()

        logging.info(f"Batch F1Acc. MEAN: {torch.mean(torch.tensor(self.batch_f1_accs_list))}")
        logging.info(f"Batch Precision MEAN: {torch.mean(torch.tensor(self.batch_precision_list))}")
        logging.info(f"Batch Recall MEAN: {torch.mean(torch.tensor(self.batch_recall_list))}")
        logging.info(f"Batch ATS F1Acc. MEAN: {torch.mean(torch.tensor(self.batch_f1_accs_ats_list))}")

    def on_validation_epoch_end(self) -> Optional[dict[str, dict[str, torch.Tensor]]]:
        """Run validation with sync_dist=True."""
        return super().on_validation_epoch_end(sync_metrics=True)

    @torch.no_grad()
    def diarize(
        self,
        audio: Union[str, List[str], np.ndarray, DataLoader],
        batch_size: int = 1,
        include_tensor_outputs: bool = False,
        postprocessing_yaml: Optional[str] = None,
        num_workers: int = 0,
        verbose: bool = True,
        override_config: Optional[DiarizeConfig] = None,
    ) -> Union[List[List[str]], Tuple[List[List[str]], List[torch.Tensor]]]:
        """One-click runner function for diarization.

        Args:
            audio: (a single or list) of paths to audio files or path to a manifest file.
            batch_size: (int) Batch size to use during inference.
                Bigger will result in better throughput performance but would use more memory.
            include_tensor_outputs: (bool) Include raw speaker activity probabilities to the output.
                See Returns: for more details.
            postprocessing_yaml: Optional(str) Path to .yaml file with postprocessing parameters.
            num_workers: (int) Number of workers for DataLoader.
            verbose: (bool) Whether to display tqdm progress bar.
            override_config: (Optional[DiarizeConfig]) A config to override the default config.

        Returns:
            *if include_tensor_outputs is False: A list of lists of speech segments with a corresponding speaker index,
                in format "[begin_seconds, end_seconds, speaker_index]".
            *if include_tensor_outputs is True: A tuple of the above list
                and list of tensors of raw speaker activity probabilities.
        """
        return super().diarize(
            audio=audio,
            batch_size=batch_size,
            include_tensor_outputs=include_tensor_outputs,
            postprocessing_yaml=postprocessing_yaml,
            num_workers=num_workers,
            verbose=verbose,
            override_config=override_config,
        )
