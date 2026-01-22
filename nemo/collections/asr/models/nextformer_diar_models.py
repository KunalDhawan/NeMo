#Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import torch.nn.functional as F

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
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
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_ats_targets, get_pil_targets
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import get_pil_targets_hungarian
from nemo.collections.asr.parts.utils.speaker_utils import generate_diarization_output_lines
from nemo.collections.asr.parts.utils.vad_utils import ts_vad_post_processing
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType, LogitsType
from nemo.core.neural_types.elements import ProbsType
from nemo.utils import logging

__all__ = ['NextformerEncLabelModel']

class NextformerEncLabelModel(ModelPT, ExportableEncDecModel, SpkDiarizationMixin):
    """
    Encoder class for Nextformer diarization model.
    Model class creates training, validation methods for setting up data performing model forward pass.

    This model class expects config dict for:
        * preprocessor
        * Transformer Encoder
        * FastConformer Encoder
        * Nextformer Modules
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
        Initialize an Nextformer Diarizer model and a pretrained NEST encoder.
        In this init function, training and validation datasets are prepared.
        """
        torch.set_printoptions(precision=2, sci_mode=False)
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
        super().__init__(cfg=self._cfg, trainer=trainer)
        self.preprocessor = NextformerEncLabelModel.from_config_dict(self._cfg.preprocessor)

        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = NextformerEncLabelModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

        self.encoder = NextformerEncLabelModel.from_config_dict(self._cfg.encoder).to(self.device)
        
        # Get dual decoder mode before decoder initialization
        self.dual_decoder_mode = self._cfg.get("dual_decoder_mode", "concat")  # "concat" or "init" or "init_concat"
        
        # Handle optional query_decoder
        query_decoder_cfg = self._cfg.get('query_decoder', None)
        if query_decoder_cfg is not None:
            self.query_decoder = NextformerEncLabelModel.from_config_dict(query_decoder_cfg).to(self.device)
        else:
            self.query_decoder = None
            
        # Handle optional query_decoder_raw
        query_decoder_raw_cfg = self._cfg.get('query_decoder_raw', None)
        if query_decoder_raw_cfg is not None:
            self.query_decoder_raw = NextformerEncLabelModel.from_config_dict(query_decoder_raw_cfg).to(self.device)
        else:
            self.query_decoder_raw = None

        query_enhancer_cfg = self._cfg.get('query_enhancer', None)
        if query_enhancer_cfg is not None:
            self.query_enhancer = NextformerEncLabelModel.from_config_dict(query_enhancer_cfg).to(self.device)
        else:
            self.query_enhancer = None
            
        self.transformer_encoder = NextformerEncLabelModel.from_config_dict(self._cfg.transformer_encoder).to(
            self.device
        )

        self.nextformer_modules = NextformerEncLabelModel.from_config_dict(self._cfg.nextformer_modules).to(
            self.device
        )
        if self._cfg.encoder.d_model != self._cfg.model_defaults.tf_d_model:
            self.nextformer_modules.encoder_proj = self.nextformer_modules.encoder_proj.to(self.device)
        else:
            self.nextformer_modules.encoder_proj = None
        if self.query_decoder is not None:
            self.nextformer_modules.query_proj = self.nextformer_modules.query_proj.to(self.device)
        else:
            self.nextformer_modules.query_proj = None
        if self.query_decoder_raw is not None:
            self.nextformer_modules.query_raw_proj = self.nextformer_modules.query_raw_proj.to(self.device)
        else:
            self.nextformer_modules.query_raw_proj = None
        if self.query_decoder is not None and self.query_decoder_raw is not None and self.dual_decoder_mode != "init":
            self.nextformer_modules.query_combiner = self.nextformer_modules.query_combiner.to(self.device)
            logging.info(f"Both query_decoder and query_decoder_raw are enabled with mode '{self.dual_decoder_mode}'. Using query_combiner to merge outputs.")
        else:
            self.nextformer_modules.query_combiner = None
            if self.query_decoder is None and self.query_decoder_raw is None:
                raise ValueError("At least one of query_decoder or query_decoder_raw must be specified in the config.")
            elif self.query_decoder is not None and self.query_decoder_raw is not None:
                logging.info(f"Both query_decoder and query_decoder_raw are enabled with mode '{self.dual_decoder_mode}' (init mode - no combiner).")
            elif self.query_decoder is not None:
                logging.info("Using only query_decoder (query_decoder_raw not specified).")
            else:
                logging.info("Using only query_decoder_raw (query_decoder not specified).")
        
        self._init_loss_weights()

        self.eps = 1e-3
        self.negative_init_val = -99
        self.loss = instantiate(self._cfg.loss)
        self.q_sim_loss = instantiate(self._cfg.q_sim_loss)
        self.emb_sim_loss = instantiate(self._cfg.emb_sim_loss)
        self.local_mask_threshold = self._cfg.get("local_mask_threshold", 0.5)
        self.initialize_queries = self._cfg.get("initialize_queries", True)
        self.initialize_mask = self._cfg.get("initialize_mask", True)
        self.pil_metric = self._cfg.get("pil_metric", "bce")
        self.oracle_queries_train = self._cfg.get("oracle_queries_train", False)
        self.oracle_queries_test = self._cfg.get("oracle_queries_test", False)
        self.oracle_assignment = self._cfg.get("oracle_assignment", False)
        
        self.streaming_mode = self.cfg.get("streaming_mode", False)
        self.save_hyperparameters("cfg")
        self._init_eval_metrics()
        speaker_inds = list(range(self._cfg.local_num_spks))
        self.speaker_permutations = torch.tensor(list(itertools.permutations(speaker_inds)))  # Get all permutations

        self.max_batch_dur = self._cfg.get("max_batch_dur", 20000)

    def _init_loss_weights(self):
        pil_weight = self._cfg.get("pil_weight", 1.0)
        ats_weight = self._cfg.get("ats_weight", 0.0)
        total_weight = pil_weight + ats_weight
        if total_weight == 0:
            raise ValueError(
                f"weights for PIL {pil_weight} and ATS {ats_weight} cannot sum to 0"
            )
        self.pil_weight = pil_weight / total_weight
        self.ats_weight = ats_weight / total_weight
        self.global_pil_weight = self._cfg.get("global_pil_weight", 1.0)

    def _init_eval_metrics(self):
        """
        If there is no label, then the evaluation metrics will be based on Permutation Invariant Loss (PIL).
        """
        self._accuracy_test = MultiBinaryAccuracy()
        self._accuracy_test_op = MultiBinaryAccuracy()
        self._accuracy_test_local = MultiBinaryAccuracy()
        self._accuracy_test_local_ats = MultiBinaryAccuracy()
        self._accuracy_train = MultiBinaryAccuracy()
        self._accuracy_train_global = MultiBinaryAccuracy()
        self._accuracy_valid = MultiBinaryAccuracy()
        self._accuracy_valid_global = MultiBinaryAccuracy()
        self._accuracy_train_global_op = MultiBinaryAccuracy()
        self._accuracy_valid_global_op = MultiBinaryAccuracy()

        self._accuracy_train_ats = MultiBinaryAccuracy()
        self._accuracy_valid_ats = MultiBinaryAccuracy()

    def _reset_train_metrics(self):
        self._accuracy_train.reset()
        self._accuracy_train_ats.reset()
        self._accuracy_train_global.reset()
        self._accuracy_train_global_op.reset()

    def _reset_valid_metrics(self):
        self._accuracy_valid.reset()
        self._accuracy_valid_ats.reset()
        self._accuracy_valid_global.reset()
        self._accuracy_valid_global_op.reset()

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
            global_rank=global_rank,
            soft_targets=config.soft_targets if 'soft_targets' in config else False,
            device=self.device,
            subsegment_mode=config.get('subsegment_mode', False),
            subsegment_min_len_sec=config.get('subsegment_min_len_sec', 15.0),
            subsegment_two_chunks_rate=config.get('subsegment_two_chunks_rate', 0.0),
            subsegment_min_chunk_len_sec=config.get('subsegment_min_chunk_len_sec', 10.0),
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
                "logits": NeuralType(('B', 'T', 'C'), LogitsType()),
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
        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)
        emb_seq, emb_seq_length = self.encoder(
            audio_signal=processed_signal,
            length=processed_signal_length,
            bypass_pre_encode=bypass_pre_encode,
        )
        emb_seq = emb_seq.transpose(1, 2)
        return emb_seq, emb_seq_length

    def forward(
        self,
        audio_signal,
        audio_signal_length,
        targets: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for training and inference.

        Args:
            audio_signal (torch.Tensor): Tensor containing audio waveform
                Shape: (batch_size, num_samples)
            audio_signal_length (torch.Tensor): Tensor containing lengths of audio waveforms
                Shape: (batch_size,)
            targets (torch.Tensor, optional): Ground truth speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers). Defaults to None.

        Returns:
            logits (torch.Tensor): Sorted tensor containing predicted speaker labels
                Shape: (batch_size, max. diar frame count, max_num_speakers)
            local_logits (torch.Tensor): Tensor containing local speaker logits.
                Shape: (num_chunks * batch_size, lc+chunk_len+rc, local_num_spks)
            local_queries (torch.Tensor): Tensor containing local speaker queries.
                Shape: (num_chunks * batch_size, local_num_spks, query_dim)
            active_frames_per_query (torch.Tensor): Tensor containing the number of active frames per query
                Shape: (num_chunks * batch_size, local_num_spks)
        """
        processed_signal, processed_signal_length = self.process_signal(
            audio_signal=audio_signal, audio_signal_length=audio_signal_length
        )
        processed_signal = processed_signal[:, :, : processed_signal_length.max()]
        if self.streaming_mode:
            raise NotImplementedError("Streaming mode is not implemented for Nextformer model.")
        else:
            logits, local_logits, local_queries, active_frames_per_query = self.forward_offline(
                processed_signal=processed_signal, processed_signal_length=processed_signal_length, targets=targets
            )
            return logits, local_logits, local_queries, active_frames_per_query

    def _create_batch_of_chunks_extra(
        self,
        input_tensor: torch.Tensor,
        input_lengths: Optional[torch.Tensor],
        lc: int,
        chunk_len: int,
        rc: int,
        extra_lc: int = 0,
        extra_rc: int = 0,
        silence_frames: int = 3,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """
        Create a batch of chunks with extra context, reordered to exploit ATS ordering.
        
        Reordering: [lc+chunk+rc | extra_rc | silence | extra_lc]
        This ensures speakers in the prediction window appear first in ATS order.
        
        Args:
            input_tensor: Input tensor to chunk
                Shape: (batch_size, n_frames, feature_dim)
            input_lengths: Optional tensor containing lengths of input sequences
                Shape: (batch_size,)
            lc: Left context size for prediction window
            chunk_len: Main chunk length for prediction window
            rc: Right context size for prediction window
            extra_lc: Extra left context (for encoder acoustic context only)
            extra_rc: Extra right context (for encoder acoustic context only)
            silence_frames: Number of silence frames to insert between sections
            
        Returns:
            batch_chunks: Batched chunks tensor with reordered frames
                Shape: (batch_size * num_chunks, chunk_total, feature_dim)
                where chunk_total = lc + chunk_len + rc + extra_rc + silence_frames + extra_lc
            batch_chunk_lengths: Full lengths including extra context (for encoder)
                Shape: (batch_size * num_chunks,) or None
            batch_chunk_prediction_lengths: Prediction window lengths only (lc+chunk+rc)
                Shape: (batch_size * num_chunks,) or None
            num_chunks: Number of chunks created
        """
        batch_size = input_tensor.shape[0]
        total_n_frames = input_tensor.shape[1]
        feature_dim = input_tensor.shape[-1]
        num_chunks = math.ceil(total_n_frames / chunk_len)
        
        # Calculate total chunk size after reordering
        prediction_window_size = lc + chunk_len + rc
        # Only include silence if we have extra_lc (silence separates extra_right from extra_left)
        actual_silence_frames = silence_frames if extra_lc > 0 else 0
        chunk_total = prediction_window_size + extra_rc + actual_silence_frames + extra_lc
        
        # Pre-allocate batch tensors
        batch_chunks = torch.zeros(
            (batch_size * num_chunks, chunk_total, feature_dim),
            dtype=input_tensor.dtype,
            device=input_tensor.device
        )
        
        batch_chunk_lengths = None
        batch_chunk_prediction_lengths = None
        if input_lengths is not None:
            batch_chunk_lengths = torch.zeros(
                (batch_size * num_chunks,),
                dtype=input_lengths.dtype,
                device=input_lengths.device
            )
            batch_chunk_prediction_lengths = torch.zeros(
                (batch_size * num_chunks,),
                dtype=input_lengths.dtype,
                device=input_lengths.device
            )
        
        # Fill pre-allocated tensors directly
        for chunk_idx in range(num_chunks):
            # Calculate positions in original sequence
            chunk_start = chunk_idx * chunk_len
            chunk_end = min(chunk_start + chunk_len, total_n_frames)
            
            # Main prediction window: [lc + chunk + rc]
            main_left = max(0, chunk_start - lc)
            main_right = min(chunk_end + rc, total_n_frames)
            
            # Extra context regions (in original sequence)
            extra_left_start = max(0, chunk_start - lc - extra_lc)
            extra_left_end = max(0, chunk_start - lc)
            extra_right_start = min(chunk_end + rc, total_n_frames)
            extra_right_end = min(chunk_end + rc + extra_rc, total_n_frames)

            #logging.info(f"chunk {chunk_idx} start: {chunk_start}, end: {chunk_end}, main_left: {main_left}, main_right: {main_right}, extra_left_start: {extra_left_start}, extra_left_end: {extra_left_end}, extra_right_start: {extra_right_start}, extra_right_end: {extra_right_end}")

            # Calculate actual sizes
            main_size = main_right - main_left
            extra_left_size = extra_left_end - extra_left_start
            extra_right_size = extra_right_end - extra_right_start

            #logging.info(f"chunk {chunk_idx} main_size: {main_size}, extra_left_size: {extra_left_size}, extra_right_size: {extra_right_size}")
            
            # Extract data from input_tensor
            main_window = input_tensor[:, main_left:main_right, :]  # (batch_size, main_size, feature_dim)
            extra_left_data = input_tensor[:, extra_left_start:extra_left_end, :] if extra_lc > 0 else None
            extra_right_data = input_tensor[:, extra_right_start:extra_right_end, :] if extra_rc > 0 else None
            
            # Calculate indices in the batch tensor
            batch_start_idx = chunk_idx * batch_size
            batch_end_idx = (chunk_idx + 1) * batch_size
            
            # Reorder: [main_window | extra_right | silence | extra_left]
            pos = 0
            
            # 1. Main window (lc + chunk + rc)
            batch_chunks[batch_start_idx:batch_end_idx, pos:pos+main_size, :] = main_window
            pos += main_size
            
            # 2. Extra right context (immediately after main data)
            if extra_right_data is not None and extra_right_size > 0:
                batch_chunks[batch_start_idx:batch_end_idx, pos:pos+extra_right_size, :] = extra_right_data
            pos += extra_right_size
            
            # 3. Silence frames (only if we have extra_lc)
            pos += actual_silence_frames
            
            # 4. Extra left context
            if extra_left_data is not None and extra_left_size > 0:
                batch_chunks[batch_start_idx:batch_end_idx, pos:pos+extra_left_size, :] = extra_left_data
            
            # Calculate chunk lengths if input_lengths provided
            if batch_chunk_lengths is not None:
                # Prediction window length: valid frames in main window
                pred_window_len = torch.clamp(input_lengths - main_left, min=0, max=main_size)
                batch_chunk_prediction_lengths[batch_start_idx:batch_end_idx] = pred_window_len
                
                # Full chunk length: sum of valid frames across all sections
                # Section 1: main window
                valid_main = pred_window_len
                
                # Section 2: extra right (frames from extra_right_start to min(input_lengths, extra_right_end))
                valid_extra_right = torch.clamp(input_lengths - extra_right_start, min=0, max=extra_right_size)
                
                # Section 4: extra left (frames from extra_left_start to min(input_lengths, extra_left_end))
                valid_extra_left = torch.clamp(input_lengths - extra_left_start, min=0, max=extra_left_size)
                
                # Section 3: silence (only if we have extra_lc and valid extra_left data)
                valid_silence = torch.where(valid_extra_left > 0, actual_silence_frames, 0)
                
                total_valid = valid_main + valid_extra_right + valid_silence + valid_extra_left
                batch_chunk_lengths[batch_start_idx:batch_end_idx] = total_valid
        
        return batch_chunks, batch_chunk_lengths, batch_chunk_prediction_lengths, num_chunks

    def _create_batch_of_chunks(
        self,
        input_tensor: torch.Tensor,
        input_lengths: Optional[torch.Tensor],
        lc: int,
        chunk_len: int,
        rc: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[int], List[int], int, int]:
        """
        Create a batch of chunks from input tensor by slicing with left/right context.
        
        Args:
            input_tensor: Input tensor to chunk
                Shape: (batch_size, n_frames, feature_dim)
            input_lengths: Optional tensor containing lengths of input sequences
                Shape: (batch_size,)
            lc: Left context size
            chunk_len: Main chunk length
            rc: Right context size
            
        Returns:
            batch_chunks: Batched chunks tensor
                Shape: (batch_size * num_chunks, chunk_total, feature_dim)
            batch_chunk_lengths: Optional tensor containing lengths for each chunk
                Shape: (batch_size * num_chunks,) or None if input_lengths is None
            num_chunks: Number of chunks created
        """
        batch_size = input_tensor.shape[0]
        chunk_total = lc + chunk_len + rc
        total_n_frames = input_tensor.shape[1]
        num_chunks = math.ceil(total_n_frames / chunk_len)
        feature_dim = input_tensor.shape[-1]
        
        # Pre-allocate batch tensors
        batch_chunks = torch.zeros(
            (batch_size * num_chunks, chunk_total, feature_dim),
            dtype=input_tensor.dtype,
            device=input_tensor.device
        )
        
        batch_chunk_lengths = None
        if input_lengths is not None:
            batch_chunk_lengths = torch.zeros(
                (batch_size * num_chunks,),
                dtype=input_lengths.dtype,
                device=input_lengths.device
            )
        

        # Fill pre-allocated tensors directly
        for chunk_idx in range(num_chunks):
            # Calculate start and end positions in pre_encode space
            chunk_start = chunk_idx * chunk_len
            chunk_end = min(chunk_start + chunk_len, total_n_frames)
            
            # Calculate left context (0 for first chunk, lc otherwise)
            left_context_start = max(0, chunk_start - lc)
            
            # Calculate right context
            right_context_end = min(chunk_end + rc, total_n_frames)
            
            # Calculate indices in the batch tensor
            batch_start_idx = chunk_idx * batch_size
            batch_end_idx = (chunk_idx + 1) * batch_size
            
            # Extract chunk with context from input_tensor
            chunk_data = input_tensor[:, left_context_start:right_context_end, :]
            chunk_data_size = chunk_data.shape[1]
            
            # Fill the batch_chunks tensor directly (no left padding, only right padding if needed)
            # Copy data starting from the beginning of the chunk
            batch_chunks[batch_start_idx:batch_end_idx, :chunk_data_size, :] = chunk_data
            # Right padding is already zeros (from initialization), so no need to fill
            
            # Calculate chunk lengths if input_lengths provided
            if batch_chunk_lengths is not None:
                # The valid length is the number of valid frames in the chunk
                # Valid data length: remaining_length_from_left_context_start, but not more than chunk_data_size
                chunk_lengths = torch.clamp(input_lengths - left_context_start, min=0, max=chunk_data_size)
                batch_chunk_lengths[batch_start_idx:batch_end_idx] = chunk_lengths
        
        return batch_chunks, batch_chunk_lengths, num_chunks

    def forward_offline(
        self,
        processed_signal,
        processed_signal_length,
        targets: Optional[torch.Tensor] = None,
    ):
        """
        The main forward pass for diarization in offline mode (for training/validation).
        Processes the entire signal at once by creating a batch of chunks.

        Args:
            processed_signal (torch.Tensor): Tensor containing preprocessed audio features
                Shape: (batch_size, channels, feature_length)
            processed_signal_length (torch.Tensor): Tensor containing lengths of audio features
                Shape: (batch_size,)
            targets (torch.Tensor, optional): Ground truth speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers). Defaults to None.

        Returns:
            logits (torch.Tensor): Tensor containing predicted speaker labels
                Shape: (batch_size, total_n_frames, max_num_spks)
            local_logits (torch.Tensor): Tensor containing local speaker logits
                Shape: (num_chunks * batch_size, lc+chunk_len+rc, local_num_spks)
            local_queries (torch.Tensor): Tensor containing local speaker queries
                Shape: (num_chunks * batch_size, local_num_spks, query_dim)
            active_frames_per_query (torch.Tensor): Tensor containing the number of active frames per query
                Shape: (num_chunks * batch_size, local_num_spks)
        """
        batch_size, ch, sig_length = processed_signal.shape

        # Step 1: Pad processed_signal similarly to forward_streaming
        if dist.is_available() and dist.is_initialized():
            local_tensor = torch.tensor([sig_length], device=processed_signal.device)
            dist.all_reduce(
                local_tensor, op=dist.ReduceOp.MAX, async_op=False
            )  # get max feature length across all GPUs
            max_n_frames = local_tensor.item()
            #if dist.get_rank() == 0:
            #    logging.info(f"Maximum feature length across all GPUs: {max_n_frames}")
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
            if rand_num < self.nextformer_modules.causal_attn_rate:
                self.encoder.att_context_size = [-1, self.nextformer_modules.causal_attn_rc]
                self.transformer_encoder.diag = self.nextformer_modules.causal_attn_rc
                att_mod = True

        # Step 2: Get pre_encode_embs and pre_encode_lengths for the whole batch
        # Transpose to (batch_size, feature_length, channels) for pre_encode
        processed_signal_t = processed_signal.transpose(1, 2)
        pre_encode_embs, pre_encode_lengths = self.encoder.pre_encode(
            x=processed_signal_t, lengths=processed_signal_length
        )
        total_n_frames = pre_encode_embs.shape[1]  # Total number of frames after pre_encode
        # pre_encode_embs shape: (batch_size, n_frames, fc_d_model)
        # pre_encode_lengths shape: (batch_size,)

        # Step 3: Create a batch of chunks by slicing pre_encode_embs
        lc = self.nextformer_modules.chunk_left_context
        chunk_len = self.nextformer_modules.chunk_len
        rc = self.nextformer_modules.chunk_right_context
        extra_lc = self.nextformer_modules.extra_left_context
        extra_rc = self.nextformer_modules.extra_right_context
        
        # Use extra context chunking if either extra context is enabled
        if extra_lc > 0 or extra_rc > 0:
            batch_chunks, batch_chunk_lengths, batch_chunk_prediction_lengths, num_chunks = self._create_batch_of_chunks_extra(
                input_tensor=pre_encode_embs,
                input_lengths=pre_encode_lengths,
                lc=lc,
                chunk_len=chunk_len,
                rc=rc,
                extra_lc=extra_lc,
                extra_rc=extra_rc,
                silence_frames=self.nextformer_modules.extra_silence_frames,
            )
            #logging.info(f"batch_chunks shape: {batch_chunks.shape}")
            #logging.info(f"batch_chunk_lengths: {batch_chunk_lengths}")
            #logging.info(f"batch_chunk_prediction_lengths: {batch_chunk_prediction_lengths}")
            #logging.info(f"num_chunks: {num_chunks}")
        else:
            batch_chunks, batch_chunk_lengths, num_chunks = self._create_batch_of_chunks(
                input_tensor=pre_encode_embs,
                input_lengths=pre_encode_lengths,
                lc=lc,
                chunk_len=chunk_len,
                rc=rc,
            )
            batch_chunk_prediction_lengths = batch_chunk_lengths  # Same as full length when no extra context
        
        # Extract prediction window from reordered chunks (if using extra contexts)
        if extra_lc > 0 or extra_rc > 0:
            prediction_window_size = lc + chunk_len + rc
            batch_chunks_prediction_window = batch_chunks[:, :prediction_window_size, :]
        else:
            batch_chunks_prediction_window = batch_chunks
        
        # Apply query_raw_proj if available
        if self.nextformer_modules.query_raw_proj is not None:
            query_raw_proj = self.nextformer_modules.query_raw_proj(batch_chunks_prediction_window)
        else:
            query_raw_proj = batch_chunks_prediction_window

        # Step 5: Run frontend_encoder, forward_infer, query_decoder in one pass
        # Get encoder embeddings for all chunks
        emb_seq, emb_seq_length = self.frontend_encoder(
            processed_signal=batch_chunks, processed_signal_length=batch_chunk_lengths, bypass_pre_encode=True
        )
        
        # Step 6: Extract only prediction window from encoder output (if using extra contexts)
        # The encoder has processed the full reordered sequence, but we only need predictions for the main window
        if extra_lc > 0 or extra_rc > 0:
            prediction_window_size = lc + chunk_len + rc
            emb_seq = emb_seq[:, :prediction_window_size, :]  # Extract only prediction window
            emb_seq_length = batch_chunk_prediction_lengths  # Use prediction window lengths
            #logging.info(f"emb_seq shape: {emb_seq.shape}")
            #logging.info(f"emb_seq_length: {emb_seq_length}")
        
        if self.nextformer_modules.encoder_proj is not None:
            emb_seq_enc_proj = self.nextformer_modules.encoder_proj(emb_seq)
        else:
            emb_seq_enc_proj = emb_seq

        if self.nextformer_modules.query_proj is not None:
            emb_seq_query_proj = self.nextformer_modules.query_proj(emb_seq)
        else:
            emb_seq_query_proj = emb_seq
        
        # Get local logits for all chunks
        local_logits = self.forward_infer(emb_seq_enc_proj, emb_seq_length)
        #logging.info(f"local logits shape: {local_logits.shape}")
        #logging.info(f"emb_seq_length: {emb_seq_length}")
        # logits shape: (batch_size * num_chunks, chunk_total, local_num_spks)

        # Get speaker queries for all chunks
        encoder_len_mask = self.nextformer_modules.length_to_mask(emb_seq_length, emb_seq.shape[1])
        encoder_len_mask = ~encoder_len_mask
        #logging.info(f"encoder_len_mask: {encoder_len_mask.to(int).sum(dim=1)}")

        # Handle oracle mode (targets) if provided
        local_target_indices = None
        if targets is not None:
            # Pad targets to match pre_encode_embs if necessary (when signal was padded for multi-GPU)
            target_n_frames = targets.shape[1]
            if target_n_frames < total_n_frames:
                # Pad targets with zeros to match pre_encode_embs length
                pad_size = total_n_frames - target_n_frames
                targets = torch.nn.functional.pad(targets, (0, 0, 0, pad_size), mode='constant', value=0)
                logging.info(f"Padded targets from {target_n_frames} to {total_n_frames} frames to match padded signal")
            elif target_n_frames > total_n_frames:
                logging.info(f"WARNING! targets has more frames than pre_encode_embs ({target_n_frames} > {total_n_frames}). Truncating targets.")
                targets = targets[:, :total_n_frames, :]
            
            # Create batch of target chunks using the same chunking function
            batch_targets, _, _ = self._create_batch_of_chunks(
                input_tensor=targets,
                input_lengths=None,  # Targets don't need length tracking for chunking
                lc=lc,
                chunk_len=chunk_len,
                rc=rc,
            )
            
            # Get oracle predictions using Hungarian algorithm
            logits_len = min(local_logits.shape[1], batch_targets.shape[1])
            local_pil_targets, local_target_indices = get_pil_targets_hungarian(
                labels=batch_targets[:, :logits_len, :],
                logits=local_logits[:, :logits_len, :],
                metric=self.pil_metric
            )
            if (self.oracle_queries_train and self.training) or (self.oracle_queries_test and not self.training):
                preds = local_pil_targets
            else:
                preds = torch.sigmoid(local_logits)
        else:
            preds = torch.sigmoid(local_logits)

        # masks for the target speaker and non-target speakers in extra cross-attention
        # encoder_mask_extra should have False (0) on frames where current speaker is inactive and any other speaker is active
        if self.initialize_mask:
            encoder_query_mask = ~(preds > self.local_mask_threshold).transpose(1, 2)  # (batch, local_num_spks, n_frames)
            any_speaker_active = (preds.max(dim=2)[0] > self.local_mask_threshold).unsqueeze(1)  # (batch, 1, n_frames)
            any_speaker_active = any_speaker_active.expand(-1, preds.shape[2], -1)  # (batch, local_num_spks, n_frames)
            encoder_mask_extra = ~(encoder_query_mask & any_speaker_active)
            #logging.info(f"encoder_mask_extra: {encoder_mask_extra.to(int)[0, :, :20]}")
            #logging.info(f"encoder_query_mask: {encoder_query_mask.to(int)[0, :, :20]}")
        else:
            encoder_query_mask = None
            encoder_mask_extra = None
          
        local_num_spks = preds.shape[-1]
        active_frames_per_query = (preds > self.local_mask_threshold).to(int).sum(dim=1) # (num_chunks * batch_size, local_num_spks)
        #logging.info(f"active frames per query: {active_frames_per_query}")
        spk_detected = active_frames_per_query > 0 # (num_chunks * batch_size, local_num_spks)
        spk_not_detected = ~spk_detected # (num_chunks * batch_size, local_num_spks)
        query_mask_from = spk_not_detected.unsqueeze(2).expand(-1, -1, local_num_spks)  # (num_chunks * batch_size, local_num_spks, local_num_spks)
        query_mask_to = spk_not_detected.unsqueeze(1).expand(-1, local_num_spks, -1)  # (num_chunks * batch_size, local_num_spks, local_num_spks)
        query_mask = query_mask_from | query_mask_to  # (num_chunks * batch_size, local_num_spks, local_num_spks)
        #logging.info(f"query_mask: {query_mask}")
        #logging.info(f"spk_not_detected: {spk_not_detected.to(int).sum(dim=1)}")

        # Step 3: Run both query_decoder and query_decoder_raw with query_mask
        if self.query_decoder_raw is not None:
            if self.initialize_queries:
                init_queries_raw = self.nextformer_modules.get_init_queries(preds, query_raw_proj)
            else:
                init_queries_raw = None
            spk_queries_raw = self.query_decoder_raw(
                encoder_states=query_raw_proj,
                encoder_len_mask=encoder_len_mask,
                encoder_mask=encoder_query_mask,
                encoder_mask_extra=encoder_mask_extra,
                query_states=init_queries_raw,
                query_mask=query_mask
            )
            # spk_queries_raw shape: (num_chunks * batch_size, local_num_spks, emb_dim)
            # Zero out queries for undetected speakers
            spk_queries_raw = spk_queries_raw.masked_fill(spk_not_detected.unsqueeze(2), 0)
        else:
            spk_queries_raw = None

        if self.query_decoder is not None:
            if spk_queries_raw is not None and (self.dual_decoder_mode == "init" or self.dual_decoder_mode == "init_concat"):
                init_queries = spk_queries_raw
            elif self.initialize_queries:
                init_queries = self.nextformer_modules.get_init_queries(preds, emb_seq_query_proj)                    
            else:
                init_queries = None

            spk_queries = self.query_decoder(
                encoder_states=emb_seq_query_proj,
                encoder_len_mask=encoder_len_mask,
                encoder_mask=encoder_query_mask,
                encoder_mask_extra=encoder_mask_extra,
                query_states=init_queries,
                query_mask=query_mask
            )
            # spk_queries shape: (num_chunks * batch_size, local_num_spks, emb_dim)
            # Zero out queries for undetected speakers
            spk_queries = spk_queries.masked_fill(spk_not_detected.unsqueeze(2), 0)
        else:
            spk_queries = None

        # Combine queries from both decoders if both are used
        if spk_queries is not None and spk_queries_raw is not None:
            if self.nextformer_modules.query_combiner is not None:
                # Store original dtype to preserve it
                original_dtype = spk_queries.dtype
                # Concatenate along embedding dimension and project back
                spk_queries_combined = torch.cat([spk_queries, spk_queries_raw], dim=-1)
                spk_queries = self.nextformer_modules.query_combiner(spk_queries_combined)
                # Ensure output has the same dtype as input
                if spk_queries.dtype != original_dtype:
                    spk_queries = spk_queries.to(original_dtype)
            else:
                # If no combiner is provided, just use the first decoder output
                # (alternatively could raise an error or average the two)
                logging.warning("Both query decoders are used but no query_combiner provided. Using only query_decoder output.")
                spk_queries = spk_queries
        elif spk_queries_raw is not None:
            # Only raw decoder is used
            spk_queries = spk_queries_raw
        # else: only spk_queries is not None, use it as is

        if att_mod:
            self.encoder.att_context_size = [-1, -1]
            self.transformer_encoder.diag = None

        if self.query_enhancer is not None:
            emb_dim = spk_queries.shape[-1]
            spk_queries_reshaped = spk_queries.view(num_chunks, batch_size, local_num_spks, emb_dim).transpose(0, 1)
            spk_queries_reshaped = spk_queries_reshaped.reshape(batch_size, num_chunks * local_num_spks, emb_dim)
            valid_query_mask = spk_detected.view(num_chunks, batch_size, local_num_spks).transpose(0, 1).reshape(batch_size, num_chunks * local_num_spks)
            spk_queries_enhanced = self.query_enhancer(encoder_states=spk_queries_reshaped, encoder_mask=valid_query_mask)
            spk_queries_enhanced = spk_queries_enhanced.masked_fill(~valid_query_mask.unsqueeze(-1), 0)
            spk_queries = spk_queries_enhanced.view(batch_size, num_chunks, local_num_spks, emb_dim).transpose(0, 1)
            spk_queries = spk_queries.reshape(num_chunks * batch_size, local_num_spks, emb_dim)

        # Collect per-chunk logits and concatenate for differentiable assembly
        logits_list = []

        streaming_state = self.nextformer_modules.init_streaming_state(
            batch_size=processed_signal.shape[0], device=self.device
        )

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_len
            end = min(start + chunk_len, total_n_frames)
            dur = end - start
            offset = min(lc, start)
            spk_queries_chunk = spk_queries[chunk_idx * batch_size:(chunk_idx + 1) * batch_size, :, :] # (batch_size, local_num_spks, emb_dim)
            local_logits_chunk = local_logits[chunk_idx * batch_size:(chunk_idx + 1) * batch_size, :, :] # (batch_size, lc+chunk_len+rc, local_num_spks)

            if self.oracle_assignment and local_target_indices is not None:
                # Use oracle local-to-global assignments
                global_spk_indices_oracle = local_target_indices[
                    chunk_idx * batch_size:(chunk_idx + 1) * batch_size, :
                ]  # (batch_size, local_num_spks)
                valid_mask = global_spk_indices_oracle >= 0
                safe_indices = global_spk_indices_oracle.clamp(min=0)
                spk_assignments_chunk = F.one_hot(
                    safe_indices, num_classes=self.nextformer_modules.max_num_spks
                ).to(local_logits_chunk.dtype)
                spk_assignments_chunk = spk_assignments_chunk * valid_mask.unsqueeze(-1)
            else:
                spk_assignments_chunk = self.nextformer_modules.get_local_to_global_assignments(
                    spk_queries=spk_queries_chunk,
                    streaming_state=streaming_state,
                )

            active_frames_chunk = active_frames_per_query[chunk_idx * batch_size:(chunk_idx + 1) * batch_size, :]
            self.nextformer_modules.update_streaming_state(
                streaming_state=streaming_state,
                spk_queries=spk_queries_chunk,
                spk_assignments=spk_assignments_chunk,
                active_frames_per_query=active_frames_chunk
            )
            logits_chunk = self.nextformer_modules.get_global_logits(
                local_logits=local_logits_chunk,
                spk_assignments=spk_assignments_chunk,
                offset=offset,
                dur=dur
            )
            logits_list.append(logits_chunk)

        logits = torch.cat(logits_list, dim=1)
        
        # Remove padding from logits if necessary
        if sig_length < max_n_frames:  # Discard preds corresponding to padding
            n_frames = math.ceil(sig_length / self.encoder.subsampling_factor)
            logits = logits[:, :n_frames, :]

        return logits, local_logits, spk_queries, active_frames_per_query

    def forward_infer(self, emb_seq, emb_seq_length):
        """
        The main forward pass for diarization for offline diarization inference.

        Args:
            emb_seq (torch.Tensor): Tensor containing FastConformer encoder states (embedding vectors).
                Shape: (batch_size, diar_frame_count, emb_dim)
            emb_seq_length (torch.Tensor): Tensor containing lengths of FastConformer encoder states.
                Shape: (batch_size,)

        Returns:
            logits (torch.Tensor): Tensor containing local speaker logits.
                Shape: (batch_size, diar_frame_count, num_speakers)
        """
        encoder_mask = self.nextformer_modules.length_to_mask(emb_seq_length, emb_seq.shape[1])
        trans_emb_seq = self.transformer_encoder(encoder_states=emb_seq, encoder_mask=encoder_mask)
        logits = self.nextformer_modules.forward_spk_logits(trans_emb_seq)
        mask = encoder_mask.unsqueeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        return logits

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
            _, logits, _, _, _, _ = self.forward(audio_signal=batch[0], audio_signal_length=batch[1])
            preds = torch.sigmoid(logits)
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
                    unit_10ms_frame_count=int(self._cfg.encoder.subsampling_factor),
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
            'num_spks': config.get('num_spks', self._cfg.max_num_spks),
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
        #if not self.streaming_mode:
        #    audio_signal = (1 / (audio_signal.max() + self.eps)) * audio_signal

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

    def _process_logits_and_targets(
        self,
        local_logits: torch.Tensor,
        targets: torch.Tensor,
        target_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process logits and targets tensors (not lists) to get batched tensors for offline mode.
        
        Args:
            local_logits (torch.Tensor): Speaker logits for all chunks
                Shape: (batch_size * num_chunks, chunk_total, local_num_spks)
            targets (torch.Tensor): Ground truth speaker labels
                Shape: (batch_size, total_n_frames, max_num_spks)
            target_lens (torch.Tensor): Lengths of target sequences
                Shape: (batch_size,)
        
        Returns:
            local_pil_targets (torch.Tensor): PIL targets for all chunks
                Shape: (batch_size * num_chunks, chunk_total, max_num_spks)
            local_ats_targets (torch.Tensor): ATS targets for all chunks
                Shape: (batch_size * num_chunks, chunk_total, max_num_spks)
            local_target_lens (torch.Tensor): Target lengths for all chunks
                Shape: (batch_size * num_chunks,)
            local_target_indices (torch.Tensor): Target indices mapping
                Shape: (num_chunks, batch_size, local_num_spks)
            total_logits_op (torch.Tensor): Oracle-permuted logits
                Shape: (batch_size, total_n_frames, max_num_spks)
        """
        # Extract dimensions
        batch_size = targets.shape[0]
        targets_num_spks = targets.shape[-1]
        total_n_frames = targets.shape[1]
        
        # Create batch of target chunks using the same chunking function
        local_targets, local_target_lens, _ = self._create_batch_of_chunks(
            input_tensor=targets,
            input_lengths=target_lens,
            lc=self.nextformer_modules.chunk_left_context,
            chunk_len=self.nextformer_modules.chunk_len,
            rc=self.nextformer_modules.chunk_right_context,
        )
        # local_targets shape: (batch_size * num_chunks, chunk_total, max_num_spks)
        # local_target_lens shape: (batch_size * num_chunks,)

        #logging.info(f"local logits shape: {local_logits.shape}")
        #logging.info(f"local targets shape: {local_targets.shape}")

        local_preds = torch.sigmoid(local_logits)
        if local_targets.shape[0] < local_logits.shape[0]:
            pad_size = local_logits.shape[0] - local_targets.shape[0]
            logging.info(f"Padding local targets from {local_targets.shape[0]} to {local_logits.shape[0]}")
            # For 3D tensor: pad dimension 0 (first dim) on the right
            # Padding tuple format: (pad_dim2_left, pad_dim2_right, pad_dim1_left, pad_dim1_right, pad_dim0_left, pad_dim0_right)
            local_targets = torch.nn.functional.pad(local_targets, (0, 0, 0, 0, 0, pad_size), mode='constant', value=0)
            # For 1D tensor: pad dimension 0 on the right
            # Padding tuple format: (pad_dim0_left, pad_dim0_right)
            local_target_lens = torch.nn.functional.pad(local_target_lens, (0, pad_size), mode='constant', value=0)
        elif local_targets.shape[0] > local_logits.shape[0]:
            logging.info(f"Truncating local targets from {local_targets.shape[0]} to {local_logits.shape[0]}")
            local_targets = local_targets[:local_logits.shape[0], :, :]
            local_target_lens = local_target_lens[:local_logits.shape[0]]

        local_pil_targets, local_target_indices = get_pil_targets_hungarian(labels=local_targets.clone(), logits=local_logits, metric=self.pil_metric)
        local_ats_targets = get_ats_targets(labels=local_pil_targets.clone(), preds=local_preds, speaker_permutations=self.speaker_permutations)

        #logging.info(f"local_targets: {local_targets.to(int).sum(dim=1)}")
        #logging.info(f"local_preds: {(local_preds > 0.5).to(int).sum(dim=1)}")
        #logging.info(f"local_pil_targets: {local_pil_targets.to(int).sum(dim=1)}")
        #logging.info(f"local_ats_targets: {local_ats_targets.to(int).sum(dim=1)}")
        #logging.info(f"local_target_indices: {local_target_indices}")
        #logging.info(f"local_target_lens: {local_target_lens}")

        total_logits_op = torch.full(
            (batch_size, total_n_frames, targets_num_spks),
            -1e9,
            dtype=local_logits.dtype,
            device=local_logits.device
        )

        lc=self.nextformer_modules.chunk_left_context
        chunk_len=self.nextformer_modules.chunk_len
        num_chunks = local_targets.shape[0] // batch_size
        local_num_spks = local_logits.shape[2]

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_len
            end = min(start + chunk_len, total_n_frames)
            dur = end - start
            #logging.info(f"chunk_idx: {chunk_idx}, start: {start}, end: {end}, dur: {dur}, lc: {lc}, chunk_len: {chunk_len}")
            offset = min(lc, start)
            local_logits_chunk = local_logits[chunk_idx * batch_size:(chunk_idx + 1) * batch_size, :, :] # (batch_size, lc+chunk_len+rc, local_num_spks)
            global_spk_indices = local_target_indices[chunk_idx * batch_size:(chunk_idx + 1) * batch_size, :]  # (batch_size, local_num_spks)
            
            # Vectorized version: create mask for valid mappings
            valid_mask = global_spk_indices != -1  # (batch_size, local_num_spks)
            
            if valid_mask.any():
                # Get indices of valid (batch, local_speaker) pairs
                batch_indices, local_spk_indices = torch.where(valid_mask)  # 1D tensors of length num_valid
                global_spk_idx_flat = global_spk_indices[batch_indices, local_spk_indices]  # (num_valid,)
                num_valid = len(batch_indices)
                
                # Extract source slices: local_logits_chunk[b, offset:offset+dur, j] for each valid (b, j)
                # First, extract the time slices for all batches: (batch_size, dur, local_num_spks)
                time_slice = local_logits_chunk[:, offset:offset+dur, :]  # (batch_size, dur, local_num_spks)
                
                # Extract slices for valid batch indices: (num_valid, dur, local_num_spks)
                time_slice_valid = time_slice[batch_indices]  # (num_valid, dur, local_num_spks)
                
                # Extract the specific local speaker for each valid (batch, speaker) pair
                # Use advanced indexing: for each batch i, select speaker local_spk_indices[i] across all time steps
                # Shape: (num_valid, dur)
                batch_idx_tensor = torch.arange(num_valid, device=local_logits.device)  # (num_valid,)
                source_slices = time_slice_valid[batch_idx_tensor, :, local_spk_indices]  # (num_valid, dur)
                
                # Create index arrays for assignment to total_logits_op[b, start:end, global_index]
                # We need to assign for each (batch, time, global_speaker) combination
                time_indices = torch.arange(start, end, device=local_logits.device)  # (dur,)
                
                # Expand indices: for each valid (batch, speaker) pair, assign to all time steps
                # batch_indices_expanded: (dur, num_valid) - repeat batch indices for each time step
                # time_indices_expanded: (dur, num_valid) - repeat time indices for each valid pair
                # global_spk_idx_expanded: (dur, num_valid) - repeat global speaker indices for each time step
                batch_indices_expanded = batch_indices.unsqueeze(0).expand(dur, -1)  # (dur, num_valid)
                time_indices_expanded = time_indices.unsqueeze(1).expand(-1, num_valid)  # (dur, num_valid)
                global_spk_idx_expanded = global_spk_idx_flat.unsqueeze(0).expand(dur, -1)  # (dur, num_valid)
                
                # Flatten all tensors for vectorized assignment
                batch_flat = batch_indices_expanded.flatten()  # (dur * num_valid,)
                time_flat = time_indices_expanded.flatten()  # (dur * num_valid,)
                global_spk_flat = global_spk_idx_expanded.flatten()  # (dur * num_valid,)
                source_flat = source_slices.transpose(0, 1).flatten()  # (dur * num_valid,) - transpose to match time-first order
                
                # Assign using advanced indexing for efficient vectorized assignment
                total_logits_op[batch_flat, time_flat, global_spk_flat] = source_flat
        
        return local_pil_targets, local_ats_targets, local_target_lens, local_target_indices, total_logits_op

    def _get_aux_train_evaluations(
        self, logits, local_logits, local_queries, active_frames_per_query, targets, target_lens
    ) -> dict:
        """
        Compute auxiliary training evaluations including losses and metrics.

        This function calculates various losses and metrics for the training process,
        including Arrival Time Sort (ATS) Loss and Permutation Invariant Loss (PIL)
        based evaluations.

        Args:
            logits (torch.Tensor): Predicted speaker labels for the entire audio.
                Shape: (batch_size, total_n_frames, local_num_spks)
            local_logits (torch.Tensor): Speaker logits for the entire audio.
                Shape: (batch_size * num_chunks, lc+chunk_len+rc, local_num_spks)
            local_queries (torch.Tensor): Local speaker queries.
                Shape: (batch_size * num_chunks, local_num_spks, emb_dim)
            active_frames_per_query (torch.Tensor): Tensor containing the number of active frames per query
                Shape: (num_chunks * batch_size, local_num_spks)
            targets (torch.Tensor): Ground truth speaker labels.
                Shape: (batch_size, total_n_frames, max_num_spks)
            target_lens (torch.Tensor): Lengths of target sequences.
                Shape: (batch_size,)

        Returns:
            (dict): A dictionary containing the following training metrics.
        """
        preds = torch.sigmoid(logits)
        targets = targets.to(preds.dtype)
        if preds.shape[1] < targets.shape[1]:
            logging.info(
                f"WARNING! preds has less frames than targets ({preds.shape[1]} < {targets.shape[1]}). "
                "Truncating targets and clamping target_lens."
            )
            targets = targets[:, : preds.shape[1], :]
            target_lens = target_lens.clamp(max=preds.shape[1])

        # get global PIL targets using Hungarian algorithm
        targets_pil, _ = get_pil_targets_hungarian(labels=targets.clone(), logits=logits, metric=self.pil_metric)
        self._accuracy_train_global(preds, targets_pil, target_lens)
        train_f1_acc_global, _, _ = self._accuracy_train_global.compute()
      
        global_loss_scale = logits.shape[-1] / local_logits.shape[-1]
        global_pil_loss = global_loss_scale * self.loss(logits=logits, labels=targets_pil, target_lens=target_lens)

        local_pil_targets, local_ats_targets, local_target_lens, local_target_indices, total_logits_op = self._process_logits_and_targets(
            local_logits, targets, target_lens
        )
        
        preds_op = torch.sigmoid(total_logits_op)
        self._accuracy_train_global_op(preds_op, targets, target_lens)
        train_f1_acc_global_op, _, _ = self._accuracy_train_global_op.compute()
        
        pil_loss = self.loss(logits=local_logits, labels=local_pil_targets, target_lens=local_target_lens)
        ats_loss = self.loss(logits=local_logits, labels=local_ats_targets, target_lens=local_target_lens)

        loss = (
            self.ats_weight * ats_loss
            + self.pil_weight * pil_loss
            + self.global_pil_weight * global_pil_loss
        )

        local_preds = torch.sigmoid(local_logits)
        self._accuracy_train(local_preds, local_pil_targets, local_target_lens)
        train_f1_acc, train_precision, train_recall = self._accuracy_train.compute()

        self._accuracy_train_ats(local_preds, local_ats_targets, local_target_lens)
        train_f1_acc_ats, _, _ = self._accuracy_train_ats.compute()

        train_metrics = {
            'loss': loss,
            'ats_loss': ats_loss,
            'pil_loss': pil_loss,
            'global_pil_loss': global_pil_loss,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'train_f1_acc': train_f1_acc,
            'train_f1_acc_global': train_f1_acc_global,
            'train_f1_acc_global_op': train_f1_acc_global_op,
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
        logits, local_logits, local_queries, active_frames_per_query = self.forward(
            audio_signal=audio_signal, audio_signal_length=audio_signal_length, targets=targets
        )
        train_metrics = self._get_aux_train_evaluations(
            logits, local_logits, local_queries, active_frames_per_query, targets, target_lens
        )
        logging.info(f"dustbin parameter value: {self.nextformer_modules.sinkhorn_dustbin_val.item()}")
        self._reset_train_metrics()
        self.log_dict(train_metrics, sync_dist=True, on_step=True, on_epoch=False, logger=True)
        return {'loss': train_metrics['loss']}

    def _get_aux_validation_evaluations(
        self, logits, local_logits, local_queries, active_frames_per_query, targets, target_lens
    ) -> dict:
        """
        Compute auxiliary validation evaluations including losses and metrics.

        This function calculates various losses and metrics for the training process,
        including Arrival Time Sort (ATS) Loss and Permutation Invariant Loss (PIL)
        based evaluations.

        Args:
            logits (torch.Tensor): Predicted speaker labels for the entire audio.
                Shape: (batch_size, total_n_frames, local_num_spks)
            local_logits (torch.Tensor): Speaker logits for the entire audio.
                Shape: (batch_size * num_chunks, lc+chunk_len+rc, local_num_spks)
            local_queries (torch.Tensor): Local speaker queries.
                Shape: (batch_size * num_chunks, local_num_spks, emb_dim)
            active_frames_per_query (torch.Tensor): Tensor containing the number of active frames per query
                Shape: (num_chunks * batch_size, local_num_spks)
            targets (torch.Tensor): Ground truth speaker labels.
                Shape: (batch_size, total_n_frames, max_num_spks)
            target_lens (torch.Tensor): Lengths of target sequences.
                Shape: (batch_size,)

        Returns:
            val_metrics (dict): A dictionary containing the following validation metrics
        """
        preds = torch.sigmoid(logits)
        targets = targets.to(preds.dtype)
        if preds.shape[1] < targets.shape[1]:
            logging.info(
                f"WARNING! preds has less frames than targets ({preds.shape[1]} < {targets.shape[1]}). "
                "Truncating targets and clamping target_lens."
            )
            targets = targets[:, : preds.shape[1], :]
            target_lens = target_lens.clamp(max=preds.shape[1])

        #Global PIL targets using Hungarian
        targets_pil, _ = get_pil_targets_hungarian(labels=targets.clone(), logits=logits, metric=self.pil_metric)
        self._accuracy_valid_global(preds, targets_pil, target_lens)
        val_f1_acc_global, _, _ = self._accuracy_valid_global.compute()

        global_loss_scale = logits.shape[-1] / local_logits.shape[-1]
        val_global_pil_loss = global_loss_scale * self.loss(logits=logits, labels=targets_pil, target_lens=target_lens)

        local_pil_targets, local_ats_targets, local_target_lens, local_target_indices, total_logits_op = self._process_logits_and_targets(
            local_logits, targets, target_lens
        )

        preds_op = torch.sigmoid(total_logits_op)
        self._accuracy_valid_global_op(preds_op, targets, target_lens)
        val_f1_acc_global_op, _, _ = self._accuracy_valid_global_op.compute()

        val_pil_loss = self.loss(logits=local_logits, labels=local_pil_targets, target_lens=local_target_lens)
        val_ats_loss = self.loss(logits=local_logits, labels=local_ats_targets, target_lens=local_target_lens)

        val_loss = (
            self.ats_weight * val_ats_loss
            + self.pil_weight * val_pil_loss
            + self.global_pil_weight * val_global_pil_loss
        )

        local_preds = torch.sigmoid(local_logits)
        self._accuracy_valid(local_preds, local_pil_targets, local_target_lens)
        val_f1_acc, val_precision, val_recall = self._accuracy_valid.compute()

        self._accuracy_valid_ats(local_preds, local_ats_targets, local_target_lens)
        val_f1_acc_ats, _, _ = self._accuracy_valid_ats.compute()

        self._accuracy_valid.reset()
        self._accuracy_valid_ats.reset()
        self._accuracy_valid_global.reset()
        self._accuracy_valid_global_op.reset()

        val_metrics = {
            'val_loss': val_loss,
            'val_ats_loss': val_ats_loss,
            'val_pil_loss': val_pil_loss,
            'val_global_pil_loss': val_global_pil_loss,
            'val_f1_acc': val_f1_acc,
            'val_f1_acc_global': val_f1_acc_global,
            'val_f1_acc_global_op': val_f1_acc_global_op,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1_acc_ats': val_f1_acc_ats,
        }
        return val_metrics

    def validation_step(self, batch: list, batch_idx: int, dataloader_idx: int = 0) -> dict:
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
        logits, local_logits, local_queries, active_frames_per_query = self.forward(
            audio_signal=audio_signal, audio_signal_length=audio_signal_length, targets=targets
        )
        val_metrics = self._get_aux_validation_evaluations(
            logits, local_logits, local_queries, active_frames_per_query, targets, target_lens
        )
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
        val_global_pil_loss_mean = torch.stack([x['val_global_pil_loss'] for x in outputs]).mean()
        val_f1_acc_mean = torch.stack([x['val_f1_acc'] for x in outputs]).mean()
        val_f1_acc_global_mean = torch.stack([x['val_f1_acc_global'] for x in outputs]).mean()
        val_f1_acc_global_op_mean = torch.stack([x['val_f1_acc_global_op'] for x in outputs]).mean()
        val_precision_mean = torch.stack([x['val_precision'] for x in outputs]).mean()
        val_recall_mean = torch.stack([x['val_recall'] for x in outputs]).mean()
        val_f1_acc_ats_mean = torch.stack([x['val_f1_acc_ats'] for x in outputs]).mean()

        self._reset_valid_metrics()

        multi_val_metrics = {
            'val_loss': val_loss_mean,
            'val_ats_loss': val_ats_loss_mean,
            'val_pil_loss': val_pil_loss_mean,
            'val_global_pil_loss': val_global_pil_loss_mean,
            'val_f1_acc': val_f1_acc_mean,
            'val_f1_acc_global': val_f1_acc_global_mean,
            'val_f1_acc_global_op': val_f1_acc_global_op_mean,
            'val_precision': val_precision_mean,
            'val_recall': val_recall_mean,
            'val_f1_acc_ats': val_f1_acc_ats_mean,
        }
        return {'log': multi_val_metrics}

    def _get_aux_test_batch_evaluations(
        self, batch_idx: int, logits, local_logits, local_queries, active_frames_per_query, targets, target_lens
    ):
        preds = torch.sigmoid(logits)
        targets = targets.to(preds.dtype)
        if preds.shape[1] < targets.shape[1]:
            logging.info(
                f"WARNING! preds has less frames than targets ({preds.shape[1]} < {targets.shape[1]}). "
                "Truncating targets and clamping target_lens."
            )
            targets = targets[:, : preds.shape[1], :]
            target_lens = target_lens.clamp(max=preds.shape[1])

        # get global f1 accuracy
        targets_pil, _ = get_pil_targets_hungarian(labels=targets.clone(), logits=logits, metric=self.pil_metric)
        self._accuracy_test(preds, targets_pil, target_lens)
        f1_acc, precision, recall = self._accuracy_test.compute()
        self.batch_f1_accs_list.append(f1_acc)
        logging.info(f"batch {batch_idx}: f1_acc={f1_acc}, precision={precision}, recall={recall}")

        local_pil_targets, local_ats_targets, local_target_lens, local_target_indices, total_logits_op = self._process_logits_and_targets(
            local_logits, targets, target_lens
        )

        # get global optimally-permuted f1 accuracy (upper bound)
        preds_op = torch.sigmoid(total_logits_op)
        self._accuracy_test_op(preds_op, targets, target_lens)
        f1_acc_op, precision_op, recall_op = self._accuracy_test_op.compute()
        self.batch_f1_accs_op_list.append(f1_acc_op)
        logging.info(f"batch {batch_idx}: f1_acc_op={f1_acc_op}, precision_op={precision_op}, recall_op={recall_op}")

        # get local f1 accuracy
        local_preds = torch.sigmoid(local_logits)
        self._accuracy_test_local(local_preds, local_pil_targets, local_target_lens)
        f1_acc_local, precision_local, recall_local = self._accuracy_test_local.compute()
        self.batch_f1_accs_local_list.append(f1_acc_local)
        logging.info(f"batch {batch_idx}: f1_acc_local={f1_acc_local}, precision_local={precision_local}, recall_local={recall_local}")

        self._accuracy_test_local_ats(local_preds, local_ats_targets, local_target_lens)
        f1_acc_local_ats, precision_local_ats, recall_local_ats = self._accuracy_test_local_ats.compute()
        self.batch_f1_accs_local_ats_list.append(f1_acc_local_ats)
        logging.info(f"batch {batch_idx}: f1_acc_local_ats={f1_acc_local_ats}, precision_local_ats={precision_local_ats}, recall_local_ats={recall_local_ats}")

        self._accuracy_test.reset()
        self._accuracy_test_op.reset()
        self._accuracy_test_local.reset()
        self._accuracy_test_local_ats.reset()

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
            self.batch_f1_accs_op_list,
            self.batch_f1_accs_local_list,
            self.batch_f1_accs_local_ats_list,
        ) = ([], [], [], [], [])

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self._test_dl)):
                audio_signal, audio_signal_length, targets, target_lens = batch
                audio_signal = audio_signal.to(self.device)
                audio_signal_length = audio_signal_length.to(self.device)
                targets = targets.to(self.device)
                logits, local_logits, local_queries, active_frames_per_query = self.forward(
                    audio_signal=audio_signal, audio_signal_length=audio_signal_length, targets=targets
                )
                self._get_aux_test_batch_evaluations(
                    batch_idx, logits, local_logits, local_queries, active_frames_per_query, targets, target_lens
                )
                preds = torch.sigmoid(logits).detach().to('cpu')
                if preds.shape[0] == 1:  # batch size = 1
                    self.preds_total_list.append(preds)
                else:
                    self.preds_total_list.extend(torch.split(preds, [1] * preds.shape[0]))
                torch.cuda.empty_cache()

        logging.info(f"Batch F1Acc. MEAN: {torch.mean(torch.tensor(self.batch_f1_accs_list))}")
        logging.info(f"Batch OP-F1Acc. MEAN: {torch.mean(torch.tensor(self.batch_f1_accs_op_list))}")
        logging.info(f"Batch Local F1Acc. MEAN: {torch.mean(torch.tensor(self.batch_f1_accs_local_list))}")
        logging.info(f"Batch Local F1Acc. ATS MEAN: {torch.mean(torch.tensor(self.batch_f1_accs_local_ats_list))}")

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