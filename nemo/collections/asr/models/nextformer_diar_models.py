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
import torch.nn as nn

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm

from nemo.collections.asr.data.audio_to_diar_label import (
    AudioToSpeechE2ESpkDiarDataset,
    extract_global_speaker_ids,
)
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
from nemo.collections.asr.parts.utils.offline_clustering import SpeakerClustering
from nemo.utils import logging

__all__ = ['NextformerEncLabelModel']


class _OversamplingDistributedSampler(torch.utils.data.DistributedSampler):
    """DistributedSampler that cycles through the dataset to guarantee a
    minimum number of samples per GPU per epoch.  Because it inherits from
    DistributedSampler, PyTorch Lightning will *not* replace it with its own
    sampler in DDP mode.

    Args:
        num_samples_per_epoch: desired total samples across ALL GPUs.
            Each GPU will yield ``num_samples_per_epoch // num_replicas``
            samples, cycling through the dataset as many times as needed.
    """

    def __init__(self, dataset, *, num_samples_per_epoch: int, **kwargs):
        super().__init__(dataset, **kwargs)
        self._target = max(math.ceil(num_samples_per_epoch / self.num_replicas), 1)

    def __iter__(self):
        base = list(super().__iter__())
        if len(base) == 0:
            raise ValueError(
                "OversamplingDistributedSampler received no indices from DistributedSampler. "
                "This usually means the training dataset is empty."
            )
        if len(base) >= self._target:
            return iter(base[: self._target])
        result = []
        cycle = 0
        while len(result) < self._target:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch * 100_000 + cycle)
            perm = torch.randperm(len(base), generator=g).tolist()
            result.extend(base[i] for i in perm)
            cycle += 1
        return iter(result[: self._target])

    def __len__(self):
        return self._target


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

        # Set before super().__init__() because setup_training_data() is called during super init
        self.min_speaker_duration_sec = self._cfg.get("min_speaker_duration_sec", 0.0)

        super().__init__(cfg=self._cfg, trainer=trainer)
        self.preprocessor = NextformerEncLabelModel.from_config_dict(self._cfg.preprocessor)

        if (
            hasattr(self._cfg, 'spec_augment')
            and self._cfg.spec_augment is not None
            and self._cfg.spec_augment.get('freq_masks', 0) + self._cfg.spec_augment.get('time_masks', 0) > 0
        ):
            self.spec_augmentation = NextformerEncLabelModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

        self.encoder = NextformerEncLabelModel.from_config_dict(self._cfg.encoder).to(self.device)
        
        if not hasattr(self.encoder, 'num_cls_tokens') or self.encoder.num_cls_tokens < self._cfg.local_num_spks:
            raise ValueError("CLS-based encoder must be specified in the config and have at least local_num_spks CLS tokens")

        logging.info(
            f"Using CLS-based speaker embeddings from encoder (num_cls_tokens={self.encoder.num_cls_tokens}, "
            f"cls_pos_mode='{self.encoder.cls_pos_mode}')"
        )

        spk_embs_decoder_cfg = self._cfg.get('spk_embs_decoder', None)
        if spk_embs_decoder_cfg is not None and spk_embs_decoder_cfg.get('num_layers', 0) > 0:
            self.spk_embs_decoder = NextformerEncLabelModel.from_config_dict(spk_embs_decoder_cfg).to(self.device)
        else:
            self.spk_embs_decoder = None

        spk_embs_enhancer_cfg = self._cfg.get('spk_embs_enhancer', None)
        if spk_embs_enhancer_cfg is not None and spk_embs_enhancer_cfg.get('num_layers', 0) > 0:
            self.spk_embs_enhancer = NextformerEncLabelModel.from_config_dict(spk_embs_enhancer_cfg).to(self.device)
        else:
            self.spk_embs_enhancer = None

        profile_updater_cfg = self._cfg.get('profile_updater', None)
        if profile_updater_cfg is not None and profile_updater_cfg.get('num_layers', 0) > 0:
            self.profile_updater = NextformerEncLabelModel.from_config_dict(profile_updater_cfg).to(self.device)
        else:
            self.profile_updater = None
        self.profile_updater_detach_inputs = self._cfg.get('profile_updater_detach_inputs', False)

        self.nextformer_modules = NextformerEncLabelModel.from_config_dict(self._cfg.nextformer_modules).to(
            self.device
        )
       
        self._init_loss_weights()

        self.eps = 1e-3
        self.negative_init_val = -99
        self.loss = instantiate(self._cfg.loss)
        self.q_sim_loss = instantiate(self._cfg.q_sim_loss)
        self.emb_sim_loss = instantiate(self._cfg.emb_sim_loss)
        self.local_mask_threshold = self._cfg.get("local_mask_threshold", 0.5)
        self.pil_metric = self._cfg.get("pil_metric", "bce")
        self.oracle_assignment = self._cfg.get("oracle_assignment", False)
        self.oracle_centroids_train = self._cfg.get("oracle_centroids_train", False)
        self.oracle_centroids_test = self._cfg.get("oracle_centroids_test", False)
        # Interpolation weight for oracle centroids: 0.0 = pure Sinkhorn, 1.0 = pure oracle.
        self.oracle_centroids_weight = self._cfg.get("oracle_centroids_weight", 1.0)
        self.streaming_mode = self.cfg.get("streaming_mode", False)

        self.clustering_assignment = self._cfg.get("clustering_assignment", False)
        self.clustering_method = self._cfg.get("clustering_method", "nmesc")  # "nmesc" or "ahc"
        self.clustering_threshold = self._cfg.get("clustering_threshold", -1.0)
        if self.clustering_assignment:
            use_cuda = torch.cuda.is_available()
            self.speaker_clustering = SpeakerClustering(cuda=use_cuda)
        else:
            self.speaker_clustering = None

        # Cross-chunk speaker embedding swap augmentation
        self.cross_chunk_swap_p = self._cfg.get("cross_chunk_swap_p", 0.0)
        self.cross_chunk_swap_detach = self._cfg.get("cross_chunk_swap_detach", True)
        self.cross_chunk_swap_min_frames = self._cfg.get("cross_chunk_swap_min_frames", 0)

        # SupCon auxiliary loss on speaker embeddings
        self.supcon_weight = self._cfg.get("supcon_weight", 0.0)
        self.supcon_min_active_frames = self._cfg.get("supcon_min_active_frames", 5)
        self.supcon_cross_batch = self._cfg.get("supcon_cross_batch", True)
        self.supcon_aam = self._cfg.get("supcon_aam", 0.0)
        self.supcon_temperature = self._cfg.get("supcon_temperature", 0.1)
        self.supcon_decoupled = self._cfg.get("supcon_decoupled", False)
        self.supcon_dustbin_margin = self._cfg.get("supcon_dustbin_margin", 0.0)

        # AAM-Softmax loss on speaker embeddings
        self.aam_weight = self._cfg.get("aam_weight", 0.0)
        self.aam_scale = self._cfg.get("aam_scale", 30.0)
        self.aam_margin = self._cfg.get("aam_margin", 0.2)
        self.aam_min_active_frames = self._cfg.get("aam_min_active_frames", 10)
        self.aam_min_confidence = self._cfg.get("aam_min_confidence", 0.0)

        # Profile update mode: "frame" (legacy frame-embedding average) or "cls" (CLS-embedding average)
        self.profile_update_mode = self._cfg.get("profile_update_mode", "cls")

        # Backend and fusion settings come from NextformerModules config
        logging.info(f"Using backend: {self.nextformer_modules.backend}")

        # Optional Transformer backend modules
        if self.nextformer_modules.backend == "trff":
            if not hasattr(self._cfg, "transformer_encoder"):
                raise ValueError("transformer backend requires 'transformer_encoder' config")
            self.transformer_encoder = NextformerEncLabelModel.from_config_dict(self._cfg.transformer_encoder).to(self.device)
        else:
            self.transformer_encoder = None

        # Optional ISD backend modules
        if self.nextformer_modules.backend == "isd":
            if not hasattr(self._cfg, "isd_encoder"):
                raise ValueError("isd backend requires 'isd_encoder' config")
            self.isd_encoder = NextformerEncLabelModel.from_config_dict(self._cfg.isd_encoder).to(self.device)
        else:
            self.isd_encoder = None

        # Optional JSD (Joint Speaker Detection) backend modules
        if self.nextformer_modules.backend == "jsd":
            if not hasattr(self._cfg, "jsd_encoder"):
                raise ValueError("jsd backend requires 'jsd_encoder' config")
            self.jsd_encoder = NextformerEncLabelModel.from_config_dict(self._cfg.jsd_encoder).to(self.device)
        else:
            self.jsd_encoder = None

        # Log ISD/JSD backend configuration (fusion layers are in nextformer_modules)
        if self.nextformer_modules.backend in ["isd", "jsd"]:
            logging.info(
                f"{self.nextformer_modules.backend.upper()} backend initialized with fusion_type: "
                f"{self.nextformer_modules.fusion_type}"
            )
        # Initialize AAM-Softmax head if speaker vocabulary is available
        if hasattr(self, 'num_speaker_classes') and self.num_speaker_classes > 0:
            self.nextformer_modules.init_aam_head(self.num_speaker_classes)
        elif self.aam_weight > 0:
            logging.warning(
                "aam_weight > 0 but speaker vocabulary is not available at init time. "
                "AAM-Softmax head was NOT initialized and the loss will be disabled. "
                "This can happen with Lhotse dataloaders or deferred data setup."
            )

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
            subsegment_margin_frames=config.get('subsegment_margin_frames', 0),
        )

        self.data_collection = dataset.collection
        self.collate_ds = dataset

        if not hasattr(self, 'speaker_to_id'):
            self.speaker_to_id = extract_global_speaker_ids(
                dataset.collection,
                min_speaker_duration_sec=self.min_speaker_duration_sec,
            )
            self.num_speaker_classes = len(self.speaker_to_id)

        sampler = None
        shuffle = config.get('shuffle', False)
        num_samples = config.get('num_samples_per_epoch', 0)
        if num_samples > 0:
            sampler = _OversamplingDistributedSampler(
                dataset,
                num_samples_per_epoch=num_samples,
                num_replicas=self.world_size,
                rank=global_rank,
                shuffle=shuffle,
            )
            shuffle = False

        dataloader_instance = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            collate_fn=self.collate_ds.eesd_train_collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=config.get('num_workers', 1),
            pin_memory=config.get('pin_memory', False),
        )
        return dataloader_instance

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        self._train_dl = self.__setup_dataloader_from_config(
            config=train_data_config,
        )

    def _save_speaker_vocab(self):
        """Save speaker_to_id mapping as JSON to the trainer's log directory."""
        if not hasattr(self, 'speaker_to_id') or self.speaker_to_id is None:
            return
        log_dir = getattr(self.trainer, 'log_dir', None)
        if log_dir is None:
            return
        if not (hasattr(self, 'global_rank') and self.global_rank == 0):
            if not (hasattr(self, '_trainer') and hasattr(self._trainer, 'global_rank') and self._trainer.global_rank == 0):
                return
        import json
        save_path = os.path.join(log_dir, 'speaker_to_id.json')
        with open(save_path, 'w') as f:
            json.dump(self.speaker_to_id, f, indent=2)
        logging.info(
            f"Saved speaker_to_id mapping ({len(self.speaker_to_id)} speakers) to {save_path}"
        )

    def _speaker_names_to_ids(self, speaker_names_batch):
        """
        Convert a batch of speaker name lists to a global-speaker-ID tensor.

        Args:
            speaker_names_batch (list[list[str|None]]): Length B, each inner
                list has length max_spks.  Entries are RTTM speaker name
                strings or None for unused slots.

        Returns:
            global_speaker_ids (torch.Tensor): Shape (B, max_spks), dtype long,
                on the current model device.  Each entry is the global integer
                ID for that speaker, or -1 if the slot is unused or the speaker
                is not in the vocabulary.
        """
        speaker_to_id = getattr(self, 'speaker_to_id', None)
        B = len(speaker_names_batch)
        max_spks = len(speaker_names_batch[0]) if B > 0 else 0
        ids = torch.full((B, max_spks), -1, dtype=torch.long, device=self.device)
        if speaker_to_id is None:
            return ids
        for b, names in enumerate(speaker_names_batch):
            for s, name in enumerate(names):
                if name is not None:
                    ids[b, s] = speaker_to_id.get(name, -1)
        return ids

    def on_train_start(self):
        super().on_train_start()
        self._save_speaker_vocab()

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
                tensor containing encoder outputs (B, T, D).
            emb_seq_length (torch.Tensor):
                tensor containing lengths of encoder outputs.
            cls_embs (torch.Tensor or None):
                CLS token embeddings if using CLS-based encoder, None otherwise.
                Shape: (B, num_cls_tokens, D)
        """      
        # CLS-based encoder returns (audio_emb, length, cls_emb)
        emb_seq, emb_seq_length, cls_embs = self.encoder(
            audio_signal=processed_signal,
            length=processed_signal_length,
            bypass_pre_encode=bypass_pre_encode,
        )
        emb_seq = emb_seq.transpose(1, 2)  # (B, D, T) -> (B, T, D)
        return emb_seq, emb_seq_length, cls_embs

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
            spk_embs (torch.Tensor): Tensor containing local speaker embeddings.
                Shape: (num_chunks * batch_size, local_num_spks, spk_emb_dim)
            active_frames_per_spk (torch.Tensor): Tensor containing the number of active frames per speaker
                Shape: (num_chunks * batch_size, local_num_spks)
        """
        processed_signal, processed_signal_length = self.process_signal(
            audio_signal=audio_signal, audio_signal_length=audio_signal_length
        )
        processed_signal = processed_signal[:, :, : processed_signal_length.max()]
        if self.streaming_mode:
            raise NotImplementedError("Streaming mode is not implemented for Nextformer model.")
        else:
            logits, local_logits, spk_embs, active_frames_per_spk = self.forward_offline(
                processed_signal=processed_signal, processed_signal_length=processed_signal_length, targets=targets
            )
            return logits, local_logits, spk_embs, active_frames_per_spk

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

    def _create_swapped_embeddings(
        self,
        spk_embs: torch.Tensor,
        local_target_indices: torch.Tensor,
        active_frames_per_spk: torch.Tensor,
        batch_size: int,
        num_chunks: int,
        p_swap: float = 0.5,
        detach: bool = True,
        min_src_frames: int = 0,
    ) -> torch.Tensor:
        """
        Create cross-chunk swapped speaker embeddings for augmentation.

        For each active local speaker, with probability p_swap, replace its embedding
        with the embedding of the same global speaker from a randomly selected different chunk.
        This breaks the positional shortcut where CLS tokens encode speaker arrival order
        rather than speaker identity.

        Args:
            spk_embs (torch.Tensor): Speaker embeddings for all chunks.
                Shape: (num_chunks * batch_size, local_num_spks, emb_dim)
            local_target_indices (torch.Tensor): Global speaker index for each local speaker.
                Shape: (num_chunks * batch_size, local_num_spks)
                Value of -1 means unmatched/inactive.
            active_frames_per_spk (torch.Tensor): Number of active frames per local speaker.
                Shape: (num_chunks * batch_size, local_num_spks)
                Used to filter out unreliable source embeddings.
            batch_size (int): Batch size.
            num_chunks (int): Number of chunks.
            p_swap (float): Probability of swapping each embedding. Default: 0.5.
            detach (bool): Whether to detach swapped embeddings from the computation graph.
                If True, only the backend receives gradients (regularizer mode).
                If False, the source embedding also receives gradients. Default: True.
            min_src_frames (int): Minimum number of active frames a source speaker must
                have to be eligible as a swap source. Embeddings from speakers with fewer
                active frames are unreliable and can destabilize training. Default: 0.

        Returns:
            swapped_embs (torch.Tensor): Speaker embeddings with some entries swapped.
                Shape: (num_chunks * batch_size, local_num_spks, emb_dim)
        """
        local_num_spks = spk_embs.shape[1]
        emb_dim = spk_embs.shape[2]

        # Clone to avoid modifying original
        swapped_embs = spk_embs.clone()

        # Reshape for easier indexing: (num_chunks, batch_size, local_num_spks, ...)
        indices_view = local_target_indices.view(num_chunks, batch_size, local_num_spks)
        embs_view = spk_embs.view(num_chunks, batch_size, local_num_spks, emb_dim)
        swapped_view = swapped_embs.view(num_chunks, batch_size, local_num_spks, emb_dim)
        af_view = active_frames_per_spk.view(num_chunks, batch_size, local_num_spks)

        for b in range(batch_size):
            # Build lookup: global_spk_id -> [(chunk_idx, local_slot_idx, active_frames)]
            spk_to_locations: dict = {}
            for c in range(num_chunks):
                for s in range(local_num_spks):
                    g = indices_view[c, b, s].item()
                    if g < 0:
                        continue
                    af = af_view[c, b, s].item()
                    if g not in spk_to_locations:
                        spk_to_locations[g] = []
                    spk_to_locations[g].append((c, s, af))

            # For each location, potentially swap
            for c in range(num_chunks):
                for s in range(local_num_spks):
                    g = indices_view[c, b, s].item()
                    if g < 0:
                        continue

                    if random.random() >= p_swap:
                        continue

                    # Find alternative locations for the same global speaker
                    # that have enough active frames to be a reliable source
                    other_locations = [
                        (cc, ss) for (cc, ss, af) in spk_to_locations[g]
                        if cc != c and af >= min_src_frames
                    ]
                    if not other_locations:
                        continue

                    src_c, src_s = random.choice(other_locations)
                    src_emb = embs_view[src_c, b, src_s, :]
                    if detach:
                        src_emb = src_emb.detach()
                    swapped_view[c, b, s, :] = src_emb

        return swapped_embs

    @staticmethod
    def _constrained_ahc(
        cos_sim: torch.Tensor,
        cannot_link: set,
        threshold: float,
    ) -> torch.LongTensor:
        """Constrained Agglomerative Hierarchical Clustering (average linkage).

        Merges the closest pair of clusters whose merge does not violate any
        cannot-link constraint, stopping when the minimum valid distance
        exceeds *threshold*.

        Args:
            cos_sim: (N, N) cosine similarity matrix.
            cannot_link: set of frozenset({i, j}) pairs that must stay apart.
            threshold: maximum cosine *distance* (1 - similarity) for merging.

        Returns:
            labels: (N,) contiguous integer cluster labels on the same device
                as *cos_sim*.
        """
        device = cos_sim.device
        distance = (1.0 - cos_sim).float().cpu().numpy()
        N = distance.shape[0]

        clusters: dict = {i: [i] for i in range(N)}
        point_to_cluster = list(range(N))
        active = set(range(N))

        while len(active) > 1:
            best_dist = float("inf")
            best_pair = None

            active_list = sorted(active)
            for ii in range(len(active_list)):
                ci = active_list[ii]
                for jj in range(ii + 1, len(active_list)):
                    cj = active_list[jj]
                    # Check cannot-link
                    violates = False
                    for pi in clusters[ci]:
                        for pj in clusters[cj]:
                            if frozenset({pi, pj}) in cannot_link:
                                violates = True
                                break
                        if violates:
                            break
                    if violates:
                        continue
                    # Average linkage distance
                    total = sum(
                        distance[pi, pj]
                        for pi in clusters[ci]
                        for pj in clusters[cj]
                    )
                    d = total / (len(clusters[ci]) * len(clusters[cj]))
                    if d < best_dist:
                        best_dist = d
                        best_pair = (ci, cj)

            if best_pair is None or best_dist > threshold:
                break

            ci, cj = best_pair
            clusters[ci] = clusters[ci] + clusters[cj]
            for p in clusters[cj]:
                point_to_cluster[p] = ci
            del clusters[cj]
            active.discard(cj)

        label_map = {cid: label for label, cid in enumerate(sorted(active))}
        labels = [label_map[point_to_cluster[i]] for i in range(N)]
        return torch.tensor(labels, device=device, dtype=torch.long)

    def _compute_clustering_assignments(
        self,
        spk_embs: torch.Tensor,
        spk_detected: torch.Tensor,
        batch_size: int,
        num_chunks: int,
        local_num_spks: int,
    ) -> torch.Tensor:
        """Cluster all local speaker embeddings across chunks to produce
        global assignment matrices.

        Args:
            spk_embs: (num_chunks * batch_size, local_num_spks, emb_dim)
            spk_detected: (num_chunks * batch_size, local_num_spks) bool
            batch_size: batch size
            num_chunks: number of chunks
            local_num_spks: number of local speakers per chunk

        Returns:
            assignments: (num_chunks * batch_size, local_num_spks, max_num_spks)
                one-hot assignment matrices.
        """
        emb_dim = spk_embs.shape[-1]
        max_num_spks = self.nextformer_modules.max_num_spks
        N = num_chunks * local_num_spks

        # Reshape to (B, num_chunks * local_num_spks, emb_dim)
        spk_embs_flat = (
            spk_embs.view(num_chunks, batch_size, local_num_spks, emb_dim)
            .transpose(0, 1)
            .reshape(batch_size, N, emb_dim)
        )
        valid_mask = (
            spk_detected.view(num_chunks, batch_size, local_num_spks)
            .transpose(0, 1)
            .reshape(batch_size, N)
        )

        # Build cannot-link pairs: speakers from the same chunk must not merge
        cannot_link: set = set()
        for c in range(num_chunks):
            base = c * local_num_spks
            for i in range(local_num_spks):
                for j in range(i + 1, local_num_spks):
                    cannot_link.add(frozenset({base + i, base + j}))

        all_assignments = torch.zeros(
            batch_size, N, max_num_spks,
            device=spk_embs.device, dtype=spk_embs.dtype,
        )

        for b in range(batch_size):
            valid_b = valid_mask[b]
            num_valid = valid_b.sum().item()
            if num_valid == 0:
                continue

            valid_indices = torch.where(valid_b)[0]
            valid_embs = F.normalize(spk_embs_flat[b, valid_b, :], p=2, dim=1)
            cos_sim = torch.mm(valid_embs, valid_embs.T)

            if self.clustering_method == "nmesc":
                cluster_labels = self.speaker_clustering.forward_unit_infer(
                    cos_sim,
                    max_num_speakers=max_num_spks,
                    fixed_thres=self.clustering_threshold,
                )
                cluster_labels = cluster_labels.to(self.device)

                # Post-check: warn if same-chunk speakers share a cluster
                for c in range(num_chunks):
                    base = c * local_num_spks
                    chunk_valid_local = []
                    for vi, idx in enumerate(valid_indices.tolist()):
                        if base <= idx < base + local_num_spks:
                            chunk_valid_local.append(vi)
                    chunk_labels = [cluster_labels[vi].item() for vi in chunk_valid_local]
                    if len(chunk_labels) != len(set(chunk_labels)):
                        logging.warning(
                            f"NMESC: same-chunk speakers share a cluster in chunk {c} "
                            f"(labels={chunk_labels}). Consider using clustering_method='ahc'."
                        )

            elif self.clustering_method == "ahc":
                # Remap cannot-link to valid-only indices
                idx_map = {orig.item(): vi for vi, orig in enumerate(valid_indices)}
                valid_cl: set = set()
                for pair in cannot_link:
                    mapped = [idx_map[p] for p in pair if p in idx_map]
                    if len(mapped) == 2:
                        valid_cl.add(frozenset(mapped))

                threshold = self.clustering_threshold if self.clustering_threshold > 0 else 0.3
                cluster_labels = self._constrained_ahc(cos_sim, valid_cl, threshold)
            else:
                raise ValueError(
                    f"Unknown clustering_method='{self.clustering_method}'. "
                    f"Expected 'nmesc' or 'ahc'."
                )

            cluster_labels = cluster_labels.clamp(0, max_num_spks - 1)
            one_hot = F.one_hot(cluster_labels, num_classes=max_num_spks).to(all_assignments.dtype)
            all_assignments[b, valid_indices, :] = one_hot

        # Reshape back to (num_chunks * batch_size, local_num_spks, max_num_spks)
        all_assignments = (
            all_assignments.view(batch_size, num_chunks, local_num_spks, max_num_spks)
            .transpose(0, 1)
            .reshape(num_chunks * batch_size, local_num_spks, max_num_spks)
        )
        return all_assignments

    def _forward_offline_clustering(
        self,
        local_logits: torch.Tensor,
        spk_embs: torch.Tensor,
        spk_detected: torch.Tensor,
        batch_size: int,
        num_chunks: int,
        local_num_spks: int,
        chunk_len: int,
        lc: int,
        total_n_frames: int,
    ) -> torch.Tensor:
        """Build global logits using clustering-based assignment.

        Replaces the Sinkhorn / streaming-state chunk loop when
        ``clustering_assignment`` is enabled at inference time.
        """
        clustering_assignments = self._compute_clustering_assignments(
            spk_embs=spk_embs,
            spk_detected=spk_detected,
            batch_size=batch_size,
            num_chunks=num_chunks,
            local_num_spks=local_num_spks,
        )

        logits_list = []
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_len
            end = min(start + chunk_len, total_n_frames)
            dur = end - start
            offset = min(lc, start)

            local_logits_chunk = local_logits[
                chunk_idx * batch_size : (chunk_idx + 1) * batch_size, :, :
            ]
            spk_assignments_chunk = clustering_assignments[
                chunk_idx * batch_size : (chunk_idx + 1) * batch_size, :, :
            ]

            logits_chunk = self.nextformer_modules.get_global_logits(
                local_logits=local_logits_chunk,
                spk_assignments=spk_assignments_chunk,
                offset=offset,
                dur=dur,
            )
            logits_list.append(logits_chunk)

        return torch.cat(logits_list, dim=1)

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
            spk_embs (torch.Tensor): Tensor containing local speaker embeddings
                Shape: (num_chunks * batch_size, local_num_spks, spk_emb_dim)
            active_frames_per_spk (torch.Tensor): Tensor containing the number of active frames per speaker
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
        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

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
        else:
            batch_chunks, batch_chunk_lengths, num_chunks = self._create_batch_of_chunks(
                input_tensor=pre_encode_embs,
                input_lengths=pre_encode_lengths,
                lc=lc,
                chunk_len=chunk_len,
                rc=rc,
            )
            batch_chunk_prediction_lengths = batch_chunk_lengths  # Same as full length when no extra context
        
        # Step 5: Run frontend_encoder and forward_infer in one pass
        # Get encoder and CLS embeddings for all chunks
        emb_seq, emb_seq_length, cls_embs = self.frontend_encoder(
            processed_signal=batch_chunks, processed_signal_length=batch_chunk_lengths, bypass_pre_encode=True
        )
        
        # Step 6: Extract only prediction window from encoder output (if using extra contexts)
        # The encoder has processed the full reordered sequence, but we only need predictions for the main window
        if extra_lc > 0 or extra_rc > 0:
            prediction_window_size = lc + chunk_len + rc
            emb_seq = emb_seq[:, :prediction_window_size, :]  # Extract only prediction window
            emb_seq_length = batch_chunk_prediction_lengths  # Use prediction window lengths
        
        local_num_spks = self._cfg.local_num_spks
        # tf projections are for backend
        emb_seq_proj_tf = self.nextformer_modules.encoder_proj_tf(emb_seq)
        spk_embs_proj_tf = self.nextformer_modules.spk_emb_proj_tf(cls_embs[:, :local_num_spks, :])
     
        # Get local logits for all chunks
        local_logits = self.forward_backend(emb_seq_proj_tf, emb_seq_length, spk_embs=spk_embs_proj_tf)
        # (batch_size * num_chunks, chunk_total, local_num_spks)

        # se projections are for between-chunk spk embs matching
        cls_embs_proj_se = self.nextformer_modules.project_spk_embs_for_se(cls_embs)
        emb_seq_proj_se = self.nextformer_modules.project_encoder_for_se(emb_seq)
        if self.spk_embs_decoder is not None:
            encoder_len_mask = self.nextformer_modules.length_to_mask(emb_seq_length, emb_seq_proj_se.shape[1])
            encoder_len_mask = ~encoder_len_mask
            cls_embs_decoded = self.spk_embs_decoder(
                encoder_states=emb_seq_proj_se,
                encoder_len_mask=encoder_len_mask,
                query_states=cls_embs_proj_se,
            )
            spk_embs = cls_embs_decoded[:, :local_num_spks, :]
        else:
            spk_embs = cls_embs_proj_se[:, :local_num_spks, :]

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
            _, local_target_indices = get_pil_targets_hungarian(
                labels=batch_targets[:, :logits_len, :],
                logits=local_logits[:, :logits_len, :],
                metric=self.pil_metric
            )

        preds = torch.sigmoid(local_logits)
        active_frames_per_spk = (preds > self.local_mask_threshold).to(int).sum(dim=1) # (num_chunks * batch_size, local_num_spks)
        spk_detected = active_frames_per_spk > 0 # (num_chunks * batch_size, local_num_spks)
        spk_not_detected = ~spk_detected # (num_chunks * batch_size, local_num_spks)

        # Zero out embeddings for undetected speakers (spk_embs already computed earlier)
        spk_embs = spk_embs.masked_fill(spk_not_detected.unsqueeze(2), 0)

        if att_mod:
            self.encoder.att_context_size = [-1, -1]
            if self.transformer_encoder is not None:
                self.transformer_encoder.diag = None

        if self.spk_embs_enhancer is not None:
            emb_dim = spk_embs.shape[-1]
            spk_embs_reshaped = spk_embs.view(num_chunks, batch_size, local_num_spks, emb_dim).transpose(0, 1)
            spk_embs_reshaped = spk_embs_reshaped.reshape(batch_size, num_chunks * local_num_spks, emb_dim)
            valid_spk_mask = spk_detected.view(num_chunks, batch_size, local_num_spks).transpose(0, 1).reshape(batch_size, num_chunks * local_num_spks)
            spk_embs_enhanced = self.spk_embs_enhancer(encoder_states=spk_embs_reshaped, encoder_mask=valid_spk_mask)
            spk_embs_enhanced = spk_embs_enhanced.masked_fill(~valid_spk_mask.unsqueeze(-1), 0)
            spk_embs = spk_embs_enhanced.view(batch_size, num_chunks, local_num_spks, emb_dim).transpose(0, 1)
            spk_embs = spk_embs.reshape(num_chunks * batch_size, local_num_spks, emb_dim)

        if self.clustering_assignment and not self.training:
            logits = self._forward_offline_clustering(
                local_logits=local_logits,
                spk_embs=spk_embs,
                spk_detected=spk_detected,
                batch_size=batch_size,
                num_chunks=num_chunks,
                local_num_spks=local_num_spks,
                chunk_len=chunk_len,
                lc=lc,
                total_n_frames=total_n_frames,
            )
            if sig_length < max_n_frames:
                n_frames = math.ceil(sig_length / self.encoder.subsampling_factor)
                logits = logits[:, :n_frames, :]
            return logits, local_logits, spk_embs, active_frames_per_spk

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
            spk_embs_chunk = spk_embs[chunk_idx * batch_size:(chunk_idx + 1) * batch_size, :, :] # (batch_size, local_num_spks, emb_dim)
            local_logits_chunk = local_logits[chunk_idx * batch_size:(chunk_idx + 1) * batch_size, :, :] # (batch_size, lc+chunk_len+rc, local_num_spks)
            emb_seq_proj_se_chunk = emb_seq_proj_se[chunk_idx * batch_size:(chunk_idx + 1) * batch_size, :, :] # (batch_size, lc+chunk_len+rc, se_d_model)

            if local_target_indices is not None:
                # Use oracle local-to-global assignments
                global_spk_indices_oracle = local_target_indices[
                    chunk_idx * batch_size:(chunk_idx + 1) * batch_size, :
                ]  # (batch_size, local_num_spks)
                valid_mask = global_spk_indices_oracle >= 0
                safe_indices = global_spk_indices_oracle.clamp(min=0)
                oracle_spk_assignments_chunk = F.one_hot(
                    safe_indices, num_classes=self.nextformer_modules.max_num_spks
                ).to(local_logits_chunk.dtype)
                oracle_spk_assignments_chunk = oracle_spk_assignments_chunk * valid_mask.unsqueeze(-1)
            else:
                raise ValueError("local_target_indices is None")

            if self.oracle_assignment:  
                centroid_spk_assignments_chunk = oracle_spk_assignments_chunk
                spk_assignments_chunk = oracle_spk_assignments_chunk
                profile_sinkhorn_scores = oracle_spk_assignments_chunk
            else:
                # get real local-to-global assignments
                spk_assignments_chunk, sinkhorn_scores_chunk = self.nextformer_modules.get_local_to_global_assignments(
                    spk_embs=spk_embs_chunk,
                    streaming_state=streaming_state,
                )

                # oracle_centroids flags gate whether oracle is used at all.
                # oracle_centroids_weight (if set) controls the softness: 1.0 = pure oracle, <1 = interpolation.
                use_oracle = (self.oracle_centroids_train and self.training) or (
                    self.oracle_centroids_test and not self.training
                )
                if use_oracle:
                    ocw = self.oracle_centroids_weight
                else:
                    ocw = 0.0

                if ocw > 0:
                    # Interpolate between oracle (one-hot) and Sinkhorn (soft).
                    # ocw=1 → pure oracle, ocw=0 → pure Sinkhorn.
                    # Both are in the same coordinate system because oracle_centroids
                    # maintains slot j = target speaker j alignment.
                    centroid_spk_assignments_chunk = (
                        ocw * oracle_spk_assignments_chunk
                        + (1 - ocw) * spk_assignments_chunk
                    )
                    profile_sinkhorn_scores = (
                        ocw * oracle_spk_assignments_chunk
                        + (1 - ocw) * sinkhorn_scores_chunk
                    )
                    # On the first chunk, use oracle for logit assembly to establish
                    # the slot→target column alignment that oracle_centroids depends on.
                    if chunk_idx == 0:
                        spk_assignments_chunk = oracle_spk_assignments_chunk
                else:
                    centroid_spk_assignments_chunk = spk_assignments_chunk
                    profile_sinkhorn_scores = sinkhorn_scores_chunk

            active_frames_chunk = active_frames_per_spk[chunk_idx * batch_size:(chunk_idx + 1) * batch_size, :]

            if self.profile_updater is not None:
                # --- Sinkhorn-masked self-attention profile update ---
                was_active = streaming_state.global_spk_total_confidence > 0  # (B, max_num_spks)

                # Filter unreliable speakers (too few active frames) for the profile
                # updater only.  Sinkhorn matching and get_global_logits still see
                # the original embeddings, so local predictions are preserved.
                min_frames = self.nextformer_modules.spk_emb_update_min_frames
                reliable_spk = active_frames_chunk >= max(min_frames, 1)  # (B, local_num_spks)

                updater_spk_embs = spk_embs_chunk.clone()
                updater_spk_embs[~reliable_spk] = 0.0

                updater_sinkhorn = profile_sinkhorn_scores.clone()
                updater_sinkhorn[~reliable_spk] = 0.0

                if self.profile_updater_detach_inputs:
                    pu_global = streaming_state.global_spk_embs.detach()
                    pu_local = updater_spk_embs.detach()
                    pu_sinkhorn = updater_sinkhorn.detach()
                else:
                    pu_global = streaming_state.global_spk_embs
                    pu_local = updater_spk_embs
                    pu_sinkhorn = updater_sinkhorn

                updated_profiles, updated_local_embs = self.profile_updater(
                    global_profiles=pu_global,
                    local_embs=pu_local,
                    sinkhorn_scores=pu_sinkhorn,
                )
                # Clone to avoid in-place autograd issues when initializing new speakers below
                streaming_state.global_spk_embs = updated_profiles.clone()

                # Handle new speakers: find local speakers assigned to previously-inactive global slots.
                # When oracle_centroids is active (ocw > 0), use oracle assignments for new speaker
                # slot allocation to maintain the slot→target column alignment. The interpolated
                # centroid_spk_assignments_chunk could route new speakers to wrong slots when
                # Sinkhorn's dustbin allocation disagrees with the oracle's target column.
                _local_num_spks = spk_embs_chunk.shape[1]
                if ocw > 0:
                    assigned_globals = oracle_spk_assignments_chunk.argmax(dim=2)
                    has_assignment = oracle_spk_assignments_chunk.max(dim=2).values > 0.01
                else:
                    assigned_globals = centroid_spk_assignments_chunk.argmax(dim=2)
                    has_assignment = centroid_spk_assignments_chunk.max(dim=2).values > 0.01

                _batch_idx = torch.arange(batch_size, device=spk_embs_chunk.device).unsqueeze(1).expand(-1, _local_num_spks)
                # Only allocate new slots for reliable speakers
                is_new_speaker = has_assignment & reliable_spk & ~was_active[_batch_idx, assigned_globals]

                if is_new_speaker.any():
                    new_b, new_s = torch.where(is_new_speaker)
                    new_g = assigned_globals[new_b, new_s]
                    # Initialize new global slot with the (self-attention-refined) local embedding
                    streaming_state.global_spk_embs[new_b, new_g] = updated_local_embs[new_b, new_s]
                    streaming_state.global_spk_total_confidence[new_b, new_g] = 1.0

                # Mark existing assigned speakers as active (increment confidence counter)
                is_existing = has_assignment & reliable_spk & was_active[_batch_idx, assigned_globals]
                if is_existing.any():
                    ex_b, ex_s = torch.where(is_existing)
                    ex_g = assigned_globals[ex_b, ex_s]
                    streaming_state.global_spk_total_confidence[ex_b, ex_g] = (
                        streaming_state.global_spk_total_confidence[ex_b, ex_g] + 1.0
                    )

                logging.info(
                    f"streaming_state.global_spk_total_confidence: "
                    f"{streaming_state.global_spk_total_confidence[0, 0:17]}"
                )
            else:
                if self.profile_update_mode == "cls":
                    self.nextformer_modules.update_streaming_state_cls(
                        streaming_state=streaming_state,
                        spk_embs=spk_embs_chunk,
                        local_logits=local_logits_chunk,
                        spk_assignments=centroid_spk_assignments_chunk,
                        active_frames_per_spk=active_frames_chunk,
                    )
                elif self.profile_update_mode == "frame":
                    self.nextformer_modules.update_streaming_state_frame(
                        streaming_state=streaming_state,
                        emb_seq_proj=emb_seq_proj_se_chunk,
                        local_logits=local_logits_chunk,
                        spk_assignments=centroid_spk_assignments_chunk,
                        active_frames_per_spk=active_frames_chunk,
                    )
                else:
                    raise ValueError(f"Invalid profile update mode: {self.profile_update_mode}")
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

        # Cross-chunk speaker embedding swap augmentation (training only, non-trff backends).
        # Recompute local_logits with swapped embeddings so the regular local loss (PIL + ATS)
        # is computed on augmented inputs. This breaks the positional shortcut where CLS tokens
        # encode speaker arrival order instead of speaker identity.
        # Done after the chunk loop so global logits assembly uses original local_logits.
        if (
            self.training
            and self.cross_chunk_swap_p > 0
            and local_target_indices is not None
            and self.nextformer_modules.backend != "trff"
            and num_chunks > 1
        ):
            swapped_spk_embs_proj_tf = self._create_swapped_embeddings(
                spk_embs=spk_embs_proj_tf,
                local_target_indices=local_target_indices,
                active_frames_per_spk=active_frames_per_spk,
                batch_size=batch_size,
                num_chunks=num_chunks,
                p_swap=self.cross_chunk_swap_p,
                detach=self.cross_chunk_swap_detach,
                min_src_frames=self.cross_chunk_swap_min_frames,
            )
            local_logits = self.forward_backend(
                emb_seq_proj_tf, emb_seq_length, spk_embs=swapped_spk_embs_proj_tf
            )

        return logits, local_logits, spk_embs, active_frames_per_spk

    def forward_backend(self, emb_seq, emb_seq_length, spk_embs=None):
        """
        The main forward pass for diarization for offline diarization inference.
        Dispatches to the appropriate backend based on nextformer_modules.backend.

        Args:
            emb_seq (torch.Tensor): Tensor containing FastConformer encoder states (embedding vectors).
                Shape: (batch_size, diar_frame_count, emb_dim)
            emb_seq_length (torch.Tensor): Tensor containing lengths of FastConformer encoder states.
                Shape: (batch_size,)
            spk_embs (torch.Tensor, optional): Speaker embeddings from CLS tokens.
                Shape: (batch_size, local_num_spks, emb_dim). Required for "dotp" backend.

        Returns:
            logits (torch.Tensor): Tensor containing local speaker logits.
                Shape: (batch_size, diar_frame_count, num_speakers)
        """
        encoder_mask = self.nextformer_modules.length_to_mask(emb_seq_length, emb_seq.shape[1])

        if self.nextformer_modules.backend == "trff":
            logits = self.backend_trff(emb_seq, encoder_mask)
        elif self.nextformer_modules.backend == "dotp":
            if spk_embs is None:
                raise ValueError("spk_embs is required for 'dotp' backend")
            logits = self.backend_dotp(emb_seq, spk_embs)
        elif self.nextformer_modules.backend == "isd":
            if spk_embs is None:
                raise ValueError("spk_embs is required for 'isd' backend")
            logits = self.backend_isd(emb_seq, emb_seq_length, spk_embs)
        elif self.nextformer_modules.backend == "jsd":
            if spk_embs is None:
                raise ValueError("spk_embs is required for 'jsd' backend")
            logits = self.backend_jsd(emb_seq, emb_seq_length, spk_embs)
        else:
            raise ValueError(f"Unknown backend: {self.nextformer_modules.backend}")
        
        # Apply length mask (common to all backends)
        mask = encoder_mask.unsqueeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        return logits

    def backend_trff(self, emb_seq, encoder_mask):
        """
        Transformer + feedforward backend for computing local speaker logits.

        Args:
            emb_seq (torch.Tensor): Tensor containing encoder states.
                Shape: (batch_size, diar_frame_count, emb_dim)
            encoder_mask (torch.Tensor): Boolean mask for encoder states.
                Shape: (batch_size, diar_frame_count)

        Returns:
            logits (torch.Tensor): Tensor containing local speaker logits (unmasked).
                Shape: (batch_size, diar_frame_count, local_num_spks)
        """
        if self.transformer_encoder is not None:
            emb_seq = self.transformer_encoder(encoder_states=emb_seq, encoder_mask=encoder_mask)
        logits = self.nextformer_modules.forward_spk_logits(emb_seq)
        return logits

    def backend_dotp(self, emb_seq, spk_embs):
        """
        Dot product backend for computing local speaker logits.
        Computes logits as dot product between frame embeddings and speaker embeddings.

        Args:
            emb_seq (torch.Tensor): Tensor containing encoder states (projected).
                Shape: (batch_size, diar_frame_count, emb_dim)
            spk_embs (torch.Tensor): Speaker embeddings from CLS tokens (projected).
                Shape: (batch_size, local_num_spks, emb_dim)

        Returns:
            logits (torch.Tensor): Tensor containing local speaker logits (unmasked).
                Shape: (batch_size, diar_frame_count, local_num_spks)
        """
        # Compute dot product: (B, T, D) @ (B, D, local_num_spks) -> (B, T, local_num_spks)
        logits = torch.bmm(emb_seq, spk_embs.transpose(1, 2))
        return logits

    def backend_isd(self, emb_seq, emb_seq_length, spk_embs):
        """
        Individual Speaker Detection (ISD) backend for computing local speaker logits.
        Fuses per-speaker embeddings with frame embeddings using configurable fusion
        and runs a dedicated encoder.

        Args:
            emb_seq (torch.Tensor): Tensor containing encoder states (projected).
                Shape: (batch_size, diar_frame_count, tf_d_model)
            emb_seq_length (torch.Tensor): Tensor containing lengths of encoder states.
                Shape: (batch_size,)
            spk_embs (torch.Tensor): Speaker embeddings from CLS tokens (projected).
                Shape: (batch_size, local_num_spks, se_d_model)

        Returns:
            logits (torch.Tensor): Tensor containing local speaker logits (unmasked).
                Shape: (batch_size, diar_frame_count, local_num_spks)
        """
        if self.isd_encoder is None or self.nextformer_modules.backend_output_proj is None:
            raise RuntimeError("ISD backend modules are not initialized")

        batch_size, seq_len, _ = emb_seq.shape
        local_num_spks = spk_embs.shape[1]

        # Expand frame embeddings to per-speaker sequences: (B, S, T, tf_d_model)
        emb_seq_expanded = emb_seq.unsqueeze(1).expand(-1, local_num_spks, -1, -1)

        # Apply configurable fusion
        combined = self.nextformer_modules.apply_fusion(emb_seq_expanded, spk_embs)

        # Reshape to (B * local_num_spks, T, tf_d_model)
        combined = combined.reshape(batch_size * local_num_spks, seq_len, -1)

        # Repeat lengths for each speaker
        isd_lengths = emb_seq_length.unsqueeze(1).expand(-1, local_num_spks).reshape(-1)
        encoder_mask = self.nextformer_modules.length_to_mask(isd_lengths, seq_len)

        # Run ISD encoder and output projection
        encoded = self.isd_encoder(encoder_states=combined, encoder_mask=encoder_mask)
        logits = self.nextformer_modules.backend_output_proj(encoded)  # (B * local_num_spks, T, 1)

        # Reshape back to (B, T, local_num_spks)
        logits = logits.reshape(batch_size, local_num_spks, seq_len, 1).squeeze(-1).transpose(1, 2)
        return logits

    def backend_jsd(self, emb_seq, emb_seq_length, spk_embs):
        """
        Joint Speaker Detection (JSD) backend for computing local speaker logits.
        Alternates time-wise and speaker-wise self-attention to model both
        temporal patterns and speaker interactions jointly.

        Args:
            emb_seq (torch.Tensor): Tensor containing encoder states (projected).
                Shape: (batch_size, diar_frame_count, tf_d_model)
            emb_seq_length (torch.Tensor): Tensor containing lengths of encoder states.
                Shape: (batch_size,)
            spk_embs (torch.Tensor): Speaker embeddings from CLS tokens (projected).
                Shape: (batch_size, local_num_spks, se_d_model)

        Returns:
            logits (torch.Tensor): Tensor containing local speaker logits (unmasked).
                Shape: (batch_size, diar_frame_count, local_num_spks)
        """
        if self.jsd_encoder is None or self.nextformer_modules.backend_output_proj is None:
            raise RuntimeError("JSD backend modules are not initialized")

        local_num_spks = spk_embs.shape[1]

        # Expand frame embeddings to per-speaker sequences: (B, S, T, tf_d_model)
        emb_seq_expanded = emb_seq.unsqueeze(1).expand(-1, local_num_spks, -1, -1)

        # Apply configurable fusion
        combined = self.nextformer_modules.apply_fusion(emb_seq_expanded, spk_embs)  # (B, S, T, tf_d_model)

        # Run JSD encoder - input/output is (B, S, T, D)
        encoded = self.jsd_encoder(combined, time_lengths=emb_seq_length)  # (B, S, T, D)

        # Project to logits
        logits = self.nextformer_modules.backend_output_proj(encoded)  # (B, S, T, 1)
        logits = logits.squeeze(-1).transpose(1, 2)  # (B, T, S)
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
            logits, _, _, _ = self.forward(audio_signal=batch[0], audio_signal_length=batch[1])
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

    def compute_supcon_loss(
        self,
        spk_embs,
        local_target_indices,
        global_speaker_ids,
        active_frames_per_spk,
        batch_size,
        num_chunks,
        force_within_session=False,
    ):
        """
        Supervised Contrastive (SupCon) loss on speaker embeddings.

        When cross-batch mode is active (and not overridden), uses globally
        unique speaker IDs so that same-speaker embeddings across different
        sessions are treated as positives.  When within-session mode is active,
        each session is processed independently -- no cross-session interactions.

        Args:
            spk_embs: SE-projected speaker embeddings.
                Shape: (num_chunks * B, local_num_spks, emb_dim)
            local_target_indices: Oracle mapping from local speaker slots to
                target matrix columns.
                Shape: (num_chunks * B, local_num_spks)
            global_speaker_ids: Global integer speaker ID per target column.
                Shape: (B, max_num_spks)
            active_frames_per_spk: Number of active frames per speaker.
                Shape: (num_chunks * B, local_num_spks)
            batch_size: Batch size.
            num_chunks: Number of chunks per batch item.
            force_within_session: If True, always use within-session mode
                regardless of self.supcon_cross_batch. Used for validation
                where cross-batch matching is not meaningful.

        Returns:
            loss (torch.Tensor): Scalar SupCon loss.
        """
        local_num_spks = spk_embs.shape[1]
        emb_dim = spk_embs.shape[2]
        max_num_spks = global_speaker_ids.shape[1]

        # Reshape for indexing: (num_chunks, B, local_num_spks, ...)
        target_cols = local_target_indices.view(num_chunks, batch_size, local_num_spks)
        af = active_frames_per_spk.view(num_chunks, batch_size, local_num_spks)

        # Track which batch item (session) each embedding belongs to
        batch_idx = torch.arange(batch_size, device=spk_embs.device)
        batch_idx = batch_idx.unsqueeze(0).unsqueeze(-1).expand(num_chunks, batch_size, local_num_spks)

        use_cross_batch = self.supcon_cross_batch and not force_within_session

        # Resolve each CLS embedding's speaker ID
        emb_speaker_ids = torch.full(
            (num_chunks, batch_size, local_num_spks), -1,
            dtype=torch.long, device=spk_embs.device,
        )
        valid_col = target_cols >= 0
        safe_cols = target_cols.clamp(min=0)

        if use_cross_batch:
            # Use globally unique speaker IDs from RTTM vocabulary
            emb_speaker_ids[valid_col] = global_speaker_ids[
                batch_idx[valid_col], safe_cols[valid_col]
            ]
        else:
            # Use session-local IDs: offset column by batch item so that
            # same column in different sessions maps to different IDs.
            # This avoids needing the shared vocabulary for within-session mode.
            emb_speaker_ids[valid_col] = (
                batch_idx[valid_col] * max_num_spks + safe_cols[valid_col]
            )

        # Flatten and filter by validity and minimum active frames
        all_embs = spk_embs.view(num_chunks, batch_size, local_num_spks, emb_dim)
        all_embs_flat = all_embs.reshape(-1, emb_dim)
        all_ids_flat = emb_speaker_ids.reshape(-1)
        all_af_flat = af.reshape(-1)
        all_batch_flat = batch_idx.reshape(-1)

        valid_mask = (all_ids_flat >= 0) & (all_af_flat >= self.supcon_min_active_frames)
        valid_embs = all_embs_flat[valid_mask]
        valid_ids = all_ids_flat[valid_mask]
        valid_batch = all_batch_flat[valid_mask]
        N = valid_embs.shape[0]

        if N < 2:
            return torch.tensor(0.0, device=spk_embs.device, requires_grad=True)

        # L2-normalize for cosine similarity
        z = F.normalize(valid_embs, dim=-1)

        # Raw pairwise cosine similarity in [-1, 1]
        sim_raw = z @ z.T  # (N, N)

        # Masks
        self_mask = torch.eye(N, dtype=torch.bool, device=sim_raw.device)
        same_speaker = (valid_ids.unsqueeze(0) == valid_ids.unsqueeze(1))  # (N, N)
        same_session = (valid_batch.unsqueeze(0) == valid_batch.unsqueeze(1))  # (N, N)

        if use_cross_batch:
            pos_mask = same_speaker & ~self_mask
            if self.supcon_decoupled:
                denom_mask = ~same_speaker & ~self_mask
            else:
                denom_mask = ~self_mask
        else:
            pos_mask = same_speaker & same_session & ~self_mask
            if self.supcon_decoupled:
                denom_mask = ~same_speaker & same_session & ~self_mask
            else:
                denom_mask = same_session & ~self_mask

        # Apply additive angular margin (ArcFace) to positive pairs before temperature scaling
        if self.supcon_aam > 0:
            cos_m = math.cos(self.supcon_aam)
            sin_m = math.sin(self.supcon_aam)
            threshold = -cos_m
            mm = sin_m * self.supcon_aam
            cos_theta = torch.clamp(sim_raw, -1.0, 1.0)
            sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta ** 2, min=1e-8))
            cos_theta_plus_m = cos_theta * cos_m - sin_theta * sin_m
            cos_theta_plus_m = torch.where(cos_theta < threshold, cos_theta - mm, cos_theta_plus_m)
            sim_raw = torch.where(pos_mask, cos_theta_plus_m, sim_raw)

        # Scale by SupCon temperature (separate from Sinkhorn's cosine_temperature
        # to allow balanced gradients across positives and negatives)
        sim = sim_raw / self.supcon_temperature  # (N, N)

        # Only anchors with both positives and denominator entries can contribute
        has_positive = pos_mask.any(dim=1)
        has_denom = denom_mask.any(dim=1)
        active_anchors = has_positive & has_denom
        if not active_anchors.any():
            return torch.tensor(0.0, device=spk_embs.device, requires_grad=True)

        # Numerical stability: subtract row-wise max over denom entries.
        # For rows without denom entries, set sim_max to 0 to avoid -inf → overflow.
        NEG_INF = float('-inf')
        sim_for_max = sim.detach().masked_fill(~denom_mask, NEG_INF)
        sim_max = sim_for_max.max(dim=1, keepdim=True).values
        sim_max = sim_max.masked_fill(~has_denom.unsqueeze(1), 0.0)
        sim_shifted = sim - sim_max

        # Mask non-denom entries to -inf BEFORE exp so that exp(-inf) = 0 exactly,
        # avoiding the inf * 0 = NaN hazard from post-multiply masking.
        sim_shifted_denom = sim_shifted.masked_fill(~denom_mask, NEG_INF)
        exp_sim = torch.exp(sim_shifted_denom)

        # Add dustbin class to denominator: creates a reference point that
        # negatives above dustbin_val get stronger gradient to push down
        dustbin_score = (self.nextformer_modules.sinkhorn_dustbin_val.detach() - self.supcon_dustbin_margin) / self.supcon_temperature
        exp_dustbin = torch.exp(dustbin_score - sim_max.squeeze(1))  # (N,) shifted for stability
        log_denom = torch.log(exp_sim.sum(dim=1) + exp_dustbin + 1e-12)  # (N,)

        # Numerator: use sim_shifted (not sim_shifted_denom) so that positive
        # entries retain valid values in decoupled mode where pos_mask and
        # denom_mask are disjoint.
        pos_sim_sum = torch.where(pos_mask, sim_shifted, torch.zeros_like(sim_shifted)).sum(dim=1)  # (N,)
        num_positives = pos_mask.float().sum(dim=1).clamp(min=1)  # (N,)

        loss_per_anchor = -(pos_sim_sum / num_positives - log_denom)
        loss = loss_per_anchor[active_anchors].mean()
        return loss

    def compute_aam_softmax_loss(
        self,
        spk_embs,
        local_target_indices,
        global_speaker_ids,
        active_frames_per_spk,
        batch_size,
        num_chunks,
        total_confidence_per_spk=None,
    ):
        """
        AAM-Softmax (ArcFace) loss on speaker embeddings.

        Each valid CLS embedding is classified to its global speaker class using
        cosine similarity against learnable class weight vectors with additive
        angular margin on the correct class.

        Args:
            spk_embs: SE-projected speaker embeddings.
                Shape: (num_chunks * B, local_num_spks, emb_dim)
            local_target_indices: Oracle mapping from local speaker slots to
                target matrix columns.
                Shape: (num_chunks * B, local_num_spks)
            global_speaker_ids: Global integer speaker ID per target column.
                Shape: (B, max_num_spks)
            active_frames_per_spk: Number of active frames per speaker.
                Shape: (num_chunks * B, local_num_spks)
            batch_size: Batch size.
            num_chunks: Number of chunks per batch item.
            total_confidence_per_spk: Accumulated per-frame confidence per speaker.
                Shape: (num_chunks * B, local_num_spks). If provided and
                aam_min_confidence > 0, used as additional quality filter.

        Returns:
            loss (torch.Tensor): Scalar AAM-Softmax loss.
        """
        if self.nextformer_modules.aam_head is None:
            logging.warning("AAM-Softmax head is not initialized, returning zero loss.")
            return torch.tensor(0.0, device=spk_embs.device)

        local_num_spks = spk_embs.shape[1]
        emb_dim = spk_embs.shape[2]
        max_num_spks = global_speaker_ids.shape[1]
        num_classes = self.nextformer_modules.aam_head.weight.shape[0]

        # Resolve per-embedding global speaker IDs (same chain as SupCon)
        target_cols = local_target_indices.view(num_chunks, batch_size, local_num_spks)
        af = active_frames_per_spk.view(num_chunks, batch_size, local_num_spks)
        batch_idx = torch.arange(batch_size, device=spk_embs.device)
        batch_idx = batch_idx.unsqueeze(0).unsqueeze(-1).expand(num_chunks, batch_size, local_num_spks)

        emb_speaker_ids = torch.full(
            (num_chunks, batch_size, local_num_spks), -1,
            dtype=torch.long, device=spk_embs.device,
        )
        valid_col = target_cols >= 0
        safe_cols = target_cols.clamp(min=0)
        emb_speaker_ids[valid_col] = global_speaker_ids[
            batch_idx[valid_col], safe_cols[valid_col]
        ]

        # Flatten and filter
        all_embs = spk_embs.view(num_chunks, batch_size, local_num_spks, emb_dim).reshape(-1, emb_dim)
        all_ids = emb_speaker_ids.reshape(-1)
        all_af = af.reshape(-1)
        emb_norms = all_embs.norm(dim=-1)

        valid_mask = (
            (all_ids >= 0)
            & (all_ids < num_classes)
            & (all_af >= self.aam_min_active_frames)
            & (emb_norms > 1e-6)
            & torch.isfinite(emb_norms)
        )

        # Confidence-based filtering: adaptive threshold that is strict when
        # predictions are uncertain (early training) and permissive when confident.
        if total_confidence_per_spk is not None and self.aam_min_confidence > 0:
            all_conf = total_confidence_per_spk.view(num_chunks, batch_size, local_num_spks).reshape(-1)
            valid_mask = valid_mask & (all_conf >= self.aam_min_confidence)
        valid_embs = all_embs[valid_mask]
        valid_labels = all_ids[valid_mask]
        N = valid_embs.shape[0]

        if N < 1:
            return torch.tensor(0.0, device=spk_embs.device, requires_grad=True)

        # L2-normalize embeddings and weight matrix
        embs_norm = F.normalize(valid_embs, dim=1)
        W_norm = F.normalize(self.nextformer_modules.aam_head.weight, dim=1)  # (C, D)

        # Cosine similarities to all class centers
        cos_theta = F.linear(embs_norm, W_norm)  # (N, C)

        # Apply angular margin to the correct class
        if self.aam_margin > 0:
            cos_m = math.cos(self.aam_margin)
            sin_m = math.sin(self.aam_margin)
            threshold = -cos_m
            mm = sin_m * self.aam_margin

            cos_theta_clamped = torch.clamp(cos_theta, -1.0, 1.0)
            sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta_clamped ** 2, min=1e-8))
            cos_theta_plus_m = cos_theta_clamped * cos_m - sin_theta * sin_m

            # Canonical ArcFace fallback: linear approximation when theta + m > pi
            cos_theta_plus_m = torch.where(
                cos_theta_clamped < threshold,
                cos_theta_clamped - mm,
                cos_theta_plus_m,
            )

            one_hot = F.one_hot(valid_labels, num_classes=num_classes).bool()
            cos_theta = torch.where(one_hot, cos_theta_plus_m, cos_theta)

        # Scale and cross-entropy
        logits = cos_theta * self.aam_scale
        loss = F.cross_entropy(logits, valid_labels)

        if not torch.isfinite(loss):
            logging.warning("AAM-Softmax loss is NaN/inf, returning zero.")
            return torch.tensor(0.0, device=spk_embs.device, requires_grad=True)

        return loss

    def _get_aux_train_evaluations(
        self, logits, local_logits, spk_embs, active_frames_per_spk, targets, target_lens,
        global_speaker_ids=None,
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
                When cross-chunk swap augmentation is active, these are computed with
                swapped speaker embeddings (some slots use same-speaker embeddings
                from different chunks).
            spk_embs (torch.Tensor): Local speaker embeddings.
                Shape: (batch_size * num_chunks, local_num_spks, emb_dim)
            active_frames_per_spk (torch.Tensor): Tensor containing the number of active frames per speaker
                Shape: (num_chunks * batch_size, local_num_spks)
            targets (torch.Tensor): Ground truth speaker labels.
                Shape: (batch_size, total_n_frames, max_num_spks)
            target_lens (torch.Tensor): Lengths of target sequences.
                Shape: (batch_size,)
            global_speaker_ids (torch.Tensor, optional): Global speaker integer IDs per target column.
                Shape: (batch_size, max_num_spks). Used for cross-batch contrastive losses.

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

        # Compute chunk-level confidence for embedding quality filtering.
        # confidence = sum of per-frame C[i] = P[i] * Prod(1-P[j]) over active frames.
        # Used by SupCon and AAM to filter unreliable embeddings.
        batch_size = targets.shape[0]
        num_chunks = local_logits.shape[0] // batch_size
        local_preds_for_conf = torch.sigmoid(local_logits)
        per_frame_confidence = self.nextformer_modules._get_confidence(local_preds_for_conf)
        per_frame_confidence = per_frame_confidence.masked_fill(local_preds_for_conf <= self.local_mask_threshold, 0)
        total_confidence_per_spk = per_frame_confidence.sum(dim=1)  # (num_chunks * B, local_num_spks)

        # SupCon auxiliary loss on speaker embeddings
        supcon_loss = torch.tensor(0.0, device=logits.device)
        if self.supcon_weight >= 0 and global_speaker_ids is not None:
            supcon_loss = self.compute_supcon_loss(
                spk_embs=spk_embs,
                local_target_indices=local_target_indices,
                global_speaker_ids=global_speaker_ids,
                active_frames_per_spk=active_frames_per_spk,
                batch_size=batch_size,
                num_chunks=num_chunks,
            )
            if self.supcon_weight > 0:
                loss = loss + self.supcon_weight * supcon_loss

        # AAM-Softmax auxiliary loss on speaker embeddings
        aam_loss = torch.tensor(0.0, device=logits.device)
        if self.aam_weight >= 0 and global_speaker_ids is not None:
            aam_loss = self.compute_aam_softmax_loss(
                spk_embs=spk_embs,
                local_target_indices=local_target_indices,
                global_speaker_ids=global_speaker_ids,
                active_frames_per_spk=active_frames_per_spk,
                total_confidence_per_spk=total_confidence_per_spk,
                batch_size=batch_size,
                num_chunks=num_chunks,
            )
            loss = loss + self.aam_weight * aam_loss

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
            'supcon_loss': supcon_loss,
            'aam_loss': aam_loss,
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
                - speaker_names (list[list[str|None]]): RTTM speaker names per target column.
            batch_idx (int): The index of the current batch.

        Returns:
            (dict): A dictionary containing the 'loss' key with the calculated loss value.
        """
        audio_signal, audio_signal_length, targets, target_lens, speaker_names = batch
        global_speaker_ids = self._speaker_names_to_ids(speaker_names)
        logits, local_logits, spk_embs, active_frames_per_spk = self.forward(
            audio_signal=audio_signal, audio_signal_length=audio_signal_length, targets=targets
        )
        train_metrics = self._get_aux_train_evaluations(
            logits, local_logits, spk_embs, active_frames_per_spk, targets, target_lens,
            global_speaker_ids=global_speaker_ids,
        )
        logging.info(f"dustbin parameter value: {self.nextformer_modules.sinkhorn_dustbin_val.item()}")
        self._reset_train_metrics()
        self.log_dict(train_metrics, sync_dist=True, on_step=True, on_epoch=False, logger=True)
        return {'loss': train_metrics['loss']}

    def _get_aux_validation_evaluations(
        self, logits, local_logits, spk_embs, active_frames_per_spk, targets, target_lens,
        global_speaker_ids=None,
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
            spk_embs (torch.Tensor): Local speaker embeddings.
                Shape: (batch_size * num_chunks, local_num_spks, emb_dim)
            active_frames_per_spk (torch.Tensor): Tensor containing the number of active frames per speaker
                Shape: (num_chunks * batch_size, local_num_spks)
            targets (torch.Tensor): Ground truth speaker labels.
                Shape: (batch_size, total_n_frames, max_num_spks)
            target_lens (torch.Tensor): Lengths of target sequences.
                Shape: (batch_size,)
            global_speaker_ids (torch.Tensor, optional): Global speaker integer IDs per target column.
                Shape: (batch_size, max_num_spks). Used for cross-batch contrastive losses.

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

        # SupCon for monitoring (within-session only, regardless of cross_batch setting)
        batch_size = targets.shape[0]
        num_chunks = local_logits.shape[0] // batch_size
        val_supcon_loss = torch.tensor(0.0, device=logits.device)
        if self.supcon_weight >= 0 and global_speaker_ids is not None:
            val_supcon_loss = self.compute_supcon_loss(
                spk_embs=spk_embs,
                local_target_indices=local_target_indices,
                global_speaker_ids=global_speaker_ids,
                active_frames_per_spk=active_frames_per_spk,
                batch_size=batch_size,
                num_chunks=num_chunks,
                force_within_session=True,
            )
            if self.supcon_weight > 0:
                val_loss = val_loss + self.supcon_weight * val_supcon_loss

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
            'val_supcon_loss': val_supcon_loss,
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
                - speaker_names (list[list[str|None]]): RTTM speaker names per target column.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int, optional): The index of the dataloader in case of multiple
                                            validation dataloaders. Defaults to 0.

        Returns:
            dict: A dictionary containing various validation metrics for this batch.
        """
        audio_signal, audio_signal_length, targets, target_lens, speaker_names = batch
        global_speaker_ids = self._speaker_names_to_ids(speaker_names)
        logits, local_logits, spk_embs, active_frames_per_spk = self.forward(
            audio_signal=audio_signal, audio_signal_length=audio_signal_length, targets=targets
        )
        val_metrics = self._get_aux_validation_evaluations(
            logits, local_logits, spk_embs, active_frames_per_spk, targets, target_lens,
            global_speaker_ids=global_speaker_ids,
        )
        if isinstance(self.trainer.val_dataloaders, list) and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(val_metrics)
        else:
            self.validation_step_outputs.append(val_metrics)
        return val_metrics

    def test_step(self, batch: list, batch_idx: int, dataloader_idx: int = 0):
        """
        Performs a single test step (delegates to validation_step).

        Args:
            batch (list): A list containing the following elements:
                - audio_signal (torch.Tensor): The input audio signal.
                - audio_signal_length (torch.Tensor): The length of each audio signal in the batch.
                - targets (torch.Tensor): The target labels for the batch.
                - target_lens (torch.Tensor): The length of each target sequence in the batch.
                - speaker_names (list[list[str|None]]): RTTM speaker names per target column.
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
        val_supcon_loss_mean = torch.stack([x['val_supcon_loss'] for x in outputs]).mean()
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
            'val_supcon_loss': val_supcon_loss_mean,
            'val_f1_acc': val_f1_acc_mean,
            'val_f1_acc_global': val_f1_acc_global_mean,
            'val_f1_acc_global_op': val_f1_acc_global_op_mean,
            'val_precision': val_precision_mean,
            'val_recall': val_recall_mean,
            'val_f1_acc_ats': val_f1_acc_ats_mean,
        }
        return {'log': multi_val_metrics}

    def _get_aux_test_batch_evaluations(
        self, batch_idx: int, logits, local_logits, spk_embs, active_frames_per_spk, targets, target_lens
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
                audio_signal, audio_signal_length, targets, target_lens, _speaker_names = batch
                audio_signal = audio_signal.to(self.device)
                audio_signal_length = audio_signal_length.to(self.device)
                targets = targets.to(self.device)
                logits, local_logits, spk_embs, active_frames_per_spk = self.forward(
                    audio_signal=audio_signal, audio_signal_length=audio_signal_length, targets=targets
                )
                self._get_aux_test_batch_evaluations(
                    batch_idx, logits, local_logits, spk_embs, active_frames_per_spk, targets, target_lens
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