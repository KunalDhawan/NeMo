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
from dataclasses import dataclass

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
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import (
    find_first_nonzero,
    get_ats_targets,
    get_ats_targets_hungarian,
    get_pil_targets,
    get_pil_targets_hungarian,
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

# Long sessions can have combinatorially many post-cache chunk subsets. Search
# exhaustively below this limit and sample a bounded subset above it.
_MAX_CHUNK_REPLACEMENT_PLANS_PER_COUNT = 32
# Count and copy every positive soft target consistently. This conservatively
# rejects some plans with tiny soft tails rather than dropping labeled speakers.
_CHUNK_REPLACEMENT_ACTIVITY_THRESHOLD = 0.0


class _ChunkReplacementInvariantError(RuntimeError):
    """A prevalidated replacement plan violated a target-construction invariant."""


@dataclass(frozen=True)
class _ChunkReplacementPlan:
    recipient_index: int
    donor_index: int
    destination_chunks: Tuple[int, ...]


@dataclass(frozen=True)
class _ChunkReplacementMetadata:
    target_values: torch.Tensor
    valid_speaker_activity: torch.Tensor
    feature_lengths: torch.Tensor
    target_lengths: torch.Tensor
    channel_speaker_indices: torch.Tensor
    full_speaker_presence: torch.Tensor
    chunk_speaker_presence: torch.Tensor
    compatible_pairs: torch.Tensor
    feature_frames_per_chunk: int
    target_frames_per_chunk: int


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

    def __init__(self, dataset, *, num_samples_per_epoch: int, trainer=None, **kwargs):
        super().__init__(dataset, **kwargs)
        self._target = max(math.ceil(num_samples_per_epoch / self.num_replicas), 1)
        # Handle on the PTL trainer so the shuffle epoch can follow the *restored* global
        # epoch on resume. A freshly-constructed sampler starts at epoch 0, and on a mid-epoch
        # resume its iterator is built (FitLoop.setup_data) before Lightning calls set_epoch(),
        # which would otherwise replay the epoch-0 ordering instead of continuing the schedule.
        self._trainer = trainer

    def _partial_resume_offset(self) -> int:
        """Return a nonzero shuffle offset when resuming into a partially completed epoch.

        On a mid-epoch resume the trainer restores ``current_epoch`` with some batches
        already consumed, then replays the *same* epoch ordering for the remaining steps.
        Because the shuffle seed is derived from the epoch index alone, those remaining
        steps would re-show samples already seen earlier in this very epoch. Deriving an
        offset from the number of already-processed batches lets us shift to a disjoint
        seed so the rest of the epoch draws fresh data instead.

        The offset is deterministic for a given resume point and is 0 at normal epoch
        starts (``processed == 0``), so uninterrupted runs are unaffected and only the
        partially completed epoch is perturbed. The offset may occasionally coincide with
        another epoch's seed; such collisions are rare and harmless.
        """
        if self._trainer is None:
            return 0
        try:
            processed = int(self._trainer.fit_loop.epoch_loop.batch_progress.current.processed)
        except AttributeError:
            return 0
        return max(processed, 0)

    def __iter__(self):
        # Align the shuffle epoch with the trainer's (restored) epoch before the base
        # DistributedSampler permutation is computed, so a resumed run continues the data
        # schedule rather than rewinding to the epoch-0 ordering.
        if self._trainer is not None:
            current_epoch = getattr(self._trainer, "current_epoch", None)
            if current_epoch is not None:
                self.epoch = current_epoch
        # On a mid-epoch resume, shift to a disjoint seed namespace so that both the base
        # DistributedSampler permutation and the oversampling cycles below produce a fresh
        # ordering, rather than replaying the samples already consumed this epoch.
        self.epoch += self._partial_resume_offset()
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
        self.upsample_mode = self._cfg.get("upsample_mode", "single")
        upsample_kernel_sizes_cfg = self._cfg.get("upsample_kernel_sizes", [3, 5, 7])
        try:
            self.upsample_kernel_sizes = list(upsample_kernel_sizes_cfg)
        except TypeError as e:
            raise TypeError(
                "upsample_kernel_sizes must be an iterable of positive odd integers, "
                f"but got {type(upsample_kernel_sizes_cfg).__name__}: {upsample_kernel_sizes_cfg}"
            ) from e

        if self.upsample_factor > 1:
            if not isinstance(self.upsample_smooth_kernel, int):
                raise TypeError(
                    "upsample_smooth_kernel must be an integer when upsampling is enabled, "
                    f"but got {type(self.upsample_smooth_kernel).__name__}: {self.upsample_smooth_kernel}"
                )
            if self.upsample_smooth_kernel < 1:
                raise ValueError(
                    f"upsample_smooth_kernel must be >= 1 when upsampling is enabled, got {self.upsample_smooth_kernel}"
                )
            if self.upsample_smooth_kernel % 2 == 0:
                raise ValueError(
                    "upsample_smooth_kernel must be odd when upsampling is enabled to preserve output length, "
                    f"got {self.upsample_smooth_kernel}"
                )

        # When val_upsample_preds is True, training runs at coarse encoder resolution
        # (output_subsampling_factor) while validation targets are prepared at 10 ms
        # resolution and predictions are upsampled with upsample_preds.
        self.val_upsample_preds = self._cfg.get("val_upsample_preds", False)
        if self.val_upsample_preds and self.upsample_factor > 1:
            raise ValueError(
                "val_upsample_preds cannot be combined with learnable upsampling "
                "(output_subsampling_factor < encoder.subsampling_factor). "
                "Set output_subsampling_factor equal to encoder.subsampling_factor "
                "when using val_upsample_preds."
            )
        self.val_upsample_smooth_kernel = self._cfg.get(
            "val_upsample_smooth_kernel", self.output_subsampling_factor + 1
        )

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
        self.spec_augment_per_chunk = self._cfg.get("spec_augment_per_chunk", True)

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

        # Set up learnable sub-pixel upsampling for high-resolution output
        if self.upsample_factor > 1:
            self.sortformer_modules.upsample_factor = self.upsample_factor
            self.sortformer_modules.upsample_mode = self.upsample_mode
            self.sortformer_modules.upsample_kernel_sizes = self.upsample_kernel_sizes
            self.sortformer_modules.preenc_proj_dim = self._cfg.get("preenc_proj_dim", 0)
            self.sortformer_modules._init_subpixel_upsample()
            if self.sortformer_modules.subpixel_convs is not None:
                self.sortformer_modules.subpixel_convs = self.sortformer_modules.subpixel_convs.to(self.device)
            if self.sortformer_modules.subpixel_norms is not None:
                self.sortformer_modules.subpixel_norms = self.sortformer_modules.subpixel_norms.to(self.device)
            if self.sortformer_modules.subpixel_upsample is not None:
                self.sortformer_modules.subpixel_upsample = self.sortformer_modules.subpixel_upsample.to(self.device)
            if self.sortformer_modules.preenc_proj is not None:
                self.sortformer_modules.preenc_proj = self.sortformer_modules.preenc_proj.to(self.device)
            if self.sortformer_modules.preenc_norm is not None:
                self.sortformer_modules.preenc_norm = self.sortformer_modules.preenc_norm.to(self.device)
            if self.sortformer_modules.fusion_block is not None:
                self.sortformer_modules.fusion_block = self.sortformer_modules.fusion_block.to(self.device)

        self._init_loss_weights()
        self._init_batch_noise_augmentation()
        self._init_batch_chunk_replace_augmentation()
        self._init_activity_auxiliary_heads()

        self.eps = 0.004   # bf16-safe epsilon
        self.negative_init_val = -99
        self._reported_nonfinite_grad = False
        self.loss = instantiate(self._cfg.loss)

        # Loss-only target softening. These transforms are applied exclusively to the
        # targets fed to the BCE loss; the F1/accuracy metrics keep using the original
        # hard (0/1) targets, so reported metrics remain comparable across runs.
        #   (1) loss_target_smooth_kernel: temporal boundary smoothing window (in output
        #       frames, must be odd). 1 disables it. Softens the abrupt 0->1 jumps at
        #       speaker onsets/offsets, which is especially relevant at fine output
        #       resolution (output_subsampling_factor close to 1).
        #   (2) loss_label_smoothing: clamps targets to [eps, 1 - eps] to curb BCE
        #       overconfidence. 0.0 disables it.
        # Both default to disabled, so existing configs are unaffected.
        self.loss_target_smooth_kernel = int(self._cfg.get("loss_target_smooth_kernel", 1))
        self.loss_target_smooth_type = self._cfg.get("loss_target_smooth_type", "gaussian")
        self.loss_target_smooth_sigma = self._cfg.get("loss_target_smooth_sigma", None)
        self.loss_label_smoothing = float(self._cfg.get("loss_label_smoothing", 0.0))
        if not 0.0 <= self.loss_label_smoothing < 0.5:
            raise ValueError(
                f"loss_label_smoothing must be in [0.0, 0.5), got {self.loss_label_smoothing}"
            )
        self._loss_smooth_kernel_1d = None
        if self.loss_target_smooth_kernel > 1:
            if self.loss_target_smooth_kernel % 2 == 0:
                raise ValueError(
                    "loss_target_smooth_kernel must be odd to preserve sequence length, "
                    f"got {self.loss_target_smooth_kernel}"
                )
            if self.loss_target_smooth_type not in ("gaussian", "uniform"):
                raise ValueError(
                    "loss_target_smooth_type must be 'gaussian' or 'uniform', "
                    f"got {self.loss_target_smooth_type}"
                )
            self._loss_smooth_kernel_1d = self._build_loss_smooth_kernel()

        self.async_streaming = self._cfg.get("async_streaming", False)
        self.streaming_mode = self._cfg.get("streaming_mode", False)
        if self.batch_chunk_replace_probability > 0.0 and not self.streaming_mode:
            raise ValueError("batch_chunk_replace_augmentation requires streaming_mode=True")
        if self.streaming_mode:
            # Validate streaming parameters once at initialization for streaming models
            self.sortformer_modules._check_streaming_parameters()
        self.save_hyperparameters("cfg")
        self._init_eval_metrics()
        speaker_inds = list(range(self._cfg.max_num_of_spks))
        self.speaker_permutations = torch.tensor(list(itertools.permutations(speaker_inds)))  # Get all permutations

        self.max_batch_dur = self._cfg.get("max_batch_dur", 20000)
        self.concat_and_pad_script = torch.jit.script(self.sortformer_modules.concat_and_pad)

    def _init_batch_noise_augmentation(self):
        """Initialize waveform-level in-batch babble augmentation settings."""
        config = self._cfg.get("batch_noise_augmentation", None)
        self.batch_noise_probability = 0.0
        self.batch_noise_min_num_samples = 2
        self.batch_noise_max_num_samples = 3
        self.batch_noise_min_snr_db = 30.0
        self.batch_noise_max_snr_db = 40.0

        if config is None or config is False:
            return

        self.batch_noise_probability = float(config.get("probability", 0.5))
        self.batch_noise_min_num_samples = config.get("min_num_noise_samples", 2)
        self.batch_noise_max_num_samples = config.get("max_num_noise_samples", 3)
        self.batch_noise_min_snr_db = float(config.get("min_snr_db", 30.0))
        self.batch_noise_max_snr_db = float(config.get("max_snr_db", 40.0))

        if not 0.0 <= self.batch_noise_probability <= 1.0:
            raise ValueError(
                f"batch_noise_augmentation.probability must be in [0, 1], "
                f"got {self.batch_noise_probability}"
            )
        for name, value in (
            ("min_num_noise_samples", self.batch_noise_min_num_samples),
            ("max_num_noise_samples", self.batch_noise_max_num_samples),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"batch_noise_augmentation.{name} must be an integer, got {value!r}")
            if value < 1:
                raise ValueError(f"batch_noise_augmentation.{name} must be >= 1, got {value}")
        if self.batch_noise_min_num_samples > self.batch_noise_max_num_samples:
            raise ValueError(
                "batch_noise_augmentation.min_num_noise_samples must be <= "
                "batch_noise_augmentation.max_num_noise_samples"
            )
        if not math.isfinite(self.batch_noise_min_snr_db) or not math.isfinite(self.batch_noise_max_snr_db):
            raise ValueError("batch_noise_augmentation SNR bounds must be finite")
        if self.batch_noise_min_snr_db > self.batch_noise_max_snr_db:
            raise ValueError(
                "batch_noise_augmentation.min_snr_db must be <= batch_noise_augmentation.max_snr_db"
            )

    def _init_batch_chunk_replace_augmentation(self):
        """Initialize streaming in-batch chunk replacement settings."""
        config = self._cfg.get("batch_chunk_replace_augmentation", None)
        self.batch_chunk_replace_probability = 0.0
        self.batch_chunk_replace_min_num_chunks = 1
        self.batch_chunk_replace_max_num_chunks = 3
        self.batch_chunk_replace_num_preserved_chunks = 2
        self._batch_chunk_replace_warned_missing_speaker_ids = False

        if config is None or config is False:
            return

        self.batch_chunk_replace_probability = float(config.get("probability", 0.25))
        self.batch_chunk_replace_min_num_chunks = config.get("min_num_chunks", 1)
        self.batch_chunk_replace_max_num_chunks = config.get("max_num_chunks", 3)
        self.batch_chunk_replace_num_preserved_chunks = config.get("num_preserved_chunks", 2)

        if not 0.0 <= self.batch_chunk_replace_probability <= 1.0:
            raise ValueError(
                f"batch_chunk_replace_augmentation.probability must be in [0, 1], "
                f"got {self.batch_chunk_replace_probability}"
            )
        for name, value, minimum in (
            ("min_num_chunks", self.batch_chunk_replace_min_num_chunks, 1),
            ("max_num_chunks", self.batch_chunk_replace_max_num_chunks, 1),
            ("num_preserved_chunks", self.batch_chunk_replace_num_preserved_chunks, 0),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"batch_chunk_replace_augmentation.{name} must be an integer, got {value!r}")
            if value < minimum:
                raise ValueError(
                    f"batch_chunk_replace_augmentation.{name} must be >= {minimum}, got {value}"
                )
        if self.batch_chunk_replace_min_num_chunks > self.batch_chunk_replace_max_num_chunks:
            raise ValueError(
                "batch_chunk_replace_augmentation.min_num_chunks must be <= "
                "batch_chunk_replace_augmentation.max_num_chunks"
            )

    def _init_loss_weights(self):
        pil_weight = self._cfg.get("pil_weight", 0.0)
        ats_weight = self._cfg.get("ats_weight", 1.0)
        total_weight = pil_weight + ats_weight
        if total_weight == 0:
            raise ValueError(f"weights for PIL {pil_weight} and ATS {ats_weight} cannot sum to 0")
        self.pil_weight = pil_weight / total_weight
        self.ats_weight = ats_weight / total_weight
        # PIL-aligned logit ranking on clean exactly-one-speaker frames. This is a raw
        # multiplier on top of the normalized PIL/ATS combination; 0 disables the loss.
        self.rank_weight = float(self._cfg.get("rank_weight", 0.0))
        if self.rank_weight < 0.0:
            raise ValueError(f"rank_weight must be >= 0, got {self.rank_weight}")
        self.rank_margin = float(self._cfg.get("rank_margin", 0.0))
        if self.rank_margin < 0.0:
            raise ValueError(f"rank_margin must be >= 0, got {self.rank_margin}")
        # Number of output frames excluded on each side of every target-speaker
        # transition. Zero keeps every valid exactly-one-speaker frame.
        self.rank_collar_frames = self._cfg.get("rank_collar_frames", 0)
        if not isinstance(self.rank_collar_frames, int) or isinstance(self.rank_collar_frames, bool):
            raise TypeError(
                "rank_collar_frames must be a non-negative integer, "
                f"got {type(self.rank_collar_frames).__name__}: {self.rank_collar_frames}"
            )
        if self.rank_collar_frames < 0:
            raise ValueError(f"rank_collar_frames must be >= 0, got {self.rank_collar_frames}")
        # PIL-aligned BCE restricted to non-silent frames. The per-frame BCE is
        # averaged over the fixed speaker slots, then globally averaged over eligible
        # frames across DDP workers.
        self.speech_bce_weight = float(self._cfg.get("speech_bce_weight", 0.0))
        if self.speech_bce_weight < 0.0:
            raise ValueError(f"speech_bce_weight must be >= 0, got {self.speech_bce_weight}")
        self.speech_bce_collar_frames = self._cfg.get("speech_bce_collar_frames", 0)
        if not isinstance(self.speech_bce_collar_frames, int) or isinstance(
            self.speech_bce_collar_frames, bool
        ):
            raise TypeError(
                "speech_bce_collar_frames must be a non-negative integer, "
                f"got {type(self.speech_bce_collar_frames).__name__}: {self.speech_bce_collar_frames}"
            )
        if self.speech_bce_collar_frames < 0:
            raise ValueError(
                f"speech_bce_collar_frames must be >= 0, got {self.speech_bce_collar_frames}"
            )
        # PIL-aligned focal BCE on stable interiors of positive and negative regions.
        # Positive and negative terms share one global eligible frame/channel denominator,
        # preserving their natural class-frequency imbalance.
        self.interior_focal_weight = float(self._cfg.get("interior_focal_weight", 0.0))
        if self.interior_focal_weight < 0.0:
            raise ValueError(
                f"interior_focal_weight must be >= 0, got {self.interior_focal_weight}"
            )
        self.interior_focal_gamma = float(self._cfg.get("interior_focal_gamma", 2.0))
        if self.interior_focal_gamma < 0.0:
            raise ValueError(
                f"interior_focal_gamma must be >= 0, got {self.interior_focal_gamma}"
            )
        self.interior_focal_positive_radius = self._cfg.get(
            "interior_focal_positive_radius", 0
        )
        self.interior_focal_negative_radius = self._cfg.get(
            "interior_focal_negative_radius", 0
        )
        for name, value in (
            ("interior_focal_positive_radius", self.interior_focal_positive_radius),
            ("interior_focal_negative_radius", self.interior_focal_negative_radius),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{name} must be a non-negative integer, "
                    f"got {type(value).__name__}: {value}"
                )
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        # Soft-purity focal BCE. Center targets weight their positive/negative terms,
        # and local target purity smoothly suppresses boundaries and short regions.
        self.purity_focal_weight = float(self._cfg.get("purity_focal_weight", 0.0))
        if self.purity_focal_weight < 0.0:
            raise ValueError(
                f"purity_focal_weight must be >= 0, got {self.purity_focal_weight}"
            )
        self.purity_focal_gamma = float(self._cfg.get("purity_focal_gamma", 2.0))
        if self.purity_focal_gamma < 0.0:
            raise ValueError(
                f"purity_focal_gamma must be >= 0, got {self.purity_focal_gamma}"
            )
        self.purity_focal_power = float(self._cfg.get("purity_focal_power", 2.0))
        if self.purity_focal_power < 0.0:
            raise ValueError(
                f"purity_focal_power must be >= 0, got {self.purity_focal_power}"
            )
        self.purity_focal_positive_radius = self._cfg.get(
            "purity_focal_positive_radius", 0
        )
        self.purity_focal_negative_radius = self._cfg.get(
            "purity_focal_negative_radius", 0
        )
        for name, value in (
            ("purity_focal_positive_radius", self.purity_focal_positive_radius),
            ("purity_focal_negative_radius", self.purity_focal_negative_radius),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{name} must be a non-negative integer, "
                    f"got {type(value).__name__}: {value}"
                )
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        # pairwise_ats_weight and self_ats_weight are applied as raw multipliers on top of the
        # normalized PIL/ATS combination (intentionally NOT normalized with them).
        self.pairwise_ats_weight = self._cfg.get("pairwise_ats_weight", 0.0)
        # Temperature for the pairwise-ATS loss: sharpens predictions (T < 1) before the survival
        # product so faint/low-confidence hedges don't accumulate into a confident onset (relaxing
        # the penalty on tentative onsets) while confident out-of-order onsets stay suppressible;
        # T == 1.0 disables.
        self.pairwise_ats_temperature = float(self._cfg.get("pairwise_ats_temperature", 1.0))
        if self.pairwise_ats_temperature <= 0:
            raise ValueError(
                f"pairwise_ats_temperature must be > 0, got {self.pairwise_ats_temperature}"
            )
        self.self_ats_weight = self._cfg.get("self_ats_weight", 0.0)
        # Auxiliary supervised loss on the cumulative number of arrived (onset-so-far) speakers.
        # GT-anchored and assignment-free; it turns a brief early-onset "blip" into a persistent
        # arrival-schedule error (a Wasserstein/EMD-style cost), which counters the pairwise-ATS
        # "early blip" cheat. Raw multiplier on top of the normalized PIL/ATS combination
        # (intentionally NOT normalized with them); 0 disables.
        self.spkcount_weight = self._cfg.get("spkcount_weight", 0.0)
        # Temperature for the spkcount loss: sharpens predictions (T < 1) before the survival
        # product so faint/low-confidence hedges don't inflate the cumulative arrival count;
        # T == 1.0 disables.
        self.spkcount_temperature = float(self._cfg.get("spkcount_temperature", 1.0))
        if self.spkcount_temperature <= 0:
            raise ValueError(f"spkcount_temperature must be > 0, got {self.spkcount_temperature}")
        # Assignment-free side task with three mutually exclusive classes: silence,
        # single-speaker speech, and overlapped speech (two or more active speakers).
        # It can operate on either a dedicated hidden-state head or the speaker predictions.
        self.activity_weight = float(self._cfg.get("activity_weight", 0.0))
        if self.activity_weight < 0.0:
            raise ValueError(f"activity_weight must be >= 0, got {self.activity_weight}")
        # Source for the three-class activity loss:
        #   'aux_head': classify post-encoder hidden states with a dedicated head.
        #   'speaker_preds': derive silence/exactly-one/overlap probabilities directly
        #       from the existing per-speaker activity probabilities.
        self.activity_loss_mode = str(self._cfg.get("activity_loss_mode", "aux_head")).lower()
        if self.activity_loss_mode not in ("aux_head", "speaker_preds"):
            raise ValueError(
                "activity_loss_mode must be 'aux_head' or 'speaker_preds', "
                f"got '{self.activity_loss_mode}'"
            )
        # PIL-aligned local speaker-presence loss. For every output frame and speaker
        # channel, max-pool predictions and PIL-matched targets over [t-d, t+d].
        # This makes a short high-confidence false speaker incur a neighborhood-wide
        # penalty while preserving short ground-truth speakers.
        self.presence_weight = float(self._cfg.get("presence_weight", 0.0))
        if self.presence_weight < 0.0:
            raise ValueError(f"presence_weight must be >= 0, got {self.presence_weight}")
        self.presence_window_radius = self._cfg.get("presence_window_radius", 3)
        if not isinstance(self.presence_window_radius, int) or isinstance(
            self.presence_window_radius, bool
        ):
            raise TypeError(
                "presence_window_radius must be a non-negative integer, "
                f"got {type(self.presence_window_radius).__name__}: {self.presence_window_radius}"
            )
        if self.presence_window_radius < 0:
            raise ValueError(f"presence_window_radius must be >= 0, got {self.presence_window_radius}")
        self.presence_negative_margin = float(self._cfg.get("presence_negative_margin", 0.4))
        if not 0.0 <= self.presence_negative_margin < 1.0:
            raise ValueError(
                f"presence_negative_margin must be in [0, 1), got {self.presence_negative_margin}"
            )
        # PIL-aligned soft Dice loss over every sample/speaker channel, including
        # channels with no target activity so phantom speakers are penalized.
        self.dice_weight = float(self._cfg.get("dice_weight", 0.0))
        if self.dice_weight < 0.0:
            raise ValueError(f"dice_weight must be >= 0, got {self.dice_weight}")
        self.dice_min_target_frames = self._cfg.get("dice_min_target_frames", 15)
        if not isinstance(self.dice_min_target_frames, int) or isinstance(
            self.dice_min_target_frames, bool
        ):
            raise TypeError(
                "dice_min_target_frames must be a positive integer, "
                f"got {type(self.dice_min_target_frames).__name__}: {self.dice_min_target_frames}"
            )
        if self.dice_min_target_frames < 1:
            raise ValueError(
                f"dice_min_target_frames must be >= 1, got {self.dice_min_target_frames}"
            )
        self.dice_duration_gamma = float(self._cfg.get("dice_duration_gamma", 0.0))
        if not 0.0 <= self.dice_duration_gamma <= 1.0:
            raise ValueError(
                "dice_duration_gamma must be in [0, 1], "
                f"got {self.dice_duration_gamma}"
            )
        # Hard-negative BCE for phantom speakers. Only PIL-aligned channels with no
        # target activity are eligible, and only predictions above the threshold are
        # penalized. Each channel is reduced over its selected frames before the channel
        # losses are summed and divided by the fixed number of model speaker slots.
        self.phantom_weight = float(self._cfg.get("phantom_weight", 0.0))
        if self.phantom_weight < 0.0:
            raise ValueError(f"phantom_weight must be >= 0, got {self.phantom_weight}")
        self.phantom_threshold = float(self._cfg.get("phantom_threshold", 0.25))
        if not 0.0 <= self.phantom_threshold < 1.0:
            raise ValueError(f"phantom_threshold must be in [0, 1), got {self.phantom_threshold}")
        self.phantom_logmeanexp = self._cfg.get("phantom_logmeanexp", False)
        if not isinstance(self.phantom_logmeanexp, bool):
            raise TypeError(
                f"phantom_logmeanexp must be a boolean, got {self.phantom_logmeanexp!r}"
            )
        self.phantom_logmeanexp_temperature = float(
            self._cfg.get("phantom_logmeanexp_temperature", 0.5)
        )
        if self.phantom_logmeanexp_temperature <= 0.0:
            raise ValueError(
                "phantom_logmeanexp_temperature must be > 0, "
                f"got {self.phantom_logmeanexp_temperature}"
            )
        # Positive channel-level counterpart to phantom loss. For sufficiently long
        # target-present speakers, active-frame logits are pooled with normalized
        # log-mean-exp and trained as one positive speaker-existence decision.
        self.speaker_existence_weight = float(
            self._cfg.get("speaker_existence_weight", 0.0)
        )
        if self.speaker_existence_weight < 0.0:
            raise ValueError(
                "speaker_existence_weight must be >= 0, "
                f"got {self.speaker_existence_weight}"
            )
        self.speaker_existence_min_frames = self._cfg.get(
            "speaker_existence_min_frames", 1
        )
        if not isinstance(self.speaker_existence_min_frames, int) or isinstance(
            self.speaker_existence_min_frames, bool
        ):
            raise TypeError(
                "speaker_existence_min_frames must be a positive integer, "
                f"got {type(self.speaker_existence_min_frames).__name__}: "
                f"{self.speaker_existence_min_frames}"
            )
        if self.speaker_existence_min_frames < 1:
            raise ValueError(
                "speaker_existence_min_frames must be >= 1, "
                f"got {self.speaker_existence_min_frames}"
            )
        self.speaker_existence_temperature = float(
            self._cfg.get("speaker_existence_temperature", 0.5)
        )
        if self.speaker_existence_temperature <= 0.0:
            raise ValueError(
                "speaker_existence_temperature must be > 0, "
                f"got {self.speaker_existence_temperature}"
            )
        self.speaker_existence_threshold = float(
            self._cfg.get("speaker_existence_threshold", 1.0)
        )
        if not 0.0 < self.speaker_existence_threshold <= 1.0:
            raise ValueError(
                "speaker_existence_threshold must be in (0, 1], "
                f"got {self.speaker_existence_threshold}"
            )
        self.speaker_existence_target = str(
            self._cfg.get("speaker_existence_target", "pil")
        ).lower()
        if self.speaker_existence_target not in ("pil", "ats"):
            raise ValueError(
                "speaker_existence_target must be 'pil' or 'ats', "
                f"got '{self.speaker_existence_target}'"
            )
        # Entry-focused companion to phantom loss. It penalizes only the first
        # fixed-size streaming chunk where an empty channel exceeds the entry threshold.
        self.phantom_entry_weight = float(self._cfg.get("phantom_entry_weight", 0.0))
        if self.phantom_entry_weight < 0.0:
            raise ValueError(
                f"phantom_entry_weight must be >= 0, got {self.phantom_entry_weight}"
            )
        self.phantom_entry_threshold = float(
            self._cfg.get("phantom_entry_threshold", 0.5)
        )
        if not 0.0 <= self.phantom_entry_threshold < 1.0:
            raise ValueError(
                "phantom_entry_threshold must be in [0, 1), "
                f"got {self.phantom_entry_threshold}"
            )
        # Hard-negative BCE for speaker activity before the speaker's first PIL-aligned
        # target arrival. Predictions within the grace window before arrival are tolerated;
        # never-active target channels remain eligible throughout the valid sequence.
        self.prearrival_weight = float(self._cfg.get("prearrival_weight", 0.0))
        if self.prearrival_weight < 0.0:
            raise ValueError(f"prearrival_weight must be >= 0, got {self.prearrival_weight}")
        self.prearrival_threshold = float(self._cfg.get("prearrival_threshold", 0.25))
        if not 0.0 <= self.prearrival_threshold < 1.0:
            raise ValueError(
                f"prearrival_threshold must be in [0, 1), got {self.prearrival_threshold}"
            )
        self.prearrival_grace_frames = self._cfg.get("prearrival_grace_frames", 0)
        if not isinstance(self.prearrival_grace_frames, int) or isinstance(
            self.prearrival_grace_frames, bool
        ):
            raise TypeError(
                "prearrival_grace_frames must be a non-negative integer, "
                f"got {type(self.prearrival_grace_frames).__name__}: {self.prearrival_grace_frames}"
            )
        if self.prearrival_grace_frames < 0:
            raise ValueError(
                f"prearrival_grace_frames must be >= 0, got {self.prearrival_grace_frames}"
            )
        self.prearrival_logmeanexp = self._cfg.get("prearrival_logmeanexp", False)
        if not isinstance(self.prearrival_logmeanexp, bool):
            raise TypeError(
                f"prearrival_logmeanexp must be a boolean, got {self.prearrival_logmeanexp!r}"
            )
        self.prearrival_logmeanexp_temperature = float(
            self._cfg.get("prearrival_logmeanexp_temperature", 0.5)
        )
        if self.prearrival_logmeanexp_temperature <= 0.0:
            raise ValueError(
                "prearrival_logmeanexp_temperature must be > 0, "
                f"got {self.prearrival_logmeanexp_temperature}"
            )
        # Distance used by the self-ATS loss: 'bce' (default; matches the other losses but has
        # an entropy floor, so the logged value is > 0 even when perfectly self-consistent) or
        # 'mse' (no floor: value is 0 when self-consistent, and a gentler bounded gradient).
        self.self_ats_metric = str(self._cfg.get("self_ats_metric", "bce")).lower()
        if self.self_ats_metric not in ("bce", "mse"):
            raise ValueError(f"self_ats_metric must be 'bce' or 'mse', got '{self.self_ats_metric}'")
        # Temperature for the self-ATS loss, applied symmetrically to both the probs side and
        # the target. T < 1 sharpens predictions toward 0/1 (countering the softening a soft
        # self-target induces and amplifying the reordering gradient by ~1/T) while preserving
        # the zero-loss / zero-gradient-when-sorted property; T == 1.0 disables sharpening.
        self.self_ats_temperature = float(self._cfg.get("self_ats_temperature", 1.0))
        if self.self_ats_temperature <= 0:
            raise ValueError(f"self_ats_temperature must be > 0, got {self.self_ats_temperature}")
        # Arrival-time tolerance (in output frames) for ATS target construction. Speakers whose
        # arrival times differ by at most this value are treated as tied and may be reordered by
        # the model (the Hungarian step then uses predictions to assign them). Relaxes ordering
        # only for near-simultaneous onsets, reducing label-noise harm; 0 = exact arrival sort.
        self.ats_tolerance = self._cfg.get("ats_tolerance", 0)
        # match_metric: metric used by the Hungarian assignment when building both PIL and ATS targets.
        #   'dot_product' (default): scores only true-positive overlap (label==1 & high prob) and
        #       ignores the silence/negative class, so it does not penalize false alarms. Bounded
        #       and numerically stable; matches the previous behavior.
        #   'accuracy': accounts for both the positive and negative classes (soft accuracy).
        #   'bce': matches the BCE training objective directly (most assignment-consistent), at
        #       the cost of the usual BCE numerical sensitivity near 0/1 probabilities.
        self.match_metric = str(self._cfg.get("match_metric", "dot_product")).lower()
        if self.match_metric not in ("dot_product", "accuracy", "bce"):
            raise ValueError(
                f"match_metric must be 'dot_product', 'accuracy', or 'bce', got '{self.match_metric}'"
            )

    def _init_activity_auxiliary_heads(self):
        """Create the training-only three-class activity head when enabled."""
        if self.activity_weight > 0.0 and self.activity_loss_mode == "aux_head":
            self.sortformer_modules.init_activity_head()

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

    def __setup_dataloader_from_config(self, config, subsampling_factor=None):
        sf = subsampling_factor if subsampling_factor is not None else self.output_subsampling_factor

        # Switch to lhotse dataloader if specified in the config
        if config.get("use_lhotse"):
            if subsampling_factor is not None:
                from omegaconf import OmegaConf

                config = DictConfig(OmegaConf.to_container(config, resolve=True))
                config.subsampling_factor = sf
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
            subsampling_factor=sf,
            global_rank=global_rank,
            soft_targets=config.soft_targets if 'soft_targets' in config else False,
            device=self.device,
            subsegment_mode=config.get('subsegment_mode', False),
            subsegment_min_len_sec=config.get('subsegment_min_len_sec', 15.0),
            subsegment_two_chunks_rate=config.get('subsegment_two_chunks_rate', 0.0),
            subsegment_min_chunk_len_sec=config.get('subsegment_min_chunk_len_sec', 10.0),
            subsegment_margin_frames=config.get('subsegment_margin_frames', 0),
            subsegment_nspk_bias=config.get('subsegment_nspk_bias', 1.0),
            subsegment_min_first_spk_frames=config.get('subsegment_min_first_spk_frames', 50),
            subsegment_boundary_silence_frames=config.get('subsegment_boundary_silence_frames', 10),
            subsegment_preload_sec=config.get('subsegment_preload_sec', 0.0),
            opus_roundtrip_prob=config.get('opus_roundtrip_prob', 0.0),
            opus_roundtrip_compression_level=config.get('opus_roundtrip_compression_level', None),
            validate_manifest_paths=config.get('validate_manifest_paths', True),
        )

        self.data_collection = dataset.collection
        self.collate_ds = dataset

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
                trainer=self._trainer,
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

    def setup_validation_data(self, val_data_layer_config: Optional[Union[DictConfig, Dict]]):
        sf = 1 if self.val_upsample_preds else None
        self._validation_dl = self.__setup_dataloader_from_config(
            config=val_data_layer_config, subsampling_factor=sf,
        )

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        sf = 1 if self.val_upsample_preds else None
        self._test_dl = self.__setup_dataloader_from_config(
            config=test_data_config, subsampling_factor=sf,
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

    _PREENC_UPSAMPLE_MODES = ("single_preenc", "single_preenc_full", "single_preenc_mlp")

    def _call_pre_encode(self, x, lengths):
        """Call ``self.encoder.pre_encode`` with correct input layout.

        Most pre-encoders (``ConvSubsampling``, ``StackingSubsampling``,
        ``nn.Linear``) expect *time-major* input ``(B, T, C)``.
        ``FeatureStacking`` is the exception — it expects ``(B, C, T)``
        and transposes internally.  This helper hides that difference so
        every call site can pass ``(B, T, C)`` uniformly.

        Returns:
            ``(features, lengths)`` with features in ``(B, T', D)`` layout.
        """
        from nemo.collections.asr.parts.submodules.subsampling import FeatureStacking

        if isinstance(self.encoder.pre_encode, torch.nn.Linear):
            return self.encoder.pre_encode(x), lengths
        if isinstance(self.encoder.pre_encode, FeatureStacking):
            return self.encoder.pre_encode(x.transpose(1, 2), lengths)
        return self.encoder.pre_encode(x=x, lengths=lengths)

    def frontend_encoder(self, processed_signal, processed_signal_length, bypass_pre_encode: bool = False):
        """
        Generate encoder outputs from frontend encoder.

        When the upsample mode requires pre-encoder features and
        ``bypass_pre_encode`` is False, the encoder call is split so that
        the ConvSubsampling output is captured before the conformer layers.

        Args:
            processed_signal (torch.Tensor):
                tensor containing audio-feature (mel spectrogram, mfcc, etc.).
            processed_signal_length (torch.Tensor):
                tensor containing lengths of audio signal in integers.
            bypass_pre_encode (bool):
                if True, ``processed_signal`` already contains pre-encoded
                embeddings and the subsampling step is skipped.

        Returns:
            emb_seq (torch.Tensor):
                tensor containing encoder outputs.
            emb_seq_length (torch.Tensor):
                tensor containing lengths of encoder outputs.
            pre_encode_feats (torch.Tensor or None):
                ConvSubsampling output at ``fc_d_model`` dimensionality,
                or None when not needed / not available.
        """
        # Spec augment is not applied during evaluation/testing.
        # In streaming mode (bypass_pre_encode=True), spec augment is applied per-chunk
        # in forward_streaming_step before pre_encode, so skip it here.
        if self.spec_augmentation is not None and self.training and not bypass_pre_encode:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        need_preenc = (
            self.upsample_factor > 1
            and self.upsample_mode in self._PREENC_UPSAMPLE_MODES
            and not bypass_pre_encode
        )
        pre_encode_feats = None

        if need_preenc:
            processed_signal_t = processed_signal.transpose(1, 2)
            pre_encode_feats, pre_encode_lengths = self._call_pre_encode(
                processed_signal_t, processed_signal_length
            )
            pre_encode_lengths = pre_encode_lengths.to(torch.int64)
            emb_seq, emb_seq_length = self.encoder(
                audio_signal=pre_encode_feats,
                length=pre_encode_lengths,
                bypass_pre_encode=True,
            )
        else:
            emb_seq, emb_seq_length = self.encoder(
                audio_signal=processed_signal,
                length=processed_signal_length,
                bypass_pre_encode=bypass_pre_encode,
            )

        emb_seq = emb_seq.transpose(1, 2)
        if self.sortformer_modules.encoder_proj is not None:
            emb_seq = self.sortformer_modules.encoder_proj(emb_seq)
        return emb_seq, emb_seq_length, pre_encode_feats

    def forward_infer(
        self,
        emb_seq,
        emb_seq_length,
        pre_encode_feats=None,
    ):
        """
        The main forward pass for diarization for offline diarization inference.

        Args:
            emb_seq (torch.Tensor): Tensor containing FastConformer encoder states (embedding vectors).
                Shape: (batch_size, diar_frame_count, emb_dim)
            emb_seq_length (torch.Tensor): Tensor containing lengths of FastConformer encoder states.
                Shape: (batch_size,)
            pre_encode_feats (torch.Tensor, optional): Pre-encoder embeddings from
                ConvSubsampling, passed through to the upsampler for ``single_preenc*`` modes.
                Shape: (batch_size, diar_frame_count, fc_d_model)

        Returns:
            tuple:
                - Speaker probabilities of shape (batch_size, frames, num_speakers).
                - Raw speaker logits with the same shape.
                - Three-class activity logits, or None when not requested.
        """
        encoder_mask = self.sortformer_modules.length_to_mask(emb_seq_length, emb_seq.shape[1])
        trans_emb_seq = self.transformer_encoder(encoder_states=emb_seq, encoder_mask=encoder_mask)
        trans_emb_seq = self.sortformer_modules.upsample_hidden(trans_emb_seq, pre_encode_feats=pre_encode_feats)
        if self.upsample_factor > 1:
            encoder_mask = encoder_mask.repeat_interleave(self.upsample_factor, dim=1)
        speaker_logits = self.sortformer_modules.forward_speaker_logits(trans_emb_seq)
        _preds = torch.sigmoid(speaker_logits)
        preds = _preds * encoder_mask.unsqueeze(-1)
        activity_logits = self.sortformer_modules.forward_activity_logits(trans_emb_seq)
        return preds, speaker_logits, activity_logits

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
            preds, _, _ = self.forward(audio_signal=batch[0], audio_signal_length=batch[1])
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
            'subsegment_nspk_bias': config.get('subsegment_nspk_bias', 1.0),
            'subsegment_min_first_spk_frames': config.get('subsegment_min_first_spk_frames', 50),
            'subsegment_boundary_silence_frames': config.get('subsegment_boundary_silence_frames', 10),
            'subsegment_preload_sec': config.get('subsegment_preload_sec', 0.0),
            'opus_roundtrip_prob': config.get('opus_roundtrip_prob', 0.0),
            'opus_roundtrip_compression_level': config.get('opus_roundtrip_compression_level', None),
            'validate_manifest_paths': config.get('validate_manifest_paths', True),
        }
        temporary_datalayer = self.__setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    def _get_batch_active_speech_rms(self, audio_signal, audio_signal_length, targets, target_lens):
        """Calculate one active-speech waveform RMS value per batch sample."""
        if audio_signal.ndim != 2:
            raise ValueError(f"batch noise augmentation expects audio shape (B, T), got {audio_signal.shape}")
        if targets.ndim != 3:
            raise ValueError(f"batch noise augmentation expects target shape (B, T, S), got {targets.shape}")
        if audio_signal.shape[0] != targets.shape[0]:
            raise ValueError(
                f"audio and target batch sizes must match, got {audio_signal.shape[0]} and {targets.shape[0]}"
            )

        batch_size, max_audio_len = audio_signal.shape
        max_target_len = targets.shape[1]
        device = audio_signal.device
        compute_dtype = (
            torch.float32 if audio_signal.dtype in (torch.float16, torch.bfloat16) else audio_signal.dtype
        )
        if not audio_signal.is_floating_point():
            compute_dtype = torch.float32

        audio_lens = audio_signal_length.to(device=device, dtype=torch.long).clamp(min=0, max=max_audio_len)
        target_lens = target_lens.to(device=device, dtype=torch.long).clamp(min=0, max=max_target_len)
        sample_positions = torch.arange(max_audio_len, device=device).unsqueeze(0)
        valid_audio_mask = sample_positions < audio_lens.unsqueeze(1)

        if max_audio_len == 0 or max_target_len == 0:
            return torch.zeros(batch_size, device=device, dtype=compute_dtype), valid_audio_mask

        target_positions = torch.arange(max_target_len, device=device).unsqueeze(0)
        valid_target_mask = target_positions < target_lens.unsqueeze(1)
        frame_activity = (targets.to(device=device) >= 0.5).any(dim=-1) & valid_target_mask

        # Each target frame covers a fixed number of waveform samples. For example,
        # a 10 ms feature hop (160 samples) with 8x subsampling gives 1280 samples
        # per target frame. Repeat each frame's activity over exactly those samples.
        samples_per_target_frame = self.preprocessor.hop_length * self.output_subsampling_factor
        active_audio_mask = frame_activity.repeat_interleave(samples_per_target_frame, dim=1)
        if active_audio_mask.shape[1] < max_audio_len:
            missing_samples = max_audio_len - active_audio_mask.shape[1]
            inactive_padding = torch.zeros(batch_size, missing_samples, dtype=torch.bool, device=device)
            active_audio_mask = torch.cat((active_audio_mask, inactive_padding), dim=1)
        active_audio_mask = active_audio_mask[:, :max_audio_len] & valid_audio_mask

        waveform = audio_signal.to(dtype=compute_dtype)
        active_sample_count = active_audio_mask.sum(dim=1)
        active_energy = torch.where(active_audio_mask, waveform.square(), 0.0).sum(dim=1)
        active_rms = torch.sqrt(active_energy / active_sample_count.clamp_min(1).to(compute_dtype))
        active_rms = active_rms.masked_fill(active_sample_count == 0, 0.0)
        return active_rms, valid_audio_mask

    def _apply_batch_noise_augmentation(self, audio_signal, audio_signal_length, targets, target_lens):
        """Mix whole in-batch waveforms using independently RMS-scaled donor gains."""
        if self.batch_noise_probability <= 0.0 or audio_signal.shape[0] < 2:
            return audio_signal

        active_rms, valid_audio_mask = self._get_batch_active_speech_rms(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
            targets=targets,
            target_lens=target_lens,
        )
        rms_epsilon = torch.finfo(active_rms.dtype).eps
        eligible = active_rms > rms_epsilon
        eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
        if eligible_indices.numel() < 2:
            return audio_signal

        if self.batch_noise_probability >= 1.0:
            augment = eligible
        else:
            augment = torch.rand(audio_signal.shape[0], device=audio_signal.device) < self.batch_noise_probability
            augment &= eligible

        waveform = audio_signal.to(dtype=active_rms.dtype)
        augmented_audio = None
        for target_idx in torch.nonzero(augment, as_tuple=False).flatten().tolist():
            donor_candidates = eligible_indices[eligible_indices != target_idx]
            max_num_samples = min(self.batch_noise_max_num_samples, donor_candidates.numel())
            if max_num_samples == 0:
                continue
            min_num_samples = min(self.batch_noise_min_num_samples, max_num_samples)
            if min_num_samples == max_num_samples:
                num_noise_samples = max_num_samples
            else:
                num_noise_samples = int(
                    torch.randint(
                        min_num_samples,
                        max_num_samples + 1,
                        size=(1,),
                        device=audio_signal.device,
                    ).item()
                )

            if num_noise_samples < donor_candidates.numel():
                donor_order = torch.randperm(donor_candidates.numel(), device=audio_signal.device)
                donor_indices = donor_candidates[donor_order[:num_noise_samples]]
            else:
                donor_indices = donor_candidates

            if self.batch_noise_min_snr_db == self.batch_noise_max_snr_db:
                snr_db = torch.full(
                    (num_noise_samples,),
                    self.batch_noise_min_snr_db,
                    device=audio_signal.device,
                    dtype=active_rms.dtype,
                )
            else:
                snr_db = torch.empty(
                    num_noise_samples, device=audio_signal.device, dtype=active_rms.dtype
                ).uniform_(self.batch_noise_min_snr_db, self.batch_noise_max_snr_db)

            gains = (
                active_rms[target_idx]
                / active_rms[donor_indices]
                * torch.pow(10.0, -snr_db / 20.0)
            )
            donor_audio = waveform[donor_indices] * valid_audio_mask[donor_indices].to(active_rms.dtype)
            noise = (donor_audio * gains.unsqueeze(1)).sum(dim=0)
            mixed_audio = waveform[target_idx] + noise * valid_audio_mask[target_idx].to(active_rms.dtype)

            if augmented_audio is None:
                augmented_audio = audio_signal.clone()
            augmented_audio[target_idx] = mixed_audio.to(dtype=audio_signal.dtype)

        return audio_signal if augmented_audio is None else augmented_audio

    @staticmethod
    def _channel_activity_to_speaker_presence(
        channel_activity, channel_speaker_indices, num_speaker_ids
    ):
        """Convert per-channel activity to globally indexed speaker presence."""
        squeeze_batch = channel_activity.ndim == 1
        if squeeze_batch:
            channel_activity = channel_activity.unsqueeze(0)
            channel_speaker_indices = channel_speaker_indices.unsqueeze(0)

        batch_size, num_channels = channel_activity.shape
        presence = torch.zeros(
            batch_size, num_speaker_ids, dtype=torch.bool, device=channel_activity.device
        )
        batch_indices = torch.arange(batch_size, device=channel_activity.device)
        for channel in range(num_channels):
            speaker_indices = channel_speaker_indices[:, channel]
            valid = channel_activity[:, channel] & (speaker_indices >= 0)
            presence[batch_indices[valid], speaker_indices[valid]] = True
        return presence[0] if squeeze_batch else presence

    def _build_chunk_replacement_metadata(
        self,
        processed_signal,
        processed_signal_length,
        targets,
        target_lens,
        speaker_names,
    ):
        """Precompute speaker tensors and pair compatibility once per batch."""
        batch_size, _, max_feature_length = processed_signal.shape
        max_target_length, num_channels = targets.shape[1:]
        feature_lengths = (
            processed_signal_length.detach().long().cpu().clamp(min=0, max=max_feature_length)
        )
        target_lengths = target_lens.detach().long().cpu().clamp(min=0, max=max_target_length)
        target_values = targets.detach().cpu()
        speaker_activity = target_values > _CHUNK_REPLACEMENT_ACTIVITY_THRESHOLD

        feature_frames_per_chunk = (
            self.sortformer_modules.chunk_len * self.sortformer_modules.subsampling_factor
        )
        if feature_frames_per_chunk % self.output_subsampling_factor != 0:
            raise ValueError(
                "Streaming chunk feature length must be divisible by output_subsampling_factor "
                "for batch chunk replacement."
            )
        target_frames_per_chunk = feature_frames_per_chunk // self.output_subsampling_factor

        speaker_id_to_index = {}
        channel_speaker_indices = torch.full((batch_size, num_channels), -1, dtype=torch.long)
        valid_samples = torch.ones(batch_size, dtype=torch.bool)
        for batch_index, sample_names in enumerate(speaker_names):
            if not isinstance(sample_names, (list, tuple)):
                valid_samples[batch_index] = False
                continue
            for channel in range(min(num_channels, len(sample_names))):
                speaker_id = sample_names[channel]
                if speaker_id is None or speaker_id == "":
                    continue
                if speaker_id not in speaker_id_to_index:
                    speaker_id_to_index[speaker_id] = len(speaker_id_to_index)
                channel_speaker_indices[batch_index, channel] = speaker_id_to_index[speaker_id]

        valid_target_mask = torch.arange(max_target_length).unsqueeze(0) < target_lengths.unsqueeze(1)
        valid_speaker_activity = speaker_activity & valid_target_mask.unsqueeze(-1)
        full_channel_activity = valid_speaker_activity.any(dim=1)
        valid_samples &= ~((channel_speaker_indices < 0) & full_channel_activity).any(dim=1)
        full_speaker_presence = self._channel_activity_to_speaker_presence(
            full_channel_activity,
            channel_speaker_indices,
            len(speaker_id_to_index),
        )

        num_target_chunks = math.ceil(max_target_length / target_frames_per_chunk)
        padded_target_length = num_target_chunks * target_frames_per_chunk
        padded_targets = torch.nn.functional.pad(
            valid_speaker_activity,
            (0, 0, 0, padded_target_length - max_target_length),
        )
        chunk_channel_activity = padded_targets.reshape(
            batch_size,
            num_target_chunks,
            target_frames_per_chunk,
            num_channels,
        ).any(dim=2)
        chunk_speaker_presence = self._channel_activity_to_speaker_presence(
            chunk_channel_activity.flatten(0, 1),
            channel_speaker_indices.repeat_interleave(num_target_chunks, dim=0),
            len(speaker_id_to_index),
        ).reshape(batch_size, num_target_chunks, len(speaker_id_to_index))

        shared_speakers = (
            full_speaker_presence[:, None, :] & full_speaker_presence[None, :, :]
        ).any(dim=-1)
        compatible_pairs = valid_samples[:, None] & valid_samples[None, :] & ~shared_speakers
        compatible_pairs.fill_diagonal_(False)

        return _ChunkReplacementMetadata(
            target_values=target_values,
            valid_speaker_activity=valid_speaker_activity,
            feature_lengths=feature_lengths,
            target_lengths=target_lengths,
            channel_speaker_indices=channel_speaker_indices,
            full_speaker_presence=full_speaker_presence,
            chunk_speaker_presence=chunk_speaker_presence,
            compatible_pairs=compatible_pairs,
            feature_frames_per_chunk=feature_frames_per_chunk,
            target_frames_per_chunk=target_frames_per_chunk,
        )

    def _candidate_chunk_replacement_plans(self, recipient_index, metadata):
        """Return randomized destination-chunk combinations for one recipient."""
        feature_length = int(metadata.feature_lengths[recipient_index])
        target_length = int(metadata.target_lengths[recipient_index])
        recipient_chunk_count = math.ceil(feature_length / metadata.feature_frames_per_chunk)
        candidates = [
            chunk_index
            for chunk_index in range(
                self.batch_chunk_replace_num_preserved_chunks, recipient_chunk_count
            )
            if chunk_index * metadata.target_frames_per_chunk < target_length
        ]
        max_num_chunks = min(self.batch_chunk_replace_max_num_chunks, len(candidates))
        if max_num_chunks < self.batch_chunk_replace_min_num_chunks:
            return []

        chunk_counts = list(range(self.batch_chunk_replace_min_num_chunks, max_num_chunks + 1))
        random.shuffle(chunk_counts)
        plans = []
        for num_replacements in chunk_counts:
            total_combinations = math.comb(len(candidates), num_replacements)
            if total_combinations <= _MAX_CHUNK_REPLACEMENT_PLANS_PER_COUNT:
                plans_for_count = list(itertools.combinations(candidates, num_replacements))
            else:
                plans_for_count = set()
                while len(plans_for_count) < _MAX_CHUNK_REPLACEMENT_PLANS_PER_COUNT:
                    plans_for_count.add(
                        tuple(sorted(random.sample(candidates, num_replacements)))
                    )
                plans_for_count = list(plans_for_count)
            random.shuffle(plans_for_count)
            plans.extend(plans_for_count)
        return plans

    def _select_chunk_replacement_plan(self, recipient_index, metadata):
        """Select one valid plan while evaluating all donors in parallel."""
        target_length = int(metadata.target_lengths[recipient_index])
        feature_length = int(metadata.feature_lengths[recipient_index])
        original_speaker_count = int(metadata.full_speaker_presence[recipient_index].sum())
        num_speaker_ids = metadata.full_speaker_presence.shape[1]
        recipient_chunk_count = math.ceil(target_length / metadata.target_frames_per_chunk)
        max_output_speakers = min(
            self._cfg.max_num_of_spks, metadata.channel_speaker_indices.shape[1]
        )

        for destination_chunks in self._candidate_chunk_replacement_plans(recipient_index, metadata):
            retained_chunk_mask = torch.ones(recipient_chunk_count, dtype=torch.bool)
            retained_chunk_mask[list(destination_chunks)] = False
            retained_presence = metadata.chunk_speaker_presence[
                recipient_index, :recipient_chunk_count
            ][retained_chunk_mask].any(dim=0)

            inserted_presence = torch.zeros_like(metadata.full_speaker_presence)
            donors_have_data = torch.ones(metadata.feature_lengths.shape[0], dtype=torch.bool)
            for source_rank, destination_chunk in enumerate(destination_chunks):
                destination_feature_start = destination_chunk * metadata.feature_frames_per_chunk
                destination_feature_length = min(
                    metadata.feature_frames_per_chunk,
                    feature_length - destination_feature_start,
                )
                source_feature_end = (
                    source_rank * metadata.feature_frames_per_chunk + destination_feature_length
                )
                donors_have_data &= metadata.feature_lengths >= source_feature_end

                destination_target_start = destination_chunk * metadata.target_frames_per_chunk
                destination_target_length = min(
                    metadata.target_frames_per_chunk,
                    target_length - destination_target_start,
                )
                source_target_start = source_rank * metadata.target_frames_per_chunk
                source_target_end = source_target_start + destination_target_length
                donors_have_data &= metadata.target_lengths >= source_target_end
                if destination_target_length == metadata.target_frames_per_chunk:
                    inserted_presence |= metadata.chunk_speaker_presence[:, source_rank]
                else:
                    source_channel_activity = metadata.valid_speaker_activity[
                        :, source_target_start:source_target_end
                    ].any(dim=1)
                    inserted_presence |= self._channel_activity_to_speaker_presence(
                        source_channel_activity,
                        metadata.channel_speaker_indices,
                        num_speaker_ids,
                    )

            final_presence = inserted_presence | retained_presence.unsqueeze(0)
            final_speaker_count = final_presence.sum(dim=1)
            valid_donors = (
                metadata.compatible_pairs[recipient_index]
                & donors_have_data
                & inserted_presence.any(dim=1)
                & (final_speaker_count > original_speaker_count)
                & (final_speaker_count <= max_output_speakers)
            )
            donor_indices = torch.nonzero(valid_donors, as_tuple=False).flatten().tolist()
            if donor_indices:
                return _ChunkReplacementPlan(
                    recipient_index=recipient_index,
                    donor_index=random.choice(donor_indices),
                    destination_chunks=destination_chunks,
                )
        return None

    def _sample_chunk_replacement_plans(self, metadata):
        """Sample at most one replacement plan per eligible recipient."""
        plans = []
        for recipient_index in range(metadata.feature_lengths.shape[0]):
            if not metadata.compatible_pairs[recipient_index].any():
                continue
            if (
                self.batch_chunk_replace_probability < 1.0
                and random.random() >= self.batch_chunk_replace_probability
            ):
                continue
            plan = self._select_chunk_replacement_plan(recipient_index, metadata)
            if plan is not None:
                plans.append(plan)
        return plans

    def _build_chunk_replaced_targets(
        self,
        recipient_targets,
        donor_targets,
        recipient_speaker_names,
        donor_speaker_names,
        recipient_target_len,
        donor_target_len,
        target_frames_per_chunk,
        destination_chunks,
    ):
        """Build speaker-ID-consistent targets for an explicit chunk replacement plan."""
        recipient_target_len = min(int(recipient_target_len), recipient_targets.shape[0])
        donor_target_len = min(int(donor_target_len), donor_targets.shape[0])
        replaced_targets = recipient_targets.clone()
        for chunk_index in destination_chunks:
            start = chunk_index * target_frames_per_chunk
            replaced_targets[start : start + target_frames_per_chunk] = 0.0

        retained_channels = (
            replaced_targets[:recipient_target_len] > _CHUNK_REPLACEMENT_ACTIVITY_THRESHOLD
        ).any(dim=0)
        used_channels = torch.nonzero(retained_channels, as_tuple=False).flatten().tolist()
        speaker_to_output_channel = {}
        for channel in used_channels:
            if channel >= len(recipient_speaker_names):
                raise _ChunkReplacementInvariantError(
                    f"Recipient active channel {channel} has no RTTM speaker-ID entry"
                )
            speaker_id = recipient_speaker_names[channel]
            if speaker_id is None or speaker_id == "":
                raise _ChunkReplacementInvariantError(
                    f"Recipient active channel {channel} has an empty RTTM speaker ID"
                )
            speaker_to_output_channel[speaker_id] = channel

        recipient_speaker_ids = set(speaker_to_output_channel)
        free_channels = [
            channel for channel in range(recipient_targets.shape[1]) if channel not in used_channels
        ]
        for source_rank, destination_chunk in enumerate(destination_chunks):
            destination_start = destination_chunk * target_frames_per_chunk
            destination_end = min(destination_start + target_frames_per_chunk, recipient_target_len)
            destination_length = destination_end - destination_start
            source_start = source_rank * target_frames_per_chunk
            source_end = source_start + destination_length
            if source_end > donor_target_len:
                raise _ChunkReplacementInvariantError(
                    f"Donor target ends at frame {donor_target_len}, but plan requires {source_end}"
                )
            source_targets = donor_targets[source_start:source_end]
            active_donor_channels = torch.nonzero(
                (source_targets > _CHUNK_REPLACEMENT_ACTIVITY_THRESHOLD).any(dim=0),
                as_tuple=False,
            ).flatten().tolist()
            for source_channel in active_donor_channels:
                if source_channel >= len(donor_speaker_names):
                    raise _ChunkReplacementInvariantError(
                        f"Donor active channel {source_channel} has no RTTM speaker-ID entry"
                    )
                speaker_id = donor_speaker_names[source_channel]
                if speaker_id is None or speaker_id == "":
                    raise _ChunkReplacementInvariantError(
                        f"Donor active channel {source_channel} has an empty RTTM speaker ID"
                    )
                if speaker_id in recipient_speaker_ids:
                    raise _ChunkReplacementInvariantError(
                        f"Donor speaker ID {speaker_id!r} overlaps the retained recipient speakers"
                    )
                if speaker_id not in speaker_to_output_channel:
                    if not free_channels:
                        raise _ChunkReplacementInvariantError(
                            "Replacement plan has more speakers than available target channels"
                        )
                    output_channel = free_channels.pop(0)
                    speaker_to_output_channel[speaker_id] = output_channel
                    replaced_targets[:recipient_target_len, output_channel] = 0.0
                output_channel = speaker_to_output_channel[speaker_id]
                current = replaced_targets[destination_start:destination_end, output_channel]
                replaced_targets[destination_start:destination_end, output_channel] = torch.maximum(
                    current, source_targets[:, source_channel]
                )
        return replaced_targets

    def _apply_chunk_replacement_plans(
        self,
        processed_signal,
        targets,
        speaker_names,
        metadata,
        plans,
    ):
        """Apply prevalidated plans while reading every donor from the original batch."""
        replacement_rate = processed_signal.new_tensor(0.0)
        if not plans:
            return processed_signal, targets, replacement_rate

        resolved_targets = []
        for plan in plans:
            recipient_index = plan.recipient_index
            donor_index = plan.donor_index
            replaced_targets = self._build_chunk_replaced_targets(
                recipient_targets=metadata.target_values[recipient_index],
                donor_targets=metadata.target_values[donor_index],
                recipient_speaker_names=speaker_names[recipient_index],
                donor_speaker_names=speaker_names[donor_index],
                recipient_target_len=metadata.target_lengths[recipient_index],
                donor_target_len=metadata.target_lengths[donor_index],
                target_frames_per_chunk=metadata.target_frames_per_chunk,
                destination_chunks=plan.destination_chunks,
            )
            resolved_targets.append(replaced_targets)

        augmented_signal = processed_signal.clone()
        augmented_targets = targets.clone()
        for plan, replaced_targets in zip(plans, resolved_targets):
            recipient_index = plan.recipient_index
            donor_index = plan.donor_index
            recipient_length = int(metadata.feature_lengths[recipient_index])
            for source_rank, destination_chunk in enumerate(plan.destination_chunks):
                destination_start = destination_chunk * metadata.feature_frames_per_chunk
                destination_length = min(
                    metadata.feature_frames_per_chunk,
                    recipient_length - destination_start,
                )
                source_start = source_rank * metadata.feature_frames_per_chunk
                augmented_signal[
                    recipient_index, :, destination_start : destination_start + destination_length
                ] = processed_signal[
                    donor_index, :, source_start : source_start + destination_length
                ]
            augmented_targets[recipient_index] = replaced_targets.to(
                device=targets.device, dtype=targets.dtype
            )

        replacement_rate = processed_signal.new_tensor(len(plans) / processed_signal.shape[0])
        return augmented_signal, augmented_targets, replacement_rate

    def _apply_batch_chunk_replace_augmentation(
        self,
        processed_signal,
        processed_signal_length,
        targets,
        target_lens,
        speaker_names,
    ):
        """Replace post-compression chunks with valid donor prefixes and remap targets by RTTM speaker ID."""
        batch_size = processed_signal.shape[0]
        replacement_rate = processed_signal.new_tensor(0.0)
        if self.batch_chunk_replace_probability <= 0.0 or batch_size < 2:
            return processed_signal, targets, replacement_rate
        if speaker_names is None or len(speaker_names) != batch_size:
            if not self._batch_chunk_replace_warned_missing_speaker_ids:
                logging.warning(
                    "Skipping batch_chunk_replace_augmentation because the batch does not contain "
                    "per-channel RTTM speaker IDs."
                )
                self._batch_chunk_replace_warned_missing_speaker_ids = True
            return processed_signal, targets, replacement_rate

        metadata = self._build_chunk_replacement_metadata(
            processed_signal,
            processed_signal_length,
            targets,
            target_lens,
            speaker_names,
        )
        plans = self._sample_chunk_replacement_plans(metadata)
        return self._apply_chunk_replacement_plans(
            processed_signal,
            targets,
            speaker_names,
            metadata,
            plans,
        )

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
            tuple:
                - Speaker probabilities of shape (batch_size, frames, num_speakers).
                - Raw speaker logits with the same shape.
                - Three-class activity logits, or None when no activity head exists.
        """
        processed_signal, processed_signal_length = self.process_signal(
            audio_signal=audio_signal, audio_signal_length=audio_signal_length
        )
        processed_signal = processed_signal[:, :, : processed_signal_length.max()]
        if self.streaming_mode:
            preds, speaker_logits, activity_logits = self.forward_streaming(
                processed_signal,
                processed_signal_length,
            )
            # When upsample_factor > 1, forward_streaming_step already collects
            # fine-resolution chunk preds from the learnable upsampler, so
            # total_preds is already at target resolution — no further upsampling.
            if self.upsample_factor <= 1 and self.val_upsample_preds and not self.training:
                preds = self.sortformer_modules.upsample_preds(
                    preds,
                    upsample_factor=self.output_subsampling_factor,
                    smooth_kernel=self.val_upsample_smooth_kernel,
                )
                speaker_logits = self._speaker_logits_from_upsampled_preds(preds, speaker_logits.dtype)
                activity_logits = self._upsample_activity_logits(activity_logits)
        else:
            emb_seq, emb_seq_length, pre_encode_feats = self.frontend_encoder(
                processed_signal=processed_signal, processed_signal_length=processed_signal_length
            )
            preds, speaker_logits, activity_logits = self.forward_infer(
                emb_seq,
                emb_seq_length,
                pre_encode_feats=pre_encode_feats,
            )
            if self.val_upsample_preds and not self.training:
                preds = self.sortformer_modules.upsample_preds(
                    preds,
                    upsample_factor=self.output_subsampling_factor,
                    smooth_kernel=self.val_upsample_smooth_kernel,
                )
                speaker_logits = self._speaker_logits_from_upsampled_preds(preds, speaker_logits.dtype)
                activity_logits = self._upsample_activity_logits(activity_logits)
        return preds, speaker_logits, activity_logits

    def _speaker_logits_from_upsampled_preds(self, preds, output_dtype):
        """Return canonical logits for validation probabilities after temporal upsampling."""
        with torch.autocast(device_type=preds.device.type, enabled=False):
            logits = torch.logit(preds.float().clamp(min=self.eps, max=1.0 - self.eps))
        return logits.to(output_dtype)

    def _upsample_activity_logits(self, logits):
        """Upsample three-class probabilities consistently with validation predictions."""
        if logits is None:
            return None
        probs = self.sortformer_modules.upsample_preds(
            torch.softmax(logits, dim=-1),
            upsample_factor=self.output_subsampling_factor,
            smooth_kernel=self.val_upsample_smooth_kernel,
        )
        return torch.log(probs.clamp_min(self.eps))

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
        chunk_pre_encode_embs, chunk_pre_encode_lengths = self._call_pre_encode(chunk, chunk_lengths)
        chunk_pre_encode_lengths = chunk_pre_encode_lengths.to(torch.int64)

        # concat the embeddings from speaker cache, FIFO queue and the chunk
        spkcache_fifo_chunk_pre_encode_embs, spkcache_fifo_chunk_pre_encode_lengths = self.concat_and_pad_script(
            [spkcache, fifo, chunk_pre_encode_embs], [spkcache_lengths, fifo_lengths, chunk_pre_encode_lengths]
        )

        # encode the concatenated embeddings
        spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths, _ = self.frontend_encoder(
            processed_signal=spkcache_fifo_chunk_pre_encode_embs,
            processed_signal_length=spkcache_fifo_chunk_pre_encode_lengths,
            bypass_pre_encode=True,
        )

        # forward pass for inference
        spkcache_fifo_chunk_preds, _, _ = self.forward_infer(
            spkcache_fifo_chunk_fc_encoder_embs,
            spkcache_fifo_chunk_fc_encoder_lengths,
            pre_encode_feats=spkcache_fifo_chunk_pre_encode_embs,
        )
        if self.upsample_factor > 1:
            spkcache_fifo_chunk_preds = self.sortformer_modules.downsample_preds(
                spkcache_fifo_chunk_preds, downsample_factor=self.upsample_factor
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

        if self.spec_augmentation is not None and self.training and not self.spec_augment_per_chunk:
            processed_signal = self.spec_augmentation(
                input_spec=processed_signal, length=processed_signal_length
            )

        # Tail-attention regularization (TransformerEncoder backbone only; ConformerEncoder has no
        # ``tail_len`` and is skipped). The encoder's attn_mode is fixed (e.g. "tail_causal"); only
        # the tail size varies. During training, with probability tail_attn_rate the tail is sampled
        # from [1, max_tail_attn_len]; otherwise tail_len=0 (full attention), so a fraction of steps
        # stay full-context for high-right-context inference. Always 0 at eval (and max_tail_attn_len=0
        # disables it). NOTE: only the backbone is masked; the post-encoder (self.transformer_encoder)
        # stays at full attention, so future can still leak through it — mask it too for a fully
        # faithful low-latency simulation.
        if hasattr(self.encoder, 'tail_len'):
            max_tail = getattr(self.sortformer_modules, 'max_tail_attn_len', 0)
            if self.training and max_tail > 0:
                tail_rate = getattr(self.sortformer_modules, 'tail_attn_rate', 1.0)
                self.encoder.tail_len = random.randint(1, max_tail) if random.random() < tail_rate else 0
            else:
                self.encoder.tail_len = 0

        att_mod = False
        if self.training:
            rand_num = random.random()
            if rand_num < self.sortformer_modules.causal_attn_rate:
                if hasattr(self.encoder, 'att_context_size'):
                    self.encoder.att_context_size = [-1, self.sortformer_modules.causal_attn_rc]
                self.transformer_encoder.diag = self.sortformer_modules.causal_attn_rc
                att_mod = True

        total_preds = torch.zeros((batch_size, 0, self.sortformer_modules.n_spk), device=self.device)
        total_speaker_logits = torch.empty(
            (batch_size, 0, self.sortformer_modules.n_spk), device=self.device
        )
        total_activity_logits = (
            torch.zeros((batch_size, 0, 3), device=self.device)
            if self.sortformer_modules.activity_head is not None
            else None
        )

        feat_len = processed_signal.shape[2]
        num_chunks = math.ceil(
            feat_len / (self.sortformer_modules.chunk_len * self.sortformer_modules.subsampling_factor)
        )
        streaming_loader = self.sortformer_modules.streaming_feat_loader(
            feat_seq=processed_signal,
            feat_seq_length=processed_signal_length,
            feat_seq_offset=processed_signal_offset,
        )
        for _, chunk_feat_seq_t, feat_lengths, left_offset, right_offset in tqdm(
            streaming_loader,
            total=num_chunks,
            desc="Streaming Steps",
            disable=self.training,
        ):
            step_outputs = self.forward_streaming_step(
                processed_signal=chunk_feat_seq_t,
                processed_signal_length=feat_lengths,
                streaming_state=streaming_state,
                total_preds=total_preds,
                left_offset=left_offset,
                right_offset=right_offset,
            )
            streaming_state, total_preds, chunk_speaker_logits, chunk_activity_logits = step_outputs
            total_speaker_logits = torch.cat([total_speaker_logits, chunk_speaker_logits], dim=1)
            if total_activity_logits is not None:
                total_activity_logits = torch.cat([total_activity_logits, chunk_activity_logits], dim=1)

        if att_mod:
            if hasattr(self.encoder, 'att_context_size'):
                self.encoder.att_context_size = [-1, -1]
            self.transformer_encoder.diag = None

        del processed_signal, processed_signal_length

        if sig_length < max_n_frames:  # Discard preds corresponding to padding
            n_frames = math.ceil(sig_length / self.encoder.subsampling_factor)
            if self.upsample_factor > 1:
                n_frames *= self.upsample_factor
            total_preds = total_preds[:, :n_frames, :]
            total_speaker_logits = total_speaker_logits[:, :n_frames, :]
            if total_activity_logits is not None:
                total_activity_logits = total_activity_logits[:, :n_frames, :]
        return total_preds, total_speaker_logits, total_activity_logits

    def forward_streaming_step(
        self,
        processed_signal,
        processed_signal_length,
        streaming_state,
        total_preds,
        left_offset=0,
        right_offset=0,
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

        Returns:
            streaming_state (SortformerStreamingState):
                    Tensor variables that contain the updated streaming state of the model from
                    this function call.
            total_preds (torch.Tensor):
                Tensor containing the updated total predicted speaker activity probabilities.
                Shape: (batch_size, cumulative pred length, num_speakers)
        """
        # When spec_augment_per_chunk=True, each chunk gets independently sampled masks,
        # simulating acoustic condition mismatch between cached and current embeddings.
        # When False, a single global mask was already applied in forward_streaming.
        # Note: processed_signal arrives as (B, T, D) from streaming_feat_loader (transposed),
        # but SpecAugment expects (B, D, T), so we transpose before and after.
        if self.spec_augmentation is not None and self.training and self.spec_augment_per_chunk:
            processed_signal = self.spec_augmentation(
                input_spec=processed_signal.transpose(1, 2), length=processed_signal_length
            ).transpose(1, 2)

        chunk_pre_encode_embs, chunk_pre_encode_lengths = self._call_pre_encode(
            processed_signal, processed_signal_length
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
        spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths, _ = self.frontend_encoder(
            processed_signal=spkcache_fifo_chunk_pre_encode_embs,
            processed_signal_length=spkcache_fifo_chunk_pre_encode_lengths,
            bypass_pre_encode=True,
        )
        spkcache_fifo_chunk_preds, all_speaker_logits, all_activity_logits = self.forward_infer(
            emb_seq=spkcache_fifo_chunk_fc_encoder_embs,
            emb_seq_length=spkcache_fifo_chunk_fc_encoder_lengths,
            pre_encode_feats=spkcache_fifo_chunk_pre_encode_embs,
        )

        lc_enc = round(left_offset / self.encoder.subsampling_factor)
        rc_enc = math.ceil(right_offset / self.encoder.subsampling_factor)
        uf = self.upsample_factor

        # The activity task is speaker-assignment invariant, so it only needs the same temporal
        # cache/FIFO/chunk slicing as the diarization output, not the speaker permutation.
        chunk_activity_logits = None
        if all_activity_logits is not None:
            if self.async_streaming:
                saved_spkcache_lengths_aux = streaming_state.spkcache_lengths.clone()
                saved_fifo_lengths_aux = streaming_state.fifo_lengths.clone()
                max_chunk_len = chunk_pre_encode_embs.shape[1] - lc_enc - rc_enc
                chunk_lengths_enc = (chunk_pre_encode_lengths - lc_enc).clamp(min=0, max=max_chunk_len)
                chunk_activity_logits = torch.zeros(
                    (all_activity_logits.shape[0], max_chunk_len * uf, 3),
                    device=all_activity_logits.device,
                    dtype=all_activity_logits.dtype,
                )
                for batch_index in range(all_activity_logits.shape[0]):
                    start = (
                        saved_spkcache_lengths_aux[batch_index]
                        + saved_fifo_lengths_aux[batch_index]
                        + lc_enc
                    ).item() * uf
                    length = chunk_lengths_enc[batch_index].item() * uf
                    chunk_activity_logits[batch_index, :length, :] = all_activity_logits[
                        batch_index, start : start + length, :
                    ]
            else:
                saved_spkcache_len_aux = streaming_state.spkcache.shape[1]
                saved_fifo_len_aux = streaming_state.fifo.shape[1]
                chunk_len_aux = chunk_pre_encode_embs.shape[1] - lc_enc - rc_enc
                start_aux = (saved_spkcache_len_aux + saved_fifo_len_aux + lc_enc) * uf
                end_aux = start_aux + chunk_len_aux * uf
                chunk_activity_logits = all_activity_logits[:, start_aux:end_aux, :]

        # Speaker logits must follow the same temporal slicing and inverse cache
        # permutation as the probabilities returned for the current chunk.
        ordered_speaker_logits = all_speaker_logits
        if not self.async_streaming and streaming_state.spk_perm is not None:
            inverse_speaker_permutation = torch.stack(
                [
                    torch.argsort(streaming_state.spk_perm[batch_index])
                    for batch_index in range(all_speaker_logits.shape[0])
                ]
            )
            ordered_speaker_logits = torch.stack(
                [
                    all_speaker_logits[batch_index, :, inverse_speaker_permutation[batch_index]]
                    for batch_index in range(all_speaker_logits.shape[0])
                ]
            )

        if self.async_streaming:
            max_chunk_len = chunk_pre_encode_embs.shape[1] - lc_enc - rc_enc
            chunk_lengths_enc = (chunk_pre_encode_lengths - lc_enc).clamp(min=0, max=max_chunk_len)
            chunk_speaker_logits = torch.full(
                (ordered_speaker_logits.shape[0], max_chunk_len * uf, ordered_speaker_logits.shape[2]),
                self.negative_init_val,
                device=ordered_speaker_logits.device,
                dtype=ordered_speaker_logits.dtype,
            )
            for batch_index in range(ordered_speaker_logits.shape[0]):
                start = (
                    streaming_state.spkcache_lengths[batch_index]
                    + streaming_state.fifo_lengths[batch_index]
                    + lc_enc
                ).item() * uf
                length = chunk_lengths_enc[batch_index].item() * uf
                chunk_speaker_logits[batch_index, :length, :] = ordered_speaker_logits[
                    batch_index, start : start + length, :
                ]
        else:
            start = (
                streaming_state.spkcache.shape[1]
                + streaming_state.fifo.shape[1]
                + lc_enc
            ) * uf
            chunk_len = (chunk_pre_encode_embs.shape[1] - lc_enc - rc_enc) * uf
            chunk_speaker_logits = ordered_speaker_logits[:, start : start + chunk_len, :]

        if uf > 1:
            preds_fine = spkcache_fifo_chunk_preds
            spkcache_fifo_chunk_preds = self.sortformer_modules.downsample_preds(
                preds_fine, downsample_factor=uf
            ).detach()
            # Apply the same inverse speaker permutation that streaming_update
            # will apply to the coarse preds (sync mode only, training only).
            if not self.async_streaming and streaming_state.spk_perm is not None:
                batch_size_pf = preds_fine.shape[0]
                inv_spk_perm = torch.stack(
                    [torch.argsort(streaming_state.spk_perm[bi]) for bi in range(batch_size_pf)]
                )
                preds_fine = torch.stack(
                    [preds_fine[bi, :, inv_spk_perm[bi]] for bi in range(batch_size_pf)]
                )

        if self.async_streaming:
            if uf > 1:
                saved_spkcache_lengths = streaming_state.spkcache_lengths.clone()
                saved_fifo_lengths = streaming_state.fifo_lengths.clone()
            streaming_state, chunk_preds = self.sortformer_modules.streaming_update_async(
                streaming_state=streaming_state,
                chunk=chunk_pre_encode_embs,
                chunk_lengths=chunk_pre_encode_lengths,
                preds=spkcache_fifo_chunk_preds,
                lc=lc_enc,
                rc=rc_enc,
            )
            if uf > 1:
                batch_size_cp = chunk_pre_encode_embs.shape[0]
                max_chunk_len = chunk_pre_encode_embs.shape[1] - lc_enc - rc_enc
                cl_enc = (chunk_pre_encode_lengths - lc_enc).clamp(min=0, max=max_chunk_len)
                chunk_preds = torch.zeros(
                    (batch_size_cp, max_chunk_len * uf, preds_fine.shape[2]),
                    device=preds_fine.device, dtype=preds_fine.dtype,
                )
                for bi in range(batch_size_cp):
                    sl = saved_spkcache_lengths[bi].item()
                    fl = saved_fifo_lengths[bi].item()
                    cl = cl_enc[bi].item()
                    start = (sl + fl + lc_enc) * uf
                    chunk_preds[bi, : cl * uf, :] = preds_fine[bi, start : start + cl * uf, :]
        else:
            saved_spkcache_len = streaming_state.spkcache.shape[1]
            saved_fifo_len = streaming_state.fifo.shape[1]
            streaming_state, chunk_preds = self.sortformer_modules.streaming_update(
                streaming_state=streaming_state,
                chunk=chunk_pre_encode_embs,
                preds=spkcache_fifo_chunk_preds,
                lc=lc_enc,
                rc=rc_enc,
            )
            if uf > 1:
                chunk_len = chunk_pre_encode_embs.shape[1] - lc_enc - rc_enc
                start = (saved_spkcache_len + saved_fifo_len + lc_enc) * uf
                chunk_preds = preds_fine[:, start : start + chunk_len * uf, :]

        total_preds = torch.cat([total_preds, chunk_preds], dim=1)

        return streaming_state, total_preds, chunk_speaker_logits, chunk_activity_logits

    def _build_loss_smooth_kernel(self) -> torch.Tensor:
        """Build the 1-D temporal smoothing kernel for loss-target softening.

        Returns a length-``loss_target_smooth_kernel`` float32 tensor that sums to 1.
        For 'gaussian', the standard deviation defaults to ``(K - 1) / 4`` (so the
        window spans roughly +/- 2 sigma) unless ``loss_target_smooth_sigma`` is set.
        """
        kernel_len = self.loss_target_smooth_kernel
        if self.loss_target_smooth_type == "uniform":
            kernel = torch.ones(kernel_len, dtype=torch.float32)
        else:
            sigma = self.loss_target_smooth_sigma
            if sigma is None or sigma <= 0:
                sigma = max((kernel_len - 1) / 4.0, 1e-6)
            positions = torch.arange(kernel_len, dtype=torch.float32) - (kernel_len - 1) / 2.0
            kernel = torch.exp(-0.5 * (positions / sigma) ** 2)
        return kernel / kernel.sum()

    def _soften_targets_for_loss(self, targets: torch.Tensor, target_lens: torch.Tensor) -> torch.Tensor:
        """Produce soft training targets for the BCE loss without affecting metrics.

        Applies, in order: (1) temporal boundary smoothing via a mask-normalized
        depthwise 1-D convolution along time (independently per speaker channel), and
        (2) label smoothing by clamping to ``[eps, 1 - eps]``. Returns ``targets``
        unchanged when both transforms are disabled.

        The convolution is normalized using a validity mask derived from
        ``target_lens`` so that zero-padded frames beyond each sample's length do not
        bleed into valid frames (and the activity plateau at a sample's true end is
        not spuriously ramped toward zero).

        Args:
            targets (torch.Tensor): Hard targets of shape (B, T, S).
            target_lens (torch.Tensor): Valid sequence lengths of shape (B,).

        Returns:
            torch.Tensor: Softened targets of shape (B, T, S).
        """
        if self.loss_target_smooth_kernel <= 1 and self.loss_label_smoothing <= 0.0:
            return targets

        soft = targets
        if self.loss_target_smooth_kernel > 1:
            _, num_frames, num_spks = targets.shape
            kernel_len = self.loss_target_smooth_kernel
            pad = kernel_len // 2
            kernel = self._loss_smooth_kernel_1d.to(device=targets.device, dtype=targets.dtype)

            # Validity mask from target_lens: (B, 1, T), 1 inside each sample, 0 on padding.
            frame_idx = torch.arange(num_frames, device=targets.device).unsqueeze(0)  # (1, T)
            valid = (frame_idx < target_lens.to(targets.device).unsqueeze(1)).to(targets.dtype)  # (B, T)
            valid = valid.unsqueeze(1)  # (B, 1, T)

            signal = targets.transpose(1, 2) * valid  # (B, S, T), zeroed outside valid region
            depthwise_w = kernel.view(1, 1, kernel_len).repeat(num_spks, 1, 1).contiguous()  # (S, 1, K)
            numerator = torch.nn.functional.conv1d(signal, depthwise_w, padding=pad, groups=num_spks)
            denominator = torch.nn.functional.conv1d(valid, kernel.view(1, 1, kernel_len), padding=pad)
            smoothed = numerator / denominator.clamp_min(self.eps)  # (B, S, T)
            soft = (smoothed * valid).transpose(1, 2)  # (B, T, S), padding restored to 0

        if self.loss_label_smoothing > 0.0:
            eps = self.loss_label_smoothing
            soft = soft.clamp(min=eps, max=1.0 - eps)
        return soft

    def _sharpen_probs(self, probs, temperature):
        """
        Temperature-sharpen probabilities in logit space: ``sigmoid(logit(p) / T)``.

        ``T < 1`` sharpens toward 0/1, so faint/low-confidence predictions are squashed toward 0
        and no longer accumulate into a confident first onset in the survival product (while
        confident predictions stay near 1). ``T == 1.0`` is a no-op. The transform is monotonic,
        so it preserves onset ordering and the 0.5 threshold.

        Args:
            probs (torch.Tensor): Probabilities in [0, 1].
            temperature (float): Sharpening temperature (> 0); < 1 sharpens, 1.0 disables.

        Returns:
            torch.Tensor: Sharpened probabilities (same shape/dtype as ``probs``).
        """
        if temperature == 1.0:
            return probs
        return torch.sigmoid(torch.logit(probs, eps=self.eps) / temperature)

    def _pairwise_ats_loss(self, preds, target_lens):
        """
        Onset-based pairwise Arrival Time Sort loss (duration-invariant, self-referential).

        For each output channel a differentiable first-onset distribution is built from the
        predictions via a survival product, and the pairwise "arrives-before" probabilities

            P[i, j] = Pr(onset_i < onset_j) = sum_t f_i(t) * Sv_j(t)

        are computed, where ``Sv_i(t)`` is the probability that channel ``i`` is still silent
        through frame ``t`` and ``f_i(t) = Sv_i(t-1) * preds[t, i]`` is its first-onset pmf.
        The canonical convention is that lower channel indices onset earlier, so the target is
        ``P[i, j] = 0`` for every ``i > j`` (the strictly-lower triangle); the upper triangle is
        left unsupervised (it is pulled toward ``1 - tie`` automatically and must stay free to
        allow genuine ties and undetected speakers). The penalty is a single scalar per channel
        pair, so unlike a per-frame loss it does NOT depend on how much each speaker talks (a
        brief speaker's ordering counts as much as a talkative one). Undetected channels (no
        predicted activity) contribute no spurious penalty when correctly placed at high indices
        (``P = 0`` there), and an empty channel sitting below a detected one is penalized
        (``P ~= 1``), pushing detected speakers toward the low channels.

        Note: this is computed purely from predictions (no ground-truth arrival times), so like
        any self-referential ordering term it relies on PIL to keep the predictions faithful
        (otherwise the ordering can be satisfied by suppressing or fabricating onsets).

        Args:
            preds (torch.Tensor): Predicted probabilities of shape (B, T, N).
            target_lens (torch.Tensor): Valid sequence lengths of shape (B,).

        Returns:
            torch.Tensor: Scalar pairwise ATS loss (mean BCE-to-0 over i>j channel pairs).
        """
        batch_size, num_frames, num_spks = preds.shape
        if num_spks < 2:
            return torch.zeros((), device=preds.device, dtype=preds.dtype)

        # Zero out padding frames so they never create spurious onset mass, and compute the
        # survival product in float32/log-space for numerical stability over long sequences.
        valid = (
            torch.arange(num_frames, device=preds.device).unsqueeze(0)
            < target_lens.to(preds.device).unsqueeze(1)
        ).to(preds.dtype).unsqueeze(-1)  # (B, T, 1)
        # Optionally sharpen first so faint hedges don't accumulate into a confident onset.
        preds_s = self._sharpen_probs(preds, self.pairwise_ats_temperature)
        p = (preds_s * valid).clamp(max=1.0 - self.eps).float()  # (B, T, N)

        log_sv = torch.cumsum(torch.log1p(-p), dim=1)  # log Sv_i(t) = log P(onset_i > t)
        sv = log_sv.exp()  # Sv_i(t)
        sv_prev = torch.cat([torch.ones_like(sv[:, :1]), sv[:, :-1]], dim=1)  # Sv_i(t-1)
        onset_pmf = sv_prev * p  # f_i(t) = P(first onset at t), (B, T, N)

        # P[i, j] = sum_t f_i(t) * Sv_j(t) = Pr(onset_i < onset_j), shape (B, N, N).
        p_before = torch.einsum('bti,btj->bij', onset_pmf, sv)

        # Target P[i, j] = 0 for i > j (strictly-lower triangle); penalize via BCE toward 0.
        lower_tri = torch.tril(
            torch.ones(num_spks, num_spks, device=preds.device, dtype=torch.bool), diagonal=-1
        )  # (N, N), True where i > j
        p_lower = p_before[:, lower_tri].clamp(max=1.0 - self.eps)  # (B, num_pairs)
        loss = -torch.log1p(-p_lower)  # BCE(P, target=0) = -log(1 - P)
        return loss.mean()

    def _self_ats_loss(self, speaker_logits, target_lens):
        """
        Compute a self-referential Arrival Time Sort (self-ATS) loss.

        Unlike ats_loss / pairwise_ats_loss, whose ordering target is derived from the
        ground-truth arrival times, self-ATS derives the target from the model's OWN
        predictions: it sorts the predicted channels by their predicted onset and asks the
        model to already be in that order. The loss is therefore (near) zero whenever the
        predicted channels are self-consistently ordered by onset, regardless of ground
        truth, and produces a gradient that pushes mis-ordered channels toward onset order.
        PIL is expected to supply the separation/detection signal; this term only
        canonicalizes the ordering of whatever the model predicts. By construction it does
        not penalize ordering for speakers the model fails to detect (e.g. a missed short
        early speaker), since the target is built only from predicted channels.

        Notes:
            - The target is the (detached) onset-sorted copy of preds. With ``self_ats_metric
              == 'bce'`` (default) the BCE has an entropy floor, so the logged value is not
              exactly zero at perfect ordering (its gradient is still ~zero when sorted). With
              ``self_ats_metric == 'mse'`` there is no floor: the value is 0 when
              self-consistent and the gradient is bounded/gentler.
            - ``self_ats_temperature`` < 1 sharpens predictions toward 0/1, applied
              symmetrically to both the probs side and the target. Because both sides use the
              same sharpened tensor, the loss is still 0 / zero-gradient when already sorted
              (no self-distillation bias), but mis-ordered channels are pulled toward sharp
              targets (less softening) with a ~1/T-amplified gradient (more reordering muscle).
            - Ties in predicted onset are broken by a stable sort (current channel order is
              kept), avoiding spurious swap gradients between equally-onset channels.

        Args:
            speaker_logits (torch.Tensor): Raw speaker logits of shape (B, T, N).
            target_lens (torch.Tensor): Valid sequence lengths of shape (B,).

        Returns:
            torch.Tensor: Scalar self-ATS loss.
        """
        num_spks = speaker_logits.shape[2]
        if num_spks < 2:
            return torch.zeros(
                (), device=speaker_logits.device, dtype=speaker_logits.dtype
            )
        valid = (
            torch.arange(
                speaker_logits.shape[1], device=speaker_logits.device
            ).unsqueeze(0)
            < target_lens.to(speaker_logits.device).unsqueeze(1)
        )

        # Predicted onset (first frame above threshold) per channel; empty channels -> T.
        # Temperature sharpening is monotonic and preserves the 0.5 threshold, so onsets (and
        # therefore the permutation) are computed from the original predictions.
        preds = torch.sigmoid(speaker_logits) * valid.unsqueeze(-1)
        onsets = find_first_nonzero(
            preds.detach(), max_cap_val=speaker_logits.shape[1]
        )  # (B, N)
        # Permutation sorting channels by predicted onset; stable keeps current order on ties.
        perm = torch.argsort(onsets, dim=1, stable=True)  # (B, N)

        # Sharpen in logit space; applying sigmoid produces the same probabilities as
        # sigmoid(logit(preds) / T) without recovering logits from rounded probabilities.
        logits_s = speaker_logits / self.self_ats_temperature
        preds_s = torch.sigmoid(logits_s)

        # Onset-sorted target: position k receives the content of the k-th earliest channel.
        index = perm.unsqueeze(1).expand(
            -1, speaker_logits.shape[1], -1
        )  # (B, T, N)
        self_ats_target = torch.gather(preds_s, dim=2, index=index).detach()  # (B, T, N)

        if self.self_ats_metric == "mse":
            # Masked mean squared error over valid frames (no entropy floor; 0 when sorted).
            valid_float = valid.to(speaker_logits.dtype)
            sq_err = (
                (preds_s - self_ats_target) ** 2
            ) * valid_float.unsqueeze(-1)  # (B, T, N)
            denom = (
                valid_float.sum() * speaker_logits.shape[2]
            )  # valid (frame, channel) element count
            return sq_err.sum() / denom.clamp_min(1.0)

        with torch.autocast(device_type=speaker_logits.device.type, enabled=False):
            element_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits_s.float(),
                self_ats_target.float(),
                reduction="none",
            )
            denominator = valid.sum() * speaker_logits.shape[2]
            loss = (
                element_loss * valid.unsqueeze(-1)
            ).sum() / denominator.clamp_min(1)
        return loss

    def _spkcount_loss(self, preds, targets, target_lens):
        """
        Auxiliary supervised loss on the cumulative number of arrived speakers.

        For every frame ``t`` this matches the expected number of speakers whose onset has
        occurred by ``t`` against the ground-truth count. The predicted count reuses the same
        first-onset survival product as the pairwise-ATS loss: the probability that channel
        ``i`` has arrived by ``t`` is its onset CDF ``A_i(t) = 1 - Sv_i(t)``, and by linearity
        of expectation (exact regardless of inter-channel correlation)

            C_pred(t) = sum_i A_i(t) = N - sum_i Sv_i(t).

        The target is the GT arrival staircase ``C_gt(t) = #{speakers active at some s <= t}``,
        built as a cumulative-max over time summed over channels (e.g. 0 0 0 1 1 1 2 2 ...). Both
        curves are normalized by ``N`` (fraction arrived) and compared with a Huber/Smooth-L1
        distance averaged over valid frames.

        Why this helps: both curves are non-decreasing (each ``Sv_i`` is monotone), so the L1
        distance between them equals the 1-Wasserstein (earth-mover) distance between the
        predicted and true arrival schedules. A spurious early "blip" on a low channel collapses
        that channel's survival permanently, inflating ``C_pred`` for the entire tail; the penalty
        therefore scales with how early the blip is displaced from the true onset. This is a
        persistent, GT-anchored cost that matches the persistent self-referential reward the
        pairwise-ATS loss would otherwise grant the blip (a 1-frame blip is only a 1-frame false
        positive to PIL/ATS, but corrupts this whole curve). It is permutation-invariant across
        channels (a sum over channels), so it constrains only the arrival schedule, never the
        channel assignment, and needs no Hungarian matching. As a symmetric regressor it also
        penalizes under-counting, reinforcing detection of missed speakers.

        Args:
            preds (torch.Tensor): Predicted probabilities of shape (B, T, N).
            targets (torch.Tensor): Ground-truth (hard 0/1) labels of shape (B, T, N).
            target_lens (torch.Tensor): Valid sequence lengths of shape (B,).

        Returns:
            torch.Tensor: Scalar arrival-count loss (Huber on the normalized count curve).
        """
        num_frames, num_spks = preds.shape[1], preds.shape[2]

        # Validity mask (B, T): 1 inside each sample, 0 on padding. Built in float32 since the
        # survival product runs in float32 for numerical stability over long sequences.
        valid = (
            torch.arange(num_frames, device=preds.device).unsqueeze(0)
            < target_lens.to(preds.device).unsqueeze(1)
        ).float()  # (B, T)

        # Predicted expected arrival fraction C_pred(t) / N via the first-onset survival product.
        # Padding frames are zeroed so they never create spurious arrival mass. Optionally sharpen
        # first so faint hedges don't accumulate into a confident onset / inflate the count.
        preds_s = self._sharpen_probs(preds, self.spkcount_temperature)
        p = (preds_s * valid.unsqueeze(-1)).clamp(min=0.0, max=1.0 - self.eps).float()  # (B, T, N)
        sv = torch.cumsum(torch.log1p(-p), dim=1).exp()  # Sv_i(t) = P(onset_i > t)
        c_pred = (1.0 - sv).sum(dim=2) / num_spks  # (B, T)

        # GT arrival fraction C_gt(t) / N: "arrived" = ever active by t -> cumulative max in time.
        c_gt = torch.cummax(targets.float(), dim=1).values.sum(dim=2) / num_spks  # (B, T)

        # Huber on the normalized count curve, averaged over valid frames. beta = one speaker's
        # worth of count error, so it is L1-like (= EMD, bounded gradient) for the large persistent
        # errors that matter and only quadratic once the schedule is essentially correct.
        err = torch.nn.functional.smooth_l1_loss(
            c_pred, c_gt, beta=1.0 / num_spks, reduction="none"
        ) * valid  # (B, T)
        loss = err.sum() / valid.sum().clamp_min(1.0)
        return loss

    @staticmethod
    def _masked_cross_entropy(logits, labels, target_lens):
        """Compute three-class cross entropy over valid frames in float32."""
        if logits is None:
            return labels.new_zeros((), dtype=torch.float32)

        loss = torch.nn.functional.cross_entropy(
            logits.float().transpose(1, 2),
            labels.to(device=logits.device, dtype=torch.long),
            reduction="none",
        )
        valid = (
            torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
            < target_lens.to(logits.device).unsqueeze(1)
        )
        loss = loss * valid
        return loss.sum() / valid.sum().clamp_min(1)

    def _activity_logits_from_speaker_preds(self, preds):
        """
        Derive silence, exactly-one-speaker, and overlap logits from speaker probabilities.

        The recurrence computes a Poisson-binomial count distribution while merging all
        counts of two or more active speakers into the overlap class. It is permutation
        invariant across speaker channels and remains fully differentiable.
        """
        with torch.autocast(device_type=preds.device.type, enabled=False):
            speaker_probs = preds.float().clamp(min=0.0, max=1.0)
            inactive_probs = 1.0 - speaker_probs

            # Prefix/suffix products provide prod_{j != i}(1 - p_j) for every
            # speaker without division, remaining well-defined when p_i == 1.
            prefix_products = torch.cumprod(inactive_probs, dim=-1)
            suffix_products = torch.flip(
                torch.cumprod(torch.flip(inactive_probs, dims=(-1,)), dim=-1),
                dims=(-1,),
            )
            ones = torch.ones_like(inactive_probs[..., :1])
            exclusive_inactive_products = torch.cat((ones, prefix_products[..., :-1]), dim=-1) * torch.cat(
                (suffix_products[..., 1:], ones), dim=-1
            )

            prob_zero = prefix_products[..., -1]
            prob_one = (speaker_probs * exclusive_inactive_products).sum(dim=-1)
            prob_overlap = (1.0 - prob_zero - prob_one).clamp_min(0.0)

            activity_probs = torch.stack((prob_zero, prob_one, prob_overlap), dim=-1)
            # The common additive floor keeps cross entropy finite for saturated
            # probabilities; cross_entropy normalizes these log scores internally.
            return torch.log(activity_probs + self.eps)

    def _activity_loss(self, activity_logits, targets, target_lens):
        """Classify each frame as silence, single-speaker speech, or overlap."""
        hard_activity = targets > 0.5
        activity_targets = hard_activity.sum(dim=-1).clamp(max=2)
        return self._masked_cross_entropy(activity_logits, activity_targets, target_lens)

    @staticmethod
    def _exclude_transition_collar(eligible, hard_targets, valid, collar_frames):
        """Remove ``collar_frames`` on each side of every valid target transition."""
        if collar_frames <= 0 or hard_targets.shape[1] <= 1:
            return eligible

        adjacent_valid = valid[:, :-1] & valid[:, 1:]
        target_changes = (hard_targets[:, 1:] != hard_targets[:, :-1]).any(dim=-1)
        target_changes &= adjacent_valid
        transition_frames = torch.zeros_like(valid)
        transition_frames[:, :-1] |= target_changes
        transition_frames[:, 1:] |= target_changes

        # The two transition endpoints already remove one frame on each side.
        collar_extension = collar_frames - 1
        if collar_extension > 0:
            transition_frames = (
                torch.nn.functional.max_pool1d(
                    transition_frames.float().unsqueeze(1),
                    kernel_size=2 * collar_extension + 1,
                    stride=1,
                    padding=collar_extension,
                )
                .squeeze(1)
                .bool()
            )
        return eligible & ~transition_frames

    def _speaker_rank_loss(self, speaker_logits, targets_pil, target_lens):
        """
        Rank the PIL-matched active speaker above every inactive slot on exclusive frames.

        For an eligible frame with active speaker ``y``, this computes

            log(1 + sum_{j != y} exp(z_j - z_y + margin)).

        The loss is reduced over all eligible frames in the effective DDP batch. The
        Hungarian assignment is already detached when ``targets_pil`` is constructed;
        this function consumes those hard aligned targets without recomputing it.
        """
        if speaker_logits is None:
            return targets_pil.new_zeros((), dtype=torch.float32)

        num_frames = speaker_logits.shape[1]
        hard_targets = targets_pil > 0.5
        valid = (
            torch.arange(num_frames, device=speaker_logits.device).unsqueeze(0)
            < target_lens.to(speaker_logits.device).unsqueeze(1)
        )
        eligible = hard_targets.sum(dim=-1) == 1
        eligible = self._exclude_transition_collar(
            eligible, hard_targets, valid, self.rank_collar_frames
        )
        eligible &= valid

        with torch.autocast(device_type=speaker_logits.device.type, enabled=False):
            logits_f = speaker_logits.float()
            positive_logits = (logits_f * hard_targets).sum(dim=-1, keepdim=True)
            competitor_terms = logits_f - positive_logits + self.rank_margin
            competitor_terms = competitor_terms.masked_fill(hard_targets, float('-inf'))
            zero_term = torch.zeros_like(positive_logits)
            per_frame_loss = torch.logsumexp(
                torch.cat((zero_term, competitor_terms), dim=-1),
                dim=-1,
            )
            local_loss_sum = per_frame_loss.masked_select(eligible).sum()

        global_eligible_count = eligible.sum().to(device=speaker_logits.device, dtype=torch.float32)
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_eligible_count, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()

        loss = world_size * local_loss_sum / global_eligible_count.clamp_min(1.0)
        return loss

    def _speech_bce_loss(self, speaker_logits, targets_pil, target_lens):
        """
        Compute PIL-aligned BCE on clean frames containing at least one active speaker.

        BCE is first averaged over the fixed speaker-slot dimension, then reduced over
        all eligible frames in the effective DDP batch. Silence, padding, and the
        configured transition collar do not contribute.
        """
        if speaker_logits is None:
            return targets_pil.new_zeros((), dtype=torch.float32)

        num_frames = speaker_logits.shape[1]
        hard_targets = targets_pil > 0.5
        valid = (
            torch.arange(num_frames, device=speaker_logits.device).unsqueeze(0)
            < target_lens.to(speaker_logits.device).unsqueeze(1)
        )
        eligible = hard_targets.any(dim=-1)
        eligible = self._exclude_transition_collar(
            eligible, hard_targets, valid, self.speech_bce_collar_frames
        )
        eligible &= valid

        with torch.autocast(device_type=speaker_logits.device.type, enabled=False):
            element_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                speaker_logits.float(),
                targets_pil.float(),
                reduction="none",
            )
            per_frame_loss = element_loss.mean(dim=-1)
            local_loss_sum = per_frame_loss.masked_select(eligible).sum()

        global_eligible_count = eligible.sum().to(
            device=speaker_logits.device, dtype=torch.float32
        )
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_eligible_count, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()

        return world_size * local_loss_sum / global_eligible_count.clamp_min(1.0)

    @staticmethod
    def _full_window_region_mask(frame_condition, valid, radius):
        """Keep frames whose complete centered window satisfies ``frame_condition``."""
        stable = frame_condition & valid.unsqueeze(-1)
        if radius <= 0:
            return stable

        batch_size, num_frames, num_spks = stable.shape
        kernel_size = 2 * radius + 1
        flat_stable = stable.transpose(1, 2).reshape(
            batch_size * num_spks, 1, num_frames
        )
        kernel = torch.ones(
            (1, 1, kernel_size),
            device=stable.device,
            dtype=torch.float32,
        )
        window_count = torch.nn.functional.conv1d(
            flat_stable.float(),
            kernel,
            padding=radius,
        )
        return (
            window_count.eq(float(kernel_size))
            .reshape(batch_size, num_spks, num_frames)
            .transpose(1, 2)
        )

    @staticmethod
    def _local_window_mean(values, radius):
        """Compute a centered local mean with zero support outside the sequence."""
        if radius <= 0:
            return values

        batch_size, num_frames, num_spks = values.shape
        kernel_size = 2 * radius + 1
        flat_values = values.transpose(1, 2).reshape(
            batch_size * num_spks, 1, num_frames
        )
        kernel = torch.full(
            (1, 1, kernel_size),
            1.0 / kernel_size,
            device=values.device,
            dtype=torch.float32,
        )
        local_mean = torch.nn.functional.conv1d(
            flat_values.float(),
            kernel,
            padding=radius,
        )
        return (
            local_mean
            .reshape(batch_size, num_spks, num_frames)
            .transpose(1, 2)
        )

    def _interior_focal_loss(self, speaker_logits, targets_pil, target_lens):
        """
        Compute focal BCE on strict interiors of PIL-aligned activity and silence.

        Positive frames require every target in the centered positive-radius window
        to exceed 0.5. Negative frames analogously require every target in the
        negative-radius window to be below 0.5. Windows crossing padding or sequence
        boundaries are ineligible. Positive and negative losses share one global DDP
        denominator, preserving their natural eligible-frame class imbalance.
        """
        if speaker_logits is None:
            return targets_pil.new_zeros((), dtype=torch.float32)

        num_frames = speaker_logits.shape[1]
        valid = (
            torch.arange(num_frames, device=speaker_logits.device).unsqueeze(0)
            < target_lens.to(speaker_logits.device).unsqueeze(1)
        )
        positive_eligible = self._full_window_region_mask(
            targets_pil > 0.5,
            valid,
            self.interior_focal_positive_radius,
        )
        negative_eligible = self._full_window_region_mask(
            targets_pil < 0.5,
            valid,
            self.interior_focal_negative_radius,
        )

        with torch.autocast(device_type=speaker_logits.device.type, enabled=False):
            logits_f = speaker_logits.float()
            positive_loss = (
                torch.sigmoid(-logits_f).pow(self.interior_focal_gamma)
                * torch.nn.functional.softplus(-logits_f)
            )
            negative_loss = (
                torch.sigmoid(logits_f).pow(self.interior_focal_gamma)
                * torch.nn.functional.softplus(logits_f)
            )
            local_loss_sum = positive_loss.masked_select(
                positive_eligible
            ).sum() + negative_loss.masked_select(negative_eligible).sum()

        global_eligible_count = (
            positive_eligible.sum() + negative_eligible.sum()
        ).to(device=speaker_logits.device, dtype=torch.float32)
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_eligible_count, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()

        return world_size * local_loss_sum / global_eligible_count.clamp_min(1.0)

    def _purity_focal_loss(self, speaker_logits, targets_pil, target_lens):
        """
        Compute focal BCE weighted by local PIL target purity.

        Positive and negative center weights are multiplied by their corresponding
        centered-window target purity raised to ``purity_focal_power``.
        Invalid and out-of-sequence frames contribute zero support, smoothly reducing
        boundary weights. Positive and negative terms share one global DDP weight
        denominator, preserving their effective weighted class imbalance.
        """
        if speaker_logits is None:
            return targets_pil.new_zeros((), dtype=torch.float32)

        num_frames = speaker_logits.shape[1]
        valid = (
            torch.arange(num_frames, device=speaker_logits.device).unsqueeze(0)
            < target_lens.to(speaker_logits.device).unsqueeze(1)
        )
        valid_expanded = valid.unsqueeze(-1)
        targets_f = targets_pil.float().clamp(min=0.0, max=1.0)
        positive_support = targets_f * valid_expanded
        negative_support = (1.0 - targets_f) * valid_expanded
        positive_purity = self._local_window_mean(
            positive_support,
            self.purity_focal_positive_radius,
        )
        negative_purity = self._local_window_mean(
            negative_support,
            self.purity_focal_negative_radius,
        )
        positive_weight = (
            positive_support
            * positive_purity.pow(self.purity_focal_power)
        ).detach()
        negative_weight = (
            negative_support
            * negative_purity.pow(self.purity_focal_power)
        ).detach()

        with torch.autocast(device_type=speaker_logits.device.type, enabled=False):
            logits_f = speaker_logits.float()
            positive_loss = (
                torch.sigmoid(-logits_f).pow(self.purity_focal_gamma)
                * torch.nn.functional.softplus(-logits_f)
            )
            negative_loss = (
                torch.sigmoid(logits_f).pow(self.purity_focal_gamma)
                * torch.nn.functional.softplus(logits_f)
            )
            local_loss_sum = (
                positive_weight * positive_loss
                + negative_weight * negative_loss
            ).sum()

        global_weight_sum = positive_weight.sum() + negative_weight.sum()
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_weight_sum, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()

        return world_size * local_loss_sum / global_weight_sum.clamp_min(1.0)

    @staticmethod
    def _speaker_count_metrics(preds, targets, target_lens):
        """Compute per-sample global speaker-count MAE and exact-match accuracy."""
        valid = (
            torch.arange(preds.shape[1], device=preds.device).unsqueeze(0)
            < target_lens.to(preds.device).unsqueeze(1)
        ).unsqueeze(-1)
        pred_present = ((preds > 0.5) & valid).any(dim=1)
        target_present = ((targets > 0.5) & valid).any(dim=1)
        pred_count = pred_present.sum(dim=1)
        target_count = target_present.sum(dim=1)
        count_error = (pred_count - target_count).abs().float()
        count_mae = count_error.mean()
        count_accuracy = (pred_count == target_count).float().mean()
        return count_mae, count_accuracy

    def _presence_loss(self, preds, targets_pil, target_lens):
        """
        Compare local speaker presence after PIL alignment.

        For every frame ``t`` and channel ``s``, presence is the maximum activity
        in ``[t-d, t+d]`` where ``d`` is ``presence_window_radius``. Padding is
        zeroed before pooling and excluded from the final mean. Locally absent
        channels incur no penalty below ``presence_negative_margin``, preserving
        low-confidence hedging while suppressing cache-threatening false speakers.
        """
        num_frames, num_spks = preds.shape[1], preds.shape[2]
        valid = (
            torch.arange(num_frames, device=preds.device).unsqueeze(0)
            < target_lens.to(preds.device).unsqueeze(1)
        )
        valid_expanded = valid.unsqueeze(-1)
        preds_valid = preds * valid_expanded
        targets_valid = (targets_pil > 0.5).to(preds.dtype) * valid_expanded

        radius = self.presence_window_radius
        kernel_size = 2 * radius + 1
        pred_presence = torch.nn.functional.max_pool1d(
            preds_valid.transpose(1, 2),
            kernel_size=kernel_size,
            stride=1,
            padding=radius,
        ).transpose(1, 2)
        target_presence = torch.nn.functional.max_pool1d(
            targets_valid.transpose(1, 2),
            kernel_size=kernel_size,
            stride=1,
            padding=radius,
        ).transpose(1, 2)

        with torch.autocast(device_type=preds.device.type, enabled=False):
            pred_presence_f = pred_presence.float().clamp(min=self.eps, max=1.0 - self.eps)
            target_presence_f = target_presence.float()
            positive_loss = -target_presence_f * torch.log(pred_presence_f)
            negative_excess = (
                (pred_presence_f - self.presence_negative_margin)
                / (1.0 - self.presence_negative_margin)
            ).clamp(min=0.0, max=1.0 - self.eps)
            negative_loss = -(1.0 - target_presence_f) * torch.log1p(-negative_excess)
            element_loss = positive_loss + negative_loss
        element_loss = element_loss * valid_expanded
        denominator = valid.sum() * num_spks
        return element_loss.sum() / denominator.clamp_min(1)

    def _dice_loss(self, preds, targets_pil, target_lens):
        """
        Compute PIL-aligned soft Dice loss for target-present speaker channels.

        Dice is reduced independently over time for every sample/speaker channel.
        Channels with fewer than ``dice_min_target_frames`` positive hard-target frames
        are excluded. Each remaining channel is weighted by
        ``target_mass ** dice_duration_gamma`` and the weighted mean is reduced over
        the effective DDP batch. Gamma zero gives every eligible speaker equal weight;
        gamma one weights speakers in proportion to target duration.
        """
        valid = (
            torch.arange(preds.shape[1], device=preds.device).unsqueeze(0)
            < target_lens.to(preds.device).unsqueeze(1)
        ).unsqueeze(-1)

        with torch.autocast(device_type=preds.device.type, enabled=False):
            preds_valid = preds.float() * valid
            targets_valid = targets_pil.float() * valid
            intersection = (preds_valid * targets_valid).sum(dim=1)
            pred_mass = preds_valid.sum(dim=1)
            target_mass = targets_valid.sum(dim=1)
            dice_score = (2.0 * intersection + 1.0) / (pred_mass + target_mass + 1.0)
            target_frame_count = (targets_valid > 0.5).sum(dim=1)
            eligible_channels = target_frame_count >= self.dice_min_target_frames
            channel_weights = torch.where(
                eligible_channels,
                target_mass.pow(self.dice_duration_gamma),
                torch.zeros_like(target_mass),
            ).detach()
            local_weighted_loss = ((1.0 - dice_score) * channel_weights).sum()

        global_weight_sum = channel_weights.sum()
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_weight_sum, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()

        return world_size * local_weighted_loss / global_weight_sum.clamp_min(1.0)

    @staticmethod
    def _aggregate_selected_frame_losses(
        frame_losses,
        selected,
        use_logmeanexp,
        temperature,
    ):
        """Aggregate selected frame losses independently for each batch/channel."""
        if frame_losses.shape[1] == 0:
            return frame_losses.new_zeros((frame_losses.shape[0], frame_losses.shape[2]))

        selected_count = selected.sum(dim=1)
        if not use_logmeanexp:
            selected_losses = torch.where(
                selected, frame_losses, torch.zeros_like(frame_losses)
            )
            return selected_losses.sum(dim=1) / selected_count.clamp_min(1)

        scaled_losses = frame_losses / temperature
        masked_losses = scaled_losses.masked_fill(~selected, float('-inf'))
        has_selected = selected_count > 0
        # Give empty channels one finite dummy value so logsumexp stays finite; they
        # are explicitly reset to zero after aggregation.
        safe_first_frame = torch.where(
            has_selected.unsqueeze(1),
            masked_losses[:, :1, :],
            torch.zeros_like(masked_losses[:, :1, :]),
        )
        masked_losses = torch.cat((safe_first_frame, masked_losses[:, 1:, :]), dim=1)
        channel_loss = temperature * (
            torch.logsumexp(masked_losses, dim=1)
            - torch.log(selected_count.clamp_min(1).to(frame_losses.dtype))
        )
        return torch.where(has_selected, channel_loss, torch.zeros_like(channel_loss))

    def _phantom_loss(self, speaker_logits, targets_pil, target_lens):
        """
        Penalize high-confidence predictions on PIL-aligned empty speaker channels.

        A channel is eligible only when its hard PIL target is zero over every valid
        frame. Within each eligible channel, negative BCE is aggregated over frames whose
        detached prediction exceeds ``phantom_threshold`` using either the mean or
        temperature-controlled log-mean-exp. Channel losses are then summed and divided
        by the fixed number of output speaker slots, so an additional phantom channel
        adds an additional penalty while the scale remains independent of the number of
        ground-truth speakers. Active channels, sub-threshold predictions, and padding
        contribute zero.
        """
        num_frames, num_spks = speaker_logits.shape[1], speaker_logits.shape[2]
        valid = (
            torch.arange(num_frames, device=speaker_logits.device).unsqueeze(0)
            < target_lens.to(speaker_logits.device).unsqueeze(1)
        )
        valid_expanded = valid.unsqueeze(-1)
        empty_channels = ~((targets_pil > 0.5) & valid_expanded).any(dim=1)
        preds = torch.sigmoid(speaker_logits.detach())
        selected = (
            valid_expanded
            & empty_channels.unsqueeze(1)
            & (preds > self.phantom_threshold)
        )

        with torch.autocast(device_type=speaker_logits.device.type, enabled=False):
            negative_bce = torch.nn.functional.softplus(speaker_logits.float())
            channel_loss = self._aggregate_selected_frame_losses(
                negative_bce,
                selected,
                use_logmeanexp=self.phantom_logmeanexp,
                temperature=self.phantom_logmeanexp_temperature,
            )
            loss = (channel_loss.sum(dim=1) / num_spks).mean()
        return loss

    def _speaker_existence_loss(self, speaker_logits, targets_pil, target_lens):
        """
        Require one channel-level detection decision for each sufficiently long speaker.

        Active-frame logits are pooled per PIL-aligned channel using normalized
        temperature-controlled log-mean-exp. Positive BCE on that pooled logit then
        penalizes channels whose detached pooled confidence remains below
        ``speaker_existence_threshold``. Ineligible short, empty, and already-confident
        channels contribute zero. Channel terms are summed over the fixed output slots,
        divided by ``num_spks``, and averaged over samples like phantom loss.
        """
        num_frames, num_spks = speaker_logits.shape[1], speaker_logits.shape[2]
        valid = (
            torch.arange(num_frames, device=speaker_logits.device).unsqueeze(0)
            < target_lens.to(speaker_logits.device).unsqueeze(1)
        ).unsqueeze(-1)
        active = (targets_pil > 0.5) & valid
        eligible_channels = (
            active.sum(dim=1) >= self.speaker_existence_min_frames
        )

        with torch.autocast(device_type=speaker_logits.device.type, enabled=False):
            pooled_logits = self._aggregate_selected_frame_losses(
                speaker_logits.float(),
                active,
                use_logmeanexp=True,
                temperature=self.speaker_existence_temperature,
            )
            if self.speaker_existence_threshold >= 1.0:
                needs_detection = torch.ones_like(eligible_channels)
            else:
                needs_detection = (
                    torch.sigmoid(pooled_logits.detach())
                    < self.speaker_existence_threshold
                )
            channel_loss = torch.where(
                eligible_channels & needs_detection,
                torch.nn.functional.softplus(-pooled_logits),
                torch.zeros_like(pooled_logits),
            )
            loss = (channel_loss.sum(dim=1) / num_spks).mean()
        return loss

    def _phantom_entry_loss(self, speaker_logits, targets_pil, target_lens):
        """
        Penalize the first streaming chunk where a target-empty channel appears.

        This is a simple proxy for cache-entry prevention: output frames are grouped
        using the model's fixed training chunk length. The earliest chunk containing a
        detached prediction above ``phantom_entry_threshold`` is selected per empty
        channel, then negative BCE is averaged over frames in that chunk whose prediction
        exceeds the lower regular ``phantom_threshold``. Later persistent phantom
        activity is left to the regular phantom and primary BCE losses. If a channel
        never crosses the entry threshold, all of its frames above ``phantom_threshold``
        are used as a fallback. Frame aggregation reuses the regular phantom
        mean/log-mean-exp settings.
        """
        num_frames, num_spks = speaker_logits.shape[1], speaker_logits.shape[2]
        valid = (
            torch.arange(num_frames, device=speaker_logits.device).unsqueeze(0)
            < target_lens.to(speaker_logits.device).unsqueeze(1)
        )
        valid_expanded = valid.unsqueeze(-1)
        empty_channels = ~((targets_pil > 0.5) & valid_expanded).any(dim=1)
        preds = torch.sigmoid(speaker_logits.detach())
        entry_trigger = (
            valid_expanded
            & empty_channels.unsqueeze(1)
            & (preds > self.phantom_entry_threshold)
        )
        aggregate_selected = (
            valid_expanded
            & empty_channels.unsqueeze(1)
            & (preds > self.phantom_threshold)
        )

        chunk_frames = max(
            int(self.sortformer_modules.chunk_len * self.upsample_factor),
            1,
        )
        first_selected_frame = find_first_nonzero(
            entry_trigger.float(),
            max_cap_val=num_frames,
        )
        has_entry = entry_trigger.any(dim=1)
        first_selected_chunk = torch.div(
            first_selected_frame,
            chunk_frames,
            rounding_mode='floor',
        )
        frame_chunk = torch.div(
            torch.arange(num_frames, device=speaker_logits.device),
            chunk_frames,
            rounding_mode='floor',
        ).view(1, -1, 1)
        entry_or_fallback = (
            has_entry.unsqueeze(1)
            & (frame_chunk == first_selected_chunk.unsqueeze(1))
        ) | ~has_entry.unsqueeze(1)
        entry_selected = (
            aggregate_selected
            & entry_or_fallback
        )

        with torch.autocast(device_type=speaker_logits.device.type, enabled=False):
            negative_bce = torch.nn.functional.softplus(speaker_logits.float())
            channel_loss = self._aggregate_selected_frame_losses(
                negative_bce,
                entry_selected,
                use_logmeanexp=self.phantom_logmeanexp,
                temperature=self.phantom_logmeanexp_temperature,
            )
            loss = (channel_loss.sum(dim=1) / num_spks).mean()
        return loss

    def _prearrival_loss(self, speaker_logits, targets_pil, target_lens):
        """
        Penalize high-confidence speaker activity substantially before first arrival.

        A frame/channel pair is eligible when its detached prediction exceeds
        ``prearrival_threshold`` and the PIL-aligned speaker has no target onset within
        the prefix ending ``prearrival_grace_frames`` after that frame. Speakers with no
        target activity in the valid sequence remain eligible throughout. Selected-frame
        negative BCE is aggregated per channel using either the mean or
        temperature-controlled log-mean-exp, then affected channels are summed and
        divided by the fixed number of model speaker slots before the batch mean.
        """
        num_frames, num_spks = speaker_logits.shape[1], speaker_logits.shape[2]
        valid = (
            torch.arange(num_frames, device=speaker_logits.device).unsqueeze(0)
            < target_lens.to(speaker_logits.device).unsqueeze(1)
        )
        valid_expanded = valid.unsqueeze(-1)
        hard_targets = (targets_pil > 0.5) & valid_expanded

        first_onsets = find_first_nonzero(
            hard_targets,
            max_cap_val=num_frames,
        )
        has_onset = first_onsets < num_frames
        frame_indices = torch.arange(
            num_frames, device=speaker_logits.device
        ).view(1, -1, 1)
        safely_before_arrival = (
            ~has_onset.unsqueeze(1)
            | (
                frame_indices + self.prearrival_grace_frames
                < first_onsets.unsqueeze(1)
            )
        )
        selected = (
            valid_expanded
            & safely_before_arrival
            & (
                torch.sigmoid(speaker_logits.detach())
                > self.prearrival_threshold
            )
        )

        with torch.autocast(device_type=speaker_logits.device.type, enabled=False):
            negative_bce = torch.nn.functional.softplus(speaker_logits.float())
            channel_loss = self._aggregate_selected_frame_losses(
                negative_bce,
                selected,
                use_logmeanexp=self.prearrival_logmeanexp,
                temperature=self.prearrival_logmeanexp_temperature,
            )
            loss = (channel_loss.sum(dim=1) / num_spks).mean()
        return loss

    def _get_aux_train_evaluations(
        self, preds, targets, target_lens, activity_logits=None, speaker_logits=None
    ) -> dict:
        """
        Compute auxiliary training evaluations including losses and metrics.

        This function calculates various losses and metrics for the training process,
        including Arrival Time Sort (ATS) Loss and Permutation Invariant Loss (PIL)
        based evaluations.

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
            if speaker_logits is not None:
                speaker_logits = speaker_logits[:, : targets.shape[1], :]
            if activity_logits is not None:
                activity_logits = activity_logits[:, : targets.shape[1], :]
        targets_ats, _ = get_ats_targets_hungarian(
            targets.clone(), preds, tolerance=self.ats_tolerance, metric=self.match_metric, apply_sigmoid=False
        )
        targets_pil, _ = get_pil_targets_hungarian(
            targets.clone(), preds, metric=self.match_metric, apply_sigmoid=False
        )
        # Soften only the loss targets (boundary + label smoothing); metrics below
        # keep using the original hard targets so F1/accuracy stay unaffected.
        targets_ats_loss = self._soften_targets_for_loss(targets_ats, target_lens)
        targets_pil_loss = self._soften_targets_for_loss(targets_pil, target_lens)
        ats_loss = self.loss(
            logits=speaker_logits,
            labels=targets_ats_loss,
            target_lens=target_lens,
        )
        pil_loss = self.loss(
            logits=speaker_logits,
            labels=targets_pil_loss,
            target_lens=target_lens,
        )
        zero_loss = speaker_logits.new_zeros((), dtype=torch.float32)
        rank_loss = (
            self._speaker_rank_loss(speaker_logits, targets_pil, target_lens)
            if self.rank_weight > 0.0
            else zero_loss
        )
        speech_bce_loss = (
            self._speech_bce_loss(speaker_logits, targets_pil, target_lens)
            if self.speech_bce_weight > 0.0
            else zero_loss
        )
        interior_focal_loss = (
            self._interior_focal_loss(speaker_logits, targets_pil, target_lens)
            if self.interior_focal_weight > 0.0
            else zero_loss
        )
        purity_focal_loss = (
            self._purity_focal_loss(speaker_logits, targets_pil, target_lens)
            if self.purity_focal_weight > 0.0
            else zero_loss
        )
        pairwise_ats_loss = (
            self._pairwise_ats_loss(preds, target_lens)
            if self.pairwise_ats_weight > 0.0
            else zero_loss
        )
        self_ats_loss = self._self_ats_loss(speaker_logits, target_lens)
        spkcount_loss = self._spkcount_loss(preds, targets, target_lens)
        if self.activity_weight > 0.0:
            if self.activity_loss_mode == "speaker_preds":
                activity_logits = self._activity_logits_from_speaker_preds(preds)
            activity_loss = self._activity_loss(activity_logits, targets, target_lens)
        else:
            activity_loss = zero_loss
        presence_loss = (
            self._presence_loss(preds, targets_pil, target_lens)
            if self.presence_weight > 0.0
            else zero_loss
        )
        dice_loss = (
            self._dice_loss(preds, targets_pil, target_lens)
            if self.dice_weight > 0.0
            else zero_loss
        )
        phantom_loss = self._phantom_loss(speaker_logits, targets_pil, target_lens)
        speaker_existence_targets = (
            targets_ats
            if self.speaker_existence_target == "ats"
            else targets_pil
        )
        speaker_existence_loss = self._speaker_existence_loss(
            speaker_logits, speaker_existence_targets, target_lens
        )
        phantom_entry_loss = (
            self._phantom_entry_loss(speaker_logits, targets_pil, target_lens)
            if self.phantom_entry_weight > 0.0
            else zero_loss
        )
        prearrival_loss = (
            self._prearrival_loss(speaker_logits, targets_pil, target_lens)
            if self.prearrival_weight > 0.0
            else zero_loss
        )
        loss = (
            self.ats_weight * ats_loss
            + self.pil_weight * pil_loss
            + self.rank_weight * rank_loss
            + self.speech_bce_weight * speech_bce_loss
            + self.interior_focal_weight * interior_focal_loss
            + self.purity_focal_weight * purity_focal_loss
            + self.pairwise_ats_weight * pairwise_ats_loss
            + self.self_ats_weight * self_ats_loss
            + self.spkcount_weight * spkcount_loss
            + self.activity_weight * activity_loss
            + self.presence_weight * presence_loss
            + self.dice_weight * dice_loss
            + self.phantom_weight * phantom_loss
            + self.speaker_existence_weight * speaker_existence_loss
            + self.phantom_entry_weight * phantom_entry_loss
            + self.prearrival_weight * prearrival_loss
        )

        self._accuracy_train(preds, targets_pil, target_lens)
        train_f1_acc, train_precision, train_recall = self._accuracy_train.compute()

        self._accuracy_train_ats(preds, targets_ats, target_lens)
        train_f1_acc_ats, _, _ = self._accuracy_train_ats.compute()
        train_speaker_count_mae, train_speaker_count_accuracy = self._speaker_count_metrics(
            preds, targets, target_lens
        )

        train_metrics = {
            'loss': loss,
            'ats_loss': ats_loss,
            'pil_loss': pil_loss,
            'rank_loss': rank_loss,
            'speech_bce_loss': speech_bce_loss,
            'interior_focal_loss': interior_focal_loss,
            'purity_focal_loss': purity_focal_loss,
            'pairwise_ats_loss': pairwise_ats_loss,
            'self_ats_loss': self_ats_loss,
            'spkcount_loss': spkcount_loss,
            'activity_loss': activity_loss,
            'presence_loss': presence_loss,
            'dice_loss': dice_loss,
            'phantom_loss': phantom_loss,
            'speaker_existence_loss': speaker_existence_loss,
            'phantom_entry_loss': phantom_entry_loss,
            'prearrival_loss': prearrival_loss,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'train_f1_acc': train_f1_acc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1_acc_ats': train_f1_acc_ats,
            'train_speaker_count_mae': train_speaker_count_mae,
            'train_speaker_count_accuracy': train_speaker_count_accuracy,
        }
        return train_metrics

    def on_train_start(self):
        super().on_train_start()
        # Ad-hoc fix: on resume PL restores the LR scheduler's `max_steps` from the
        # checkpoint, overwriting the (possibly extended) value built from config.
        # For InverseSquareRootAnnealing this re-activates the
        # `step > max_steps -> min_lr` clamp and pins the LR at min_lr. Re-apply the
        # trainer's max_steps here, after restoration and before training begins.
        max_steps = self.trainer.max_steps if self.trainer is not None else None
        if max_steps is not None and max_steps > 0:
            schedulers = self.lr_schedulers()
            if schedulers is None:
                schedulers = []
            elif not isinstance(schedulers, (list, tuple)):
                schedulers = [schedulers]
            for sched in schedulers:
                if hasattr(sched, "max_steps") and sched.max_steps != max_steps:
                    logging.info(
                        f"Overriding restored scheduler max_steps "
                        f"{sched.max_steps} -> {max_steps} from trainer config."
                    )
                    sched.max_steps = max_steps

    def on_before_optimizer_step(self, optimizer):
        """
        Report the first non-finite gradient encountered before an optimizer step.

        Lightning calls this hook once gradients are accumulated and all-reduced, but
        before clipping. That ordering is what makes the check useful: norm-based
        clipping scales every gradient by a single shared coefficient, so one NaN
        element turns the global norm into NaN and writes NaN into all parameters and
        into the optimizer moment buffers, which training never recovers from. Under
        bf16 there is no gradient scaler to detect this, so nothing else in the step
        inspects finiteness.

        Only the first event is reported; afterwards every step would repeat it.
        """
        if self._reported_nonfinite_grad:
            return

        grads = [param.grad for param in self.parameters() if param.grad is not None]
        if not grads:
            return

        # Single host sync for the whole model. The per-parameter scan that names the
        # culprit runs only after a hit, so the healthy path stays cheap.
        if torch.stack([torch.isfinite(grad).all() for grad in grads]).all():
            return

        self._reported_nonfinite_grad = True
        for name, param in self.named_parameters():
            if param.grad is None or torch.isfinite(param.grad).all():
                continue
            logging.error(
                f"Non-finite gradient at step {self.global_step} in '{name}': "
                f"{torch.isnan(param.grad).sum().item()} NaNs, "
                f"{torch.isinf(param.grad).sum().item()} Infs out of {param.grad.numel()} "
                "elements. Gradient clipping will propagate this to every parameter. "
                "Further occurrences will not be reported."
            )
            break

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
        audio_signal, audio_signal_length, targets, target_lens, *batch_metadata = batch
        speaker_names = batch_metadata[0] if batch_metadata else None
        logging.info(f"audio_signal.shape: {audio_signal.shape}, targets.shape: {targets.shape}, target_lens: {target_lens}")
        audio_signal = self._apply_batch_noise_augmentation(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
            targets=targets,
            target_lens=target_lens,
        )
        chunk_replace_rate = None
        if self.streaming_mode and self.batch_chunk_replace_probability > 0.0:
            processed_signal, processed_signal_length = self.process_signal(
                audio_signal=audio_signal,
                audio_signal_length=audio_signal_length,
            )
            processed_signal = processed_signal[:, :, : processed_signal_length.max()]
            processed_signal, targets, chunk_replace_rate = (
                self._apply_batch_chunk_replace_augmentation(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    targets=targets,
                    target_lens=target_lens,
                    speaker_names=speaker_names,
                )
            )
            outputs = self.forward_streaming(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
            )
        else:
            outputs = self.forward(
                audio_signal=audio_signal,
                audio_signal_length=audio_signal_length,
            )
        preds, speaker_logits, activity_logits = outputs
        train_metrics = self._get_aux_train_evaluations(
            preds,
            targets,
            target_lens,
            activity_logits=activity_logits,
            speaker_logits=speaker_logits,
        )
        if chunk_replace_rate is not None:
            train_metrics['batch_chunk_replace_rate'] = chunk_replace_rate
        self._reset_train_metrics()
        self.log_dict(train_metrics, sync_dist=True, on_step=True, on_epoch=False, logger=True)
        return {'loss': train_metrics['loss']}

    def _get_aux_validation_evaluations(
        self, preds, targets, target_lens, activity_logits=None, speaker_logits=None
    ) -> dict:
        """
        Compute auxiliary validation evaluations including losses and metrics.

        This function calculates various losses and metrics for the training process,
        including Arrival Time Sort (ATS) Loss and Permutation Invariant Loss (PIL)
        based evaluations.

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
            if speaker_logits is not None:
                speaker_logits = speaker_logits[:, : targets.shape[1], :]
            if activity_logits is not None:
                activity_logits = activity_logits[:, : targets.shape[1], :]
        targets_ats, _ = get_ats_targets_hungarian(
            targets.clone(), preds, tolerance=self.ats_tolerance, metric=self.match_metric, apply_sigmoid=False
        )
        targets_pil, _ = get_pil_targets_hungarian(
            targets.clone(), preds, metric=self.match_metric, apply_sigmoid=False
        )

        # Soften only the loss targets (boundary + label smoothing); metrics below
        # keep using the original hard targets so F1/accuracy stay unaffected.
        targets_ats_loss = self._soften_targets_for_loss(targets_ats, target_lens)
        targets_pil_loss = self._soften_targets_for_loss(targets_pil, target_lens)
        val_ats_loss = self.loss(
            logits=speaker_logits,
            labels=targets_ats_loss,
            target_lens=target_lens,
        )
        val_pil_loss = self.loss(
            logits=speaker_logits,
            labels=targets_pil_loss,
            target_lens=target_lens,
        )
        val_rank_loss = self._speaker_rank_loss(speaker_logits, targets_pil, target_lens)
        val_speech_bce_loss = self._speech_bce_loss(speaker_logits, targets_pil, target_lens)
        val_interior_focal_loss = self._interior_focal_loss(
            speaker_logits, targets_pil, target_lens
        )
        val_purity_focal_loss = self._purity_focal_loss(
            speaker_logits, targets_pil, target_lens
        )
        val_pairwise_ats_loss = self._pairwise_ats_loss(preds, target_lens)
        val_self_ats_loss = self._self_ats_loss(speaker_logits, target_lens)
        val_spkcount_loss = self._spkcount_loss(preds, targets, target_lens)
        if self.activity_weight > 0.0 and self.activity_loss_mode == "speaker_preds":
            activity_logits = self._activity_logits_from_speaker_preds(preds)
        val_activity_loss = self._activity_loss(activity_logits, targets, target_lens)
        val_presence_loss = self._presence_loss(preds, targets_pil, target_lens)
        val_dice_loss = self._dice_loss(preds, targets_pil, target_lens)
        val_phantom_loss = self._phantom_loss(
            speaker_logits, targets_pil, target_lens
        )
        speaker_existence_targets = (
            targets_ats
            if self.speaker_existence_target == "ats"
            else targets_pil
        )
        val_speaker_existence_loss = self._speaker_existence_loss(
            speaker_logits, speaker_existence_targets, target_lens
        )
        val_phantom_entry_loss = self._phantom_entry_loss(
            speaker_logits, targets_pil, target_lens
        )
        val_prearrival_loss = self._prearrival_loss(
            speaker_logits, targets_pil, target_lens
        )
        val_loss = (
            self.ats_weight * val_ats_loss
            + self.pil_weight * val_pil_loss
            + self.rank_weight * val_rank_loss
            + self.speech_bce_weight * val_speech_bce_loss
            + self.interior_focal_weight * val_interior_focal_loss
            + self.purity_focal_weight * val_purity_focal_loss
            + self.pairwise_ats_weight * val_pairwise_ats_loss
            + self.self_ats_weight * val_self_ats_loss
            + self.spkcount_weight * val_spkcount_loss
            + self.activity_weight * val_activity_loss
            + self.presence_weight * val_presence_loss
            + self.dice_weight * val_dice_loss
            + self.phantom_weight * val_phantom_loss
            + self.speaker_existence_weight * val_speaker_existence_loss
            + self.phantom_entry_weight * val_phantom_entry_loss
            + self.prearrival_weight * val_prearrival_loss
        )

        self._accuracy_valid(preds, targets_pil, target_lens)
        val_f1_acc, val_precision, val_recall = self._accuracy_valid.compute()

        self._accuracy_valid_ats(preds, targets_ats, target_lens)
        valid_f1_acc_ats, _, _ = self._accuracy_valid_ats.compute()
        val_speaker_count_mae, val_speaker_count_accuracy = self._speaker_count_metrics(
            preds, targets, target_lens
        )

        self._accuracy_valid.reset()
        self._accuracy_valid_ats.reset()

        val_metrics = {
            'val_loss': val_loss,
            'val_ats_loss': val_ats_loss,
            'val_pil_loss': val_pil_loss,
            'val_rank_loss': val_rank_loss,
            'val_speech_bce_loss': val_speech_bce_loss,
            'val_interior_focal_loss': val_interior_focal_loss,
            'val_purity_focal_loss': val_purity_focal_loss,
            'val_pairwise_ats_loss': val_pairwise_ats_loss,
            'val_self_ats_loss': val_self_ats_loss,
            'val_spkcount_loss': val_spkcount_loss,
            'val_activity_loss': val_activity_loss,
            'val_presence_loss': val_presence_loss,
            'val_dice_loss': val_dice_loss,
            'val_phantom_loss': val_phantom_loss,
            'val_speaker_existence_loss': val_speaker_existence_loss,
            'val_phantom_entry_loss': val_phantom_entry_loss,
            'val_prearrival_loss': val_prearrival_loss,
            'val_f1_acc': val_f1_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1_acc_ats': valid_f1_acc_ats,
            'val_speaker_count_mae': val_speaker_count_mae,
            'val_speaker_count_accuracy': val_speaker_count_accuracy,
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
        audio_signal, audio_signal_length, targets, target_lens, *_ = batch
        logging.info(f"audio_signal.shape: {audio_signal.shape}, targets.shape: {targets.shape}, target_lens: {target_lens}")
        outputs = self.forward(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
        )
        preds, speaker_logits, activity_logits = outputs
        val_metrics = self._get_aux_validation_evaluations(
            preds,
            targets,
            target_lens,
            activity_logits=activity_logits,
            speaker_logits=speaker_logits,
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
        val_rank_loss_mean = torch.stack([x['val_rank_loss'] for x in outputs]).mean()
        val_speech_bce_loss_mean = torch.stack([x['val_speech_bce_loss'] for x in outputs]).mean()
        val_interior_focal_loss_mean = torch.stack(
            [x['val_interior_focal_loss'] for x in outputs]
        ).mean()
        val_purity_focal_loss_mean = torch.stack(
            [x['val_purity_focal_loss'] for x in outputs]
        ).mean()
        val_pairwise_ats_loss_mean = torch.stack([x['val_pairwise_ats_loss'] for x in outputs]).mean()
        val_self_ats_loss_mean = torch.stack([x['val_self_ats_loss'] for x in outputs]).mean()
        val_spkcount_loss_mean = torch.stack([x['val_spkcount_loss'] for x in outputs]).mean()
        val_activity_loss_mean = torch.stack([x['val_activity_loss'] for x in outputs]).mean()
        val_presence_loss_mean = torch.stack([x['val_presence_loss'] for x in outputs]).mean()
        val_dice_loss_mean = torch.stack([x['val_dice_loss'] for x in outputs]).mean()
        val_phantom_loss_mean = torch.stack([x['val_phantom_loss'] for x in outputs]).mean()
        val_speaker_existence_loss_mean = torch.stack(
            [x['val_speaker_existence_loss'] for x in outputs]
        ).mean()
        val_phantom_entry_loss_mean = torch.stack(
            [x['val_phantom_entry_loss'] for x in outputs]
        ).mean()
        val_prearrival_loss_mean = torch.stack([x['val_prearrival_loss'] for x in outputs]).mean()
        val_f1_acc_mean = torch.stack([x['val_f1_acc'] for x in outputs]).mean()
        val_precision_mean = torch.stack([x['val_precision'] for x in outputs]).mean()
        val_recall_mean = torch.stack([x['val_recall'] for x in outputs]).mean()
        val_f1_acc_ats_mean = torch.stack([x['val_f1_acc_ats'] for x in outputs]).mean()
        val_speaker_count_mae_mean = torch.stack([x['val_speaker_count_mae'] for x in outputs]).mean()
        val_speaker_count_accuracy_mean = torch.stack(
            [x['val_speaker_count_accuracy'] for x in outputs]
        ).mean()

        self._reset_valid_metrics()

        multi_val_metrics = {
            'val_loss': val_loss_mean,
            'val_ats_loss': val_ats_loss_mean,
            'val_pil_loss': val_pil_loss_mean,
            'val_rank_loss': val_rank_loss_mean,
            'val_speech_bce_loss': val_speech_bce_loss_mean,
            'val_interior_focal_loss': val_interior_focal_loss_mean,
            'val_purity_focal_loss': val_purity_focal_loss_mean,
            'val_pairwise_ats_loss': val_pairwise_ats_loss_mean,
            'val_self_ats_loss': val_self_ats_loss_mean,
            'val_spkcount_loss': val_spkcount_loss_mean,
            'val_activity_loss': val_activity_loss_mean,
            'val_presence_loss': val_presence_loss_mean,
            'val_dice_loss': val_dice_loss_mean,
            'val_phantom_loss': val_phantom_loss_mean,
            'val_speaker_existence_loss': val_speaker_existence_loss_mean,
            'val_phantom_entry_loss': val_phantom_entry_loss_mean,
            'val_prearrival_loss': val_prearrival_loss_mean,
            'val_f1_acc': val_f1_acc_mean,
            'val_precision': val_precision_mean,
            'val_recall': val_recall_mean,
            'val_f1_acc_ats': val_f1_acc_ats_mean,
            'val_speaker_count_mae': val_speaker_count_mae_mean,
            'val_speaker_count_accuracy': val_speaker_count_accuracy_mean,
        }
        return {'log': multi_val_metrics}

    def _get_aux_test_batch_evaluations(self, batch_idx: int, preds, targets, target_lens):
        """
        Compute auxiliary validation evaluations including losses and metrics.

        This function calculates various losses and metrics for the training process,
        including Arrival Time Sort (ATS) Loss and Permutation Invariant Loss (PIL)
        based evaluations.

        Args:
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
        targets_ats, _ = get_ats_targets_hungarian(
            targets.clone(), preds, tolerance=self.ats_tolerance, metric=self.match_metric, apply_sigmoid=False
        )
        targets_pil, _ = get_pil_targets_hungarian(
            targets.clone(), preds, metric=self.match_metric, apply_sigmoid=False
        )
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
                audio_signal, audio_signal_length, targets, target_lens, *_ = batch
                audio_signal = audio_signal.to(self.device)
                audio_signal_length = audio_signal_length.to(self.device)
                targets = targets.to(self.device)
                preds, _, _ = self.forward(
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
