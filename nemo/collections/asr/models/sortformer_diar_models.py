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

        self.eps = 0.004   # bf16-safe epsilon
        self.negative_init_val = -99
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
        total_weight = pil_weight + ats_weight
        if total_weight == 0:
            raise ValueError(f"weights for PIL {pil_weight} and ATS {ats_weight} cannot sum to 0")
        self.pil_weight = pil_weight / total_weight
        self.ats_weight = ats_weight / total_weight
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

    def forward_infer(self, emb_seq, emb_seq_length, pre_encode_feats=None):
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
            preds (torch.Tensor): Sorted tensor containing Sigmoid values for predicted speaker labels.
                Shape: (batch_size, diar_frame_count [* upsample_factor], num_speakers)
        """
        encoder_mask = self.sortformer_modules.length_to_mask(emb_seq_length, emb_seq.shape[1])
        trans_emb_seq = self.transformer_encoder(encoder_states=emb_seq, encoder_mask=encoder_mask)
        trans_emb_seq = self.sortformer_modules.upsample_hidden(trans_emb_seq, pre_encode_feats=pre_encode_feats)
        if self.upsample_factor > 1:
            encoder_mask = encoder_mask.repeat_interleave(self.upsample_factor, dim=1)
        _preds = self.sortformer_modules.forward_speaker_sigmoids(trans_emb_seq)
        preds = _preds * encoder_mask.unsqueeze(-1)
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
            'subsegment_nspk_bias': config.get('subsegment_nspk_bias', 1.0),
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
            preds = self.forward_streaming(processed_signal, processed_signal_length)
            # When upsample_factor > 1, forward_streaming_step already collects
            # fine-resolution chunk preds from the learnable upsampler, so
            # total_preds is already at target resolution — no further upsampling.
            if self.upsample_factor <= 1 and self.val_upsample_preds and not self.training:
                preds = self.sortformer_modules.upsample_preds(
                    preds,
                    upsample_factor=self.output_subsampling_factor,
                    smooth_kernel=self.val_upsample_smooth_kernel,
                )
        else:
            emb_seq, emb_seq_length, pre_encode_feats = self.frontend_encoder(
                processed_signal=processed_signal, processed_signal_length=processed_signal_length
            )
            preds = self.forward_infer(emb_seq, emb_seq_length, pre_encode_feats=pre_encode_feats)
            if self.val_upsample_preds and not self.training:
                preds = self.sortformer_modules.upsample_preds(
                    preds,
                    upsample_factor=self.output_subsampling_factor,
                    smooth_kernel=self.val_upsample_smooth_kernel,
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
        spkcache_fifo_chunk_preds = self.forward_infer(
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

        att_mod = False
        if self.training:
            rand_num = random.random()
            if rand_num < self.sortformer_modules.causal_attn_rate:
                if hasattr(self.encoder, 'att_context_size'):
                    self.encoder.att_context_size = [-1, self.sortformer_modules.causal_attn_rc]
                elif hasattr(self.encoder, 'attn_mode'):
                    self.encoder.attn_mode = "causal"
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
        for _, chunk_feat_seq_t, feat_lengths, left_offset, right_offset in tqdm(
            streaming_loader,
            total=num_chunks,
            desc="Streaming Steps",
            disable=self.training,
        ):
            streaming_state, total_preds = self.forward_streaming_step(
                processed_signal=chunk_feat_seq_t,
                processed_signal_length=feat_lengths,
                streaming_state=streaming_state,
                total_preds=total_preds,
                left_offset=left_offset,
                right_offset=right_offset,
            )

        if att_mod:
            if hasattr(self.encoder, 'att_context_size'):
                self.encoder.att_context_size = [-1, -1]
            elif hasattr(self.encoder, 'attn_mode'):
                self.encoder.attn_mode = "full"
            self.transformer_encoder.diag = None

        del processed_signal, processed_signal_length

        if sig_length < max_n_frames:  # Discard preds corresponding to padding
            n_frames = math.ceil(sig_length / self.encoder.subsampling_factor)
            if self.upsample_factor > 1:
                n_frames *= self.upsample_factor
            total_preds = total_preds[:, :n_frames, :]
        return total_preds

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
        spkcache_fifo_chunk_preds = self.forward_infer(
            emb_seq=spkcache_fifo_chunk_fc_encoder_embs,
            emb_seq_length=spkcache_fifo_chunk_fc_encoder_lengths,
            pre_encode_feats=spkcache_fifo_chunk_pre_encode_embs,
        )

        lc_enc = round(left_offset / self.encoder.subsampling_factor)
        rc_enc = math.ceil(right_offset / self.encoder.subsampling_factor)
        uf = self.upsample_factor

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

        return streaming_state, total_preds

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
        return loss.mean().to(preds.dtype)

    def _self_ats_loss(self, preds, target_lens):
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
            preds (torch.Tensor): Predicted probabilities of shape (B, T, N).
            target_lens (torch.Tensor): Valid sequence lengths of shape (B,).

        Returns:
            torch.Tensor: Scalar self-ATS loss.
        """
        num_spks = preds.shape[2]
        if num_spks < 2:
            return torch.zeros((), device=preds.device, dtype=preds.dtype)

        # Predicted onset (first frame above threshold) per channel; empty channels -> T.
        # Temperature sharpening is monotonic and preserves the 0.5 threshold, so onsets (and
        # therefore the permutation) are computed from the original predictions.
        onsets = find_first_nonzero(preds.detach(), max_cap_val=preds.shape[1])  # (B, N)
        # Permutation sorting channels by predicted onset; stable keeps current order on ties.
        perm = torch.argsort(onsets, dim=1, stable=True)  # (B, N)

        # Optionally sharpen predictions; applied symmetrically so the loss stays
        # zero/zero-gradient when sorted (see docstring). T == 1.0 leaves preds untouched.
        preds_s = self._sharpen_probs(preds, self.self_ats_temperature)

        # Onset-sorted target: position k receives the content of the k-th earliest channel.
        index = perm.unsqueeze(1).expand(-1, preds.shape[1], -1)  # (B, T, N)
        self_ats_target = torch.gather(preds_s, dim=2, index=index).detach()  # (B, T, N)

        if self.self_ats_metric == "mse":
            # Masked mean squared error over valid frames (no entropy floor; 0 when sorted).
            num_frames = preds.shape[1]
            valid = (
                torch.arange(num_frames, device=preds.device).unsqueeze(0)
                < target_lens.to(preds.device).unsqueeze(1)
            ).to(preds.dtype)  # (B, T)
            sq_err = ((preds_s - self_ats_target) ** 2) * valid.unsqueeze(-1)  # (B, T, N)
            denom = valid.sum() * preds.shape[2]  # valid (frame, channel) element count
            return sq_err.sum() / denom.clamp_min(1.0)

        return self.loss(probs=preds_s, labels=self_ats_target, target_lens=target_lens)

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
        return loss.to(preds.dtype)

    def _get_aux_train_evaluations(self, preds, targets, target_lens) -> dict:
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
        targets_ats, _ = get_ats_targets_hungarian(
            targets.clone(), preds, tolerance=self.ats_tolerance, apply_sigmoid=False
        )
        targets_pil, _ = get_pil_targets_hungarian(targets.clone(), preds, apply_sigmoid=False)
        # Soften only the loss targets (boundary + label smoothing); metrics below
        # keep using the original hard targets so F1/accuracy stay unaffected.
        ats_loss = self.loss(
            probs=preds, labels=self._soften_targets_for_loss(targets_ats, target_lens), target_lens=target_lens
        )
        pil_loss = self.loss(
            probs=preds, labels=self._soften_targets_for_loss(targets_pil, target_lens), target_lens=target_lens
        )
        pairwise_ats_loss = self._pairwise_ats_loss(preds, target_lens)
        self_ats_loss = self._self_ats_loss(preds, target_lens)
        spkcount_loss = self._spkcount_loss(preds, targets, target_lens)
        loss = (
            self.ats_weight * ats_loss
            + self.pil_weight * pil_loss
            + self.pairwise_ats_weight * pairwise_ats_loss
            + self.self_ats_weight * self_ats_loss
            + self.spkcount_weight * spkcount_loss
        )

        self._accuracy_train(preds, targets_pil, target_lens)
        train_f1_acc, train_precision, train_recall = self._accuracy_train.compute()

        self._accuracy_train_ats(preds, targets_ats, target_lens)
        train_f1_acc_ats, _, _ = self._accuracy_train_ats.compute()

        train_metrics = {
            'loss': loss,
            'ats_loss': ats_loss,
            'pil_loss': pil_loss,
            'pairwise_ats_loss': pairwise_ats_loss,
            'self_ats_loss': self_ats_loss,
            'spkcount_loss': spkcount_loss,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'train_f1_acc': train_f1_acc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1_acc_ats': train_f1_acc_ats,
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
        audio_signal, audio_signal_length, targets, target_lens, *_ = batch
        logging.info(f"audio_signal.shape: {audio_signal.shape}, targets.shape: {targets.shape}, target_lens: {target_lens}")
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
        targets_ats, _ = get_ats_targets_hungarian(
            targets.clone(), preds, tolerance=self.ats_tolerance, apply_sigmoid=False
        )
        targets_pil, _ = get_pil_targets_hungarian(targets.clone(), preds, apply_sigmoid=False)

        # Soften only the loss targets (boundary + label smoothing); metrics below
        # keep using the original hard targets so F1/accuracy stay unaffected.
        val_ats_loss = self.loss(
            probs=preds, labels=self._soften_targets_for_loss(targets_ats, target_lens), target_lens=target_lens
        )
        val_pil_loss = self.loss(
            probs=preds, labels=self._soften_targets_for_loss(targets_pil, target_lens), target_lens=target_lens
        )
        val_pairwise_ats_loss = self._pairwise_ats_loss(preds, target_lens)
        val_self_ats_loss = self._self_ats_loss(preds, target_lens)
        val_spkcount_loss = self._spkcount_loss(preds, targets, target_lens)
        val_loss = (
            self.ats_weight * val_ats_loss
            + self.pil_weight * val_pil_loss
            + self.pairwise_ats_weight * val_pairwise_ats_loss
            + self.self_ats_weight * val_self_ats_loss
            + self.spkcount_weight * val_spkcount_loss
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
            'val_pairwise_ats_loss': val_pairwise_ats_loss,
            'val_self_ats_loss': val_self_ats_loss,
            'val_spkcount_loss': val_spkcount_loss,
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
        audio_signal, audio_signal_length, targets, target_lens, *_ = batch
        logging.info(f"audio_signal.shape: {audio_signal.shape}, targets.shape: {targets.shape}, target_lens: {target_lens}")
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
        val_pairwise_ats_loss_mean = torch.stack([x['val_pairwise_ats_loss'] for x in outputs]).mean()
        val_self_ats_loss_mean = torch.stack([x['val_self_ats_loss'] for x in outputs]).mean()
        val_spkcount_loss_mean = torch.stack([x['val_spkcount_loss'] for x in outputs]).mean()
        val_f1_acc_mean = torch.stack([x['val_f1_acc'] for x in outputs]).mean()
        val_precision_mean = torch.stack([x['val_precision'] for x in outputs]).mean()
        val_recall_mean = torch.stack([x['val_recall'] for x in outputs]).mean()
        val_f1_acc_ats_mean = torch.stack([x['val_f1_acc_ats'] for x in outputs]).mean()

        self._reset_valid_metrics()

        multi_val_metrics = {
            'val_loss': val_loss_mean,
            'val_ats_loss': val_ats_loss_mean,
            'val_pil_loss': val_pil_loss_mean,
            'val_pairwise_ats_loss': val_pairwise_ats_loss_mean,
            'val_self_ats_loss': val_self_ats_loss_mean,
            'val_spkcount_loss': val_spkcount_loss_mean,
            'val_f1_acc': val_f1_acc_mean,
            'val_precision': val_precision_mean,
            'val_recall': val_recall_mean,
            'val_f1_acc_ats': val_f1_acc_ats_mean,
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
            targets.clone(), preds, tolerance=self.ats_tolerance, apply_sigmoid=False
        )
        targets_pil, _ = get_pil_targets_hungarian(targets.clone(), preds, apply_sigmoid=False)
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
