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

import os

import lightning.pytorch as pl
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

"""
Example training session (single node training)
For training, you can use the following precisions: 32, bf16 and bf16-mixed.
You can train with a larger batch size using BF16 mixed precision.

python ./sortformer_diar_train.py --config-path='../conf/neural_diarizer' \
    --config-name='sortformer_diarizer_hybrid_loss_4spk-v1.yaml' \
    trainer.precision='bf16' \
    trainer.devices=1 \
    model.train_ds.manifest_filepath="<train_manifest_path>" \
    model.validation_ds.manifest_filepath="<dev_manifest_path>" \
    exp_manager.name='sample_train' \
    exp_manager.exp_dir='./sortformer_diar_train'

Same initialization seed but different training noise (sampling/dropout):
python ./sortformer_diar_train.py --config-path='../conf/neural_diarizer' \
    --config-name='sortformer_diarizer_hybrid_loss_4spk-v1.yaml' \
    init_seed=42 \
    train_seed=101
"""


def _get_checkpoint_callback(trainer: pl.Trainer):
    """Return the checkpoint callback created by exp_manager."""
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            return callback
    return None


def _rank_zero_average_and_save(trainer: pl.Trainer, model: SortformerEncLabelModel) -> str:
    """Load top-k checkpoints, average them, and save as .nemo (rank 0 only)."""
    checkpoint_callback = _get_checkpoint_callback(trainer)
    if checkpoint_callback is None:
        logging.warning("No ModelCheckpoint callback was found. Skipping post-training checkpoint averaging.")
        return ""

    best_k_models = getattr(checkpoint_callback, "best_k_models", None)
    if not best_k_models:
        logging.warning("No top-k checkpoints were found. Skipping post-training checkpoint averaging.")
        return ""

    reverse = getattr(checkpoint_callback, "mode", "max") == "max"
    ranked_ckpts = sorted(best_k_models.items(), key=lambda item: float(item[1]), reverse=reverse)
    selected_ckpts = [str(path) for path, _ in ranked_ckpts]

    selected_ckpts = [path for path in selected_ckpts if not path.endswith("-last.ckpt")]
    if checkpoint_callback.save_top_k not in (None, -1):
        selected_ckpts = selected_ckpts[: int(checkpoint_callback.save_top_k)]

    if len(selected_ckpts) == 0:
        logging.warning("No checkpoints remained after filtering. Skipping averaging.")
        return ""

    logging.info(f"Averaging {len(selected_ckpts)} checkpoints.")
    device = torch.device("cpu")
    avg_state = {}
    non_float_state = {}
    ref_dtypes = {}
    ref_keys = None

    for idx, path in enumerate(selected_ckpts):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        if idx == 0:
            ref_keys = list(state_dict.keys())
            ref_key_set = set(ref_keys)
            for key, value in state_dict.items():
                tensor = value.detach().cpu()
                ref_dtypes[key] = tensor.dtype
                if torch.is_floating_point(tensor):
                    avg_state[key] = tensor.to(torch.float32)
                else:
                    non_float_state[key] = tensor
            continue

        if set(state_dict.keys()) != ref_key_set:
            raise RuntimeError(f"State dict mismatch while averaging checkpoints: {path}")

        for key in ref_keys:
            tensor = state_dict[key].detach().cpu()
            if torch.is_floating_point(tensor):
                avg_state[key] = avg_state[key] + tensor.to(torch.float32)

    num_ckpts = len(selected_ckpts)
    merged_state = {}
    for key in ref_keys:
        if key in avg_state:
            merged_state[key] = (avg_state[key] / num_ckpts).to(ref_dtypes[key])
        else:
            merged_state[key] = non_float_state[key]

    model.load_state_dict(merged_state, strict=True)

    output_name = "model_averaged.nemo"
    output_dir = str(getattr(checkpoint_callback, "dirpath", "") or os.path.dirname(selected_ckpts[0]))
    output_path = os.path.join(output_dir, output_name)

    model.save_to(output_path)
    logging.info(f"Averaged .nemo model exported to: {output_path}")
    return output_path


def _average_checkpoints_after_training(trainer: pl.Trainer, model: SortformerEncLabelModel) -> str:
    """Average top-k checkpoints and export an averaged .nemo model.

    Only the global-zero rank performs I/O and averaging; all ranks
    synchronize via a barrier before returning.
    """
    output_path = _rank_zero_average_and_save(trainer, model) if trainer.is_global_zero else ""
    trainer.strategy.barrier("checkpoint_averaging_done")
    return output_path


@hydra_runner(config_path="../conf/neural_diarizer", config_name="sortformer_diarizer_hybrid_loss_4spk-v1.yaml")
def main(cfg):
    """Main function for training the sortformer diarizer model."""
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    init_seed = int(cfg.get("init_seed", 42))
    train_seed_cfg = cfg.get("train_seed", None)
    train_seed = init_seed if train_seed_cfg is None else int(train_seed_cfg)
    seed_workers = bool(cfg.get("seed_workers", True))

    # Seed before model construction so parameter initialization is reproducible.
    seed_everything(init_seed, workers=seed_workers)
    logging.info(f"Seeding: init_seed={init_seed}, train_seed={train_seed}, seed_workers={seed_workers}")

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    sortformer_model = SortformerEncLabelModel(cfg=cfg.model, trainer=trainer)
    sortformer_model.maybe_init_from_pretrained_checkpoint(cfg)

    # Optional re-seed before fit() to vary training noise while keeping init fixed.
    if train_seed != init_seed:
        seed_everything(train_seed, workers=seed_workers)
        logging.info("RNGs reseeded before trainer.fit() to vary training/sampling noise.")

    trainer.fit(sortformer_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if sortformer_model.prepare_test(trainer):
            trainer.test(sortformer_model)

    avg_model_path = _average_checkpoints_after_training(trainer=trainer, model=sortformer_model)
    if avg_model_path:
        logging.info(f"Finished post-training checkpoint averaging: {avg_model_path}")


if __name__ == '__main__':
    main()
