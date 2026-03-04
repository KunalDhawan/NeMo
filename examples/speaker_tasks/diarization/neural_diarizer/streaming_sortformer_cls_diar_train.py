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

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from nemo.collections.asr.models import SortformerCLSEncLabelModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

from checkpoint_averaging import average_checkpoints_after_training

"""
Example training session (single node training)
For training, you can use the following precisions: 32, bf16 and bf16-mixed.
You can train with a larger batch size using BF16 mixed precision.

python ./streaming_sortformer_cls_diar_train.py --config-path='../conf/neural_diarizer' \
    --config-name='streaming_sortformer_cls_diarizer_4spk-v2.yaml' \
    trainer.precision='bf16' \
    trainer.devices=1 \
    model.train_ds.manifest_filepath="<train_manifest_path>" \
    model.validation_ds.manifest_filepath="<dev_manifest_path>" \
    exp_manager.name='sample_train' \
    exp_manager.exp_dir='./streaming_sortformer_cls_diar_train'
"""

@hydra_runner(config_path="../conf/neural_diarizer", config_name="streaming_sortformer_cls_diarizer_4spk-v2.yaml")
def main(cfg):
    """Main function for training the SortformerCLS diarizer model."""
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    init_seed = int(cfg.get("init_seed", 42))
    train_seed_cfg = cfg.get("train_seed", None)
    train_seed = init_seed if train_seed_cfg is None else int(train_seed_cfg)
    seed_workers = bool(cfg.get("seed_workers", True))

    seed_everything(init_seed, workers=seed_workers)
    logging.info(f"Seeding: init_seed={init_seed}, train_seed={train_seed}, seed_workers={seed_workers}")

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    sortformer_model = SortformerCLSEncLabelModel(cfg=cfg.model, trainer=trainer)
    sortformer_model.maybe_init_from_pretrained_checkpoint(cfg)

    if train_seed != init_seed:
        seed_everything(train_seed, workers=seed_workers)
        logging.info("RNGs reseeded before trainer.fit() to vary training/sampling noise.")

    trainer.fit(sortformer_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if sortformer_model.prepare_test(trainer):
            trainer.test(sortformer_model)

    avg_model_path = average_checkpoints_after_training(trainer=trainer, model=sortformer_model)
    if avg_model_path:
        logging.info(f"Finished post-training checkpoint averaging: {avg_model_path}")
        if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
            if sortformer_model.prepare_test(trainer):
                logging.info("Testing averaged model...")
                trainer.test(sortformer_model)


if __name__ == '__main__':
    main()
