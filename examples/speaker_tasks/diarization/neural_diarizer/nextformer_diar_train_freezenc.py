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

from nemo.collections.asr.models import NextformerEncLabelModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
import torch
"""
Example training session (single node training)
For training, you can use the following precisions: 32, bf16 and bf16-mixed.
You can train with a larger batch size using BF16 mixed precision.

python ./nextformer_diar_train.py --config-path='../conf/neural_diarizer' \
    --config-name='nextformer_diarizer.yaml' \
    trainer.precision='bf16' \
    trainer.devices=1 \
    model.train_ds.manifest_filepath="<train_manifest_path>" \
    model.validation_ds.manifest_filepath="<dev_manifest_path>" \
    exp_manager.name='sample_train' \
    exp_manager.exp_dir='./nextformer_diar_train'
"""

seed_everything(42)

def enable_bn_se(m):
    if type(m) == torch.nn.BatchNorm1d:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

@hydra_runner(config_path="../conf/neural_diarizer", config_name="nextformer_diarizer.yaml")
def main(cfg):
    """Main function for training the nextformer diarizer model."""
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    nextformer_model = NextformerEncLabelModel(cfg=cfg.model, trainer=trainer)
    nextformer_model.maybe_init_from_pretrained_checkpoint(cfg)

    logging.info("freezing encoder")
    nextformer_model.encoder.freeze()
    nextformer_model.encoder.apply(enable_bn_se)

    trainer.fit(nextformer_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if nextformer_model.prepare_test(trainer):
            trainer.test(nextformer_model)


if __name__ == '__main__':
    main()
