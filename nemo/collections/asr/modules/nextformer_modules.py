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

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.utils import logging
from nemo.collections.asr.modules.transformer.transformer_modules import PositionWiseFF

__all__ = ['NextformerModules']

class NextformerModules(NeuralModule, Exportable):
    """
    A class including auxiliary functions for Nextformer models.
    This class contains and will contain the following functions that performs streaming features,
    and any neural layers that are not included in the NeMo neural modules
    (e.g. Transformer, Fast-Conformer).
    """

    def init_weights(self, m):
        """Init weights for linear layers."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(
        self,
        ff_dropout_rate: float = 0.5,
        fc_d_model: int = 512,
        tf_d_model: int = 192,
        subsampling_factor: int = 8,
    ):
        super().__init__()
        # General params
        self.subsampling_factor = subsampling_factor
        self.fc_d_model = fc_d_model
        self.tf_d_model = tf_d_model

        self.encoder_proj = nn.Linear(self.fc_d_model, self.tf_d_model)
        self.query_classifier = nn.Linear(self.tf_d_model, 1)
        self.ff = PositionWiseFF(hidden_size=self.tf_d_model, inner_size=4*self.tf_d_model, ffn_dropout=ff_dropout_rate)
        self.log = False

    @staticmethod
    def length_to_mask(lengths, max_length: int):
        """
        Convert length values to encoder mask input tensor

        Args:
            lengths (torch.Tensor): Tensor containing lengths of sequences
            max_length (int): maximum sequence length

        Returns:
            mask (torch.Tensor): Tensor of shape (batch_size, max_len) containing 0's
                                 in the padded region and 1's elsewhere
        """
        batch_size = lengths.shape[0]
        arange = torch.arange(max_length, device=lengths.device)
        mask = arange.expand(batch_size, max_length) < lengths.unsqueeze(1)
        return mask

    def forward_spk_logits(self, emb_seq, spk_queries):
        """
        The final layer that outputs speaker activations (logits).

        Args:
            emb_seq (torch.Tensor): Tensor containing hidden states from the encoder
                Shape: (batch_size, n_frames, emb_dim)
            spk_queries (torch.Tensor): Tensor containing speaker queries
                Shape: (batch_size, n_spk, query_dim)

        Returns:
            logits (torch.Tensor): Tensor containing speaker activations (logits)
                Shape: (batch_size, n_frames, n_spk)
        """
        query_ff_out = self.ff(spk_queries)
        logits = torch.matmul(emb_seq, query_ff_out.transpose(1, 2))
        return logits

    def forward_query_logits(self, spk_queries):
        """
        The final layer that outputs query existence logits.
        """
        query_logits = self.query_classifier(spk_queries).squeeze(-1)
        return query_logits

