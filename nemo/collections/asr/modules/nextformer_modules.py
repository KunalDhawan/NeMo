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
        local_num_spks: int = 4,
        pred_score_threshold: float = 0.25,
    ):
        super().__init__()
        # General params
        self.subsampling_factor = subsampling_factor
        self.fc_d_model = fc_d_model
        self.tf_d_model = tf_d_model
        self.local_num_spks = local_num_spks
        self.pred_score_threshold = pred_score_threshold

        self.encoder_proj = nn.Linear(self.fc_d_model, self.tf_d_model)
        self.query_classifier = nn.Linear(self.tf_d_model, 1)
        self.ff = PositionWiseFF(hidden_size=self.tf_d_model, inner_size=4 * self.tf_d_model, ffn_dropout=ff_dropout_rate)
        self.first_hidden_to_hidden = nn.Linear(self.tf_d_model, self.tf_d_model)
        self.single_hidden_to_spks = nn.Linear(self.tf_d_model, self.local_num_spks)
        self.dropout = nn.Dropout(ff_dropout_rate)
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

    def forward_local_spk_logits(self, emb_seq):
        """
        The final layer that outputs local speaker logits

        Args:
            emb_seq (torch.Tensor): Tensor containing hidden states from the encoder
                Shape: (batch_size, n_frames, emb_dim)

        Returns:
            local_logits (torch.Tensor): Tensor containing local speaker logits computed using
                Shape: (batch_size, n_frames, n_spk)
        """
        emb_seq_ = self.dropout(F.relu(emb_seq))
        emb_seq_ = self.first_hidden_to_hidden(emb_seq_)
        emb_seq_ = self.dropout(F.relu(emb_seq_))
        local_logits = self.single_hidden_to_spks(emb_seq_)
        return local_logits

    def get_init_queries(self, preds, emb_seq):
        """
        Get initial queries as a weighted average of encoder embeddings.
        Predictions with values lower than 0.5 are excluded.
        Args:
            preds (torch.Tensor): Tensor containing speaker predictions (weights).
                Shape: (batch_size, n_frames, n_spk)
            emb_seq (torch.Tensor): Tensor containing hidden states from the encoder.
                Shape: (batch_size, n_frames, emb_dim)
        Returns:
            init_queries (torch.Tensor): Tensor containing initial speaker queries.
                Shape: (batch_size, n_spk, emb_dim)
        """
        scores = self._get_log_pred_scores(preds)
        is_speech = preds > 0.5
        scores = torch.where(is_speech, scores, torch.tensor(float('-inf')))
        weights = torch.where(scores > 0, preds, torch.tensor(0.0, device=preds.device))

        # Fallback for speakers with no positive scores
        no_pos_scores_mask = (weights.sum(dim=1) == 0) & (preds.max(dim=1)[0] > 0.5)  # Shape: (B, N_spk)
        if no_pos_scores_mask.any():
            logging.info(f"Fallback! no_pos_scores_mask: {no_pos_scores_mask}")
            fallback_weights = torch.where(preds > 0.5, preds, torch.tensor(0.0, device=preds.device))
            expanded_mask = no_pos_scores_mask.unsqueeze(1).expand_as(weights)
            weights = torch.where(expanded_mask, fallback_weights, weights)

        init_queries_sum = torch.matmul(weights.transpose(1, 2), emb_seq)
        sum_weights = weights.sum(dim=1)
        logging.info(f"sum_weights: {sum_weights}")
        init_queries = init_queries_sum / (sum_weights.unsqueeze(-1) + 1e-8)
        return init_queries

    def _get_log_pred_scores(self, preds):
        """
        Get per-frame scores for speakers based on their activity probabilities.
        Scores are log-based and designed to be high for confident prediction of non-overlapped speech.

        Args:
            preds (torch.Tensor): Tensor containing speaker activity probabilities
                Shape: (batch_size, n_frames, n_spk)

        Returns:
            scores (torch.Tensor): Tensor containing speaker scores
                Shape: (batch_size, n_frames, n_spk)
        """
        log_probs = torch.log(torch.clamp(preds, min=self.pred_score_threshold))
        log_1_probs = torch.log(torch.clamp(1.0 - preds, min=self.pred_score_threshold))
        log_1_probs_sum = log_1_probs.sum(dim=2).unsqueeze(-1).expand(-1, -1, self.local_num_spks)
        scores = log_probs - log_1_probs + log_1_probs_sum - math.log(0.5)
        return scores

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

