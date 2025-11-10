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
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.utils import logging
from nemo.collections.asr.modules.transformer.transformer_modules import PositionWiseFF
from nemo.collections.asr.modules.transformer.perceiver_encoders import SimplePerceiverBlock
from nemo.collections.asr.parts.submodules.multi_head_attention import PositionalEncoding

__all__ = ['NextformerModules', 'MaskedQueryDecoder']

@dataclass
class StreamingNextformerState:
    """
    This class creates a class instance that will be used to store the state of the
    streaming Nextformer model.

    Attributes:
        global_emb_set (torch.Tensor): global speaker embeddings set to store embeddings from start.
            Shape: (B, max_num_spks, global_emb_set_size, emb_dim)
        global_emb_set_lengths (torch.Tensor): number of embeddings stored in the global speaker embeddings set for each speaker
            Shape: (B, max_num_spks)
        global_spk_centroids (torch.Tensor): global speaker centroid embeddings for each speaker
            Shape: (B, max_num_spks, emb_dim)
    """
    global_emb_set = None
    global_emb_set_lengths = None
    global_spk_centroids = None

class MaskedQueryDecoder(nn.Module):
    def __init__(
        self,
        num_queries: int,
        num_layers: int,
        hidden_size: int,
        inner_size: int,
        num_cross_attention_heads: int = 4,
        num_self_attention_heads: int = 4,
        cross_attn_dropout: float = 0.0,
        self_attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        pre_ln: bool = True,
        pre_ln_final_layer_norm: bool = True,
        use_query_pos_emb: bool = False,
        use_encoder_pos_emb: bool = False,
        encoder_pos_emb_max_len: int = 5000,
    ):
        super().__init__()

        if pre_ln and pre_ln_final_layer_norm:
            self.final_layer_norm = torch.nn.LayerNorm(hidden_size, eps=1e-5)
        else:
            self.final_layer_norm = None

        self.num_queries = num_queries
        self.num_layers = num_layers
        self.use_query_pos_emb = use_query_pos_emb
        self.use_encoder_pos_emb = use_encoder_pos_emb

        # Absolute positional encoding for input features
        if self.use_encoder_pos_emb:
            self.pos_enc_encoder = PositionalEncoding(
                d_model=hidden_size,
                dropout_rate=0.0,
                max_len=encoder_pos_emb_max_len,
            )
        else:
            self.pos_enc_encoder = None

        # learnable positional embeddings for queries
        if self.use_query_pos_emb:
            self.query_pos_emb = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(torch.empty(num_queries, hidden_size))
            )
        else:
            self.query_pos_emb = None

        layer = SimplePerceiverBlock(
            hidden_size=hidden_size,
            inner_size=inner_size,
            num_cross_attention_heads=num_cross_attention_heads,
            num_self_attention_heads=num_self_attention_heads,
            cross_attn_dropout=cross_attn_dropout,
            self_attn_dropout=self_attn_dropout,
            ffn_dropout=ffn_dropout,
            hidden_act=hidden_act,
            pre_ln=pre_ln,
        )
        self.layers = torch.nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_layers)])

        self.mask_head = PositionWiseFF(hidden_size=hidden_size, inner_size=inner_size, ffn_dropout=ffn_dropout)

    def forward(self, encoder_states, encoder_len_mask, encoder_mask=None, query_states=None, query_mask=None):
        """
        Args:
            encoder_states (torch.Tensor): outputs of the encoder
                Shape: (B, n_frames, hidden_size)
            encoder_len_mask (torch.Tensor): lengths-based mask of encoder states for cross-attention, True means masking out
                Shape: (B, n_frames)
            encoder_mask (torch.Tensor): encoder inputs mask for cross-attention, True means masking out
                Shape: (B, num_queries, n_frames)
            query_states (torch.Tensor): optional initial query states
                Shape: (B, num_queries, hidden_size)    
            query_mask (torch.Tensor): optional query mask for self-attention, True means masking out
                Shape: (B, num_queries, num_queries)
        Returns:
            query_states (torch.Tensor): final query states
                Shape: (B, num_queries, hidden_size)
            intermediate_logits (list of torch.Tensor): list of intermediate mask_logits for all num_layers layers
                Shape: [(B, n_frames, num_queries)]
        """
        encoder_pos_emb = None
        if self.use_encoder_pos_emb:
            self.pos_enc_encoder.extend_pe(encoder_states.size(1), encoder_states.device, encoder_states.dtype)
            encoder_pos_emb = self.pos_enc_encoder.pe[:, : encoder_states.size(1)]

        # Get positional embedding for queries if enabled
        query_pos_emb = None
        if self.use_query_pos_emb:
            query_pos_emb = self.query_pos_emb.unsqueeze(0).expand(encoder_states.shape[0], -1, -1)
        
        # initialize query states
        if query_states is None:
            if self.use_query_pos_emb:
                query_states = query_pos_emb
            else:
                query_states = torch.zeros((encoder_states.shape[0], self.num_queries, encoder_states.shape[2]), device=encoder_states.device, dtype=encoder_states.dtype)
        
        encoder_len_mask_expand = encoder_len_mask.unsqueeze(-1).expand(-1, -1, self.num_queries)
        #logging.info(f"encoder_len_mask: {encoder_len_mask.to(int).sum(dim=1)}")

        if encoder_mask is None:
            encoder_mask = encoder_len_mask_expand.transpose(1, 2)

        for i, layer in enumerate(self.layers):
            query_states = layer(
                latent_states=query_states,
                latent_mask=query_mask,
                encoder_states=encoder_states,
                encoder_mask=encoder_mask,
                latent_pos_emb=query_pos_emb,
                encoder_pos_emb=encoder_pos_emb,
            )

        if self.final_layer_norm is not None:
            query_states = self.final_layer_norm(query_states)

        return query_states

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
        max_num_spks: int = 4,
        chunk_len: int = 125,
        chunk_left_context: int = 62,
        chunk_right_context: int = 1,
        causal_attn_rate: float = 0,
        causal_attn_rc: int = 7,
        pred_score_threshold: float = 0.25,
        global_emb_set_size: int = 100,
        matching_threshold: float = 0.5,
    ):
        super().__init__()
        # General params
        self.subsampling_factor = subsampling_factor
        self.fc_d_model = fc_d_model
        self.tf_d_model = tf_d_model
        self.local_num_spks = local_num_spks
        self.max_num_spks = max_num_spks
        self.pred_score_threshold = pred_score_threshold
        self.matching_threshold = matching_threshold

        self.encoder_proj = nn.Linear(self.fc_d_model, self.tf_d_model)
        self.first_hidden_to_hidden = nn.Linear(self.tf_d_model, self.tf_d_model)
        self.single_hidden_to_spks = nn.Linear(self.tf_d_model, self.local_num_spks)
        self.dropout = nn.Dropout(ff_dropout_rate)
        self.log = False

        # Streaming-related params
        self.chunk_len = chunk_len
        self.chunk_left_context = chunk_left_context
        self.chunk_right_context = chunk_right_context
        self.causal_attn_rate = causal_attn_rate
        self.causal_attn_rc = causal_attn_rc

        self.global_emb_set_size = global_emb_set_size

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

    def streaming_feat_loader(
        self, feat_seq, feat_seq_length, feat_seq_offset
    ) -> Tuple[int, torch.Tensor, torch.Tensor, int, int]:
        """
        Load a chunk of feature sequence for streaming inference.

        Args:
            feat_seq (torch.Tensor): Tensor containing feature sequence
                Shape: (batch_size, feat_dim, feat frame count)
            feat_seq_length (torch.Tensor): Tensor containing feature sequence lengths
                Shape: (batch_size,)
            feat_seq_offset (torch.Tensor): Tensor containing feature sequence offsets
                Shape: (batch_size,)

        Returns:
            chunk_idx (int): Index of the current chunk
            chunk_feat_seq (torch.Tensor): Tensor containing the chunk of feature sequence
                Shape: (batch_size, diar frame count, feat_dim)
            feat_lengths (torch.Tensor): Tensor containing lengths of the chunk of feature sequence
                Shape: (batch_size,)
        """
        feat_len = feat_seq.shape[2]
        num_chunks = math.ceil(feat_len / (self.chunk_len * self.subsampling_factor))
        if self.log:
            logging.info(
                f"feat_len={feat_len}, num_chunks={num_chunks}, "
                f"feat_seq_length={feat_seq_length}, feat_seq_offset={feat_seq_offset}"
            )

        stt_feat, end_feat, chunk_idx = 0, 0, 0
        while end_feat < feat_len:
            left_offset = min(self.chunk_left_context * self.subsampling_factor, stt_feat)
            end_feat = min(stt_feat + self.chunk_len * self.subsampling_factor, feat_len)
            right_offset = min(self.chunk_right_context * self.subsampling_factor, feat_len - end_feat)
            chunk_feat_seq = feat_seq[:, :, stt_feat - left_offset : end_feat + right_offset]
            feat_lengths = (feat_seq_length + feat_seq_offset - stt_feat + left_offset).clamp(
                0, chunk_feat_seq.shape[2]
            )
            feat_lengths = feat_lengths * (feat_seq_offset < end_feat)
            stt_feat = end_feat
            chunk_feat_seq_t = torch.transpose(chunk_feat_seq, 1, 2)
            if self.log:
                logging.info(
                    f"chunk_idx: {chunk_idx}, "
                    f"chunk_feat_seq_t shape: {chunk_feat_seq_t.shape}, "
                    f"chunk_feat_lengths: {feat_lengths}"
                )
            yield chunk_idx, chunk_feat_seq_t, feat_lengths, left_offset, right_offset
            chunk_idx += 1

    def forward_spk_logits(self, emb_seq):
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
        logits = self.single_hidden_to_spks(emb_seq_)
        return logits

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
            #logging.info(f"Fallback! no_pos_scores_mask: {no_pos_scores_mask}")
            fallback_weights = torch.where(preds > 0.5, preds, torch.tensor(0.0, device=preds.device))
            expanded_mask = no_pos_scores_mask.unsqueeze(1).expand_as(weights)
            weights = torch.where(expanded_mask, fallback_weights, weights)

        init_queries_sum = torch.matmul(weights.transpose(1, 2), emb_seq)
        sum_weights = weights.sum(dim=1)
        #logging.info(f"sum_weights: {sum_weights}")
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

    def init_streaming_state(self, batch_size: int = 1, device: torch.device = None):
        """
        Initializes StreamingNextformerState with empty tensors or zero-valued tensors.

        Args:
            batch_size (int): Batch size for tensors in streaming state
            device (torch.device): Device for tensors in streaming state

        Returns:
            streaming_state (StreamingNextformerState): initialized streaming state
        """
        streaming_state = StreamingNextformerState()
        streaming_state.global_emb_set = torch.zeros((batch_size, self.max_num_spks, self.global_emb_set_size, self.fc_d_model), device=device)
        streaming_state.global_emb_set_lengths = torch.zeros((batch_size, self.max_num_spks), dtype=torch.long, device=device)
        streaming_state.global_spk_centroids = torch.zeros((batch_size, self.max_num_spks, self.fc_d_model), device=device)
        return streaming_state

    def get_global_indices(self, spk_queries, global_spk_centroids):
        """
        Get global indices for speaker queries. This function performs speaker matching between
        local speaker queries from the current chunk and global speaker centroids maintained
        throughout the stream. It uses cosine similarity and the Hungarian algorithm for matching.

        Args:
            spk_queries (torch.Tensor): Speaker queries for the current chunk.
                Shape: (B, local_num_spks, emb_dim)
            global_spk_centroids (torch.Tensor): Global speaker centroids.
                Shape: (B, max_num_spks, emb_dim)

        Returns:
            global_spk_indices (torch.Tensor): Tensor with global speaker indices for each local query.
                Shape: (B, local_num_spks)
        """
        batch_size, local_num_spks, _ = spk_queries.shape
        device = spk_queries.device
        global_spk_indices = torch.full((batch_size, local_num_spks), -1, dtype=torch.long, device=device)

        # Calculate norms for the entire batch to find zero-vectors
        local_norms = spk_queries.norm(dim=2)
        global_norms = global_spk_centroids.norm(dim=2)

        for b in range(batch_size):
            # Filter out zero-norm queries (undetected speakers)
            is_nonzero_local = local_norms[b] > 1e-8
            nonzero_local_indices = torch.where(is_nonzero_local)[0]
            if len(nonzero_local_indices) == 0:
                continue
            nonzero_local_queries = spk_queries[b][nonzero_local_indices]

            # Filter out zero-norm global centroids (inactive global speakers)
            is_nonzero_global = global_norms[b] > 1e-8
            nonzero_global_indices = torch.where(is_nonzero_global)[0]

            logging.info(f"b={b}, nonzero_local_indices: {nonzero_local_indices}, nonzero_global_indices: {nonzero_global_indices}")

            if len(nonzero_global_indices) > 0:
                nonzero_global_centroids = global_spk_centroids[b][nonzero_global_indices]

                # Compute cosine similarity matrix
                sim_matrix = F.cosine_similarity(
                    nonzero_local_queries.unsqueeze(1), nonzero_global_centroids.unsqueeze(0), dim=2
                )

                logging.info(f"b={b}, sim_matrix: {sim_matrix}")

                # Use Hungarian algorithm to find optimal assignment
                cost_matrix = 1 - sim_matrix
                row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
                hungarian_map = {r: (c, sim_matrix[r, c].item()) for r, c in zip(row_ind, col_ind)}

                logging.info(f"b={b}, hungarian_map: {hungarian_map}")

                # Get available slots BEFORE making assignments
                all_global_indices = torch.arange(self.max_num_spks, device=device)
                currently_used_mask = torch.zeros(self.max_num_spks, dtype=torch.bool, device=device)
                if len(nonzero_global_indices) > 0:
                    currently_used_mask[nonzero_global_indices] = True
                available_global_indices = all_global_indices[~currently_used_mask]

                # Iterate through all local speakers and decide their assignment
                for r in range(len(nonzero_local_indices)):
                    local_idx = nonzero_local_indices[r]
                    if r in hungarian_map:
                        c, sim = hungarian_map[r]
                        global_idx = nonzero_global_indices[c]
                        if sim > self.matching_threshold:
                            # Strong match
                            global_spk_indices[b, local_idx] = global_idx
                        else:
                            # Weak match, prefer new slot
                            if len(available_global_indices) > 0:
                                global_spk_indices[b, local_idx] = available_global_indices[0]
                                available_global_indices = available_global_indices[1:]
                            else:
                                global_spk_indices[b, local_idx] = global_idx # Fallback
                    else:
                        # No match, assign to new slot
                        if len(available_global_indices) > 0:
                            global_spk_indices[b, local_idx] = available_global_indices[0]
                            available_global_indices = available_global_indices[1:]
            else:
                # This is the first chunk with speakers, assign all to new slots
                num_to_assign = min(len(nonzero_local_indices), self.max_num_spks)
                for i in range(num_to_assign):
                    global_spk_indices[b, nonzero_local_indices[i]] = i

            logging.info(f"b={b}, global_spk_indices: {global_spk_indices[b]}")

        return global_spk_indices

    def update_streaming_state(self, streaming_state, spk_queries, global_spk_indices):
        """
        Update the streaming state with new speaker queries based on their global indices.
        This function updates the global embedding set and recalculates speaker centroids.

        Args:
            streaming_state (StreamingNextformerState): The current streaming state.
            spk_queries (torch.Tensor): Speaker queries from the current chunk.
                Shape: (B, local_num_spks, emb_dim)
            global_spk_indices (torch.Tensor): Global speaker indices for each local query.
                Shape: (B, local_num_spks)

        Returns:
            streaming_state (StreamingNextformerState): The updated streaming state.
        """
        batch_size, local_num_spks, _ = spk_queries.shape
        for b in range(batch_size):
            for i in range(local_num_spks):
                global_index = global_spk_indices[b, i]
                if global_index != -1:
                    # Update global_emb_set (circular buffer)
                    count = streaming_state.global_emb_set_lengths[b, global_index]
                    insert_idx = count % self.global_emb_set_size
                    streaming_state.global_emb_set[b, global_index, insert_idx] = spk_queries[b, i]
                    streaming_state.global_emb_set_lengths[b, global_index] += 1

                    # Update global_spk_centroids
                    num_valid_embs = min(count + 1, self.global_emb_set_size)
                    valid_embs = streaming_state.global_emb_set[b, global_index, :num_valid_embs]
                    streaming_state.global_spk_centroids[b, global_index] = valid_embs.mean(dim=0)

        logging.info(f"updated streaming_state.global_emb_set_lengths: {streaming_state.global_emb_set_lengths}")
        return streaming_state

