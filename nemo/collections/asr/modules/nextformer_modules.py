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
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.utils import logging
from nemo.collections.asr.modules.transformer.transformer_modules import PositionWiseFF
from nemo.collections.asr.parts.submodules.multi_head_attention import PositionalEncoding, MultiHeadAttention

__all__ = ['NextformerModules', 'MaskedQueryDecoder']

@dataclass
class StreamingNextformerState:
    """
    This class creates a class instance that will be used to store the state of the
    streaming Nextformer model.

    Attributes:
        past_spk_queries (torch.Tensor): Past speaker queries.
            Shape: (B, num_past_spk_queries, emb_dim)
        past_spk_assignments (torch.Tensor): Past speaker assignments.
            Shape: (B, num_past_spk_queries, max_num_spks)
    """
    past_spk_queries = None
    past_spk_assignments = None

class MaskedQueryDecoderBlock(torch.nn.Module):
    """
    Building block of a Masked Query Decoder.
    This block is similar to TransformerDecoderBlock but with a different order of operations.
    It consists of one or two cross-attention layers, a self-attention layer, and a feed-forward network.
    This is used to build a MaskedQueryDecoder.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        inner_size: number of neurons in the intermediate part of feed-forward
            net, usually is (4-8 x hidden_size) in the papers
        extra_cross_attention: bool = False, whether to add an extra cross-attention layer after the first cross-attention layer
        num_cross_attention_heads: number of heads in multi-head attention for cross-attention
        num_self_attention_heads: number of heads in multi-head attention for self-attention
        cross_attn_dropout: probability of dropout applied to attention scores for cross-attention
        self_attn_dropout: probability of dropout applied to attention scores for self-attention
        ffn_dropout: probability of dropout applied to FFN output
        hidden_act: activation function used between two linear layers in FFN
    """

    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        extra_cross_attention: bool = False,
        num_cross_attention_heads: int = 4,
        num_self_attention_heads: int = 4,
        cross_attn_dropout: float = 0.0,
        self_attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        pre_ln: bool = False,
    ):
        super().__init__()
        self.pre_ln = pre_ln
        self.extra_cross_attention = extra_cross_attention
        # Cross-attention layer
        self.layer_norm_1 = torch.nn.LayerNorm(hidden_size, eps=1e-5)
        self.first_sub_layer = MultiHeadAttention(
            n_head=num_cross_attention_heads,
            n_feat=hidden_size,
            dropout_rate=cross_attn_dropout,
        )
        if extra_cross_attention:
            self.layer_norm_extra = torch.nn.LayerNorm(hidden_size, eps=1e-5)
            self.extra_sub_layer = MultiHeadAttention(
                n_head=num_cross_attention_heads,
                n_feat=hidden_size,
                dropout_rate=cross_attn_dropout,
            )
        # Self-attention layer
        self.layer_norm_2 = torch.nn.LayerNorm(hidden_size, eps=1e-5)
        self.second_sub_layer = MultiHeadAttention(
            n_head=num_self_attention_heads,
            n_feat=hidden_size,
            dropout_rate=self_attn_dropout,
        )
        # Feed-forward layer
        self.layer_norm_3 = torch.nn.LayerNorm(hidden_size, eps=1e-5)
        self.third_sub_layer = PositionWiseFF(hidden_size, inner_size, ffn_dropout, hidden_act)


    def forward_preln(
        self, latent_states, latent_mask, encoder_states, encoder_mask, encoder_mask_extra=None, latent_pos_emb=None, encoder_pos_emb=None
    ):
        """
        Pre-LayerNorm block
        Order of operations: LN -> Cross-Attn -> Residual -> (LN -> Cross-Attn -> Residual) -> LN -> Self-Attn -> Residual -> LN -> FFN
        """
        residual = latent_states
        latent_states = self.layer_norm_1(latent_states)
        
        # Add positional embedding to query for cross-attention
        cross_query = latent_states
        if latent_pos_emb is not None:
            cross_query = latent_states + latent_pos_emb

        if encoder_pos_emb is not None:
            encoder_key = encoder_states + encoder_pos_emb
        else:
            encoder_key = encoder_states
        cross_attn_output = self.first_sub_layer(
            query=cross_query, key=encoder_key, value=encoder_states, mask=encoder_mask
        )
        cross_attn_output += residual

        # Extra cross-attention layer (if enabled)
        if self.extra_cross_attention:
            residual = cross_attn_output
            cross_attn_output = self.layer_norm_extra(cross_attn_output)
            
            # Add positional embedding to query for extra cross-attention
            extra_cross_query = cross_attn_output
            if latent_pos_emb is not None:
                extra_cross_query = cross_attn_output + latent_pos_emb
            
            extra_cross_attn_output = self.extra_sub_layer(
                query=extra_cross_query, key=encoder_key, value=encoder_states, mask=encoder_mask_extra
            )
            cross_attn_output = extra_cross_attn_output + residual

        residual = cross_attn_output
        cross_attn_output = self.layer_norm_2(cross_attn_output)
        
        # Add positional embedding to query and key for self-attention
        self_query = cross_attn_output
        self_key = cross_attn_output
        if latent_pos_emb is not None:
            self_query = cross_attn_output + latent_pos_emb
            self_key = cross_attn_output + latent_pos_emb
            
        self_attn_output = self.second_sub_layer(query=self_query, key=self_key, value=cross_attn_output, mask=latent_mask)
        self_attn_output += residual

        residual = self_attn_output
        self_attn_output = self.layer_norm_3(self_attn_output)
        output_states = self.third_sub_layer(self_attn_output)
        output_states += residual

        return output_states

    def forward_postln(
        self, latent_states, latent_mask, encoder_states, encoder_mask, encoder_mask_extra=None, latent_pos_emb=None, encoder_pos_emb=None
    ):
        """
        Post-LayerNorm block
        Order of operations: Cross-Attn -> Residual -> LN -> (Cross-Attn -> Residual -> LN) -> Self-Attn -> Residual -> LN -> FFN -> Residual -> LN
        """
        # Add positional embedding to query for cross-attention
        cross_query = latent_states
        #logging.info(f"latent_states_before_ca: {latent_states[0, 0:20, 0:3]}")
        if latent_pos_emb is not None:
            cross_query = latent_states + latent_pos_emb

        # Add positional embedding to key for cross-attention
        encoder_key = encoder_states
        if encoder_pos_emb is not None:
            encoder_key = encoder_states + encoder_pos_emb
            
        cross_attn_output = self.first_sub_layer(
            query=cross_query, key=encoder_key, value=encoder_states, mask=encoder_mask
        )
        #logging.info(f"latent_states_ca: {cross_attn_output[0, 0:20, 0:3]}")

        cross_attn_output += latent_states
        #logging.info(f"latent_states_ca_residual: {cross_attn_output[0, 0:20, 0:3]}")
        cross_attn_output = self.layer_norm_1(cross_attn_output)

        # Extra cross-attention layer (if enabled)
        if self.extra_cross_attention:
            # Add positional embedding to query for extra cross-attention
            extra_cross_query = cross_attn_output
            if latent_pos_emb is not None:
                extra_cross_query = cross_attn_output + latent_pos_emb
            
            extra_cross_attn_output = self.extra_sub_layer(
                query=extra_cross_query, key=encoder_key, value=encoder_states, mask=encoder_mask_extra
            )
            cross_attn_output = extra_cross_attn_output + cross_attn_output
            cross_attn_output = self.layer_norm_extra(cross_attn_output)

        # Add positional embedding to query and key for self-attention
        self_query = cross_attn_output
        self_key = cross_attn_output
        if latent_pos_emb is not None:
            self_query = cross_attn_output + latent_pos_emb
            self_key = cross_attn_output + latent_pos_emb
            
        #logging.info(f"latent_states_before_sa: {cross_attn_output[0, 0:20, 0:3]}")
        self_attn_output = self.second_sub_layer(query=self_query, key=self_key, value=cross_attn_output, mask=latent_mask)
        #logging.info(f"latent_states_sa: {self_attn_output[0, 0:20, 0:3]}")
        self_attn_output += cross_attn_output
        #logging.info(f"latent_states_sa_residual: {self_attn_output[0, 0:20, 0:3]}")
        self_attn_output = self.layer_norm_2(self_attn_output)

        output_states = self.third_sub_layer(self_attn_output)
        output_states += self_attn_output
        return self.layer_norm_3(output_states)

    def forward(self, latent_states, latent_mask, encoder_states, encoder_mask, encoder_mask_extra=None, latent_pos_emb=None, encoder_pos_emb=None):
        if self.pre_ln:
            return self.forward_preln(
                latent_states, latent_mask, encoder_states, encoder_mask, encoder_mask_extra, latent_pos_emb, encoder_pos_emb
            )
        else:
            return self.forward_postln(
                latent_states, latent_mask, encoder_states, encoder_mask, encoder_mask_extra, latent_pos_emb, encoder_pos_emb
            )

class MaskedQueryDecoder(nn.Module):
    """
    Masked query decoder for speaker diarization.
    
    This decoder uses learnable query embeddings that attend to encoder features
    to extract speaker-specific representations.
    """
    
    def __init__(
        self,
        num_queries: int,
        num_layers: int,
        hidden_size: int,
        inner_size: int,
        extra_cross_attention: bool = False,
        num_cross_attention_heads: int = 4,
        num_self_attention_heads: int = 4,
        cross_attn_dropout: float = 0.0,
        self_attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        pre_ln: bool = True,
        pre_ln_final_layer_norm: bool = True,
        use_learned_init: bool = False,
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
        self.extra_cross_attention = extra_cross_attention
        self.use_learned_init = use_learned_init
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

        # Learnable initialization for query content (separate from positional embeddings)
        if self.use_learned_init:
            self.learned_init = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(torch.empty(num_queries, hidden_size))
            )
        else:
            self.learned_init = None

        # Learnable positional embeddings for queries (used in attention, not initialization)
        if self.use_query_pos_emb:
            self.query_pos_emb = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(torch.empty(num_queries, hidden_size))
            )
        else:
            self.query_pos_emb = None

        layer = MaskedQueryDecoderBlock(
            hidden_size=hidden_size,
            inner_size=inner_size,
            num_cross_attention_heads=num_cross_attention_heads,
            extra_cross_attention=extra_cross_attention,
            num_self_attention_heads=num_self_attention_heads,
            cross_attn_dropout=cross_attn_dropout,
            self_attn_dropout=self_attn_dropout,
            ffn_dropout=ffn_dropout,
            hidden_act=hidden_act,
            pre_ln=pre_ln,
        )
        self.layers = torch.nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_layers)])

        self.mask_head = PositionWiseFF(hidden_size=hidden_size, inner_size=inner_size, ffn_dropout=ffn_dropout)

    def forward(self, encoder_states, encoder_len_mask, encoder_mask=None, encoder_mask_extra=None, query_states=None, query_mask=None):
        """
        Args:
            encoder_states (torch.Tensor): outputs of the encoder
                Shape: (B, n_frames, hidden_size)
            encoder_len_mask (torch.Tensor): lengths-based mask of encoder states for cross-attention, True means masking out
                Shape: (B, n_frames)
            encoder_mask (torch.Tensor): encoder inputs mask for cross-attention, True means masking out
                Shape: (B, num_queries, n_frames)
            encoder_mask_extra (torch.Tensor): encoder inputs mask for extra cross-attention, True means masking out
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

        # Get positional embedding for queries if enabled (used in attention layers)
        query_pos_emb = None
        if self.use_query_pos_emb:
            query_pos_emb = self.query_pos_emb.unsqueeze(0).expand(encoder_states.shape[0], -1, -1)
        
        # Initialize query states (separate from positional embeddings)
        if query_states is None:
            # Zero initialization (will be populated by cross-attention)
            query_states = torch.zeros(
                (encoder_states.shape[0], self.num_queries, encoder_states.shape[2]), 
                device=encoder_states.device, 
                dtype=encoder_states.dtype
            )
        
        # Add learned initialization/bias if enabled
        if self.use_learned_init:
            query_states = query_states + self.learned_init.unsqueeze(0)
        
        encoder_len_mask_expand = encoder_len_mask.unsqueeze(-1).expand(-1, -1, self.num_queries)
        #logging.info(f"encoder_len_mask: {encoder_len_mask.to(int).sum(dim=1)}")

        if encoder_mask is None:
            encoder_mask = encoder_len_mask_expand.transpose(1, 2)

        if encoder_mask_extra is None and self.extra_cross_attention:
            encoder_mask_extra = encoder_len_mask_expand.transpose(1, 2)

        for i, layer in enumerate(self.layers):
            query_states = layer(
                latent_states=query_states,
                latent_mask=query_mask,
                encoder_states=encoder_states,
                encoder_mask=encoder_mask,
                encoder_mask_extra=encoder_mask_extra,
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
        sq_d_model: int = 192,
        subsampling_factor: int = 8,
        local_num_spks: int = 4,
        max_num_spks: int = 4,
        chunk_len: int = 125,
        chunk_left_context: int = 62,
        chunk_right_context: int = 1,
        causal_attn_rate: float = 0,
        causal_attn_rc: int = 7,
        pred_score_threshold: float = 0.25,
        extra_left_context: int = 0,
        extra_right_context: int = 0,
        extra_silence_frames: int = 3,
        sinkhorn_n_iters: int = 20,
        sinkhorn_dustbin_init: float = 0.0,
        spk_query_min_frames: int = 0,
        score_similarity: str = "cosine",
        cosine_temperature: float = 0.01,
        hard_history_assignments: bool = False,
    ):
        super().__init__()
        # General params
        self.subsampling_factor = subsampling_factor
        self.fc_d_model = fc_d_model
        self.tf_d_model = tf_d_model
        self.sq_d_model = sq_d_model
        self.local_num_spks = local_num_spks
        self.max_num_spks = max_num_spks
        self.pred_score_threshold = pred_score_threshold

        # Sinkhorn parameters
        self.sinkhorn_n_iters = sinkhorn_n_iters
        self.sinkhorn_dustbin_val = nn.Parameter(
            torch.tensor(float(sinkhorn_dustbin_init), dtype=torch.float32)
        )
        #self.sinkhorn_dustbin_val = 100 / math.sqrt(self.sq_d_model)
        #self.sinkhorn_dustbin_val=0.5

        self.spk_query_min_frames = spk_query_min_frames
        self.score_similarity = score_similarity
        self.cosine_temperature = cosine_temperature
        self.hard_history_assignments = hard_history_assignments

        self.encoder_proj = nn.Linear(self.fc_d_model, self.tf_d_model)
        self.query_proj = nn.Linear(self.fc_d_model, self.sq_d_model)
        self.query_raw_proj = nn.Linear(self.fc_d_model, self.sq_d_model)
        self.query_combiner = nn.Linear(2 * self.sq_d_model, self.sq_d_model)
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

        # Extra encoder context params (for handling high speaker density)
        self.extra_left_context = extra_left_context
        self.extra_right_context = extra_right_context
        self.extra_silence_frames = extra_silence_frames

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
        streaming_state.past_spk_queries = torch.zeros((batch_size, 0, self.sq_d_model), device=device)
        streaming_state.past_spk_assignments = torch.zeros((batch_size, 0, self.max_num_spks), device=device)
        return streaming_state

    def update_streaming_state(self, streaming_state, spk_queries, spk_assignments, active_frames_per_query=None):
        """
        Update the streaming state (in-place) with new speaker queries based on their soft assignments.

        Args:
            streaming_state (StreamingNextformerState): The current streaming state.
            spk_queries (torch.Tensor): Speaker queries from the current chunk.
                Shape: (B, local_num_spks, emb_dim)
            spk_assignments (torch.Tensor): Soft assignments for each local query.
                Shape: (B, local_num_spks, max_num_spks)
            active_frames_per_query (torch.Tensor, optional): Number of active frames per query.
                Shape: (B, local_num_spks). Used to filter out short, unreliable queries.
        """
        # Filter out queries that don't meet the minimum frame threshold
        if active_frames_per_query is not None and self.spk_query_min_frames > 0:
            insufficient_frames = active_frames_per_query < self.spk_query_min_frames
            # Zero out queries and assignments for speakers with insufficient frames
            spk_queries = spk_queries.masked_fill(insufficient_frames.unsqueeze(-1), 0)
            spk_assignments = spk_assignments.masked_fill(insufficient_frames.unsqueeze(-1), 0)
            
        streaming_state.past_spk_queries = torch.cat([streaming_state.past_spk_queries, spk_queries], dim=1)
        #logging.info(f"spk_assignments: {spk_assignments}")
        if self.hard_history_assignments or not self.training:
            # Convert soft assignments to hard one-hot and detach from computation graph.
            # This prevents gradient flow through history and avoids probability dilution
            # from repeated soft matrix multiplications.
            spk_assignments_hard = F.one_hot(
                spk_assignments.argmax(dim=-1), 
                num_classes=self.max_num_spks
            ).to(spk_assignments.dtype).detach()
            spk_assignments = spk_assignments_hard

        streaming_state.past_spk_assignments = torch.cat(
            [streaming_state.past_spk_assignments, spk_assignments], dim=1
        )
        return streaming_state

    def get_local_to_global_assignments(
        self,
        spk_queries,
        streaming_state,
        dustbin_threshold: float = 0.5,
        zero_norm_threshold: float = 1e-8,
        zero_score: float = -10000.0,
    ):
        """
        Compute soft local-to-global assignments using partial Sinkhorn and dustbin handling.

        This function aggregates past queries by global speaker BEFORE applying Sinkhorn,
        ensuring that the column constraints apply to distinct global speakers rather than
        individual past queries. This prevents the issue where two local queries could be
        incorrectly assigned to the same global speaker through different past queries.

        Args:
            spk_queries (torch.Tensor): Local speaker queries for the current chunk.
                Shape: (B, local_num_spks, emb_dim)
            streaming_state (StreamingNextformerState): Streaming state containing past queries and assignments.
            dustbin_threshold (float): Threshold for dustbin score to allocate new speakers.
            zero_norm_threshold (float): L2 norm threshold to treat a query as a zero vector.
            zero_score (float): Score value used to mask zero-vector rows/cols.

        Returns:
            spk_assignments (torch.Tensor): Soft assignments from local queries to global speakers.
                Shape: (B, local_num_spks, max_num_spks)
        """
        past_spk_queries = streaming_state.past_spk_queries
        past_spk_assignments = streaming_state.past_spk_assignments
        batch_size, local_num_spks, _ = spk_queries.shape
        num_past = past_spk_queries.shape[1]

        # Mask zero-vector local queries
        local_zero = spk_queries.norm(dim=2) < zero_norm_threshold

        if num_past > 0:
            # Mask zero-vector past queries
            past_zero = past_spk_queries.norm(dim=2) < zero_norm_threshold

            # Compute full scores: (B, local_num_spks, num_past)
            if self.score_similarity == "cosine":
                spk_queries_norm = F.normalize(spk_queries, p=2, dim=2, eps=zero_norm_threshold)
                past_spk_queries_norm = F.normalize(past_spk_queries, p=2, dim=2, eps=zero_norm_threshold)
                full_scores = torch.bmm(spk_queries_norm, past_spk_queries_norm.transpose(1, 2)) / self.cosine_temperature
            elif self.score_similarity == "scaled_dot":
                emb_dim = spk_queries.shape[-1]
                full_scores = torch.bmm(spk_queries, past_spk_queries.transpose(1, 2)) / math.sqrt(emb_dim)
            else:
                raise ValueError(f"Invalid score similarity mode: {self.score_similarity}")

            # Mask zero-vector queries in full scores
            full_scores = full_scores.masked_fill(local_zero.unsqueeze(2), zero_score)
            full_scores = full_scores.masked_fill(past_zero.unsqueeze(1), zero_score)

            # Aggregate by global speaker using max pooling
            # For each (local_query, global_speaker) pair, take the max score
            # over all past queries belonging to that global speaker.
            # This preserves the "memory bank" concept while fixing Sinkhorn column constraints.

            # Expand for broadcasting:
            # full_scores: (B, local_num_spks, num_past) -> (B, local_num_spks, num_past, 1)
            # past_spk_assignments: (B, num_past, max_num_spks) -> (B, 1, num_past, max_num_spks)
            scores_expanded = full_scores.unsqueeze(-1)  # (B, local_num_spks, num_past, 1)
            assignments_expanded = past_spk_assignments.unsqueeze(1)  # (B, 1, num_past, max_num_spks)

            # Boolean mask: which past queries belong to each global speaker
            belongs_to_g = assignments_expanded > 0.5  # (B, 1, num_past, max_num_spks)

            # Broadcast scores and apply mask
            # Where past query j doesn't belong to global speaker g, set score to zero_score
            scores_broadcasted = scores_expanded.expand(-1, -1, -1, self.max_num_spks)
            masked_scores = torch.where(belongs_to_g, scores_broadcasted, torch.tensor(zero_score, device=spk_queries.device, dtype=spk_queries.dtype))
            # Shape: (B, local_num_spks, num_past, max_num_spks)

            # Max over past queries for each global speaker
            aggregated_scores = masked_scores.max(dim=2)[0]  # (B, local_num_spks, max_num_spks)

            # Determine which global speakers are active (have at least one past query)
            global_active = past_spk_assignments.max(dim=1)[0] > 0.5  # (B, max_num_spks)

            # Mask inactive global speakers (no past queries for them)
            aggregated_scores = aggregated_scores.masked_fill(~global_active.unsqueeze(1), zero_score)

        else:
            # No past queries - all scores are zero_score (will go to dustbin)
            aggregated_scores = torch.full(
                (batch_size, local_num_spks, self.max_num_spks),
                zero_score,
                device=spk_queries.device,
                dtype=spk_queries.dtype,
            )
            global_active = torch.zeros(
                (batch_size, self.max_num_spks),
                device=spk_queries.device,
                dtype=torch.bool,
            )

        # Mask zero local queries in aggregated scores
        aggregated_scores = aggregated_scores.masked_fill(local_zero.unsqueeze(2), zero_score)
        #logging.info(f"aggregated_scores: {aggregated_scores}")

        # Apply Sinkhorn on aggregated scores: (B, local_num_spks, max_num_spks)
        # Now column constraints correctly apply to distinct global speakers
        # Returns: (B, local_num_spks + 1, max_num_spks + 1)
        assign_aug = self.partial_sinkhorn(aggregated_scores)
        logging.info(f"assign_aug: {assign_aug[:,:,0:10]}")

        # Extract assignments and dustbin scores
        # local_to_global is now directly from Sinkhorn (no matrix multiply needed)
        local_to_global = assign_aug[:, :local_num_spks, :self.max_num_spks]
        dustbin_scores = assign_aug[:, :local_num_spks, -1]

        spk_assignments = local_to_global.clone()

        # Allocate new global slots for dustbin matches
        used_global = global_active.clone()

        for b in range(batch_size):
            available = torch.where(~used_global[b])[0].tolist()
            if not available:
                continue
            for local_idx in range(local_num_spks):
                if local_zero[b, local_idx]:
                    continue
                if dustbin_scores[b, local_idx] > dustbin_threshold and available:
                    global_idx = available.pop(0)
                    spk_assignments[b, local_idx, global_idx] += dustbin_scores[b, local_idx]

        return spk_assignments

    def get_global_logits(self, local_logits, spk_assignments, offset: int, dur: int):
        """
        Build per-chunk global logits from local logits and local-to-global assignments.

        Args:
            local_logits (torch.Tensor): Local logits tensor for the current chunk.
                Shape: (batch_size, local_frames, local_num_spks)
            spk_assignments (torch.Tensor): Soft assignments for each local query.
                Shape: (batch_size, local_num_spks, max_num_spks)
            offset (int): Start offset into the local frame axis.
            dur (int): Duration (number of frames) to update.

        Returns:
            chunk_logits (torch.Tensor): Per-chunk logits in global speaker space.
                Shape: (batch_size, dur, max_num_spks)
        """
        local_logits_slice = local_logits[:, offset : offset + dur, :]
        local_preds = torch.sigmoid(local_logits_slice)
        global_preds = torch.bmm(local_preds, spk_assignments)
        global_logits = torch.logit(global_preds, eps=1e-6)
        return global_logits

    def partial_sinkhorn(self, scores):
        """
        Perform partial Sinkhorn normalization on a log-score matrix.
        
        Args:
            scores: [Batch, M, N] log-scores
        
        Returns:
            Augmented normalized assignment matrix of shape [Batch, M+1, N+1]
        """
        B, M, N = scores.shape

        # Create augmented container: (M+1) x (N+1) in log-space, initialized to zeros
        aug_scores = scores.new_zeros((B, M + 1, N + 1))
        aug_scores[:, :M, :N] = scores
        
        # Fill dustbin row/col with trainable log-score bias (dustbin_val)
        if self.score_similarity == "cosine":
            aug_scores[:, M, :] = self.sinkhorn_dustbin_val / self.cosine_temperature
            aug_scores[:, :, N] = self.sinkhorn_dustbin_val / self.cosine_temperature
        elif self.score_similarity == "scaled_dot":
            aug_scores[:, M, :] = self.sinkhorn_dustbin_val
            aug_scores[:, :, N] = self.sinkhorn_dustbin_val
        else:
            raise ValueError(f"Invalid score similarity mode: {self.score_similarity}")
        
        # Define target log-marginals
        # Real Rows/Cols sum to 1 -> log(1) = 0
        r = aug_scores.new_zeros((B, M + 1, 1))
        c = aug_scores.new_zeros((B, 1, N + 1))
        # Dustbin Row sums to N -> log(N)
        r[:, M, 0] = math.log(N) if N > 0 else -10000.0
        # Dustbin Col sums to M -> log(M)
        c[:, 0, N] = math.log(M) if M > 0 else -10000.0

        # Sinkhorn Iterations (Direct Matrix Update in Log Space)
        P = aug_scores
        for _ in range(self.sinkhorn_n_iters):
            # Row normalization: normalize each row to match target marginal r
            P = P - torch.logsumexp(P, dim=2, keepdim=True) + r
            # Column normalization: normalize each column to match target marginal c
            P = P - torch.logsumexp(P, dim=1, keepdim=True) + c

        # Final row normalization: normalize each row to match target marginal r
        P = P - torch.logsumexp(P, dim=2, keepdim=True) + r

        # Exponentiate and return the full augmented matrix
        return P.exp()