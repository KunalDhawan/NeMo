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
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    PositionalEncoding,
    MultiHeadAttention,
    RelPositionMultiHeadAttention,
    RelPositionalEncoding,
)

__all__ = ['NextformerModules', 'MaskedQueryDecoder', 'JointTimeSpeakerEncoder', 'JointTimeSpeakerEncoderBlock']

@dataclass
class StreamingNextformerState:
    """
    This class creates a class instance that will be used to store the state of the
    streaming Nextformer model.

    Attributes:
        past_spk_embs (torch.Tensor): Past speaker embeddings.
            Shape: (B, num_past_spk_embs, emb_dim)
        past_spk_assignments (torch.Tensor): Past speaker assignments.
            Shape: (B, num_past_spk_embs, max_num_spks)
    """
    past_spk_embs = None
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
        se_d_model: int = 192,
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
        spk_emb_update_min_frames: int = 0,
        spk_centroid_update_min_frames: int = 0,
        spk_assignment_min_prob: float = 0.5,
        score_similarity: str = "cosine",
        cosine_temperature: float = 0.05,
        hard_history_assignments: bool = False,
        logsumexp_temperature: float = 1.0,
        assignment_swap_prob: float = 0.0,
        backend: str = "trff",
        fusion_type: str = "concat",
        fusion_lnorm: bool = False,
        spk_extra_proj: bool = False,
    ):
        super().__init__()
        # General params
        self.subsampling_factor = subsampling_factor
        self.fc_d_model = fc_d_model
        self.tf_d_model = tf_d_model
        self.spk_extra_proj = spk_extra_proj
        # When not using separate SE projections, force se_d_model = tf_d_model
        # to avoid dimension mismatch when reusing TF projections
        self.se_d_model = se_d_model if spk_extra_proj else tf_d_model
        self.local_num_spks = local_num_spks
        self.max_num_spks = max_num_spks
        self.pred_score_threshold = pred_score_threshold

        # Backend and fusion params
        self.backend = backend
        self.fusion_type = fusion_type
        self.fusion_lnorm = fusion_lnorm

        # Sinkhorn parameters
        self.sinkhorn_n_iters = sinkhorn_n_iters
        self.sinkhorn_dustbin_val = nn.Parameter(
            torch.tensor(float(sinkhorn_dustbin_init), dtype=torch.float32)
        )
        #self.sinkhorn_dustbin_val = 100 / math.sqrt(self.se_d_model)
        #self.sinkhorn_dustbin_val=0.5

        self.spk_emb_update_min_frames = spk_emb_update_min_frames
        self.spk_assignment_min_prob = spk_assignment_min_prob
        self.score_similarity = score_similarity
        self.cosine_temperature = cosine_temperature
        self.hard_history_assignments = hard_history_assignments
        self.logsumexp_temperature = logsumexp_temperature

        self.encoder_proj_tf = nn.Linear(self.fc_d_model, self.tf_d_model)
        self.spk_emb_proj_tf = nn.Linear(self.fc_d_model, self.tf_d_model)
        
        # Only create separate SE projections when spk_extra_proj is enabled
        if self.spk_extra_proj:
            self.encoder_proj_se = nn.Linear(self.fc_d_model, self.se_d_model)
            self.spk_emb_proj_se = nn.Linear(self.fc_d_model, self.se_d_model)
        else:
            self.encoder_proj_se = None
            self.spk_emb_proj_se = None

        self.first_hidden_to_hidden = nn.Linear(self.tf_d_model, self.tf_d_model)
        self.single_hidden_to_spks = nn.Linear(self.tf_d_model, self.local_num_spks)
        self.dropout = nn.Dropout(ff_dropout_rate)
        self.log = False

        # Initialize fusion layers for ISD/JSD backends
        self._init_fusion_layers()

        # Initialize backend output projection for ISD/JSD backends
        if self.backend in ["isd", "jsd"]:
            self.backend_output_proj = nn.Linear(self.tf_d_model, 1)
        else:
            self.backend_output_proj = None

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

    def _init_fusion_layers(self):
        """
        Initialize fusion layers for combining frame embeddings and speaker embeddings.

        Fusion types:
            - 'concat': Concatenates frame and speaker embeddings, then projects back to d_model.
            - 'film': Feature-wise Linear Modulation - learns affine transformation (gamma, beta)
                      conditioned on speaker embeddings.
            - 'hdm': Hadamard (element-wise) product - no learnable parameters required.

        Note:
            Only used by ISD and JSD backends. For other backends, all fusion attributes
            are set to None.
        """
        # Initialize all fusion attributes to None by default
        self.fusion_input_proj = None
        self.fusion_film_gamma = None
        self.fusion_film_beta = None
        self.fusion_layer_norm = None

        if self.backend not in ("isd", "jsd"):
            return

        d_model = self.tf_d_model

        if self.fusion_type == "concat":
            # Concatenate embeddings [frame; speaker] and project back to d_model
            self.fusion_input_proj = nn.Linear(2 * d_model, d_model)
        elif self.fusion_type == "film":
            # FiLM: output = gamma(speaker) * frame + beta(speaker)
            self.fusion_film_gamma = nn.Linear(d_model, d_model)
            self.fusion_film_beta = nn.Linear(d_model, d_model)
        elif self.fusion_type == "hdm":
            # Hadamard: element-wise product, no additional parameters needed
            pass
        else:
            raise ValueError(
                f"Invalid fusion_type '{self.fusion_type}'. Expected one of: 'concat', 'film', 'hdm'."
            )

        if self.fusion_lnorm:
            self.fusion_layer_norm = nn.LayerNorm(d_model)

    def apply_fusion(
        self,
        emb_seq_expanded: torch.Tensor,
        spk_embs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply fusion between frame embeddings and speaker embeddings.
        Used by ISD and JSD backends.
        
        Args:
            emb_seq_expanded: Frame embeddings expanded for speakers.
                Shape: (B, S, T, tf_d_model)
            spk_embs: Speaker embeddings (not expanded).
                Shape: (B, S, tf_d_model)
        
        Returns:
            combined: Fused embeddings.
                Shape: (B, S, T, tf_d_model)
        """
        seq_len = emb_seq_expanded.shape[2]
        
        if self.fusion_type == "concat":
            # Expand speaker embeddings to match time dimension
            spk_embs_expanded = spk_embs.unsqueeze(2).expand(-1, -1, seq_len, -1)
            combined = torch.cat([emb_seq_expanded, spk_embs_expanded], dim=-1)
            combined = self.fusion_input_proj(combined)
        
        elif self.fusion_type == "film":
            # FiLM: Feature-wise Linear Modulation
            # Speaker embedding generates scale (gamma) and shift (beta) to modulate frame embeddings
            gamma = self.fusion_film_gamma(spk_embs)  # (B, S, tf_d_model)
            beta = self.fusion_film_beta(spk_embs)    # (B, S, tf_d_model)
            # Expand for time dimension: (B, S, 1, tf_d_model)
            gamma = gamma.unsqueeze(2)
            beta = beta.unsqueeze(2)
            # Modulate frame embeddings: gamma * x + beta
            combined = gamma * emb_seq_expanded + beta
        
        elif self.fusion_type == "hdm":
            # Hadamard (element-wise) product after projecting both to same space
            spk_embs_expanded = spk_embs.unsqueeze(2).expand(-1, -1, seq_len, -1)
            combined = emb_seq_expanded * spk_embs_expanded

        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        if self.fusion_lnorm and self.fusion_layer_norm is not None:
            combined = self.fusion_layer_norm(combined)

        return combined

    def project_encoder_for_se(self, emb_seq: torch.Tensor) -> torch.Tensor:
        """
        Project encoder embeddings for speaker embedding matching.
        Uses separate SE projection if spk_extra_proj is enabled, otherwise reuses TF projection.
        
        Args:
            emb_seq: Encoder embeddings of shape (B, T, fc_d_model)
        
        Returns:
            Projected embeddings of shape (B, T, se_d_model) or (B, T, tf_d_model)
        """
        if self.spk_extra_proj:
            return self.encoder_proj_se(emb_seq)
        else:
            return self.encoder_proj_tf(emb_seq)

    def project_spk_embs_for_se(self, spk_embs: torch.Tensor) -> torch.Tensor:
        """
        Project speaker embeddings for speaker embedding matching.
        Uses separate SE projection if spk_extra_proj is enabled, otherwise reuses TF projection.
        
        Args:
            spk_embs: Speaker embeddings of shape (B, num_spks, fc_d_model)
        
        Returns:
            Projected embeddings of shape (B, num_spks, se_d_model) or (B, num_spks, tf_d_model)
        """
        if self.spk_extra_proj:
            return self.spk_emb_proj_se(spk_embs)
        else:
            return self.spk_emb_proj_tf(spk_embs)

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
        streaming_state.past_spk_embs = torch.zeros((batch_size, 0, self.se_d_model), device=device)
        streaming_state.past_spk_assignments = torch.zeros((batch_size, 0, self.max_num_spks), device=device)
        return streaming_state

    def update_streaming_state(self, streaming_state, spk_embs, spk_assignments, active_frames_per_spk=None):
        """
        Update the streaming state (in-place) with new speaker embeddings based on their soft assignments.

        Args:
            streaming_state (StreamingNextformerState): The current streaming state.
            spk_embs (torch.Tensor): Speaker embeddings from the current chunk.
                Shape: (B, local_num_spks, emb_dim)
            spk_assignments (torch.Tensor): Soft assignments for each local spk embedding.
                Shape: (B, local_num_spks, max_num_spks)
            active_frames_per_spk (torch.Tensor, optional): Number of active frames per speaker.
                Shape: (B, local_num_spks). Used to filter out short, unreliable spk embeddings.
        """
        # Filter out spk embeddings that don't meet the minimum frame threshold
        if active_frames_per_spk is not None and self.spk_emb_update_min_frames > 0:
            insufficient_frames = active_frames_per_spk < self.spk_emb_update_min_frames
            # Zero out embeddings and assignments for speakers with insufficient frames
            spk_embs = spk_embs.masked_fill(insufficient_frames.unsqueeze(-1), 0)
            spk_assignments = spk_assignments.masked_fill(insufficient_frames.unsqueeze(-1), 0)

        # Filter out spk embeddings with low assignment probability
        if self.spk_assignment_min_prob > 0:
            max_assignment_prob = spk_assignments.max(dim=-1)[0]  # (B, local_num_spks)
            low_prob = max_assignment_prob < self.spk_assignment_min_prob
            # Zero out spk embeddings and assignments for low-probability speakers
            spk_embs = spk_embs.masked_fill(low_prob.unsqueeze(-1), 0)
            spk_assignments = spk_assignments.masked_fill(low_prob.unsqueeze(-1), 0)

        streaming_state.past_spk_embs = torch.cat([streaming_state.past_spk_embs, spk_embs], dim=1)
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
        spk_embs,
        streaming_state,
        dustbin_threshold: float = 0.5,
        zero_norm_threshold: float = 1e-8,
        zero_score: float = -10000.0,
    ):
        """
        Compute soft local-to-global assignments using partial Sinkhorn and dustbin handling.

        This function aggregates past local spk embedding by global speaker BEFORE applying Sinkhorn,
        ensuring that the column constraints apply to distinct global speakers rather than
        individual past local spk embeddings. This prevents the issue where two local spk embeddings could be
        incorrectly assigned to the same global speaker through different past local spk embeddings.

        Args:
            spk_embs (torch.Tensor): Local speaker embeddings for the current chunk.
                Shape: (B, local_num_spks, emb_dim)
            streaming_state (StreamingNextformerState): Streaming state containing past spk embeddings and assignments.
            dustbin_threshold (float): Threshold for dustbin score to allocate new speakers.
            zero_norm_threshold (float): L2 norm threshold to treat a local spk embedding as a zero vector.
            zero_score (float): Score value used to mask zero-vector rows/cols.

        Returns:
            spk_assignments (torch.Tensor): Soft assignments from local spk embeddings to global speakers.
                Shape: (B, local_num_spks, max_num_spks)
        """
        past_spk_embs = streaming_state.past_spk_embs
        past_spk_assignments = streaming_state.past_spk_assignments
        batch_size, local_num_spks, _ = spk_embs.shape
        num_past = past_spk_embs.shape[1]

        # Mask zero-vector local spk embeddings
        local_zero = spk_embs.norm(dim=2) < zero_norm_threshold

        if num_past > 0:
            # Mask zero-vector past spk embeddings
            past_zero = past_spk_embs.norm(dim=2) < zero_norm_threshold

            # Compute full scores: (B, local_num_spks, num_past)
            if self.score_similarity == "cosine":
                spk_embs_norm = F.normalize(spk_embs, p=2, dim=2, eps=zero_norm_threshold)
                past_spk_embs_norm = F.normalize(past_spk_embs, p=2, dim=2, eps=zero_norm_threshold)
                full_scores = torch.bmm(spk_embs_norm, past_spk_embs_norm.transpose(1, 2)) / self.cosine_temperature
            elif self.score_similarity == "scaled_dot":
                emb_dim = spk_embs.shape[-1]
                full_scores = torch.bmm(spk_embs, past_spk_embs.transpose(1, 2)) / math.sqrt(emb_dim)
            else:
                raise ValueError(f"Invalid score similarity mode: {self.score_similarity}")

            # Mask zero-vector spk embeddings in full scores
            full_scores = full_scores.masked_fill(local_zero.unsqueeze(2), zero_score)
            full_scores = full_scores.masked_fill(past_zero.unsqueeze(1), zero_score)

            # Aggregate by global speaker using LogSumExp (smooth differentiable approximation to max)
            # For each (local_spk_emb, global_speaker) pair, aggregate scores over all past spk embeddings
            # belonging to that global speaker. LogSumExp provides dense gradients for training
            # while behaving like max when temperature is low.

            # Expand for broadcasting:
            # full_scores: (B, local_num_spks, num_past) -> (B, local_num_spks, num_past, 1)
            # past_spk_assignments: (B, num_past, max_num_spks) -> (B, 1, num_past, max_num_spks)
            scores_expanded = full_scores.unsqueeze(-1)  # (B, local_num_spks, num_past, 1)
            assignments_expanded = past_spk_assignments.unsqueeze(1)  # (B, 1, num_past, max_num_spks)

            # Boolean mask: which past spk embeddings belong to each global speaker
            belongs_to_g = assignments_expanded > 0.5  # (B, 1, num_past, max_num_spks)

            # Broadcast scores and apply mask
            # Where past spk embedding j doesn't belong to global speaker g, set score to zero_score for LogSumExp
            scores_broadcasted = scores_expanded.expand(-1, -1, -1, self.max_num_spks)
            belongs_to_g_expanded = belongs_to_g.expand(-1, local_num_spks, -1, -1)
            masked_scores = torch.where(
                belongs_to_g_expanded,
                scores_broadcasted,
                zero_score  # -10000.0 effectively acts like -inf in LogSumExp
            )
            # Shape: (B, local_num_spks, num_past, max_num_spks)

            # Count valid elements per (local_spk_emb, global_speaker) pair for normalization
            num_valid = belongs_to_g_expanded.sum(dim=2).clamp(min=1).float()
            # Shape: (B, local_num_spks, max_num_spks)

            # LogMeanExp over past spk embeddings for each global speaker (generalized mean in log-space)
            # This is count-normalized to avoid bias towards speakers with more history
            # temperature controls smoothness: low = sharp (like max), high = smooth (like mean)
            aggregated_scores = self.logsumexp_temperature * (
                torch.logsumexp(masked_scores / self.logsumexp_temperature, dim=2)
                - torch.log(num_valid)
            )  # (B, local_num_spks, max_num_spks)

            # Determine which global speakers are active (have at least one past spk embedding)
            global_active = past_spk_assignments.max(dim=1)[0] > 0.5  # (B, max_num_spks)

            # Mask inactive global speakers (no past spk embeddings for them)
            aggregated_scores = aggregated_scores.masked_fill(~global_active.unsqueeze(1), zero_score)

        else:
            # No past spk embeddings - all scores are zero_score (will go to dustbin)
            aggregated_scores = torch.full(
                (batch_size, local_num_spks, self.max_num_spks),
                zero_score,
                device=spk_embs.device,
                dtype=spk_embs.dtype,
            )
            global_active = torch.zeros(
                (batch_size, self.max_num_spks),
                device=spk_embs.device,
                dtype=torch.bool,
            )

        # Mask zero local spk embeddings in aggregated scores
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
            spk_assignments (torch.Tensor): Soft assignments for each local spk embedding.
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


class JointTimeSpeakerEncoderBlock(nn.Module):
    """
    A single block of the Joint Time-Speaker Encoder.
    
    This block alternates between:
    1. Time-wise self-attention (with relative positional encoding)
    2. Speaker-wise self-attention (no positional encoding - speaker slots are permutation-invariant)
    3. Feed-forward network
    
    Args:
        hidden_size (int): Size of the hidden dimension (d_model)
        inner_size (int): Size of the feed-forward inner dimension
        num_attention_heads_t (int): Number of attention heads for time-wise attention
        num_attention_heads_s (int): Number of attention heads for speaker-wise attention
        attn_score_dropout_t (float): Dropout rate for time-wise attention scores
        attn_score_dropout_s (float): Dropout rate for speaker-wise attention scores
        attn_layer_dropout_t (float): Dropout rate for time-wise attention layer output
        attn_layer_dropout_s (float): Dropout rate for speaker-wise attention layer output
        ffn_dropout (float): Dropout rate for feed-forward network
        hidden_act (str): Activation function for feed-forward network
        pre_ln (bool): Whether to use pre-LayerNorm (True) or post-LayerNorm (False)
        pos_bias_u: Shared positional bias u for relative attention
        pos_bias_v: Shared positional bias v for relative attention
    """

    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        num_attention_heads_t: int = 4,
        num_attention_heads_s: int = 4,
        attn_score_dropout_t: float = 0.0,
        attn_score_dropout_s: float = 0.0,
        attn_layer_dropout_t: float = 0.0,
        attn_layer_dropout_s: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        pre_ln: bool = True,
        pos_bias_u=None,
        pos_bias_v=None,
    ):
        super().__init__()
        self.pre_ln = pre_ln
        self.hidden_size = hidden_size

        # Time-wise self-attention with relative positional encoding
        self.norm_time = nn.LayerNorm(hidden_size, eps=1e-5)
        self.time_attn = RelPositionMultiHeadAttention(
            n_head=num_attention_heads_t,
            n_feat=hidden_size,
            dropout_rate=attn_score_dropout_t,
            pos_bias_u=pos_bias_u,
            pos_bias_v=pos_bias_v,
        )
        self.time_dropout = nn.Dropout(attn_layer_dropout_t)

        # Speaker-wise self-attention (no positional encoding)
        self.norm_speaker = nn.LayerNorm(hidden_size, eps=1e-5)
        self.speaker_attn = MultiHeadAttention(
            n_head=num_attention_heads_s,
            n_feat=hidden_size,
            dropout_rate=attn_score_dropout_s,
        )
        self.speaker_dropout = nn.Dropout(attn_layer_dropout_s)

        # Feed-forward network
        self.norm_ff = nn.LayerNorm(hidden_size, eps=1e-5)
        self.ffn = PositionWiseFF(hidden_size, inner_size, ffn_dropout, hidden_act)

    def forward_preln(
        self,
        x: torch.Tensor,
        time_mask: Optional[torch.Tensor] = None,
        speaker_mask: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pre-LayerNorm forward pass.
        
        Args:
            x: Input tensor of shape (B, S, T, D)
            time_mask: Mask for time attention (B*S, T, T) or (B*S, 1, T), True means masked
            speaker_mask: Mask for speaker attention (B*T, S, S) or (B*T, 1, S), True means masked
            pos_emb: Relative positional embeddings for time attention
            
        Returns:
            Output tensor of shape (B, S, T, D)
        """
        batch_size, num_speakers, seq_len, _ = x.shape
        # Reshape for time-wise attention: (B, S, T, D) -> (B*S, T, D)
        x_time = x.reshape(batch_size * num_speakers, seq_len, self.hidden_size)

        # Time-wise self-attention with relative positional encoding
        residual = x_time
        x_time = self.norm_time(x_time)
        x_time = self.time_attn(query=x_time, key=x_time, value=x_time, mask=time_mask, pos_emb=pos_emb)
        x_time = self.time_dropout(x_time) + residual

        # Reshape for speaker-wise attention: (B*S, T, D) -> (B, S, T, D) -> (B, T, S, D) -> (B*T, S, D)
        x_spk = x_time.view(batch_size, num_speakers, seq_len, self.hidden_size)
        x_spk = x_spk.permute(0, 2, 1, 3)  # (B, T, S, D)
        x_spk = x_spk.reshape(batch_size * seq_len, num_speakers, self.hidden_size)

        # Speaker-wise self-attention (no positional encoding)
        residual = x_spk
        x_spk = self.norm_speaker(x_spk)
        x_spk = self.speaker_attn(query=x_spk, key=x_spk, value=x_spk, mask=speaker_mask)
        x_spk = self.speaker_dropout(x_spk) + residual

        # Reshape back: (B*T, S, D) -> (B, T, S, D) -> (B, S, T, D)
        x = x_spk.view(batch_size, seq_len, num_speakers, self.hidden_size)
        x = x.permute(0, 2, 1, 3)  # (B, S, T, D)

        # Feed-forward network (reshape to 3D for efficiency)
        x_flat = x.reshape(batch_size * num_speakers, seq_len, self.hidden_size)
        residual = x_flat
        x_flat = self.norm_ff(x_flat)
        x_flat = self.ffn(x_flat)
        x_flat = x_flat + residual

        # Reshape back to (B, S, T, D)
        x = x_flat.view(batch_size, num_speakers, seq_len, self.hidden_size)
        return x

    def forward_postln(
        self,
        x: torch.Tensor,
        time_mask: Optional[torch.Tensor] = None,
        speaker_mask: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Post-LayerNorm forward pass.
        
        Args:
            x: Input tensor of shape (B, S, T, D)
            time_mask: Mask for time attention (B*S, T, T) or (B*S, 1, T), True means masked
            speaker_mask: Mask for speaker attention (B*T, S, S) or (B*T, 1, S), True means masked
            pos_emb: Relative positional embeddings for time attention
            
        Returns:
            Output tensor of shape (B, S, T, D)
        """
        batch_size, num_speakers, seq_len, _ = x.shape
        # Reshape for time-wise attention: (B, S, T, D) -> (B*S, T, D)
        x_time = x.reshape(batch_size * num_speakers, seq_len, self.hidden_size)

        # Time-wise self-attention with relative positional encoding
        residual = x_time
        x_time = self.time_attn(query=x_time, key=x_time, value=x_time, mask=time_mask, pos_emb=pos_emb)
        x_time = self.norm_time(self.time_dropout(x_time) + residual)

        # Reshape for speaker-wise attention: (B*S, T, D) -> (B, S, T, D) -> (B, T, S, D) -> (B*T, S, D)
        x_spk = x_time.view(batch_size, num_speakers, seq_len, self.hidden_size)
        x_spk = x_spk.permute(0, 2, 1, 3)  # (B, T, S, D)
        x_spk = x_spk.reshape(batch_size * seq_len, num_speakers, self.hidden_size)

        # Speaker-wise self-attention (no positional encoding)
        residual = x_spk
        x_spk = self.speaker_attn(query=x_spk, key=x_spk, value=x_spk, mask=speaker_mask)
        x_spk = self.norm_speaker(self.speaker_dropout(x_spk) + residual)

        # Reshape back: (B*T, S, D) -> (B, T, S, D) -> (B, S, T, D)
        x = x_spk.view(batch_size, seq_len, num_speakers, self.hidden_size)
        x = x.permute(0, 2, 1, 3)  # (B, S, T, D)

        # Feed-forward network (reshape to 3D for efficiency)
        x_flat = x.reshape(batch_size * num_speakers, seq_len, self.hidden_size)
        residual = x_flat
        x_flat = self.ffn(x_flat)
        x_flat = self.norm_ff(x_flat + residual)

        # Reshape back to (B, S, T, D)
        x = x_flat.view(batch_size, num_speakers, seq_len, self.hidden_size)
        return x

    def forward(
        self,
        x: torch.Tensor,
        time_mask: Optional[torch.Tensor] = None,
        speaker_mask: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, S, T, D)
            time_mask: Mask for time attention (B*S, T, T) or (B*S, 1, T), True means masked
            speaker_mask: Mask for speaker attention (B*T, S, S) or (B*T, 1, S), True means masked
            pos_emb: Relative positional embeddings for time attention
            
        Returns:
            Output tensor of shape (B, S, T, D)
        """
        if self.pre_ln:
            return self.forward_preln(x, time_mask, speaker_mask, pos_emb)
        else:
            return self.forward_postln(x, time_mask, speaker_mask, pos_emb)


class JointTimeSpeakerEncoder(nn.Module):
    """
    Joint Time-Speaker Encoder for speaker diarization.
    
    This encoder alternates between time-wise and speaker-wise self-attention,
    allowing the model to learn both temporal patterns and speaker interactions.
    
    - Time-wise attention uses relative positional encoding (like Conformer)
    - Speaker-wise attention has no positional encoding (speaker slots are permutation-invariant)
    
    Args:
        num_layers (int): Number of encoder blocks
        hidden_size (int): Size of the hidden dimension (d_model)
        inner_size (int): Size of the feed-forward inner dimension
        num_attention_heads_t (int): Number of attention heads for time-wise attention
        num_attention_heads_s (int): Number of attention heads for speaker-wise attention
        attn_score_dropout_t (float): Dropout rate for time-wise attention scores
        attn_score_dropout_s (float): Dropout rate for speaker-wise attention scores
        attn_layer_dropout_t (float): Dropout rate for time-wise attention layer output
        attn_layer_dropout_s (float): Dropout rate for speaker-wise attention layer output
        ffn_dropout (float): Dropout rate for feed-forward network
        hidden_act (str): Activation function for feed-forward network
        pre_ln (bool): Whether to use pre-LayerNorm (True) or post-LayerNorm (False)
        pre_ln_final_layer_norm (bool): Whether to apply final layer norm when using pre-LN
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        inner_size: int,
        num_attention_heads_t: int = 4,
        num_attention_heads_s: int = 4,
        attn_score_dropout_t: float = 0.0,
        attn_score_dropout_s: float = 0.0,
        attn_layer_dropout_t: float = 0.0,
        attn_layer_dropout_s: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        pre_ln: bool = True,
        pre_ln_final_layer_norm: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.pre_ln = pre_ln

        # Final layer norm for pre-LN
        if pre_ln and pre_ln_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        else:
            self.final_layer_norm = None

        # Relative positional encoding for time-wise attention
        self.pos_enc = RelPositionalEncoding(
            d_model=hidden_size,
            dropout_rate=0.0,
            max_len=5000,
        )

        # Create encoder blocks
        self.layers = nn.ModuleList([
            JointTimeSpeakerEncoderBlock(
                hidden_size=hidden_size,
                inner_size=inner_size,
                num_attention_heads_t=num_attention_heads_t,
                num_attention_heads_s=num_attention_heads_s,
                attn_score_dropout_t=attn_score_dropout_t,
                attn_score_dropout_s=attn_score_dropout_s,
                attn_layer_dropout_t=attn_layer_dropout_t,
                attn_layer_dropout_s=attn_layer_dropout_s,
                ffn_dropout=ffn_dropout,
                hidden_act=hidden_act,
                pre_ln=pre_ln,
                pos_bias_u=None,
                pos_bias_v=None,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        time_lengths: Optional[torch.Tensor] = None,
        speaker_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, S, T, D)
            time_lengths: Lengths of valid time frames for each batch item, shape (B,)
            speaker_mask: Optional mask for speaker attention (B*T, S, S), True means masked
            
        Returns:
            Output tensor of shape (B, S, T, D)
        """
        batch_size, num_speakers, seq_len, _ = x.shape

        # Generate relative positional embeddings for time-wise attention
        # pos_enc expects (B, T, D) and returns (x, pos_emb)
        # We use a dummy input just to get pos_emb
        self.pos_enc.extend_pe(seq_len, x.device, x.dtype)
        dummy_time = x[:, 0, :, :].clone()  # (B, T, D)
        _, pos_emb = self.pos_enc(x=dummy_time)  # pos_emb: (1, 2*T-1, D) or similar

        # Create time mask from lengths if provided
        time_mask = None
        if time_lengths is not None:
            # Create mask: True where position >= length (i.e., padding positions)
            # Shape: (B, T)
            arange = torch.arange(seq_len, device=x.device)
            time_len_mask = arange.unsqueeze(0) >= time_lengths.unsqueeze(1)  # (B, T)
            # Expand for all speakers: (B, T) -> (B, 1, T) -> (B, S, T) -> (B*S, 1, T)
            time_mask = time_len_mask.unsqueeze(1).expand(-1, num_speakers, -1)
            time_mask = time_mask.reshape(batch_size * num_speakers, 1, seq_len)

        # Forward through all layers
        for layer in self.layers:
            x = layer(
                x=x,
                time_mask=time_mask,
                speaker_mask=speaker_mask,
                pos_emb=pos_emb,
            )

        # Apply final layer norm if using pre-LN
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        return x