# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import math

import torch
import torch.distributed
from omegaconf import DictConfig

from nemo.collections.asr.modules.transformer.transformer_decoders import TransformerDecoder
from nemo.collections.asr.modules.transformer.transformer_encoders import TransformerEncoder
from nemo.collections.asr.modules.transformer.transformer_modules import (
    AttentionBridge,
    MultiHeadAttention,
    PositionWiseFF,
)
from nemo.collections.asr.parts.submodules.multi_head_attention import MultiHeadAttention as MHA
from nemo.collections.asr.parts.submodules.adapters.attention_adapter_mixin import AttentionAdapterModuleMixin
from nemo.collections.asr.parts.submodules.multi_head_attention import PositionalEncoding
from nemo.collections.common.parts import form_attention_mask
from nemo.core.classes.mixins import adapter_mixins
from nemo.utils import logging

__all__ = ["PerceiverEncoder", "SimplePerceiverEncoder"]


class SimplePerceiverBlock(torch.nn.Module):
    """
    Building block of a simple Perceiver Encoder.
    This block is similar to TransformerDecoderBlock but with a different order of operations.
    It consists of a cross-attention layer, a self-attention layer, and a feed-forward network.
    This is used to build a SimplePerceiverEncoder.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        inner_size: number of neurons in the intermediate part of feed-forward
            net, usually is (4-8 x hidden_size) in the papers
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
        # Cross-attention layer
        self.layer_norm_1 = torch.nn.LayerNorm(hidden_size, eps=1e-5)
        self.first_sub_layer = MHA(
            n_head=num_cross_attention_heads,
            n_feat=hidden_size,
            dropout_rate=cross_attn_dropout,
        )
        # Self-attention layer
        self.layer_norm_2 = torch.nn.LayerNorm(hidden_size, eps=1e-5)
        self.second_sub_layer = MHA(
            n_head=num_self_attention_heads,
            n_feat=hidden_size,
            dropout_rate=self_attn_dropout,
        )
        # Feed-forward layer
        self.layer_norm_3 = torch.nn.LayerNorm(hidden_size, eps=1e-5)
        self.third_sub_layer = PositionWiseFF(hidden_size, inner_size, ffn_dropout, hidden_act)


    def forward_preln(self, hidden_states, hidden_mask, encoder_states, encoder_mask):
        """
        Pre-LayerNorm block
        Order of operations: LN -> Cross-Attn -> Residual -> LN -> Self-Attn -> Residual -> LN -> FFN
        """
        residual = hidden_states
        hidden_states = self.layer_norm_1(hidden_states)
        cross_attn_output = self.first_sub_layer(query=hidden_states, key=encoder_states, value=encoder_states, mask=encoder_mask)
        cross_attn_output += residual

        residual = cross_attn_output
        cross_attn_output = self.layer_norm_2(cross_attn_output)
        self_attn_output = self.second_sub_layer(query=cross_attn_output, key=cross_attn_output, value=cross_attn_output, mask=hidden_mask)
        self_attn_output += residual

        residual = self_attn_output
        self_attn_output = self.layer_norm_3(self_attn_output)
        output_states = self.third_sub_layer(self_attn_output)
        output_states += residual

        return output_states

    def forward_postln(self, hidden_states, hidden_mask, encoder_states, encoder_mask):
        """
        Post-LayerNorm block
        Order of operations: Cross-Attn -> Residual -> LN -> Self-Attn -> Residual -> LN -> FFN -> Residual -> LN
        """
        cross_attn_output = self.first_sub_layer(query=hidden_states, key=encoder_states, value=encoder_states, mask=encoder_mask)
        cross_attn_output += hidden_states
        cross_attn_output = self.layer_norm_1(cross_attn_output)

        self_attn_output = self.second_sub_layer(query=cross_attn_output, key=cross_attn_output, value=cross_attn_output, mask=hidden_mask)
        self_attn_output += cross_attn_output
        self_attn_output = self.layer_norm_2(self_attn_output)

        output_states = self.third_sub_layer(self_attn_output)
        output_states += self_attn_output
        return self.layer_norm_3(output_states)

    def forward(self, hidden_states, hidden_mask, encoder_states, encoder_mask):
        if self.pre_ln:
            return self.forward_preln(hidden_states, hidden_mask, encoder_states, encoder_mask)
        else:
            return self.forward_postln(hidden_states, hidden_mask, encoder_states, encoder_mask)

class SimplePerceiverEncoder(torch.nn.Module):
    def __init__(
        self,
        num_latents: int,
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
    ):
        super().__init__()

        if pre_ln and pre_ln_final_layer_norm:
            self.final_layer_norm = torch.nn.LayerNorm(hidden_size, eps=1e-5)
        else:
            self.final_layer_norm = None

        self._num_latents = num_latents
        self._num_layers = num_layers

        # learnable initial hidden values
        self.init_hidden = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(num_latents, hidden_size)))

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
        self.layers = torch.nn.ModuleList([copy.deepcopy(layer) for _ in range(self._num_layers)])

    @property
    def num_latents(self):
        return self._num_latents

    def forward(self, encoder_states, encoder_mask, hidden_states=None, hidden_mask=None):
        """
        Args:
            encoder_states: output of the encoder (B x L_enc x H)
            encoder_mask: encoder inputs mask (B x num_latents x L_enc)
            hidden_states: optional initial hidden states (B x num_latents x H)
            hidden_mask: optional initial hidden mask (B x num_latents x num_latents)
        """
        # all hidden values are active
        if hidden_mask is None:
            hidden_mask = torch.ones(
                encoder_states.shape[0], self._num_latents, self._num_latents, dtype=encoder_mask.dtype, device=encoder_mask.device
            )

        # initialize hidden state
        if hidden_states is None:
            hidden_states = self.init_hidden.unsqueeze(0).expand(encoder_states.shape[0], -1, -1)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                hidden_mask=hidden_mask,
                encoder_states=encoder_states,
                encoder_mask=encoder_mask,
            )

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states

class PerceiverEncoder(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        inner_size: int,
        mask_future: bool = False,
        num_attention_heads: int = 1,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        pre_ln: bool = False,
        pre_ln_final_layer_norm: bool = True,
        hidden_steps: int = 32,
        hidden_init_method: str = "default",
        hidden_blocks: int = 2,
    ):
        super().__init__()

        self._hidden_steps = hidden_steps
        self._hidden_init_method = hidden_init_method
        self._hidden_blocks = hidden_blocks

        if self._hidden_init_method == "default":
            self._hidden_init_method = "params"

        if self.hidden_init_method not in self.supported_init_methods:
            raise ValueError(
                "Unknown hidden_init_method = {hidden_init_method}, supported methods are {supported_init_methods}".format(
                    hidden_init_method=self.hidden_init_method, supported_init_methods=self.supported_init_methods,
                )
            )

        diagonal = 0 if mask_future else None

        if self.hidden_init_method == "params":
            # learnable initial hidden values
            self.init_hidden = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(hidden_steps, hidden_size)))
            self.init_cross_att = TransformerDecoder(
                num_layers=1,
                hidden_size=hidden_size,
                inner_size=inner_size,
                num_attention_heads=num_attention_heads,
                attn_score_dropout=attn_score_dropout,
                attn_layer_dropout=attn_layer_dropout,
                ffn_dropout=ffn_dropout,
                hidden_act=hidden_act,
                pre_ln=pre_ln,
                pre_ln_final_layer_norm=pre_ln_final_layer_norm,
            )
            self.init_cross_att.diagonal = diagonal
        elif self.hidden_init_method == "bridge":
            # initialize latent with attention bridge
            self.att_bridge = AttentionBridge(hidden_size=hidden_size, k=hidden_steps, bridge_size=inner_size,)

        # cross-attention encoder
        layer = TransformerDecoder(
            num_layers=1,
            hidden_size=hidden_size,
            inner_size=inner_size,
            num_attention_heads=num_attention_heads,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
            ffn_dropout=ffn_dropout,
            hidden_act=hidden_act,
            pre_ln=pre_ln,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
        )
        layer.diagonal = diagonal
        self.cross_att_layers = torch.nn.ModuleList([copy.deepcopy(layer) for _ in range(hidden_blocks)])

        # self-attention encoder
        layer = TransformerEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            inner_size=inner_size,
            mask_future=mask_future,
            num_attention_heads=num_attention_heads,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
            ffn_dropout=ffn_dropout,
            hidden_act=hidden_act,
            pre_ln=pre_ln,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
        )
        self.self_att_layers = torch.nn.ModuleList([copy.deepcopy(layer) for _ in range(hidden_blocks)])

    @property
    def supported_init_methods(self):
        return ["params", "bridge"]

    @property
    def hidden_steps(self):
        return self._hidden_steps

    @property
    def hidden_blocks(self):
        return self._hidden_blocks

    @property
    def hidden_init_method(self):
        return self._hidden_init_method

    def forward(self, encoder_states, encoder_mask):
        """
        Args:
            encoder_states: output of the encoder (B x L_enc x H)
            encoder_mask: encoder inputs mask (B x L_enc)
        """
        # all hidden values are active
        hidden_mask = torch.ones(
            encoder_states.shape[0], self._hidden_steps, dtype=encoder_mask.dtype, device=encoder_mask.device
        )

        # initialize hidden state
        if self._hidden_init_method == "params":
            # initialize latent with learned parameters
            hidden_states = self.init_hidden.unsqueeze(0).expand(encoder_states.shape[0], -1, -1)
            hidden_states = self.init_cross_att(
                decoder_states=hidden_states,
                decoder_mask=hidden_mask,
                encoder_states=encoder_states,
                encoder_mask=encoder_mask,
            )
        elif self._hidden_init_method == "bridge":
            # initialize latent with attention bridge
            hidden_states = self.att_bridge(hidden=encoder_states, hidden_mask=encoder_mask,)

        # apply block (cross-attention, self-attention) multiple times
        # for block in range(self._hidden_blocks):
        for self_att, cross_att in zip(self.self_att_layers, self.cross_att_layers):
            residual = hidden_states

            # cross attention of hidden over encoder states
            hidden_states = cross_att(
                decoder_states=hidden_states,
                decoder_mask=hidden_mask,
                encoder_states=encoder_states,
                encoder_mask=encoder_mask,
            )

            # self-attention over hidden
            hidden_states = self_att(encoder_states=hidden_states, encoder_mask=hidden_mask,)

            # residual connection
            hidden_states += residual

        return hidden_states, hidden_mask



