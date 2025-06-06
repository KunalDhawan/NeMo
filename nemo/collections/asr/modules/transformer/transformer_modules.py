# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import torch
from torch import nn
from torch.nn.functional import gelu

from nemo.collections.common.parts import form_attention_mask
from nemo.utils import logging

from math import pi

from torch.nn import Module
from torch.cuda.amp import autocast
from torch import einsum, broadcast_tensors, Tensor

from einops import rearrange, repeat
from typing import Optional

__all__ = ["TransformerEmbedding", "AttentionBridge"]


# helper functions
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# broadcat, as tortoise-tts was using it

def broadcat(tensors, dim = -1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim = dim)

# rotary embedding helper functions

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

@autocast(enabled = False)
def apply_rotary_emb(freqs, t, start_index = 0, scale = 1., seq_dim = -2):
    dtype = t.dtype

    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = torch.cat((t_left, t, t_right), dim = -1)

    return out.type(dtype)

# learned rotation helpers

def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
    if exists(freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
    return apply_rotary_emb(rotations, t, start_index = start_index)

# classes

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        custom_freqs: Optional[Tensor] = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False,
        use_xpos = False,
        xpos_scale_base = 512,
        interpolate_factor = 1.,
        theta_rescale_factor = 1.,
        seq_before_head_dim = False,
        cache_if_possible = True
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible

        self.tmp_store('cached_freqs', None)
        self.tmp_store('cached_scales', None)

        self.freqs = nn.Parameter(freqs, requires_grad = learned_freq)

        self.learned_freq = learned_freq

        # dummy for device

        self.tmp_store('dummy', torch.tensor(0))

        # default sequence dimension

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors

        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        # xpos

        self.use_xpos = use_xpos
        if not use_xpos:
            self.tmp_store('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.tmp_store('scale', scale)

        # add apply_rotary_emb as static method

        # self.apply_rotary_emb = staticmethod(apply_rotary_emb)
        # self.apply_rotary_emb = apply_rotary_emb

    def _apply_rotary_emb(self, freqs, t, start_index = 0, scale = 1., seq_dim = -2):
        dtype = t.dtype

        if t.ndim == 3:
            seq_len = t.shape[seq_dim]
            freqs = freqs[-seq_len:]

        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim

        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
        out = torch.cat((t_left, t, t_right), dim = -1)

        return out.type(dtype)


    @property
    def device(self):
        return self.dummy.device

    def tmp_store(self, key, value):
        self.register_buffer(key, value, persistent = False)

    def get_seq_pos(self, seq_len, device, dtype, offset = 0):
        return (torch.arange(seq_len, device = device, dtype = dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim = None, offset = 0):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos, 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        freqs = self.forward(self.get_seq_pos(seq_len, device = device, dtype = dtype, offset = offset), seq_len = seq_len, offset = offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return self._apply_rotary_emb(freqs, t, seq_dim = seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim = None, offset = 0):
        seq_dim = default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        rotated_q = self.rotate_queries_or_keys(q, seq_dim = seq_dim, offset = k_len - q_len + offset)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim = seq_dim, offset = offset)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim = None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype = dtype, device = device)

        freqs = self.forward(seq, seq_len = seq_len)
        scale = self.get_scale(seq, seq_len = seq_len).to(dtype)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')
            scale = rearrange(scale, 'n d -> n 1 d')

        rotated_q = apply_rotary_emb(freqs, q, scale = scale, seq_dim = seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale = scale ** -1, seq_dim = seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def get_scale(
        self,
        t: Tensor,
        seq_len: Optional[int] = None,
        offset = 0
    ):
        assert self.use_xpos

        should_cache = (
            self.cache_if_possible and
            exists(seq_len)
        )

        if (
            should_cache and \
            exists(self.cached_scales) and \
            (seq_len + offset) <= self.cached_scales.shape[0]
        ):
            return self.cached_scales[offset:(offset + seq_len)]

        scale = 1.
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, 'n -> n 1')
            scale = torch.cat((scale, scale), dim = -1)

        if should_cache:
            self.tmp_store('cached_scales', scale)

        return scale

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == 'pixel':
                pos = torch.linspace(-1, 1, steps = dim, device = self.device)
            else:
                pos = torch.arange(dim, device = self.device)

            freqs = self.forward(pos, seq_len = dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim = -1)

    @autocast(enabled = False)
    def forward(
        self,
        t: Tensor,
        seq_len = None,
        offset = 0
    ):
        should_cache = (
            self.cache_if_possible and \
            not self.learned_freq and \
            exists(seq_len) and \
            self.freqs_for != 'pixel'
        )

        if (
            should_cache and \
            exists(self.cached_freqs) and \
            (offset + seq_len) <= self.cached_freqs.shape[0]
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        freqs = self.freqs

        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        if should_cache:
            self.tmp_store('cached_freqs', freqs.detach())

        return freqs


class FixedPositionalEncoding(nn.Module):
    """
    Fixed positional encoding (embedding layer) from sine and cosine functions
    of different frequencies according to https://arxiv.org/abs/1706.03762

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        max_sequence_length: maximum allowed length of the input sequence
    """

    def __init__(self, hidden_size, max_sequence_length=512):
        super().__init__()

        self._hidden_size = hidden_size
        self._max_sequence_length = max_sequence_length
        self._build_pos_enc(hidden_size=self._hidden_size, max_sequence_length=self._max_sequence_length)

    def _build_pos_enc(self, hidden_size, max_sequence_length, device=None):
        """
        Builds/replaces pre-computed positional encoding.
        """
        pos_enc = torch.zeros(max_sequence_length, hidden_size, device=device)
        position = torch.arange(0.0, max_sequence_length).unsqueeze(1)
        coef = -math.log(10000.0) / hidden_size
        div_term = torch.exp(coef * torch.arange(0.0, hidden_size, 2))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc.div_(math.sqrt(hidden_size))
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, position_ids):
        max_pos_id = position_ids.max()
        # update positional encoding if needed
        if max_pos_id >= self._max_sequence_length:
            logging.warning(
                f'Max position id {max_pos_id} is greater than max sequence length {self._max_sequence_length}. Expanding position embeddings just for this batch. This is not expected to work very well. Consider chunking your input into smaller sequences.'
            )
            self._build_pos_enc(
                hidden_size=self._hidden_size, max_sequence_length=max_pos_id + 1, device=position_ids.device,
            )

        embeddings = torch.embedding(self.pos_enc, position_ids)

        # Revert expansion of position embeddings since this wall checkpoint size mismatches.
        if max_pos_id >= self._max_sequence_length:
            self._build_pos_enc(
                hidden_size=self._hidden_size,
                max_sequence_length=self._max_sequence_length,
                device=position_ids.device,
            )
        return embeddings


class TransformerEmbedding(nn.Module):
    """
    Embedding from token and position embeddings.
    Optionally add token_type embedding (e.g. type of the sentence in BERT).

    Args:
        vocab_size: size of the vocabulary
        hidden_size: size of the embeddings in the model, also known as d_model
        max_sequence_length: maximum allowed length of the input sequence
        num_token_types: number of different token types
            (e.g. tokens of sentence A and tokens of sentence B in BERT)
        embedding_dropout: probability of dropout applied to embeddings
        learn_positional_encodings: whether to learn positional encodings or
            use fixed (sine-cosine) ones
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_sequence_length=512,
        num_token_types=2,
        embedding_dropout=0.0,
        learn_positional_encodings=False,
    ):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.learn_positional_encodings = learn_positional_encodings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if learn_positional_encodings:
            self.position_embedding = nn.Embedding(max_sequence_length, hidden_size)
        else:
            self.position_embedding = FixedPositionalEncoding(hidden_size, max_sequence_length)
        if num_token_types > 0:
            self.token_type_embedding = nn.Embedding(num_token_types, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(embedding_dropout)

    def forward(self, input_ids, token_type_ids=None, start_pos=0):
        seq_length = input_ids.size(1)
        # we fail here only with parametric positional embedding. FixedPositionalEncoding automatically extends.
        if self.learn_positional_encodings and (seq_length > self.max_sequence_length):
            raise ValueError(
                f"Input sequence is longer than maximum allowed sequence length for positional encoding. "
                f"Got {seq_length} and {self.max_sequence_length}"
            )
        position_ids = torch.arange(
            start=start_pos, end=start_pos + seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).repeat(input_ids.size(0), 1)

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = token_embeddings + position_embeddings

        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embedding(token_type_ids)
            embeddings = embeddings + token_type_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention layer.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        num_attention_heads: number of heads in multi-head attention
        attn_score_dropout: probability of dropout applied to attention scores
        attn_layer_dropout: probability of dropout applied to the output of the
            whole layer, but before layer normalization
    """

    def __init__(self, hidden_size, num_attention_heads, attn_score_dropout=0.0, attn_layer_dropout=0.0, is_decoder=False):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number "
                "of attention heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.is_decoder = is_decoder
        self.attention_saved_list = []
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attn_head_size = int(hidden_size / num_attention_heads)
        self.attn_scale = math.sqrt(math.sqrt(self.attn_head_size))

        self.query_net = nn.Linear(hidden_size, hidden_size)
        self.key_net = nn.Linear(hidden_size, hidden_size)
        self.value_net = nn.Linear(hidden_size, hidden_size)
        self.out_projection = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(attn_score_dropout)
        self.layer_dropout = nn.Dropout(attn_layer_dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attn_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, queries, keys, values, attention_mask):

        # attention_mask is needed to hide the tokens which correspond to [PAD]
        # in the case of BERT, or to hide the future tokens in the case of
        # vanilla language modeling and translation
        query = self.query_net(queries)
        key = self.key_net(keys)
        value = self.value_net(values)
        query = self.transpose_for_scores(query) / self.attn_scale
        key = self.transpose_for_scores(key) / self.attn_scale
        value = self.transpose_for_scores(value)

        # for numerical stability we pre-divide query and key by sqrt(sqrt(d))
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.to(attention_scores.dtype)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size,)
        context = context.view(*new_context_shape)

        # output projection
        output_states = self.out_projection(context)
        output_states = self.layer_dropout(output_states)
        if self.is_decoder:
            # self.attention_saved_list.append(attention_probs[:, -1, :].unsqueeze_(0))
            self.attention_saved_list.append(attention_scores)
        return output_states


class PositionWiseFF(nn.Module):
    """
    Position-wise feed-forward network of Transformer block.

    Args:
        hidden_size: size of the embeddings in the model, also known as d_model
        inner_size: number of neurons in the intermediate part of feed-forward
            net, usually is (4-8 x hidden_size) in the papers
        ffn_dropout: probability of dropout applied to net output
        hidden_act: activation function used between two linear layers
    """

    def __init__(self, hidden_size, inner_size, ffn_dropout=0.0, hidden_act="relu"):
        super().__init__()
        self.dense_in = nn.Linear(hidden_size, inner_size)
        self.dense_out = nn.Linear(inner_size, hidden_size)
        self.layer_dropout = nn.Dropout(ffn_dropout)
        ACT2FN = {"gelu": gelu, "relu": torch.relu}
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_states):
        output_states = self.dense_in(hidden_states)
        output_states = self.act_fn(output_states)
        output_states = self.dense_out(output_states)
        output_states = self.layer_dropout(output_states)
        return output_states


class AttentionBridge(torch.nn.Module):
    """
    A multi-head attention bridge to project a variable-size hidden states
    to k hidden states (per attention head).

    Code is based on the paper https://arxiv.org/pdf/1703.03130.pdf
    """

    def __init__(self, hidden_size, k, bridge_size):
        """
        hidden_size - size of input hidden state
        k - number of attention heads
        bridge_size - size of internal feed forward weights (i.e., attention head size)
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.k = k
        self.bridge_size = bridge_size

        self.attn_scale = np.sqrt(np.sqrt(self.bridge_size))

        # build model

        self.W1 = torch.nn.Linear(hidden_size, bridge_size, bias=False)
        self.W2 = torch.nn.Linear(bridge_size, k, bias=False)
        self.act = torch.nn.ReLU()

    def forward(self, hidden, hidden_mask=None, return_ortho_loss=False):
        """
        Project hidden [B x N x H] to fixed-size [B x k x H]

        return_ortho_loss - if True returns loss term to encourage
                              orthogonal attention vectors
        """

        attention_scores = self.W2(self.act(self.W1(hidden) / self.attn_scale) / self.attn_scale).transpose(-1, -2)

        attention_mask = form_attention_mask(hidden_mask)
        if attention_mask is not None:
            attention_mask.squeeze_(1)
            attention_scores = attention_scores + attention_mask.to(attention_scores.dtype)

        A = torch.softmax(attention_scores, dim=-1)
        M = A @ hidden

        if return_ortho_loss:
            ortho_loss = ((A @ A.transpose(-1, -2)) - torch.eye(self.k).type_as(A)).pow(2).sum()

            return M, ortho_loss
        else:
            return M
