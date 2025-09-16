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

"""
Constrained Rotary Positional Embeddings (RoPE) for multimodal speech-language models.

This implementation extends standard RoPE with time constraints to prevent angle wrapping
in long audio sequences.
"""

from math import pi, log
from typing import Literal, Union, Optional
import warnings

import torch
from torch.nn import Module, ModuleList
from torch.cuda.amp import autocast
from torch import nn, einsum, broadcast_tensors, Tensor

from einops import rearrange, repeat

from nemo.core.classes import NeuralModule
from nemo.utils import logging

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# broadcat, as used in multimodal applications

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
    """Apply rotary positional embedding to tensor t.
    
    Args:
        freqs: Frequency tensor from RotaryEmbedding forward pass
        t: Input tensor to apply RoPE to (queries or keys)
        start_index: Index to start applying RoPE (for partial RoPE)
        scale: Scaling factor for XPos
        seq_dim: Dimension that corresponds to sequence length
        
    Returns:
        Tensor with rotary embeddings applied
    """
    ori_dtype = t.dtype
    embed_dtype = torch.float64
    t = t.to(embed_dtype)
    
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:].to(t)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim = -1).to(ori_dtype)

# learned rotation helpers

def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
    if exists(freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
    return apply_rotary_emb(rotations, t, start_index = start_index)

# main rotary embedding class

class ConstrainedRotaryEmbedding(NeuralModule):
    """
    Constrained Rotary Positional Embedding for multimodal speech-language models.
    
    This implementation adds time constraints to prevent angle wrapping in long sequences,
    which is particularly important for audio processing in multimodal models.
    
    Args:
        dim: Embedding dimension (should be head_dim for attention)
        custom_freqs: Optional custom frequency tensor
        freqs_for: Type of frequencies ('lang' for language, 'pixel' for vision, 'constant' for fixed)
        theta: Base frequency for RoPE (adjusted automatically if max_time is set)
        max_freq: Maximum frequency for pixel embeddings
        num_freqs: Number of frequencies for constant embeddings  
        learned_freq: Whether frequencies are learnable parameters
        use_xpos: Whether to use XPos scaling for length extrapolation
        xpos_scale_base: Base for XPos scaling
        interpolate_factor: Factor for RoPE interpolation
        theta_rescale_factor: Factor to rescale theta
        seq_before_head_dim: Whether sequence dim comes before head dim
        cache_if_possible: Whether to cache computed embeddings
        max_time: Maximum time duration to constrain angle wrapping (in seconds for audio)
    """
    
    def __init__(
        self,
        dim,
        custom_freqs: Optional[Tensor] = None,
        freqs_for: Union[Literal['lang', 'pixel', 'constant']] = 'lang',
        theta = 50000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False,
        use_xpos = False,
        xpos_scale_base = 512,
        interpolate_factor = 1.,
        theta_rescale_factor = 1.,
        seq_before_head_dim = False,
        cache_if_possible = True,
        max_time = 7200  # 2 hours of audio by default
    ):
        super().__init__()

        self.dim = dim
        self.freqs_for = freqs_for
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        self.learned_freq = learned_freq
        self.use_xpos = use_xpos
        self.xpos_scale_base = xpos_scale_base
        self.interpolate_factor = interpolate_factor
        self.theta_rescale_factor = theta_rescale_factor
        self.cache_if_possible = cache_if_possible
        self.max_time = max_time

        self.tmp_store('cached_freqs', None)
        self.tmp_store('cached_scales', None)

        # Adjust theta to avoid angle wrapping after large times
        if exists(max_time) and freqs_for == 'lang':
            # Make sure highest frequency completes 1 full rotation over max time
            # theta = base of exponent: higher theta → lower frequency range
            # max_time * (1/theta^(0)) = 2pi  =>  theta = max_time / (2pi)
            theta = max_time / (2 * pi)
            logging.info(f"ConstrainedRotaryEmbedding: Adjusted theta to {theta:.2f} for max_time={max_time}s")

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.theta = theta

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()

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
        if not use_xpos:
            self.tmp_store('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.tmp_store('scale', scale)

        # add apply_rotary_emb as static method
        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def tmp_store(self, key, value):
        self.register_buffer(key, value, persistent = False)

    def get_seq_pos(self, seq_len, device, dtype, offset = 0):
        """Get sequence positions with interpolation factor applied."""
        return (torch.arange(seq_len, device = device, dtype = dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim = None, offset = 0, modality_segments = None):
        """
        Rotate queries or keys tensor with rotary embeddings.
        
        Args:
            t: Input tensor (queries or keys)
            seq_dim: Sequence dimension 
            offset: Position offset for the sequence
            modality_segments: Optional list of (start, end, modality_type) tuples for mixed sequences
                               modality_type can be 'audio' or 'text' to apply different position encodings
        """
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos, 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        if modality_segments is not None:
            # Handle mixed audio/text sequences with different temporal characteristics
            return self._apply_multimodal_rope(t, modality_segments, seq_dim, offset)
        else:
            # Standard rotary embedding application
            freqs = self.forward(self.get_seq_pos(seq_len, device = device, dtype = dtype, offset = offset), seq_len = seq_len, offset = offset)

            if seq_dim == -3:
                freqs = rearrange(freqs, 'n d -> n 1 d')

            return apply_rotary_emb(freqs, t, seq_dim = seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim = None, offset = 0):
        """Rotate queries with cached keys, handling different sequence lengths."""
        seq_dim = default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        rotated_q = self.rotate_queries_or_keys(q, seq_dim = seq_dim, offset = k_len - q_len + offset)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim = seq_dim, offset = offset)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim = None):
        """Rotate both queries and keys with XPos scaling."""
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
        """Get XPos scaling factors."""
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
        """Get axial frequencies for multidimensional inputs (e.g., 2D vision)."""
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
        """
        Forward pass to compute rotary frequency embeddings.
        
        Args:
            t: Position tensor (sequence positions)
            seq_len: Length of sequence (for caching)
            offset: Offset for positions
            
        Returns:
            Tensor: Frequency embeddings for RoPE application
        """
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

        # Scale time to keep t * freq <= 2pi (critical for long audio sequences)
        if hasattr(self, 'max_time') and self.max_time is not None:
            t = t / self.max_time * (2 * pi)

        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        if should_cache:
            self.tmp_store('cached_freqs', freqs.detach())

        return freqs

    def _apply_multimodal_rope(self, t, modality_segments, seq_dim, offset):
        """
        Apply rotary embeddings to mixed audio/text sequences with segment-aware positioning.
        
        Args:
            t: Input tensor [batch, seq_len, ...] or [batch, heads, seq_len, ...]
            modality_segments: List of (start, end, modality_type, audio_time_scale) tuples
            seq_dim: Sequence dimension
            offset: Base position offset
            
        Returns:
            Tensor with rotary embeddings applied segment-wise
        """
        if not modality_segments:
            # Fallback to standard application
            return self.rotate_queries_or_keys(t, seq_dim=seq_dim, offset=offset)
        
        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]
        output = t.clone()
        
        for segment_start, segment_end, modality_type, time_scale in modality_segments:
            if segment_end <= segment_start:
                continue
                
            # Extract segment
            segment_len = segment_end - segment_start
            if seq_dim == 2:  # [batch, heads, seq_len, head_dim]
                segment = t[:, :, segment_start:segment_end]
            elif seq_dim == 1:  # [batch, seq_len, hidden_dim]
                segment = t[:, segment_start:segment_end]
            else:
                raise ValueError(f"Unsupported seq_dim: {seq_dim}")
            
            # Compute position encoding based on modality
            if modality_type == 'audio':
                # Audio segments use time-scaled positions to reflect temporal nature
                positions = torch.arange(segment_len, device=device, dtype=dtype) * time_scale
                positions = positions / self.interpolate_factor
            else:  # text or other modalities
                # Text uses standard integer positions
                positions = self.get_seq_pos(segment_len, device=device, dtype=dtype, offset=segment_start + offset)
            
            # Compute frequencies for this segment
            freqs = self.forward(positions, seq_len=segment_len)
            
            if seq_dim == 2:
                freqs = rearrange(freqs, 'n d -> 1 1 n d')  # Broadcast for [batch, heads, seq_len, head_dim]
            elif seq_dim == 1:
                freqs = rearrange(freqs, 'n d -> 1 n d')    # Broadcast for [batch, seq_len, hidden_dim]
            
            # Apply rotary embedding to segment
            rotated_segment = apply_rotary_emb(freqs, segment, seq_dim=seq_dim)
            
            # Write back to output
            if seq_dim == 2:
                output[:, :, segment_start:segment_end] = rotated_segment
            elif seq_dim == 1:
                output[:, segment_start:segment_end] = rotated_segment
                
        return output

# Utility functions for integration with existing attention mechanisms

def inject_rotary_embedding(attention_layer, rope_config):
    """
    Inject rotary embeddings into a HuggingFace attention layer.
    
    Args:
        attention_layer: HuggingFace attention module
        rope_config: Configuration dict for rotary embeddings
        
    Returns:
        Modified attention layer with RoPE injection
    """
    if not hasattr(attention_layer, '_rope_injected'):
        # Get head dimension
        head_dim = attention_layer.head_dim if hasattr(attention_layer, 'head_dim') else \
                   attention_layer.hidden_size // attention_layer.num_heads
        
        # Filter rope_config to only include valid constructor parameters
        valid_params = {
            'custom_freqs', 'freqs_for', 'theta', 'max_freq', 'num_freqs', 
            'learned_freq', 'use_xpos', 'xpos_scale_base', 'interpolate_factor',
            'theta_rescale_factor', 'seq_before_head_dim', 'cache_if_possible', 'max_time'
        }
        
        filtered_config = {k: v for k, v in rope_config.items() if k in valid_params}
        
        # Create rotary embedding
        rotary_emb = ConstrainedRotaryEmbedding(
            dim=head_dim,
            **filtered_config
        )
        
        # Store rotary embedding on the attention layer
        attention_layer.rotary_emb = rotary_emb
        attention_layer._rope_injected = True
        
        # Monkey patch the forward method
        original_forward = attention_layer.forward
        
        def forward_with_rope(*args, **kwargs):
            return _attention_forward_with_rope(attention_layer, original_forward, *args, **kwargs)
        
        attention_layer.forward = forward_with_rope
        
        logging.info(f"Injected ConstrainedRotaryEmbedding into {type(attention_layer).__name__}")
    
    return attention_layer

def _attention_forward_with_rope(attention_layer, original_forward, *args, **kwargs):
    """
    Modified attention forward pass that applies rotary embeddings to queries and keys.
    
    This function intercepts the attention forward pass and applies constrained
    rotary embeddings to the query and key projections before the attention computation.
    """
    # Call original forward but intercept at the right point to apply RoPE
    # We need to monkey patch the internal computation to apply RoPE after Q, K projections
    
    # Store original projections
    original_q_proj_forward = attention_layer.q_proj.forward
    original_k_proj_forward = attention_layer.k_proj.forward
    
    def q_proj_with_rope(hidden_states):
        # Get query projections
        query = original_q_proj_forward(hidden_states)
        
        # Reshape for multi-head attention: [batch, seq_len, num_heads, head_dim]
        bsz, seq_len, _ = query.shape
        query = query.view(bsz, seq_len, attention_layer.num_heads, attention_layer.head_dim)
        
        # Apply rotary embedding to queries
        if hasattr(attention_layer, 'rotary_emb'):
            # Get modality segments if available (set by SALM model)
            modality_segments = getattr(attention_layer, '_current_modality_segments', None)
            if modality_segments and len(modality_segments) > 0:
                # Use first batch's segments (assumes uniform batching for now)
                segments = modality_segments[0] if modality_segments else None
            else:
                segments = None
                
            query = attention_layer.rotary_emb.rotate_queries_or_keys(
                query.transpose(1, 2),  # [batch, num_heads, seq_len, head_dim]
                seq_dim=2,
                modality_segments=segments
            ).transpose(1, 2)  # Back to [batch, seq_len, num_heads, head_dim]
        
        # Reshape back to original format
        query = query.view(bsz, seq_len, -1)
        return query
    
    def k_proj_with_rope(hidden_states):
        # Get key projections
        key = original_k_proj_forward(hidden_states)
        
        # Reshape for multi-head attention: [batch, seq_len, num_heads, head_dim]
        bsz, seq_len, _ = key.shape
        key = key.view(bsz, seq_len, attention_layer.num_heads, attention_layer.head_dim)
        
        # Apply rotary embedding to keys
        if hasattr(attention_layer, 'rotary_emb'):
            # Get modality segments if available (set by SALM model)
            modality_segments = getattr(attention_layer, '_current_modality_segments', None)
            if modality_segments and len(modality_segments) > 0:
                # Use first batch's segments (assumes uniform batching for now)
                segments = modality_segments[0] if modality_segments else None
            else:
                segments = None
                
            key = attention_layer.rotary_emb.rotate_queries_or_keys(
                key.transpose(1, 2),  # [batch, num_heads, seq_len, head_dim]
                seq_dim=2,
                modality_segments=segments
            ).transpose(1, 2)  # Back to [batch, seq_len, num_heads, head_dim]
        
        # Reshape back to original format
        key = key.view(bsz, seq_len, -1)
        return key
    
    # Temporarily replace the projection forwards
    attention_layer.q_proj.forward = q_proj_with_rope
    attention_layer.k_proj.forward = k_proj_with_rope
    
    try:
        # Call the original forward with modified projections
        result = original_forward(*args, **kwargs)
    finally:
        # Restore original projection forwards
        attention_layer.q_proj.forward = original_q_proj_forward
        attention_layer.k_proj.forward = original_k_proj_forward
    
    return result
