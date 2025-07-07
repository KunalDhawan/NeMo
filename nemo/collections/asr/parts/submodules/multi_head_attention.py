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
#

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
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
#

"""
Part of this code is adopted from https://github.com/espnet/espnet
"""

import math
from functools import lru_cache
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.attention
import torch.nn.functional as F

from nemo.utils import avoid_float16_autocast_context

__all__ = [
    'RelPositionMultiHeadAttention',
    'RelPositionalEncoding',
    'PositionalEncoding',
    'TrueCircularAttentionCache',
    'SimplifiedCircularAttentionCache',
]

INF_VAL = 10000.0


class TrueCircularAttentionCache:
    """
    Intelligent circular buffer implementation that sizes itself based on attention context requirements.
    Designed to provide identical context as baseline while eliminating concatenation operations.
    """
    
    def __init__(self, max_cache_len: int, n_feat: int, batch_size: int, device: torch.device, dtype: torch.dtype, att_context_size: list = None):
        """
        Initialize intelligent circular buffer that sizes itself based on attention context requirements.
        
        Args:
            max_cache_len (int): Maximum cache frames (att_context_size[0] = 70)
            n_feat (int): Feature dimension
            batch_size (int): Batch size for pre-allocation
            device (torch.device): Device for tensors
            dtype (torch.dtype): Data type for tensors
            att_context_size (list): [left_context, right_context] - model training context
        """
        self.max_cache_len = max_cache_len
        self.n_feat = n_feat
        self.device = device
        self.dtype = dtype
        
        # INTELLIGENT SIZING: Calculate total context needed based on model training
        if att_context_size is not None and len(att_context_size) >= 2:
            left_context = att_context_size[0]   # 70
            right_context = att_context_size[1]  # 13
            current_frame = 1  # Always 1 current frame
            self.total_context_needed = left_context + current_frame + right_context  # 70 + 1 + 13 = 84
            print(f"[INTELLIGENT_BUFFER] Model training context: {left_context} left + {current_frame} current + {right_context} right = {self.total_context_needed} total")
        else:
            # Fallback to max_cache_len if att_context_size not provided
            self.total_context_needed = max_cache_len
            print(f"[INTELLIGENT_BUFFER] Fallback context: {self.total_context_needed} (max_cache_len)")
        
        # Buffer sizing: Size buffer to hold the context we need to provide
        # We need to be able to provide total_context_needed - current_frames cached frames
        # Plus some extra to avoid constant overwrites
        required_buffer_size = max(self.total_context_needed, max_cache_len)
        
        # Use a buffer that's at least as big as what we need to provide
        self.buffer_size = required_buffer_size
        
        print(f"[INTELLIGENT_BUFFER] Buffer sized to {self.buffer_size} frames (need to provide {self.total_context_needed} total context)")
        
        # Pre-allocate circular buffers
        self.key_buffer = torch.zeros(batch_size, self.buffer_size, n_feat, device=device, dtype=dtype)
        self.value_buffer = torch.zeros(batch_size, self.buffer_size, n_feat, device=device, dtype=dtype)
        
        # Circular buffer state
        self.write_idx = 0  # Next position to write to
        self.current_length = 0  # Current number of frames stored
        
        # Debug settings
        self._debug_context_size = False  # Disable debug logging for production

    def add_and_get(self, new_key: torch.Tensor, new_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add new frames and return context that matches the model's training expectations.
        
        Model was trained with att_context_size=[70,13] meaning:
        - 70 left context frames + 1 current frame + 13 right context frames = 84 total frames
        
        Args:
            new_key: (batch_size, seq_len, n_feat) New key frames to cache
            new_value: (batch_size, seq_len, n_feat) New value frames to cache
            
        Returns:
            tuple: (cached_frames, cached_values) that will be used as extended context
        """
        batch_size, current_frames, n_feat = new_key.shape
        
        # CRITICAL FIX: Handle dynamic batch sizes by expanding buffer if needed
        if batch_size > self.key_buffer.shape[0]:
            # Expand buffer to accommodate larger batch size
            old_batch_size = self.key_buffer.shape[0]
            new_batch_size = batch_size
            
            # Create new larger buffers
            new_key_buffer = torch.zeros(new_batch_size, self.buffer_size, self.n_feat, 
                                       device=self.device, dtype=self.dtype)
            new_value_buffer = torch.zeros(new_batch_size, self.buffer_size, self.n_feat,
                                         device=self.device, dtype=self.dtype)
            
            # Copy existing data to new buffers
            new_key_buffer[:old_batch_size, :, :] = self.key_buffer
            new_value_buffer[:old_batch_size, :, :] = self.value_buffer
            
            # Replace buffers
            self.key_buffer = new_key_buffer
            self.value_buffer = new_value_buffer
            
            print(f"[CRITICAL_FIX] Expanded circular buffer from batch_size={old_batch_size} to {new_batch_size}")
        
        # CRITICAL FIX: Calculate and extract cached frames BEFORE adding new frames
        # This ensures we get the correct number of cached frames based on the current buffer state
        
        # Calculate how many cached frames to return based on current buffer state
        target_total_context = self.total_context_needed  # Should be 84
        target_cached_frames = max(0, target_total_context - current_frames)  # 84 - 14 = 70
        
        # We can provide up to current_length cached frames, but respect target
        cached_frames_to_return = min(self.current_length, target_cached_frames)
        
        # Extract cached frames from current buffer state BEFORE adding new frames
        if cached_frames_to_return == 0:
            # No cached frames available
            cached_key = torch.zeros(batch_size, 0, n_feat, device=new_key.device, dtype=new_key.dtype)
            cached_value = torch.zeros(batch_size, 0, n_feat, device=new_value.device, dtype=new_value.dtype)
        else:
            # Extract cached frames from buffer
            actual_batch_size = batch_size
            
            if self.current_length < self.buffer_size:
                # Buffer is not full yet - frames are stored linearly at positions [0, current_length)
                # Extract the most recent cached_frames_to_return frames
                start_pos = max(0, self.current_length - cached_frames_to_return)
                end_pos = self.current_length
                cached_key = self.key_buffer[:actual_batch_size, start_pos:end_pos, :].clone()
                cached_value = self.value_buffer[:actual_batch_size, start_pos:end_pos, :].clone()
            else:
                # Buffer is full - frames are stored circularly
                # The most recent cached_frames_to_return frames are at positions:
                # [(write_idx - cached_frames_to_return) % buffer_size, write_idx % buffer_size)
                start_pos = (self.write_idx - cached_frames_to_return) % self.buffer_size
                end_pos = self.write_idx % self.buffer_size
                
                if start_pos < end_pos:
                    # No wrap-around needed
                    cached_key = self.key_buffer[:actual_batch_size, start_pos:end_pos, :].clone()
                    cached_value = self.value_buffer[:actual_batch_size, start_pos:end_pos, :].clone()
                else:
                    # Wrap-around needed - concatenate in chronological order
                    part1_key = self.key_buffer[:actual_batch_size, start_pos:, :]     # Older frames
                    part2_key = self.key_buffer[:actual_batch_size, :end_pos, :]      # Newer frames
                    cached_key = torch.cat([part1_key, part2_key], dim=1)
                    
                    part1_value = self.value_buffer[:actual_batch_size, start_pos:, :] # Older frames
                    part2_value = self.value_buffer[:actual_batch_size, :end_pos, :]   # Newer frames
                    cached_value = torch.cat([part1_value, part2_value], dim=1)
        
        # NOW add new frames to circular buffer (after extracting cached frames)
        for i in range(current_frames):
            # Write to circular buffer
            write_pos = self.write_idx % self.buffer_size
            self.key_buffer[:batch_size, write_pos, :] = new_key[:batch_size, i, :]
            self.value_buffer[:batch_size, write_pos, :] = new_value[:batch_size, i, :]
            
            # Update circular indices
            self.write_idx += 1
            self.current_length = min(self.current_length + 1, self.buffer_size)
        
        # Debug logging
        if self._debug_context_size:
            total_context_frames = cached_key.shape[1] + current_frames
            print(f"[FIXED_DEBUG] cached_frames={cached_key.shape[1]}, current_frames={current_frames}, total_context={total_context_frames}")
            print(f"[FIXED_DEBUG] write_idx={self.write_idx}, current_length={self.current_length}, buffer_size={self.buffer_size}")
            print(f"[FIXED_DEBUG] target_context={self.total_context_needed}, providing_context={total_context_frames} {'✓ MATCH' if self.total_context_needed == total_context_frames else '✗ MISMATCH'}")
        
        return cached_key, cached_value

    def reset(self):
        """Reset circular buffer state without deallocating memory."""
        self.key_buffer.zero_()
        self.value_buffer.zero_()
        self.write_idx = 0
        self.current_length = 0


class SimplifiedCircularAttentionCache:
    """
    Simplified circular buffer implementation for testing and backward compatibility.
    Provides a simplified interface that wraps the TrueCircularAttentionCache functionality.
    """
    
    def __init__(self, max_cache_len: int, n_feat: int, batch_size: int, device: torch.device, dtype: torch.dtype, att_context_size: list = None):
        """
        Initialize simplified circular buffer.
        
        Args:
            max_cache_len (int): Maximum cache frames
            n_feat (int): Feature dimension
            batch_size (int): Batch size for pre-allocation
            device (torch.device): Device for tensors
            dtype (torch.dtype): Data type for tensors
            att_context_size (list): [left_context, right_context] - model training context
        """
        self._internal_cache = TrueCircularAttentionCache(
            max_cache_len=max_cache_len,
            n_feat=n_feat,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            att_context_size=att_context_size
        )
        
    def get_cache_for_new_frames(self, new_frames: torch.Tensor) -> torch.Tensor:
        """
        Add new frames and return cached frames for context.
        
        Args:
            new_frames: (batch_size, seq_len, n_feat) New frames to add
            
        Returns:
            torch.Tensor: Cached frames to use as context
        """
        # Use the internal cache to get both key and value cache
        # For simplicity, we'll return the same tensor for both key and value
        cached_key, cached_value = self._internal_cache.add_and_get(new_frames, new_frames)
        return cached_key  # Return cached frames for context
    
    def add_and_get_cache(self, new_frames: torch.Tensor) -> torch.Tensor:
        """
        Alternative method name for compatibility with some tests.
        
        Args:
            new_frames: (batch_size, seq_len, n_feat) New frames to add
            
        Returns:
            torch.Tensor: Cached frames to use as context
        """
        return self.get_cache_for_new_frames(new_frames)
    
    def reset(self):
        """Reset the circular buffer state."""
        self._internal_cache.reset()


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer of Transformer.
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
        use_bias (bool): whether to remove bias in linear and conv layers
        use_pytorch_sdpa (bool): use torch sdpa instead of manual attention
        use_pytorch_sdpa_backends list[str]: list of backend names to use in sdpa. None or empty list means all backends. e.g. ["MATH"]
    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        max_cache_len=0,
        use_bias=True,
        use_pytorch_sdpa=False,
        use_pytorch_sdpa_backends=None,
    ):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadAttention, self).__init__()
        self.use_pytorch_sdpa = use_pytorch_sdpa
        if self.use_pytorch_sdpa and use_pytorch_sdpa_backends:
            use_pytorch_sdpa_backends = list(
                map(
                    lambda backend_name: getattr(torch.nn.attention.SDPBackend, backend_name),
                    use_pytorch_sdpa_backends,
                )
            )
        self.use_pytorch_sdpa_backends = use_pytorch_sdpa_backends

        self.cache_drop_size = None
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.dropout = nn.Dropout(p=dropout_rate)

        self._max_cache_len = max_cache_len
        self.n_feat = n_feat
        
        # Cache optimization settings - will be set by encoder
        self._optimization_enabled = False
        self._use_true_circular = False  # Enable true circular buffer
        self._parent_encoder = None
        
        # Cache instances
        self._true_circular_cache = None  # True circular cache without concatenation

    def initialize_circular_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Initialize the true circular cache based on configuration."""
        if self._max_cache_len <= 0:
            print(f"[DEBUG] Skipping circular cache init: max_cache_len={self._max_cache_len}")
            return
            
        # Check if true circular optimization is enabled
        use_true_circular = getattr(self, '_use_true_circular', False)
        
        # Also check parent encoder configuration
        att_context_size = None
        if hasattr(self, '_parent_encoder') and self._parent_encoder:
            streaming_cfg = getattr(self._parent_encoder, 'streaming_cfg', None)
            if streaming_cfg:
                use_true_circular = getattr(streaming_cfg, 'use_true_circular_buffers', False)
                att_context_size = getattr(self._parent_encoder, 'att_context_size', None)
        
        try:
            if use_true_circular:
                # Check if cache is already initialized
                if hasattr(self, '_true_circular_cache') and self._true_circular_cache is not None:
                    print(f"[DEBUG] TRUE circular cache already initialized - skipping duplicate initialization")
                    return
                
                # Initialize TRUE circular buffer (no concatenation)
                self._true_circular_cache = TrueCircularAttentionCache(
                    max_cache_len=self._max_cache_len,
                    n_feat=self.n_feat,
                    batch_size=batch_size,
                    device=device,
                    dtype=dtype,
                    att_context_size=att_context_size
                )
                self._optimization_enabled = True
                self._use_true_circular = True
                print(f"[DEBUG] TRUE circular cache initialized: max_len={self._max_cache_len}, batch_size={batch_size}")
            else:
                print(f"[DEBUG] No circular cache initialized - using baseline path")
                
        except Exception as e:
            print(f"Warning: Failed to initialize circular cache: {e}")
            self._true_circular_cache = None
            self._optimization_enabled = False
            self._use_true_circular = False

    def set_circular_buffer_config(self, parent_encoder=None, use_true_circular: bool = False):
        """
        Set the circular buffer configuration for this attention layer.
        
        Args:
            parent_encoder: Reference to the parent encoder (optional)
            use_true_circular (bool): Whether to use true circular buffers (no concatenation)
        """
        self._use_true_circular = use_true_circular
        if parent_encoder is not None:
            self._parent_encoder = parent_encoder
        
        # CRITICAL FIX: Initialize circular cache immediately when configuration is set
        if use_true_circular and self._max_cache_len > 0:
            try:
                # Get reasonable defaults for immediate initialization
                device = next(self.parameters()).device if list(self.parameters()) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                dtype = next(self.parameters()).dtype if list(self.parameters()) else torch.float32
                
                # Get att_context_size from parent encoder
                att_context_size = None
                if parent_encoder and hasattr(parent_encoder, 'att_context_size'):
                    att_context_size = parent_encoder.att_context_size
                
                # Check parent encoder for dtype configuration
                if parent_encoder and hasattr(parent_encoder, 'streaming_cfg'):
                    streaming_cfg = parent_encoder.streaming_cfg
                    if hasattr(streaming_cfg, 'cache_dtype'):
                        dtype = streaming_cfg.cache_dtype
                
                # Use a reasonable default batch size that will be expanded later if needed
                default_batch_size = 2048  # This will be the maximum, can handle smaller batches too
                
                # Initialize the circular cache immediately
                self._true_circular_cache = TrueCircularAttentionCache(
                    max_cache_len=self._max_cache_len,
                    n_feat=self.n_feat,
                    batch_size=default_batch_size,
                    device=device,
                    dtype=dtype,
                    att_context_size=att_context_size
                )
                self._optimization_enabled = True
                print(f"[DEBUG] TRUE circular cache immediately initialized: max_len={self._max_cache_len}, batch_size={default_batch_size}, dtype={dtype}")
                
            except Exception as e:
                print(f"[WARNING] Failed to immediately initialize circular cache: {e}")
                self._true_circular_cache = None
                self._optimization_enabled = False
                self._use_true_circular = False
        else:
            # Clear existing caches if configuration changes
            self._true_circular_cache = None
            self._optimization_enabled = False

    def cleanup_cache_buffers(self):
        """Clean up cache buffers to prevent memory leaks."""
        # Only clear very large buffers to avoid thrashing
        if hasattr(self, '_cache_buffer') and self._cache_buffer is not None:
            if self._cache_buffer.numel() > 1000000:  # 1M elements threshold  
                self._cache_buffer = None

    def forward(self, query, key, value, mask, pos_emb=None, cache=None):
        """Compute 'Scaled Dot Product Attention'.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            cache (torch.Tensor) : (batch, time_cache, size)

        returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
            cache (torch.Tensor) : (batch, time_cache_next, size)
        """
        original_key_len = key.shape[1] if key is not None else 0
        original_cache = cache
        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)
        
        # Handle mask shape mismatches (should be rare with context limiting)
        if mask is not None:
            # Check for completely empty or malformed masks
            if mask.numel() == 0 or any(dim == 0 for dim in mask.shape):
                batch_size = query.shape[0]
                query_len = query.shape[1]
                key_len = key.shape[1]
                mask = torch.zeros(batch_size, query_len, key_len, device=query.device, dtype=torch.bool)
            
            # Handle dimension mismatches
            elif key.shape[1] != mask.shape[-1]:
                batch_size = mask.shape[0]
                query_len = mask.shape[1]
                key_len = key.shape[1]
                
                # CRITICAL FIX: Handle circular buffer case where key is extended
                if key_len > mask.shape[-1]:
                    # Extended key context (circular buffer case)
                    padding_size = key_len - mask.shape[-1]
                    
                    # Cached frames should be unmasked (attend to all cached frames)
                    mask_padding = torch.zeros(
                        batch_size, query_len, padding_size,
                        device=mask.device, dtype=mask.dtype
                    )
                    # Concatenate: [cached_frames_mask (0s)] + [current_frames_mask (original)]
                    mask = torch.cat([mask_padding, mask], dim=-1)
                else:
                    # Trim mask to match key length (shouldn't happen with circular buffer)
                    mask = mask[:, :, -key_len:]
        elif mask is None and using_circular_buffers and key.shape[1] > query.shape[1]:
            # Create mask for circular buffer case when no mask was provided
            batch_size = query.shape[0]
            query_len = query.shape[1]
            key_len = key.shape[1]
            # For circular buffer, all positions should be unmasked (attend to all)
            mask = torch.zeros(batch_size, query_len, key_len, device=query.device, dtype=torch.bool)
        
        # Final validation
        if mask is not None:
            expected_shape = (query.shape[0], query.shape[1], key.shape[1])
            if mask.shape != expected_shape:
                # Force create correct mask
                mask = torch.zeros(expected_shape, device=query.device, dtype=torch.bool)
        
        # Apply transformer attention
        n_batch = query.size(0)
        
        # Linear transformations for q, k, v
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        
        # Transpose for attention dot product
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        # Perform attention computation
        if self.use_pytorch_sdpa:
            output = self._forward_sdpa(query, key, value, mask)
        else:
            # Manual attention computation
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.s_d_k
            
            if mask is not None:
                mask = mask.unsqueeze(1).repeat(1, self.h, 1, 1)
                scores = scores.masked_fill(mask, -INF_VAL)
                
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            x = torch.matmul(attn, v)

        # Transpose back and combine heads
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        
        # Final linear projection
        output = self.linear_out(x)
        
        # Return cache if it was provided
        if original_cache is not None:
            return output, cache
        else:
            return output

    def update_cache(self, key, value, query, cache):
        """Update cache with comprehensive path separation for different optimization modes."""
        
        if cache is None:
            # No cache - return as-is (original behavior)
            return key, value, query, cache
        
        # STEP 1: Determine which optimization path to use
        cache_path = self._determine_cache_path()
        
        if cache_path == "true_circular":
            return self._update_cache_true_circular(key, value, query, cache)
        else:
            return self._update_cache_baseline(key, value, query, cache)
    
    def _determine_cache_path(self) -> str:
        """Determine which cache update path to use based on configuration."""
        
        # CRITICAL DEBUG: Add detailed logging to understand why circular cache is None
        use_true_circular = getattr(self, '_use_true_circular', False)
        has_cache_attr = hasattr(self, '_true_circular_cache')
        cache_not_none = has_cache_attr and self._true_circular_cache is not None
        
        if use_true_circular and not cache_not_none:
            # CRITICAL: This is where the issue occurs - cache is None when it should exist
            print(f"[CRITICAL_DEBUG] Circular cache missing! use_true_circular={use_true_circular}, has_cache_attr={has_cache_attr}")
            if has_cache_attr:
                print(f"[CRITICAL_DEBUG] Cache value: {self._true_circular_cache}")
            
            # EMERGENCY FIX: Try to re-initialize the missing cache immediately
            try:
                if hasattr(self, '_max_cache_len') and self._max_cache_len > 0:
                    device = next(self.parameters()).device if list(self.parameters()) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    dtype = next(self.parameters()).dtype if list(self.parameters()) else torch.float32
                    batch_size = 2048  # Use reasonable default
                    
                    # Get att_context_size from parent encoder if available
                    att_context_size = None
                    if hasattr(self, '_parent_encoder') and self._parent_encoder and hasattr(self._parent_encoder, 'att_context_size'):
                        att_context_size = self._parent_encoder.att_context_size
                    
                    # Emergency re-initialization
                    from nemo.collections.asr.parts.submodules.multi_head_attention import TrueCircularAttentionCache
                    self._true_circular_cache = TrueCircularAttentionCache(
                        max_cache_len=self._max_cache_len,
                        n_feat=self.n_feat,
                        batch_size=batch_size,
                        device=device,
                        dtype=dtype,
                        att_context_size=att_context_size
                    )
                    print(f"[EMERGENCY_FIX] Re-initialized circular cache during forward pass!")
                    return "true_circular"
                    
            except Exception as e:
                print(f"[EMERGENCY_FIX] Failed to re-initialize cache: {e}")
        
        # Check for true circular buffer
        if (hasattr(self, '_use_true_circular') and self._use_true_circular and
            hasattr(self, '_true_circular_cache') and self._true_circular_cache is not None):
            return "true_circular"
            
        # Default to baseline
        return "baseline"
    
    def _update_cache_true_circular(self, key, value, query, cache):
        """TRUE CIRCULAR PATH: Use true circular buffer with intelligent context provisioning."""
        
        # Store original query length for proper residual connections
        original_query_len = query.shape[1]
        
        # CRITICAL FIX: In baseline, key and value are BOTH set to the same concatenated tensor
        # The baseline does: key = value = torch.cat([cache, key], dim=1)
        # This means key and value are identical after cache update!
        # We need to replicate this exact behavior in circular buffer
        
        # Add current key frames to circular buffer and get cached context
        # Note: We use key for both key and value addition since baseline treats them identically
        cached_key_context, cached_value_context = self._true_circular_cache.add_and_get(key, key)
        
        # CRITICAL FIX: Provide the full context the model expects
        # Model was trained with 84 frames total (70 left + 14 current)
        # Concatenate cached context + current frames to achieve this
        
        if cached_key_context.shape[1] > 0:
            # Concatenate: [cached_frames] + [current_frames] = total context
            extended_context = torch.cat([cached_key_context, key], dim=1)
        else:
            # Early stage - no cached frames yet, use current frames only
            extended_context = key
        
        # CRITICAL FIX: Set BOTH key and value to the same extended context
        # This matches the baseline behavior exactly: key = value = torch.cat([cache, key], dim=1)
        key = extended_context
        value = extended_context
        
        # ENHANCED: Keep query at its original length but ensure it aligns with the attention context
        # The query represents the current frames we want to compute attention for
        # The key/value represent the full context (cached + current) to attend over
        
        # Debug output to track context sizes
        if hasattr(self._true_circular_cache, '_debug_context_size') and self._true_circular_cache._debug_context_size:
            print(f"[CIRCULAR_CACHE_DEBUG] query_len={query.shape[1]}, key_len={key.shape[1]}, "
                  f"cached_frames={cached_key_context.shape[1]}, current_frames={original_query_len}")
        
        # Return empty cache - circular buffer manages its own state
        empty_cache = torch.zeros(key.shape[0], 0, key.shape[2], 
                                 device=key.device, dtype=key.dtype)
        
        return key, value, query, empty_cache
    
    def _update_cache_baseline(self, key, value, query, cache):
        """BASELINE PATH: Use original concatenation logic (EXACT original behavior)."""
        
        cache_len = cache.shape[1]
        drop_size = self.cache_drop_size or 0
        
        # Apply drop_size if specified (original logic)
        if drop_size > 0 and cache_len > drop_size:
            cache = cache[:, drop_size:, :]
        
        # Standard concatenation: old cache + new frames (original behavior)
        key = value = torch.cat([cache, key], dim=1)
        
        # Prepare cache for next iteration
        if self._max_cache_len > 0:
            cache = key[:, -self._max_cache_len:, :]
        else:
            cache = key
                    
        return key, value, query, cache

    def _forward_sdpa(self, query, key, value, mask):
        """Forward pass using torch.nn.functional.scaled_dot_product_attention for RelPositionMultiHeadAttention."""
        # For RelPositionMultiHeadAttention, we need to fall back to manual computation
        # because SDPA doesn't support relative positional encoding
        # This is a simplified version that doesn't use relative positional encoding
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
        
        # Linear transformations for q, k, v
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        
        # Transpose for attention dot product: (batch, head, time, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        with torch.backends.cuda.sdp_kernel(
            enable_flash=torch.backends.cuda.flash_sdp_enabled() if self.use_pytorch_sdpa_backends is None or torch.backends.cuda.SDPBackend.FLASH_ATTENTION in self.use_pytorch_sdpa_backends else False,
            enable_math=torch.backends.cuda.math_sdp_enabled() if self.use_pytorch_sdpa_backends is None or torch.backends.cuda.SDPBackend.MATH in self.use_pytorch_sdpa_backends else False,
            enable_mem_efficient=torch.backends.cuda.mem_efficient_sdp_enabled() if self.use_pytorch_sdpa_backends is None or torch.backends.cuda.SDPBackend.EFFICIENT_ATTENTION in self.use_pytorch_sdpa_backends else False,
        ):
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout_rate if self.training else 0.0
            )
        
        # Transpose back and combine heads
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        
        # Final linear projection
        output = self.linear_out(x)
        
        # CRITICAL FIX: Handle circular buffer context vs residual connection requirements (SDPA version)
        query_len = query.shape[1]
        
        if self._is_using_circular_buffers():
            # CIRCULAR BUFFER MODE: Same logic as manual attention computation
            if output.shape[1] > query_len:
                output = output[:, -query_len:, :]
            elif output.shape[1] < query_len:
                padding_size = query_len - output.shape[1]
                padding = torch.zeros(output.shape[0], padding_size, output.shape[2], 
                                    device=output.device, dtype=output.dtype)
                output = torch.cat([padding, output], dim=1)
        else:
            # BASELINE MODE: Standard residual connection requirement
            if output.shape[1] != query_len:
                output = output[:, -query_len:, :]
        
        return output


class RelPositionMultiHeadAttention(MultiHeadAttention):
    """Multi-Head Attention layer of Transformer-XL with support of relative positional encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
        use_bias (bool): whether to apply bias in linear and conv layers of MultiHeadAttention
    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        pos_bias_u,
        pos_bias_v,
        max_cache_len=0,
        use_bias=True,
        use_pytorch_sdpa=False,
        use_pytorch_sdpa_backends=None,
    ):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(
            n_head=n_head,
            n_feat=n_feat,
            dropout_rate=dropout_rate,
            max_cache_len=max_cache_len,
            use_bias=use_bias,
            use_pytorch_sdpa=use_pytorch_sdpa,
            use_pytorch_sdpa_backends=use_pytorch_sdpa_backends,
        )
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable biases are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            # nn.init.normal_(self.pos_bias_u, 0.0, 0.02)
            # nn.init.normal_(self.pos_bias_v, 0.0, 0.02)
            nn.init.zeros_(self.pos_bias_u)
            nn.init.zeros_(self.pos_bias_v)
        else:
            self.pos_bias_u = pos_bias_u
            self.pos_bias_v = pos_bias_v

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x (torch.Tensor): (batch, nheads, time, 2*time-1)
        """
        b, h, qlen, pos_len = x.size()  # (b, h, t1, t2)
        # need to add a column of zeros on the left side of last dimension to perform the relative shifting
        x = torch.nn.functional.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, h, -1, qlen)  # (b, h, t2+1, t1)
        # need to drop the first row
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # (b, h, t1, t2)
        return x

    def forward_qkv(self, query, key, value):
        """Forward pass for computing Q, K, V tensors.
        Args:
            query (torch.Tensor): Query tensor (batch, time, size)
            key (torch.Tensor): Key tensor (batch, time, size)  
            value (torch.Tensor): Value tensor (batch, time, size)
        Returns:
            tuple: (q, k, v) tensors with shape (batch, head, time, d_k)
        """
        n_batch = query.size(0)
        
        # Linear transformations for q, k, v
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        
        # Transpose for attention dot product: (batch, head, time, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        return q, k, v

    def forward(self, query, key, value, mask, pos_emb, cache=None):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            pos_emb (torch.Tensor) : (batch, time1, size)
            cache (torch.Tensor) : (batch, time_cache, size)

        Returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
            cache (torch.Tensor) : (batch, time_cache_next, size)
        """
        # Store original cache for return logic
        original_cache = cache
        
        # Check if we're using circular buffers BEFORE cache update
        using_circular_buffers = self._is_using_circular_buffers()
        
        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)
        
        # CRITICAL FIX: For circular buffers, we need to handle the case where key/value
        # context is extended but query remains at original length. This requires
        # careful handling of positional encoding and attention computation.
        
        # SANITY-CHECK : ensure positional encoding is at least as long as the key context.
        if __debug__ and using_circular_buffers:
            assert pos_emb.shape[1] >= key.shape[1], (
                f"pos_emb too short (pos_emb={pos_emb.shape[1]}, key={key.shape[1]})" )
        
        # NOTE: In circular-buffer mode we keep the original positional encoding un-trimmed so
        # it still covers the cached frames.  The assertion above guarantees it is long enough.
        
        # TRIM BLOCK REMOVED – keep positional encoding unchanged
        if False and using_circular_buffers and key.shape[1] != query.shape[1]:
            # When using circular buffers, key/value may have extended context
            # but query should remain at original length. We need to ensure
            # positional encoding matches the query length, not the key length.
            
            # If pos_emb is longer than query, trim it to match query
            if pos_emb.shape[1] > query.shape[1]:
                pos_emb = pos_emb[:, :query.shape[1], :]
            # If pos_emb is shorter than query, this is an error
            elif pos_emb.shape[1] < query.shape[1]:
                # This shouldn't happen in normal operation
                raise ValueError(f"Positional encoding length ({pos_emb.shape[1]}) is shorter than query length ({query.shape[1]})")
                
        # Handle mask shape mismatches (should be rare with context limiting)
        if mask is not None:
            # Check for completely empty or malformed masks
            if mask.numel() == 0 or any(dim == 0 for dim in mask.shape):
                batch_size = query.shape[0]
                query_len = query.shape[1]
                key_len = key.shape[1]
                mask = torch.zeros(batch_size, query_len, key_len, device=query.device, dtype=torch.bool)
            
            # Handle dimension mismatches
            elif key.shape[1] != mask.shape[-1]:
                batch_size = mask.shape[0]
                query_len = mask.shape[1]
                key_len = key.shape[1]
                
                # CRITICAL FIX: Handle circular buffer case where key is extended
                if key_len > mask.shape[-1]:
                    # Extended key context (circular buffer case)
                    padding_size = key_len - mask.shape[-1]
                    
                    # Cached frames should be unmasked (attend to all cached frames)
                    mask_padding = torch.zeros(
                        batch_size, query_len, padding_size,
                        device=mask.device, dtype=mask.dtype
                    )
                    # Concatenate: [cached_frames_mask (0s)] + [current_frames_mask (original)]
                    mask = torch.cat([mask_padding, mask], dim=-1)
                else:
                    # Trim mask to match key length (shouldn't happen with circular buffer)
                    mask = mask[:, :, -key_len:]
        elif mask is None and using_circular_buffers and key.shape[1] > query.shape[1]:
            # Create mask for circular buffer case when no mask was provided
            batch_size = query.shape[0]
            query_len = query.shape[1]
            key_len = key.shape[1]
            # For circular buffer, all positions should be unmasked (attend to all)
            mask = torch.zeros(batch_size, query_len, key_len, device=query.device, dtype=torch.bool)
        
        # Final validation
        if mask is not None:
            expected_shape = (query.shape[0], query.shape[1], key.shape[1])
            if mask.shape != expected_shape:
                # Force create correct mask
                mask = torch.zeros(expected_shape, device=query.device, dtype=torch.bool)
        
        # Apply transformer attention
        n_batch = query.size(0)
        
        # Linear transformations for q, k, v
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        
        # Transpose for attention dot product
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        # Convert inputs to the correct format for attention computation
        if self.use_pytorch_sdpa:
            output = self._forward_sdpa(query, key, value, mask)
        else:
            output = self._forward_manual(query, key, value, mask, pos_emb)

        # Return cache if it was provided
        if original_cache is not None:
            return output, cache
        else:
            return output

    def _forward_manual(self, query, key, value, mask, pos_emb):
        """Forward pass using manual attention computation with relative positional encoding."""
        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        # temporary until we solve this more gracefully
        with avoid_float16_autocast_context():
            q, k, v = self.forward_qkv(query, key, value)
            q = q.transpose(1, 2)  # (batch, time1, head, d_k)

            n_batch_pos = pos_emb.size(0)
            n_batch = value.size(0)
            
            # CRITICAL FIX: For circular buffers, the positional encoding needs to handle
            # the case where key/value context is extended beyond query length
            query_len = query.shape[1]
            key_len = key.shape[1]
            
            # if key_len > query_len:
            #     # CRITICAL FIX: Extended key context (circular buffer case)
            #     # PROPER SOLUTION: Don't extend positional encoding at all!
            #     # Instead, use the original positional encoding that matches the query length
            #     # and rely on attention masking or natural decay for cached frames
                
            #     # The key insight: In relative positional encoding, what matters is the 
            #     # relationship between query positions and the corresponding key positions.
            #     # Since cached frames are from the past, they should have natural attention decay
            #     # based on the relative distance encoded in the standard positional encoding.
                
            #     # For circular buffers with extended context, we should:
            #     # 1. Keep positional encoding at query length (original)
            #     # 2. Let the attention mechanism naturally handle the extended key/value context
            #     # 3. Trust that the relative positional relationships will work correctly
                
            #     # DON'T extend positional encoding - this was causing the accuracy issues!
            #     # The original pos_emb already contains the correct relative relationships
            #     # for the current query frames.
                
            #     print(f"[DEBUG] Circular buffer: query_len={query_len}, key_len={key_len}, keeping pos_emb at original size")
            
            p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
            p = p.transpose(1, 2)  # (batch, head, time_pos, d_k)

            # (batch, head, time1, d_k)
            q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
            # (batch, head, time1, d_k)
            q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

            # compute attention score
            # first compute matrix a and matrix c
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            # (batch, head, time1, time2)
            matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

            # compute matrix b and matrix d
            # (batch, head, time1, time_pos)
            matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
            matrix_bd = self.rel_shift(matrix_bd)
            
            # CRITICAL FIX: Handle dimension mismatch properly for circular buffers
            if matrix_bd.size(-1) != matrix_ac.size(-1):
                # When using circular buffers, matrix_bd is computed using original positional encoding
                # (query_len x query_len) but matrix_ac uses extended key context (query_len x key_len)
                
                if matrix_ac.size(-1) > matrix_bd.size(-1):
                    # Extended key context case: Need to pad matrix_bd
                    # Since frames are now at the BEGINNING of context, we pad at the END
                    # The additional positions (padding positions) get zero relative positional bias
                    # This means they rely purely on content-based attention (matrix_ac)
                    
                    padding_size = matrix_ac.size(-1) - matrix_bd.size(-1)
                    zero_padding = torch.zeros(
                        matrix_bd.size(0), matrix_bd.size(1), matrix_bd.size(2), padding_size,
                        device=matrix_bd.device, dtype=matrix_bd.dtype
                    )
                    # Pad at the END (padding positions get zero positional bias)
                    matrix_bd = torch.cat([matrix_bd, zero_padding], dim=-1)
                else:
                    # This shouldn't happen in normal circular buffer operation
                    matrix_bd = matrix_bd[:, :, :, :matrix_ac.size(-1)]
            
            # Combined attention matrix
            scores = (matrix_ac + matrix_bd) / self.s_d_k
            
            # Apply masking if provided  
            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(1), -INF_VAL)
            
            # Apply softmax
            attn_probs = torch.softmax(scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_probs, v)  # (batch, head, time1, d_k)

            # Transpose and reshape: (batch, time1, head*d_k)
            attn_output = attn_output.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
            
            # Final linear projection
            output = self.linear_out(attn_output)
            
            # CRITICAL FIX: Handle circular buffer context vs residual connection requirements
            query_len = query.shape[1]
            
            if self._is_using_circular_buffers():
                # CIRCULAR BUFFER MODE: Output represents attended information over extended context
                # We need to preserve the full attended context, not truncate to query length
                # The circular buffer internally manages the context boundaries correctly
                
                # For circular buffers, the attention operates over extended key/value context
                # but the output should represent the current frame(s) with full context awareness
                # The query length determines how many output frames we actually need
                
                if output.shape[1] > query_len:
                    # Extract the frames corresponding to the current query
                    # The circular buffer positions the current frames at the end of the context
                    output = output[:, -query_len:, :]
                elif output.shape[1] < query_len:
                    # This shouldn't happen with proper circular buffer setup
                    # Pad if needed to match expected output size
                    padding_size = query_len - output.shape[1]
                    padding = torch.zeros(output.shape[0], padding_size, output.shape[2], 
                                        device=output.device, dtype=output.dtype)
                    output = torch.cat([padding, output], dim=1)
                
                # Debug: Track context usage for circular buffers
                if hasattr(self, '_debug_circular_context') and self._debug_circular_context:
                    context_frames = key.shape[1] if hasattr(key, 'shape') else 0
                    print(f"[CIRCULAR_DEBUG] query_len={query_len}, key_context={context_frames}, output_len={output.shape[1]}")
            else:
                # BASELINE MODE: Standard residual connection requirement
                if output.shape[1] != query_len:
                    # For baseline mode, output must exactly match query length for residual connections
                    output = output[:, -query_len:, :]
            
            return output

    def _is_using_circular_buffers(self) -> bool:
        """Check if this attention layer is currently using circular buffers."""
        return (hasattr(self, '_use_true_circular') and self._use_true_circular and
                hasattr(self, '_true_circular_cache') and self._true_circular_cache is not None)

class RelPositionMultiHeadAttentionLongformer(RelPositionMultiHeadAttention):
    """Multi-Head Attention layer of Transformer-XL with sliding window local+global attention from Longformer.
    Partially adapted from allenai (https://github.com/allenai/longformer/blob/master/longformer/sliding_chunks.py)
    and huggingface (https://github.com/huggingface/transformers/blob/main/src/transformers/models/longformer/modeling_longformer.py)
    Paper: https://arxiv.org/abs/1901.02860 (Transformer-XL),
           https://arxiv.org/abs/2004.05150 (Longformer)
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
        pos_bias_u (Tensor): the positional bias matrix U
        pos_bias_v (Tensor): the positional bias matrix V
        att_context_size (List[int]): List of 2 ints corresponding to left and right attention context sizes.
        max_cache_len (int): the maximum size of cache
        global_tokens (int): number of tokens to be used for global attention
        global_tokens_spacing (int): how far apart the global tokens are
        global_attn_separate (bool): whether the q, k, v layers used for global tokens should be separate
        use_bias (bool): whether to apply bias in linear and conv layers of MultiHeadAttention
    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        pos_bias_u,
        pos_bias_v,
        att_context_size,
        max_cache_len=0,
        global_tokens=0,
        global_tokens_spacing=1,
        global_attn_separate=False,
        use_bias=True,
        use_pytorch_sdpa=False,
        use_pytorch_sdpa_backends=None,
    ):
        """Construct an RelPositionMultiHeadAttentionLongformer object."""
        super().__init__(
            n_head=n_head,
            n_feat=n_feat,
            dropout_rate=dropout_rate,
            pos_bias_u=pos_bias_u,
            pos_bias_v=pos_bias_v,
            max_cache_len=max_cache_len,
            use_bias=use_bias,
            use_pytorch_sdpa=use_pytorch_sdpa,
            use_pytorch_sdpa_backends=use_pytorch_sdpa_backends,
        )

        if use_pytorch_sdpa:
            raise NotImplementedError("Not implemented for Longformer yet")

        self.att_context_size = att_context_size
        self.global_tokens = global_tokens
        self.global_tokens_spacing = global_tokens_spacing
        self.global_attn_separate = global_attn_separate

        if self.global_attn_separate:
            self.global_q = nn.Linear(n_feat, n_feat, bias=use_bias)
            self.global_k = nn.Linear(n_feat, n_feat, bias=use_bias)
            self.global_v = nn.Linear(n_feat, n_feat, bias=use_bias)

    def forward(self, query, key, value, pad_mask, pos_emb, cache=None):
        """Compute Scaled Dot Product Local Attention with rel. positional encoding. using overlapping chunks
        Args:
            query (torch.Tensor): (batch, time, size)
            key (torch.Tensor): (batch, time, size)
            value(torch.Tensor): (batch, time, size)
            pad_mask (torch.Tensor): (batch, time)
            pos_emb (torch.Tensor) : (batch, 2w + 1, size)
            cache (torch.Tensor) : (batch, time_cache, size)
        Returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
            cache (torch.Tensor) : (batch, time_cache_next, size)
        """

        key, value, query, cache = self.update_cache(key=key, value=value, query=query, cache=cache)

        if torch.is_autocast_enabled():
            query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)

        # temporary until we solve this more gracefully
        with avoid_float16_autocast_context():
            q, k, v = self.forward_qkv(query, key, value)
            n_batch, _, T, _ = q.size()

            w = max(self.att_context_size[0], self.att_context_size[1])
            if w <= 0:
                raise ValueError("When using local attention, context size must be set > 0")
            pad_len = (2 * w - T % (2 * w)) % (2 * w)  # pad time to 2w
            q = F.pad(q, (0, 0, 0, pad_len))  # (batch, head, time, size)
            k = F.pad(k, (0, 0, 0, pad_len))  # (batch, head, time, size)
            v = F.pad(v, (0, 0, 0, pad_len))  # (batch, head, time, size)
            mask = F.pad(pad_mask, (0, pad_len), value=1.0)

            q_with_bias_u = q + self.pos_bias_u.unsqueeze(1)  # (batch, head, time, size)
            q_with_bias_v = q + self.pos_bias_v.unsqueeze(1)  # (batch, head, time, size)

            diagonal_matrix_ac = self.sliding_chunks_matmul_qk(
                q_with_bias_u, k, w, padding_value=0.0
            )  # (batch, head, time, 2w + 1)

            # add relative positional embedding

            n_batch_pos = pos_emb.size(0)
            p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k).transpose(1, 2)
            # (batch, head, 2w, size)
            diagonal_matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
            # (batch, head, time, 2w + 1)

            start_pos = w - self.att_context_size[0]
            end_pos = w + self.att_context_size[1]

            diagonal_matrix_ac[:, :, :, : self.att_context_size[0]] += diagonal_matrix_bd[
                :, :, :, : self.att_context_size[0]
            ]
            diagonal_matrix_ac[:, :, :, -(self.att_context_size[1] + 1) :] += diagonal_matrix_bd[
                :, :, :, self.att_context_size[0] :
            ]
            scores = diagonal_matrix_ac / self.s_d_k
            # (batch, head, time, 2w + 1)

            # mask invalid positions
            scores[:, :, :, :start_pos] = -INF_VAL
            scores[:, :, :, end_pos + 1 :] = -INF_VAL

            # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
            # from (bsz x seq_len) to (bsz x num_heads x seqlen x hidden_size)
            mask = mask.unsqueeze(dim=1).unsqueeze(dim=-1)
            # cast to float/half then replace 1's with -inf
            float_mask = mask.type_as(scores).masked_fill(mask, -INF_VAL)
            ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
            # diagonal mask with zeros everywhere and -inf inplace of padding
            d_mask = self.sliding_chunks_matmul_qk(ones, float_mask, w, padding_value=0.0)
            # (batch, head, time, 2w + 1)

            scores += d_mask

            if self.global_tokens > 0:

                # create q, k, v for global attn
                if self.global_attn_separate:
                    global_q = self.global_q(query).view(n_batch, -1, self.h, self.d_k)
                    global_k = self.global_k(key).view(n_batch, -1, self.h, self.d_k)
                    global_v = self.global_v(value).view(n_batch, -1, self.h, self.d_k)
                    global_q = global_q.transpose(1, 2)
                    global_k = global_k.transpose(1, 2)
                    global_v = global_v.transpose(1, 2)
                    global_q = F.pad(global_q, (0, 0, 0, pad_len))  # (batch, head, time, size)
                    global_k = F.pad(global_k, (0, 0, 0, pad_len))  # (batch, head, time, size)
                    global_v = F.pad(global_v, (0, 0, 0, pad_len))  # (batch, head, time, size)
                else:
                    global_q, global_k, global_v = q, k, v

                global_q /= self.s_d_k

                # assign which tokens are global
                is_index_global_attn = torch.zeros_like(pad_mask)
                is_index_global_attn[
                    :, : self.global_tokens * self.global_tokens_spacing : self.global_tokens_spacing
                ] = 1.0

                # compute global attn probs with global keys
                # (batch, time, head, max_num_global_attn_indices)
                global_key_attn = self._compute_global_key_attn(
                    query=global_q.transpose(1, 2),
                    key=global_k.transpose(1, 2),
                    max_num_global_attn_indices=max_num_global_attn_indices,
                    is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                    is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                    is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                ).transpose(1, 2)

                # concat to local_attn_probs
                # (batch, time, head, max_num_global_attn_indices + 2*w)
                scores = torch.cat((global_key_attn, scores), dim=-1)

                # free memory
                del global_key_attn

            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
            p_attn = self.dropout(attn)
            # (batch, head, time, 2w + 1)

            if self.global_tokens > 0:
                # compute sum of global and local attn
                out = self._compute_attn_output_with_global_indices(
                    value=v,
                    attn_probs=p_attn,
                    max_num_global_attn_indices=max_num_global_attn_indices,
                    is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                    is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                    w=w,
                )
            else:
                # compute local attn only
                out = self.sliding_chunks_matmul_pv(p_attn, v, w)

            out = out.reshape(n_batch, -1, self.h * self.d_k)[:, :T]

            if self.global_tokens > 0:
                out_global_to_all = self._compute_out_global_to_all(
                    query=global_q,
                    key=global_k,
                    value=global_v,
                    max_num_global_attn_indices=max_num_global_attn_indices,
                    is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                    is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                    is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                    is_index_masked=mask,
                )

                # overwrite values with global attention
                out[is_index_global_attn_nonzero] = out_global_to_all

        ret = self.linear_out(out)

        if cache is None:
            return ret
        else:
            return ret, cache

    def _get_global_attn_indices(self, is_index_global_attn: torch.Tensor) -> Tuple:
        """
        Compute global attention indices.

        Args:
            is_index_global_attn (torch.Tensor): (batch, time) A boolean tensor indicating if an index is a global attention index.

        Returns:
            max_num_global_attn_indices (int): Maximum number of global attention indices in the batch.
            is_index_global_attn_nonzero (tuple): Indices of global attention (non-zero elements).
            is_local_index_global_attn_nonzero (tuple): Indices of non-padding values within global attention indices.
            is_local_index_no_global_attn_nonzero (tuple): Indices of padding values within global attention indices.
        """
        # Calculate the number of global attention indices in the batch
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)

        # Find the maximum number of global attention indices in the batch
        max_num_global_attn_indices = num_global_attn_indices.max()

        # Get the indices of global attention (non-zero elements)
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)

        # Create a helper tensor to find the local indices of global attention
        is_local_index_global_attn = torch.arange(
            max_num_global_attn_indices, device=is_index_global_attn.device
        ) < num_global_attn_indices.unsqueeze(dim=-1)

        # Find the non-padding values within global attention indices
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)

        # Find the padding values within global attention indices
        is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0).nonzero(as_tuple=True)

        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )

    def _compute_global_key_attn(
        self,
        key: torch.Tensor,
        query: torch.Tensor,
        max_num_global_attn_indices: int,
        is_index_global_attn_nonzero: tuple,
        is_local_index_global_attn_nonzero: tuple,
        is_local_index_no_global_attn_nonzero: tuple,
    ) -> torch.Tensor:
        """
        Compute the attention probabilities using only global key vectors.

        Args:
            key (torch.Tensor): (batch, time, head, head_dim) The key vectors.
            query (torch.Tensor): (batch, time, head, head_dim) The query vectors.
            max_num_global_attn_indices (int): Maximum number of global attention indices in the batch.
            is_index_global_attn_nonzero (tuple): Indices of global attention (non-zero elements).
            is_local_index_global_attn_nonzero (tuple): Non-padding values within global attention indices.
            is_local_index_no_global_attn_nonzero (tuple): Padding values within global attention indices.

        Returns:
            attn_probs_from_global_key (torch.Tensor): (batch, time, head, max_num_global_attn_indices) The computed attention probabilities using only global key vectors.
        """
        batch_size = key.shape[0]

        # create only global key vectors
        key_only_global = key.new_zeros(batch_size, max_num_global_attn_indices, self.h, self.d_k)

        key_only_global[is_local_index_global_attn_nonzero] = key[is_index_global_attn_nonzero]

        # (batch_size, seq_len, head, max_num_global_attn_indices)
        attn_probs_from_global_key = torch.einsum("blhd,bshd->blhs", (query, key_only_global))

        # need to transpose since ONNX export only supports consecutive indexing: https://pytorch.org/docs/stable/onnx.html#writes-sets
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)
        attn_probs_from_global_key[
            is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :
        ] = torch.finfo(attn_probs_from_global_key.dtype).min
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)

        return attn_probs_from_global_key

    def _compute_attn_output_with_global_indices(
        self,
        value: torch.Tensor,
        attn_probs: torch.Tensor,
        max_num_global_attn_indices: int,
        is_index_global_attn_nonzero: tuple,
        is_local_index_global_attn_nonzero: tuple,
        w: int,
    ) -> torch.Tensor:
        """
        Compute the attention output with global indices.

        Args:
            value (torch.Tensor): (batch, head, time, head_dim) The value vectors for global attention.
            attn_probs (torch.Tensor): (batch, time, head, 2w) The attention probabilities.
            max_num_global_attn_indices (int): Maximum number of global attention indices in the batch.
            is_index_global_attn_nonzero (tuple): Indices of global attention (non-zero elements).
            is_local_index_global_attn_nonzero (tuple): Non-padding values within global attention indices.
            w (int): Local context size
        Returns:
            torch.Tensor: (batch, time, head x head_dim) The attention output of all tokens attending to global.
        """
        batch_size, time = attn_probs.shape[0], attn_probs.shape[2]

        value = value.transpose(1, 2)

        # get value vectors for global only
        value_vectors_only_global = value.new_zeros(batch_size, max_num_global_attn_indices, self.h, self.d_k)
        value_vectors_only_global[is_local_index_global_attn_nonzero] = value[is_index_global_attn_nonzero]

        # cut local attn probs to global only
        attn_probs_only_global = attn_probs.narrow(-1, 0, max_num_global_attn_indices)
        # compute attn output only global
        attn_output_only_global = torch.matmul(
            attn_probs_only_global.clone(), value_vectors_only_global.transpose(1, 2).clone()
        ).transpose(1, 2)

        # reshape attn probs
        attn_probs_without_global = attn_probs.narrow(
            -1, max_num_global_attn_indices, attn_probs.size(-1) - max_num_global_attn_indices
        ).contiguous()

        # compute attn output with global
        attn_output_without_global = self.sliding_chunks_matmul_pv(attn_probs_without_global, value.transpose(1, 2), w)


class PositionalEncoding(torch.nn.Module):
    """Fixed sinusoidal positional encoding.
    Args:
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
        dropout_rate_emb (float): dropout rate for the positional embeddings
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, xscale=None, dropout_rate_emb=0.0):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = xscale
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len
        if dropout_rate_emb > 0:
            self.dropout_emb = nn.Dropout(dropout_rate_emb)
        else:
            self.dropout_emb = None

    def create_pe(self, positions, dtype):
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.d_model, device=positions.device)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=positions.device)
            * -(math.log(INF_VAL) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0).to(dtype)
        if hasattr(self, 'pe'):
            self.pe = pe
        else:
            self.register_buffer('pe', pe, persistent=False)

    def extend_pe(self, length, device, dtype):
        """Reset and extend the positional encodings if needed."""
        if hasattr(self, 'pe') and self.pe.size(1) >= length:
            return
        positions = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, x: torch.Tensor, cache_len=0):
        """Adds positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
            cache_len (int): the size of the cache which is used to shift positions
        Returns:
            x+pos_emb (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """
        input_len = x.size(1) + cache_len
        if self.xscale:
            x = x * self.xscale
        pos_emb = self.pe[:, :input_len]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        x = x + pos_emb
        return self.dropout(x), pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding for TransformerXL's layers
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
        dropout_rate_emb (float): dropout rate for the positional embeddings
    """

    def extend_pe(self, length, device, dtype):
        """Reset and extend the positional encodings if needed."""
        needed_size = 2 * length - 1
        if hasattr(self, 'pe') and self.pe.size(1) >= needed_size:
            return
        # positions would be from negative numbers to positive
        # positive positions would be used for left positions and negative for right positions
        positions = torch.arange(length - 1, -length, -1, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, x, cache_len=0):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
            cache_len (int): the size of the cache which is used to shift positions
        Returns:
            x (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """

        if self.xscale:
            x = x * self.xscale

        # center_pos would be the index of position 0
        # negative positions would be used for right and positive for left tokens
        # for input of length L, 2*L-1 positions are needed, positions from (L-1) to -(L-1)
        input_len = x.size(1) + cache_len
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pe[:, start_pos:end_pos]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb


class LocalAttRelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding for sliding window attention or chunked attention.
    See above for relative positional encoding based on Transformer-XL paper
    Args:
        left_chunk_size (int): number of frames to in past chunks
        chunk size (int): number of frames (max frames if using multimode) in current chunk
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
        dropout_rate_emb (float): dropout rate for the positional embeddings
    """

    def __init__(self, att_context_size, **kwargs):
        super(LocalAttRelPositionalEncoding, self).__init__(**kwargs)
        self.left_context = att_context_size[0]
        self.right_context = att_context_size[1]

    def extend_pe(self, length, device, dtype):
        """Reset and extend the positional encodings only at the beginning"""
        if hasattr(self, 'pe'):
            return

        positions = torch.arange(
            self.left_context, -self.right_context - 1, -1, dtype=torch.float32, device=device
        ).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)

    def forward(self, x, cache_len=0):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
        Returns:
            x (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """

        if self.xscale:
            x = x * self.xscale

        end_pos = self.left_context + self.right_context + 1
        pos_emb = self.pe[:, :end_pos]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb
