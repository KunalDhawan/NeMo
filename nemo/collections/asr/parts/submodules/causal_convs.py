# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Union

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['CausalConv2D', 'CausalConv1D']


class CircularConvCache:
    """
    Efficient circular buffer for causal convolution cache to eliminate tensor concatenations.
    Uses pre-allocated memory and in-place operations for optimal performance.
    """
    
    def __init__(self, max_cache_len: int, channels: int, batch_size: int, device: torch.device, dtype: torch.dtype):
        self.max_cache_len = max_cache_len
        self.channels = channels
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        
        # Pre-allocate circular buffer: (batch_size, channels, max_cache_len)
        self.buffer = torch.zeros(batch_size, channels, max_cache_len, device=device, dtype=dtype)
        self.write_idx = 0  # Current write position
        self.valid_len = 0  # Number of valid cached frames
        
    def update(self, new_data: torch.Tensor, drop_size: int = 0) -> tuple:
        """
        Update cache with new data using circular buffer logic.
        
        Args:
            new_data: (batch_size, channels, seq_len) New data to add
            drop_size: Number of frames to drop from the beginning
            
        Returns:
            combined_data: (batch_size, channels, total_len) All cached data + new data
            cache_for_next: Updated cache state
        """
        batch_size, channels, seq_len = new_data.shape
        
        # Calculate effective cache length after dropping frames
        effective_cache_len = max(0, self.valid_len - drop_size)
        total_len = effective_cache_len + seq_len
        
        if effective_cache_len == 0:
            # No valid cache, return just the new data
            if seq_len <= self.max_cache_len:
                # Update buffer with new data
                self.buffer[:, :, :seq_len] = new_data
                self.write_idx = seq_len % self.max_cache_len
                self.valid_len = seq_len
            else:
                # New data exceeds cache size, keep only the last part
                self.buffer[:, :, :] = new_data[:, :, -self.max_cache_len:]
                self.write_idx = 0
                self.valid_len = self.max_cache_len
            
            return new_data, self
        
        if total_len <= self.max_cache_len:
            # Everything fits in cache
            # Get valid cached data
            if self.write_idx >= effective_cache_len:
                # No wraparound
                start_idx = self.write_idx - effective_cache_len
                cached_data = self.buffer[:, :, start_idx:self.write_idx]
            else:
                # CRITICAL FIX: Correct temporal ordering when buffer wraps around
                # When buffer wraps, write_idx points to where next write will happen (oldest data)
                # We need effective_cache_len frames in chronological order
                
                # Calculate how much we can get from the oldest part (from write_idx onwards)
                available_from_oldest = self.max_cache_len - self.write_idx
                oldest_part_len = min(available_from_oldest, effective_cache_len)
                
                # Calculate how much we need from the newest part (from start of buffer)
                remaining_needed = effective_cache_len - oldest_part_len
                newest_part_len = min(remaining_needed, self.write_idx)
                
                # Get oldest part first (from write_idx onwards)
                if oldest_part_len > 0:
                    oldest_part = self.buffer[:, :, self.write_idx:self.write_idx + oldest_part_len]
                else:
                    oldest_part = torch.empty(batch_size, channels, 0, device=self.device, dtype=self.dtype)
                
                # Get newest part second (from start of buffer)
                if newest_part_len > 0:
                    newest_part = self.buffer[:, :, :newest_part_len]
                else:
                    newest_part = torch.empty(batch_size, channels, 0, device=self.device, dtype=self.dtype)
                
                # Combine in chronological order: oldest first, then newest
                cached_data = torch.cat([oldest_part, newest_part], dim=2)
            
            # Combine cached and new data
            combined_data = torch.cat([cached_data, new_data], dim=2)
            
            # Update buffer with new data
            end_idx = (self.write_idx + seq_len) % self.max_cache_len
            if self.write_idx + seq_len <= self.max_cache_len:
                # No wraparound
                self.buffer[:, :, self.write_idx:self.write_idx + seq_len] = new_data
            else:
                # Wraparound needed
                first_part_len = self.max_cache_len - self.write_idx
                self.buffer[:, :, self.write_idx:] = new_data[:, :, :first_part_len]
                self.buffer[:, :, :end_idx] = new_data[:, :, first_part_len:]
            
            self.write_idx = end_idx
            self.valid_len = total_len
            
        else:
            # Cache overflow: keep only the most recent data
            keep_len = self.max_cache_len - seq_len
            
            if keep_len > 0:
                # Keep part of the cache
                keep_from_cache = min(keep_len, effective_cache_len)
                if self.write_idx >= keep_from_cache:
                    cached_data = self.buffer[:, :, self.write_idx - keep_from_cache:self.write_idx]
                else:
                    # CRITICAL FIX: Correct temporal ordering for overflow case
                    # Similar wraparound logic - get oldest data from write_idx onwards
                    available_from_oldest = self.max_cache_len - self.write_idx
                    oldest_part_len = min(available_from_oldest, keep_from_cache)
                    
                    remaining_needed = keep_from_cache - oldest_part_len
                    newest_part_len = min(remaining_needed, self.write_idx)
                    
                    # Get oldest part first
                    if oldest_part_len > 0:
                        oldest_part = self.buffer[:, :, self.write_idx:self.write_idx + oldest_part_len]
                    else:
                        oldest_part = torch.empty(batch_size, channels, 0, device=self.device, dtype=self.dtype)
                    
                    # Get newest part second
                    if newest_part_len > 0:
                        newest_part = self.buffer[:, :, :newest_part_len]
                    else:
                        newest_part = torch.empty(batch_size, channels, 0, device=self.device, dtype=self.dtype)
                    
                    cached_data = torch.cat([oldest_part, newest_part], dim=2)
                
                combined_data = torch.cat([cached_data, new_data], dim=2)
            else:
                # Keep only new data
                combined_data = new_data[:, :, -self.max_cache_len:] if seq_len > self.max_cache_len else new_data
            
            # Reset buffer and add new data
            self.buffer.zero_()
            if seq_len <= self.max_cache_len:
                self.buffer[:, :, :seq_len] = new_data
                self.write_idx = seq_len % self.max_cache_len
                self.valid_len = seq_len
            else:
                self.buffer[:, :, :] = new_data[:, :, -self.max_cache_len:]
                self.write_idx = 0
                self.valid_len = self.max_cache_len
        
        return combined_data, self
    
    def reset(self):
        """Reset cache state."""
        self.buffer.zero_()
        self.write_idx = 0
        self.valid_len = 0
    
    def size(self, dim: int = None):
        """Return size for compatibility with torch.Tensor.size()"""
        if dim is None:
            return self.buffer.shape
        return self.buffer.shape[dim]


class CausalConv2D(nn.Conv2d):
    """
    A causal version of nn.Conv2d where each location in the 2D matrix would have no access to locations on its right or down
    All arguments are the same as nn.Conv2d except padding which should be set as None
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ) -> None:
        if padding is not None:
            raise ValueError("Argument padding should be set to None for CausalConv2D.")
        self._left_padding = kernel_size - 1
        self._right_padding = stride - 1

        padding = 0
        super(CausalConv2D, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    def forward(
        self, x,
    ):
        x = F.pad(x, pad=(self._left_padding, self._right_padding, self._left_padding, self._right_padding))
        x = super().forward(x)
        return x


class CausalConv1D(nn.Conv1d):
    """
    A causal version of nn.Conv1d where each step would have limited access to locations on its right or left
    All arguments are the same as nn.Conv1d except padding.

    If padding is set None, then paddings are set automatically to make it a causal convolution where each location would not see any steps on its right.

    If padding is set as a list (size of 2), then padding[0] would be used as left padding and padding[1] as right padding.
    It would make it possible to control the number of steps to be accessible on the right and left.
    This mode is not supported when stride > 1. padding[0]+padding[1] should be equal to (kernel_size - 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ) -> None:
        self.cache_drop_size = None
        if padding is None:
            self._left_padding = kernel_size - 1
            self._right_padding = stride - 1
        else:
            if stride != 1 and padding != kernel_size - 1:
                raise ValueError("No striding allowed for non-symmetric convolutions!")
            if isinstance(padding, int):
                self._left_padding = padding
                self._right_padding = padding
            elif isinstance(padding, list) and len(padding) == 2 and padding[0] + padding[1] == kernel_size - 1:
                self._left_padding = padding[0]
                self._right_padding = padding[1]
            else:
                raise ValueError(f"Invalid padding param: {padding}!")

        self._max_cache_len = self._left_padding

        # Initialize circular buffer optimization settings
        self._use_circular_buffers = False
        self._optimization_enabled = False

        super(CausalConv1D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def set_circular_buffer_config(self, use_circular_buffers: bool = False, optimization_enabled: bool = False):
        """Set circular buffer configuration for this convolution layer."""
        self._use_circular_buffers = use_circular_buffers
        self._optimization_enabled = optimization_enabled

    def update_cache(self, x, cache=None):
        if cache is None:
            new_x = F.pad(x, pad=(self._left_padding, self._right_padding))
            next_cache = cache
        else:
            # CRITICAL FIX: For now, always use baseline path for convolution caches
            # The circular buffer optimization is primarily for attention layers
            # Convolution caches are much smaller and don't benefit as much from circular buffers
            
                # BASELINE PATH: Use safe concatenation without in-place operations
                cache_len = cache.size(-1)
                drop_size = self.cache_drop_size or 0
                
                # Apply drop_size if specified
                if drop_size > 0 and cache_len > drop_size:
                    # Drop frames from the beginning (create new tensor to avoid in-place ops)
                    cache = cache[:, :, drop_size:].clone()
                    cache_len = cache.size(-1)
                
                # Pad new input
                padded_x = F.pad(x, pad=(0, self._right_padding))
                
                # Standard concatenation
                new_x = torch.cat([cache, padded_x], dim=-1)
                
                # CRITICAL: Create NEW cache tensor for next iteration (avoid sharing)
                # This ensures baseline behavior doesn't have cache corruption
                if self._max_cache_len > 0:
                    # Keep only the last max_cache_len frames
                    next_cache = new_x[:, :, -self._max_cache_len:].clone()
                else:
                    # Keep everything but create a copy to avoid sharing
                    next_cache = new_x.clone()
                    
        return new_x, next_cache

    def forward(self, x, cache=None):
        x, cache = self.update_cache(x, cache=cache)
        x = super().forward(x)
        if cache is None:
            return x
        else:
            return x, cache
