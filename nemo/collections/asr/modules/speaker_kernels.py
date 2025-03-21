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

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SpeakerMask', 'SpeakerConcat']

class SpeakerMask(torch.nn.Module):
    def __init__(self, input_size, output_size, mask_original=True, residual=True):
        super().__init__()
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(input_size, output_size*2),
            torch.nn.ReLU(),
            torch.nn.Linear(output_size*2, output_size)
        )
        self.mask_original = mask_original
        self.residual = residual
        
    def forward(self, x, mask):
        """
        x: (B, T, D)
        mask: (B, T)
        """
        if mask.shape[1] < x.shape[1]:
            mask = F.pad(mask, (0, x.shape[1] - mask.shape[1]), mode='replicate')

        if mask.shape[1] > x.shape[1]:
            mask = mask[:, -x.shape[1]:]

        x_masked = x * mask.unsqueeze(2)
        if self.residual:
            if self.mask_original:
                print("mask_original and residual")
                x = x_masked + self.feedforward(x_masked)
            else:
                print("residual")
                x = x + self.feedforward(x_masked)
        else:
            print("no residual")
            x = self.feedforward(x_masked)

        return x
    
class SpeakerConcat(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(input_size+1, output_size*2),
            torch.nn.ReLU(),
            torch.nn.Linear(output_size*2, output_size)
        )

    def forward(self, x, mask):
        """
        x: (B, T, D)
        mask: (B, T)
        """
        if mask.shape[1] < x.shape[1]:
            mask = F.pad(mask, (0, x.shape[1] - mask.shape[1]), mode='replicate')

        if mask.shape[1] > x.shape[1]:
            mask = mask[:, -x.shape[1]:]

        x_cat = torch.cat([x, mask.unsqueeze(2)], dim=2)
        x = self.feedforward(x_cat)

        return x
