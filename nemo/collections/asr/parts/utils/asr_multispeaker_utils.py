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
from typing import Optional, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from lhotse import SupervisionSet
from lhotse.cut import MixedCut, MonoCut
from scipy.optimize import linear_sum_assignment
from nemo.utils import logging

def find_first_nonzero(mat: torch.Tensor, max_cap_val=-1, thres: float = 0.5) -> torch.Tensor:
    """
    Finds the first nonzero value in the matrix, discretizing it to the specified maximum capacity.

    Args:
        mat (Tensor): A torch tensor representing the matrix.
        max_cap_val (int): The maximum capacity to which the matrix values will be discretized.
        thres (float): The threshold value for discretizing the matrix values.

    Returns:
        mask_max_indices (Tensor): A torch tensor representing the discretized matrix with the first
        nonzero value in each row.
    """
    # Discretize the matrix to the specified maximum capacity
    labels_discrete = mat.clone()
    labels_discrete[labels_discrete < thres] = 0
    labels_discrete[labels_discrete >= thres] = 1

    # non zero values mask
    non_zero_mask = labels_discrete != 0
    # operations on the mask to find first nonzero values in the rows
    mask_max_values, mask_max_indices = torch.max(non_zero_mask, dim=1)
    # if the max-mask is zero, there is no nonzero value in the row
    mask_max_indices[mask_max_values == 0] = max_cap_val
    return mask_max_indices


def find_best_permutation(match_score: torch.Tensor, speaker_permutations: torch.Tensor) -> torch.Tensor:
    """
    Finds the best permutation indices based on the match score.

    Args:
        match_score (torch.Tensor): A tensor containing the match scores for each permutation.
            Shape: (batch_size, num_permutations)
        speaker_permutations (torch.Tensor): A tensor containing all possible speaker permutations.
            Shape: (num_permutations, num_speakers)

    Returns:
        torch.Tensor: A tensor containing the best permutation indices for each batch.
            Shape: (batch_size, num_speakers)
    """
    batch_best_perm = torch.argmax(match_score, axis=1)
    rep_speaker_permutations = speaker_permutations.repeat(batch_best_perm.shape[0], 1).to(match_score.device)
    perm_size = speaker_permutations.shape[0]
    global_inds_vec = (
        torch.arange(0, perm_size * batch_best_perm.shape[0], perm_size).to(batch_best_perm.device) + batch_best_perm
    )
    return rep_speaker_permutations[global_inds_vec.to(rep_speaker_permutations.device), :]


def reconstruct_labels(labels: torch.Tensor, batch_perm_inds: torch.Tensor) -> torch.Tensor:
    """
    Reconstructs the labels using the best permutation indices with matrix operations.

    Args:
        labels (torch.Tensor): A tensor containing the original labels.
            Shape: (batch_size, num_frames, num_speakers)
        batch_perm_inds (torch.Tensor): A tensor containing the best permutation indices for each batch.
            Shape: (batch_size, num_speakers)

    Returns:
        torch.Tensor: A tensor containing the reconstructed labels using the best permutation indices.
            Shape: (batch_size, num_frames, num_speakers)
    """
    # Expanding batch_perm_inds to align with labels dimensions
    batch_size, num_frames, num_speakers = labels.shape
    batch_perm_inds_exp = batch_perm_inds.unsqueeze(1).expand(-1, num_frames, -1)

    # Reconstructing the labels using advanced indexing
    reconstructed_labels = torch.gather(labels, 2, batch_perm_inds_exp)
    return reconstructed_labels


def get_ats_targets(
    labels: torch.Tensor,
    preds: torch.Tensor,
    speaker_permutations: torch.Tensor,
    thres: float = 0.5,
    tolerance: float = 0,
) -> torch.Tensor:
    """
    Sorts labels and predictions to get the optimal of all arrival-time ordered permutations.

    Args:
        labels (torch.Tensor): A tensor containing the original labels.
            Shape: (batch_size, num_frames, num_speakers)
        preds (torch.Tensor): A tensor containing the predictions.
            Shape: (batch_size, num_frames, num_speakers)
        speaker_permutations (torch.Tensor): A tensor containing all possible speaker permutations.
            Shape: (num_permutations, num_speakers)
        thres (float): The threshold value for discretizing the matrix values. Default is 0.5.
        tolerance (float): The tolerance for comparing the first speech frame indices. Default is 0.

    Returns:
        torch.Tensor: A tensor containing the reconstructed labels using the best permutation indices.
            Shape: (batch_size, num_frames, num_speakers)
    """
    # Find the first nonzero frame index for each speaker in each batch
    nonzero_ind = find_first_nonzero(
        mat=labels, max_cap_val=labels.shape[1], thres=thres
    )  # (batch_size, num_speakers)

    # Sort the first nonzero frame indices for arrival-time ordering
    sorted_values = torch.sort(nonzero_ind)[0]  # (batch_size, num_speakers)
    perm_size = speaker_permutations.shape[0]  # Scalar value (num_permutations)
    permed_labels = labels[:, :, speaker_permutations]  # (batch_size, num_frames, num_permutations, num_speakers)
    permed_nonzero_ind = find_first_nonzero(
        mat=permed_labels, max_cap_val=labels.shape[1]
    )  # (batch_size, num_permutations, num_speakers)

    # Compare the first frame indices of sorted labels with those of the permuted labels using tolerance
    perm_compare = (
        torch.abs(sorted_values.unsqueeze(1) - permed_nonzero_ind) <= tolerance
    )  # (batch_size, num_permutations, num_speakers)
    perm_mask = torch.all(perm_compare, dim=2).float()  # (batch_size, num_permutations)
    preds_rep = torch.unsqueeze(preds, 2).repeat(
        1, 1, perm_size, 1
    )  # Exapnd the preds: (batch_size, num_frames, num_permutations, num_speakers)

    # Compute the match score for each permutation by comparing permuted labels with preds
    match_score = (
        torch.sum(permed_labels * preds_rep, axis=1).sum(axis=2) * perm_mask
    )  # (batch_size, num_permutations)
    batch_perm_inds = find_best_permutation(match_score, speaker_permutations)  # (batch_size, num_speakers)
    max_score_permed_labels = reconstruct_labels(labels, batch_perm_inds)  # (batch_size, num_frames, num_speakers)
    return max_score_permed_labels  # (batch_size, num_frames, num_speakers)


def get_ats_targets_streaming(
    labels: torch.Tensor,
    preds: torch.Tensor,
    speaker_permutations: torch.Tensor,
    spkcache_len: int,
    spkcache_frame_indices: torch.Tensor = None,
    spk_perm: torch.Tensor = None,
    thres: float = 0.5,
    tolerance: float = 0,
) -> torch.Tensor:
    """
    Streaming-aware ATS target computation.

    When spkcache_frame_indices is provided and the spkcache has been compressed
    (indicated by presence of -1 silence markers), uses block-position-based pinning:

      Step 1: Detect speaker blocks in the compressed spkcache by finding contiguous
              runs of valid frames (>= 0) separated by silence markers (-1). For each
              block, average the ground-truth labels across its frames and take argmax
              to identify the speaker. Pin identified speakers to the canonical column
              corresponding to their block position. When spk_perm is active, block k
              maps to canonical column spk_perm[k] (because inv_spk_perm routes
              permuted column k to canonical column spk_perm[k]). Without spk_perm,
              block k maps to canonical column k (identity).
      Step 2: Run standard ATS logic with the overridden nonzero_ind. Speakers not
              identified in spkcache blocks keep their natural first-nonzero frame index
              from the full context and get assigned in arrival-time order.

    When spkcache_frame_indices is not provided (e.g., CLS model), falls back to
    pred-based content matching: binarized preds and labels on the spkcache portion
    identify which pred column corresponds to which speaker.

    Args:
        labels (torch.Tensor): Context labels (canonical speaker columns).
            Shape: (batch_size, context_len, num_speakers)
        preds (torch.Tensor): Context predictions (canonical ordering, after inv_spk_perm).
            Shape: (batch_size, context_len, num_speakers)
        speaker_permutations (torch.Tensor): All possible speaker permutations.
            Shape: (num_permutations, num_speakers)
        spkcache_len (int): Number of frames in the spkcache portion of the context.
        spkcache_frame_indices (torch.Tensor, optional): Frame indices for the spkcache
            portion. Shape: (batch_size, spkcache_len), long. -1 for silence/disabled
            frames. When provided and contains -1 markers (compressed), block-based
            detection is used. Default: None.
        spk_perm (torch.Tensor, optional): Speaker permutation from spkcache compression.
            Shape: (batch_size, num_speakers). Maps canonical index to permuted block
            position. Used to translate block positions to canonical column positions.
            None when permute_spk is inactive (eval mode). Default: None.
        thres (float): Threshold for binarizing/identifying speakers. Default 0.5.
        tolerance (float): Tolerance for comparing first speech frame indices. Default 0.

    Returns:
        torch.Tensor: Reconstructed labels using the best permutation.
            Shape: (batch_size, context_len, num_speakers)
    """
    batch_size, context_len, num_speakers = labels.shape

    # --- Step 1: speaker identification and pinning ---
    nonzero_ind = find_first_nonzero(
        mat=labels, max_cap_val=context_len, thres=thres
    )  # (B, S)

    if spkcache_len > 0 and spkcache_frame_indices is not None:
        # Block-position-based pinning: detect speaker blocks in compressed spkcache
        # using ground-truth labels (invariant to prediction errors and block ordering).
        labels_spk = labels[:, :spkcache_len, :]  # (B, spkcache_len, S)

        for b in range(batch_size):
            fi = spkcache_frame_indices[b]  # (spkcache_len,)

            # Only apply block detection if spkcache has been compressed
            # (compression introduces silence separators marked as -1)
            if not (fi < 0).any():
                continue

            # Detect contiguous speech blocks separated by silence
            blocks = []
            in_block = False
            block_start = 0
            for i in range(spkcache_len):
                val = fi[i].item()
                if val >= 0 and not in_block:
                    block_start = i
                    in_block = True
                elif val < 0 and in_block:
                    blocks.append((block_start, i))
                    in_block = False
            if in_block:
                blocks.append((block_start, spkcache_len))

            # For each block, identify speaker via ground-truth label mean + argmax
            assigned_targets = set()
            for block_idx, (bs, be) in enumerate(blocks):
                if block_idx >= num_speakers:
                    break
                block_labels = labels_spk[b, bs:be, :]  # (block_len, S)
                if block_labels.shape[0] == 0:
                    continue
                mean_labels = block_labels.mean(dim=0)  # (S,)
                if mean_labels.max() < thres:
                    continue  # No identifiable speaker in this block

                # Assign best unassigned speaker to this block's canonical column.
                # Block k -> permuted column k -> canonical column spk_perm[k].
                # Without spk_perm (eval), block k = canonical column k.
                canonical_pos = spk_perm[b, block_idx].item() if spk_perm is not None else block_idx
                sorted_spks = torch.argsort(mean_labels, descending=True)
                for spk_idx in sorted_spks:
                    spk = spk_idx.item()
                    if spk not in assigned_targets:
                        nonzero_ind[b, spk] = -num_speakers + canonical_pos
                        assigned_targets.add(spk)
                        break

    elif spkcache_len > 0:
        # Fallback: pred-based content matching (for models not providing frame indices)
        labels_spk = labels[:, :spkcache_len, :]
        preds_spk = preds[:, :spkcache_len, :]
        labels_bin = (labels_spk >= thres).float()
        preds_bin = (preds_spk >= thres).float()
        match_matrix = torch.bmm(preds_bin.transpose(1, 2), labels_bin)  # (B, S, S)

        for b in range(batch_size):
            assigned_targets = set()
            assigned_preds = set()
            mm = match_matrix[b]  # (S, S)
            flat_order = torch.argsort(mm.flatten(), descending=True)
            for flat_idx in flat_order:
                pred_col = (flat_idx // num_speakers).item()
                target_col = (flat_idx % num_speakers).item()
                if mm[pred_col, target_col] <= 0:
                    break
                if pred_col in assigned_preds or target_col in assigned_targets:
                    continue
                nonzero_ind[b, target_col] = -num_speakers + pred_col
                assigned_targets.add(target_col)
                assigned_preds.add(pred_col)

    # --- Step 2: standard ATS logic with overridden nonzero_ind ---
    sorted_values = torch.sort(nonzero_ind)[0]  # (B, S)
    perm_size = speaker_permutations.shape[0]
    permed_labels = labels[:, :, speaker_permutations]  # (B, context_len, P, S)
    permed_nonzero_ind = nonzero_ind[:, speaker_permutations]  # (B, P, S)

    perm_compare = (
        torch.abs(sorted_values.unsqueeze(1) - permed_nonzero_ind) <= tolerance
    )  # (B, P, S)
    perm_mask = torch.all(perm_compare, dim=2).float()  # (B, P)
    preds_rep = preds.unsqueeze(2).repeat(1, 1, perm_size, 1)  # (B, context_len, P, S)

    match_score = (
        torch.sum(permed_labels * preds_rep, axis=1).sum(axis=2) * perm_mask
    )  # (B, P)
    batch_perm_inds = find_best_permutation(match_score, speaker_permutations)  # (B, S)
    return reconstruct_labels(labels, batch_perm_inds)  # (B, context_len, S)


def get_pil_targets(labels: torch.Tensor, preds: torch.Tensor, speaker_permutations: torch.Tensor) -> torch.Tensor:
    """
    Sorts labels and predictions to get the optimal permutation based on the match score.

    Args:
        labels (torch.Tensor): A tensor containing the ground truth labels.
            Shape: (batch_size, num_speakers, num_classes)
        preds (torch.Tensor): A tensor containing the predicted values.
            Shape: (batch_size, num_speakers, num_classes)
        speaker_permutations (torch.Tensor): A tensor containing all possible speaker permutations.
            Shape: (num_permutations, num_speakers)

    Returns:
        torch.Tensor: A tensor of permuted labels that best match the predictions.
            Shape: (batch_size, num_speakers, num_classes)
    """
    permed_labels = labels[:, :, speaker_permutations]  # (batch_size, num_classes, num_permutations, num_speakers)
    # Repeat preds to match permutations for comparison
    preds_rep = torch.unsqueeze(preds, 2).repeat(
        1, 1, speaker_permutations.shape[0], 1
    )  # (batch_size, num_speakers, num_permutations, num_classes)
    match_score = torch.sum(permed_labels * preds_rep, axis=1).sum(axis=2)  # (batch_size, num_permutations)
    batch_perm_inds = find_best_permutation(match_score, speaker_permutations)  # (batch_size, num_speakers)
    # Reconstruct labels based on the best permutation for each batch
    max_score_permed_labels = reconstruct_labels(labels, batch_perm_inds)  # (batch_size, num_speakers, num_classes)
    return max_score_permed_labels  # (batch_size, num_speakers, num_classes)


def get_pil_targets_hungarian(
    labels: torch.Tensor,
    logits: torch.Tensor,
    cls_preds: Optional[torch.Tensor] = None,
    cls_preds_weight: float = 1.0,
    metric: str = 'dot_product',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates permutation-invariant training (PIT) targets using the Hungarian algorithm.
    This function finds the optimal permutation of labels to match the predictions by maximizing
    the dot product between them. This is useful when the number of speakers in labels and
    predictions differ.
    Args:
        labels (torch.Tensor): Ground truth labels of shape (B, T, S), where B is the batch size,
            T is the number of frames, and S is the number of speakers in labels.
        logits (torch.Tensor): Predicted speaker logits of shape (B, T, N), where N is the
            number of speakers in predictions.
        cls_preds (torch.Tensor, optional): Predicted speaker existence probabilities of shape (B, N).
            If provided, these probabilities are added to the match score to bias the assignment.
            Defaults to None.
        cls_preds_weight (float): Weight for the `cls_preds` contribution. Defaults to 1.0.
        metric (str): Metric to use for the match score. Can be 'accuracy', 'dot_product' or 'bce'.
            If 'accuracy' or 'dot_product' is used, sigmoid is applied to logits. Defaults to 'dot_product'.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - reconstructed_labels: The permuted labels that best match the predictions, of shape (B, T, N).
            - spk_indices: Speaker indices mapping of shape (B, N), where spk_indices[b, n] is the 
              original label speaker index that prediction column n corresponds to, or -1 if unmatched.
    """
    batch_size, _num_frames, num_speakers_labels = labels.shape
    _batch_size, _num_frames, num_speakers_preds = logits.shape

    # Check for NaN/Inf in input logits (can happen early in training)
    # Clip logits to [-20, 20] to prevent numerical overflow in sigmoid/BCE computation
    # sigmoid(20) ≈ 1.0 and sigmoid(-20) ≈ 0.0 within floating point precision
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        nan_count = torch.isnan(logits).sum().item()
        inf_count = torch.isinf(logits).sum().item()
        logging.warning(f"Invalid values detected in logits: {nan_count} NaNs, {inf_count} Infs. Clipping to [-20, 20].")
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
    else:
        # Even if no NaN/Inf, clamp extreme values for numerical stability
        logits = torch.clamp(logits, min=-20.0, max=20.0)

    # Allow rectangular assignment when num_speakers_labels > num_speakers_preds (N < S)

    # Expand dimensions to calculate the pair-wise match score
    # logits_expanded: (B, T, N) -> (B, T, 1, N) -> (B, T, S, N)
    # labels_expanded: (B, T, S) -> (B, T, S, 1) -> (B, T, S, N)
    logits_expanded = logits.unsqueeze(2).expand(-1, -1, num_speakers_labels, -1)
    labels_expanded = labels.unsqueeze(3).expand(-1, -1, -1, num_speakers_preds)

    # Calculate match score by averaging the element-wise product over the time dimension.
    # The result is a match score matrix of shape (B, S, N).
    if metric == 'accuracy':
        preds_expanded = torch.sigmoid(logits_expanded)
        match_score_matrix = (preds_expanded * labels_expanded + (1 - preds_expanded) * (1 - labels_expanded)).mean(
            dim=1
        )
    elif metric == 'bce':
        # Negative BCE with logits, since we want to maximize the score
        match_score_matrix = -F.binary_cross_entropy_with_logits(
            logits_expanded, labels_expanded.to(logits_expanded.dtype), reduction='none'
        ).mean(dim=1)
    elif metric == 'dot_product':
        preds_expanded = torch.sigmoid(logits_expanded)
        match_score_matrix = (preds_expanded * labels_expanded).mean(dim=1)
    else:
        raise ValueError(f"Unsupported metric for Hungarian assignment: {metric}")

    active_speaker_mask = torch.any(labels > 0.5, dim=1)  # (B, S)
    # If cls_preds is provided, add it to the match score to bias the assignment.
    if cls_preds is not None:
        # Add cls_preds contribution only for active speakers
        match_score_matrix = match_score_matrix + cls_preds_weight * cls_preds.unsqueeze(1)
        
    #logging.info(f"match_score_matrix: {match_score_matrix}")
    # Set inactive speakers to have very low match scores (high cost) so they won't be matched
    # unless absolutely necessary (e.g., when N >= S and all speakers must be matched)
    inactive_speaker_mask = (~active_speaker_mask).unsqueeze(2)  # (B, S, 1) - boolean mask
    # Use a very large negative value (becomes very large positive cost) to discourage matching inactive speakers
    # Using -1e6 ensures inactive speakers have very high cost without risking overflow
    large_negative_value = -1e6
    match_score_matrix = torch.where(
        inactive_speaker_mask.expand_as(match_score_matrix),
        torch.full_like(match_score_matrix, large_negative_value),
        match_score_matrix
    )
    #logging.info(f"match_score_matrix after active speaker mask: {match_score_matrix}")

    # linear_sum_assignment minimizes the cost, so we use the negative of the score for maximization.
    # We also convert to float32, as numpy doesn't support bfloat16.
    cost_matrix_np = -match_score_matrix.detach().cpu().to(torch.float32).numpy()

    # Additional safety check: ensure cost matrix has no NaN/Inf values
    if not np.isfinite(cost_matrix_np).all():
        nan_count = np.isnan(cost_matrix_np).sum()
        inf_count = np.isinf(cost_matrix_np).sum()
        logging.warning(
            f"Invalid values in cost matrix for Hungarian algorithm: {nan_count} NaNs, {inf_count} Infs. "
            f"Replacing with large finite values. This may indicate numerical instability in the model."
        )
        # Replace NaN with 0 and Inf with large finite values
        cost_matrix_np = np.nan_to_num(cost_matrix_np, nan=0.0, posinf=1e6, neginf=-1e6)

    # Find the best permutation using the Hungarian algorithm for each item in the batch
    batch_row_ind = []
    batch_col_ind = []
    for i in range(batch_size):
        row_ind, col_ind = linear_sum_assignment(cost_matrix_np[i])
        batch_row_ind.append(torch.from_numpy(row_ind).to(labels.device))
        batch_col_ind.append(torch.from_numpy(col_ind).to(labels.device))

    batch_row_ind = torch.stack(batch_row_ind)  # (B, K), K = min(S, N)
    batch_col_ind = torch.stack(batch_col_ind)  # (B, K)

    # Create a permutation matrix P of shape (B, S, N) with ones at matched (row, col) pairs
    P = torch.zeros(batch_size, num_speakers_labels, num_speakers_preds, device=labels.device, dtype=labels.dtype)
    b_idx = torch.arange(batch_size, device=labels.device).unsqueeze(1).expand_as(batch_col_ind)
    P[b_idx, batch_row_ind, batch_col_ind] = 1.0

    # Reconstruct labels with the best permutation; output shape is always (B, T, N)
    reconstructed_labels = torch.matmul(labels, P)
    
    # Create speaker indices mapping: (B, N) where spk_indices[b, n] is the original speaker index
    # that prediction column n corresponds to, or -1 if unmatched or if the speaker has no activity
    spk_indices = torch.full(
        (batch_size, num_speakers_preds), 
        -1, 
        device=labels.device, 
        dtype=torch.long
    )
    # Fill in the matched pairs, but only if the speaker has activity
    b_idx_expanded = torch.arange(batch_size, device=labels.device).unsqueeze(1).expand_as(batch_col_ind)
    # Check if matched speakers are active, if not, keep -1
    matched_speaker_active = active_speaker_mask[b_idx_expanded, batch_row_ind]  # (B, K)
    # Only assign speaker indices for active speakers
    spk_indices[b_idx_expanded, batch_col_ind] = torch.where(
        matched_speaker_active,
        batch_row_ind,
        torch.full_like(batch_row_ind, -1)
    )
    
    return reconstructed_labels, spk_indices


def find_segments_from_rttm(
    recording_id: str,
    rttms: SupervisionSet,
    start_after: float,
    end_before: float,
    adjust_offset: bool = True,
    tolerance: float = 0.001,
):
    """
    Finds segments from the given rttm file.
    This function is designed to replace rttm

    Args:
        recording_id (str): The recording ID in string format.
        rttms (SupervisionSet): The SupervisionSet instance.
        start_after (float): The start time after which segments are selected.
        end_before (float): The end time before which segments are selected.
        adjust_offset (bool): Whether to adjust the offset of the segments.
        tolerance (float): The tolerance for time matching. 0.001 by default.

    Returns:
        segments (List[SupervisionSegment]): A list of SupervisionSegment instances.
    """
    segment_by_recording_id = rttms._segments_by_recording_id
    if segment_by_recording_id is None:
        from cytoolz import groupby

        segment_by_recording_id = groupby(lambda seg: seg.recording_id, rttms)

    return [
        # We only modify the offset - the duration remains the same, as we're only shifting the segment
        # relative to the Cut's start, and not truncating anything.
        segment.with_offset(-start_after) if adjust_offset else segment
        for segment in segment_by_recording_id.get(recording_id, [])
        if segment.start < end_before + tolerance and segment.end > start_after + tolerance
    ]


def get_mask_from_segments(
    segments: list,
    a_cut: Optional[Union[MonoCut, MixedCut]],
    speaker_to_idx_map: torch.Tensor,
    num_speakers: int = 4,
    feat_per_sec: int = 100,
    ignore_num_spk_mismatch: bool = False,
):
    """
    Generate mask matrix from segments list.
    This function is needed for speaker diarization with ASR model trainings.

    Args:
        segments: A list of Lhotse Supervision segments iterator.
        cut (MonoCut, MixedCut): Lhotse MonoCut or MixedCut instance.
        speaker_to_idx_map (dict): A dictionary mapping speaker names to indices.
        num_speakers (int): max number of speakers for all cuts ("mask" dim0), 4 by default
        feat_per_sec (int): number of frames per second, 100 by default, 0.01s frame rate
        ignore_num_spk_mismatch (bool): This is a temporary solution to handle speaker mismatch.
                                        Will be removed in the future.

    Returns:
        mask (Tensor): A numpy array of shape (num_speakers, encoder_hidden_len).
            Dimension: (num_speakers, num_frames)
    """
    # get targets with 0.01s frame rate
    num_samples = round(a_cut.duration * feat_per_sec)
    mask = torch.zeros((num_samples, num_speakers))
    for rttm_sup in segments:
        speaker_idx = speaker_to_idx_map[rttm_sup.speaker]
        if speaker_idx >= num_speakers:
            if ignore_num_spk_mismatch:
                continue
            else:
                raise ValueError(f"Speaker Index {speaker_idx} exceeds the max index: {num_speakers-1}")
        stt = max(rttm_sup.start, 0)
        ent = min(rttm_sup.end, a_cut.duration)
        stf = int(stt * feat_per_sec)
        enf = int(ent * feat_per_sec)
        mask[stf:enf, speaker_idx] = 1.0
    return mask


def get_soft_mask(feat_level_target, num_frames, stride):
    """
    Get soft mask from feat_level_target with stride.
    This function is needed for speaker diarization with ASR model trainings.

    Args:
        feat_level_target (Tensor): A numpy array of shape (num_frames, num_speakers).
            Dimension: (num_frames, num_speakers)
        num_sample (int): The total number of samples.
        stride (int): The stride for the mask.

    Returns:
        mask: The soft mask of shape (num_frames, num_speakers).
            Dimension: (num_frames, num_speakers)
    """

    num_speakers = feat_level_target.shape[1]
    mask = torch.zeros(num_frames, num_speakers)

    for index in range(num_frames):
        if index == 0:
            seg_stt_feat = 0
        else:
            seg_stt_feat = stride * index - 1 - int(stride / 2)
        if index == num_frames - 1:
            seg_end_feat = feat_level_target.shape[0]
        else:
            seg_end_feat = stride * index - 1 + int(stride / 2)
        mask[index] = torch.mean(feat_level_target[seg_stt_feat : seg_end_feat + 1, :], axis=0)
    return mask


def get_hidden_length_from_sample_length(
    num_samples: int, num_sample_per_mel_frame: int = 160, num_mel_frame_per_asr_frame: int = 8
) -> int:
    """
    Calculate the hidden length from the given number of samples.
    This function is needed for speaker diarization with ASR model trainings.

    This function computes the number of frames required for a given number of audio samples,
    considering the number of samples per mel frame and the number of mel frames per ASR frame.

    Please refer to the following function for more on feature frame length calculation:
        NeMo/nemo/collections/asr/parts/preprocessing/features.py::FilterbankFeatures::get_seq_len

    Parameters:
        num_samples (int): The total number of audio samples.
        num_sample_per_mel_frame (int, optional): The number of samples per mel frame. Default is 160.
        num_mel_frame_per_asr_frame (int, optional): The number of mel frames per ASR frame. Default is 8.

    Returns:
        hidden_length (int): The calculated hidden length in terms of the number of frames.
    """
    mel_frame_count = math.ceil(num_samples / num_sample_per_mel_frame)
    hidden_length = math.ceil(mel_frame_count / num_mel_frame_per_asr_frame)
    return int(hidden_length)


def speaker_to_target(
    a_cut,
    num_speakers: int = 4,
    num_sample_per_mel_frame: int = 160,
    num_mel_frame_per_asr_frame: int = 8,
    spk_tar_all_zero: bool = False,
    boundary_segments: bool = False,
    soft_label: bool = False,
    ignore_num_spk_mismatch: bool = True,
    soft_thres: float = 0.5,
):
    """
    Get rttm samples corresponding to one cut, generate speaker mask numpy.ndarray with shape
    (num_speaker, hidden_length). This function is needed for speaker diarization with ASR model trainings.

    Args:
        a_cut (MonoCut, MixedCut):
            Lhotse Cut instance which is MonoCut or MixedCut instance.
        num_speakers (int):
            Max number of speakers for all cuts ("mask" dim0), 4 by default
        num_sample_per_mel_frame (int):
            Number of sample per mel frame, sample_rate / 1000 * window_stride, 160 by default (10ms window stride)
        num_mel_frame_per_asr_frame (int):
            Encoder subsampling_factor, 8 by default
        spk_tar_all_zero (Tensor):
            Set to True gives all zero "mask"
        boundary_segments (bool):
            Set to True to include segments containing the boundary of the cut,
            False by default for multi-speaker ASR training
        soft_label (bool):
            Set to True to use soft label that enables values in [0, 1] range,
            False by default and leads to binary labels.
        ignore_num_spk_mismatch (bool):
            This is a temporary solution to handle speaker mismatch. Will be removed in the future.

    Returns:
        mask (Tensor): Speaker mask with shape (num_speaker, hidden_lenght)
    """
    # get cut-related segments from rttms
    if isinstance(a_cut, MixedCut):
        cut_list = [track.cut for track in a_cut.tracks if isinstance(track.cut, MonoCut)]
        offsets = [track.offset for track in a_cut.tracks if isinstance(track.cut, MonoCut)]
    elif isinstance(a_cut, MonoCut):
        cut_list = [a_cut]
        offsets = [0]
    else:
        raise ValueError(f"Unsupported cut type type{a_cut}: only MixedCut and MonoCut are supported")

    segments_total = []
    for i, cut in enumerate(cut_list):
        rttms = SupervisionSet.from_rttm(cut.rttm_filepath)
        if boundary_segments:  # segments with seg_start < total_end and seg_end > total_start are included
            segments_iterator = find_segments_from_rttm(
                recording_id=cut.recording_id, rttms=rttms, start_after=cut.start, end_before=cut.end, tolerance=0.0
            )
        else:  # segments with seg_start > total_start and seg_end < total_end are included
            segments_iterator = rttms.find(
                recording_id=cut.recording_id, start_after=cut.start, end_before=cut.end, adjust_offset=True
            )

        for seg in segments_iterator:
            if seg.start < 0:
                seg.duration += seg.start
                seg.start = 0
            if seg.end > cut.duration:
                seg.duration -= seg.end - cut.duration
            seg.start += offsets[i]
            segments_total.append(seg)

    # apply arrival time sorting to the existing segments
    segments_total.sort(key=lambda rttm_sup: rttm_sup.start)

    seen = set()
    seen_add = seen.add
    speaker_ats = [s.speaker for s in segments_total if not (s.speaker in seen or seen_add(s.speaker))]

    speaker_to_idx_map = {spk: idx for idx, spk in enumerate(speaker_ats)}
    if len(speaker_to_idx_map) > num_speakers and not ignore_num_spk_mismatch:  # raise error if number of speakers
        raise ValueError(
            f"Number of speakers {len(speaker_to_idx_map)} is larger than "
            f"the maximum number of speakers {num_speakers}"
        )

    # initialize mask matrices (num_speaker, encoder_hidden_len)
    feat_per_sec = int(a_cut.sampling_rate / num_sample_per_mel_frame)  # 100 by default
    num_samples = get_hidden_length_from_sample_length(
        a_cut.num_samples, num_sample_per_mel_frame, num_mel_frame_per_asr_frame
    )
    if spk_tar_all_zero:
        frame_mask = torch.zeros((num_samples, num_speakers))
    else:
        frame_mask = get_mask_from_segments(
            segments_total, a_cut, speaker_to_idx_map, num_speakers, feat_per_sec, ignore_num_spk_mismatch
        )
    soft_mask = get_soft_mask(frame_mask, num_samples, num_mel_frame_per_asr_frame)

    if soft_label:
        mask = soft_mask
    else:
        mask = (soft_mask > soft_thres).float()

    return mask
