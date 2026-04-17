"""
Native-PyTorch all-to-all dispatch / combine for MoE Expert Parallelism.

This module implements the "dropless" a2a dispatch used when ``ep_size > 1``:

1.  Each rank sees ``(N_local, D)`` tokens and picks ``top_k`` experts per
    token via the (replicated) router.
2.  Tokens are replicated ``top_k`` times and sorted so that all tokens going
    to the same global expert are contiguous on the sender side.
3.  A single :func:`torch.distributed.all_to_all_single` delivers each
    contiguous group to the EP rank that owns the target expert.
4.  On the receiver, a second (local) permutation regroups incoming tokens by
    *local* expert id so that a grouped-GEMM / batched-bmm kernel can consume
    them.
5.  After expert compute, the reverse pipeline (inverse permutations +
    reverse a2a + weighted scatter-add) returns ``(N_local, D)`` outputs.

All operations are autograd-safe:

- :func:`torch.distributed.all_to_all_single` in PyTorch >= 2.1 registers a
  backward that performs a reverse a2a with swapped split sizes.
- :func:`torch.index_select` and :func:`torch.Tensor.scatter_add_` are
  standard differentiable ops.

No Megatron dependency. Works identically on A100 and H100 (the only
difference is NCCL bandwidth). No CUDA-arch-specific calls.

See ``/work/moe/docs/moe_transformer_encoder_parallelism.md`` §5 for the
algorithmic derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

__all__ = [
    "DispatchCtx",
    "dispatch_tokens",
    "combine_tokens",
]


@dataclass
class DispatchCtx:
    """Per-call state that must flow from :func:`dispatch_tokens` to
    :func:`combine_tokens`. All tensors live on the compute device.

    Attributes:
        send_perm: ``(N * K,)`` -- initial sender-side permutation that sorted
            tokens by global target-expert id. We keep its inverse for combine.
        regroup_perm: ``(total_recv,)`` -- receiver-side permutation that
            groups incoming tokens by *local* expert id.
        send_counts_ep: ``list[int]`` of length ``ep_size`` -- how many tokens
            this rank sent to each peer (for the payload a2a split sizes).
        recv_counts_ep: ``list[int]`` of length ``ep_size`` -- how many tokens
            this rank received from each peer.
        local_expert_counts: ``(L,)`` int64 tensor -- number of received
            tokens per local expert (sum across peers). Consumed by
            :class:`LocalExperts`.
        flat_weight: ``(N * K,)`` -- gate weight attached to each routed
            entry; used in combine to weight each expert output.
        N: Number of tokens on this rank before top-k replication.
        K: ``top_k``.
        D: Hidden dimension.
        dtype: Working dtype for payload tensors (used to allocate reverse
            buffers).
    """

    send_perm: torch.Tensor
    regroup_perm: torch.Tensor
    send_counts_ep: List[int]
    recv_counts_ep: List[int]
    local_expert_counts: torch.Tensor
    flat_weight: torch.Tensor
    N: int
    K: int
    D: int
    dtype: torch.dtype


def _compute_local_expert_labels(
    recv_counts_per_expert_from_peer: torch.Tensor,  # (ep_size * L,) int64
    ep_size: int,
    experts_per_rank: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a ``(total_recv,)`` tensor whose i-th entry is the *local* expert
    id of the i-th received token (in the post-a2a peer-chunked layout).

    We use this as a sort key to regroup incoming tokens by local expert id.
    """
    # Per-peer layout of local-expert labels: [0, 1, ..., L-1] repeated ep_size times.
    labels_flat = (
        torch.arange(experts_per_rank, device=device, dtype=torch.long)
        .unsqueeze(0)
        .expand(ep_size, experts_per_rank)
        .reshape(-1)
    )  # shape (ep_size * L,)
    return torch.repeat_interleave(labels_flat, recv_counts_per_expert_from_peer)


def dispatch_tokens(
    x_local: torch.Tensor,  # (N, D)
    top_k_idx: torch.Tensor,  # (N, K) int64, global expert ids
    top_k_weights: torch.Tensor,  # (N, K) float
    ep_group,  # ProcessGroup
    num_experts: int,
    ep_size: int,
    experts_per_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, DispatchCtx]:
    """Permute + a2a + regroup.

    Returns:
        (x_grouped, local_expert_counts, ctx) where:

        - ``x_grouped`` is ``(M, D)`` with tokens grouped contiguously by local
          expert id, ready for :class:`LocalExperts` to consume.
        - ``local_expert_counts`` is ``(L,)`` int64 -- how many tokens per
          local expert in ``x_grouped``.
        - ``ctx`` is a :class:`DispatchCtx` to pass to :func:`combine_tokens`.
    """
    device = x_local.device
    N, D = x_local.shape
    K = top_k_idx.shape[1]
    L = experts_per_rank
    dtype = x_local.dtype

    flat_assign = top_k_idx.reshape(-1)  # (N*K,)
    flat_weight = top_k_weights.reshape(-1)  # (N*K,)
    flat_x = x_local.unsqueeze(1).expand(N, K, D).reshape(N * K, D)

    send_perm = torch.argsort(flat_assign, stable=True)
    x_perm = flat_x.index_select(0, send_perm)

    counts_per_expert = torch.bincount(flat_assign, minlength=num_experts).to(torch.int64)
    send_counts_ep_t = counts_per_expert.view(ep_size, L).sum(dim=1)  # (ep_size,)

    recv_counts_per_expert_from_peer = torch.empty_like(counts_per_expert)
    dist.all_to_all_single(
        recv_counts_per_expert_from_peer,
        counts_per_expert,
        output_split_sizes=[L] * ep_size,
        input_split_sizes=[L] * ep_size,
        group=ep_group,
    )
    recv_counts_ep_t = recv_counts_per_expert_from_peer.view(ep_size, L).sum(dim=1)
    local_expert_counts = recv_counts_per_expert_from_peer.view(ep_size, L).sum(dim=0)

    # CPU sync: required for passing list[int] split sizes to a2a_single.
    send_counts_ep = send_counts_ep_t.tolist()
    recv_counts_ep = recv_counts_ep_t.tolist()
    total_recv = int(sum(recv_counts_ep))

    recv_x = torch.empty(total_recv, D, device=device, dtype=dtype)
    dist.all_to_all_single(
        recv_x,
        x_perm,
        output_split_sizes=recv_counts_ep,
        input_split_sizes=send_counts_ep,
        group=ep_group,
    )

    if total_recv > 0:
        labels = _compute_local_expert_labels(
            recv_counts_per_expert_from_peer,
            ep_size=ep_size,
            experts_per_rank=L,
            device=device,
        )
        regroup_perm = torch.argsort(labels, stable=True)
        x_grouped = recv_x.index_select(0, regroup_perm)
    else:
        regroup_perm = torch.empty(0, device=device, dtype=torch.long)
        x_grouped = recv_x  # shape (0, D)

    ctx = DispatchCtx(
        send_perm=send_perm,
        regroup_perm=regroup_perm,
        send_counts_ep=send_counts_ep,
        recv_counts_ep=recv_counts_ep,
        local_expert_counts=local_expert_counts,
        flat_weight=flat_weight,
        N=N,
        K=K,
        D=D,
        dtype=dtype,
    )
    return x_grouped, local_expert_counts, ctx


def combine_tokens(
    y_grouped: torch.Tensor,  # (M, D) in local-expert-grouped order
    ctx: DispatchCtx,
    ep_group,
) -> torch.Tensor:
    """Inverse-permute + reverse a2a + weighted scatter-add.

    Args:
        y_grouped: Expert outputs in the same order as ``x_grouped`` returned
            by :func:`dispatch_tokens`.
        ctx: The :class:`DispatchCtx` returned by the matching dispatch call.
        ep_group: EP process group.

    Returns:
        ``(N, D)`` tensor of combined expert outputs on this rank.
    """
    device = y_grouped.device
    N, K, D = ctx.N, ctx.K, ctx.D

    total_recv = y_grouped.shape[0]

    # (1) Inverse of the receiver-side regroup permutation.
    if total_recv > 0:
        inv_regroup = torch.empty_like(ctx.regroup_perm)
        inv_regroup[ctx.regroup_perm] = torch.arange(total_recv, device=device)
        y_peer_chunked = y_grouped.index_select(0, inv_regroup)
    else:
        y_peer_chunked = y_grouped  # (0, D)

    # (2) Reverse a2a: send each peer their outputs.
    total_send = int(sum(ctx.send_counts_ep))
    y_back = torch.empty(total_send, D, device=device, dtype=ctx.dtype)
    dist.all_to_all_single(
        y_back,
        y_peer_chunked,
        output_split_sizes=ctx.send_counts_ep,
        input_split_sizes=ctx.recv_counts_ep,
        group=ep_group,
    )

    # (3) Invert the initial sender-side sort.
    inv_send = torch.empty_like(ctx.send_perm)
    inv_send[ctx.send_perm] = torch.arange(ctx.send_perm.shape[0], device=device)
    y_flat = y_back.index_select(0, inv_send)  # (N*K, D)

    # (4) Weight and combine the K routes per token.
    y_weighted = y_flat * ctx.flat_weight.unsqueeze(-1).to(y_flat.dtype)  # (N*K, D)

    # Scatter-add from (N*K, D) back to (N, D): index = arange(N*K) // K
    token_idx = (
        torch.arange(N * K, device=device).div(K, rounding_mode="floor")
    )  # (N*K,) int64, each token id appears K times
    output = torch.zeros(N, D, device=device, dtype=y_weighted.dtype)
    output.scatter_add_(0, token_idx.unsqueeze(-1).expand(-1, D), y_weighted)

    return output
