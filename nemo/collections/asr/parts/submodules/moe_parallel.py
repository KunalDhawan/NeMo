"""
Native-PyTorch parallelism primitives for MoE encoders.

Provides the minimal infrastructure needed by :mod:`moe_transformer_encoder`
to run Expert Parallelism (EP) on top of Lightning's DDP:

- Discover world topology (rank / world size / local rank) from environment
  variables that are available *before* ``torch.distributed`` is initialized
  (Slurm / torchrun / Lightning all set these).
- Build a 2-D ``(dp, ep)`` DeviceMesh once distributed is up.
- Expose helpers to query local-expert ownership per rank.

No dependency on ``megatron.core`` or any NVIDIA-specific runtime; everything
is stock ``torch.distributed``. Works identically on A100 and H100 (the only
difference is the underlying NCCL throughput).

See ``/work/moe/docs/moe_transformer_encoder_parallelism.md`` §3 for the full
design rationale.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

__all__ = [
    "EPTopology",
    "MoEParallelContext",
    "infer_world_from_env",
    "local_expert_range",
]


# ---------------------------------------------------------------------------
# Static topology -- safe to compute before torch.distributed is initialized.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EPTopology:
    """Static description of the (dp, ep) rank layout, known at module __init__.

    Determined purely from environment variables set by the launcher (Slurm,
    torchrun, Lightning). No reliance on a live ``torch.distributed`` group,
    because the encoder is constructed *before* Lightning initializes
    distributed.

    Attributes:
        world_size: Total number of processes.
        rank: Global rank of this process.
        local_rank: Local rank within the node (from ``LOCAL_RANK`` env var).
        ep_size: Size of the EP process group. ``1`` disables EP.
        dp_size: Size of the DP process group. Always ``world_size / ep_size``.
        ep_rank: This process's index within its EP group (``rank % ep_size``
            under the default contiguous rank layout).
        dp_rank: This process's index within its DP group.
        num_experts: Number of experts per MoE layer (for local-expert math).
        experts_per_rank: ``num_experts // ep_size``.
    """

    world_size: int
    rank: int
    local_rank: int
    ep_size: int
    dp_size: int
    ep_rank: int
    dp_rank: int
    num_experts: int
    experts_per_rank: int

    @property
    def enabled(self) -> bool:
        """Whether EP is active (i.e. ``ep_size > 1``)."""
        return self.ep_size > 1

    def local_expert_ids(self) -> List[int]:
        """Return the global expert ids owned by this rank."""
        return list(range(self.ep_rank * self.experts_per_rank,
                          (self.ep_rank + 1) * self.experts_per_rank))

    def owner_rank_of(self, global_expert_id: int) -> int:
        """Return the EP-local rank index that owns ``global_expert_id``."""
        return global_expert_id // self.experts_per_rank

    def describe(self) -> str:
        """Human-readable summary for logging."""
        return (
            f"EPTopology(world={self.world_size}, rank={self.rank}, "
            f"local_rank={self.local_rank}, ep_size={self.ep_size}, "
            f"dp_size={self.dp_size}, ep_rank={self.ep_rank}, "
            f"dp_rank={self.dp_rank}, num_experts={self.num_experts}, "
            f"experts_per_rank={self.experts_per_rank}, "
            f"local_expert_ids={self.local_expert_ids()})"
        )


def infer_world_from_env() -> Tuple[int, int, int]:
    """Read ``(world_size, rank, local_rank)`` from env vars.

    Supports Slurm (``SLURM_NTASKS``, ``SLURM_PROCID``, ``SLURM_LOCALID``) and
    torchrun / Lightning (``WORLD_SIZE``, ``RANK``, ``LOCAL_RANK``). If nothing
    is set we assume a single-process run (``1, 0, 0``).

    Returns:
        A tuple of ``(world_size, rank, local_rank)``.
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        world_size = int(os.environ.get("WORLD_SIZE",
                          os.environ.get("SLURM_NTASKS", "1")))
        rank = int(os.environ.get("RANK",
                   os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK",
                     os.environ.get("SLURM_LOCALID", "0")))
    return world_size, rank, local_rank


def local_expert_range(num_experts: int, ep_size: int, ep_rank: int) -> Tuple[int, int]:
    """Return the ``[start, end)`` range of global expert ids owned by ``ep_rank``.

    Raises:
        ValueError: If ``num_experts`` is not divisible by ``ep_size``.
    """
    if num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})."
        )
    experts_per_rank = num_experts // ep_size
    return ep_rank * experts_per_rank, (ep_rank + 1) * experts_per_rank


def build_topology(num_experts: int, ep_size: int) -> EPTopology:
    """Build an :class:`EPTopology` from env vars.

    Args:
        num_experts: Total experts per MoE layer (global, not sharded).
        ep_size: Desired EP group size. ``1`` disables EP.

    Returns:
        The topology describing this rank's role.

    Behavior when ``ep_size > world_size`` (typical for inference / restoring
    an EP-trained model on a smaller world): we transparently fall back to
    ``ep_size=1`` and emit a ``UserWarning``. This is safe because:

    - Checkpoints written with EP active (``moe_ep_size>1``) are consolidated
      into full ``(num_experts, ...)`` tensors at save time, so the smaller
      world can load them without sharding.
    - With ``ep_size=1`` the encoder runs the single-rank grouped path (or
      the legacy loop, depending on ``moe_expert_backend``); no all-to-all
      occurs and no EP process group is required.

    Raises:
        ValueError: If ``ep_size < 1``, or if other divisibility constraints
            fail (these are configuration bugs that cannot be auto-fixed).
    """
    world_size, rank, local_rank = infer_world_from_env()

    if ep_size < 1:
        raise ValueError(f"ep_size must be >= 1, got {ep_size}.")

    if ep_size > world_size:
        # Common case: a model trained with moe_ep_size=8 is now being
        # restored on a single-GPU box (world_size=1) for eval. The .nemo
        # contains full expert tensors thanks to checkpoint consolidation,
        # so the only thing we need to do is stop trying to shard them.
        import warnings
        warnings.warn(
            f"moe_ep_size={ep_size} requested but world_size={world_size}; "
            f"falling back to ep_size=1 (no expert sharding). This is the "
            f"expected behavior when restoring an EP-trained .nemo onto a "
            f"smaller world (e.g. single-GPU eval). To suppress, override "
            f"`model.encoder.moe_ep_size=1` at restore time.",
            UserWarning,
            stacklevel=2,
        )
        ep_size = 1
    if world_size % ep_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by ep_size ({ep_size})."
        )
    if ep_size > 1 and num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size}) "
            f"when EP is enabled."
        )

    dp_size = world_size // ep_size
    ep_rank = rank % ep_size
    dp_rank = rank // ep_size
    experts_per_rank = num_experts // ep_size if ep_size > 1 else num_experts

    return EPTopology(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        ep_size=ep_size,
        dp_size=dp_size,
        ep_rank=ep_rank,
        dp_rank=dp_rank,
        num_experts=num_experts,
        experts_per_rank=experts_per_rank,
    )


# ---------------------------------------------------------------------------
# Dynamic context -- only usable once torch.distributed is initialized.
# ---------------------------------------------------------------------------


class MoEParallelContext:
    """Lazy-initialized holder for the ``(dp, ep)`` device mesh and its
    sub-process-groups.

    The object is safe to construct at encoder ``__init__`` time (before
    ``torch.distributed`` is up) but its ``mesh`` / ``ep_group`` / ``dp_group``
    attributes must not be accessed until :meth:`ensure_initialized` has been
    called -- typically from the Lightning model's ``on_train_start`` (or from
    the first ``forward`` as a safety net).

    The lazy design matches Lightning's lifecycle: modules are constructed in
    user code, then the ``DDPStrategy`` initializes distributed, then ``fit``
    starts.

    Args:
        topology: Static EP topology (from :func:`build_topology`).
        device: Device string for the DeviceMesh (default ``"cuda"``).
    """

    def __init__(self, topology: EPTopology, device: str = "cuda") -> None:
        self.topology = topology
        self._device = device
        self._mesh = None
        self._ep_group = None
        self._dp_group = None
        self._initialized = False

    @property
    def enabled(self) -> bool:
        return self.topology.enabled

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def ep_group(self):
        self._check_ready()
        return self._ep_group

    @property
    def dp_group(self):
        self._check_ready()
        return self._dp_group

    @property
    def mesh(self):
        self._check_ready()
        return self._mesh

    def _check_ready(self):
        if not self._initialized:
            raise RuntimeError(
                "MoEParallelContext accessed before initialization. Call "
                "ensure_initialized() after torch.distributed is up (e.g. "
                "in LightningModule.on_train_start)."
            )

    def ensure_initialized(self) -> None:
        """Build the DeviceMesh if EP is enabled and ``dist`` is up.

        Idempotent: safe to call multiple times.
        """
        if self._initialized:
            return

        if not self.topology.enabled:
            # EP disabled -- no mesh needed. Mark initialized so callers can
            # skip silently.
            self._initialized = True
            return

        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "MoE EP requires torch.distributed to be initialized before "
                "the first forward. Make sure the Lightning DDPStrategy has "
                "run setup_distributed() (normally automatic)."
            )

        current_world = dist.get_world_size()
        if current_world != self.topology.world_size:
            raise RuntimeError(
                f"World size changed since topology was built: "
                f"expected {self.topology.world_size}, got {current_world}."
            )

        # Use the torch.distributed.device_mesh API -- available in PyTorch >= 2.2.
        # No Megatron / NeMo-specific dependency.
        from torch.distributed.device_mesh import init_device_mesh

        self._mesh = init_device_mesh(
            device_type=self._device,
            mesh_shape=(self.topology.dp_size, self.topology.ep_size),
            mesh_dim_names=("dp", "ep"),
        )
        self._ep_group = self._mesh.get_group("ep")
        self._dp_group = self._mesh.get_group("dp")
        self._initialized = True

        if self.topology.rank == 0:
            print(
                f"[MoEParallelContext] Initialized EP mesh: "
                f"dp_size={self.topology.dp_size}, ep_size={self.topology.ep_size}, "
                f"device={self._device}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Gradient-sync helpers for EP-sharded parameters.
# ---------------------------------------------------------------------------


def register_dp_allreduce_hook(param: torch.nn.Parameter, dp_group, dp_size: int) -> None:
    """Register a post-accumulate-grad hook that averages the parameter's
    gradient across the DP group.

    Used for EP-local expert parameters which are intentionally excluded from
    DDP's world-level reducer (they are not replicated across the world, only
    across the DP sub-group).

    The hook is idempotent: registering it twice only attaches once (via a
    private attribute guard) to survive re-entry from ``on_train_start``.

    Args:
        param: The expert parameter.
        dp_group: The DP ``ProcessGroup`` to reduce over.
        dp_size: The DP group size (used for mean scaling).
    """
    if getattr(param, "_moe_dp_hook_registered", False):
        return
    if dp_size <= 1:
        # No peers -- hook would be a no-op, skip.
        param._moe_dp_hook_registered = True
        return

    scale = 1.0 / float(dp_size)

    def _hook(p: torch.Tensor) -> None:
        if p.grad is None:
            return
        # All-reduce in fp32 to avoid bf16 saturation when DP is large; cast back.
        grad_fp32 = p.grad.to(torch.float32, copy=True) if p.grad.dtype != torch.float32 else p.grad
        dist.all_reduce(grad_fp32, op=dist.ReduceOp.SUM, group=dp_group)
        grad_fp32.mul_(scale)
        if p.grad.dtype != torch.float32:
            p.grad.copy_(grad_fp32.to(p.grad.dtype))
        # else: grad_fp32 aliases p.grad, already updated in-place.

    param.register_post_accumulate_grad_hook(_hook)
    param._moe_dp_hook_registered = True
