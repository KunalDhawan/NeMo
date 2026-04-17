"""
``LocalExperts`` -- batched grouped-linear expert compute for MoE encoders.

Holds the feed-forward weights for the experts owned by the local rank in a
single grouped tensor, enabling three efficient execution backends:

- ``grouped_mm``: one kernel via :func:`torch._grouped_mm` (PyTorch >= 2.6,
  optimized for H100, works on A100).
- ``bmm``: padded batched matmul via :func:`torch.bmm`. Works on any
  CUDA-capable GPU, any PyTorch >= 2.0. Trades padding waste for a single
  kernel launch.
- ``loop``: Python loop over local experts. Equivalent to the pre-EP
  per-expert loop; used as the numerical reference and when ``L == 1`` (where
  "grouped" is a no-op anyway).

Selection is controlled by ``backend = {'auto', 'grouped_mm', 'bmm', 'loop'}``.
``'auto'`` picks the fastest available option and emits a one-time
``UserWarning`` describing the choice (useful for A100 runs where
``grouped_mm`` may not be present).

No Megatron dependency. The weight layout is (L, in, out) for linear 1 and
(L, out, in) for linear 2 -- which corresponds to ``x @ w + b`` rather than
the ``(out, in)`` convention used by :class:`torch.nn.Linear`. State-dict
loading transparently transposes from the legacy layout.

See ``/work/moe/docs/moe_transformer_encoder_parallelism.md`` §6.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LocalExperts", "choose_expert_backend"]


_GROUPED_MM_PROBE_DONE = False
_GROUPED_MM_WORKS = False


def _probe_grouped_mm() -> bool:
    """One-time check whether :func:`torch._grouped_mm` is importable and
    runnable on a tiny tensor. Caches the result for subsequent calls.

    We probe at import-probe time (not at forward) to avoid a per-step try /
    except cost and to surface the fallback choice in a single startup log line.
    """
    global _GROUPED_MM_PROBE_DONE, _GROUPED_MM_WORKS
    if _GROUPED_MM_PROBE_DONE:
        return _GROUPED_MM_WORKS
    _GROUPED_MM_PROBE_DONE = True

    if not hasattr(torch, "_grouped_mm"):
        _GROUPED_MM_WORKS = False
        return False
    if not torch.cuda.is_available():
        _GROUPED_MM_WORKS = False
        return False
    try:
        # Tiny bf16 probe; matches the dtype we use during training.
        a = torch.zeros(4, 2, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(2, 2, 3, device="cuda", dtype=torch.bfloat16)
        offs = torch.tensor([2, 4], device="cuda", dtype=torch.int32)
        _ = torch._grouped_mm(a, b, offs)  # type: ignore[attr-defined]
        _GROUPED_MM_WORKS = True
    except Exception as e:  # pragma: no cover -- device-dependent
        warnings.warn(
            f"[MoE] torch._grouped_mm exists but failed probe ({type(e).__name__}: {e}). "
            f"Falling back to 'bmm' / 'loop' backends.",
            UserWarning,
        )
        _GROUPED_MM_WORKS = False
    return _GROUPED_MM_WORKS


def choose_expert_backend(requested: str, num_local_experts: int) -> str:
    """Resolve ``'auto'`` to a concrete backend, validating explicit choices.

    Emits a one-line ``UserWarning`` describing the final choice -- particularly
    useful when running on A100 where ``grouped_mm`` may not be present.
    """
    requested = (requested or "auto").lower()
    if requested == "loop":
        return "loop"
    if requested == "bmm":
        return "bmm"
    if requested == "grouped_mm":
        if not _probe_grouped_mm():
            raise RuntimeError(
                "moe_expert_backend='grouped_mm' was requested but "
                "torch._grouped_mm is not available in this PyTorch build. "
                "Use 'auto', 'bmm', or 'loop' instead."
            )
        return "grouped_mm"
    if requested == "auto":
        if num_local_experts == 1:
            # Single local expert: loop is a single linear and is already optimal.
            warnings.warn(
                "[MoE] backend='auto' with num_local_experts=1 -> using 'loop' "
                "(no benefit from grouping with L=1).",
                UserWarning,
                stacklevel=2,
            )
            return "loop"
        if _probe_grouped_mm():
            warnings.warn(
                "[MoE] backend='auto' -> using 'grouped_mm' (torch._grouped_mm available).",
                UserWarning,
                stacklevel=2,
            )
            return "grouped_mm"
        warnings.warn(
            "[MoE] backend='auto' -> torch._grouped_mm not available on this "
            "PyTorch/GPU, using 'bmm' fallback. This is the expected path on "
            "A100 or older PyTorch (<2.6).",
            UserWarning,
            stacklevel=2,
        )
        return "bmm"
    raise ValueError(
        f"Unknown moe_expert_backend '{requested}'. "
        f"Choose from: auto, grouped_mm, bmm, loop."
    )


class LocalExperts(nn.Module):
    """Batched feed-forward for the local-rank's subset of MoE experts.

    Structure per expert (identical to the base ``FeedForward``):

    .. code-block:: text

        h = GELU(x @ w1 + b1)      # (D) -> (4D)
        y = h @ w2 + b2            # (4D) -> (D)

    Weights are stored in grouped tensors of shape ``(L, D, 4D)`` / ``(L, 4D, D)``
    so that a single grouped-GEMM or batched-bmm kernel can compute all local
    experts in one pass. Tokens are expected to arrive in local-expert-sorted
    order with a ``counts: (L,)`` companion tensor describing the group sizes.

    Args:
        d_model: Hidden dimension ``D``.
        num_local_experts: Number of experts owned by this rank (``L =
            num_experts // ep_size``).
        hidden_multiplier: FFN expansion factor (default ``4``, matching the
            base ``FeedForward``).
        bias: Include biases (default ``True`` for parity with
            :class:`FeedForward`).
        backend: One of ``'auto' | 'grouped_mm' | 'bmm' | 'loop'``.
    """

    def __init__(
        self,
        d_model: int,
        num_local_experts: int,
        hidden_multiplier: int = 4,
        bias: bool = True,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_local_experts = num_local_experts
        self.hidden_dim = hidden_multiplier * d_model
        self.use_bias = bias
        self.backend = choose_expert_backend(backend, num_local_experts)

        L, D, H = num_local_experts, d_model, self.hidden_dim
        # Layout matches `x @ w + b`: w1 is (L, D, H), w2 is (L, H, D).
        self.w1 = nn.Parameter(torch.empty(L, D, H))
        self.w2 = nn.Parameter(torch.empty(L, H, D))
        if bias:
            self.b1 = nn.Parameter(torch.empty(L, H))
            self.b2 = nn.Parameter(torch.empty(L, D))
        else:
            self.register_parameter("b1", None)
            self.register_parameter("b2", None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Match :class:`torch.nn.Linear`'s default initialization per expert slot."""
        for slot in range(self.num_local_experts):
            # w1: fan_in = D
            nn.init.kaiming_uniform_(self.w1[slot], a=math.sqrt(5))
            # w2: fan_in = H
            nn.init.kaiming_uniform_(self.w2[slot], a=math.sqrt(5))
        if self.use_bias:
            bound1 = 1.0 / math.sqrt(self.d_model)
            bound2 = 1.0 / math.sqrt(self.hidden_dim)
            nn.init.uniform_(self.b1, -bound1, bound1)
            nn.init.uniform_(self.b2, -bound2, bound2)

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _forward_loop(self, x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Reference implementation: Python loop over local experts."""
        out = torch.empty_like(x)
        offset = 0
        counts_list = counts.tolist()
        for j in range(self.num_local_experts):
            n = counts_list[j]
            if n == 0:
                continue
            xi = x[offset:offset + n]
            h = xi @ self.w1[j]
            if self.use_bias:
                h = h + self.b1[j]
            h = F.gelu(h)
            y = h @ self.w2[j]
            if self.use_bias:
                y = y + self.b2[j]
            out[offset:offset + n] = y
            offset += n
        return out

    def _forward_bmm(self, x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Padded batched matmul. Pads each group to ``max_count`` and uses
        :func:`torch.bmm`. Waste is proportional to count imbalance."""
        L = self.num_local_experts
        D = self.d_model
        total = x.shape[0]
        counts_list = counts.tolist()
        max_n = max(counts_list) if counts_list else 0
        if max_n == 0:
            return x.new_zeros(0, D)

        # Pack (L, max_n, D) from the flat (total, D).
        padded = x.new_zeros(L, max_n, D)
        valid_mask = x.new_zeros(L, max_n, dtype=torch.bool)
        offset = 0
        for j in range(L):
            n = counts_list[j]
            if n > 0:
                padded[j, :n] = x[offset:offset + n]
                valid_mask[j, :n] = True
                offset += n

        # bmm: (L, max_n, D) @ (L, D, H) -> (L, max_n, H)
        h = torch.bmm(padded, self.w1)
        if self.use_bias:
            h = h + self.b1.unsqueeze(1)
        h = F.gelu(h)
        y = torch.bmm(h, self.w2)  # (L, max_n, D)
        if self.use_bias:
            y = y + self.b2.unsqueeze(1)

        # Unpack back to flat (total, D). Use masked_select + reshape to preserve autograd.
        return y[valid_mask].reshape(total, D)

    def _forward_grouped_mm(self, x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Single-kernel grouped GEMM via :func:`torch._grouped_mm`."""
        # torch._grouped_mm expects cumulative offsets (end-indices) as int32.
        offs = torch.cumsum(counts, dim=0).to(torch.int32)
        # (M, D) @ (L, D, H) with ragged group splits on dim 0 of x -> (M, H)
        h = torch._grouped_mm(x, self.w1, offs)  # type: ignore[attr-defined]
        if self.use_bias:
            h = h + torch.repeat_interleave(self.b1, counts, dim=0)
        h = F.gelu(h)
        y = torch._grouped_mm(h, self.w2, offs)  # type: ignore[attr-defined]
        if self.use_bias:
            y = y + torch.repeat_interleave(self.b2, counts, dim=0)
        return y

    def forward(self, x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Compute expert FFN over ``x`` grouped by local expert id.

        Args:
            x: ``(M, D)`` tokens in local-expert-sorted order (``M = counts.sum()``).
            counts: ``(L,)`` int64 token count per local expert.

        Returns:
            ``(M, D)`` expert outputs in the same order as ``x``.
        """
        if x.shape[0] == 0:
            return x  # no tokens routed to this rank's experts
        if self.backend == "grouped_mm":
            return self._forward_grouped_mm(x, counts)
        if self.backend == "bmm":
            return self._forward_bmm(x, counts)
        return self._forward_loop(x, counts)

    # ------------------------------------------------------------------
    # State-dict interop with the legacy per-expert layout.
    # ------------------------------------------------------------------

    def load_from_feedforward(self, ffn_state: dict, expert_idx: int) -> None:
        """Copy weights from a legacy :class:`FeedForward` state-dict into slot
        ``expert_idx`` of this module's grouped parameters.

        Legacy state_dict keys (``nn.Sequential(Linear, GELU, Linear)``):

        - ``ffn.0.weight``: shape ``(H, D)`` -> transpose to ``(D, H)`` into ``w1[slot]``
        - ``ffn.0.bias``:   shape ``(H,)``           -> ``b1[slot]``
        - ``ffn.2.weight``: shape ``(D, H)`` -> transpose to ``(H, D)`` into ``w2[slot]``
        - ``ffn.2.bias``:   shape ``(D,)``           -> ``b2[slot]``

        Args:
            ffn_state: State-dict of a single :class:`FeedForward` module.
            expert_idx: Local slot (0 .. num_local_experts-1) to write into.
        """
        with torch.no_grad():
            self.w1[expert_idx].copy_(ffn_state["ffn.0.weight"].transpose(0, 1))
            self.w2[expert_idx].copy_(ffn_state["ffn.2.weight"].transpose(0, 1))
            if self.use_bias:
                if "ffn.0.bias" in ffn_state:
                    self.b1[expert_idx].copy_(ffn_state["ffn.0.bias"])
                if "ffn.2.bias" in ffn_state:
                    self.b2[expert_idx].copy_(ffn_state["ffn.2.bias"])
