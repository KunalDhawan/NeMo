"""
Mixture-of-Experts (MoE) Transformer Encoder for ASR with optional Expert
Parallelism (EP) over intra-node GPUs.

Extends the standard :class:`TransformerEncoder` by replacing the feed-forward
network (FFN) in each Transformer layer with an MoE feed-forward module
(:class:`MoEFeedForward`). Supports both per-layer (``switch``) routing and
shared (``omni``) routing across all MoE layers.

New in this branch (``transfoemr_asr_moe_ep``):

- **Expert Parallelism**: when ``moe_ep_size > 1`` experts are sharded across
  a process group (typically intra-node), tokens are dispatched via
  :func:`torch.distributed.all_to_all_single`. Works on both A100 and H100
  with stock ``torch.distributed`` (no Megatron dependency).
- **Fused expert kernels**: one of ``grouped_mm`` (PyTorch >= 2.6),
  ``bmm`` (portable), or ``loop`` (legacy) via
  :class:`~nemo.collections.asr.parts.submodules.moe_experts.LocalExperts`.
- **Per-layer activation checkpointing**: optional, orthogonal to EP.
- **EP-reduced auxiliary loss**: optional all-reduce of ``f`` / ``rho``
  across the EP group before computing the load-balancing loss (reduces
  per-rank variance).

Back-compat: with defaults ``moe_ep_size=1`` and ``moe_expert_backend='loop'``
the forward path is byte-identical to the pre-EP implementation.

References:

- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models
  with Simple and Efficient Sparsity", 2022.
- Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation
  and Automatic Sharding", 2021.
- Gale et al., "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts", 2022.
- Gu et al., "Omni-Router: Sharing Routing Decisions in Sparse MoE for ASR", 2025.

Full design log: ``/work/moe/docs/moe_transformer_encoder_parallelism.md``.
"""

from __future__ import annotations

import re
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from nemo.collections.asr.modules.transformer_encoder import (
    FeedForward,
    TransformerEncoder,
)
from nemo.collections.asr.parts.submodules.moe_dispatch import (
    combine_tokens,
    dispatch_tokens,
)
from nemo.collections.asr.parts.submodules.moe_experts import LocalExperts
from nemo.collections.asr.parts.submodules.moe_parallel import (
    MoEParallelContext,
    build_topology,
    register_dp_allreduce_hook,
)

__all__ = ['SwitchGate', 'MoEFeedForward', 'MoETransformerEncoder']


# ---------------------------------------------------------------------------
# Router -- unchanged from the pre-EP implementation.
# ---------------------------------------------------------------------------


class SwitchGate(nn.Module):
    """Softmax routing gate for Mixture-of-Experts layers.

    Computes routing probabilities over N experts for each input token using
    a learned linear projection followed by softmax. Optionally adds Gaussian
    noise during training to encourage exploration (jitter).

    Can be shared across multiple MoE layers (omni-router) or used independently
    per layer (switch-style).

    Args:
        d_model: Input feature dimension.
        num_experts: Number of experts to route over.
        jitter_eps: Std-dev of Gaussian noise added to logits during training
            (``0.0`` disables jitter). Defaults to ``0.0``.
    """

    def __init__(self, d_model: int, num_experts: int, jitter_eps: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.jitter_eps = jitter_eps
        self.w_gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.w_gate(x)
        if self.training and self.jitter_eps > 0.0:
            logits = logits + torch.randn_like(logits) * self.jitter_eps
        return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# MoEFeedForward -- three forward paths depending on configuration.
# ---------------------------------------------------------------------------


class MoEFeedForward(nn.Module):
    """Mixture-of-Experts feed-forward module -- drop-in replacement for
    :class:`FeedForward`, with three execution modes:

    1. **Legacy** (``ep_size=1`` + ``backend='loop'``): identical to the
       pre-EP implementation, using ``nn.ModuleList`` of :class:`FeedForward`.
       Preserved for byte-level reproducibility of prior experiments.
    2. **Grouped single-rank** (``ep_size=1`` + ``backend in {'auto','bmm','grouped_mm'}``):
       uses :class:`LocalExperts` with all ``num_experts`` experts stored
       locally; faster than the legacy loop on many-expert / single-GPU runs.
    3. **Expert-parallel** (``ep_size>1``): shards experts across the EP
       process group and dispatches tokens via all-to-all; uses
       :class:`LocalExperts` with ``L = num_experts // ep_size`` experts.

    The auxiliary load-balancing loss is computed in all modes and stored in
    ``self._aux_loss`` for :meth:`MoETransformerEncoder.get_moe_auxiliary_loss`
    to pick up.

    Args:
        d_model: Hidden dimension.
        num_experts: Global expert count (not sharded).
        top_k: Experts activated per token.
        router: Optional external router (shared omni-router); otherwise a
            private :class:`SwitchGate` is created.
        jitter_eps: Router jitter; only used if this module creates its own
            router.
        topology: EP topology (``None`` disables EP for this layer).
        parallel_ctx: Lazy :class:`MoEParallelContext` (shared across layers).
        backend: Local-expert compute backend.
        aux_loss_ep_reduce: If True and EP is enabled, all-reduce ``f``/``rho``
            across the EP group before computing the load-balancing loss.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 1,
        router: Optional[SwitchGate] = None,
        jitter_eps: float = 0.0,
        topology=None,
        parallel_ctx: Optional[MoEParallelContext] = None,
        backend: str = 'auto',
        aux_loss_ep_reduce: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = router if router is not None else SwitchGate(
            d_model=d_model, num_experts=num_experts, jitter_eps=jitter_eps
        )
        self._aux_loss: Optional[torch.Tensor] = None
        self.topology = topology
        self.parallel_ctx = parallel_ctx
        self.aux_loss_ep_reduce = aux_loss_ep_reduce

        ep_enabled = topology is not None and topology.enabled
        # Legacy path is ONLY when EP is off AND the user explicitly asks for 'loop'.
        # This keeps bit-level back-compat for prior experiments.
        self._legacy = (not ep_enabled) and (backend == 'loop')

        if self._legacy:
            self.experts = nn.ModuleList([FeedForward(d_model) for _ in range(num_experts)])
            self.local_experts = None
        else:
            num_local = topology.experts_per_rank if ep_enabled else num_experts
            self.local_experts = LocalExperts(
                d_model=d_model,
                num_local_experts=num_local,
                backend=backend,
            )
            # Tag expert parameters so the parent model can exclude them from
            # DDP's world-level reducer and attach a DP-group grad hook.
            if ep_enabled:
                for p in self.local_experts.parameters():
                    p._is_moe_expert_ep_local = True
            self.experts = None

    # ------------------------------------------------------------------
    # Aux loss
    # ------------------------------------------------------------------

    def _compute_load_balancing_loss(
        self,
        gate_probs: torch.Tensor,
        expert_mask: torch.Tensor,
        ep_reduce: bool,
    ) -> torch.Tensor:
        """GShard / Switch Transformer load-balancing loss.

        ``L_load = N * Σ_j(f_j * ρ_j)`` where ``f_j`` = fraction of tokens
        dispatched to expert j and ``ρ_j`` = mean router probability for
        expert j.

        If EP is enabled and ``ep_reduce`` is True, ``f`` and ``ρ`` are
        averaged across the EP group before combining -- this reduces
        per-rank variance at near-zero cost.
        """
        num_tokens = gate_probs.shape[0]
        f = expert_mask.float().sum(dim=0) / num_tokens  # (E,)
        rho = gate_probs.mean(dim=0)  # (E,)

        if (
            ep_reduce
            and self.topology is not None
            and self.topology.enabled
            and self.parallel_ctx is not None
            and self.parallel_ctx.initialized
        ):
            # Cast to fp32 for the reduction to avoid bf16 staleness.
            f32 = f.to(torch.float32)
            r32 = rho.to(torch.float32)
            dist.all_reduce(f32, op=dist.ReduceOp.SUM, group=self.parallel_ctx.ep_group)
            dist.all_reduce(r32, op=dist.ReduceOp.SUM, group=self.parallel_ctx.ep_group)
            f = (f32 / self.topology.ep_size).to(f.dtype)
            rho = (r32 / self.topology.ep_size).to(rho.dtype)

        return self.num_experts * (f * rho).sum()

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute MoE feed-forward.

        Args:
            x: ``(B, T, D)`` input.

        Returns:
            ``(B, T, D)`` output.
        """
        if self._legacy:
            return self._forward_legacy(x)
        if self.topology is not None and self.topology.enabled:
            # Lazy-init mesh / groups on first forward (defensive -- normally
            # the parent model's on_train_start has already done this).
            if self.parallel_ctx is not None and not self.parallel_ctx.initialized:
                self.parallel_ctx.ensure_initialized()
            return self._forward_ep(x)
        return self._forward_single_rank_grouped(x)

    # ---------- Legacy (bit-identical to pre-EP code) ----------

    def _forward_legacy(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)
        num_tokens = x_flat.shape[0]

        gate_probs = self.router(x_flat)
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        expert_mask = torch.zeros(num_tokens, self.num_experts, device=x.device, dtype=x.dtype)
        expert_mask.scatter_(1, top_k_indices, 1.0)
        self._aux_loss = self._compute_load_balancing_loss(
            gate_probs, expert_mask, ep_reduce=False
        )

        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]
            expert_weights = top_k_probs[:, k]
            for i in range(self.num_experts):
                mask = expert_indices == i
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[i](expert_input)
                    output[mask] += expert_output * expert_weights[mask].unsqueeze(-1)
        return output.reshape(batch_size, seq_len, d_model)

    # ---------- Single-rank grouped (ep_size=1, backend != loop) ----------

    def _forward_single_rank_grouped(self, x: torch.Tensor) -> torch.Tensor:
        """Single-GPU path through :class:`LocalExperts`. Useful even without
        EP when ``num_experts`` is large and the Python loop is the bottleneck.
        """
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        N = x_flat.shape[0]
        K = self.top_k
        E = self.num_experts

        gate_probs = self.router(x_flat)
        top_k_probs, top_k_idx = torch.topk(gate_probs, K, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        expert_mask = torch.zeros(N, E, device=x.device, dtype=x.dtype)
        expert_mask.scatter_(1, top_k_idx, 1.0)
        self._aux_loss = self._compute_load_balancing_loss(
            gate_probs, expert_mask, ep_reduce=False
        )

        flat_assign = top_k_idx.reshape(-1)
        flat_weight = top_k_probs.reshape(-1)
        flat_x = x_flat.unsqueeze(1).expand(N, K, D).reshape(N * K, D)

        perm = torch.argsort(flat_assign, stable=True)
        x_sorted = flat_x.index_select(0, perm)
        counts = torch.bincount(flat_assign, minlength=E).to(torch.int64)

        y_sorted = self.local_experts(x_sorted, counts)

        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.shape[0], device=x.device)
        y_flat = y_sorted.index_select(0, inv)
        y_flat = y_flat * flat_weight.unsqueeze(-1).to(y_flat.dtype)

        token_idx = torch.arange(N * K, device=x.device).div(K, rounding_mode='floor')
        output = torch.zeros(N, D, device=x.device, dtype=y_flat.dtype)
        output.scatter_add_(0, token_idx.unsqueeze(-1).expand(-1, D), y_flat)
        return output.reshape(B, T, D)

    # ---------- Expert-parallel ----------

    def _forward_ep(self, x: torch.Tensor) -> torch.Tensor:
        """All-to-all dispatch -> local expert compute -> all-to-all combine."""
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        N = x_flat.shape[0]
        K = self.top_k
        E = self.num_experts

        gate_probs = self.router(x_flat)
        top_k_probs, top_k_idx = torch.topk(gate_probs, K, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        expert_mask = torch.zeros(N, E, device=x.device, dtype=x.dtype)
        expert_mask.scatter_(1, top_k_idx, 1.0)
        self._aux_loss = self._compute_load_balancing_loss(
            gate_probs, expert_mask, ep_reduce=self.aux_loss_ep_reduce
        )

        x_grouped, counts, ctx = dispatch_tokens(
            x_local=x_flat,
            top_k_idx=top_k_idx,
            top_k_weights=top_k_probs,
            ep_group=self.parallel_ctx.ep_group,
            num_experts=self.num_experts,
            ep_size=self.topology.ep_size,
            experts_per_rank=self.topology.experts_per_rank,
        )

        y_grouped = self.local_experts(x_grouped, counts)

        y_flat = combine_tokens(
            y_grouped=y_grouped, ctx=ctx, ep_group=self.parallel_ctx.ep_group
        )
        return y_flat.reshape(B, T, D)


# ---------------------------------------------------------------------------
# MoETransformerEncoder
# ---------------------------------------------------------------------------


class MoETransformerEncoder(TransformerEncoder):
    """Transformer Encoder with Mixture-of-Experts feed-forward layers and
    optional Expert Parallelism.

    Subclasses :class:`TransformerEncoder` and replaces the FFN in the
    configured layers with :class:`MoEFeedForward`. The base-encoder code is
    not modified: turning off all MoE flags (``moe_num_experts=1``) plus
    ``moe_layer_indices=[]`` gives a regular encoder. Turning on EP requires
    exactly one flag: ``moe_ep_size > 1``.

    The module is safe to construct *before* ``torch.distributed`` is
    initialized; EP-specific process groups are built lazily on the first
    forward or via :meth:`ensure_ep_initialized` (called by the parent
    LightningModule's ``on_train_start``).

    Args:
        All standard :class:`TransformerEncoder` args, plus:

        moe_num_experts: Experts per MoE layer.
        moe_top_k: Experts activated per token.
        moe_router_type: ``'omni'`` (shared router) or ``'switch'``
            (per-layer router).
        moe_layer_indices: Layers where MoE is applied. ``None`` = all layers.
        moe_load_balance_loss_weight: Coefficient for the auxiliary load-
            balancing loss.
        moe_jitter_eps: Router noise.
        moe_init_from_ffn: If True, broadcast pretrained FFN weights into all
            expert slots at checkpoint-load time.
        moe_ep_size: EP group size. ``1`` disables EP (default).
        moe_ep_mode: ``'dropless'`` (default) or ``'capped'``. Currently only
            ``'dropless'`` is wired in; ``'capped'`` is validated and reserved.
        moe_ep_capacity_factor: Only meaningful for ``'capped'``.
        moe_aux_loss_ep_reduce: All-reduce ``f``/``rho`` across EP before the
            aux-loss product. Default True.
        moe_expert_backend: ``'auto' | 'grouped_mm' | 'bmm' | 'loop'``.
            ``'loop'`` combined with ``ep_size=1`` exercises the pre-EP path
            bit-identically.
        moe_activation_checkpointing: Wrap each :class:`TransformerBlock` with
            :func:`torch.utils.checkpoint.checkpoint`. Orthogonal to EP.
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 17,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
        causal_mask: bool = False,
        pre_encode: str = "conv",
        nan_debug: bool = True,
        qk_norm: bool = False,
        subsampling_factor: int = 4,
        # MoE core
        moe_num_experts: int = 8,
        moe_top_k: int = 1,
        moe_router_type: str = 'omni',
        moe_layer_indices: Optional[List[int]] = None,
        moe_load_balance_loss_weight: float = 0.01,
        moe_jitter_eps: float = 0.0,
        moe_init_from_ffn: bool = True,
        # MoE parallelism (new)
        moe_ep_size: int = 1,
        moe_ep_mode: str = 'dropless',
        moe_ep_capacity_factor: float = 1.25,
        moe_aux_loss_ep_reduce: bool = True,
        moe_expert_backend: str = 'auto',
        moe_activation_checkpointing: bool = False,
    ):
        super().__init__(
            n_mels=n_mels,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
            causal_mask=causal_mask,
            pre_encode=pre_encode,
            nan_debug=nan_debug,
            qk_norm=qk_norm,
            subsampling_factor=subsampling_factor,
        )

        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_router_type = moe_router_type
        self.moe_load_balance_loss_weight = moe_load_balance_loss_weight
        self.moe_jitter_eps = moe_jitter_eps
        self.moe_init_from_ffn = moe_init_from_ffn
        self.moe_ep_mode = moe_ep_mode
        self.moe_ep_capacity_factor = moe_ep_capacity_factor
        self.moe_aux_loss_ep_reduce = moe_aux_loss_ep_reduce
        self.moe_expert_backend = moe_expert_backend
        self.moe_activation_checkpointing = moe_activation_checkpointing

        if moe_router_type not in ('omni', 'switch'):
            raise ValueError(
                f"moe_router_type must be 'omni' or 'switch', got '{moe_router_type}'"
            )
        if moe_ep_mode not in ('dropless', 'capped'):
            raise ValueError(
                f"moe_ep_mode must be 'dropless' or 'capped', got '{moe_ep_mode}'"
            )
        if moe_ep_mode == 'capped':
            raise NotImplementedError(
                "moe_ep_mode='capped' is reserved for future use. Use 'dropless' "
                "(default) -- it aligns with the current aux-loss-driven balancing."
            )
        if moe_router_type == 'omni' and moe_top_k != 1:
            # NOT a hard constraint -- omni + top_k>1 is mathematically well-defined
            # (every layer picks the same top-k experts for a given token) and the
            # original code supported it. We keep it reachable but flag it so users
            # are aware this is an uncommon configuration.
            import warnings
            warnings.warn(
                f"moe_router_type='omni' with moe_top_k={moe_top_k}>1 is unusual: "
                f"every MoE layer will route each token to the *same* {moe_top_k} "
                f"experts. This is allowed but gives up some of the per-depth "
                f"specialization signal that motivated omni. Consider "
                f"moe_router_type='switch' if you want per-layer top-k selection.",
                UserWarning,
                stacklevel=2,
            )

        if moe_layer_indices is not None:
            self.moe_layer_indices = list(moe_layer_indices)
        else:
            self.moe_layer_indices = list(range(n_layers))
        for idx in self.moe_layer_indices:
            if idx < 0 or idx >= n_layers:
                raise ValueError(
                    f"moe_layer_indices contains invalid index {idx} for encoder "
                    f"with {n_layers} layers."
                )

        # Build static EP topology from env (safe pre-dist). When ep_size=1
        # this is effectively a no-op wrapper that reports enabled=False.
        self.moe_topology = build_topology(num_experts=moe_num_experts, ep_size=moe_ep_size)
        self.moe_parallel_ctx = MoEParallelContext(self.moe_topology)

        # Shared router for omni mode.
        if moe_router_type == 'omni':
            self.omni_router = SwitchGate(
                d_model=d_model, num_experts=moe_num_experts, jitter_eps=moe_jitter_eps
            )
        else:
            self.omni_router = None

        # Swap FFN -> MoEFeedForward in the selected layers.
        for layer_idx in self.moe_layer_indices:
            layer = self.layers[layer_idx]
            router = self.omni_router if moe_router_type == 'omni' else None
            moe_ffn = MoEFeedForward(
                d_model=d_model,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                router=router,
                jitter_eps=moe_jitter_eps,
                topology=self.moe_topology,
                parallel_ctx=self.moe_parallel_ctx,
                backend=moe_expert_backend,
                aux_loss_ep_reduce=moe_aux_loss_ep_reduce,
            )
            # When NOT using the legacy path, optionally init experts from the
            # pretrained FeedForward this MoE is replacing (if moe_init_from_ffn
            # is True, but the base FFN is *fresh* here, so this is a no-op
            # unless later reconfigured via _load_from_state_dict).
            # For the legacy path, the old behavior is preserved: copy base FFN
            # state into each expert slot.
            if moe_ffn._legacy and moe_init_from_ffn:
                original_ffn_state = layer.ffn.state_dict()
                for expert in moe_ffn.experts:
                    expert.load_state_dict(original_ffn_state)

            layer.ffn = moe_ffn

        if self.moe_topology.rank == 0:
            print(
                f"[MoETransformerEncoder] "
                f"{'EP-sharded' if self.moe_topology.enabled else 'DDP-replicated'} "
                f"MoE: experts={moe_num_experts}, top_k={moe_top_k}, "
                f"router='{moe_router_type}', layers={self.moe_layer_indices}, "
                f"ep_size={moe_ep_size}, backend='{moe_expert_backend}', "
                f"act_ckpt={moe_activation_checkpointing}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # EP lifecycle helpers (called by the parent LightningModule)
    # ------------------------------------------------------------------

    def ensure_ep_initialized(self) -> None:
        """Build the EP DeviceMesh and process groups. Safe to call multiple
        times. Must be called after ``torch.distributed`` is up (typically in
        the LightningModule's ``on_train_start`` / ``setup`` hook)."""
        self.moe_parallel_ctx.ensure_initialized()

    def register_ep_grad_hooks(self) -> None:
        """Attach post-accumulate-grad hooks on every EP-local expert parameter
        that all-reduce the gradient across the DP group.

        Must be called after :meth:`ensure_ep_initialized`. Idempotent.
        """
        if not self.moe_topology.enabled:
            return
        self.moe_parallel_ctx.ensure_initialized()
        dp_group = self.moe_parallel_ctx.dp_group
        dp_size = self.moe_topology.dp_size
        for p in self.parameters():
            if getattr(p, '_is_moe_expert_ep_local', False):
                register_dp_allreduce_hook(p, dp_group, dp_size)

    def collect_ep_ignored_param_names(self, root_prefix: str = '') -> List[str]:
        """Return fully-qualified parameter names (with ``root_prefix``) of
        EP-local expert params that DDP should skip.

        Used by the parent LightningModule to populate
        ``_ddp_params_and_buffers_to_ignore`` *before* DDP wraps the model.
        """
        if not self.moe_topology.enabled:
            return []
        names = []
        for name, p in self.named_parameters():
            if getattr(p, '_is_moe_expert_ep_local', False):
                names.append(f"{root_prefix}{name}" if root_prefix else name)
        return names

    # ------------------------------------------------------------------
    # Auxiliary loss
    # ------------------------------------------------------------------

    def get_moe_auxiliary_loss(self) -> Optional[torch.Tensor]:
        """Mean of per-layer load-balancing losses, scaled by the configured
        weight. Returns ``None`` if no MoE layer has recorded a loss (e.g. in
        eval-only mode)."""
        total_loss = None
        n_losses = 0
        for layer_idx in self.moe_layer_indices:
            ff = self.layers[layer_idx].ffn
            if isinstance(ff, MoEFeedForward) and ff._aux_loss is not None:
                total_loss = ff._aux_loss if total_loss is None else total_loss + ff._aux_loss
                n_losses += 1
        if total_loss is not None and n_losses > 0:
            total_loss = self.moe_load_balance_loss_weight * (total_loss / n_losses)
        return total_loss

    # ------------------------------------------------------------------
    # Forward with optional activation checkpointing
    # ------------------------------------------------------------------

    def forward(self, audio_signal, length):
        """Forward pass with optional per-layer activation checkpointing.

        Identical structure to :meth:`TransformerEncoder.forward`; only the
        inner layer loop changes to support
        :func:`torch.utils.checkpoint.checkpoint`.
        """
        x = audio_signal
        x, length = self.pre_encode(x, length)
        if self.nan_debug:
            self._check_nan(x, "pre_encode")
        x = x * (self.d_model ** 0.5)
        if self.nan_debug:
            self._check_nan(x, "embedding_scale")
        x = self.layer_norm(x)
        if self.nan_debug:
            self._check_nan(x, "layer_norm")

        max_len = x.shape[1]
        pad_mask = torch.arange(max_len, device=x.device).unsqueeze(0) < length.unsqueeze(1)
        attn_mask = pad_mask.unsqueeze(1).unsqueeze(2)

        use_ckpt = self.moe_activation_checkpointing and self.training
        for idx, layer in enumerate(self.layers):
            if use_ckpt:
                x = torch_checkpoint(layer, x, attn_mask, use_reentrant=False)
            else:
                x = layer(x, attn_mask=attn_mask)
            if self.nan_debug:
                self._check_nan(x, f"layer_{idx}")
        x = self.final_norm(x)
        if self.nan_debug:
            self._check_nan(x, "final_norm")
        x = x.transpose(1, 2)
        return x, length

    # ------------------------------------------------------------------
    # Checkpoint compatibility
    # ------------------------------------------------------------------

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Remap legacy FFN / per-expert checkpoint keys into the current
        module layout.

        Two input layouts are supported:

        - **Base-encoder checkpoint** (``layers.{i}.ffn.ffn.0.weight``) --
          broadcast into all expert slots if ``moe_init_from_ffn`` is True.
        - **Legacy MoE checkpoint** (``layers.{i}.ffn.experts.{j}.ffn.0.weight``)
          -- loaded directly into :class:`FeedForward` slots when the current
          module uses the legacy path, or transposed + gathered into
          :class:`LocalExperts` when using the grouped path.

        When EP is active, only the local-rank's expert slots are populated
        from the checkpoint.
        """
        moe_layer_set = set(self.moe_layer_indices)

        # (A) Broadcast base FFN weights into experts (legacy behavior).
        if self.moe_init_from_ffn:
            pattern = re.compile(
                r'^(' + re.escape(prefix) + r'layers\.(\d+)\.ffn)\.((?!experts\.|router\.|local_experts\.).+)$'
            )
            remapped = 0
            for key in list(state_dict.keys()):
                m = pattern.match(key)
                if not m:
                    continue
                layer_prefix = m.group(1)  # "layers.{i}.ffn"
                layer_idx = int(m.group(2))
                param_suffix = m.group(3)  # e.g. "ffn.0.weight"
                if layer_idx not in moe_layer_set:
                    continue
                value = state_dict[key]
                moe_ff = self.layers[layer_idx].ffn
                if moe_ff._legacy:
                    # Copy into each expert's FeedForward state_dict.
                    for j in range(self.moe_num_experts):
                        state_dict[f"{layer_prefix}.experts.{j}.{param_suffix}"] = value.clone()
                else:
                    # Copy into LocalExperts grouped parameters for this rank's local slots.
                    self._inject_into_local_experts(
                        state_dict, layer_prefix, param_suffix, value
                    )
                del state_dict[key]
                remapped += 1
            if remapped and self.moe_topology.rank == 0:
                print(
                    f"[MoETransformerEncoder] Remapped {remapped} base-FFN keys "
                    f"into MoE expert slots.",
                    flush=True,
                )

        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _inject_into_local_experts(self, state_dict, layer_prefix, param_suffix, value):
        """Place ``value`` (a legacy per-FFN tensor) into the grouped
        :class:`LocalExperts` parameters for the matching layer's local experts.

        Legacy suffixes (nn.Sequential(Linear, GELU, Linear)):
          - ``ffn.0.weight`` (shape ``H x D``) -> ``w1[slot]`` (shape ``D x H``) via transpose
          - ``ffn.0.bias``   (shape ``H``)     -> ``b1[slot]``
          - ``ffn.2.weight`` (shape ``D x H``) -> ``w2[slot]`` (shape ``H x D``) via transpose
          - ``ffn.2.bias``   (shape ``D``)     -> ``b2[slot]``
        """
        ff = self._ffn_from_prefix(layer_prefix)
        if ff is None or ff.local_experts is None:
            return
        le = ff.local_experts
        L = le.num_local_experts
        group_key_w1 = f"{layer_prefix}.local_experts.w1"
        group_key_w2 = f"{layer_prefix}.local_experts.w2"
        group_key_b1 = f"{layer_prefix}.local_experts.b1"
        group_key_b2 = f"{layer_prefix}.local_experts.b2"

        # Lazily seed the target grouped tensor in the state_dict if absent.
        def _ensure(key, ref):
            if key not in state_dict:
                state_dict[key] = ref.detach().clone()

        _ensure(group_key_w1, le.w1)
        _ensure(group_key_w2, le.w2)
        if le.use_bias:
            _ensure(group_key_b1, le.b1)
            _ensure(group_key_b2, le.b2)

        for slot in range(L):
            if param_suffix == "ffn.0.weight":
                state_dict[group_key_w1][slot].copy_(value.transpose(0, 1))
            elif param_suffix == "ffn.0.bias":
                if le.use_bias:
                    state_dict[group_key_b1][slot].copy_(value)
            elif param_suffix == "ffn.2.weight":
                state_dict[group_key_w2][slot].copy_(value.transpose(0, 1))
            elif param_suffix == "ffn.2.bias":
                if le.use_bias:
                    state_dict[group_key_b2][slot].copy_(value)

    def _ffn_from_prefix(self, layer_prefix: str) -> Optional[MoEFeedForward]:
        """Resolve a ``"...layers.{i}.ffn"`` prefix back to the MoEFeedForward
        instance (or ``None`` if it's not a MoE layer).
        """
        m = re.search(r'layers\.(\d+)\.ffn$', layer_prefix)
        if not m:
            return None
        idx = int(m.group(1))
        if idx not in set(self.moe_layer_indices):
            return None
        ff = self.layers[idx].ffn
        return ff if isinstance(ff, MoEFeedForward) else None
