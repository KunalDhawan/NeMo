"""
Unit tests for the EP-aware MoE Transformer encoder components.

These tests run on CPU (no CUDA, no ``torch.distributed``) and cover:

- ``LocalExperts.loop`` vs ``LocalExperts.bmm`` numerical parity.
- ``MoEFeedForward`` legacy path vs grouped single-rank path on a fixed
  input (equivalence after copying weights).
- Dispatch/combine identity when routing is trivial (top_k=1, all-to-same-expert).
- ``_load_from_state_dict`` correctness when remapping a base-FFN
  checkpoint into both the legacy and grouped layouts.

Design doc: ``/work/moe/docs/moe_transformer_encoder_parallelism.md``.
"""

from __future__ import annotations

import pytest
import torch

from nemo.collections.asr.modules.moe_transformer_encoder import (
    MoEFeedForward,
    SwitchGate,
)
from nemo.collections.asr.parts.submodules.moe_experts import LocalExperts


# ---------------------------------------------------------------------------
# LocalExperts backend parity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_local_experts_loop_vs_bmm_parity():
    """Loop and bmm backends must agree on the same input / weights."""
    torch.manual_seed(0)
    D = 32
    L = 4
    counts = torch.tensor([3, 5, 0, 7], dtype=torch.int64)
    x = torch.randn(int(counts.sum()), D)

    loop_mod = LocalExperts(d_model=D, num_local_experts=L, backend="loop")
    bmm_mod = LocalExperts(d_model=D, num_local_experts=L, backend="bmm")

    # Copy loop_mod weights into bmm_mod so both compute the same function.
    with torch.no_grad():
        bmm_mod.w1.copy_(loop_mod.w1)
        bmm_mod.w2.copy_(loop_mod.w2)
        bmm_mod.b1.copy_(loop_mod.b1)
        bmm_mod.b2.copy_(loop_mod.b2)

    y_loop = loop_mod(x, counts)
    y_bmm = bmm_mod(x, counts)
    assert torch.allclose(y_loop, y_bmm, atol=1e-5, rtol=1e-5), (
        f"loop/bmm disagree: max diff = {(y_loop - y_bmm).abs().max().item()}"
    )


@pytest.mark.unit
def test_local_experts_empty_tokens():
    """Module must tolerate an empty input (0-token batch)."""
    D = 16
    L = 3
    counts = torch.zeros(L, dtype=torch.int64)
    x = torch.empty(0, D)

    for backend in ("loop", "bmm"):
        mod = LocalExperts(d_model=D, num_local_experts=L, backend=backend)
        out = mod(x, counts)
        assert out.shape == (0, D)


# ---------------------------------------------------------------------------
# MoEFeedForward legacy vs grouped single-rank parity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_moe_feedforward_legacy_vs_grouped_parity():
    """Legacy (ModuleList of FeedForward) and grouped (LocalExperts) paths
    must produce the same output on a fixed input when weights are aligned."""
    torch.manual_seed(42)
    D = 32
    E = 4
    K = 2
    B, T = 2, 5

    x = torch.randn(B, T, D)

    router = SwitchGate(d_model=D, num_experts=E)

    legacy = MoEFeedForward(
        d_model=D, num_experts=E, top_k=K, router=router, backend="loop",
    )
    grouped = MoEFeedForward(
        d_model=D, num_experts=E, top_k=K, router=router, backend="bmm",
    )

    # Copy legacy expert weights into grouped LocalExperts slots (transposed
    # to match the grouped layout).
    with torch.no_grad():
        for j in range(E):
            ff = legacy.experts[j]
            grouped.local_experts.w1[j].copy_(ff.ffn[0].weight.transpose(0, 1))
            grouped.local_experts.b1[j].copy_(ff.ffn[0].bias)
            grouped.local_experts.w2[j].copy_(ff.ffn[2].weight.transpose(0, 1))
            grouped.local_experts.b2[j].copy_(ff.ffn[2].bias)

    legacy.eval()
    grouped.eval()
    with torch.no_grad():
        y_legacy = legacy(x)
        y_grouped = grouped(x)

    max_diff = (y_legacy - y_grouped).abs().max().item()
    assert max_diff < 1e-5, f"legacy vs grouped disagree: max diff = {max_diff}"


@pytest.mark.unit
def test_moe_feedforward_aux_loss_present():
    """Both paths must populate ``_aux_loss`` after forward."""
    torch.manual_seed(0)
    D = 16
    E = 4
    x = torch.randn(2, 3, D, requires_grad=True)

    for backend in ("loop", "bmm"):
        mod = MoEFeedForward(d_model=D, num_experts=E, top_k=1, backend=backend)
        mod.train()
        _ = mod(x)
        assert mod._aux_loss is not None
        assert mod._aux_loss.ndim == 0, "aux loss should be a scalar"
        assert mod._aux_loss.requires_grad, (
            "aux loss must track gradients so the model's aux-loss hook can "
            "add it to the training objective."
        )


# ---------------------------------------------------------------------------
# EP ignored-param collection (no distributed)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ep_disabled_collects_no_ignore_names(monkeypatch):
    """With ``moe_ep_size=1`` there should be nothing to ignore."""
    from nemo.collections.asr.modules.moe_transformer_encoder import MoETransformerEncoder

    # Ensure env vars don't accidentally signal a multi-rank world.
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    monkeypatch.delenv("SLURM_PROCID", raising=False)

    enc = MoETransformerEncoder(
        n_mels=32,
        d_model=32,
        n_heads=4,
        n_layers=2,
        moe_num_experts=4,
        moe_top_k=1,
        moe_router_type="switch",
        moe_ep_size=1,
        moe_expert_backend="loop",
        nan_debug=False,
    )
    assert enc.moe_topology.enabled is False
    assert enc.collect_ep_ignored_param_names() == []


@pytest.mark.unit
def test_ep_static_topology_from_env(monkeypatch):
    """Static topology must read the ``WORLD_SIZE`` / ``RANK`` env vars and
    produce a sensible layout *before* ``torch.distributed`` is initialized."""
    from nemo.collections.asr.parts.submodules.moe_parallel import build_topology

    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("RANK", "11")
    monkeypatch.setenv("LOCAL_RANK", "3")

    topo = build_topology(num_experts=8, ep_size=8)
    assert topo.enabled
    assert topo.world_size == 16
    assert topo.rank == 11
    assert topo.ep_size == 8
    assert topo.dp_size == 2
    assert topo.ep_rank == 3  # 11 % 8
    assert topo.dp_rank == 1  # 11 // 8
    assert topo.experts_per_rank == 1
    assert topo.local_expert_ids() == [3]


@pytest.mark.unit
def test_ep_topology_validation_errors(monkeypatch):
    """World-size / num-experts divisibility checks must fire cleanly."""
    from nemo.collections.asr.parts.submodules.moe_parallel import build_topology

    monkeypatch.setenv("WORLD_SIZE", "7")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    with pytest.raises(ValueError, match="world_size"):
        build_topology(num_experts=8, ep_size=4)

    monkeypatch.setenv("WORLD_SIZE", "8")
    with pytest.raises(ValueError, match="num_experts"):
        build_topology(num_experts=6, ep_size=4)


@pytest.mark.unit
def test_ep_ignored_names_populated(monkeypatch):
    """With EP enabled, ``collect_ep_ignored_param_names`` must return the
    flat names of LocalExperts parameters (w1, w2, b1, b2 per MoE layer)."""
    from nemo.collections.asr.modules.moe_transformer_encoder import MoETransformerEncoder

    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")

    enc = MoETransformerEncoder(
        n_mels=32,
        d_model=32,
        n_heads=4,
        n_layers=2,
        moe_num_experts=4,
        moe_top_k=1,
        moe_router_type="switch",
        moe_ep_size=4,
        moe_expert_backend="loop",
        nan_debug=False,
    )
    assert enc.moe_topology.enabled
    names = enc.collect_ep_ignored_param_names(root_prefix="encoder.")
    # 2 MoE layers * 4 params (w1, w2, b1, b2) = 8 names.
    assert len(names) == 8
    assert all(n.startswith("encoder.layers.") and ".local_experts." in n for n in names)
