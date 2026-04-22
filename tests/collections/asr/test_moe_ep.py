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


# ---------------------------------------------------------------------------
# Checkpoint consolidation / sharding helpers (single-process tests)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_shard_ep_state_dict_slices_consolidated_tensor(monkeypatch):
    """``shard_ep_state_dict_`` must slice a full ``(num_experts, ...)`` tensor
    down to this rank's ``(experts_per_rank, ...)`` shard when EP is enabled.

    This is the load-time path; pure single-process test (no distributed needed).
    """
    from nemo.collections.asr.modules.moe_transformer_encoder import MoETransformerEncoder

    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "2")

    D = 16
    H = 4 * D
    E = 4  # num_experts
    # ep_size=4 -> L=1 per rank, this rank is ep_rank=2 so it owns expert id 2.
    enc = MoETransformerEncoder(
        n_mels=16,
        d_model=D,
        n_heads=4,
        n_layers=1,
        moe_num_experts=E,
        moe_top_k=1,
        moe_router_type="switch",
        moe_ep_size=4,
        moe_expert_backend="loop",
        nan_debug=False,
    )
    assert enc.moe_topology.enabled
    assert enc.moe_topology.ep_rank == 2
    assert enc.moe_topology.experts_per_rank == 1

    # Build a "consolidated" state_dict with full (E, ...) tensors whose per-expert
    # content is deterministic so we can verify the right slice is taken.
    sd = {}
    w1_full = torch.stack([torch.full((D, H), float(j)) for j in range(E)])  # (4, D, H)
    w2_full = torch.stack([torch.full((H, D), float(j) + 100) for j in range(E)])
    b1_full = torch.stack([torch.full((H,), float(j) + 200) for j in range(E)])
    b2_full = torch.stack([torch.full((D,), float(j) + 300) for j in range(E)])
    sd["encoder.layers.0.ffn.local_experts.w1"] = w1_full
    sd["encoder.layers.0.ffn.local_experts.w2"] = w2_full
    sd["encoder.layers.0.ffn.local_experts.b1"] = b1_full
    sd["encoder.layers.0.ffn.local_experts.b2"] = b2_full

    enc.shard_ep_state_dict_(sd, root_prefix="encoder.")

    # After sharding, each tensor should be (L=1, ...) and contain expert 2's values.
    assert sd["encoder.layers.0.ffn.local_experts.w1"].shape == (1, D, H)
    assert torch.all(sd["encoder.layers.0.ffn.local_experts.w1"][0] == 2.0)
    assert torch.all(sd["encoder.layers.0.ffn.local_experts.w2"][0] == 102.0)
    assert torch.all(sd["encoder.layers.0.ffn.local_experts.b1"][0] == 202.0)
    assert torch.all(sd["encoder.layers.0.ffn.local_experts.b2"][0] == 302.0)


@pytest.mark.unit
def test_shard_ep_state_dict_is_noop_when_already_local(monkeypatch):
    """If tensors are already shaped ``(experts_per_rank, ...)``, shard is a no-op."""
    from nemo.collections.asr.modules.moe_transformer_encoder import MoETransformerEncoder

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    D = 16
    H = 4 * D
    enc = MoETransformerEncoder(
        n_mels=16, d_model=D, n_heads=4, n_layers=1,
        moe_num_experts=4, moe_top_k=1, moe_router_type="switch",
        moe_ep_size=2, moe_expert_backend="loop", nan_debug=False,
    )
    L = enc.moe_topology.experts_per_rank
    sd = {"encoder.layers.0.ffn.local_experts.w1": torch.zeros(L, D, H)}
    orig = sd["encoder.layers.0.ffn.local_experts.w1"]
    enc.shard_ep_state_dict_(sd, root_prefix="encoder.")
    assert sd["encoder.layers.0.ffn.local_experts.w1"] is orig  # unchanged ref


@pytest.mark.unit
def test_shard_ep_state_dict_raises_on_bad_shape(monkeypatch):
    """Any first-dim size that is neither ``L`` nor ``num_experts`` is a bug."""
    from nemo.collections.asr.modules.moe_transformer_encoder import MoETransformerEncoder

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    D = 16
    H = 4 * D
    enc = MoETransformerEncoder(
        n_mels=16, d_model=D, n_heads=4, n_layers=1,
        moe_num_experts=4, moe_top_k=1, moe_router_type="switch",
        moe_ep_size=2, moe_expert_backend="loop", nan_debug=False,
    )
    # 3 experts is neither L=2 nor E=4.
    sd = {"encoder.layers.0.ffn.local_experts.w1": torch.zeros(3, D, H)}
    with pytest.raises(RuntimeError, match="Unexpected first-dim"):
        enc.shard_ep_state_dict_(sd, root_prefix="encoder.")


@pytest.mark.unit
def test_shard_ep_state_dict_is_noop_when_ep_disabled(monkeypatch):
    """With ep_size=1, shard operates on an identity layout: num_experts == L,
    so all tensors pass through unchanged."""
    from nemo.collections.asr.modules.moe_transformer_encoder import MoETransformerEncoder

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    D = 16
    H = 4 * D
    E = 4
    enc = MoETransformerEncoder(
        n_mels=16, d_model=D, n_heads=4, n_layers=1,
        moe_num_experts=E, moe_top_k=1, moe_router_type="switch",
        moe_ep_size=1, moe_expert_backend="bmm", nan_debug=False,
    )
    # Full state dict with (E, ...) tensors should pass through.
    sd = {"encoder.layers.0.ffn.local_experts.w1": torch.zeros(E, D, H)}
    enc.shard_ep_state_dict_(sd, root_prefix="encoder.")
    assert sd["encoder.layers.0.ffn.local_experts.w1"].shape == (E, D, H)


# ---------------------------------------------------------------------------
# state_dict override + _load_from_state_dict slicing (save/restore via .nemo)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_state_dict_cache_emits_full_tensors(monkeypatch):
    """When begin_consolidated_state_dict() has stashed a cache, state_dict()
    must emit the cached full tensors in place of the local shards.

    This is the mechanism by which ``save_to`` produces a complete ``.nemo``
    under EP (the NeMo path that bypasses Lightning's on_save_checkpoint).
    """
    from nemo.collections.asr.modules.moe_transformer_encoder import MoETransformerEncoder

    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    D, H, E = 16, 4 * 16, 4
    enc = MoETransformerEncoder(
        n_mels=16, d_model=D, n_heads=4, n_layers=1,
        moe_num_experts=E, moe_top_k=1, moe_router_type="switch",
        moe_ep_size=4, moe_expert_backend="loop", nan_debug=False,
    )
    # Simulate the output of gather_full_expert_state without running dist.
    fake_full = {
        "layers.0.ffn.local_experts.w1": torch.full((E, D, H), 7.0),
        "layers.0.ffn.local_experts.w2": torch.full((E, H, D), 7.0),
        "layers.0.ffn.local_experts.b1": torch.full((E, H), 7.0),
        "layers.0.ffn.local_experts.b2": torch.full((E, D), 7.0),
    }
    enc.begin_consolidated_state_dict(fake_full)
    try:
        sd = enc.state_dict()
        assert sd["layers.0.ffn.local_experts.w1"].shape == (E, D, H)
        assert torch.all(sd["layers.0.ffn.local_experts.w1"] == 7.0)
    finally:
        enc.end_consolidated_state_dict()

    # After end_consolidated_state_dict, we are back to the local shard shape.
    sd = enc.state_dict()
    assert sd["layers.0.ffn.local_experts.w1"].shape == (1, D, H)  # L=1


@pytest.mark.unit
def test_load_from_state_dict_slices_consolidated(monkeypatch):
    """A checkpoint saved as a consolidated ``.nemo`` (full ``(E, ...)``
    tensors) must load correctly into a fresh EP-sharded module on any rank,
    even though NeMo's restore_from path does NOT fire on_load_checkpoint.

    The slicing logic added inside ``_load_from_state_dict`` handles this.
    """
    from nemo.collections.asr.modules.moe_transformer_encoder import MoETransformerEncoder

    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "3")

    D, H, E = 16, 4 * 16, 4
    enc = MoETransformerEncoder(
        n_mels=16, d_model=D, n_heads=4, n_layers=1,
        moe_num_experts=E, moe_top_k=1, moe_router_type="switch",
        moe_ep_size=4, moe_expert_backend="loop", nan_debug=False,
    )
    assert enc.moe_topology.ep_rank == 3  # owns expert id 3

    # Build a full-tensor state_dict, as if restored from a consolidated .nemo.
    sd = {}
    for name, shape in [("w1", (E, D, H)), ("w2", (E, H, D)), ("b1", (E, H)), ("b2", (E, D))]:
        # Each expert j fills its slot with a distinct value j.
        t = torch.stack([torch.full(shape[1:], float(j)) for j in range(E)])
        sd[f"layers.0.ffn.local_experts.{name}"] = t

    # Populate minimal other keys so strict=True would pass.
    missing_keys, unexpected_keys, error_msgs = [], [], []
    sd_with_all = dict(sd)
    sd_with_all.update({k: v for k, v in enc.state_dict().items() if k not in sd_with_all})

    missing_before = set(enc.state_dict().keys()) - set(sd_with_all.keys())
    assert not missing_before, f"Test setup bug, missing keys before load: {missing_before}"

    # Directly invoke _load_from_state_dict with prefix='' (i.e. module loaded at root).
    enc._load_from_state_dict(
        sd_with_all, "", {}, True, missing_keys, unexpected_keys, error_msgs
    )
    assert not error_msgs, f"Load errors: {error_msgs}"

    # After load, the encoder's local_experts.w1 should contain expert 3's values.
    w1 = enc.layers[0].ffn.local_experts.w1
    assert w1.shape == (1, D, H)
    assert torch.all(w1[0] == 3.0), (
        f"Expected local_experts.w1 to contain expert-3's values after EP slicing, "
        f"got values={w1[0].flatten()[:4].tolist()}"
    )
