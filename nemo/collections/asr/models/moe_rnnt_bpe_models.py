"""
RNNT BPE model with Mixture-of-Experts (MoE) encoder auxiliary loss support.

Extends :class:`EncDecRNNTBPEModel` to:

1. Collect and add the MoE load-balancing auxiliary loss from
   :class:`MoETransformerEncoder` during training (as before).
2. When EP (Expert Parallelism) is enabled on the encoder, exclude EP-local
   expert parameters from DDP's world-level gradient reducer and register a
   post-accumulate-grad hook on each that all-reduces their gradients across
   the DP sub-group. See the companion design doc
   ``/work/moe/docs/moe_transformer_encoder_parallelism.md`` §7 for
   rationale.

Lifecycle:

- ``__init__``: collects fully-qualified expert-parameter names from the
  encoder and writes them to ``self._ddp_params_and_buffers_to_ignore`` so
  PyTorch DDP skips them when it wraps the model during strategy setup.
- ``on_train_start``: initializes the EP process group (requires
  ``torch.distributed`` to be up) and registers DP-group grad hooks on each
  expert parameter. Idempotent.

No Megatron dependency; uses only ``torch.distributed``.
"""

from __future__ import annotations

import torch

from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel

__all__ = ['EncDecMoERNNTBPEModel']


class EncDecMoERNNTBPEModel(EncDecRNNTBPEModel):
    """Encoder-Decoder RNNT BPE model with MoE encoder auxiliary loss +
    optional Expert Parallelism wiring.

    Inherits all functionality from :class:`EncDecRNNTBPEModel` and adds:

    - :meth:`add_auxiliary_losses`: injects the MoE load-balancing loss.
    - DDP-ignore list population in ``__init__`` (no-op if EP is disabled).
    - EP hook registration in :meth:`on_train_start`.
    - **Checkpoint consolidation in :meth:`on_save_checkpoint`**: gathers all
      ranks' EP-local expert shards into full tensors on rank 0 before the
      checkpoint is written, so a single ``.nemo`` contains the complete MoE
      weights rather than just rank 0's shard.
    - **Checkpoint sharding in :meth:`on_load_checkpoint`**: slices the full
      expert tensors back down to this rank's local shard before they are
      applied to the model.

    Use this model class together with :class:`MoETransformerEncoder`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _apply_moe_ep_ddp_ignore(self)

    def on_train_start(self) -> None:
        super().on_train_start()
        _register_moe_ep_grad_hooks(self)

    def on_save_checkpoint(self, checkpoint) -> None:
        super().on_save_checkpoint(checkpoint)
        _consolidate_moe_ep_checkpoint(self, checkpoint)

    def on_load_checkpoint(self, checkpoint) -> None:
        _shard_moe_ep_checkpoint(self, checkpoint)
        super().on_load_checkpoint(checkpoint)

    def save_to(self, save_path: str):
        """Save a ``.nemo`` archive with consolidated MoE expert weights.

        Overrides :meth:`ModelPT.save_to` to run an all-gather across the EP
        group before the rank-0-only tar write. See
        :func:`_save_to_with_moe_ep_consolidation` for details.
        """
        return _save_to_with_moe_ep_consolidation(self, save_path)

    def add_auxiliary_losses(self, loss: torch.Tensor, reset_registry: bool = False) -> torch.Tensor:
        """Add auxiliary losses, including the MoE load-balancing loss.

        Args:
            loss: The primary loss value.
            reset_registry: Whether to reset the :class:`AccessMixin` registry.

        Returns:
            Loss tensor with auxiliary losses added.
        """
        loss = super().add_auxiliary_losses(loss, reset_registry=reset_registry)

        if hasattr(self.encoder, 'get_moe_auxiliary_loss'):
            moe_loss = self.encoder.get_moe_auxiliary_loss()
            if moe_loss is not None and moe_loss.requires_grad:
                loss = loss + moe_loss
                self.log('moe_aux_loss', moe_loss.detach())

        return loss


# ---------------------------------------------------------------------------
# Shared helpers (also imported by the Hybrid RNNT-CTC MoE model).
# ---------------------------------------------------------------------------


def _apply_moe_ep_ddp_ignore(pl_module) -> None:
    """If the encoder has EP enabled, populate
    ``pl_module._ddp_params_and_buffers_to_ignore`` with the fully-qualified
    names of EP-local expert parameters.

    PyTorch's DDP reads this attribute on the wrapped module and skips those
    parameters in its reducer. This is how we get DDP to leave the EP-sharded
    expert weights alone (they are only replicated across the DP sub-group,
    and their gradient sync is handled by the hook registered in
    :func:`_register_moe_ep_grad_hooks`).

    Must be called in ``__init__`` so the attribute is present before
    Lightning's ``DDPStrategy`` wraps the model.
    """
    if not hasattr(pl_module, 'encoder'):
        return
    encoder = pl_module.encoder
    if not hasattr(encoder, 'collect_ep_ignored_param_names'):
        return
    ignore_names = encoder.collect_ep_ignored_param_names(root_prefix='encoder.')
    if not ignore_names:
        return
    # Merge with anything a parent class may have set (paranoia).
    existing = list(getattr(pl_module, '_ddp_params_and_buffers_to_ignore', []) or [])
    pl_module._ddp_params_and_buffers_to_ignore = list(dict.fromkeys(existing + ignore_names))


def _register_moe_ep_grad_hooks(pl_module) -> None:
    """Initialize the EP DeviceMesh / process groups and attach DP-group
    all-reduce hooks on every EP-local expert parameter.

    Must be called after ``torch.distributed`` is initialized (e.g. in
    ``on_train_start``). Idempotent.
    """
    if not hasattr(pl_module, 'encoder'):
        return
    encoder = pl_module.encoder
    if not hasattr(encoder, 'register_ep_grad_hooks'):
        return
    encoder.ensure_ep_initialized()
    encoder.register_ep_grad_hooks()


def _consolidate_moe_ep_checkpoint(pl_module, checkpoint) -> None:
    """On save, all-gather EP-local expert shards into full tensors so the
    written ``.nemo`` / ``.ckpt`` contains every expert, not just rank 0's
    slice.

    Must run on every rank (it issues collective ops). Internally a no-op
    when ``moe_ep_size == 1``. Modifies ``checkpoint['state_dict']`` in place.

    Without this hook, training with ``moe_ep_size > 1`` produces checkpoints
    that contain only ``experts_per_rank`` of ``num_experts`` MoE weights per
    layer (i.e. rank 0's shard). Those checkpoints are unusable for eval or
    resumption, so this hook is required for any EP training run.
    """
    if not hasattr(pl_module, 'encoder'):
        return
    encoder = pl_module.encoder
    if not hasattr(encoder, 'consolidate_ep_state_dict_'):
        return
    if 'state_dict' not in checkpoint or not isinstance(checkpoint['state_dict'], dict):
        return
    encoder.consolidate_ep_state_dict_(checkpoint['state_dict'], root_prefix='encoder.')


def _shard_moe_ep_checkpoint(pl_module, checkpoint) -> None:
    """On load, slice full ``(num_experts, ...)`` expert tensors down to this
    rank's local ``(experts_per_rank, ...)`` shard before Lightning applies
    the state_dict to the model.

    Safe to call when EP is disabled -- becomes a no-op.
    """
    if not hasattr(pl_module, 'encoder'):
        return
    encoder = pl_module.encoder
    if not hasattr(encoder, 'shard_ep_state_dict_'):
        return
    if 'state_dict' not in checkpoint or not isinstance(checkpoint['state_dict'], dict):
        return
    encoder.shard_ep_state_dict_(checkpoint['state_dict'], root_prefix='encoder.')


def _save_to_with_moe_ep_consolidation(pl_module, save_path: str):
    """Implementation of :meth:`ModelPT.save_to` that first gathers every
    EP-local expert shard across the EP group and emits full tensors via the
    encoder's ``state_dict()`` override, so the ``.nemo`` archive written on
    rank 0 contains complete MoE weights rather than only rank 0's shard.

    Why this is needed in addition to ``on_save_checkpoint``:

    NeMo's ``.nemo`` packaging path (``ModelPT.save_to`` ->
    ``SaveRestoreConnector.save_to``) runs only on rank 0 and does NOT fire
    Lightning's ``on_save_checkpoint`` hook. Without this wrapper the
    consolidation that already protects the Lightning ``.ckpt`` path would
    not reach the ``.nemo`` file.

    Call from within an override of ``save_to`` on the MoE model class.
    """
    import torch.distributed as dist  # local import to keep module load cheap

    encoder = getattr(pl_module, 'encoder', None)
    topo = getattr(encoder, 'moe_topology', None) if encoder is not None else None

    # Fast path: no MoE encoder, or EP disabled -- fall through unchanged.
    if encoder is None or topo is None or not topo.enabled:
        # Call the immediate parent's save_to (not ours), i.e. ModelPT.save_to
        # via super() in the model class.
        return _call_parent_save_to(pl_module, save_path)

    # All ranks: run the collective gather so every rank can finish it.
    full_expert_state = encoder.gather_full_expert_state()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Activate the cache on ALL ranks (non-rank-0 won't actually write but
    # it is harmless for them to have the cache set; simplifies reasoning).
    encoder.begin_consolidated_state_dict(full_expert_state)
    try:
        result = _call_parent_save_to(pl_module, save_path)
    finally:
        encoder.end_consolidated_state_dict()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    return result


def _call_parent_save_to(pl_module, save_path: str):
    """Invoke :meth:`ModelPT.save_to` (the immediate parent) on ``pl_module``.

    Uses MRO lookup rather than ``super()`` so the helper can be shared
    between the RNNT and Hybrid RNNT-CTC MoE model classes without either
    knowing the other's hierarchy.
    """
    from nemo.core.classes.modelPT import ModelPT
    return ModelPT.save_to(pl_module, save_path)
