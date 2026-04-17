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

    Use this model class together with :class:`MoETransformerEncoder`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _apply_moe_ep_ddp_ignore(self)

    def on_train_start(self) -> None:
        super().on_train_start()
        _register_moe_ep_grad_hooks(self)

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
