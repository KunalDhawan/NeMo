"""
Hybrid RNNT-CTC BPE model with Mixture-of-Experts (MoE) encoder auxiliary loss
and optional Expert Parallelism wiring.

Same pattern as :class:`EncDecMoERNNTBPEModel` -- collects MoE aux loss,
populates ``_ddp_params_and_buffers_to_ignore`` for EP-local expert params at
``__init__``, and registers DP-group grad hooks at :meth:`on_train_start`.

No Megatron dependency; uses only ``torch.distributed``.
"""

from __future__ import annotations

import torch

from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel
from nemo.collections.asr.models.moe_rnnt_bpe_models import (
    _apply_moe_ep_ddp_ignore,
    _consolidate_moe_ep_checkpoint,
    _register_moe_ep_grad_hooks,
    _save_to_with_moe_ep_consolidation,
    _shard_moe_ep_checkpoint,
)

__all__ = ['EncDecMoEHybridRNNTCTCBPEModel']


class EncDecMoEHybridRNNTCTCBPEModel(EncDecHybridRNNTCTCBPEModel):
    """Hybrid RNNT-CTC BPE model with MoE encoder auxiliary loss + optional
    Expert Parallelism wiring.

    Inherits all functionality from :class:`EncDecHybridRNNTCTCBPEModel`.
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

        See :class:`EncDecMoERNNTBPEModel.save_to` for details.
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
