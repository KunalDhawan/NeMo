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
    _register_moe_ep_grad_hooks,
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
