from typing import Optional

import torch

from .base import DriftMetric


class LatentL2Drift(DriftMetric):
    """Masked L2 drift in packed latent space.

    Computes mean L2 distance between source and edited latents over
    masked token positions, averaged across the batch dimension.
    """

    def compute(
        self,
        source_latent: torch.Tensor,
        edited_latents: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> float:
        # source_latent: (1, T, C), edited_latents: (B, T, C)
        diff = edited_latents - source_latent  # (B, T, C)

        if mask is not None:
            # mask: (T,) bool -> select only preserve-region tokens
            diff = diff[:, mask, :]  # (B, T_masked, C)

        if diff.numel() == 0:
            return 0.0

        # Per-token L2 norm, then mean over tokens and batch
        per_token_l2 = diff.norm(dim=-1)  # (B, T_masked)
        return per_token_l2.mean().item()
