from abc import ABC, abstractmethod
from typing import Optional

import torch


class DriftMetric(ABC):
    """Base class for drift metrics between source and edited latents."""

    @abstractmethod
    def compute(
        self,
        source_latent: torch.Tensor,
        edited_latents: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> float:
        """Compute drift between source and edited latents.

        Args:
            source_latent: Source latent, shape (1, num_tokens, channels).
            edited_latents: Edited latents, shape (B, num_tokens, channels).
            mask: Boolean mask, shape (num_tokens,). True = include in drift.
                  If None, all tokens are included.

        Returns:
            Scalar drift value.
        """
        ...
