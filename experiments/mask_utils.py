"""Mask utilities for loading and converting masks to packed latent-token space."""

from typing import Optional

import torch
import numpy as np
from PIL import Image


def load_mask(
    path: str,
    height: int = 1024,
    width: int = 1024,
    device: str = "cuda",
) -> torch.Tensor:
    """Load a binary mask image and convert to packed token-space boolean tensor.

    The mask image should be: white (255) = preserve, black (0) = edit.

    For a 1024x1024 image with FLUX packing:
      image (1024x1024) -> VAE latent (128x128) -> 2x2 packed (64x64) -> flat (4096,)

    Args:
        path: Path to mask image.
        height: Image height in pixels.
        width: Image width in pixels.
        device: Target device.

    Returns:
        Boolean tensor of shape (num_tokens,) where True = preserve.
    """
    mask_img = Image.open(path).convert("L")

    # Resize to VAE latent resolution (height/8, width/8 = 128x128 for 1024x1024)
    latent_h = height // 8
    latent_w = width // 8
    mask_img = mask_img.resize((latent_w, latent_h), Image.NEAREST)

    mask_np = np.array(mask_img) > 127  # binary: True = preserve

    # FLUX packing: 2x2 patches from (latent_h, latent_w) to (latent_h//2, latent_w//2)
    # A patch is "preserve" if ANY pixel in the 2x2 block is preserve
    mask_np = mask_np.reshape(latent_h // 2, 2, latent_w // 2, 2)
    mask_np = mask_np.any(axis=(1, 3))  # (64, 64)

    # Flatten to token dimension
    mask_flat = mask_np.reshape(-1)  # (4096,)
    return torch.tensor(mask_flat, dtype=torch.bool, device=device)


def default_mask(num_tokens: int = 4096, device: str = "cuda") -> torch.Tensor:
    """All-preserve mask (measure drift everywhere)."""
    return torch.ones(num_tokens, dtype=torch.bool, device=device)
