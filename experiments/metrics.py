"""Optional evaluation metrics for comparing original vs adaptive runs."""

from typing import Optional

import numpy as np
from PIL import Image


def masked_l2(
    img_source: np.ndarray,
    img_edited: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute L2 distance between source and edited images in masked regions.

    Args:
        img_source: Source image as uint8 array (H, W, 3).
        img_edited: Edited image as uint8 array (H, W, 3).
        mask: Binary mask (H, W), True = region to measure. If None, full image.

    Returns:
        Mean L2 per pixel in the masked region.
    """
    diff = img_source.astype(np.float32) - img_edited.astype(np.float32)
    per_pixel_l2 = np.sqrt((diff ** 2).sum(axis=-1))  # (H, W)

    if mask is not None:
        if per_pixel_l2[mask].size == 0:
            return 0.0
        return float(per_pixel_l2[mask].mean())
    return float(per_pixel_l2.mean())


def masked_lpips(
    img_source: np.ndarray,
    img_edited: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Compute LPIPS between source and edited images, optionally masked.

    Requires the `lpips` package. Returns -1.0 if not available.

    Args:
        img_source: Source image as uint8 array (H, W, 3).
        img_edited: Edited image as uint8 array (H, W, 3).
        mask: Binary mask (H, W), True = region to measure.
              Applied by zeroing out non-masked regions before LPIPS.

    Returns:
        LPIPS score, or -1.0 if lpips package is not installed.
    """
    try:
        import lpips
        import torch
    except ImportError:
        return -1.0

    loss_fn = lpips.LPIPS(net="alex", verbose=False)

    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        # (H,W,3) uint8 -> (1,3,H,W) float in [-1,1]
        t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        return t * 2 - 1

    src_t = _to_tensor(img_source)
    edt_t = _to_tensor(img_edited)

    if mask is not None:
        mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        src_t = src_t * mask_t
        edt_t = edt_t * mask_t

    with torch.no_grad():
        score = loss_fn(src_t, edt_t)
    return float(score.item())


def load_evaluation_mask(
    mask_path: str,
    height: int,
    width: int,
) -> np.ndarray:
    """Load mask image and resize to target dimensions.

    White = preserve (True in returned array).
    """
    mask_img = Image.open(mask_path).convert("L").resize((width, height), Image.NEAREST)
    return np.array(mask_img) > 127
