"""Experiment runner for adaptive injection control in Stable Flow.

All runs use real image editing (inversion + editing). An input image is required.

Supports multiple modes:
  - original:        Unchanged Stable Flow (hard K/V copy)
  - pd_adaptive:     PD-controlled injection strength
  - pid_adaptive:    PID-controlled injection strength
  - fixed_soft:      Constant soft blending alpha
  - scheduled_fixed: Time-varying alpha (linear/cosine schedule)

Usage:
    # Original Stable Flow
    python -m experiments.run_adaptive --mode original \\
        --hf_token YOUR_TOKEN \\
        --input_img_path inputs/bottle.jpg \\
        --prompts "A photo of a bottle" "A photo of a bottle next to an apple"

    # PD adaptive
    python -m experiments.run_adaptive --mode pd_adaptive \\
        --hf_token YOUR_TOKEN \\
        --input_img_path inputs/bottle.jpg \\
        --prompts "A photo of a bottle" "A photo of a bottle next to an apple" \\
        --kp 0.5 --kd 0.1 --target_drift 0.1
"""

import os

import numpy as np
import torch
from diffusers import FluxPipeline
from PIL import Image

from .config import AdaptiveConfig
from .adapter import TransformerAdapter
from .callback import AdaptiveCallback
from .controllers.pd_controller import PDController
from .controllers.pid_controller import PIDController
from .controllers.fixed import (
    FixedAlphaController,
    ScheduledAlphaController,
    linear_decay,
    cosine_decay,
)
from .drift.latent_drift import LatentL2Drift
from .logging_utils import StepLogger
from .mask_utils import load_mask, default_mask


# Vital layers from the Stable Flow paper
MULTIMODAL_VITAL_LAYERS = [0, 1, 17, 18]
SINGLE_MODAL_VITAL_LAYERS = list(np.array([28, 53, 54, 56, 25]) - 19)

# Real image editing uses 50 inversion/editing steps with these guidance scales
INVERSION_STEPS = 50
INVERSION_GUIDANCE = 1
EDITING_GUIDANCE_SOURCE = 1
EDITING_GUIDANCE_EDITS = 3


def build_controller(cfg: AdaptiveConfig):
    """Create the appropriate controller based on config mode."""
    if cfg.mode == "pd_adaptive":
        return PDController(
            kp=cfg.kp,
            kd=cfg.kd,
            target_drift=cfg.target_drift,
            base_alpha=cfg.base_alpha,
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
        )
    elif cfg.mode == "pid_adaptive":
        return PIDController(
            kp=cfg.kp,
            ki=cfg.ki,
            kd=cfg.kd,
            target_drift=cfg.target_drift,
            base_alpha=cfg.base_alpha,
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
            integral_clamp=cfg.integral_clamp,
        )
    elif cfg.mode == "fixed_soft":
        return FixedAlphaController(alpha=cfg.base_alpha)
    elif cfg.mode == "scheduled_fixed":
        if cfg.schedule_type == "cosine":
            schedule_fn = cosine_decay(start=cfg.schedule_start, end=cfg.schedule_end)
        else:
            schedule_fn = linear_decay(start=cfg.schedule_start, end=cfg.schedule_end)
        return ScheduledAlphaController(
            schedule_fn=schedule_fn,
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
        )
    else:
        raise ValueError(f"Unknown mode for controller: {cfg.mode}")


def build_drift_metric(cfg: AdaptiveConfig):
    """Create drift metric based on config."""
    if cfg.drift_metric == "latent_l2":
        return LatentL2Drift()
    raise ValueError(f"Unknown drift metric: {cfg.drift_metric}")


def load_pipeline(cfg: AdaptiveConfig) -> FluxPipeline:
    """Load the FluxPipeline (same as original run_stable_flow.py)."""
    pipe = FluxPipeline.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.float16,
        visualize_attention=False,
        token=cfg.hf_token,
    )
    if cfg.cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(cfg.device)
    return pipe


def image2latent(pipe, image, device="cuda", latent_nudging_scalar=1.15):
    """Encode a PIL image to packed FLUX latent space (from run_stable_flow.py)."""
    image = pipe.image_processor.preprocess(image).type(pipe.vae.dtype).to(device)
    latents = pipe.vae.encode(image)["latent_dist"].mean
    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    latents = latents * latent_nudging_scalar
    latents = pipe._pack_latents(
        latents=latents,
        batch_size=1,
        num_channels_latents=16,
        height=128,
        width=128,
    )
    return latents


def save_images(images, output_dir, prefix="result"):
    """Save a list of PIL images both individually and as a horizontal strip."""
    os.makedirs(output_dir, exist_ok=True)
    arrays = [np.array(img) for img in images]
    strip = Image.fromarray(np.hstack(arrays))
    strip.save(os.path.join(output_dir, f"{prefix}.jpg"))
    for i, img in enumerate(images):
        img.save(os.path.join(output_dir, f"{prefix}_{i}.jpg"))


def invert_image(pipe, cfg: AdaptiveConfig):
    """Run inversion pass to get latent trajectory from real image."""
    print("Inverting image...")
    inversion_prompt = cfg.prompts[0:1]
    inverted_latent_list = pipe(
        inversion_prompt,
        height=cfg.height,
        width=cfg.width,
        guidance_scale=INVERSION_GUIDANCE,
        output_type="pil",
        num_inference_steps=INVERSION_STEPS,
        max_sequence_length=512,
        latents=image2latent(pipe, Image.open(cfg.input_img_path), cfg.device),
        invert_image=True,
    )
    return inverted_latent_list


@torch.no_grad()
def run_original(pipe, cfg: AdaptiveConfig):
    """Run original Stable Flow (identical to run_stable_flow.py invert_and_save)."""
    prompts = cfg.prompts
    inverted_latent_list = invert_image(pipe, cfg)

    print("Editing with original Stable Flow...")
    images = pipe(
        prompts,
        height=cfg.height,
        width=cfg.width,
        guidance_scale=[EDITING_GUIDANCE_SOURCE] + [EDITING_GUIDANCE_EDITS] * (len(prompts) - 1),
        output_type="pil",
        num_inference_steps=INVERSION_STEPS,
        max_sequence_length=512,
        latents=inverted_latent_list[-1].tile(len(prompts), 1, 1),
        inverted_latent_list=inverted_latent_list,
        mm_copy_blocks=MULTIMODAL_VITAL_LAYERS,
        single_copy_blocks=SINGLE_MODAL_VITAL_LAYERS,
    ).images
    return images


@torch.no_grad()
def run_adaptive(pipe, cfg: AdaptiveConfig):
    """Run adaptive injection control experiment with real image editing."""
    prompts = cfg.prompts

    # Inversion pass (no adapter needed)
    inverted_latent_list = invert_image(pipe, cfg)

    # Install adapter for editing pass
    adapter = TransformerAdapter(pipe.transformer)
    adapter.install()

    try:
        controller = build_controller(cfg)
        controller.reset()
        drift_metric = build_drift_metric(cfg)

        if cfg.mask_path is not None:
            mask = load_mask(cfg.mask_path, cfg.height, cfg.width, cfg.device)
        else:
            mask = default_mask(device=cfg.device)

        logger = None
        if cfg.log_steps:
            run_dir = os.path.join(cfg.output_dir, cfg.mode)
            logger = StepLogger(run_dir)

        adapter.set_alpha(cfg.base_alpha)

        callback = AdaptiveCallback(
            controller=controller,
            drift_metric=drift_metric,
            adapter=adapter,
            mask=mask,
            logger=logger,
        )
        callback.total_steps = INVERSION_STEPS

        print(f"Editing with {cfg.mode} controller...")
        images = pipe(
            prompts,
            height=cfg.height,
            width=cfg.width,
            guidance_scale=[EDITING_GUIDANCE_SOURCE] + [EDITING_GUIDANCE_EDITS] * (len(prompts) - 1),
            output_type="pil",
            num_inference_steps=INVERSION_STEPS,
            max_sequence_length=512,
            latents=inverted_latent_list[-1].tile(len(prompts), 1, 1),
            inverted_latent_list=inverted_latent_list,
            mm_copy_blocks=MULTIMODAL_VITAL_LAYERS,
            single_copy_blocks=SINGLE_MODAL_VITAL_LAYERS,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images

        if logger is not None:
            logger.finalize()

        return images
    finally:
        adapter.uninstall()


def run_metrics(images, cfg: AdaptiveConfig):
    """Optionally compute evaluation metrics."""
    if not cfg.enable_metrics:
        return

    if cfg.mask_path is None:
        print("No --mask_path provided. Computing full-image metrics.")

    from .metrics import masked_l2, masked_lpips, load_evaluation_mask

    source_img = np.array(images[0])
    eval_mask = None
    if cfg.mask_path is not None:
        eval_mask = load_evaluation_mask(cfg.mask_path, source_img.shape[0], source_img.shape[1])

    metrics_dir = os.path.join(cfg.output_dir, cfg.mode)
    os.makedirs(metrics_dir, exist_ok=True)

    import json
    results = {}
    for i, img in enumerate(images[1:], start=1):
        edited_img = np.array(img)
        l2 = masked_l2(source_img, edited_img, eval_mask)
        lpips_score = masked_lpips(source_img, edited_img, eval_mask)
        results[f"edit_{i}"] = {"masked_l2": l2, "masked_lpips": lpips_score}
        print(f"  Edit {i}: L2={l2:.4f}, LPIPS={lpips_score:.4f}")

    with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)


def main():
    cfg = AdaptiveConfig.from_args()

    print(f"Mode: {cfg.mode}")
    print(f"Input image: {cfg.input_img_path}")
    print(f"Prompts: {cfg.prompts}")

    pipe = load_pipeline(cfg)

    if cfg.mode == "original":
        images = run_original(pipe, cfg)
    else:
        images = run_adaptive(pipe, cfg)

    run_dir = os.path.join(cfg.output_dir, cfg.mode)
    save_images(images, run_dir)
    print(f"Images saved to {run_dir}/")

    run_metrics(images, cfg)


if __name__ == "__main__":
    main()
