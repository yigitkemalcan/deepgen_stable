"""Configuration for adaptive injection experiments."""

import argparse
from dataclasses import dataclass, field
from typing import List, Optional


MODES = ["original", "pd_adaptive", "pid_adaptive", "fixed_soft", "scheduled_fixed"]


@dataclass
class AdaptiveConfig:
    # --- Mode ---
    mode: str = "original"

    # --- Controller ---
    kp: float = 0.5
    kd: float = 0.1
    ki: float = 0.05
    target_drift: float = 0.1
    base_alpha: float = 0.8
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    integral_clamp: float = 2.0

    # --- Schedule (for scheduled_fixed mode) ---
    schedule_type: str = "linear"  # linear | cosine
    schedule_start: float = 1.0
    schedule_end: float = 0.0

    # --- Drift ---
    drift_metric: str = "latent_l2"

    # --- Mask ---
    mask_path: Optional[str] = None

    # --- Pipeline ---
    model_path: str = "black-forest-labs/FLUX.1-dev"
    hf_token: str = ""
    prompts: List[str] = field(default_factory=list)
    input_img_path: str = ""
    seed: int = 42
    device: str = "cuda"
    cpu_offload: bool = False
    num_inference_steps: int = 15
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 3.5

    # --- Output ---
    output_dir: str = "outputs/adaptive"
    log_steps: bool = True
    enable_metrics: bool = False

    @classmethod
    def from_args(cls) -> "AdaptiveConfig":
        parser = argparse.ArgumentParser(
            description="Adaptive injection control for Stable Flow"
        )

        # Mode
        parser.add_argument("--mode", type=str, default="original", choices=MODES)

        # Controller
        parser.add_argument("--kp", type=float, default=0.5)
        parser.add_argument("--kd", type=float, default=0.1)
        parser.add_argument("--ki", type=float, default=0.05)
        parser.add_argument("--target_drift", type=float, default=0.1)
        parser.add_argument("--base_alpha", type=float, default=0.8)
        parser.add_argument("--alpha_min", type=float, default=0.0)
        parser.add_argument("--alpha_max", type=float, default=1.0)
        parser.add_argument("--integral_clamp", type=float, default=2.0)

        # Schedule
        parser.add_argument("--schedule_type", type=str, default="linear",
                            choices=["linear", "cosine"])
        parser.add_argument("--schedule_start", type=float, default=1.0)
        parser.add_argument("--schedule_end", type=float, default=0.0)

        # Drift
        parser.add_argument("--drift_metric", type=str, default="latent_l2")

        # Mask
        parser.add_argument("--mask_path", type=str, default=None)

        # Pipeline
        parser.add_argument("--model_path", type=str, default="black-forest-labs/FLUX.1-dev")
        parser.add_argument("--hf_token", type=str, required=True)
        parser.add_argument("--prompts", type=str, nargs="+", required=True)
        parser.add_argument("--input_img_path", type=str, required=True)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--cpu_offload", action="store_true")
        parser.add_argument("--num_inference_steps", type=int, default=15)
        parser.add_argument("--height", type=int, default=1024)
        parser.add_argument("--width", type=int, default=1024)
        parser.add_argument("--guidance_scale", type=float, default=3.5)

        # Output
        parser.add_argument("--output_dir", type=str, default="outputs/adaptive")
        parser.add_argument("--no_log_steps", action="store_true")
        parser.add_argument("--enable_metrics", action="store_true")

        args = parser.parse_args()

        return cls(
            mode=args.mode,
            kp=args.kp,
            kd=args.kd,
            ki=args.ki,
            target_drift=args.target_drift,
            base_alpha=args.base_alpha,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            integral_clamp=args.integral_clamp,
            schedule_type=args.schedule_type,
            schedule_start=args.schedule_start,
            schedule_end=args.schedule_end,
            drift_metric=args.drift_metric,
            mask_path=args.mask_path,
            model_path=args.model_path,
            hf_token=args.hf_token,
            prompts=args.prompts,
            input_img_path=args.input_img_path,
            seed=args.seed,
            device=args.device,
            cpu_offload=args.cpu_offload,
            num_inference_steps=args.num_inference_steps,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            output_dir=args.output_dir,
            log_steps=not args.no_log_steps,
            enable_metrics=args.enable_metrics,
        )
