"""Head-to-head comparison: Original Stable Flow vs Adaptive methods.

Runs the original pipeline and several adaptive configurations on the same
input image, computes metrics, saves side-by-side visuals, and generates
an HTML report showing which method performed best.

Usage:
    python -m experiments.run_comparison \
        --hf_token YOUR_TOKEN \
        --input_img_path inputs/cat.jpg \
        --prompts "A photo of a cat" "A photo of a cat wearing sunglasses" \
        --output_dir outputs/comparison

    # With a mask (white=preserve, black=edit region):
    python -m experiments.run_comparison \
        --hf_token YOUR_TOKEN \
        --input_img_path inputs/cat.jpg \
        --prompts "A photo of a cat" "A photo of a cat wearing sunglasses" \
        --mask_path inputs/cat_mask.png \
        --output_dir outputs/comparison

    # With CPU offload (for T4 / low VRAM):
    python -m experiments.run_comparison \
        --hf_token YOUR_TOKEN \
        --input_img_path inputs/cat.jpg \
        --prompts "A photo of a cat" "A photo of a cat wearing sunglasses" \
        --cpu_offload \
        --output_dir outputs/comparison
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .run_adaptive import (
    MULTIMODAL_VITAL_LAYERS,
    SINGLE_MODAL_VITAL_LAYERS,
    INVERSION_STEPS,
    INVERSION_GUIDANCE,
    EDITING_GUIDANCE_SOURCE,
    EDITING_GUIDANCE_EDITS,
    load_pipeline,
    image2latent,
    invert_image,
    run_original,
    run_adaptive,
    build_controller,
    build_drift_metric,
)
from .config import AdaptiveConfig
from .adapter import TransformerAdapter
from .callback import AdaptiveCallback
from .logging_utils import StepLogger
from .mask_utils import load_mask, default_mask
from .metrics import masked_l2, masked_lpips, load_evaluation_mask


# ── Experiment configurations ──────────────────────────────────────────────
# Each entry defines a method to run.  "original" has no controller params.
# Add/remove entries here to change what gets compared.

EXPERIMENT_CONFIGS = [
    {
        "name": "original",
        "mode": "original",
        "label": "Original Stable Flow",
    },
    {
        "name": "pd_conservative",
        "mode": "pd_adaptive",
        "label": "PD Adaptive (conservative)",
        "kp": 0.3,
        "kd": 0.1,
        "target_drift": 0.3,
        "base_alpha": 0.6,
    },
    {
        "name": "pd_aggressive",
        "mode": "pd_adaptive",
        "label": "PD Adaptive (aggressive)",
        "kp": 0.5,
        "kd": 0.1,
        "target_drift": 0.5,
        "base_alpha": 0.4,
    },
    {
        "name": "cosine_schedule",
        "mode": "scheduled_fixed",
        "label": "Cosine Schedule (1.0 -> 0.1)",
        "schedule_type": "cosine",
        "schedule_start": 1.0,
        "schedule_end": 0.1,
    },
    {
        "name": "fixed_soft_05",
        "mode": "fixed_soft",
        "label": "Fixed Alpha = 0.5",
        "base_alpha": 0.5,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare original Stable Flow vs adaptive methods"
    )
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--input_img_path", type=str, required=True)
    parser.add_argument("--prompts", type=str, nargs="+", required=True)
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs/comparison")
    parser.add_argument("--model_path", type=str, default="black-forest-labs/FLUX.1-dev")
    return parser.parse_args()


def build_config_for_experiment(args, exp: dict) -> AdaptiveConfig:
    """Create an AdaptiveConfig from CLI args + experiment dict overrides."""
    cfg = AdaptiveConfig(
        mode=exp["mode"],
        hf_token=args.hf_token,
        input_img_path=args.input_img_path,
        prompts=list(args.prompts),
        mask_path=args.mask_path,
        seed=args.seed,
        device=args.device,
        cpu_offload=args.cpu_offload,
        model_path=args.model_path,
        output_dir=os.path.join(args.output_dir, exp["name"]),
        log_steps=True,
        enable_metrics=True,
    )
    # Apply experiment-specific overrides
    for key in ("kp", "kd", "ki", "target_drift", "base_alpha",
                "schedule_type", "schedule_start", "schedule_end",
                "alpha_min", "alpha_max"):
        if key in exp:
            setattr(cfg, key, exp[key])
    return cfg


def compute_metrics(source_img, edited_img, eval_mask=None):
    """Compute all metrics for a single source-edit pair."""
    l2 = masked_l2(source_img, edited_img, eval_mask)
    lpips_val = masked_lpips(source_img, edited_img, eval_mask)
    return {"l2": l2, "lpips": lpips_val}


def compute_edit_strength(input_img_array, edited_img_array):
    """Measure how much the edit changed the image (full-image L2)."""
    diff = input_img_array.astype(np.float32) - edited_img_array.astype(np.float32)
    return float(np.sqrt((diff ** 2).sum(axis=-1)).mean())


def add_label_to_image(img, label, font_size=28):
    """Add a text label at the top of an image."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    x = (img.width - text_w) // 2
    # Draw shadow then text
    draw.text((x + 1, 11), label, fill="black", font=font)
    draw.text((x, 10), label, fill="white", font=font)
    return img


def save_side_by_side(input_image, all_results, output_dir):
    """Save a single image with input + all method results side by side."""
    # For each method, take the first edit (index 1)
    images = [input_image.copy()]
    labels = ["Input"]

    for result in all_results:
        if len(result["images"]) > 1:
            images.append(result["images"][1])
            labels.append(result["label"])

    # Resize all to same height
    h = min(img.height for img in images)
    resized = []
    for img, label in zip(images, labels):
        ratio = h / img.height
        w = int(img.width * ratio)
        img_resized = img.resize((w, h), Image.LANCZOS)
        add_label_to_image(img_resized, label)
        resized.append(img_resized)

    total_w = sum(img.width for img in resized)
    strip = Image.new("RGB", (total_w, h))
    x_offset = 0
    for img in resized:
        strip.paste(img, (x_offset, 0))
        x_offset += img.width

    path = os.path.join(output_dir, "side_by_side.jpg")
    strip.save(path, quality=95)
    print(f"Side-by-side saved to {path}")
    return path


def generate_report(args, all_results, output_dir):
    """Generate an HTML report comparing all methods."""
    report_data = {
        "input_image": os.path.basename(args.input_img_path),
        "source_prompt": args.prompts[0],
        "edit_prompts": args.prompts[1:],
        "methods": [],
    }

    for result in all_results:
        method_data = {
            "name": result["name"],
            "label": result["label"],
            "mode": result["mode"],
            "time_seconds": result["time"],
            "metrics": result.get("metrics", {}),
            "edit_strength": result.get("edit_strength", {}),
        }
        if "config_params" in result:
            method_data["params"] = result["config_params"]
        report_data["methods"].append(method_data)

    # Save raw JSON
    json_path = os.path.join(output_dir, "report.json")
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    # Generate HTML report
    html = _build_html_report(report_data, output_dir)
    html_path = os.path.join(output_dir, "report.html")
    with open(html_path, "w") as f:
        f.write(html)

    print(f"Report saved to {html_path}")
    return report_data


def _build_html_report(data, output_dir):
    """Build an HTML report string."""
    methods = data["methods"]

    # Find best method per metric for each edit
    all_edit_keys = set()
    for m in methods:
        all_edit_keys.update(m.get("metrics", {}).keys())

    # Collect values for ranking
    # Lower L2 in preserve region = better preservation
    # Lower LPIPS in preserve region = better preservation
    # Higher edit_strength = stronger edit
    best = {}
    for edit_key in all_edit_keys:
        l2_vals = [(m["name"], m["metrics"].get(edit_key, {}).get("l2", float("inf"))) for m in methods]
        lpips_vals = [(m["name"], m["metrics"].get(edit_key, {}).get("lpips", float("inf"))) for m in methods]
        strength_vals = [(m["name"], m.get("edit_strength", {}).get(edit_key, 0)) for m in methods]

        best[f"{edit_key}_l2"] = min(l2_vals, key=lambda x: x[1])[0]
        best[f"{edit_key}_lpips"] = min(lpips_vals, key=lambda x: x[1])[0]
        best[f"{edit_key}_strength"] = max(strength_vals, key=lambda x: x[1])[0]

    edit_prompts_str = ", ".join(f'"{p}"' for p in data["edit_prompts"])

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Stable Flow Comparison Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 1200px; margin: 40px auto; padding: 0 20px;
           background: #f8f9fa; color: #333; }}
    h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
    h2 {{ color: #16213e; margin-top: 30px; }}
    .info {{ background: #e8eaf6; padding: 15px; border-radius: 8px; margin: 15px 0; }}
    .info strong {{ color: #1a237e; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0;
             background: white; border-radius: 8px; overflow: hidden;
             box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
    th {{ background: #1a1a2e; color: white; padding: 12px 16px; text-align: left; }}
    td {{ padding: 10px 16px; border-bottom: 1px solid #eee; }}
    tr:hover td {{ background: #f5f5f5; }}
    .best {{ background: #c8e6c9 !important; font-weight: bold; }}
    .worst {{ background: #ffcdd2 !important; }}
    .metric-note {{ font-size: 0.85em; color: #666; margin-top: 5px; }}
    .side-by-side {{ max-width: 100%; border-radius: 8px;
                     box-shadow: 0 2px 8px rgba(0,0,0,0.15); margin: 15px 0; }}
    .verdict {{ background: #e8f5e9; padding: 20px; border-radius: 8px;
                border-left: 4px solid #4caf50; margin: 20px 0; }}
    .verdict h3 {{ margin-top: 0; color: #2e7d32; }}
    .params {{ font-family: monospace; font-size: 0.9em; color: #555; }}
</style>
</head>
<body>
<h1>Stable Flow vs Adaptive: Comparison Report</h1>

<div class="info">
    <strong>Input Image:</strong> {data["input_image"]}<br>
    <strong>Source Prompt:</strong> "{data["source_prompt"]}"<br>
    <strong>Edit Prompt(s):</strong> {edit_prompts_str}
</div>

<h2>Side-by-Side Results</h2>
<img class="side-by-side" src="side_by_side.jpg" alt="Side by side comparison">
"""

    # Per-method images
    html += "<h2>Individual Results</h2>\n<table><tr>"
    for m in methods:
        html += f'<th>{m["label"]}</th>'
    html += "</tr><tr>"
    for m in methods:
        img_path = f'{m["name"]}/result.jpg'
        html += f'<td><img src="{img_path}" style="max-width:100%"></td>'
    html += "</tr></table>\n"

    # Metrics table
    html += """<h2>Metrics</h2>
<p class="metric-note">
    <b>Preserve L2 / LPIPS:</b> Lower = better background preservation (measures how much the
    non-edit region changed from source). <b>Edit Strength:</b> Higher = stronger edit applied
    (full-image L2 between input and output).
    Green = best, Red = worst for each metric.
</p>\n"""

    for edit_key in sorted(all_edit_keys):
        edit_idx = edit_key.replace("edit_", "")
        if len(data["edit_prompts"]) > int(edit_idx) - 1:
            prompt = data["edit_prompts"][int(edit_idx) - 1]
        else:
            prompt = f"Edit {edit_idx}"

        html += f'<h3>Edit {edit_idx}: "{prompt}"</h3>\n'
        html += "<table><tr><th>Method</th><th>Preserve L2</th><th>Preserve LPIPS</th>"
        html += "<th>Edit Strength (L2)</th><th>Time (s)</th></tr>\n"

        # Collect for worst marking
        l2_vals = []
        lpips_vals = []
        strength_vals = []
        for m in methods:
            metrics = m.get("metrics", {}).get(edit_key, {})
            l2_vals.append((m["name"], metrics.get("l2", float("inf"))))
            lpips_vals.append((m["name"], metrics.get("lpips", float("inf"))))
            strength_vals.append((m["name"], m.get("edit_strength", {}).get(edit_key, 0)))

        worst_l2 = max(l2_vals, key=lambda x: x[1])[0]
        worst_lpips = max(lpips_vals, key=lambda x: x[1])[0]
        worst_strength = min(strength_vals, key=lambda x: x[1])[0]

        for m in methods:
            metrics = m.get("metrics", {}).get(edit_key, {})
            l2_val = metrics.get("l2", -1)
            lpips_val = metrics.get("lpips", -1)
            strength = m.get("edit_strength", {}).get(edit_key, -1)

            l2_class = "best" if m["name"] == best.get(f"{edit_key}_l2") else ("worst" if m["name"] == worst_l2 else "")
            lpips_class = "best" if m["name"] == best.get(f"{edit_key}_lpips") else ("worst" if m["name"] == worst_lpips else "")
            strength_class = "best" if m["name"] == best.get(f"{edit_key}_strength") else ("worst" if m["name"] == worst_strength else "")

            params_str = ""
            if m.get("params"):
                params_str = f'<br><span class="params">{m["params"]}</span>'

            html += f'<tr><td>{m["label"]}{params_str}</td>'
            html += f'<td class="{l2_class}">{l2_val:.4f}</td>'
            html += f'<td class="{lpips_class}">{lpips_val:.4f}</td>'
            html += f'<td class="{strength_class}">{strength:.4f}</td>'
            html += f'<td>{m["time_seconds"]:.1f}</td></tr>\n'

        html += "</table>\n"

    # Verdict
    html += _build_verdict(methods, all_edit_keys, best)

    html += "</body></html>"
    return html


def _build_verdict(methods, edit_keys, best):
    """Build a verdict section summarizing which method won."""
    # Count wins per method
    wins = {}
    for m in methods:
        wins[m["name"]] = {"label": m["label"], "preserve": 0, "edit": 0}

    for edit_key in edit_keys:
        for suffix, category in [("_l2", "preserve"), ("_lpips", "preserve"), ("_strength", "edit")]:
            winner = best.get(f"{edit_key}{suffix}")
            if winner and winner in wins:
                wins[winner][category] += 1

    html = '<div class="verdict"><h3>Verdict</h3><table>'
    html += "<tr><th>Method</th><th>Preservation Wins</th><th>Edit Strength Wins</th><th>Total Wins</th></tr>\n"

    ranked = sorted(wins.items(), key=lambda x: x[1]["preserve"] + x[1]["edit"], reverse=True)
    for name, w in ranked:
        total = w["preserve"] + w["edit"]
        html += f'<tr><td>{w["label"]}</td><td>{w["preserve"]}</td><td>{w["edit"]}</td><td><b>{total}</b></td></tr>\n'

    html += "</table>"

    if ranked:
        winner = ranked[0]
        html += f'<p><b>Overall winner: {winner[1]["label"]}</b> '
        html += f'with {winner[1]["preserve"]} preservation wins '
        html += f'and {winner[1]["edit"]} edit strength wins.</p>'

    html += "</div>"
    return html


def run_single_experiment(pipe, args, exp, inverted_latent_list, input_image_array, eval_mask):
    """Run a single experiment config and return results."""
    cfg = build_config_for_experiment(args, exp)
    name = exp["name"]
    label = exp["label"]
    mode = exp["mode"]

    print(f"\n{'='*60}")
    print(f"Running: {label} ({mode})")
    print(f"{'='*60}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    t0 = time.time()

    if mode == "original":
        # Run original using the shared inverted latents
        prompts = cfg.prompts
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
    else:
        # Run adaptive using shared inverted latents
        prompts = cfg.prompts
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

            logger = StepLogger(cfg.output_dir)
            adapter.set_alpha(cfg.base_alpha)

            callback = AdaptiveCallback(
                controller=controller,
                drift_metric=drift_metric,
                adapter=adapter,
                mask=mask,
                logger=logger,
            )
            callback.total_steps = INVERSION_STEPS

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

            logger.finalize()
        finally:
            adapter.uninstall()

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    # Save images
    arrays = [np.array(img) for img in images]
    strip = Image.fromarray(np.hstack(arrays))
    strip.save(os.path.join(cfg.output_dir, "result.jpg"), quality=95)
    for i, img in enumerate(images):
        img.save(os.path.join(cfg.output_dir, f"result_{i}.jpg"), quality=95)

    # Compute metrics for each edit
    source_array = arrays[0]
    metrics = {}
    edit_strength = {}
    for i, edited_array in enumerate(arrays[1:], start=1):
        edit_key = f"edit_{i}"
        metrics[edit_key] = compute_metrics(source_array, edited_array, eval_mask)
        edit_strength[edit_key] = compute_edit_strength(input_image_array, edited_array)
        print(f"  Edit {i}: preserve_L2={metrics[edit_key]['l2']:.4f}, "
              f"preserve_LPIPS={metrics[edit_key]['lpips']:.4f}, "
              f"edit_strength={edit_strength[edit_key]:.4f}")

    # Save metrics
    with open(os.path.join(cfg.output_dir, "metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "edit_strength": edit_strength}, f, indent=2)

    # Config params for the report
    config_params = {}
    if mode != "original":
        for key in ("kp", "kd", "ki", "target_drift", "base_alpha",
                    "schedule_type", "schedule_start", "schedule_end"):
            if key in exp:
                config_params[key] = exp[key]

    return {
        "name": name,
        "label": label,
        "mode": mode,
        "images": images,
        "metrics": metrics,
        "edit_strength": edit_strength,
        "time": elapsed,
        "config_params": config_params,
    }


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Input: {args.input_img_path}")
    print(f"Prompts: {args.prompts}")
    print(f"Output: {args.output_dir}")
    print(f"Methods to compare: {len(EXPERIMENT_CONFIGS)}")

    # Load pipeline once
    print("\nLoading pipeline...")
    cfg_for_pipeline = AdaptiveConfig(
        model_path=args.model_path,
        hf_token=args.hf_token,
        device=args.device,
        cpu_offload=args.cpu_offload,
    )
    pipe = load_pipeline(cfg_for_pipeline)

    # Load and prepare input image
    input_image = Image.open(args.input_img_path)
    input_image_array = np.array(input_image)

    # Prepare evaluation mask
    eval_mask = None
    if args.mask_path is not None:
        eval_mask = load_evaluation_mask(args.mask_path, input_image.height, input_image.width)
        print(f"Using mask from {args.mask_path}")
    else:
        print("No mask provided - metrics computed on full image")

    # Run inversion ONCE (shared across all experiments)
    print("\nInverting image (shared across all methods)...")
    cfg_for_inversion = AdaptiveConfig(
        hf_token=args.hf_token,
        input_img_path=args.input_img_path,
        prompts=list(args.prompts),
        device=args.device,
        cpu_offload=args.cpu_offload,
    )
    inverted_latent_list = invert_image(pipe, cfg_for_inversion)
    print("Inversion complete.")

    # Run all experiments
    all_results = []
    for exp in EXPERIMENT_CONFIGS:
        result = run_single_experiment(
            pipe, args, exp, inverted_latent_list, input_image_array, eval_mask
        )
        all_results.append(result)
        torch.cuda.empty_cache()

    # Save side-by-side comparison
    save_side_by_side(input_image, all_results, args.output_dir)

    # Generate report
    report = generate_report(args, all_results, args.output_dir)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for m in report["methods"]:
        print(f"\n{m['label']}:")
        print(f"  Time: {m['time_seconds']:.1f}s")
        for edit_key, met in m.get("metrics", {}).items():
            strength = m.get("edit_strength", {}).get(edit_key, -1)
            print(f"  {edit_key}: L2={met['l2']:.4f}, LPIPS={met['lpips']:.4f}, "
                  f"Edit Strength={strength:.4f}")

    print(f"\nFull report: {os.path.join(args.output_dir, 'report.html')}")
    print(f"Side-by-side: {os.path.join(args.output_dir, 'side_by_side.jpg')}")


if __name__ == "__main__":
    main()
