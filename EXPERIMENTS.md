# Adaptive Injection Control for Stable Flow

Experimental extension that adds closed-loop control of source-to-target attention injection strength in Stable Flow, using a drift signal to regulate preservation vs editability during image editing.

The original Stable Flow implementation is **completely untouched**. All experimental code lives in `experiments/`.

## Prerequisites

**HuggingFace token (required):** The model (`black-forest-labs/FLUX.1-dev`) is gated. You must:
1. Create a token at https://huggingface.co/settings/tokens
2. Accept the FLUX.1-dev license at https://huggingface.co/black-forest-labs/FLUX.1-dev
3. Pass the token via `--hf_token` in every run

## How It Works

All runs edit a **real input image**. The pipeline:
1. **Inverts** the input image into the latent noise trajectory (50 steps)
2. **Edits** by denoising from the inverted noise with new prompts, injecting source K/V at vital layers

The first prompt must **describe the input image**. The remaining prompts describe the desired edits.

In adaptive modes, a controller modulates the injection strength `alpha` at each editing step based on how much the preserve-regions drift from the source:
- `alpha = 1.0` is identical to original Stable Flow (hard K/V copy)
- `alpha = 0.0` means no injection (free editing)
- The PD controller increases alpha when drift is high, decreases it when drift is low

## Quick Start

### Original Stable Flow (unchanged)

```bash
python run_stable_flow.py \
    --hf_token YOUR_TOKEN \
    --input_img_path inputs/bottle.jpg \
    --prompts "A photo of a bottle" "A photo of a bottle next to an apple"
```

### Original via experiment runner

```bash
python -m experiments.run_adaptive --mode original \
    --hf_token YOUR_TOKEN \
    --input_img_path inputs/bottle.jpg \
    --prompts "A photo of a bottle" "A photo of a bottle next to an apple"
```

### Adaptive PD Controller

```bash
python -m experiments.run_adaptive --mode pd_adaptive \
    --hf_token YOUR_TOKEN \
    --input_img_path inputs/bottle.jpg \
    --prompts "A photo of a bottle" "A photo of a bottle next to an apple" \
    --kp 0.5 --kd 0.1 --target_drift 0.1 --base_alpha 0.8
```

### Fixed Soft Blending (constant alpha baseline)

```bash
python -m experiments.run_adaptive --mode fixed_soft \
    --hf_token YOUR_TOKEN \
    --input_img_path inputs/bottle.jpg \
    --prompts "A photo of a bottle" "A photo of a bottle next to an apple" \
    --base_alpha 0.7
```

### Scheduled Alpha (cosine decay)

```bash
python -m experiments.run_adaptive --mode scheduled_fixed \
    --hf_token YOUR_TOKEN \
    --input_img_path inputs/bottle.jpg \
    --prompts "A photo of a bottle" "A photo of a bottle next to an apple" \
    --schedule_type cosine --schedule_start 1.0 --schedule_end 0.2
```

### With Metrics

```bash
python -m experiments.run_adaptive --mode pd_adaptive \
    --hf_token YOUR_TOKEN \
    --input_img_path inputs/bottle.jpg \
    --prompts "A photo of a bottle" "A photo of a bottle next to an apple" \
    --mask_path path/to/mask.png \
    --enable_metrics
```

### Without Step Logging

```bash
python -m experiments.run_adaptive --mode pd_adaptive \
    --hf_token YOUR_TOKEN \
    --input_img_path inputs/bottle.jpg \
    --prompts "A photo of a bottle" "A photo of a bottle next to an apple" \
    --no_log_steps
```

## Modes

| Mode | Description |
|------|-------------|
| `original` | Unchanged Stable Flow (hard K/V copy) |
| `pd_adaptive` | PD-controlled injection alpha based on drift |
| `pid_adaptive` | PID-controlled (adds integral term) |
| `fixed_soft` | Constant alpha soft blending |
| `scheduled_fixed` | Time-varying alpha (linear or cosine decay) |

### Mask Convention

Mask images: **white = preserve**, black = edit. The mask is downsampled to token space (4096 tokens for 1024x1024 images). Optional — without it, drift is measured globally.

## Output Structure

```
outputs/adaptive/{mode}/
    result.jpg           # Horizontal strip of all images
    result_0.jpg         # Reconstruction of input image
    result_1.jpg         # Edit 1
    result_2.jpg         # Edit 2 (if multiple edit prompts)
    step_log.jsonl       # Per-step drift and alpha values
    summary.json         # Trajectory summary
    metrics.json         # Optional evaluation metrics
```

## Configuration Reference

### Controller Parameters
| Flag | Default | Description |
|------|---------|-------------|
| `--kp` | 0.5 | Proportional gain |
| `--kd` | 0.1 | Derivative gain |
| `--ki` | 0.05 | Integral gain (PID only) |
| `--target_drift` | 0.1 | Target drift setpoint |
| `--base_alpha` | 0.8 | Base injection strength |
| `--alpha_min` | 0.0 | Minimum alpha |
| `--alpha_max` | 1.0 | Maximum alpha |

### Schedule Parameters (scheduled_fixed mode)
| Flag | Default | Description |
|------|---------|-------------|
| `--schedule_type` | linear | `linear` or `cosine` |
| `--schedule_start` | 1.0 | Alpha at step 0 |
| `--schedule_end` | 0.0 | Alpha at final step |

### Pipeline Parameters
| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | black-forest-labs/FLUX.1-dev | HuggingFace model |
| `--hf_token` | (required) | HuggingFace access token |
| `--input_img_path` | (required) | Path to the input image to edit |
| `--prompts` | (required) | First = description of input image, rest = edit descriptions |
| `--seed` | 42 | Random seed |
| `--cpu_offload` | false | Enable CPU offloading for low VRAM |

### Output Parameters
| Flag | Default | Description |
|------|---------|-------------|
| `--output_dir` | outputs/adaptive | Base output directory |
| `--no_log_steps` | false | Disable step logging |
| `--enable_metrics` | false | Compute evaluation metrics |
| `--mask_path` | None | Mask image (white=preserve, black=edit) |

## Files Added

```
experiments/
    __init__.py
    run_adaptive.py          # Main entry point
    config.py                # Configuration + argparse
    adapter.py               # Processor swap at runtime
    callback.py              # Denoising loop callback
    mask_utils.py            # Mask loading utilities
    logging_utils.py         # JSONL step logger
    metrics.py               # Optional LPIPS / L2 metrics
    processors/
        __init__.py
        adaptive_flux.py     # Soft-blend attention processors
    controllers/
        __init__.py
        base.py              # Controller ABC
        pd_controller.py     # PD controller
        pid_controller.py    # PID controller
        fixed.py             # Fixed and scheduled baselines
    drift/
        __init__.py
        base.py              # Drift metric ABC
        latent_drift.py      # Masked L2 in latent space
EXPERIMENTS.md               # This file
```

No existing files were modified.
