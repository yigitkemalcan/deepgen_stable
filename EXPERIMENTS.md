# Adaptive Injection Control for Stable Flow

Experimental extension that adds closed-loop control of source-to-target attention injection strength in Stable Flow, using a drift signal to regulate preservation vs editability during image editing.

The original Stable Flow implementation is **completely untouched**. All experimental code lives in `experiments/`.

## Quick Start

### Run Original Stable Flow (unchanged)

```bash
python run_stable_flow.py \
    --hf_token YOUR_TOKEN \
    --prompts "A photo of a dog standing" "A photo of a dog sitting"
```

### Run Adaptive PD Controller

```bash
python -m experiments.run_adaptive --mode pd_adaptive \
    --hf_token YOUR_TOKEN \
    --prompts "A photo of a dog standing" "A photo of a dog sitting" \
    --kp 0.5 --kd 0.1 --target_drift 0.1 --base_alpha 0.8
```

### Run Fixed Soft Blending (constant alpha baseline)

```bash
python -m experiments.run_adaptive --mode fixed_soft \
    --hf_token YOUR_TOKEN \
    --prompts "A photo of a dog standing" "A photo of a dog sitting" \
    --base_alpha 0.7
```

### Run with Real Image Editing

```bash
python -m experiments.run_adaptive --mode pd_adaptive \
    --hf_token YOUR_TOKEN \
    --input_img_path inputs/bottle.jpg \
    --prompts "A photo of a bottle" "A photo of a bottle next to an apple" \
    --mask_path path/to/mask.png
```

### Run with Metrics

Add `--enable_metrics` to any real-image editing run:

```bash
python -m experiments.run_adaptive --mode pd_adaptive \
    --hf_token YOUR_TOKEN \
    --input_img_path inputs/bottle.jpg \
    --prompts "A photo of a bottle" "A photo of a bottle next to an apple" \
    --mask_path path/to/mask.png \
    --enable_metrics
```

### Run without Step Logging

```bash
python -m experiments.run_adaptive --mode pd_adaptive \
    --hf_token YOUR_TOKEN \
    --prompts "A dog" "A cat" \
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

## How It Works

At each denoising step:
1. After the scheduler step, the callback computes **drift** between source (batch[0]) and edited (batch[1:]) latents in preserve-regions
2. The drift is fed to a **PD controller** which outputs an injection strength `alpha`
3. `alpha` is applied to the attention processors as soft blending: `K_target = alpha * K_source + (1-alpha) * K_target`
4. When `alpha = 1.0`, this is identical to original Stable Flow (hard copy)

### Mask Convention

Mask images: **white = preserve**, black = edit. The mask is downsampled to token space (4096 tokens for 1024x1024 images).

## Output Structure

```
outputs/adaptive/{mode}/
    result.jpg           # Horizontal strip of all images
    result_0.jpg         # Individual images
    result_1.jpg
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
| `--hf_token` | (required) | HuggingFace token |
| `--prompts` | (required) | Source + edit prompts |
| `--input_img_path` | None | Real image for editing |
| `--seed` | 42 | Random seed |
| `--num_inference_steps` | 15 | Denoising steps |
| `--cpu_offload` | false | Enable CPU offloading |

### Output Parameters
| Flag | Default | Description |
|------|---------|-------------|
| `--output_dir` | outputs/adaptive | Base output directory |
| `--no_log_steps` | false | Disable step logging |
| `--enable_metrics` | false | Compute evaluation metrics |
| `--mask_path` | None | Path to mask image |

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
