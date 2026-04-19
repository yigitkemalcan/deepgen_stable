# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

**Stable Flow** (CVPR 2025) — a training-free image editing method built on top of FLUX.1-dev. It works by injecting attention features from a reference generation pass into edited generation passes at specific "vital layers" of the DiT (Diffusion Transformer). The repo is a modified fork of HuggingFace Diffusers (`src/diffusers/`), installed in editable mode alongside a single entry-point script (`run_stable_flow.py`).

## Setup

```bash
conda env create -f environment.yml
conda activate stable-flow
# The package is installed as editable via '-e .' in environment.yml
```

If the CUDA version in `environment.yml` doesn't match your system, update `pytorch-cuda` before creating the environment.

## Running the Code

**Generated image editing** (no input image):
```bash
python run_stable_flow.py \
  --hf_token YOUR_HF_TOKEN \
  --prompts "A photo of a dog standing in the street" \
             "A photo of a dog sitting in the street"
```

**Real image editing** (with inversion):
```bash
python run_stable_flow.py \
  --hf_token YOUR_HF_TOKEN \
  --input_img_path inputs/bottle.jpg \
  --prompts "A photo of a bottle" \
             "A photo of a bottle next to an apple"
```

Key flags:
- `--output_path`: defaults to `outputs/result.jpg`
- `--seed`: defaults to 42
- `--cpu_offload`: enables sequential CPU offloading to reduce VRAM (increases inference time significantly); needed for GPUs below ~80GB
- `--device`: defaults to `cuda`
- `--model_path`: defaults to `black-forest-labs/FLUX.1-dev`

## Architecture

The core mechanism is **selective attention injection** through "vital layers":

```
run_stable_flow.py (StableFlow class)
  └─ FluxPipeline  (src/diffusers/pipelines/flux/pipeline_flux.py)
       ├─ FluxTransformer2DModel  (src/diffusers/models/transformers/transformer_flux.py)
       │    ├─ transformer_blocks: 19× FluxTransformerBlock  (MMDiT, multimodal)
       │    └─ single_transformer_blocks: 38× FluxSingleTransformerBlock  (single-modal)
       ├─ AutoencoderKL (VAE)
       ├─ CLIPTextModel + T5EncoderModel (dual text encoders)
       └─ FlowMatchEulerDiscreteScheduler
```

**Vital layers** (hardcoded in `run_stable_flow.py`):
- Multimodal blocks: `[0, 1, 17, 18]` → `mm_copy_blocks`
- Single-modal blocks: `[9, 34, 35, 37, 6]` (computed from `[28, 53, 54, 56, 25] - 19`) → `single_copy_blocks`

**How injection works**: The pipeline `__call__` accepts `mm_copy_blocks` and `single_copy_blocks` lists. These are forwarded into each transformer block's `forward()`. Inside `FluxAttnProcessor2_0` and `FluxSingleAttnProcessor2_0` (`src/diffusers/models/attention_processor.py`), when `index_block in copy_blocks`, the K/V tensors from batch index 0 (the reference/source prompt) are copied into all other batch entries before the attention operation.

**Inversion flow**: For real images, the pipeline is called with `invert_image=True` and `guidance_scale=1`, which runs the denoising loop in reverse (noise-prediction → latent update reversed) to recover the noise trajectory. The resulting `inverted_latent_list` is then passed back in the editing call to supply per-timestep starting latents.

**Batch structure**: All prompts (source + edits) are run as a single batch. The source prompt is always index 0; edits are indices 1..N. The shared initial latent (tiled from a single random draw) ensures structural consistency.

## Key Modified Files

The diffusers fork modifies primarily:
- `src/diffusers/pipelines/flux/pipeline_flux.py` — added `invert_image`, `mm_copy_blocks`, `single_copy_blocks`, `mm_skip_blocks`, `single_skip_blocks`, `percentage_of_steps`, `inverted_latent_list` parameters to `__call__`
- `src/diffusers/models/transformers/transformer_flux.py` — passes block index and copy/skip lists through `forward()`
- `src/diffusers/models/attention_processor.py` — implements K/V injection logic in `FluxAttnProcessor2_0` and `FluxSingleAttnProcessor2_0`
