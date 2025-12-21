# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DiffSketcher is a text-guided vector sketch synthesis system using Latent Diffusion Models. It generates high-quality SVG sketches from text prompts by combining:
- Stable Diffusion models (SD 1.4/1.5/2.1/XL) for guidance
- DiffVG for differentiable vector graphics rendering
- Attention Score Distillation Sampling (ASDS) for optimization
- CLIP-based perceptual losses

The system initializes strokes using cross-attention maps from the diffusion model, then optimizes their positions, colors, and widths to match the text prompt.

## Installation & Setup

### Environment Setup
```bash
# Standard installation (creates conda env 'svgrender')
chmod +x script/install.sh
bash script/install.sh

# Or using Docker
chmod +x script/run_docker.sh
sudo bash script/run_docker.sh
```

**Critical Dependencies:**
- PyTorch 1.12.1 with CUDA 11.3 (or CPU-only)
- diffusers==0.20.2 (version is important)
- DiffVG (cloned and built from source)
- xformers (optional, for speed)
- CLIP from OpenAI repo

**Note:** The setup.py contains custom install commands but the install.sh script is the recommended method.

### Model Downloads
The U2Net model for edge detection needs manual download if using `xdog_intersec=True`:
- Download from: https://huggingface.co/akhaliq/CLIPasso/blob/main/u2net.pth
- Place in: `checkpoint/u2net/u2net.pth`

## Running DiffSketcher

### Basic Command Structure
```bash
python run_painterly_render.py \
  -c <config.yaml> \
  -eval_step 10 -save_step 10 \
  -update "key1=value1 key2=value2" \
  -pt "your text prompt" \
  -respath ./workdir/output_name \
  -d <seed> \
  --download  # First run only
```

### Key Arguments
- `-c, --config`: Configuration file from `config/` directory
  - `diffsketcher.yaml` - Standard black/white sketches
  - `diffsketcher-width.yaml` - Variable stroke width (ink painting style)
  - `diffsketcher-color.yaml` - Colorful/oil painting results
  - `diffsketcher-style.yaml` - Style transfer mode
- `-tk, --task`: Task type (`diffsketcher` or `style-diffsketcher`)
- `-update`: Override config params without creating new YAML
- `-pt, --prompt`: Text prompt for generation
- `-npt, --negative_prompt`: Negative prompt
- `-respath, --results_path`: Output directory
- `-d, --seed`: Random seed
- `--download`: Auto-download HuggingFace models on first run

### Critical Parameters (via -update)
- `token_ind`: Index of cross-attention map for stroke initialization (start from 1, 0 is start token)
- `num_paths`: Number of strokes (16-2000 depending on complexity)
- `num_iter`: Training iterations (500-2000)
- `guidance_scale`: Diffusion guidance strength (default 7.5)
- `prune_start_step`: Start applying pruning loss after this many steps (default 200)
- `prune_ema_threshold`: EMA threshold to identify dead strokes (default 0.02)
- `prune_loss_weight`: Weight for the pruning regularization loss (default 10.0)

### Performance Options
- `enable_xformers=True`: Speed up attention computation
- `gradient_checkpoint=True`: Reduce VRAM usage (slower)
- `ldm_speed_up=True`: Additional LDM optimizations

### Style Transfer
```bash
python run_painterly_render.py \
  -tk style-diffsketcher -c diffsketcher-style.yaml \
  -update "style_warmup=0 style_strength=1 sds.grad_scale=0" \
  -pt "your prompt" \
  -style ./path/to/style_image.jpg \
  -respath ./workdir/output
```

### Batch Rendering
```bash
python run_painterly_render.py \
  -c diffsketcher.yaml \
  -rdbz -srange 100 200 \
  -pt "your prompt"
# Renders with seeds 100-199
```

## Code Architecture

### High-Level Flow
1. **run_painterly_render.py** - Entry point, handles CLI args and config merging
2. **Pipeline Selection** (in pipelines/painter/)
   - `DiffSketcherPipeline` - Standard text-to-sketch
   - `StylizedDiffSketcherPipeline` - With style transfer
3. **Painter Initialization** (methods/painter/diffsketcher/)
   - `Painter` class manages SVG strokes as PyTorch parameters
   - Initializes strokes from attention maps or randomly
4. **Optimization Loop**
   - Renders SVG to raster using DiffVG
   - Computes losses (ASDS, CLIP, perceptual)
   - Updates stroke parameters via SGD/Adam

### Key Components

**Config System** (`libs/engine/`)
- Uses OmegaConf for hierarchical configs
- `merge_and_update_config()` merges YAML + CLI args
- `-update` param allows inline config overrides

**Diffusion Integration** (`methods/diffusers_warp/`)
- Custom pipelines: `Token2AttnMixinASDSPipeline` (SD 1.x/2.x) and `Token2AttnMixinASDSSDXLPipeline` (SDXL)
- Stores attention maps via `AttentionStore` controller
- ASDS loss implemented in pipeline classes

**Stroke Representation** (`methods/painter/diffsketcher/painter_params.py`)
- Each stroke is a Bézier path with control points
- Optimizable parameters:
  - `points_vars`: Control point positions
  - `color_vars`: RGBA values (if `optim_rgba=True`)
  - `stroke_width_vars`: Width values (if `optim_width=True`)
  - Opacity (if `optim_opacity=True`)

**Attention-Based Initialization** (`methods/token2attn/`)
- Extracts cross-attention maps at resolution `cross_attn_res` (16 or 32)
- `token_ind` selects which token's attention to use
- Can combine with XDoG edge detection if `xdog_intersec=True`
- Softmax temperature `softmax_temp` controls attention sharpness

**Loss Functions**
- **ASDS**: Attention Score Distillation Sampling (main guidance)
  - Configured in `sds:` section of YAML
  - `grad_scale`, `guidance_scale`, `t_range`, `warmup`
- **CLIP Visual**: Multi-layer perceptual loss from CLIP vision encoder
  - `clip.feats_loss_weights` sets per-layer weights
  - `clip.fc_loss_weight` for final FC layer
- **Perceptual**: LPIPS or DISTS
  - `perceptual.name`, `perceptual.coeff`
- **Pruning Loss (Soft-OBR)**: Dynamic stroke pruning with EMA tracking
  - Forces low-opacity strokes to become fully transparent (alpha=0)
  - Uses Exponential Moving Average (EMA) to avoid misidentifying temporarily low strokes
  - Starts after `prune_start_step` warmup period
  - Dead strokes identified when `alphas_ema < prune_ema_threshold`
  - Loss weight controlled by `prune_loss_weight`

**Rendering** (`methods/diffvg_warp/`)
- Wraps DiffVG library for differentiable SVG rendering
- Converts stroke parameters to pydiffvg shapes
- Renders at `image_size` resolution (default 224)

### Directory Structure
```
DiffSketcher/
├── config/               # YAML configuration files
├── libs/
│   ├── engine/          # Config merging, base ModelState class
│   ├── metric/          # CLIP, LPIPS, FID implementations
│   ├── modules/         # Edge detection (XDoG), visual utils
│   ├── solver/          # Optimizers and LR schedulers
│   └── utils/           # Logging, argparse, misc utilities
├── methods/
│   ├── diffusers_warp/  # Custom diffusion pipelines with attention control
│   ├── diffvg_warp/     # DiffVG initialization and state management
│   ├── painter/         # Core Painter class and optimizer
│   │   └── diffsketcher/ # ASDS pipelines, stroke utils, mask utils
│   └── token2attn/      # Attention extraction and control (Prompt-to-Prompt style)
└── pipelines/
    └── painter/         # High-level rendering pipelines
```

## Configuration Files

All configs in `config/` directory share base structure but differ in stroke optimization settings:

**diffsketcher.yaml** (Standard)
- Fixed width strokes
- `optim_width=False`, `optim_rgba=False`
- Best for clean sketches

**diffsketcher-width.yaml** (Variable Width)
- `optim_width=True`, `width_lr=0.1`, `max_width=50`
- Creates ink painting effects

**diffsketcher-color.yaml** (Color/Oil Painting)
- `optim_rgba=True`, `color_lr=0.01`
- Usually needs more paths (1000+)

**diffsketcher-style.yaml** (Style Transfer)
- Adds `style_warmup`, `style_strength` parameters
- Often sets `sds.grad_scale=0` to disable ASDS

## Development Patterns

### Adding New Loss Functions
1. Implement in `methods/painter/diffsketcher/` or `libs/metric/`
2. Add computation in pipeline's optimization loop
3. Add config parameters to YAML
4. Weight via config (e.g., `new_loss.coeff`)

### Supporting New Diffusion Models
1. Add model ID to `methods/diffusers_warp/`
2. Set appropriate pipeline class in `DiffSketcherPipeline.__init__`
3. Adjust `cross_attn_res` if needed (SDXL uses 2x)

### Stroke Initialization Methods
Current: attention maps + optional XDoG edges
- Modify `Painter.get_path()` in `painter_params.py`
- Control via `attention_init`, `xdog_intersec` flags

### Output Files
Results saved to `-respath`:
- `png_logs/` - Rasterized renders at intervals
- `svg_logs/` - SVG files at intervals (automatically filtered to remove dead strokes)
- `attn_logs/` - Attention map visualizations
- `best.svg` / `best.png` - Final result
- `frame_logs/` - Video frames (if `-mv` flag set)

**Note**: All SVG files are automatically cleaned during export - strokes with opacity < 0.01 are filtered out, resulting in smaller file sizes and cleaner vector graphics.

## Model Compatibility

**Stable Diffusion Versions:**
- `sd14`, `sd15` - Use `DDIMScheduler`, 16x16 cross-attn
- `sd21` - Use `DDIMScheduler`, 16x16 cross-attn
- `sdxl` - Use `DPMSolverMultistepScheduler`, 32x32 cross-attn (auto-adjusted)

Set via `model_id` in config or `-update "model_id=sdxl"`.

## Common Issues

**CUDA out of memory:**
- Reduce `num_paths` or `image_size`
- Enable `gradient_checkpoint=True`
- Lower `sds.crop_size`

**Poor stroke initialization:**
- Adjust `token_ind` (try different words in prompt)
- Tune `softmax_temp` (higher = more diffuse, lower = more peaked)
- Enable `xdog_intersec=True` for edge-aware init

**Strokes disappear during training:**
- Check `opacity_delta` (default 0, increase to prevent pruning)
- Verify `pruning_freq` isn't too aggressive

**Style transfer not working:**
- Ensure `style_warmup` allows enough ASDS warmup first
- Balance `style_strength` vs `sds.grad_scale`
