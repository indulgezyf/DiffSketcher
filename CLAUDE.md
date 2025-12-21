# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DiffSketcher is a text-guided vector sketch synthesis system using Latent Diffusion Models (NeurIPS 2023). It generates high-quality SVG sketches from text prompts using a combination of Stable Diffusion attention maps, differentiable rendering (DiffVG), and CLIP-based optimization.

## Installation

### Standard Installation
```bash
chmod +x script/install.sh
bash script/install.sh
```

This creates a conda environment `svgrender` with:
- Python 3.10
- PyTorch 1.12.1 with CUDA 11.3
- DiffVG (differentiable vector graphics renderer)
- Diffusers 0.20.2
- Required dependencies (CLIP, xformers, etc.)

### Docker Installation
```bash
chmod +x script/run_docker.sh
sudo bash script/run_docker.sh
```

### Alternative: Manual Setup
```bash
python setup.py install
```

Note: The custom install command in setup.py handles all dependencies including DiffVG compilation.

## Running the Code

### Basic Text-to-Sketch
```bash
python run_painterly_render.py \
  -c diffsketcher.yaml \
  -eval_step 10 -save_step 10 \
  -update "token_ind=4 num_paths=96 num_iter=800" \
  -pt "a photo of Sydney opera house" \
  -respath ./workdir/sydney_opera_house \
  -d 8019 \
  --download
```

### Style Transfer Variant
```bash
python run_painterly_render.py \
  -tk style-diffsketcher \
  -c diffsketcher-style.yaml \
  -eval_step 10 -save_step 10 \
  -update "token_ind=4 num_paths=2000" \
  -pt "The French Revolution, highly detailed" \
  -style ./img/starry.jpg \
  -respath ./workdir/style_transfer \
  -d 876809
```

### Key Parameters
- `-c/--config`: Config file from `config/` directory (diffsketcher.yaml, diffsketcher-width.yaml, diffsketcher-color.yaml, diffsketcher-style.yaml)
- `-update`: Override config params inline (e.g., "token_ind=4 num_paths=96")
- `-pt/--prompt`: Text prompt for generation
- `-npt/--negative_prompt`: Negative prompt
- `-d/--seed`: Random seed
- `--download`: Auto-download models from HuggingFace on first run
- `-mv/--make_video`: Create rendering process video (requires ffmpeg)

### Critical Config Parameters
- `token_ind`: Index of cross-attention map for stroke initialization (start from 1, 0 is start token)
- `num_paths`: Number of strokes (16-2000 depending on complexity)
- `num_iter`: Optimization iterations (500-2000)
- `guidance_scale`: Diffusion guidance scale (default 7.5)

## Architecture

### High-Level Pipeline Flow

1. **Attention Extraction** (`extract_ldm_attn`):
   - Generate target image using Stable Diffusion
   - Extract cross-attention maps (prompt → image correspondence)
   - Extract self-attention maps (image structure)
   - Fuse attention maps for stroke initialization

2. **Stroke Initialization** (`Painter.init_image`):
   - Use fused attention maps to place initial strokes
   - Optional: Combine with XDoG edge detection (`xdog_intersec=True`)
   - Initialize stroke parameters (control points, width, opacity, color)

3. **Optimization Loop** (`painterly_rendering`):
   - Render strokes to raster image via DiffVG
   - Compute losses:
     - **ASDS (Attention-based SDS)**: Score Distillation Sampling loss from diffusion model
     - **CLIP Visual**: Perceptual similarity using CLIP conv/fc layers
     - **CLIP Text-Visual**: Text-image alignment
     - **Perceptual**: LPIPS or DISTS loss
   - Backpropagate and update stroke parameters
   - Save best results based on visual/semantic metrics

### Key Components

**Pipelines** (`pipelines/painter/`):
- `DiffSketcherPipeline`: Main text-to-sketch pipeline
- `StylizedDiffSketcherPipeline`: Adds style transfer using reference image

**Painter** (`methods/painter/diffsketcher/painter_params.py`):
- `Painter`: Manages stroke parameters and differentiable rendering
- `SketchPainterOptimizer`: Handles optimization of points, width, opacity, and color

**Diffusion Integration** (`methods/painter/diffsketcher/`):
- `Token2AttnMixinASDSPipeline`: SD 1.5/2.1 with attention extraction and SDS
- `Token2AttnMixinASDSSDXLPipeline`: SDXL variant

**Attention Control** (`methods/token2attn/`):
- `AttentionStore`: Captures cross/self-attention during diffusion
- `attn_control.py`: Hooks into UNet attention layers

**DiffVG Wrapper** (`methods/diffvg_warp/`):
- Initializes and manages pydiffvg for differentiable SVG rendering

**Metrics** (`libs/metric/`):
- CLIP score (visual and text-visual distance)
- LPIPS/DISTS perceptual loss
- FID for evaluation

### Configuration System

Uses OmegaConf for hierarchical config management:
- Base configs in `config/*.yaml`
- Runtime override via `-update` flag using dotlist notation
- Config merging: YAML base → command-line args → `-update` overrides

See `libs/engine/config_processor.py` for merge logic.

### Model Variants

**Sketch Styles** (controlled by config):
- `diffsketcher.yaml`: Black-and-white sketches (optim_opacity=True, optim_rgba=False)
- `diffsketcher-width.yaml`: Ink painting style (optim_width=True, wider strokes)
- `diffsketcher-color.yaml`: Oil painting (optim_rgba=True, 1000+ strokes)
- `diffsketcher-style.yaml`: Style transfer enabled

**Diffusion Models**:
- `sd15` (default): Stable Diffusion 1.5
- `sd21`: Stable Diffusion 2.1
- `sdxl`: SDXL (requires cross_attn_res adjustment)

### Loss Functions

**ASDS (Attention-based Score Distillation Sampling)**:
- Activated after warmup period (default: 2000 steps)
- Guides sketch toward diffusion prior
- Controlled by `sds.grad_scale` and `sds.guidance_scale`

**CLIP Visual Loss**:
- Feature-level similarity using ResNet or ViT CLIP
- `feats_loss_weights`: Layer-wise weighting for conv layers
- `fc_loss_weight`: Final fully-connected layer weight

**Perceptual Loss**:
- LPIPS (VGG-based) or DISTS
- Ensures low-level texture similarity

### Optimization

**Learnable Parameters**:
- Control points (via `lr`)
- Stroke width (via `width_lr` if `optim_width=True`)
- Opacity (if `optim_opacity=True`)
- RGBA color (via `color_lr` if `optim_rgba=True`)

**Learning Rate Scheduling**:
- Optional step-based decay (`lr_scheduler=True`)
- Decay at specified steps (`decay_steps`)

**Data Augmentation**:
- Affine transformations before CLIP computation
- Configurable via `clip.augmentations` and `sds.augmentations`

## Output Structure

Results saved to `-respath` directory:
```
workdir/
└── <experiment_name>/
    ├── ldm_generated_image.png       # Initial diffusion generation
    ├── cross_attn.png                # Cross-attention visualization
    ├── self-attn-final.png           # Self-attention visualization
    ├── attention_map.jpg             # Fused attention + stroke init
    ├── visual_best.png/.svg          # Best result by visual metric
    ├── semantic_best.png/.svg        # Best result by semantic metric
    ├── final_render.png              # Final raster output
    ├── final_svg.svg                 # Final SVG (opacity-pruned)
    ├── png_logs/                     # Periodic PNG snapshots
    ├── svg_logs/                     # Periodic SVG snapshots
    ├── attn_logs/                    # Cross-attention logs (if enabled)
    └── frame_logs/                   # Video frames (if -mv enabled)
```

## Common Tasks

### Adjust Sketch Complexity
```bash
-update "num_paths=48"     # Fewer strokes (abstract)
-update "num_paths=500"    # More strokes (detailed)
```

### Enable Video Generation
```bash
-mv -frame_freq 10 -framerate 36
```
Generates `out.mp4` in results directory (requires ffmpeg).

### Use Different Attention Token
The `token_ind` selects which word in the prompt initializes strokes:
```bash
-pt "a red house by the lake"
-update "token_ind=1"  # "a"
-update "token_ind=2"  # "red"
-update "token_ind=3"  # "house"
```
Experiment to find which token gives best stroke placement.

### Speed Up / Reduce VRAM
```bash
-update "enable_xformers=True gradient_checkpoint=True"
```

### Generate Multiple Sketches
```bash
--render_batch -srange 100 200  # Generate 100 sketches with seeds 100-199
```

## Important Notes

- **Attention Initialization**: Critical for quality. The `token_ind`, `attn_coeff`, and `softmax_temp` parameters significantly affect stroke placement.
- **Two-Phase Optimization**: ASDS loss typically starts after warmup (e.g., step 2000) to allow initial CLIP-guided convergence.
- **U2Net Masking**: For images with backgrounds, download U2Net model to `checkpoint/u2net/u2net.pth` and set `mask_object=True`.
- **DiffVG Compilation**: Requires cmake, ffmpeg, and CUDA (if available). The install script handles this but may fail on unsupported platforms.
- **Model Downloads**: First run with `--download` flag will download SD models from HuggingFace (~4-7GB depending on model).

## Related Projects

- **PyTorch-SVGRender**: State-of-the-art differentiable rendering library (reference implementation)
- **SVGDreamer**: Extended version with better editing and synthesis quality
- **VectorFusion**: Alternative text-to-SVG approach

## Troubleshooting

**DiffVG install fails**: Ensure cmake, ffmpeg, and build tools are installed. On macOS, xformers may not be available (remove from install script).

**CUDA OOM**: Reduce `image_size`, enable `gradient_checkpoint=True`, or reduce `num_paths`.

**Poor stroke placement**: Adjust `token_ind` (try different prompt words), increase `softmax_temp`, or modify `attn_coeff` to balance cross/self attention.

**Sketches too abstract/detailed**: Modify `num_paths` and `num_iter`. More iterations with fewer strokes → abstract. More strokes with moderate iterations → detailed.
