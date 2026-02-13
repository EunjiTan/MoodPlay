# MoodPlay — Instance-Guided Semantic Video Colorization

An AI-powered pipeline that colorizes grayscale video with **instance-level accuracy**, **temporal consistency**, and **strict palette adherence**. Each object in the scene receives its own assigned colour from a user-defined palette, maintained consistently across every frame.

## Architecture

```
Input Video  -->  YOLOv11 Detect  -->  SAM-2 Segment  -->  CoTracker Track
                                                                 |
                                                                 v
Output Video  <--  Quality Report  <--  Validate/Correct  <--  Diffuse (ControlNet + LoRA)
```

**Per-frame loop (F1–F7):**

| Step | Component | Purpose |
|------|-----------|---------|
| F1 | YOLOv11 | Detect objects, assign persistent Instance IDs |
| F2 | SAM-2 | Box-prompted segmentation, zero-overlap masks |
| F3 | CoTracker | Joint multi-instance point tracking |
| F4 | Temporal Engine | Warp previous frame colours to current frame |
| F5 | ControlNet + LoRA | Conditional diffusion colorization |
| F6 | LAB Validator | Palette correction if ΔE > 5.0, BLS check |
| F7 | Registry Update | Log colour history, update temporal state |

## Project Structure

```
MoodPlay/
├── run.py                          # CLI entry point
├── requirements.txt                # Python dependencies
├── backend/
│   ├── core/                       # Core infrastructure
│   │   ├── pipeline_config.py      # All thresholds & constants
│   │   ├── lab_color_engine.py     # CIELAB conversion, ΔE CIE-1994
│   │   ├── instance_registry.py    # Persistent ID & palette binding
│   │   └── quality_metrics.py      # ICA, TCV, BLS, GPC, SSIM
│   ├── models/                     # AI model wrappers
│   │   ├── sam2_segmenter.py       # SAM-2 with temporal propagation
│   │   └── cotracker_motion.py     # CoTracker with joint tracking
│   ├── services/                   # Service layer
│   │   ├── yolo_service.py         # YOLOv11 detection + registry
│   │   ├── controlnet_service.py   # ControlNet conditioning
│   │   └── lora_service.py         # LoRA palette modulation
│   ├── conditioning/               # Colour conditioning
│   │   ├── four_color_palette.py   # Mood-based palette generator
│   │   ├── instance_hints.py       # Per-instance colour hints
│   │   └── style_conditioning.py   # Style reference encoding
│   ├── pipelines/                  # Pipeline orchestration
│   │   ├── instance_guided_pipeline.py  # Main F1-F7 orchestrator
│   │   └── temporal_coherence.py        # Temporal consistency engine
│   └── tests/
│       └── test_instance_pipeline.py    # Integration test suite
├── configs/                        # Model configurations
├── checkpoints/                    # Model weights (gitignored)
├── frontend/                       # Web UI
├── uploads/                        # Input videos
└── results/                        # Output videos
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required model checkpoints** (place in `checkpoints/`):
- SAM-2: `checkpoints/sam2/sam2_hiera_tiny.pt`
- ControlNet Canny: `checkpoints/controlnet/canny/`
- YOLOv11: `yolo11n.pt` (auto-downloaded)

### 2. Run Tests

```bash
python run.py --test
```

### 3. Colorize a Video

```bash
# Basic usage
python run.py uploads/my_video.mp4

# With options
python run.py uploads/my_video.mp4 --mood golden_hour --seed 42 --steps 20

# Specify output path
python run.py uploads/my_video.mp4 --output results/output.mp4
```

### 4. Python API

```python
from backend.pipelines.instance_guided_pipeline import (
    InstanceGuidedPipeline, PipelineRunConfig,
)

pipeline = InstanceGuidedPipeline()
config = PipelineRunConfig(
    mood="golden_hour",
    seed=42,
    num_inference_steps=20,
    keyframe_interval=5,
)

result = pipeline.process_video(frames, config)

colorized_frames = result["colorized_frames"]
quality_report   = result["quality_report"]
instance_state   = result["registry"]
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mood` | `sunny_day` | Palette mood preset |
| `--seed` | None | Random seed for reproducibility |
| `--steps` | 20 | Diffusion inference steps |
| `--max-frames` | 60 | Maximum frames to process |
| `--keyframe-interval` | 5 | Full segmentation every N frames |
| `--no-tracking` | False | Disable CoTracker motion tracking |
| `--output` | `results/<name>_colorized.mp4` | Output video path |
| `--test` | False | Run integration test suite |

## Quality Metrics

The pipeline computes and reports these metrics after each run:

| Metric | Target | Description |
|--------|--------|-------------|
| **TCV** | < 8.0 | Temporal Color Variance — frame-to-frame colour stability |
| **BLS** | 0.0 | Boundary Leakage Score — colour bleeding across masks |
| **GPC** | > 0.5% each | Global Palette Coverage — all palette entries represented |
| **SSIM** | > 0.8 | Structural preservation vs. grayscale input |

## Available Mood Presets

`sunny_day` · `golden_hour` · `winter` · `autumn` · · `cinematic` · `neon_cyberpunk` ·

## Requirements

- Python 3.10+
- CUDA-capable GPU (8+ GB VRAM recommended)
- PyTorch 2.0+ with CUDA support
