"""
Configuration parameters for video colorization pipeline.
"""

# =============================================================================
# Processing Parameters
# =============================================================================

# Frame extraction settings
DEFAULT_FPS = 24  # Default frames per second
SUPPORTED_FPS = [24, 30, 60]

# Resolution settings
RESOLUTION_SD15 = (512, 512)   # Stable Diffusion 1.5
RESOLUTION_SDXL = (1024, 1024)  # SDXL
DEFAULT_RESOLUTION = RESOLUTION_SD15

# Batch processing
BATCH_SIZE = 4  # Number of frames per batch (adjust based on VRAM)
OVERLAP_FRAMES = 2  # Frame overlap between batches for consistency

# =============================================================================
# ControlNet Settings
# =============================================================================

CONTROLNET_TYPE = "canny"  # Options: "canny", "depth"
CONTROLNET_WEIGHT = 0.8  # 0.7-1.0 Higher = more structural adherence
GUIDANCE_SCALE = 7.5  # 7.0-9.0

# Canny edge detection parameters
CANNY_LOW_THRESHOLD = 100
CANNY_HIGH_THRESHOLD = 200

# =============================================================================
# LoRA Settings
# =============================================================================

LORA_COLORIZATION_WEIGHT = 0.7  # 0.6-0.8 for colorization LoRA
LORA_STYLE_WEIGHT = 0.4  # 0.3-0.5 for optional style LoRA
LORA_MERGE_METHOD = "additive"  # Options: "additive", "weighted"

# Default LoRA paths (relative to checkpoints folder)
LORA_COLORIZATION_PATH = "lora/colorization_v1.safetensors"
LORA_STYLE_PATH = None  # Optional style LoRA

# =============================================================================
# Temporal Consistency Settings
# =============================================================================

# Optical flow warping
WARP_STRENGTH = 0.4  # 0.3-0.5

# Frame blending weights
BLEND_CURRENT_WEIGHT = 0.7  # Weight for current frame
BLEND_PREVIOUS_WEIGHT = 0.3  # Weight for warped previous frame

# CLIP similarity threshold
CLIP_SIMILARITY_THRESHOLD = 0.85  # Must be > 0.85 for consecutive frames

# Histogram matching
HISTOGRAM_BLEND_STRENGTH = 0.3

# =============================================================================
# SAM-2 Segmentation Settings
# =============================================================================

SAM2_MODEL_SIZE = "large"  # Options: "small", "base_plus", "large"
SAM2_MODEL_PATHS = {
    "small": "checkpoints/sam2_hiera_small.pt",
    "base_plus": "checkpoints/sam2_hiera_base_plus.pt", 
    "large": "checkpoints/sam2_hiera_large.pt",
}
SAM2_CONFIG_PATHS = {
    "small": "configs/sam2_hiera_s.yaml",
    "base_plus": "configs/sam2_hiera_b+.yaml",
    "large": "configs/sam2_hiera_l.yaml",
}

# Segmentation parameters
SAM2_CONFIDENCE_THRESHOLD = 0.7  # 0.6-0.8
SAM2_TEMPORAL_TRACKING = True
SAM2_MASK_REFINEMENT = True
SAM2_MAX_OBJECTS = 10  # Maximum objects to track

# =============================================================================
# Memory Management
# =============================================================================

# GPU memory allocation (in GB)
SAM2_MEMORY_LIMIT = 6.0  # 4-6GB for SAM-2
GENERATION_MEMORY_LIMIT = 10.0  # 8-12GB for SD + ControlNet

# Enable memory optimizations
ENABLE_CPU_OFFLOAD = True
ENABLE_ATTENTION_SLICING = True
ENABLE_VAE_SLICING = True

# =============================================================================
# Prompt Templates
# =============================================================================

DEFAULT_PROMPT_TEMPLATE = (
    "{main_prompt}, vibrant colors, high quality, detailed, photorealistic, "
    "temporally consistent, smooth motion, {style_keywords}, masterpiece"
)

DEFAULT_NEGATIVE_PROMPT = (
    "flickering, inconsistent colors, blurry, low quality, artifacts, "
    "color bleeding, temporal inconsistency, frame jumps, distorted, "
    "oversaturated, undersaturated, washed out"
)

# =============================================================================
# Output Settings
# =============================================================================

OUTPUT_CODEC = "libx264"
OUTPUT_QUALITY = 23  # CRF value (lower = higher quality, 18-28 typical)
OUTPUT_PRESET = "medium"  # Options: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
