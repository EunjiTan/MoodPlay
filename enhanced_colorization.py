"""
Enhanced Video Colorization with Palette Constraints & Temporal Smoothing
- Predefined color palettes for consistent aesthetics
- Process ALL frames (no skipping)
- Temporal smoothing for flicker-free output
- Palette quantization for controlled colors
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# ============================================================
# PREDEFINED COLOR PALETTES
# ============================================================

PALETTES = {
    "cinematic": np.array([
        # Skin tones
        [255, 224, 189], [241, 194, 125], [198, 134, 66],
        # Blues
        [70, 130, 180], [135, 206, 235], [25, 25, 112],
        # Warm accents (teal/orange)
        [0, 128, 128], [255, 140, 0], [255, 69, 0],
        # Neutrals
        [105, 105, 105], [169, 169, 169], [245, 245, 220]
    ]),
    
    "cozy_bedroom": np.array([
        # Warm wood tones
        [139, 90, 43], [160, 120, 80], [180, 140, 100],
        # Soft whites/creams
        [255, 250, 240], [245, 245, 220], [255, 248, 220],
        # Soft blues/greens (calming)
        [176, 196, 222], [144, 238, 144], [173, 216, 230],
        # Warm lamp light
        [255, 218, 185], [255, 228, 181], [255, 239, 213]
    ]),
    
    "winter_cold": np.array([
        # Ice blues
        [176, 224, 230], [135, 206, 250], [70, 130, 180],
        # Snow whites
        [255, 250, 250], [240, 248, 255], [245, 245, 245],
        # Cold grays
        [119, 136, 153], [176, 196, 222], [192, 192, 192],
        # Subtle purple hints
        [230, 230, 250], [216, 191, 216]
    ]),
    
    "vintage_warm": np.array([
        # Sepia tones
        [255, 218, 185], [245, 222, 179], [210, 180, 140],
        # Faded colors
        [188, 143, 143], [205, 192, 176], [244, 164, 96],
        # Soft greens
        [143, 188, 143], [152, 251, 152],
        # Cream whites
        [255, 253, 208], [255, 250, 205]
    ])
}


# ============================================================
# PALETTE QUANTIZATION
# ============================================================

def quantize_to_palette(image_np, palette, method="soft"):
    """
    Quantize image colors to nearest palette colors.
    
    Args:
        image_np: numpy array (H, W, 3) in [0, 255]
        palette: numpy array (N, 3) in [0, 255]
        method: "nearest" (hard) or "soft" (blended)
    
    Returns:
        Quantized image numpy array
    """
    H, W, C = image_np.shape
    pixels = image_np.reshape(-1, 3).astype(np.float32)
    palette_f = palette.astype(np.float32)
    
    # Compute distances to all palette colors
    # (N_pixels, 1, 3) - (1, N_palette, 3) -> (N_pixels, N_palette)
    distances = np.linalg.norm(
        pixels[:, np.newaxis, :] - palette_f[np.newaxis, :, :],
        axis=2
    )
    
    if method == "nearest":
        # Hard quantization: pick closest color
        nearest_idx = np.argmin(distances, axis=1)
        quantized = palette[nearest_idx]
    
    elif method == "soft":
        # Soft quantization: weighted blend of nearby colors
        # Temperature controls softness (lower = harder)
        temperature = 30.0
        weights = np.exp(-distances / temperature)
        weights = weights / weights.sum(axis=1, keepdims=True)
        quantized = np.matmul(weights, palette_f)
    
    return quantized.reshape(H, W, C).astype(np.uint8)


# ============================================================
# TEMPORAL SMOOTHING
# ============================================================

def temporal_smooth(frames, alpha=0.3):
    """
    Exponential moving average smoothing across frames.
    
    Args:
        frames: list of numpy arrays (H, W, 3)
        alpha: smoothing factor (0 = full smooth, 1 = no smooth)
    """
    if len(frames) <= 1:
        return frames
    
    smoothed = [frames[0].astype(np.float32)]
    
    for i in range(1, len(frames)):
        smooth_frame = alpha * frames[i].astype(np.float32) + (1 - alpha) * smoothed[i-1]
        smoothed.append(smooth_frame)
    
    return [f.astype(np.uint8) for f in smoothed]


# ============================================================
# EDGE EXTRACTION
# ============================================================

def get_canny_edges(image, low=50, high=150):
    """Extract Canny edges for ControlNet."""
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, low, high)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)


# ============================================================
# MAIN COLORIZATION PIPELINE
# ============================================================

def colorize_video_enhanced(
    input_path,
    output_path,
    prompt,
    palette_name="cozy_bedroom",
    frame_skip=1,  # 1 = process ALL frames
    temporal_smoothing_alpha=0.4
):
    """
    Enhanced video colorization with palette constraints and temporal smoothing.
    
    Key improvements:
    1. Predefined color palette enforcement
    2. Process all frames (no skipping by default)
    3. Temporal smoothing for flicker reduction
    """
    print("="*70)
    print("ENHANCED VIDEO COLORIZATION")
    print("Palette-Constrained + Temporally Smooth")
    print("="*70)
    
    palette = PALETTES[palette_name]
    print(f"\nPalette: {palette_name} ({len(palette)} colors)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load models
    print("\n[1/5] Loading ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16
    )
    
    print("[2/5] Loading SD1.5...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    
    pipe = pipe.to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
    
    # Load LoRA
    lora_path = "checkpoints/lora/colorize_slider_v1.safetensors"
    print(f"[3/5] Loading LoRA...")
    pipe.load_lora_weights(lora_path)
    print("[OK] Pipeline ready!")
    
    # Load video
    print(f"\n[4/5] Loading video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {input_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Frame Skip: {frame_skip} (1 = ALL frames)")
    
    frames_to_process = total_frames // frame_skip
    print(f"  Frames to colorize: {frames_to_process}")
    
    # Colorization settings
    CONTROLNET_SCALE = 0.7   # Guidance, not override
    GUIDANCE_SCALE = 7.5
    NUM_STEPS = 15           # Faster for video
    LORA_SCALE = 0.6
    
    print(f"\n[5/5] Colorizing frames...")
    print(f"  Prompt: {prompt}")
    print(f"  ControlNet: {CONTROLNET_SCALE} | LoRA: {LORA_SCALE}")
    print(f"  Palette Quantization: soft")
    print(f"  Temporal Smoothing: alpha={temporal_smoothing_alpha}")
    
    # Collect all colorized frames
    colorized_frames = []
    frame_idx = 0
    
    pbar = tqdm(total=frames_to_process, desc="Colorizing")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames if configured
        if frame_skip > 1 and frame_idx % frame_skip != 0:
            frame_idx += 1
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        gray_pil = Image.fromarray(gray_rgb).resize((512, 512), Image.Resampling.LANCZOS)
        
        # Get edge map
        canny_image = get_canny_edges(gray_pil)
        
        # Colorize with SD1.5 + ControlNet + LoRA
        result = pipe(
            prompt=prompt,
            negative_prompt="blurry, distorted, ugly, oversaturated, unnatural colors",
            image=canny_image,
            num_inference_steps=NUM_STEPS,
            controlnet_conditioning_scale=CONTROLNET_SCALE,
            guidance_scale=GUIDANCE_SCALE,
            cross_attention_kwargs={"scale": LORA_SCALE}
        ).images[0]
        
        result_np = np.array(result)
        
        # Apply palette quantization
        quantized = quantize_to_palette(result_np, palette, method="soft")
        
        colorized_frames.append(quantized)
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    print(f"\n  Colorized {len(colorized_frames)} frames")
    
    # Apply temporal smoothing
    print("  Applying temporal smoothing...")
    smoothed_frames = temporal_smooth(colorized_frames, alpha=temporal_smoothing_alpha)
    
    # Write output video
    print("  Writing output video...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_fps = fps / frame_skip if frame_skip > 1 else fps
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (512, 512))
    
    for frame in smoothed_frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    
    print(f"\n[OK] Output saved: {output_path}")
    print("="*70)
    print("COMPLETE!")
    print("="*70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    colorize_video_enhanced(
        input_path="uploads/bedroom.mp4",
        output_path="results/bedroom_enhanced.mp4",
        prompt="cozy bedroom, warm natural lighting, soft morning atmosphere, comfortable, photorealistic",
        palette_name="cozy_bedroom",
        frame_skip=2,  # Process every 2nd frame for 5-sec video
        temporal_smoothing_alpha=0.5
    )
