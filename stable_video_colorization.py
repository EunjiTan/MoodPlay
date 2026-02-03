"""
Stable Video Colorization with Flow-Guided Feedback
Prevents "changing sceneries" by using Optical Flow + Img2Img Feedback.

Algorithm:
1. Frame 0: Generate full colorization (Text2Img + ControlNet).
2. Frame i:
   - Calculate Optical Flow (Gray[i-1] -> Gray[i])
   - Warp Color[i-1] using Flow -> Guess[i]
   - Use Guess[i] as starting point for SD 1.5 (Img2Img) with Low Denoising Strength (0.35)
   - Condition with ControlNet(Gray[i]) + Text Prompt
   - Force Palette Constraints
   - Result[i] becomes feedback for next frame.

This forces the model to "stick" to the previous frame's look, only updating pixels that moved or disoccluded.
"""

import torch
import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler

# ============================================================
# OPTICAL FLOW UTILS
# ============================================================

def warp_flow(img, flow):
    """Warp an image using the optical flow field."""
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def compute_optical_flow(prev_gray, curr_gray):
    """Compute dense optical flow using Farneback algorithm."""
    # Ensure inputs are uint8
    if prev_gray.dtype != np.uint8: prev_gray = prev_gray.astype(np.uint8)
    if curr_gray.dtype != np.uint8: curr_gray = curr_gray.astype(np.uint8)
    
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow

# ============================================================
# PALETTE UTILS
# ============================================================

PALETTES = {
    "cozy_bedroom": np.array([
        [139, 90, 43], [160, 120, 80], [180, 140, 100], # Wood/Browns
        [255, 250, 240], [245, 245, 220], [255, 248, 220], # Creams
        [176, 196, 222], [100, 149, 237], # Soft Blues (Bedding?)
        [255, 218, 185], [255, 228, 181] # Warm Light
    ])
}

def quantize_to_palette(image_np, palette, temperature=20.0):
    """Soft palette quantization."""
    H, W, C = image_np.shape
    pixels = image_np.reshape(-1, 3).astype(np.float32)
    palette_f = palette.astype(np.float32)
    
    distances = np.linalg.norm(pixels[:, None, :] - palette_f[None, :, :], axis=2)
    weights = np.exp(-distances / temperature)
    weights /= weights.sum(axis=1, keepdims=True)
    
    quantized = weights @ palette_f
    return quantized.reshape(H, W, C).astype(np.uint8)

# ============================================================
# MAIN PIPELINE
# ============================================================

def stable_video_colorization(
    input_path,
    output_path,
    prompt,
    palette_name="cozy_bedroom",
    denoising_strength=0.35  # CRITICAL: Low strength = keeps previous frame consistency
):
    print("="*70)
    print("STABLE VIDEO COLORIZATION (FLOW FEEDBACK)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load Models (Img2Img Pipeline for loop)
    print("Loading models...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16
    )
    
    # Note: Using Img2ImgPipeline for temporal feedback
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
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
    if os.path.exists(lora_path):
        pipe.load_lora_weights(lora_path)
        print("LoRA loaded.")
    
    # 2. Setup Video
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {total_frames} frames from {input_path}")
    
    # Writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (512, 512))
    
    # State variables
    prev_gray = None
    prev_colorized = None
    palette = PALETTES[palette_name]
    
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret: break
        
        # Prepare Inputs
        # 1. Grayscale (Structure)
        gray_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray_orig, (512, 512))
        
        # 2. Canny (ControlNet)
        gray_blur = cv2.GaussianBlur(gray_resized, (3, 3), 0)
        edges = cv2.Canny(gray_blur, 50, 150)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        control_image = Image.fromarray(edges_rgb)
        
        # 3. Latent Initialization Image (The "Guess")
        if prev_colorized is None:
            # First Frame: Use Grayscale as base hint (or noise)
            init_image = Image.fromarray(cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2RGB))
            strength = 1.0 # Full generation for first frame
            print(f"Frame 0: Full generation...")
        else:
            # Subsequent Frames: Warp previous result
            # Calculate Flow
            flow = compute_optical_flow(prev_gray, gray_resized)
            
            # Warp previous colorized to current position
            warped_color = warp_flow(prev_colorized, flow)
            
            # Use Warped Image as Init
            init_image = Image.fromarray(cv2.cvtColor(warped_color, cv2.COLOR_BGR2RGB))
            strength = denoising_strength # Low strength to preserve consistency
        
        # GENERATE
        result = pipe(
            prompt=prompt,
            negative_prompt="changing scenery, distorted, glitchy, weird colors, blurry",
            image=init_image,           # Initial image for Img2Img
            control_image=control_image, # ControlNet Structure constraints
            num_inference_steps=20,
            strength=strength,          # How much to change the Init Image (Low = Stable)
            controlnet_conditioning_scale=0.7,
            guidance_scale=7.5,
            cross_attention_kwargs={"scale": 0.6}
        ).images[0]
        
        # Post-process
        res_np = np.array(result)
        
        # Palette Constraint (Soft)
        res_quant = quantize_to_palette(res_np, palette)
        
        # Convert to BGR for OpenCV
        res_bgr = cv2.cvtColor(res_quant, cv2.COLOR_RGB2BGR)
        
        # Update State
        prev_gray = gray_resized
        prev_colorized = res_bgr
        
        out.write(res_bgr)
        
    cap.release()
    out.release()
    print(f"Done! Saved to {output_path}")

if __name__ == "__main__":
    stable_video_colorization(
        "uploads/bedroom.mp4",
        "results/bedroom_stable.mp4",
        "cozy bedroom, white bedding, wooden furniture, soft blue walls, sunlight, interior design, photorealistic",
        "cozy_bedroom",
        denoising_strength=0.35 # Keep this low (0.3-0.4) for stability!
    )
