"""
Full Video Colorization Pipeline
Colorizes multiple frames from bedroom.mp4 and outputs video
Using proper ControlNet + SD1.5 + LoRA architecture
"""

import torch
import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

def get_canny_edges(image, low=50, high=150):
    """Extract Canny edges for spatial conditioning."""
    image_np = np.array(image)
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, low, high)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)

def colorize_video(input_path, output_path, prompt, frame_skip=3):
    """
    Colorize a video using ControlNet + SD1.5 + LoRA.
    
    Args:
        input_path: Path to input video
        output_path: Path to output colorized video
        prompt: Text prompt for colorization style
        frame_skip: Process every Nth frame (for speed)
    """
    print("="*70)
    print("VIDEO COLORIZATION PIPELINE")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Load models
    print("\n[1/5] Loading ControlNet (Spatial Guide)...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16
    )
    
    print("[2/5] Loading SD1.5 (Generative Core)...")
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
    print(f"[3/5] Loading LoRA (Domain Adapter)...")
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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Frame Skip: {frame_skip} (processing every {frame_skip}th frame)")
    
    frames_to_process = total_frames // frame_skip
    print(f"  Frames to colorize: {frames_to_process}")
    
    # Prepare output video
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps / frame_skip, (512, 512))
    
    # Colorization settings (proper architecture)
    CONTROLNET_SCALE = 0.7   # Guidance, not override
    GUIDANCE_SCALE = 7.5
    NUM_STEPS = 20           # Faster for video
    LORA_SCALE = 0.6
    
    print(f"\n[5/5] Colorizing frames...")
    print(f"  Prompt: {prompt}")
    print(f"  ControlNet: {CONTROLNET_SCALE} | LoRA: {LORA_SCALE}")
    
    frame_idx = 0
    colorized_count = 0
    
    pbar = tqdm(total=frames_to_process, desc="Colorizing")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for speed
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        gray_pil = Image.fromarray(gray_rgb).resize((512, 512), Image.Resampling.LANCZOS)
        
        # Get edge map
        canny_image = get_canny_edges(gray_pil)
        
        # Colorize
        result = pipe(
            prompt=prompt,
            negative_prompt="blurry, distorted, ugly, oversaturated",
            image=canny_image,
            num_inference_steps=NUM_STEPS,
            controlnet_conditioning_scale=CONTROLNET_SCALE,
            guidance_scale=GUIDANCE_SCALE,
            cross_attention_kwargs={"scale": LORA_SCALE}
        ).images[0]
        
        # Convert to OpenCV format and write
        result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        out.write(result_np)
        
        colorized_count += 1
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"\n[OK] Colorized {colorized_count} frames")
    print(f"[OK] Output saved: {output_path}")
    print("="*70)

if __name__ == "__main__":
    # Find bedroom.mp4
    possible_paths = [
        "uploads/bedroom.mp4",
        "bedroom.mp4",
        "uploads/Bedroom.mp4"
    ]
    
    input_video = None
    for path in possible_paths:
        if os.path.exists(path):
            input_video = path
            break
    
    if not input_video:
        print("ERROR: Could not find bedroom.mp4")
        print("Please ensure the video is in the uploads/ folder")
        exit(1)
    
    # Colorize with cinematic style
    colorize_video(
        input_path=input_video,
        output_path="results/bedroom_colorized.mp4",
        prompt="cozy bedroom, warm natural lighting, soft colors, comfortable atmosphere, photorealistic",
        frame_skip=2  # Process every 2nd frame for 5-second video
    )
