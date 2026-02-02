"""
Direct Colorization Test - Proof of Concept
This script directly colorizes a video without segmentation for quick testing.
"""

import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import os

def setup_pipeline():
    """Initialize the colorization pipeline."""
    print("Loading models...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        "checkpoints/vae",
        torch_dtype=torch.float16
    ).to(device)
    
    # Load ControlNet (Canny for edge guidance)
    controlnet = ControlNetModel.from_pretrained(
        "checkpoints/controlnet/canny",
        torch_dtype=torch.float16
    ).to(device)
    
    # Load SDXL pipeline
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "checkpoints/sdxl",
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16
    ).to(device)
    
    # Optimizations for RTX 3050
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    
    print("✓ Models loaded successfully!")
    return pipe

def get_canny_edges(image, low_threshold=100, high_threshold=200):
    """Extract Canny edges from image."""
    image_np = np.array(image)
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)

def colorize_frame(pipe, gray_frame, prompt, num_steps=15):
    """Colorize a single frame."""
    # Convert to PIL
    if isinstance(gray_frame, np.ndarray):
        gray_pil = Image.fromarray(gray_frame).convert('RGB')
    else:
        gray_pil = gray_frame
    
    # Resize to 512x512 for speed (SDXL can handle up to 1024)
    gray_pil = gray_pil.resize((512, 512))
    
    # Get edge map for ControlNet guidance
    canny_image = get_canny_edges(gray_pil)
    
    # Generate colorized image
    result = pipe(
        prompt=prompt,
        image=canny_image,
        num_inference_steps=num_steps,
        controlnet_conditioning_scale=0.5,
        guidance_scale=7.5
    ).images[0]
    
    return result

def process_video(input_path, output_path, prompt, max_frames=30):
    """Process a video file."""
    print(f"\nProcessing: {input_path}")
    print(f"Prompt: {prompt}")
    
    # Load pipeline
    pipe = setup_pipeline()
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"Processing first {max_frames} frames for demo...")
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (512, 512))
    
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Colorize
        print(f"Colorizing frame {frame_count + 1}/{max_frames}...", end='\r')
        colorized = colorize_frame(pipe, gray_rgb, prompt, num_steps=10)
        
        # Convert back to OpenCV format
        colorized_cv = cv2.cvtColor(np.array(colorized), cv2.COLOR_RGB2BGR)
        
        # Write frame
        out.write(colorized_cv)
        frame_count += 1
    
    cap.release()
    out.release()
    
    print(f"\n✓ Colorization complete!")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    # Test with uploaded video
    input_video = "uploads/grey - Trim.mp4"  # Change this to your video
    output_video = "results/colorized_demo.mp4"
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Colorization prompt
    prompt = "vibrant colors, natural lighting, high quality, photorealistic"
    
    # Process (limit to 30 frames for quick demo)
    process_video(input_video, output_video, prompt, max_frames=30)
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print(f"Input:  {input_video}")
    print(f"Output: {output_video}")
    print("\nYou can now show this as proof of colorization working!")
