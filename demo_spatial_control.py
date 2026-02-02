"""
SD 1.5 Colorization Demo - Works on RTX 3050
Uses lighter Stable Diffusion 1.5 with ControlNet for spatial conditioning
"""

import torch
import cv2
import numpy as np
from PIL import Image
import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler

def setup_sd15_pipeline():
    """Initialize SD 1.5 pipeline with ControlNet (lighter than SDXL)."""
    print("Loading Stable Diffusion 1.5 + ControlNet...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load ControlNet Canny
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16
    )
    
    # Load SD 1.5 pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    
    pipe = pipe.to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Memory optimizations
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
    
    print("✓ SD 1.5 loaded successfully!")
    return pipe

def get_canny_edges(image, low=100, high=200):
    """Extract Canny edges for ControlNet."""
    image_np = np.array(image)
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    
    edges = cv2.Canny(gray, low, high)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)

def colorize_with_spatial_control(pipe, gray_frame, prompt, negative_prompt=""):
    """
    Colorize frame using SD 1.5 + ControlNet.
    This demonstrates SPATIAL CONDITIONING - the edge map guides where colors go.
    """
    # Resize to 512x512 (SD 1.5 optimal size)
    gray_pil = Image.fromarray(gray_frame).convert('RGB').resize((512, 512))
    
    # Get edge map (SPATIAL CONDITIONING)
    canny_image = get_canny_edges(gray_pil)
    
    # Generate with spatial control
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=canny_image,
        num_inference_steps=20,
        controlnet_conditioning_scale=0.8,  # How much to follow edges
        guidance_scale=7.5
    ).images[0]
    
    return result, canny_image

def demo_spatial_conditioning(input_path, output_dir="results/demo"):
    """
    Demonstrate spatial conditioning with multiple prompts.
    Shows how the SAME structure gets DIFFERENT colors based on text.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("SPATIAL CONDITIONING DEMONSTRATION")
    print("="*70)
    
    # Load pipeline
    pipe = setup_sd15_pipeline()
    
    # Load video and get first frame
    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error loading video")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Test different prompts on SAME frame
    test_cases = [
        {
            "name": "vibrant",
            "prompt": "vibrant colors, bright sunny day, saturated colors, high quality",
            "negative": "dull, washed out, grayscale"
        },
        {
            "name": "sunset",
            "prompt": "warm sunset colors, golden hour, orange and pink sky, cinematic",
            "negative": "cold, blue, grayscale"
        },
        {
            "name": "winter",
            "prompt": "cold winter scene, blue tones, snowy atmosphere, crisp",
            "negative": "warm, summer, grayscale"
        }
    ]
    
    print(f"\nProcessing {len(test_cases)} variations to show spatial control...")
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Generating: {test['name']}")
        print(f"  Prompt: {test['prompt']}")
        
        # Colorize with spatial conditioning
        colorized, edges = colorize_with_spatial_control(
            pipe, gray, test['prompt'], test['negative']
        )
        
        # Save outputs
        colorized.save(f"{output_dir}/{test['name']}_colorized.png")
        edges.save(f"{output_dir}/{test['name']}_edges.png")
        
        print(f"  ✓ Saved to {output_dir}/{test['name']}_*.png")
    
    # Save original grayscale
    Image.fromarray(gray).save(f"{output_dir}/original_grayscale.png")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE!")
    print("="*70)
    print(f"\nGenerated {len(test_cases)} variations in: {output_dir}/")
    print("\nThis demonstrates:")
    print("  ✓ Spatial conditioning (edges guide color placement)")
    print("  ✓ Text-based control (different prompts = different colors)")
    print("  ✓ Same structure, different moods")
    print("\nFor your report, show:")
    print("  1. Original grayscale")
    print("  2. Edge map (spatial conditioning input)")
    print("  3. Multiple colorized versions (text control)")

if __name__ == "__main__":
    input_video = "uploads/grey - Trim.mp4"
    demo_spatial_conditioning(input_video)
