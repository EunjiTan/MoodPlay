"""
SD 1.5 + LoRA Colorization Test
Tests colorization LoRAs with high structure preservation settings
"""

import torch
import cv2
import numpy as np
from PIL import Image
import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

def get_canny_edges(image):
    """Extract Canny edges for spatial conditioning."""
    image_np = np.array(image)
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    
    edges = cv2.Canny(gray, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)

def test_colorization_lora(lora_path=None):
    """
    Test colorization with LoRA.
    Uses high ControlNet conditioning to preserve structure.
    """
    print("="*70)
    print("SD 1.5 + LoRA COLORIZATION TEST")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Load ControlNet
    print("\n[1/4] Loading ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16
    )
    
    # Load SD 1.5
    print("[2/4] Loading SD 1.5...")
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
    
    # Load LoRA if provided
    if lora_path:
        print(f"[3/4] Loading LoRA: {lora_path}...")
        pipe.load_lora_weights(lora_path)
        print("[OK] LoRA loaded!")
    else:
        print("[3/4] No LoRA specified, using base SD 1.5")
    
    # Load test image
    print("[4/4] Loading test image...")
    input_video = "uploads/grey - Trim.mp4"
    cap = cv2.VideoCapture(input_video)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("ERROR: Could not load video")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    gray_pil = Image.fromarray(gray_rgb).resize((512, 512))
    
    # Get edge map
    canny_image = get_canny_edges(gray_pil)
    
    print("\n[OK] Setup complete! Generating colorization...\n")
    
    # Test with different settings for structure preservation
    test_configs = [
        {
            "name": "high_preservation",
            "prompt": "add natural colors, photorealistic",
            "negative": "blurry, distorted, deformed, ugly",
            "controlnet_scale": 0.95,  # Very high - strong structure preservation
            "guidance_scale": 5.0,      # Lower - less creative freedom
            "num_steps": 15
        },
        {
            "name": "balanced",
            "prompt": "colorize with vibrant natural colors",
            "negative": "grayscale, monochrome, blurry",
            "controlnet_scale": 0.85,
            "guidance_scale": 7.0,
            "num_steps": 20
        }
    ]
    
    os.makedirs("results/lora_test", exist_ok=True)
    
    for i, config in enumerate(test_configs, 1):
        print(f"[{i}/{len(test_configs)}] Testing: {config['name']}")
        print(f"  ControlNet Scale: {config['controlnet_scale']}")
        print(f"  Guidance Scale: {config['guidance_scale']}")
        
        result = pipe(
            prompt=config['prompt'],
            negative_prompt=config['negative'],
            image=canny_image,
            num_inference_steps=config['num_steps'],
            controlnet_conditioning_scale=config['controlnet_scale'],
            guidance_scale=config['guidance_scale']
        ).images[0]
        
        output_path = f"results/lora_test/{config['name']}.png"
        result.save(output_path)
        print(f"  [OK] Saved to {output_path}\n")
    
    # Save references
    gray_pil.save("results/lora_test/original_grayscale.png")
    canny_image.save("results/lora_test/edges.png")
    
    print("="*70)
    print("TEST COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - original_grayscale.png")
    print("  - edges.png")
    print("  - high_preservation.png (ControlNet 0.95)")
    print("  - balanced.png (ControlNet 0.85)")
    print("\nCompare results to see which preserves structure best!")

if __name__ == "__main__":
    # Test with the downloaded Colorize Slider LoRA
    lora_path = "checkpoints/lora/colorize_slider_v1.safetensors"
    
    if os.path.exists(lora_path):
        print(f"Found LoRA: {lora_path}")
        test_colorization_lora(lora_path=lora_path)
    else:
        print(f"LoRA not found at: {lora_path}")
        print("Run: python download_colorize_lora.py")
        print("\nTesting without LoRA (baseline)...")
        test_colorization_lora(lora_path=None)
