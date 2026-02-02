"""
Working SD 1.5 Colorization - RTX 3050 Compatible
Demonstrates spatial conditioning with ControlNet
"""

import torch
import cv2
import numpy as np
from PIL import Image
import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

print("="*70)
print("MOODPLAY - SD 1.5 COLORIZATION DEMO")
print("="*70)

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

print("\n[1/4] Loading models...")
print("  - ControlNet Canny (for spatial conditioning)")
print("  - Stable Diffusion 1.5 (lightweight)")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  - Device: {device}")

# Load ControlNet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

# Load SD 1.5
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
)

pipe = pipe.to(device)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Memory optimizations for RTX 3050
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()

print("[OK] Models loaded!\n")

# Load video
print("[2/4] Loading video...")
input_video = "uploads/grey - Trim.mp4"
cap = cv2.VideoCapture(input_video)
ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Could not load video")
    exit(1)

# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
gray_pil = Image.fromarray(gray_rgb).resize((512, 512))

print(f"[OK] Loaded frame: {frame.shape}\n")

# Get edge map (SPATIAL CONDITIONING)
print("[3/4] Extracting spatial conditioning (edges)...")
canny_image = get_canny_edges(gray_pil)
print("[OK] Edge map extracted!\n")

# Test different prompts
print("[4/4] Generating colorized variations...")
print("  This demonstrates TEXT-GUIDED + SPATIALLY-CONDITIONED colorization\n")

test_cases = [
    ("vibrant", "vibrant colors, bright sunny day, saturated, photorealistic", "dull, grayscale"),
    ("sunset", "warm sunset, golden hour, orange pink sky, cinematic", "cold, blue"),
    ("winter", "cold winter, blue tones, snowy, crisp", "warm, summer")
]

os.makedirs("results/sd15_demo", exist_ok=True)

for i, (name, prompt, negative) in enumerate(test_cases, 1):
    print(f"  [{i}/3] {name}: {prompt[:50]}...")
    
    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=canny_image,
        num_inference_steps=20,
        controlnet_conditioning_scale=0.8,
        guidance_scale=7.5
    ).images[0]
    
    result.save(f"results/sd15_demo/{name}.png")
    print(f"       [OK] Saved to results/sd15_demo/{name}.png")

# Save reference images
gray_pil.save("results/sd15_demo/original_grayscale.png")
canny_image.save("results/sd15_demo/edges_spatial_conditioning.png")

print("\n" + "="*70)
print("DEMO COMPLETE!")
print("="*70)
print("\nGenerated files in results/sd15_demo/:")
print("  - original_grayscale.png (input)")
print("  - edges_spatial_conditioning.png (ControlNet input)")
print("  - vibrant.png (sunny day colors)")
print("  - sunset.png (warm golden colors)")
print("  - winter.png (cold blue tones)")
print("\nThis demonstrates:")
print("  [OK] Spatial conditioning via ControlNet edges")
print("  [OK] Text-guided color control")
print("  [OK] Same structure, different moods")
print("\n" + "="*70)
print("READY FOR WEDNESDAY REPORT!")
print("="*70)

