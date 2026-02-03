"""
Professional Colorization Test - Pixel-Perfect Quality
High-resolution, cinematic-grade colorization with LoRA
"""

import torch
import cv2
import numpy as np
from PIL import Image
import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

def get_canny_edges(image, low=50, high=150):
    """Extract refined Canny edges for precise spatial conditioning."""
    image_np = np.array(image)
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    
    # Apply Gaussian blur for cleaner edges
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, low, high)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)

print("="*70)
print("PROFESSIONAL COLORIZATION - PIXEL-PERFECT QUALITY")
print("="*70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

# Load ControlNet
print("\n[1/4] Loading ControlNet Canny...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

# Load SD 1.5
print("[2/4] Loading Stable Diffusion 1.5...")
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

# Load Colorize LoRA
lora_path = "checkpoints/lora/colorize_slider_v1.safetensors"
print(f"[3/4] Loading Colorize LoRA: {lora_path}...")
pipe.load_lora_weights(lora_path)
print("[OK] LoRA loaded!")

# Load test image
print("[4/4] Loading test frame...")
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

# Standard resolution for SD 1.5 (Native training resolution = best quality/speed balance)
gray_pil = Image.fromarray(gray_rgb).resize((512, 512), Image.Resampling.LANCZOS)

# Get refined edge map
canny_image = get_canny_edges(gray_pil, low=50, high=150)

print("\n[OK] Setup complete! Generating professional colorizations...\n")

# Professional test cases
test_cases = [
    {
        "name": "winter_redo",
        "prompt": "cold winter atmosphere, blue and white color palette, frost, cold lighting, photorealistic, 8k quality",
        "negative": "warm colors, orange, summer, blurry, distorted, changed details, extra objects",
        "lora_scale": 0.8,  # Increased LoRA influence
    },
    {
        "name": "sad_mood",
        "prompt": "sad melancholic atmosphere, desaturated muted colors, cool dark tones, gloomy lighting, emotional, photorealistic",
        "negative": "vibrant, happy, bright, warm, colorful, blurry, distorted",
        "lora_scale": 0.7,
    },
    {
        "name": "sunday_morning",
        "prompt": "sunday morning vibes, soft cozy lighting, gentle pastel colors, warm peaceful atmosphere, morning glow, photorealistic",
        "negative": "harsh lighting, dark, gloomy, high contrast, blurry, distorted",
        "lora_scale": 0.7,
    },
    {
        "name": "cinematic_redo",
        "prompt": "cinematic film look, teal and orange, dramatic lighting, moody, professional color grading, photorealistic",
        "negative": "flat, amateur, oversaturated, blurry, distorted",
        "lora_scale": 0.7,
    }
]

os.makedirs("results/professional_colorization", exist_ok=True)

# Stricter settings for structure preservation
CONTROLNET_SCALE = 1.0   # Maximum structure preservation (Restricts SD1.5 from changing geometry)
GUIDANCE_SCALE = 5.0     # Lower guidance to let ControlNet/LoRA lead
NUM_STEPS = 45           # Higher steps for refinement

for i, config in enumerate(test_cases, 1):
    print(f"[{i}/{len(test_cases)}] Generating: {config['name'].upper()}")
    print(f"  Style: {config['prompt'][:60]}...")
    print(f"  LoRA Scale: {config['lora_scale']}")
    print(f"  ControlNet: {CONTROLNET_SCALE} | Steps: {NUM_STEPS}")
    
    result = pipe(
        prompt=config['prompt'],
        negative_prompt=config['negative'],
        image=canny_image,
        num_inference_steps=NUM_STEPS,
        controlnet_conditioning_scale=CONTROLNET_SCALE,
        guidance_scale=GUIDANCE_SCALE,
        cross_attention_kwargs={"scale": config['lora_scale']}  # LoRA weight
    ).images[0]
    
    output_path = f"results/professional_colorization/{config['name']}.png"
    result.save(output_path)
    print(f"  [OK] Saved: {output_path}\n")

# Save references
gray_pil.save("results/professional_colorization/original_grayscale.png")
canny_image.save("results/professional_colorization/edges.png")

print("="*70)
print("PROFESSIONAL COLORIZATION COMPLETE! (STRICT MODE)")
print("="*70)
print("\nGenerated Files (768x768, Strict Structure):")
print("  - winter_redo.png")
print("  - sad_mood.png")
print("  - sunday_morning.png")
print("  - cinematic_redo.png")
print("\nSettings:")
print(f"  ControlNet Scale: {CONTROLNET_SCALE} (Max structure preservation)")
print(f"  LoRA Weights: 0.7-0.8 (Dominant colorization)")
print("\n" + "="*70)
print("READY FOR WEDNESDAY REPORT!")
print("="*70)
