"""
Proper ControlNet + SD1.5 + LoRA Colorization
Following the correct architectural principles:
- SD1.5: Generative core (does actual colorization)
- ControlNet: Spatial guidance (WHERE to colorize) - additive, not override
- LoRA: Domain adaptation (HOW to colorize better)
"""

import torch
import cv2
import numpy as np
from PIL import Image
import os
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

print("="*70)
print("CONTROLNET + SD1.5 + LORA COLORIZATION")
print("Proper Architecture Implementation")
print("="*70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

# Load ControlNet (Spatial Guide)
print("\n[1/4] Loading ControlNet (Spatial Guide)...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

# Load SD1.5 (Generative Core)
print("[2/4] Loading SD1.5 (Generative Core)...")
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

# Load LoRA (Domain Adapter)
lora_path = "checkpoints/lora/colorize_slider_v1.safetensors"
print(f"[3/4] Loading LoRA (Domain Adapter): {lora_path}...")
pipe.load_lora_weights(lora_path)
print("[OK] LoRA loaded - adapts SD1.5 for colorization domain")

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
gray_pil = Image.fromarray(gray_rgb).resize((512, 512), Image.Resampling.LANCZOS)

# Get edge map (ControlNet input)
canny_image = get_canny_edges(gray_pil, low=50, high=150)

print("\n[OK] Setup complete!")
print("\nArchitectural Configuration:")
print("  - SD1.5: Generates colors via denoising (primary)")
print("  - ControlNet: Provides spatial boundaries (guidance only)")
print("  - LoRA: Adapts attention for colorization domain")
print("\nGenerating mood variations...\n")

# Mood test cases
# ControlNet scale 0.7 = guidance without override (as per architecture doc)
# LoRA scale varies to control colorization intensity
test_cases = [
    {
        "name": "winter",
        "prompt": "cold winter atmosphere, blue and white tones, frost, crisp cold lighting, photorealistic",
        "negative": "warm, orange, summer, oversaturated",
        "lora_scale": 0.6,
    },
    {
        "name": "sad_mood",
        "prompt": "melancholic atmosphere, desaturated muted colors, cool dark tones, emotional, photorealistic",
        "negative": "vibrant, happy, bright, warm, colorful",
        "lora_scale": 0.6,
    },
    {
        "name": "sunday_morning",
        "prompt": "cozy sunday morning, soft warm pastels, gentle sunlight, peaceful, photorealistic",
        "negative": "dark, gloomy, harsh, high contrast",
        "lora_scale": 0.6,
    },
    {
        "name": "cinematic",
        "prompt": "cinematic film look, teal and orange color grading, dramatic, professional, photorealistic",
        "negative": "flat, amateur, oversaturated",
        "lora_scale": 0.6,
    },
    {
        "name": "sunny_day",
        "prompt": "bright sunny day, warm golden colors, vibrant natural, clear skies, photorealistic",
        "negative": "dark, cold, gloomy, desaturated",
        "lora_scale": 0.7,
    }
]

os.makedirs("results/proper_architecture", exist_ok=True)

# Correct architectural settings
CONTROLNET_SCALE = 0.7   # Guidance, not override (per architecture doc)
GUIDANCE_SCALE = 7.5     # Standard for good prompt adherence
NUM_STEPS = 30           # Reasonable quality/speed balance

for i, config in enumerate(test_cases, 1):
    print(f"[{i}/{len(test_cases)}] {config['name'].upper()}")
    print(f"  Prompt: {config['prompt'][:50]}...")
    print(f"  ControlNet: {CONTROLNET_SCALE} (spatial guide)")
    print(f"  LoRA: {config['lora_scale']} (domain adaptation)")
    
    result = pipe(
        prompt=config['prompt'],
        negative_prompt=config['negative'],
        image=canny_image,
        num_inference_steps=NUM_STEPS,
        controlnet_conditioning_scale=CONTROLNET_SCALE,  # Guidance, not override
        guidance_scale=GUIDANCE_SCALE,
        cross_attention_kwargs={"scale": config['lora_scale']}  # LoRA influence
    ).images[0]
    
    output_path = f"results/proper_architecture/{config['name']}.png"
    result.save(output_path)
    print(f"  [OK] Saved: {output_path}\n")

# Save references
gray_pil.save("results/proper_architecture/original_grayscale.png")
canny_image.save("results/proper_architecture/edges_controlnet_input.png")

print("="*70)
print("COLORIZATION COMPLETE!")
print("="*70)
print("\nGenerated Files:")
print("  - winter.png")
print("  - sad_mood.png")
print("  - sunday_morning.png")
print("  - cinematic.png")
print("  - sunny_day.png")
print("\nArchitectural Notes:")
print("  SD1.5 = Artist (generates all colors)")
print("  ControlNet = Blueprint (shows where to paint)")
print("  LoRA = Training (teaches colorization patterns)")
print("\n" + "="*70)
