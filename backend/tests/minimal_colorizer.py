
import cv2
import numpy as np
import sys
from pathlib import Path
from PIL import Image
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.diffusion.sd15_runner import sd15_runner
from backend.models.controlnet_helpers import extract_canny
from backend.core.device import device_manager

def main():
    video_path = "uploads/bedroom.mp4"
    output_path = "results/minimal_test.mp4"
    prompt = "scandi style bedroom, sunny day, vibrant colors, blue textured bedsheets, wooden furniture, photorealistic, 8k, highly detailed"
    negative_prompt = "b&w, grayscale, black and white, cartoon, anime, low quality"
    
    print(f"Processing {video_path}...")
    
    # Extract frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    frames = frames[:5] # Limit to 5 frames for speed
    print(f"Extracted {len(frames)} frames. Colorizing frame by frame...")
    
    # Load pipeline
    sd15_runner.load_pipeline()
    
    colorized_frames = []
    
    try:
        for i, frame in enumerate(frames):
            print(f"Colorizing frame {i+1}/{len(frames)}...")
            
            # Resize for speed (max 512 width)
            h, w = frame.shape[:2]
            if w > 512:
                scale = 512 / w
                new_w = 512
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
                print(f"Resized frame to {new_w}x{new_h} for speed")
            
            # Prepare input
            pil_frame = Image.fromarray(frame)
            edges = extract_canny(pil_frame)
            
            # Generate
            # Using high guidance scale to force colors
            result = sd15_runner.generate_frame(
                image=edges,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=10,     # Reduced for speed
                guidance_scale=9.0,         # High CFG for color adherence
                controlnet_scale=0.6,       # Lower ControlNet
                lora_scale=0.0,             # No LoRA
                denoise_strength=0.0,       # No post-processing
                seed=42
            )
            
            # Save debug image immediately
            result.save(f"results/minimal_debug_{i}.png")
            print(f"Saved results/minimal_debug_{i}.png")
            
            colorized_frames.append(np.array(result))
            
        # Save video
        if colorized_frames:
            h, w = colorized_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))
            
            for frame in colorized_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            
            writer.release()
            print(f"Saved to {output_path}")
            
    finally:
        sd15_runner.unload_pipeline()

if __name__ == "__main__":
    main()
