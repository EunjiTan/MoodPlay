"""
FFmpeg-Based Video Colorization Pipeline
VRAM-Optimized: Uses FFmpeg for frame I/O, processes from disk

Flow:
1. FFmpeg: Extract frames at low res/fps to folder
2. Python: Process every Nth frame with diffusion
3. FFmpeg: Recompose frames back to video
"""

import os
import subprocess
import cv2
import gc
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler

# SETTINGS
PROCESS_SIZE = 256  # Resolution for diffusion
OUTPUT_SIZE = 512   # Final output resolution
INFERENCE_STEPS = 12
FRAME_SKIP = 3  # Process every Nth frame
EXTRACT_FPS = 10  # Extract at lower FPS to reduce frame count

def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class FFmpegColorizationPipeline:
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self):
        """Load diffusion model with VRAM optimizations"""
        if self.pipe is not None:
            return
        
        print("Loading SD1.5 + ControlNet (VRAM-Optimized)...")
        clear_vram()
        
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16
        )
        
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        
        # VRAM optimizations
        self.pipe.enable_attention_slicing(1)
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        print(f"Model loaded. Process: {PROCESS_SIZE}px, Output: {OUTPUT_SIZE}px")
    
    def extract_frames(self, video_path: str, output_dir: Path) -> tuple:
        """
        Step 1: Extract frames at low resolution using OpenCV
        Returns: (frame_count, fps)
        """
        frames_dir = output_dir / "frames_input"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame skip to achieve target FPS
        fps_skip = max(1, int(original_fps / EXTRACT_FPS))
        
        print(f"Extracting at {PROCESS_SIZE}px, every {fps_skip}th frame (~{EXTRACT_FPS}fps)")
        
        frame_idx = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames to reduce count
            if frame_idx % fps_skip == 0:
                # Resize to small processing size
                small_frame = cv2.resize(frame, (PROCESS_SIZE, PROCESS_SIZE))
                out_path = frames_dir / f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(str(out_path), small_frame)
                saved_count += 1
            
            frame_idx += 1
        
        cap.release()
        print(f"Extracted {saved_count} frames")
        return saved_count, original_fps
    
    def process_frames(self, output_dir: Path, prompt: str, websocket=None):
        """
        Step 2: Process frames from disk, one at a time
        """
        frames_dir = output_dir / "frames_input"
        colorized_dir = output_dir / "frames_colorized"
        colorized_dir.mkdir(parents=True, exist_ok=True)
        
        frame_files = sorted(frames_dir.glob("*.jpg"))
        total = len(frame_files)
        
        prev_colorized = None
        prev_gray = None
        
        for idx, frame_path in enumerate(frame_files):
            # Skip frames (copy previous for skipped)
            if idx % FRAME_SKIP != 0 and prev_colorized is not None:
                out_path = colorized_dir / f"frame_{idx:04d}.jpg"
                cv2.imwrite(str(out_path), prev_colorized)
                continue
            
            print(f"Processing frame {idx+1}/{total}")
            
            # Load frame from disk
            frame = cv2.imread(str(frame_path))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            with torch.no_grad():
                # Canny edges
                edges = cv2.Canny(gray, 50, 150)
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                control_image = Image.fromarray(edges_rgb)
                
                # Init image
                if prev_colorized is None:
                    init_image = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
                    strength = 0.85
                else:
                    init_image = Image.fromarray(cv2.cvtColor(prev_colorized, cv2.COLOR_BGR2RGB))
                    strength = 0.3
                
                # Generate
                result = self.pipe(
                    prompt=f"{prompt}, vivid colors, high quality",
                    negative_prompt="blurry, grainy, distorted",
                    image=init_image,
                    control_image=control_image,
                    num_inference_steps=INFERENCE_STEPS,
                    strength=strength,
                    controlnet_conditioning_scale=0.7,
                    guidance_scale=7.0,
                ).images[0]
                
                result_np = np.array(result)
                result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
                
                # Upscale
                result_upscaled = cv2.resize(result_bgr, (OUTPUT_SIZE, OUTPUT_SIZE), 
                                            interpolation=cv2.INTER_LANCZOS4)
                
                # Save
                out_path = colorized_dir / f"frame_{idx:04d}.jpg"
                cv2.imwrite(str(out_path), result_upscaled)
                
                prev_colorized = result_bgr
                prev_gray = gray
            
            # Clear VRAM after each frame
            clear_vram()
            
            # Send progress if websocket
            if websocket:
                import asyncio
                progress = (idx + 1) / total * 100
                asyncio.create_task(websocket.send_json({
                    "frame": idx,
                    "total_frames": total,
                    "progress": progress,
                    "status": "colorizing"
                }))
        
        print(f"Processed {total} frames")
    
    def recompose_video(self, output_dir: Path, fps: float) -> str:
        """
        Step 3: Use OpenCV to recompose frames back to video
        """
        colorized_dir = output_dir / "frames_colorized"
        output_video = output_dir / "colorized.mp4"
        
        frame_files = sorted(colorized_dir.glob("*.jpg"))
        if not frame_files:
            raise ValueError("No colorized frames found!")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_files[0]))
        h, w = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video), fourcc, EXTRACT_FPS, (w, h))
        
        print(f"Recomposing {len(frame_files)} frames at {EXTRACT_FPS}fps...")
        
        for frame_path in frame_files:
            frame = cv2.imread(str(frame_path))
            out.write(frame)
        
        out.release()
        print(f"Output saved to: {output_video}")
        return str(output_video)
    
    async def colorize_video(self, video_path: str, session_id: str, prompt: str, websocket=None):
        """Main entry point"""
        print(f"Starting disk-based colorization for {session_id}")
        
        try:
            # Setup directories
            base_dir = Path(os.getcwd())
            output_dir = base_dir / "results" / session_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Extract frames
            print("Step 1/3: Extracting frames to disk...")
            if websocket:
                await websocket.send_json({"status": "extracting", "progress": 0})
            
            frame_count, original_fps = self.extract_frames(video_path, output_dir)
            
            # Step 2: Load model and process
            print("Step 2/3: Processing frames...")
            if websocket:
                await websocket.send_json({"status": "loading_model", "progress": 10})
            
            self.load_model()
            self.process_frames(output_dir, prompt, websocket)
            
            # Step 3: Recompose video
            print("Step 3/3: Recomposing video...")
            if websocket:
                await websocket.send_json({"status": "recomposing", "progress": 95})
            
            output_video = self.recompose_video(output_dir, original_fps)
            
            # Done
            if websocket:
                await websocket.send_json({
                    "status": "completed",
                    "progress": 100,
                    "output_path": output_video
                })
            
            print("Colorization complete!")
            return output_video
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            if websocket:
                await websocket.send_json({"status": "error", "error": str(e)})
            raise

# Global instance
ffmpeg_colorization = FFmpegColorizationPipeline()
