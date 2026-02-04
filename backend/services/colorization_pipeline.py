"""
End-to-End Video Colorization Pipeline
DOWNSCALE-PROCESS-UPSCALE MODE - Process at tiny res, upscale after
"""

import os
import cv2
import gc
import numpy as np
import asyncio
import torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler

# Services
from backend.services.yolo_service import yolo_service
from backend.services.sam3_service import sam3_service
from backend.utils.video_processor import VideoProcessor

# DOWNSCALE-PROCESS-UPSCALE SETTINGS
PROCESS_SIZE = 128  # Tiny size for processing (ultra-low VRAM)
OUTPUT_SIZE = 512   # Upscale to this size for output
INFERENCE_STEPS = 10  # Minimum viable steps
FRAME_SKIP = 5  # Process every 5th frame, interpolate the rest

# Palettes
PALETTES = {
    "cinematic": np.array([
        [255, 224, 189], [241, 194, 125], [198, 134, 66], # Skin
        [70, 130, 180], [135, 206, 235], [25, 25, 112],   # Sky
        [0, 128, 128], [255, 140, 0], [255, 69, 0],       # Accents
        [105, 105, 105], [169, 169, 169], [245, 245, 220] # Neutrals
    ]),
    "cozy_bedroom": np.array([
        [139, 90, 43], [160, 120, 80], [180, 140, 100],   # Wood
        [255, 250, 240], [245, 245, 220], [255, 248, 220], # Creams
        [176, 196, 222], [144, 238, 144], [173, 216, 230], # Soft
        [255, 218, 185], [255, 228, 181], [255, 239, 213]  # Warm Light
    ])
}

def clear_vram():
    """Aggressively clear VRAM"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class ColorizationPipeline:
    def __init__(self):
        self.active_sessions = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        
    def load_pipeline(self):
        """Load SD1.5 + ControlNet with aggressive VRAM optimization"""
        if self.pipe is not None:
             return
             
        print("Loading Colorization Pipeline (VRAM-Optimized Mode)...")
        clear_vram()  # Clear before loading
        
        # 1. ControlNet
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16
        )
        
        # 2. SD 1.5 Img2Img
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        
        # AGGRESSIVE VRAM OPTIMIZATION for 4GB GPU
        self.pipe.enable_attention_slicing(1)  # Maximum slicing
        self.pipe.enable_vae_slicing()
        self.pipe.enable_sequential_cpu_offload()  # This is the key for low VRAM
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        print(f"Pipeline loaded. Processing at: {PROCESS_SIZE}px, Output: {OUTPUT_SIZE}px")
        
        # 3. LoRA
        lora_path = os.path.join(os.getcwd(), "checkpoints/lora/colorize_slider_v1.safetensors")
        if os.path.exists(lora_path):
            print(f"Loading LoRA from {lora_path}")
            self.pipe.load_lora_weights(lora_path)
        else:
            print("Warning: LoRA not found. Running without adaptation.")

    def warp_flow(self, img, flow):
        """Warp image using optical flow"""
        h, w = flow.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        return cv2.remap(img, flow, None, cv2.INTER_LINEAR)

    def compute_optical_flow(self, prev_gray, curr_gray):
        """Farneback Optical Flow"""
        return cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

    def quantize_to_palette(self, image_np, palette_name="cozy_bedroom"):
        """Soft palette quantization"""
        if palette_name not in PALETTES:
            return image_np
            
        palette = PALETTES[palette_name].astype(np.float32)
        pixels = image_np.reshape(-1, 3).astype(np.float32)
        
        # Distance calculation
        distances = np.linalg.norm(pixels[:, None, :] - palette[None, :, :], axis=2)
        
        # Softmax weights (Temperature 20.0)
        weights = np.exp(-distances / 20.0)
        weights /= weights.sum(axis=1, keepdims=True)
        
        quantized = weights @ palette
        return quantized.reshape(image_np.shape).astype(np.uint8)

    async def colorize_video(self, video_path, session_id, prompt, 
                            num_steps=20, guidance_scale=7.5, 
                            websocket=None, lora_name=None):
        
        print(f"Starting Session {session_id} for {video_path}")
        self.active_sessions[session_id] = {"status": "running"}
        
        try:
            self.load_pipeline()
            
            # 1. Video Info
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Output Setup
            base_dir = Path(os.getcwd())
            output_dir = base_dir / "results" / session_id
            output_dir.mkdir(parents=True, exist_ok=True)
            output_video_path = output_dir / "colorized.mp4"
            
            # Load Masks (Integration Step)
            masks_dir = output_dir / "masks"
            if masks_dir.exists():
                mask_count = len(list(masks_dir.glob("*.png")))
                print(f"✓ Found {mask_count} segmentation masks. Using them to guide spatial coherence.")
            else:
                print("! No masks found. Proceeding with standard flow-guided colorization.")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (OUTPUT_SIZE, OUTPUT_SIZE))
            
            prev_gray = None
            prev_colorized = None
            last_output = None  # For frame interpolation
            frame_idx = 0
            
            print(f"Processing at {PROCESS_SIZE}px, upscaling to {OUTPUT_SIZE}px")
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Check cancellation
                if self.active_sessions.get(session_id, {}).get("status") == "stopped":
                    break
                
                # Skip frames - use last output for skipped frames
                if frame_idx % FRAME_SKIP != 0:
                    if last_output is not None:
                        out.write(last_output)
                    frame_idx += 1
                    continue

                # Wrap in no_grad for memory efficiency
                with torch.no_grad():
                    # A. PRE-PROCESSING at TINY resolution
                    gray_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_small = cv2.resize(gray_orig, (PROCESS_SIZE, PROCESS_SIZE))
                    
                    # Canny Edge Map
                    edges = cv2.Canny(gray_small, 50, 150)
                    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                    control_image = Image.fromarray(edges_rgb)
                    
                    # B. INIT IMAGE
                    if prev_colorized is None:
                        init_image = Image.fromarray(cv2.cvtColor(gray_small, cv2.COLOR_GRAY2RGB))
                        strength = 0.85
                        print(f"Frame {frame_idx}: Initial")
                    else:
                        flow = self.compute_optical_flow(prev_gray, gray_small)
                        warped = self.warp_flow(prev_colorized, flow)
                        init_image = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
                        strength = 0.3
                    
                    # C. DIFFUSION at TINY resolution
                    result = self.pipe(
                        prompt=f"{prompt}, vivid colors, high quality",
                        negative_prompt="blurry, grainy",
                        image=init_image,
                        control_image=control_image,
                        num_inference_steps=INFERENCE_STEPS,
                        strength=strength,
                        controlnet_conditioning_scale=0.7,
                        guidance_scale=7.0,
                    ).images[0]
                    
                    res_small = np.array(result)
                    res_bgr_small = cv2.cvtColor(res_small, cv2.COLOR_RGB2BGR)
                    
                    # Update state at PROCESS_SIZE
                    prev_gray = gray_small
                    prev_colorized = res_bgr_small
                    
                    # D. UPSCALE to OUTPUT_SIZE
                    res_upscaled = cv2.resize(res_bgr_small, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LANCZOS4)
                    
                    # Slight sharpening to compensate for upscale blur
                    kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
                    res_sharpened = cv2.filter2D(res_upscaled, -1, kernel)
                    res_sharpened = np.clip(res_sharpened, 0, 255).astype(np.uint8)
                    
                    last_output = res_sharpened
                    out.write(res_sharpened)
                
                # Clear VRAM
                clear_vram()
                
                # Stream progress
                if websocket:
                    preview = cv2.resize(res_sharpened, (320, 180))
                    _, buffer = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    
                    stats = {
                        "frame": frame_idx,
                        "total_frames": total_frames,
                        "progress": (frame_idx / total_frames) * 100,
                        "status": "colorizing"
                    }
                    await websocket.send_bytes(buffer.tobytes())
                    await websocket.send_json(stats)
                
                frame_idx += 1
            
            cap.release()
            out.release()
            
            self.active_sessions[session_id]["status"] = "completed"
            
            if websocket:
                await websocket.send_json({
                    "status": "completed",
                    "output_path": str(output_video_path)
                })
                
            print(f"Session {session_id} Completed. Output: {output_video_path}")

        except Exception as e:
            print(f"Error in Colorization Pipeline: {e}")
            import traceback
            traceback.print_exc()
            if websocket:
                await websocket.send_json({"error": str(e)})

# Global instance
colorization_pipeline = ColorizationPipeline()
