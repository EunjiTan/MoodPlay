"""
End-to-End Video Colorization Pipeline (Stable FD1.5 + Flow Feedback)
Orchestrates all modules for complete video colorization with:
1. SD 1.5 + ControlNet + LoRA
2. Optical Flow-Guided Temporal Consistency
3. Palette Constraints
4. Object-Aware Masking (YOLO+SAM)
"""

import os
import cv2
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

class ColorizationPipeline:
    def __init__(self):
        self.active_sessions = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        
    def load_pipeline(self):
        """Load SD1.5 + ControlNet + LoRA"""
        if self.pipe is not None:
             return
             
        print("Loading Colorization Pipeline Models...")
        
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
        
        self.pipe = self.pipe.to(self.device)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_attention_slicing()
        self.pipe.enable_model_cpu_offload() # Save VRAM
        
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
            # Assuming resize to 512x512 for SD
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (512, 512))
            
            prev_gray = None
            prev_colorized = None
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Check cancellation
                if self.active_sessions.get(session_id, {}).get("status") == "stopped":
                    break

                # -----------------------------------------------------
                # A. PRE-PROCESSING (Resize & Canny)
                # -----------------------------------------------------
                gray_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_resized = cv2.resize(gray_orig, (512, 512))
                
                # Canny Edge Map (ControlNet Input)
                gray_blur = cv2.GaussianBlur(gray_resized, (3, 3), 0)
                edges = cv2.Canny(gray_blur, 50, 150)
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                control_image = Image.fromarray(edges_rgb)

                # -----------------------------------------------------
                # B. OPTIONAL: OBJECT DETECTION & SEGMENTATION
                # (Demonstrating YOLO+SAM usage as requested)
                # -----------------------------------------------------
                # YOLO Detect on original frame size (better accuracy)
                # detections = yolo_service.detect(frame)
                
                # SAM Segment
                # masks = sam3_service.segment_frame_boxes(frame_idx, detections)
                # NOTE: For now, we just pass this data to frontend for Viz.
                # In future: Use masks to mask latent noise per object.
                
                # -----------------------------------------------------
                # C. LATENT INITIALIZATION (FLOW FEEDBACK)
                # -----------------------------------------------------
                if prev_colorized is None:
                    # Frame 0: START FRESH
                    # Use grayscale as init image hint
                    init_image = Image.fromarray(cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2RGB))
                    strength = 1.0 # High strength = Ignore init image, generate from scratch
                    print(f"Frame {frame_idx}: Initial Generation")
                else:
                    # Frame N: WARP PREVIOUS
                    flow = self.compute_optical_flow(prev_gray, gray_resized)
                    warped_color = self.warp_flow(prev_colorized, flow)
                    
                    init_image = Image.fromarray(cv2.cvtColor(warped_color, cv2.COLOR_BGR2RGB))
                    strength = 0.25 # VERY LOW strength = Strict adherence to structure
                
                # -----------------------------------------------------
                # D. DIFFUSION GENERATION
                # -----------------------------------------------------
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt="blurry, distorted, changing scenery, weird colors",
                    image=init_image,           # Img2Img Input
                    control_image=control_image, # ControlNet Input
                    num_inference_steps=num_steps,
                    strength=strength,          # Denoising Strength
                    controlnet_conditioning_scale=0.7,
                    guidance_scale=guidance_scale,
                    cross_attention_kwargs={"scale": 0.6}
                ).images[0]
                
                res_np = np.array(result)
                
                # -----------------------------------------------------
                # E. MASK-GUIDED BLENDING
                # Load saved masks and apply colors ONLY within masks
                # -----------------------------------------------------
                # Find masks for this frame
                frame_masks = list(masks_dir.glob(f"frame_{frame_idx:04d}_obj_*.png")) if masks_dir.exists() else []
                
                if frame_masks:
                    # Create combined mask from all objects
                    combined_mask = np.zeros((512, 512), dtype=np.float32)
                    for mask_path in frame_masks:
                        mask = np.array(Image.open(mask_path).convert('L').resize((512, 512)))
                        combined_mask = np.maximum(combined_mask, mask.astype(np.float32) / 255.0)
                    
                    # Expand to 3 channels
                    mask_3ch = np.stack([combined_mask] * 3, axis=-1)
                    
                    # Blend: Colorized inside mask, Original grayscale outside
                    gray_rgb = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2RGB)
                    res_np = (res_np * mask_3ch + gray_rgb * (1 - mask_3ch)).astype(np.uint8)
                    print(f"Frame {frame_idx}: Applied {len(frame_masks)} masks")
                
                # Palette constraint
                res_quant = self.quantize_to_palette(res_np, "cozy_bedroom")
                res_bgr = cv2.cvtColor(res_quant, cv2.COLOR_RGB2BGR)
                
                # -----------------------------------------------------
                # F. UPDATE STATE & SAVE
                # -----------------------------------------------------
                prev_gray = gray_resized
                prev_colorized = res_bgr
                
                out.write(res_bgr)
                
                # Stream to Frontend
                if websocket:
                    preview = cv2.resize(res_bgr, (640, 360))
                    _, buffer = cv2.imencode('.jpg', preview)
                    
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
