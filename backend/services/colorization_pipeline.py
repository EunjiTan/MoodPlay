"""
End-to-End Video Colorization Pipeline
Orchestrates all modules for complete video colorization.
"""

import os
import cv2
import numpy as np
import asyncio
from pathlib import Path
from PIL import Image

# Import all modules
from backend.services.sdxl_service import sdxl_service
from backend.services.controlnet_service import controlnet_service
from backend.services.lora_service import lora_service
from backend.utils.video_processor import VideoProcessor

from backend.pipelines.conditioning_builder import condition_builder
from backend.pipelines.latent_preparation import LatentPreparation
from backend.pipelines.guidance_fusion import GuidanceFusion
from backend.pipelines.diffusion_core import DiffusionCore
from backend.pipelines.temporal_coherence import TemporalCoherence
from backend.pipelines.reconstruction import ReconstructionStabilization

class ColorizationPipeline:
    def __init__(self):
        # Initialize all modules
        self.latent_prep = LatentPreparation(sdxl_service)
        self.guidance_fusion = GuidanceFusion(sdxl_service, controlnet_service)
        self.diffusion_core = DiffusionCore(sdxl_service, controlnet_service, lora_service)
        self.temporal_coherence = TemporalCoherence()
        self.reconstruction = ReconstructionStabilization(sdxl_service)
        
        self.active_sessions = {}
    
    async def colorize_video(self, video_path, session_id, prompt, 
                            num_steps=15, guidance_scale=7.5, 
                            websocket=None, lora_name=None):
        """
        Complete video colorization pipeline.
        
        Args:
            video_path (str): Path to grayscale video
            session_id (str): Session identifier
            prompt (str): Text prompt for colorization
            num_steps (int): Diffusion steps (lower for faster, 15-20 recommended)
            guidance_scale (float): CFG scale
            websocket: WebSocket for progress streaming
            lora_name (str): Optional LoRA to apply
        """
        print(f"\n{'='*60}")
        print(f"Starting Colorization Pipeline: {session_id}")
        print(f"Video: {video_path}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}\n")
        
        self.active_sessions[session_id] = {"status": "running"}
        
        try:
            # 1. Load models
            print("Loading models...")
            sdxl_service.load_models()
            self.diffusion_core.load_controlnet_pipeline()
            
            # Apply LoRA if specified
            if lora_name:
                self.diffusion_core.apply_lora(lora_name)
            
            # 2. Initialize video
            info = VideoProcessor.get_video_info(video_path)
            frames = VideoProcessor.frame_generator(video_path)
            
            # 3. Setup output
            base_dir = Path(__file__).parent.parent.parent
            output_dir = base_dir / "results" / session_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            colorized_frames = []
            frame_idx = 0
            
            # 4. Process each frame
            for frame in frames:
                print(f"\nProcessing frame {frame_idx + 1}/{info['frame_count']}...")
                
                # Convert to grayscale if needed
                if len(frame.shape) == 3:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray_frame = frame
                
                # A. Build conditioning
                conditioning_data = condition_builder.build_full_conditioning(
                    frame, frame_idx, prompt, video_path
                )
                
                # B. Create conditioning bundle
                conditioning_bundle = self.guidance_fusion.create_conditioning_bundle(
                    conditioning_data, prompt
                )
                
                # C. Colorize with diffusion
                gray_pil = Image.fromarray(gray_frame).convert('RGB')
                colorized_pil = self.diffusion_core.colorize_frame(
                    gray_pil,
                    conditioning_bundle,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale
                )
                
                # D. Convert to numpy
                colorized_np = np.array(colorized_pil)
                
                # E. Apply temporal smoothing if not first frame
                if self.temporal_coherence.prev_colorized is not None:
                    colorized_np = self.temporal_coherence.apply_temporal_smoothing(
                        colorized_np,
                        self.temporal_coherence.prev_colorized
                    )
                
                # F. Preserve chrominance
                colorized_np = self.reconstruction.preserve_chrominance(
                    gray_frame, colorized_np
                )
                
                # G. Update color memory for tracked objects
                for track_id, mask in conditioning_data['instance_data']['masks'].items():
                    self.temporal_coherence.update_color_memory(
                        track_id, colorized_np, mask
                    )
                
                # H. Update temporal history
                self.temporal_coherence.update_frame_history(gray_frame, colorized_np)
                
                colorized_frames.append(colorized_np)
                
                # I. Stream progress
                if websocket:
                    # Resize for preview
                    preview = cv2.resize(colorized_np, (640, 360))
                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
                    
                    stats = {
                        "frame": frame_idx,
                        "total_frames": info['frame_count'],
                        "progress": (frame_idx + 1) / info['frame_count'] * 100,
                        "objects": len(conditioning_data['instance_data']['detections'])
                    }
                    
                    await websocket.send_bytes(buffer.tobytes())
                    await websocket.send_json(stats)
                
                frame_idx += 1
            
            # 5. Final temporal reassembly and flicker reduction
            print("\nApplying final stabilization...")
            colorized_frames = self.reconstruction.reduce_flicker(colorized_frames)
            
            # 6. Save output video
            output_path = output_dir / "colorized_output.mp4"
            print(f"Saving to {output_path}...")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                info['fps'],
                (colorized_frames[0].shape[1], colorized_frames[0].shape[0])
            )
            
            for frame in colorized_frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            out.release()
            
            self.active_sessions[session_id]["status"] = "completed"
            self.active_sessions[session_id]["output_path"] = str(output_path)
            
            print(f"\n{'='*60}")
            print(f"✓ Colorization Complete!")
            print(f"Output: {output_path}")
            print(f"{'='*60}\n")
            
            if websocket:
                await websocket.send_json({
                    "status": "completed",
                    "output_path": str(output_path)
                })
        
        except Exception as e:
            print(f"\n✗ Colorization Error: {e}")
            import traceback
            traceback.print_exc()
            
            self.active_sessions[session_id]["status"] = "error"
            self.active_sessions[session_id]["error"] = str(e)
            
            if websocket:
                await websocket.send_json({"error": str(e)})

# Global instance
colorization_pipeline = ColorizationPipeline()
