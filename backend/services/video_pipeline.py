
import os
import cv2
import time
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from .yolo_service import yolo_service
from .sam3_service import sam3_service
from ..utils.video_processor import VideoProcessor
from ..utils.visualization import VisualizationUtils

class VideoPipeline:
    def __init__(self):
        self.active_sessions = {}
        self.executor = ThreadPoolExecutor(max_workers=2)

    async def process_video(self, video_path, session_id, websocket=None):
        """
        Main processing loop.
        """
        print(f"Starting pipeline for {session_id}")
        self.active_sessions[session_id] = {"status": "running"}
        
        try:
            # 1. Initialize Video
            info = VideoProcessor.get_video_info(video_path)
            frames = VideoProcessor.frame_generator(video_path)
            
            # 2. Initialize SAM Session
            sam3_service.init_session(video_path)
            
            # 3. Setup Export Paths
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(base_dir, "results", session_id)
            masks_dir = os.path.join(output_dir, "masks")
            os.makedirs(masks_dir, exist_ok=True)
            
            frame_idx = 0
            start_time = time.time()
            
            for frame in frames:
                if self.active_sessions.get(session_id, {}).get("status") == "stopped":
                    break
                
                # DEMO OPTIMIZATION: Skip frames to speed up analysis
                # Process every 5th frame (Speed x5)
                if frame_idx % 5 != 0:
                     frame_idx += 1
                     continue

                
                # Convert RGB to BGR for OpenCV/YOLO
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # A. Detect (YOLO)
                detections = yolo_service.detect(bgr_frame)
                
                # B. Segment (SAM)
                # Note: SAM Video Predictor needs the *original* video path and frame indices
                # It handles loading internally if we use init_state(video_path).
                # But here we are iterating frames manually. 
                # SAM2 Video Predictor actually expects us to pass inputs (prompts) and it manages state.
                # Since we already called init_session(video_path), SAM2 has loaded the video frames internally?
                # YES: build_sam2_video_predictor reads the video frames into memory/cache.
                
                # ISSUE: If SAM2 reads the video itself, we need to ensure we are on the sync frame index.
                # YES: build_sam2_video_predictor reads the video frames into memory/cache.
                
                # OPTIMIZATION: Switched to Ultralytics SAM for stability. Requires passing the frame image.
                masks_dict = sam3_service.segment_frame_boxes_with_image(bgr_frame, detections)
                
                # C. Visualize
                masks_list = [masks_dict.get(d.get('track_id')) for d in detections]
                annotated_frame = VisualizationUtils.draw_detections(bgr_frame, detections)
                annotated_frame = VisualizationUtils.overlay_masks(annotated_frame, masks_list)
                
                # D. Export Mask
                # Save combined mask or individual? Requirement: "Export segmentation masks for each object"
                for obj_id, mask in masks_dict.items():
                    if mask is not None:
                        mask_path = os.path.join(masks_dir, f"frame_{frame_idx:04d}_obj_{obj_id}.png")
                        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
                
                # E. Stream Update
                if websocket:
                    # Resize for faster transfer if needed
                    preview = cv2.resize(annotated_frame, (640, 360))
                    _, buffer = cv2.imencode('.jpg', preview)
                    
                    stats = {
                        "frame": frame_idx,
                        "total_frames": info['frame_count'],
                        "fps": frame_idx / (time.time() - start_time + 1e-6),
                        "objects": len(detections),
                        "detections": detections
                    }
                    
                    await websocket.send_bytes(buffer.tobytes())
                    await websocket.send_json(stats)
                
                frame_idx += 1
                
            self.active_sessions[session_id]["status"] = "completed"
            
            if websocket:
                await websocket.send_json({"status": "completed", "progress": 100})
            print(f"Pipeline {session_id} finished successfully.")
            
            # CRITICAL OPTIMIZATION: Unload SAM to free VRAM for Colorization
            print("Unloading SAM model to free VRAM...")
            sam3_service.model = None
            sam3_service.model_loaded = False
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            print("VRAM cleared.")
            
        except Exception as e:
            print(f"Pipeline Error: {e}")
            self.active_sessions[session_id]["status"] = "error"
            self.active_sessions[session_id]["error"] = str(e)
            if websocket:
                await websocket.send_json({"error": str(e)})

# Global Pipeline
video_pipeline = VideoPipeline()
