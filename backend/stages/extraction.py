"""
Stage 1: Frame Extraction
Extracts frames from video and prepares downscaled versions for processing.
"""

from PIL import Image
from backend.core.stage_runner import StageRunner
from backend.core.data_manager import DataManager
from backend.io.video_reader import VideoReader

class FrameExtractionStage(StageRunner):
    """
    Reads the video file and saves all frames to the workspace.
    Saves two versions: 
    - 0_raw_frames: Original resolution
    - 1_process_frames: Downscaled (512p) for VRAM safety
    """
    
    def run(self, video_path: str, max_frames: int = None):
        self.logger.info(f"Extracting frames from {video_path} (Limit: {max_frames})")
        
        # We use a context manager to ensure the reader closes
        with VideoReader(video_path, max_dimension=4096) as reader: # Read full res initially
            
            for idx, frame_pil in reader:
                if max_frames and idx >= max_frames:
                    break
                # 1. Save Raw
                self.dm.save_frame(frame_pil, idx, stage="raw")
                
                # 2. Downscale for processing (512px long edge)
                # SD1.5 usually trained on 512x512.
                # Maintaining aspect ratio is important.
                w, h = frame_pil.size
                scale = 512 / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                # Ensure dimensions are multiple of 8 (for VAE)
                new_w = (new_w // 8) * 8
                new_h = (new_h // 8) * 8
                
                proc_frame = frame_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
                self.dm.save_frame(proc_frame, idx, stage="proc")
                
                if idx % 50 == 0:
                    self.logger.info(f"Extracted frame {idx} (Proc Size: {new_w}x{new_h})")
                    
        self.logger.info(f"Extraction complete. Total frames: {self.dm.get_frame_count('raw')}")
