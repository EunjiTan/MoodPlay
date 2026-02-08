"""
Orchestrator V2 (Staged)
Executes the pipeline stages sequentially.
"""

import gc
import torch
import shutil
from pathlib import Path
from backend.core.logging import get_logger
from backend.core.data_manager import get_data_manager
from backend.core.device import device_manager

# Import all stages
from backend.stages.extraction import FrameExtractionStage
from backend.stages.segmentation import SegmentationStage
from backend.stages.tracking import TrackingStage
from backend.stages.keyframe_colorization import KeyframeColorizationStage
from backend.stages.propagation import PropagationStage

logger = get_logger("orchestrator.v2")

class StagedOrchestrator:
    
    def __init__(self):
        pass
        
    def run(self, 
            input_video: str, 
            output_name: str = "output_staged.mp4",
            style_name: str = "cinematic",
            keyframe_interval: int = 5,
            job_id: str = "latest_job",
            clean_start: bool = True,
            max_frames: int = None,
            seed: int = None,
            segment_frames: bool = False,
            track_motion: bool = False,
            num_inference_steps: int = 20):
        
        logger.info(f"Starting Staged Pipeline (Job: {job_id})")
        
        # 1. Setup Workspace
        dm = get_data_manager(job_id)
        if clean_start:
            logger.warning(f"Cleaning workspace: {dm.base_dir}")
            dm.clear_workspace()
            
        # 2. Stage 1: Extraction
        # Only run if raw frames are empty or we want to overwrite
        if dm.get_frame_count("raw") == 0:
            stage1 = FrameExtractionStage(dm)
            stage1.execute(input_video, max_frames=max_frames)
            del stage1
            
        # 3. Stage 2: Segmentation (Optional)
        if segment_frames:
            stage2 = SegmentationStage(dm)
            stage2.execute()
            del stage2
        
        # 4. Stage 3: Tracking (Optional)
        if track_motion:
            stage3 = TrackingStage(dm)
            stage3.execute()
            del stage3
        
        # 5. Stage 4: Keyframe Colorization
        stage4 = KeyframeColorizationStage(dm)
        stage4.execute(mood=style_name, keyframe_interval=keyframe_interval, seed=seed, num_inference_steps=num_inference_steps)
        del stage4
        
        # 6. Stage 5: Propagation / Assembly
        output_path = str(Path("results") / output_name)
        Path("results").mkdir(exist_ok=True)
        
        stage5 = PropagationStage(dm)
        stage5.execute(keyframe_interval=keyframe_interval, output_video_path=output_path)
        del stage5
        
        logger.info(f"Staged Pipeline Complete. Output: {output_path}")

staged_pipeline = StagedOrchestrator()
