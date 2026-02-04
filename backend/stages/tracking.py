"""
Stage 3: Tracking / Flow
Computes optical flow or point tracks between frames.
Essential for propagating colors from keyframes to intermediate frames.
"""

import cv2
import numpy as np
from tqdm import tqdm
from backend.core.stage_runner import StageRunner

class TrackingStage(StageRunner):
    """
    Computes dense optical flow (Forward and Backward) for propagation.
    Saves flow fields to disk.
    """
    
    def run(self):
        self.logger.info("Starting Tracking/Flow Stage")
        
        frame_count = self.dm.get_frame_count("proc")
        
        # We need pairs of frames: (t, t+1) and (t+1, t)
        # For simple propagation:
        # Flow (t -> t+1): Tells us where pixels at t went to at t+1
        # Flow (t+1 -> t): Tells us where pixels at t+1 came from (for backward warp)
        
        # Using Farneback for now as a robust fallback if CoTracker isn't installed.
        # Ideally we wrap CoTracker here if available.
        
        prev_gray = None
        
        for idx in tqdm(range(frame_count - 1), desc="Computing Flow"):
            # Load frames t and t+1
            if idx == 0:
                curr = self.dm.load_frame(idx, stage="proc")
                curr_gray = cv2.cvtColor(np.array(curr), cv2.COLOR_RGB2GRAY)
            else:
                curr_gray = next_gray
                
            next_frame = self.dm.load_frame(idx + 1, stage="proc")
            next_gray = cv2.cvtColor(np.array(next_frame), cv2.COLOR_RGB2GRAY)
            
            # Compute Forward Flow (t -> t+1)
            # "Where does pixel (x,y) in Curr move to in Next?"
            flow_fwd = cv2.calcOpticalFlowFarneback(
                curr_gray, next_gray, None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Compute Backward Flow (t+1 -> t)
            # "Where does pixel (x,y) in Next come from in Curr?"
            flow_bwd = cv2.calcOpticalFlowFarneback(
                next_gray, curr_gray, None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Save flow. We save full dense flow.
            # Filename convention: flow_{idx} contains flow usually to idx+1
            # But we need both. Let's bundle.
            
            # Save structure:
            # idx.npz -> {'fwd': flow to idx+1, 'bwd': flow from idx+1 to idx}
            
            save_path = self.dm.dirs["flow"] / f"{idx:06d}.npz"
            np.savez_compressed(save_path, fwd=flow_fwd, bwd=flow_bwd)
            
        self.logger.info("Flow computation complete.")
        self.release_resources()
