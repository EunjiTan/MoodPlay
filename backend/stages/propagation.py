"""
Stage 5: Propagation
Propagates color from keyframes to intermediate frames.
Uses CoTracker motion data or interpolated flow fields.
"""

import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from backend.core.stage_runner import StageRunner
from backend.io.video_reader import VideoReader


class PropagationStage(StageRunner):
    """
    Uses computed motion data to warp colors from keyframes to fill gaps.
    Supports both CoTracker sparse tracks and dense flow fields.
    """
    
    def _warp_image(self, img: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp image using dense flow field."""
        h, w = flow.shape[:2]
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Compute source coordinates
        map_x = (x + flow[..., 0]).astype(np.float32)
        map_y = (y + flow[..., 1]).astype(np.float32)
        
        warped = cv2.remap(
            img, 
            map_x, 
            map_y, 
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return warped
    
    def _load_cotracker_data(self) -> dict:
        """Load CoTracker motion data if available."""
        tracks_path = self.dm.dirs["flow"] / "cotracker_data.npz"
        if tracks_path.exists():
            data = np.load(tracks_path)
            return {
                'tracks': data['tracks'],
                'visibility': data['visibility'],
                'grid_size': int(data['grid_size'])
            }
        return None

    def run(self, keyframe_interval: int = 10, output_video_path: str = "output.mp4"):
        """
        Run propagation stage.
        
        Args:
            keyframe_interval: Interval used during keyframe colorization
            output_video_path: Output video file path
        """
        self.logger.info(f"Starting Propagation Stage (Interval: {keyframe_interval})")
        
        frame_count = self.dm.get_frame_count("proc")
        
        # Load CoTracker data if available
        cotracker_data = self._load_cotracker_data()
        if cotracker_data:
            self.logger.info(f"Using CoTracker data ({cotracker_data['grid_size']}x{cotracker_data['grid_size']} grid)")
        else:
            self.logger.info("Using interpolated flow fields")
        
        # Prepare Video Writer
        first_frame = self.dm.load_frame(0, stage="proc")
        w, h = first_frame.size
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (w, h))
        
        try:
            # Find all keyframe indices
            indices = list(range(0, frame_count, keyframe_interval))
            if indices[-1] != frame_count - 1:
                indices.append(frame_count - 1)
            
            # Process segment by segment
            for k_idx in range(len(indices) - 1):
                start_idx = indices[k_idx]
                end_idx = indices[k_idx + 1]
                
                # Load Keyframes
                img_start = np.array(self.dm.load_frame(
                    start_idx, 
                    stage="keys" if self.dm.frame_exists(start_idx, stage="keys") else "proc"
                ))
                img_end = np.array(self.dm.load_frame(
                    end_idx, 
                    stage="keys" if self.dm.frame_exists(end_idx, stage="keys") else "proc"
                ))
                
                # Propagate Forward from Start
                fwd_preds = {start_idx: img_start}
                curr_img = img_start
                for f_idx in range(start_idx, end_idx):
                    try:
                        flow_data = self.dm.load_flow(f_idx)
                        flow_bwd = flow_data['bwd']
                        
                        next_img_warped = self._warp_image(curr_img, flow_bwd)
                        fwd_preds[f_idx + 1] = next_img_warped
                        curr_img = next_img_warped
                    except Exception as e:
                        self.logger.warning(f"Flow failure at {f_idx}: {e}")
                        fwd_preds[f_idx + 1] = curr_img

                # Propagate Backward from End
                bwd_preds = {end_idx: img_end}
                curr_img = img_end
                for f_idx in range(end_idx, start_idx, -1):
                    try:
                        flow_data = self.dm.load_flow(f_idx - 1)
                        flow_fwd = flow_data['fwd']
                        
                        prev_img_warped = self._warp_image(curr_img, flow_fwd)
                        bwd_preds[f_idx - 1] = prev_img_warped
                        curr_img = prev_img_warped
                    except Exception as e:
                        self.logger.warning(f"Flow failure at {f_idx}: {e}")
                        bwd_preds[f_idx - 1] = curr_img

                # Blend and Write
                for i in range(start_idx, end_idx):
                    res_fwd = fwd_preds.get(i, fwd_preds[start_idx])
                    res_bwd = bwd_preds.get(i, bwd_preds[end_idx])
                    
                    # Linear crossfade weighting
                    dist_start = i - start_idx
                    dist_end = end_idx - i
                    total_dist = dist_start + dist_end
                    
                    alpha = dist_start / total_dist if total_dist > 0 else 0.5
                    
                    blended = cv2.addWeighted(res_fwd, 1.0 - alpha, res_bwd, alpha, 0)
                    out = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
                    writer.write(out)

            # Write the very last frame
            last_img = np.array(self.dm.load_frame(
                frame_count - 1, 
                stage="keys" if self.dm.frame_exists(frame_count - 1, stage="keys") else "proc"
            ))
            writer.write(cv2.cvtColor(last_img, cv2.COLOR_RGB2BGR))

        finally:
            writer.release()
            self.logger.info(f"Video saved to {output_video_path}")
            self.release_resources()
