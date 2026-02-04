"""
Stage 5: Propagation
Propagates color from keyframes to intermediate frames using optical flow.
"""

import cv2
import numpy as np
from tqdm import tqdm
from backend.core.stage_runner import StageRunner
from backend.io.video_reader import VideoReader

class PropagationStage(StageRunner):
    """
    Uses computed flow to warp colors from keyframes to fill gaps.
    """
    
    def _warp_image(self, img: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp image using flow field."""
        h, w = flow.shape[:2]
        flow = -flow # Inverse flow for remap? 
        # Actually standard backward warp: Destination(x,y) pulls from Source(x+u, y+v)
        # Flow computed as "Where does pixel at T go to at T+1" is Forward Flow.
        # Flow computed as "Where does pixel at T+1 come from at T" is Backward Flow.
        
        # We need to pull pixels.
        flow_map = np.column_stack((np.repeat(np.arange(h), w), np.tile(np.arange(w), h))).reshape(h, w, 2)
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # remap expects: map_x(y,x), map_y(y,x) containing the source coordinates
        # If flow is "Displacement from Current to Prev", then Src = Curr + Flow
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

    def run(self, keyframe_interval: int = 5, output_video_path: str = "output.mp4"):
        self.logger.info(f"Starting Propagation Stage (Interval: {keyframe_interval})")
        
        frame_count = self.dm.get_frame_count("proc")
        
        # Prepare Video Writer
        # We assume 30fps for now, ideally read from metadata
        first_frame = self.dm.load_frame(0, stage="proc")
        w, h = first_frame.size
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (w, h))
        
        try:
        try:
            # We process segment by segment (Keyframe to Keyframe)
            # Find all keyframe indices
            indices = list(range(0, frame_count, keyframe_interval))
            if indices[-1] != frame_count - 1:
                indices.append(frame_count - 1) # Ensure last frame is handled
            
            # Dictionary to store final outputs before writing (buffer for blending)
            # Since video is linear, we can write as we resolve segments.
            # Segment: [K_start, ..., K_end]
            
            # Initialization
            for k_idx in range(len(indices) - 1):
                start_idx = indices[k_idx]
                end_idx = indices[k_idx + 1]
                
                # Load Keyframes
                img_start = np.array(self.dm.load_frame(start_idx, stage="keys" if self.dm.frame_exists(start_idx, stage="keys") else "proc"))
                img_end = np.array(self.dm.load_frame(end_idx, stage="keys" if self.dm.frame_exists(end_idx, stage="keys") else "proc"))
                
                # Propagate Forward from Start
                # We need a list to hold forward propagated frames for this segment
                fwd_preds = {start_idx: img_start}
                curr_img = img_start
                for f_idx in range(start_idx, end_idx):
                    # To go f_idx -> f_idx + 1:
                    # We need Flow(From f_idx+1 to f_idx).
                    # 'flow.npz' at f_idx contains 'bwd' which is Flow(f_idx+1 -> f_idx).
                    try:
                        flow_data = self.dm.load_flow(f_idx)
                        flow_bwd = flow_data['bwd'] # Vector pointing from next to curr
                        
                        next_img_warped = self._warp_image(curr_img, flow_bwd)
                        fwd_preds[f_idx + 1] = next_img_warped
                        curr_img = next_img_warped
                    except Exception as e:
                        self.logger.warning(f"Flow failure at {f_idx}: {e}")
                        fwd_preds[f_idx + 1] = curr_img # Fallback copy

                # Propagate Backward from End
                bwd_preds = {end_idx: img_end}
                curr_img = img_end
                for f_idx in range(end_idx, start_idx, -1):
                    # To go f_idx -> f_idx - 1:
                    # We need Flow(From f_idx-1 to f_idx).
                    # 'flow.npz' at f_idx-1 contains 'fwd' which is Flow(f_idx-1 -> f_idx).
                    # We use this as the "lookup" for remap to pull pixels from f_idx to f_idx-1.
                    try:
                        flow_data = self.dm.load_flow(f_idx - 1)
                        flow_fwd = flow_data['fwd'] # Vector pointing from curr to next
                        
                        # We want to create Image at f_idx-1 using Image at f_idx.
                        # Standard Remap: Dest(x,y) = Source(Map(x,y))
                        # Here Source is Image at f_idx. Dest is Image at f_idx-1.
                        # Flow_fwd(u,v) at (x,y) in f_idx-1 points to (x',y') in f_idx.
                        # So Dest(x,y) should take color from Source(x+u, y+v).
                        # This works perfectly with _warp_image logic if we pass flow_fwd!
                        
                        prev_img_warped = self._warp_image(curr_img, flow_fwd)
                        bwd_preds[f_idx - 1] = prev_img_warped
                        curr_img = prev_img_warped
                    except Exception as e:
                        self.logger.warning(f"Flow failure at {f_idx}: {e}")
                        bwd_preds[f_idx - 1] = curr_img

                # Blend and Write
                for i in range(start_idx, end_idx):
                    # Frame index i
                    res_fwd = fwd_preds.get(i, fwd_preds[start_idx])
                    res_bwd = bwd_preds.get(i, bwd_preds[end_idx])
                    
                    # Weighting
                    dist_start = i - start_idx
                    dist_end = end_idx - i
                    total_dist = dist_start + dist_end
                    
                    alpha = dist_start / total_dist # Weight for Ending (Linear crossfade)
                    
                    # Advanced: Use Gaussian weighting or occlusion masks?
                    # Linear is stable for V1.
                    
                    blended = cv2.addWeighted(res_fwd, 1.0 - alpha, res_bwd, alpha, 0)
                    out = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
                    writer.write(out)

            # Write the very last frame
            last_img = np.array(self.dm.load_frame(frame_count - 1, stage="keys" if self.dm.frame_exists(frame_count - 1, stage="keys") else "proc"))
            writer.write(cv2.cvtColor(last_img, cv2.COLOR_RGB2BGR))

        finally:
            writer.release()
            self.logger.info(f"Video saved to {output_video_path}")
            self.release_resources()
