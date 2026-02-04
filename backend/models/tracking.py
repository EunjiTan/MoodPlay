"""
Tracking Module
Handles instance identity propagation across video frames.
Uses optical flow + IOU to match masks between frames.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment

from backend.core.logging import get_logger, time_execution

logger = get_logger("models.tracking")

class InstanceTracker:
    """
    Propagates instance IDs from frame t-1 to frame t.
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        self.next_id = 1
        self.active_tracks = {}  # {id: {'mask': mask, 'last_seen': frame_idx}}
        self.iou_threshold = iou_threshold
        self.max_missed_frames = 5
        
    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute Intersection over Union between two binary masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return intersection / union

    def _compute_flow_warped_iou(self, prev_mask: np.ndarray, curr_mask: np.ndarray, flow: np.ndarray) -> float:
        """Warp previous mask with optical flow and compute IOU with current mask."""
        h, w = prev_mask.shape
        
        # Warp prev_mask using flow
        # Flow is (H, W, 2)
        flow_map = np.column_stack((np.repeat(np.arange(h), w), np.tile(np.arange(w), h))).reshape(h, w, 2)
        
        # CV2 remap expects coordinates, not deltas. flow contains dx, dy.
        # We need to map DESTINATION pixels back to SOURCE pixels.
        # pos_src = pos_dst - flow(pos_dst)  (approx)
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Inverse mapping: pixel at (x,y) in current frame came from (x-dx, y-dy) in prev frame
        map_x = (x - flow[..., 0]).astype(np.float32)
        map_y = (y - flow[..., 1]).astype(np.float32)
        
        warped_prev = cv2.remap(
            prev_mask.astype(np.uint8), 
            map_x, 
            map_y, 
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return self._compute_iou(warped_prev, curr_mask)

    @time_execution(logger)
    def update(self, frame_idx: int, current_masks: List[Dict], optical_flow: np.ndarray = None) -> List[Dict]:
        """
        Assign IDs to current masks based on active tracks.
        """
        if not self.active_tracks:
            # Initialize tracks
            for mask_data in current_masks:
                mask_data['id'] = self.next_id
                self.active_tracks[self.next_id] = {
                    'mask': mask_data['segmentation'],
                    'last_seen': frame_idx
                }
                self.next_id += 1
            return current_masks

        # Match existing tracks to new masks
        track_ids = list(self.active_tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(current_masks)))
        
        # Fill cost matrix (1 - IOU)
        for i, tid in enumerate(track_ids):
            prev_mask = self.active_tracks[tid]['mask']
            
            for j, curr_mask_data in enumerate(current_masks):
                curr_mask = curr_mask_data['segmentation']
                
                if optical_flow is not None:
                     iou = self._compute_flow_warped_iou(prev_mask, curr_mask, optical_flow)
                else:
                    iou = self._compute_iou(prev_mask, curr_mask)
                    
                cost_matrix[i, j] = 1.0 - iou

        # Hungarian algorithm matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Assign matched IDs
        matched_indices = set()
        for r, c in zip(row_ind, col_ind):
            iou = 1.0 - cost_matrix[r, c]
            tid = track_ids[r]
            
            if iou > self.iou_threshold:
                current_masks[c]['id'] = tid
                self.active_tracks[tid]['mask'] = current_masks[c]['segmentation']
                self.active_tracks[tid]['last_seen'] = frame_idx
                matched_indices.add(c)
        
        # Create new tracks for unmatched
        for j in range(len(current_masks)):
            if j not in matched_indices:
                current_masks[j]['id'] = self.next_id
                self.active_tracks[self.next_id] = {
                    'mask': current_masks[j]['segmentation'],
                    'last_seen': frame_idx
                }
                self.next_id += 1
                
        # Prune old tracks
        ids_to_remove = []
        for tid, data in self.active_tracks.items():
            if frame_idx - data['last_seen'] > self.max_missed_frames:
                ids_to_remove.append(tid)
        
        for tid in ids_to_remove:
            del self.active_tracks[tid]
            
        logger.debug(f"Frame {frame_idx}: {len(current_masks)} masks, {len(self.active_tracks)} active tracks")
        return current_masks
