"""
SAM-2 Video Segmentation Tracker with multi-modal prompting support.
Supports text prompts, visual prompts (clicks), and bounding box prompts.
"""

import os
import numpy as np
from PIL import Image
import cv2
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Generator

# Global flags
TORCH_AVAILABLE = False
SAM2_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    print("Warning: Torch not available. SAM-2 tracking will be mocked.")

try:
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    print("Warning: SAM-2 not available. Using mock segmentation.")


@dataclass
class SegmentationPrompt:
    """Represents a segmentation prompt with multiple modalities."""
    object_id: int
    frame_idx: int = 0
    
    # Click prompts (positive and negative)
    points: Optional[np.ndarray] = None  # (N, 2) array of [x, y] coordinates
    labels: Optional[np.ndarray] = None  # (N,) array: 1=positive, 0=negative
    
    # Bounding box prompt
    box: Optional[np.ndarray] = None  # [x1, y1, x2, y2]
    
    # Text prompt (requires CLIP integration)
    text: Optional[str] = None


class SAM2Tracker:
    """
    SAM-2 based video object segmentation with temporal tracking.
    Supports multi-modal prompting and mask propagation.
    """
    
    def __init__(self, 
                 checkpoint_path: str, 
                 model_cfg_path: str,
                 confidence_threshold: float = 0.7):
        self.device = "cpu"
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"
            
        self.checkpoint_path = checkpoint_path
        self.model_cfg_path = model_cfg_path
        self.confidence_threshold = confidence_threshold
        
        self.predictor = None
        self.inference_state = None
        self.video_path = None
        self.frame_count = 0
        
        # Track objects and their prompts
        self.objects: Dict[int, SegmentationPrompt] = {}
        self.masks: Dict[int, Dict[int, np.ndarray]] = {}  # object_id -> frame_idx -> mask

    def _load_model(self):
        """Load SAM-2 video predictor."""
        if not SAM2_AVAILABLE or not TORCH_AVAILABLE:
            print("SAM-2 not available, running in mock mode.")
            return
            
        print(f"Loading SAM-2 from {self.checkpoint_path}...")
        self.predictor = build_sam2_video_predictor(
            self.model_cfg_path, 
            self.checkpoint_path,
            device=self.device
        )
        print("SAM-2 Loaded.")

    def init_video(self, video_path: str) -> int:
        """
        Initialize the inference state for a new video.
        
        Args:
            video_path: Path to video file or directory of frames
            
        Returns:
            Number of frames in the video
        """
        print(f"Initializing video state for: {video_path}")
        self.video_path = video_path
        self.objects.clear()
        self.masks.clear()
        
        if self.predictor is None and SAM2_AVAILABLE:
            self._load_model()
        
        if SAM2_AVAILABLE and self.predictor:
            self.inference_state = self.predictor.init_state(video_path=video_path)
            # Get frame count from video
            cap = cv2.VideoCapture(video_path)
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            # Mock: assume 30 frames
            self.frame_count = 30
        
        return self.frame_count

    def add_click(self, 
                  frame_idx: int, 
                  object_id: int, 
                  points: List[Tuple[int, int]], 
                  labels: List[int]) -> np.ndarray:
        """
        Add click prompts to segment an object.
        
        Args:
            frame_idx: Frame index to add clicks on
            object_id: Unique ID for the object
            points: List of (x, y) click coordinates
            labels: List of labels (1=positive/include, 0=negative/exclude)
            
        Returns:
            Initial mask for the object on this frame
        """
        points_np = np.array(points, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.int32)
        
        prompt = SegmentationPrompt(
            object_id=object_id,
            frame_idx=frame_idx,
            points=points_np,
            labels=labels_np
        )
        self.objects[object_id] = prompt
        
        print(f"Adding clicks at frame {frame_idx} for object {object_id}: {len(points)} points")
        
        if SAM2_AVAILABLE and self.predictor:
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=object_id,
                points=points_np,
                labels=labels_np
            )
            
            # Convert logits to binary mask
            mask = (out_mask_logits[0] > 0).cpu().numpy().squeeze()
            
            # Store mask
            if object_id not in self.masks:
                self.masks[object_id] = {}
            self.masks[object_id][frame_idx] = mask
            
            return mask
        else:
            # Mock: return a circular mask around the first click
            if len(points) > 0:
                mask = self._mock_mask_from_click(points[0], (512, 512))
            else:
                mask = np.zeros((512, 512), dtype=bool)
            
            if object_id not in self.masks:
                self.masks[object_id] = {}
            self.masks[object_id][frame_idx] = mask
            
            return mask

    def add_box(self, 
                frame_idx: int, 
                object_id: int, 
                box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Add bounding box prompt to segment an object.
        
        Args:
            frame_idx: Frame index
            object_id: Unique ID for the object
            box: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Initial mask for the object
        """
        box_np = np.array(box, dtype=np.float32)
        
        prompt = SegmentationPrompt(
            object_id=object_id,
            frame_idx=frame_idx,
            box=box_np
        )
        self.objects[object_id] = prompt
        
        print(f"Adding box at frame {frame_idx} for object {object_id}: {box}")
        
        if SAM2_AVAILABLE and self.predictor:
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=object_id,
                box=box_np
            )
            
            mask = (out_mask_logits[0] > 0).cpu().numpy().squeeze()
            
            if object_id not in self.masks:
                self.masks[object_id] = {}
            self.masks[object_id][frame_idx] = mask
            
            return mask
        else:
            # Mock: return rectangular mask
            mask = self._mock_mask_from_box(box, (512, 512))
            
            if object_id not in self.masks:
                self.masks[object_id] = {}
            self.masks[object_id][frame_idx] = mask
            
            return mask

    def add_text_prompt(self, 
                        frame_idx: int, 
                        object_id: int, 
                        text: str) -> np.ndarray:
        """
        Add text prompt for object segmentation (requires CLIP integration).
        
        Args:
            frame_idx: Frame index
            object_id: Unique ID for the object
            text: Text description (e.g., "red car", "person in blue shirt")
            
        Returns:
            Initial mask for the object
        """
        prompt = SegmentationPrompt(
            object_id=object_id,
            frame_idx=frame_idx,
            text=text
        )
        self.objects[object_id] = prompt
        
        print(f"Adding text prompt at frame {frame_idx} for object {object_id}: '{text}'")
        
        # Text prompts require additional CLIP-based grounding
        # For now, this is a placeholder that would integrate with
        # Grounding DINO or similar for text-to-box then SAM
        
        # Mock response
        mask = np.zeros((512, 512), dtype=bool)
        mask[100:300, 100:300] = True  # Placeholder region
        
        if object_id not in self.masks:
            self.masks[object_id] = {}
        self.masks[object_id][frame_idx] = mask
        
        return mask

    def add_negative_region(self, 
                            frame_idx: int, 
                            object_id: int, 
                            points: List[Tuple[int, int]]) -> np.ndarray:
        """
        Add negative clicks to exclude regions from an existing object mask.
        
        Args:
            frame_idx: Frame index
            object_id: ID of the object to refine
            points: List of (x, y) coordinates to exclude
            
        Returns:
            Refined mask
        """
        if object_id not in self.objects:
            raise ValueError(f"Object {object_id} not found. Add positive prompt first.")
        
        # Add negative points to existing prompt
        existing = self.objects[object_id]
        neg_points = np.array(points, dtype=np.float32)
        neg_labels = np.zeros(len(points), dtype=np.int32)  # 0 = negative
        
        if existing.points is not None:
            existing.points = np.vstack([existing.points, neg_points])
            existing.labels = np.concatenate([existing.labels, neg_labels])
        else:
            existing.points = neg_points
            existing.labels = neg_labels
        
        print(f"Adding {len(points)} negative points to object {object_id}")
        
        # Re-run segmentation with updated prompts
        return self.add_click(
            frame_idx,
            object_id,
            existing.points.tolist(),
            existing.labels.tolist()
        )

    def propagate(self, 
                  start_frame: int = 0,
                  direction: str = "both") -> Generator[Tuple[int, List[int], Dict[int, np.ndarray]], None, None]:
        """
        Propagate masks throughout the video.
        
        Args:
            start_frame: Starting frame index
            direction: "forward", "backward", or "both"
            
        Yields:
            Tuple of (frame_idx, object_ids, masks_dict)
        """
        print(f"Propagating masks from frame {start_frame}, direction: {direction}...")
        
        if SAM2_AVAILABLE and self.predictor:
            # Real SAM-2 propagation
            for frame_idx, object_ids, mask_logits in self.predictor.propagate_in_video(
                self.inference_state,
                start_frame_idx=start_frame,
                max_frame_num_to_track=self.frame_count
            ):
                masks_dict = {}
                for i, obj_id in enumerate(object_ids):
                    mask = (mask_logits[i] > 0).cpu().numpy().squeeze()
                    masks_dict[obj_id] = mask
                    
                    # Store mask
                    if obj_id not in self.masks:
                        self.masks[obj_id] = {}
                    self.masks[obj_id][frame_idx] = mask
                
                yield frame_idx, list(object_ids), masks_dict
        else:
            # Mock propagation
            object_ids = list(self.objects.keys())
            
            for frame_idx in range(self.frame_count):
                masks_dict = {}
                for obj_id in object_ids:
                    # Simulate mask movement/scaling
                    base_mask = self._get_or_create_mock_mask(obj_id, frame_idx)
                    masks_dict[obj_id] = base_mask
                    
                    if obj_id not in self.masks:
                        self.masks[obj_id] = {}
                    self.masks[obj_id][frame_idx] = base_mask
                
                yield frame_idx, object_ids, masks_dict

    def get_mask(self, object_id: int, frame_idx: int) -> Optional[np.ndarray]:
        """Get the mask for a specific object and frame."""
        if object_id in self.masks and frame_idx in self.masks[object_id]:
            return self.masks[object_id][frame_idx]
        return None

    def get_combined_mask(self, frame_idx: int) -> np.ndarray:
        """Get combined mask of all objects for a frame."""
        combined = None
        for obj_id in self.objects:
            mask = self.get_mask(obj_id, frame_idx)
            if mask is not None:
                if combined is None:
                    combined = mask.copy()
                else:
                    combined = combined | mask
        
        if combined is None:
            combined = np.zeros((512, 512), dtype=bool)
        
        return combined

    def export_masks(self, output_dir: str, prefix: str = "mask") -> Dict[int, List[str]]:
        """
        Export all masks to files.
        
        Args:
            output_dir: Directory to save masks
            prefix: Filename prefix
            
        Returns:
            Dict mapping object_id to list of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        for obj_id, frame_masks in self.masks.items():
            saved_files[obj_id] = []
            
            for frame_idx, mask in frame_masks.items():
                filename = f"{prefix}_obj{obj_id}_frame{frame_idx:05d}.png"
                filepath = os.path.join(output_dir, filename)
                
                # Convert boolean mask to 8-bit image
                mask_img = (mask.astype(np.uint8) * 255)
                cv2.imwrite(filepath, mask_img)
                saved_files[obj_id].append(filepath)
        
        return saved_files

    # =========================================================================
    # Mock helpers for when SAM-2 is not available
    # =========================================================================
    
    def _mock_mask_from_click(self, point: Tuple[int, int], size: Tuple[int, int]) -> np.ndarray:
        """Create a circular mask around a click point."""
        mask = np.zeros(size, dtype=bool)
        y, x = np.ogrid[:size[0], :size[1]]
        cx, cy = point
        radius = min(size) // 8
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask[dist <= radius] = True
        return mask

    def _mock_mask_from_box(self, box: Tuple[int, int, int, int], size: Tuple[int, int]) -> np.ndarray:
        """Create a rectangular mask from bounding box."""
        mask = np.zeros(size, dtype=bool)
        x1, y1, x2, y2 = box
        x1, x2 = max(0, x1), min(size[1], x2)
        y1, y2 = max(0, y1), min(size[0], y2)
        mask[y1:y2, x1:x2] = True
        return mask

    def _get_or_create_mock_mask(self, obj_id: int, frame_idx: int) -> np.ndarray:
        """Get existing mask or create a mock one with simulated movement."""
        if obj_id in self.masks and frame_idx in self.masks[obj_id]:
            return self.masks[obj_id][frame_idx]
        
        # Find nearest existing mask and shift it slightly
        if obj_id in self.masks and len(self.masks[obj_id]) > 0:
            nearest_frame = min(self.masks[obj_id].keys(), key=lambda f: abs(f - frame_idx))
            base_mask = self.masks[obj_id][nearest_frame]
            
            # Simulate movement: shift mask by a few pixels
            shift_x = (frame_idx - nearest_frame) * 2
            shift_y = frame_idx - nearest_frame
            
            shifted = np.roll(np.roll(base_mask, shift_x, axis=1), shift_y, axis=0)
            return shifted
        
        # Default empty mask
        return np.zeros((512, 512), dtype=bool)


# Example usage
if __name__ == "__main__":
    tracker = SAM2Tracker("checkpoints/sam2_hiera_large.pt", "configs/sam2_hiera_l.yaml")
    
    # Initialize video
    tracker.init_video("test_video.mp4")
    
    # Add click to segment object
    mask = tracker.add_click(
        frame_idx=0,
        object_id=1,
        points=[(256, 256)],
        labels=[1]  # Positive click
    )
    
    # Propagate through video
    for frame_idx, obj_ids, masks in tracker.propagate():
        print(f"Frame {frame_idx}: {len(obj_ids)} objects tracked")
