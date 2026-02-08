"""
CoTracker Motion Tracking Module
Replaces optical flow with point-based tracking across video frames.
Better temporal consistency and accuracy than Farneback optical flow.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

from backend.core.logging import get_logger, log_gpu_memory, time_execution
from backend.core.device import get_device, get_dtype, oom_safe, device_manager
from backend.core.exceptions import ModelLoadError, ResourceError

logger = get_logger("models.cotracker_motion")


class CoTrackerMotion:
    """
    CoTracker wrapper for point tracking across video frames.
    Replaces optical flow with better temporal consistency.
    
    Features:
    - Track points across entire video
    - Configurable grid density
    - Handles occlusions
    - Better accuracy than dense optical flow
    """
    
    CHECKPOINT_MAP = {
        "cotracker2": "cotracker2.pth",
        "cotracker": "cotracker.pth",
        "cotracker3": "scaled_offline.pth"
    }
    
    def __init__(self, model_name: str = "cotracker3"):
        """
        Initialize CoTracker.
        
        Args:
            model_name: 'cotracker3' (recommended), 'cotracker2', or 'cotracker'
        """
        if model_name not in self.CHECKPOINT_MAP:
            raise ValueError(f"Unknown model: {model_name}. "
                           f"Choose from: {list(self.CHECKPOINT_MAP.keys())}")
        
        self.model_name = model_name
        self.device = get_device()
        self.model = None
        
        # Checkpoint paths
        self.checkpoint_dir = Path("checkpoints/cotracker")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CoTracker initialized (model={model_name})")
    
    def _get_checkpoint_path(self) -> Path:
        """Get path to checkpoint file."""
        return self.checkpoint_dir / self.CHECKPOINT_MAP[self.model_name]
    
    def _check_checkpoint_exists(self) -> bool:
        """Check if checkpoint file exists."""
        return self._get_checkpoint_path().exists()
    
    @time_execution(logger)
    def load_model(self):
        """Load CoTracker model to GPU."""
        if self.model is not None:
            return
        
        checkpoint_path = self._get_checkpoint_path()
        
        try:
            logger.info(f"Loading CoTracker {self.model_name}...")
            
            # Try importing cotracker
            try:
                from cotracker.predictor import CoTrackerPredictor
            except ImportError:
                # Try alternative import path
                try:
                    from co_tracker.predictor import CoTrackerPredictor
                except ImportError:
                    raise ModelLoadError(
                        "CoTracker package not installed. "
                        "Install with: pip install git+https://github.com/facebookresearch/co-tracker.git"
                    )
            
            # Load model - CoTracker can auto-download if checkpoint not found
            if checkpoint_path.exists():
                self.model = CoTrackerPredictor(
                    checkpoint=str(checkpoint_path)
                )
            else:
                # Try online loading (auto-download)
                logger.info("Checkpoint not found locally, attempting online load...")
                self.model = CoTrackerPredictor(
                    checkpoint=f"facebook/{self.model_name}"
                )
            
            self.model = self.model.to(self.device)
            
            log_gpu_memory(logger, "CoTracker Loaded")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load CoTracker model: {e}")
    
    @oom_safe(logger)
    def track_video(
        self, 
        frames: List[np.ndarray],
        grid_size: int = 20,
        backward_tracking: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Track points across entire video using a grid.
        
        Args:
            frames: List of frames as numpy arrays (H, W, C) in RGB
            grid_size: Grid density (grid_size x grid_size points tracked)
            backward_tracking: Also track backwards for bidirectional consistency
            
        Returns:
            Dict with keys:
            - 'tracks': (N, T, 2) array of point positions
            - 'visibility': (N, T) boolean array of point visibility
            - 'grid_size': int, the grid size used
        """
        if self.model is None:
            self.load_model()
        
        if len(frames) == 0:
            raise ValueError("No frames provided")
        
        logger.info(f"Tracking {len(frames)} frames with {grid_size}x{grid_size} grid...")
        
        # Convert frames to tensor: (1, T, C, H, W)
        frames_np = np.stack(frames, axis=0)  # (T, H, W, C)
        frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # (T, C, H, W)
        frames_tensor = frames_tensor.unsqueeze(0).float().to(self.device)  # (1, T, C, H, W)
        
        with torch.no_grad():
            # Track with grid query
            pred_tracks, pred_visibility = self.model(
                frames_tensor,
                grid_size=grid_size,
                backward_tracking=backward_tracking
            )
        
        # Convert to numpy
        tracks = pred_tracks[0].cpu().numpy()  # (N, T, 2)
        visibility = pred_visibility[0].cpu().numpy()  # (N, T)
        
        logger.info(f"Tracked {tracks.shape[0]} points")
        
        return {
            'tracks': tracks,
            'visibility': visibility,
            'grid_size': grid_size
        }
    
    @oom_safe(logger)
    def track_points(
        self,
        frames: List[np.ndarray],
        query_points: np.ndarray,
        query_frame: int = 0
    ) -> Dict[str, np.ndarray]:
        """
        Track specific points across video.
        
        Args:
            frames: List of frames as numpy arrays
            query_points: (N, 2) array of starting point coordinates
            query_frame: Frame index where points are defined
            
        Returns:
            Dict with keys:
            - 'tracks': (N, T, 2) array of point positions
            - 'visibility': (N, T) boolean array
        """
        if self.model is None:
            self.load_model()
        
        # Convert frames to tensor
        frames_np = np.stack(frames, axis=0)
        frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)
        frames_tensor = frames_tensor.unsqueeze(0).float().to(self.device)
        
        # Prepare query points: (1, N, 3) with [t, x, y]
        n_points = query_points.shape[0]
        queries = np.zeros((n_points, 3), dtype=np.float32)
        queries[:, 0] = query_frame  # Time index
        queries[:, 1:] = query_points  # x, y coordinates
        queries_tensor = torch.from_numpy(queries).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_tracks, pred_visibility = self.model(
                frames_tensor,
                queries=queries_tensor
            )
        
        return {
            'tracks': pred_tracks[0].cpu().numpy(),
            'visibility': pred_visibility[0].cpu().numpy()
        }
    
    def compute_motion_vectors(
        self,
        motion_data: Dict[str, np.ndarray],
        source_frame: int,
        target_frame: int
    ) -> np.ndarray:
        """
        Compute motion vectors from source to target frame.
        
        Args:
            motion_data: Output from track_video()
            source_frame: Source frame index
            target_frame: Target frame index
            
        Returns:
            (N, 2) array of motion vectors [dx, dy] for each tracked point
        """
        tracks = motion_data['tracks']  # (N, T, 2)
        
        source_pos = tracks[:, source_frame, :]  # (N, 2)
        target_pos = tracks[:, target_frame, :]  # (N, 2)
        
        motion = target_pos - source_pos  # (N, 2)
        
        return motion
    
    def interpolate_dense_flow(
        self,
        motion_data: Dict[str, np.ndarray],
        source_frame: int,
        target_frame: int,
        frame_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Interpolate sparse point tracks to dense optical flow field.
        For compatibility with existing propagation code.
        
        Args:
            motion_data: Output from track_video()
            source_frame: Source frame index
            target_frame: Target frame index  
            frame_shape: (H, W) output shape
            
        Returns:
            (H, W, 2) dense flow field
        """
        import cv2
        
        tracks = motion_data['tracks']
        visibility = motion_data['visibility']
        
        h, w = frame_shape
        
        # Get source and target positions
        source_pos = tracks[:, source_frame, :]  # (N, 2)
        target_pos = tracks[:, target_frame, :]  # (N, 2)
        vis = visibility[:, source_frame] & visibility[:, target_frame]
        
        # Filter visible points
        src_pts = source_pos[vis].astype(np.float32)
        tgt_pts = target_pos[vis].astype(np.float32)
        
        if len(src_pts) < 4:
            logger.warning("Not enough visible points for flow interpolation")
            return np.zeros((h, w, 2), dtype=np.float32)
        
        # Compute displacement vectors
        displacements = tgt_pts - src_pts  # (N_vis, 2)
        
        # Create dense flow via interpolation
        # Use griddata for scattered interpolation
        from scipy.interpolate import griddata
        
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        
        flow_x = griddata(
            src_pts, displacements[:, 0], 
            (grid_x, grid_y), 
            method='linear',
            fill_value=0
        )
        flow_y = griddata(
            src_pts, displacements[:, 1],
            (grid_x, grid_y),
            method='linear', 
            fill_value=0
        )
        
        flow = np.stack([flow_x, flow_y], axis=-1).astype(np.float32)
        
        return flow
    
    def unload_model(self):
        """Free GPU resources."""
        if self.model is not None:
            del self.model
            self.model = None
            device_manager.empty_cache()
            logger.info("CoTracker model unloaded")


# Global instance
cotracker_motion = CoTrackerMotion(model_name="cotracker2")
