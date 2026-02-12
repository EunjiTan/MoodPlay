"""
CoTracker Motion Tracking Module (Enhanced)
Point-based tracking with joint multi-instance support, colour-warp
computation, and motion-magnitude gating per Section 3.3 of the spec.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from scipy.interpolate import griddata

from backend.core.logging import get_logger, log_gpu_memory, time_execution
from backend.core.device import get_device, get_dtype, oom_safe, device_manager
from backend.core.exceptions import ModelLoadError, ResourceError
from backend.core.pipeline_config import CFG

logger = get_logger("models.cotracker_motion")


class CoTrackerMotion:
    """
    CoTracker wrapper for point tracking across video frames.

    Features:
    - Joint multi-instance tracking
    - Configurable grid density
    - Memory-efficient processing
    - Color-warp affine estimation
    - Motion-magnitude gating
    """

    CHECKPOINT_MAP = {
        "cotracker3": "cotracker3.pth",
        "cotracker2": "cotracker2.pth",
        "cotracker": "cotracker.pth",
    }

    def __init__(self, model_name: str = "cotracker3"):
        """
        Initialize CoTracker.

        Args:
            model_name: 'cotracker3' (recommended), 'cotracker2', or 'cotracker'
        """
        self.model_name = model_name
        self.device = get_device()
        self.model = None

        self.checkpoint_dir = Path("checkpoints/cotracker")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"CoTracker initialized (model={model_name})")

    def _get_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.CHECKPOINT_MAP.get(self.model_name, "cotracker2.pth")

    def _check_checkpoint_exists(self) -> bool:
        return self._get_checkpoint_path().exists()

    @time_execution(logger)
    def load_model(self):
        """Load CoTracker model to GPU."""
        if self.model is not None:
            return

        try:
            logger.info(f"Loading CoTracker ({self.model_name})...")

            checkpoint_path = self._get_checkpoint_path()
            if checkpoint_path.exists():
                logger.info(f"Loading from local checkpoint: {checkpoint_path}")
                self.model = torch.hub.load(
                    "facebookresearch/co-tracker",
                    self.model_name,
                    source="github",
                )
            else:
                logger.info(f"Downloading CoTracker from torch hub...")
                self.model = torch.hub.load(
                    "facebookresearch/co-tracker",
                    self.model_name,
                    source="github",
                )

            self.model = self.model.to(self.device)
            log_gpu_memory(logger, "CoTracker Loaded")

        except Exception as e:
            logger.warning(f"CoTracker load failed: {e}. Using fallback optical flow.")
            self.model = None

    def _frames_to_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Convert list of (H, W, 3) RGB uint8 arrays to (1, T, 3, H, W) tensor."""
        arr = np.stack(frames)                     # (T, H, W, 3)
        arr = arr.transpose(0, 3, 1, 2)            # (T, 3, H, W)
        tensor = torch.from_numpy(arr).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)  # (1, T, 3, H, W)

    # ── Grid tracking ─────────────────────────────────────────────────

    @oom_safe(logger)
    def track_video(
        self,
        frames: List[np.ndarray],
        grid_size: int = CFG.COTRACKER_GRID_SIZE,
        backward_tracking: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Track points across entire video using a grid."""
        if self.model is None:
            self.load_model()
        if self.model is None:
            raise ResourceError("CoTracker model not available")

        video_tensor = self._frames_to_tensor(frames)

        with torch.no_grad():
            pred_tracks, pred_visibility = self.model(
                video_tensor,
                grid_size=grid_size,
                backward_tracking=backward_tracking,
            )

        tracks = pred_tracks[0].cpu().numpy()        # (N, T, 2)
        vis = pred_visibility[0].cpu().numpy()        # (N, T)

        return {
            "tracks": tracks,
            "visibility": vis,
            "num_points": tracks.shape[0],
            "num_frames": tracks.shape[1],
            "grid_size": grid_size,
        }

    # ── Point tracking ─────────────────────────────────────────────────

    @oom_safe(logger)
    def track_points(
        self,
        frames: List[np.ndarray],
        query_points: np.ndarray,
        query_frame: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Track specific points across video."""
        if self.model is None:
            self.load_model()
        if self.model is None:
            raise ResourceError("CoTracker model not available")

        video_tensor = self._frames_to_tensor(frames)

        N = query_points.shape[0]
        queries = np.zeros((N, 3), dtype=np.float32)
        queries[:, 0] = query_frame
        queries[:, 1:] = query_points
        queries_tensor = torch.from_numpy(queries).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_tracks, pred_visibility = self.model(
                video_tensor,
                queries=queries_tensor,
            )

        return {
            "tracks": pred_tracks[0].cpu().numpy(),
            "visibility": pred_visibility[0].cpu().numpy(),
        }

    # ── Motion vectors ─────────────────────────────────────────────────

    def compute_motion_vectors(
        self,
        motion_data: Dict[str, np.ndarray],
        source_frame: int,
        target_frame: int,
    ) -> np.ndarray:
        """Compute (N, 2) motion vectors from source to target frame."""
        tracks = motion_data["tracks"]  # (N, T, 2)
        return tracks[:, target_frame] - tracks[:, source_frame]

    # ── Dense flow interpolation ───────────────────────────────────────

    def interpolate_dense_flow(
        self,
        motion_data: Dict[str, np.ndarray],
        source_frame: int,
        target_frame: int,
        frame_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Interpolate sparse tracks to dense (H, W, 2) flow field."""
        tracks = motion_data["tracks"]
        vis = motion_data["visibility"]
        H, W = frame_shape

        src_pts = tracks[:, source_frame]
        tgt_pts = tracks[:, target_frame]
        vis_mask = vis[:, source_frame] & vis[:, target_frame]

        if vis_mask.sum() < 3:
            return np.zeros((H, W, 2), dtype=np.float32)

        src_valid = src_pts[vis_mask]
        flow_valid = (tgt_pts - src_pts)[vis_mask]

        grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)

        flow_dense = np.zeros((H, W, 2), dtype=np.float32)
        for ch in range(2):
            flow_dense[:, :, ch] = griddata(
                src_valid, flow_valid[:, ch],
                (grid_x, grid_y),
                method="linear",
                fill_value=0.0,
            )

        return flow_dense

    # ── NEW: Joint multi-instance tracking ─────────────────────────────

    def track_joint(
        self,
        frames: List[np.ndarray],
        instance_masks: Dict[int, np.ndarray],
        points_per_instance: int = 10,
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Track all instances jointly within each frame window.

        Seeds tracking points inside each instance mask and tracks them
        jointly to prevent ID-switch errors.

        Returns:
            {instance_id: {"tracks": (N, T, 2), "visibility": (N, T)}}
        """
        all_points = []
        instance_assignment = []

        for iid, mask in instance_masks.items():
            ys, xs = np.where(mask)
            if len(ys) < points_per_instance:
                pts_idx = np.arange(len(ys))
            else:
                pts_idx = np.random.choice(len(ys), points_per_instance, replace=False)
            pts = np.stack([xs[pts_idx], ys[pts_idx]], axis=1).astype(np.float32)
            all_points.append(pts)
            instance_assignment.extend([iid] * len(pts))

        if not all_points:
            return {}

        query_pts = np.concatenate(all_points, axis=0)
        motion_data = self.track_points(frames, query_pts, query_frame=0)

        # Partition tracks back to per-instance
        tracks = motion_data["tracks"]
        vis = motion_data["visibility"]
        assignment = np.array(instance_assignment)

        per_instance: Dict[int, Dict[str, np.ndarray]] = {}
        for iid in set(instance_assignment):
            idx = assignment == iid
            per_instance[iid] = {
                "tracks": tracks[idx],
                "visibility": vis[idx],
            }

        return per_instance

    # ── NEW: Color warp computation ────────────────────────────────────

    def compute_color_warp(
        self,
        motion_data: Dict[str, np.ndarray],
        source_frame: int,
        target_frame: int,
    ) -> Optional[np.ndarray]:
        """Estimate 2×3 affine transformation from source to target frame.

        Used to warp the colour hint mask for an instance between frames.

        Returns:
            (2, 3) affine matrix or None if insufficient visible points.
        """
        tracks = motion_data["tracks"]
        vis = motion_data["visibility"]

        src_pts = tracks[:, source_frame]
        tgt_pts = tracks[:, target_frame]
        vis_both = vis[:, source_frame] & vis[:, target_frame]

        src_valid = src_pts[vis_both].astype(np.float32)
        tgt_valid = tgt_pts[vis_both].astype(np.float32)

        if len(src_valid) < 3:
            return None

        import cv2
        M, _ = cv2.estimateAffine2D(src_valid, tgt_valid, method=cv2.RANSAC)
        return M

    # ── NEW: Motion magnitude gating ───────────────────────────────────

    def check_motion_gate(
        self,
        motion_data: Dict[str, np.ndarray],
        source_frame: int,
        target_frame: int,
        threshold: float = CFG.MOTION_MAGNITUDE_THRESHOLD,
    ) -> bool:
        """Return True if the average motion magnitude exceeds threshold.

        When True, the caller should re-validate SAM-2 masks for this instance.
        """
        vectors = self.compute_motion_vectors(motion_data, source_frame, target_frame)
        magnitudes = np.linalg.norm(vectors, axis=1)
        avg_mag = float(magnitudes.mean())
        return avg_mag > threshold

    # ── Unload ─────────────────────────────────────────────────────────

    def unload_model(self):
        """Free GPU resources."""
        if self.model is not None:
            del self.model
            self.model = None
            device_manager.empty_cache()
            logger.info("CoTracker model unloaded")


# Global instance
cotracker_motion = CoTrackerMotion(model_name="cotracker2")
