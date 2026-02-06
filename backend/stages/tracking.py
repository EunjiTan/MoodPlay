"""
Stage 3: Tracking / Motion
Computes point tracks across frames using CoTracker.
Replaces optical flow with better temporal consistency.
"""

import numpy as np
from tqdm import tqdm
from PIL import Image
from backend.core.stage_runner import StageRunner
from backend.models.cotracker_motion import cotracker_motion


class TrackingStage(StageRunner):
    """
    Uses CoTracker for point tracking across video frames.
    Replaces Farneback optical flow with better accuracy and temporal consistency.
    Saves point tracks and visibility data to disk.
    """
    
    def run(self, grid_size: int = 20, batch_size: int = 50):
        """
        Run CoTracker motion tracking.
        
        Args:
            grid_size: Grid density (grid_size x grid_size points)
            batch_size: Number of frames to process at once (for memory)
        """
        self.logger.info("Starting Tracking/Motion Stage (CoTracker)")
        
        frame_count = self.dm.get_frame_count("proc")
        self.logger.info(f"Tracking {frame_count} frames with {grid_size}x{grid_size} grid...")
        
        # Check if tracking data already exists
        tracks_path = self.dm.dirs["flow"] / "cotracker_data.npz"
        if tracks_path.exists():
            self.logger.info("CoTracker data already exists, skipping...")
            return
        
        try:
            # Load CoTracker model
            cotracker_motion.load_model()
            
            # Load all frames (or batch if too many)
            if frame_count <= batch_size:
                # Process all at once
                frames = self._load_frames_batch(0, frame_count)
                motion_data = cotracker_motion.track_video(
                    frames, 
                    grid_size=grid_size,
                    backward_tracking=True
                )
                
                # Save tracking data
                self._save_motion_data(motion_data, tracks_path)
                
                # Also generate flow fields for backward compatibility
                self._generate_flow_fields(motion_data, frame_count)
                
            else:
                # Process in overlapping batches
                self._process_batched(frame_count, grid_size, batch_size)
            
            self.logger.info("CoTracker tracking complete.")
            
        finally:
            cotracker_motion.unload_model()
            self.release_resources()
    
    def _load_frames_batch(self, start_idx: int, end_idx: int) -> list:
        """Load a batch of frames as numpy arrays."""
        frames = []
        for idx in range(start_idx, end_idx):
            frame_pil = self.dm.load_frame(idx, stage="proc")
            frame_np = np.array(frame_pil)
            frames.append(frame_np)
        return frames
    
    def _save_motion_data(self, motion_data: dict, path):
        """Save CoTracker motion data to disk."""
        np.savez_compressed(
            path,
            tracks=motion_data['tracks'],
            visibility=motion_data['visibility'],
            grid_size=motion_data['grid_size']
        )
        self.logger.info(f"Saved CoTracker data to {path}")
    
    def _generate_flow_fields(self, motion_data: dict, frame_count: int):
        """
        Generate dense flow fields from CoTracker data.
        For backward compatibility with propagation stage.
        """
        self.logger.info("Generating flow fields from CoTracker data...")
        
        # Get frame dimensions from first frame
        first_frame = self.dm.load_frame(0, stage="proc")
        h, w = first_frame.size[1], first_frame.size[0]
        
        for idx in tqdm(range(frame_count - 1), desc="Generating Flow"):
            # Skip if already exists
            save_path = self.dm.dirs["flow"] / f"{idx:06d}.npz"
            if save_path.exists():
                continue
            
            # Interpolate dense flow from sparse tracks
            flow_fwd = cotracker_motion.interpolate_dense_flow(
                motion_data, 
                source_frame=idx,
                target_frame=idx + 1,
                frame_shape=(h, w)
            )
            
            flow_bwd = cotracker_motion.interpolate_dense_flow(
                motion_data,
                source_frame=idx + 1,
                target_frame=idx,
                frame_shape=(h, w)
            )
            
            # Save in same format as old optical flow for compatibility
            np.savez_compressed(save_path, fwd=flow_fwd, bwd=flow_bwd)
    
    def _process_batched(self, frame_count: int, grid_size: int, batch_size: int):
        """Process video in overlapping batches for large videos."""
        self.logger.info(f"Processing in batches of {batch_size} frames...")
        
        # For now, just process first batch as primary
        # Full implementation would stitch batches together
        frames = self._load_frames_batch(0, min(batch_size, frame_count))
        motion_data = cotracker_motion.track_video(
            frames,
            grid_size=grid_size,
            backward_tracking=True
        )
        
        tracks_path = self.dm.dirs["flow"] / "cotracker_data.npz"
        self._save_motion_data(motion_data, tracks_path)
        
        # Generate flow for available frames
        self._generate_flow_fields(motion_data, len(frames))
        
        self.logger.warning(
            f"Batched processing: Only processed first {len(frames)} frames. "
            f"Full video has {frame_count} frames."
        )
