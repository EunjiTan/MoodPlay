"""
Stage 2: Segmentation
Generates instance masks for frames using SAM-2.
Updated to use SAM-2 for better quality segmentation.
"""

from tqdm import tqdm
import numpy as np
from backend.core.stage_runner import StageRunner
from backend.models.sam2_segmenter import sam2_segmenter


class SegmentationStage(StageRunner):
    """
    Loads SAM-2, segments all processed frames, saves masks, unloads SAM-2.
    Uses SAM-2 (Segment Anything Model 2) for high-quality instance segmentation.
    """
    
    def run(self):
        self.logger.info("Starting Segmentation Stage (SAM-2)")
        
        # Load Model
        sam2_segmenter.load_model()
        
        frame_count = self.dm.get_frame_count("proc")
        self.logger.info(f"Processing {frame_count} frames with SAM-2...")
        
        try:
            for idx in tqdm(range(frame_count), desc="Segmenting (SAM-2)"):
                # Skip if already exists (resume capability)
                if (self.dm.dirs["masks"] / f"{idx:06d}.npz").exists():
                    continue

                # Load Frame
                frame_pil = self.dm.load_frame(idx, stage="proc")
                
                # Run SAM-2 (Automatic Mask Generator)
                masks_data = sam2_segmenter.segment_frame(frame_pil)
                
                # Compress masks into a single ID map for storage efficiency
                w, h = frame_pil.size
                id_map = np.zeros((h, w), dtype=np.uint16)
                
                # Sort masks by area (large first, small on top)
                masks_data.sort(key=lambda x: x['area'], reverse=True)
                
                for i, m_data in enumerate(masks_data, 1):
                    # m_data['segmentation'] is boolean mask
                    id_map[m_data['segmentation']] = i
                    
                # Save to disk
                self.dm.save_mask(id_map, idx)
                
        finally:
            # Unload Model (CRITICAL)
            sam2_segmenter.unload_model()
            self.release_resources()
