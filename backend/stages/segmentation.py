"""
Stage 2: Segmentation
Generates instance masks for frames using SAM.
"""

from tqdm import tqdm
import numpy as np
from backend.core.stage_runner import StageRunner
from backend.models.segmentation import segmenter

class SegmentationStage(StageRunner):
    """
    Loads SAM, segments all processed frames, saves masks, unloads SAM.
    """
    
    def run(self):
        self.logger.info("Starting Segmentation Stage (SAM)")
        
        # 1. Load Model
        # Segmenter internals handle lazy loading, but we force it here
        segmenter.load_model()
        
        frame_count = self.dm.get_frame_count("proc")
        self.logger.info(f"Processing {frame_count} frames...")
        
        try:
            for idx in tqdm(range(frame_count), desc="Segmenting"):
                # Skip if already exists (resume capability)
                if (self.dm.dirs["masks"] / f"{idx:06d}.npz").exists():
                    continue

                # Load Frame
                frame_pil = self.dm.load_frame(idx, stage="proc")
                
                # Run SAM (Automatic Mask Generator)
                # Returns list of dicts
                masks_data = segmenter.segment_frame(frame_pil)
                
                # Compress masks into a single ID map for storage efficiency
                # Create blank canvas
                w, h = frame_pil.size
                id_map = np.zeros((h, w), dtype=np.uint16)
                
                # Sort masks by area (large first, small on top)
                masks_data.sort(key=lambda x: x['area'], reverse=True)
                
                for i, m_data in enumerate(masks_data, 1):
                    # m_data['segmentation'] is boolean mask
                    # We assign ID i to these pixels
                    id_map[m_data['segmentation']] = i
                    
                # Save to disk
                self.dm.save_mask(id_map, idx)
                
        finally:
            # 2. Unload Model (CRITICAL)
            segmenter.unload_model()
            self.release_resources()
