"""
Video Reader Module
Provides streaming access to video frames to minimize memory usage.
"""

import cv2
import os
import logging
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict
from PIL import Image

from backend.core.logging import get_logger, time_execution
from backend.core.exceptions import VideoReadError, ResourceError

logger = get_logger("io.video_reader")

class VideoReader:
    """
    Streaming video reader that yields frames one by one.
    Ensures low memory footprint by not loading full video.
    """
    
    def __init__(self, video_path: str, max_dimension: int = 1024):
        self.path = Path(video_path)
        if not self.path.exists():
            raise ResourceError(f"Video file not found: {video_path}")
            
        self.cap = None
        self.max_dimension = max_dimension
        self.meta = self._extract_metadata()
        
    def _extract_metadata(self) -> Dict:
        """Extract video metadata without fully reading it."""
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise VideoReadError(f"Failed to open video: {self.path}")
            
        meta = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": 0.0
        }
        
        if meta["fps"] > 0:
            meta["duration"] = meta["frame_count"] / meta["fps"]
            
        cap.release()
        logger.info(f"Loaded video metadata: {meta}")
        return meta

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame if it exceeds max dimension while preserving aspect ratio."""
        h, w = frame.shape[:2]
        if max(h, w) <= self.max_dimension:
            return frame
            
        scale = self.max_dimension / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def __iter__(self) -> Generator[Tuple[int, Image.Image], None, None]:
        """
        Yields (frame_idx, frame_pil) tuples.
        Context management handled by generator cleanup or explicit close.
        """
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise VideoReadError(f"Failed to open video stream: {self.path}")
            
        frame_idx = 0
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Convert BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize if needed
                frame_resized = self._resize_frame(frame_rgb)
                
                # Convert to PIL
                frame_pil = Image.fromarray(frame_resized)
                
                yield frame_idx, frame_pil
                
                frame_idx += 1
                
        except Exception as e:
            logger.error(f"Error reading frame {frame_idx}: {e}")
            raise VideoReadError(f"Stream failure at frame {frame_idx}") from e
        finally:
            self.close()

    def close(self):
        """Release resources."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.debug("Video reader released resources")
        self.cap = None

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        vpath = sys.argv[1]
        with VideoReader(vpath) as reader:
            print(f"Metadata: {reader.meta}")
            for idx, frame in reader:
                if idx % 30 == 0:
                    print(f"Read frame {idx} size={frame.size}")
