import cv2
import os
import numpy as np
from typing import Iterator

class VideoProcessor:
    @staticmethod
    def get_video_info(video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        cap.release()
        return info

    @staticmethod
    def frame_generator(video_path: str) -> Iterator[np.ndarray]:
        """Yields frames from video as RGB numpy arrays."""
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()

    @staticmethod
    def save_video(frames: Iterator[np.ndarray], output_path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert RGB back to BGR
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
            
        out.release()
        return output_path
