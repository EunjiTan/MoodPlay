import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.pipelines.updated_colorization_pipeline import UpdatedColorizationPipeline
from backend.io.video_reader import VideoReader

def save_video(frames, output_path, fps=30):
    if not frames:
        print("No frames to save")
        return
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Saved video to {output_path}")

def main():
    video_path = "uploads/bedroom.mp4"
    output_path = "results/reproduction_output.mp4"
    
    # Ensure results dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # limits
    max_frames = 60 # 2 seconds at 30fps
    
    print(f"Reading video from {video_path}...")
    frames = []
    try:
        with VideoReader(video_path, max_dimension=320) as reader:
            for idx, frame_pil in reader:
                frames.append(np.array(frame_pil)) # PIPELINE EXPECTS NUMPY ARRAYS
                if len(frames) >= max_frames:
                    break
    except Exception as e:
        print(f"Error reading video: {e}")
        return

    print(f"Read {len(frames)} frames.")
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = UpdatedColorizationPipeline(
        sam2_model_size="tiny",     # Fast segmentation
        cotracker_model="cotracker3"
    )
    
    # Run processing
    print("Running processing loop...")
    # mood="sunny_day" is default
    result_frames = pipeline.process_video(
        frames=frames,
        mood="sunny_day",
        keyframe_interval=25, # 2-3 keyframes for 60 frames
        denoise_strength=0.3,
        track_motion=True,
        segment_frames=False,
        num_inference_steps=15
    )
    
    # Save output
    print("Saving output...")
    save_video(result_frames, output_path)
    print("Done!")

if __name__ == "__main__":
    main()
