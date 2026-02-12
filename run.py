"""
MoodPlay — Instance-Guided Semantic Video Colorization
System entry point for running the pipeline on a video file.

Usage:
    python run.py <video_path> [--mood MOOD] [--seed SEED] [--steps STEPS]

Example:
    python run.py uploads/bedroom.mp4 --mood golden_hour --seed 42
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def extract_frames(video_path: str, max_frames: int = 60) -> list:
    """Extract RGB frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}")
    print(f"  FPS: {fps:.1f} | Total frames: {total}")

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    print(f"  Extracted: {len(frames)} frames")
    return frames, fps


def save_video(frames: list, output_path: str, fps: float):
    """Save RGB frames to an MP4 video."""
    if not frames:
        print("No frames to save.")
        return

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    print(f"\nOutput saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MoodPlay Instance-Guided Video Colorization"
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--mood", default="sunny_day",
                        help="Palette mood (default: sunny_day)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--steps", type=int, default=20,
                        help="Diffusion inference steps (default: 20)")
    parser.add_argument("--max-frames", type=int, default=60,
                        help="Max frames to process (default: 60)")
    parser.add_argument("--output", default=None,
                        help="Output video path (default: results/<name>_colorized.mp4)")
    parser.add_argument("--keyframe-interval", type=int, default=5,
                        help="Full processing every N frames (default: 5)")
    parser.add_argument("--no-tracking", action="store_true",
                        help="Disable CoTracker motion tracking")
    parser.add_argument("--test", action="store_true",
                        help="Run test suite instead of processing video")

    args = parser.parse_args()

    # Test mode
    if args.test:
        print("Running test suite...\n")
        from backend.tests.test_instance_pipeline import run_all_tests
        success = run_all_tests()
        sys.exit(0 if success else 1)

    # Validate input
    if not Path(args.video).exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    # Extract frames
    frames, fps = extract_frames(args.video, args.max_frames)
    if not frames:
        print("Error: No frames extracted.")
        sys.exit(1)

    # Import pipeline
    from backend.pipelines.instance_guided_pipeline import (
        InstanceGuidedPipeline, PipelineRunConfig,
    )

    # Configure
    config = PipelineRunConfig(
        mood=args.mood,
        seed=args.seed,
        num_inference_steps=args.steps,
        keyframe_interval=args.keyframe_interval,
        track_motion=not args.no_tracking,
    )

    # Run pipeline
    pipeline = InstanceGuidedPipeline()
    result = pipeline.process_video(frames, config)

    # Save output
    colorized = result["colorized_frames"]
    output_path = args.output
    if not output_path:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        stem = Path(args.video).stem
        output_path = str(results_dir / f"{stem}_colorized.mp4")

    save_video(colorized, output_path, fps)

    # Print quality summary
    report = result["quality_report"]
    print(f"\nQuality Summary:")
    print(f"  Avg ICA (dE): {report.avg_ica:.2f}")
    print(f"  Avg TCV:      {report.avg_tcv:.2f}")
    print(f"  Max BLS:      {report.max_bls:.4f}")
    print(f"  GPC pass:     {report.gpc_pass}")
    print(f"  Instances:    {len(result['registry'])}")


if __name__ == "__main__":
    main()
