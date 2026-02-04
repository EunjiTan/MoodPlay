"""
CLI Runner for Staged Pipeline
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.logging import get_logger
from backend.pipeline.orchestrator_v2 import staged_pipeline

logger = get_logger("cli.staged")

def main():
    parser = argparse.ArgumentParser(description="Staged Disk-Based Video Colorization")
    
    parser.add_argument("--input", "-i", type=str, required=True, help="Input video path")
    parser.add_argument("--output", "-o", type=str, default="output_staged.mp4", help="Output video name")
    parser.add_argument("--style", "-s", type=str, default="cinematic", help="Style preset")
    parser.add_argument("--interval", "-k", type=int, default=5, help="Keyframe interval")
    parser.add_argument("--job", "-j", type=str, default="job_001", help="Job ID (folder name)")
    parser.add_argument("--clean", action="store_true", help="Clean workspace before starting")
    
    args = parser.parse_args()
    
    try:
        staged_pipeline.run(
            input_video=args.input,
            output_name=args.output,
            style_name=args.style,
            keyframe_interval=args.interval,
            job_id=args.job,
            clean_start=args.clean
        )
    except Exception as e:
        logger.error(f"Staged execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
