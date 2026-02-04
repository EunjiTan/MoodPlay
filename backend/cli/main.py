"""
CLI Entrypoint
Main command-line interface for the video colorization system.
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.logging import get_logger
from backend.pipeline.orchestrator import pipeline

logger = get_logger("cli")

def main():
    parser = argparse.ArgumentParser(description="SD1.5 + ControlNext + LoRA Video Colorization")
    
    parser.add_argument("--input", "-i", type=str, required=True, help="Input video path")
    parser.add_argument("--output", "-o", type=str, default="output.mp4", help="Output video name")
    parser.add_argument("--style", "-s", type=str, default="cinematic", help="Style preset (cinematic, winter, sunny_day)")
    parser.add_argument("--prompt", "-p", type=str, default="", help="Additional prompt instructions")
    parser.add_argument("--segmentation", action="store_true", help="Enable semantic segmentation (experimental, slow)")
    parser.add_argument("--lora", "-l", type=str, help="Path to LoRA file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
        
    try:
        pipeline.run(
            input_video=args.input,
            output_name=args.output,
            style_name=args.style,
            base_prompt=args.prompt,
            use_segmentation=args.segmentation,
            lora_path=args.lora
        )
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
