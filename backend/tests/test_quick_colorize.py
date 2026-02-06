"""
Quick Pipeline Test
Test the updated colorization pipeline with a sample video.
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

sys.path.append(str(Path(__file__).parent.parent))

from backend.core.logging import get_logger
from backend.conditioning.four_color_palette import FourColorPaletteGenerator
from backend.conditioning.style_conditioning import style_manager
from backend.diffusion.sd15_runner import sd15_runner
from backend.models.controlnet_helpers import extract_canny

logger = get_logger("test_pipeline")


def test_colorize_single_frame(
    input_path: str,
    output_path: str,
    mood: str = "sunny_day"
):
    """
    Test colorizing a single frame from the video.
    
    Args:
        input_path: Path to input video
        output_path: Path to save colorized frame
        mood: Mood preset to use
    """
    logger.info(f"Testing Pipeline with mood: {mood}")
    
    # Load first frame from video
    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise Exception(f"Could not read video: {input_path}")
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    
    logger.info(f"Loaded frame: {frame_pil.size}")
    
    # Get palette info
    palette = FourColorPaletteGenerator.get_palette(mood)
    logger.info(f"Using palette: {palette['name']}")
    for i, (color, name) in enumerate(zip(palette['colors'], palette['color_names'])):
        logger.info(f"  Color {i+1} ({name}): RGB{color}")
    
    # Build prompt
    prompt, negative = style_manager.build_prompt("", mood)
    config = style_manager.get_style_config(mood)
    
    logger.info(f"Prompt: {prompt[:100]}...")
    logger.info(f"LoRA Scale: {config['lora_scale']}")
    
    # Extract Canny edges as ControlNet input
    edges = extract_canny(frame_pil)
    logger.info("Extracted Canny edges")
    
    # Load pipeline
    logger.info("Loading SD1.5 pipeline...")
    sd15_runner.load_pipeline()
    
    # Colorize
    logger.info("Colorizing frame...")
    colorized = sd15_runner.generate_frame(
        image=edges,
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=30,  # Faster for testing
        guidance_scale=5.5,
        controlnet_scale=0.95,
        lora_scale=config['lora_scale'],
        denoise_strength=0.3,
        seed=42
    )
    
    # Save result
    colorized.save(output_path)
    logger.info(f"✓ Saved colorized frame to: {output_path}")
    
    # Also save a comparison image
    comparison_path = output_path.replace('.png', '_comparison.png')
    
    # Create side-by-side comparison
    original = frame_pil.resize(colorized.size)
    comparison = Image.new('RGB', (colorized.width * 2, colorized.height))
    comparison.paste(original, (0, 0))
    comparison.paste(colorized, (colorized.width, 0))
    comparison.save(comparison_path)
    logger.info(f"✓ Saved comparison to: {comparison_path}")
    
    # Cleanup
    sd15_runner.unload_pipeline()
    logger.info("Pipeline unloaded")
    
    return colorized


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="uploads/bedroom.mp4", help="Input video")
    parser.add_argument("--output", default="results/test_colorized.png", help="Output image")
    parser.add_argument("--mood", default="sunny_day", help="Mood preset")
    args = parser.parse_args()
    
    # Show available moods
    print("\n" + "="*60)
    print("Available Moods:")
    for mood_id in FourColorPaletteGenerator.PALETTES.keys():
        palette = FourColorPaletteGenerator.get_palette(mood_id)
        print(f"  - {mood_id}: {palette['name']}")
    print("="*60 + "\n")
    
    # Run test
    test_colorize_single_frame(
        input_path=args.input,
        output_path=args.output,
        mood=args.mood
    )
