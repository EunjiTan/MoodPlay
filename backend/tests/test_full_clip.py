
import cv2
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.tests.enhanced_video_colorizer import EnhancedVideoColorizer
from backend.diffusion.sd15_runner import sd15_runner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    video_path = "uploads/bedroom.mp4"
    output_path = "results/enhanced_segmented_5s.mp4"
    
    # Initialize colorizer
    # Keyframe interval 10 = ~15 keyframes for 5s video = ~60 mins processing
    # Steps 20-30 are set inside EnhancedVideoColorizer
    colorizer = EnhancedVideoColorizer(
        use_segmentation=True,
        num_inference_steps=30 
    )
    
    try:
        sd15_runner.load_pipeline()
        
        logger.info(f"Processing full clip {video_path}...")
        
        # Process
        # We process the whole video (or limit to 150 frames if it's long)
        colorizer.colorize_video(
            input_path=video_path,
            output_path=output_path
        )
        
        logger.info(f"Done! Saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sd15_runner.unload_pipeline()

if __name__ == "__main__":
    main()
