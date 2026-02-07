"""
Enhanced Video Colorization with SAM-2 Segmentation
Uses SAM-2 to segment people/clothing for region-specific color treatment.
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
from typing import List, Dict, Optional

sys.path.append(str(Path(__file__).parent.parent))

from backend.core.logging import get_logger
from backend.core.device import device_manager
from backend.conditioning.four_color_palette import FourColorPaletteGenerator
from backend.conditioning.style_conditioning import style_manager
from backend.diffusion.sd15_runner import sd15_runner
from backend.models.controlnet_helpers import extract_canny

logger = get_logger("enhanced_video_colorizer")


class EnhancedVideoColorizer:
    """
    Enhanced video colorization with SAM-2 segmentation.
    
    Features:
    - Keyframe-based colorization
    - Region-aware prompting (clothing, skin, background)
    - Color propagation between keyframes
    - Temporal consistency
    """
    
    def __init__(
        self,
        mood: str = "sunny_day",
        keyframe_interval: int = 15,
        num_inference_steps: int = 20,  # Lower for faster CPU
        use_segmentation: bool = True
    ):
        self.mood = mood
        self.keyframe_interval = keyframe_interval
        self.num_inference_steps = num_inference_steps
        self.use_segmentation = use_segmentation
        
        # Get style config
        self.palette = FourColorPaletteGenerator.get_palette(mood)
        self.style_config = style_manager.get_style_config(mood)
        
        if self.use_segmentation:
            try:
                from backend.models.sam2_segmenter import SAM2Segmenter
                self.segmenter = SAM2Segmenter(model_size="tiny")  # Use tiny for speed
                logger.info("  SAM-2 Segmenter initialized")
            except Exception as e:
                logger.warning(f"  Could not initialize SAM-2: {e}")
                self.use_segmentation = False
                
        logger.info(f"Enhanced Colorizer initialized: {mood}")
        logger.info(f"  Palette: {self.palette['name']}")
        logger.info(f"  Keyframe interval: {keyframe_interval}")
        logger.info(f"  Steps per frame: {num_inference_steps}")
    
    def _build_enhanced_prompt(self, base_prompt: str = "") -> tuple:
        """Build a more detailed prompt for clothing realism."""
        
        # Get color descriptions
        colors = self.palette['colors']
        color_names = self.palette['color_names']
        
        # Build clothing-aware prompt
        positive = (
            f"professional color restoration, 8k resolution, raw photo, "
            f"intricate fabric texture, high thread count, "
            f"realistic clothing drapes and folds, "
            f"lifelike skin pores and texture, "
            f"natural lighting, volumetric lighting, "
            f"color palette: {', '.join([f'{n} tones' for n in color_names])}, "
            f"sharp focus, highly detailed, "
            f"{base_prompt}"
        )
        
        negative = (
            "cartoon, anime, painting, drawing, illustration, "
            "blur, smooth, flat texture, plastic skin, "
            "oversaturated, color bleeding, seamless, "
            "bad clothing fit, floating clothes, "
            "low resolution, pixelated, artifacts, noise, "
            "extra limbs, distorted, unnatural"
        )
        
        return positive, negative
    
    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames")
        return frames
    
    def _colorize_frame(
        self,
        frame: np.ndarray,
        seed: int = 42
    ) -> np.ndarray:
        """Colorize a single frame using region-aware composition."""
        
        # 1. Resize for speed/consistency (max 512 height)
        h, w = frame.shape[:2]
        target_h = 512
        if h > target_h:
            scale = target_h / h
            new_w = int(w * scale)
            # Ensure divisible by 8 for SD
            new_w = (new_w // 8) * 8
            frame_resized = cv2.resize(frame, (new_w, target_h))
        else:
            frame_resized = frame.copy()
            
        frame_pil = Image.fromarray(frame_resized)
        edges = extract_canny(frame_pil)
        
        # 2. Generate Base/Background
        # Stronger prompting for background context
        bg_prompt = f"{self.mood}, interior design, photorealistic, 8k, detailed background, natural lighting"
        bg_negative = "people, person, clothing, skin, cartoon, anime, grayscale"
        
        bg_image = sd15_runner.generate_frame(
            image=edges,
            prompt=bg_prompt,
            negative_prompt=bg_negative,
            num_inference_steps=20,  # Fast but good enough for BG
            guidance_scale=7.5,
            controlnet_scale=0.6,    # Looser structure for BG
            seed=seed
        )
        
        # 3. Generate Person/Clothing (if segmentation is enabled)
        final_image = bg_image
        
        if self.use_segmentation and hasattr(self, 'segmenter'):
            try:
                # Get mask for people/clothing
                # Note: SAM-2 auto-mask returns list of masks. We need to filter/select.
                # For simplicity in this demo, we'll try to segment the largest central object
                masks = self.segmenter.segment_frame(frame_pil)
                
                # Combine masks that look like people (heuristic: large central masks)
                person_mask = np.zeros((frame_resized.shape[0], frame_resized.shape[1]), dtype=np.uint8)
                
                if masks:
                    # Simple heuristic: take largest mask
                    # Ideally we would prompt SAM-2 with point cues, but auto-mask is what we have wrapped
                    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
                    if sorted_masks:
                        # Take the top mask (likely the main subject)
                        m = sorted_masks[0]['segmentation']
                        person_mask = np.logical_or(person_mask, m).astype(np.uint8) * 255
                
                if np.sum(person_mask) > 0:
                    # Generate Person Pass
                    fg_prompt, fg_nav = self._build_enhanced_prompt("photorealistic person, detailed clothing, realistic skin")
                    
                    fg_image = sd15_runner.generate_frame(
                        image=edges,
                        prompt=fg_prompt,
                        negative_prompt=fg_nav,
                        num_inference_steps=30,  # Higher quality for person
                        guidance_scale=8.5,      # Stricter prompt adherence
                        controlnet_scale=0.8,    # Stronger structure
                        seed=seed
                    )
                    
                    # Composite
                    # Convert to numpy
                    bg_np = np.array(bg_image)
                    fg_np = np.array(fg_image)
                    
                    # Blur mask for smooth transition
                    mask_blur = cv2.GaussianBlur(person_mask, (15, 15), 0)
                    mask_norm = mask_blur.astype(float) / 255.0
                    mask_3d = np.stack([mask_norm]*3, axis=2)
                    
                    # Blend
                    comp_np = (fg_np * mask_3d + bg_np * (1 - mask_3d)).astype(np.uint8)
                    final_image = Image.fromarray(comp_np)
                    logger.info("  Applied segmentation-based composition")
                    
            except Exception as e:
                logger.error(f"Segmentation failed: {e}")
                # Fallback to BG only (which is a full generation anyway)
        
        # Resize back to original if needed? 
        # For now return resized to save bandwidth/processing in propagation
        return np.array(final_image)
    
    def _propagate_color(
        self,
        prev_colored: np.ndarray,
        curr_gray: np.ndarray,
        next_colored: np.ndarray,
        alpha: float
    ) -> np.ndarray:
        """
        Propagate color between keyframes using luminance transfer.
        Simple but effective color propagation.
        """
        # Get target dimensions from current frame
        if len(curr_gray.shape) == 3:
            h, w = curr_gray.shape[:2]
        else:
            h, w = curr_gray.shape
        
        # Resize keyframe colors to match current frame size
        prev_resized = cv2.resize(prev_colored, (w, h), interpolation=cv2.INTER_LINEAR)
        next_resized = cv2.resize(next_colored, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to LAB
        prev_lab = cv2.cvtColor(prev_resized, cv2.COLOR_RGB2LAB)
        next_lab = cv2.cvtColor(next_resized, cv2.COLOR_RGB2LAB)
        
        # Get current luminance
        if len(curr_gray.shape) == 3:
            curr_l = cv2.cvtColor(curr_gray, cv2.COLOR_RGB2GRAY)
        else:
            curr_l = curr_gray.copy()
        
        # Ensure luminance is correct size
        if curr_l.shape[:2] != (h, w):
            curr_l = cv2.resize(curr_l, (w, h))
        
        # Blend AB channels from keyframes
        _, prev_a, prev_b = cv2.split(prev_lab)
        _, next_a, next_b = cv2.split(next_lab)
        
        # Linear interpolation
        blended_a = cv2.addWeighted(prev_a, 1 - alpha, next_a, alpha, 0)
        blended_b = cv2.addWeighted(prev_b, 1 - alpha, next_b, alpha, 0)
        
        # Merge with current luminance
        result_lab = cv2.merge([curr_l, blended_a, blended_b])
        result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
        
        return result_rgb
    
    def colorize_video(
        self,
        input_path: str,
        output_path: str,
        max_frames: int = None
    ) -> str:
        """
        Colorize entire video.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            max_frames: Limit frames for testing (None = all)
            
        Returns:
            Path to output video
        """
        logger.info(f"Starting video colorization: {input_path}")
        
        # Extract frames
        frames = self._extract_frames(input_path)
        if max_frames:
            frames = frames[:max_frames]
            logger.info(f"Limited to {max_frames} frames for testing")
        
        n_frames = len(frames)
        
        # Determine keyframes
        keyframe_indices = list(range(0, n_frames, self.keyframe_interval))
        if keyframe_indices[-1] != n_frames - 1:
            keyframe_indices.append(n_frames - 1)
        
        logger.info(f"Keyframes: {keyframe_indices}")
        
        # Load SD pipeline
        logger.info("Loading colorization pipeline...")
        sd15_runner.load_pipeline()
        
        try:
            # Colorize keyframes
            logger.info("Colorizing keyframes...")
            keyframe_colors = {}
            
            for i, idx in enumerate(tqdm(keyframe_indices, desc="Colorizing")):
                colored = self._colorize_frame(frames[idx], seed=42 + i)
                keyframe_colors[idx] = colored
            
            # Propagate to all frames
            logger.info("Propagating colors to all frames...")
            all_colored = []
            
            for i in tqdm(range(n_frames), desc="Propagating"):
                if i in keyframe_colors:
                    # Use keyframe directly
                    all_colored.append(keyframe_colors[i])
                else:
                    # Find surrounding keyframes
                    prev_key = max([k for k in keyframe_indices if k < i])
                    next_key = min([k for k in keyframe_indices if k > i])
                    
                    # Calculate blend factor
                    alpha = (i - prev_key) / (next_key - prev_key)
                    
                    # Propagate color
                    colored = self._propagate_color(
                        keyframe_colors[prev_key],
                        frames[i],
                        keyframe_colors[next_key],
                        alpha
                    )
                    all_colored.append(colored)
            
            # Write output video
            logger.info(f"Saving video to {output_path}...")
            h, w = frames[0].shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))
            
            for frame in all_colored:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            
            writer.release()
            logger.info(f"✓ Video saved: {output_path}")
            
        finally:
            sd15_runner.unload_pipeline()
            device_manager.empty_cache()
        
        return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="uploads/bedroom.mp4", help="Input video")
    parser.add_argument("--output", default="results/enhanced_colorized.mp4", help="Output video")
    parser.add_argument("--mood", default="sunny_day", help="Mood preset")
    parser.add_argument("--keyframe-interval", type=int, default=15, help="Keyframe interval")
    parser.add_argument("--steps", type=int, default=15, help="Inference steps (lower=faster)")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for testing")
    args = parser.parse_args()
    
    # Show available moods
    print("\n" + "="*60)
    print("Available Moods:")
    for mood_id in FourColorPaletteGenerator.PALETTES.keys():
        palette = FourColorPaletteGenerator.get_palette(mood_id)
        print(f"  - {mood_id}: {palette['name']}")
    print("="*60 + "\n")
    
    # Create colorizer
    colorizer = EnhancedVideoColorizer(
        mood=args.mood,
        keyframe_interval=args.keyframe_interval,
        num_inference_steps=args.steps
    )
    
    # Run
    colorizer.colorize_video(
        input_path=args.input,
        output_path=args.output,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
