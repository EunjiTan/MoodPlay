"""
Updated Video Colorization Pipeline
Unified pipeline combining SAM-2, CoTracker, 4-Color LoRA, and denoising.

Architecture:
    Stage 1: Frame Extraction
    Stage 2: SAM-2 Segmentation
    Stage 3: CoTracker Motion Tracking
    Stage 4: 4-Color LoRA + SD1.5 + Denoising
    Stage 5: Color Propagation
"""

import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from backend.core.logging import get_logger
from backend.core.device import device_manager
from backend.models.sam2_segmenter import SAM2Segmenter, sam2_segmenter
from backend.models.cotracker_motion import CoTrackerMotion, cotracker_motion
from backend.diffusion.sd15_runner import SD15Runner, sd15_runner
from backend.conditioning.style_conditioning import style_manager
from backend.conditioning.four_color_palette import FourColorPaletteGenerator
from backend.models.controlnet_helpers import extract_canny

logger = get_logger("pipelines.updated_colorization")


class UpdatedColorizationPipeline:
    """
    Complete video colorization pipeline with:
    - SAM-2 for high-quality segmentation
    - CoTracker for accurate motion tracking
    - 4-Color palette system for mood-based colorization
    - DPMSolver denoising for photorealistic output
    
    Usage:
        pipeline = UpdatedColorizationPipeline()
        colorized = pipeline.process_video(
            frames=grayscale_frames,
            mood="sunny_day",
            keyframe_interval=10,
            denoise_strength=0.3
        )
    """
    
    def __init__(
        self,
        sam2_model_size: str = "tiny",
        cotracker_model: str = "cotracker2",
        sd_model_id: str = "runwayml/stable-diffusion-v1-5",
        lora_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initialize the pipeline.
        
        Args:
            sam2_model_size: SAM-2 size ('tiny', 'small', 'base_plus', 'large')
            cotracker_model: CoTracker version ('cotracker2', 'cotracker')
            sd_model_id: Stable Diffusion model ID
            lora_path: Optional path to 4-color LoRA weights
            device: Device to use ('cuda' or 'cpu')
        """
        self.sam2_model_size = sam2_model_size
        self.cotracker_model = cotracker_model
        self.sd_model_id = sd_model_id
        self.lora_path = lora_path
        self.device = device
        
        # Components (lazy loaded)
        self._segmenter = None
        self._tracker = None
        self._colorizer = None
        
        logger.info(f"UpdatedColorizationPipeline initialized")
        logger.info(f"  SAM-2: {sam2_model_size}")
        logger.info(f"  CoTracker: {cotracker_model}")
        logger.info(f"  LoRA: {lora_path or 'None'}")
    
    @property
    def segmenter(self) -> SAM2Segmenter:
        """Get or create SAM-2 segmenter."""
        if self._segmenter is None:
            self._segmenter = SAM2Segmenter(model_size=self.sam2_model_size)
        return self._segmenter
    
    @property
    def tracker(self) -> CoTrackerMotion:
        """Get or create CoTracker motion tracker."""
        if self._tracker is None:
            self._tracker = CoTrackerMotion(model_name=self.cotracker_model)
        return self._tracker
    
    @property
    def colorizer(self) -> SD15Runner:
        """Get or create SD1.5 colorizer."""
        if self._colorizer is None:
            self._colorizer = SD15Runner(model_id=self.sd_model_id)
        return self._colorizer
    
    def segment_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Segment a single frame using SAM-2.
        
        Args:
            frame: RGB numpy array (H, W, 3)
            
        Returns:
            Tuple of (id_map, masks_data)
        """
        frame_pil = Image.fromarray(frame)
        masks_data = self.segmenter.segment_frame(frame_pil)
        
        # Create ID map
        h, w = frame.shape[:2]
        id_map = np.zeros((h, w), dtype=np.uint16)
        masks_data.sort(key=lambda x: x['area'], reverse=True)
        
        for i, m_data in enumerate(masks_data, 1):
            id_map[m_data['segmentation']] = i
        
        return id_map, masks_data
    
    def track_video(
        self, 
        frames: List[np.ndarray],
        grid_size: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        Track motion across video frames using CoTracker.
        
        Args:
            frames: List of RGB numpy arrays
            grid_size: Tracking grid density
            
        Returns:
            Motion data dict with 'tracks' and 'visibility'
        """
        return self.tracker.track_video(frames, grid_size=grid_size)
    
    def colorize_frame(
        self,
        frame: np.ndarray,
        mood: str = "sunny_day",
        base_prompt: str = "",
        denoise_strength: float = 0.3,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Colorize a single grayscale frame.
        
        Args:
            frame: Grayscale or RGB numpy array
            mood: Mood preset name
            base_prompt: Additional prompt text
            denoise_strength: Post-processing strength (0-1)
            seed: Random seed
            
        Returns:
            Colorized RGB numpy array
        """
        # Ensure RGB
        if len(frame.shape) == 2:
            frame = np.stack([frame] * 3, axis=-1)
        
        frame_pil = Image.fromarray(frame)
        
        # Extract edges for ControlNet
        edges = extract_canny(frame_pil)
        
        # Build prompt
        prompt, negative = style_manager.build_prompt(base_prompt, mood)
        style_config = style_manager.get_style_config(mood)
        lora_scale = style_config.get("lora_scale", 0.5)
        
        # Load LoRA if specified and not loaded
        if self.lora_path and self.colorizer.pipe:
            self.colorizer.load_lora(self.lora_path, weight=lora_scale)
        
        # Generate
        result = self.colorizer.generate_frame(
            image=edges,
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=50,
            guidance_scale=5.5,
            controlnet_scale=0.95,
            lora_scale=lora_scale,
            denoise_strength=denoise_strength,
            seed=seed
        )
        
        return np.array(result)
    
    def process_video(
        self,
        frames: List[np.ndarray],
        mood: str = "sunny_day",
        keyframe_interval: int = 10,
        denoise_strength: float = 0.3,
        base_prompt: str = "",
        seed: Optional[int] = None,
        track_motion: bool = True,
        segment_frames: bool = False
    ) -> List[np.ndarray]:
        """
        Process entire video through the pipeline.
        
        Args:
            frames: List of grayscale/RGB numpy arrays
            mood: Mood preset name  
            keyframe_interval: Colorize every N frames
            denoise_strength: Post-processing strength (0-1)
            base_prompt: Additional prompt text
            seed: Random seed
            track_motion: Whether to run CoTracker
            segment_frames: Whether to run SAM-2 segmentation
            
        Returns:
            List of colorized RGB numpy arrays
        """
        n_frames = len(frames)
        logger.info(f"Processing {n_frames} frames with mood '{mood}'")
        
        # Step 1: Optional segmentation
        if segment_frames:
            logger.info("Running SAM-2 segmentation...")
            self.segmenter.load_model()
            masks = []
            for i, frame in enumerate(frames):
                if i % 10 == 0:
                    logger.info(f"Segmenting frame {i}/{n_frames}")
                id_map, _ = self.segment_frame(frame)
                masks.append(id_map)
            self.segmenter.unload_model()
            device_manager.empty_cache()
        
        # Step 2: Optional motion tracking
        motion_data = None
        if track_motion and n_frames > 1:
            logger.info("Running CoTracker motion tracking...")
            self.tracker.load_model()
            motion_data = self.track_video(frames, grid_size=20)
            self.tracker.unload_model()
            device_manager.empty_cache()
        
        # Step 3: Colorize keyframes
        logger.info(f"Colorizing keyframes (interval={keyframe_interval})...")
        self.colorizer.load_pipeline()
        if self.lora_path:
            style_config = style_manager.get_style_config(mood)
            self.colorizer.load_lora(self.lora_path, style_config.get("lora_scale", 0.5))
        
        keyframe_indices = list(range(0, n_frames, keyframe_interval))
        colorized_keyframes = {}
        
        for idx in keyframe_indices:
            logger.info(f"Colorizing keyframe {idx}/{n_frames}")
            colorized = self.colorize_frame(
                frames[idx],
                mood=mood,
                base_prompt=base_prompt,
                denoise_strength=denoise_strength,
                seed=seed
            )
            colorized_keyframes[idx] = colorized
        
        self.colorizer.unload_pipeline()
        device_manager.empty_cache()
        
        # Step 4: Propagate colors
        logger.info("Propagating colors to all frames...")
        result = self._propagate_colors(
            frames, colorized_keyframes, motion_data, keyframe_interval
        )
        
        logger.info("Pipeline complete!")
        return result
    
    def _propagate_colors(
        self,
        frames: List[np.ndarray],
        keyframes: Dict[int, np.ndarray],
        motion_data: Optional[Dict],
        interval: int
    ) -> List[np.ndarray]:
        """Propagate keyframe colors to intermediate frames."""
        import cv2
        
        n_frames = len(frames)
        result = [None] * n_frames
        
        # Set keyframes
        for idx, colorized in keyframes.items():
            result[idx] = colorized
        
        # Get sorted keyframe indices
        key_indices = sorted(keyframes.keys())
        
        # Propagate between keyframes
        for i in range(len(key_indices) - 1):
            start_idx = key_indices[i]
            end_idx = key_indices[i + 1]
            
            start_frame = keyframes[start_idx]
            end_frame = keyframes[end_idx]
            
            for j in range(start_idx + 1, end_idx):
                # Linear blend as fallback
                alpha = (j - start_idx) / (end_idx - start_idx)
                blended = cv2.addWeighted(
                    start_frame, 1 - alpha,
                    end_frame, alpha, 0
                )
                result[j] = blended
        
        # Handle frames after last keyframe
        if key_indices[-1] < n_frames - 1:
            last_keyframe = keyframes[key_indices[-1]]
            for j in range(key_indices[-1] + 1, n_frames):
                result[j] = last_keyframe.copy()
        
        return result
    
    def get_palette_preview(self, mood: str, size: Tuple[int, int] = (256, 64)) -> Image.Image:
        """Get a visual preview of the 4-color palette for a mood."""
        return FourColorPaletteGenerator.create_palette_image(mood, size)
    
    def list_available_moods(self) -> List[str]:
        """Get list of available mood presets."""
        return style_manager.list_styles()
    
    def cleanup(self):
        """Release all resources."""
        if self._segmenter:
            self._segmenter.unload_model()
        if self._tracker:
            self._tracker.unload_model()
        if self._colorizer:
            self._colorizer.unload_pipeline()
        device_manager.empty_cache()
        logger.info("Pipeline cleanup complete")


# Convenience function
def create_pipeline(**kwargs) -> UpdatedColorizationPipeline:
    """Create a new pipeline instance."""
    return UpdatedColorizationPipeline(**kwargs)
