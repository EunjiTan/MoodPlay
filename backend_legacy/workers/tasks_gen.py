"""
Video Generation/Colorization Worker Task.
"""

import os
import time
from typing import Dict, Any, Optional, List
from backend.workers.celery_config import celery_app
from backend.workers.colorization_config import (
    LORA_COLORIZATION_PATH, LORA_STYLE_PATH,
    LORA_COLORIZATION_WEIGHT, LORA_STYLE_WEIGHT,
    DEFAULT_PROMPT_TEMPLATE, DEFAULT_NEGATIVE_PROMPT
)

# Lazy load pipeline
_pipeline = None

def get_pipeline():
    """Lazy load the render pipeline."""
    global _pipeline
    if _pipeline is None:
        from backend.workers.pipeline import RenderPipeline
        _pipeline = RenderPipeline()
    return _pipeline


@celery_app.task(bind=True, name='backend.workers.tasks_gen.generate_video_task')
def generate_video_task(self, 
                        video_path: str, 
                        mask_path: str = None,
                        prompt: str = "vibrant colors, high quality",
                        negative_prompt: str = None,
                        style_lora: str = None,
                        colorization_lora: str = None,
                        **kwargs) -> Dict[str, Any]:
    """
    Enhanced Video Generation Task with full colorization pipeline.
    """
    import traceback
    try:
        self.update_state(state='PROGRESS', meta={
            'status': 'Initializing Generation Pipeline...',
            'progress': 0
        })
        print(f"[GEN] Processing video: {video_path}")
        print(f"[GEN] Mask path: {mask_path}")
        print(f"[GEN] Prompt: {prompt}")
        
        pipeline = get_pipeline()
        
        # Load masks if provided
        masks_by_frame = None
        if mask_path and os.path.isdir(mask_path):
            self.update_state(state='PROGRESS', meta={
                'status': 'Loading segmentation masks...',
                'progress': 5
            })
            masks_by_frame = _load_masks_from_dir(mask_path)
        
        # Resolve LoRA paths
        lora_colorization = colorization_lora or LORA_COLORIZATION_PATH
        lora_style = style_lora or LORA_STYLE_PATH
        
        # Check if LoRAs exist
        if lora_colorization and not os.path.exists(lora_colorization):
            lora_colorization = None
            print(f"[GEN] Colorization LoRA not found, skipping")
        if lora_style and not os.path.exists(lora_style):
            lora_style = None
            print(f"[GEN] Style LoRA not found, skipping")
        
        # Generate output path
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(
            os.path.dirname(video_path).replace('uploads', 'results'),
            f"{base_name}_colorized.mp4"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Progress callback
        def progress_callback(current, total, status):
            progress = 10 + int(85 * current / max(total, 1))
            self.update_state(state='PROGRESS', meta={
                'status': status,
                'progress': progress,
                'current_frame': current,
                'total_frames': total
            })
        
        # Process video
        self.update_state(state='PROGRESS', meta={
            'status': 'Starting colorization...',
            'progress': 10
        })
        
        result_path = pipeline.process_video(
            video_path=video_path,
            prompt=prompt,
            masks_by_frame=masks_by_frame,
            output_path=output_path,
            lora_colorization=lora_colorization,
            lora_style=lora_style,
            progress_callback=progress_callback
        )
        
        self.update_state(state='PROGRESS', meta={
            'status': 'Finalizing output...',
            'progress': 95
        })
        
        return {
            'status': 'Generation Complete',
            'result': result_path,
            'prompt': prompt,
            'masks_used': masks_by_frame is not None
        }

    except Exception as e:
        err_msg = traceback.format_exc()
        # Logging to file for user/agent visibility
        with open("generation_error.log", "w") as f:
            f.write(err_msg)
        print(f"[GEN] CRITICAL ERROR: {err_msg}")
        raise e  # Re-raise to fail the celery task

@celery_app.task(bind=True, name='backend.workers.tasks_gen.colorize_frame_task')
def colorize_frame_task(self,
                        image_path: str,
                        prompt: str,
                        mask: List[List[int]] = None,
                        **kwargs) -> Dict[str, Any]:
    """
    Colorize a single frame/image.
    
    Args:
        image_path: Path to input image
        prompt: Colorization prompt
        mask: Optional 2D mask array
        
    Returns:
        Dict with output path
    """
    from PIL import Image
    import numpy as np
    
    pipeline = get_pipeline()
    
    # Load image
    image = Image.open(image_path)
    
    # Convert mask if provided
    mask_np = None
    if mask:
        mask_np = np.array(mask, dtype=bool)
    
    # Process
    result = pipeline.process_frame(image, prompt, mask=mask_np, **kwargs)
    
    # Save result
    output_path = image_path.rsplit('.', 1)[0] + '_colorized.png'
    result.save(output_path)
    
    return {
        'status': 'Complete',
        'result': output_path
    }


@celery_app.task(bind=True, name='backend.workers.tasks_gen.batch_colorize_task')
def batch_colorize_task(self,
                        frame_paths: List[str],
                        prompt: str,
                        masks_dir: str = None,
                        **kwargs) -> Dict[str, Any]:
    """
    Colorize a batch of frames with temporal consistency.
    
    Args:
        frame_paths: List of paths to frame images
        prompt: Colorization prompt
        masks_dir: Optional directory containing masks
        
    Returns:
        Dict with list of output paths
    """
    from PIL import Image
    import numpy as np
    
    pipeline = get_pipeline()
    
    # Load frames
    frames = []
    for path in frame_paths:
        img = Image.open(path)
        frames.append(np.array(img))
    
    # Load masks if provided
    masks = None
    if masks_dir and os.path.isdir(masks_dir):
        masks = []
        for i, path in enumerate(frame_paths):
            mask_file = os.path.join(masks_dir, f"mask_frame{i:05d}.png")
            if os.path.exists(mask_file):
                mask_img = Image.open(mask_file).convert('L')
                masks.append(np.array(mask_img) > 127)
            else:
                masks.append(None)
    
    # Process batch
    results = pipeline.process_batch(frames, [prompt], masks)
    
    # Save results
    output_paths = []
    for i, (frame, orig_path) in enumerate(zip(results, frame_paths)):
        output_path = orig_path.rsplit('.', 1)[0] + '_colorized.png'
        Image.fromarray(frame).save(output_path)
        output_paths.append(output_path)
        
        self.update_state(state='PROGRESS', meta={
            'current': i + 1,
            'total': len(frames)
        })
    
    return {
        'status': 'Batch Complete',
        'results': output_paths
    }


def _load_masks_from_dir(mask_dir: str) -> Dict[int, Any]:
    """Load masks from a directory into a frame-indexed dict."""
    import cv2
    import numpy as np
    import re
    
    masks_by_frame = {}
    
    for filename in os.listdir(mask_dir):
        if not filename.endswith('.png'):
            continue
        
        # Parse frame number from filename
        match = re.search(r'frame(\d+)', filename)
        if match:
            frame_idx = int(match.group(1))
            mask_path = os.path.join(mask_dir, filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Combine all object masks for same frame
                if frame_idx in masks_by_frame:
                    masks_by_frame[frame_idx] = masks_by_frame[frame_idx] | (mask > 127)
                else:
                    masks_by_frame[frame_idx] = mask > 127
    
    return masks_by_frame
