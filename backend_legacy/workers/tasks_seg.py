"""
SAM-2 Segmentation Worker Task with multi-modal prompt support.
"""

import os
import time
from typing import List, Dict, Any, Optional
from backend.workers.celery_config import celery_app
from backend.workers.colorization_config import (
    SAM2_MODEL_SIZE, SAM2_MODEL_PATHS, SAM2_CONFIG_PATHS,
    SAM2_CONFIDENCE_THRESHOLD, SAM2_TEMPORAL_TRACKING
)

# Lazy load tracker
_tracker = None

def get_tracker():
    """Lazy load the SAM-2 tracker."""
    global _tracker
    if _tracker is None:
        from backend.processing.tracking import SAM2Tracker
        checkpoint = SAM2_MODEL_PATHS.get(SAM2_MODEL_SIZE, SAM2_MODEL_PATHS["large"])
        config = SAM2_CONFIG_PATHS.get(SAM2_MODEL_SIZE, SAM2_CONFIG_PATHS["large"])
        _tracker = SAM2Tracker(checkpoint, config, SAM2_CONFIDENCE_THRESHOLD)
    return _tracker


@celery_app.task(bind=True, name='backend.workers.tasks_seg.segment_video_task')
def segment_video_task(self, 
                       video_path: str, 
                       prompts: List[Dict[str, Any]],
                       output_dir: str = None) -> Dict[str, Any]:
    """
    SAM-2 Segmentation Task with multi-modal prompting support.
    
    Args:
        video_path: Path to input video
        prompts: List of prompt dicts, each containing:
            - object_id: Unique ID for the object
            - frame_idx: Frame index for the prompt (default: 0)
            - type: "click", "box", "text", "negative"
            - data: Prompt-specific data
                - click: {"points": [[x, y], ...], "labels": [1, ...]}
                - box: {"box": [x1, y1, x2, y2]}
                - text: {"text": "description"}
                - negative: {"points": [[x, y], ...]}
        output_dir: Directory to save mask outputs
        
    Returns:
        Dict with mask_dir, object_ids, and frame_count
    """
    self.update_state(state='PROGRESS', meta={'status': 'Initializing SAM-2...', 'progress': 0})
    print(f"[SAM-2] Processing video: {video_path}")
    print(f"[SAM-2] Prompts: {prompts}")
    
    tracker = get_tracker()
    
    # Initialize video
    self.update_state(state='PROGRESS', meta={'status': 'Loading video...', 'progress': 5})
    frame_count = tracker.init_video(video_path)
    
    # Process prompts
    self.update_state(state='PROGRESS', meta={'status': 'Processing prompts...', 'progress': 10})
    
    for prompt in prompts:
        object_id = prompt.get('object_id', 1)
        frame_idx = prompt.get('frame_idx', 0)
        prompt_type = prompt.get('type', 'click')
        data = prompt.get('data', {})
        
        if prompt_type == 'click':
            points = data.get('points', [])
            labels = data.get('labels', [1] * len(points))
            tracker.add_click(frame_idx, object_id, points, labels)
            
        elif prompt_type == 'box':
            box = data.get('box', [0, 0, 100, 100])
            tracker.add_box(frame_idx, object_id, tuple(box))
            
        elif prompt_type == 'text':
            text = data.get('text', '')
            tracker.add_text_prompt(frame_idx, object_id, text)
            
        elif prompt_type == 'negative':
            points = data.get('points', [])
            tracker.add_negative_region(frame_idx, object_id, points)
    
    # Propagate masks through video
    self.update_state(state='PROGRESS', meta={'status': 'Propagating masks...', 'progress': 20})
    
    processed_frames = 0
    for frame_idx, obj_ids, masks in tracker.propagate():
        processed_frames += 1
        progress = 20 + int(70 * processed_frames / frame_count)
        self.update_state(state='PROGRESS', meta={
            'status': f'Segmenting frame {frame_idx}/{frame_count}',
            'progress': progress,
            'current_frame': frame_idx,
            'total_frames': frame_count
        })
    
    # Export masks
    self.update_state(state='PROGRESS', meta={'status': 'Exporting masks...', 'progress': 90})
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(video_path), 'masks')
    
    saved_files = tracker.export_masks(output_dir, prefix='mask')
    
    # Create mask video for preview
    mask_video_path = _create_mask_video(tracker, video_path, output_dir)
    
    return {
        'status': 'Segmentation Complete',
        'mask_dir': output_dir,
        'mask_video': mask_video_path,
        'object_ids': list(tracker.objects.keys()),
        'frame_count': frame_count,
        'files_saved': len(saved_files)
    }


@celery_app.task(bind=True, name='backend.workers.tasks_seg.refine_segmentation_task')
def refine_segmentation_task(self,
                              video_path: str,
                              object_id: int,
                              frame_idx: int,
                              refinement_prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Refine an existing segmentation with additional prompts.
    
    Args:
        video_path: Path to video (for context)
        object_id: ID of object to refine
        frame_idx: Frame to apply refinement
        refinement_prompts: List of additional prompts to apply
        
    Returns:
        Updated mask information
    """
    self.update_state(state='PROGRESS', meta={'status': 'Refining segmentation...'})
    
    tracker = get_tracker()
    
    for prompt in refinement_prompts:
        prompt_type = prompt.get('type', 'negative')
        data = prompt.get('data', {})
        
        if prompt_type == 'negative':
            points = data.get('points', [])
            tracker.add_negative_region(frame_idx, object_id, points)
        elif prompt_type == 'click':
            points = data.get('points', [])
            labels = data.get('labels', [1] * len(points))
            tracker.add_click(frame_idx, object_id, points, labels)
    
    # Re-propagate from the refined frame
    for frame_idx, obj_ids, masks in tracker.propagate(start_frame=frame_idx):
        pass  # Just iterate to update masks
    
    return {
        'status': 'Refinement Complete',
        'object_id': object_id,
        'frame_idx': frame_idx
    }


def _create_mask_video(tracker, video_path: str, output_dir: str) -> str:
    """Create a visualization video of the masks."""
    import cv2
    import numpy as np
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = os.path.join(output_dir, 'mask_preview.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Overlay masks with semi-transparent colors
        for i, obj_id in enumerate(tracker.objects.keys()):
            mask = tracker.get_mask(obj_id, frame_idx)
            if mask is not None:
                color = colors[i % len(colors)]
                # Resize mask if needed
                if mask.shape[:2] != (height, width):
                    mask = cv2.resize(mask.astype(np.uint8), (width, height)).astype(bool)
                # Apply color overlay
                overlay = frame.copy()
                overlay[mask] = color
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    
    return output_path
