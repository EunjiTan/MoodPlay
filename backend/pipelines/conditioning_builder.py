"""
Module (a): Spatiotemporal Condition Builder
Builds comprehensive conditioning from video frames using YOLO, SAM, and ControlNet.
"""

import numpy as np
import torch
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.services.yolo_service import yolo_service
from backend.services.sam3_service import sam3_service
from backend.services.controlnet_service import controlnet_service

class SpatiotemporalConditionBuilder:
    def __init__(self):
        self.track_memory = {}  # Store object colors across frames
        self.frame_history = []  # Store recent frames for temporal consistency
        self.max_history = 5
    
    def build_instance_masks(self, frame, frame_idx, video_path=None):
        """
        Sub-module (a1): Instance Mask + Identity Tracking
        Detect objects with YOLO and segment with SAM-2, maintaining TrackIDs.
        
        Args:
            frame (np.ndarray): Current video frame (BGR)
            frame_idx (int): Frame index
            video_path (str): Path to video (for SAM-2 initialization)
        
        Returns:
            dict: {
                'detections': List of YOLO detections with TrackIDs,
                'masks': Dict of {track_id: segmentation_mask},
                'track_ids': List of active track IDs
            }
        """
        # A. Detect objects with YOLO (with tracking)
        detections = yolo_service.detect(frame)
        
        # B. Segment with SAM-2
        masks_dict = {}
        if detections:
            # Initialize SAM session if needed
            if not sam3_service.inference_state and video_path:
                sam3_service.init_session(video_path)
            
            # Get segmentation masks for detected objects
            masks_dict = sam3_service.segment_frame_boxes(frame_idx, detections)
        
        # C. Update track memory
        for det in detections:
            track_id = det.get('track_id')
            if track_id and track_id not in self.track_memory:
                self.track_memory[track_id] = {
                    'first_seen': frame_idx,
                    'last_seen': frame_idx,
                    'class': det['label'],
                    'color': None,  # Will be set during colorization
                }
            elif track_id:
                self.track_memory[track_id]['last_seen'] = frame_idx
        
        return {
            'detections': detections,
            'masks': masks_dict,
            'track_ids': [d.get('track_id') for d in detections if d.get('track_id')]
        }
    
    def build_spatial_constraints(self, masks_dict, user_hints=None):
        """
        Sub-module (a2): Spatial Constraint Constructor
        Convert masks and user inputs into spatial constraints.
        
        Args:
            masks_dict (dict): Segmentation masks from SAM
            user_hints (dict): User-defined color hints {track_id: color}
        
        Returns:
            dict: Spatial constraint maps
        """
        constraints = {
            'semantic_map': None,
            'boundary_map': None,
            'user_color_map': None,
        }
        
        # Create boundary map from masks
        if masks_dict:
            # Combine all masks into a single boundary map
            import cv2
            combined_mask = np.zeros_like(list(masks_dict.values())[0], dtype=np.uint8)
            
            for track_id, mask in masks_dict.items():
                # Find contours for boundaries
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(combined_mask, contours, -1, 255, 2)
            
            constraints['boundary_map'] = combined_mask
        
        # Apply user color hints if provided
        if user_hints:
            constraints['user_color_map'] = user_hints
        
        return constraints
    
    def extract_structural_controls(self, frame):
        """
        Sub-module (a3): Structural Control Extractor
        Extract Canny edges and depth maps using ControlNet preprocessors.
        
        Args:
            frame (np.ndarray): Input frame (BGR)
        
        Returns:
            dict: {
                'canny': PIL.Image (edge map),
                'depth': PIL.Image (depth map)
            }
        """
        from PIL import Image
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Extract structural controls
        controls = controlnet_service.get_controlnet_conditioning(
            pil_image,
            use_canny=True,
            use_depth=True
        )
        
        return controls
    
    def encode_global_conditions(self, prompt, style_image=None):
        """
        Sub-module (a4): Global Conditioning Encoders
        Encode text prompts and style references using CLIP.
        
        Args:
            prompt (str): Text description for colorization
            style_image (PIL.Image): Optional style reference image
        
        Returns:
            dict: {
                'text_embedding': torch.Tensor,
                'style_embedding': torch.Tensor (if style_image provided)
            }
        """
        conditions = {
            'text_prompt': prompt,
            'style_embedding': None
        }
        
        # Style embedding would be extracted here if style_image is provided
        # For now, we'll rely on SDXL's built-in CLIP encoding
        
        return conditions
    
    def build_full_conditioning(self, frame, frame_idx, prompt, video_path=None, user_hints=None, style_image=None):
        """
        Build complete conditioning bundle for a frame.
        
        Args:
            frame (np.ndarray): Current frame
            frame_idx (int): Frame index
            prompt (str): Text prompt
            video_path (str): Video path for SAM initialization
            user_hints (dict): User color hints
            style_image (PIL.Image): Style reference
        
        Returns:
            dict: Complete conditioning bundle
        """
        # 1. Instance masks + TrackID
        instance_data = self.build_instance_masks(frame, frame_idx, video_path)
        
        # 2. Spatial constraints
        spatial_constraints = self.build_spatial_constraints(
            instance_data['masks'],
            user_hints
        )
        
        # 3. Structural controls
        structural_controls = self.extract_structural_controls(frame)
        
        # 4. Global conditions
        global_conditions = self.encode_global_conditions(prompt, style_image)
        
        # Update frame history for temporal consistency
        self.frame_history.append({
            'frame_idx': frame_idx,
            'track_ids': instance_data['track_ids'],
            'masks': instance_data['masks']
        })
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)
        
        return {
            'instance_data': instance_data,
            'spatial_constraints': spatial_constraints,
            'structural_controls': structural_controls,
            'global_conditions': global_conditions,
            'track_memory': self.track_memory,
            'frame_history': self.frame_history
        }
    
    def get_object_color_memory(self, track_id):
        """Get stored color for a tracked object."""
        if track_id in self.track_memory:
            return self.track_memory[track_id].get('color')
        return None
    
    def set_object_color_memory(self, track_id, color):
        """Store color for a tracked object."""
        if track_id in self.track_memory:
            self.track_memory[track_id]['color'] = color

# Global instance
condition_builder = SpatiotemporalConditionBuilder()
