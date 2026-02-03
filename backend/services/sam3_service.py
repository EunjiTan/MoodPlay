import os
import numpy as np
from ultralytics import SAM

class SAM3Service:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.load_error = None
        # Ultralytics handles model loading and downloading automatically
        # 'sam2_t.pt' is the tiny version of SAM2, perfect for 4GB VRAM
        self.model_name = "sam2_t.pt" 

    def _lazy_load_model(self):
        try:
            print(f"Loading SAM-2 (via Ultralytics): {self.model_name}...")
            self.model = SAM(self.model_name)
            self.model_loaded = True
            print("SAM-2 Loaded Successfully.")
        except Exception as e:
            self.load_error = f"Error loading SAM model: {e}"
            print(f"CRITICAL ERROR: {self.load_error}")

    def init_session(self, video_path):
        """Initialize segmentation session."""
        if not self.model_loaded:
            self._lazy_load_model()
            if not self.model_loaded:
                raise RuntimeError(f"Model not loaded: {self.load_error}")
        
        # Ultralytics SAM doesn't need explicit state initialization like native SAM2
        # We just need to ensure the model is ready.
        return {"status": "initialized", "video": video_path}

    def segment_frame_boxes(self, frame_idx, detections):
        """
        Segment objects in a frame using bounding boxes from YOLO.
        detections: List of dicts with 'box' (x1,y1,x2,y2) and 'class_id'/'track_id'.
        """
        if not self.model_loaded:
            self._lazy_load_model()

        masks = {}
        
        # Prepare boxes for batch prediction
        bboxes = [d['box'] for d in detections]
        if not bboxes:
            return masks
            
        # Predict masks using SAM
        # Note: Ultralytics SAM predict takes image path or numpy array
        # We assume the caller might need to pass the frame image to us.
        # BUT: The current signature only has frame_idx and detections.
        # We need the FRAME IMAGE. 
        # Checking video_pipeline.py: it calls segment_frame_boxes(frame_idx, detections)
        # It DOES NOT pass the frame. This is a design flaw in the original service interface for Ultralytics usage.
        
        # WORKAROUND: We can't use Ultralytics inference easily without the image.
        # Native SAM2 had state. Ultralytics does not maintain video state in the same way for simple box prompting.
        
        # Re-reading video_pipeline.py:
        # line 50: detections = yolo_service.detect(bgr_frame)
        # line 62: masks_dict = sam3_service.segment_frame_boxes(frame_idx, detections)
        
        # I MUST update video_pipeline.py to pass the frame to segment_frame_boxes.
        pass 
        return {} 

    # Since I need to change the interface, I will provide a dummy implementation here 
    # and then immediately update video_pipeline.py to pass the frame.
    def segment_frame_boxes_with_image(self, frame, detections):
        if not self.model_loaded:
            self._lazy_load_model()
            
        masks = {}
        bboxes = [d['box'] for d in detections]
        if not bboxes:
            return masks
            
        # Run inference
        results = self.model(frame, bboxes=bboxes, verbose=False)
        
        # Process results
        if results and results[0].masks:
            # Match results back to detections
            # Ultralytics returns masks in order of boxes
            for i, result_mask in enumerate(results[0].masks.data):
                # result_mask is a tensor (H, W) or (1, H, W)
                mask_np = result_mask.cpu().numpy().squeeze()
                
                # Get object ID
                obj_id = int(detections[i].get('track_id', i + 1))
                masks[obj_id] = mask_np
                
        return masks

# Global Instance
sam3_service = SAM3Service()
