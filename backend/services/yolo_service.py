
import os
import cv2
import numpy as np
from ultralytics import YOLO

class YoloService:
    def __init__(self, model_name="yolo11n.pt"):
        """
        Initialize YOLO model.
        Args:
            model_name (str): Name of the YOLO model to load.
        """
        self.model_name = model_name
        self.model = None
        self.load_model()
    
    def load_model(self):
        try:
            print(f"Loading YOLO model: {self.model_name}...")
            # Ultralytics auto-downloads the model if not found
            self.model = YOLO(self.model_name)
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise e

    def detect(self, frame):
        """
        Detect objects in a frame.
        Args:
            frame (np.ndarray): Image frame (BGR).
        Returns:
            list: List of detections, each dict containing 'box', 'label', 'conf', 'class_id'.
        """
        if self.model is None:
             raise RuntimeError("YOLO Model not loaded.")

                
        # Use track() for persistent tracking
        results = self.model.track(frame, persist=True, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                label = self.model.names[cls]
                
                # Get ID if available
                id = int(box.id[0].cpu().numpy()) if box.id is not None else None
                
                detections.append({
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "label": label,
                    "class_id": cls,
                    "confidence": conf,
                    "track_id": id
                })
        
        return detections

# Global instance
yolo_service = YoloService()
