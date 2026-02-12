"""
YOLOv11 Service (Enhanced)
Object detection and persistent Instance ID assignment.
Section 3.2: detect, assign persistent IDs, bind to Instance Registry.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Optional, Tuple

from backend.core.pipeline_config import CFG


class YoloService:
    def __init__(self, model_name: str = "yolo11n.pt"):
        """
        Initialize YOLO model.
        Args:
            model_name: Name of the YOLO model to load.
        """
        self.model_name = model_name
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            print(f"Loading YOLO model: {self.model_name}...")
            self.model = YOLO(self.model_name)
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise e

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a frame with persistent tracking IDs.
        Args:
            frame: Image frame (BGR), np.ndarray.
        Returns:
            list of dicts: 'box', 'label', 'conf', 'class_id', 'track_id'.
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

                # Get track ID if available
                tid = int(box.id[0].cpu().numpy()) if box.id is not None else None

                detections.append({
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "label": label,
                    "class_id": cls,
                    "confidence": conf,
                    "track_id": tid,
                })

        return detections

    # ── NEW: Enhanced detection for pipeline (Section 3.2) ─────────────

    def detect_and_register(
        self,
        frame: np.ndarray,
        registry,  # InstanceRegistry
        frame_idx: int,
        palette_assigner=None,
    ) -> List[Dict]:
        """Detect objects, assign persistent IDs via registry, filter low confidence.

        Args:
            frame: BGR image.
            registry: InstanceRegistry instance.
            frame_idx: Current frame index.
            palette_assigner: Optional callable(class_label, instance_id) -> rgb.

        Returns:
            list of dicts with keys:
              'box', 'label', 'class_id', 'confidence', 'track_id',
              'instance_id' (registry-assigned), 'is_new'.
        """
        raw = self.detect(frame)
        enriched = []

        for det in raw:
            # Suppress low-confidence detections (Section 3.2)
            if det["confidence"] < CFG.DETECTION_CONFIDENCE_MIN:
                continue

            yolo_tid = det["track_id"]
            box = np.array(det["box"])
            label = det["label"]

            # Check if YOLO already gave a stable track_id
            if yolo_tid is not None:
                existing = registry.get(yolo_tid)
                if existing is not None:
                    # Confirm in this frame
                    registry.confirm_frame(yolo_tid, frame_idx, bbox=box)
                    det["instance_id"] = yolo_tid
                    det["is_new"] = False
                    enriched.append(det)
                    continue

                # Try re-entry match for deactivated instances
                match_id = registry.find_reentry_match(label, box)
                if match_id is not None:
                    registry.reactivate(match_id, frame_idx)
                    registry.confirm_frame(match_id, frame_idx, bbox=box)
                    det["instance_id"] = match_id
                    det["is_new"] = False
                    enriched.append(det)
                    continue

            # New instance — register it
            palette_rgb = np.array([128, 128, 128], dtype=np.uint8)
            if palette_assigner is not None:
                iid_temp = registry._next_id
                palette_rgb = palette_assigner(label, iid_temp)

            new_id = registry.register(
                class_label=label,
                palette_color_rgb=palette_rgb,
                frame_idx=frame_idx,
                bbox=box,
            )
            det["instance_id"] = new_id
            det["is_new"] = True
            enriched.append(det)

        return enriched

    def filter_low_confidence(
        self, detections: List[Dict], threshold: float = CFG.DETECTION_CONFIDENCE_MIN
    ) -> List[Dict]:
        """Remove detections below confidence threshold."""
        return [d for d in detections if d["confidence"] >= threshold]

    def unload_model(self):
        """Free resources."""
        self.model = None


# Global instance
yolo_service = YoloService()
