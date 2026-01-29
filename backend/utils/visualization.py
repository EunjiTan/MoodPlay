
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class VisualizationUtils:
    @staticmethod
    def draw_detections(frame, detections):
        """
        Draw bounding boxes and labels on the frame.
        """
        out_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            label = det.get('label', 'Object')
            conf = det.get('confidence', 0.0)
            
            # Color for box (Green)
            color = (0, 255, 0) 
            
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
            
            text = f"{label} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(out_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        return out_frame

    @staticmethod
    def overlay_masks(frame, masks, colors=None, alpha=0.5):
        """
        Overlay segmentation masks on the frame.
        masks: List of boolean masks or a single combined mask.
        """
        if not masks:
            return frame
            
        out_frame = frame.copy()
        img = frame.astype(np.float32)
        
        if colors is None:
            # Generate random colors
            colors = [np.random.randint(0, 255, (1, 3), dtype=np.uint8) for _ in range(len(masks))]
        
        for i, mask in enumerate(masks):
            if mask is None: continue
            
            color = colors[i % len(colors)][0]
            
            # Create colored mask
            colored_mask = np.zeros_like(frame)
            colored_mask[mask > 0] = color
            
            # Blend
            cv2.addWeighted(colored_mask, alpha, out_frame, 1 - alpha, 0, out_frame)
            
            # Draw contours
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out_frame, contours, -1, (255, 255, 255), 1)

        return out_frame
