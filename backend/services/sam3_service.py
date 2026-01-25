import os
import numpy as np
from PIL import Image

class SAM3Service:
    def __init__(self):
        self.device = "cpu" # Default fallback
        self.predictor = None
        self.inference_state = None
        self.model_loaded = False
        self.load_error = None
        
        # Use absolute paths for robustness
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.checkpoint = os.path.join(base_dir, "checkpoints/sam2/sam2_hiera_large.pt")
        self.config = os.path.join(base_dir, "checkpoints/sam2/sam2_hiera_large.yaml")

        # self._lazy_load_model() # Don't load on init to prevent slow startup/crashes

    def _lazy_load_model(self):
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if os.path.exists(self.checkpoint) and os.path.exists(self.config):
                print(f"Loading SAM-2 from {self.checkpoint}...")
                self.predictor = build_sam2_video_predictor(self.config, self.checkpoint, device=self.device)
                self.model_loaded = True
                print("SAM-2 Loaded Successfully.")
            else:
                self.load_error = f"Checkpoint not found at {self.checkpoint}"
                print(f"Warning: {self.load_error}")
                
        except OSError as e:
            # Catch DLL errors (WinError 1114)
            self.load_error = f"System Error (DLL Missing): {e}. Please install Visual C++ Redistributable."
            print(f"CRITICAL ERROR: {self.load_error}")
        except ImportError as e:
            self.load_error = f"Missing Dependency: {e}"
            print(f"CRITICAL ERROR: {self.load_error}")
        except Exception as e:
            self.load_error = f"Unknown Error: {e}"
            print(f"CRITICAL ERROR: {self.load_error}")

    def init_session(self, video_path):
        """Initialize a video segmentation session."""
        if not self.model_loaded:
            # Try loading again just in case
            self._lazy_load_model()
            if not self.model_loaded:
                raise RuntimeError(f"Model not loaded: {self.load_error}")
        
        if not self.predictor:
             raise RuntimeError(f"Predictor is None. Error: {self.load_error}")
        
        print(f"Initializing video: {video_path}")
        self.inference_state = self.predictor.init_state(video_path=video_path)
        self.predictor.reset_state(self.inference_state)
        return {"status": "initialized", "video": video_path}

    def add_click(self, frame_idx, points, labels, object_id=1):
        """Add a click to segment an object."""
        if not self.model_loaded:
             raise RuntimeError(f"Model not loaded: {self.load_error}")
             
        if not self.inference_state:
            raise RuntimeError("Session not initialized")

        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=object_id,
            points=points,
            labels=labels,
        )
        
        return {"status": "click_added", "object_id": object_id}

    def propagate(self):
        """Propagate masks across the entire video."""
        if not self.inference_state:
            raise RuntimeError("Session not initialized")
            
        print("Propagating masks...")
        results = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
             results[out_frame_idx] = [int(id) for id in out_obj_ids]
             
        return {"status": "propagation_complete", "frames_processed": len(results)}

# Global Instance
sam3_service = SAM3Service()
