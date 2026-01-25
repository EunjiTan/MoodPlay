"""
Temporal consistency module for video colorization.
Implements CLIP-based similarity checking and color histogram matching.
"""

import numpy as np
from PIL import Image
import cv2

# Global flags
TORCH_AVAILABLE = False
CLIP_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    print("Warning: Torch not available.")

try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    print("Warning: OpenCLIP not available. Using histogram-based similarity.")


class CLIPSimilarityChecker:
    """
    Uses CLIP embeddings to check visual similarity between frames.
    Falls back to histogram comparison when CLIP is unavailable.
    """
    
    def __init__(self, 
                 model_name: str = "ViT-B-32", 
                 pretrained: str = "laion2b_s34b_b79k",
                 threshold: float = 0.85):
        self.threshold = threshold
        self.device = "cpu"
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"
        
        self.model = None
        self.preprocess = None
        self.model_name = model_name
        self.pretrained = pretrained
    
    def _load_model(self):
        """Load CLIP model."""
        if not CLIP_AVAILABLE or not TORCH_AVAILABLE:
            return
        
        print(f"Loading CLIP model {self.model_name}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, 
            pretrained=self.pretrained,
            device=self.device
        )
        self.model.eval()
        print("CLIP model loaded.")
    
    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Get CLIP embedding for an image.
        
        Args:
            image: Input image (H, W, 3) in RGB
            
        Returns:
            Embedding vector
        """
        if not CLIP_AVAILABLE or not TORCH_AVAILABLE:
            # Fallback: use color histogram as "embedding"
            return self._get_histogram_embedding(image)
        
        if self.model is None:
            self._load_model()
        
        pil_image = Image.fromarray(image)
        image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy().flatten()
    
    def _get_histogram_embedding(self, image: np.ndarray) -> np.ndarray:
        """Fallback: use color histogram as embedding."""
        # Compute color histogram in HSV space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        # Concatenate and normalize
        hist = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()])
        hist = hist / (np.linalg.norm(hist) + 1e-8)
        
        return hist
    
    def compute_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Compute visual similarity between two images.
        
        Args:
            image1: First image (H, W, 3)
            image2: Second image (H, W, 3)
            
        Returns:
            Similarity score between 0 and 1
        """
        emb1 = self.get_embedding(image1)
        emb2 = self.get_embedding(image2)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2)
        return float(similarity)
    
    def check_consistency(self, image1: np.ndarray, image2: np.ndarray) -> tuple:
        """
        Check if two consecutive frames are temporally consistent.
        
        Args:
            image1: First frame
            image2: Second frame
            
        Returns:
            Tuple of (is_consistent, similarity_score)
        """
        similarity = self.compute_similarity(image1, image2)
        is_consistent = similarity >= self.threshold
        return is_consistent, similarity


class ColorHistogramMatcher:
    """
    Matches color histograms between frames for consistency.
    """
    
    def match_histogram(self, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Match the histogram of source image to reference image.
        
        Args:
            source: Source image to adjust (H, W, 3)
            reference: Reference image with target histogram (H, W, 3)
            
        Returns:
            Source image with matched histogram
        """
        matched = np.zeros_like(source)
        
        for channel in range(3):
            matched[:, :, channel] = self._match_channel(
                source[:, :, channel],
                reference[:, :, channel]
            )
        
        return matched
    
    def _match_channel(self, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Match histogram for a single channel."""
        # Get histograms
        src_hist, bins = np.histogram(source.flatten(), 256, [0, 256])
        ref_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])
        
        # Compute CDFs
        src_cdf = src_hist.cumsum()
        ref_cdf = ref_hist.cumsum()
        
        # Normalize
        src_cdf = src_cdf / src_cdf[-1]
        ref_cdf = ref_cdf / ref_cdf[-1]
        
        # Create lookup table
        lookup = np.zeros(256)
        for i in range(256):
            j = 255
            while j > 0 and ref_cdf[j] > src_cdf[i]:
                j -= 1
            lookup[i] = j
        
        # Apply lookup
        matched = lookup[source.astype(np.uint8)]
        return matched.astype(np.uint8)
    
    def blend_histogram(self, 
                        current: np.ndarray, 
                        reference: np.ndarray,
                        strength: float = 0.5) -> np.ndarray:
        """
        Blend current frame's colors towards reference histogram.
        
        Args:
            current: Current frame
            reference: Reference frame
            strength: Blending strength (0 = no change, 1 = full match)
            
        Returns:
            Blended frame
        """
        matched = self.match_histogram(current, reference)
        blended = current * (1 - strength) + matched * strength
        return np.clip(blended, 0, 255).astype(np.uint8)


class TemporalConsistencyEnforcer:
    """
    Main class for enforcing temporal consistency in colorized videos.
    Combines multiple techniques: flow-based warping, CLIP similarity,
    and histogram matching.
    """
    
    def __init__(self,
                 clip_threshold: float = 0.85,
                 histogram_strength: float = 0.3,
                 max_correction_iterations: int = 3):
        self.clip_checker = CLIPSimilarityChecker(threshold=clip_threshold)
        self.histogram_matcher = ColorHistogramMatcher()
        self.histogram_strength = histogram_strength
        self.max_iterations = max_correction_iterations
    
    def enforce_consistency(self, 
                            current_frame: np.ndarray,
                            previous_frame: np.ndarray) -> tuple:
        """
        Enforce temporal consistency between consecutive frames.
        
        Args:
            current_frame: Current colorized frame
            previous_frame: Previous colorized frame
            
        Returns:
            Tuple of (corrected_frame, consistency_score, iterations_used)
        """
        corrected = current_frame.copy()
        
        for iteration in range(self.max_iterations):
            is_consistent, score = self.clip_checker.check_consistency(
                corrected, previous_frame
            )
            
            if is_consistent:
                return corrected, score, iteration
            
            # Apply histogram matching to improve consistency
            corrected = self.histogram_matcher.blend_histogram(
                corrected,
                previous_frame,
                self.histogram_strength * (iteration + 1) / self.max_iterations
            )
        
        # Final check
        _, final_score = self.clip_checker.check_consistency(corrected, previous_frame)
        return corrected, final_score, self.max_iterations
    
    def process_video_frames(self, 
                             frames: list,
                             progress_callback=None) -> tuple:
        """
        Process a list of frames ensuring temporal consistency.
        
        Args:
            frames: List of colorized frames
            progress_callback: Optional callback(current, total, score)
            
        Returns:
            Tuple of (processed_frames, consistency_scores)
        """
        if len(frames) == 0:
            return [], []
        
        processed = [frames[0]]
        scores = [1.0]  # First frame is reference
        
        for i in range(1, len(frames)):
            corrected, score, iters = self.enforce_consistency(
                frames[i], 
                processed[-1]
            )
            processed.append(corrected)
            scores.append(score)
            
            if progress_callback:
                progress_callback(i, len(frames), score)
        
        return processed, scores
