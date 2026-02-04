"""
Instance Hints Module
Manages user-specified color hints for specific instances.
"""

from typing import  Dict, List, Optional, Tuple
import numpy as np

class InstanceHintManager:
    """
    Manages color hints for specific instance IDs.
    e.g. ID #5 (car) -> "red sports car"
    """
    
    def __init__(self):
        # Maps instance_id -> text prompt or color tuple
        self.hints: Dict[int, Dict] = {} 
        
    def add_hint(self, instance_id: int, prompt: str = None, color: Tuple[int, int, int] = None):
        """Add a hint for a specific instance."""
        self.hints[instance_id] = {
            "prompt": prompt,
            "color": color
        }
    
    def get_hint(self, instance_id: int) -> Optional[Dict]:
        """Get hint for an instance."""
        return self.hints.get(instance_id)

    def apply_hints_to_prompt(self, base_prompt: str, active_instance_ids: List[int]) -> str:
        """
        Dynamically modify prompt based on visible instances.
        Note: This is a simple concatenation strategy. 
        Advanced implementations uses multi-diffusion or attention masking.
        """
        # For simple global prompt augmentation:
        hints_text = []
        for iid in active_instance_ids:
            hint = self.get_hint(iid)
            if hint and hint.get("prompt"):
                hints_text.append(hint["prompt"])
        
        if not hints_text:
            return base_prompt
            
        full_prompt = base_prompt + ", " + ", ".join(hints_text)
        return full_prompt

# Global instance
instance_hint_manager = InstanceHintManager()
