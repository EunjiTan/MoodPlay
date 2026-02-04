"""
Style Conditioning Module
Manages global style presets, prompt engineering, and palette application.
"""

from typing import Dict, List, Optional, Tuple
import random

class StyleManager:
    """
    Manages global style conditioning for video colorization.
    """
    
    PRESETS = {
        "cinematic": {
            "prompt": "cinematic film look, teal and orange color grading, dramatic lighting, professional photography, 8k, detailed texture",
            "negative": "flat, amateur, oversaturated, cartoon, anime, low quality, jpeg artifacts, grain, noise",
            "lora_scale": 0.6
        },
        "winter": {
            "prompt": "cold winter atmosphere, blue and white tones, frost, snow, crisp cold lighting, photorealistic, 8k",
            "negative": "warm, orange, summer, sunny, yellow tones, tropical",
            "lora_scale": 0.7
        },
        "sunny_day": {
            "prompt": "bright sunny day, warm golden hour colors, vibrant natural lighting, clear blue skies, happiness, photorealistic, 8k",
            "negative": "dark, gloomy, cold, rain, night, storm, desaturated",
            "lora_scale": 0.7
        },
        "vintage": {
            "prompt": "vintage 1970s film look, kodak portra, warm retro aesthetic, analog photography style, film grain, soft focus",
            "negative": "modern, digital, sharp, high contrast, hd, 4k",
            "lora_scale": 0.8
        },
        "noir": {
            "prompt": "film noir style, high contrast, dramatic shadows, moody atmosphere, cinematic lighting, 1940s",
            "negative": "color, vibrant, bright, flat lighting",
            "lora_scale": 0.5
        },
        "natural": {
            "prompt": "natural lighting, neutral color grade, photorealistic documentary style, balanced colors, 8k",
            "negative": "stylized, artistic, filter, oversaturated, cartoon",
            "lora_scale": 0.5
        }
    }

    def __init__(self):
        pass

    def get_style_config(self, style_name: str) -> Dict:
        """Get configuration for a specific style."""
        style_name = style_name.lower()
        if style_name not in self.PRESETS:
            # Fallback for custom prompts treated as a "natural" base? 
            # Or just return a default configuration if unknown
            return self.PRESETS["natural"]
        return self.PRESETS[style_name]

    def build_prompt(self, base_prompt: str, style_name: str) -> Tuple[str, str]:
        """Upgrade a simple prompt with style-specific keywords."""
        config = self.get_style_config(style_name)
        
        style_prompt = config["prompt"]
        negative_prompt = config["negative"]
        
        # Combine base instruction with style style
        # e.g. "a living room" + "cinematic film look..."
        full_prompt = f"{base_prompt}, {style_prompt}"
        
        return full_prompt, negative_prompt

# Global instance
style_manager = StyleManager()
