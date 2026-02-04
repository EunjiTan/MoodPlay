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
        "sunny_day": {
            "prompt": "bright sunny day, clear blue sky, warm sunlight, vibrant colors, happiness, sharp focus, 8k, photorealistic surroundings, natural lighting",
            "negative": "dark, gloomy, rain, fog, snow, cartoon, anime, artificial, sketch, blurry, watermark, extra objects, illustration",
            "lora_scale": 0.25
        },
        "winter": {
            "prompt": "cold winter atmosphere, snow-covered, frost, ice, cool blue tones, winter lighting, soft diffuse light, 8k, photorealistic",
            "negative": "warm, yellow, orange, summer, sunny, grass, flowers, cartoon, anime, illustration, extra objects, painting",
            "lora_scale": 0.25
        },
        "golden_hour": {
            "prompt": "golden hour lighting, warm glow, long shadows, romantic atmosphere, sunset colors, orange and purple hues, photorealistic, 8k",
            "negative": "cold, blue, harsh light, mid-day, cartoon, anime, drawing, painting, extra people, artificial colors",
            "lora_scale": 0.25
        },
        "overcast": {
            "prompt": "overcast day, soft diffused lighting, neutral tones, cloud cover, shadowless, balanced exposure, photorealistic, 8k",
            "negative": "harsh shadows, bright sun, high contrast, cartoon, anime, painting, saturation, vibrant",
            "lora_scale": 0.2
        },
        "night_scene": {
            "prompt": "night scene, cinematic lighting, dark moody atmosphere, artificial lights, bioluminescence, deep shadows, high contrast, 8k",
            "negative": "daytime, sun, bright, washed out, cartoon, anime, noise, grain, low quality",
            "lora_scale": 0.3
        },
        "autumn": {
            "prompt": "autumn season, fall colors, orange and red leaves, cozy atmosphere, earth tones, soft sunlight, photorealistic, 8k",
            "negative": "green, summer, winter, snow, spring, cartoon, anime, neon colors, artificial",
            "lora_scale": 0.25
        },
        "spring": {
            "prompt": "spring season, fresh green grass, blooming flowers, pastel colors, soft sunlight, vibrant nature, photorealistic, 8k",
            "negative": "autumn, dry, dead, brown, winter, snow, cartoon, anime, dark, gloomy",
            "lora_scale": 0.25
        },
        "desert": {
            "prompt": "desert landscape, hot sun, sand dunes, dry atmosphere, warm yellow and orange tones, harsh lighting, heat haze, photorealistic",
            "negative": "water, rain, snow, green, lush, forest, cold, cartoon, anime, moist",
            "lora_scale": 0.3
        },
        "tropical": {
            "prompt": "tropical paradise, vibrant turquoise water, lush green palm trees, saturated colors, exotic, bright sunlight, holiday vibe, 8k",
            "negative": "cold, grey, dull, desaturated, urban, city, cartoon, anime, drawing",
            "lora_scale": 0.35
        },
        "vintage": {
            "prompt": "vintage film look, classic movie style, slightly faded colors, film grain, analog aesthetic, nostalgic, photorealistic",
            "negative": "digital, sharp, modern, hd, 4k, neon, futuristic, cartoon, anime",
            "lora_scale": 0.3
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
