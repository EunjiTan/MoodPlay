"""
Style Conditioning Module
Manages global style presets, prompt engineering, and 4-color palette integration.
Updated to use LoRA weight 0.5 for moderate palette guidance.
"""

from typing import Dict, List, Optional, Tuple

from backend.conditioning.four_color_palette import FourColorPaletteGenerator


class StyleManager:
    """
    Manages global style conditioning for video colorization.
    Integrates with 4-color palette system for LoRA guidance.
    """
    
    # Updated presets with 4-color palette integration
    # lora_scale set to 0.5 for moderate guidance (not forcing colors)
    PRESETS = {
        "sunny_day": {
            "prompt": "bright sunny day, clear blue sky, warm sunlight, vibrant colors, sharp focus, 8k, photorealistic, natural lighting",
            "negative": "dark, gloomy, rain, fog, snow, cartoon, anime, artificial, sketch, blurry, watermark, extra objects, illustration, painting",
            "lora_scale": 0.5,
            "palette_key": "sunny_day"
        },
        "winter": {
            "prompt": "cold winter atmosphere, snow-covered, frost, ice, cool blue tones, winter lighting, soft diffuse light, 8k, photorealistic",
            "negative": "warm, yellow, orange, summer, sunny, grass, flowers, cartoon, anime, illustration, extra objects, painting",
            "lora_scale": 0.5,
            "palette_key": "winter"
        },
        "golden_hour": {
            "prompt": "golden hour lighting, warm glow, long shadows, romantic atmosphere, sunset colors, orange and purple hues, photorealistic, 8k",
            "negative": "cold, blue, harsh light, mid-day, cartoon, anime, drawing, painting, extra people, artificial colors",
            "lora_scale": 0.5,
            "palette_key": "golden_hour"
        },
        "overcast": {
            "prompt": "overcast day, soft diffused lighting, neutral tones, cloud cover, shadowless, balanced exposure, photorealistic, 8k",
            "negative": "harsh shadows, bright sun, high contrast, cartoon, anime, painting, saturation, vibrant",
            "lora_scale": 0.5,
            "palette_key": "overcast"
        },
        "night_scene": {
            "prompt": "night scene, cinematic lighting, dark moody atmosphere, artificial lights, deep shadows, high contrast, 8k, photorealistic",
            "negative": "daytime, sun, bright, washed out, cartoon, anime, noise, grain, low quality",
            "lora_scale": 0.5,
            "palette_key": "night_scene"
        },
        "autumn": {
            "prompt": "autumn season, fall colors, orange and red leaves, cozy atmosphere, earth tones, soft sunlight, photorealistic, 8k",
            "negative": "green, summer, winter, snow, spring, cartoon, anime, neon colors, artificial",
            "lora_scale": 0.5,
            "palette_key": "autumn"
        },
        "spring": {
            "prompt": "spring season, fresh green grass, blooming flowers, pastel colors, soft sunlight, vibrant nature, photorealistic, 8k",
            "negative": "autumn, dry, dead, brown, winter, snow, cartoon, anime, dark, gloomy",
            "lora_scale": 0.5,
            "palette_key": "spring"
        },
        "desert": {
            "prompt": "desert landscape, hot sun, sand dunes, dry atmosphere, warm yellow and orange tones, harsh lighting, heat haze, photorealistic",
            "negative": "water, rain, snow, green, lush, forest, cold, cartoon, anime, moist",
            "lora_scale": 0.5,
            "palette_key": "desert"
        },
        "tropical": {
            "prompt": "tropical paradise, vibrant turquoise water, lush green palm trees, saturated colors, exotic, bright sunlight, 8k, photorealistic",
            "negative": "cold, grey, dull, desaturated, urban, city, cartoon, anime, drawing",
            "lora_scale": 0.5,
            "palette_key": "tropical"
        },
        "vintage": {
            "prompt": "vintage film look, classic movie style, slightly faded colors, film grain, analog aesthetic, nostalgic, photorealistic",
            "negative": "digital, sharp, modern, hd, 4k, neon, futuristic, cartoon, anime",
            "lora_scale": 0.5,
            "palette_key": "vintage"
        }
    }

    def __init__(self):
        self.palette_generator = FourColorPaletteGenerator()

    def get_style_config(self, style_name: str) -> Dict:
        """Get configuration for a specific style."""
        style_name = style_name.lower().replace(" ", "_")
        if style_name not in self.PRESETS:
            # Default to sunny_day
            return self.PRESETS["sunny_day"]
        return self.PRESETS[style_name]

    def build_prompt(self, base_prompt: str, style_name: str) -> Tuple[str, str]:
        """Upgrade a simple prompt with style-specific keywords."""
        config = self.get_style_config(style_name)
        
        style_prompt = config["prompt"]
        negative_prompt = config["negative"]
        
        # Combine base instruction with style
        if base_prompt:
            full_prompt = f"{base_prompt}, {style_prompt}"
        else:
            full_prompt = style_prompt
        
        return full_prompt, negative_prompt
    
    def get_palette_colors(self, style_name: str) -> Dict:
        """Get the 4-color palette for a style."""
        config = self.get_style_config(style_name)
        palette_key = config.get("palette_key", style_name)
        return FourColorPaletteGenerator.get_palette(palette_key)
    
    def get_palette_image(self, style_name: str, size: Tuple[int, int] = (512, 512)):
        """Get the 4-color conditioning image for LoRA guidance."""
        config = self.get_style_config(style_name)
        palette_key = config.get("palette_key", style_name)
        return FourColorPaletteGenerator.create_conditioning_image(palette_key, size)
    
    def list_styles(self) -> List[str]:
        """Return list of available style names."""
        return list(self.PRESETS.keys())
    
    def get_style_display_info(self, style_name: str) -> Dict:
        """Get display info for UI including palette colors."""
        config = self.get_style_config(style_name)
        palette = self.get_palette_colors(style_name)
        
        return {
            "name": style_name,
            "display_name": palette.get("name", style_name.replace("_", " ").title()),
            "colors": palette.get("colors", []),
            "color_names": palette.get("color_names", []),
            "lora_scale": config.get("lora_scale", 0.5)
        }


# Global instance
style_manager = StyleManager()
