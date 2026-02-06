"""
4-Color Palette Generator Module
Creates simplified 4-color palettes for each mood for LoRA guidance.
Each mood has exactly 4 semantically meaningful colors.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import Dict, List, Tuple, Optional

from backend.core.logging import get_logger

logger = get_logger("conditioning.four_color_palette")


class FourColorPaletteGenerator:
    """
    Generates 4-color palette images for LoRA guidance.
    
    Each mood has exactly 4 colors with semantic roles:
    1. Primary (sky/atmosphere)
    2. Accent (highlights/light)
    3. Secondary (foliage/nature)
    4. Tertiary (ground/shadows)
    """
    
    # 10 Mood Palettes - Each with exactly 4 colors
    # Format: (R, G, B) tuples
    PALETTES = {
        "sunny_day": {
            "name": "☀️ Sunny Day",
            "colors": [
                (135, 206, 235),   # Sky Blue - sky/atmosphere
                (255, 215, 0),     # Golden Yellow - sunlight/highlights
                (34, 139, 34),     # Forest Green - foliage/vegetation
                (205, 133, 63)     # Warm Brown - earth/skin tones
            ],
            "color_names": ["Sky", "Sunlight", "Foliage", "Earth"]
        },
        "winter": {
            "name": "❄️ Winter",
            "colors": [
                (176, 196, 222),   # Light Steel Blue - cold sky
                (248, 248, 255),   # Ghost White - snow/ice
                (112, 128, 144),   # Slate Gray - shadows
                (85, 107, 47)      # Dark Olive - winter foliage
            ],
            "color_names": ["Cold Sky", "Snow", "Shadows", "Pine"]
        },
        "golden_hour": {
            "name": "🌅 Golden Hour",
            "colors": [
                (255, 140, 0),     # Dark Orange - sunset sky
                (255, 215, 0),     # Gold - warm light
                (139, 69, 19),     # Saddle Brown - earth
                (154, 205, 50)     # Yellow Green - lit foliage
            ],
            "color_names": ["Sunset", "Warm Light", "Earth", "Lit Foliage"]
        },
        "overcast": {
            "name": "☁️ Overcast",
            "colors": [
                (192, 192, 192),   # Silver - cloudy sky
                (169, 169, 169),   # Dark Gray - clouds
                (128, 128, 128),   # Gray - neutral tones
                (105, 105, 105)    # Dim Gray - shadows
            ],
            "color_names": ["Cloudy Sky", "Clouds", "Neutral", "Shadows"]
        },
        "night_scene": {
            "name": "🌙 Night",
            "colors": [
                (25, 25, 112),     # Midnight Blue - night sky
                (255, 215, 0),     # Gold - artificial lights
                (0, 0, 0),         # Black - darkness
                (64, 64, 64)       # Dark Gray - shadows
            ],
            "color_names": ["Night Sky", "Lights", "Darkness", "Shadows"]
        },
        "autumn": {
            "name": "🍂 Autumn",
            "colors": [
                (135, 206, 235),   # Sky Blue - clear fall sky
                (255, 140, 0),     # Dark Orange - fall leaves
                (139, 69, 19),     # Saddle Brown - tree bark
                (107, 142, 35)     # Olive Drab - fading leaves
            ],
            "color_names": ["Fall Sky", "Leaves", "Bark", "Fading Green"]
        },
        "spring": {
            "name": "🌸 Spring",
            "colors": [
                (135, 206, 250),   # Light Sky Blue - spring sky
                (144, 238, 144),   # Light Green - fresh grass
                (255, 182, 193),   # Light Pink - blossoms
                (245, 222, 179)    # Wheat - earth/pathways
            ],
            "color_names": ["Spring Sky", "Fresh Grass", "Blossoms", "Earth"]
        },
        "desert": {
            "name": "🏜️ Desert",
            "colors": [
                (0, 191, 255),     # Deep Sky Blue - desert sky
                (244, 164, 96),    # Sandy Brown - sand
                (210, 180, 140),   # Tan - light sand
                (160, 82, 45)      # Sienna - rocks
            ],
            "color_names": ["Desert Sky", "Sand", "Light Sand", "Rocks"]
        },
        "tropical": {
            "name": "🌴 Tropical",
            "colors": [
                (0, 191, 255),     # Deep Sky Blue - tropical sky
                (64, 224, 208),    # Turquoise - water
                (34, 139, 34),     # Forest Green - palm trees
                (255, 69, 0)       # Red Orange - flowers/sunset
            ],
            "color_names": ["Tropical Sky", "Water", "Palms", "Accents"]
        },
        "vintage": {
            "name": "📽️ Vintage",
            "colors": [
                (176, 196, 222),   # Light Steel Blue - faded sky
                (222, 184, 135),   # Burlywood - sepia tones
                (188, 143, 143),   # Rosy Brown - faded colors
                (105, 105, 105)    # Dim Gray - shadows
            ],
            "color_names": ["Faded Sky", "Sepia", "Faded Tones", "Shadows"]
        }
    }
    
    @classmethod
    def get_palette(cls, mood: str) -> Dict:
        """
        Get palette info for a mood.
        
        Args:
            mood: Mood name (e.g., 'sunny_day', 'winter')
            
        Returns:
            Dict with 'name', 'colors', 'color_names'
        """
        mood_key = mood.lower().replace(" ", "_")
        
        if mood_key not in cls.PALETTES:
            logger.warning(f"Unknown mood '{mood}', defaulting to 'sunny_day'")
            mood_key = "sunny_day"
        
        return cls.PALETTES[mood_key]
    
    @classmethod
    def list_moods(cls) -> List[str]:
        """Return list of available mood names."""
        return list(cls.PALETTES.keys())
    
    @classmethod
    def create_palette_image(
        cls,
        mood: str,
        size: Tuple[int, int] = (512, 128),
        blur_radius: int = 0
    ) -> Image.Image:
        """
        Create a visual palette image showing the 4 colors.
        
        Args:
            mood: Mood name
            size: (width, height) of output image
            blur_radius: Gaussian blur radius (0 = no blur)
            
        Returns:
            PIL Image with 4 color stripes
        """
        palette = cls.get_palette(mood)
        colors = palette['colors']
        
        width, height = size
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)
        
        # Draw 4 vertical stripes
        stripe_width = width // 4
        for i, color in enumerate(colors):
            x0 = i * stripe_width
            x1 = (i + 1) * stripe_width if i < 3 else width
            draw.rectangle([x0, 0, x1, height], fill=color)
        
        if blur_radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        return img
    
    @classmethod
    def create_2x2_grid(
        cls,
        mood: str,
        size: int = 512,
        blur_radius: int = 50
    ) -> Image.Image:
        """
        Create a blurred 2x2 grid of the 4 colors.
        This format is ideal for LoRA palette guidance.
        
        Args:
            mood: Mood name
            size: Output square size
            blur_radius: Gaussian blur radius for smooth blending
            
        Returns:
            PIL Image with blurred 2x2 color grid
        """
        palette = cls.get_palette(mood)
        colors = palette['colors']
        
        img = Image.new('RGB', (size, size))
        draw = ImageDraw.Draw(img)
        
        half = size // 2
        
        # Top-left: Color 1 (Sky/Primary)
        draw.rectangle([0, 0, half, half], fill=colors[0])
        
        # Top-right: Color 2 (Accent)
        draw.rectangle([half, 0, size, half], fill=colors[1])
        
        # Bottom-left: Color 3 (Secondary)
        draw.rectangle([0, half, half, size], fill=colors[2])
        
        # Bottom-right: Color 4 (Tertiary)
        draw.rectangle([half, half, size, size], fill=colors[3])
        
        # Apply blur for smooth transitions
        if blur_radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        return img
    
    @classmethod
    def create_conditioning_image(
        cls,
        mood: str,
        size: Tuple[int, int] = (512, 512),
        blur_radius: int = 80
    ) -> Image.Image:
        """
        Create a conditioning image for SD1.5 pipeline.
        Heavily blurred 2x2 grid that provides color hints without structure.
        
        Args:
            mood: Mood name
            size: Output (width, height)
            blur_radius: Strong blur for subtle guidance
            
        Returns:
            PIL Image for pipeline conditioning
        """
        width, height = size
        
        # Create at square size, resize after
        grid = cls.create_2x2_grid(mood, max(width, height), blur_radius)
        
        if width != height:
            grid = grid.resize(size, Image.Resampling.LANCZOS)
        
        return grid
    
    @classmethod
    def get_color_array(cls, mood: str) -> np.ndarray:
        """
        Get palette colors as numpy array.
        
        Args:
            mood: Mood name
            
        Returns:
            (4, 3) uint8 array of RGB colors
        """
        palette = cls.get_palette(mood)
        return np.array(palette['colors'], dtype=np.uint8)
    
    @classmethod  
    def quantize_image(
        cls,
        image: Image.Image,
        mood: str,
        blend_strength: float = 0.3
    ) -> Image.Image:
        """
        Soft-quantize image towards the 4-color palette.
        Useful for post-processing to reinforce palette.
        
        Args:
            image: Input image
            mood: Target mood palette
            blend_strength: How strongly to blend (0-1)
            
        Returns:
            Quantized image
        """
        palette_colors = cls.get_color_array(mood)
        img_np = np.array(image).astype(np.float32)
        
        # For each pixel, find nearest palette color
        h, w, c = img_np.shape
        pixels = img_np.reshape(-1, 3)
        
        # Compute distances to each palette color
        distances = np.zeros((len(pixels), 4))
        for i, color in enumerate(palette_colors.astype(np.float32)):
            distances[:, i] = np.sqrt(np.sum((pixels - color) ** 2, axis=1))
        
        # Find nearest palette color for each pixel
        nearest_idx = np.argmin(distances, axis=1)
        quantized = palette_colors[nearest_idx].astype(np.float32)
        
        # Blend with original
        blended = pixels * (1 - blend_strength) + quantized * blend_strength
        blended = np.clip(blended, 0, 255).astype(np.uint8).reshape(h, w, c)
        
        return Image.fromarray(blended)


# Convenience function
def get_palette_for_mood(mood: str) -> Dict:
    """Get 4-color palette for a mood."""
    return FourColorPaletteGenerator.get_palette(mood)
