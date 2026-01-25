import os

class LoRAService:
    def __init__(self):
        # Use absolute paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.lora_dir = os.path.join(self.base_dir, "checkpoints/lora_styles")
        os.makedirs(self.lora_dir, exist_ok=True)
        
        self.styles = {
            "anime": "anime_style.safetensors",
            "cyberpunk": "cyberpunk.safetensors",
            "vintage": "vintage.safetensors"
        }

    def list_styles(self):
        """List available style presets."""
        available = []
        for name, filename in self.styles.items():
            path = os.path.join(self.lora_dir, filename)
            status = "available" if os.path.exists(path) else "missing"
            available.append({"name": name, "status": status, "filename": filename})
        return available

    def get_lora_path(self, style_name):
        """Get absolute path for a style."""
        if style_name not in self.styles:
            return None
        return os.path.join(self.lora_dir, self.styles[style_name])

# Global Instance
lora_service = LoRAService()
