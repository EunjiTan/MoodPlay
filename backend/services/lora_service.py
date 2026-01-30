"""
LoRA Service for SDXL Fine-Tuning
Loads and applies LoRA weights for style adaptation and colorization.
"""

import torch
from pathlib import Path
from huggingface_hub import hf_hub_download

class LoRAService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded_loras = {}
        
        # Paths
        base_dir = Path(__file__).parent.parent.parent
        self.lora_dir = base_dir / "checkpoints" / "lora"
        self.lora_dir.mkdir(parents=True, exist_ok=True)
    
    def download_lora(self, repo_id, filename=None):
        """
        Download LoRA weights from HuggingFace.
        Args:
            repo_id (str): HuggingFace repo ID
            filename (str): Specific file to download (optional)
        Returns:
            Path: Path to downloaded LoRA
        """
        try:
            print(f"Downloading LoRA from {repo_id}...")
            
            lora_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename if filename else "pytorch_lora_weights.safetensors",
                local_dir=str(self.lora_dir / repo_id.replace("/", "_")),
                local_dir_use_symlinks=False,
            )
            
            print(f"✓ LoRA downloaded to {lora_path}")
            return Path(lora_path)
            
        except Exception as e:
            print(f"✗ Error downloading LoRA: {e}")
            return None
    
    def load_lora(self, pipeline, lora_path_or_repo, adapter_name="default", weight=1.0):
        """
        Load LoRA weights into SDXL pipeline.
        Args:
            pipeline: SDXL pipeline
            lora_path_or_repo (str or Path): Local path or HuggingFace repo
            adapter_name (str): Name for this LoRA adapter
            weight (float): LoRA weight/strength
        """
        try:
            # Check if it's a local path or repo ID
            if isinstance(lora_path_or_repo, (str, Path)) and Path(lora_path_or_repo).exists():
                lora_path = Path(lora_path_or_repo)
            else:
                # Download from HuggingFace
                lora_path = self.download_lora(lora_path_or_repo)
                if not lora_path:
                    return False
            
            print(f"Loading LoRA: {adapter_name} from {lora_path}...")
            
            # Load LoRA weights
            pipeline.load_lora_weights(
                str(lora_path.parent),
                weight_name=lora_path.name,
                adapter_name=adapter_name,
            )
            
            # Set adapter weight
            pipeline.set_adapters([adapter_name], adapter_weights=[weight])
            
            self.loaded_loras[adapter_name] = {
                "path": lora_path,
                "weight": weight
            }
            
            print(f"✓ LoRA '{adapter_name}' loaded with weight {weight}")
            return True
            
        except Exception as e:
            print(f"✗ Error loading LoRA: {e}")
            return False
    
    def unload_lora(self, pipeline, adapter_name="default"):
        """Unload a specific LoRA adapter."""
        try:
            pipeline.delete_adapters(adapter_name)
            if adapter_name in self.loaded_loras:
                del self.loaded_loras[adapter_name]
            print(f"✓ LoRA '{adapter_name}' unloaded")
        except Exception as e:
            print(f"✗ Error unloading LoRA: {e}")
    
    def set_lora_weight(self, pipeline, adapter_name, weight):
        """Adjust LoRA weight dynamically."""
        try:
            pipeline.set_adapters([adapter_name], adapter_weights=[weight])
            if adapter_name in self.loaded_loras:
                self.loaded_loras[adapter_name]["weight"] = weight
            print(f"✓ LoRA '{adapter_name}' weight set to {weight}")
        except Exception as e:
            print(f"✗ Error setting LoRA weight: {e}")
    
    def list_available_loras(self):
        """List all downloaded LoRAs."""
        loras = []
        if self.lora_dir.exists():
            for lora_file in self.lora_dir.rglob("*.safetensors"):
                loras.append(str(lora_file.relative_to(self.lora_dir)))
        return loras

# Global instance
lora_service = LoRAService()
