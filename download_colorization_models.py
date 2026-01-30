"""
Download pre-trained models for video colorization pipeline.
Downloads: SDXL 1.0, ControlNext SDXL, and colorization LoRA weights.
"""

import os
from huggingface_hub import snapshot_download, hf_hub_download
from pathlib import Path

# Base directory for checkpoints
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Model configurations
MODELS = {
    "sdxl": {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "local_dir": CHECKPOINT_DIR / "sdxl",
        "allow_patterns": ["*.safetensors", "*.json", "*.txt"],
    },
    "controlnet_canny": {
        "repo_id": "diffusers/controlnet-canny-sdxl-1.0",
        "local_dir": CHECKPOINT_DIR / "controlnet" / "canny",
    },
    "controlnet_depth": {
        "repo_id": "diffusers/controlnet-depth-sdxl-1.0",
        "local_dir": CHECKPOINT_DIR / "controlnet" / "depth",
    },
    "vae": {
        "repo_id": "madebyollin/sdxl-vae-fp16-fix",
        "local_dir": CHECKPOINT_DIR / "vae",
    },
}

def download_model(name, config):
    """Download a model from HuggingFace Hub."""
    print(f"\n{'='*60}")
    print(f"Downloading {name.upper()}...")
    print(f"Repository: {config['repo_id']}")
    print(f"Destination: {config['local_dir']}")
    print(f"{'='*60}\n")
    
    try:
        local_dir = config["local_dir"]
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        if local_dir.exists() and any(local_dir.iterdir()):
            print(f"✓ {name} already exists at {local_dir}")
            return
        
        # Download with optional patterns
        snapshot_download(
            repo_id=config["repo_id"],
            local_dir=str(local_dir),
            allow_patterns=config.get("allow_patterns"),
            resume_download=True,
        )
        
        print(f"✓ {name} downloaded successfully!")
        
    except Exception as e:
        print(f"✗ Error downloading {name}: {e}")
        raise

def main():
    print("\n" + "="*60)
    print("MoodPlay Colorization Models Downloader")
    print("="*60)
    print("\nThis will download ~15-20GB of models:")
    print("  • SDXL 1.0 Base (~6.9GB)")
    print("  • ControlNet Canny SDXL (~2.5GB)")
    print("  • ControlNet Depth SDXL (~2.5GB)")
    print("  • VAE FP16 Fix (~335MB)")
    print("\nEnsure you have sufficient disk space and internet bandwidth.")
    print("="*60 + "\n")
    
    # Download each model
    for name, config in MODELS.items():
        download_model(name, config)
    
    print("\n" + "="*60)
    print("✓ All models downloaded successfully!")
    print("="*60)
    print("\nModels saved to:")
    for name, config in MODELS.items():
        print(f"  • {name}: {config['local_dir']}")
    print("\nYou can now run the colorization pipeline!")

if __name__ == "__main__":
    main()
