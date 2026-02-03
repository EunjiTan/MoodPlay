import requests
import os

def download_file(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Done!")

os.makedirs("checkpoints/sam2", exist_ok=True)

# tiny model weights
weights_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
weights_path = "checkpoints/sam2/sam2_hiera_tiny.pt"

# tiny model config
config_url = "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2_configs/sam2_hiera_t.yaml"
config_path = "checkpoints/sam2/sam2_hiera_tiny.yaml" 

download_file(weights_url, weights_path)
download_file(config_url, config_path)
