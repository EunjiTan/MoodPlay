import os
import requests
from pathlib import Path

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunks
                f.write(chunk)
        print("Download complete.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

def main():
    checkpoint_dir = Path("checkpoints/cotracker")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target_path = checkpoint_dir / "scaled_offline.pth"
    
    url = "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
    
    if target_path.exists():
        print(f"File already exists at {target_path}")
        # optional: verify size or overwrite. For now assuming if it exists it might be partial or corrupt if errors occurred, but likely it doesn't exist.
        # let's overwrite to be safe or check size.
        # simpler: just download.
    
    download_file(url, target_path)

if __name__ == "__main__":
    main()
