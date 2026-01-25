import os
import urllib.request
from tqdm import tqdm

CHECKPOINT_DIR = "checkpoints/sam2"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODELS = {
    "sam2_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
    "sam2_hiera_large.yaml": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2/sam2_hiera_l.yaml"
}

def download_file(url, filename):
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(filepath):
        print(f"File already exists: {filename}")
        return

    print(f"Downloading {filename}...")
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
        def reporthook(blocknum, blocksize, totalsize):
            t.total = totalsize
            t.update(blocknum * blocksize - t.n)
            
        urllib.request.urlretrieve(url, filepath, reporthook)

if __name__ == "__main__":
    print("Downloading SAM-2 Models...")
    for filename, url in MODELS.items():
        download_file(url, filename)
    print("Download Complete.")
