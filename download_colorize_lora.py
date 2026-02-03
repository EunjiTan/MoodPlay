"""
Download and test Colorize Slider LoRA from CivitAI
"""

import requests
import os

def download_lora():
    """Download Colorize Slider LoRA from CivitAI."""
    
    # CivitAI model page: https://civitai.com/models/133421/colorize-slider-lora
    # Direct download link (you may need to get this from the page)
    
    print("="*70)
    print("DOWNLOADING COLORIZE SLIDER LORA")
    print("="*70)
    
    # Create lora directory
    os.makedirs("checkpoints/lora", exist_ok=True)
    
    # CivitAI download URL (version 1.0)
    # Note: CivitAI may require authentication or direct link
    lora_url = "https://civitai.com/api/download/models/148933"  # v1.0
    output_path = "checkpoints/lora/colorize_slider_v1.safetensors"
    
    if os.path.exists(output_path):
        print(f"[OK] LoRA already downloaded: {output_path}")
        return output_path
    
    print(f"\nDownloading from CivitAI...")
    print(f"URL: {lora_url}")
    print(f"Output: {output_path}")
    
    try:
        response = requests.get(lora_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end='')
        
        print(f"\n[OK] Downloaded successfully!")
        return output_path
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nManual download instructions:")
        print("1. Go to: https://civitai.com/models/133421/colorize-slider-lora")
        print("2. Click 'Download' button")
        print(f"3. Save file to: {output_path}")
        return None

if __name__ == "__main__":
    lora_path = download_lora()
    
    if lora_path:
        print("\n" + "="*70)
        print("READY TO TEST!")
        print("="*70)
        print(f"\nLoRA saved to: {lora_path}")
        print("\nNext step: Run test_lora_colorization.py")
        print("  python test_lora_colorization.py")
    else:
        print("\nPlease download manually and then run:")
        print("  python test_lora_colorization.py")
