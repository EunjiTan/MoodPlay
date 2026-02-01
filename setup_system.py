import subprocess
import sys
import os

def run_command(command):
    print(f"\nExecuting: {command}")
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        # Build environments are messy, sometimes uninstall fails if not installed. Continue.
        pass

def main():
    print("="*60)
    print("MOODPLAY SYSTEM REPAIR - CLEAN INSTALL")
    print("="*60)
    
    # 1. Uninstall EVERYTHING related to AI (Deep Clean)
    print("\n[1/3] Cleaning existing libraries...")
    packages = [
        "torch", "torchvision", "torchaudio", 
        "diffusers", "transformers", "accelerate", 
        "huggingface_hub", "safetensors", "xformers"
    ]
    run_command(f"{sys.executable} -m pip uninstall -y {' '.join(packages)}")
    
    # 2. Install PyTorch with CUDA 12.1 (Crucial Step)
    print("\n[2/3] Installing PyTorch (CUDA 12.1)...")
    # This must be done FIRST and ALONE to avoid pip picking CPU versions
    run_command(f"{sys.executable} -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121")
    
    # 3. Install AI Logic Libraries (Stable Versions)
    print("\n[3/3] Installing compatible AI libraries...")
    # Known working combination for SDXL
    libs = [
        "diffusers==0.27.2",       # Stable for SDXL
        "transformers==4.40.2",    # Matches diffusers
        "accelerate==0.30.1",      # Required for GPU offloading
        "huggingface_hub==0.23.0", # Fixes dynamic_module import error
        "safetensors",
        "opencv-python",
        "pillow",
        "numpy<2.0" # SDXL sometimes hates numpy 2.x
    ]
    run_command(f"{sys.executable} -m pip install {' '.join(libs)}")
    
    # 4. Final Verification
    print("\n[Verification] Checking CUDA...")
    verification_script = """
import torch
print(f'Torch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('❌ ERROR: CUDA NOT AVAILABLE - REINSTALL FAILED')
    exit(1)
"""
    with open("verify_install.py", "w") as f:
        f.write(verification_script)
        
    try:
        subprocess.check_call([sys.executable, "verify_install.py"])
        print("\n✅ SYSTEM REPAIR COMPLETE! You can now run 'python run.py'")
    except:
        print("\n❌ Verification Failed.")

if __name__ == "__main__":
    main()
