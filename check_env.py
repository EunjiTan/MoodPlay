import sys
import os

print(f"Python: {sys.version}")
try:
    import torch
    print(f"Torch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Check for the attribute causing issues
    print(f"Has torch.backends.flash_attention: {hasattr(torch.backends, 'flash_attention')}")
    print(f"Has torch.backends.cuda.flash_sdp_enabled: {hasattr(torch.backends.cuda, 'flash_sdp_enabled')}")

except ImportError as e:
    print(f"Torch Import Failed: {e}")

try:
    import diffusers
    print(f"Diffusers: {diffusers.__version__}")
except ImportError:
    print("Diffusers not installed")

try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except ImportError:
    print("Transformers not installed")

try:
    import accelerate
    print(f"Accelerate: {accelerate.__version__}")
except ImportError:
    print("Accelerate not installed")
