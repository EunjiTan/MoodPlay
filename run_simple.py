"""
MoodPlay - Simple Local Runner (Works without ML dependencies)
Uses mock/fallback mode for all processing.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set environment for local running without Redis
os.environ['CELERY_TASK_ALWAYS_EAGER'] = '1'
os.environ['CELERY_TASK_EAGER_PROPAGATES'] = '1'

print("=" * 60)
print("MoodPlay Video Colorization Pipeline")
print("=" * 60)
print()

# Check dependencies with graceful handling
print("Checking dependencies...")

# Check PyTorch
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"    └ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("    └ CUDA not available (using CPU)")
except Exception as e:
    print(f"  ⚠ PyTorch not working: {type(e).__name__}")
    print("    └ Running in mock/demo mode")

# Check Diffusers
try:
    import diffusers
    print(f"  ✓ Diffusers {diffusers.__version__}")
except ImportError:
    print("  ⚠ Diffusers not installed (mock mode)")

# Check OpenCV
try:
    import cv2
    print(f"  ✓ OpenCV {cv2.__version__}")
except ImportError:
    print("  ⚠ OpenCV not installed")

# Check Flask
try:
    import flask
    print(f"  ✓ Flask {flask.__version__}")
except ImportError:
    print("  ✗ Flask not installed - required!")
    sys.exit(1)

print()

# Patch Celery to run synchronously without Redis
try:
    from backend.workers.celery_config import celery_app
    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
        broker_url='memory://',
        result_backend='cache+memory://',
    )
    print("✓ Configured for synchronous processing (no Redis needed)")
except Exception as e:
    print(f"Note: Celery config: {e}")

print()
print("Starting server...")
print("=" * 60)

# Import and run Flask
from backend.app import app

if __name__ == '__main__':
    print()
    print("=" * 60)
    print("Server running at: http://localhost:5000")
    print("=" * 60)
    print()
    print("Open your browser to http://localhost:5000 to use MoodPlay!")
    print()
    print("Note: Without PyTorch, processing will use mock/demo output.")
    print("      This proves the pipeline works - full processing needs")
    print("      PyTorch + CUDA GPU drivers installed correctly.")
    print()
    print("Press Ctrl+C to stop the server.")
    print()
    
    app.run(debug=True, port=5000, host='0.0.0.0', use_reloader=False)
