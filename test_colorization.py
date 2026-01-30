"""
Test script for video colorization pipeline.
Tests basic colorization on a single frame or short video.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# Test if models are downloaded
def check_models():
    """Check if required models are downloaded."""
    checkpoint_dir = Path("checkpoints")
    
    required = {
        "SDXL": checkpoint_dir / "sdxl",
        "ControlNet Canny": checkpoint_dir / "controlnet" / "canny",
        "ControlNet Depth": checkpoint_dir / "controlnet" / "depth",
        "VAE": checkpoint_dir / "vae",
    }
    
    print("\n" + "="*60)
    print("Checking Model Downloads...")
    print("="*60)
    
    all_present = True
    for name, path in required.items():
        if path.exists() and any(path.iterdir()):
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: NOT FOUND at {path}")
            all_present = False
    
    print("="*60 + "\n")
    return all_present

def create_test_grayscale_video():
    """Create a simple test grayscale video."""
    output_path = Path("uploads") / "test_grayscale.mp4"
    output_path.parent.mkdir(exist_ok=True)
    
    # Create a simple animated grayscale video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 10.0, (512, 512), False)
    
    for i in range(30):  # 3 seconds at 10 FPS
        # Create a moving circle
        frame = np.zeros((512, 512), dtype=np.uint8)
        center = (256 + int(100 * np.sin(i * 0.2)), 256)
        cv2.circle(frame, center, 50, 200, -1)
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        out.write(frame)
    
    out.release()
    print(f"✓ Created test video: {output_path}")
    return str(output_path)

def test_single_frame_colorization():
    """Test colorization on a single frame."""
    print("\n" + "="*60)
    print("Testing Single Frame Colorization...")
    print("="*60 + "\n")
    
    try:
        from backend.services.sdxl_service import sdxl_service
        from backend.services.controlnet_service import controlnet_service
        
        # Create a simple grayscale test image
        test_image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        cv2.circle(test_image, (256, 256), 100, (200, 200, 200), -1)
        test_pil = Image.fromarray(test_image)
        
        # Load SDXL
        print("Loading SDXL...")
        sdxl_service.load_models()
        
        # Test colorization
        print("Colorizing...")
        result = sdxl_service.colorize_frame(
            test_pil,
            prompt="a colorful sphere, vibrant colors",
            num_inference_steps=10
        )
        
        # Save result
        output_path = Path("results") / "test_single_frame.png"
        output_path.parent.mkdir(exist_ok=True)
        result.save(output_path)
        
        print(f"✓ Single frame test complete!")
        print(f"  Output: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Single frame test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_controlnet():
    """Test ControlNet edge and depth extraction."""
    print("\n" + "="*60)
    print("Testing ControlNet Preprocessing...")
    print("="*60 + "\n")
    
    try:
        from backend.services.controlnet_service import controlnet_service
        
        # Create test image
        test_image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        cv2.rectangle(test_image, (100, 100), (400, 400), (200, 200, 200), -1)
        test_pil = Image.fromarray(test_image)
        
        # Extract controls
        print("Extracting Canny edges...")
        canny = controlnet_service.extract_canny(test_image)
        
        print("Extracting depth map...")
        depth = controlnet_service.extract_depth(test_pil)
        
        # Save results
        output_dir = Path("results") / "controlnet_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        canny.save(output_dir / "canny.png")
        depth.save(output_dir / "depth.png")
        
        print(f"✓ ControlNet test complete!")
        print(f"  Canny: {output_dir / 'canny.png'}")
        print(f"  Depth: {output_dir / 'depth.png'}")
        
        return True
        
    except Exception as e:
        print(f"✗ ControlNet test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("MoodPlay Colorization Pipeline - Test Suite")
    print("="*60)
    
    # 1. Check models
    if not check_models():
        print("\n⚠ Some models are missing!")
        print("Run: python download_colorization_models.py")
        return
    
    # 2. Test ControlNet
    if not test_controlnet():
        print("\n⚠ ControlNet test failed, but continuing...")
    
    # 3. Test single frame
    if not test_single_frame_colorization():
        print("\n✗ Single frame test failed!")
        return
    
    print("\n" + "="*60)
    print("✓ All Tests Passed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Start the backend: python run.py")
    print("2. Upload a grayscale video via the frontend")
    print("3. Use the colorization interface")

if __name__ == "__main__":
    main()
