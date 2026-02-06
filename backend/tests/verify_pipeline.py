"""
Pipeline Verification Script
Tests core components and runs a short end-to-end test.
Updated for SAM-3, CoTracker, 4-Color LoRA pipeline.
"""

import sys
import os
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.logging import get_logger
from backend.core.device import get_device
from PIL import Image
import numpy as np

logger = get_logger("verify")


def test_imports():
    """Test that all new modules can be imported."""
    logger.info("Testing imports...")
    
    # Core modules
    import backend.core.logging
    import backend.core.device
    import backend.io.video_reader
    
    # Updated model modules
    from backend.models.sam3_segmenter import SAM3Segmenter
    from backend.models.cotracker_motion import CoTrackerMotion
    
    # Updated diffusion module
    from backend.diffusion.sd15_runner import SD15Runner
    
    # New palette system
    from backend.conditioning.four_color_palette import FourColorPaletteGenerator
    from backend.conditioning.style_conditioning import style_manager
    
    # Updated pipeline
    from backend.pipelines.updated_colorization_pipeline import UpdatedColorizationPipeline
    
    logger.info("All imports OK")


def test_device():
    """Test device detection."""
    device = get_device()
    logger.info(f"Active Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


def test_canny():
    """Test Canny extraction."""
    logger.info("Testing Canny extraction...")
    from backend.models.controlnet_helpers import extract_canny
    
    img = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
    edges = extract_canny(img)
    if edges.size == (512, 512):
        logger.info("Canny OK")
    else:
        logger.error(f"Canny extraction failed size check: {edges.size}")


def test_palette_system():
    """Test 4-color palette generation."""
    logger.info("Testing 4-Color Palette System...")
    from backend.conditioning.four_color_palette import FourColorPaletteGenerator
    from backend.conditioning.style_conditioning import style_manager
    
    # Test palette retrieval
    palette = FourColorPaletteGenerator.get_palette("sunny_day")
    assert len(palette['colors']) == 4, "Should have 4 colors"
    logger.info(f"Sunny Day palette: {palette['name']}")
    
    # Test palette image generation
    img = FourColorPaletteGenerator.create_2x2_grid("winter", size=128)
    assert img.size == (128, 128), "Should be 128x128"
    logger.info("2x2 Grid generation OK")
    
    # Test style manager integration
    styles = style_manager.list_styles()
    assert len(styles) >= 10, "Should have at least 10 styles"
    logger.info(f"Available styles: {len(styles)}")
    
    config = style_manager.get_style_config("golden_hour")
    assert config['lora_scale'] == 0.5, "LoRA scale should be 0.5"
    logger.info("Style Manager integration OK")
    
    logger.info("4-Color Palette System OK")


def test_sam2_module():
    """Test SAM-2 module structure (without loading model)."""
    logger.info("Testing SAM-2 module structure...")
    from backend.models.sam2_segmenter import SAM2Segmenter
    
    segmenter = SAM2Segmenter(model_size="tiny")
    assert segmenter.model_size == "tiny"
    assert segmenter.model is None  # Should not auto-load
    logger.info("SAM-2 module structure OK")


def test_cotracker_module():
    """Test CoTracker module structure (without loading model)."""
    logger.info("Testing CoTracker module structure...")
    from backend.models.cotracker_motion import CoTrackerMotion
    
    tracker = CoTrackerMotion(model_name="cotracker2")
    assert tracker.model_name == "cotracker2"
    assert tracker.model is None  # Should not auto-load
    logger.info("CoTracker module structure OK")


def test_sd15_runner():
    """Test SD1.5 runner configuration (without loading model)."""
    logger.info("Testing SD1.5 Runner configuration...")
    from backend.diffusion.sd15_runner import SD15Runner
    
    runner = SD15Runner()
    assert runner.pipe is None  # Should not auto-load
    logger.info("SD1.5 Runner configuration OK")


def test_updated_pipeline():
    """Test updated pipeline initialization."""
    logger.info("Testing Updated Pipeline initialization...")
    from backend.pipelines.updated_colorization_pipeline import UpdatedColorizationPipeline
    
    pipeline = UpdatedColorizationPipeline(
        sam2_model_size="tiny",
        cotracker_model="cotracker2",
        lora_path=None
    )
    
    # Test mood listing
    moods = pipeline.list_available_moods()
    assert len(moods) >= 10
    logger.info(f"Available moods: {moods}")
    
    # Test palette preview
    preview = pipeline.get_palette_preview("autumn")
    assert preview.size[0] > 0 and preview.size[1] > 0
    logger.info("Palette preview OK")
    
    pipeline.cleanup()
    logger.info("Updated Pipeline initialization OK")


def run_full_verification():
    """Run all verification tests."""
    tests = [
        ("Imports", test_imports),
        ("Device", test_device),
        ("Canny", test_canny),
        ("Palette System", test_palette_system),
        ("SAM-2 Module", test_sam2_module),
        ("CoTracker Module", test_cotracker_module),
        ("SD1.5 Runner", test_sd15_runner),
        ("Updated Pipeline", test_updated_pipeline),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            logger.error(f"Test '{name}' FAILED: {e}")
            failed += 1
    
    logger.info("=" * 50)
    logger.info(f"Verification Complete: {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    logger.info("Starting Verification for Updated Pipeline...")
    logger.info("=" * 50)
    
    try:
        success = run_full_verification()
        if success:
            logger.info("✓ All tests passed - System Ready")
            sys.exit(0)
        else:
            logger.error("✗ Some tests failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Verification Failed: {e}")
        sys.exit(1)
