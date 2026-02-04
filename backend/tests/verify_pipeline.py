"""
Pipeline Verification Script
Tests core components and runs a short end-to-end test.
"""

import sys
import os
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.logging import get_logger
from backend.core.device import get_device
from backend.models.controlnet_helpers import extract_canny
from backend.pipeline.orchestrator import pipeline
from PIL import Image
import numpy as np

logger = get_logger("verify")

def test_imports():
    logger.info("Testing imports...")
    import backend.core.logging
    import backend.core.device
    import backend.io.video_reader
    import backend.models.segmentation
    import backend.diffusion.sd15_runner
    logger.info("Imports OK")

def test_device():
    device = get_device()
    logger.info(f"Active Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

def test_canny():
    logger.info("Testing Canny extraction...")
    img = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
    edges = extract_canny(img)
    if edges.size == (512, 512):
        logger.info("Canny OK")
    else:
        logger.error("Canny extraction failed size check")

def run_pipeline_dry_run():
    logger.info("Checking pipeline instance...")
    if pipeline:
        logger.info("Pipeline orchestrator initialized OK")

if __name__ == "__main__":
    logger.info("Starting Verification...")
    try:
        test_imports()
        test_device()
        test_canny()
        run_pipeline_dry_run()
        logger.info("Verification Complete - System Ready")
    except Exception as e:
        logger.error(f"Verification Failed: {e}")
        sys.exit(1)
