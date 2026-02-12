"""
Integration Tests for Instance-Guided Semantic Video Colorization Pipeline.

Tests cover:
  1. Core infrastructure modules (InstanceRegistry, LABColorEngine, PipelineConfig)
  2. Quality metrics (ICA, TCV, BLS, GPC, SSIM)
  3. Enhanced component interfaces
  4. End-to-end pipeline smoke test
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_pipeline_config():
    """Test: PipelineConfig loads and has correct thresholds."""
    from backend.core.pipeline_config import CFG

    assert CFG.DELTA_E_THRESHOLD == 5.0, f"Expected 5.0, got {CFG.DELTA_E_THRESHOLD}"
    assert CFG.TCV_THRESHOLD == 8.0
    assert CFG.BLS_THRESHOLD == 0.001
    assert CFG.GPC_MIN_COVERAGE == 0.005
    assert CFG.CONTROLNET_WEIGHT_EARLY == 0.9
    assert CFG.CONTROLNET_WEIGHT_LATE == 0.4
    assert CFG.LORA_RANK_STANDARD == 16
    assert CFG.LORA_RANK_STYLIZED == 32
    assert isinstance(CFG.TARGET_SIZE, tuple)
    print("  PASS PipelineConfig – all thresholds correct")
    return True


def test_lab_color_engine():
    """Test: RGB↔LAB round-trip and ΔE computation."""
    from backend.core.lab_color_engine import (
        rgb_to_lab, lab_to_rgb, delta_e_cie94, mean_delta_e,
        validate_palette, correct_instance_color,
    )

    # Round-trip
    red = np.array([255, 0, 0], dtype=np.uint8)
    lab = rgb_to_lab(red)
    recovered = lab_to_rgb(lab)
    diff = np.abs(red.astype(int) - recovered.astype(int)).max()
    assert diff < 5, f"Round-trip error too large: {diff}"

    # ΔE of identical colours should be 0
    de = delta_e_cie94(lab.reshape(1, 1, 3), lab.reshape(1, 1, 3))
    assert de.item() < 0.01, f"ΔE of same colour should be ~0, got {de.item()}"

    # ΔE of very different colours should be large
    blue_lab = rgb_to_lab(np.array([0, 0, 255], dtype=np.uint8))
    de2 = delta_e_cie94(lab.reshape(1, 1, 3), blue_lab.reshape(1, 1, 3))
    assert de2.item() > 10, f"ΔE of red vs blue should be large, got {de2.item()}"

    # Image conversion
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img_lab = rgb_to_lab(img)
    assert img_lab.shape == (64, 64, 3)
    img_back = lab_to_rgb(img_lab)
    assert img_back.shape == (64, 64, 3)

    # Palette validation
    palette = {"sky": [135, 206, 235], "grass": [34, 139, 34]}
    palette_lab = validate_palette(palette)
    assert "sky" in palette_lab
    assert palette_lab["sky"].shape == (3,)

    # Palette correction
    output = np.ones((32, 32, 3), dtype=np.uint8) * 128
    target_lab = rgb_to_lab(np.array([255, 0, 0], dtype=np.uint8))
    mask = np.ones((32, 32), dtype=bool)
    corrected, final_de = correct_instance_color(output, target_lab, mask)
    assert corrected.shape == (32, 32, 3)

    print("  PASS LABColorEngine – conversions, ΔE, palette correction all ok")
    return True


def test_instance_registry():
    """Test: InstanceRegistry ID persistence, palette binding, re-entry."""
    from backend.core.instance_registry import InstanceRegistry

    reg = InstanceRegistry()

    # Background entry at 0
    assert 0 in reg.all_ids
    assert reg.get(0).class_label == "background"

    # Register instances
    id1 = reg.register("person", np.array([255, 0, 0]), frame_idx=0)
    id2 = reg.register("car", np.array([0, 255, 0]), frame_idx=0)
    assert id1 == 1
    assert id2 == 2

    # Palette immutability
    assert np.array_equal(reg.get(id1).palette_color_rgb, [255, 0, 0])

    # Confirm frame
    reg.confirm_frame(id1, 5, mean_color_lab=np.array([50.0, 20.0, -10.0]))
    assert reg.get(id1).last_confirmed_frame == 5
    assert len(reg.get(id1).color_history_lab) == 1

    # Deactivate + re-entry
    reg.deactivate(id1)
    assert not reg.get(id1).active

    box = np.array([10, 10, 100, 100], dtype=np.float32)
    reg.get(id1).last_bbox = box
    match = reg.find_reentry_match("person", box)
    assert match == id1

    # Override
    reg.override_palette(id2, np.array([0, 0, 255]))
    assert np.array_equal(reg.get(id2).palette_color_rgb, [0, 0, 255])

    # Serialisation
    state = reg.to_dict()
    assert isinstance(state, dict)

    print("  PASS InstanceRegistry – register, confirm, deactivate, re-entry, override all ok")
    return True


def test_quality_metrics():
    """Test: ICA, TCV, BLS, GPC, SSIM metrics."""
    from backend.core.quality_metrics import (
        compute_ica, compute_tcv, compute_bls, compute_gpc, compute_ssim,
        gpc_passes, FrameMetrics, build_video_report,
    )
    from backend.core.lab_color_engine import rgb_to_lab

    H, W = 64, 64

    # ICA – same colour → ΔE ≈ 0
    solid = np.full((H, W, 3), [255, 0, 0], dtype=np.uint8)
    target = rgb_to_lab(np.array([255, 0, 0], dtype=np.uint8))
    mask = np.ones((H, W), dtype=bool)
    ica = compute_ica(solid, target, mask)
    assert ica < 1.0, f"ICA of identical colour should be ~0, got {ica}"

    # TCV – constant history → 0
    history = [np.array([50, 10, -10])] * 5
    tcv = compute_tcv(history)
    assert tcv == 0.0

    # TCV – varying history → > 0
    history2 = [np.array([50, 10, -10]), np.array([60, 20, 0])]
    tcv2 = compute_tcv(history2)
    assert tcv2 > 0

    # BLS – single instance, all correct → 0
    masks = {1: np.ones((H, W), dtype=bool)}
    pal = {1: target}
    bls = compute_bls(solid, masks, pal)
    assert bls == 0.0, f"Expected BLS=0, got {bls}"

    # GPC
    palette_lab = {"red": target}
    gpc = compute_gpc(solid, palette_lab)
    assert "red" in gpc
    assert gpc["red"] > 0.9  # all pixels are red

    # SSIM
    gray = np.random.randint(0, 255, (H, W), dtype=np.uint8)
    colorized = np.stack([gray] * 3, axis=-1)
    ssim = compute_ssim(gray, colorized)
    assert 0 <= ssim <= 1.0

    # Report builder
    fr = FrameMetrics(frame_idx=0, per_instance_ica={1: 2.0}, bls=0.0, gpc=gpc)
    report = build_video_report([fr])
    assert report.total_frames == 1

    print("  PASS QualityMetrics – ICA, TCV, BLS, GPC, SSIM, report all ok")
    return True


def test_enhanced_sam2():
    """Test: SAM2Segmenter enhanced API exists."""
    from backend.models.sam2_segmenter import SAM2Segmenter

    sam = SAM2Segmenter(model_size="tiny")

    # Check new methods exist
    assert hasattr(sam, "segment_frame_with_boxes")
    assert hasattr(sam, "propagate_masks")
    assert hasattr(sam, "handle_occlusion")
    assert hasattr(sam, "resolve_overlaps")
    assert hasattr(sam, "dilate_soft_boundaries")
    assert hasattr(sam, "clear_memory")

    # Test resolve_overlaps statically
    m1 = np.zeros((10, 10), dtype=bool)
    m2 = np.zeros((10, 10), dtype=bool)
    m1[2:8, 2:8] = True
    m2[5:9, 5:9] = True  # overlaps with m1
    resolved = SAM2Segmenter.resolve_overlaps({1: m1, 2: m2}, {1: 0.9, 2: 0.7})
    overlap = resolved[1] & resolved[2]
    assert not overlap.any(), "Resolved masks should have zero overlap"

    # Test soft boundary dilation
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    dilated = SAM2Segmenter.dilate_soft_boundaries(mask, "person")
    assert dilated.sum() >= mask.sum()  # dilated ≥ original

    # Non-soft class should not dilate
    not_dilated = SAM2Segmenter.dilate_soft_boundaries(mask, "car")
    assert np.array_equal(not_dilated, mask)

    print("  PASS SAM2Segmenter – enhanced API and static methods verified")
    return True


def test_enhanced_controlnet():
    """Test: ControlNetService enhanced API."""
    from backend.services.controlnet_service import ControlNetService

    cs = ControlNetService()

    assert hasattr(cs, "inject_mask_conditioning")
    assert hasattr(cs, "fuse_edge_signals")
    assert hasattr(cs, "get_weight_schedule")
    assert hasattr(cs, "suppress_boundary_leakage")

    # Weight schedule
    w_early = ControlNetService.get_weight_schedule(2, 20)
    w_late = ControlNetService.get_weight_schedule(18, 20)
    assert w_early > w_late, f"Early weight ({w_early}) should exceed late ({w_late})"

    # Mask conditioning
    masks = {1: np.ones((64, 64), dtype=bool)}
    boundary = cs.inject_mask_conditioning(masks, (64, 64))
    assert boundary.size == (64, 64)

    print("  PASS ControlNetService – enhanced API verified")
    return True


def test_enhanced_lora():
    """Test: LoRAService enhanced API."""
    from backend.services.lora_service import LoRAService

    ls = LoRAService()

    assert hasattr(ls, "load_palette_lora")
    assert hasattr(ls, "get_rank_for_style")
    assert hasattr(ls, "validate_temporal_stability")

    # Rank selection
    assert LoRAService.get_rank_for_style("sunny_day") == 16
    assert LoRAService.get_rank_for_style("neon_cyberpunk") == 32

    # Temporal stability (no LoRA loaded → should pass)
    assert ls.validate_temporal_stability()

    print("  PASS LoRAService – enhanced API verified")
    return True


def test_enhanced_instance_hints():
    """Test: InstanceHintManager enhanced API."""
    from backend.conditioning.instance_hints import InstanceHintManager

    ihm = InstanceHintManager()

    ihm.set_palette_color(1, (255, 0, 0))
    lab = ihm.get_palette_lab(1)
    assert lab is not None
    assert lab.shape == (3,)

    # Colour hint mask
    masks = {1: np.ones((32, 32), dtype=bool)}
    palette_rgb = {1: np.array([255, 0, 0], dtype=np.uint8)}
    hint = InstanceHintManager.generate_color_hint_mask(masks, palette_rgb, (32, 32))
    assert hint.shape == (32, 32, 3)
    assert np.all(hint[:, :, 0] == 255)  # all red

    # Warp
    M = np.eye(2, 3, dtype=np.float32)
    M[0, 2] = 5  # translate x
    warped = InstanceHintManager.warp_hint_mask(hint, M)
    assert warped.shape == hint.shape

    print("  PASS InstanceHintManager – enhanced API verified")
    return True


def test_temporal_coherence():
    """Test: TemporalCoherence enhanced API."""
    from backend.pipelines.temporal_coherence import TemporalCoherence

    tc = TemporalCoherence()

    assert hasattr(tc, "compute_tcv")
    assert hasattr(tc, "apply_temporal_median_filter")
    assert hasattr(tc, "get_warped_color_prior")

    # TCV static
    history = [np.array([50, 10, -10]), np.array([55, 12, -8])]
    tcv = TemporalCoherence.compute_tcv(history)
    assert tcv > 0

    # Median filter (needs history)
    frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    tc._frame_history = [frame.copy() for _ in range(5)]
    median = tc.apply_temporal_median_filter(frame)
    assert median.shape == frame.shape

    print("  PASS TemporalCoherence – enhanced API verified")
    return True


def test_pipeline_structure():
    """Test: InstanceGuidedPipeline can be imported and instantiation shape is correct."""
    from backend.pipelines.instance_guided_pipeline import (
        InstanceGuidedPipeline, PipelineRunConfig,
    )

    config = PipelineRunConfig()
    assert config.mood == "sunny_day"
    assert config.num_inference_steps == 20

    # Verify the class has required methods
    assert hasattr(InstanceGuidedPipeline, "process_video")
    assert hasattr(InstanceGuidedPipeline, "_process_single_frame")
    assert hasattr(InstanceGuidedPipeline, "_validate_and_correct")
    assert hasattr(InstanceGuidedPipeline, "_run_diffusion")

    print("  PASS InstanceGuidedPipeline – structure and imports verified")
    return True


# ── Runner ─────────────────────────────────────────────────────────────

def run_all_tests():
    tests = [
        ("PipelineConfig", test_pipeline_config),
        ("LABColorEngine", test_lab_color_engine),
        ("InstanceRegistry", test_instance_registry),
        ("QualityMetrics", test_quality_metrics),
        ("SAM2Segmenter (enhanced)", test_enhanced_sam2),
        ("ControlNetService (enhanced)", test_enhanced_controlnet),
        ("LoRAService (enhanced)", test_enhanced_lora),
        ("InstanceHintManager (enhanced)", test_enhanced_instance_hints),
        ("TemporalCoherence (enhanced)", test_temporal_coherence),
        ("InstanceGuidedPipeline (structure)", test_pipeline_structure),
    ]

    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print(" Instance-Guided Pipeline — Test Suite")
    print("=" * 60)

    for name, fn in tests:
        try:
            result = fn()
            if result:
                passed += 1
            else:
                failed += 1
                errors.append((name, "Returned False"))
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  FAIL {name}: {e}")

    print("\n" + "=" * 60)
    print(f" Results: {passed}/{len(tests)} passed, {failed} failed")
    if errors:
        print("\n Failures:")
        for name, err in errors:
            print(f"   • {name}: {err}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
