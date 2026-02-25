"""
Instance-Guided Diffusion Pipeline
===================================
The central orchestrator that integrates every component into the
per-frame processing loop described in Section 7 of the specification.

Processing loop per frame (F1–F7):
  F1. Detect  — YOLOv11 → persistent IDs
  F2. Segment — SAM-2 box-prompted, overlap-resolved
  F3. Track   — CoTracker joint multi-instance
  F4. Warp    — Temporal colour prior from Frame N-1
  F5. Diffuse — ControlNet-conditioned latent diffusion with LoRA
  F6. Validate — BLS / ICA / TCV checks; palette correction if needed
  F7. Update  — Registry, temporal coherence, metrics
"""

import gc
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from backend.core.pipeline_config import CFG
from backend.core.instance_registry import InstanceRegistry
from backend.core.lab_color_engine import (
    rgb_to_lab, lab_to_rgb, mean_delta_e, correct_instance_color,
)
from backend.core.quality_metrics import (
    FrameMetrics, VideoMetrics,
    compute_ica, compute_tcv, compute_bls, compute_gpc, compute_ssim,
    gpc_passes, build_video_report,
)

# Component imports
from backend.services.yolo_service import YoloService
from backend.models.sam2_segmenter import SAM2Segmenter
from backend.models.cotracker_motion import CoTrackerMotion
from backend.services.controlnet_service import ControlNetService
from backend.services.lora_service import LoRAService
from backend.conditioning.instance_hints import InstanceHintManager
from backend.pipelines.temporal_coherence import TemporalCoherence
from backend.conditioning.four_color_palette import FourColorPaletteGenerator


# ─── Configuration dataclass ──────────────────────────────────────────

@dataclass
class PipelineRunConfig:
    """Per-run settings (can be overridden by the caller)."""
    mood: str = "sunny_day"
    base_prompt: str = ""
    num_inference_steps: int = CFG.NUM_INFERENCE_STEPS
    guidance_scale: float = CFG.GUIDANCE_SCALE
    denoise_strength: float = CFG.DENOISE_STRENGTH
    target_size: Tuple[int, int] = CFG.TARGET_SIZE
    keyframe_interval: int = 5
    seed: Optional[int] = None
    track_motion: bool = True
    segment_every_frame: bool = False
    lora_path: Optional[str] = None
    custom_palette: Optional[Dict[str, List[int]]] = None


# ─── Main pipeline class ──────────────────────────────────────────────

class InstanceGuidedPipeline:
    """
    Orchestrator for Instance-Guided Semantic Video Colorization.

    Usage::

        pipeline = InstanceGuidedPipeline()
        frames_rgb = [...]   # list of (H, W, 3) uint8 RGB arrays
        config = PipelineRunConfig(mood="golden_hour")
        result = pipeline.process_video(frames_rgb, config)
        colorized = result["colorized_frames"]
        report = result["quality_report"]
    """

    def __init__(
        self,
        yolo_model: str = "yolo11n.pt",
        sam_size: str = "tiny",
        cotracker_model: str = "cotracker2",
    ):
        # Registry (created fresh per video)
        self.registry: Optional[InstanceRegistry] = None

        # Components (lazy-loaded)
        self._yolo = YoloService(model_name=yolo_model)
        self._sam = SAM2Segmenter(model_size=sam_size)
        self._tracker = CoTrackerMotion(model_name=cotracker_model)
        self._controlnet = ControlNetService()
        self._lora = LoRAService()
        self._hints = InstanceHintManager()
        self._temporal = TemporalCoherence()
        self._palette_gen = FourColorPaletteGenerator()

        # Diffusion pipeline handle (loaded once during init)
        self._diffusion_pipe = None

        # Runtime state
        self._current_masks: Dict[int, np.ndarray] = {}
        self._current_confs: Dict[int, float] = {}
        self._motion_data: Optional[Dict] = None
        self._frame_reports: List[FrameMetrics] = []

    # ── Public API ─────────────────────────────────────────────────────

    def process_video(
        self,
        frames: List[np.ndarray],
        config: Optional[PipelineRunConfig] = None,
    ) -> Dict:
        """Process an entire video through the instance-guided pipeline.

        Args:
            frames: List of (H, W, 3) uint8 RGB arrays.
            config: Run configuration (defaults used if None).

        Returns:
            dict with keys:
              "colorized_frames": list of (H, W, 3) uint8 RGB,
              "quality_report": VideoMetrics,
              "registry": InstanceRegistry state dict.
        """
        if config is None:
            config = PipelineRunConfig()

        print(f"\n{'='*60}")
        print(f" Instance-Guided Video Colorization")
        print(f" Frames: {len(frames)} | Mood: {config.mood}")
        print(f" Target: {config.target_size} | Steps: {config.num_inference_steps}")
        print(f"{'='*60}\n")

        # ── Initialise ────────────────────────────────────────────────
        self.registry = InstanceRegistry()
        self._temporal.reset()
        self._frame_reports.clear()
        self._current_masks.clear()
        self._current_confs.clear()

        # Build palette
        palette_rgb = self._resolve_palette(config)

        # Pre-compute motion tracks if enabled
        if config.track_motion and len(frames) > 1:
            self._precompute_motion(frames, config)

        # Load diffusion backbone (lazy)
        self._ensure_diffusion_loaded(config)

        # ── Per-frame loop ────────────────────────────────────────────
        colorized_frames: List[np.ndarray] = []

        for idx, frame in enumerate(frames):
            is_keyframe = (idx % config.keyframe_interval == 0) or (idx == 0)
            print(f"\n[Frame {idx}/{len(frames)-1}] {'KEYFRAME' if is_keyframe else 'propagate'}")

            colorized = self._process_single_frame(
                frame, idx, is_keyframe, config, palette_rgb,
            )
            colorized_frames.append(colorized)

        # ── Temporal post-processing ──────────────────────────────────
        print("\n▸ Applying temporal median filter…")
        for i in range(len(colorized_frames)):
            self._temporal._frame_history = colorized_frames[max(0, i - CFG.TEMPORAL_SMOOTHING_WINDOW):i + 1]
            if len(self._temporal._frame_history) >= CFG.TEMPORAL_SMOOTHING_WINDOW:
                colorized_frames[i] = self._temporal.apply_temporal_median_filter(colorized_frames[i])

        # ── Build quality report ──────────────────────────────────────
        report = build_video_report(self._frame_reports)
        print(f"\n{'='*60}")
        print(f" QUALITY REPORT")
        print(f"   Avg ICA (ΔE): {report.avg_ica:.2f}  (target < {CFG.DELTA_E_THRESHOLD})")
        print(f"   Avg TCV:      {report.avg_tcv:.2f}  (target < {CFG.TCV_THRESHOLD})")
        print(f"   Max BLS:      {report.max_bls:.4f}  (target = 0)")
        print(f"   GPC pass:     {report.gpc_pass}")
        print(f"{'='*60}\n")

        # ── Cleanup ───────────────────────────────────────────────────
        self._unload_components()

        return {
            "colorized_frames": colorized_frames,
            "quality_report": report,
            "registry": self.registry.to_dict(),
        }

    # ── Per-frame processing (F1–F7) ──────────────────────────────────

    def _process_single_frame(
        self,
        frame: np.ndarray,
        idx: int,
        is_keyframe: bool,
        config: PipelineRunConfig,
        palette_rgb: Dict[str, List[int]],
    ) -> np.ndarray:
        """Execute the F1–F7 loop for one frame."""

        H, W = frame.shape[:2]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # ── F1: Detect ────────────────────────────────────────────────
        if is_keyframe or config.segment_every_frame:
            detections = self._yolo.detect_and_register(
                frame_bgr, self.registry, idx,
                palette_assigner=lambda label, iid: self._assign_palette_color(
                    label, iid, palette_rgb
                ),
            )
            boxes = {
                d["instance_id"]: np.array(d["box"])
                for d in detections if d.get("instance_id") is not None
            }
        else:
            detections = []
            boxes = {}

        # ── F2: Segment ───────────────────────────────────────────────
        if is_keyframe or config.segment_every_frame:
            if boxes:
                seg_results = self._sam.segment_frame_with_boxes(frame, list(boxes.values()))
                new_masks = {}
                new_confs = {}
                box_ids = list(boxes.keys())
                for i, (mask, score) in enumerate(seg_results):
                    iid = box_ids[i]
                    entry = self.registry.get(iid)
                    label = entry.class_label if entry else ""
                    mask = SAM2Segmenter.dilate_soft_boundaries(mask, label)
                    new_masks[iid] = mask
                    new_confs[iid] = score
                self._current_masks = SAM2Segmenter.resolve_overlaps(new_masks, new_confs)
                self._current_confs = new_confs
            else:
                # Auto-segment fallback
                auto_masks = self._sam.segment_frame(Image.fromarray(frame))
                self._current_masks = {}
                self._current_confs = {}
                for i, m in enumerate(auto_masks[:10]):  # limit to 10
                    self._current_masks[i] = m["segmentation"]
                    self._current_confs[i] = m.get("predicted_iou", 0.8)
        else:
            # Propagate masks from previous frame
            if self._current_masks:
                prop = self._sam.propagate_masks(
                    frame, self._current_masks, self._current_confs, boxes or None,
                )
                for iid, (mask, conf) in prop.items():
                    self._current_masks[iid] = mask
                    self._current_confs[iid] = conf

        # ── F3: Track (motion magnitude gating) ──────────────────────
        motion_triggered_reseg = set()
        if self._motion_data is not None and idx > 0:
            for iid in list(self._current_masks.keys()):
                if self._tracker.check_motion_gate(
                    self._motion_data, idx - 1, min(idx, self._motion_data["tracks"].shape[1] - 1),
                ):
                    motion_triggered_reseg.add(iid)

            # Re-segment instances with large motion
            if motion_triggered_reseg and not is_keyframe:
                reseg_boxes = {iid: boxes[iid] for iid in motion_triggered_reseg if iid in boxes}
                if reseg_boxes:
                    seg_results = self._sam.segment_frame_with_boxes(frame, list(reseg_boxes.values()))
                    for i, (mask, score) in enumerate(seg_results):
                        iid = list(reseg_boxes.keys())[i]
                        self._current_masks[iid] = mask
                        self._current_confs[iid] = score

        # ── F4: Warp colour prior ─────────────────────────────────────
        warped_prior = None
        if self._temporal.prev_colorized is not None:
            warped_prior, _ = self._temporal.get_warped_color_prior(
                frame, self._temporal.prev_colorized,
                self._current_masks,
            )

        # ── F5: Diffuse ───────────────────────────────────────────────
        colorized = self._run_diffusion(
            frame, self._current_masks, config, warped_prior,
        )

        # ── F6: Validate & correct ────────────────────────────────────
        colorized, frame_metrics = self._validate_and_correct(
            colorized, idx, self._current_masks, palette_rgb,
        )

        # ── F7: Update state ──────────────────────────────────────────
        self._update_state(colorized, frame, idx)
        self._frame_reports.append(frame_metrics)

        return colorized

    # ── Diffusion step (F5) ────────────────────────────────────────────

    def _run_diffusion(
        self,
        frame: np.ndarray,
        masks: Dict[int, np.ndarray],
        config: PipelineRunConfig,
        warped_prior: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run conditional latent diffusion with ControlNet + LoRA.

        This is the core colourisation step.  When a full diffusion pipeline
        is loaded, it runs through sd-pipe; otherwise, falls back to the
        palette-correction-only path.
        """
        H, W = frame.shape[:2]

        # Build conditioning images
        frame_pil = Image.fromarray(frame)
        canny = self._controlnet.extract_canny(frame_pil)

        # Create mask boundary conditioning
        boundary_img = self._controlnet.inject_mask_conditioning(masks, (H, W))
        fused_control = self._controlnet.fuse_edge_signals(canny, boundary_img)

        # Build colour hint mask
        palette_rgb_dict = {}
        for iid in masks:
            entry = self.registry.get(iid) if self.registry else None
            if entry is not None:
                palette_rgb_dict[iid] = entry.palette_color_rgb

        hint_mask = InstanceHintManager.generate_color_hint_mask(masks, palette_rgb_dict, (H, W))

        # Blend warped prior with hint mask
        if warped_prior is not None:
            alpha = 0.5
            combined_hint = cv2.addWeighted(hint_mask, alpha, warped_prior, 1 - alpha, 0)
        else:
            combined_hint = hint_mask

        # ── Attempt diffusion pipeline ────────────────────────────────
        if self._diffusion_pipe is not None:
            try:
                seed = config.seed
                generator = torch.Generator(device="cuda").manual_seed(seed) if seed else None

                # Resize inputs
                tw, th = config.target_size
                control_resized = fused_control.resize((tw, th))
                input_image = frame_pil.resize((tw, th))

                result = self._diffusion_pipe(
                    prompt=config.base_prompt or f"photorealistic {config.mood} scene",
                    image=input_image,
                    control_image=control_resized,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    strength=config.denoise_strength,
                    generator=generator,
                ).images[0]

                colorized = np.array(result.resize((W, H)))

                # Apply palette colours via mask overlay
                colorized = self._apply_palette_via_masks(colorized, masks, palette_rgb_dict)

                return colorized

            except Exception as e:
                print(f"  ⚠ Diffusion failed: {e}, using palette-only fallback")

        # ── Palette-only fallback ─────────────────────────────────────
        # When diffusion model is not available, apply palette colours directly
        colorized = self._apply_palette_fallback(frame, masks, palette_rgb_dict, combined_hint)
        return colorized

    def _apply_palette_via_masks(
        self,
        colorized: np.ndarray,
        masks: Dict[int, np.ndarray],
        palette_rgb: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """Blend diffusion output towards palette targets inside each mask."""
        result = colorized.copy()
        lab = rgb_to_lab(result)

        for iid, mask in masks.items():
            if iid not in palette_rgb:
                continue
            target_lab = rgb_to_lab(palette_rgb[iid])
            if mask.shape[:2] != result.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (result.shape[1], result.shape[0])) > 0

            # Soft shift: move ab channels towards target while preserving L
            region_lab = lab[mask].copy()
            blend = 0.6
            region_lab[:, 1] = region_lab[:, 1] * (1 - blend) + target_lab[1] * blend
            region_lab[:, 2] = region_lab[:, 2] * (1 - blend) + target_lab[2] * blend
            lab[mask] = region_lab

        return lab_to_rgb(lab)

    def _apply_palette_fallback(
        self,
        frame: np.ndarray,
        masks: Dict[int, np.ndarray],
        palette_rgb: Dict[int, np.ndarray],
        hint_mask: np.ndarray,
    ) -> np.ndarray:
        """Apply palette colours directly when no diffusion model is available.

        Uses luminance from grayscale + chrominance from palette.
        """
        # Convert frame to LAB
        frame_lab = rgb_to_lab(frame)
        L_channel = frame_lab[..., 0]

        result_lab = frame_lab.copy()

        for iid, mask in masks.items():
            if iid not in palette_rgb:
                continue
            target_lab = rgb_to_lab(palette_rgb[iid])
            if mask.shape[:2] != result_lab.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (result_lab.shape[1], result_lab.shape[0])) > 0

            # Apply palette chrominance, keep luminance
            result_lab[mask, 1] = target_lab[1]
            result_lab[mask, 2] = target_lab[2]

        return lab_to_rgb(result_lab)

    # ── Validation & correction (F6) ───────────────────────────────────

    def _validate_and_correct(
        self,
        colorized: np.ndarray,
        frame_idx: int,
        masks: Dict[int, np.ndarray],
        palette_rgb: Dict[str, List[int]],
    ) -> Tuple[np.ndarray, FrameMetrics]:
        """Run boundary validation and palette correction protocol."""
        fm = FrameMetrics(frame_idx=frame_idx)

        # Build palette LAB map for BLS / GPC
        palette_lab_map: Dict[int, np.ndarray] = {}
        for iid in masks:
            entry = self.registry.get(iid) if self.registry else None
            if entry is not None:
                palette_lab_map[iid] = entry.palette_color_lab

        # ── ICA per instance ──────────────────────────────────────────
        for iid, mask in masks.items():
            target_lab = palette_lab_map.get(iid)
            if target_lab is None:
                continue
            if mask.shape[:2] != colorized.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (colorized.shape[1], colorized.shape[0])) > 0

            ica = compute_ica(colorized, target_lab, mask)
            fm.per_instance_ica[iid] = ica

            # Palette correction if ICA exceeds threshold
            if ica > CFG.DELTA_E_THRESHOLD:
                print(f"  ⚠ Instance {iid} ΔE={ica:.1f} > {CFG.DELTA_E_THRESHOLD}, correcting…")
                colorized, final_de = correct_instance_color(colorized, target_lab, mask)
                fm.per_instance_ica[iid] = final_de

        # ── TCV per instance ──────────────────────────────────────────
        for iid in masks:
            entry = self.registry.get(iid) if self.registry else None
            if entry and len(entry.color_history_lab) >= 2:
                tcv = compute_tcv(entry.color_history_lab)
                fm.per_instance_tcv[iid] = tcv

        # ── BLS ───────────────────────────────────────────────────────
        fm.bls = compute_bls(colorized, masks, palette_lab_map)
        if fm.bls > CFG.BLS_THRESHOLD:
            print(f"  ⚠ BLS={fm.bls:.4f} exceeds threshold, boundary leakage detected")

        # ── GPC ───────────────────────────────────────────────────────
        named_palette_lab = {}
        for name, rgb in palette_rgb.items():
            named_palette_lab[name] = rgb_to_lab(np.array(rgb, dtype=np.uint8))
        fm.gpc = compute_gpc(colorized, named_palette_lab)

        return colorized, fm

    # ── State update (F7) ──────────────────────────────────────────────

    def _update_state(
        self,
        colorized: np.ndarray,
        frame: np.ndarray,
        frame_idx: int,
    ) -> None:
        """Update registry colour history and temporal coherence state."""
        colorized_lab = rgb_to_lab(colorized)

        for iid, mask in self._current_masks.items():
            if mask.shape[:2] != colorized.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8),
                                  (colorized.shape[1], colorized.shape[0])) > 0
            if mask.sum() > 0:
                mean_lab = colorized_lab[mask].mean(axis=0)
                self.registry.confirm_frame(iid, frame_idx, mean_color_lab=mean_lab)

        # Temporal coherence
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        if self._temporal.prev_colorized is not None:
            colorized = self._temporal.apply_temporal_smoothing(colorized, self._temporal.prev_colorized)
        self._temporal.update_frame_history(gray, colorized)

    # ── Helper: palette resolution ─────────────────────────────────────

    def _resolve_palette(self, config: PipelineRunConfig) -> Dict[str, List[int]]:
        """Build the 4-colour palette from mood or custom input."""
        if config.custom_palette:
            return config.custom_palette

        # Use FourColorPaletteGenerator
        try:
            palette_info = self._palette_gen.get_palette(config.mood)
            if palette_info and "colors" in palette_info:
                colors = palette_info["colors"]
                names = palette_info.get("color_names", ["primary", "accent", "secondary", "tertiary"])
                result = {}
                for i, color in enumerate(colors):
                    role = names[i].lower().replace(" ", "_") if i < len(names) else f"color_{i}"
                    result[role] = list(color)
                return result
        except Exception as e:
            print(f"  ⚠ Palette generation failed for mood '{config.mood}': {e}, using default palette")

        # Default palette
        return {
            "background": [200, 210, 220],
            "primary": [120, 80, 60],
            "secondary": [80, 130, 90],
            "accent": [180, 140, 100],
        }

    def _assign_palette_color(
        self,
        class_label: str,
        instance_id: int,
        palette: Dict[str, List[int]],
    ) -> np.ndarray:
        """Assign a palette colour to a newly detected instance.

        Uses a simple round-robin across non-background palette entries.
        """
        keys = [k for k in palette.keys() if k != "background"]
        if not keys:
            return np.array([128, 128, 128], dtype=np.uint8)

        idx = (instance_id - 1) % len(keys)
        color = palette[keys[idx]]
        return np.array(color, dtype=np.uint8)

    # ── Motion pre-computation ─────────────────────────────────────────

    def _precompute_motion(self, frames: List[np.ndarray], config: PipelineRunConfig):
        """Run CoTracker on the full video upfront."""
        print("▸ Pre-computing CoTracker motion tracks…")
        try:
            self._tracker.load_model()
            self._motion_data = self._tracker.track_video(
                frames, grid_size=CFG.COTRACKER_GRID_SIZE,
            )
            print(f"  ✓ Tracked {self._motion_data['num_points']} points "
                  f"across {self._motion_data['num_frames']} frames")
            self._tracker.unload_model()
        except Exception as e:
            print(f"  ⚠ CoTracker failed: {e}, proceeding without motion tracking")
            self._motion_data = None

    # ── Diffusion model management ─────────────────────────────────────

    def _ensure_diffusion_loaded(self, config: PipelineRunConfig):
        """Load the diffusion pipeline (ControlNet + LoRA) if available."""
        if self._diffusion_pipe is not None:
            return

        try:
            from diffusers import (
                StableDiffusionControlNetImg2ImgPipeline,
                ControlNetModel,
                UniPCMultistepScheduler,
            )

            print("▸ Loading diffusion pipeline…")
            controlnet = None
            canny_path = Path("checkpoints/controlnet/canny")
            if canny_path.exists():
                controlnet = ControlNetModel.from_pretrained(
                    str(canny_path), torch_dtype=torch.float16,
                )

            if controlnet:
                self._diffusion_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                ).to("cuda")
            else:
                from diffusers import StableDiffusionImg2ImgPipeline
                self._diffusion_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16,
                    safety_checker=None,
                ).to("cuda")

            self._diffusion_pipe.scheduler = UniPCMultistepScheduler.from_config(
                self._diffusion_pipe.scheduler.config,
            )
            self._diffusion_pipe.enable_attention_slicing()

            # Load LoRA if requested
            if config.lora_path:
                self._lora.load_palette_lora(
                    self._diffusion_pipe, config.mood, config.lora_path,
                )

            print("  ✓ Diffusion pipeline ready")

        except Exception as e:
            print(f"  ⚠ Diffusion pipeline not available: {e}")
            print("    → Will use palette-only colorization fallback")
            self._diffusion_pipe = None

    # ── Cleanup ────────────────────────────────────────────────────────

    def _unload_components(self):
        """Free GPU memory after processing."""
        print("▸ Unloading models…")
        try:
            self._sam.unload_model()
        except Exception as e:
            print(f"  ⚠ SAM unload failed: {e}")

        if self._diffusion_pipe is not None:
            del self._diffusion_pipe
            self._diffusion_pipe = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  ✓ GPU memory cleared")
