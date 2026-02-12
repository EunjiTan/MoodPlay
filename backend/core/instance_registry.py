"""
Instance Registry — Persistent per-object identity and palette binding store.

Every tracked object in the video receives a unique Instance ID at first
detection.  The registry guarantees:
  • ID persistence: once assigned, an ID never changes.
  • Palette binding immutability: each ID maps to exactly one palette colour
    for the entire video duration (unless explicitly overridden).
  • Re-entry matching: objects that leave and return are reconciled to their
    original ID.
  • Colour history: a sliding window of per-frame LAB means for temporal
    consistency evaluation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from backend.core.pipeline_config import CFG
from backend.core.lab_color_engine import rgb_to_lab


@dataclass
class InstanceEntry:
    """Single tracked-instance record."""
    instance_id: int
    class_label: str
    palette_color_rgb: np.ndarray          # (3,) uint8
    palette_color_lab: np.ndarray          # (3,) float32
    mask_id: Optional[int] = None
    first_detected_frame: int = 0
    last_confirmed_frame: int = 0
    color_history_lab: List[np.ndarray] = field(default_factory=list)
    last_bbox: Optional[np.ndarray] = None  # (4,) xyxy
    last_appearance_features: Optional[np.ndarray] = None
    active: bool = True


class InstanceRegistry:
    """Thread-safe, video-scoped store for tracked instances."""

    def __init__(self) -> None:
        self._entries: Dict[int, InstanceEntry] = {}
        self._next_id: int = 1  # 0 is reserved for background
        # Background entry
        self._entries[0] = InstanceEntry(
            instance_id=0,
            class_label="background",
            palette_color_rgb=np.array([0, 0, 0], dtype=np.uint8),
            palette_color_lab=rgb_to_lab(np.array([0, 0, 0], dtype=np.uint8)),
        )

    # ── Registration ───────────────────────────────────────────────────

    def register(
        self,
        class_label: str,
        palette_color_rgb: np.ndarray,
        frame_idx: int,
        bbox: Optional[np.ndarray] = None,
        appearance_features: Optional[np.ndarray] = None,
    ) -> int:
        """Create a new instance entry and return its unique ID."""
        iid = self._next_id
        self._next_id += 1
        rgb = np.asarray(palette_color_rgb, dtype=np.uint8)
        entry = InstanceEntry(
            instance_id=iid,
            class_label=class_label,
            palette_color_rgb=rgb,
            palette_color_lab=rgb_to_lab(rgb),
            first_detected_frame=frame_idx,
            last_confirmed_frame=frame_idx,
            last_bbox=bbox,
            last_appearance_features=appearance_features,
        )
        self._entries[iid] = entry
        return iid

    # ── Lookup ─────────────────────────────────────────────────────────

    def get(self, instance_id: int) -> Optional[InstanceEntry]:
        return self._entries.get(instance_id)

    def get_all_active(self) -> Dict[int, InstanceEntry]:
        return {k: v for k, v in self._entries.items() if v.active}

    def get_palette_color_lab(self, instance_id: int) -> Optional[np.ndarray]:
        e = self._entries.get(instance_id)
        return e.palette_color_lab if e else None

    @property
    def all_ids(self) -> List[int]:
        return list(self._entries.keys())

    @property
    def active_ids(self) -> List[int]:
        return [k for k, v in self._entries.items() if v.active]

    # ── Update ─────────────────────────────────────────────────────────

    def confirm_frame(
        self,
        instance_id: int,
        frame_idx: int,
        mean_color_lab: Optional[np.ndarray] = None,
        bbox: Optional[np.ndarray] = None,
    ) -> None:
        """Mark an instance as seen in *frame_idx* and optionally log colour."""
        e = self._entries.get(instance_id)
        if e is None:
            return
        e.last_confirmed_frame = frame_idx
        if bbox is not None:
            e.last_bbox = bbox
        if mean_color_lab is not None:
            e.color_history_lab.append(mean_color_lab.copy())
            # Keep sliding window
            max_hist = max(CFG.TCV_WINDOW_SIZE, 30)
            if len(e.color_history_lab) > max_hist:
                e.color_history_lab = e.color_history_lab[-max_hist:]

    def deactivate(self, instance_id: int) -> None:
        e = self._entries.get(instance_id)
        if e:
            e.active = False

    def reactivate(self, instance_id: int, frame_idx: int) -> None:
        e = self._entries.get(instance_id)
        if e:
            e.active = True
            e.last_confirmed_frame = frame_idx

    # ── Re-entry Matching ──────────────────────────────────────────────

    def find_reentry_match(
        self,
        class_label: str,
        bbox: np.ndarray,
        appearance_features: Optional[np.ndarray] = None,
    ) -> Optional[int]:
        """Try to match a new detection with a previously deactivated instance.

        Uses class-label + bounding-box IoU + optional appearance similarity.
        Returns matched instance_id or None.
        """
        best_id: Optional[int] = None
        best_score: float = 0.0

        for iid, entry in self._entries.items():
            if iid == 0:
                continue
            if entry.active:
                continue
            if entry.class_label != class_label:
                continue
            if entry.last_bbox is None:
                continue

            iou = self._bbox_iou(entry.last_bbox, bbox)
            score = iou

            # Boost with appearance similarity when available
            if (appearance_features is not None
                    and entry.last_appearance_features is not None):
                cos_sim = np.dot(appearance_features, entry.last_appearance_features) / (
                    np.linalg.norm(appearance_features)
                    * np.linalg.norm(entry.last_appearance_features)
                    + 1e-8
                )
                score = 0.5 * iou + 0.5 * max(cos_sim, 0.0)

            if score > best_score and score > CFG.REENTRY_IOU_THRESHOLD:
                best_score = score
                best_id = iid

        return best_id

    # ── Palette Override ───────────────────────────────────────────────

    def override_palette(
        self, instance_id: int, new_color_rgb: np.ndarray
    ) -> None:
        """Explicitly reassign palette colour (user-initiated only)."""
        e = self._entries.get(instance_id)
        if e is None:
            return
        rgb = np.asarray(new_color_rgb, dtype=np.uint8)
        e.palette_color_rgb = rgb
        e.palette_color_lab = rgb_to_lab(rgb)
        e.color_history_lab.clear()

    # ── Background Palette ─────────────────────────────────────────────

    def set_background_color(self, color_rgb: np.ndarray) -> None:
        rgb = np.asarray(color_rgb, dtype=np.uint8)
        bg = self._entries[0]
        bg.palette_color_rgb = rgb
        bg.palette_color_lab = rgb_to_lab(rgb)

    # ── Serialisation helpers ──────────────────────────────────────────

    def to_dict(self) -> Dict:
        """Export registry state for the quality report."""
        out = {}
        for iid, e in self._entries.items():
            out[iid] = {
                "class_label": e.class_label,
                "palette_rgb": e.palette_color_rgb.tolist(),
                "first_frame": e.first_detected_frame,
                "last_frame": e.last_confirmed_frame,
                "color_history_len": len(e.color_history_lab),
                "active": e.active,
            }
        return out

    # ── Internal ───────────────────────────────────────────────────────

    @staticmethod
    def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Compute intersection-over-union for two [x1,y1,x2,y2] boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return (f"InstanceRegistry(total={len(self._entries)}, "
                f"active={len(self.active_ids)})")
