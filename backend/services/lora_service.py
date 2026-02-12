"""
LoRA Service (Enhanced)
Low-Rank Adaptation for global palette modulation.
Section 3.5: palette-aligned loading, rank configuration, temporal stability.
"""

import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from typing import Dict, List, Optional

from backend.core.pipeline_config import CFG


class LoRAService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded_loras: Dict[str, Dict] = {}

        base_dir = Path(__file__).parent.parent.parent
        self.lora_dir = base_dir / "checkpoints" / "lora"
        self.lora_dir.mkdir(parents=True, exist_ok=True)

        # Track applied LoRA weights for temporal stability enforcement
        self._active_weights_hash: Optional[str] = None

    def download_lora(self, repo_id: str, filename: str = None) -> Optional[Path]:
        """
        Download LoRA weights from HuggingFace.
        """
        try:
            print(f"Downloading LoRA from {repo_id}...")
            lora_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename if filename else "pytorch_lora_weights.safetensors",
                local_dir=str(self.lora_dir / repo_id.replace("/", "_")),
                local_dir_use_symlinks=False,
            )
            print(f"✓ LoRA downloaded to {lora_path}")
            return Path(lora_path)
        except Exception as e:
            print(f"✗ Error downloading LoRA: {e}")
            return None

    def load_lora(
        self,
        pipeline,
        lora_path_or_repo,
        adapter_name: str = "default",
        weight: float = 1.0,
    ) -> bool:
        """Load LoRA weights into a diffusion pipeline."""
        try:
            if isinstance(lora_path_or_repo, (str, Path)) and Path(lora_path_or_repo).exists():
                lora_path = Path(lora_path_or_repo)
            else:
                lora_path = self.download_lora(str(lora_path_or_repo))
                if not lora_path:
                    return False

            print(f"Loading LoRA: {adapter_name} from {lora_path}...")

            pipeline.load_lora_weights(
                str(lora_path.parent),
                weight_name=lora_path.name,
                adapter_name=adapter_name,
            )
            pipeline.set_adapters([adapter_name], adapter_weights=[weight])

            self.loaded_loras[adapter_name] = {
                "path": lora_path,
                "weight": weight,
            }
            self._active_weights_hash = f"{lora_path}:{weight}"

            print(f"✓ LoRA '{adapter_name}' loaded with weight {weight}")
            return True

        except Exception as e:
            print(f"✗ Error loading LoRA: {e}")
            return False

    def unload_lora(self, pipeline, adapter_name: str = "default"):
        """Unload a specific LoRA adapter."""
        try:
            pipeline.delete_adapters(adapter_name)
            if adapter_name in self.loaded_loras:
                del self.loaded_loras[adapter_name]
            print(f"✓ LoRA '{adapter_name}' unloaded")
        except Exception as e:
            print(f"✗ Error unloading LoRA: {e}")

    def set_lora_weight(self, pipeline, adapter_name: str, weight: float):
        """Adjust LoRA weight dynamically."""
        try:
            pipeline.set_adapters([adapter_name], adapter_weights=[weight])
            if adapter_name in self.loaded_loras:
                self.loaded_loras[adapter_name]["weight"] = weight
            print(f"✓ LoRA '{adapter_name}' weight set to {weight}")
        except Exception as e:
            print(f"✗ Error setting LoRA weight: {e}")

    def list_available_loras(self) -> List[str]:
        """List all downloaded LoRAs."""
        loras = []
        if self.lora_dir.exists():
            for lora_file in self.lora_dir.rglob("*.safetensors"):
                loras.append(str(lora_file.relative_to(self.lora_dir)))
        return loras

    # ── NEW: Palette-aligned LoRA loading (Section 3.5) ────────────────

    def load_palette_lora(
        self,
        pipeline,
        style_name: str,
        lora_path_or_repo=None,
        weight: Optional[float] = None,
    ) -> bool:
        """Load LoRA appropriate for the given style/palette.

        Automatically selects rank and weight based on whether the style
        is standard or highly stylised.

        Args:
            pipeline: Diffusion pipeline.
            style_name: Name of the style/palette category.
            lora_path_or_repo: Path or HF repo for LoRA weights.
            weight: Override weight (uses default from config if None).
        """
        if lora_path_or_repo is None:
            print(f"⚠ No LoRA path provided for style '{style_name}', skipping.")
            return False

        rank = self.get_rank_for_style(style_name)
        if weight is None:
            weight = CFG.LORA_DEFAULT_WEIGHT

        print(f"Loading palette LoRA for '{style_name}' (rank={rank}, weight={weight})")
        return self.load_lora(pipeline, lora_path_or_repo, adapter_name=style_name, weight=weight)

    # ── NEW: Rank configuration (Section 3.5) ──────────────────────────

    @staticmethod
    def get_rank_for_style(style_name: str) -> int:
        """Return appropriate LoRA rank for the given style.

        Standard palettes → r=16.
        Highly stylised palettes → r=32.
        """
        if style_name.lower() in CFG.STYLIZED_PALETTES:
            return CFG.LORA_RANK_STYLIZED
        return CFG.LORA_RANK_STANDARD

    # ── NEW: Temporal stability enforcement (Section 3.5) ──────────────

    def validate_temporal_stability(self) -> bool:
        """Confirm that the active LoRA weights have not changed.

        The same LoRA weights MUST be applied identically to every frame.
        This check ensures no accidental weight mutations occurred.
        """
        if not self.loaded_loras:
            return True

        current_hash = "|".join(
            f"{k}:{v['path']}:{v['weight']}" for k, v in sorted(self.loaded_loras.items())
        )
        if self._active_weights_hash is None:
            self._active_weights_hash = current_hash
            return True

        is_stable = current_hash == self._active_weights_hash
        if not is_stable:
            print("⚠ LoRA weights changed between frames! Temporal flicker risk.")
        return is_stable


# Global instance
lora_service = LoRAService()
