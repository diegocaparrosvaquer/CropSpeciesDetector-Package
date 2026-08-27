"""High-level hierarchical inference pipeline.

Stage 1 -> cropland / no cropland
Stage 2 -> crop species (only run when Stage 1 says "cropland")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Union

import torch
from PIL import Image
from torchvision import transforms

from .constants import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    STAGE1_CLASSES,
    STAGE2_CLASSES,
)
from .model import DINOv2Classifier
from .checkpoints import get_checkpoint_path

ImageInput = Union[str, "os.PathLike[str]", Image.Image]

# A real checkpoint is on the order of hundreds of MB. If the file is only
# a few KB, Git LFS did not actually download the binary content and what
# we have on disk is an LFS pointer file instead.
_LFS_POINTER_SIZE_THRESHOLD_MB = 1.0


class PredictionResult(TypedDict):
    stage1_prediction: str
    stage1_confidence: float
    stage2_prediction: Optional[str]
    stage2_confidence: Optional[float]
    final_prediction: str
    final_confidence: float
    stage2_used: bool
    stage1_probs: Dict[str, float]
    stage2_probs: Optional[Dict[str, float]]


def _check_checkpoint_file(path: Union[str, "os.PathLike[str]"], label: str) -> float:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{label} checkpoint not found.\nExpected file at: {path}"
        )

    size_mb = path.stat().st_size / 1024**2

    if size_mb < _LFS_POINTER_SIZE_THRESHOLD_MB:
        raise RuntimeError(
            f"{label} checkpoint at '{path}' is only {size_mb:.3f} MB, "
            f"which is far too small to be a real model checkpoint.\n"
            f"This usually means Git LFS did not download the actual file "
            f"and you are looking at an LFS pointer file instead.\n\n"
            f"Fix: run the following in the repository directory:\n"
            f"    git lfs install\n"
            f"    git lfs pull"
        )

    return size_mb


def _verify_classes(actual_classes, expected_classes, label: str) -> None:
    actual_set = set(actual_classes)
    if actual_set != set(expected_classes):
        missing = set(expected_classes) - actual_set
        unexpected = actual_set - set(expected_classes)
        raise ValueError(
            f"{label} checkpoint classes do not match the expected classes.\n"
            f"Missing: {sorted(missing) if missing else 'none'}\n"
            f"Unexpected: {sorted(unexpected) if unexpected else 'none'}"
        )


class CropSpeciesDetector:
    """Runs the Stage 1 -> Stage 2 hierarchical inference pipeline.

    Prefer :meth:`from_checkpoints` over calling the constructor directly.
    """

    def __init__(
        self,
        stage1_model: DINOv2Classifier,
        stage2_model: DINOv2Classifier,
        stage1_idx_to_class: Dict[int, str],
        stage2_idx_to_class: Dict[int, str],
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.stage1_model = stage1_model.to(self.device).eval()
        self.stage2_model = stage2_model.to(self.device).eval()

        self.stage1_idx_to_class = stage1_idx_to_class
        self.stage2_idx_to_class = stage2_idx_to_class

        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_checkpoints(
        cls,
        stage1_path: Union[str, "os.PathLike[str]"],
        stage2_path: Union[str, "os.PathLike[str]"],
        device: Optional[Union[str, torch.device]] = None,
        verify_classes: bool = True,
    ) -> "CropSpeciesDetector":
        """Load Stage 1 and Stage 2 checkpoints and build the detector.

        Parameters
        ----------
        stage1_path, stage2_path:
            Paths to ``stage1_best.pt`` / ``stage2_best.pt`` (or
            equivalently-shaped checkpoints containing
            ``model_state_dict``, ``classes`` and ``class_to_idx``).
        device:
            ``"cuda"``, ``"cpu"``, or a ``torch.device``. Defaults to CUDA
            if available.
        verify_classes:
            If True, raise if the checkpoint's classes don't match the
            expected Stage 1 / Stage 2 label sets.
        """
        _check_checkpoint_file(stage1_path, "Stage 1")
        _check_checkpoint_file(stage2_path, "Stage 2")

        stage1_checkpoint = torch.load(stage1_path, map_location="cpu")
        stage2_checkpoint = torch.load(stage2_path, map_location="cpu")

        stage1_classes = stage1_checkpoint["classes"]
        stage2_classes = stage2_checkpoint["classes"]

        if verify_classes:
            _verify_classes(stage1_classes, STAGE1_CLASSES, "Stage 1")
            _verify_classes(stage2_classes, STAGE2_CLASSES, "Stage 2")

        stage1_idx_to_class = {
            int(idx): name for name, idx in stage1_checkpoint["class_to_idx"].items()
        }
        stage2_idx_to_class = {
            int(idx): name for name, idx in stage2_checkpoint["class_to_idx"].items()
        }

        stage1_model = DINOv2Classifier(num_classes=len(stage1_classes))
        stage2_model = DINOv2Classifier(num_classes=len(stage2_classes))

        cls._load_state_dict_strict(stage1_model, stage1_checkpoint, "Stage 1")
        cls._load_state_dict_strict(stage2_model, stage2_checkpoint, "Stage 2")

        return cls(
            stage1_model,
            stage2_model,
            stage1_idx_to_class,
            stage2_idx_to_class,
            device=device,
        )
    @classmethod
    def from_pretrained(cls, stage1_path: str | None = None, stage2_path: str | None = None, **kwargs):
        """Like from_checkpoints, but auto-downloads from the Hub if a path isn't given."""
        stage1_path = stage1_path or get_checkpoint_path("stage1")
        stage2_path = stage2_path or get_checkpoint_path("stage2")
        return cls.from_checkpoints(stage1_path, stage2_path, **kwargs)
    @staticmethod
    def _load_state_dict_strict(model, checkpoint, label: str) -> None:
        try:
            result = model.load_state_dict(checkpoint["model_state_dict"])
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise RuntimeError(
                f"Failed to load {label} checkpoint into DINOv2Classifier.\n"
                f"The checkpoint does not match the model architecture.\n\n"
                f"Underlying error:\n{exc}"
            ) from exc

        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"{label} checkpoint loaded with mismatched keys.\n"
                f"Missing keys: {result.missing_keys}\n"
                f"Unexpected keys: {result.unexpected_keys}"
            )

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    def _load_image(self, image: ImageInput) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    def preprocess(self, image: ImageInput) -> torch.Tensor:
        pil_image = self._load_image(image)
        tensor = self.transform(pil_image).unsqueeze(0)  # add batch dim
        return tensor

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict(self, image: ImageInput) -> PredictionResult:
        """Run the hierarchical (Stage 1 -> Stage 2) pipeline on one image."""
        image_tensor = self.preprocess(image).to(self.device)

        with torch.inference_mode():
            stage1_logits = self.stage1_model(image_tensor)
            stage1_probs_tensor = torch.softmax(stage1_logits, dim=1)[0].cpu()

        stage1_pred_idx = int(torch.argmax(stage1_probs_tensor).item())
        stage1_prediction = self.stage1_idx_to_class[stage1_pred_idx]
        stage1_confidence = float(stage1_probs_tensor[stage1_pred_idx].item())
        stage1_probs = {
            self.stage1_idx_to_class[i]: float(p)
            for i, p in enumerate(stage1_probs_tensor.tolist())
        }

        result: PredictionResult = {
            "stage1_prediction": stage1_prediction,
            "stage1_confidence": stage1_confidence,
            "stage2_prediction": None,
            "stage2_confidence": None,
            "final_prediction": "no cropland",
            "final_confidence": stage1_confidence,
            "stage2_used": False,
            "stage1_probs": stage1_probs,
            "stage2_probs": None,
        }

        if stage1_prediction != "cropland":
            return result

        with torch.inference_mode():
            stage2_logits = self.stage2_model(image_tensor)
            stage2_probs_tensor = torch.softmax(stage2_logits, dim=1)[0].cpu()

        stage2_pred_idx = int(torch.argmax(stage2_probs_tensor).item())
        stage2_prediction = self.stage2_idx_to_class[stage2_pred_idx]
        stage2_confidence = float(stage2_probs_tensor[stage2_pred_idx].item())
        stage2_probs = {
            self.stage2_idx_to_class[i]: float(p)
            for i, p in enumerate(stage2_probs_tensor.tolist())
        }

        result.update({
            "stage2_prediction": stage2_prediction,
            "stage2_confidence": stage2_confidence,
            "stage2_used": True,
            "stage2_probs": stage2_probs,
            "final_prediction": stage2_prediction,
            "final_confidence": stage2_confidence,
        })

        return result

    def predict_batch(self, images: List[ImageInput]) -> List[PredictionResult]:
        """Run :meth:`predict` over a list of images (one at a time)."""
        return [self.predict(image) for image in images]
