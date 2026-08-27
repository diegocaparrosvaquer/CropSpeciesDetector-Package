"""
crop_species_detection
=======================

Hierarchical crop species classification with a DINOv2 ViT-B/14 backbone.

Stage 1 decides whether an image shows cropland at all.
Stage 2 (run only when Stage 1 says "cropland") classifies the image
into one of nine crop species.

Quick start
-----------
>>> from crop_species_detection import CropSpeciesDetector
>>> detector = CropSpeciesDetector.from_checkpoints(
...     "models/stage1_best.pt", "models/stage2_best.pt"
... )
>>> result = detector.predict("field.jpg")
>>> result["final_prediction"]
'maize'
"""

from .model import DINOv2Classifier
from .pipeline import CropSpeciesDetector, PredictionResult
from .constants import STAGE1_CLASSES, STAGE2_CLASSES

__all__ = [
    "DINOv2Classifier",
    "CropSpeciesDetector",
    "PredictionResult",
    "STAGE1_CLASSES",
    "STAGE2_CLASSES",
]

__version__ = "0.1.0"
