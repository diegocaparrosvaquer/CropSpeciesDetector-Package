"""DINOv2-backed classification head.

This architecture is kept identical to the original demo's ``model.py``
so that the ``stage1_best.pt`` / ``stage2_best.pt`` checkpoints load with
``strict=True`` and no key mismatches.
"""

import torch
import torch.nn as nn


class DINOv2Classifier(nn.Module):
    """DINOv2 ViT-B/14 backbone with a small MLP classification head."""

    def __init__(self, num_classes, embedding_dim=768, dropout=0.3):
        super().__init__()

        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
        )

        self.embedding_dim = embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
