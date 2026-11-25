"""
GoogLeNet (Inception-v1) model builder.

Reference:
  - Szegedy et al. (2015), "Going deeper with convolutions", CVPR.

Includes built-in BatchNorm2d after every convolution across the main stem,
nine four-branch Inception modules, and two auxiliary training classifiers
(59 BatchNorm2d modules in total; auxiliary loss weight = 0.3 during training).
"""

import torch.nn as nn
from torchvision.models import googlenet
from .base import patch_dropout


def build_googlenet(dropout: float = 0.5) -> nn.Module:
    model = googlenet(weights=None, aux_logits=True, init_weights=True)
    patch_dropout(model, dropout)
    bn_count = sum(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert bn_count == 59, f"Expected 59 BN layers in googlenet, got {bn_count}"
    assert model.fc.out_features == 1000, "Expected canonical 1000-output main head"
    return model
