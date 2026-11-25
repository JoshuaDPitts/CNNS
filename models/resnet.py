"""
ResNet family model builders (ResNet-18, ResNet-34, ResNet-50).

Reference:
  - He et al. (2016), "Deep residual learning for image recognition", CVPR.

Built-in BatchNorm2d immediately follows every convolutional layer, including
stage-transition 1x1 projection shortcuts (20, 36, and 53 BN modules).
"""

import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from .base import patch_dropout

_EXPECTED_BN = {
    "resnet18": 20,
    "resnet34": 36,
    "resnet50": 53,
}


def _verify_resnet(model: nn.Module, name: str) -> nn.Module:
    bn_count = sum(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert bn_count == _EXPECTED_BN[name], (
        f"Expected {_EXPECTED_BN[name]} BN layers in {name}, got {bn_count}"
    )
    assert model.fc.out_features == 1000, "Expected canonical 1000-output fc head"
    return model


def build_resnet18(dropout: float = 0.0) -> nn.Module:
    model = resnet18(weights=None)
    patch_dropout(model, dropout)
    return _verify_resnet(model, "resnet18")


def build_resnet34(dropout: float = 0.0) -> nn.Module:
    model = resnet34(weights=None)
    patch_dropout(model, dropout)
    return _verify_resnet(model, "resnet34")


def build_resnet50(dropout: float = 0.0) -> nn.Module:
    model = resnet50(weights=None)
    patch_dropout(model, dropout)
    return _verify_resnet(model, "resnet50")
