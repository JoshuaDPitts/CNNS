"""
VGG family model builders (with and without Batch Normalization).

Reference:
  - Simonyan & Zisserman (2015), "Very deep convolutional networks for
    large-scale image recognition", ICLR.
  - Ioffe & Szegedy (2015), "Batch normalization: accelerating deep network
    training by reducing internal covariate shift", ICML.

All architectures preserve canonical 1,000-unit torchvision output heads so
parameter counts and profiled MACs remain strictly commensurate across cohorts.
"""

import torch.nn as nn
from torchvision.models import (
    vgg11,
    vgg11_bn,
    vgg13,
    vgg13_bn,
    vgg16,
    vgg16_bn,
    vgg19,
    vgg19_bn,
)
from .base import patch_dropout

_EXPECTED_BN = {
    "vgg11": 0,
    "vgg13": 0,
    "vgg16": 0,
    "vgg19": 0,
    "vgg11_bn": 8,
    "vgg13_bn": 10,
    "vgg16_bn": 13,
    "vgg19_bn": 16,
}

_TORCHVISION_BUILDERS = {
    "vgg11": vgg11,
    "vgg13": vgg13,
    "vgg16": vgg16,
    "vgg19": vgg19,
    "vgg11_bn": vgg11_bn,
    "vgg13_bn": vgg13_bn,
    "vgg16_bn": vgg16_bn,
    "vgg19_bn": vgg19_bn,
}


def _build_vgg_variant(name: str, dropout: float = 0.5) -> nn.Module:
    model = _TORCHVISION_BUILDERS[name](weights=None)
    patch_dropout(model, dropout)
    bn_count = sum(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert bn_count == _EXPECTED_BN[name], (
        f"Expected {_EXPECTED_BN[name]} BN layers in {name}, got {bn_count}"
    )
    assert model.classifier[6].out_features == 1000, (
        "Expected canonical 1000-output classifier head"
    )
    return model


def build_vgg11_bn(dropout: float = 0.5) -> nn.Module:
    return _build_vgg_variant("vgg11_bn", dropout)


def build_vgg13_bn(dropout: float = 0.5) -> nn.Module:
    return _build_vgg_variant("vgg13_bn", dropout)


def build_vgg16_bn(dropout: float = 0.5) -> nn.Module:
    return _build_vgg_variant("vgg16_bn", dropout)


def build_vgg19_bn(dropout: float = 0.5) -> nn.Module:
    return _build_vgg_variant("vgg19_bn", dropout)


def build_vgg11(dropout: float = 0.5) -> nn.Module:
    return _build_vgg_variant("vgg11", dropout)


def build_vgg13(dropout: float = 0.5) -> nn.Module:
    return _build_vgg_variant("vgg13", dropout)


def build_vgg16(dropout: float = 0.5) -> nn.Module:
    return _build_vgg_variant("vgg16", dropout)


def build_vgg19(dropout: float = 0.5) -> nn.Module:
    return _build_vgg_variant("vgg19", dropout)
