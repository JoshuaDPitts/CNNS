"""
Model registry for all CNN architectures evaluated in the study.

Primary BN-equalized cohort (8 models x 5 seeds = 40 runs):
  vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn, resnet18, resnet34, resnet50, googlenet

Canonical unnormalized VGG baseline cohort (3 models x 5 runs = 15 runs):
  vgg11, vgg16, vgg19
"""

from .depth_info import MODEL_DEPTH_INFO, get_depth_info

PRIMARY_BN_MODELS = [
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
    "resnet18",
    "resnet34",
    "resnet50",
    "googlenet",
]

UNNORMALIZED_VGG_MODELS = [
    "vgg11",
    "vgg16",
    "vgg19",
]

ALL_MODEL_NAMES = PRIMARY_BN_MODELS + ["vgg11", "vgg13", "vgg16", "vgg19"]


def _load_builders() -> dict:
    from .googlenet import build_googlenet
    from .resnet import build_resnet18, build_resnet34, build_resnet50
    from .vgg import (
        build_vgg11,
        build_vgg11_bn,
        build_vgg13,
        build_vgg13_bn,
        build_vgg16,
        build_vgg16_bn,
        build_vgg19,
        build_vgg19_bn,
    )

    return {
        "vgg11_bn": build_vgg11_bn,
        "vgg13_bn": build_vgg13_bn,
        "vgg16_bn": build_vgg16_bn,
        "vgg19_bn": build_vgg19_bn,
        "resnet18": build_resnet18,
        "resnet34": build_resnet34,
        "resnet50": build_resnet50,
        "googlenet": build_googlenet,
        "vgg11": build_vgg11,
        "vgg13": build_vgg13,
        "vgg16": build_vgg16,
        "vgg19": build_vgg19,
    }


def get_model(name: str, dropout: float = 0.5):
    """Instantiate an evaluated architecture by name (requires PyTorch & torchvision)."""
    builders = _load_builders()
    key = name.lower()
    if key not in builders:
        available = ", ".join(sorted(builders.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return builders[key](dropout)


def list_models() -> list[str]:
    return list(ALL_MODEL_NAMES)
