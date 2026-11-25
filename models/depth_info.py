"""
Closed-form nominal and structural effective-depth calculators (Equations 1-4).

Companion code for:
  Fischer, M. M. & Pitts, J. D. "Batch normalization rather than nominal or
  effective depth governs convolutional network accuracy while topology governs
  efficiency."

Definitions (Methods & Supplementary Note):
  - Nominal depth (D_nom): count of weight-bearing convolutional and fully
    connected (FC) transformations along the longest main-output path.
  - Structural effective depth (D_eff): expected path length across all feasible
    main-output paths under a pre-training uniform branch-selection prior:
        D_eff = (1 / |P|) * sum_{p in P} ell(p)                         (Eq. 1)

Closed-form topology expressions:
  - Sequential VGG (with or without BN): D_eff = D_nom                  (Eq. 2)
  - Residual networks (ResNet)         : D_eff = (ell_min + ell_max)/2  (Eq. 3)
  - Multi-branch Inception (GoogLeNet) : D_eff = D_base + sum_m mean(d) (Eq. 4)
"""

from __future__ import annotations


def effective_depth_vgg(d_nom: int) -> float:
    """Strictly sequential stack (|P| = 1): D_eff = D_nom (Eq. 2)."""
    return float(d_nom)


def effective_depth_resnet(ell_min: int, ell_max: int) -> float:
    """
    Residual network path ensemble under independent uniform block selection:
    D_eff = (ell_min + ell_max) / 2 (Eq. 3).

    ell_min counts the mandatory stem convolution, final FC layer, and
    stage-transition 1x1 projection shortcuts (5 for ResNet-18/34; 6 for ResNet-50).
    ell_max equals D_nom (18, 34, 50).
    """
    return (ell_min + ell_max) / 2.0


def effective_depth_googlenet(
    branch_depths_per_module: list[list[int]],
    d_base: float = 4.0,
) -> float:
    """
    Parallel multi-branch Inception network (Eq. 4):
    D_eff = D_base + sum_{m=1}^M (1/B) * sum_{b=1}^B d_{m,b},
    where D_base = 4 counts the 3 stem convolutions + 1 main FC classifier.
    """
    module_sum = sum(sum(b) / len(b) for b in branch_depths_per_module if b)
    return float(d_base + module_sum)


_GOOGLENET_BRANCHES = [[1, 2, 2, 1]] * 9

MODEL_DEPTH_INFO: dict[str, dict] = {
    # Primary BN-equalized cohort (n = 8 configurations, 5 unique seeds each)
    "vgg11_bn": {
        "cohort": "main_bn",
        "family": "vgg",
        "conv_longest": 8,
        "fc_longest": 3,
        "nominal_depth": 11,
        "effective_depth": effective_depth_vgg(11),
        "bn_layers": 8,
    },
    "vgg13_bn": {
        "cohort": "main_bn",
        "family": "vgg",
        "conv_longest": 10,
        "fc_longest": 3,
        "nominal_depth": 13,
        "effective_depth": effective_depth_vgg(13),
        "bn_layers": 10,
    },
    "vgg16_bn": {
        "cohort": "main_bn",
        "family": "vgg",
        "conv_longest": 13,
        "fc_longest": 3,
        "nominal_depth": 16,
        "effective_depth": effective_depth_vgg(16),
        "bn_layers": 13,
    },
    "vgg19_bn": {
        "cohort": "main_bn",
        "family": "vgg",
        "conv_longest": 16,
        "fc_longest": 3,
        "nominal_depth": 19,
        "effective_depth": effective_depth_vgg(19),
        "bn_layers": 16,
    },
    "resnet18": {
        "cohort": "main_bn",
        "family": "resnet",
        "conv_longest": 17,
        "fc_longest": 1,
        "nominal_depth": 18,
        "effective_depth": effective_depth_resnet(5, 18),
        "bn_layers": 20,
    },
    "resnet34": {
        "cohort": "main_bn",
        "family": "resnet",
        "conv_longest": 33,
        "fc_longest": 1,
        "nominal_depth": 34,
        "effective_depth": effective_depth_resnet(5, 34),
        "bn_layers": 36,
    },
    "resnet50": {
        "cohort": "main_bn",
        "family": "resnet",
        "conv_longest": 49,
        "fc_longest": 1,
        "nominal_depth": 50,
        "effective_depth": effective_depth_resnet(6, 50),
        "bn_layers": 53,
    },
    "googlenet": {
        "cohort": "main_bn",
        "family": "googlenet",
        "conv_longest": 21,
        "fc_longest": 1,
        "nominal_depth": 22,
        "effective_depth": effective_depth_googlenet(_GOOGLENET_BRANCHES, d_base=4.0),
        "bn_layers": 59,
    },
    # Canonical unnormalized VGG baseline cohort (n = 3 evaluated depths, 5 runs each)
    "vgg11": {
        "cohort": "unnormalized_vgg",
        "family": "vgg",
        "conv_longest": 8,
        "fc_longest": 3,
        "nominal_depth": 11,
        "effective_depth": effective_depth_vgg(11),
        "bn_layers": 0,
    },
    "vgg16": {
        "cohort": "unnormalized_vgg",
        "family": "vgg",
        "conv_longest": 13,
        "fc_longest": 3,
        "nominal_depth": 16,
        "effective_depth": effective_depth_vgg(16),
        "bn_layers": 0,
    },
    "vgg19": {
        "cohort": "unnormalized_vgg",
        "family": "vgg",
        "conv_longest": 16,
        "fc_longest": 3,
        "nominal_depth": 19,
        "effective_depth": effective_depth_vgg(19),
        "bn_layers": 0,
    },
    # Unnormalized VGG-13 retained in builder registry for completeness (not evaluated in study)
    "vgg13": {
        "cohort": "not_evaluated",
        "family": "vgg",
        "conv_longest": 10,
        "fc_longest": 3,
        "nominal_depth": 13,
        "effective_depth": effective_depth_vgg(13),
        "bn_layers": 0,
    },
}


def get_depth_info(model_name: str) -> dict:
    """Return structural depth metadata for `model_name` (case-insensitive)."""
    name = model_name.lower()
    if name not in MODEL_DEPTH_INFO:
        raise KeyError(
            f"Unknown model '{model_name}'. Available: {sorted(MODEL_DEPTH_INFO)}"
        )
    return MODEL_DEPTH_INFO[name].copy()
