# Batch Normalization, Effective Depth, and Topology in Convolutional Neural Networks

**Authors:** Manfred M. Fischer<sup>1,*</sup> and Joshua D. Pitts<sup>2</sup>  
<sup>1</sup>Department of Socioeconomics, Vienna University of Economics and Business (WU Wien), Vienna, Austria  
<sup>2</sup>Boston University, Boston, MA, USA  
<sup>*</sup>Correspondence: `manfred.fischer@wu.ac.at`

Official reproducibility repository for the manuscript:

> **Batch normalization rather than nominal or effective depth governs convolutional network accuracy while topology governs efficiency**

---

## Overview

Convolutional neural networks (CNNs) are routinely compared by nominal layer count ($D_{\mathrm{nom}}$), yet residual shortcuts and parallel multi-branch modules shorten forward signal paths ($D_{\mathrm{eff}}$) and co-occur with batch normalization (BN) in standard reference implementations. This repository provides:

1. **Closed-form structural effective-depth calculators** (`models/depth_info.py`, Equations 1–4) placing sequential stacks (VGG with or without BN), shortcut networks (ResNet), and parallel multi-branch networks (GoogLeNet / Inception-v1) onto a common pre-training path-length scale.
2. **Unified 200-epoch PyTorch training harness** (`models/` and `scripts/train.py`) executing all 11 evaluated configurations on CIFAR-10 resized to $224 \times 224$ under an identical stochastic gradient descent schedule.
3. **Complete empirical measurements (`results/`)** covering all **55 completed 200-epoch training runs** (**11,000 per-epoch records**):
   - **Primary BN-equalized cohort (40 runs):** `VGG-11-BN`, `VGG-13-BN`, `VGG-16-BN`, `VGG-19-BN`, `ResNet-18`, `ResNet-34`, `ResNet-50`, and `GoogLeNet` across five identical random seeds (`42, 123, 456, 789, 1024`).
   - **Canonical unnormalized VGG baseline cohort (15 runs):** `VGG-11`, `VGG-16`, and `VGG-19` without `BatchNorm2d` ($n = 5$ completed runs per depth).
4. **Deterministic statistical verification & figure generation** (`scripts/verify_paper_results.py` and `scripts/generate_figures.py`) reproducing all manuscript tables, statistical tests, and Figures 1–6 / Supplementary Figures S1–S3 directly from the archived CSV logs.

---

## Summary of Main Empirical Findings (Epoch 200)

| Cohort / Model | Conv + FC | $D_{\mathrm{nom}}$ | $D_{\mathrm{eff}}$ | BN Layers | Params (M) | G-MACs | Top-1 Accuracy (%, [range]) | Test Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary BN cohort ($n = 5$ seeds)** | | | | | | | | |
| `VGG-11-BN` | 8 + 3 | 11 | 11.0 | 8 | 132.869 | 7.639 | 90.66 ± 0.06 [90.56, 90.72] | 0.416 ± 0.008 |
| `VGG-13-BN` | 10 + 3 | 13 | 13.0 | 10 | 133.054 | 11.357 | 90.66 ± 0.22 [90.47, 91.04] | 0.423 ± 0.021 |
| `VGG-16-BN` | 13 + 3 | 16 | 16.0 | 13 | 138.366 | 15.525 | 91.42 ± 0.20 [91.16, 91.62] | 0.421 ± 0.018 |
| `VGG-19-BN` | 16 + 3 | 19 | 19.0 | 16 | 143.678 | 19.692 | 91.61 ± 0.37 [91.18, 91.96] | 0.433 ± 0.012 |
| `ResNet-18` | 17 + 1 | 18 | 11.5 | 20 | 11.690 | 1.824 | 90.66 ± 0.23 [90.36, 90.97] | 0.323 ± 0.011 |
| `ResNet-34` | 33 + 1 | 34 | 19.5 | 36 | 21.798 | 3.679 | 91.18 ± 0.17 [90.98, 91.44] | 0.325 ± 0.005 |
| `ResNet-50` | 49 + 1 | 50 | 28.0 | 53 | 25.557 | 4.134 | 90.49 ± 0.15 [90.23, 90.59] | 0.362 ± 0.016 |
| `GoogLeNet` | 21 + 1 | 22 | 17.5 | 59 | 13.005 | 1.511 | 92.02 ± 0.16 [91.84, 92.20] | 0.294 ± 0.002 |
| **Unnormalized VGG baseline ($n = 5$ runs)** | | | | | | | | |
| `VGG-11 (no BN)` | 8 + 3 | 11 | 11.0 | 0 | 132.863 | 7.609 | 87.25 ± 0.32 [86.90, 87.70] | 0.706 ± 0.012 |
| `VGG-16 (no BN)` | 13 + 3 | 16 | 16.0 | 0 | 138.358 | 15.470 | 87.81 ± 0.11 [87.64, 87.93] | 0.773 ± 0.023 |
| `VGG-19 (no BN)` | 16 + 3 | 19 | 19.0 | 0 | 143.667 | 19.632 | 87.83 ± 0.46 [87.19, 88.32] | 0.835 ± 0.053 |

* **Normalization resolves VGG's accuracy gap:** Adding BN lifts VGG accuracy by **+3.41 to +3.78 pp** (Mann–Whitney $U = 25$, exact two-sided $P = 0.0079$; matched-seed paired $t$-test Holm-adjusted $P \le 0.0043$), bringing VGG-BN on par with or slightly ahead of ResNet-18/34/50.
* **Depth proxies do not linearly predict accuracy once BN is equalized:** Across the 8 BN architectures, neither nominal depth ($r = -0.179, P = 0.671$) nor structural effective depth ($r = 0.064, P = 0.880$) correlates linearly with Top-1 accuracy.
* **Topology governs compute efficiency and probabilistic calibration:** GoogLeNet and ResNet-18/34 match or surpass VGG-BN using **76–92% fewer MACs**, **84–92% fewer trainable parameters**, and substantially lower main-head test cross-entropy (`0.294` and `0.323–0.362` vs. `0.416–0.433` in VGG-BN and `0.706–0.835` in unnormalized VGG).

---

## Repository Structure

```text
.
├── models/
│   ├── __init__.py               # Unified registry of all 11 evaluated configurations
│   ├── base.py                   # Recursive dropout configuration helper
│   ├── depth_info.py             # Equations (1)-(4): nominal & structural effective depth
│   ├── googlenet.py              # GoogLeNet (Inception-v1, 59 BN layers, 2 aux heads)
│   ├── resnet.py                 # ResNet-18, ResNet-34, ResNet-50 builders
│   └── vgg.py                    # VGG-11/13/16/19 with and without BatchNorm2d
├── results/
│   ├── data/
│   │   ├── all_run_epoch_measurements.csv   # All 11,000 run-epoch records (55 runs x 200 epochs)
│   │   ├── per_run_endpoints.csv            # Per-seed endpoints at epochs 100 and 200 (110 rows)
│   │   ├── curves_all_models.csv            # Per-epoch mean +/- sample s.d. trajectories
│   │   ├── matched_seed_differences.csv     # Paired BN minus non-BN seed differences
│   │   ├── checkpoint_metadata.json         # Verified parameter & BN module counts from checkpoints
│   │   ├── run_inventory.csv                # Complete run provenance manifest
│   │   └── statistics.json                  # Exact correlation & regression statistics
│   └── tables/
│       ├── all_cohort_summary.csv           # Table 1 & Supplementary Table S1 summary metrics
│       ├── bn_vs_nonbn_accuracy.csv         # Full-cohort BN gains & exact Mann-Whitney tests
│       ├── matched_seed_statistics.csv      # Table 2 Part A & Table S3 Part B paired t / Holm / sign-flip
│       ├── paired_100_200.csv               # Supplementary Table S2 (epoch 200 vs epoch 100)
│       ├── bn_vgg_adjacent_depth_comparisons.csv  # Supplementary Table S3 Part A
│       ├── vgg11_seed_sensitivity.csv       # Supplementary Table S3 Part C
│       ├── correlations.csv                 # Table 2 Part B & Supplementary Table S5
│       ├── table1_architectures.csv         # Architectural specifications
│       ├── table2_epoch100.csv              # Primary BN cohort at epoch 100
│       └── table2_epoch200.csv              # Primary BN cohort at epoch 200
├── scripts/
│   ├── verify_paper_results.py   # End-to-end deterministic numerical & statistical verifier
│   ├── generate_figures.py       # Regenerates Figs. 1-6 and Supplementary Figs. S1-S3
│   ├── bn_analysis_utils.py      # Statistical test & correlation utilities
│   ├── compute_mean.py           # Computes native CIFAR-10 training RGB channel means
│   ├── preprocess_cifar.py       # Optional uint8 224x224 CIFAR-10 tensor cache builder
│   ├── train.py                  # Unified 200-epoch SGD + StepLR + AMP training loop
│   ├── run_sweep.py              # Multi-seed campaign runner (seeds: 42, 123, 456, 789, 1024)
│   ├── run_logger.py             # CSV and TensorBoard epoch logger
│   └── metrics_sampler.py        # Hardware telemetry sampler
├── Dockerfile                    # Reproducible CUDA 12.4 / PyTorch 2.2 container definition
├── requirements.txt              # Python package dependencies
└── LICENSE                       # MIT License
```

---

## Quickstart

### 1. Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

*(Note: `verify_paper_results.py` and `generate_figures.py` require only Python 3.10+, `numpy`, `scipy`, and `matplotlib`, and run in seconds on any CPU laptop.)*

### 2. Verify All Numerical Results & Statistical Tests

Recompute every endpoint summary, Mann–Whitney rank-sum test, matched-seed paired $t$-test, Holm step-down adjustment, exact paired sign-flip bound, and cross-architecture depth correlation directly from the 11,000 per-epoch measurements:

```bash
python3 scripts/verify_paper_results.py
```

### 3. Regenerate All Manuscript & Supplementary Figures

Generate vector PDFs and 300 DPI PNGs (`fig1`–`fig6` and `supp_fig1`–`supp_fig3`) into `figures/`:

```bash
python3 scripts/generate_figures.py
```

### 4. Train Individual Models or Multi-Seed Sweeps (GPU)

Optional pre-resizing of CIFAR-10 to $224 \times 224$ `uint8` tensor caches (~4.2 GB disk) for fast multi-epoch I/O:

```bash
python3 scripts/compute_mean.py
python3 scripts/preprocess_cifar.py --img_size 224 --output_dir data/preresized
```

Train a single 200-epoch run under the unified protocol (e.g., `vgg19_bn`, seed `42`):

```bash
python3 scripts/train.py \
  --model vgg19_bn \
  --seed 42 \
  --epochs 200 \
  --preresized_dir data/preresized \
  --ckpt_dir runs/vgg19_bn_s42
```

Run the full 40-run primary BN-equalized campaign (`8 models × 5 seeds`):

```bash
python3 scripts/run_sweep.py --cohort primary_bn --preresized_dir data/preresized
```

---

## License

Released under the [MIT License](LICENSE).
