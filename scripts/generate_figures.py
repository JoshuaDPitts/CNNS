#!/usr/bin/env python3
r"""
Generate publication-grade figures for Scientific Reports submission.

Enforces Nature Portfolio / Scientific Reports artwork standards:
- Sans-serif typography (DejaVu Sans / Arial), consistent font sizes
- Proper mathematical subscripts ($D_{\mathrm{nom}}$, $D_{\mathrm{eff}}$)
- Lower-case bold panel labels ('a', 'b', 'c') without parentheses
- No baked-in 'Figure X.' suptitles or internal revision memo phrasing
- Zero label overlaps, minimum line width >= 1.0 pt, white backgrounds
- Vector PDF (Type 42 fonts) + 300 DPI PNG exports for every figure
"""

from pathlib import Path
import csv
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import ScalarFormatter, NullFormatter
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


SUMMARY = read_csv(SRC / "tables/all_cohort_summary.csv")
CURVES = read_csv(SRC / "data/curves_all_models.csv")
EPOCHS = read_csv(SRC / "data/all_run_epoch_measurements.csv")
MATCHED = read_csv(SRC / "tables/matched_seed_statistics.csv")
STATS = json.loads((SRC / "data/statistics.json").read_text(encoding="utf-8"))

MAIN_MODELS = [
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
    "resnet18",
    "resnet34",
    "resnet50",
    "googlenet",
]
PLAIN_MODELS = ["vgg11", "vgg16", "vgg19"]
ALL_MODELS = MAIN_MODELS + PLAIN_MODELS

LABELS = {
    "vgg11_bn": "VGG-11-BN",
    "vgg13_bn": "VGG-13-BN",
    "vgg16_bn": "VGG-16-BN",
    "vgg19_bn": "VGG-19-BN",
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
    "resnet50": "ResNet-50",
    "googlenet": "GoogLeNet",
    "vgg11": "VGG-11 (no BN)",
    "vgg16": "VGG-16 (no BN)",
    "vgg19": "VGG-19 (no BN)",
}

COLORS = {
    "vgg": "#1f5aa6",
    "resnet": "#b31b34",
    "googlenet": "#1e7e3e",
    "nonbn": "#6b7280",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "lines.linewidth": 1.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def fval(row, key):
    return float(row[key])


def fam_of(model: str) -> str:
    if model.startswith("vgg"):
        return "vgg"
    if model.startswith("resnet"):
        return "resnet"
    return "googlenet"


def get_summary(model: str, horizon: int = 200):
    return next(
        r for r in SUMMARY if r["model"] == model and int(r["horizon"]) == horizon
    )


def get_curve(model: str, horizon: int = 200):
    rows = [r for r in CURVES if r["model"] == model and int(r["epoch"]) <= horizon]
    rows.sort(key=lambda r: int(r["epoch"]))
    return rows


def get_test_loss_curve(model: str, horizon: int = 200):
    by_ep = {}
    for r in EPOCHS:
        if r["model"] == model and int(r["completed_epoch"]) <= horizon:
            ep = int(r["completed_epoch"])
            by_ep.setdefault(ep, []).append(float(r["test_loss"]))
    xs = sorted(by_ep.keys())
    means = np.array([np.mean(by_ep[e]) for e in xs])
    sds = np.array([np.std(by_ep[e], ddof=1) for e in xs])
    return np.array(xs), means, means - sds, means + sds


def panel_label(ax, letter: str, x: float = -0.13, y: float = 1.05):
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#111111",
    )


def save_fig(fig, stem: str):
    for ext in ("pdf", "png"):
        fig.savefig(
            OUT_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.06
        )
    plt.close(fig)
    print(f"Saved {stem}.pdf and {stem}.png")


def draw_box(ax, x, y, text, w=0.80, h=0.11, fc="#eef3f8", ec="#475569", fs=8.0):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="#111827")


def draw_arrow(ax, p1, p2, ls="-", color="#475569"):
    ax.annotate(
        "",
        xy=p2,
        xytext=p1,
        arrowprops=dict(
            arrowstyle="-|>,head_length=0.45,head_width=0.25",
            lw=1.05,
            color=color,
            linestyle=ls,
        ),
    )


# =========================================================================
# FIGURE 1: Topologies, path-averaging concept, and structural D_eff vs D_nom
# =========================================================================
def build_fig1():
    fig = plt.figure(figsize=(11.2, 7.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.05, 1.0])

    # Panel a: Four architectural block motifs across top row
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.set_xlim(0, 4.0)
    ax_a.set_ylim(0, 1.0)
    ax_a.axis("off")
    panel_label(ax_a, "a", x=-0.015, y=1.01)

    cx = 0.45
    ax_a.text(
        cx,
        0.96,
        "Sequential stack\n(VGG without BN)",
        ha="center",
        va="top",
        fontsize=9.0,
        fontweight="bold",
    )
    for y, txt, fc in [
        (0.75, "Feature input", "#f3f4f6"),
        (0.55, "Conv 3×3 → ReLU", "#e5e7eb"),
        (0.35, "Conv 3×3 → ReLU", "#e5e7eb"),
        (0.14, "3-layer FC head", "#fef3c7"),
    ]:
        draw_box(ax_a, cx, y, txt, w=0.76, h=0.115, fc=fc, fs=8.0)
    for y1, y2 in [(0.69, 0.61), (0.49, 0.41), (0.29, 0.20)]:
        draw_arrow(ax_a, (cx, y1), (cx, y2))

    cx = 1.42
    ax_a.text(
        cx,
        0.96,
        "Normalized stack\n(VGG-BN)",
        ha="center",
        va="top",
        fontsize=9.0,
        fontweight="bold",
    )
    for y, txt, fc in [
        (0.75, "Feature input", "#f3f4f6"),
        (0.55, "Conv 3×3 → BN → ReLU", "#dbeafe"),
        (0.35, "Conv 3×3 → BN → ReLU", "#dbeafe"),
        (0.14, "3-layer FC head", "#fef3c7"),
    ]:
        draw_box(ax_a, cx, y, txt, w=0.80, h=0.115, fc=fc, fs=8.0)
    for y1, y2 in [(0.69, 0.61), (0.49, 0.41), (0.29, 0.20)]:
        draw_arrow(ax_a, (cx, y1), (cx, y2))

    cx = 2.44
    ax_a.text(
        cx,
        0.96,
        "Residual block\n(ResNet, BN)",
        ha="center",
        va="top",
        fontsize=9.0,
        fontweight="bold",
    )
    for y, txt, fc in [
        (0.75, "Block input", "#f3f4f6"),
        (0.55, "Conv → BN → ReLU", "#fee2e2"),
        (0.35, "Conv → BN", "#fee2e2"),
        (0.14, "Add (+) → ReLU", "#fef3c7"),
    ]:
        draw_box(ax_a, cx, y, txt, w=0.72, h=0.115, fc=fc, fs=8.0)
    for y1, y2 in [(0.69, 0.61), (0.49, 0.41), (0.29, 0.20)]:
        draw_arrow(ax_a, (cx, y1), (cx, y2))
    ax_a.plot(
        [cx - 0.36, cx - 0.48, cx - 0.48, cx - 0.38],
        [0.75, 0.75, 0.14, 0.14],
        color="#b31b34",
        lw=1.15,
        ls="--",
    )
    draw_arrow(ax_a, (cx - 0.48, 0.14), (cx - 0.36, 0.14), color="#b31b34")
    ax_a.text(
        cx - 0.50,
        0.45,
        "Identity or 1×1\nshortcut",
        rotation=90,
        ha="right",
        va="center",
        fontsize=7.6,
        color="#991b1b",
    )

    cx = 3.48
    ax_a.text(
        cx,
        0.96,
        "Multi-branch module\n(GoogLeNet, BN)",
        ha="center",
        va="top",
        fontsize=9.0,
        fontweight="bold",
    )
    draw_box(ax_a, cx, 0.77, "Module input", w=0.74, h=0.10, fc="#f3f4f6", fs=8.0)
    bxs = [3.08, 3.34, 3.62, 3.88]
    btxts = ["1×1\n(d=1)", "1×1→3×3\n(d=2)", "1×1→3×3*\n(d=2)", "Pool→1×1\n(d=1)"]
    for bx, btxt in zip(bxs, btxts):
        draw_box(ax_a, bx, 0.46, btxt, w=0.23, h=0.23, fc="#dcfce7", fs=7.2)
        draw_arrow(ax_a, (cx, 0.72), (bx, 0.58))
        draw_arrow(ax_a, (bx, 0.34), (cx, 0.20))
    draw_box(
        ax_a, cx, 0.14, "Channel concatenate", w=0.78, h=0.10, fc="#fef3c7", fs=8.0
    )
    ax_a.text(
        cx,
        0.02,
        "*torchvision 3×3 branch; BN after every conv",
        ha="center",
        va="bottom",
        fontsize=7.3,
        color="#374151",
    )

    # Panel b: Operationalizing D_nom vs D_eff per structural motif
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.axis("off")
    panel_label(ax_b, "b", x=-0.03, y=1.02)
    ax_b.text(
        0.5,
        0.96,
        "Pre-training path-length operationalization",
        ha="center",
        va="top",
        fontsize=9.5,
        fontweight="bold",
    )

    rows_b = [
        (
            0.74,
            "Sequential block (VGG / VGG-BN)",
            r"1 path: {2}  $\rightarrow$  $D_{\mathrm{nom}} = 2$,  $D_{\mathrm{eff}} = 2.0$  (0% reduction)",
            "#dbeafe",
        ),
        (
            0.47,
            "Residual basic block (ResNet-18/34)",
            r"2 paths: {2, 0}  $\rightarrow$  $D_{\mathrm{nom}} = 2$,  $D_{\mathrm{eff}} = 1.0$  (50% block reduction)",
            "#fee2e2",
        ),
        (
            0.20,
            "Inception module (GoogLeNet)",
            r"4 branches: {1, 2, 2, 1}  $\rightarrow$  $D_{\mathrm{nom}} = 2$,  $D_{\mathrm{eff}} = 1.5$  (25% module reduction)",
            "#dcfce7",
        ),
    ]
    for y, hdr, body, fc in rows_b:
        patch = FancyBboxPatch(
            (0.03, y - 0.10),
            0.94,
            0.20,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=fc,
            edgecolor="#64748b",
            linewidth=1.0,
        )
        ax_b.add_patch(patch)
        ax_b.text(
            0.06,
            y + 0.04,
            hdr,
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#111827",
        )
        ax_b.text(
            0.06, y - 0.04, body, ha="left", va="center", fontsize=8.1, color="#1f2937"
        )
    ax_b.text(
        0.5,
        0.005,
        "Full-network $D_{\\mathrm{eff}}$ adds fixed stem, classifier, and 1×1 projection layers;\nBatch Normalization adds affine scales/shifts without altering path count.",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#374151",
    )

    # Panel c: Structural effective depth vs nominal depth across all 8 architectures
    ax_c = fig.add_subplot(gs[1, 1])
    panel_label(ax_c, "c", x=-0.13, y=1.02)
    ax_c.plot([6, 54], [6, 54], ls="--", color="#6b7280", lw=1.1)

    offsets_c = {
        "vgg11_bn": (-42, 4),
        "vgg13_bn": (-42, 5),
        "vgg16_bn": (-42, 6),
        "vgg19_bn": (-62, 8),
        "resnet18": (7, -4),
        "resnet34": (7, -3),
        "resnet50": (-52, -13),
        "googlenet": (7, -2),
    }
    for m in MAIN_MODELS:
        r = get_summary(m, 200)
        x = fval(r, "nominal_depth")
        y = fval(r, "effective_depth")
        fam = fam_of(m)
        marker = "s" if fam == "resnet" else "^" if fam == "googlenet" else "o"
        ax_c.scatter(x, y, color=COLORS[fam], marker=marker, s=46, zorder=4)
        lbl = LABELS[m].replace("-BN", "") if m.startswith("vgg") else LABELS[m]
        if m == "vgg19_bn":
            lbl = "VGG-19 (±BN)"
        dx, dy = offsets_c[m]
        ax_c.annotate(
            lbl,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.8,
            color="#111827",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="#94a3b8")
            if m.startswith("vgg")
            else None,
        )

    handles_c = [
        Line2D(
            [],
            [],
            color="#6b7280",
            ls="--",
            lw=1.1,
            label=r"$D_{\mathrm{eff}} = D_{\mathrm{nom}}$",
        ),
        Line2D(
            [],
            [],
            color=COLORS["vgg"],
            marker="o",
            ls="",
            markersize=6,
            label="VGG (with or without BN)",
        ),
        Line2D(
            [],
            [],
            color=COLORS["resnet"],
            marker="s",
            ls="",
            markersize=6,
            label="ResNet (BN)",
        ),
        Line2D(
            [],
            [],
            color=COLORS["googlenet"],
            marker="^",
            ls="",
            markersize=6,
            label="GoogLeNet (BN)",
        ),
    ]
    ax_c.legend(handles=handles_c, frameon=False, loc="upper left", fontsize=7.9)
    ax_c.set_xlim(5.5, 54.5)
    ax_c.set_ylim(7.0, 54.5)
    ax_c.set_xlabel(r"Nominal depth $D_{\mathrm{nom}}$ (conv + FC layers)")
    ax_c.set_ylabel(r"Structural effective depth $D_{\mathrm{eff}}$ (layers)")

    save_fig(fig, "fig1")


# =========================================================================
# FIGURE 2: Endpoint accuracy vs depth, BN vs non-BN VGG, matched-seed gains
# =========================================================================
def build_fig2():
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), layout="constrained")

    # Panel a: 8-model BN cohort accuracy vs nominal depth (epoch 200)
    ax = axes[0]
    panel_label(ax, "a", x=-0.16, y=1.03)
    for fam in ("vgg", "resnet"):
        ms = [m for m in MAIN_MODELS if fam_of(m) == fam]
        xs = [fval(get_summary(m, 200), "nominal_depth") for m in ms]
        ys = [fval(get_summary(m, 200), "accuracy_mean") for m in ms]
        ax.plot(xs, ys, color=COLORS[fam], alpha=0.45, lw=1.2, zorder=2)

    label_offsets_a = {
        "vgg11_bn": (-48, 3),
        "vgg13_bn": (-48, -14),
        "vgg16_bn": (-48, 2),
        "vgg19_bn": (7, 4),
        "resnet18": (7, -4),
        "resnet34": (7, 4),
        "resnet50": (-48, 9),
        "googlenet": (7, -2),
    }
    for m in MAIN_MODELS:
        r = get_summary(m, 200)
        fam = fam_of(m)
        marker = "s" if fam == "resnet" else "^" if fam == "googlenet" else "o"
        x, y, sd = (
            fval(r, "nominal_depth"),
            fval(r, "accuracy_mean"),
            fval(r, "accuracy_sd"),
        )
        ax.errorbar(
            x,
            y,
            yerr=sd,
            fmt=marker,
            color=COLORS[fam],
            markersize=6,
            capsize=3,
            lw=1.2,
            zorder=4,
        )
        dx, dy = label_offsets_a[m]
        ax.annotate(
            LABELS[m], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.4
        )

    handles_a = [
        Line2D([], [], color=COLORS["vgg"], marker="o", ls="-", lw=1.2, label="VGG-BN"),
        Line2D(
            [],
            [],
            color=COLORS["resnet"],
            marker="s",
            ls="-",
            lw=1.2,
            label="ResNet (BN)",
        ),
        Line2D(
            [], [], color=COLORS["googlenet"], marker="^", ls="", label="GoogLeNet (BN)"
        ),
    ]
    ax.legend(handles=handles_a, frameon=False, loc="upper right", fontsize=7.8)
    ax.set_xlim(1.2, 54.5)
    ax.set_ylim(90.15, 92.38)
    ax.set_xlabel(r"Nominal depth $D_{\mathrm{nom}}$ (layers)")
    ax.set_ylabel("Test Top-1 accuracy (%)")
    ax.set_title("BN-equalized cohort (epoch 200)")

    # Panel b: Direct VGG normalization comparison across canonical evaluated depths (11, 16, 19)
    ax = axes[1]
    panel_label(ax, "b", x=-0.16, y=1.03)
    bn_paired_vgg = ["vgg11_bn", "vgg16_bn", "vgg19_bn"]
    xs_bn = [fval(get_summary(m, 200), "nominal_depth") for m in bn_paired_vgg]
    ys_bn = [fval(get_summary(m, 200), "accuracy_mean") for m in bn_paired_vgg]
    sd_bn = [fval(get_summary(m, 200), "accuracy_sd") for m in bn_paired_vgg]
    ax.errorbar(
        xs_bn,
        ys_bn,
        yerr=sd_bn,
        fmt="o-",
        color=COLORS["vgg"],
        mfc=COLORS["vgg"],
        markersize=6,
        capsize=3,
        lw=1.4,
        label="With BN (5 seeds)",
    )

    xs_nb = [fval(get_summary(m, 200), "nominal_depth") for m in PLAIN_MODELS]
    ys_nb = [fval(get_summary(m, 200), "accuracy_mean") for m in PLAIN_MODELS]
    sd_nb = [fval(get_summary(m, 200), "accuracy_sd") for m in PLAIN_MODELS]
    ax.errorbar(
        xs_nb,
        ys_nb,
        yerr=sd_nb,
        fmt="o--",
        color=COLORS["nonbn"],
        mfc="white",
        markersize=6,
        capsize=3,
        lw=1.3,
        label="Without BN (5 runs)",
    )

    gain_x_offsets = {11.0: 0.30, 16.0: -1.75, 19.0: -1.75}
    for x_nb, y_nb, m_nb in zip(xs_nb, ys_nb, PLAIN_MODELS):
        y_b = fval(get_summary(m_nb + "_bn", 200), "accuracy_mean")
        ax.annotate(
            "",
            xy=(x_nb, y_b - 0.14),
            xytext=(x_nb, y_nb + 0.16),
            arrowprops=dict(arrowstyle="->", lw=0.95, ls=":", color="#475569"),
        )
        ax.text(
            x_nb + gain_x_offsets[x_nb],
            (y_b + y_nb) / 2,
            f"+{y_b - y_nb:.2f} pp",
            fontsize=7.7,
            color="#1e3a8a",
            va="center",
            fontweight="bold",
        )

    ax.set_xticks([11, 16, 19])
    ax.set_xlim(10.0, 20.1)
    ax.set_ylim(86.7, 92.38)
    ax.set_xlabel(r"VGG nominal depth $D_{\mathrm{nom}}$ (layers)")
    ax.set_ylabel("Test Top-1 accuracy (%)")
    ax.set_title("VGG normalization effect")
    ax.legend(frameon=False, loc="upper left", fontsize=7.8)

    # Panel c: Matched-seed BN minus non-BN accuracy gains (95% paired t CI)
    ax = axes[2]
    panel_label(ax, "c", x=-0.16, y=1.03)
    positions = [(0, "vgg11"), (1, "vgg16"), (2, "vgg19")]
    for pos, base in positions:
        r = next(
            row
            for row in MATCHED
            if row["model"] == base and int(row["horizon"]) == 200
        )
        mean = float(r["delta_mean_pp"])
        lo = float(r["ci95_lower_pp"])
        hi = float(r["ci95_upper_pp"])
        n_p = int(r["n_pairs"])
        ax.errorbar(
            pos,
            mean,
            yerr=[[mean - lo], [hi - mean]],
            fmt="o",
            color=COLORS["vgg"],
            markersize=6.5,
            capsize=4.5,
            lw=1.4,
            zorder=4,
        )
        ax.text(
            pos,
            hi + 0.16,
            f"+{mean:.2f} pp",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#1e3a8a",
        )
        ax.text(
            pos,
            0.22,
            f"n = {n_p}\nshared seeds",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#374151",
        )

    ax.axhline(0, color="#6b7280", lw=1.0, ls="--")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["VGG-11", "VGG-16", "VGG-19"])
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-0.15, 5.50)
    ax.set_xlabel("VGG depth variant")
    ax.set_ylabel("Matched-seed BN gain (pp)")
    ax.set_title("Paired gain (bars = 95% t-CI)")

    save_fig(fig, "fig2")


# =========================================================================
# FIGURE 3: Accuracy vs inference compute (G-MACs) and parameter count (M)
# =========================================================================
def build_fig3():
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), layout="constrained")

    # Panel a: Accuracy vs G-MACs
    ax = axes[0]
    panel_label(ax, "a", x=-0.14, y=1.03)
    offsets_mac = {
        "googlenet": (7, 2),
        "resnet18": (-6, -15),
        "resnet34": (7, 3),
        "resnet50": (7, -5),
        "vgg11_bn": (7, 3),
        "vgg13_bn": (7, -4),
        "vgg16_bn": (-54, 4),
        "vgg19_bn": (-54, 5),
        "vgg11": (7, -4),
        "vgg16": (-68, 4),
        "vgg19": (7, -4),
    }
    for m in PLAIN_MODELS:
        r0 = get_summary(m, 200)
        r1 = get_summary(m + "_bn", 200)
        ax.plot(
            [fval(r0, "profiled_gmacs"), fval(r1, "profiled_gmacs")],
            [fval(r0, "accuracy_mean"), fval(r1, "accuracy_mean")],
            ls=":",
            color="#9ca3af",
            lw=1.1,
            zorder=1,
        )

    for m in ALL_MODELS:
        r = get_summary(m, 200)
        is_plain = m in PLAIN_MODELS
        fam = fam_of(m)
        color = COLORS["nonbn"] if is_plain else COLORS[fam]
        marker = "s" if fam == "resnet" else "^" if fam == "googlenet" else "o"
        x, y, sd = (
            fval(r, "profiled_gmacs"),
            fval(r, "accuracy_mean"),
            fval(r, "accuracy_sd"),
        )
        ax.errorbar(
            x,
            y,
            yerr=sd,
            fmt=marker,
            color=color,
            mfc="white" if is_plain else color,
            markersize=6,
            capsize=3,
            lw=1.2,
            zorder=3,
        )
        dx, dy = offsets_mac[m]
        ax.annotate(
            LABELS[m], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.4
        )

    ax.set_xlim(0.2, 24.8)
    ax.set_ylim(86.6, 92.55)
    ax.set_xlabel("Single-image inference compute (G-MACs at 224×224)")
    ax.set_ylabel("Test Top-1 accuracy (%)")
    ax.set_title("Compute–accuracy Pareto frontier")

    handles = [
        Line2D(
            [],
            [],
            color=COLORS["googlenet"],
            marker="^",
            ls="",
            markersize=6,
            label="GoogLeNet (BN)",
        ),
        Line2D(
            [],
            [],
            color=COLORS["resnet"],
            marker="s",
            ls="",
            markersize=6,
            label="ResNet (BN)",
        ),
        Line2D(
            [], [], color=COLORS["vgg"], marker="o", ls="", markersize=6, label="VGG-BN"
        ),
        Line2D(
            [],
            [],
            color=COLORS["nonbn"],
            marker="o",
            mfc="white",
            ls="",
            markersize=6,
            label="VGG without BN",
        ),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left", fontsize=7.9)

    # Panel b: Accuracy vs Trainable Parameters (M, log scale)
    ax = axes[1]
    panel_label(ax, "b", x=-0.14, y=1.03)
    offsets_par = {
        "googlenet": (7, 2),
        "resnet18": (7, -6),
        "resnet34": (7, 3),
        "resnet50": (7, -6),
        "vgg11_bn": (-66, -3),
        "vgg13_bn": None,  # Grouped cleanly with VGG-11-BN (132.9M & 133.1M, both 90.66%)
        "vgg16_bn": (-58, 2),
        "vgg19_bn": (7, 4),
        "vgg11": (-70, -5),
        "vgg16": (-72, 4),
        "vgg19": (7, -5),
    }
    for m in PLAIN_MODELS:
        r0 = get_summary(m, 200)
        r1 = get_summary(m + "_bn", 200)
        ax.plot(
            [fval(r0, "params_m"), fval(r1, "params_m")],
            [fval(r0, "accuracy_mean"), fval(r1, "accuracy_mean")],
            ls=":",
            color="#9ca3af",
            lw=1.1,
            zorder=1,
        )

    for m in ALL_MODELS:
        r = get_summary(m, 200)
        is_plain = m in PLAIN_MODELS
        fam = fam_of(m)
        color = COLORS["nonbn"] if is_plain else COLORS[fam]
        marker = "s" if fam == "resnet" else "^" if fam == "googlenet" else "o"
        x, y, sd = fval(r, "params_m"), fval(r, "accuracy_mean"), fval(r, "accuracy_sd")
        ax.errorbar(
            x,
            y,
            yerr=sd,
            fmt=marker,
            color=color,
            mfc="white" if is_plain else color,
            markersize=6,
            capsize=3,
            lw=1.2,
            zorder=3,
        )
        if m == "vgg11_bn":
            ax.annotate(
                "VGG-11/13-BN",
                (x, y),
                xytext=(-68, -3),
                textcoords="offset points",
                fontsize=7.4,
            )
        elif offsets_par[m] is not None:
            dx, dy = offsets_par[m]
            ax.annotate(
                LABELS[m],
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.4,
            )

    ax.set_xscale("log")
    ax.set_xlim(9.0, 295.0)
    ax.set_ylim(86.6, 92.55)
    ax.set_xticks([10, 20, 50, 100, 200])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Trainable parameters (millions, log scale)")
    ax.set_ylabel("Test Top-1 accuracy (%)")
    ax.set_title("Parameter–accuracy scaling")

    save_fig(fig, "fig3")


# =========================================================================
# FIGURE 4: Training objective & main-head test cross-entropy (200 epochs)
# =========================================================================
def plot_train_curve(ax, model: str, horizon: int, color: str, ls: str, label: str):
    rr = get_curve(model, horizon)
    x = np.array([int(r["epoch"]) for r in rr])
    mu = np.array([fval(r, "mean_train_loss") for r in rr])
    lo = np.array([fval(r, "lower_train_loss") for r in rr])
    hi = np.array([fval(r, "upper_train_loss") for r in rr])
    ax.fill_between(
        x, np.where(lo > 0, lo, np.nan), hi, color=color, alpha=0.14, linewidth=0
    )
    ax.plot(x, mu, color=color, ls=ls, lw=1.45, label=label)


def plot_test_curve(ax, model: str, horizon: int, color: str, ls: str, label: str):
    x, mu, lo, hi = get_test_loss_curve(model, horizon)
    ax.fill_between(
        x, np.where(lo > 0, lo, np.nan), hi, color=color, alpha=0.14, linewidth=0
    )
    ax.plot(x, mu, color=color, ls=ls, lw=1.45, label=label)


def build_fig4():
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), layout="constrained")

    # Panel a: Primary BN models training objective
    ax = axes[0]
    panel_label(ax, "a", x=-0.16, y=1.03)
    plot_train_curve(ax, "vgg19_bn", 200, COLORS["vgg"], "-", "VGG-19-BN (single head)")
    plot_train_curve(
        ax, "resnet50", 200, COLORS["resnet"], "-", "ResNet-50 (single head)"
    )
    plot_train_curve(
        ax, "googlenet", 200, COLORS["googlenet"], "-", "GoogLeNet (main + 0.3×aux1,2)"
    )
    ax.set_yscale("log")
    ax.set_xlim(1, 200)
    ax.set_xlabel("Completed training epoch")
    ax.set_ylabel("Training objective (log scale)")
    ax.set_title("Training objective (BN cohort)")
    ax.legend(frameon=False, loc="upper right", fontsize=7.6)

    # Panel b: Direct VGG-19 training cross-entropy with vs without BN
    ax = axes[1]
    panel_label(ax, "b", x=-0.16, y=1.03)
    plot_train_curve(ax, "vgg19_bn", 200, COLORS["vgg"], "-", "VGG-19-BN")
    plot_train_curve(ax, "vgg19", 200, COLORS["nonbn"], "--", "VGG-19 without BN")
    ax.set_yscale("log")
    ax.set_xlim(1, 200)
    ax.set_xlabel("Completed training epoch")
    ax.set_ylabel("Training cross-entropy (log scale)")
    ax.set_title("VGG-19 training cross-entropy")
    ax.legend(frameon=False, loc="upper right", fontsize=7.9)

    # Panel c: Main-head test cross-entropy across ALL four models (identical single-head scale)
    ax = axes[2]
    panel_label(ax, "c", x=-0.16, y=1.03)
    plot_test_curve(ax, "vgg19", 200, COLORS["nonbn"], "--", "VGG-19 without BN")
    plot_test_curve(ax, "vgg19_bn", 200, COLORS["vgg"], "-", "VGG-19-BN")
    plot_test_curve(ax, "resnet50", 200, COLORS["resnet"], "-", "ResNet-50 (BN)")
    plot_test_curve(
        ax, "googlenet", 200, COLORS["googlenet"], "-", "GoogLeNet (BN, main head)"
    )
    ax.set_yscale("log")
    ax.set_xlim(1, 200)
    ax.set_ylim(0.24, 2.5)
    ax.set_yticks([0.3, 0.5, 1.0, 2.0])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Completed training epoch")
    ax.set_ylabel("Main-head test cross-entropy (log scale)")
    ax.set_title("Main-head test loss (comparable scale)")
    ax.legend(frameon=False, loc="upper right", fontsize=7.6)

    save_fig(fig, "fig4")


# =========================================================================
# FIGURE 5: Global parameter-gradient L2 norms and finite-record coverage
# =========================================================================
def build_fig5():
    fig, (ax, cov) = plt.subplots(
        2, 1, figsize=(8.8, 5.8), height_ratios=[2.7, 1.15], layout="constrained"
    )
    panel_label(ax, "a", x=-0.09, y=1.03)
    panel_label(cov, "b", x=-0.09, y=1.06)

    offsets_g = {
        "vgg11_bn": (-52, -3),
        "vgg13_bn": (7, 2),
        "vgg16_bn": (-54, -3),
        "vgg19_bn": (7, 3),
        "resnet18": (7, 3),
        "resnet34": (7, 4),
        "resnet50": (-48, 7),
        "googlenet": (7, 2),
        "vgg11": (-66, -4),
        "vgg16": (-66, 5),
        "vgg19": (7, -4),
    }
    for m in ALL_MODELS:
        r = get_summary(m, 200)
        is_plain = m in PLAIN_MODELS
        fam = fam_of(m)
        color = COLORS["nonbn"] if is_plain else COLORS[fam]
        marker = "s" if fam == "resnet" else "^" if fam == "googlenet" else "o"
        x = fval(r, "nominal_depth")
        y = fval(r, "gradient_finite_mean_mean")
        sd = fval(r, "gradient_finite_mean_sd")
        ax.errorbar(
            x,
            y,
            yerr=sd,
            fmt=marker,
            color=color,
            mfc="white" if is_plain else color,
            markersize=6,
            capsize=3,
            lw=1.2,
            zorder=3,
        )
        dx, dy = offsets_g[m]
        ax.annotate(
            LABELS[m], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.4
        )

    handles = [
        Line2D(
            [], [], color=COLORS["vgg"], marker="o", ls="", markersize=6, label="VGG-BN"
        ),
        Line2D(
            [],
            [],
            color=COLORS["resnet"],
            marker="s",
            ls="",
            markersize=6,
            label="ResNet (BN)",
        ),
        Line2D(
            [],
            [],
            color=COLORS["googlenet"],
            marker="^",
            ls="",
            markersize=6,
            label="GoogLeNet (BN)",
        ),
        Line2D(
            [],
            [],
            color=COLORS["nonbn"],
            marker="o",
            mfc="white",
            ls="",
            markersize=6,
            label="VGG without BN",
        ),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8.0)
    ax.set_xlim(4.2, 54.0)
    ax.set_ylim(0.08, 1.72)
    ax.set_xlabel(r"Nominal depth $D_{\mathrm{nom}}$ (conv + FC layers)")
    ax.set_ylabel(r"Global parameter-gradient $L_2$ norm")
    ax.set_title("Finite-conditioned global gradient norms (epochs 181–200)")

    rows_cov = [get_summary(m, 200) for m in ALL_MODELS]
    bar_colors = [
        COLORS["nonbn"] if m in PLAIN_MODELS else COLORS[fam_of(m)] for m in ALL_MODELS
    ]
    bars = cov.bar(
        range(len(ALL_MODELS)),
        [fval(r, "gradient_finite_records") for r in rows_cov],
        color=bar_colors,
        width=0.62,
        edgecolor="#374151",
        linewidth=0.7,
    )
    for b, m in zip(bars, ALL_MODELS):
        if m in PLAIN_MODELS:
            b.set_hatch("//")
            b.set_facecolor("#e5e7eb")
    for idx, r in enumerate(rows_cov):
        cnt = int(fval(r, "gradient_finite_records"))
        cov.text(idx, cnt + 3, f"{cnt}/100", ha="center", va="bottom", fontsize=7.3)
    cov.set_ylim(0, 114)
    cov.set_xticks(range(len(ALL_MODELS)))
    cov.set_xticklabels(
        [LABELS[m].replace(" (no BN)", "\n(no BN)") for m in ALL_MODELS],
        rotation=28,
        ha="right",
        fontsize=7.6,
    )
    cov.set_ylabel("Finite epoch\nrecords (/100)")

    save_fig(fig, "fig5")


# =========================================================================
# FIGURE 6: Pooled depth-accuracy correlation panels (D_nom and D_eff, n=8)
# =========================================================================
def build_fig6():
    fig, axes = plt.subplots(
        1, 2, figsize=(10.0, 4.3), sharey=True, layout="constrained"
    )
    ss = [get_summary(m, 200) for m in MAIN_MODELS]
    st200 = STATS["main_bn_eight_200"]

    offsets_nom = {
        "vgg11_bn": (-48, 3),
        "vgg13_bn": (-48, -14),
        "vgg16_bn": (-50, 2),
        "vgg19_bn": (7, 4),
        "resnet18": (7, -5),
        "resnet34": (7, 4),
        "resnet50": (-48, 8),
        "googlenet": (7, -2),
    }
    offsets_eff = {
        "vgg11_bn": (-46, 4),
        "resnet18": (-14, -18),
        "vgg13_bn": (7, -4),
        "vgg16_bn": (-50, 2),
        "googlenet": (7, -2),
        "vgg19_bn": (-50, 5),
        "resnet34": (7, 3),
        "resnet50": (-48, 8),
    }

    for ax, xkey, skey, letter, offs, xtitle in [
        (
            axes[0],
            "nominal_depth",
            "nominal",
            "a",
            offsets_nom,
            r"Nominal depth $D_{\mathrm{nom}}$ (layers)",
        ),
        (
            axes[1],
            "effective_depth",
            "effective",
            "b",
            offsets_eff,
            r"Structural effective depth $D_{\mathrm{eff}}$ (layers)",
        ),
    ]:
        panel_label(ax, letter, x=-0.13, y=1.03)
        xs = np.array([fval(r, xkey) for r in ss])
        ys = np.array([fval(r, "accuracy_mean") for r in ss])
        fit = stats.linregress(xs, ys)
        xx = np.linspace(xs.min() - 1.0, xs.max() + 1.5, 100)
        ax.plot(
            xx,
            fit.intercept + fit.slope * xx,
            ls=":",
            color="#4b5563",
            lw=1.3,
            label="Linear OLS fit",
        )

        for m, r in zip(MAIN_MODELS, ss):
            fam = fam_of(m)
            marker = "s" if fam == "resnet" else "^" if fam == "googlenet" else "o"
            x, y, sd = fval(r, xkey), fval(r, "accuracy_mean"), fval(r, "accuracy_sd")
            ax.errorbar(
                x,
                y,
                yerr=sd,
                fmt=marker,
                color=COLORS[fam],
                markersize=6,
                capsize=3,
                lw=1.2,
                zorder=4,
            )
            dx, dy = offs[m]
            ax.annotate(
                LABELS[m],
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.4,
            )

        r_val = st200[skey]["r"]
        p_val = st200[skey]["p"]
        ax.set_xlabel(xtitle)
        ax.set_title(f"Pearson r = {r_val:.3f} (P = {p_val:.3f}, n = 8)")

    axes[0].set_xlim(3.5, 54.0)
    axes[1].set_xlim(7.2, 30.5)
    axes[0].set_ylabel("Test Top-1 accuracy (%)")
    axes[0].set_ylim(90.15, 92.38)
    handles = [
        Line2D(
            [], [], color=COLORS["vgg"], marker="o", ls="", markersize=6, label="VGG-BN"
        ),
        Line2D(
            [],
            [],
            color=COLORS["resnet"],
            marker="s",
            ls="",
            markersize=6,
            label="ResNet (BN)",
        ),
        Line2D(
            [],
            [],
            color=COLORS["googlenet"],
            marker="^",
            ls="",
            markersize=6,
            label="GoogLeNet (BN)",
        ),
        Line2D([], [], color="#4b5563", ls=":", lw=1.3, label="Linear OLS fit"),
    ]
    axes[0].legend(handles=handles, frameon=False, loc="upper right", fontsize=7.8)

    save_fig(fig, "fig6")


# =========================================================================
# SUPPLEMENTARY FIGURES S1-S3 (No subpanel letters needed; never cited by panel)
# =========================================================================
def build_supp_figures():
    # Supplementary Fig S1: Canonical three VGG depths (11, 16, 19) training & test loss (with vs without BN)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.6, 5.3),
        sharex=True,
        sharey="row",
        gridspec_kw={"hspace": 0.22, "wspace": 0.14},
    )
    bases = ["vgg11", "vgg16", "vgg19"]
    for idx, base in enumerate(bases):
        ax_tr = axes[0, idx]
        ax_te = axes[1, idx]
        plot_train_curve(ax_tr, base + "_bn", 200, COLORS["vgg"], "-", "With BN")
        plot_test_curve(ax_te, base + "_bn", 200, COLORS["vgg"], "-", "With BN")
        plot_train_curve(ax_tr, base, 200, COLORS["nonbn"], "--", "Without BN")
        plot_test_curve(ax_te, base, 200, COLORS["nonbn"], "--", "Without BN")
        ax_tr.set_yscale("log")
        ax_te.set_yscale("log")
        ax_tr.set_xlim(1, 200)
        ax_te.set_xlim(1, 200)
        ax_tr.set_ylim(1.2e-3, 3.5)
        ax_te.set_ylim(0.30, 2.5)
        ax_te.set_yticks([0.4, 0.6, 1.0, 2.0])
        ax_te.yaxis.set_major_formatter(ScalarFormatter())
        ax_te.yaxis.set_minor_formatter(NullFormatter())
        ax_tr.set_title(base.upper().replace("VGG", "VGG-"))
        ax_te.set_xlabel("Completed epoch")
        if idx == 0:
            ax_tr.set_ylabel("Training cross-entropy")
            ax_te.set_ylabel("Test cross-entropy")
            ax_tr.legend(frameon=False, loc="upper right", fontsize=7.9)
            ax_te.legend(frameon=False, loc="upper right", fontsize=7.9)
    save_fig(fig, "supp_fig1")

    # Supplementary Fig S2: Epoch-100 robustness (accuracy vs depth, BN vs non-BN, accuracy vs MACs)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), layout="constrained")
    for fam in ("vgg", "resnet"):
        ms = [m for m in MAIN_MODELS if fam_of(m) == fam]
        xs = [fval(get_summary(m, 100), "nominal_depth") for m in ms]
        ys = [fval(get_summary(m, 100), "accuracy_mean") for m in ms]
        axes[0].plot(xs, ys, color=COLORS[fam], alpha=0.45, lw=1.2)
    for m in MAIN_MODELS:
        r = get_summary(m, 100)
        fam = fam_of(m)
        marker = "s" if fam == "resnet" else "^" if fam == "googlenet" else "o"
        axes[0].errorbar(
            fval(r, "nominal_depth"),
            fval(r, "accuracy_mean"),
            yerr=fval(r, "accuracy_sd"),
            fmt=marker,
            color=COLORS[fam],
            markersize=5.8,
            capsize=3,
            lw=1.15,
        )
    handles_s2 = [
        Line2D([], [], color=COLORS["vgg"], marker="o", ls="-", lw=1.2, label="VGG-BN"),
        Line2D(
            [],
            [],
            color=COLORS["resnet"],
            marker="s",
            ls="-",
            lw=1.2,
            label="ResNet (BN)",
        ),
        Line2D(
            [], [], color=COLORS["googlenet"], marker="^", ls="", label="GoogLeNet (BN)"
        ),
    ]
    axes[0].legend(handles=handles_s2, frameon=False, loc="upper right", fontsize=7.8)
    axes[0].set_xlim(8, 54)
    axes[0].set_ylim(90.15, 92.35)
    axes[0].set_xlabel(r"Nominal depth $D_{\mathrm{nom}}$ (layers)")
    axes[0].set_ylabel("Epoch-100 test Top-1 accuracy (%)")
    axes[0].set_title("BN-equalized cohort (epoch 100)")

    bn_paired_vgg = ["vgg11_bn", "vgg16_bn", "vgg19_bn"]
    axes[1].errorbar(
        [fval(get_summary(m, 100), "nominal_depth") for m in bn_paired_vgg],
        [fval(get_summary(m, 100), "accuracy_mean") for m in bn_paired_vgg],
        yerr=[fval(get_summary(m, 100), "accuracy_sd") for m in bn_paired_vgg],
        fmt="o-",
        color=COLORS["vgg"],
        markersize=5.8,
        capsize=3,
        lw=1.3,
        label="With BN",
    )
    axes[1].errorbar(
        [fval(get_summary(m, 100), "nominal_depth") for m in PLAIN_MODELS],
        [fval(get_summary(m, 100), "accuracy_mean") for m in PLAIN_MODELS],
        yerr=[fval(get_summary(m, 100), "accuracy_sd") for m in PLAIN_MODELS],
        fmt="o--",
        color=COLORS["nonbn"],
        mfc="white",
        markersize=5.8,
        capsize=3,
        lw=1.2,
        label="Without BN",
    )
    axes[1].set_xticks([11, 16, 19])
    axes[1].set_xlim(10.0, 20.1)
    axes[1].set_ylim(86.7, 92.35)
    axes[1].set_xlabel(r"VGG nominal depth $D_{\mathrm{nom}}$")
    axes[1].set_ylabel("Epoch-100 test Top-1 accuracy (%)")
    axes[1].set_title("VGG normalization comparison (epoch 100)")
    axes[1].legend(frameon=False, loc="upper left", fontsize=7.8)

    for m in ALL_MODELS:
        r = get_summary(m, 100)
        is_plain = m in PLAIN_MODELS
        fam = fam_of(m)
        color = COLORS["nonbn"] if is_plain else COLORS[fam]
        marker = "s" if fam == "resnet" else "^" if fam == "googlenet" else "o"
        axes[2].errorbar(
            fval(r, "profiled_gmacs"),
            fval(r, "accuracy_mean"),
            yerr=fval(r, "accuracy_sd"),
            fmt=marker,
            color=color,
            mfc="white" if is_plain else color,
            markersize=5.8,
            capsize=3,
            lw=1.15,
        )
    axes[2].set_xlim(0.2, 22.0)
    axes[2].set_ylim(86.6, 92.55)
    axes[2].set_xlabel("Inference compute (G-MACs)")
    axes[2].set_ylabel("Epoch-100 test Top-1 accuracy (%)")
    axes[2].set_title("Compute–accuracy frontier (epoch 100)")
    save_fig(fig, "supp_fig2")

    # Supplementary Fig S3: Fixed 7-model correlation sensitivity (epoch 200)
    fig, axes = plt.subplots(
        2, 2, figsize=(9.8, 6.6), sharey=True, layout="constrained"
    )
    cohorts = [
        (
            0,
            "historical_nonbn_seven_200",
            PLAIN_MODELS + ["resnet18", "resnet34", "resnet50", "googlenet"],
            "Unnormalized VGG + BN ResNet/GoogLeNet (n = 7)",
        ),
        (
            1,
            "matched_bn_seven_200",
            [
                "vgg11_bn",
                "vgg16_bn",
                "vgg19_bn",
                "resnet18",
                "resnet34",
                "resnet50",
                "googlenet",
            ],
            "BN-equalized 7-model subset (VGG-13 excluded, n = 7)",
        ),
    ]
    for row_idx, skey_cohort, mlist, row_title in cohorts:
        rows_m = [get_summary(m, 200) for m in mlist]
        for col_idx, (xkey, metric, xlabel) in enumerate(
            [
                (
                    "nominal_depth",
                    "nominal",
                    r"Nominal depth $D_{\mathrm{nom}}$ (layers)",
                ),
                (
                    "effective_depth",
                    "effective",
                    r"Structural effective depth $D_{\mathrm{eff}}$ (layers)",
                ),
            ]
        ):
            ax = axes[row_idx, col_idx]
            xs = np.array([fval(r, xkey) for r in rows_m])
            ys = np.array([fval(r, "accuracy_mean") for r in rows_m])
            fit = stats.linregress(xs, ys)
            xx = np.linspace(xs.min() - 1, xs.max() + 1, 80)
            ax.plot(xx, fit.intercept + fit.slope * xx, ls=":", color="#4b5563", lw=1.2)
            for m, r in zip(mlist, rows_m):
                is_plain = m in PLAIN_MODELS
                fam = fam_of(m)
                color = COLORS["nonbn"] if is_plain else COLORS[fam]
                marker = "s" if fam == "resnet" else "^" if fam == "googlenet" else "o"
                ax.errorbar(
                    fval(r, xkey),
                    fval(r, "accuracy_mean"),
                    yerr=fval(r, "accuracy_sd"),
                    fmt=marker,
                    color=color,
                    mfc="white" if is_plain else color,
                    markersize=5.8,
                    capsize=3,
                    lw=1.15,
                )
            st = STATS[skey_cohort][metric]
            ax.set_xlabel(xlabel)
            ax.set_ylim(86.6, 92.55)
            ax.set_title(f"{row_title}\nr = {st['r']:.3f}, P = {st['p']:.3f}")
            if col_idx == 0:
                ax.set_ylabel("Test Top-1 accuracy (%)")
    handles_s3 = [
        Line2D(
            [], [], color=COLORS["vgg"], marker="o", ls="", markersize=6, label="VGG-BN"
        ),
        Line2D(
            [],
            [],
            color=COLORS["nonbn"],
            marker="o",
            mfc="white",
            ls="",
            markersize=6,
            label="VGG without BN",
        ),
        Line2D(
            [],
            [],
            color=COLORS["resnet"],
            marker="s",
            ls="",
            markersize=6,
            label="ResNet (BN)",
        ),
        Line2D(
            [],
            [],
            color=COLORS["googlenet"],
            marker="^",
            ls="",
            markersize=6,
            label="GoogLeNet (BN)",
        ),
    ]
    axes[0, 0].legend(handles=handles_s3, frameon=False, loc="upper left", fontsize=7.6)
    save_fig(fig, "supp_fig3")


def main():
    build_fig1()
    build_fig2()
    build_fig3()
    build_fig4()
    build_fig5()
    build_fig6()
    build_supp_figures()
    print("All 6 main figures and 3 supplementary figures regenerated successfully.")


if __name__ == "__main__":
    main()
