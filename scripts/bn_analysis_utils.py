"""
Deterministic statistical helpers for endpoint summaries, non-parametric
Mann-Whitney U tests, matched-seed paired t-tests, Holm step-down adjustments,
exact sign-flip permutation bounds, and cross-architecture correlations.
"""

import itertools
import math
import statistics
from collections import defaultdict
import numpy as np
from scipy import stats


def moments(values):
    values = list(values)
    return statistics.mean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def matched_differences(baseline, bn, metric: str = "accuracy"):
    a = defaultdict(list)
    b = defaultdict(list)
    for r in baseline:
        a[int(r["seed"])].append(float(r[metric]))
    for r in bn:
        b[int(r["seed"])].append(float(r[metric]))
    return [
        dict(
            seed=seed,
            baseline=statistics.mean(a[seed]),
            bn=statistics.mean(b[seed]),
            difference=statistics.mean(b[seed]) - statistics.mean(a[seed]),
            baseline_runs_averaged=len(a[seed]),
            bn_runs_averaged=len(b[seed]),
        )
        for seed in sorted(a.keys() & b.keys())
    ]


def holm(pvalues):
    out = [None] * len(pvalues)
    previous = 0.0
    for rank, idx in enumerate(sorted(range(len(pvalues)), key=lambda i: pvalues[i])):
        previous = max(previous, min(1.0, pvalues[idx] * (len(pvalues) - rank)))
        out[idx] = previous
    return out


def signflip(differences):
    a = np.asarray(differences, dtype=float)
    observed = abs(float(a.mean()))
    null = [
        abs(float(np.mean(a * np.asarray(signs))))
        for signs in itertools.product((-1, 1), repeat=len(a))
    ]
    return sum(x >= observed - 1e-12 for x in null) / len(null)


def paired_summary(differences):
    a = np.asarray(differences, dtype=float)
    n = len(a)
    mean, sd = moments(a)
    if sd == 0:
        p = 1.0 if mean == 0 else 0.0
        half = 0.0
    else:
        p = float(stats.ttest_1samp(a, 0.0).pvalue)
        half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return dict(
        n_pairs=n,
        delta_mean_pp=mean,
        delta_sample_sd_pp=sd,
        ci95_lower_pp=mean - half,
        ci95_upper_pp=mean + half,
        p_raw=p,
        p_exact_signflip=signflip(a),
    )


def exact_mannwhitney(group_a, group_b):
    u_stat, p_val = stats.mannwhitneyu(
        group_a, group_b, alternative="two-sided", method="exact"
    )
    return dict(u_stat=float(u_stat), p_exact=float(p_val))


def correlation(x, y):
    res = stats.pearsonr(x, y)
    return dict(n_models=len(x), r=float(res.statistic), p=float(res.pvalue))


def partial_correlation(x, y, z):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    design = np.column_stack([np.ones(len(z)), z])
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    r = float(stats.pearsonr(rx, ry).statistic)
    df = len(x) - 3
    p = float(2.0 * stats.t.sf(abs(r) * np.sqrt(df / (1.0 - r * r)), df))
    return dict(n_models=len(x), df=df, r=r, p=p)
