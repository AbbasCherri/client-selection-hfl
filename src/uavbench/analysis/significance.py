"""Paired statistical significance tests over multi-seed run tables.

The seed design guarantees every method sees an *identical* problem
instance per seed (Tier-1's shared instance stream; the sweeps'
method-free ``sweep_job_seed``), so per-seed metric values are paired
samples and a paired test (Wilcoxon signed-rank by default, paired t-test
optionally) is the correct — and stronger — choice over an unpaired test.
State this pairing property explicitly in the paper as the justification.

Input is a final-round-per-seed table (one row per (method, group, seed)),
typically derived from ``runs.parquet`` / ``*_rounds.parquet`` outputs.
Multiple comparisons across method pairs are corrected with Holm-Bonferroni
(implemented inline; scipy is the only dependency).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

_TESTS = ("wilcoxon", "ttest_rel")


def _rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (Kerby 2014) — Wilcoxon effect size.

    Rank the nonzero |differences|; ``r = (W+ - W-) / (W+ + W-)`` where W+/W- are
    the summed ranks of positive/negative differences. Range [-1, 1]; sign matches
    ``mean_diff``. Returns 0.0 when every pair is tied.
    """
    nz = diff[diff != 0.0]
    if nz.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    w_pos = ranks[nz > 0].sum()
    w_neg = ranks[nz < 0].sum()
    total = w_pos + w_neg
    return float((w_pos - w_neg) / total) if total > 0 else 0.0


def _bootstrap_ci(
    diff: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean paired difference.

    Resamples the paired differences with replacement (the pairing is preserved
    because we resample the diff vector itself). Deterministic given ``seed``.
    """
    if diff.size == 0:
        return (float("nan"), float("nan"))
    if np.allclose(diff, 0.0):
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def _paired_values(
    df: pd.DataFrame,
    metric: str,
    method_a: str,
    method_b: str,
    group_cols: list[str],
    seed_col: str,
) -> list[tuple[tuple, np.ndarray, np.ndarray]]:
    """Per group: seed-aligned metric arrays for the two methods.

    Raises if the two methods' seed sets differ within any group — pairing
    is only valid when both methods ran the identical seeds.
    """
    out = []
    for key, sub in df.groupby(list(group_cols)):
        a = sub[sub["method"] == method_a].set_index(seed_col)[metric].sort_index()
        b = sub[sub["method"] == method_b].set_index(seed_col)[metric].sort_index()
        if list(a.index) != list(b.index):
            raise ValueError(
                f"seed sets differ for {method_a} vs {method_b} in group {key}: "
                f"{list(a.index)} vs {list(b.index)} — pairing invalid"
            )
        if a.index.has_duplicates:
            raise ValueError(
                f"duplicate seeds for group {key} — aggregate to one row per "
                "(method, group, seed) before testing"
            )
        out.append((key if isinstance(key, tuple) else (key,), a.to_numpy(), b.to_numpy()))
    return out


def paired_seed_test(
    df: pd.DataFrame,
    metric: str,
    method_a: str,
    method_b: str,
    group_cols: list[str] | tuple[str, ...] = ("scenario",),
    seed_col: str = "seed",
    test: str = "wilcoxon",
) -> pd.DataFrame:
    """Paired test of ``method_a`` vs ``method_b`` per group.

    One row per group: ``{**group, n_pairs, statistic, p_value, mean_diff,
    method_a, method_b}``. ``mean_diff > 0`` means ``method_a`` scores
    higher on ``metric``. A Wilcoxon with all-zero differences (methods
    literally tied on every seed) reports ``p_value = 1.0``.
    """
    if test not in _TESTS:
        raise ValueError(f"test must be one of {_TESTS}; got {test!r}")

    rows = []
    for key, a, b in _paired_values(df, metric, method_a, method_b, list(group_cols), seed_col):
        diff = a - b
        if test == "wilcoxon":
            if np.allclose(diff, 0.0):
                statistic, p = 0.0, 1.0
            else:
                statistic, p = stats.wilcoxon(a, b)
        else:
            statistic, p = stats.ttest_rel(a, b)
        ci_low, ci_high = _bootstrap_ci(diff)
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "method_a": method_a,
                "method_b": method_b,
                "n_pairs": len(a),
                "statistic": float(statistic),
                "p_value": float(p),
                "mean_diff": float(diff.mean()),
                "effect_size": _rank_biserial(diff),  # matched-pairs rank-biserial
                "ci_low": ci_low,                      # 95% bootstrap CI on mean_diff
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def holm_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Holm-Bonferroni step-down: boolean reject mask at family level alpha."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    reject = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        if p[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break  # step-down stops at the first non-rejection
    return reject


def pairwise_significance_table(
    df: pd.DataFrame,
    metric: str,
    methods: list[str],
    group_cols: list[str] | tuple[str, ...] = ("scenario",),
    seed_col: str = "seed",
    test: str = "wilcoxon",
    alpha: float = 0.05,
    correction: str = "holm",
    reference: str | None = None,
    correction_scope: str = "table",
) -> pd.DataFrame:
    """Paired tests across ``methods``, multiplicity-corrected.

    ``reference`` — when given, only ``reference`` vs each other method is
    tested (M-1 comparisons) instead of all M(M-1)/2 pairs. This is both the
    question a system paper actually asks ("does the proposed method beat each
    baseline?") and a decisive power matter: a paired Wilcoxon on ``n`` seeds
    cannot return a p below ``2/2**n`` (n=10 → 0.00195), so an all-pairs family
    large enough to push the Holm threshold under that floor makes *every*
    comparison structurally non-significant regardless of effect size. With 13
    methods × 4 groups the all-pairs family is 312 and the threshold 1.6e-4 —
    unreachable at 10 seeds. Reference-only within one group is 12 comparisons
    (threshold 4.2e-3), which the floor clears.

    ``correction_scope`` — ``"table"`` corrects over every row (most
    conservative); ``"group"`` corrects within each ``group_cols`` combination,
    appropriate when each group is a separate experiment/figure panel rather
    than one joint family.

    ``significant`` reflects the corrected decision at ``alpha``; raw p-values,
    effect sizes and CIs are kept alongside for transparency.
    """
    if correction not in ("holm", "none"):
        raise ValueError(f"correction must be 'holm' or 'none'; got {correction!r}")
    if correction_scope not in ("table", "group"):
        raise ValueError(f"correction_scope must be 'table' or 'group'; got {correction_scope!r}")
    if reference is not None and reference not in methods:
        raise ValueError(f"reference {reference!r} not among methods {methods}")

    if reference is not None:
        pairs = [(reference, m) for m in methods if m != reference]
    else:
        pairs = list(itertools.combinations(methods, 2))

    parts = [
        paired_seed_test(df, metric, a, b, group_cols=group_cols, seed_col=seed_col, test=test)
        for a, b in pairs
    ]
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if table.empty:
        return table

    if correction == "holm":
        if correction_scope == "group" and list(group_cols):
            table["significant"] = False
            for _, idx in table.groupby(list(group_cols)).groups.items():
                rows = list(idx)
                table.loc[rows, "significant"] = holm_correction(
                    table.loc[rows, "p_value"].to_numpy(), alpha=alpha
                )
        else:
            table["significant"] = holm_correction(table["p_value"].to_numpy(), alpha=alpha)
    else:
        table["significant"] = table["p_value"] < alpha
    table["correction_scope"] = correction_scope
    table["reference"] = reference if reference is not None else ""
    table["metric"] = metric
    table["test"] = test
    table["correction"] = correction
    return table.sort_values("p_value", ignore_index=True)
