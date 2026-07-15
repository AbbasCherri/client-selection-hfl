"""Significance testing: paired Wilcoxon direction/null, Holm correction, and
the seed-mismatch refusal (manual check #8 — this must HARD-FAIL, not warn)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from _lib import check, finish

from uavbench.analysis import paired_seed_test, pairwise_significance_table
from uavbench.analysis.significance import holm_correction


def _paired_df(effects, n_seeds=12, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for scenario in ("uniform", "clustered"):
        base = rng.normal(0.7, 0.05, size=n_seeds)  # shared per-seed difficulty
        for method, effect in effects.items():
            for s in range(n_seeds):
                rows.append({
                    "method": method, "scenario": scenario, "seed": s,
                    "accuracy": base[s] + effect + rng.normal(0, noise),
                })
    return pd.DataFrame(rows)


def detects_real_effect():
    df = _paired_df({"a": 0.05, "b": 0.0})
    ab = paired_seed_test(df, "accuracy", "a", "b")
    assert (ab["p_value"] < 0.05).all() and (ab["mean_diff"] > 0).all()
    ba = paired_seed_test(df, "accuracy", "b", "a")
    assert (ba["p_value"] < 0.05).all() and (ba["mean_diff"] < 0).all()


def null_not_significant():
    df = _paired_df({"a": 0.0, "b": 0.0}, noise=0.02, seed=7)
    assert (paired_seed_test(df, "accuracy", "a", "b")["p_value"] > 0.05).all()
    ident = _paired_df({"a": 0.0}, noise=0.0)
    out = paired_seed_test(pd.concat([ident, ident.assign(method="b")]), "accuracy", "a", "b")
    assert (out["p_value"] == 1.0).all()


def mismatched_seeds_refused():
    df = _paired_df({"a": 0.0, "b": 0.0})
    broken = df[~((df["method"] == "b") & (df["seed"] == 0))]
    try:
        paired_seed_test(broken, "accuracy", "a", "b")
    except ValueError as e:
        assert "seed" in str(e)
    else:
        raise AssertionError("mismatched seed sets must raise, not silently drop")
    dup = pd.concat([df, df[df["seed"] == 0]])
    try:
        paired_seed_test(dup, "accuracy", "a", "b")
    except ValueError as e:
        assert "duplicate" in str(e)
    else:
        raise AssertionError("duplicate seeds must raise")


def holm_step_down():
    p = np.array([0.001, 0.012, 0.02, 0.04, 0.049])
    assert holm_correction(p, alpha=0.05).sum() < (p < 0.05).sum()
    p = np.array([0.001, 0.03, 0.002, 0.2])
    reject = holm_correction(p, alpha=0.05)
    if reject.any():
        assert p[~reject].min() > p[reject].max()  # step-down monotone


def pairwise_table_corrected():
    df = _paired_df({"a": 0.06, "b": 0.0, "c": 0.0})
    table = pairwise_significance_table(df, "accuracy", ["a", "b", "c"])
    assert len(table) == 6  # 3 pairs x 2 scenarios
    assert set(table["correction"]) == {"holm"}
    sig = set(map(tuple, table[table["significant"]][["method_a", "method_b"]].values))
    assert ("a", "b") in sig and ("a", "c") in sig and ("b", "c") not in sig


check("paired Wilcoxon detects a real effect in both directions", detects_real_effect)
check("null case not significant; identical methods -> p=1", null_not_significant)
check("mismatched or duplicate seed sets REFUSED with a raise", mismatched_seeds_refused)
check("Holm correction is step-down monotone and conservative", holm_step_down)
check("pairwise table: all pairs, Holm-corrected, only real effects survive", pairwise_table_corrected)
finish()
