"""Paired significance tests: direction, null behaviour, Holm correction."""

import numpy as np
import pandas as pd
import pytest

from uavbench.analysis import paired_seed_test, pairwise_significance_table
from uavbench.analysis.significance import holm_correction


def _paired_df(effects: dict[str, float], n_seeds: int = 12, noise: float = 0.01, seed: int = 0):
    """One row per (method, scenario, seed); per-seed offsets are shared
    across methods (the paired-instance property the harness guarantees)."""
    rng = np.random.default_rng(seed)
    rows = []
    for scenario in ("uniform", "clustered"):
        base = rng.normal(0.7, 0.05, size=n_seeds)  # shared per-seed difficulty
        for method, effect in effects.items():
            for s in range(n_seeds):
                rows.append({
                    "method": method,
                    "scenario": scenario,
                    "seed": s,
                    "accuracy": base[s] + effect + rng.normal(0, noise),
                })
    return pd.DataFrame(rows)


class TestPairedSeedTest:
    def test_detects_real_effect_both_directions(self):
        df = _paired_df({"a": 0.05, "b": 0.0})
        ab = paired_seed_test(df, "accuracy", "a", "b")
        assert (ab["p_value"] < 0.05).all()
        assert (ab["mean_diff"] > 0).all()
        ba = paired_seed_test(df, "accuracy", "b", "a")
        assert (ba["p_value"] < 0.05).all()
        assert (ba["mean_diff"] < 0).all()

    def test_null_case_not_significant(self):
        df = _paired_df({"a": 0.0, "b": 0.0}, noise=0.02, seed=7)
        out = paired_seed_test(df, "accuracy", "a", "b")
        assert (out["p_value"] > 0.05).all()

    def test_identical_methods_p_value_one(self):
        df = _paired_df({"a": 0.0}, noise=0.0)
        clone = df.assign(method="b")
        out = paired_seed_test(pd.concat([df, clone]), "accuracy", "a", "b")
        assert (out["p_value"] == 1.0).all()

    def test_ttest_variant_runs(self):
        df = _paired_df({"a": 0.05, "b": 0.0})
        out = paired_seed_test(df, "accuracy", "a", "b", test="ttest_rel")
        assert (out["p_value"] < 0.05).all()

    def test_mismatched_seeds_raise(self):
        df = _paired_df({"a": 0.0, "b": 0.0})
        df = df[~((df["method"] == "b") & (df["seed"] == 0))]
        with pytest.raises(ValueError, match="seed sets differ"):
            paired_seed_test(df, "accuracy", "a", "b")

    def test_duplicate_seeds_raise(self):
        df = _paired_df({"a": 0.0, "b": 0.0}, n_seeds=3)
        dup = df[df["seed"] == 0]
        with pytest.raises(ValueError, match="duplicate seeds"):
            paired_seed_test(pd.concat([df, dup]), "accuracy", "a", "b")


class TestHolmCorrection:
    def test_reduces_significant_count_vs_uncorrected(self):
        p = np.array([0.001, 0.012, 0.02, 0.04, 0.049])
        assert holm_correction(p, alpha=0.05).sum() < (p < 0.05).sum()

    def test_step_down_monotone(self):
        # A rejected p implies every smaller p is also rejected.
        p = np.array([0.001, 0.03, 0.002, 0.2])
        reject = holm_correction(p, alpha=0.05)
        rejected_ps = p[reject]
        if rejected_ps.size:
            assert p[~reject].min() > rejected_ps.max()


class TestPairwiseTable:
    def test_all_pairs_present_with_metadata(self):
        df = _paired_df({"a": 0.06, "b": 0.0, "c": 0.0})
        table = pairwise_significance_table(df, "accuracy", ["a", "b", "c"])
        # 3 pairs x 2 scenarios
        assert len(table) == 6
        assert set(table["correction"]) == {"holm"}
        # The genuine effects (a vs b, a vs c) survive correction; b vs c doesn't.
        sig_pairs = set(map(tuple, table[table["significant"]][["method_a", "method_b"]].values))
        assert ("a", "b") in sig_pairs and ("a", "c") in sig_pairs
        assert ("b", "c") not in sig_pairs
