"""Tests for the selection-isolation benchmark (fl/selection_isolation.py)."""

import tempfile

import numpy as np
import pandas as pd

from uavbench.fl.selection_isolation import (
    elbow_k,
    run_selection_isolation,
    static_uav_layout,
)

from .synthetic_fixture import build_synthetic_raw

# ── elbow_k ───────────────────────────────────────────────────────────────────


class TestElbowK:
    def _blobs(self, centres, n_per=30, spread=50.0, seed=0):
        rng = np.random.default_rng(seed)
        pts = [c + rng.normal(0, spread, size=(n_per, 2)) for c in np.asarray(centres, float)]
        return np.vstack(pts)

    def test_three_separated_blobs_gives_k3(self):
        xy = self._blobs([(0, 0), (10_000, 0), (0, 10_000)])
        k, centres = elbow_k(xy, k_min=2, k_max=8, seed=0)
        assert k == 3
        assert centres.shape == (3, 2)

    def test_k_respects_bounds(self):
        xy = self._blobs([(0, 0), (10_000, 0), (0, 10_000), (10_000, 10_000)])
        k, _ = elbow_k(xy, k_min=2, k_max=6, seed=0)
        assert 2 <= k <= 6

    def test_k_max_capped_below_n(self):
        rng = np.random.default_rng(1)
        xy = rng.normal(0, 100, size=(5, 2))
        k, centres = elbow_k(xy, k_min=2, k_max=30, seed=0)
        assert k <= 4  # n − 1

    def test_single_k_range(self):
        rng = np.random.default_rng(2)
        xy = rng.normal(0, 100, size=(10, 2))
        k, centres = elbow_k(xy, k_min=3, k_max=3, seed=0)
        assert k == 3
        assert centres.shape == (3, 2)

    def test_deterministic(self):
        xy = self._blobs([(0, 0), (10_000, 0)])
        k1, c1 = elbow_k(xy, 2, 6, seed=7)
        k2, c2 = elbow_k(xy, 2, 6, seed=7)
        assert k1 == k2
        np.testing.assert_allclose(c1, c2)


# ── static_uav_layout ─────────────────────────────────────────────────────────


class TestStaticLayout:
    def _coord_map(self, n=20):
        rng = np.random.default_rng(3)
        return {
            i: (37.2 + float(rng.uniform(0, 0.4)), 137.0 + float(rng.uniform(0, 0.4)))
            for i in range(n)
        }

    def test_layout_shapes_and_coverage(self):
        coords = self._coord_map()
        K, uav_latlon, covered = static_uav_layout(coords, 2, 4, seed=0, R_comm=50_000.0)
        assert len(uav_latlon) == K
        assert set(covered.keys()) <= set(coords.keys())
        assert all(0 <= u < K for u in covered.values())
        # 50 km range over a ~40 km box → everyone covered
        assert len(covered) == len(coords)

    def test_deterministic_across_calls(self):
        """The isolation guarantee: same layout for every mode/seed job."""
        coords = self._coord_map()
        out1 = static_uav_layout(coords, 2, 5, seed=42, R_comm=50_000.0)
        out2 = static_uav_layout(coords, 2, 5, seed=42, R_comm=50_000.0)
        assert out1[0] == out2[0]
        assert out1[1] == out2[1]
        assert out1[2] == out2[2]


# Jain-index unit tests moved to tests/uavbench/test_fairness.py alongside
# the shared fairness.py module (the _jain_index import stays: it pins the
# alias this harness still uses).

# ── run_selection_isolation (synthetic smoke) ────────────────────────────────


def _iso_cfg(results_dir: str, modes: list[str], n_rounds: int = 4) -> dict:
    return {
        "results_dir": results_dir,
        "modes": modes,
        "fl": {
            "n_rounds": n_rounds,
            "n_local_epochs": 1,
            "n_uav_epochs": 1,
            "lr": 0.01,
            "uav_lr": 0.01,
            "batch_size": 4,
            # Synthetic mode only: fl.K is the number of generated clients
            # (data-gen role); the UAV count comes from the elbow method.
            "K": 12,
            "R_comm": 200_000.0,
            "capacity": 10,
            "T_sel": 2,
            "lambda_min": 0.0,
            "target_accuracy": 0.99,
            "seed": 42,
        },
        "elbow": {"k_min": 2, "k_max": 4},
        "data": {
            "source": "prebuilt",
            "prebuilt": build_synthetic_raw(N=240, K=12, seed=42),
            "seed": 42,
        },
        "optimizer_seed": 42,
    }


class TestRunSelectionIsolation:
    def test_smoke_all_modes(self):
        modes = ["ucb", "random", "fedcs", "rep_cap", "fair_mab"]
        with tempfile.TemporaryDirectory() as d:
            out = run_selection_isolation(_iso_cfg(d, modes))
        df = out["rounds"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(modes) * 4
        assert set(df["method"].unique()) == set(modes)
        for col in (
            "accuracy",
            "macro_f1",
            "coverage_pct",
            "n_selected",
            "n_eligible",
            "K_uav",
            "jain_fairness",
            "n_unique_selected",
            "mean_battery",
            "comm_mb_round",
            "rounds_to_target",
        ):
            assert col in df.columns, f"missing column {col}"
        assert df["accuracy"].between(0, 1).all()
        assert df["jain_fairness"].between(0, 1).all()
        assert (df["n_selected"] > 0).all()

    def test_isolation_same_layout_across_modes(self):
        """Every mode must see the identical static placement and coverage."""
        with tempfile.TemporaryDirectory() as d:
            out = run_selection_isolation(_iso_cfg(d, ["ucb", "fedcs"]))
        df = out["rounds"]
        assert df["K_uav"].nunique() == 1
        assert df["coverage_pct"].nunique() == 1

    def test_deterministic_rerun(self):
        """Identical config → identical metrics (fully seeded pipeline)."""
        with tempfile.TemporaryDirectory() as d1:
            df1 = run_selection_isolation(_iso_cfg(d1, ["fedcs"]))["rounds"]
        with tempfile.TemporaryDirectory() as d2:
            df2 = run_selection_isolation(_iso_cfg(d2, ["fedcs"]))["rounds"]
        pd.testing.assert_frame_equal(
            df1.drop(columns=["round_time_s"]),
            df2.drop(columns=["round_time_s"]),
        )

    def test_k_uav_from_elbow_within_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            out = run_selection_isolation(_iso_cfg(d, ["random"]))
        # k_max additionally capped at N_clients // 5 = 12 // 5 = 2
        assert out["K_uav"] == 2

    def test_static_uavs_no_roster_change_between_reselects(self):
        """With T_sel=2, participation changes only on reselection rounds."""
        with tempfile.TemporaryDirectory() as d:
            cfg = _iso_cfg(d, ["fedcs"], n_rounds=4)
            df = run_selection_isolation(cfg)["rounds"]
        # Rounds 1-2 share a roster, rounds 3-4 share a roster.
        assert df["n_selected"].iloc[0] == df["n_selected"].iloc[1]
        assert df["n_selected"].iloc[2] == df["n_selected"].iloc[3]
