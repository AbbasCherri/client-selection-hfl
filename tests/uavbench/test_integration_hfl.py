"""Integration tests for run_full_hfl — smoke tests with synthetic data.

These tests avoid any HuggingFace I/O by using data.source=synthetic.
They run a minimal 2-round simulation to verify end-to-end correctness
without needing the full paper grid.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _minimal_cfg(results_dir: str, methods: list[str], n_rounds: int = 2,
                 T_sel: int = 1, K: int = 2, capacity: int = 10,
                 R_comm: float = 200_000.0) -> dict:
    return {
        "results_dir": results_dir,
        "methods": methods,
        "fl": {
            "n_rounds": n_rounds,
            "n_local_epochs": 1,
            "lr": 0.01,
            "batch_size": 4,
            "K": K,
            "R_comm": R_comm,
            "capacity": capacity,
            "T_sel": T_sel,
            "target_accuracy": 0.99,  # high so rounds_to_target is usually None
            "seed": 42,
        },
        "budget": {"P": 5, "G_max": 3},
        "data": {
            "source": "synthetic",
            "N_clients": 12,
            "seed": 42,
        },
        "optimizer_seed": 42,
    }


# ── run_full_hfl smoke ────────────────────────────────────────────────────────

class TestRunFullHflSmoke:
    def test_proposed_hfl_returns_dataframe(self):
        from uavbench.fl.federated import run_full_hfl
        with tempfile.TemporaryDirectory() as d:
            cfg = _minimal_cfg(d, ["proposed_hfl"])
            out = run_full_hfl(cfg)
        assert isinstance(out["rounds"], pd.DataFrame)
        assert len(out["rounds"]) == 2  # n_rounds

    def test_proposed_hfl_has_required_columns(self):
        from uavbench.fl.federated import run_full_hfl
        with tempfile.TemporaryDirectory() as d:
            cfg = _minimal_cfg(d, ["proposed_hfl"])
            out = run_full_hfl(cfg)
        df = out["rounds"]
        for col in ("method", "round", "accuracy", "macro_f1",
                    "coverage_pct", "n_selected", "placement_fitness",
                    "comm_mb_round", "cumulative_energy_j", "round_time_s"):
            assert col in df.columns, f"Missing column: {col}"

    def test_accuracy_in_01(self):
        from uavbench.fl.federated import run_full_hfl
        with tempfile.TemporaryDirectory() as d:
            cfg = _minimal_cfg(d, ["proposed_hfl"])
            out = run_full_hfl(cfg)
        df = out["rounds"]
        assert df["accuracy"].between(0.0, 1.0).all()
        assert df["macro_f1"].between(0.0, 1.0).all()

    def test_coverage_pct_in_0_100(self):
        from uavbench.fl.federated import run_full_hfl
        with tempfile.TemporaryDirectory() as d:
            cfg = _minimal_cfg(d, ["proposed_hfl"])
            out = run_full_hfl(cfg)
        df = out["rounds"]
        assert df["coverage_pct"].between(0.0, 100.0).all()

    def test_round_numbers_correct(self):
        from uavbench.fl.federated import run_full_hfl
        with tempfile.TemporaryDirectory() as d:
            cfg = _minimal_cfg(d, ["proposed_hfl"], n_rounds=3)
            out = run_full_hfl(cfg)
        df = out["rounds"]
        assert sorted(df["round"].tolist()) == [1, 2, 3]

    def test_flat_fl_skips_uav_placement(self):
        from uavbench.fl.federated import run_full_hfl
        with tempfile.TemporaryDirectory() as d:
            cfg = _minimal_cfg(d, ["flat_fl"])
            out = run_full_hfl(cfg)
        df = out["rounds"]
        assert all(df["method"] == "flat_fl")
        # placement_fitness fixed at 1.0 for flat_fl
        assert df["placement_fitness"].iloc[0] == pytest.approx(1.0)

    def test_centralized_produces_rows(self):
        from uavbench.fl.federated import run_full_hfl
        with tempfile.TemporaryDirectory() as d:
            cfg = _minimal_cfg(d, ["centralized"])
            out = run_full_hfl(cfg)
        df = out["rounds"]
        assert all(df["method"] == "centralized")
        assert len(df) == 2

    def test_multiple_methods_all_present(self):
        from uavbench.fl.federated import run_full_hfl
        methods = ["proposed_hfl", "flat_fl", "hfl_no_reputation"]
        with tempfile.TemporaryDirectory() as d:
            cfg = _minimal_cfg(d, methods)
            out = run_full_hfl(cfg)
        df = out["rounds"]
        for m in methods:
            assert m in df["method"].values, f"Method {m!r} missing from results"

    def test_unknown_method_skipped_gracefully(self):
        from uavbench.fl.federated import run_full_hfl
        with tempfile.TemporaryDirectory() as d:
            cfg = _minimal_cfg(d, ["proposed_hfl", "does_not_exist"])
            out = run_full_hfl(cfg)
        df = out["rounds"]
        assert "does_not_exist" not in df["method"].values

    def test_results_written_to_disk(self):
        from uavbench.fl.federated import run_full_hfl
        with tempfile.TemporaryDirectory() as d:
            cfg = _minimal_cfg(d, ["proposed_hfl"])
            run_full_hfl(cfg)
            parquet = Path(d) / "fullsim_rounds.parquet"
            csv = Path(d) / "fullsim_rounds.csv"
            assert parquet.exists() or csv.exists()

    def test_comm_mb_positive_for_hfl(self):
        from uavbench.fl.federated import run_full_hfl
        with tempfile.TemporaryDirectory() as d:
            cfg = _minimal_cfg(d, ["proposed_hfl"])
            out = run_full_hfl(cfg)
        df = out["rounds"]
        # At least one round should have positive communication cost
        # (assuming at least one client gets selected)
        n_selected_total = df["n_selected"].sum()
        if n_selected_total > 0:
            assert df["comm_mb_round"].sum() > 0

    def test_comm_mb_flat_fl_lower_than_hfl(self):
        """flat_fl has no UAV hop → fewer model transfers than proposed_hfl."""
        from uavbench.fl.federated import run_full_hfl
        with tempfile.TemporaryDirectory() as d:
            cfg = _minimal_cfg(d, ["proposed_hfl", "flat_fl"])
            out = run_full_hfl(cfg)
        df = out["rounds"]
        hfl_comm = df[df["method"] == "proposed_hfl"]["comm_mb_round"].mean()
        flat_comm = df[df["method"] == "flat_fl"]["comm_mb_round"].mean()
        # HFL includes UAV→server hop on top of client→UAV, so should be higher
        # (only meaningful if both have clients selected)
        if hfl_comm > 0 and flat_comm > 0:
            assert hfl_comm >= flat_comm


# ── placement_fitness preservation ───────────────────────────────────────────

class TestPlacementFitnessPreservation:
    def test_fitness_preserved_on_non_repositioning_rounds(self):
        """With T_sel=3, round 1 repositions; rounds 2 & 3 should carry the same fitness."""
        from uavbench.fl.federated import run_full_hfl
        with tempfile.TemporaryDirectory() as d:
            cfg = _minimal_cfg(d, ["proposed_hfl"], n_rounds=4, T_sel=3)
            out = run_full_hfl(cfg)
        df = out["rounds"][out["rounds"]["method"] == "proposed_hfl"]
        fitness = df.set_index("round")["placement_fitness"]
        # Round 1 → repositions
        # Round 2 & 3 → carry round-1 fitness (no repositioning)
        # Round 4 → repositions again ((4-1) % 3 == 0)
        assert fitness[2] == pytest.approx(fitness[1])
        assert fitness[3] == pytest.approx(fitness[1])

    def test_static_uavs_fitness_constant(self):
        """hfl_static places UAVs once and never moves them."""
        from uavbench.fl.federated import run_full_hfl
        with tempfile.TemporaryDirectory() as d:
            cfg = _minimal_cfg(d, ["hfl_static"], n_rounds=4, T_sel=2)
            out = run_full_hfl(cfg)
        df = out["rounds"][out["rounds"]["method"] == "hfl_static"]
        fitness = df["placement_fitness"].values
        # All rounds must carry the same fitness value (placed once)
        assert np.all(np.abs(fitness - fitness[0]) < 1e-9)


# ── multi-seed diversity ──────────────────────────────────────────────────────

class TestMultiSeedDiversity:
    def test_different_seeds_produce_different_selections(self):
        """hfl_no_selection uses random mode; different seeds → different n_selected."""
        from uavbench.fl.federated import run_full_hfl
        results = []
        for seed in [0, 1, 2]:
            with tempfile.TemporaryDirectory() as d:
                cfg = _minimal_cfg(d, ["hfl_no_selection"], n_rounds=3)
                cfg["fl"]["seed"] = seed * 10000
                out = run_full_hfl(cfg)
            n_sel = out["rounds"]["n_selected"].tolist()
            results.append(n_sel)
        # At least two seeds should have different selection sequences
        assert len(set(map(tuple, results))) > 1, \
            "All seeds produced identical selections — rng not being used"
