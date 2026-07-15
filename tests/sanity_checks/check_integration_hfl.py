"""End-to-end run_full_hfl smoke on the prebuilt fixture: schema, bounds,
ablation semantics, reproducibility. The slowest check (~1-2 min CPU)."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from _fixture import build_synthetic_raw
from _lib import check, finish

from uavbench.fl.federated import run_full_hfl

REQUIRED_COLS = (
    "method", "round", "accuracy", "macro_f1", "coverage_pct", "n_selected",
    "placement_fitness", "comm_mb_round", "cumulative_energy_j", "round_time_s",
    "jain_fairness", "rounds_to_target",
)


def _cfg(results_dir, methods, n_rounds=2, T_sel=1, seed=42):
    return {
        "results_dir": results_dir,
        "methods": methods,
        "fl": {
            "n_rounds": n_rounds, "n_local_epochs": 1, "n_uav_epochs": 1,
            "lr": 0.01, "uav_lr": 0.01, "batch_size": 4, "K": 2,
            "R_comm": 200_000.0, "capacity": 10, "T_sel": T_sel,
            "lambda_min": 0.0, "target_accuracy": 0.99, "seed": seed,
        },
        "budget": {"P": 5, "G_max": 3},
        "data": {"source": "prebuilt", "prebuilt": build_synthetic_raw(N=12, K=2, seed=42), "seed": 42},
        "optimizer_seed": 42,
    }


def schema_and_bounds():
    methods = ["proposed_hfl", "flat_fl", "centralized", "hfl_no_reputation"]
    with tempfile.TemporaryDirectory() as d:
        out = run_full_hfl(_cfg(d, methods))
        df = out["rounds"]
        assert isinstance(df, pd.DataFrame)
        for col in REQUIRED_COLS:
            assert col in df.columns, f"missing column {col}"
        for m in methods:
            assert m in df["method"].values, f"{m} missing — ablations must share one schema"
        assert df["accuracy"].between(0, 1).all() and df["macro_f1"].between(0, 1).all()
        assert df["coverage_pct"].between(0, 100).all()
        # centralized is the no-selection oracle: jain is NaN there by design.
        fl_rows = df[df["method"] != "centralized"]
        assert fl_rows["jain_fairness"].between(0, 1).all()
        assert df[df["method"] == "centralized"]["jain_fairness"].isna().all()
        assert sorted(df[df["method"] == "proposed_hfl"]["round"]) == [1, 2]
        # flat_fl has no placement problem: fitness pinned at 1.0.
        assert abs(df[df["method"] == "flat_fl"]["placement_fitness"].iloc[0] - 1.0) < 1e-9
        # Results actually on disk.
        assert (Path(d) / "fullsim_rounds.parquet").exists() or (Path(d) / "fullsim_rounds.csv").exists()
        assert (Path(d) / "confusion.parquet").exists() or (Path(d) / "confusion.csv").exists()
        # HFL pays the UAV<->server hop on top of flat_fl's IoT payloads.
        hfl = df[df["method"] == "proposed_hfl"]["comm_mb_round"].mean()
        flat = df[df["method"] == "flat_fl"]["comm_mb_round"].mean()
        if hfl > 0 and flat > 0:
            assert hfl >= flat


def placement_cadence():
    with tempfile.TemporaryDirectory() as d:
        out = run_full_hfl(_cfg(d, ["proposed_hfl", "hfl_static"], n_rounds=4, T_sel=3))
        df = out["rounds"]
        dyn = df[df["method"] == "proposed_hfl"].set_index("round")["placement_fitness"]
        # T_sel=3: rounds 2 and 3 carry round-1 fitness (no repositioning).
        assert abs(dyn[2] - dyn[1]) < 1e-9 and abs(dyn[3] - dyn[1]) < 1e-9
        stat = df[df["method"] == "hfl_static"]["placement_fitness"].values
        assert np.all(np.abs(stat - stat[0]) < 1e-9)  # placed once, never moved


def same_seed_reproduces():
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        a = run_full_hfl(_cfg(d1, ["proposed_hfl"]))["rounds"]
        b = run_full_hfl(_cfg(d2, ["proposed_hfl"]))["rounds"]
        assert a["accuracy"].tolist() == b["accuracy"].tolist()
        assert a["n_selected"].tolist() == b["n_selected"].tolist()


def different_seeds_differ():
    results = []
    for seed in (0, 10_000, 20_000):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(d, ["hfl_no_selection"], n_rounds=3, seed=seed)
            results.append(tuple(run_full_hfl(cfg)["rounds"]["n_selected"]))
    assert len(set(results)) > 1, "all seeds produced identical selections — rng unused"


check("all ablations share one schema; bounds hold; files on disk", schema_and_bounds)
check("T_sel repositioning cadence; hfl_static truly static", placement_cadence)
check("same seed -> identical accuracy/selection trajectories", same_seed_reproduces)
check("different seeds -> different random selections", different_seeds_differ)
finish()
