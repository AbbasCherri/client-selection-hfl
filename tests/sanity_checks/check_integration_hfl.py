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


def class_source_ladder_runs_end_to_end():
    """Every rung of the oracle ladder must be reachable from run_full_hfl.

    `pseudo` was not: run_full_hfl called build_class_info without a model, so
    the rung raised on every run and the realism answer existed only in the
    selection-isolation harness. Exercising all four here is what keeps the
    ladder honest — a rung nobody can run is not a rung.
    """
    for source in ("true", "pseudo", "dp", "none"):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(d, ["proposed_hfl", "hfl_no_selection"], n_rounds=2)
            cfg["fl"]["class_source"] = source
            cfg["fl"]["dp_epsilon"] = 1.0
            df = run_full_hfl(cfg)["rounds"]
            for m in ("proposed_hfl", "hfl_no_selection"):
                rows = df[df["method"] == m]
                assert len(rows) == 2, f"{source}/{m}: got {len(rows)} rounds, want 2"
                assert rows["macro_f1"].between(0, 1).all(), f"{source}/{m}: macro_f1 out of range"


def unquoted_class_source_is_rejected():
    """YAML turns `class_source: true` into a bool and `class_source:` into None.

    str().lower() would coerce both — the second into "none", i.e. the
    lower-anchor arm running under the default's name. A silently wrong arm is
    indistinguishable from a real result, so this must raise.
    """
    for bad in (True, None, 1):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(d, ["proposed_hfl"], n_rounds=1)
            cfg["fl"]["class_source"] = bad
            try:
                run_full_hfl(cfg)
            except ValueError:
                continue
            raise AssertionError(f"class_source={bad!r} was silently coerced")


def clone_model_matches_deepcopy():
    # clone_model is a hand-rolled fast path replacing copy.deepcopy. It must
    # stay bit-identical: same param values, same per-parameter requires_grad,
    # same training mode, independent storage, AND (load-bearing for
    # reproducibility) it must consume zero random numbers from torch's global
    # RNG — a shifted stream would change every downstream draw.
    import copy as _copy

    import torch

    from uavbench.fl.model import CachedFusionModel, clone_model

    torch.manual_seed(1)
    m = CachedFusionModel()
    m.freeze_img_proj()          # realistic: global model has img_proj frozen
    m.train(False)               # exercise training-mode preservation
    ref = _copy.deepcopy(m)
    got = clone_model(m)
    assert all(torch.equal(a, b) for a, b in zip(ref.parameters(), got.parameters()))
    assert all(a.requires_grad == b.requires_grad for a, b in zip(ref.parameters(), got.parameters()))
    assert all(a.training == b.training for a, b in zip(ref.modules(), got.modules()))
    # independent storage
    before = next(m.parameters()).clone()
    with torch.no_grad():
        next(got.parameters()).add_(1.0)
    assert torch.equal(next(m.parameters()), before)
    # RNG stream untouched
    torch.manual_seed(7)
    s = torch.get_rng_state()
    expect = torch.rand(4)
    torch.set_rng_state(s)
    _ = clone_model(m)
    assert torch.equal(expect, torch.rand(4))


def fedavg_inplace_matches_naive():
    # fedavg / reputation_fedavg accumulate in place; pin numerical identity
    # to the out-of-place reference formula they replaced.
    import torch

    from uavbench.fl.model import fedavg, reputation_fedavg

    rng = np.random.default_rng(0)
    for _ in range(50):
        m = int(rng.integers(1, 6))
        updates = [({"w": torch.randn(8, 3), "b": torch.randn(8)}, int(rng.integers(1, 40))) for _ in range(m)]
        total = sum(n for _, n in updates)
        ref = {}
        for sd, n in updates:
            w = n / total
            for k, v in sd.items():
                ref[k] = ref.get(k, torch.zeros_like(v)) + w * v.float()
        got = fedavg(updates)
        assert all(torch.equal(ref[k], got[k]) for k in ref)

        rep_updates = [(sd, n, float(rng.uniform(0, 1))) for sd, n in updates]
        weights = [max(r, 0.0) * n for _, n, r in rep_updates]
        tw = sum(weights)
        ref_r = {}
        for (sd, _n, _r), w in zip(rep_updates, weights):
            wn = w / tw
            for k, v in sd.items():
                ref_r[k] = ref_r.get(k, torch.zeros_like(v)) + wn * v.float()
        got_r = reputation_fedavg(rep_updates)
        assert all(torch.equal(ref_r[k], got_r[k]) for k in ref_r)


check("all ablations share one schema; bounds hold; files on disk", schema_and_bounds)
check("T_sel repositioning cadence; hfl_static truly static", placement_cadence)
check("same seed -> identical accuracy/selection trajectories", same_seed_reproduces)
check("different seeds -> different random selections", different_seeds_differ)
check("every class_source rung runs end-to-end", class_source_ladder_runs_end_to_end)
check("unquoted/empty class_source is rejected, not coerced", unquoted_class_source_is_rejected)
check("clone_model is bit-identical to deepcopy and preserves the RNG stream", clone_model_matches_deepcopy)
check("fedavg/reputation_fedavg in-place accumulation matches naive formula", fedavg_inplace_matches_naive)
finish()
