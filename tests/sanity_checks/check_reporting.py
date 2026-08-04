"""Results reporting: shared table writer (no stale-file shadowing), seed
manifests match the live seed derivations, Tier-1 runner end-to-end."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
from _lib import check, finish

from uavbench.fl.seeds import fullsim_method_seed, sweep_job_seed, tier2_seed
from uavbench.reporting.seed_manifest import build_seed_manifest
from uavbench.reporting.tables import write_table
from uavbench.runner import _instance_seed, run_experiment


def writer_parquet_and_fallback():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        df = pd.DataFrame({"a": [1, 2]})
        p = write_table(df, d / "t.parquet")
        assert p.suffix == ".parquet" and p.exists()
        # Simulate a parquet-write failure with a STALE parquet already on
        # disk from a previous run: the fallback must remove it so readers
        # preferring parquet can never pick up old data.
        (d / "s.parquet").write_bytes(b"stale-old-run")

        class Bad(pd.DataFrame):
            def to_parquet(self, *a, **k):
                raise RuntimeError("no pyarrow")

        p2 = write_table(Bad({"a": [3]}), d / "s.parquet")
        assert p2.suffix == ".csv" and p2.exists()
        assert not (d / "s.parquet").exists(), "stale parquet must be removed on fallback"
        # And the mirror: a successful parquet write removes a stale CSV.
        (d / "t.csv").write_text("stale")
        write_table(df, d / "t.parquet")
        assert not (d / "t.csv").exists()


def manifest_matches_live_seeds():
    cfg = {
        "instance_seed": 1234, "optimizer_seed": 9876, "n_seeds": 2,
        "scenarios": [{"distribution": "uniform", "N": 40, "K": 4}],
        "methods": ["pso", "ga"],
    }
    m = build_seed_manifest(cfg, "tier1")
    assert len(m) == 1 * 2 * 2
    for _, row in m.iterrows():
        assert row["instance_seed"] == _instance_seed(1234, 0, row["seed_idx"])
    # Paired-comparison property: instance seed identical across methods.
    assert (m.groupby("seed_idx")["instance_seed"].nunique() == 1).all()

    m2 = build_seed_manifest(
        {"optimizer_seed": 9876, "n_seeds": 3, "N_values": [30, 50],
         "methods": ["proposed_hfl", "fedcs"]},
        "paper_sweep",
    )
    row = m2[(m2["N"] == 50) & (m2["method"] == "fedcs") & (m2["seed_idx"] == 1)].iloc[0]
    job = sweep_job_seed(9876, 1, 50)
    assert row["job_seed"] == job and row["seed"] == fullsim_method_seed(job, "fedcs")

    m3 = build_seed_manifest(
        {"optimizer_seed": 9876, "n_seeds": 2, "N_values": [30],
         "modes": ["ucb", "random", "fedcs"]},
        "selection_sweep",
    )
    # Selection isolation: identical seed for every mode at a (N, seed_idx).
    assert (m3.groupby(["N", "seed_idx"])["seed"].nunique() == 1).all()

    m4 = build_seed_manifest(
        {"optimizer_seed": 9876, "data": {"N_clients": 40}, "methods": ["pso"]}, "tier2"
    )
    assert m4.iloc[0]["seed"] == tier2_seed(9876, 40, "pso")

    try:
        build_seed_manifest({}, "nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown harness must raise")


def tier1_runner_end_to_end():
    # Miniature Tier-1 grid through run_experiment: exercises the runtime
    # seed-disjointness assert, compute_metrics, manifest, and table writes.
    with tempfile.TemporaryDirectory() as d:
        cfg = {
            "results_dir": d,
            "instance_seed": 1234,
            "optimizer_seed": 9876,
            "n_seeds": 2,
            "n_workers": 1,
            "methods": ["pso", "centroid"],
            "scenarios": [{"distribution": "uniform", "N": 20, "K": 2}],
            "area": {"x": [0.0, 1000.0], "y": [0.0, 1000.0], "z": [20.0, 120.0]},
            "problem": {"capacity": 15, "uav_battery": 1.0, "R_comm": 500.0, "B_min_uav": 0.2},
            "value": {"beta_mode": "pinned", "t": 0, "T_decay": 20},
            "fitness": {"w1": 0.6, "w2": 0.3, "w3": 0.1},
            "budget": {"P": 8, "G_max": 5},
        }
        out = run_experiment(cfg)
        runs = out["runs"]
        assert len(runs) == 2 * 2  # methods x seeds
        assert set(runs["method"]) == {"pso", "centroid"}
        # (seed_manifest.csv is written by the CLI wrapper, not run_experiment)
        assert (Path(d) / "runs.parquet").exists() or (Path(d) / "runs.csv").exists()
        assert (Path(d) / "convergence.parquet").exists() or (Path(d) / "convergence.csv").exists()


def headline_table_keeps_N_and_the_final_round():
    """§0.6: accuracy + per-class F1 beside macro-F1, per (method, N).

    Two ways this table can lie, both guarded here:
      * pooling over N — the selection effect is null at N=30 and decisive at
        N=500, so a pooled row is an average of two different findings;
      * reading a non-final round — round order on disk is not guaranteed, so
        the summary must pick max(round), not the last row.
    """
    from uavbench.plotting import headline_metrics

    rows = []
    for method in ("ucb", "class_greedy"):
        for N in (30, 500):
            for seed in (0, 1):
                for rnd, f1 in ((0, 0.10), (5, 0.50), (2, 0.20)):  # out of order on purpose
                    rows.append({
                        "method": method, "N": N, "seed": seed, "round": rnd,
                        "accuracy": f1 + 0.05, "macro_f1": f1,
                        "f1_survived": f1, "f1_obstructed": f1 / 2,
                    })
    df = pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as d:
        paths = headline_metrics(df, Path(d))
        assert paths, "headline_metrics wrote nothing"
        out = pd.read_csv(paths[0])

    assert len(out) == 4, f"expected one row per (method, N), got {len(out)}"
    assert {"accuracy_mean", "macro_f1_mean", "f1_survived_mean",
            "f1_obstructed_mean", "n_seeds"}.issubset(out.columns), (
        f"missing headline columns: {sorted(out.columns)}"
    )
    assert (out["n_seeds"] == 2).all(), f"seed count wrong: {out['n_seeds'].tolist()}"
    # round 5 is the final round; picking the last ROW would give 0.20
    assert (out["macro_f1_mean"] == 0.50).all(), (
        f"summary did not use the final round: {out['macro_f1_mean'].tolist()}"
    )


check("shared writer: parquet, loud CSV fallback, stale files removed", writer_parquet_and_fallback)
check("headline table keeps N and uses the final round", headline_table_keeps_N_and_the_final_round)
check("seed manifests reproduce the live seed derivations exactly", manifest_matches_live_seeds)
check("Tier-1 runner end-to-end on a miniature grid", tier1_runner_end_to_end)
finish()
