"""Wall-clock summary over persisted run tables."""

import pandas as pd

from uavbench.reporting import summarize_wall_clock


def test_summarizes_tier1_runs(tmp_path):
    df = pd.DataFrame(
        {
            "method": ["pso", "pso", "ga", "ga"],
            "wall_time_s": [1.0, 3.0, 2.0, 2.0],
        }
    )
    df.to_parquet(tmp_path / "runs.parquet", index=False)
    out = summarize_wall_clock(tmp_path)
    assert set(out["method"]) == {"pso", "ga"}
    pso = out[out["method"] == "pso"].iloc[0]
    assert pso["mean_s"] == 2.0
    assert pso["total_s"] == 4.0
    assert pso["n"] == 2
    assert pso["source"] == "runs.parquet"


def test_summarizes_round_tables(tmp_path):
    df = pd.DataFrame(
        {
            "method": ["proposed_hfl"] * 3,
            "round_time_s": [0.5, 0.7, 0.6],
        }
    )
    df.to_parquet(tmp_path / "fullsim_rounds.parquet", index=False)
    out = summarize_wall_clock(tmp_path)
    assert len(out) == 1
    assert abs(out.iloc[0]["total_s"] - 1.8) < 1e-9


def test_empty_dir_returns_empty(tmp_path):
    out = summarize_wall_clock(tmp_path)
    assert out.empty


def test_optimizer_wall_time_is_real_perf_counter_delta():
    """wall_time is a genuine measurement, not just a present column."""
    import time

    import numpy as np

    from uavbench.optimizers import Centroid
    from uavbench.problem.fitness import Fitness
    from uavbench.problem.instance import generate_instance

    inst = generate_instance(
        "uniform", N=30, K=3,
        area={"x": [0.0, 1000.0], "y": [0.0, 1000.0], "z": [20.0, 120.0]}, seed=0,
    )
    t0 = time.perf_counter()
    result = Centroid().optimize(inst, Fitness(inst), np.random.default_rng(0))
    elapsed = time.perf_counter() - t0
    assert 0.0 < result.wall_time <= elapsed


def test_round_time_recorded_by_run_tier2(tmp_path):
    from uavbench.fl.federated import run_tier2

    cfg = {
        "results_dir": str(tmp_path),
        "optimizer_seed": 42,
        "data": {"source": "synthetic", "N_clients": 60, "n_synthetic_clients": 6, "seed": 1},
        "fl": {"K": 2, "R_comm": 50000.0, "capacity": 5, "n_rounds": 2,
               "n_local_epochs": 1, "lr": 1e-3, "batch_size": 8, "T_sel": 2},
        "budget": {"P": 5, "G_max": 3},
        "methods": ["centroid"],
    }
    df = run_tier2(cfg)["rounds"]
    assert (df["round_time_s"] > 0.0).all()
