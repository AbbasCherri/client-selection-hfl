"""Experiment orchestration: build the run grid and execute it in parallel.

A *run* is one (method, scenario, seed) triple. Instances depend only on
(scenario, seed) so every method sees identical instances (paired comparison);
optimizer stochasticity uses a separate seed stream keyed by method too, so each
run is independently reproducible (simulation plan Section 9).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from .metrics.placement import compute_metrics
from .optimizers import REGISTRY
from .problem.energy import EnergyModel
from .problem.fitness import Fitness
from .problem.instance import generate_instance

logger = logging.getLogger("uavbench.runner")


def load_config(path: str | Path) -> dict:
    """Load a YAML experiment config."""
    with open(path) as f:
        return yaml.safe_load(f)


def _build_optimizer(method: str, budget: dict):
    """Instantiate an optimizer with the shared evaluation budget."""
    cls = REGISTRY[method]
    if method in ("pso", "ga"):
        return cls(P=budget["P"], G_max=budget["G_max"])
    return cls()


def _instance_seed(base: int, scenario_idx: int, seed_i: int) -> int:
    ss = np.random.SeedSequence([base, scenario_idx, seed_i])
    return int(ss.generate_state(1)[0])


def _optimizer_rng(base: int, method_idx: int, scenario_idx: int, seed_i: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([base, method_idx, scenario_idx, seed_i]))


def _run_one(cfg: dict, method: str, method_idx: int, scenario_idx: int, seed_i: int) -> dict:
    """Execute a single (method, scenario, seed) run; return metrics + convergence."""
    scenario = cfg["scenarios"][scenario_idx]
    inst_seed = _instance_seed(cfg["instance_seed"], scenario_idx, seed_i)

    instance = generate_instance(
        distribution=scenario["distribution"],
        N=scenario["N"],
        K=scenario["K"],
        area=cfg["area"],
        seed=inst_seed,
        capacity=cfg["problem"]["capacity"],
        uav_battery=cfg["problem"]["uav_battery"],
        R_comm=cfg["problem"]["R_comm"],
        B_min_uav=cfg["problem"]["B_min_uav"],
        beta_mode=cfg["value"]["beta_mode"],
        t=cfg["value"]["t"],
        T_decay=cfg["value"]["T_decay"],
        prev_mode=cfg["problem"].get("prev_mode", "stale"),
    )

    fw = (cfg["fitness"]["w1"], cfg["fitness"]["w2"], cfg["fitness"]["w3"])
    fitness = Fitness(instance, *fw)
    rng = _optimizer_rng(cfg["optimizer_seed"], method_idx, scenario_idx, seed_i)

    optimizer = _build_optimizer(method, cfg["budget"])
    result = optimizer.optimize(instance, fitness, rng)

    metrics = compute_metrics(instance, result, fitness_weights=fw, energy_model=EnergyModel())
    metrics.update(
        scenario=f"{scenario['distribution']}_N{scenario['N']}_K{scenario['K']}",
        distribution=scenario["distribution"],
        N=scenario["N"],
        K=scenario["K"],
        seed=seed_i,
    )
    return {"metrics": metrics, "convergence": result.convergence}


def _write_table(df: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame to Parquet, falling back to CSV if pyarrow is missing."""
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Parquet write failed (%s); falling back to CSV", exc)
        csv = path.with_suffix(".csv")
        df.to_csv(csv, index=False)
        return csv


def _dir_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def run_experiment(cfg: dict) -> dict:
    """Run the full grid and persist per-run metrics + convergence traces."""
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Allow a per-machine worker override (e.g. from run_tier1.sh WORKERS=...).
    env_workers = os.environ.get("UAVBENCH_N_WORKERS")
    if env_workers:
        cfg["n_workers"] = int(env_workers)

    methods = cfg["methods"]
    n_seeds = cfg["n_seeds"]
    n_scen = len(cfg["scenarios"])

    jobs = [
        (m_idx, m, s_idx, seed_i)
        for s_idx in range(n_scen)
        for m_idx, m in enumerate(methods)
        for seed_i in range(n_seeds)
    ]
    logger.info(
        "Running %d jobs (%d methods x %d scenarios x %d seeds) on %d workers",
        len(jobs), len(methods), n_scen, n_seeds, cfg["n_workers"],
    )

    outputs = Parallel(n_jobs=cfg["n_workers"])(
        delayed(_run_one)(cfg, m, m_idx, s_idx, seed_i) for (m_idx, m, s_idx, seed_i) in jobs
    )

    rows = [o["metrics"] for o in outputs]
    conv_rows = []
    for o in outputs:
        m = o["metrics"]
        for it, val in enumerate(o["convergence"]):
            conv_rows.append(
                {"method": m["method"], "scenario": m["scenario"], "seed": m["seed"],
                 "iteration": it, "best_fitness": val}
            )

    runs_df = pd.DataFrame(rows)
    conv_df = pd.DataFrame(conv_rows)

    runs_path = _write_table(runs_df, results_dir / "runs.parquet")
    conv_path = _write_table(conv_df, results_dir / "convergence.parquet")

    # Persist the fully-resolved config next to the results.
    with open(results_dir / "config.resolved.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    size_mb = _dir_size_mb(results_dir)
    logger.info("Wrote %s and %s", runs_path.name, conv_path.name)
    logger.info("Results dir %s footprint: %.2f MB", results_dir, size_mb)

    return {"runs": runs_df, "convergence": conv_df, "results_dir": results_dir, "size_mb": size_mb}
