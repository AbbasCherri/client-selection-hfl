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
from .optimizers import build_optimizer
from .problem.energy import EnergyModel
from .problem.fitness import Fitness
from .problem.instance import generate_instance
from .reporting.checkpoint import load_checkpoint, save_checkpoint
from .reporting.tables import write_table as _write_table

logger = logging.getLogger("uavbench.runner")


def load_config(path: str | Path) -> dict:
    """Load a YAML experiment config."""
    with open(path) as f:
        return yaml.safe_load(f)


def _build_optimizer(method: str, budget: dict, method_params: dict | None = None):
    """Instantiate an optimizer with the shared evaluation budget.

    Thin wrapper kept for backward compatibility; the shared construction
    logic (including the budget-precedence rule) lives in
    :func:`uavbench.optimizers.build_optimizer`.
    """
    return build_optimizer(method, params=method_params, budget=budget)


# Stream tags disambiguate the two seed families structurally. Without them,
# SeedSequence's trailing-zero equivalence ([b, s, i] == [b, s, i, 0]) lets an
# optimizer stream with seed_i=0 collide with an instance stream whenever a
# config sets instance_seed == optimizer_seed (verified by the runtime assert
# in _run_one and tests/sanity_checks/check_seeds_and_budget.py).
_INSTANCE_STREAM = 0
_OPTIMIZER_STREAM = 1


def _instance_seed(base: int, scenario_idx: int, seed_i: int) -> int:
    ss = np.random.SeedSequence([base, _INSTANCE_STREAM, scenario_idx, seed_i])
    return int(ss.generate_state(1)[0])


def _optimizer_rng(
    base: int, method_idx: int, scenario_idx: int, seed_i: int
) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([base, _OPTIMIZER_STREAM, method_idx, scenario_idx, seed_i])
    )


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
    # The two seed families must stay disjoint even when a config sets
    # instance_seed == optimizer_seed (SeedSequence trailing-zero hazard).
    assert inst_seed != int(rng.bit_generator.seed_seq.generate_state(1)[0]), (  # type: ignore[union-attr]
        "instance/optimizer seed streams collided — stream tags broken"
    )

    opt_params = cfg.get("optimizer_params", {}).get(method, {})
    optimizer = _build_optimizer(method, cfg["budget"], opt_params)
    result = optimizer.optimize(instance, fitness, rng)

    metrics = compute_metrics(
        instance,
        result,
        fitness_weights=fw,
        energy_model=EnergyModel(),
        G_max=cfg["budget"]["G_max"],
        radii=result.meta.get("radii"),
    )
    metrics.update(
        scenario=f"{scenario['distribution']}_N{scenario['N']}_K{scenario['K']}",
        distribution=scenario["distribution"],
        N=scenario["N"],
        K=scenario["K"],
        seed=seed_i,
    )
    return {"metrics": metrics, "convergence": result.convergence}


def _dir_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def _checkpoint_path(results_dir: Path, method: str, scenario_idx: int, seed_i: int) -> Path:
    return results_dir / ".checkpoints" / f"{method}__s{scenario_idx}__seed{seed_i}.pkl"


def _run_one_checkpointed(
    cfg: dict, method: str, method_idx: int, scenario_idx: int, seed_i: int, ckpt: Path
) -> dict:
    """Run one job and checkpoint it — executed inside a joblib worker."""
    out = _run_one(cfg, method, method_idx, scenario_idx, seed_i)
    save_checkpoint(ckpt, out)
    return out


def run_experiment(cfg: dict) -> dict:
    """Run the full grid and persist per-run metrics + convergence traces.

    Resumable: each (method, scenario, seed) job is checkpointed to
    ``results_dir/.checkpoints/`` as it finishes. A rerun with the same
    ``results_dir`` (e.g. after the process was killed mid-grid) reloads
    already-checkpointed jobs instead of recomputing them and only submits
    the remaining ones to the pool.
    """
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

    job_outputs: dict[tuple, dict] = {}
    pending = []
    for job in jobs:
        _, m, s_idx, seed_i = job
        ckpt = _checkpoint_path(results_dir, m, s_idx, seed_i)
        cached = load_checkpoint(ckpt)
        if cached is not None:
            job_outputs[job] = cached
        else:
            pending.append(job)

    n_done = len(jobs) - len(pending)
    if n_done:
        logger.info(
            "Resuming: %d/%d jobs already checkpointed, running the remaining %d",
            n_done,
            len(jobs),
            len(pending),
        )
    logger.info(
        "Running %d jobs (%d methods x %d scenarios x %d seeds) on %d workers",
        len(jobs),
        len(methods),
        n_scen,
        n_seeds,
        cfg["n_workers"],
    )

    if pending:
        computed = Parallel(n_jobs=cfg["n_workers"])(
            delayed(_run_one_checkpointed)(
                cfg, m, m_idx, s_idx, seed_i, _checkpoint_path(results_dir, m, s_idx, seed_i)
            )
            for (m_idx, m, s_idx, seed_i) in pending
        )
        for job, out in zip(pending, computed):
            job_outputs[job] = out

    outputs = [job_outputs[job] for job in jobs]

    rows = [o["metrics"] for o in outputs]
    conv_rows = []
    for o in outputs:
        m = o["metrics"]
        for it, val in enumerate(o["convergence"]):
            conv_rows.append(
                {
                    "method": m["method"],
                    "scenario": m["scenario"],
                    "seed": m["seed"],
                    "iteration": it,
                    "best_fitness": val,
                }
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
