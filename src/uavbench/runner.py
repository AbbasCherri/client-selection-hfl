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
from .problem.link import LinkModel
from .reporting.checkpoint import load_checkpoint, save_checkpoint
from .reporting.tables import write_table as _write_table

logger = logging.getLogger("uavbench.runner")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto ``base``; ``override`` always wins.

    Nested mappings merge key-by-key (so a config can override a single ``fl``
    entry without restating the block); every other type replaces wholesale.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict:
    """Load a YAML experiment config, resolving ``extends:`` inheritance.

    ``extends: <file>`` (relative to the config's own directory) loads that
    file first and deep-merges this config over it, so shared values — notably
    the Optuna-tuned training weights in ``configs/tuned_weights.yaml`` — live
    in exactly one place. Chains are followed recursively; cycles raise.
    """
    return _load_config_inner(Path(path), _seen=())


def _load_config_inner(path: Path, _seen: tuple[Path, ...]) -> dict:
    resolved = path.resolve()
    if resolved in _seen:
        chain = " -> ".join(p.name for p in (*_seen, resolved))
        raise ValueError(f"circular 'extends' in config chain: {chain}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    parent = cfg.pop("extends", None)
    if parent is None:
        return cfg

    parent_path = path.parent / parent
    if not parent_path.exists():
        raise FileNotFoundError(f"{path}: 'extends' target not found: {parent_path}")
    base = _load_config_inner(parent_path, _seen=(*_seen, resolved))
    return _deep_merge(base, cfg)


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
        capacity_cv=cfg["problem"].get("capacity_cv", 0.0),
        battery_cv=cfg["problem"].get("battery_cv", 0.0),
        data_dir=cfg.get("data", {}).get("data_dir", "./data"),
    )

    fw = (cfg["fitness"]["w1"], cfg["fitness"]["w2"], cfg["fitness"]["w3"])
    # Air-to-ground link. "path_loss" derives each UAV's coverage radius from its
    # own altitude through the shared Al-Hourani channel; "range_gate" is the
    # legacy flat |x - c| <= R_comm sphere.
    #
    # The flat gate makes the vertical decision a pure penalty — altitude only
    # ever adds slant distance, so every optimizer drives z to the floor and the
    # "3D" benchmark is a 2D one carrying a height column. That is the same
    # degeneracy the FL pipeline was corrected for on 2026-08-08, and Tier-1 is
    # the paper's headline placement table, so it cannot keep reporting under
    # physics the rest of the benchmark has retired.
    #
    # It also dissolves the scoring-fairness problem recorded in
    # REPORTS/results_provenance.md: the literature baselines (mozaffari2016,
    # alzenad2017) derive their own radius from altitude and were previously
    # scored at a larger radius than PSO's flat 500 m. Under a shared channel
    # nobody has a private radius left — every method is gated by the same
    # function of the altitude it chose.
    link = None
    link_model = str(cfg["problem"].get("link_model", "range_gate"))
    if link_model == "path_loss":
        area_z = cfg["area"]["z"]
        link = LinkModel(
            r_comm_m=float(cfg["problem"]["R_comm"]),
            z_min_m=float(area_z[0]),
            z_max_m=float(area_z[1]),
            environment=str(cfg["problem"].get("environment", "suburban")),
            freq_ghz=float(cfg["problem"].get("freq_ghz", 2.0)),
        )
    elif link_model != "range_gate":
        raise ValueError(
            f"problem.link_model must be 'path_loss' or 'range_gate', got {link_model!r}"
        )
    fitness = Fitness(instance, *fw, link=link)
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
        # Under a shared channel there are no private radii: Fitness lets an
        # explicit `radii` override the link, so forwarding a baseline's own
        # meta["radii"] here would score mozaffari/alzenad on their derived
        # radius while every other method is gated by the channel — the exact
        # 618-vs-500 m unfairness recorded in REPORTS/results_provenance.md,
        # reintroduced through the back door. The link already derives a radius
        # from whatever altitude each method chose, which is the fair version of
        # the same idea. Only the legacy flat gate still needs the override.
        radii=None if link is not None else result.meta.get("radii"),
        # Same link the optimizer was scored under — see compute_metrics.
        link=link,
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
