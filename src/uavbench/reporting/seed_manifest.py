"""Enumerate the exact seeds a configured run will use, before it runs.

The manifest calls the harnesses' own seed-derivation functions
(:mod:`uavbench.runner` for Tier-1, :mod:`uavbench.fl.seeds` for the FL
harnesses) rather than reimplementing the formulas, so it can never drift
from what actually executes. One ``seed_manifest.csv`` is written next to
each run's resolved config; checked in per reported paper result, it makes
every number traceable to an exact, rerunnable seed.

Caveat (columns note it): ``run_tier2`` folds the number of *loaded*
clients into its seed, and real-data loading can drop clients with empty
shards — for the tier2/sweep harnesses the manifest assumes
``n_clients == configured N`` (geographic K-means keeps all N on real data
unless a client's shard is empty).
"""

from __future__ import annotations

import pandas as pd

from uavbench.fl.seeds import (
    fullsim_method_seed,
    partition_seed_for,
    sweep_job_seed,
    tier2_seed,
)

HARNESSES = (
    "tier1",
    "tier2",
    "full_hfl",
    "sweep",
    "paper_sweep",
    "selection_sweep",
    "stress_sweep",
)


def build_seed_manifest(cfg: dict, harness: str) -> pd.DataFrame:
    """Flat (method/scenario/seed_idx -> resolved seed) table for a config.

    ``harness`` selects the seed-derivation scheme:

    * ``tier1``           — runner.py's SeedSequence streams (run / smoke)
    * ``tier2``           — run_tier2's per-method hash fold
    * ``full_hfl``        — run_full_hfl's per-method XOR fold
    * ``sweep``           — run_sweep: tier2 fold per (N, method)
    * ``paper_sweep``     — run_paper_sim: job seed + per-method XOR fold
    * ``selection_sweep`` — run_selection_sim: job seed shared across modes
    * ``stress_sweep``    — run_stress_sweep: knob-independent job seed +
                            per-method XOR fold (identical base problem in
                            every stress cell)
    """
    rows: list[dict] = []

    if harness == "tier1":
        # Deferred import: runner imports reporting.tables at module level, so
        # a top-level import here would be a circular dependency (it breaks
        # joblib workers, which import each module fresh).
        from uavbench.runner import _instance_seed

        base = cfg["instance_seed"]
        opt_base = cfg["optimizer_seed"]
        for s_idx, scen in enumerate(cfg["scenarios"]):
            scen_name = f"{scen['distribution']}_N{scen['N']}_K{scen['K']}"
            for m_idx, method in enumerate(cfg["methods"]):
                for seed_i in range(cfg["n_seeds"]):
                    rows.append(
                        {
                            "harness": harness,
                            "method": method,
                            "scenario": scen_name,
                            "seed_idx": seed_i,
                            "instance_seed": _instance_seed(base, s_idx, seed_i),
                            "optimizer_stream": f"SeedSequence([{opt_base}, 1, {m_idx}, {s_idx}, {seed_i}])",
                        }
                    )

    elif harness == "tier2":
        opt_seed = cfg.get("optimizer_seed", 9876)
        n = cfg["data"]["N_clients"]
        for method in cfg["methods"]:
            rows.append(
                {
                    "harness": harness,
                    "method": method,
                    "N": n,
                    "seed": tier2_seed(opt_seed, n, method),
                    "note": "assumes n_clients == N (empty shards may reduce it on real data)",
                }
            )

    elif harness == "full_hfl":
        run_seed = cfg["fl"].get("seed", cfg.get("optimizer_seed", 42))
        for method in cfg["methods"]:
            rows.append(
                {
                    "harness": harness,
                    "method": method,
                    "run_seed": run_seed,
                    "seed": fullsim_method_seed(run_seed, method),
                }
            )

    elif harness == "sweep":
        opt_seed = cfg.get("optimizer_seed", 9876)
        for N in cfg["N_values"]:
            for method in cfg["methods"]:
                rows.append(
                    {
                        "harness": harness,
                        "method": method,
                        "N": N,
                        "seed": tier2_seed(opt_seed, N, method),
                        "note": "assumes n_clients == N (empty shards may reduce it on real data)",
                    }
                )

    elif harness == "paper_sweep":
        opt_seed = cfg.get("optimizer_seed", 9876)
        for N in cfg["N_values"]:
            for method in cfg["methods"]:
                for seed_idx in range(cfg.get("n_seeds", 1)):
                    job_seed = sweep_job_seed(opt_seed, seed_idx, N)
                    rows.append(
                        {
                            "harness": harness,
                            "method": method,
                            "N": N,
                            "seed_idx": seed_idx,
                            "job_seed": job_seed,
                            "seed": fullsim_method_seed(job_seed, method),
                            "partition_seed": partition_seed_for(seed_idx),
                        }
                    )

    elif harness == "selection_sweep":
        opt_seed = cfg.get("optimizer_seed", 9876)
        modes = cfg.get("modes", [])
        for N in cfg["N_values"]:
            for mode in modes:
                for seed_idx in range(cfg.get("n_seeds", 1)):
                    rows.append(
                        {
                            "harness": harness,
                            "method": mode,
                            "N": N,
                            "seed_idx": seed_idx,
                            # Shared across modes by design: identical problem
                            # instance per (N, seed) isolates the selection rule.
                            "seed": sweep_job_seed(opt_seed, seed_idx, N),
                            "partition_seed": partition_seed_for(seed_idx),
                        }
                    )

    elif harness == "stress_sweep":
        from uavbench.fl.stress_sweep import build_stress_grid

        opt_seed = cfg.get("optimizer_seed", 9876)
        n = cfg["data"]["N_clients"]
        for d, s, c in build_stress_grid(cfg["sweep"]):
            for method in cfg["methods"]:
                for seed_idx in range(cfg.get("n_seeds", 1)):
                    job_seed = sweep_job_seed(opt_seed, seed_idx, n)
                    rows.append(
                        {
                            "harness": harness,
                            "method": method,
                            "dropout_rate": d,
                            "snr_degradation_db": s,
                            "black_chip_rate": c,
                            "seed_idx": seed_idx,
                            # Knob-independent by design: every stress cell sees
                            # the identical base problem per seed.
                            "job_seed": job_seed,
                            "seed": fullsim_method_seed(job_seed, method),
                            "partition_seed": partition_seed_for(seed_idx),
                        }
                    )

    else:
        raise ValueError(f"unknown harness {harness!r}; expected one of {HARNESSES}")

    return pd.DataFrame(rows)
