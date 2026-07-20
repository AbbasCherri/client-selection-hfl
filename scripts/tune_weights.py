#!/usr/bin/env python
"""Optuna weight search for proposed_hfl's tunable weights.

Every weight below was picked by reasoning (paper defaults or engineering
judgment), never fit to data. This script searches them experimentally
against real data at reduced scale, so a full-grid run isn't the first time
any of these values gets tested.

Weights searched
-----------------
Tier A training recipe (uavbench.fl.federated, via job config):
    logit_adjust_tau, server_momentum, lr, ema_decay, lr_decay

Tier C selection blend (uavbench.fl.client_selection module constants):
    W_LEARNING_NB, W_UTILITY_NB (simplex-2)
    SEL_STATIC_BLEND, SEL_GUMBEL_SCALE

Paper-specified weights (also module constants — searching these means
deviating from the values stated in the paper; that trade-off was a
deliberate choice, not an oversight):
    W_EPI, W_SNR, W_DENS, W_PROX      (client_selection.py, simplex-4)
    W_CONTRIB, W_ANOMALY, W_TEMP      (reputation.py, simplex-3 — this is
                                        the Dirichlet PRIOR; it still adapts
                                        during a run, so we're tuning the
                                        starting point)
    w1, w2, w3                        (problem.fitness.Fitness, simplex-3)

Objective
---------
mean(last-5-round macro_f1) - 0.5*std(last-10-round macro_f1) of
`proposed_hfl`, averaged over a small (N, seed) grid. The stability penalty
is deliberate: Tier A specifically targeted round-to-round oscillation, so a
config that wins by luck on one volatile round shouldn't outscore a
config that's consistently good.

Correctness note on parallelism
--------------------------------
The searched weights are hardcoded module globals (not config), so a naive
"monkeypatch then call run_full_hfl" is not safe across joblib's *reused*
worker processes — a later job on the same worker could silently inherit a
previous trial's patched values. `_run_one_job` (the joblib worker function)
always sets every patched value explicitly from its own arguments before
using them, so every call is self-contained regardless of process reuse.
Optuna's own `n_jobs` is intentionally left at 1 (thread-based, and threads
DO share module state within one process — unsafe for this monkeypatch
approach) — parallelism instead comes from running each trial's (N, seed)
mini-grid concurrently via joblib (process-based).

Does NOT modify any committed source file. Results go to the Optuna
storage (sqlite) plus a leaderboard CSV. Adopting a winning value into the
module defaults is a separate, deliberate edit after reviewing the results.

Usage (on the VM, .venv activated, from repo root; reuses the paper_smoke
feature/partition caches — zero HF calls if that smoke run already ran):
    python scripts/tune_weights.py --n-trials 60
"""

from __future__ import annotations

import argparse
import copy
import tempfile
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed

import uavbench.fl.client_selection as cs
import uavbench.fl.federated as fed
import uavbench.fl.reputation as rep
from uavbench.fl.seeds import partition_seed_for, sweep_job_seed
from uavbench.problem.fitness import Fitness as _RealFitness

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURE_CACHE_SOURCE_DIR = REPO_ROOT / "results" / "paper_smoke"  # reuse its caches


def _simplex(trial: optuna.Trial, prefix: str, n: int) -> list[float]:
    """n raw Uniform(0.01, 1) draws, normalized to sum to 1."""
    raw = [trial.suggest_float(f"{prefix}_{i}", 0.01, 1.0) for i in range(n)]
    total = sum(raw)
    return [r / total for r in raw]


def _sample_weights(trial: optuna.Trial) -> dict:
    w_learning_nb, w_utility_nb = _simplex(trial, "sel_priority", 2)
    w_epi, w_snr, w_dens, w_prox = _simplex(trial, "utility", 4)
    w_contrib, w_anomaly, w_temp = _simplex(trial, "reputation", 3)
    w1, w2, w3 = _simplex(trial, "fitness", 3)
    return dict(
        # Tier A recipe (config-driven, no monkeypatch needed)
        logit_adjust_tau=trial.suggest_float("logit_adjust_tau", 0.0, 2.0),
        server_momentum=trial.suggest_float("server_momentum", 0.0, 0.99),
        lr=trial.suggest_float("lr", 1e-3, 1e-1, log=True),
        ema_decay=trial.suggest_float("ema_decay", 0.0, 0.99),
        lr_decay=trial.suggest_categorical("lr_decay", ["cosine", "sqrt", "none"]),
        # Tier C selection blend (module constants)
        sel_static_blend=trial.suggest_float("sel_static_blend", 0.0, 1.0),
        sel_gumbel_scale=trial.suggest_float("sel_gumbel_scale", 0.0, 1.5),
        w_learning_nb=w_learning_nb,
        w_utility_nb=w_utility_nb,
        # Paper-specified weights (module constants)
        w_epi=w_epi, w_snr=w_snr, w_dens=w_dens, w_prox=w_prox,
        w_contrib=w_contrib, w_anomaly=w_anomaly, w_temp=w_temp,
        fitness_w1=w1, fitness_w2=w2, fitness_w3=w3,
    )


def _run_one_job(job_cfg: dict, w: dict) -> pd.DataFrame | None:
    """One (N, seed) proposed_hfl run under the given weight set.

    Runs inside a joblib worker process. Every patched module attribute is
    set explicitly from `w` (never assumed inherited from a prior call on a
    reused worker process) — see module docstring.
    """
    cs.W_LEARNING_NB = w["w_learning_nb"]
    cs.W_UTILITY_NB = w["w_utility_nb"]
    cs.SEL_STATIC_BLEND = w["sel_static_blend"]
    cs.SEL_GUMBEL_SCALE = w["sel_gumbel_scale"]
    cs.W_EPI = w["w_epi"]
    cs.W_SNR = w["w_snr"]
    cs.W_DENS = w["w_dens"]
    cs.W_PROX = w["w_prox"]
    rep.W_CONTRIB = w["w_contrib"]
    rep.W_ANOMALY = w["w_anomaly"]
    rep.W_TEMP = w["w_temp"]
    fed.Fitness = lambda instance, w1=w["fitness_w1"], w2=w["fitness_w2"], w3=w["fitness_w3"]: (
        _RealFitness(instance, w1, w2, w3)
    )

    with tempfile.TemporaryDirectory() as d:
        job_cfg = copy.deepcopy(job_cfg)
        job_cfg["results_dir"] = d
        try:
            out = fed.run_full_hfl(job_cfg)
        except Exception as exc:  # a bad weight combination should not kill the study
            print(f"[trial job FAILED] N={job_cfg['data']['N_clients']} "
                  f"seed={job_cfg['fl']['seed']}: {exc!r}")
            return None
        df = out["rounds"]
        return df[df["method"] == "proposed_hfl"].copy()


def _score(df: pd.DataFrame) -> float:
    df = df.sort_values("round")
    last5 = df.tail(5)["macro_f1"].mean()
    last10_std = df.tail(10)["macro_f1"].std(ddof=0)
    return float(last5 - 0.5 * last10_std)


def build_objective(n_values: list[int], seeds: list[int], n_rounds: int, n_parallel: int):
    def objective(trial: optuna.Trial) -> float:
        w = _sample_weights(trial)

        base_cfg = {
            "methods": ["proposed_hfl"],
            "optimizer_seed": 9876,
            "data": {
                "source": "real", "subsample": 0.2, "seed": 42,
                "data_dir": str(REPO_ROOT / "data"),
                "feature_batch_size": 32, "feature_num_workers": 8,
            },
            "fl": {
                "K": 20, "R_comm": 20000.0, "capacity": 6,
                "n_rounds": n_rounds, "n_local_epochs": 2, "n_uav_epochs": 2,
                "batch_size": 32, "T_sel": 5, "reselect_every": 1,
                "lambda_min": 0.5, "R_min": 0.3, "placement_method": "pso",
                "target_metric": "macro_f1", "target_value": 0.45,
                "logit_adjust_tau": w["logit_adjust_tau"],
                "server_momentum": w["server_momentum"],
                "lr": w["lr"], "uav_lr": w["lr"],
                "ema_decay": w["ema_decay"], "lr_decay": w["lr_decay"],
                "balanced_sampling": False, "fusion_owner": "uav",
                "placement_class_aware": True,
            },
            "budget": {"P": 50, "G_max": 30},
        }

        jobs = []
        for N in n_values:
            cache_path = FEATURE_CACHE_SOURCE_DIR / f"N{N}" / "img_features.npy"
            for seed_idx in seeds:
                job_cfg = copy.deepcopy(base_cfg)
                job_cfg["data"]["N_clients"] = N
                job_cfg["data"]["partition_seed"] = partition_seed_for(seed_idx)
                job_cfg["data"]["feature_cache_path"] = str(cache_path)
                job_cfg["fl"]["seed"] = sweep_job_seed(9876, seed_idx, N)
                jobs.append(job_cfg)

        results = Parallel(n_jobs=n_parallel)(delayed(_run_one_job)(j, w) for j in jobs)
        scores = [_score(df) for df in results if df is not None and len(df)]
        if not scores:
            return float("-inf")  # every job in this trial crashed
        return float(np.mean(scores))

    return objective


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--n-parallel-jobs-per-trial", type=int, default=4)
    ap.add_argument("--n-rounds", type=int, default=20)
    ap.add_argument("--n-values", type=int, nargs="+", default=[30, 50])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--study-name", default="proposed_hfl_weights")
    ap.add_argument(
        "--storage",
        default=f"sqlite:///{REPO_ROOT}/results/hpo/weights.db",
        help="Optuna storage URL; sqlite makes the study resumable/inspectable.",
    )
    args = ap.parse_args()

    Path(args.storage.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name, storage=args.storage,
        direction="maximize", load_if_exists=True,
    )
    objective = build_objective(
        args.n_values, args.seeds, args.n_rounds, args.n_parallel_jobs_per_trial
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("\n=== Best trial ===")
    print(f"value: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    leaderboard = study.trials_dataframe().sort_values("value", ascending=False)
    out_csv = Path(args.storage.replace("sqlite:///", "")).parent / f"{args.study_name}_leaderboard.csv"
    leaderboard.to_csv(out_csv, index=False)
    print(f"\nFull leaderboard: {out_csv}")


if __name__ == "__main__":
    main()
