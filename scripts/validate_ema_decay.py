#!/usr/bin/env python
"""Longer-horizon grid check for ema_decay, the one Optuna-searched value
NOT adopted from the 2026-07-20 weight search (see tune_weights.py).

The search's 20-round evaluation window structurally penalizes heavy EMA
smoothing (a decay=0.9 EMA hasn't caught up to the live model in 20 rounds
regardless of whether it's better at full 100-round scale), so its
ema_decay≈0.17 result there isn't trustworthy on its own. This script fixes
every other weight at the now-adopted defaults and sweeps only ema_decay,
at 80 rounds instead of 20, to see which value actually wins with room to
converge.

Usage (VM, .venv activated, from repo root — reuses the paper_smoke caches):
    python scripts/validate_ema_decay.py
"""

from __future__ import annotations

import copy
import statistics
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

import tune_weights as tw
from uavbench.fl.seeds import partition_seed_for, sweep_job_seed

# Everything except ema_decay is already the committed default (module
# constants + config), so this only needs to vary the one config knob.
DEFAULT_WEIGHTS = dict(
    logit_adjust_tau=0.601, server_momentum=0.528, lr=0.01775, lr_decay="cosine",
    sel_static_blend=0.435, sel_gumbel_scale=1.475,
    w_learning_nb=0.702, w_utility_nb=0.298,
    w_epi=0.043, w_snr=0.078, w_dens=0.295, w_prox=0.584,
    w_contrib=0.091, w_anomaly=0.134, w_temp=0.775,
    fitness_w1=0.811, fitness_w2=0.03, fitness_w3=0.159,
    ema_decay=None,  # set per grid point below
)

GRID = [0.0, 0.1, 0.17, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
N_ROUNDS = 80
N_VALUES = [30, 50]
SEEDS = [0, 1]


def _score_ema_decay(ema_decay: float) -> tuple[float, list[float]]:
    w = dict(DEFAULT_WEIGHTS, ema_decay=ema_decay)
    base_cfg = {
        "methods": ["proposed_hfl"], "optimizer_seed": 9876,
        "data": {"source": "real", "subsample": 0.2, "seed": 42,
                  "data_dir": str(tw.REPO_ROOT / "data"),
                  "feature_batch_size": 32, "feature_num_workers": 8},
        "fl": {"K": 20, "R_comm": 20000.0, "capacity": 6, "n_rounds": N_ROUNDS,
               "n_local_epochs": 2, "n_uav_epochs": 2, "batch_size": 32, "T_sel": 5,
               "reselect_every": 1, "lambda_min": 0.5, "R_min": 0.3, "placement_method": "pso",
               "target_metric": "macro_f1", "target_value": 0.45,
               "logit_adjust_tau": w["logit_adjust_tau"], "server_momentum": w["server_momentum"],
               "lr": w["lr"], "uav_lr": w["lr"], "ema_decay": ema_decay, "lr_decay": w["lr_decay"],
               "balanced_sampling": False, "fusion_owner": "uav", "placement_class_aware": True},
        "budget": {"P": 50, "G_max": 30},
    }
    jobs = []
    for N in N_VALUES:
        cache_path = tw.FEATURE_CACHE_SOURCE_DIR / f"N{N}" / "img_features.npy"
        for seed_idx in SEEDS:
            job_cfg = copy.deepcopy(base_cfg)
            job_cfg["data"]["N_clients"] = N
            job_cfg["data"]["partition_seed"] = partition_seed_for(seed_idx)
            job_cfg["data"]["feature_cache_path"] = str(cache_path)
            job_cfg["fl"]["seed"] = sweep_job_seed(9876, seed_idx, N)
            jobs.append(job_cfg)
    results = Parallel(n_jobs=4)(delayed(tw._run_one_job)(j, w) for j in jobs)
    scores = [tw._score(df) for df in results if df is not None and len(df)]
    return (float(statistics.mean(scores)) if scores else float("-inf")), scores


def main() -> None:
    rows = []
    for ema_decay in GRID:
        mean_score, scores = _score_ema_decay(ema_decay)
        print(f"ema_decay={ema_decay:.2f}  mean={mean_score:.4f}  per-job={scores}")
        rows.append({"ema_decay": ema_decay, "mean_score": mean_score, "per_job_scores": scores})

    df = pd.DataFrame(rows).sort_values("mean_score", ascending=False)
    out_csv = tw.REPO_ROOT / "results" / "hpo" / "ema_decay_validation.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nBest: ema_decay={df.iloc[0]['ema_decay']} (mean={df.iloc[0]['mean_score']:.4f})")
    print(f"Full results: {out_csv}")


if __name__ == "__main__":
    main()
