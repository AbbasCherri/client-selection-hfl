#!/usr/bin/env python
"""Optuna hyperparameter search — scored on VALIDATION data, per method.

This script had two flaws that a reviewer would treat as disqualifying, both
fixed here (REPORTS/rigor_plan_2026-08.md §0.2 / §0.3b):

**F1 — the objective read the reported test set.** 22 hyperparameters were
chosen by maximising `macro_f1`, which is the held-out *test* column. That is
selection on the answer, and every headline number inherited the optimism. The
objective now reads `val_macro_f1` and :func:`_score` REFUSES to run if that
column is absent or all-NaN — a silent fallback to the test column is exactly
the failure being fixed, so it must be impossible rather than merely avoided.

**F2 — only the proposed method was tuned.** The recipe (lr, schedule, τ,
momentum) was fit to `proposed_hfl` and then imposed on every baseline, so the
comparison was tuned-vs-untuned. `--method` tunes any single method, and
`--space recipe` restricts the search to the shared training recipe, which is
the part every method is entitled to have fit to it. Run once per method and
each is reported at its own best recipe.

Search spaces
-------------
``recipe`` (every method — this is the fairness fix):
    lr, lr_decay, logit_adjust_tau, server_momentum, ema_decay

``full`` (``proposed_hfl`` only — recipe plus the constants that exist only in
our selector; a baseline has no analogue of these, so tuning them for it would
be meaningless rather than generous):
    SEL_STATIC_BLEND, SEL_GUMBEL_SCALE, W_LEARNING_NB, W_UTILITY_NB
    W_EPI, W_SNR, W_DENS, W_PROX          (client_selection, simplex-4)
    W_CONTRIB, W_ANOMALY, W_TEMP          (reputation, simplex-3 — Dirichlet
                                           PRIOR; still adapts during a run)
    w1, w2, w3                            (problem.fitness.Fitness, simplex-3)

Placement weights (w1,w2,w3) stay OUT of the per-baseline space on purpose:
they are paper-specified and are studied directly by the Phase 3 weight sweep.
Letting each baseline re-fit them would confound "better selector" with
"different placement objective".

Seed hygiene
------------
Tuning seeds must be disjoint from evaluation seeds. Evaluation uses seed
indices 0..19; `--seeds` defaults to 20,21,22 and the script errors out if any
tuning seed falls in the reserved evaluation range. `--transfer-seeds`
(default 23,24) are held out from tuning as well, for `--transfer-check`.

Objective
---------
mean(last-5-round val_macro_f1) - 0.5*std(last-10-round val_macro_f1),
averaged over a small (N, seed) grid. The stability penalty is deliberate:
Tier A specifically targeted round-to-round oscillation, so a config that wins
on one lucky volatile round should not outscore a consistently good one.

Regime note (F3, NOT fixed — reported as a limitation)
------------------------------------------------------
Tuning runs at reduced N / subsample / rounds; the paper grid runs at full
scale. Tuning at the evaluation regime would cost ~134 h per method. The
`--transfer-check` mode is the partial mitigation: it re-scores the top-3
configs on held-out seeds, so at least the *ranking* is shown to be stable
rather than assumed.

Does NOT modify any committed source file. Adopting a winner into
configs/tuned_weights.yaml is a separate, deliberate edit.

Usage (on the VM, from repo root):
    python scripts/tune_weights.py --method proposed_hfl --space full --n-trials 60
    python scripts/tune_weights.py --method fedcs --n-trials 30
    python scripts/tune_weights.py --method fedcs --transfer-check
"""

from __future__ import annotations

import argparse
import copy
import tempfile
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from joblib import Parallel, delayed

import uavbench.fl.client_selection as cs
import uavbench.fl.federated as fed
import uavbench.fl.reputation as rep
from uavbench.fl.seeds import partition_seed_for, sweep_job_seed
from uavbench.problem.fitness import Fitness as _RealFitness

REPO_ROOT = Path(__file__).resolve().parent.parent

# Seed indices 0..19 are reserved for evaluation sweeps (paper_full uses 0..9,
# the 15-seed re-run 0..14). Tuning on any of them would leak the evaluation
# draw into the hyperparameter choice, which is the same class of error as
# tuning on the test split.
RESERVED_EVAL_SEEDS = range(0, 20)

# Methods whose selector has our own tunable constants. Everything else gets
# the shared training recipe only.
FULL_SPACE_METHODS = {"proposed_hfl"}

# Captured before any trial patches them, so a worker process that ran a
# `full`-space trial can be restored to stock for a `recipe`-space one.
_STOCK = {
    "cs.W_LEARNING_NB": cs.W_LEARNING_NB,
    "cs.W_UTILITY_NB": cs.W_UTILITY_NB,
    "cs.SEL_STATIC_BLEND": cs.SEL_STATIC_BLEND,
    "cs.SEL_GUMBEL_SCALE": cs.SEL_GUMBEL_SCALE,
    "cs.W_EPI": cs.W_EPI,
    "cs.W_SNR": cs.W_SNR,
    "cs.W_DENS": cs.W_DENS,
    "cs.W_PROX": cs.W_PROX,
    "rep.W_CONTRIB": rep.W_CONTRIB,
    "rep.W_ANOMALY": rep.W_ANOMALY,
    "rep.W_TEMP": rep.W_TEMP,
}


def _simplex(trial: optuna.Trial, prefix: str, n: int) -> list[float]:
    """n raw Uniform(0.01, 1) draws, normalized to sum to 1."""
    raw = [trial.suggest_float(f"{prefix}_{i}", 0.01, 1.0) for i in range(n)]
    total = sum(raw)
    return [r / total for r in raw]


def _sample_weights(trial: optuna.Trial, space: str) -> dict:
    """Sample the search space. Works with a real Trial or an optuna FixedTrial.

    FixedTrial replay is what `--transfer-check` uses to re-run a stored
    parameter set, so this function must ask for exactly the same parameter
    names in the same space it originally did — hence `space` is stored in the
    study's user attrs rather than re-derived.
    """
    w = dict(
        logit_adjust_tau=trial.suggest_float("logit_adjust_tau", 0.0, 2.0),
        server_momentum=trial.suggest_float("server_momentum", 0.0, 0.99),
        lr=trial.suggest_float("lr", 1e-3, 1e-1, log=True),
        ema_decay=trial.suggest_float("ema_decay", 0.0, 0.99),
        lr_decay=trial.suggest_categorical("lr_decay", ["cosine", "sqrt", "none"]),
    )
    if space == "full":
        w_learning_nb, w_utility_nb = _simplex(trial, "sel_priority", 2)
        w_epi, w_snr, w_dens, w_prox = _simplex(trial, "utility", 4)
        w_contrib, w_anomaly, w_temp = _simplex(trial, "reputation", 3)
        w1, w2, w3 = _simplex(trial, "fitness", 3)
        w.update(
            sel_static_blend=trial.suggest_float("sel_static_blend", 0.0, 1.0),
            sel_gumbel_scale=trial.suggest_float("sel_gumbel_scale", 0.0, 1.5),
            w_learning_nb=w_learning_nb, w_utility_nb=w_utility_nb,
            w_epi=w_epi, w_snr=w_snr, w_dens=w_dens, w_prox=w_prox,
            w_contrib=w_contrib, w_anomaly=w_anomaly, w_temp=w_temp,
            fitness_w1=w1, fitness_w2=w2, fitness_w3=w3,
        )
    return w


def _run_one_job(job_cfg: dict, w: dict, method: str) -> pd.DataFrame | None:
    """One (N, seed) run of `method` under the given weight set.

    Runs inside a joblib worker process. Every patched module attribute is set
    explicitly — from `w` in the `full` space, from `_STOCK` otherwise — and
    never assumed inherited, because joblib REUSES worker processes and these
    are module globals. A `recipe` trial landing on a worker that previously
    ran a `full` trial would otherwise silently inherit its constants.
    """
    # Same reason every other parallel worker in this codebase does this
    # (sweep.py, stress_sweep.py): without it, each worker's own BLAS/MKL
    # intra-op parallelism fights every other worker for the same cores.
    torch.set_num_threads(1)

    cs.W_LEARNING_NB = w.get("w_learning_nb", _STOCK["cs.W_LEARNING_NB"])
    cs.W_UTILITY_NB = w.get("w_utility_nb", _STOCK["cs.W_UTILITY_NB"])
    cs.SEL_STATIC_BLEND = w.get("sel_static_blend", _STOCK["cs.SEL_STATIC_BLEND"])
    cs.SEL_GUMBEL_SCALE = w.get("sel_gumbel_scale", _STOCK["cs.SEL_GUMBEL_SCALE"])
    cs.W_EPI = w.get("w_epi", _STOCK["cs.W_EPI"])
    cs.W_SNR = w.get("w_snr", _STOCK["cs.W_SNR"])
    cs.W_DENS = w.get("w_dens", _STOCK["cs.W_DENS"])
    cs.W_PROX = w.get("w_prox", _STOCK["cs.W_PROX"])
    rep.W_CONTRIB = w.get("w_contrib", _STOCK["rep.W_CONTRIB"])
    rep.W_ANOMALY = w.get("w_anomaly", _STOCK["rep.W_ANOMALY"])
    rep.W_TEMP = w.get("w_temp", _STOCK["rep.W_TEMP"])
    if "fitness_w1" in w:
        fed.Fitness = lambda instance, w1=w["fitness_w1"], w2=w["fitness_w2"], w3=w["fitness_w3"]: (
            _RealFitness(instance, w1, w2, w3)
        )

    with tempfile.TemporaryDirectory() as d:
        job_cfg = copy.deepcopy(job_cfg)
        job_cfg["results_dir"] = d
        try:
            out = fed.run_full_hfl(job_cfg)
        except Exception as exc:  # a bad weight combination should not kill the study
            print(f"[trial job FAILED] method={method} "
                  f"N={job_cfg['data']['N_clients']} seed={job_cfg['fl']['seed']}: {exc!r}")
            return None
        df = out["rounds"]
        return df[df["method"] == method].copy()


def _score(df: pd.DataFrame) -> float:
    """Validation score for one run. Never reads the test columns.

    Hard-fails on a missing/empty val column instead of falling back. The bug
    this script exists to fix was a *silent* read of the test metric; a
    fallback path here would reintroduce it the first time someone ran with a
    config whose `data.val_ratio` was 0.
    """
    if "val_macro_f1" not in df.columns:
        raise KeyError(
            "run produced no 'val_macro_f1' column — the harness predates the "
            "three-way split. Tuning against 'macro_f1' is selection on the "
            "test set and is exactly what this script was rewritten to prevent."
        )
    vals = df["val_macro_f1"]
    if vals.isna().all():
        raise ValueError(
            "'val_macro_f1' is all-NaN — data.val_ratio is 0 or unset, so there "
            "is no validation split to tune against. Refusing to score."
        )
    df = df.sort_values("round")
    last5 = df.tail(5)["val_macro_f1"].mean()
    last10_std = df.tail(10)["val_macro_f1"].std(ddof=0)
    return float(last5 - 0.5 * last10_std)


def _build_jobs(method: str, w: dict, n_values, seeds, n_rounds: int,
                subsample: float, feature_cache_dir: Path) -> list[dict]:
    base_cfg = {
        "methods": [method],
        "optimizer_seed": 9876,
        "data": {
            "source": "real", "subsample": subsample, "seed": 42,
            "data_dir": str(REPO_ROOT / "data"),
            "feature_batch_size": 32, "feature_num_workers": 8,
            # Without this the run emits no val columns and _score refuses.
            "val_ratio": 0.1,
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
        cache_path = feature_cache_dir / f"N{N}" / "img_features.npy"
        for seed_idx in seeds:
            job_cfg = copy.deepcopy(base_cfg)
            job_cfg["data"]["N_clients"] = N
            job_cfg["data"]["partition_seed"] = partition_seed_for(seed_idx)
            job_cfg["data"]["feature_cache_path"] = str(cache_path)
            job_cfg["fl"]["seed"] = sweep_job_seed(9876, seed_idx, N)
            jobs.append(job_cfg)
    return jobs


def _evaluate(method: str, w: dict, n_values, seeds, n_rounds: int, subsample: float,
              feature_cache_dir: Path, n_parallel: int) -> float:
    jobs = _build_jobs(method, w, n_values, seeds, n_rounds, subsample, feature_cache_dir)
    results = Parallel(n_jobs=n_parallel)(delayed(_run_one_job)(j, w, method) for j in jobs)
    scores = [_score(df) for df in results if df is not None and len(df)]
    if not scores:
        return float("-inf")  # every job in this trial crashed
    return float(np.mean(scores))


def build_objective(method: str, space: str, n_values, seeds, n_rounds: int,
                    subsample: float, feature_cache_dir: Path, n_parallel: int):
    def objective(trial: optuna.Trial) -> float:
        w = _sample_weights(trial, space)
        return _evaluate(method, w, n_values, seeds, n_rounds, subsample,
                         feature_cache_dir, n_parallel)

    return objective


def run_transfer_check(study: optuna.Study, args, space: str) -> None:
    """Re-score the top-3 val configs on seeds never used during tuning.

    A hyperparameter set chosen on 3 tuning seeds can be fit to those seeds'
    partition draw rather than to the method. If the top-3 ranking survives on
    held-out seeds, the winner is a property of the method; if it reshuffles,
    the search resolved noise and the paper should say so.
    """
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(done) < 3:
        raise SystemExit(f"need >=3 completed trials to transfer-check, found {len(done)}")
    top3 = sorted(done, key=lambda t: t.value, reverse=True)[:3]

    rows = []
    for rank, t in enumerate(top3, 1):
        w = _sample_weights(optuna.trial.FixedTrial(t.params), space)
        transfer = _evaluate(args.method, w, args.n_values, args.transfer_seeds,
                             args.n_rounds, args.subsample,
                             Path(args.feature_cache_dir), args.n_parallel_jobs_per_trial)
        rows.append({"tune_rank": rank, "trial": t.number,
                     "val_tune": t.value, "val_transfer": transfer})
        print(f"rank {rank} (trial {t.number}): tune={t.value:.4f} transfer={transfer:.4f}")

    df = pd.DataFrame(rows)
    winner_holds = bool(df.loc[df["val_transfer"].idxmax(), "tune_rank"] == 1)
    print(f"\ntop-1 config still best on held-out seeds: {winner_holds}")
    if not winner_holds:
        print("  -> the search is resolving seed noise, not method quality. "
              "Report this and prefer the config that wins on transfer.")

    out = Path(args.storage.replace("sqlite:///", "")).parent / f"{args.study_name}_transfer.csv"
    df.to_csv(out, index=False)
    print(f"transfer table: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="proposed_hfl",
                    help="method to tune; one study per method (F2 fairness fix)")
    ap.add_argument("--space", choices=["auto", "recipe", "full"], default="auto",
                    help="auto = full for proposed_hfl, recipe for everything else")
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--n-parallel-jobs-per-trial", type=int, default=4)
    ap.add_argument("--n-rounds", type=int, default=20)
    ap.add_argument("--subsample", type=float, default=0.2)
    ap.add_argument("--n-values", type=int, nargs="+", default=[30, 50])
    ap.add_argument("--seeds", type=int, nargs="+", default=[20, 21, 22],
                    help="tuning seed indices; must not overlap evaluation seeds 0-19")
    ap.add_argument("--transfer-seeds", type=int, nargs="+", default=[23, 24])
    ap.add_argument("--transfer-check", action="store_true",
                    help="skip tuning; re-score the study's top-3 on --transfer-seeds")
    ap.add_argument("--feature-cache-dir", default=str(REPO_ROOT / "results" / "paper_smoke"),
                    help="reuse an existing sweep's ResNet feature caches")
    ap.add_argument("--study-name", default=None,
                    help="defaults to weights_<method>, so studies never collide")
    ap.add_argument(
        "--storage",
        default=f"sqlite:///{REPO_ROOT}/results/hpo/weights.db",
        help="Optuna storage URL; sqlite makes the study resumable/inspectable.",
    )
    args = ap.parse_args()

    leaked = sorted(set(args.seeds) & set(RESERVED_EVAL_SEEDS))
    if leaked:
        raise SystemExit(
            f"tuning seeds {leaked} are in the reserved evaluation range "
            f"{RESERVED_EVAL_SEEDS.start}-{RESERVED_EVAL_SEEDS.stop - 1}. Tuning on a "
            "seed the paper also evaluates on leaks the evaluation draw into the "
            "hyperparameter choice. Use 20+."
        )
    if set(args.seeds) & set(args.transfer_seeds):
        raise SystemExit("--transfer-seeds must be disjoint from --seeds")

    space = args.space
    if space == "auto":
        space = "full" if args.method in FULL_SPACE_METHODS else "recipe"
    study_name = args.study_name or f"weights_{args.method}"
    args.study_name = study_name

    Path(args.storage.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name, storage=args.storage,
        direction="maximize", load_if_exists=True,
    )
    # Replaying a stored trial requires knowing which space produced it.
    prior_space = study.user_attrs.get("space")
    if prior_space is None:
        study.set_user_attr("space", space)
        study.set_user_attr("method", args.method)
    elif prior_space != space:
        raise SystemExit(
            f"study {study_name!r} was created with space={prior_space!r} but "
            f"space={space!r} was requested. Mixing spaces in one study makes its "
            "trials non-comparable; use a different --study-name."
        )

    if args.transfer_check:
        run_transfer_check(study, args, space)
        return

    print(f"tuning method={args.method} space={space} seeds={args.seeds} "
          f"N={args.n_values} rounds={args.n_rounds} subsample={args.subsample}")
    objective = build_objective(
        args.method, space, args.n_values, args.seeds, args.n_rounds,
        args.subsample, Path(args.feature_cache_dir), args.n_parallel_jobs_per_trial,
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("\n=== Best trial (validation score) ===")
    print(f"value: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    leaderboard = study.trials_dataframe().sort_values("value", ascending=False)
    out_csv = Path(args.storage.replace("sqlite:///", "")).parent / f"{study_name}_leaderboard.csv"
    leaderboard.to_csv(out_csv, index=False)
    print(f"\nFull leaderboard: {out_csv}")
    print("Next: --transfer-check to confirm the winner survives held-out seeds.")


if __name__ == "__main__":
    main()
