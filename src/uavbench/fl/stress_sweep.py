"""Synthetic stress-test sweep: robustness evidence for the single-event scope.

The paper's real-data evaluation covers one event (Noto Peninsula 2024);
this sweep is the complementary controlled stress test over conditions
that single event may not exhibit: per-round device dropout, aftershock-
triggered area-wide SNR degradation, and black-chip (unusable image) rate.
Synthetic-only by construction — the knobs plug into
:class:`~uavbench.fl.dataset.SyntheticClientData` and
:class:`~uavbench.fl.device_state.DeviceStateManager`.

Grid modes
----------
Default ("one-axis-at-a-time"): each axis is varied alone while the other
two sit at their baseline (first listed) value — this is the paper-body
grid. ``sweep.full_grid: true`` runs the full Cartesian product for an
appendix-level exhaustive check.

Pairing: the run seed depends only on ``seed_idx`` (via ``sweep_job_seed``
with a constant N), never on the knob values — so for a given seed every
grid cell and method sees the identical base problem, and along-axis
comparisons are paired.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import pandas as pd
import yaml
from joblib import Parallel, delayed

from .seeds import sweep_job_seed

logger = logging.getLogger("uavbench.fl.stress_sweep")

AXES = ("dropout_rates", "snr_degradations_db", "black_chip_rates")


def build_stress_grid(sweep_cfg: dict) -> list[tuple[float, float, float]]:
    """(dropout, snr_degradation_db, black_chip) cells for the configured mode.

    One-axis-at-a-time by default (union of varying each axis with the other
    two at their first-listed baseline, deduplicated, baseline cell included);
    ``full_grid: true`` yields the Cartesian product.
    """
    dropouts = list(sweep_cfg["dropout_rates"])
    snrs = list(sweep_cfg["snr_degradations_db"])
    chips = list(sweep_cfg["black_chip_rates"])

    if sweep_cfg.get("full_grid", False):
        return [(d, s, c) for d in dropouts for s in snrs for c in chips]

    base = (dropouts[0], snrs[0], chips[0])
    cells: list[tuple[float, float, float]] = [base]
    cells += [(d, base[1], base[2]) for d in dropouts[1:]]
    cells += [(base[0], s, base[2]) for s in snrs[1:]]
    cells += [(base[0], base[1], c) for c in chips[1:]]
    return cells


def _job(
    dropout: float, snr_deg: float, black_chip: float,
    method: str, seed_idx: int, cfg: dict,
) -> pd.DataFrame:
    """One (knob-cell, method, seed) full-system run inside a joblib worker."""
    import torch
    torch.set_num_threads(1)

    from .federated import run_full_hfl

    job_cfg = copy.deepcopy(cfg)
    job_cfg["methods"] = [method]
    job_cfg["fl"]["dropout_rate"] = dropout
    job_cfg["fl"]["snr_degradation_db"] = snr_deg
    job_cfg["data"]["black_chip_rate"] = black_chip
    # Seed depends only on seed_idx (constant N): identical base problem in
    # every cell → paired along-axis comparisons. Method identity is folded
    # in exactly once inside run_full_hfl.
    job_cfg["fl"]["seed"] = sweep_job_seed(
        cfg.get("optimizer_seed", 9876), seed_idx, cfg["data"]["N_clients"]
    )
    job_cfg["results_dir"] = str(
        Path(cfg["results_dir"])
        / f"d{dropout}_s{snr_deg}_b{black_chip}" / f"seed{seed_idx}" / method
    )

    logger.info(
        "[d=%.2f snr-%.0fdB chip=%.2f  %-16s seed=%d] starting",
        dropout, snr_deg, black_chip, method, seed_idx,
    )
    out = run_full_hfl(job_cfg)
    df = out["rounds"].copy()
    df.insert(0, "seed", seed_idx)
    df.insert(0, "black_chip_rate", black_chip)
    df.insert(0, "snr_degradation_db", snr_deg)
    df.insert(0, "dropout_rate", dropout)
    return df


def run_stress_sweep(cfg: dict) -> dict:
    """Run the (stress-cell × method × seed) grid; write stress_rounds.parquet.

    Synthetic-only: raises if ``data.source`` isn't ``synthetic`` — the
    knobs have no effect on the real pipeline and a silent no-op sweep
    would be worse than an error.
    """
    if cfg["data"].get("source", "synthetic") != "synthetic":
        raise ValueError("stress sweep is synthetic-only; set data.source: synthetic")

    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    cells = build_stress_grid(cfg["sweep"])
    methods: list[str] = cfg["methods"]
    n_seeds: int = cfg.get("n_seeds", 1)
    n_workers: int = cfg.get("n_workers", 8)

    jobs = [
        (d, s, c, method, seed_idx)
        for (d, s, c) in cells
        for method in methods
        for seed_idx in range(n_seeds)
    ]
    logger.info(
        "Stress sweep: %d cells × %d methods × %d seeds = %d jobs — %d workers",
        len(cells), len(methods), n_seeds, len(jobs), n_workers,
    )

    dfs = Parallel(n_jobs=n_workers, backend="loky", verbose=5)(
        delayed(_job)(d, s, c, method, seed_idx, cfg)
        for d, s, c, method, seed_idx in jobs
    )

    full_df = pd.concat(dfs, ignore_index=True)

    out_path = results_dir / "stress_rounds.parquet"
    try:
        full_df.to_parquet(out_path, index=False)
    except Exception:
        full_df.to_csv(out_path.with_suffix(".csv"), index=False)

    with open(results_dir / "config.stress.resolved.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    size_mb = sum(p.stat().st_size for p in results_dir.rglob("*") if p.is_file()) / 1e6
    logger.info("Stress sweep complete — %.2f MB at %s", size_mb, results_dir)

    return {"rounds": full_df, "results_dir": results_dir, "size_mb": size_mb}
