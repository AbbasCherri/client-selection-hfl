"""N-scalability sweep: run the full (N × method) grid in parallel.

Each job is one (N, method) combination — a complete 30-round FL run for
a specific number of clients and placement strategy. Jobs are parallelised
with joblib across the 12 vCPUs of the GCP instance.

Thread budget:
    Each worker calls ``torch.set_num_threads(1)`` so MKL/OpenBLAS does not
    spawn extra threads per job. Total active threads = n_workers × 1 = 12,
    matching the 12-vCPU budget exactly.

HuggingFace rate-limit strategy:
    All N-value datasets are streamed and cached **sequentially** in
    ``_prefetch_all_N()`` before any parallel worker starts. Workers then
    load only from the local partition cache and image-feature .npy files —
    zero HF API calls during parallel execution.
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

logger = logging.getLogger("uavbench.fl.sweep")


# ---------------------------------------------------------------------------
# Sequential pre-fetch — runs before any parallel worker
# ---------------------------------------------------------------------------

def _prefetch_all_N(cfg: dict) -> None:
    """Stream + cache data for every N value sequentially.

    For each N this function:
    1. Calls ``get_hfl_data_partitions`` (which writes to .partition_cache/).
    2. Runs ``compute_feature_cache`` to save the ResNet-18 .npy file.

    After this step, parallel workers only touch local disk.
    """
    if cfg["data"].get("source", "synthetic") == "synthetic":
        return  # synthetic data is generated in-process; nothing to pre-fetch

    from hflsim.data import get_hfl_data_partitions
    from .features import compute_feature_cache

    data_cfg = cfg["data"]
    hf_token = os.environ.get("HF_TOKEN", data_cfg.get("hf_token"))
    results_dir = Path(cfg["results_dir"])

    for N in cfg["N_values"]:
        logger.info("[prefetch] N=%d — streaming dataset from HuggingFace …", N)
        full_dataset, _, _, _, _ = get_hfl_data_partitions(
            csv_path=data_cfg.get("csv_path"),
            data_dir=data_cfg.get("data_dir", "./data"),
            N=N,
            subsample=data_cfg.get("subsample", 0.05),
            random_seed=data_cfg.get("seed", 42),
            hf_token=hf_token,
        )
        cache_path = str(results_dir / f"N{N}" / "img_features.npy")
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        compute_feature_cache(
            full_dataset,
            cache_path=cache_path,
            batch_size=data_cfg.get("feature_batch_size", 32),
            num_workers=0,
        )
        logger.info("[prefetch] N=%d — done.", N)


# ---------------------------------------------------------------------------
# Per-job worker
# ---------------------------------------------------------------------------

def _job(N: int, method: str, cfg: dict) -> pd.DataFrame:
    """Single (N, method) FL run, executed inside a joblib worker process."""
    import torch
    torch.set_num_threads(1)  # intra-op; interop pinned via OMP_NUM_THREADS in run_gcp.sh

    from .federated import run_tier2  # import inside worker to avoid fork issues

    job_cfg = copy.deepcopy(cfg)
    job_cfg["data"]["N_clients"] = N
    job_cfg["methods"] = [method]
    # Each N gets its own sub-directory; the pre-fetched img_features.npy lives here.
    job_cfg["results_dir"] = str(Path(cfg["results_dir"]) / f"N{N}")

    logger.info("[N=%d  method=%-10s] starting", N, method)
    out = run_tier2(job_cfg)
    df = out["rounds"].copy()
    df.insert(0, "N", N)
    final_acc = float(df["accuracy"].iloc[-1]) if len(df) else float("nan")
    final_f1  = float(df["macro_f1"].iloc[-1])  if len(df) else float("nan")
    logger.info(
        "[N=%d  method=%-10s] done | acc=%.3f  macro-F1=%.3f",
        N, method, final_acc, final_f1,
    )
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_sweep(cfg: dict) -> dict:
    """Run the full (N × method) scalability sweep and write consolidated results.

    Phase 1 (sequential): stream and cache all N-value datasets from HuggingFace.
    Phase 2 (parallel):   run FL for every (N, method) job with n_workers workers.

    Returns
    -------
    dict with keys ``"rounds"`` (full DataFrame), ``"results_dir"``, ``"size_mb"``.
    """
    N_values: list[int] = cfg["N_values"]
    methods: list[str] = cfg["methods"]
    n_workers: int = cfg.get("n_workers", 12)
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: sequential data pre-fetch (avoids parallel HF rate limits) ---
    logger.info("Phase 1: pre-fetching data for %d N-values (sequential)…", len(N_values))
    _prefetch_all_N(cfg)
    logger.info("Phase 1 complete — all caches ready.")

    # --- Phase 2: parallel FL sweep ---
    jobs = [(N, method) for N in N_values for method in methods]
    logger.info(
        "Phase 2: %d N-values × %d methods = %d jobs — %d parallel workers",
        len(N_values), len(methods), len(jobs), n_workers,
    )

    dfs = Parallel(n_jobs=n_workers, backend="loky", verbose=5)(
        delayed(_job)(N, method, cfg) for N, method in jobs
    )

    full_df = pd.concat(dfs, ignore_index=True)

    out_path = results_dir / "sweep_rounds.parquet"
    try:
        full_df.to_parquet(out_path, index=False)
    except Exception:
        full_df.to_csv(out_path.with_suffix(".csv"), index=False)

    with open(results_dir / "config.sweep.resolved.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    size_mb = sum(p.stat().st_size for p in results_dir.rglob("*") if p.is_file()) / 1e6
    logger.info("Sweep complete — %.2f MB at %s", size_mb, results_dir)

    return {"rounds": full_df, "results_dir": results_dir, "size_mb": size_mb}
