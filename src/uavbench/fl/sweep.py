"""N-scalability sweep: run the full (N × method) grid in parallel.

Each job is one (N, method) combination — a complete 30-round FL run for
a specific number of clients and placement strategy. Jobs are parallelised
with joblib across the 8 vCPUs of the n1-standard-8 GCP instance.

Thread budget:
    Each worker calls ``torch.set_num_threads(1)`` so MKL/OpenBLAS does not
    spawn extra threads per job. Total active threads = n_workers × 1 = 12,
    matching the 12-vCPU budget exactly.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import pandas as pd
import yaml
from joblib import Parallel, delayed

logger = logging.getLogger("uavbench.fl.sweep")


def _job(N: int, method: str, cfg: dict) -> pd.DataFrame:
    """Single (N, method) FL run, executed inside a joblib worker process."""
    import torch
    torch.set_num_threads(1)  # intra-op; interop set via OMP_NUM_THREADS env var in run_gcp.sh

    from .federated import run_tier2  # import inside worker to avoid fork issues

    job_cfg = copy.deepcopy(cfg)
    job_cfg["data"]["N_clients"] = N
    job_cfg["methods"] = [method]
    # Each N gets its own cache sub-directory so image-feature .npy files
    # for different partition sizes never collide.
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


def run_sweep(cfg: dict) -> dict:
    """Run the full (N × method) scalability sweep and write consolidated results.

    The config must include an ``N_values`` list. All other keys follow the
    same schema as ``tier2_fl.yaml``.

    Returns
    -------
    dict with keys ``"rounds"`` (full DataFrame), ``"results_dir"``, ``"size_mb"``.
    """
    N_values: list[int] = cfg["N_values"]
    methods: list[str] = cfg["methods"]
    n_workers: int = cfg.get("n_workers", 8)
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(N, method) for N in N_values for method in methods]
    total = len(jobs)
    logger.info(
        "Sweep: %d N-values × %d methods = %d jobs — %d parallel workers",
        len(N_values), len(methods), total, n_workers,
    )

    dfs = Parallel(n_jobs=n_workers, backend="loky", verbose=5)(
        delayed(_job)(N, method, cfg) for N, method in jobs
    )

    full_df = pd.concat(dfs, ignore_index=True)

    # Consolidated output
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
