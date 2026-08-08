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
import traceback
from pathlib import Path

import pandas as pd
import yaml
from joblib import Parallel, delayed

from ..reporting.tables import read_table, write_table
from .federated import _dump_resolved_cfg

logger = logging.getLogger("uavbench.fl.sweep")

# Bump whenever a code change alters RNG draw order or numerics (e.g. the
# 2026-07-17 tensor-sliced shard loader): checkpoints written by an older
# version are no longer reproducible under the current code, so the resume
# gate refuses them and the affected jobs rerun.
#
# v4 (2026-08-06): `placement_fitness` is now the equal-radius, canonical-
# normaliser re-score of the returned layout rather than each optimizer's own
# self-reported best_fitness (see _place_uavs in federated.py). FL numerics are
# untouched, but resuming a v3 checkpoint would splice old-convention rounds
# onto new-convention rounds inside one column, so v3 checkpoints are refused.
PIPELINE_VERSION = 4

# Checkpoint-resume config comparison: sections that define what a job
# computes. Keys in _RESUME_VOLATILE_DATA_KEYS are locations/credentials, not
# semantics (the feature cache is separately validated against the dataset by
# compute_feature_cache/CachedDataset).
_RESUME_SIG_KEYS = ("data", "fl", "budget", "methods", "optimizer_params", "epicentre")
_RESUME_VOLATILE_DATA_KEYS = ("hf_token", "prebuilt", "feature_cache_path")


def _resume_signature(cfg: dict) -> dict:
    """The job-defining subset of a config, normalized through YAML.

    The YAML round-trip maps tuples/np scalars to the plain types a reloaded
    checkpoint config carries, so stored and freshly-built signatures compare
    equal iff the job would compute the same thing.
    """
    sig: dict = {"_pipeline_version": cfg.get("_pipeline_version")}
    for key in _RESUME_SIG_KEYS:
        val = copy.deepcopy(cfg.get(key))
        if key == "data" and isinstance(val, dict):
            for vk in _RESUME_VOLATILE_DATA_KEYS:
                val.pop(vk, None)
        sig[key] = val
    return yaml.safe_load(yaml.safe_dump(sig))


def _stale_checkpoint_reason(ckpt_path: Path, job_cfg: dict) -> str | None:
    """None if the checkpoint at ``ckpt_path`` matches ``job_cfg``; else why not.

    Guards the resume gate: a resolved-config file left behind by an earlier
    run with different settings (or an older pipeline version) must not be
    silently reused as this job's result — that is how the 2026-07-14 smoke
    leftovers ended up standing in for full paper runs.
    """
    try:
        with open(ckpt_path) as f:
            stored = yaml.safe_load(f)
    except Exception as exc:  # unreadable/corrupt checkpoint → rerun
        return f"unreadable checkpoint config ({exc})"
    if not isinstance(stored, dict):
        return "checkpoint config is not a mapping"

    stored_sig = _resume_signature(stored)
    current_sig = _resume_signature(job_cfg)
    if stored_sig == current_sig:
        return None
    for key, cur in current_sig.items():
        if stored_sig.get(key) != cur:
            return f"config mismatch in {key!r} (stored {stored_sig.get(key)!r} != current {cur!r})"
    return "config mismatch"


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
    if cfg["data"].get("source", "real") != "real":
        return  # prebuilt test injections carry their own features; nothing to pre-fetch

    from hflsim.data import get_hfl_data_partitions

    from .features import compute_feature_cache
    from .seeds import partition_seed_for

    data_cfg = cfg["data"]
    hf_token = os.environ.get("HF_TOKEN", data_cfg.get("hf_token"))
    results_dir = Path(cfg["results_dir"])

    for N in cfg["N_values"]:
        logger.info(
            "[prefetch] N=%d — loading dataset (first call streams HF, subsequent calls hit disk cache) …",
            N,
        )
        full_dataset, *_ = get_hfl_data_partitions(
            csv_path=data_cfg.get("csv_path"),
            data_dir=data_cfg.get("data_dir", "./data"),
            N=N,
            subsample=data_cfg.get("subsample", 0.05),
            random_seed=data_cfg.get("seed", 42),
            hf_token=hf_token,
        )
        # Warm the per-seed partition caches too (cheap K-means only — the
        # rows and feature cache are seed-independent), so parallel workers
        # never race on computing the same partition.
        for seed_idx in range(cfg.get("n_seeds", 1)):
            get_hfl_data_partitions(
                csv_path=data_cfg.get("csv_path"),
                data_dir=data_cfg.get("data_dir", "./data"),
                N=N,
                subsample=data_cfg.get("subsample", 0.05),
                random_seed=data_cfg.get("seed", 42),
                hf_token=hf_token,
                partition_seed=partition_seed_for(seed_idx),
            )
        # data.feature_cache_dir lets sweep variants share one cache. The cache
        # keys only on (N, data.seed, subsample), so every variant of a config
        # would otherwise recompute a byte-identical 132 MB file — 12 of them
        # for the Phase 6 environment screen alone.
        cache_root = Path(data_cfg.get("feature_cache_dir") or results_dir)
        cache_path = str(cache_root / f"N{N}" / "img_features.npy")
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        compute_feature_cache(
            full_dataset,
            cache_path=cache_path,
            batch_size=data_cfg.get("feature_batch_size", 32),
            num_workers=data_cfg.get("feature_num_workers", 0),
        )
        logger.info("[prefetch] N=%d — done.", N)


# ---------------------------------------------------------------------------
# Per-job worker
# ---------------------------------------------------------------------------


def _job(N: int, method: str, cfg: dict) -> pd.DataFrame:
    """Single (N, method) FL run, executed inside a joblib worker process.

    Resumable: ``run_tier2`` writes ``config.tier2.resolved.yaml`` as its
    last step, so that file's presence means the job fully finished on a
    prior attempt — reload its already-written table instead of redoing the
    FL run. Each (N, method) gets its own sub-directory: sharing one dir per
    N across methods (as a pre-checkpointing version of this function did)
    made every method's write silently clobber the previous method's output
    file, which was harmless when nothing ever read it back but would make
    the resume check reload the wrong method's results.
    """
    job_cfg = copy.deepcopy(cfg)
    job_cfg["data"]["N_clients"] = N
    job_cfg["methods"] = [method]
    job_cfg["_pipeline_version"] = PIPELINE_VERSION
    job_results_dir = Path(cfg["results_dir"]) / f"N{N}" / method
    job_cfg["results_dir"] = str(job_results_dir)

    ckpt = job_results_dir / "config.tier2.resolved.yaml"
    if ckpt.exists():
        stale = _stale_checkpoint_reason(ckpt, job_cfg)
        if stale is None:
            logger.info("[N=%d  method=%-10s] checkpoint found — skipping (resume)", N, method)
            df = read_table(job_results_dir / "tier2_rounds.parquet")
            df.insert(0, "N", N)
            return df
        logger.warning(
            "[N=%d  method=%-10s] STALE checkpoint at %s — rerunning (%s)",
            N,
            method,
            job_results_dir,
            stale,
        )

    # Point to the shared N-level feature cache (one dir above) produced by
    # _prefetch_all_N so parallel method workers don't each re-run the full
    # ResNet forward pass.
    if job_cfg["data"].get("source", "real") == "real":
        job_cfg["data"]["feature_cache_path"] = str(
            Path(cfg["results_dir"]) / f"N{N}" / "img_features.npy"
        )

    import torch

    torch.set_num_threads(1)  # limit intra-op parallelism so 12 workers don't thrash BLAS

    from .federated import run_tier2  # import inside worker to avoid fork issues

    logger.info("[N=%d  method=%-10s] starting", N, method)
    out = run_tier2(job_cfg)
    df = out["rounds"].copy()
    df.insert(0, "N", N)
    final_acc = float(df["accuracy"].iloc[-1]) if len(df) else float("nan")
    final_f1 = float(df["macro_f1"].iloc[-1]) if len(df) else float("nan")
    logger.info(
        "[N=%d  method=%-10s] done | acc=%.3f  macro-F1=%.3f",
        N,
        method,
        final_acc,
        final_f1,
    )
    return df


# ---------------------------------------------------------------------------
# Job failure isolation
# ---------------------------------------------------------------------------


def _isolated(job_fn, tag: str, *args) -> tuple[str, pd.DataFrame | None, str | None]:
    """Run one sweep job, converting any exception into a returned record.

    joblib's default fail-fast aborts the whole sweep on the first job error,
    discarding hours of sibling-job compute (the 2026-07-16 run lost ~2.5 h to
    one stale-cache IndexError). Instead every job runs to completion or
    failure; the caller collects the failures and raises once, after the
    surviving jobs have written their checkpoints.
    """
    try:
        return tag, job_fn(*args), None
    except Exception:
        logger.exception("[%s] job FAILED", tag)
        return tag, None, traceback.format_exc()


def _collect_or_raise(results: list[tuple[str, pd.DataFrame | None, str | None]]) -> pd.DataFrame:
    """Concatenate successful job frames; raise (loudly, with every traceback) if any failed."""
    failures = [(tag, err) for tag, _, err in results if err is not None]
    if failures:
        for tag, err in failures:
            logger.error("Job %s failed:\n%s", tag, err)
        raise RuntimeError(
            f"{len(failures)}/{len(results)} sweep jobs failed: "
            f"{', '.join(tag for tag, _ in failures)}. Completed jobs are "
            "checkpointed in their per-job directories — fix the cause and "
            "rerun to resume. Full tracebacks are in the log above."
        )
    return pd.concat([df for _, df, _ in results], ignore_index=True)


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
        len(N_values),
        len(methods),
        len(jobs),
        n_workers,
    )

    results = Parallel(n_jobs=n_workers, backend="loky", verbose=5)(
        delayed(_isolated)(_job, f"N{N}/{method}", N, method, cfg) for N, method in jobs
    )

    full_df = _collect_or_raise(results)

    write_table(full_df, results_dir / "sweep_rounds.parquet")

    _dump_resolved_cfg(cfg, results_dir / "config.sweep.resolved.yaml")

    size_mb = sum(p.stat().st_size for p in results_dir.rglob("*") if p.is_file()) / 1e6
    logger.info("Sweep complete — %.2f MB at %s", size_mb, results_dir)

    return {"rounds": full_df, "results_dir": results_dir, "size_mb": size_mb}


# ---------------------------------------------------------------------------
# Paper full-system sweep (N × method × seed)
# ---------------------------------------------------------------------------


def _paper_job(N: int, method: str, seed_idx: int, cfg: dict) -> pd.DataFrame:
    """Single (N, method, seed) full-system run, executed inside a joblib worker.

    Resumable: see ``_job`` above — ``run_full_hfl`` writes
    ``config.fullsim.resolved.yaml`` last, so its presence gates the skip.
    """
    from .seeds import partition_seed_for, sweep_job_seed

    job_cfg = copy.deepcopy(cfg)
    job_cfg["data"]["N_clients"] = N
    # Partition/test-split varies per seed repetition (method-free — every
    # method sees the identical partition for a given seed, keeping the
    # Wilcoxon pairing valid).
    job_cfg["data"]["partition_seed"] = partition_seed_for(seed_idx)
    job_cfg["methods"] = [method]
    # Per-method training recipe (rigor plan §0.3b). lr / schedule / tau /
    # momentum / ema were previously fit to proposed_hfl alone and imposed on
    # every baseline, making the headline comparison tuned-vs-untuned. Each
    # method now carries the recipe from its OWN validation study; methods with
    # no entry (ablations, flat_fl, centralized) keep the base recipe, which is
    # proposed_hfl's — an ablation must differ from it only in the ablated part.
    #
    # POPPED, not merged: leaving the whole map in job_cfg would put every
    # method's recipe into every job's resume signature, so editing one
    # method's numbers would invalidate all the others' checkpoints. Popping
    # also keeps methods without an entry byte-identical to a pre-0.3b config,
    # so their existing checkpoints stay valid.
    _per_method = job_cfg["fl"].pop("per_method", None) or {}
    job_cfg["fl"].update(_per_method.get(method, {}))
    # Method identity is folded into the seed exactly once, inside run_full_hfl
    # (via fullsim_method_seed). Do NOT add a method hash here too, or every
    # (N, seed_idx) job's seed double-counts the method and silently shifts
    # the per-method RNG draw in a way that is no longer reproducible from
    # this function's inputs alone.
    job_cfg["fl"]["seed"] = sweep_job_seed(cfg.get("optimizer_seed", 9876), seed_idx, N)
    job_cfg["_pipeline_version"] = PIPELINE_VERSION
    # Each (N, method, seed) gets its own sub-directory so parallel workers
    # never write to the same fullsim_rounds.parquet simultaneously.  Without
    # the method segment, all 6 methods for a given (N, seed) share one dir
    # and race to overwrite each other's output file — the last writer wins
    # and earlier results are silently lost.
    job_results_dir = Path(cfg["results_dir"]) / f"N{N}" / f"seed{seed_idx}" / method
    job_cfg["results_dir"] = str(job_results_dir)

    ckpt = job_results_dir / "config.fullsim.resolved.yaml"
    if ckpt.exists():
        stale = _stale_checkpoint_reason(ckpt, job_cfg)
        if stale is None:
            logger.info(
                "[N=%d  method=%-20s  seed=%d] checkpoint found — skipping (resume)",
                N,
                method,
                seed_idx,
            )
            df = read_table(job_results_dir / "fullsim_rounds.parquet")
            df.insert(0, "seed", seed_idx)
            df.insert(0, "N", N)
            return df
        logger.warning(
            "[N=%d  method=%-20s  seed=%d] STALE checkpoint at %s — rerunning (%s)",
            N,
            method,
            seed_idx,
            job_results_dir,
            stale,
        )

    # Point to the shared N-level feature cache (one dir above) produced by
    # _prefetch_all_N so parallel method/seed workers don't each re-run the
    # full ResNet forward pass.
    if job_cfg["data"].get("source", "real") == "real":
        job_cfg["data"]["feature_cache_path"] = str(
            Path(cfg["results_dir"]) / f"N{N}" / "img_features.npy"
        )

    import torch

    torch.set_num_threads(1)

    from .federated import run_full_hfl

    logger.info("[N=%d  method=%-20s  seed=%d] starting", N, method, seed_idx)
    out = run_full_hfl(job_cfg)
    df = out["rounds"].copy()
    df.insert(0, "seed", seed_idx)
    df.insert(0, "N", N)
    final_acc = float(df["accuracy"].iloc[-1]) if len(df) else float("nan")
    final_f1 = float(df["macro_f1"].iloc[-1]) if len(df) else float("nan")
    logger.info(
        "[N=%d  method=%-20s  seed=%d] done | acc=%.3f  macro-F1=%.3f",
        N,
        method,
        seed_idx,
        final_acc,
        final_f1,
    )
    return df


def run_paper_sweep(cfg: dict) -> dict:
    """Run the full paper simulation sweep: N × method × seed grid.

    Phase 1 (sequential): pre-fetch dataset for all N values.
    Phase 2 (parallel):   run (N, method, seed) jobs with n_workers workers.

    Returns dict with ``"rounds"`` (full DataFrame), ``"results_dir"``, ``"size_mb"``.
    """
    N_values: list[int] = cfg["N_values"]
    methods: list[str] = cfg["methods"]
    n_seeds: int = cfg.get("n_seeds", 1)
    n_workers: int = cfg.get("n_workers", 12)
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: sequential data pre-fetch ---
    logger.info("Phase 1: pre-fetching data for %d N-values (sequential)…", len(N_values))
    _prefetch_all_N(cfg)
    logger.info("Phase 1 complete — all caches ready.")

    # --- Phase 2: parallel full-system sweep ---
    jobs = [
        (N, method, seed_idx) for N in N_values for method in methods for seed_idx in range(n_seeds)
    ]
    logger.info(
        "Phase 2: %d N × %d methods × %d seeds = %d jobs — %d parallel workers",
        len(N_values),
        len(methods),
        n_seeds,
        len(jobs),
        n_workers,
    )

    results = Parallel(n_jobs=n_workers, backend="loky", verbose=5)(
        delayed(_isolated)(_paper_job, f"N{N}/seed{seed_idx}/{method}", N, method, seed_idx, cfg)
        for N, method, seed_idx in jobs
    )

    full_df = _collect_or_raise(results)

    write_table(full_df, results_dir / "paper_sweep_rounds.parquet")

    _dump_resolved_cfg(cfg, results_dir / "config.paper_sweep.resolved.yaml")

    size_mb = sum(p.stat().st_size for p in results_dir.rglob("*") if p.is_file()) / 1e6
    logger.info("Paper sweep complete — %.2f MB at %s", size_mb, results_dir)

    return {"rounds": full_df, "results_dir": results_dir, "size_mb": size_mb}


# ---------------------------------------------------------------------------
# Coverage-constrained sweep (R_comm × method × seed, fixed N)
# ---------------------------------------------------------------------------


def _coverage_job(r_comm: float, method: str, seed_idx: int, cfg: dict) -> pd.DataFrame:
    """Single (R_comm, method, seed) run at fixed N — mirrors ``_paper_job``.

    Only ``fl.R_comm`` varies across jobs of one seed; N, partition, and the
    feature cache are shared, so this isolates *how much placement quality
    matters as coverage binds*. At large R_comm every client is covered
    (placement moot); as R_comm shrinks, a better placement covers more (and,
    class-aware, more rare-class) clients — so macro_f1 separates by placement.
    """
    from .seeds import partition_seed_for, sweep_job_seed

    N = int(cfg["N"])
    job_cfg = copy.deepcopy(cfg)
    job_cfg["data"]["N_clients"] = N
    job_cfg["data"]["partition_seed"] = partition_seed_for(seed_idx)
    job_cfg["methods"] = [method]
    job_cfg["fl"]["seed"] = sweep_job_seed(cfg.get("optimizer_seed", 9876), seed_idx, N)
    job_cfg["fl"]["R_comm"] = float(r_comm)

    # Equal-radius calibration: the path-loss baselines derive their own coverage
    # radius from the link budget, so at a swept R_comm they must be re-calibrated
    # or they place for the wrong radius and lose on the mismatch, not the rule.
    #
    # Only the method this job actually runs is calibrated. run_full_hfl reads
    # optimizer_params[method], so touching the other entries changes nothing it
    # computes — but optimizer_params IS part of the resume signature, so editing
    # them all would mark every previously-finished job stale and force a full
    # recompute (measured: 300/300 existing jobs invalidated instead of 120).
    if job_cfg.get("calibrate_path_loss", True) and method in ("mozaffari2016", "alzenad2017"):
        from ..problem.path_loss import ENV_PRESETS, path_loss_db_for_radius

        op = job_cfg.setdefault("optimizer_params", {})
        for base in (method,):
            if base not in op:
                continue
            env = ENV_PRESETS[op[base].get("environment", "suburban")]
            freq_hz = float(op[base].get("freq_ghz", 2.0)) * 1e9
            op[base]["max_path_loss_db"] = path_loss_db_for_radius(
                float(r_comm), freq_hz, env["a"], env["b"],
                env["eta_los_db"], env["eta_nlos_db"], 20.0, 120.0,
            )

    job_cfg["_pipeline_version"] = PIPELINE_VERSION

    r_tag = f"R{int(round(r_comm))}"
    job_results_dir = Path(cfg["results_dir"]) / r_tag / f"seed{seed_idx}" / method
    job_cfg["results_dir"] = str(job_results_dir)

    ckpt = job_results_dir / "config.fullsim.resolved.yaml"
    if ckpt.exists() and _stale_checkpoint_reason(ckpt, job_cfg) is None:
        logger.info("[%s method=%-20s seed=%d] checkpoint — skipping", r_tag, method, seed_idx)
        df = read_table(job_results_dir / "fullsim_rounds.parquet")
        df.insert(0, "seed", seed_idx)
        df.insert(0, "R_comm", float(r_comm))
        return df

    # Share the single-N feature cache (N is fixed across the R_comm sweep).
    if job_cfg["data"].get("source", "real") == "real":
        job_cfg["data"]["feature_cache_path"] = str(Path(cfg["results_dir"]) / f"N{N}" / "img_features.npy")

    import torch

    torch.set_num_threads(1)
    from .federated import run_full_hfl

    logger.info("[%s method=%-20s seed=%d] starting", r_tag, method, seed_idx)
    df = run_full_hfl(job_cfg)["rounds"].copy()
    df.insert(0, "seed", seed_idx)
    df.insert(0, "R_comm", float(r_comm))
    return df


def _uav_job(k: int, capacity: int, method: str, seed_idx: int, cfg: dict) -> pd.DataFrame:
    """Single (K, capacity, method, seed) run at fixed N and R_comm.

    Mirrors ``_coverage_job`` but sweeps the *fleet size* instead of the radio
    range. Both make coverage bind; K is the more direct knob, because at the
    paper's R_comm = 20 km a single disc already covers a large fraction of the
    Noto peninsula, so coverage saturates by K ≈ 4-6 and every placement rule
    looks alike above that (measured: 99.95% at K=20).

    ``capacity`` is swept *with* K rather than held fixed, and that is the whole
    point of the design. Holding per-UAV capacity constant would make total
    training slots K·C scale with K, so a small-K cell would starve the model of
    data at the same time as it stresses placement — and a flat macro_f1 across
    methods would then be unreadable: data starvation and placement irrelevance
    look identical. Pairing each K with a capacity that keeps K·C roughly
    constant holds the per-round data volume fixed and leaves coverage geometry
    as the only thing that changes. The per-UAV capacities at small K are
    correspondingly unrealistic; that is the price of a controlled ablation and
    must be stated when reporting it.
    """
    from .seeds import partition_seed_for, sweep_job_seed

    N = int(cfg["N"])
    job_cfg = copy.deepcopy(cfg)
    job_cfg["data"]["N_clients"] = N
    job_cfg["data"]["partition_seed"] = partition_seed_for(seed_idx)
    job_cfg["methods"] = [method]
    job_cfg["fl"]["seed"] = sweep_job_seed(cfg.get("optimizer_seed", 9876), seed_idx, N)
    job_cfg["fl"]["K"] = int(k)
    job_cfg["fl"]["capacity"] = int(capacity)
    job_cfg["_pipeline_version"] = PIPELINE_VERSION

    k_tag = f"K{int(k)}"
    job_results_dir = Path(cfg["results_dir"]) / k_tag / f"seed{seed_idx}" / method
    job_cfg["results_dir"] = str(job_results_dir)

    ckpt = job_results_dir / "config.fullsim.resolved.yaml"
    if ckpt.exists() and _stale_checkpoint_reason(ckpt, job_cfg) is None:
        logger.info("[%s method=%-20s seed=%d] checkpoint — skipping", k_tag, method, seed_idx)
        df = read_table(job_results_dir / "fullsim_rounds.parquet")
        df.insert(0, "seed", seed_idx)
        df.insert(0, "capacity", int(capacity))
        df.insert(0, "K", int(k))
        return df

    # Share the single-N feature cache (N is fixed across the K sweep).
    if job_cfg["data"].get("source", "real") == "real":
        job_cfg["data"]["feature_cache_path"] = str(Path(cfg["results_dir"]) / f"N{N}" / "img_features.npy")

    import torch

    torch.set_num_threads(1)
    from .federated import run_full_hfl

    logger.info("[%s cap=%d method=%-20s seed=%d] starting", k_tag, capacity, method, seed_idx)
    df = run_full_hfl(job_cfg)["rounds"].copy()
    df.insert(0, "seed", seed_idx)
    df.insert(0, "capacity", int(capacity))
    df.insert(0, "K", int(k))
    return df


def run_uav_sweep(cfg: dict) -> dict:
    """Run the fleet-size sweep: (K, capacity) × method × seed at fixed N, R_comm.

    ``K_values`` is a list of ``{K: <int>, capacity: <int>}`` mappings — the
    pairing is explicit in the config rather than derived here, so the design is
    auditable from the YAML alone.
    """
    N = int(cfg["N"])
    raw: list = cfg["K_values"]
    methods: list[str] = cfg["methods"]
    n_seeds: int = cfg.get("n_seeds", 1)
    n_workers: int = cfg.get("n_workers", 12)
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[int, int]] = []
    for entry in raw:
        if not isinstance(entry, dict) or "K" not in entry or "capacity" not in entry:
            raise ValueError(
                f"K_values entries must be mappings with 'K' and 'capacity'; got {entry!r}. "
                "Capacity is swept with K deliberately — see _uav_job's docstring."
            )
        pairs.append((int(entry["K"]), int(entry["capacity"])))

    slots = [k * c for k, c in pairs]
    logger.info(
        "Fleet-size sweep: K=%s, capacity=%s -> total slots %s (N=%d)",
        [k for k, _ in pairs], [c for _, c in pairs], slots, N,
    )
    if max(slots) > 2 * min(slots):
        logger.warning(
            "Total slots K*C vary by more than 2x across cells (%s). The sweep will "
            "confound data volume with coverage geometry — a flat macro_f1 could mean "
            "either. Re-pair the capacities unless this is intended.",
            slots,
        )

    logger.info("Phase 1: pre-fetching data for N=%d …", N)
    _prefetch_all_N({**cfg, "N_values": [N]})
    logger.info("Phase 1 complete.")

    jobs = [(k, c, m, s) for k, c in pairs for m in methods for s in range(n_seeds)]
    logger.info(
        "Phase 2: %d K-values × %d methods × %d seeds = %d jobs — %d workers",
        len(pairs), len(methods), n_seeds, len(jobs), n_workers,
    )
    results = Parallel(n_jobs=n_workers, backend="loky", verbose=5)(
        delayed(_isolated)(_uav_job, f"K{k}/seed{s}/{m}", k, c, m, s, cfg)
        for k, c, m, s in jobs
    )
    full_df = _collect_or_raise(results)
    write_table(full_df, results_dir / "uav_sweep_rounds.parquet")
    _dump_resolved_cfg(cfg, results_dir / "config.uav.resolved.yaml")
    size_mb = sum(p.stat().st_size for p in results_dir.rglob("*") if p.is_file()) / 1e6
    logger.info("Fleet-size sweep complete — %.2f MB at %s", size_mb, results_dir)
    return {"rounds": full_df, "results_dir": results_dir, "size_mb": size_mb}


def run_coverage_sweep(cfg: dict) -> dict:
    """Run the coverage-constrained sweep: R_comm × method × seed at fixed N."""
    N = int(cfg["N"])
    r_values: list[float] = cfg["R_comm_values"]
    methods: list[str] = cfg["methods"]
    n_seeds: int = cfg.get("n_seeds", 1)
    n_workers: int = cfg.get("n_workers", 12)
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: one-time data + feature prefetch for the single N.
    logger.info("Phase 1: pre-fetching data for N=%d …", N)
    _prefetch_all_N({**cfg, "N_values": [N]})
    logger.info("Phase 1 complete.")

    jobs = [(r, m, s) for r in r_values for m in methods for s in range(n_seeds)]
    logger.info(
        "Phase 2: %d R_comm × %d methods × %d seeds = %d jobs — %d workers",
        len(r_values), len(methods), n_seeds, len(jobs), n_workers,
    )
    results = Parallel(n_jobs=n_workers, backend="loky", verbose=5)(
        delayed(_isolated)(_coverage_job, f"R{int(round(r))}/seed{s}/{m}", r, m, s, cfg)
        for r, m, s in jobs
    )
    full_df = _collect_or_raise(results)
    write_table(full_df, results_dir / "coverage_sweep_rounds.parquet")
    _dump_resolved_cfg(cfg, results_dir / "config.coverage.resolved.yaml")
    size_mb = sum(p.stat().st_size for p in results_dir.rglob("*") if p.is_file()) / 1e6
    logger.info("Coverage sweep complete — %.2f MB at %s", size_mb, results_dir)
    return {"rounds": full_df, "results_dir": results_dir, "size_mb": size_mb}
