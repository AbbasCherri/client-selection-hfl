"""Selection-isolation benchmark: client-selection rules head-to-head.

Compares the client-selection algorithms (proposed UCB, random, and the
literature baselines B1-B3 from REPORTS/master_implementation_reference.md Appendix C) under
conditions where the selection rule is the *only* experimental variable:

- **Static UAVs.** UAV positions are the K-means cluster centres of the
  client coordinates, with K chosen by the elbow method on the K-means
  inertia curve. Placed once before round 1, never moved — no PSO, no
  repositioning, no placement-driven coverage churn.
- **Shared seed across modes.** For a given (N, seed) every mode gets the
  identical torch/numpy seed: same model init, same device heterogeneity,
  same UAV layout. (``run_full_hfl`` deliberately folds the method name into
  the seed; here we deliberately do not — isolation over independence.)
- **Identical FL pipeline.** Local training, UAV image training, reputation
  tracking, trimmed-mean cluster reputation, R_min-gated reputation FedAvg —
  all reused verbatim from ``federated.py`` for every mode.

Efficiency (logic-identical optimizations only)
-----------------------------------------------
- Placement + coverage computed once per run (static ground sensors + UAVs).
- Per-client shard loaders built once (shard indices never change); UAV pooled
  loaders rebuilt only when the roster changes (every T_sel rounds).
- Test loader built once; evaluation reuses it every round.
- ``run_selection_sweep`` parallelises the (N × mode × seed) grid with joblib;
  each worker pins torch to one thread. Data/feature caches are pre-fetched
  sequentially per N (same pattern as ``sweep.run_paper_sweep``).

Entry points
------------
``run_selection_isolation(cfg)`` — all modes for one (N, seed).
``run_selection_sweep(cfg)``     — full (N × mode × seed) grid, parallel.
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.cluster import KMeans

from hflsim.shared.coords import latlon_to_meters
from uavbench.metrics.fl import (
    confusion_rows as _confusion_rows,
)
from uavbench.metrics.fl import (
    TARGET_CONSEC_ROUNDS as _TARGET_CONSEC_ROUNDS,
)
from uavbench.metrics.fl import (
    evaluate_subset as _evaluate_subset,
)
from uavbench.metrics.fl import (
    jain_index,
    round_comm_mb,
)

from ..reporting.tables import read_table
from .client_selection import ClientSelector
from .dataset import BalancedShardLoader, CachedDataset, ClientData, make_client_loader
from .device_state import DeviceStateManager
from .federated import (
    _covered_clients,
    _dump_resolved_cfg,
    _load_data,
    _local_train,
    _uav_local_train,
    _uav_pos_to_latlon,
    _write_table,
)
from .model import CachedFusionModel, fedavg, reputation_fedavg
from .reputation import ReputationManager, trimmed_mean
from .seeds import partition_seed_for, sweep_job_seed

logger = logging.getLogger("uavbench.fl.selection_isolation")

# Selection rules under test. "ucb" is the proposed system (Algorithms 1-4);
# the rest are the random ablation and literature baselines B1-B3.
DEFAULT_MODES: list[str] = [
    "ucb",
    "random",
    "fedcs",
    "rep_cap",
    "fair_mab",
    "oort",
    "power_of_choice",
]

_UAV_ALTITUDE_M = 70.0  # hover altitude used throughout the repo


# ---------------------------------------------------------------------------
# Static placement: elbow-method K-means
# ---------------------------------------------------------------------------


def elbow_k(
    xy_m: np.ndarray,
    k_min: int,
    k_max: int,
    seed: int = 0,
) -> tuple[int, np.ndarray]:
    """Choose K by the elbow method and return (K, cluster centres in metres).

    Fits K-means for every K in [k_min, k_max] and picks the elbow of the
    inertia curve: the K whose (K, inertia) point, after min-max normalising
    both axes, lies farthest from the chord joining the curve's endpoints
    (the standard geometric "max distance to line" elbow criterion).
    """
    n = len(xy_m)
    k_max = max(k_min, min(k_max, n - 1))
    ks = list(range(k_min, k_max + 1))

    fits = {k: KMeans(n_clusters=k, n_init=10, random_state=seed).fit(xy_m) for k in ks}
    if len(ks) == 1:
        return ks[0], fits[ks[0]].cluster_centers_

    inertia = np.array([fits[k].inertia_ for k in ks], dtype=np.float64)
    x = (np.array(ks, dtype=np.float64) - ks[0]) / (ks[-1] - ks[0])
    y = (inertia - inertia.min()) / (inertia.max() - inertia.min() + 1e-12)

    # Perpendicular distance of each point to the chord (x0,y0)→(x1,y1).
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    chord_len = np.hypot(x1 - x0, y1 - y0)
    dist = np.abs((x1 - x0) * (y0 - y) - (x0 - x) * (y1 - y0)) / (chord_len + 1e-12)

    k_star = ks[int(np.argmax(dist))]
    return k_star, fits[k_star].cluster_centers_


def static_uav_layout(
    client_coord_map: dict[int, tuple[float, float]],
    k_min: int,
    k_max: int,
    seed: int,
    R_comm: float,
) -> tuple[int, list[tuple[float, float]], dict[int, int]]:
    """Elbow K-means static placement: (K, UAV latlon list, covered clients).

    Deterministic in (client layout, seed): every mode/seed job for the same N
    reproduces the identical layout, which is what guarantees the selection
    rule is the only cross-mode difference.
    """
    latlon = np.array(list(client_coord_map.values()), dtype=np.float64)
    xy_m, ref = latlon_to_meters(latlon)
    K, centres = elbow_k(xy_m, k_min, k_max, seed=seed)

    uav_pos_m = np.column_stack([centres, np.full(K, _UAV_ALTITUDE_M)])
    uav_latlon = _uav_pos_to_latlon(uav_pos_m, np.array(ref))
    covered = _covered_clients(client_coord_map, uav_pos_m, np.array(ref), R_comm)
    return K, uav_latlon, covered


# Jain's index lives in metrics.fl so run_tier2/run_full_hfl report the
# same metric; the private alias keeps this module's historical import path.
_jain_index = jain_index


# ---------------------------------------------------------------------------
# Single run: all modes for one (N, seed)
# ---------------------------------------------------------------------------


def run_selection_isolation(cfg: dict) -> dict:
    """Run every selection mode on the identical static-UAV FL problem.

    Config schema mirrors ``paper_full.yaml`` with two additions:
      ``modes``  — selection rules to compare (default ``DEFAULT_MODES``)
      ``elbow``  — ``{k_min, k_max}`` bounds for the elbow K search; k_max is
                   additionally capped at N//5 so each UAV serves ≥5 devices
                   on average ("comparable UAVs" as N scales)

    Returns dict with ``"rounds"`` DataFrame, ``"results_dir"``, ``"size_mb"``,
    and ``"K_uav"`` (the elbow-selected UAV count).
    """
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    fl = cfg["fl"]
    n_rounds = fl["n_rounds"]
    n_local_epochs = fl["n_local_epochs"]
    lr = fl["lr"]
    batch_size = fl["batch_size"]
    R_comm = fl["R_comm"]
    capacity = fl["capacity"]
    T_sel = fl.get("T_sel", 5)
    lambda_min = fl.get("lambda_min", 0.5)
    R_min = fl.get("R_min", 0.3)
    target_accuracy = fl.get("target_accuracy", 0.70)
    run_seed = fl.get("seed", cfg.get("optimizer_seed", 42))
    n_uav_epochs = fl.get("n_uav_epochs", n_local_epochs)
    uav_lr = fl.get("uav_lr", lr)

    modes: list[str] = cfg.get("modes", list(DEFAULT_MODES))
    epicentre = tuple(cfg.get("epicentre", [37.488, 137.272]))

    # ── 1. Data (shared across modes) ───────────────────────────────────────
    full_dataset, client_train_indices, global_test_indices, client_coords, img_features = (
        _load_data(cfg, results_dir)
    )
    cached_dataset = CachedDataset(full_dataset, img_features)

    clients: list[ClientData] = [
        ClientData(
            client_id=cid,
            coords=client_coords[cid],
            train_indices=client_train_indices[cid],
            test_indices=[],
        )
        for cid in client_coords
        if client_train_indices.get(cid)
    ]
    client_ids = [c.client_id for c in clients]
    client_coord_map = {c.client_id: c.coords for c in clients}
    N_clients = len(clients)
    logger.info("%d clients loaded (selection isolation).", N_clients)

    # ── 2. Static placement (once, shared by every mode) ───────────────────
    elbow_cfg = cfg.get("elbow", {})
    k_min = elbow_cfg.get("k_min", 2)
    k_max = min(elbow_cfg.get("k_max", 30), max(k_min, N_clients // 5))
    # Placement depends only on the client layout — seed with the data seed so
    # every (mode, seed) job for this N reproduces the identical layout.
    placement_seed = cfg["data"].get("seed", 42)
    K, uav_latlon, covered_all = static_uav_layout(
        client_coord_map, k_min, k_max, placement_seed, R_comm
    )
    coverage_pct = 100.0 * len(covered_all) / max(N_clients, 1)
    logger.info(
        "Elbow K-means placement: K=%d UAVs (k range [%d, %d]) | coverage %.1f%%",
        K,
        k_min,
        k_max,
        coverage_pct,
    )

    # ── 3. Loaders built once (shard indices never change) ─────────────────
    client_loaders: dict[int, BalancedShardLoader] = {
        c.client_id: make_client_loader(cached_dataset, c.train_indices, batch_size)
        for c in clients
    }
    client_by_id = {c.client_id: c for c in clients}

    all_rows: list[dict] = []
    confusion_rows: list[dict] = []

    # ── 4. Per-mode runs — identical seed, identical everything but the rule ─
    for mode in modes:
        logger.info("=== Selection mode: %s ===", mode)

        rng = np.random.default_rng(run_seed)
        torch.manual_seed(run_seed)

        global_model = CachedFusionModel()
        global_model.freeze_img_proj()

        device_mgr = DeviceStateManager(
            client_ids,
            rng,
            dropout_rate=fl.get("dropout_rate", 0.0),
            snr_degradation_db=fl.get("snr_degradation_db", 0.0),
        )
        rep_mgr = ReputationManager(client_ids)
        selector = ClientSelector(client_ids, epicentre=epicentre)

        selected: dict[int, int] = {}
        uav_loaders: dict[int, BalancedShardLoader] = {}
        sel_counts = {cid: 0 for cid in client_ids}
        rounds_to_target: int | None = None
        target_streak = 0
        method_start_idx = len(all_rows)

        for rnd in range(1, n_rounds + 1):
            t0 = time.perf_counter()

            device_states = device_mgr.get_all_states()
            rep_scores = rep_mgr.get_all_scores()

            # Early-reselection trigger (§IV-E6) — same rule as run_full_hfl.
            n_eligible = sum(1 for st in device_states.values() if st.eligible())
            low_eligible = n_eligible < lambda_min * min(K * capacity, N_clients)
            reselect = (rnd - 1) % T_sel == 0 or low_eligible or not selected

            # ── Client selection (the experimental variable) ─────────────
            if reselect:
                selected = selector.select(
                    covered=covered_all,
                    device_states=device_states,
                    reputation_scores=rep_scores,
                    client_coords=client_coord_map,
                    uav_coords_latlon=uav_latlon,
                    round_num=rnd,
                    uav_capacity=capacity,
                    mode=mode,
                    rng=rng,
                    R_comm=R_comm,
                    t_stale_cap=T_sel,
                )
                # Roster changed → rebuild the UAV pooled-shard loaders.
                uav_indices: dict[int, list[int]] = {}
                for cid, uav_idx in selected.items():
                    uav_indices.setdefault(uav_idx, []).extend(client_by_id[cid].train_indices)
                uav_loaders = {
                    uav_idx: make_client_loader(cached_dataset, idxs, batch_size)
                    for uav_idx, idxs in uav_indices.items()
                    if idxs
                }

            n_selected = len(selected)
            participation_pct = 100.0 * n_selected / max(N_clients, 1)

            # ── UAV image training (paper §IV-A Step 3) ──────────────────
            uav_img_updates: dict[int, tuple[dict, int]] = {}
            for uav_idx, loader in uav_loaders.items():
                sd, n = _uav_local_train(global_model, loader, n_uav_epochs, uav_lr)
                uav_img_updates[uav_idx] = (sd, n)

            # ── IoT local training (paper §IV-A Step 5) ──────────────────
            global_trainable = global_model.trainable_state_dict()
            client_updates: dict[int, tuple[dict, int, float]] = {}
            client_deltas: dict[int, dict] = {}
            client_losses: dict[int, float] = {}
            for cid in selected:
                loader = client_loaders.get(cid)
                if loader is None:
                    continue
                sd, n, mean_loss = _local_train(global_model, loader, n_local_epochs, lr)
                rep = rep_scores.get(cid, 0.5)
                client_updates[cid] = (sd, n, rep)
                client_losses[cid] = mean_loss
                client_deltas[cid] = {k: v - global_trainable[k] for k, v in sd.items()}
            # Statistical-utility feed for the oort / power_of_choice modes.
            selector.update_losses(client_losses)

            for cid in selected:
                if cid not in client_updates:
                    rep_mgr.mark_absent(cid)

            # ── Reputation update ────────────────────────────────────────
            if client_deltas:
                rep_mgr.update_batch(
                    client_deltas,
                    global_update_vec=None,
                    response_times={
                        cid: device_states[cid].compute_time_s
                        for cid in client_deltas
                        if cid in device_states
                    },
                )

            # ── UAV-level aggregation (paper §IV-A Step 6) ───────────────
            iot_by_uav: dict[int, list[tuple[dict, int, float]]] = {}
            for cid, triple in client_updates.items():
                iot_by_uav.setdefault(selected[cid], []).append(triple)

            uav_updates: list[tuple[dict, int, float]] = []
            for uav_idx in range(K):
                iot_upds = iot_by_uav.get(uav_idx, [])
                uav_img = uav_img_updates.get(uav_idx)

                if not iot_upds and uav_img is None:
                    continue

                if iot_upds:
                    sf_agg = fedavg([(sd, n) for sd, n, _ in iot_upds])
                    total_n = sum(n for _, n, _ in iot_upds)
                else:
                    sf_agg = global_model.trainable_state_dict()
                    total_n = uav_img[1]

                if uav_img is not None:
                    img_part = {k: v for k, v in uav_img[0].items() if k.startswith("img_proj.")}
                else:
                    img_part = {
                        f"img_proj.{k}": v.clone()
                        for k, v in global_model.img_proj.state_dict().items()
                    }
                full_agg = {**img_part, **sf_agg}

                cluster_reps = [
                    rep_scores.get(cid, 0.5) for cid, u in selected.items() if u == uav_idx
                ]
                uav_rep = trimmed_mean(cluster_reps) if cluster_reps else 1.0
                uav_updates.append((full_agg, total_n, uav_rep))

            # ── Server aggregation: reputation FedAvg, R_min gate (§IV-D) ─
            # Fixed for every mode — aggregation is not the variable here.
            if uav_updates:
                active = [u for u in uav_updates if u[2] >= R_min]
                if active:
                    server_agg = reputation_fedavg(active)
                    global_model.load_full_trainable_state_dict(server_agg)

            # ── Device state + fairness bookkeeping ──────────────────────
            device_mgr.update_round(set(selected.keys()))
            for cid in selected:
                sel_counts[cid] += 1

            # ── Evaluate ─────────────────────────────────────────────────
            # evaluate_subset handles empty test indices (zero metrics) and
            # takes the tensor-sliced fast path on CachedDataset.
            metrics = _evaluate_subset(global_model, cached_dataset, global_test_indices)
            elapsed = time.perf_counter() - t0

            if metrics["accuracy"] >= target_accuracy:
                target_streak += 1
                if rounds_to_target is None and target_streak >= _TARGET_CONSEC_ROUNDS:
                    rounds_to_target = rnd - _TARGET_CONSEC_ROUNDS + 1
            else:
                target_streak = 0

            counts_arr = np.fromiter(sel_counts.values(), dtype=np.float64)
            comm_mb = round_comm_mb(n_selected, n_active_uavs=len(uav_img_updates))

            all_rows.append(
                {
                    "method": mode,
                    "round": rnd,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "coverage_pct": coverage_pct,
                    "participation_pct": participation_pct,
                    "n_selected": n_selected,
                    "n_eligible": n_eligible,
                    "K_uav": K,
                    "jain_fairness": _jain_index(counts_arr),
                    "n_unique_selected": int((counts_arr > 0).sum()),
                    "mean_battery": float(np.mean([st.battery for st in device_states.values()])),
                    "comm_mb_round": comm_mb,
                    "round_time_s": elapsed,
                    **{f"f1_{cls}": v for cls, v in metrics["f1_per_class"].items()},
                }
            )
            confusion_rows.extend(_confusion_rows(mode, rnd, metrics["confusion_matrix"]))
            if rnd % 10 == 0 or rnd == n_rounds:
                logger.info(
                    "[%s] round %d/%d | acc=%.3f | F1=%.3f | sel=%d | jain=%.3f | %.1fs",
                    mode,
                    rnd,
                    n_rounds,
                    metrics["accuracy"],
                    metrics["macro_f1"],
                    n_selected,
                    all_rows[-1]["jain_fairness"],
                    elapsed,
                )

        for row in all_rows[method_start_idx:]:
            row["rounds_to_target"] = rounds_to_target

        logger.info(
            "%s done. Rounds to %.0f%%: %s",
            mode,
            target_accuracy * 100,
            rounds_to_target if rounds_to_target else "not reached",
        )

    # ── 5. Persist ───────────────────────────────────────────────────────────
    rounds_df = pd.DataFrame(all_rows)
    _write_table(rounds_df, results_dir / "selection_rounds.parquet")
    _write_table(pd.DataFrame(confusion_rows), results_dir / "confusion.parquet")

    _dump_resolved_cfg(cfg, results_dir / "config.selection.resolved.yaml")

    size_mb = sum(p.stat().st_size for p in results_dir.rglob("*") if p.is_file()) / 1e6
    return {"rounds": rounds_df, "results_dir": results_dir, "size_mb": size_mb, "K_uav": K}


# ---------------------------------------------------------------------------
# Parallel sweep: (N × mode × seed)
# ---------------------------------------------------------------------------


def _selection_job(N: int, mode: str, seed_idx: int, cfg: dict) -> pd.DataFrame:
    """One (N, mode, seed) run inside a joblib worker.

    Resumable: see ``sweep._job`` — ``run_selection_isolation`` writes
    ``config.selection.resolved.yaml`` last, so its presence gates the skip.
    """
    job_cfg = copy.deepcopy(cfg)
    job_cfg["data"]["N_clients"] = N
    # Partition varies per seed repetition, mode-free (see seeds.partition_seed_for).
    job_cfg["data"]["partition_seed"] = partition_seed_for(seed_idx)
    job_cfg["modes"] = [mode]
    # Deliberately NO mode hash in the seed: every mode for a given (N, seed)
    # must share model init, device heterogeneity, and the static UAV layout,
    # so the selection rule is the only cross-mode difference. (Contrast with
    # sweep._paper_job, where run_full_hfl folds in a method hash.)
    job_cfg["fl"]["seed"] = sweep_job_seed(cfg.get("optimizer_seed", 9876), seed_idx, N)
    job_results_dir = Path(cfg["results_dir"]) / f"N{N}" / f"seed{seed_idx}" / mode
    job_cfg["results_dir"] = str(job_results_dir)

    if (job_results_dir / "config.selection.resolved.yaml").exists():
        logger.info(
            "[N=%d  mode=%-9s seed=%d] checkpoint found — skipping (resume)", N, mode, seed_idx
        )
        df = read_table(job_results_dir / "selection_rounds.parquet")
        df.insert(0, "seed", seed_idx)
        df.insert(0, "N", N)
        return df

    import torch as _torch

    _torch.set_num_threads(1)  # 1 thread × n_workers = full CPU budget, no BLAS thrash

    if job_cfg["data"].get("source", "real") == "real":
        job_cfg["data"]["feature_cache_path"] = str(
            Path(cfg["results_dir"]) / f"N{N}" / "img_features.npy"
        )

    logger.info("[N=%d  mode=%-9s seed=%d] starting", N, mode, seed_idx)
    out = run_selection_isolation(job_cfg)
    df = out["rounds"].copy()
    df.insert(0, "seed", seed_idx)
    df.insert(0, "N", N)
    final_acc = float(df["accuracy"].iloc[-1]) if len(df) else float("nan")
    logger.info("[N=%d  mode=%-9s seed=%d] done | acc=%.3f", N, mode, seed_idx, final_acc)
    return df


def run_selection_sweep(cfg: dict) -> dict:
    """Full selection-isolation grid: N × mode × seed, joblib-parallel.

    Phase 1 (sequential): pre-fetch dataset + feature caches for all N values
    (zero HuggingFace calls during parallel execution).
    Phase 2 (parallel):   one worker per (N, mode, seed) job.
    """
    from .sweep import _prefetch_all_N

    N_values: list[int] = cfg["N_values"]
    modes: list[str] = cfg.get("modes", list(DEFAULT_MODES))
    n_seeds: int = cfg.get("n_seeds", 1)
    n_workers: int = cfg.get("n_workers", 12)
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 1: pre-fetching data for %d N-values (sequential)…", len(N_values))
    _prefetch_all_N(cfg)
    logger.info("Phase 1 complete — all caches ready.")

    jobs = [(N, mode, seed_idx) for N in N_values for mode in modes for seed_idx in range(n_seeds)]
    logger.info(
        "Phase 2: %d N × %d modes × %d seeds = %d jobs — %d parallel workers",
        len(N_values),
        len(modes),
        n_seeds,
        len(jobs),
        n_workers,
    )

    dfs = Parallel(n_jobs=n_workers, backend="loky", verbose=5)(
        delayed(_selection_job)(N, mode, seed_idx, cfg) for N, mode, seed_idx in jobs
    )

    full_df = pd.concat(dfs, ignore_index=True)
    _write_table(full_df, results_dir / "selection_sweep_rounds.parquet")

    _dump_resolved_cfg(cfg, results_dir / "config.selection_sweep.resolved.yaml")

    size_mb = sum(p.stat().st_size for p in results_dir.rglob("*") if p.is_file()) / 1e6
    logger.info("Selection sweep complete — %.2f MB at %s", size_mb, results_dir)

    return {"rounds": full_df, "results_dir": results_dir, "size_mb": size_mb}
