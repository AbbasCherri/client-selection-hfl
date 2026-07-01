"""Tier-2 FL harness: placement → covered clients → FedAvg → metrics.

For each placement method (e.g. PSO, GA, centroid, static) the harness runs
``n_rounds`` of hierarchical federated learning on the post-earthquake damage
dataset. Each round:

1. **Placement.** Run the placement optimizer on the client geographic
   coordinates to find K UAV hover positions.
2. **Coverage.** Determine which clients are within ``R_comm`` of any UAV.
   Only covered clients participate this round.
3. **Local training.** Each covered client trains a ``CachedFusionModel`` for
   ``n_local_epochs`` on its shard, using the precomputed ResNet-18 features
   instead of running the image backbone (CPU feasibility).
4. **UAV-level FedAvg.** Within each UAV's coverage zone, aggregate client
   updates by sample count.
5. **Server-level FedAvg.** Aggregate UAV updates at the central server.
6. **Evaluation.** Compute global accuracy, per-class F1, and macro-F1 on the
   pooled test set.

Metrics are written to ``results_dir/tier2/`` as Parquet/CSV and are
regenerable from saved files via the ``analyze`` CLI command.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

from hflsim.shared.coords import haversine, latlon_to_meters
from uavbench.optimizers import REGISTRY
from uavbench.problem.energy import EnergyModel
from uavbench.problem.fitness import Fitness
from uavbench.problem.instance import ProblemInstance

_ENERGY_MODEL = EnergyModel()

from .client_selection import ClientSelector
from .dataset import CachedDataset, ClientData, SyntheticClientData, make_client_loader
from .device_state import DeviceStateManager
from .features import compute_feature_cache, synthetic_feature_cache
from .model import (
    CachedFusionModel, clone_model,
    fedavg, reputation_fedavg,
    mixed_fedavg, mixed_reputation_fedavg,
)
from .reputation import ReputationManager

logger = logging.getLogger("uavbench.fl.federated")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_problem_instance(
    client_coords: dict[int, tuple[float, float]],
    K: int,
    R_comm: float,
    capacity: int,
    prev_positions_m: np.ndarray | None,
) -> ProblemInstance:
    """Construct a ProblemInstance from client geographic coordinates."""
    latlon = np.array(list(client_coords.values()), dtype=np.float64)
    xy_m, ref = latlon_to_meters(latlon)
    N = len(xy_m)
    device_coords = np.column_stack([xy_m, np.zeros(N)])

    lower = np.array([xy_m[:, 0].min(), xy_m[:, 1].min(), 20.0])
    upper = np.array([xy_m[:, 0].max(), xy_m[:, 1].max(), 120.0])

    if prev_positions_m is None:
        # Spread initial positions evenly across the bounding box.
        xs = np.linspace(lower[0], upper[0], K)
        ys = np.linspace(lower[1], upper[1], K)
        prev_positions_m = np.column_stack([xs, ys, np.full(K, 70.0)])

    return ProblemInstance(
        device_coords=device_coords,
        value=np.ones(N),   # uniform — coverage drives FL quality, not value weighting
        capacity=np.full(K, float(capacity)),
        battery=np.ones(K),
        prev_positions=prev_positions_m,
        lower=lower,
        upper=upper,
        R_comm=R_comm,
        B_min_uav=0.0,      # battery is always 1 in this bridge
    ), ref


def _place_uavs(
    client_coords: dict[int, tuple[float, float]],
    K: int,
    R_comm: float,
    capacity: int,
    method: str,
    rng: np.random.Generator,
    P: int,
    G_max: int,
    prev_positions_m: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run a placement optimizer and return UAV positions in metres.

    Returns
    -------
    uav_positions_m : (K, 3) metres in the projected frame
    ref             : (lat0, lon0) reference for back-projection
    best_fitness    : placement fitness score
    """
    instance, ref = _build_problem_instance(
        client_coords, K, R_comm, capacity, prev_positions_m
    )
    fitness = Fitness(instance)

    cls = REGISTRY[method]
    optimizer = cls(P=P, G_max=G_max) if method in ("pso", "ga") else cls()
    result = optimizer.optimize(instance, fitness, rng)

    uav_pos = result.best_position.reshape(K, 3)
    return uav_pos, np.array(ref), result.best_fitness


def _covered_clients(
    client_coords: dict[int, tuple[float, float]],
    uav_pos_m: np.ndarray,
    ref: np.ndarray,
    R_comm: float,
) -> dict[int, int]:
    """Return {client_id: assigned_uav_idx} for clients within R_comm of any UAV.

    Converts UAV metre positions back to (lat, lon) for Haversine range check.
    Assigns each covered client to its nearest UAV.
    """
    lat0, lon0 = float(ref[0]), float(ref[1])
    lat0_rad = math.radians(lat0)
    R = 6_371_000.0

    uav_latlon: list[tuple[float, float]] = []
    for x, y, _z in uav_pos_m:
        lat = lat0 + math.degrees(y / R)
        lon = lon0 + math.degrees(x / (R * math.cos(lat0_rad)))
        uav_latlon.append((lat, lon))

    assignment: dict[int, int] = {}
    for cid, coord in client_coords.items():
        dists = [haversine(coord, ul) for ul in uav_latlon]
        nearest = int(np.argmin(dists))
        if dists[nearest] <= R_comm:
            assignment[cid] = nearest
    return assignment


def _local_train(
    model: CachedFusionModel,
    loader: DataLoader,
    n_epochs: int,
    lr: float,
) -> tuple[dict, int]:
    """Train a client-local copy of the model; return (trainable_state_dict, n_samples).

    img_proj is frozen on the clone (inherited from global_model which has
    freeze_img_proj() called after construction).  Only struct_branch + fusion
    are updated — the IoT-level payload per paper §IV-B.
    """
    local = clone_model(model)
    local.train()
    trainable = [p for p in local.parameters() if p.requires_grad]
    opt = optim.Adam(trainable, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n_seen = 0
    for _ in range(n_epochs):
        for img_feat, struct, labels in loader:
            opt.zero_grad()
            logits = local(img_feat, struct)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            n_seen += labels.shape[0]

    return local.trainable_state_dict(), n_seen // max(n_epochs, 1)


def _uav_local_train(
    model: CachedFusionModel,
    loader: DataLoader,
    n_epochs: int,
    lr: float,
) -> tuple[dict, int]:
    """Train a UAV-local copy with img_proj unfrozen (full model, paper §IV-A Step 3).

    Uses the same cached 512-dim ResNet features as IoT clients — no raw image
    loading or backbone forward pass required.  img_proj learns to map ImageNet
    features to damage-relevant representations; IoT devices cannot do this.
    Returns (full_trainable_state_dict, n_samples).
    """
    local = clone_model(model)
    local.unfreeze_img_proj()
    local.train()
    trainable = [p for p in local.parameters() if p.requires_grad]
    opt = optim.Adam(trainable, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n_seen = 0
    for _ in range(n_epochs):
        for img_feat, struct, labels in loader:
            opt.zero_grad()
            logits = local(img_feat, struct)
            loss_fn(logits, labels).backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            n_seen += labels.shape[0]

    return local.full_trainable_state_dict(), n_seen // max(n_epochs, 1)


def _evaluate(
    model: CachedFusionModel,
    dataset: CachedDataset,
    indices: list[int],
    batch_size: int = 64,
) -> dict:
    """Compute global accuracy, per-class F1, and macro-F1 on the test set."""
    if not indices:
        return {"accuracy": 0.0, "macro_f1": 0.0, "f1_per_class": {}}

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for img_feat, struct, labels in loader:
            preds = model(img_feat, struct).argmax(dim=1)
            all_preds.append(preds.numpy())
            all_labels.append(labels.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0, labels=[0, 1, 2, 3]))
    per_class = f1_score(labels, preds, average=None, zero_division=0, labels=[0, 1, 2, 3])
    class_names = ["survived", "collapsed", "obstructed", "missing"]
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "f1_per_class": dict(zip(class_names, per_class.tolist())),
    }


def _write_table(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_csv(path.with_suffix(".csv"), index=False)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_tier2(cfg: dict) -> dict:
    """Run the Tier-2 FL benchmark for all configured placement methods.

    The config schema mirrors ``tier2_reduced.yaml`` / ``tier2_fl.yaml``.

    Returns
    -------
    dict with keys:
        ``"rounds"``      — per-round metrics DataFrame (method × round)
        ``"results_dir"`` — where the Parquets were written
        ``"size_mb"``     — disk footprint of results
    """
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    n_rounds: int = cfg["fl"]["n_rounds"]
    n_local_epochs: int = cfg["fl"]["n_local_epochs"]
    lr: float = cfg["fl"]["lr"]
    batch_size: int = cfg["fl"]["batch_size"]
    K: int = cfg["fl"]["K"]
    R_comm: float = cfg["fl"]["R_comm"]
    capacity: int = cfg["fl"]["capacity"]
    P: int = cfg["budget"]["P"]
    G_max: int = cfg["budget"]["G_max"]
    methods: list[str] = cfg["methods"]
    data_cfg: dict = cfg["data"]
    target_accuracy: float = cfg["fl"].get("target_accuracy", 0.70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    data_source = data_cfg.get("source", "synthetic")

    if data_source == "synthetic":
        logger.info(
            "Using SYNTHETIC data (N=%d, K=%d) — no HF token required.",
            data_cfg["N_clients"], K,
        )
        synth = SyntheticClientData(N=data_cfg["N_clients"], K=K, seed=data_cfg.get("seed", 42))
        raw = synth.build()

        full_dataset = raw["full_dataset"]
        client_train_indices: dict[int, list[int]] = raw["client_train_indices"]
        global_test_indices: list[int] = raw["global_test_indices"]
        client_coords: dict[int, tuple[float, float]] = raw["client_coords"]
        img_features: np.ndarray = raw["img_features"]

    else:
        logger.info("Loading real HFL dataset (N=%d clients)…", data_cfg["N_clients"])
        import os
        from hflsim.data import get_hfl_data_partitions

        hf_token = os.environ.get("HF_TOKEN", data_cfg.get("hf_token"))
        full_dataset, client_train_indices, _, global_test_indices, client_coords = (
            get_hfl_data_partitions(
                csv_path=data_cfg.get("csv_path"),
                data_dir=data_cfg.get("data_dir", "./data"),
                N=data_cfg["N_clients"],
                subsample=data_cfg.get("subsample", 0.05),
                random_seed=data_cfg.get("seed", 42),
                hf_token=hf_token,
            )
        )
        cache_path = str(results_dir / "img_features.npy")
        img_features = compute_feature_cache(
            full_dataset,
            cache_path=cache_path,
            batch_size=data_cfg.get("feature_batch_size", 32),
            num_workers=0,
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
    logger.info("%d clients loaded.", len(clients))

    all_rows: list[dict] = []

    # ------------------------------------------------------------------
    # 2. Outer loop: one full FL run per placement method
    # ------------------------------------------------------------------
    for method in methods:
        logger.info("=== Method: %s ===", method)
        N_clients = len(clients)
        _method_hash = int(hashlib.md5(method.encode()).hexdigest(), 16) % (2**31)
        _seed = (cfg.get("optimizer_seed", 9876) + N_clients * 7919 + _method_hash) % (2**31)
        rng = np.random.default_rng(_seed)
        torch.manual_seed(_seed)   # deterministic model init across runs

        global_model = CachedFusionModel()
        prev_uav_positions_m: np.ndarray | None = None
        rounds_to_target: int | None = None
        cumulative_energy_j: float = 0.0

        method_start_idx = len(all_rows)

        for rnd in range(1, n_rounds + 1):
            t0 = time.perf_counter()

            # ---- Placement + Coverage ----
            if method == "no_uav":
                # Baseline: every client participates every round, no UAV filter.
                # Models the upper-bound FL scenario — full participation, zero movement cost.
                covered = {c.client_id: 0 for c in clients}
                placement_fitness = 1.0
            else:
                uav_pos_m, ref, placement_fitness = _place_uavs(
                    client_coords={c.client_id: c.coords for c in clients},
                    K=K,
                    R_comm=R_comm,
                    capacity=capacity,
                    method=method,
                    rng=rng,
                    P=P,
                    G_max=G_max,
                    prev_positions_m=prev_uav_positions_m,
                )
                if prev_uav_positions_m is not None:
                    move_m = float(
                        np.sum(np.sqrt(np.sum((uav_pos_m - prev_uav_positions_m) ** 2, axis=1)))
                    )
                    cumulative_energy_j += _ENERGY_MODEL.energy_joules(move_m)
                prev_uav_positions_m = uav_pos_m.copy()
                covered = _covered_clients(
                    {c.client_id: c.coords for c in clients}, uav_pos_m, ref, R_comm
                )

            coverage_pct = 100.0 * len(covered) / max(len(clients), 1)

            # ---- Per-UAV grouping ----
            uav_groups: dict[int, list[ClientData]] = {j: [] for j in range(K)}
            for c in clients:
                if c.client_id in covered:
                    uav_groups[covered[c.client_id]].append(c)

            # ---- Local training ----
            client_updates: dict[int, list[tuple[dict, int]]] = {j: [] for j in range(K)}
            for uav_idx, group in uav_groups.items():
                for client in group:
                    if not client.train_indices:
                        continue
                    loader = make_client_loader(cached_dataset, client.train_indices, batch_size)
                    sd, n = _local_train(global_model, loader, n_local_epochs, lr)
                    client_updates[uav_idx].append((sd, n))

            # ---- UAV-level FedAvg ----
            uav_updates: list[tuple[dict, int]] = []
            for uav_idx in range(K):
                upds = client_updates[uav_idx]
                if upds:
                    agg = fedavg(upds)
                    total_n = sum(n for _, n in upds)
                    uav_updates.append((agg, total_n))

            # ---- Server-level FedAvg ----
            if uav_updates:
                server_agg = fedavg(uav_updates)
                global_model.load_trainable_state_dict(server_agg)

            # ---- Evaluate ----
            metrics = _evaluate(global_model, cached_dataset, global_test_indices)
            elapsed = time.perf_counter() - t0

            n_covered = len(covered)
            # Uplink + downlink, no UAV→server hop (Tier-2 flat placement harness).
            comm_mb_round = 2.0 * n_covered * _MODEL_SIZE_MB

            if rounds_to_target is None and metrics["accuracy"] >= target_accuracy:
                rounds_to_target = rnd

            row = {
                "method": method,
                "round": rnd,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "coverage_pct": coverage_pct,
                "n_covered": n_covered,
                "placement_fitness": placement_fitness,
                "comm_mb_round": comm_mb_round,
                "cumulative_energy_j": cumulative_energy_j,
                "round_time_s": elapsed,
                **{f"f1_{cls}": v for cls, v in metrics["f1_per_class"].items()},
            }
            all_rows.append(row)
            logger.info(
                "Round %d/%d | acc=%.3f | macro-F1=%.3f | covered=%d/%.0f%% | %.1fs",
                rnd, n_rounds, metrics["accuracy"], metrics["macro_f1"],
                n_covered, coverage_pct, elapsed,
            )

        # Backfill per-method scalar onto all rows for this method.
        for row in all_rows[method_start_idx:]:
            row["rounds_to_target"] = rounds_to_target

        logger.info(
            "%s finished. Rounds to target (%.0f%%): %s | Cumulative energy: %.1f kJ",
            method, target_accuracy * 100,
            rounds_to_target if rounds_to_target else "not reached",
            cumulative_energy_j / 1000,
        )

    # ------------------------------------------------------------------
    # 3. Persist results
    # ------------------------------------------------------------------
    rounds_df = pd.DataFrame(all_rows)
    _write_table(rounds_df, results_dir / "tier2_rounds.parquet")

    with open(results_dir / "config.tier2.resolved.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    size_mb = sum(p.stat().st_size for p in results_dir.rglob("*") if p.is_file()) / 1e6
    logger.info("Tier-2 results at %s (%.2f MB)", results_dir, size_mb)

    return {"rounds": rounds_df, "results_dir": results_dir, "size_mb": size_mb}


# ---------------------------------------------------------------------------
# Full paper system simulation
# ---------------------------------------------------------------------------

# IoT payload:  struct_branch (17,216) + fusion (50,436)         = 67,652 params ≈ 0.271 MB
# UAV payload:  img_proj (65,664) + struct_branch + fusion       = 133,316 params ≈ 0.533 MB
_IOT_MODEL_SIZE_MB: float = 67_652  * 4 / 1_000_000
_UAV_MODEL_SIZE_MB: float = 133_316 * 4 / 1_000_000
_MODEL_SIZE_MB: float = _IOT_MODEL_SIZE_MB   # kept for run_tier2 back-compat


# Method configuration: (placement_method, selection_mode, reputation_weighted, dynamic)
# placement_method: "ga" or None (flat/centralized)
# selection_mode:   "ucb" | "random" | "all"
# reputation_weighted: True → reputation_fedavg; False → uniform sample-weight fedavg
# dynamic:          True → reposition every T_sel rounds; False → place once at round 1
_METHOD_CFG: dict[str, tuple] = {
    "proposed_hfl":      ("ga",  "ucb",    True,  True),
    "flat_fl":           (None,  "all",    False, False),
    "centralized":       (None,  "all",    False, False),   # handled specially
    "hfl_no_selection":  ("ga",  "random", True,  True),
    "hfl_static":        ("ga",  "ucb",    True,  False),
    "hfl_no_reputation": ("ga",  "ucb",    False, True),
}


def _uav_pos_to_latlon(
    uav_pos_m: np.ndarray,
    ref: np.ndarray,
) -> list[tuple[float, float]]:
    """Convert UAV metre positions back to (lat, lon) tuples."""
    lat0, lon0 = float(ref[0]), float(ref[1])
    lat0_rad = math.radians(lat0)
    R = 6_371_000.0
    latlon = []
    for x, y, _z in uav_pos_m:
        lat = lat0 + math.degrees(y / R)
        lon = lon0 + math.degrees(x / (R * math.cos(lat0_rad)))
        latlon.append((lat, lon))
    return latlon


def _load_data(cfg: dict, results_dir: Path) -> tuple:
    """Shared data loading for run_full_hfl (real or synthetic)."""
    data_cfg = cfg["data"]
    data_source = data_cfg.get("source", "synthetic")

    if data_source == "synthetic":
        K = cfg["fl"]["K"]
        synth = SyntheticClientData(N=data_cfg["N_clients"], K=K, seed=data_cfg.get("seed", 42))
        raw = synth.build()
        return (
            raw["full_dataset"],
            raw["client_train_indices"],
            raw["global_test_indices"],
            raw["client_coords"],
            raw["img_features"],
        )

    import os
    from hflsim.data import get_hfl_data_partitions

    hf_token = os.environ.get("HF_TOKEN", data_cfg.get("hf_token"))
    full_dataset, client_train_indices, _, global_test_indices, client_coords = (
        get_hfl_data_partitions(
            csv_path=data_cfg.get("csv_path"),
            data_dir=data_cfg.get("data_dir", "./data"),
            N=data_cfg["N_clients"],
            subsample=data_cfg.get("subsample", 0.05),
            random_seed=data_cfg.get("seed", 42),
            hf_token=hf_token,
        )
    )
    # Allow the sweep to provide a shared N-level cache (avoids recomputing per seed).
    cache_path = data_cfg.get("feature_cache_path") or str(results_dir / "img_features.npy")
    img_features = compute_feature_cache(
        full_dataset,
        cache_path=cache_path,
        batch_size=data_cfg.get("feature_batch_size", 32),
        num_workers=0,
    )
    return full_dataset, client_train_indices, global_test_indices, client_coords, img_features


def _run_centralized(
    global_model: CachedFusionModel,
    cached_dataset: CachedDataset,
    all_train_indices: list[int],
    global_test_indices: list[int],
    n_rounds: int,
    n_local_epochs: int,
    lr: float,
    batch_size: int,
) -> list[dict]:
    """Oracle: train on all data at one node, report metrics every n_local_epochs epochs."""
    rows: list[dict] = []
    loader = make_client_loader(cached_dataset, all_train_indices, batch_size)
    # Centralized has full compute — train the entire model including img_proj.
    global_model.unfreeze_img_proj()
    trainable = [p for p in global_model.parameters() if p.requires_grad]
    opt = optim.Adam(trainable, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for rnd in range(1, n_rounds + 1):
        t0 = time.perf_counter()
        global_model.train()
        for _ in range(n_local_epochs):
            for img_feat, struct, labels in loader:
                opt.zero_grad()
                logits = global_model(img_feat, struct)
                loss_fn(logits, labels).backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
        metrics = _evaluate(global_model, cached_dataset, global_test_indices)
        rows.append({
            "method": "centralized",
            "round": rnd,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "coverage_pct": 100.0,
            "n_selected": len(all_train_indices),
            "placement_fitness": 1.0,
            "comm_mb_round": 0.0,          # no communication
            "cumulative_energy_j": 0.0,
            "round_time_s": time.perf_counter() - t0,
            **{f"f1_{cls}": v for cls, v in metrics["f1_per_class"].items()},
        })
        logger.info(
            "Centralized round %d/%d | acc=%.3f | macro-F1=%.3f",
            rnd, n_rounds, metrics["accuracy"], metrics["macro_f1"],
        )
    return rows


def run_full_hfl(cfg: dict) -> dict:
    """Full paper system simulation — all methods from §V including ablations.

    Supported methods (``cfg["methods"]``):
      proposed_hfl      — GA placement every T_sel rounds + UCB selection + reputation FedAvg
      flat_fl           — no UAV hierarchy, all clients, server aggregates directly
      centralized       — oracle upper bound (all data at one node, no federation)
      hfl_no_selection  — GA every T_sel rounds + random selection + reputation FedAvg
      hfl_static        — GA once (no repositioning) + UCB selection + reputation FedAvg
      hfl_no_reputation — GA every T_sel rounds + UCB selection + uniform FedAvg

    Additional config keys vs run_tier2
    ------------------------------------
    fl.T_sel            : int   — reposition interval in rounds (default 5)
    fl.seed             : int   — per-run RNG seed (for multi-seed sweeps)
    client_simulation   : dict  — optional; enabled by default
    """
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    fl = cfg["fl"]
    n_rounds        = fl["n_rounds"]
    n_local_epochs  = fl["n_local_epochs"]
    lr              = fl["lr"]
    batch_size      = fl["batch_size"]
    K               = fl["K"]
    R_comm          = fl["R_comm"]
    capacity        = fl["capacity"]
    T_sel           = fl.get("T_sel", 5)
    target_accuracy = fl.get("target_accuracy", 0.70)
    run_seed        = fl.get("seed", cfg.get("optimizer_seed", 42))
    n_uav_epochs    = fl.get("n_uav_epochs", n_local_epochs)
    uav_lr          = fl.get("uav_lr", lr)

    P     = cfg["budget"]["P"]
    G_max = cfg["budget"]["G_max"]

    # ── 1. Load data ────────────────────────────────────────────────────────
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
    all_train_indices: list[int] = [idx for c in clients for idx in c.train_indices]
    logger.info("%d clients loaded (full system).", len(clients))

    # Epicentre — use config override or default to Noto Peninsula 2024
    epicentre = tuple(cfg.get("epicentre", [37.488, 137.272]))   # type: ignore[assignment]

    all_rows: list[dict] = []
    models_by_method: dict[str, "CachedFusionModel"] = {}

    # ── 2. Per-method outer loop ─────────────────────────────────────────────
    for method in cfg["methods"]:
        logger.info("=== Full-system method: %s ===", method)

        if method not in _METHOD_CFG:
            logger.warning("Unknown method %s — skipping.", method)
            continue

        placement_method, selection_mode, rep_weighted, dynamic = _METHOD_CFG[method]

        # Per-method seed: combine the run-level seed with a stable method hash.
        # The hash is applied exactly once here; callers (e.g. _paper_job) must NOT
        # pre-encode the method identity into run_seed to avoid double-counting.
        _method_hash = int(hashlib.md5(method.encode()).hexdigest(), 16) % (2**16)
        _seed = (run_seed ^ _method_hash) % (2**31)
        rng = np.random.default_rng(_seed)
        torch.manual_seed(_seed)   # deterministic model init across runs

        global_model = CachedFusionModel()
        # img_proj frozen for IoT clients; UAV training unfreezes on its own clone.
        global_model.freeze_img_proj()

        # ── Centralized baseline: no federation at all ───────────────────
        if method == "centralized":
            rows = _run_centralized(
                global_model, cached_dataset, all_train_indices, global_test_indices,
                n_rounds, n_local_epochs, lr, batch_size,
            )
            all_rows.extend(rows)
            models_by_method[method] = global_model
            continue

        # ── Federated path ───────────────────────────────────────────────
        device_mgr  = DeviceStateManager(client_ids, rng)
        rep_mgr     = ReputationManager(client_ids)
        selector    = ClientSelector(client_ids, epicentre=epicentre)

        # Precompute static client-coord lookup (avoid rebuilding each round)
        client_coord_map: dict[int, tuple[float, float]] = {
            c.client_id: c.coords for c in clients
        }

        prev_uav_pos_m:       np.ndarray | None = None
        uav_pos_m:            np.ndarray | None = None
        uav_latlon:           list[tuple[float, float]] = []
        ref:                  np.ndarray | None = None
        covered_all:          dict[int, int] = {}
        last_placement_fitness: float = 0.0
        cumulative_energy     = 0.0
        rounds_to_target:     int | None = None
        method_start_idx:     int = len(all_rows)

        for rnd in range(1, n_rounds + 1):
            t0 = time.perf_counter()

            # ── Placement ────────────────────────────────────────────────
            if placement_method is None:
                # flat_fl: no UAV filter — all clients always covered (static, no dropouts).
                covered_all = {c.client_id: 0 for c in clients}
                placement_fitness = 1.0
            else:
                needs_placement = (uav_pos_m is None) or (dynamic and (rnd - 1) % T_sel == 0)
                if needs_placement:
                    uav_pos_m, ref, last_placement_fitness = _place_uavs(
                        client_coords=client_coord_map,
                        K=K,
                        R_comm=R_comm,
                        capacity=capacity,
                        method=placement_method,
                        rng=rng,
                        P=P,
                        G_max=G_max,
                        prev_positions_m=prev_uav_pos_m,
                    )
                    if prev_uav_pos_m is not None:
                        move_m = float(
                            np.sum(np.sqrt(np.sum((uav_pos_m - prev_uav_pos_m) ** 2, axis=1)))
                        )
                        cumulative_energy += _ENERGY_MODEL.energy_joules(move_m)
                    prev_uav_pos_m = uav_pos_m.copy()
                    uav_latlon = _uav_pos_to_latlon(uav_pos_m, ref)
                    covered_all = _covered_clients(client_coord_map, uav_pos_m, ref, R_comm)
                placement_fitness = last_placement_fitness

            # ── Client state + selection ──────────────────────────────────
            device_states = device_mgr.get_all_states()
            rep_scores    = rep_mgr.get_all_scores()

            selected: dict[int, int] = selector.select(
                covered           = covered_all,
                device_states     = device_states,
                reputation_scores = rep_scores,
                client_coords     = client_coord_map,
                uav_coords_latlon = uav_latlon,
                round_num         = rnd,
                uav_capacity      = capacity,
                mode              = selection_mode,
                rng               = rng,
            )

            coverage_pct = 100.0 * len(covered_all) / max(len(clients), 1)
            participation_pct = 100.0 * len(selected) / max(len(clients), 1)
            n_selected   = len(selected)

            # ── Build UAV groups from selection map ───────────────────────
            # Maps uav_idx → list of ClientData for clients assigned to that UAV.
            client_by_id = {c.client_id: c for c in clients}
            uav_groups: dict[int, list] = {j: [] for j in range(K)}
            for cid, uav_idx in selected.items():
                if cid in client_by_id:
                    uav_groups[uav_idx].append(client_by_id[cid])

            # ── UAV local training on imagery (paper §IV-A Step 3) ───────
            # Each UAV trains the full model (img_proj + struct + fusion) on
            # the pooled shard of all its assigned clients.  Uses the existing
            # 512-dim ResNet feature cache — no backbone forward pass needed.
            # flat_fl has no UAVs (placement_method is None), so it skips this.
            uav_img_updates: dict[int, tuple[dict, int]] = {}
            if placement_method is not None:
                for uav_idx, group in uav_groups.items():
                    uav_indices = [idx for c in group for idx in c.train_indices]
                    if not uav_indices:
                        continue
                    uav_loader = make_client_loader(cached_dataset, uav_indices, batch_size)
                    sd, n = _uav_local_train(global_model, uav_loader, n_uav_epochs, uav_lr)
                    uav_img_updates[uav_idx] = (sd, n)

            # ── IoT local training on structured data (paper §IV-A Step 5) ─
            # img_proj is frozen on global_model → clone inherits the freeze →
            # only struct_branch + fusion are updated (IoT-level payload).
            client_updates: dict[int, tuple[dict, int, float]] = {}
            for c in clients:
                if c.client_id not in selected or not c.train_indices:
                    continue
                loader = make_client_loader(cached_dataset, c.train_indices, batch_size)
                sd, n = _local_train(global_model, loader, n_local_epochs, lr)
                rep  = rep_scores.get(c.client_id, 0.5)
                client_updates[c.client_id] = (sd, n, rep)

            # Clients chosen for this round but unable to train (e.g. empty shard)
            # count as absent for the temporal-reliability term of their reputation.
            for cid in selected:
                if cid not in client_updates:
                    rep_mgr.mark_absent(cid)

            # ── Reputation update ─────────────────────────────────────────
            if client_updates:
                rep_mgr.update_batch(
                    {cid: sd for cid, (sd, _, _) in client_updates.items()},
                    global_update_vec=None,
                )

            # ── UAV-level aggregation (paper §IV-A Step 6) ───────────────
            # w̃_u = (n_img·w_img + Σ n_i·w_i) / (n_img + Σ n_i)
            # img_proj: UAV's update only.  struct+fusion: FedAvg of UAV+IoT.
            iot_by_uav: dict[int, list[tuple[dict, int, float]]] = {}
            for cid, triple in client_updates.items():
                uav_idx = selected[cid]
                iot_by_uav.setdefault(uav_idx, []).append(triple)

            uav_updates: list[tuple[dict, int, float]] = []
            for uav_idx in range(K):
                iot_upds = iot_by_uav.get(uav_idx, [])
                uav_img  = uav_img_updates.get(uav_idx)

                if not iot_upds and uav_img is None:
                    continue

                if uav_img is None:
                    # No UAV image data (empty coverage zone); IoT-only FedAvg.
                    # Carry current img_proj weights so server aggregation has
                    # a complete dict regardless of method.
                    total_n = sum(n for _, n, _ in iot_upds)
                    partial = (reputation_fedavg(iot_upds) if rep_weighted
                               else fedavg([(sd, n) for sd, n, _ in iot_upds]))
                    full_agg = {
                        **{f"img_proj.{k}": v.clone()
                           for k, v in global_model.img_proj.state_dict().items()},
                        **partial,
                    }
                    uav_rep = (sum(r * n for _, n, r in iot_upds) / max(total_n, 1)
                               if rep_weighted else 1.0)
                else:
                    total_n = uav_img[1] + sum(n for _, n, _ in iot_upds)
                    if rep_weighted:
                        full_agg = mixed_reputation_fedavg((*uav_img, 1.0), iot_upds)
                        iot_n = sum(n for _, n, _ in iot_upds)
                        uav_rep = (sum(r * n for _, n, r in iot_upds) / max(iot_n, 1)
                                   if iot_upds else 1.0)
                    else:
                        full_agg = mixed_fedavg(uav_img, [(sd, n) for sd, n, _ in iot_upds])
                        uav_rep = 1.0

                uav_updates.append((full_agg, total_n, uav_rep))

            # ── Server-level aggregation ──────────────────────────────────
            if uav_updates:
                if rep_weighted:
                    server_agg = reputation_fedavg(uav_updates)
                else:
                    server_agg = fedavg([(sd, n) for sd, n, _ in uav_updates])
                global_model.load_full_trainable_state_dict(server_agg)

            # ── Device state update ───────────────────────────────────────
            device_mgr.update_round(set(selected.keys()))

            # ── Evaluate ─────────────────────────────────────────────────
            metrics = _evaluate(global_model, cached_dataset, global_test_indices)
            elapsed = time.perf_counter() - t0

            if rounds_to_target is None and metrics["accuracy"] >= target_accuracy:
                rounds_to_target = rnd

            # Communication cost (MB): uplink + downlink
            # IoT↔UAV: IoT payload (struct+fusion only, _IOT_MODEL_SIZE_MB)
            # UAV↔server: UAV payload (img_proj+struct+fusion, _UAV_MODEL_SIZE_MB)
            # flat_fl: IoT↔server directly, IoT payload only
            if placement_method is None:
                comm_mb = 2.0 * n_selected * _IOT_MODEL_SIZE_MB
            else:
                n_active_uavs = len(uav_img_updates)
                comm_mb = (2.0 * n_selected * _IOT_MODEL_SIZE_MB
                           + 2.0 * n_active_uavs * _UAV_MODEL_SIZE_MB)

            all_rows.append({
                "method":             method,
                "round":              rnd,
                "accuracy":           metrics["accuracy"],
                "macro_f1":           metrics["macro_f1"],
                "coverage_pct":       coverage_pct,
                "participation_pct":  participation_pct,
                "n_selected":         n_selected,
                "placement_fitness":  placement_fitness,
                "comm_mb_round":      comm_mb,
                "cumulative_energy_j": cumulative_energy,
                "round_time_s":       elapsed,
                **{f"f1_{cls}": v for cls, v in metrics["f1_per_class"].items()},
            })
            logger.info(
                "Round %d/%d | acc=%.3f | macro-F1=%.3f | selected=%d/%.0f%% | %.1fs",
                rnd, n_rounds, metrics["accuracy"], metrics["macro_f1"],
                n_selected, coverage_pct, elapsed,
            )

        # Backfill per-method scalar onto all rows for this method.
        for row in all_rows[method_start_idx:]:
            row["rounds_to_target"] = rounds_to_target

        models_by_method[method] = global_model
        logger.info(
            "%s done. Rounds to %.0f%%: %s | Energy: %.1f kJ",
            method, target_accuracy * 100,
            rounds_to_target if rounds_to_target else "not reached",
            cumulative_energy / 1000,
        )

    # ── 3. Persist ───────────────────────────────────────────────────────────
    rounds_df = pd.DataFrame(all_rows)
    _write_table(rounds_df, results_dir / "fullsim_rounds.parquet")

    with open(results_dir / "config.fullsim.resolved.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    size_mb = sum(p.stat().st_size for p in results_dir.rglob("*") if p.is_file()) / 1e6
    logger.info("Full-system results at %s (%.2f MB)", results_dir, size_mb)

    return {
        "rounds": rounds_df,
        "results_dir": results_dir,
        "size_mb": size_mb,
        "models": models_by_method,
    }
