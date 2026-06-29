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

import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Subset

from hflsim.shared.coords import haversine, latlon_to_meters
from uavbench.optimizers import REGISTRY
from uavbench.problem.fitness import Fitness
from uavbench.problem.instance import ProblemInstance

from .dataset import CachedDataset, ClientData, SyntheticClientData, make_client_loader
from .features import compute_feature_cache, synthetic_feature_cache
from .model import CachedFusionModel, clone_model, fedavg

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
        prev_positions_m = np.zeros((K, 3))
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
    """Train a client-local copy of the model; return (trainable_state_dict, n_samples)."""
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
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
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
        _seed = (cfg.get("optimizer_seed", 9876) + N_clients * 7919 + hash(method) % (2**31)) % (2**31)
        rng = np.random.default_rng(_seed)

        global_model = CachedFusionModel()
        prev_uav_positions_m: np.ndarray | None = None
        rounds_to_target: int | None = None
        cumulative_energy_j: float = 0.0

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
                    cumulative_energy_j += 250.0 * (move_m / 15.0)
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
            comm_cost = n_covered  # one upload per covered client per round

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
                "comm_cost": comm_cost,
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

    import yaml
    with open(results_dir / "config.tier2.resolved.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    size_mb = sum(p.stat().st_size for p in results_dir.rglob("*") if p.is_file()) / 1e6
    logger.info("Tier-2 results at %s (%.2f MB)", results_dir, size_mb)

    return {"rounds": rounds_df, "results_dir": results_dir, "size_mb": size_mb}
