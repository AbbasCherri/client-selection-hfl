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
import yaml
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset

from hflsim.shared.coords import haversine_matrix, latlon_to_meters
from hflsim.shared.value import compute_value
from uavbench.optimizers import build_optimizer
from uavbench.problem.energy import EnergyModel
from uavbench.problem.fitness import Fitness
from uavbench.problem.instance import ProblemInstance

from .client_selection import ClientSelector
from .dataset import CachedDataset, ClientData, make_client_loader
from .device_state import DeviceStateManager
from .fairness import jain_index
from .features import compute_feature_cache
from .model import (
    CachedFusionModel,
    clone_model,
    fedavg,
    reputation_fedavg,
)
from .reputation import ReputationManager, trimmed_mean
from .seeds import fullsim_method_seed, tier2_seed

logger = logging.getLogger("uavbench.fl.federated")

_ENERGY_MODEL = EnergyModel()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_problem_instance(
    client_coords: dict[int, tuple[float, float]],
    K: int,
    R_comm: float,
    capacity: int,
    prev_positions_m: np.ndarray | None,
    value: np.ndarray | None = None,
) -> ProblemInstance:
    """Construct a ProblemInstance from client geographic coordinates.

    ``value`` carries the per-device V_i(t) = β·U_i + (1−β)·R_i scores so the
    placement fitness weights coverage by device value (paper §IV-E1). Callers
    without utility/reputation state (e.g. the Tier-2 placement benchmark) may
    omit it, in which case coverage is unweighted.
    """
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

    return (
        ProblemInstance(
            device_coords=device_coords,
            value=np.ones(N) if value is None else np.asarray(value, dtype=np.float64),
            capacity=np.full(K, float(capacity)),
            battery=np.ones(K),
            prev_positions=prev_positions_m,
            lower=lower,
            upper=upper,
            R_comm=R_comm,
            B_min_uav=0.0,  # battery is always 1 in this bridge
        ),
        ref,
    )


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
    value: np.ndarray | None = None,
    method_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray | None]:
    """Run a placement optimizer and return UAV positions in metres.

    Returns
    -------
    uav_positions_m : (K, 3) metres in the projected frame
    ref             : (lat0, lon0) reference for back-projection
    best_fitness    : placement fitness score
    radii           : (K,) per-UAV communication radii in metres, or None for
                      methods using the shared scalar R_comm (set by
                      path-loss-based optimizers via ``result.meta["radii"]``)
    """
    instance, ref = _build_problem_instance(
        client_coords, K, R_comm, capacity, prev_positions_m, value=value
    )
    fitness = Fitness(instance)

    optimizer = build_optimizer(method, params=method_params, budget={"P": P, "G_max": G_max})
    result = optimizer.optimize(instance, fitness, rng)

    uav_pos = result.best_position.reshape(K, 3)
    return uav_pos, np.array(ref), result.best_fitness, result.meta.get("radii")


def _covered_clients(
    client_coords: dict[int, tuple[float, float]],
    uav_pos_m: np.ndarray,
    ref: np.ndarray,
    R_comm: float,
    radii: np.ndarray | None = None,
) -> dict[int, int]:
    """Return {client_id: assigned_uav_idx} for clients within R_comm of any UAV.

    Converts UAV metre positions back to (lat, lon) for Haversine range check.
    Assigns each covered client to its nearest UAV. ``radii`` optionally
    supplies a per-UAV ``(K,)`` range (metres) overriding the scalar R_comm.
    """
    lat0, lon0 = float(ref[0]), float(ref[1])
    lat0_rad = math.radians(lat0)
    R = 6_371_000.0

    uav_latlon = np.column_stack(
        [
            lat0 + np.degrees(uav_pos_m[:, 1] / R),
            lon0 + np.degrees(uav_pos_m[:, 0] / (R * math.cos(lat0_rad))),
        ]
    )

    cids = list(client_coords.keys())
    client_latlon = np.array([client_coords[c] for c in cids], dtype=np.float64)
    dists = haversine_matrix(client_latlon, uav_latlon)  # (N, K)
    nearest = dists.argmin(axis=1)
    nearest_dist = dists[np.arange(len(cids)), nearest]
    limits = radii[nearest] if radii is not None else R_comm

    return {
        cid: int(uav)
        for cid, uav, d, lim in zip(
            cids, nearest, nearest_dist, np.broadcast_to(limits, nearest_dist.shape)
        )
        if d <= lim
    }


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
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "f1_per_class": {},
            "confusion_matrix": np.zeros((4, 4), dtype=int),
        }

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    return _evaluate_loader(model, loader)


def _evaluate_loader(model: CachedFusionModel, loader: DataLoader) -> dict:
    """Metric computation on a prebuilt (non-empty) test loader.

    Split out of ``_evaluate`` so callers that evaluate every round can build
    the loader once instead of re-wrapping the test subset per round.
    """
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
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1, 2, 3]),
    }


# Damage classes in label order 0-3 (shared by confusion reporting/plots).
CLASS_NAMES = ["survived", "collapsed", "obstructed", "missing"]


def _confusion_rows(method: str, rnd: int, cm: np.ndarray) -> list[dict]:
    """Flatten a (4,4) confusion matrix into long-form rows for parquet.

    The 4x4 matrix doesn't belong in the flat per-round row; long form keeps
    the rounds table schema stable and the matrix queryable per (method, round).
    """
    return [
        {
            "method": method,
            "round": rnd,
            "true_label": CLASS_NAMES[t],
            "pred_label": CLASS_NAMES[p],
            "count": int(cm[t, p]),
        }
        for t in range(4)
        for p in range(4)
    ]


def _report_black_chip_rate(full_dataset, cfg: dict) -> None:
    """Log and persist the real-data black-chip rate diagnostic.

    A black chip (GSI tile fetch failure) carries zero image signal; a high
    rate silently degrades accuracy toward majority-class collapse, so the
    measured rate is stored under ``cfg["_diagnostics"]`` — which the harness
    already dumps to its resolved-config YAML — as auditable evidence that
    the image modality was informative. The counters accumulate during the
    feature-cache build (the only phase that loads images), so call this
    after ``compute_feature_cache``. No-op for datasets without the counter
    (prebuilt test fixtures).
    """
    rate_fn = getattr(full_dataset, "black_chip_rate", None)
    if rate_fn is None:
        return
    rate = float(rate_fn())
    level = logging.WARNING if rate > 0.5 else logging.INFO
    logger.log(level, "Black-chip rate: %.1f%% of image loads", 100.0 * rate)
    cfg.setdefault("_diagnostics", {})["black_chip_rate"] = rate


def _dump_resolved_cfg(cfg: dict, path: Path) -> None:
    """Persist the fully-resolved config, eliding non-YAML-safe injections.

    The ``data.prebuilt`` test seam carries live datasets/arrays; the dump
    records its presence without trying to serialize it.
    """
    out = dict(cfg)
    if isinstance(out.get("data"), dict) and "prebuilt" in out["data"]:
        out["data"] = {**out["data"], "prebuilt": "<injected prebuilt data>"}
    with open(path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)


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
    T_sel: int = cfg["fl"].get("T_sel", 5)
    P: int = cfg["budget"]["P"]
    G_max: int = cfg["budget"]["G_max"]
    # optimizer_params.<method> config blocks (same convention as the Tier-1
    # runner) — placement-method kwargs like the path-loss baselines' link
    # budget. P/G_max always come from `budget` above (build_optimizer rule).
    optimizer_params: dict = cfg.get("optimizer_params", {})
    methods: list[str] = cfg["methods"]
    target_accuracy: float = cfg["fl"].get("target_accuracy", 0.70)

    # ------------------------------------------------------------------
    # 1. Load data (shared loader; real-data only, prebuilt = test seam)
    # ------------------------------------------------------------------
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
    logger.info("%d clients loaded.", len(clients))

    all_rows: list[dict] = []
    confusion_rows: list[dict] = []

    # ------------------------------------------------------------------
    # 2. Outer loop: one full FL run per placement method
    # ------------------------------------------------------------------
    for method in methods:
        logger.info("=== Method: %s ===", method)
        _seed = tier2_seed(cfg.get("optimizer_seed", 9876), len(clients), method)
        rng = np.random.default_rng(_seed)
        torch.manual_seed(_seed)  # deterministic model init across runs

        global_model = CachedFusionModel()
        prev_uav_positions_m: np.ndarray | None = None
        uav_pos_m: np.ndarray | None = None
        covered: dict[int, int] = {}
        last_placement_fitness: float = 0.0
        rounds_to_target: int | None = None
        cumulative_energy_j: float = 0.0
        # Tier-2 has no selection layer: every covered client trains each
        # round, so participation counts track coverage persistence.
        sel_counts: dict[int, int] = {c.client_id: 0 for c in clients}

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
                # Reposition only every T_sel rounds (paper §IV-E6), not per round.
                needs_placement = (uav_pos_m is None) or ((rnd - 1) % T_sel == 0)
                if needs_placement:
                    uav_pos_m, ref, last_placement_fitness, uav_radii = _place_uavs(
                        client_coords={c.client_id: c.coords for c in clients},
                        K=K,
                        R_comm=R_comm,
                        capacity=capacity,
                        method=method,
                        rng=rng,
                        P=P,
                        G_max=G_max,
                        prev_positions_m=prev_uav_positions_m,
                        method_params=optimizer_params.get(method, {}),
                    )
                    if prev_uav_positions_m is not None:
                        move_m = float(
                            np.sum(np.sqrt(np.sum((uav_pos_m - prev_uav_positions_m) ** 2, axis=1)))
                        )
                        cumulative_energy_j += _ENERGY_MODEL.energy_joules(move_m)
                    prev_uav_positions_m = uav_pos_m.copy()
                    covered = _covered_clients(
                        {c.client_id: c.coords for c in clients},
                        uav_pos_m,
                        ref,
                        R_comm,
                        radii=uav_radii,
                    )
                placement_fitness = last_placement_fitness

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

            for cid in covered:
                sel_counts[cid] += 1
            counts_arr = np.fromiter(sel_counts.values(), dtype=np.float64)

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
                "jain_fairness": jain_index(counts_arr),
                "n_unique_selected": int((counts_arr > 0).sum()),
                **{f"f1_{cls}": v for cls, v in metrics["f1_per_class"].items()},
            }
            all_rows.append(row)
            confusion_rows.extend(_confusion_rows(method, rnd, metrics["confusion_matrix"]))
            logger.info(
                "Round %d/%d | acc=%.3f | macro-F1=%.3f | covered=%d/%.0f%% | %.1fs",
                rnd,
                n_rounds,
                metrics["accuracy"],
                metrics["macro_f1"],
                n_covered,
                coverage_pct,
                elapsed,
            )

        # Backfill per-method scalar onto all rows for this method.
        for row in all_rows[method_start_idx:]:
            row["rounds_to_target"] = rounds_to_target

        logger.info(
            "%s finished. Rounds to target (%.0f%%): %s | Cumulative energy: %.1f kJ",
            method,
            target_accuracy * 100,
            rounds_to_target if rounds_to_target else "not reached",
            cumulative_energy_j / 1000,
        )

    # ------------------------------------------------------------------
    # 3. Persist results
    # ------------------------------------------------------------------
    rounds_df = pd.DataFrame(all_rows)
    _write_table(rounds_df, results_dir / "tier2_rounds.parquet")
    _write_table(pd.DataFrame(confusion_rows), results_dir / "confusion.parquet")

    _dump_resolved_cfg(cfg, results_dir / "config.tier2.resolved.yaml")

    size_mb = sum(p.stat().st_size for p in results_dir.rglob("*") if p.is_file()) / 1e6
    logger.info("Tier-2 results at %s (%.2f MB)", results_dir, size_mb)

    return {"rounds": rounds_df, "results_dir": results_dir, "size_mb": size_mb}


# ---------------------------------------------------------------------------
# Full paper system simulation
# ---------------------------------------------------------------------------

# IoT payload:  struct_branch (17,216) + fusion (50,436)         = 67,652 params ≈ 0.271 MB
# UAV payload:  img_proj (65,664) + struct_branch + fusion       = 133,316 params ≈ 0.533 MB
_IOT_MODEL_SIZE_MB: float = 67_652 * 4 / 1_000_000
_UAV_MODEL_SIZE_MB: float = 133_316 * 4 / 1_000_000
_MODEL_SIZE_MB: float = _IOT_MODEL_SIZE_MB  # kept for run_tier2 back-compat


# Method configuration: (placement_method, selection_mode, reputation_weighted, dynamic)
# placement_method: "pso" (authoritative placement optimizer) or None (flat/centralized);
#                   override per run via cfg["fl"]["placement_method"]
# selection_mode:   "ucb" | "random" | "all" | "fedcs" | "rep_cap" | "fair_mab"
# reputation_weighted: True → reputation_fedavg; False → uniform sample-weight fedavg
# dynamic:          True → reposition every T_sel rounds; False → place once at round 1
_METHOD_CFG: dict[str, tuple] = {
    "proposed_hfl": ("pso", "ucb", True, True),
    "flat_fl": (None, "all", False, False),
    "centralized": (None, "all", False, False),  # handled specially
    "hfl_no_selection": ("pso", "random", True, True),
    "hfl_static": ("pso", "ucb", True, False),
    "hfl_no_reputation": ("pso", "ucb", False, True),
    # Literature baselines (Algorithms B1-B3, REPORTS/literature_baselines.md):
    # identical PSO placement, reputation FedAvg, and T_sel cadence as
    # proposed_hfl — only the client-selection rule differs, isolating it as
    # the experimental variable.
    "fedcs": ("pso", "fedcs", True, True),  # Nishio & Yonetani, ICC 2019
    "rep_cap": ("pso", "rep_cap", True, True),  # Zhao et al., Chin. J. Aeronaut. 2024
    "fair_mab": ("pso", "fair_mab", True, True),  # Zhu et al., Sensors 2024
    # Placement literature baselines: identical UCB selection, reputation
    # FedAvg, and T_sel cadence as proposed_hfl — only the placement rule
    # differs, isolating it as the experimental variable (mirror image of
    # the selection baselines above).
    "mozaffari2016": ("mozaffari2016", "ucb", True, True),  # IEEE Comm. Lett. 2016
    "alzenad2017": ("alzenad2017", "ucb", True, True),  # IEEE WCL 2017
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


def _apply_black_chips(img_features: np.ndarray, rate: float, seed: int) -> np.ndarray:
    """Zero a deterministic fraction of image-feature rows (stress-test knob).

    Models additional unusable-imagery degradation on top of whatever
    fetch-failure black chips the real pipeline already produced (the
    measured natural rate is reported separately via
    ``_report_black_chip_rate``). Drawn from a dedicated RNG stream keyed
    off the data seed so nothing else in the run shifts with the rate, and
    applied to a copy so on-disk feature caches stay pristine.
    """
    if rate <= 0:
        return img_features
    out = img_features.copy()
    n_black = int(round(len(out) * rate))
    chip_rng = np.random.default_rng(seed + 977)
    out[chip_rng.choice(len(out), size=n_black, replace=False)] = 0.0
    return out


def _load_data(cfg: dict, results_dir: Path) -> tuple:
    """Shared data loading for the FL harnesses (run_tier2 / run_full_hfl).

    ``data.source``:

    * ``real`` (default) — stream/cache the pinned HF dataset
      (`AbbasABC/HFL-Dataset`); the **only** source that produces
      reportable results.
    * ``prebuilt`` — a caller-injected raw dict under
      ``cfg["data"]["prebuilt"]`` with the exact shape the real pipeline
      produces. Test seam only (tests/uavbench/synthetic_fixture.py);
      never used by checked-in configs.

    ``synthetic`` was removed 2026-07-14: the experimental pipeline is
    real-data only, and a config still requesting it fails loudly here.
    """
    data_cfg = cfg["data"]
    data_source = data_cfg.get("source", "real")
    black_chip_rate = data_cfg.get("black_chip_rate", 0.0)
    seed = data_cfg.get("seed", 42)

    if data_source == "prebuilt":
        raw = data_cfg["prebuilt"]
        return (
            raw["full_dataset"],
            raw["client_train_indices"],
            raw["global_test_indices"],
            raw["client_coords"],
            _apply_black_chips(raw["img_features"], black_chip_rate, seed),
        )
    if data_source != "real":
        raise ValueError(
            f"data.source={data_source!r} is not supported: the experimental "
            "pipeline is real-data only ('real'; 'prebuilt' is the test-injection "
            "seam). Synthetic data was removed from the library."
        )

    import os

    from hflsim.data import get_hfl_data_partitions

    logger.info("Loading real HFL dataset (N=%d clients)…", data_cfg["N_clients"])
    hf_token = os.environ.get("HF_TOKEN", data_cfg.get("hf_token"))
    full_dataset, client_train_indices, _, global_test_indices, client_coords = (
        get_hfl_data_partitions(
            csv_path=data_cfg.get("csv_path"),
            data_dir=data_cfg.get("data_dir", "./data"),
            N=data_cfg["N_clients"],
            subsample=data_cfg.get("subsample", 0.05),
            random_seed=seed,
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
    _report_black_chip_rate(full_dataset, cfg)
    return (
        full_dataset,
        client_train_indices,
        global_test_indices,
        client_coords,
        _apply_black_chips(img_features, black_chip_rate, seed),
    )


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
        rows.append(
            {
                "method": "centralized",
                "round": rnd,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "coverage_pct": 100.0,
                "n_selected": len(all_train_indices),
                "placement_fitness": 1.0,
                "comm_mb_round": 0.0,  # no communication
                "cumulative_energy_j": 0.0,
                "round_time_s": time.perf_counter() - t0,
                **{f"f1_{cls}": v for cls, v in metrics["f1_per_class"].items()},
            }
        )
        logger.info(
            "Centralized round %d/%d | acc=%.3f | macro-F1=%.3f",
            rnd,
            n_rounds,
            metrics["accuracy"],
            metrics["macro_f1"],
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
      fedcs             — literature B1: FedCS greedy deadline selection (Nishio & Yonetani 2019)
      rep_cap           — literature B2: reputation-capability ranking (Zhao et al. 2024)
      fair_mab          — literature B3: fairness/energy MAB selection (Zhu et al. 2024)

    Additional config keys vs run_tier2
    ------------------------------------
    fl.T_sel            : int   — reposition interval in rounds (default 5)
    fl.seed             : int   — per-run RNG seed (for multi-seed sweeps)
    client_simulation   : dict  — optional; enabled by default
    """
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    fl = cfg["fl"]
    n_rounds = fl["n_rounds"]
    n_local_epochs = fl["n_local_epochs"]
    lr = fl["lr"]
    batch_size = fl["batch_size"]
    K = fl["K"]
    R_comm = fl["R_comm"]
    capacity = fl["capacity"]
    T_sel = fl.get("T_sel", 5)
    lambda_min = fl.get("lambda_min", 0.5)  # early-reselection trigger (paper §IV-E6)
    R_min = fl.get("R_min", 0.3)  # min cluster reputation for aggregation (§IV-D)
    target_accuracy = fl.get("target_accuracy", 0.70)
    run_seed = fl.get("seed", cfg.get("optimizer_seed", 42))
    n_uav_epochs = fl.get("n_uav_epochs", n_local_epochs)
    uav_lr = fl.get("uav_lr", lr)
    placement_override = fl.get("placement_method")

    P = cfg["budget"]["P"]
    G_max = cfg["budget"]["G_max"]
    optimizer_params: dict = cfg.get("optimizer_params", {})

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
    epicentre = tuple(cfg.get("epicentre", [37.488, 137.272]))  # type: ignore[assignment]

    all_rows: list[dict] = []
    confusion_rows: list[dict] = []
    models_by_method: dict[str, CachedFusionModel] = {}

    # ── 2. Per-method outer loop ─────────────────────────────────────────────
    for method in cfg["methods"]:
        logger.info("=== Full-system method: %s ===", method)

        if method not in _METHOD_CFG:
            logger.warning("Unknown method %s — skipping.", method)
            continue

        placement_method, selection_mode, rep_weighted, dynamic = _METHOD_CFG[method]
        # fl.placement_method swaps the authoritative optimizer for the
        # proposed system and its ablations. Placement-literature baselines
        # (entries whose method name IS their placement rule) are exempt:
        # their placement is the experimental variable being compared.
        is_placement_baseline = placement_method == method
        if placement_method is not None and placement_override and not is_placement_baseline:
            placement_method = placement_override

        # Per-method seed: run-level seed folded with a stable method hash,
        # exactly once, inside fullsim_method_seed (see seeds.py for why
        # sweep callers must not pre-encode the method into run_seed).
        _seed = fullsim_method_seed(run_seed, method)
        rng = np.random.default_rng(_seed)
        torch.manual_seed(_seed)  # deterministic model init across runs

        global_model = CachedFusionModel()
        # img_proj frozen for IoT clients; UAV training unfreezes on its own clone.
        global_model.freeze_img_proj()

        # ── Centralized baseline: no federation at all ───────────────────
        if method == "centralized":
            rows = _run_centralized(
                global_model,
                cached_dataset,
                all_train_indices,
                global_test_indices,
                n_rounds,
                n_local_epochs,
                lr,
                batch_size,
            )
            all_rows.extend(rows)
            models_by_method[method] = global_model
            continue

        # ── Federated path ───────────────────────────────────────────────
        device_mgr = DeviceStateManager(
            client_ids,
            rng,
            dropout_rate=fl.get("dropout_rate", 0.0),
            snr_degradation_db=fl.get("snr_degradation_db", 0.0),
        )
        rep_mgr = ReputationManager(client_ids)
        selector = ClientSelector(client_ids, epicentre=epicentre)

        # Precompute static client-coord lookup (avoid rebuilding each round)
        client_coord_map: dict[int, tuple[float, float]] = {c.client_id: c.coords for c in clients}

        prev_uav_pos_m: np.ndarray | None = None
        uav_pos_m: np.ndarray | None = None
        uav_latlon: list[tuple[float, float]] = []
        ref: np.ndarray | None = None
        covered_all: dict[int, int] = {}
        selected: dict[int, int] = {}
        last_placement_fitness: float = 0.0
        cumulative_energy = 0.0
        rounds_to_target: int | None = None
        sel_counts: dict[int, int] = {cid: 0 for cid in client_ids}
        method_start_idx: int = len(all_rows)

        # Pre-project client coordinates once (static ground sensors): shared by
        # the per-selection V_i(t) computation. Order matches client_coord_map.
        _client_latlon = np.array([client_coord_map[c.client_id] for c in clients])
        _client_xy_m, _value_ref = latlon_to_meters(_client_latlon)
        _epi_xy_m, _ = latlon_to_meters(np.array([epicentre]), ref=_value_ref)
        _device_coords_m = np.column_stack([_client_xy_m, np.zeros(len(clients))])
        _epicentre_m = np.append(_epi_xy_m[0], 0.0)
        _samples_arr = np.array([len(c.train_indices) for c in clients], dtype=np.float64)

        for rnd in range(1, n_rounds + 1):
            t0 = time.perf_counter()

            # ── Client state (needed for both triggers and selection) ─────
            device_states = device_mgr.get_all_states()
            rep_scores = rep_mgr.get_all_scores()

            # Early-reselection trigger: eligible devices < λ_min · ΣC_u (§IV-E6).
            # The paper assumes N ≫ ΣC_u; when total UAV capacity exceeds the
            # client population (as in the paper_full config) the literal form
            # is always true, so the threshold is capped at the population size.
            n_eligible = sum(1 for st in device_states.values() if st.eligible())
            low_eligible = n_eligible < lambda_min * min(K * capacity, len(clients))

            # ── Placement ────────────────────────────────────────────────
            if placement_method is None:
                # flat_fl: no UAV filter — all clients always covered (static, no dropouts).
                covered_all = {c.client_id: 0 for c in clients}
                placement_fitness = 1.0
                reselect = True
            else:
                needs_placement = (uav_pos_m is None) or (
                    dynamic and ((rnd - 1) % T_sel == 0 or low_eligible)
                )
                if needs_placement:
                    # Per-device value V_i(t) = β(t)·U_i + (1−β(t))·R_i drives the
                    # placement fitness coverage term (paper §IV-E1).
                    snr_arr = np.array([device_states[c.client_id].snr_db for c in clients])
                    rep_arr = np.array([rep_scores.get(c.client_id, 0.5) for c in clients])
                    prev_for_value = (
                        prev_uav_pos_m
                        if prev_uav_pos_m is not None
                        else np.column_stack(
                            [
                                np.linspace(_client_xy_m[:, 0].min(), _client_xy_m[:, 0].max(), K),
                                np.linspace(_client_xy_m[:, 1].min(), _client_xy_m[:, 1].max(), K),
                                np.full(K, 70.0),
                            ]
                        )
                    )
                    device_values = compute_value(
                        _device_coords_m,
                        _epicentre_m,
                        snr_arr,
                        _samples_arr,
                        prev_for_value,
                        rep_arr,
                        t=rnd,
                        beta_mode="scheduled",
                        R_comm=R_comm,
                    )
                    uav_pos_m, ref, last_placement_fitness, uav_radii = _place_uavs(
                        client_coords=client_coord_map,
                        K=K,
                        R_comm=R_comm,
                        capacity=capacity,
                        method=placement_method,
                        rng=rng,
                        P=P,
                        G_max=G_max,
                        prev_positions_m=prev_uav_pos_m,
                        value=device_values,
                        method_params=optimizer_params.get(placement_method, {}),
                    )
                    if prev_uav_pos_m is not None:
                        move_m = float(
                            np.sum(np.sqrt(np.sum((uav_pos_m - prev_uav_pos_m) ** 2, axis=1)))
                        )
                        cumulative_energy += _ENERGY_MODEL.energy_joules(move_m)
                    prev_uav_pos_m = uav_pos_m.copy()
                    uav_latlon = _uav_pos_to_latlon(uav_pos_m, ref)
                    covered_all = _covered_clients(
                        client_coord_map, uav_pos_m, ref, R_comm, radii=uav_radii
                    )
                placement_fitness = last_placement_fitness
                # Client selection runs every T_sel rounds or on trigger (§IV-C),
                # not every round; between selections the roster persists.
                reselect = (rnd - 1) % T_sel == 0 or low_eligible or not selected

            # ── Client selection ──────────────────────────────────────────
            if reselect:
                selected = selector.select(
                    covered=covered_all,
                    device_states=device_states,
                    reputation_scores=rep_scores,
                    client_coords=client_coord_map,
                    uav_coords_latlon=uav_latlon,
                    round_num=rnd,
                    uav_capacity=capacity,
                    mode=selection_mode,
                    rng=rng,
                    R_comm=R_comm,
                    t_stale_cap=T_sel,  # fair_mab staleness saturates on the reselection cadence
                )

            coverage_pct = 100.0 * len(covered_all) / max(len(clients), 1)
            participation_pct = 100.0 * len(selected) / max(len(clients), 1)
            n_selected = len(selected)

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
            global_trainable = global_model.trainable_state_dict()
            client_updates: dict[int, tuple[dict, int, float]] = {}
            client_deltas: dict[int, dict] = {}
            for c in clients:
                if c.client_id not in selected or not c.train_indices:
                    continue
                loader = make_client_loader(cached_dataset, c.train_indices, batch_size)
                sd, n = _local_train(global_model, loader, n_local_epochs, lr)
                rep = rep_scores.get(c.client_id, 0.5)
                client_updates[c.client_id] = (sd, n, rep)
                # Reputation scores the update *delta* Δw_n, not absolute weights.
                client_deltas[c.client_id] = {k: v - global_trainable[k] for k, v in sd.items()}

            # Clients chosen for this round but unable to train (e.g. empty shard)
            # count as absent for the temporal-reliability term of their reputation.
            for cid in selected:
                if cid not in client_updates:
                    rep_mgr.mark_absent(cid)

            # ── Reputation update ─────────────────────────────────────────
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
            # struct_branch + fusion: plain data-size FedAvg over the IoT
            # updates (reputation weighting is server-level only, §IV-A Step 7).
            # img_proj: overlaid from the UAV's own image training. The UAV's
            # struct+fusion update is *not* mixed in: in this simulation the
            # UAV trains on the pooled shards of its assigned IoT clients (no
            # separate aerial dataset exists), so including it would count
            # every sample twice; its unique contribution is the vision head.
            iot_by_uav: dict[int, list[tuple[dict, int, float]]] = {}
            for cid, triple in client_updates.items():
                uav_idx = selected[cid]
                iot_by_uav.setdefault(uav_idx, []).append(triple)

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
                    # Coverage zone with UAV imagery but no IoT deliveries:
                    # struct+fusion stay at the current global weights.
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

                # UAV reputation = trimmed mean (10% per tail) of its assigned
                # cluster's IoT reputations (paper §IV-C7).
                cluster_reps = [rep_scores.get(c.client_id, 0.5) for c in uav_groups[uav_idx]]
                uav_rep = trimmed_mean(cluster_reps) if cluster_reps else 1.0

                uav_updates.append((full_agg, total_n, uav_rep))

            # ── Server-level aggregation ──────────────────────────────────
            # Reputation-weighted FedAvg; UAVs whose cluster trimmed-mean
            # reputation falls below R_min are excluded this round (§IV-D).
            if uav_updates:
                if rep_weighted:
                    active = [u for u in uav_updates if u[2] >= R_min]
                    if active:
                        server_agg = reputation_fedavg(active)
                        global_model.load_full_trainable_state_dict(server_agg)
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
                comm_mb = (
                    2.0 * n_selected * _IOT_MODEL_SIZE_MB + 2.0 * n_active_uavs * _UAV_MODEL_SIZE_MB
                )

            for cid in selected:
                sel_counts[cid] += 1
            counts_arr = np.fromiter(sel_counts.values(), dtype=np.float64)

            all_rows.append(
                {
                    "method": method,
                    "round": rnd,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "coverage_pct": coverage_pct,
                    "participation_pct": participation_pct,
                    "n_selected": n_selected,
                    "placement_fitness": placement_fitness,
                    "comm_mb_round": comm_mb,
                    "cumulative_energy_j": cumulative_energy,
                    "round_time_s": elapsed,
                    "jain_fairness": jain_index(counts_arr),
                    "n_unique_selected": int((counts_arr > 0).sum()),
                    **{f"f1_{cls}": v for cls, v in metrics["f1_per_class"].items()},
                }
            )
            confusion_rows.extend(_confusion_rows(method, rnd, metrics["confusion_matrix"]))
            logger.info(
                "Round %d/%d | acc=%.3f | macro-F1=%.3f | selected=%d/%.0f%% | %.1fs",
                rnd,
                n_rounds,
                metrics["accuracy"],
                metrics["macro_f1"],
                n_selected,
                coverage_pct,
                elapsed,
            )

        # Backfill per-method scalar onto all rows for this method.
        for row in all_rows[method_start_idx:]:
            row["rounds_to_target"] = rounds_to_target

        models_by_method[method] = global_model
        logger.info(
            "%s done. Rounds to %.0f%%: %s | Energy: %.1f kJ",
            method,
            target_accuracy * 100,
            rounds_to_target if rounds_to_target else "not reached",
            cumulative_energy / 1000,
        )

    # ── 3. Persist ───────────────────────────────────────────────────────────
    rounds_df = pd.DataFrame(all_rows)
    _write_table(rounds_df, results_dir / "fullsim_rounds.parquet")
    _write_table(pd.DataFrame(confusion_rows), results_dir / "confusion.parquet")

    _dump_resolved_cfg(cfg, results_dir / "config.fullsim.resolved.yaml")

    size_mb = sum(p.stat().st_size for p in results_dir.rglob("*") if p.is_file()) / 1e6
    logger.info("Full-system results at %s (%.2f MB)", results_dir, size_mb)

    return {
        "rounds": rounds_df,
        "results_dir": results_dir,
        "size_mb": size_mb,
        "models": models_by_method,
    }
