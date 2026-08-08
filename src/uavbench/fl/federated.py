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

from hflsim.shared.coords import haversine_matrix, latlon_to_meters
from hflsim.shared.value import compute_value

# Reported-metric computation is shared across every harness via metrics.fl;
# the leading-underscore aliases keep this module's historical names.
from uavbench.metrics.fl import (
    CLASS_NAMES,  # noqa: F401 — re-exported for confusion plots/tests
    TARGET_CONSEC_ROUNDS as _TARGET_CONSEC_ROUNDS,
    jain_index,
    round_comm_mb,
)
from uavbench.metrics.fl import (
    IOT_MODEL_SIZE_MB as _IOT_MODEL_SIZE_MB,
)
from uavbench.metrics.fl import (
    UAV_MODEL_SIZE_MB as _UAV_MODEL_SIZE_MB,  # noqa: F401 — re-exported for tests
)
from uavbench.metrics.fl import (
    confusion_rows as _confusion_rows,
)
from uavbench.metrics.fl import (
    evaluate_subset as _evaluate,
)
from uavbench.optimizers import build_optimizer
from uavbench.problem.energy import EnergyModel
from uavbench.problem.fitness import Fitness
from uavbench.problem.instance import ProblemInstance
from uavbench.problem.link import LinkModel
from uavbench.reporting.tables import write_table as _write_table

from .class_histograms import build_class_info
from . import client_selection
from .client_selection import ClientSelector
from .dataset import BalancedShardLoader, CachedDataset, ClientData, make_client_loader
from .device_state import DeviceStateManager
from .features import compute_feature_cache
from .model import (
    CachedFusionModel,
    MomentumSGD,
    clip_grad_norm_,
    clone_model,
    fedavg,
    make_loss_fn,
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
    z_min_m: float = Z_MIN_M_DEFAULT,
    z_max_m: float = Z_MAX_M_DEFAULT,
) -> ProblemInstance:
    """Construct a ProblemInstance from client geographic coordinates.

    ``value`` carries the per-device V_i(t) = β·U_i + (1−β)·R_i scores so the
    placement fitness weights coverage by device value (paper §IV-E1). Callers
    without utility/reputation state (e.g. the Tier-2 placement benchmark) may
    omit it, in which case coverage is unweighted.

    ``z_min_m``/``z_max_m`` bound the altitude search — see the module constants
    for why the band is not the 120 m small-UAS ceiling.
    """
    latlon = np.array(list(client_coords.values()), dtype=np.float64)
    xy_m, ref = latlon_to_meters(latlon)
    N = len(xy_m)
    device_coords = np.column_stack([xy_m, np.zeros(N)])

    if not 0.0 < z_min_m < z_max_m:
        raise ValueError(f"need 0 < z_min_m < z_max_m; got {z_min_m}, {z_max_m}")
    lower = np.array([xy_m[:, 0].min(), xy_m[:, 1].min(), float(z_min_m)])
    upper = np.array([xy_m[:, 0].max(), xy_m[:, 1].max(), float(z_max_m)])

    if prev_positions_m is None:
        # Spread initial positions evenly across the bounding box.
        xs = np.linspace(lower[0], upper[0], K)
        ys = np.linspace(lower[1], upper[1], K)
        z0 = 0.5 * (float(z_min_m) + float(z_max_m))
        prev_positions_m = np.column_stack([xs, ys, np.full(K, z0)])

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


# Altitude band for UAV placement, metres AGL.
#
# This is NOT the 120 m (400 ft) small-UAS ceiling, and the reason is physical
# rather than convenient. The Al-Hourani channel's radius-versus-altitude curve
# peaks at an elevation angle theta_opt that depends only on the environment
# (20.34 deg suburban), so the channel-optimal ground radius is z/tan(theta_opt).
# Under a 20-120 m band that is 54-324 m: every configured R_comm above ~324 m
# pins the optimum at the ceiling, the vertical decision goes degenerate, and
# "3D placement" collapses back to a planar placement carrying a height column —
# exactly the failure link.py was written to remove.
#
# It is worse than degenerate at the top of the old sweep grid. At z=120 m and a
# 20 km radius the elevation angle is 0.34 deg, giving P(LoS) ~ 2.8%: the link is
# 97% NLoS, so the line-of-sight advantage that motivates an aerial base station
# is switched off entirely, and R_comm=20 km is only reachable because link.py
# back-solves whatever path-loss budget makes the configured radius achievable.
#
# 100-1000 m is the band the UAV-base-station literature this benchmark compares
# against actually assumes — Mozaffari 2016 and Alzenad 2017 both derive
# altitudes of hundreds of metres to kilometres from r/tan(theta), and clamping
# them to 120 m distorts the published methods rather than reproducing them.
# Operationally it corresponds to a disaster-response waiver or a larger UAS
# class, not routine small-UAS flight; state that assumption when reporting.
#
# The guard against silently re-introducing the degeneracy is
# tests/sanity_checks/check_altitude_band.py, which requires the radius-
# maximizing altitude to be strictly interior to the band at the configured
# R_comm.
Z_MIN_M_DEFAULT = 100.0
Z_MAX_M_DEFAULT = 1000.0


def _class_value_vector(
    clients: list[ClientData],
    counts: dict[int, np.ndarray] | None,
    scarcity: np.ndarray | None,
) -> np.ndarray:
    """Per-client minority-information weight ``Σ_c scarcity_c · count_c``.

    Normalized to mean 1, so it re-weights placement value without changing its
    overall scale. ``None`` counts (``class_source="none"``, or a ``pseudo``
    histogram not yet refreshed) return all-ones: class-aware placement must
    degrade together with the histogram it is derived from, or a "no class
    information" run would still be steering UAVs with it.
    """
    if counts is None or scarcity is None:
        return np.ones(len(clients))
    value = np.array([float(np.dot(scarcity, counts[c.client_id])) for c in clients])
    mean = float(value.mean())
    return value / mean if mean > 0 else np.ones(len(clients))


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
    link_model: str = "path_loss",
    z_min_m: float = Z_MIN_M_DEFAULT,
    z_max_m: float = Z_MAX_M_DEFAULT,
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
        client_coords, K, R_comm, capacity, prev_positions_m, value=value,
        z_min_m=z_min_m, z_max_m=z_max_m,
    )
    link = None
    if link_model == "path_loss":
        link = LinkModel(
            r_comm_m=R_comm,
            z_min_m=float(instance.lower[2]),
            z_max_m=float(instance.upper[2]),
        )
    elif link_model != "range_gate":
        raise ValueError(f"link_model must be 'path_loss' or 'range_gate'; got {link_model!r}")
    fitness = Fitness(instance, link=link)

    optimizer = build_optimizer(method, params=method_params, budget={"P": P, "G_max": G_max})
    result = optimizer.optimize(instance, fitness, rng)

    uav_pos = result.best_position.reshape(K, 3)

    # Equal-radius, canonical-normaliser re-score. `result.best_fitness` is
    # whatever the method measured ITSELF at, and two methods do not measure the
    # same thing:
    #   * mozaffari2016/alzenad2017 score through `fitness(x, radii=r*)` with the
    #     altitude-derived r* (618 m at the suburban preset), while PSO is scored
    #     at the instance R_comm. At R_comm=500 m that is 1.24x the radius; at
    #     R_comm=2 km it is 0.31x. The direction of the bias flips with R_comm, so
    #     the column is not merely shifted — it is uninterpretable across methods.
    #   * pso_plus with move_norm="reachable" optimises under a tighter d_max, so
    #     its returned fitness is on a different scale again.
    # Re-scoring the returned layout once on a fresh Fitness at the shared
    # R_comm puts every method on one ruler. Fresh so the optimizer's own
    # eval_count (already read back by Optimizer.optimize) stays untouched.
    placement_score = float(Fitness(instance, link=link)(uav_pos.reshape(-1)))
    radii = result.meta.get("radii")
    if link is not None and radii is None:
        # The coverage gate the placement was scored under, handed to the system
        # gate so participation matches the objective the UAVs were placed for.
        radii = np.asarray(link.slant_radius(uav_pos[:, 2]), dtype=np.float64)
    return uav_pos, np.array(ref), placement_score, radii


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


def _reuse_or_clone(
    model: CachedFusionModel,
    scratch: CachedFusionModel | None,
) -> CachedFusionModel:
    """``clone_model(model)``, reusing ``scratch`` as the destination if given.

    Constructing a fresh module re-runs every layer initializer and the whole
    nn.Module registration dance only to overwrite it — measurably the dominant
    cost of cloning (1377 us vs 168 us for the copy alone). Reusing a
    caller-owned module is safe wherever the caller consumes the returned state
    dict before the next call (it is cloned out by ``*_state_dict``) and builds
    its own optimizer per call.

    Copies per-parameter ``requires_grad`` as well as the values, which is
    load-bearing: ``run_tier2`` never freezes img_proj while
    ``run_selection_isolation`` does, and both reach here — the frozen/unfrozen
    split must follow the source model, not a fixed block set.
    """
    if scratch is None:
        return clone_model(model)
    with torch.no_grad():
        for src, dst in zip(model.parameters(), scratch.parameters()):
            dst.copy_(src)
            dst.requires_grad_(src.requires_grad)
    scratch.zero_grad(set_to_none=True)
    return scratch


def _local_train(
    model: CachedFusionModel,
    loader: BalancedShardLoader,
    n_epochs: int,
    lr: float,
    scratch: CachedFusionModel | None = None,
) -> tuple[dict, int, float]:
    """Train a client-local copy of the model; return
    (trainable_state_dict, n_samples, mean_loss).

    img_proj is frozen on the clone (inherited from global_model which has
    freeze_img_proj() called after construction).  Only struct_branch + fusion
    are updated — the IoT-level payload per paper §IV-B. mean_loss is the
    sample-weighted mean training loss over all epochs; the oort /
    power_of_choice baselines consume it as their statistical utility.

    ``scratch`` is an optional caller-owned reusable module (see
    :func:`_reuse_or_clone`); it must be built before any per-method
    ``torch.manual_seed`` so its construction draws cannot shift a seeded stream.
    """
    local = _reuse_or_clone(model, scratch)
    local.train()
    trainable = [p for p in local.parameters() if p.requires_grad]
    opt = optim.Adam(trainable, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n_seen = 0
    loss_sum = 0.0
    for _ in range(n_epochs):
        for img_feat, struct, labels in loader:
            opt.zero_grad()
            logits = local(img_feat, struct)
            loss = loss_fn(logits, labels)
            loss.backward()
            clip_grad_norm_(trainable, 1.0)
            opt.step()
            n_seen += labels.shape[0]
            loss_sum += float(loss.detach()) * labels.shape[0]

    mean_loss = loss_sum / max(n_seen, 1)
    return local.trainable_state_dict(), n_seen // max(n_epochs, 1), mean_loss


def _uav_local_train(
    model: CachedFusionModel,
    loader: BalancedShardLoader,
    n_epochs: int,
    lr: float,
    scratch: CachedFusionModel | None = None,
) -> tuple[dict, int]:
    """Train a UAV-local copy with img_proj unfrozen (full model, paper §IV-A Step 3).

    Uses the same cached 512-dim ResNet features as IoT clients — no raw image
    loading or backbone forward pass required.  img_proj learns to map ImageNet
    features to damage-relevant representations; IoT devices cannot do this.
    Returns (full_trainable_state_dict, n_samples).

    ``scratch`` — see :func:`_local_train`.
    """
    local = _reuse_or_clone(model, scratch)
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
            clip_grad_norm_(trainable, 1.0)
            opt.step()
            n_seen += labels.shape[0]

    return local.full_trainable_state_dict(), n_seen // max(n_epochs, 1)


def _make_optimizer(params, opt_cfg: dict, lr: float):
    """Build the per-round local optimizer from ``fl.local_optimizer`` config.

    Default SGD+momentum: unlike Adam (whose moment estimates reset on every
    fresh per-round clone and never amortize), momentum SGD carries no stale
    per-round warm-up bias and is the standard choice under FedAvg.
    """
    name = str(opt_cfg.get("name", "sgd")).lower()
    wd = float(opt_cfg.get("weight_decay", 0.0))
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    momentum = float(opt_cfg.get("momentum", 0.9))
    nesterov = bool(opt_cfg.get("nesterov", False))
    if wd == 0.0 and momentum != 0.0 and not nesterov:
        # The configured default. MomentumSGD is bit-identical to torch's SGD
        # here (see its docstring) and ~2x cheaper per step; anything outside
        # that restriction falls through to torch.optim below.
        return MomentumSGD(params, lr, momentum)
    return optim.SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=wd,
        nesterov=nesterov,
    )


def _train_blocks(
    model: CachedFusionModel,
    loader: BalancedShardLoader,
    n_epochs: int,
    lr: float,
    loss_fn,
    blocks: tuple[str, ...],
    opt_cfg: dict,
    scratch: CachedFusionModel | None = None,
) -> tuple[dict, int, float]:
    """Train a clone with gradients enabled only on ``blocks`` (Tier B owner).

    Returns ``(block_state_dict, n_samples, mean_loss)``. The frozen blocks
    still participate in the forward pass at their current global weights, so
    the trained block is optimized against the exact value of its counterpart.
    ``mean_loss`` is the sample-weighted training loss (statistical-utility feed
    for the oort / power_of_choice selectors).

    ``scratch``, if given, is a reusable same-architecture module the caller
    owns: its params are overwritten from ``model`` instead of constructing a
    fresh module per call (construction re-runs every layer initializer and the
    nn.Module bookkeeping just to be overwritten — measurably the dominant cost
    of cloning). Safe to reuse across sequential calls because the returned
    state dict is cloned (block_state_dict) and the optimizer is per-call;
    requires_grad and train() are (re)set below either way. Bit-identical:
    param copy consumes no RNG (clone_model already guaranteed that), stale
    grads are cleared, and training then follows the identical op sequence.
    """
    local = _reuse_or_clone(model, scratch)
    local.set_trainable_blocks(blocks)
    local.train()
    trainable = [p for p in local.parameters() if p.requires_grad]
    opt = _make_optimizer(trainable, opt_cfg, lr)

    n_seen = 0
    loss_sum = 0.0
    for _ in range(n_epochs):
        for img_feat, struct, labels in loader:
            opt.zero_grad()
            logits = local(img_feat, struct)
            loss = loss_fn(logits, labels)
            loss.backward()
            clip_grad_norm_(trainable, 1.0)
            opt.step()
            n_seen += labels.shape[0]
            loss_sum += float(loss.detach()) * labels.shape[0]

    mean_loss = loss_sum / max(n_seen, 1)
    return local.block_state_dict(blocks), n_seen // max(n_epochs, 1), mean_loss


def _lr_scale(rnd: int, n_rounds: int, schedule: str) -> float:
    """Across-round learning-rate multiplier in (0, 1].

    A fixed client LR under non-IID FedAvg orbits the optimum indefinitely
    (the plateau-plus-oscillation seen in the diagnostics); decaying it lets the
    global model settle. ``rnd`` is 1-indexed.
    """
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * (rnd - 1) / max(n_rounds - 1, 1)))
    if schedule in ("sqrt", "inv_sqrt"):
        return 1.0 / math.sqrt(rnd)
    return 1.0


def _apply_server_momentum(
    global_state: dict[str, torch.Tensor],
    aggregate: dict[str, torch.Tensor],
    velocity: dict[str, torch.Tensor],
    momentum: float,
    server_lr: float,
) -> dict[str, torch.Tensor]:
    """FedAvgM server update over the blocks present in ``aggregate``.

    Treats ``(global − aggregate)`` as the server pseudo-gradient, applies
    heavy-ball momentum, and returns the new weights for exactly those keys.
    Reduces to plain FedAvg when ``momentum=0`` and ``server_lr=1``. ``velocity``
    is mutated in place so it persists across rounds.
    """
    # Batched with torch._foreach_* (one call per step across all keys instead
    # of ~5 small torch ops per key). Per-element op sequence is unchanged —
    # pseudo = g − agg; v = m·v + pseudo; out = g − lr·v — so the result is
    # bit-identical to the per-key loop this replaces.
    keys = list(aggregate.keys())
    g_list = [global_state[k].float() for k in keys]
    pseudo = torch._foreach_sub(g_list, [aggregate[k].float() for k in keys])
    if all(k in velocity for k in keys):
        new_v = torch._foreach_mul([velocity[k] for k in keys], momentum)
        torch._foreach_add_(new_v, pseudo)
    else:  # first round for (some of) these keys — seed velocity with pseudo
        new_v = [
            pseudo[i] if k not in velocity else momentum * velocity[k] + pseudo[i]
            for i, k in enumerate(keys)
        ]
    for k, v in zip(keys, new_v):
        velocity[k] = v
    out_list = torch._foreach_sub(g_list, torch._foreach_mul(new_v, server_lr))
    return dict(zip(keys, out_list))


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
    # Hard gate: above this rate the image modality is effectively absent and
    # accuracy silently collapses to the majority class (observed failure
    # mode). Refuse to produce results that would look valid but aren't;
    # override via data.max_black_chip_rate only for deliberate stress runs.
    max_rate = float(cfg.get("data", {}).get("max_black_chip_rate", 0.5))
    if rate > max_rate:
        raise RuntimeError(
            f"Black-chip rate {rate:.1%} exceeds data.max_black_chip_rate={max_rate:.1%}: "
            "the image modality is effectively missing (tile fetch failures). "
            "Fix the tile source/cache or raise the threshold explicitly for a stress run."
        )


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
    # Gate every method's coverage at the scalar R_comm, ignoring any
    # path-loss-derived per-UAV radius — same knob as run_full_hfl. Referenced
    # below but never defined here from e79701f5 until 2026-07-26, so every
    # placement method in this harness raised NameError on its first placement.
    # Air-to-ground link model gating coverage. "path_loss" derives each UAV's
    # radius from its own altitude through one shared channel, which is what
    # makes altitude a real decision and removes the need for a uniform-radius
    # override — every method is already on the same model, so the override is
    # forced off. "range_gate" is the legacy flat R_comm behaviour.
    link_model: str = str(cfg["fl"].get("link_model", "path_loss"))
    z_min_m: float = float(cfg["fl"].get("z_min_m", Z_MIN_M_DEFAULT))
    z_max_m: float = float(cfg["fl"].get("z_max_m", Z_MAX_M_DEFAULT))
    uniform_coverage_radius: bool = bool(cfg["fl"].get("uniform_coverage_radius", False))
    if link_model == "path_loss":
        uniform_coverage_radius = False
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
    (
        full_dataset,
        client_train_indices,
        global_test_indices,
        client_coords,
        img_features,
        global_val_indices,
    ) = _load_data(cfg, results_dir)

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

    # Reusable per-client training module (see _reuse_or_clone). Built BEFORE
    # the per-method torch.manual_seed so its construction-time RNG draws
    # cannot shift any seeded stream.
    scratch_model = CachedFusionModel()

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
        target_streak = 0
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
                        link_model=link_model,
                        z_min_m=z_min_m,
                        z_max_m=z_max_m,
                    )
                    # cumulative_energy_j counts repositioning (movement) energy
                    # only — hover/communication energy has no simulated-time
                    # model here, so non-repositioning methods legitimately
                    # report 0 J. Label it as movement energy in figures.
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
                        radii=None if uniform_coverage_radius else uav_radii,
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
                    sd, n, _loss = _local_train(
                        global_model, loader, n_local_epochs, lr, scratch=scratch_model
                    )
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
            # No UAV→server hop (Tier-2 flat placement harness).
            comm_mb_round = round_comm_mb(n_covered)

            if metrics["accuracy"] >= target_accuracy:
                target_streak += 1
                if rounds_to_target is None and target_streak >= _TARGET_CONSEC_ROUNDS:
                    rounds_to_target = rnd - _TARGET_CONSEC_ROUNDS + 1
            else:
                target_streak = 0

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

# Payload-size constants live in uavbench.metrics.fl (imported above).
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
    # Literature baselines (Algorithms B1-B3, REPORTS/master_implementation_reference.md Appendix C):
    # identical PSO placement, reputation FedAvg, and T_sel cadence as
    # proposed_hfl — only the client-selection rule differs, isolating it as
    # the experimental variable.
    "fedcs": ("pso", "fedcs", True, True),  # Nishio & Yonetani, ICC 2019
    "rep_cap": ("pso", "rep_cap", True, True),  # Zhao et al., Chin. J. Aeronaut. 2024
    "fair_mab": ("pso", "fair_mab", True, True),  # Zhu et al., Sensors 2024
    "oort": ("pso", "oort", True, True),  # Lai et al., OSDI 2021
    "power_of_choice": ("pso", "power_of_choice", True, True),  # Cho et al., 2020
    # Placement literature baselines: identical UCB selection, reputation
    # FedAvg, and T_sel cadence as proposed_hfl — only the placement rule
    # differs, isolating it as the experimental variable (mirror image of
    # the selection baselines above).
    "mozaffari2016": ("mozaffari2016", "ucb", True, True),  # IEEE Comm. Lett. 2016
    "alzenad2017": ("alzenad2017", "ucb", True, True),  # IEEE WCL 2017
    # Naive placement arms. hfl_static is NOT a bad-placement baseline — it uses
    # PSO placement, just once instead of every T_sel rounds — so with static
    # clients it ties the proposed system and isolates *cadence*, not placement
    # quality. These two supply the missing contrast: identical UCB selection /
    # reputation FedAvg / cadence, but a naive placement rule.
    "centroid_place": ("centroid", "ucb", True, True),  # k-means centroids
    "random_place": ("random", "ucb", True, True),  # uniform-random UAV positions
    # Candidate-set placement: the proposed placement rule, solving the vertical
    # subproblem in closed form and the horizontal one over the exact
    # circle-intersection candidate set (Church 1984). Measured at 97-99% of the
    # capacitated-MCLP optimum against PSO's 52% at R_comm = 500 m.
    "mclp_place": ("mclp_ls", "ucb", True, True),
    # Further placement literature, all on the identical selection/reputation/
    # cadence stack so the placement rule stays the only variable.
    "spiral_place": ("spiral", "ucb", True, True),  # Lyu et al., IEEE Comm. Lett. 2017
    "cap_kmeans_place": ("cap_kmeans", "ucb", True, True),  # capacity-constrained k-means
    "pso_cluster_place": ("pso_cluster", "ucb", True, True),  # Sawalmeh et al., Sensors 2021
    "ahc_place": ("ahc", "ucb", True, True),  # agglomerative clustering + minimax centre
    "moon2022": ("moon2022", "ucb", True, True),  # Moon et al., Electronics 11(7):1036, 2022
    "mogoa": ("mogoa", "ucb", True, True),  # Almaameri & Blazovics, Cluster Comput. 29:392, 2026
    "ga_place": ("ga", "ucb", True, True),  # GA over raw UAV coordinates
    "de_place": ("de", "ucb", True, True),  # differential evolution control
    "gwo_place": ("gwo", "ucb", True, True),  # grey wolf control
}

# Methods whose placement rule IS the experimental variable — exempt from the
# fl.placement_method override (which otherwise swaps every method to PSO).
_PLACEMENT_BASELINES = frozenset(
    {
        "mozaffari2016", "alzenad2017", "centroid_place", "random_place",
        "mclp_place", "spiral_place", "cap_kmeans_place", "pso_cluster_place",
        "ahc_place", "ga_place", "de_place", "gwo_place", "moon2022", "mogoa",
    }
)


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
            # Test fixtures predate the val split; an absent key means "no
            # validation set", which callers must treat as "cannot select on
            # val" rather than silently falling back to test.
            raw.get("global_val_indices", []),
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
    (
        full_dataset,
        client_train_indices,
        _,
        global_test_indices,
        client_coords,
        _client_val_indices,
        global_val_indices,
    ) = get_hfl_data_partitions(
        csv_path=data_cfg.get("csv_path"),
        data_dir=data_cfg.get("data_dir", "./data"),
        N=data_cfg["N_clients"],
        subsample=data_cfg.get("subsample", 0.05),
        random_seed=seed,
        hf_token=hf_token,
        partition_seed=data_cfg.get("partition_seed"),
        val_ratio=data_cfg.get("val_ratio", 0.0),
    )
    # Allow the sweep to provide a shared N-level cache (avoids recomputing per seed).
    cache_path = data_cfg.get("feature_cache_path") or str(results_dir / "img_features.npy")
    img_features = compute_feature_cache(
        full_dataset,
        cache_path=cache_path,
        batch_size=data_cfg.get("feature_batch_size", 32),
        # Image decode parallelism for the one-time feature pass; 0 = main
        # process (the safe default). The sweeps prefetch caches sequentially
        # before forking workers, so raising this never multiplies processes.
        num_workers=data_cfg.get("feature_num_workers", 0),
    )
    _report_black_chip_rate(full_dataset, cfg)
    return (
        full_dataset,
        client_train_indices,
        global_test_indices,
        client_coords,
        _apply_black_chips(img_features, black_chip_rate, seed),
        global_val_indices,
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
    loss_fn,
    opt_cfg: dict,
    balanced: bool,
) -> tuple[list[dict], list[dict]]:
    """Oracle: train on all data at one node, report metrics every n_local_epochs epochs.

    Shares the paper pipeline's logit-adjusted loss and configured optimizer so
    the upper bound is measured under the same training recipe as the federated
    methods. The optimizer is created once and persists across rounds (single
    continuous optimization), so momentum/Adam state amortizes here as intended.

    Returns (round_rows, confusion_rows) — confusion was silently dropped for
    this baseline before 2026-07-18 (empty confusion.parquet columns for the
    centralized method)."""
    rows: list[dict] = []
    conf_rows: list[dict] = []
    loader = make_client_loader(cached_dataset, all_train_indices, batch_size, balanced=balanced)
    # Centralized has full compute — train the entire model including img_proj.
    global_model.unfreeze_img_proj()
    trainable = [p for p in global_model.parameters() if p.requires_grad]
    opt = _make_optimizer(trainable, opt_cfg, lr)

    for rnd in range(1, n_rounds + 1):
        t0 = time.perf_counter()
        global_model.train()
        for _ in range(n_local_epochs):
            for img_feat, struct, labels in loader:
                opt.zero_grad()
                logits = global_model(img_feat, struct)
                loss_fn(logits, labels).backward()
                clip_grad_norm_(trainable, 1.0)
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
        conf_rows.extend(_confusion_rows("centralized", rnd, metrics["confusion_matrix"]))
        logger.info(
            "Centralized round %d/%d | acc=%.3f | macro-F1=%.3f",
            rnd,
            n_rounds,
            metrics["accuracy"],
            metrics["macro_f1"],
        )
    return rows, conf_rows


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
      oort              — literature B4: Oort guided participant selection (Lai et al. 2021)
      power_of_choice   — literature B5: Power-of-Choice loss-based sampling (Cho et al. 2020)

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
    # When True, every method's coverage is gated at the scalar R_comm, ignoring
    # any path-loss-derived per-UAV radius (mozaffari/alzenad). Needed for the
    # coverage sweep so their fixed ~20 km radius can't bypass a tight R_comm —
    # only PLACEMENT differs, not the coverage radius (Tier-1 equal-radius rule).
    # Air-to-ground link model gating coverage. "path_loss" derives each UAV's
    # radius from its own altitude through one shared channel, which is what
    # makes altitude a real decision and removes the need for a uniform-radius
    # override — every method is already on the same model, so the override is
    # forced off. "range_gate" is the legacy flat R_comm behaviour.
    link_model = str(fl.get("link_model", "path_loss"))
    # Altitude band — see Z_MIN_M_DEFAULT for why it is not the 120 m ceiling.
    # Lives in fl.* so it enters the resume signature: changing the band must
    # invalidate checkpoints, or a rerun would silently reuse placements
    # computed under different physics.
    z_min_m = float(fl.get("z_min_m", Z_MIN_M_DEFAULT))
    z_max_m = float(fl.get("z_max_m", Z_MAX_M_DEFAULT))
    uniform_coverage_radius = bool(fl.get("uniform_coverage_radius", False))
    if link_model == "path_loss":
        uniform_coverage_radius = False
    capacity = fl["capacity"]
    T_sel = fl.get("T_sel", 5)
    lambda_min = fl.get("lambda_min", 0.5)  # early-reselection trigger (paper §IV-E6)
    R_min = fl.get("R_min", 0.3)  # min cluster reputation for aggregation (§IV-D)
    target_accuracy = fl.get("target_accuracy", 0.70)
    run_seed = fl.get("seed", cfg.get("optimizer_seed", 42))
    n_uav_epochs = fl.get("n_uav_epochs", n_local_epochs)
    uav_lr = fl.get("uav_lr", lr)
    placement_override = fl.get("placement_method")

    # ── Training-recipe knobs (Tier A/B) ────────────────────────────────────
    # Logit-adjusted loss + uniform sampling (A1), momentum SGD (A2), server
    # momentum + LR decay (A3), EMA-of-global evaluation (A4), and modality-
    # aligned block ownership (B). Defaults are the improved recipe; every knob
    # is captured in the resume signature so a config change reruns the job.
    logit_tau = float(fl.get("logit_adjust_tau", 1.0))  # 0 → plain cross-entropy
    balanced_sampling = bool(fl.get("balanced_sampling", False))
    local_opt_cfg: dict = fl.get("local_optimizer", {"name": "sgd", "momentum": 0.9})
    server_momentum = float(fl.get("server_momentum", 0.9))
    server_lr = float(fl.get("server_lr", 1.0))
    lr_decay = str(fl.get("lr_decay", "cosine")).lower()  # "cosine" | "none"
    ema_decay = float(fl.get("ema_decay", 0.9))  # 0 → evaluate the raw global model
    fusion_owner = str(fl.get("fusion_owner", "uav")).lower()  # "uav" | "client"
    reselect_every = int(fl.get("reselect_every", 1))  # selection cadence (≠ placement T_sel)
    # Class-aware placement biases UAV coverage toward clients holding rare
    # classes. It carries an information assumption that class-aware *selection*
    # does not, and the difference is worth stating precisely:
    #
    #   Selection only ever ranks clients that some UAV already covers, so their
    #   histograms are observable by construction. Placement decides *who
    #   becomes reachable*, so conditioning it on per-client histograms needs
    #   those histograms from clients no UAV covers yet — circular unless the
    #   report path is separate from the data path.
    #
    # The assumption this benchmark makes, explicitly: the placer already
    # receives every client's coordinates (`client_coord_map`), exactly as every
    # placement baseline in the comparison set does — a demand map is the
    # standard input to maximal-covering placement. A 4-bin histogram is a few
    # bytes over that same low-rate control path, whereas `R_comm` is calibrated
    # for sustained model-update throughput. Going from "I know where the
    # devices are" to "I know their label mix" is an increment on an assumption
    # already made, not a new class of oracle.
    #
    # It is nonetheless an assumption, which is why this is an ablation axis and
    # not an unexamined default: report it on/off against the class_source
    # ladder rather than folding it into the headline claim.
    placement_class_aware = bool(fl.get("placement_class_aware", True))
    # Histogram source for BOTH selection and class-aware placement. Applied
    # globally (every HFL method shares it), so it is not a fairness asymmetry
    # between methods — but it is the same oracle-realism question, and the
    # degradation ladder has to cover placement too, not just selection.
    # Must be an explicit string. Unquoted YAML scalars are a trap here:
    # `class_source: true` parses as the bool True and `class_source:` (empty)
    # parses as None, both of which str().lower() would silently coerce — the
    # second one into the *lower-anchor* arm, i.e. a run with no class
    # information at all wearing the default's name. Fail loudly instead.
    _cs_raw = fl.get("class_source", "true")
    if not isinstance(_cs_raw, str):
        raise ValueError(
            f"fl.class_source must be a quoted string, got {_cs_raw!r} "
            f"({type(_cs_raw).__name__}) — write class_source: \"true\", not true"
        )
    class_source = _cs_raw.lower()
    dp_epsilon = float(fl.get("dp_epsilon", 1.0))
    # Rounds-to-target now tracks the reported primary metric (macro-F1) against
    # a meaningful floor, not raw accuracy against an 0.82 majority-class ceiling.
    target_metric = str(fl.get("target_metric", "macro_f1"))
    target_value = float(fl.get("target_value", fl.get("target_accuracy", 0.45)))

    P = cfg["budget"]["P"]
    G_max = cfg["budget"]["G_max"]
    optimizer_params: dict = cfg.get("optimizer_params", {})

    # ── 1. Load data ────────────────────────────────────────────────────────
    (
        full_dataset,
        client_train_indices,
        global_test_indices,
        client_coords,
        img_features,
        global_val_indices,
    ) = _load_data(cfg, results_dir)
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

    # ── Class statistics (shared by logit adjustment, selection, placement) ──
    # Global training-label prior → logit-adjustment shift (A1). Per-client
    # class histograms → class-coverage selection utility and class-scarcity
    # placement value (Tier C).
    #
    # The histogram SOURCE is an experimental variable, not a constant
    # (fl.class_source, see fl/class_histograms.py): through 2026-07 it was the
    # ground-truth bincount — a statistic no participant discloses. "pseudo"
    # (global-model predictions, zero disclosure) and "dp" (Laplace-noised,
    # one-shot release) are the realistic rungs. Note this histogram is built
    # ONCE, outside the round loop, from fixed train_indices: it is a one-time
    # release, not a per-round leak.
    _all_train_t = torch.as_tensor(all_train_indices, dtype=torch.long)
    _train_label_counts = torch.bincount(
        cached_dataset.labels[_all_train_t].to(torch.long), minlength=4
    ).to(torch.float64)
    _prior = _train_label_counts / _train_label_counts.sum().clamp_min(1.0)
    log_prior = torch.log(_prior.clamp_min(1e-8)).to(torch.float32) if logit_tau > 0 else None
    loss_fn = make_loss_fn(log_prior, tau=logit_tau)
    _client_train_idx = {c.client_id: list(c.train_indices) for c in clients}
    if class_source == "pseudo":
        # Needs a trained model, which does not exist yet — refreshed inside
        # each method's round loop below. Starting at None means a bug in that
        # refresh degrades to the documented no-class fallback rather than
        # silently reusing an oracle histogram. Mirrors selection_isolation.py.
        client_class_counts, class_scarcity = None, None
    else:
        client_class_counts, class_scarcity = build_class_info(
            class_source,
            labels=cached_dataset.labels,
            client_indices=_client_train_idx,
            global_prior=_prior.numpy() if class_source == "true" else None,
            epsilon=dp_epsilon,
            # run_seed, not data.seed: the DP noise draw belongs to the run's RNG
            # family so each seed repetition gets an independent release, and the
            # noise enters the confidence intervals like every other stochastic
            # component instead of being frozen across the sweep.
            rng=np.random.default_rng(run_seed),
        )
    # Epicentre — use config override or default to Noto Peninsula 2024
    epicentre = tuple(cfg.get("epicentre", [37.488, 137.272]))  # type: ignore[assignment]

    all_rows: list[dict] = []
    confusion_rows: list[dict] = []
    models_by_method: dict[str, CachedFusionModel] = {}

    # Reusable per-client/per-UAV training module (see _train_blocks). Built
    # BEFORE the per-method torch.manual_seed so its construction-time RNG
    # draws cannot shift any seeded stream.
    scratch_model = CachedFusionModel()

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
        is_placement_baseline = method in _PLACEMENT_BASELINES
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
            rows, cent_conf = _run_centralized(
                global_model,
                cached_dataset,
                all_train_indices,
                global_test_indices,
                n_rounds,
                n_local_epochs,
                lr,
                batch_size,
                loss_fn,
                local_opt_cfg,
                balanced_sampling,
            )
            all_rows.extend(rows)
            confusion_rows.extend(cent_conf)
            models_by_method[method] = global_model
            continue

        # Block ownership (Tier B). Hierarchical methods let the UAV tier own
        # img_proj + fusion (both modalities co-located there) and clients own
        # struct_branch alone. flat_fl has no UAV tier, so its clients must
        # still own struct_branch + fusion (img_proj stays at init — it cannot
        # use imagery, which is precisely the limitation the hierarchy removes).
        has_uav_tier = placement_method is not None
        if has_uav_tier:
            uav_blocks = ("img_proj", "fusion") if fusion_owner == "uav" else ("img_proj",)
            client_blocks = ("struct_branch",) if fusion_owner == "uav" else ("struct_branch", "fusion")
        else:
            uav_blocks = ()
            client_blocks = ("struct_branch", "fusion")

        # EMA-of-global evaluation model (A4) and server-momentum buffer (A3),
        # both per-method (reset each method).
        ema_model = clone_model(global_model) if ema_decay > 0 else None
        server_velocity: dict[str, torch.Tensor] = {}

        # ── Federated path ───────────────────────────────────────────────
        device_mgr = DeviceStateManager(
            client_ids,
            rng,
            dropout_rate=fl.get("dropout_rate", 0.0),
            snr_degradation_db=fl.get("snr_degradation_db", 0.0),
        )
        rep_mgr = ReputationManager(client_ids)
        selector = ClientSelector(client_ids, epicentre=epicentre, seed=_seed)

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
        target_streak = 0
        sel_counts: dict[int, int] = {cid: 0 for cid in client_ids}
        method_start_idx: int = len(all_rows)
        # Static lookup, hoisted out of the round loop.
        client_by_id = {c.client_id: c for c in clients}

        # Pre-project client coordinates once (static ground sensors): shared by
        # the per-selection V_i(t) computation. Order matches client_coord_map.
        _client_latlon = np.array([client_coord_map[c.client_id] for c in clients])
        _client_xy_m, _value_ref = latlon_to_meters(_client_latlon)
        _epi_xy_m, _ = latlon_to_meters(np.array([epicentre]), ref=_value_ref)
        _device_coords_m = np.column_stack([_client_xy_m, np.zeros(len(clients))])
        _epicentre_m = np.append(_epi_xy_m[0], 0.0)
        _samples_arr = np.array([len(c.train_indices) for c in clients], dtype=np.float64)
        # Per-client minority-information weight (Σ_c scarcity_c · count_c),
        # normalized to mean 1 so it re-weights placement value without changing
        # its overall scale. Positions UAV capacity toward rare-class-rich zones.
        # class_source="none" removes the histogram entirely, which also
        # disables class-aware placement — the two must degrade together, or a
        # "no class information" run would still be steering UAVs with it.
        #
        # Per-method copies: under class_source="pseudo" these are refreshed
        # from *this* method's global model, so they must not leak across
        # methods in the loop above.
        _cc_counts, _cc_scarcity = client_class_counts, class_scarcity
        _client_class_value = _class_value_vector(clients, _cc_counts, _cc_scarcity)

        for rnd in range(1, n_rounds + 1):
            t0 = time.perf_counter()

            # ── Pseudo-label histogram refresh ────────────────────────────
            # Refreshed from the *current* global model on the scheduled
            # reselection cadence, ahead of placement so that placement and
            # selection within a round both see the same histogram. Round 1
            # therefore runs on an untrained model — that is the honest cost of
            # removing the oracle, and warm-starting it from the true counts
            # would defeat the purpose of this rung. An early trigger
            # (low_eligible) reselects on the last scheduled histogram rather
            # than forcing an extra pass.
            if class_source == "pseudo" and (rnd - 1) % reselect_every == 0:
                _cc_counts, _cc_scarcity = build_class_info(
                    "pseudo",
                    labels=cached_dataset.labels,
                    client_indices=_client_train_idx,
                    model=global_model,
                    cached_dataset=cached_dataset,
                )
                _client_class_value = _class_value_vector(clients, _cc_counts, _cc_scarcity)

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
                # flat_fl: no UAV filter — all clients always covered. Selection
                # mode "all" still applies the device-eligibility gate, so
                # battery/SNR/memory physics (and the stress knobs) bind here too.
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
                    # Class-scarcity re-weighting (Tier C placement): bias UAV
                    # coverage toward clients carrying rare-class information, so
                    # per-UAV capacity lands where the minority classes are.
                    if placement_class_aware:
                        device_values = device_values * _client_class_value
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
                        link_model=link_model,
                        z_min_m=z_min_m,
                        z_max_m=z_max_m,
                    )
                    # Movement (repositioning) energy only — see run_tier2 note.
                    if prev_uav_pos_m is not None:
                        move_m = float(
                            np.sum(np.sqrt(np.sum((uav_pos_m - prev_uav_pos_m) ** 2, axis=1)))
                        )
                        cumulative_energy += _ENERGY_MODEL.energy_joules(move_m)
                    prev_uav_pos_m = uav_pos_m.copy()
                    uav_latlon = _uav_pos_to_latlon(uav_pos_m, ref)
                    covered_all = _covered_clients(
                        client_coord_map, uav_pos_m, ref, R_comm,
                        radii=None if uniform_coverage_radius else uav_radii
                    )
                placement_fitness = last_placement_fitness
                # Selection cadence is decoupled from the (expensive) placement
                # cadence: reselecting every round lets the roster rotate
                # smoothly under the UCB count bonus instead of the whole cohort
                # being held for T_sel rounds and then swapped wholesale — the
                # non-IID ping-pong that made the frozen-roster method the most
                # volatile of all. Placement still runs only every T_sel rounds.
                reselect = (rnd - 1) % reselect_every == 0 or low_eligible or not selected

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
                    # At the 1x default this saturates on the reselection
                    # cadence, which makes fair_mab's staleness term a constant
                    # and the baseline invariant to its own weights — see
                    class_counts=_cc_counts,  # proposed (ucb) class-coverage utility
                    class_scarcity=_cc_scarcity,
                )

            coverage_pct = 100.0 * len(covered_all) / max(len(clients), 1)
            participation_pct = 100.0 * len(selected) / max(len(clients), 1)
            n_selected = len(selected)

            # ── Build UAV groups from selection map ───────────────────────
            # Maps uav_idx → list of ClientData for clients assigned to that UAV.
            uav_groups: dict[int, list] = {j: [] for j in range(K)}
            for cid, uav_idx in selected.items():
                if cid in client_by_id:
                    uav_groups[uav_idx].append(client_by_id[cid])

            # Per-round learning-rate decay (A3).
            lr_r = lr * _lr_scale(rnd, n_rounds, lr_decay)
            uav_lr_r = uav_lr * _lr_scale(rnd, n_rounds, lr_decay)

            # ── UAV-tier training (Tier B) ────────────────────────────────
            # Each active UAV trains its owned blocks (img_proj [+ fusion]) on
            # the pooled shard of its assigned clients, with the client-owned
            # block frozen at the current global weights — vision/fusion learn
            # against the exact structured representation they deploy with,
            # rather than two tiers drifting in parallel and being glued
            # together. Uses the cached 512-dim ResNet features (no image
            # forward pass). flat_fl has no UAV tier (uav_blocks empty).
            uav_own_updates: dict[int, tuple[dict, int]] = {}
            if uav_blocks:
                for uav_idx, group in uav_groups.items():
                    uav_indices = [idx for c in group for idx in c.train_indices]
                    if not uav_indices:
                        continue
                    uav_loader = make_client_loader(
                        cached_dataset, uav_indices, batch_size, balanced=balanced_sampling
                    )
                    sd, n, _uloss = _train_blocks(
                        global_model,
                        uav_loader,
                        n_uav_epochs,
                        uav_lr_r,
                        loss_fn,
                        uav_blocks,
                        local_opt_cfg,
                        scratch=scratch_model,
                    )
                    uav_own_updates[uav_idx] = (sd, n)

            # ── Client-tier training (Tier B): owns struct_branch [+ fusion] ─
            global_owned = global_model.block_state_dict(client_blocks)
            client_updates: dict[int, tuple[dict, int, float]] = {}
            client_deltas: dict[int, dict] = {}
            client_losses: dict[int, float] = {}
            for c in clients:
                if c.client_id not in selected or not c.train_indices:
                    continue
                loader = make_client_loader(
                    cached_dataset, c.train_indices, batch_size, balanced=balanced_sampling
                )
                sd, n, mean_loss = _train_blocks(
                    global_model,
                    loader,
                    n_local_epochs,
                    lr_r,
                    loss_fn,
                    client_blocks,
                    local_opt_cfg,
                    scratch=scratch_model,
                )
                rep = rep_scores.get(c.client_id, 0.5)
                client_updates[c.client_id] = (sd, n, rep)
                client_losses[c.client_id] = mean_loss
                # Reputation scores the update *delta* Δw_n, not absolute
                # weights. One batched _foreach_sub call per client (same
                # per-key subtraction as the dict comprehension it replaces).
                ks = list(sd.keys())
                client_deltas[c.client_id] = dict(
                    zip(ks, torch._foreach_sub([sd[k] for k in ks], [global_owned[k] for k in ks]))
                )
            # Statistical-utility feed for the oort / power_of_choice baselines
            # (harmless no-op for every other mode).
            selector.update_losses(client_losses)

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

            # ── Hierarchical aggregation (client → UAV → server) ──────────
            # Client-owned block(s) flow client→UAV as data-size FedAvg within
            # each coverage zone; UAV-owned block(s) originate at the UAV. Both
            # are assembled per UAV, then combined at the server with reputation
            # weighting. A zone missing either contribution falls back to the
            # current global weights for those blocks (no double counting).
            iot_by_uav: dict[int, list[tuple[dict, int, float]]] = {}
            for cid, triple in client_updates.items():
                iot_by_uav.setdefault(selected[cid], []).append(triple)

            uav_updates: list[tuple[dict, int, float]] = []
            for uav_idx in range(K):
                iot_upds = iot_by_uav.get(uav_idx, [])
                uav_own = uav_own_updates.get(uav_idx)
                if not iot_upds and uav_own is None:
                    continue

                parts: dict = {}
                if iot_upds:
                    parts.update(fedavg([(sd, n) for sd, n, _ in iot_upds]))
                    total_n = sum(n for _, n, _ in iot_upds)
                else:
                    parts.update(global_model.block_state_dict(client_blocks))
                    total_n = uav_own[1] if uav_own else 0

                if uav_own is not None:
                    parts.update(uav_own[0])
                elif uav_blocks:
                    parts.update(global_model.block_state_dict(uav_blocks))

                # UAV reputation = trimmed mean (10% per tail) of its assigned
                # cluster's IoT reputations (paper §IV-C7).
                cluster_reps = [rep_scores.get(c.client_id, 0.5) for c in uav_groups[uav_idx]]
                uav_rep = trimmed_mean(cluster_reps) if cluster_reps else 1.0
                uav_updates.append((parts, total_n, uav_rep))

            # ── Server-level aggregation + momentum (A3) ──────────────────
            # Reputation-weighted FedAvg; UAVs whose cluster trimmed-mean
            # reputation falls below R_min are excluded this round (§IV-D). The
            # aggregate is then applied through a heavy-ball server-momentum step
            # (FedAvgM) to damp the non-IID round-to-round oscillation.
            if uav_updates:
                active = [u for u in uav_updates if u[2] >= R_min] if rep_weighted else uav_updates
                if active:
                    server_agg = (
                        reputation_fedavg(active)
                        if rep_weighted
                        else fedavg([(sd, n) for sd, n, _ in active])
                    )
                    # state_dict() (same "<block>.<param>" keys, no buffers in
                    # this model) is read-only here — the momentum step never
                    # mutates global_state — so skip full_trainable_state_dict's
                    # per-round full-model clone.
                    new_full = _apply_server_momentum(
                        global_model.state_dict(),
                        server_agg,
                        server_velocity,
                        server_momentum,
                        server_lr,
                    )
                    global_model.load_full_trainable_state_dict(new_full)

            # ── EMA of the global model for evaluation (A4) ───────────────
            # The round-100 snapshot is a lottery ticket under ±0.05/round
            # oscillation; evaluate a slow EMA of the global weights instead.
            if ema_model is not None:
                # In-place blend on the EMA params (d·ema then += (1−d)·cur —
                # the same mul/add sequence the out-of-place dict blend
                # performed, so bit-identical) instead of cloning both full
                # state dicts and load_state_dict-ing the result back every
                # round. parameters() order is aligned across the two
                # same-class instances; the model registers no buffers.
                with torch.no_grad():
                    ema_params = list(ema_model.parameters())
                    torch._foreach_mul_(ema_params, ema_decay)
                    torch._foreach_add_(
                        ema_params,
                        torch._foreach_mul(list(global_model.parameters()), 1.0 - ema_decay),
                    )
                eval_model = ema_model
            else:
                eval_model = global_model

            # ── Device state update ───────────────────────────────────────
            device_mgr.update_round(set(selected.keys()))

            # ── Evaluate ─────────────────────────────────────────────────
            # Test is the *reported* stream and is never used to select
            # anything (significance consumes the final round, not a best
            # round). Val is logged alongside it purely so hyperparameter
            # search has a legitimate stream to optimize against — before
            # 2026-08 there was no val split and scripts/tune_weights.py
            # maximized test macro-F1 directly.
            metrics = _evaluate(eval_model, cached_dataset, global_test_indices)
            val_metrics = (
                _evaluate(eval_model, cached_dataset, global_val_indices)
                if global_val_indices
                else {}
            )
            elapsed = time.perf_counter() - t0

            if metrics.get(target_metric, 0.0) >= target_value:
                target_streak += 1
                if rounds_to_target is None and target_streak >= _TARGET_CONSEC_ROUNDS:
                    rounds_to_target = rnd - _TARGET_CONSEC_ROUNDS + 1
            else:
                target_streak = 0

            # Communication cost: shared accounting rule in metrics.fl.
            # flat_fl (placement_method None): IoT↔server directly, IoT payload only.
            if placement_method is None:
                comm_mb = round_comm_mb(n_selected)
            else:
                comm_mb = round_comm_mb(n_selected, n_active_uavs=len(uav_own_updates))

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
                    # val_* mirrors the reported metrics on the validation
                    # split (NaN when no val split is configured). Anything
                    # that *selects* — hyperparameter search, early stopping —
                    # must read these, never the unprefixed test columns.
                    "val_accuracy": val_metrics.get("accuracy", float("nan")),
                    "val_macro_f1": val_metrics.get("macro_f1", float("nan")),
                    **{
                        f"val_f1_{cls}": v
                        for cls, v in val_metrics.get("f1_per_class", {}).items()
                    },
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
