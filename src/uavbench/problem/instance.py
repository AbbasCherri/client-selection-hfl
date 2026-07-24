"""Problem instance definition and the scenario generator.

Coordinate convention
---------------------
All coordinates are stored internally in a **projected metric** system (meters):
``x`` and ``y`` span the disaster-area bounding box, ``z`` is altitude in
``[z_min, z_max]``. Ground IoT devices sit at ``z = 0``.

3D distance used by the range gate and the movement cost is the Euclidean
distance in this projected space. For geographic (lat/lon) inputs, convert once
with :func:`latlon_to_meters`, which applies an equirectangular projection about
a reference point so that horizontal distance matches the great-circle
(Haversine) distance to within a fraction of a percent over the <30 km zones
considered here; the vertical altitude difference is then combined in quadrature.
This mirrors the Haversine-horizontal + vertical-difference definition used by
the parent project's ``simulation.py`` while keeping the hot path vectorizable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from hflsim.shared.coords import latlon_to_meters  # noqa: F401 — re-exported for callers

from .value import compute_value

_EARTH_RADIUS_M = 6_371_000.0

Distribution = Literal["uniform", "clustered", "epicenter_biased", "real"]

# Noto Peninsula 2024 earthquake epicentre (lat °N, lon °E) — used to anchor the
# value model's proximity term on the real-geometry scenario.
_NOTO_EPICENTRE = (37.488, 137.272)


@dataclass
class ProblemInstance:
    """One placement problem: a cluster of devices plus K hover-position slots.

    Attributes
    ----------
    device_coords:
        ``(N, 3)`` device positions in projected meters (z = 0 for ground IoT).
    value:
        ``(N,)`` fixed per-device value score ``V_i(t)`` (see :mod:`.value`).
    capacity:
        ``(K,)`` max devices each position may serve.
    battery:
        ``(K,)`` UAV battery fraction per position in ``[0, 1]``.
    prev_positions:
        ``(K, 3)`` previous UAV locations, used by the movement penalty.
    lower / upper:
        ``(3,)`` per-dimension search bounds ``[L_d, U_d]``.
    R_comm:
        UAV-IoT communication range (meters) — a feasibility gate.
    B_min_uav:
        Minimum UAV battery for a position to be usable.
    """

    device_coords: np.ndarray
    value: np.ndarray
    capacity: np.ndarray
    battery: np.ndarray
    prev_positions: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    R_comm: float = 500.0
    B_min_uav: float = 0.2
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.device_coords = np.asarray(self.device_coords, dtype=np.float64)
        self.value = np.asarray(self.value, dtype=np.float64)
        self.capacity = np.asarray(self.capacity, dtype=np.float64)
        self.battery = np.asarray(self.battery, dtype=np.float64)
        self.prev_positions = np.asarray(self.prev_positions, dtype=np.float64)
        self.lower = np.asarray(self.lower, dtype=np.float64)
        self.upper = np.asarray(self.upper, dtype=np.float64)

    @property
    def N(self) -> int:
        """Number of devices."""
        return int(self.device_coords.shape[0])

    @property
    def K(self) -> int:
        """Number of hover positions."""
        return int(self.capacity.shape[0])

    @property
    def dim(self) -> int:
        """Search-space dimension (3K)."""
        return 3 * self.K

    @property
    def box_diagonal(self) -> float:
        """Diagonal of the 3D search box (used to normalize movement cost)."""
        span = self.upper - self.lower
        return float(np.sqrt(np.sum(span * span)))

    def positions_from_vector(self, x: np.ndarray) -> np.ndarray:
        """Reshape a flat ``3K`` particle vector to ``(K, 3)`` positions."""
        return np.asarray(x, dtype=np.float64).reshape(self.K, 3)

    def distances(self, positions: np.ndarray) -> np.ndarray:
        """Return the ``(N, K)`` 3D Euclidean device-to-position distance matrix."""
        positions = np.asarray(positions, dtype=np.float64).reshape(self.K, 3)
        diff = self.device_coords[:, None, :] - positions[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2))


def _sample_devices(
    rng: np.random.Generator,
    distribution: Distribution,
    N: int,
    box_xy: np.ndarray,
    epicenter: np.ndarray,
) -> np.ndarray:
    """Sample N ground-device (x, y) positions for the chosen spatial geometry."""
    lo, hi = box_xy[0], box_xy[1]
    span = hi - lo
    if distribution == "uniform":
        xy = rng.uniform(lo, hi, size=(N, 2))
    elif distribution == "clustered":
        n_clusters = max(2, int(np.sqrt(N) / 2))
        centers = rng.uniform(lo, hi, size=(n_clusters, 2))
        sigma = 0.06 * span
        idx = rng.integers(0, n_clusters, size=N)
        xy = centers[idx] + rng.normal(0.0, sigma, size=(N, 2))
        xy = np.clip(xy, lo, hi)
    elif distribution == "epicenter_biased":
        sigma = 0.12 * span
        xy = epicenter + rng.normal(0.0, sigma, size=(N, 2))
        xy = np.clip(xy, lo, hi)
    else:  # pragma: no cover - guarded by Literal typing
        raise ValueError(f"unknown distribution: {distribution!r}")
    return xy


def _load_real_latlon(data_dir: str) -> np.ndarray:
    """Load (M, 2) lat/lon from the largest cached Noto metadata parquet."""
    from pathlib import Path

    caches = sorted(
        Path(data_dir).glob(".metadata_df_cache_sub*_seed*.parquet"),
        key=lambda p: p.stat().st_size,
    )
    if not caches:
        raise FileNotFoundError(
            f"no .metadata_df_cache_sub*.parquet under {data_dir!r}; the real-geometry "
            "scenario needs the cached Noto metadata (build it once via the FL data loader)."
        )
    import pandas as pd

    df = pd.read_parquet(caches[-1], columns=["latitude", "longitude"])
    return df[["latitude", "longitude"]].to_numpy(dtype=np.float64)


def _real_device_xy(N: int, box_xy: np.ndarray, seed: int, data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """N real client (x, y) centroids, normalized into ``box_xy``.

    K-means-clusters the real Noto sample coordinates into N geographic clients
    (the same partition convention the FL harness uses), projects lat/lon to
    metres, then min-max normalizes the layout into the benchmark box so R_comm
    and K stay comparable to the synthetic scenarios — only the *spatial
    structure* is real. Returns ``(client_xy, epicenter_xy)``.
    """
    from pathlib import Path

    from hflsim.shared.coords import latlon_to_meters

    # Cache the expensive K-means centres per (N, seed); projection/normalization
    # below are cheap and box-dependent so they stay out of the cache.
    cache = Path(data_dir) / f".tier1_real_centers_N{N}_seed{seed}.npy"
    if cache.exists():
        centers_latlon = np.load(cache)
    else:
        from sklearn.cluster import KMeans

        latlon = _load_real_latlon(data_dir)
        centers_latlon = KMeans(n_clusters=N, n_init=10, random_state=seed).fit(latlon).cluster_centers_
        try:
            np.save(cache, centers_latlon)
        except OSError:
            pass
    # project client centroids + the epicentre in one frame, then normalize together
    pts_latlon = np.vstack([centers_latlon, np.array([_NOTO_EPICENTRE])])
    xy_m, _ = latlon_to_meters(pts_latlon)
    lo, hi = box_xy[0], box_xy[1]
    mn, mx = xy_m.min(axis=0), xy_m.max(axis=0)
    span = np.where((mx - mn) > 0, mx - mn, 1.0)
    xy_norm = lo + (xy_m - mn) / span * (hi - lo)
    return xy_norm[:-1], xy_norm[-1]


def generate_instance(
    distribution: Distribution,
    N: int,
    K: int,
    area: dict,
    *,
    seed: int,
    capacity: float = 20.0,
    uav_battery: float = 1.0,
    R_comm: float = 500.0,
    B_min_uav: float = 0.2,
    beta_mode: str = "scheduled",
    t: int = 0,
    T_decay: int = 20,
    prev_mode: str = "stale",
    capacity_cv: float = 0.0,
    battery_cv: float = 0.0,
    data_dir: str = "./data",
) -> ProblemInstance:
    """Deterministically generate a placement instance from a seed.

    The seed governs *only* instance generation; optimizer stochasticity uses a
    separate stream so methods are compared on identical instances (Section 9 of
    the simulation plan).

    ``prev_mode`` controls the previous UAV layout that the movement penalty is
    measured against:

    * ``"stale"`` (default) — a displaced layout from before the reconfiguration
      trigger (positions drawn near a shifted epicenter). This models *why* a
      trigger fires: the situation changed and the old layout no longer fits, so
      repositioning can pay off and ``static`` is a genuine floor.
    * ``"warm"`` — positions at current device sub-centroids (already near
      optimal). Useful to study the conservative regime where holding position is
      nearly best and a good optimizer should barely move.

    ``capacity_cv`` / ``battery_cv`` make the UAV fleet heterogeneous: each
    per-UAV capacity (resp. battery) is drawn uniformly in ``mean * (1 +/- cv)``
    around the scalar mean, so placement must co-optimize *which* UAV (large vs
    small) serves *which* region — a joint problem greedy physics placement
    cannot see. Both default to ``0.0`` (homogeneous fleet), and the draw
    happens only when ``cv > 0`` and only *after* every other RNG draw, so a
    ``cv=0`` instance is byte-identical to before and a heterogeneous instance
    shares the exact device layout of its homogeneous twin at the same seed.
    """
    rng = np.random.default_rng(seed)
    box_xy = np.array(
        [[area["x"][0], area["y"][0]], [area["x"][1], area["y"][1]]], dtype=np.float64
    )
    z_lo, z_hi = float(area["z"][0]), float(area["z"][1])

    if distribution == "real":
        xy, epicenter = _real_device_xy(N, box_xy, seed, data_dir)
    else:
        epicenter = rng.uniform(box_xy[0], box_xy[1])
        xy = _sample_devices(rng, distribution, N, box_xy, epicenter)
    device_coords = np.column_stack([xy, np.zeros(N)])

    # Raw per-device features feeding the utility/value score.
    snr = rng.uniform(0.0, 30.0, size=N)  # dB, spans the 3 dB eligibility gate
    samples = rng.integers(20, 200, size=N).astype(np.float64)
    reputation = rng.beta(2.0, 2.0, size=N)  # synthesized history, held fixed

    # Previous UAV layout the movement penalty is measured against.
    prev_z = np.full(K, 0.5 * (z_lo + z_hi))
    if prev_mode == "warm":
        prev_idx = rng.choice(N, size=K, replace=N < K)
        prev_xy = device_coords[prev_idx, :2]
    elif prev_mode == "stale":
        # Layout fitted to a *prior* state: devices around a shifted epicenter.
        span = box_xy[1] - box_xy[0]
        old_epicenter = np.clip(
            epicenter + rng.normal(0.0, 0.25 * span, size=2), box_xy[0], box_xy[1]
        )
        # For the real-geometry scenario the prior layout is a displaced
        # epicentre-biased cluster (real coords aren't re-sampleable); synthetic
        # scenarios keep their own distribution shape (byte-identical to before).
        old_dist = "epicenter_biased" if distribution == "real" else distribution
        old_xy = _sample_devices(rng, old_dist, max(K, 8), box_xy, old_epicenter)
        prev_centers = old_xy[rng.choice(old_xy.shape[0], size=K, replace=old_xy.shape[0] < K)]
        prev_xy = prev_centers[:, :2]
    else:
        raise ValueError(f"unknown prev_mode: {prev_mode!r}")
    prev_positions = np.column_stack([prev_xy, prev_z])

    value = compute_value(
        device_coords=device_coords,
        epicenter=np.append(epicenter, 0.0),
        snr=snr,
        samples=samples,
        prev_positions=prev_positions,
        reputation=reputation,
        t=t,
        T_decay=T_decay,
        beta_mode=beta_mode,
        R_comm=R_comm,
    )

    lower = np.array([area["x"][0], area["y"][0], z_lo], dtype=np.float64)
    upper = np.array([area["x"][1], area["y"][1], z_hi], dtype=np.float64)

    # Per-UAV capacity/battery. Drawn last so cv=0 stays byte-identical and a
    # heterogeneous instance shares its homogeneous twin's device layout.
    if capacity_cv > 0.0:
        caps = rng.uniform(capacity * (1.0 - capacity_cv), capacity * (1.0 + capacity_cv), size=K)
        cap_arr = np.maximum(np.rint(caps), 1.0)
    else:
        cap_arr = np.full(K, float(capacity))
    if battery_cv > 0.0:
        bats = rng.uniform(uav_battery * (1.0 - battery_cv), uav_battery * (1.0 + battery_cv), size=K)
        bat_arr = np.clip(bats, B_min_uav + 1e-6, 1.0)
    else:
        bat_arr = np.full(K, float(uav_battery))

    return ProblemInstance(
        device_coords=device_coords,
        value=value,
        capacity=cap_arr,
        battery=bat_arr,
        prev_positions=prev_positions,
        lower=lower,
        upper=upper,
        R_comm=R_comm,
        B_min_uav=B_min_uav,
        meta={
            "distribution": distribution,
            "N": N,
            "K": K,
            "seed": seed,
            "epicenter": epicenter.tolist(),
            "beta_mode": beta_mode,
        },
    )
