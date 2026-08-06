"""Value-weighted k-means++ seeding shared by PSO and the centroid baseline."""

from __future__ import annotations

import numpy as np

from ..problem.instance import ProblemInstance


def kmeanspp_centers(
    rng: np.random.Generator,
    points: np.ndarray,
    K: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """k-means++ initial centers, with sampling probability optionally ``∝ weights``.

    Parameters
    ----------
    points:
        ``(N, D)`` data points (here device x,y).
    K:
        Number of centers.
    weights:
        ``(N,)`` non-negative weights (device value). When given, both the first
        pick and the D^2 distance weighting are scaled by value, pulling centers
        toward high-value devices.
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if weights is None:
        weights = np.ones(n)
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 0.0, None)

    centers = np.empty((K, points.shape[1]), dtype=np.float64)

    # First center: sample ∝ value.
    p0 = weights / (weights.sum() + 1e-12)
    if not np.isfinite(p0).all() or p0.sum() <= 0:
        p0 = np.full(n, 1.0 / n)
    centers[0] = points[rng.choice(n, p=p0)]

    closest_sq = np.sum((points - centers[0]) ** 2, axis=1)
    for k in range(1, K):
        prob = closest_sq * weights
        total = prob.sum()
        if total <= 0 or not np.isfinite(total):
            centers[k] = points[rng.integers(n)]
        else:
            centers[k] = points[rng.choice(n, p=prob / total)]
        new_sq = np.sum((points - centers[k]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)
    return centers


def seeded_population(
    rng: np.random.Generator,
    instance: ProblemInstance,
    P: int,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    seeding: str = "value_kmeans",
    jitter_m: float = 10.0,
) -> np.ndarray:
    """``(P, 3K)`` initial population: half k-means++ seeds, half uniform.

    This is PSO's initialization rule, factored out so the alternative
    metaheuristics (DE, GWO) start from the *identical* distribution. Without
    that, "DE beats PSO" would be confounded by initialization and would say
    nothing about the search dynamics, which is the only thing those baselines
    exist to isolate.

    :meth:`PSO._init_positions` deliberately keeps its own copy of this logic
    rather than calling here: every published PSO number depends on its exact RNG
    draw order, and ``check_pso_plus.py`` asserts bit-identity against it.
    """
    dim, K = instance.dim, instance.K
    if seeding == "uniform":
        return rng.uniform(lo, hi, size=(P, dim))

    n_seed = P // 2
    device_xy = instance.device_coords[:, :2]
    weights = instance.value if seeding == "value_kmeans" else None
    z_lo, z_hi = instance.lower[2], instance.upper[2]

    seeded = np.empty((n_seed, dim), dtype=np.float64)
    for p in range(n_seed):
        centers = kmeanspp_centers(rng, device_xy, K, weights)
        xy = centers + rng.normal(0.0, jitter_m, size=(K, 2))
        z = rng.uniform(z_lo, z_hi, size=(K, 1))
        seeded[p] = np.column_stack([xy, z]).reshape(dim)

    seeded = np.clip(seeded, lo, hi)
    uniform = rng.uniform(lo, hi, size=(P - n_seed, dim))
    return np.vstack([seeded, uniform])


def weighted_kmeans(
    rng: np.random.Generator,
    points: np.ndarray,
    K: int,
    weights: np.ndarray | None = None,
    n_iter: int = 25,
) -> np.ndarray:
    """Lloyd's algorithm with weighted centroids; returns ``(K, D)`` centers."""
    points = np.asarray(points, dtype=np.float64)
    if weights is None:
        weights = np.ones(points.shape[0])
    weights = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)

    centers = kmeanspp_centers(rng, points, K, weights)
    for _ in range(n_iter):
        d = np.sqrt(((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
        labels = d.argmin(axis=1)
        new_centers = centers.copy()
        for k in range(K):
            mask = labels == k
            w = weights[mask]
            if w.sum() > 0:
                new_centers[k] = (points[mask] * w[:, None]).sum(axis=0) / w.sum()
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers
