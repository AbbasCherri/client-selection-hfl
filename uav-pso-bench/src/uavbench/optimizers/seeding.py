"""Value-weighted k-means++ seeding shared by PSO and the centroid baseline."""

from __future__ import annotations

import numpy as np


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
