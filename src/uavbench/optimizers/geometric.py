"""Deterministic geometric placement baselines from the UAV literature.

Two families that between them cover how non-metaheuristic placement is usually
done, so the candidate-set method in :mod:`.mclp_ls` is not being compared only
against swarms:

* :class:`SpiralPlacement` — successive boundary-anchored disc placement
  (Lyu, Zeng, Zhang & Lim, "Placement Optimization of UAV-Mounted Mobile Base
  Stations", IEEE Communications Letters 21(3), 2017).
* :class:`CapacitatedKMeans` — clustering with a hard per-cluster size limit,
  the natural fix to the ``centroid`` baseline's blindness to capacity.

Adaptation notes (same convention as :mod:`.mozaffari2016`: state the deviation,
do not quietly reproduce something else under the paper's name).
"""

from __future__ import annotations

import numpy as np

from ..problem.fitness import Fitness
from ..problem.instance import ProblemInstance
from .altitude import optimal_shared_altitude, optimize_altitudes
from .base import Optimizer, Result
from .candidates import circle_intersection_points
from .seeding import kmeanspp_centers

_EPS = 1e-12


class SpiralPlacement(Optimizer):
    """Boundary-inward successive disc placement (Lyu et al. 2017), K-capped.

    Adaptation, stated for the paper's baseline-methodology section:

    * Lyu et al. solve the *minimum-UAV* problem — spiral inward until every
      terminal is covered, and report how many discs that took. Here ``K`` is
      fixed by the fleet, so the spiral is truncated at ``K`` discs. Under a
      fleet too small to cover everything this becomes a coverage heuristic
      rather than a feasibility one, which is the regime the benchmark runs in.
      This truncation is forced by the benchmark, not a choice.
    * The per-step disc is chosen among those of radius ``r`` whose boundary
      passes through the anchor terminal — the family the paper's
      smallest-enclosing-circle construction searches — maximizing the **count**
      of newly covered terminals, as published. It is deliberately *not*
      value-weighted: Lyu et al. treat all ground terminals alike, and weighting
      by this benchmark's per-device value would be a different algorithm
      wearing the paper's name. That the benchmark's objective *is*
      value-weighted while this baseline optimizes an unweighted count is a
      property of the baseline, faithfully represented.
    * Altitude is unspecified by the paper, which works in the plane at an
      assumed coverage radius ``r``. The altitude that delivers the largest such
      ``r`` under the shared channel is used, which is the reading that leaves
      the published 2D rule untouched.
    """

    name = "spiral"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)

    def _run(self, instance: ProblemInstance, fitness: Fitness, rng: np.random.Generator) -> Result:
        device_xy = instance.device_coords[:, :2]
        z, r = optimal_shared_altitude(instance, getattr(fitness, "link", None))

        residual = np.ones(instance.N, dtype=bool)
        centers: list[np.ndarray] = []

        for _ in range(instance.K):
            if not residual.any():
                centers.append(centers[-1] if centers else device_xy.mean(axis=0))
                continue

            # Spiral anchor: the residual terminal farthest from the residual
            # centroid, i.e. the current boundary. Working inward from the
            # boundary is the whole point of the spiral rule — an interior
            # anchor leaves stragglers that each need their own disc later.
            pts = device_xy[residual]
            anchor_local = int(np.argmax(np.sum((pts - pts.mean(axis=0)) ** 2, axis=1)))
            anchor = pts[anchor_local]

            # Discs of radius r whose boundary passes through the anchor: their
            # centres lie on the circle of radius r about it. Intersecting that
            # circle with the circles about nearby residual terminals gives the
            # combinatorially distinct choices; the anchor itself covers the
            # degenerate case of an isolated terminal.
            near = pts[np.sum((pts - anchor) ** 2, axis=1) <= (2.0 * r) ** 2]
            cand = np.vstack([anchor[None, :], circle_intersection_points(near, r)])
            cand = cand[np.sum((cand - anchor) ** 2, axis=1) <= r * r + 1e-6]
            if cand.shape[0] == 0:
                cand = anchor[None, :]

            dx = cand[:, None, 0] - device_xy[None, residual, 0]
            dy = cand[:, None, 1] - device_xy[None, residual, 1]
            covered = (dx * dx + dy * dy) <= r * r  # (C, n_residual)
            gain = covered.sum(axis=1)  # count, as published — not value-weighted
            b = int(np.argmax(gain))

            centers.append(cand[b])
            hit = np.zeros(instance.N, dtype=bool)
            hit[np.flatnonzero(residual)[covered[b]]] = True
            residual &= ~hit

        positions = np.column_stack([np.array(centers), np.full(instance.K, z)])
        x = positions.reshape(instance.dim)
        f = fitness(x)
        return Result(
            method=self.name,
            best_position=x,
            best_fitness=f,
            convergence=[f],
            n_iterations=1,
            meta={"altitude_m": float(positions[:, 2].mean()), "r_eff_m": r},
        )


class CapacitatedKMeans(Optimizer):
    """Lloyd's algorithm with a hard per-cluster capacity.

    ``centroid`` places each UAV at a value-weighted k-means centre and is blind
    to capacity: on a clustered layout it stacks centres on the dense region
    where most of the demand cannot be served anyway. This enforces the limit
    inside the assignment step — devices claim their nearest UAV in descending
    value order and a full UAV stops accepting — so the centres migrate toward a
    partition the fleet can actually serve.

    Adaptation: the exact capacitated k-means of Bradley, Bennett & Demiriz
    (2000) solves a min-cost-flow assignment each iteration. The greedy
    value-ordered assignment used here is the standard cheap surrogate and,
    importantly, is the *same* rule the shared :class:`Fitness` scores with, so
    the clustering optimises the assignment it will be graded on.
    """

    name = "cap_kmeans"

    def __init__(
        self, n_iter: int = 30, altitude_frac: float = 0.0, optimize_z: bool = True, **kw
    ) -> None:
        super().__init__(**kw)
        self.n_iter = n_iter
        self.altitude_frac = altitude_frac
        self.optimize_z = optimize_z

    def _run(self, instance: ProblemInstance, fitness: Fitness, rng: np.random.Generator) -> Result:
        device_xy = instance.device_coords[:, :2]
        value = instance.value
        K = instance.K
        z = instance.lower[2] + self.altitude_frac * (instance.upper[2] - instance.lower[2])

        centers = kmeanspp_centers(rng, device_xy, K, value)
        order = instance.value_order

        for _ in range(self.n_iter):
            d = np.sqrt(
                (device_xy[:, None, 0] - centers[None, :, 0]) ** 2
                + (device_xy[:, None, 1] - centers[None, :, 1]) ** 2
            )
            loads = np.zeros(K, dtype=np.int64)
            labels = np.full(instance.N, -1, dtype=np.int64)
            for i in order:
                open_k = np.flatnonzero(loads < instance.capacity)
                if open_k.size == 0:
                    break
                j = int(open_k[np.argmin(d[i, open_k])])
                labels[i] = j
                loads[j] += 1

            new_centers = centers.copy()
            for k in range(K):
                mask = labels == k
                w = value[mask]
                if w.sum() > 0:
                    new_centers[k] = (device_xy[mask] * w[:, None]).sum(axis=0) / w.sum()
            if np.allclose(new_centers, centers):
                break
            centers = new_centers

        positions = np.column_stack([centers, np.full(K, z)])
        if self.optimize_z:
            positions = optimize_altitudes(instance, fitness, positions)
        x = positions.reshape(instance.dim)
        f = fitness(x)
        return Result(
            method=self.name,
            best_position=x,
            best_fitness=f,
            convergence=[f],
            n_iterations=1,
            meta={"altitude_m": float(positions[:, 2].mean())},
        )
