"""Post-2020 UAV placement baselines.

The 2016-17 classics (Mozaffari, Al-Zenad, Lyu) are the ones everyone cites, but
beating only those invites the obvious objection that the field has moved on.
These two cover what recent placement work actually does, which is overwhelmingly
*cluster-then-place* rather than direct search over UAV coordinates.

Provenance, stated separately for each because it differs:

* :class:`PSOClusterPlacement` — Sawalmeh, Othman, Shakhatreh, Khreishah et al.,
  "Power-Efficient Wireless Coverage Using Minimum Number of UAVs", *Sensors*
  22(1):223, 2021 (open access). The algorithm was read from the source: users
  are partitioned by a **PSO-based clustering** step whose particles are the
  cluster centres and whose fitness is the user-to-centre distance sum, then each
  UAV is placed on its cluster. The paper reports this beating K-means clustering
  and running faster than GA- and ABC-based variants.

* :class:`HierarchicalPlacement` — agglomerative hierarchical clustering with a
  smallest-enclosing-circle centre per cluster. This one is a **family
  representative, not a reproduction of a specific paper**: AHC-based aerial base
  station placement recurs across recent work, but the papers stating it in full
  are paywalled and were not read, so no single citation is claimed for it. It is
  included because it is the strongest *deterministic* clustering placement, and
  because Ward linkage optimizes a different criterion (within-cluster variance)
  from k-means' — which is the point of having it.

Both illustrate the same structural weakness, and it is the reason
:mod:`.mclp_ls` exists: they optimize a **proxy** (compactness) and then hope
compact clusters are coverable. Coverage under a hard range gate is not a
compactness objective — a tight cluster wider than 2r cannot be covered by one
disc no matter how well centred, and a loose cluster that happens to fit inside
one disc is covered perfectly.
"""

from __future__ import annotations

import numpy as np

from ..problem.fitness import Fitness
from ..problem.instance import ProblemInstance
from .altitude import optimize_altitudes
from .base import Optimizer, Result

_EPS = 1e-12


def smallest_enclosing_circle_center(points: np.ndarray, n_iter: int = 64) -> np.ndarray:
    """Approximate the 1-centre (minimax-distance point) of ``points``.

    Bădoiu-Clarkson core-set iteration: start at the centroid and repeatedly step
    a shrinking fraction of the way toward the currently farthest point. It
    converges to within ``(1 + eps)`` of the true smallest enclosing circle and
    needs no convex-hull machinery.

    The centroid is the wrong centre for a coverage disc — it minimizes *mean*
    distance while coverage is decided by the *maximum*, so a centroid-placed
    disc drops the outlying members of an elongated cluster that a minimax centre
    would hold.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return np.zeros(2)
    c = pts.mean(axis=0)
    for i in range(1, n_iter + 1):
        far = pts[np.argmax(np.sum((pts - c) ** 2, axis=1))]
        c = c + (far - c) / (i + 1)
    return c


class PSOClusterPlacement(Optimizer):
    """PSO over cluster centres, then one UAV per cluster (Sawalmeh et al. 2021).

    The distinction from :class:`~.pso.PSO` is the search space, and it is not
    cosmetic: this searches ``2K`` cluster-centre coordinates against a clustering
    objective, whereas ``pso`` searches ``3K`` UAV coordinates against the
    deployment objective itself. The lower-dimensional space is easier to search
    — which is the paper's argument — but the objective it searches is a proxy.

    ``objective="shared"`` swaps the clustering fitness for this benchmark's
    actual objective while keeping the centre-space encoding. That variant is not
    the published method; it exists so a loss can be attributed to the *encoding*
    rather than dismissed as the baseline having been scored on the wrong goal.
    """

    name = "pso_cluster"

    def __init__(
        self,
        P: int = 100,
        G_max: int = 200,
        objective: str = "distance",  # "distance" (as published) | "shared"
        inertia: float = 0.7298,
        c1: float = 1.4962,
        c2: float = 1.4962,
        altitude_frac: float = 0.0,
        optimize_z: bool = True,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.P, self.G_max = P, G_max
        if objective not in ("distance", "shared"):
            raise ValueError(f"objective must be 'distance' or 'shared'; got {objective!r}")
        self.objective = objective
        self.inertia, self.c1, self.c2 = inertia, c1, c2
        self.altitude_frac = altitude_frac
        # The source optimizes altitude against a transmit-power model this
        # benchmark does not carry, so altitude is re-derived here against the
        # shared objective rather than pinned. See :mod:`.altitude`.
        self.optimize_z = optimize_z

    def _cluster_cost(self, centers: np.ndarray, device_xy: np.ndarray) -> np.ndarray:
        """``(P,)`` sum of each user's distance to its nearest centre.

        The published clustering fitness. Unweighted by device value, as in the
        source — the method has no notion of per-user value.
        """
        c = centers.reshape(centers.shape[0], -1, 2)  # (P, K, 2)
        dx = device_xy[None, :, 0, None] - c[:, None, :, 0]
        dy = device_xy[None, :, 1, None] - c[:, None, :, 1]
        return np.sqrt(dx * dx + dy * dy).min(axis=2).sum(axis=1)

    def _run(self, instance: ProblemInstance, fitness: Fitness, rng: np.random.Generator) -> Result:
        K = instance.K
        device_xy = instance.device_coords[:, :2]
        z = instance.lower[2] + self.altitude_frac * (instance.upper[2] - instance.lower[2])
        lo = np.tile(instance.lower[:2], K)
        hi = np.tile(instance.upper[:2], K)

        def to_positions(flat: np.ndarray) -> np.ndarray:
            return np.column_stack([flat.reshape(K, 2), np.full(K, z)]).reshape(3 * K)

        def score(X: np.ndarray) -> np.ndarray:
            if self.objective == "distance":
                return -self._cluster_cost(X, device_xy)  # maximization convention
            return fitness.batch(np.array([to_positions(x) for x in X]))

        X = rng.uniform(lo, hi, size=(self.P, 2 * K))
        vmax = 0.2 * (hi - lo)
        V = rng.uniform(-vmax, vmax, size=(self.P, 2 * K))

        pbest, pbest_fit = X.copy(), score(X)
        g = int(pbest_fit.argmax())
        gbest, gbest_fit = pbest[g].copy(), float(pbest_fit[g])
        convergence = [float(fitness(to_positions(gbest)))]

        for _ in range(self.G_max):
            r1 = rng.random((self.P, 2 * K))
            r2 = rng.random((self.P, 2 * K))
            V = (
                self.inertia * V
                + self.c1 * r1 * (pbest - X)
                + self.c2 * r2 * (gbest[None, :] - X)
            )
            np.clip(V, -vmax, vmax, out=V)
            X = np.clip(X + V, lo, hi)

            fit = score(X)
            improved = fit > pbest_fit
            pbest[improved] = X[improved]
            pbest_fit[improved] = fit[improved]
            g = int(pbest_fit.argmax())
            if pbest_fit[g] > gbest_fit:
                gbest_fit = float(pbest_fit[g])
                gbest = pbest[g].copy()
            convergence.append(convergence[-1])

        # The clustering objective is not the deployment objective, so the score
        # reported must be the shared one — otherwise this method's column would
        # be a distance sum sitting in a fitness table.
        positions = np.asarray(to_positions(gbest)).reshape(K, 3)
        if self.optimize_z:
            positions = optimize_altitudes(instance, fitness, positions)
        x = positions.reshape(instance.dim)
        f = float(fitness(x))
        convergence[-1] = f
        return Result(
            method=self.name,
            best_position=x,
            best_fitness=f,
            convergence=convergence,
            n_iterations=self.G_max,
            meta={"objective": self.objective, "altitude_m": float(positions[:, 2].mean())},
        )


class HierarchicalPlacement(Optimizer):
    """Ward agglomerative clustering, one UAV at each cluster's minimax centre."""

    name = "ahc"

    def __init__(
        self,
        altitude_frac: float = 0.0,
        linkage_method: str = "ward",
        optimize_z: bool = True,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.altitude_frac = altitude_frac
        self.linkage_method = linkage_method
        self.optimize_z = optimize_z

    def _run(self, instance: ProblemInstance, fitness: Fitness, rng: np.random.Generator) -> Result:
        from scipy.cluster.hierarchy import fcluster, linkage

        device_xy = instance.device_coords[:, :2]
        K = instance.K
        z = instance.lower[2] + self.altitude_frac * (instance.upper[2] - instance.lower[2])

        if device_xy.shape[0] <= K:
            centers = np.resize(device_xy, (K, 2))
        else:
            Z = linkage(device_xy, method=self.linkage_method)
            labels = fcluster(Z, t=K, criterion="maxclust")
            centers = np.empty((K, 2), dtype=np.float64)
            uniq = np.unique(labels)
            for i in range(K):
                # fcluster can return fewer than K clusters when points coincide;
                # spare UAVs go to the largest cluster's centre rather than being
                # emitted at an arbitrary coordinate.
                lab = uniq[i] if i < uniq.size else uniq[np.argmax(np.bincount(labels)[uniq])]
                centers[i] = smallest_enclosing_circle_center(device_xy[labels == lab])

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
            meta={
                "altitude_m": float(positions[:, 2].mean()),
                "linkage": self.linkage_method,
            },
        )
