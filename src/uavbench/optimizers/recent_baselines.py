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
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.P, self.G_max = P, G_max
        if objective not in ("distance", "shared"):
            raise ValueError(f"objective must be 'distance' or 'shared'; got {objective!r}")
        self.objective = objective
        self.inertia, self.c1, self.c2 = inertia, c1, c2

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
        link = getattr(fitness, "link", None)
        lo = np.tile(instance.lower[:2], K)
        hi = np.tile(instance.upper[:2], K)

        def cluster_altitudes(centers_xy: np.ndarray) -> np.ndarray:
            """Per-cluster altitude minimizing the transmit power it needs.

            The paper's altitude stage minimizes required transmit power subject
            to serving the cluster. This benchmark has no power model, so the
            equivalent under the shared channel is the lowest altitude whose
            coverage radius reaches the cluster's farthest member — least link
            margin spent, same rule. That keeps the published two-stage
            structure (cluster, then place) intact rather than substituting a
            sweep of this benchmark's objective, which the paper does not do.
            """
            z_lo, z_hi = float(instance.lower[2]), float(instance.upper[2])
            if link is None:
                return np.full(K, z_lo)
            d = np.sqrt(
                (device_xy[:, None, 0] - centers_xy[None, :, 0]) ** 2
                + (device_xy[:, None, 1] - centers_xy[None, :, 1]) ** 2
            )
            owner = d.argmin(axis=1)
            out = np.empty(K)
            for k in range(K):
                members = d[owner == k, k]
                req = float(members.max()) if members.size else 1.0
                out[k] = link.min_altitude_for_radius(max(req, 1.0))
            return np.clip(out, z_lo, z_hi)

        def to_positions(flat: np.ndarray) -> np.ndarray:
            c = flat.reshape(K, 2)
            return np.column_stack([c, cluster_altitudes(c)]).reshape(3 * K)

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


class Moon2022Priority(Optimizer):
    """Priority-aware maximal coverage (Moon, Dung & Kim, Electronics 11(7):1036, 2022).

    Verified against the source PDF. The paper's Algorithm 1:

        1. theta_opt from Eq. (9)  — environment only
        2. R_max from Eq. (11), the coverage radius at theta_opt
        3. h_opt = R_max * tan(theta_opt)
        4. Eq. (16): place the disc to maximize ``sum(u_high) + sum(u_low)/(M+N)``

    Step 4 is the reason this baseline is here: it is the only published method
    among these that optimizes **weighted** coverage, which is what this
    benchmark's objective does. The weighting is lexicographic — the low-priority
    term is bounded below 1, so it can only ever break ties among placements
    covering equally many high-priority nodes.

    Adaptations, stated:

    * **Multi-UAV.** The paper places one UAV-BS. Here ``K`` are placed
      sequentially over the residual device set, as with :class:`.Alzenad2017`.
    * **Priority split.** The paper is given two labelled user classes. This
      benchmark has a continuous per-device value, so devices above the median
      value are treated as high priority and the rest as low. That is the
      closest faithful reading; it deliberately does *not* hand the method the
      full continuous value vector, which would be a different algorithm.
    * Step 4 is solved exactly over the circle-intersection candidate set, which
      provably contains an optimal centre — the paper solves the same problem to
      optimality with MOSEK.
    """

    name = "moon2022"

    def __init__(self, environment: str = "suburban", **kw) -> None:
        super().__init__(**kw)
        self.environment = environment

    def _run(self, instance: ProblemInstance, fitness: Fitness, rng: np.random.Generator) -> Result:
        from ..problem.path_loss import ENV_PRESETS, optimal_elevation_angle
        from .candidates import build_candidate_set, coverage_matrix

        env = ENV_PRESETS[self.environment]
        theta_opt = optimal_elevation_angle(**env)
        tan_theta = float(np.tan(np.radians(theta_opt)))

        link = getattr(fitness, "link", None)
        z_lo, z_hi = float(instance.lower[2]), float(instance.upper[2])
        if link is not None:
            r_max = float(link.radius(link.z_star_m))
            h_opt = float(np.clip(link.z_star_m, z_lo, z_hi))
        else:
            r_max = float(np.sqrt(max(instance.R_comm ** 2 - z_lo * z_lo, 0.0)))
            h_opt = float(np.clip(r_max * tan_theta, z_lo, z_hi))

        device_xy = instance.device_coords[:, :2]
        # Lexicographic priority weights: high = 1, low = 1/(M+N) < 1 in total.
        median_value = float(np.median(instance.value))
        is_high = instance.value > median_value
        weight = np.where(is_high, 1.0, 1.0 / max(instance.N, 1))

        centres = np.empty((instance.K, 2))
        residual = np.ones(instance.N, dtype=bool)
        for k in range(instance.K):
            if not residual.any():
                centres[k] = centres[k - 1] if k else device_xy.mean(axis=0)
                continue
            pts = device_xy[residual]
            cands = build_candidate_set(
                pts, r_max, instance.lower[:2], instance.upper[:2],
                max_candidates=8000, rng=rng,
            )
            cover = coverage_matrix(cands, pts, r_max)
            gain = cover @ weight[residual]
            best = int(np.argmax(gain))
            centres[k] = cands[best]
            hit = np.zeros(instance.N, dtype=bool)
            hit[np.flatnonzero(residual)[cover[best]]] = True
            residual &= ~hit

        centres = np.clip(centres, instance.lower[:2], instance.upper[:2])
        positions = np.column_stack([centres, np.full(instance.K, h_opt)])
        x = positions.reshape(instance.dim)
        f = fitness(x)
        return Result(
            method=self.name, best_position=x, best_fitness=f, convergence=[f],
            n_iterations=1,
            meta={"altitude_m": h_opt, "theta_opt_deg": theta_opt, "r_max_m": r_max},
        )


class MOGOA(Optimizer):
    """Multi-objective grasshopper optimization (Almaameri & Blazovics, Cluster Computing 29:392, 2026).

    Verified against the source PDF. Swarm update, Eq. (8):

        X_i = c * sum_j [ c * (ub-lb)/2 * S(d_ij) * dhat_ij ] + T_hat
        S(r) = f * exp(-r/l) - exp(-r),      f = 0.5, l = 1.5
        c    = c_max - iter * (c_max - c_min) / iter_max,   c_max = 1, c_min = 1e-4

    Cost, Eq. (25): ``C_T = alpha*C1 + beta*C2 + gamma*C3`` with
    ``alpha=0.4, beta=0.3, gamma=0.3``, where ``C1 = N_uncovered * P_c`` (penalty
    ``P_c = 0.3``) and ``C2 = sum max(0, 0.2*r_d - dist)`` — an explicit
    **overlap penalty**, which is what distinguishes this method: it deliberately
    spreads UAVs apart to limit interference, at some cost in raw coverage.

    Adaptations, stated:

    * **C3 dropped.** The third objective is end-to-end delay over a multi-hop
      route from each drone back to a ground base station. This benchmark has no
      GBS and no multi-hop backhaul, so there is nothing to compute; the weights
      on C1 and C2 are renormalized to 0.4/0.3 -> 4/7 and 3/7 so their ratio is
      the published one.
    * **Altitude.** The paper states a *uniform* flight altitude and does all its
      distance arithmetic in 2D. Uniform altitude is kept, chosen on the shared
      objective so the method is not disadvantaged by an arbitrary constant.
    """

    name = "mogoa"

    def __init__(
        self,
        P: int = 50,
        G_max: int = 100,
        f_social: float = 0.5,
        l_scale: float = 1.5,
        c_max: float = 1.0,
        c_min: float = 1e-4,
        penalty_uncovered: float = 0.3,
        overlap_frac: float = 0.2,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.P, self.G_max = P, G_max
        self.f_social, self.l_scale = f_social, l_scale
        self.c_max, self.c_min = c_max, c_min
        self.penalty_uncovered = penalty_uncovered
        self.overlap_frac = overlap_frac

    def _social(self, r: np.ndarray) -> np.ndarray:
        """S(r) = f e^(-r/l) - e^(-r) — Eq. (6), attraction minus repulsion."""
        return self.f_social * np.exp(-r / self.l_scale) - np.exp(-r)

    def _cost(self, layout_xy: np.ndarray, device_xy: np.ndarray, r_d: float) -> float:
        d = np.sqrt(
            (device_xy[:, None, 0] - layout_xy[None, :, 0]) ** 2
            + (device_xy[:, None, 1] - layout_xy[None, :, 1]) ** 2
        )
        n_uncovered = int(np.sum(~(d <= r_d).any(axis=1)))
        c1 = n_uncovered * self.penalty_uncovered
        sep = np.sqrt(
            np.sum((layout_xy[:, None, :] - layout_xy[None, :, :]) ** 2, axis=2)
        )
        np.fill_diagonal(sep, np.inf)
        c2 = float(np.sum(np.maximum(0.0, self.overlap_frac * r_d - sep)) / 2.0)
        return (4.0 * c1 + 3.0 * c2) / 7.0  # published 0.4:0.3, C3 dropped

    def _run(self, instance: ProblemInstance, fitness: Fitness, rng: np.random.Generator) -> Result:
        K = instance.K
        device_xy = instance.device_coords[:, :2]
        link = getattr(fitness, "link", None)
        z_lo, z_hi = float(instance.lower[2]), float(instance.upper[2])
        if link is not None:
            z = float(np.clip(link.z_star_m, z_lo, z_hi))
            r_d = float(link.radius(z))
        else:
            z = z_lo
            r_d = float(np.sqrt(max(instance.R_comm ** 2 - z * z, 0.0)))

        lo_xy, hi_xy = instance.lower[:2], instance.upper[:2]
        span = float(np.mean(hi_xy - lo_xy))
        X = rng.uniform(np.tile(lo_xy, K), np.tile(hi_xy, K), size=(self.P, 2 * K))

        costs = np.array([self._cost(x.reshape(K, 2), device_xy, r_d) for x in X])
        best = int(np.argmin(costs))
        target, target_cost = X[best].copy(), float(costs[best])

        def to_positions(flat: np.ndarray) -> np.ndarray:
            return np.column_stack([flat.reshape(K, 2), np.full(K, z)]).reshape(instance.dim)

        convergence = [float(fitness(to_positions(target)))]

        for it in range(self.G_max):
            c = self.c_max - it * (self.c_max - self.c_min) / max(self.G_max, 1)
            for i in range(self.P):
                pos = X[i].reshape(K, 2)
                delta = pos[:, None, :] - pos[None, :, :]
                dist = np.sqrt(np.sum(delta * delta, axis=2))
                np.fill_diagonal(dist, np.inf)
                # Distances are normalized into [1,4] as in the source's own
                # implementation; S(r) is only meaningful on that scale.
                dn = 1.0 + 3.0 * np.clip(dist / max(span, 1e-9), 0.0, 1.0)
                unit = delta / np.maximum(dist, 1e-12)[:, :, None]
                contrib = c * (span / 2.0) * self._social(dn)[:, :, None] * unit
                X[i] = (c * np.nansum(contrib, axis=1)).reshape(2 * K) + target
            np.clip(X, np.tile(lo_xy, K), np.tile(hi_xy, K), out=X)

            costs = np.array([self._cost(x.reshape(K, 2), device_xy, r_d) for x in X])
            b = int(np.argmin(costs))
            if costs[b] < target_cost:
                target_cost = float(costs[b])
                target = X[b].copy()
            convergence.append(convergence[-1])

        x = to_positions(target)
        f = float(fitness(x))
        convergence[-1] = f
        return Result(
            method=self.name, best_position=x, best_fitness=f, convergence=convergence,
            n_iterations=self.G_max,
            meta={"altitude_m": z, "r_d_m": r_d, "mogoa_cost": target_cost},
        )
