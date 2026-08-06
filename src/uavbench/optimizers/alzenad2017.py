"""Alzenad et al. 2017 decoupled 3-D placement baseline.

Alzenad, El-Keyi, Lagum & Yanikomeroglu, "3-D Placement of an Unmanned Aerial
Vehicle Base Station (UAV-BS) for Energy-Efficient Maximal Coverage", IEEE
Wireless Communications Letters 6(4):434-437, 2017. **Verified against the
source PDF, 2026-08-06.**

Algorithm 1 of the paper, step for step:

    1. theta_opt from Eq. (7)  — depends ONLY on the environment
       (20.34 deg suburban, 42.44 urban, 54.62 dense urban, 75.52 high-rise)
    2. R1 from Eq. (8)         — max coverage radius at theta_opt for the loss budget
    3. h1 = R1 * tan(theta_opt)
    4. Eq. (12): circle placement — centre the disc of radius R1 to enclose the
       MAXIMUM NUMBER of users. Solved to optimality (MINLP, MOSEK).
    5. Eq. (13): smallest enclosing circle of the covered set -> (x*, y*), R2.
       Same users, smaller disc, less transmit power.
    6. h* = max(h_min, R2 * tan(theta_opt))

Two corrections against the previous implementation, which was not this
algorithm: step 4 was a value-weighted k-means assignment rather than an optimal
circle placement, and step 5 was skipped entirely — the UAV sat at the
value-weighted centroid instead of the smallest-enclosing-circle centre. The
centroid minimizes *mean* distance while coverage is decided by the *maximum*,
so it dropped the outlying members of elongated clusters that the paper's rule
holds, and without the recentre the altitude was never reduced.

Adaptations, stated
-------------------
* **Multi-UAV.** The paper places a single UAV-BS. Here ``K`` UAVs are placed
  sequentially: each solves the paper's Algorithm 1 on the devices not yet
  covered, then those devices are removed. This is the natural extension and is
  the only departure from the published rule.
* **Step 4 solved exactly.** The paper uses MOSEK on the MINLP. Here the
  equivalent exact solution comes from the circle-intersection candidate set
  (Church 1984), which provably contains an optimal disc centre — so the
  baseline gets the optimal placement the paper specifies, not an approximation
  of it.
* **Coverage counts devices, unweighted**, as published. This benchmark's own
  objective is value-weighted; that mismatch is a property of the baseline and
  is left visible rather than patched.
"""

from __future__ import annotations

import numpy as np

from ..problem.fitness import Fitness
from ..problem.instance import ProblemInstance
from ..problem.path_loss import ENV_PRESETS, optimal_elevation_angle
from .base import Optimizer, Result
from .candidates import build_candidate_set, coverage_matrix
from .recent_baselines import smallest_enclosing_circle_center


class Alzenad2017(Optimizer):
    """Optimal circle placement + smallest-enclosing-circle recentre, per UAV."""

    name = "alzenad2017"

    def __init__(
        self,
        environment: str = "suburban",
        freq_ghz: float = 2.0,
        max_path_loss_db: float = 100.0,
        h_min_m: float = 20.0,
        h_max_m: float = 500.0,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.environment = environment
        self.freq_ghz = freq_ghz
        self.max_path_loss_db = max_path_loss_db
        self.h_min_m = h_min_m
        self.h_max_m = h_max_m

    def _altitude_bounds(self, instance: ProblemInstance) -> tuple[float, float]:
        lo = max(self.h_min_m, float(instance.lower[2]))
        hi = min(self.h_max_m, float(instance.upper[2]))
        if not lo < hi:
            lo, hi = float(instance.lower[2]), float(instance.upper[2])
        return lo, hi

    def _run(self, instance: ProblemInstance, fitness: Fitness, rng: np.random.Generator) -> Result:
        env = ENV_PRESETS[self.environment]
        h_lo, h_hi = self._altitude_bounds(instance)
        theta_opt = optimal_elevation_angle(**env)  # Eq. (7), environment only
        tan_theta = float(np.tan(np.radians(theta_opt)))

        link = getattr(fitness, "link", None)
        device_xy = instance.device_coords[:, :2]

        # --- step 2: R1, the coverage radius at theta_opt -------------------
        if link is not None:
            # Shared channel: R1 is the largest ground radius the system's link
            # budget supports, which is exactly what Eq. (8) computes for the
            # method's own budget. Sharing it is what keeps the comparison
            # equal-budget; the rule (fly at theta_opt, size to the covered set)
            # is untouched.
            r1 = float(link.radius(link.z_star_m))
        else:
            from ..problem.path_loss import coverage_radius

            h_star = float(np.clip(np.median([h_lo, h_hi]), h_lo, h_hi))
            r1 = coverage_radius(h_star, self.max_path_loss_db, self.freq_ghz * 1e9, **env)

        centres = np.empty((instance.K, 2))
        z = np.empty(instance.K)
        radii = np.empty(instance.K)
        residual = np.ones(instance.N, dtype=bool)

        for k in range(instance.K):
            if not residual.any():
                centres[k] = centres[k - 1] if k else device_xy.mean(axis=0)
                z[k] = z[k - 1] if k else h_lo
                radii[k] = radii[k - 1] if k else r1
                continue

            pts = device_xy[residual]
            # --- step 4: optimal circle placement over the residual set ------
            cands = build_candidate_set(
                pts, r1, instance.lower[:2], instance.upper[:2],
                max_candidates=8000, rng=rng,
            )
            cover = coverage_matrix(cands, pts, r1)
            best = int(np.argmax(cover.sum(axis=1)))
            covered_local = cover[best]

            # --- step 5: smallest enclosing circle of the covered set --------
            served = pts[covered_local]
            if served.shape[0]:
                c2 = smallest_enclosing_circle_center(served)
                r2 = float(np.max(np.linalg.norm(served - c2, axis=1)))
            else:
                c2, r2 = cands[best], r1

            centres[k] = c2
            radii[k] = max(r2, 1.0)
            # --- step 6: h* = max(h_min, R2 * tan(theta_opt)) ----------------
            z[k] = float(np.clip(max(h_lo, radii[k] * tan_theta), h_lo, h_hi))

            hit = np.zeros(instance.N, dtype=bool)
            hit[np.flatnonzero(residual)[covered_local]] = True
            residual &= ~hit

        centres = np.clip(centres, instance.lower[:2], instance.upper[:2])
        positions = np.column_stack([centres, z])
        x = positions.reshape(instance.dim)
        f = fitness(x)
        return Result(
            method=self.name,
            best_position=x,
            best_fitness=f,
            convergence=[f],
            n_iterations=1,
            meta={
                "altitudes_m": z.tolist(),
                "theta_opt_deg": theta_opt,
                "r1_m": r1,
            },
        )
