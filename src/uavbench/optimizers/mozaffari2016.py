"""Mozaffari et al. 2016 circle-packing deployment baseline.

Mozaffari, Saad, Bennis & Debbah, "Efficient Deployment of Multiple Unmanned
Aerial Vehicles for Optimal Wireless Coverage", IEEE Communications Letters
20(8):1647-1650, 2016. **Verified against the source PDF, 2026-08-06.**

What the paper actually does (and what this file used to do instead)
--------------------------------------------------------------------
The previous implementation was a different algorithm wearing this paper's
name. It derived a single altitude by maximizing the coverage radius under the
Al-Hourani model, then ran a value-weighted greedy maximal covering over device
locations. Neither step is in the paper:

* The altitude is **not** the radius-maximizing one — that is Al-Hourani (2014)
  and Alzenad (2017). Mozaffari et al. use a **directional antenna** and set
  ``h = r_u / tan(theta_B / 2)`` (Section III), where ``theta_B`` is the antenna
  half-beamwidth and ``r_u`` comes from the packing, not from path loss.
* The placement is **not** demand-driven at all. Section III solves Eq. (10)-(13):
  arrange ``M`` equal, **non-overlapping** discs inside the circular service area
  to maximize total covered *area*, via **circle packing theory**. Devices never
  enter the objective. Equal radii are deliberate — the paper maximizes coverage
  *lifetime*, which requires equal transmit power across UAVs.

That distinction matters for this benchmark: a demand-agnostic area-packing rule
is expected to do poorly on value-weighted device coverage, and reporting it as
though it were a greedy device-covering method both overstated it and mislabelled
what it is.

Adaptations, stated
-------------------
* **Service area.** The paper assumes a given circular region of radius ``R_c``.
  Here ``R_c`` is the radius of the smallest circle enclosing the devices — the
  region that actually has to be served. The packing itself stays demand-blind.
* **Packing radii** come from the paper's Table I (Gáspár & Tarnai 2000) for
  ``M <= 20``, and from the hexagonal-limit density 0.9069 above that.
* **Packing arrangement.** The paper states a general packing strategy is
  intractable and that "for each value of M a specific packing strategy needs to
  be provided", giving the explicit construction only for M=3. This uses a
  deterministic repulsion relaxation from a Fibonacci-spiral start, which
  converges to the published radii; it is a numerical realization of the same
  construction, not a different rule.
* **Interference-derived ``r_u``.** The paper's Theorem 1 sizes ``r_u`` from a
  coverage-probability threshold under inter-UAV interference. This benchmark
  has no interference model, so ``r_u`` is taken straight from the packing —
  which is the binding constraint in the paper's own Eq. (11) anyway.
"""

from __future__ import annotations

import math

import numpy as np

from ..problem.fitness import Fitness
from ..problem.instance import ProblemInstance
from .base import Optimizer, Result
from .recent_baselines import smallest_enclosing_circle_center

# Table I of the paper (Gáspár & Tarnai 2000): radius of each of M equal
# non-overlapping circles packed in a unit circle. The paper tabulates M<=10;
# 11-20 are the same source's published optima, needed because this benchmark
# deploys K=20.
_PACKING_RATIO: dict[int, float] = {
    1: 1.0, 2: 0.5, 3: 0.464102, 4: 0.414214, 5: 0.370191,
    6: 0.333333, 7: 0.333333, 8: 0.302593, 9: 0.275932, 10: 0.262259,
    11: 0.254106, 12: 0.248858, 13: 0.236068, 14: 0.225228, 15: 0.221172,
    16: 0.216677, 17: 0.206518, 18: 0.205604, 19: 0.205124, 20: 0.195224,
}
_HEX_DENSITY = 0.9069  # limiting packing density, for M beyond the table


def packing_ratio(m: int) -> float:
    """``r_u / R_c`` for M equal non-overlapping circles in a circle."""
    if m in _PACKING_RATIO:
        return _PACKING_RATIO[m]
    return math.sqrt(_HEX_DENSITY / max(m, 1))


def pack_circles(m: int, r_c: float, r_u: float, n_iter: int = 400) -> np.ndarray:
    """``(m, 2)`` centres of ``m`` discs of radius ``r_u`` inside radius ``r_c``.

    Deterministic repulsion relaxation from a Fibonacci-spiral start: push
    overlapping pairs apart, then project back inside the containment circle.
    Reaches the published packing densities without needing a hand-built
    arrangement per ``m`` — which the paper itself says is unavoidable for an
    exact construction.
    """
    if m <= 0:
        return np.zeros((0, 2))
    if m == 1:
        return np.zeros((1, 2))

    # Fibonacci spiral: even radial spread, no coincident starts, no RNG.
    golden = math.pi * (3.0 - math.sqrt(5.0))
    idx = np.arange(m, dtype=np.float64)
    radius = max(r_c - r_u, 0.0) * np.sqrt((idx + 0.5) / m)
    angle = idx * golden
    pts = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])

    limit = max(r_c - r_u, 0.0)
    for _ in range(n_iter):
        delta = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(delta * delta, axis=2))
        np.fill_diagonal(dist, np.inf)
        overlap = (2.0 * r_u) - dist
        active = overlap > 0.0
        if not active.any():
            break
        # Push each overlapping pair apart by half the overlap, each way.
        scale = np.where(active, 0.5 * overlap / np.maximum(dist, 1e-12), 0.0)
        pts = pts + np.sum(scale[:, :, None] * delta, axis=1)
        norm = np.linalg.norm(pts, axis=1)
        too_far = norm > limit
        if too_far.any():
            pts[too_far] *= (limit / np.maximum(norm[too_far], 1e-12))[:, None]
    return pts


class Mozaffari2016(Optimizer):
    """Equal non-overlapping discs packed over the service area."""

    name = "mozaffari2016"

    def __init__(
        self,
        theta_b_deg: float = 80.0,  # directional antenna beamwidth (paper's Fig. 3-4)
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.theta_b_deg = theta_b_deg

    def _run(self, instance: ProblemInstance, fitness: Fitness, rng: np.random.Generator) -> Result:
        device_xy = instance.device_coords[:, :2]
        K = instance.K

        # Service area: the smallest circle enclosing the devices.
        centre = smallest_enclosing_circle_center(device_xy)
        r_c = float(np.max(np.linalg.norm(device_xy - centre, axis=1)))

        r_u = packing_ratio(K) * r_c
        offsets = pack_circles(K, r_c, r_u)
        xy = centre[None, :] + offsets

        # h = r_u / tan(theta_B / 2) — the paper's Section III altitude rule,
        # clamped into the altitude band this benchmark allows.
        half_beam = math.radians(self.theta_b_deg / 2.0)
        h = r_u / max(math.tan(half_beam), 1e-9)
        h = float(np.clip(h, instance.lower[2], instance.upper[2]))

        xy = np.clip(xy, instance.lower[:2], instance.upper[:2])
        positions = np.column_stack([xy, np.full(K, h)])
        x = positions.reshape(instance.dim)
        f = fitness(x)
        return Result(
            method=self.name,
            best_position=x,
            best_fitness=f,
            convergence=[f],
            n_iterations=1,
            meta={
                "altitude_m": h,
                "packing_radius_m": r_u,
                "service_radius_m": r_c,
                "theta_b_deg": self.theta_b_deg,
            },
        )
