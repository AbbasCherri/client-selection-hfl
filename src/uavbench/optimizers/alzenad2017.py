"""Alzenad et al. 2017 decoupled 3-D placement baseline.

Reimplements the placement rule of Alzenad, El-Keyi, Lagum &
Yanikomeroglu, "3-D Placement of an Unmanned Aerial Vehicle Base Station
(UAV-BS) for Energy-Efficient Maximal Coverage," IEEE Wireless
Communications Letters, vol. 6, no. 4, 2017: the horizontal placement is
solved first as a 2-D coverage problem, then the altitude is optimized
separately — the *lowest* altitude whose path-loss-derived radius still
covers the served devices (energy-efficient: minimum transmit power).

Adaptation notes (report in the paper's baseline-methodology section,
matching the REPCAP_GAMMA / FAIRMAB_ALPHA convention in client_selection.py):
the source paper places a single UAV-BS; here devices are partitioned into
K value-weighted k-means clusters and the paper's decoupled 2D+altitude
rule is applied per cluster, yielding genuinely non-uniform per-UAV radii.
The 2-D step uses the value-weighted centroid rather than the paper's
smallest-enclosing-circle center.
Known deviation (not verified against the source derivation): whether an
exact smallest-enclosing-circle center (Welzl) would materially tighten
`required_radius` versus the centroid used here is not evaluated — the
centroid is a deliberate, stated stand-in. Full pseudocode and fidelity
notes: Appendix A.7 of `REPORTS/master_implementation_reference.md`.

Scoring is one shot through the shared :class:`Fitness` with the per-UAV
radii applied, so all reported numbers are comparable with PSO/GA.
"""

from __future__ import annotations

import numpy as np

from ..problem.fitness import Fitness
from ..problem.instance import ProblemInstance
from ..problem.path_loss import ENV_PRESETS, min_altitude_for_radius
from .base import Optimizer, Result
from .seeding import weighted_kmeans


class Alzenad2017(Optimizer):
    """Per-cluster 2-D placement + minimum-altitude coverage optimization."""

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
        """Intersect the configured altitude range with the instance z-bounds."""
        lo = max(self.h_min_m, float(instance.lower[2]))
        hi = min(self.h_max_m, float(instance.upper[2]))
        if not lo < hi:
            lo, hi = float(instance.lower[2]), float(instance.upper[2])
        return lo, hi

    def _run(self, instance: ProblemInstance, fitness: Fitness, rng: np.random.Generator) -> Result:
        link = getattr(fitness, "link", None)
        env = ENV_PRESETS[self.environment]
        h_lo, h_hi = self._altitude_bounds(instance)

        device_xy = instance.device_coords[:, :2]
        centers = weighted_kmeans(rng, device_xy, instance.K, instance.value)  # (K, 2)
        d_to_centers = np.linalg.norm(device_xy[:, None, :] - centers[None, :, :], axis=2)  # (N, K)
        cluster_of = d_to_centers.argmin(axis=1)

        z = np.empty(instance.K)
        radii = np.empty(instance.K)
        for k in range(instance.K):
            members = device_xy[cluster_of == k]
            req_r = (
                float(np.max(np.linalg.norm(members - centers[k], axis=1)))
                if members.shape[0]
                else 0.0
            )
            # If no altitude in-band reaches req_r, min_altitude_for_radius
            # falls back to the best achievable radius (partial coverage,
            # reported honestly through the shared fitness).
            if link is not None:
                # Same rule — lowest altitude that reaches the cluster's required
                # radius — resolved against the shared channel instead of this
                # method's private path-loss budget, so no radii override is
                # published and the equal-radius question does not arise.
                h_k = link.min_altitude_for_radius(max(req_r, 1.0))
                r_k = float(link.radius(h_k))
            else:
                h_k, r_k = min_altitude_for_radius(
                    max(req_r, 1.0),
                    self.max_path_loss_db,
                    self.freq_ghz * 1e9,
                    **env,
                    h_min_m=h_lo,
                    h_max_m=h_hi,
                )
            z[k] = h_k
            radii[k] = r_k

        positions = np.column_stack([centers, z])
        x = positions.reshape(instance.dim)
        if link is not None:
            f = fitness(x)
            meta = {"altitudes_m": z.tolist()}
        else:
            f = fitness(x, radii=radii)
            meta = {"radii": radii, "altitudes_m": z.tolist()}
        return Result(
            method=self.name,
            best_position=x,
            best_fitness=f,
            convergence=[f],
            n_iterations=1,
            meta=meta,
        )
