"""Mozaffari et al. 2016 circle-packing coverage deployment baseline.

Reimplements the placement rule of Mozaffari, Saad, Bennis & Debbah,
"Efficient Deployment of Multiple Unmanned Aerial Vehicles for Optimal
Wireless Coverage," IEEE Communications Letters, vol. 20, no. 8, 2016:
a shared optimal altitude/coverage-radius pair is derived from the
air-to-ground path-loss model (see :mod:`uavbench.problem.path_loss`),
then UAV disc positions are chosen to maximize covered ground area.

Adaptation note (report in the paper's baseline-methodology section, the
same convention as REPCAP_GAMMA / FAIRMAB_W_* in client_selection.py):
the paper's core result is the single-(h*, r*) derivation; its multi-UAV
deployment packs equal discs to maximize total coverage. Here that packing
is realized as greedy maximal covering discretized at the device
locations: each of the K discs is centered on the device position that
maximizes the value of not-yet-covered devices within r*, which are then
removed before the next disc — a deterministic K-UAV generalization
aligned with this benchmark's value-weighted coverage objective, not a
verbatim reproduction.
Known deviation (not verified against the source derivation): the paper's
own multi-UAV packing arrangement is not reproduced verbatim — this greedy
covering is a deliberate, stated stand-in for it. Full pseudocode and
fidelity notes: Appendix A.6 of `REPORTS/master_implementation_reference.md`.

Scoring is one shot through the shared :class:`Fitness` with the derived
per-UAV radius applied (``radii``), so coverage/movement/load numbers are
on identical footing with PSO/GA.
"""

from __future__ import annotations

import numpy as np

from ..problem.fitness import Fitness
from ..problem.instance import ProblemInstance
from ..problem.path_loss import ENV_PRESETS, optimal_altitude_mozaffari
from .base import Optimizer, Result


class Mozaffari2016(Optimizer):
    """Greedy equal-disc coverage placement at the path-loss-optimal altitude."""

    name = "mozaffari2016"

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
        """Intersect the configured altitude range with the instance z-bounds.

        Searching directly inside the feasible band keeps the derived radius
        consistent with the altitude actually flown (no post-hoc clipping).
        """
        lo = max(self.h_min_m, float(instance.lower[2]))
        hi = min(self.h_max_m, float(instance.upper[2]))
        if not lo < hi:
            lo, hi = float(instance.lower[2]), float(instance.upper[2])
        return lo, hi

    def _run(self, instance: ProblemInstance, fitness: Fitness, rng: np.random.Generator) -> Result:
        # When the objective carries a shared air-to-ground link model, this
        # method's own (h*, r*) derivation is *the same derivation* run against
        # the shared budget, so it defers to it and publishes no radii override.
        # That is what finally removes the 618-vs-500 m handicap: the radius is
        # no longer this method's private property, it is a function of the
        # altitude it chose, exactly as it is for every other method.
        link = getattr(fitness, "link", None)
        if link is not None:
            h_star, r_star = float(link.z_star_m), float(link.radius(link.z_star_m))
        else:
            env = ENV_PRESETS[self.environment]
            h_lo, h_hi = self._altitude_bounds(instance)
            h_star, r_star = optimal_altitude_mozaffari(
                self.max_path_loss_db,
                self.freq_ghz * 1e9,
                **env,
                h_min_m=h_lo,
                h_max_m=h_hi,
            )

        device_xy = instance.device_coords[:, :2]
        # Greedy maximal covering: candidate disc centers are the device
        # locations; each disc takes the candidate covering the most residual
        # value, then covered devices are removed. Centering on a device
        # guarantees every placed disc covers at least that device, avoiding
        # the empty-disc degeneracy a residual-centroid rule has on clustered
        # layouts (centroid farther than r* from every cluster).
        pairwise = np.linalg.norm(
            device_xy[:, None, :] - device_xy[None, :, :], axis=2
        )  # (N, N): pairwise[i, j] = distance from device i to candidate j
        within = pairwise <= r_star  # (N, N)
        remaining = np.ones(instance.N, dtype=bool)
        centers: list[np.ndarray] = []
        for _ in range(instance.K):
            if remaining.any():
                covered_value = (instance.value * remaining) @ within  # (N,)
                j = int(np.argmax(covered_value))
                centers.append(device_xy[j])
                remaining &= ~within[:, j]
            else:
                # Devices exhausted before K discs placed: co-locate the spare
                # disc (K positions are always emitted, as with Centroid).
                centers.append(centers[-1] if centers else device_xy.mean(axis=0))

        positions = np.column_stack([np.array(centers), np.full(instance.K, h_star)])
        x = positions.reshape(instance.dim)
        if link is not None:
            f = fitness(x)  # radius derived from altitude by the shared model
            meta = {"altitude_m": h_star, "coverage_radius_m": r_star}
        else:
            radii = np.full(instance.K, r_star)
            f = fitness(x, radii=radii)
            meta = {"radii": radii, "altitude_m": h_star, "coverage_radius_m": r_star}
        return Result(
            method=self.name,
            best_position=x,
            best_fitness=f,
            convergence=[f],
            n_iterations=1,
            meta=meta,
        )
