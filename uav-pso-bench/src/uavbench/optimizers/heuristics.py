"""Deterministic / trivial baselines: centroid, random, static.

These bracket the metaheuristics from below (random, static) and provide the
canonical fast deterministic competitor (value-weighted k-means centroid).
"""

from __future__ import annotations

import numpy as np

from ..problem.fitness import Fitness
from ..problem.instance import ProblemInstance
from .base import Optimizer, Result
from .seeding import weighted_kmeans


class Centroid(Optimizer):
    """Place each UAV at a value-weighted k-means centroid of the devices.

    Fast and value-aware, but blind to capacity saturation and the movement
    penalty — expected to lose on the joint objective while winning on runtime.
    """

    name = "centroid"

    def __init__(self, altitude_frac: float = 0.5, value_weighted: bool = True, **kw) -> None:
        super().__init__(**kw)
        self.altitude_frac = altitude_frac
        self.value_weighted = value_weighted

    def _run(
        self, instance: ProblemInstance, fitness: Fitness, rng: np.random.Generator
    ) -> Result:
        device_xy = instance.device_coords[:, :2]
        weights = instance.value if self.value_weighted else None
        centers = weighted_kmeans(rng, device_xy, instance.K, weights)
        z = instance.lower[2] + self.altitude_frac * (instance.upper[2] - instance.lower[2])
        positions = np.column_stack([centers, np.full(instance.K, z)])
        x = positions.reshape(instance.dim)
        f = fitness(x)
        return Result(
            method=self.name,
            best_position=x,
            best_fitness=f,
            convergence=[f],
            n_iterations=1,
        )


class RandomPlacement(Optimizer):
    """Best of ``n_draws`` uniform random placements — the no-intelligence floor."""

    name = "random"

    def __init__(self, n_draws: int = 20, **kw) -> None:
        super().__init__(**kw)
        self.n_draws = n_draws

    def _run(
        self, instance: ProblemInstance, fitness: Fitness, rng: np.random.Generator
    ) -> Result:
        lo, hi = self._tile_bounds(instance)
        best_x, best_f = None, -np.inf
        conv = []
        for _ in range(self.n_draws):
            x = rng.uniform(lo, hi)
            f = fitness(x)
            if f > best_f:
                best_f, best_x = f, x
            conv.append(best_f)
        return Result(
            method=self.name,
            best_position=best_x,
            best_fitness=float(best_f),
            convergence=conv,
            n_iterations=self.n_draws,
        )


class Static(Optimizer):
    """No repositioning: UAVs stay at their previous positions.

    The key ablation-style baseline that measures the value of dynamic placement.
    """

    name = "static"

    def _run(
        self, instance: ProblemInstance, fitness: Fitness, rng: np.random.Generator
    ) -> Result:
        x = instance.prev_positions.reshape(instance.dim)
        f = fitness(x)
        return Result(
            method=self.name,
            best_position=x,
            best_fitness=f,
            convergence=[f],
            n_iterations=1,
        )
