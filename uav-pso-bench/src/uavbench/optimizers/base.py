"""Optimizer abstract base class and the standard Result container.

Every optimizer scores candidates *only* through the supplied :class:`Fitness`
callable, so all methods share one objective, one greedy assignment, and one
evaluation budget. The runner constructs the ``Fitness`` and reads its
``eval_count`` back from the returned :class:`Result`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..problem.fitness import Fitness
from ..problem.instance import ProblemInstance


@dataclass
class Result:
    """Standard optimizer output."""

    method: str
    best_position: np.ndarray
    best_fitness: float
    convergence: list[float] = field(default_factory=list)  # best-so-far per iteration
    eval_count: int = 0
    wall_time: float = 0.0
    n_iterations: int = 0
    meta: dict = field(default_factory=dict)


class Optimizer(ABC):
    """Base class. Subclasses implement :meth:`_run`; :meth:`optimize` times it."""

    name: str = "base"

    def __init__(self, **params) -> None:
        self.params = params

    def optimize(
        self, instance: ProblemInstance, fitness: Fitness, rng: np.random.Generator
    ) -> Result:
        """Run the optimizer, timing wall-clock and recording the eval count."""
        start = time.perf_counter()
        result = self._run(instance, fitness, rng)
        result.wall_time = time.perf_counter() - start
        result.eval_count = fitness.eval_count
        result.method = self.name
        return result

    @abstractmethod
    def _run(
        self, instance: ProblemInstance, fitness: Fitness, rng: np.random.Generator
    ) -> Result:
        """Optimize and return a Result (wall_time/eval_count filled by caller)."""
        raise NotImplementedError

    # --- shared helpers -------------------------------------------------

    @staticmethod
    def _tile_bounds(instance: ProblemInstance) -> tuple[np.ndarray, np.ndarray]:
        """Return per-gene lower/upper bounds of length 3K (x,y,z tiled K times)."""
        lo = np.tile(instance.lower, instance.K)
        hi = np.tile(instance.upper, instance.K)
        return lo, hi

    @staticmethod
    def _uniform_population(
        rng: np.random.Generator, n: int, lo: np.ndarray, hi: np.ndarray
    ) -> np.ndarray:
        """Sample ``n`` uniform candidates within the per-gene bounds."""
        return rng.uniform(lo, hi, size=(n, lo.shape[0]))
