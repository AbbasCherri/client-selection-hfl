"""Alternative population metaheuristics: Differential Evolution and Grey Wolf.

These are controls, not contenders. The interesting question raised by PSO's
tight-radius behaviour is *which* of two explanations holds:

  (a) PSO specifically is a poor fit for this landscape, or
  (b) population metaheuristics as a family are the wrong tool here, because the
      coverage term is piecewise constant and the informative structure is
      combinatorial (see :mod:`.mclp_ls`).

Only (b) justifies replacing the search paradigm rather than tuning PSO. Two
independent metaheuristics with different move operators — DE's difference-vector
recombination and GWO's leader-following contraction — separate the two: if both
land near PSO while the candidate-set method pulls clear, the answer is (b).

Both take the identical evaluation budget and the identical initial population
(:func:`.seeding.seeded_population`), so any difference is the search dynamics
and nothing else.

References
----------
DE: Storn & Price, "Differential Evolution — A Simple and Efficient Heuristic for
Global Optimization over Continuous Spaces", J. Global Optimization 11, 1997
(``DE/rand/1/bin``, the canonical variant).
GWO: Mirjalili, Mirjalili & Lewis, "Grey Wolf Optimizer", Advances in Engineering
Software 69, 2014.
"""

from __future__ import annotations

import numpy as np

from ..problem.fitness import Fitness
from ..problem.instance import ProblemInstance
from .base import Optimizer, Result
from .seeding import seeded_population


class DifferentialEvolution(Optimizer):
    """``DE/rand/1/bin`` on the shared 3K-dimensional placement vector."""

    name = "de"

    def __init__(
        self,
        P: int = 100,
        G_max: int = 200,
        F: float = 0.5,
        CR: float = 0.9,
        seeding: str = "value_kmeans",
        jitter_m: float = 10.0,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.P, self.G_max = P, G_max
        self.F, self.CR = F, CR
        self.seeding = seeding
        self.jitter_m = jitter_m

    def _run(self, instance: ProblemInstance, fitness: Fitness, rng: np.random.Generator) -> Result:
        lo, hi = self._tile_bounds(instance)
        dim = instance.dim

        X = seeded_population(
            rng, instance, self.P, lo, hi, seeding=self.seeding, jitter_m=self.jitter_m
        )
        fit = fitness.batch(X)
        g = int(fit.argmax())
        best_pos, best_fit = X[g].copy(), float(fit[g])
        convergence = [best_fit]

        idx = np.arange(self.P)
        for _ in range(self.G_max):
            # Three distinct donors per target, none equal to the target itself.
            # Sampling by argsort of random keys draws a permutation per row in
            # one vectorized call; rejection-sampling per individual would put a
            # Python loop inside the generation loop.
            keys = rng.random((self.P, self.P))
            keys[idx, idx] = np.inf  # exclude self
            donors = np.argsort(keys, axis=1)[:, :3]
            r1, r2, r3 = donors[:, 0], donors[:, 1], donors[:, 2]

            V = X[r1] + self.F * (X[r2] - X[r3])
            np.clip(V, lo, hi, out=V)

            cross = rng.random((self.P, dim)) < self.CR
            # Guarantee at least one inherited gene, else CR<1 can reproduce the
            # target exactly and the generation is wasted.
            forced = rng.integers(0, dim, size=self.P)
            cross[idx, forced] = True
            U = np.where(cross, V, X)

            u_fit = fitness.batch(U)
            better = u_fit > fit
            X[better] = U[better]
            fit[better] = u_fit[better]

            g = int(fit.argmax())
            if fit[g] > best_fit:
                best_fit = float(fit[g])
                best_pos = X[g].copy()
            convergence.append(best_fit)

        return Result(
            method=self.name,
            best_position=best_pos,
            best_fitness=best_fit,
            convergence=convergence,
            n_iterations=self.G_max,
            meta={"F": self.F, "CR": self.CR},
        )


class GreyWolfOptimizer(Optimizer):
    """Grey Wolf Optimizer — alpha/beta/delta leader-following contraction."""

    name = "gwo"

    def __init__(
        self,
        P: int = 100,
        G_max: int = 200,
        a_max: float = 2.0,
        seeding: str = "value_kmeans",
        jitter_m: float = 10.0,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.P, self.G_max = P, G_max
        self.a_max = a_max
        self.seeding = seeding
        self.jitter_m = jitter_m

    def _run(self, instance: ProblemInstance, fitness: Fitness, rng: np.random.Generator) -> Result:
        lo, hi = self._tile_bounds(instance)
        dim = instance.dim

        X = seeded_population(
            rng, instance, self.P, lo, hi, seeding=self.seeding, jitter_m=self.jitter_m
        )
        fit = fitness.batch(X)
        top3 = np.argsort(-fit)[:3]
        leaders = X[top3].copy()  # (3, dim): alpha, beta, delta
        leader_fit = fit[top3].copy()
        convergence = [float(leader_fit[0])]

        for t in range(self.G_max):
            # `a` anneals 2 -> 0, shrinking |A| and turning exploration into
            # exploitation. This is the whole of GWO's schedule.
            a = self.a_max * (1.0 - t / self.G_max)

            A = 2.0 * a * rng.random((3, self.P, dim)) - a
            C = 2.0 * rng.random((3, self.P, dim))
            D = np.abs(C * leaders[:, None, :] - X[None, :, :])
            X = np.mean(leaders[:, None, :] - A * D, axis=0)
            np.clip(X, lo, hi, out=X)

            fit = fitness.batch(X)
            # Leaders persist across generations: merge the new population with
            # the incumbent leaders and re-rank, so a good alpha is never lost to
            # a bad generation (plain GWO re-ranks the population only, which can
            # regress the best-so-far).
            pool = np.vstack([leaders, X])
            pool_fit = np.concatenate([leader_fit, fit])
            top3 = np.argsort(-pool_fit)[:3]
            leaders = pool[top3].copy()
            leader_fit = pool_fit[top3].copy()
            convergence.append(float(leader_fit[0]))

        return Result(
            method=self.name,
            best_position=leaders[0].copy(),
            best_fitness=float(leader_fit[0]),
            convergence=convergence,
            n_iterations=self.G_max,
            meta={"a_max": self.a_max},
        )
