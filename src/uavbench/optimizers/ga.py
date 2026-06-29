"""Genetic Algorithm baseline (the incumbent PSO replaces).

Real-coded GA with SBX crossover, polynomial mutation, tournament selection, and
elitism, run at the same population/generation budget as PSO so the head-to-head
spends the same number of fitness evaluations.
"""

from __future__ import annotations

import numpy as np

from ..problem.fitness import Fitness
from ..problem.instance import ProblemInstance
from .base import Optimizer, Result


class GA(Optimizer):
    """Real-coded genetic algorithm with SBX + polynomial mutation."""

    name = "ga"

    def __init__(
        self,
        P: int = 100,
        G_max: int = 200,
        crossover_prob: float = 0.9,
        eta_c: float = 15.0,
        eta_m: float = 20.0,
        mutation_prob: float | None = None,  # default 1/dim
        tournament_size: int = 3,
        n_elite: int = 2,
        early_stop_frac: float = 0.95,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.P, self.G_max = P, G_max
        self.crossover_prob = crossover_prob
        self.eta_c, self.eta_m = eta_c, eta_m
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.n_elite = n_elite
        self.early_stop_frac = early_stop_frac

    def _tournament(self, rng: np.random.Generator, fit: np.ndarray) -> int:
        contenders = rng.integers(0, fit.shape[0], size=self.tournament_size)
        return int(contenders[fit[contenders].argmax()])

    def _sbx(self, rng, p1, p2, lo, hi):
        dim = p1.shape[0]
        c1, c2 = p1.copy(), p2.copy()
        do = rng.random(dim) <= 0.5
        u = rng.random(dim)
        beta = np.where(
            u <= 0.5,
            (2.0 * u) ** (1.0 / (self.eta_c + 1.0)),
            (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta_c + 1.0)),
        )
        ch1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        ch2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        c1 = np.where(do, ch1, c1)
        c2 = np.where(do, ch2, c2)
        return np.clip(c1, lo, hi), np.clip(c2, lo, hi)

    def _mutate(self, rng, x, lo, hi):
        """Bounded polynomial mutation (Deb & Agrawal 1999).

        The perturbation magnitude scales with the distance to the nearer boundary,
        ensuring the offspring always lies in [lo, hi] without clipping.
        """
        dim = x.shape[0]
        pm = self.mutation_prob if self.mutation_prob is not None else 1.0 / dim
        do = rng.random(dim) < pm
        u = rng.random(dim)
        delta = np.where(
            u < 0.5,
            (2.0 * u) ** (1.0 / (self.eta_m + 1.0)) - 1.0,        # range [-1, 0]
            1.0 - (2.0 * (1.0 - u)) ** (1.0 / (self.eta_m + 1.0)), # range [0,  1]
        )
        x_new = np.where(
            do,
            np.where(u < 0.5, x + delta * (x - lo), x + delta * (hi - x)),
            x,
        )
        return np.clip(x_new, lo, hi)

    def _run(
        self, instance: ProblemInstance, fitness: Fitness, rng: np.random.Generator
    ) -> Result:
        lo, hi = self._tile_bounds(instance)

        pop = self._uniform_population(rng, self.P, lo, hi)
        fit = np.array([fitness(pop[i]) for i in range(self.P)])

        g = int(fit.argmax())
        best_pos, best_fit = pop[g].copy(), float(fit[g])
        threshold = self.early_stop_frac * fitness.w1
        convergence = [best_fit]
        n_iter = 0

        for _ in range(self.G_max):
            n_iter += 1
            elite_idx = np.argsort(fit)[-self.n_elite:]
            new_pop = [pop[i].copy() for i in elite_idx]

            while len(new_pop) < self.P:
                a = self._tournament(rng, fit)
                b = self._tournament(rng, fit)
                if rng.random() < self.crossover_prob:
                    c1, c2 = self._sbx(rng, pop[a], pop[b], lo, hi)
                else:
                    c1, c2 = pop[a].copy(), pop[b].copy()
                new_pop.append(self._mutate(rng, c1, lo, hi))
                if len(new_pop) < self.P:
                    new_pop.append(self._mutate(rng, c2, lo, hi))

            pop = np.array(new_pop[: self.P])
            fit = np.array([fitness(pop[i]) for i in range(self.P)])

            g = int(fit.argmax())
            if fit[g] > best_fit:
                best_fit, best_pos = float(fit[g]), pop[g].copy()
            convergence.append(best_fit)

            if best_fit >= threshold:
                break

        return Result(
            method=self.name,
            best_position=best_pos,
            best_fitness=best_fit,
            convergence=convergence,
            n_iterations=n_iter,
        )
