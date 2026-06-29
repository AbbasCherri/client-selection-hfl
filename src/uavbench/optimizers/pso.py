"""Our Particle Swarm Optimizer for 3D UAV placement (PSO guide Section 5).

Constriction-factor PSO with an lbest ring topology, per-dimension velocity
clamping, absorbing walls, value-weighted k-means++ warm starting, stagnation
reinitialization, and mild turbulence. Every design choice is a config toggle so
the ablations are one-line changes.
"""

from __future__ import annotations

import numpy as np

from ..problem.fitness import Fitness
from ..problem.instance import ProblemInstance
from .base import Optimizer, Result
from .seeding import kmeanspp_centers


def constriction_factor(phi: float) -> float:
    """chi = 2 / |2 - phi - sqrt(phi^2 - 4*phi)|  (Clerc & Kennedy 2002).

    Requires phi > 4 for the discriminant to be positive.
    """
    if phi <= 4.0:
        raise ValueError(f"phi must be > 4 for constriction PSO; got {phi}")
    return 2.0 / abs(2.0 - phi - np.sqrt(phi * phi - 4.0 * phi))


class PSO(Optimizer):
    """Constriction PSO with ring topology and diversity safeguards."""

    name = "pso"

    def __init__(
        self,
        P: int = 100,
        G_max: int = 200,
        c1: float = 2.05,
        c2: float = 2.05,
        vmax_frac: float = 0.2,       # per-dim velocity clamp = vmax_frac*(hi-lo)
        ring_k: int = 2,              # neighbours on each side of the ring (neighbourhood size = 2k+1)
        delta_stag: float = 1e-4,
        G_stag: int = 20,
        rho: float = 0.2,
        p_turb: float = 0.1,
        early_stop_frac: float = 0.95,
        jitter_m: float = 10.0,
        # --- design toggles (for ablations) ---
        use_constriction: bool = True,
        topology: str = "ring",          # "ring" | "gbest"
        use_clamp: bool = True,
        use_stagnation: bool = True,
        use_turbulence: bool = True,
        seeding: str = "value_kmeans",   # "value_kmeans" | "plain_kmeans" | "uniform"
        inertia_max: float = 0.9,
        inertia_min: float = 0.4,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.P, self.G_max = P, G_max
        self.c1, self.c2 = c1, c2
        # phi = c1 + c2 is required by the constriction formula (Clerc & Kennedy).
        # Deriving it here prevents silent chi errors when c1/c2 are changed in ablations.
        self.phi = c1 + c2
        self.chi = constriction_factor(self.phi)
        self.vmax_frac = vmax_frac
        self.ring_k = ring_k
        self.delta_stag, self.G_stag, self.rho = delta_stag, G_stag, rho
        self.p_turb = p_turb
        self.early_stop_frac = early_stop_frac
        self.jitter_m = jitter_m
        self.use_constriction = use_constriction
        self.topology = topology
        self.use_clamp = use_clamp
        self.use_stagnation = use_stagnation
        self.use_turbulence = use_turbulence
        self.seeding = seeding
        self.inertia_max, self.inertia_min = inertia_max, inertia_min

    # -- initialization --------------------------------------------------

    def _init_positions(
        self, instance: ProblemInstance, lo: np.ndarray, hi: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """50% value-weighted k-means++ seeds + 50% uniform (per config)."""
        P, dim, K = self.P, instance.dim, instance.K
        if self.seeding == "uniform":
            return self._uniform_population(rng, P, lo, hi)

        n_seed = P // 2
        device_xy = instance.device_coords[:, :2]
        weights = instance.value if self.seeding == "value_kmeans" else None
        z_lo, z_hi = instance.lower[2], instance.upper[2]

        seeded = np.empty((n_seed, dim), dtype=np.float64)
        for p in range(n_seed):
            centers = kmeanspp_centers(rng, device_xy, K, weights)
            xy = centers + rng.normal(0.0, self.jitter_m, size=(K, 2))
            z = rng.uniform(z_lo, z_hi, size=(K, 1))
            seeded[p] = np.column_stack([xy, z]).reshape(dim)

        seeded = np.clip(seeded, lo, hi)
        uniform = self._uniform_population(rng, P - n_seed, lo, hi)
        return np.vstack([seeded, uniform])

    # -- neighborhood best ----------------------------------------------

    def _neighborhood_best(
        self, pbest: np.ndarray, pbest_fit: np.ndarray, gbest_pos: np.ndarray
    ) -> np.ndarray:
        """Return the (P, dim) array of each particle's neighbourhood-best position.

        Ring topology: each particle i has a symmetric neighbourhood of size
        ``2*ring_k + 1`` (ring_k neighbours on each side plus itself).
        ``ring_k=1`` → classic lbest ring; ``ring_k=2`` → lbest2 ring (5 particles).
        """
        if self.topology == "gbest":
            return np.tile(gbest_pos, (self.P, 1))

        # Vectorised lbest ring — O(P * ring_k) but entirely in NumPy.
        P = self.P
        idx = np.arange(P)
        offsets = np.arange(-self.ring_k, self.ring_k + 1)   # shape (2k+1,)
        neighbors = (idx[:, None] + offsets[None, :]) % P    # shape (P, 2k+1)
        neighbor_fits = pbest_fit[neighbors]                   # shape (P, 2k+1)
        best_col = neighbor_fits.argmax(axis=1)               # shape (P,)
        best_idx = neighbors[idx, best_col]                   # shape (P,)
        return pbest[best_idx]

    # -- main loop -------------------------------------------------------

    def _run(
        self, instance: ProblemInstance, fitness: Fitness, rng: np.random.Generator
    ) -> Result:
        lo, hi = self._tile_bounds(instance)
        dim = instance.dim
        vmax = self.vmax_frac * (hi - lo)

        X = self._init_positions(instance, lo, hi, rng)
        Vel = 0.5 * rng.uniform(-vmax, vmax, size=(self.P, dim))

        pbest = X.copy()
        pbest_fit = np.array([fitness(X[i]) for i in range(self.P)])
        g = int(pbest_fit.argmax())
        gbest_pos = pbest[g].copy()
        gbest_fit = float(pbest_fit[g])

        # F_max = w1*1 - w2*0 - w3*0 = w1 (all coverage, zero penalty terms).
        threshold = self.early_stop_frac * fitness.w1
        convergence = [gbest_fit]
        stagnation = 0
        n_iter = 0

        for tau in range(self.G_max):
            n_iter += 1
            nbest = self._neighborhood_best(pbest, pbest_fit, gbest_pos)

            r1 = rng.random((self.P, dim))
            r2 = rng.random((self.P, dim))
            cognitive = self.c1 * r1 * (pbest - X)
            social = self.c2 * r2 * (nbest - X)

            if self.use_constriction:
                Vel = self.chi * (Vel + cognitive + social)
            else:
                w = self.inertia_max - (self.inertia_max - self.inertia_min) * (tau / self.G_max)
                Vel = w * Vel + cognitive + social

            if self.use_turbulence:
                kick_mask = rng.random(self.P) < self.p_turb
                if kick_mask.any():
                    kick = rng.uniform(-0.1 * vmax, 0.1 * vmax, size=(int(kick_mask.sum()), dim))
                    Vel[kick_mask] += kick

            if self.use_clamp:
                np.clip(Vel, -vmax, vmax, out=Vel)

            X = X + Vel

            # Absorbing walls: clamp out-of-bound coords and zero their velocity.
            out = (X < lo) | (X > hi)
            np.clip(X, lo, hi, out=X)
            Vel[out] = 0.0

            fit = np.array([fitness(X[i]) for i in range(self.P)])
            improved = fit > pbest_fit
            pbest[improved] = X[improved]
            pbest_fit[improved] = fit[improved]

            g = int(pbest_fit.argmax())
            if pbest_fit[g] > gbest_fit:  # gbest never overwritten by a worse value
                gbest_fit = float(pbest_fit[g])
                gbest_pos = pbest[g].copy()

            # Stagnation tracking on the global best.
            if convergence and (gbest_fit - convergence[-1]) > self.delta_stag:
                stagnation = 0
            else:
                stagnation += 1
            convergence.append(gbest_fit)

            if self.use_stagnation and stagnation >= self.G_stag:
                n_worst = max(1, int(self.rho * self.P))
                worst = np.argsort(fit)[:n_worst]
                X[worst] = self._uniform_population(rng, n_worst, lo, hi)
                Vel[worst] = 0.5 * rng.uniform(-vmax, vmax, size=(n_worst, dim))
                wf = np.array([fitness(X[i]) for i in worst])
                pbest[worst] = X[worst]
                pbest_fit[worst] = wf
                stagnation = 0
                # Update gbest immediately — a reinit particle may have improved it.
                g_new = int(pbest_fit.argmax())
                if pbest_fit[g_new] > gbest_fit:
                    gbest_fit = float(pbest_fit[g_new])
                    gbest_pos = pbest[g_new].copy()

            if gbest_fit >= threshold:
                break

        return Result(
            method=self.name,
            best_position=gbest_pos,
            best_fitness=gbest_fit,
            convergence=convergence,
            n_iterations=n_iter,
            meta={"chi": self.chi, "phi": self.phi},
        )
