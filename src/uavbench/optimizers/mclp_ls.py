"""Candidate-reduction placement: greedy capacitated covering + budgeted local search.

The idea in one line: this problem does not need a metaheuristic.

Every other optimizer here treats placement as continuous 3K-dimensional search
and spends its budget sampling it. But the objective's dominant term (``w1``
carries 0.811 of the tuned weight) is a coverage count, which is piecewise
constant — it only changes when a disc edge crosses a device. Church (1984)
showed the planar maximal covering optimum can always be attained at a *circle
intersection point*, so the continuous search space collapses, without loss, to
an ``O(N^2)`` finite set (see :mod:`.candidates`). A swarm searching the
continuum is rediscovering, approximately and at 20k evaluations, positions this
method writes down exactly.

Architecture: the 3D problem is split in two, and the split is exact
--------------------------------------------------------------------
Altitude enters the objective through exactly one channel — it sets the ground
radius ``r(z)`` — and coverage is monotonically non-decreasing in that radius.
Every UAV therefore wants the radius-maximizing altitude ``z*``, none wants any
other, and the joint problem collapses:

    max_{z, xy} F(z, xy)  =  max_xy F_2D(xy ; r(z*))

So placement is solved as **a 1D vertical problem then a 2D horizontal one**,
which is the decoupling Mozaffari (2016) and Alzenad (2017) assume — here a
consequence of the objective rather than a modelling convenience. Full argument
and its limits in :func:`.altitude.optimal_shared_altitude`.

Under the air-to-ground link model (:mod:`uavbench.problem.link`) the vertical
problem is non-trivial: climbing raises P(LoS) and sheds the NLoS excess loss
until free-space loss takes over, so ``r(z)`` is unimodal with an interior
optimum. Under the legacy hard range gate it degenerates to the altitude floor.

Pipeline
--------
1. **Vertical (1D)** — ``z*`` in closed form from the link model.
2. **Candidate set** at radius ``r(z*)`` (:func:`.build_candidate_set`).
3. **Capacitated greedy seed** — each UAV in turn takes the candidate covering
   the most residual device *value*, counting only as many devices as it can
   actually serve. Truncating at capacity matters: an uncapacitated greedy piles
   UAVs onto the densest cluster where most of the coverage it is credited with
   is unservable.
4. **Budgeted local search** — remove-and-reinsert per UAV, screened on the
   candidate set and scored on the true shared :class:`Fitness`, so acceptance
   sees capacity, load imbalance and movement exactly as the objective does.
5. **Movement polish** — slide each UAV back toward where it already is, as far
   as it can go without dropping a device it serves. Pure ``d_move`` savings at
   identical coverage.
6. **Per-UAV altitude refinement** — expected to be a no-op by step 1's argument;
   run so that "it changes nothing" is measured rather than assumed.

It consumes the same ``P * (G_max + 1)`` evaluation budget as PSO/GA, so a win
here is not a win bought with extra evaluations.
"""

from __future__ import annotations

import numpy as np

from ..problem.fitness import Fitness
from ..problem.instance import ProblemInstance
from .altitude import optimal_shared_altitude, optimize_altitudes
from .base import Optimizer, Result
from .candidates import build_candidate_set, capped_covered_value, coverage_matrix

_EPS = 1e-12


class MCLPLocalSearch(Optimizer):
    """Deterministic candidate-set covering heuristic with local refinement."""

    name = "mclp_ls"

    def __init__(
        self,
        P: int = 100,
        G_max: int = 200,
        max_candidates: int = 4000,
        screen_top: int = 48,
        dedupe_grid_m: float = 1.0,
        n_altitudes: int = 5,
        polish: bool = True,
        polish_iters: int = 24,
        seed_from_prev: bool = True,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.P, self.G_max = P, G_max
        self.max_candidates = int(max_candidates)
        self.screen_top = int(screen_top)
        self.dedupe_grid_m = float(dedupe_grid_m)
        self.n_altitudes = int(n_altitudes)
        self.polish = bool(polish)
        self.polish_iters = int(polish_iters)
        self.seed_from_prev = bool(seed_from_prev)

    # -- budget ----------------------------------------------------------

    @property
    def max_evals(self) -> int:
        """Match PSO's spend exactly: P initial + P per generation."""
        return self.P * (self.G_max + 1)

    def _tail_evals(self, instance: ProblemInstance) -> int:
        """Evaluations the polish and altitude stages will need after the search.

        Held back from the local search rather than spent on top of it: a method
        that quietly overruns the shared budget makes every comparison against it
        meaningless, and this is exactly where an overrun would hide. The
        altitude descent is per-UAV, so its cost scales with K.
        """
        polish = 2 if self.polish else 0
        alt = 2 + instance.K * self.n_altitudes if self.n_altitudes > 1 else 0
        return polish + alt

    # -- greedy seed -----------------------------------------------------

    def _greedy_seed(
        self,
        instance: ProblemInstance,
        cover_sorted: np.ndarray,
        value_sorted: np.ndarray,
    ) -> np.ndarray:
        """Return ``(K,)`` candidate indices from capacitated greedy covering.

        UAVs are filled in descending capacity so the large-capacity units claim
        the dense clusters; doing it in index order lets a small UAV take a
        cluster it cannot serve and strands the value.
        """
        K = instance.K
        residual = np.ones(value_sorted.shape[0], dtype=bool)
        chosen = np.zeros(K, dtype=np.int64)
        usable = instance.battery >= instance.B_min_uav
        order = np.argsort(-instance.capacity, kind="stable")

        for j in order:
            if not usable[j]:
                # A grounded UAV covers nothing; park it on the densest point so
                # it still emits a well-formed position.
                chosen[j] = int(np.argmax(cover_sorted @ value_sorted))
                continue
            live = cover_sorted & residual
            marginal = capped_covered_value(live, value_sorted, float(instance.capacity[j]))
            m = int(np.argmax(marginal))
            chosen[j] = m
            if marginal[m] <= _EPS:
                continue
            row = live[m]
            take = row & (np.cumsum(row) <= instance.capacity[j])
            residual &= ~take
        return chosen

    # -- local search ----------------------------------------------------

    def _covered_by_others(
        self, instance: ProblemInstance, positions: np.ndarray, j: int
    ) -> np.ndarray:
        """``(N,)`` mask of devices already in range of some UAV other than ``j``."""
        dist = instance.distances(positions)  # (N, K)
        usable = instance.battery >= instance.B_min_uav
        in_range = (dist <= instance.R_comm) & usable[None, :]
        in_range[:, j] = False
        return in_range.any(axis=1)

    # -- movement polish -------------------------------------------------

    def _polish_toward_prev(
        self,
        instance: ProblemInstance,
        positions: np.ndarray,
        assignment: np.ndarray,
        link=None,
    ) -> np.ndarray:
        """Slide each UAV toward its previous position without dropping a device.

        The set of points covering a fixed device set is an intersection of discs
        and hence convex, so along the segment from the found position to the
        previous one the feasible portion is a single interval anchored at t=0.
        Bisection therefore finds the exact furthest feasible step.

        Horizontal only: altitude is held at whatever the UAV is already flying,
        because under the link model moving vertically changes the UAV's own
        coverage radius and the disc would resize mid-slide. The altitude descent
        is a separate stage that scores z on the full objective.
        """
        out = positions.copy()
        prev = instance.prev_positions
        dev_xy = instance.device_coords[:, :2]
        for j in range(instance.K):
            served = dev_xy[assignment == j]
            if served.shape[0] == 0:
                # Serving nobody: going home is free coverage-wise.
                out[j, :2] = prev[j, :2]
                continue
            p0, p1 = positions[j, :2], prev[j, :2]
            if np.allclose(p0, p1):
                continue

            z_j = positions[j, 2]
            r_j = (
                float(link.radius(z_j))
                if link is not None
                else float(np.sqrt(max(instance.R_comm ** 2 - z_j * z_j, 0.0)))
            )
            r2 = r_j * r_j

            def feasible(t: float) -> bool:
                p = p0 + t * (p1 - p0)
                d = served - p
                return bool(np.max(np.sum(d * d, axis=1)) <= r2)

            if feasible(1.0):
                out[j] = p1
                continue
            lo, hi = 0.0, 1.0
            for _ in range(self.polish_iters):
                mid = 0.5 * (lo + hi)
                if feasible(mid):
                    lo = mid
                else:
                    hi = mid
            out[j] = p0 + lo * (p1 - p0)
        return out

    # -- main ------------------------------------------------------------

    def _run(self, instance: ProblemInstance, fitness: Fitness, rng: np.random.Generator) -> Result:
        K, dim = instance.K, instance.dim
        device_xy = instance.device_coords[:, :2]
        z_min, z_max = float(instance.lower[2]), float(instance.upper[2])

        # === STAGE 1: the vertical subproblem (1D) ===========================
        # Solved in closed form off the link model. See optimal_shared_altitude
        # for why this decoupling is exact rather than an approximation.
        link = getattr(fitness, "link", None)
        z0, r_eff = optimal_shared_altitude(instance, link)

        # === STAGE 2: the horizontal subproblem (2D) =========================
        # A pure planar capacitated covering problem at the fixed radius r(z*),
        # which the circle-intersection candidate set renders finite and exact.
        cands = build_candidate_set(
            device_xy,
            r_eff,
            instance.lower[:2],
            instance.upper[:2],
            max_candidates=self.max_candidates,
            dedupe_grid_m=self.dedupe_grid_m,
            rng=rng,
        )
        cover = coverage_matrix(cands, device_xy, r_eff)

        order = instance.value_order
        cover_sorted = cover[:, order]
        value_sorted = instance.value[order]

        chosen = self._greedy_seed(instance, cover_sorted, value_sorted)
        positions = np.column_stack([cands[chosen], np.full(K, z0)])

        best_pos = positions.copy()
        best_fit = float(fitness(best_pos.reshape(dim)))
        convergence = [best_fit]

        # The current layout is a legitimate solution and costs zero movement;
        # carrying it as an incumbent is what makes "never worse than standing
        # still" a property of this method rather than a hope (the failure mode
        # plain PSO showed at tight R_comm).
        if self.seed_from_prev:
            f_prev = float(fitness(instance.prev_positions.reshape(dim)))
            if f_prev > best_fit:
                best_pos, best_fit = instance.prev_positions.copy(), f_prev
            convergence.append(best_fit)

        # --- local search: remove one UAV, reinsert it at the best candidate ---
        cur_pos, cur_fit = positions.copy(), float(convergence[0])
        search_budget = self.max_evals - self._tail_evals(instance)
        n_iter = 0
        while fitness.eval_count + self.screen_top <= search_budget:
            n_iter += 1
            improved = False
            for j in rng.permutation(K):
                if fitness.eval_count + self.screen_top > search_budget:
                    break
                if instance.battery[j] < instance.B_min_uav:
                    continue
                others = self._covered_by_others(instance, cur_pos, int(j))
                marginal = capped_covered_value(
                    cover_sorted & ~others[order],
                    value_sorted,
                    float(instance.capacity[j]),
                )
                top = np.argpartition(-marginal, min(self.screen_top, marginal.size - 1))[
                    : self.screen_top
                ]
                trials = np.repeat(cur_pos[None, :, :], top.size, axis=0)
                trials[:, j, :2] = cands[top]
                trials[:, j, 2] = cur_pos[j, 2]
                scores = fitness.batch(trials)
                b = int(np.argmax(scores))
                if scores[b] > cur_fit + _EPS:
                    cur_fit = float(scores[b])
                    cur_pos = trials[b].copy()
                    improved = True
            convergence.append(max(cur_fit, best_fit))
            if not improved:
                break

        if cur_fit > best_fit:
            best_pos, best_fit = cur_pos.copy(), cur_fit

        # --- movement polish -------------------------------------------------
        if self.polish:
            comp = fitness.components(best_pos.reshape(dim))
            polished = self._polish_toward_prev(
                instance, best_pos, comp.assignment.assignment, link=link
            )
            f_pol = float(fitness(polished.reshape(dim)))
            if f_pol > best_fit:
                best_pos, best_fit = polished, f_pol
            convergence.append(best_fit)

        # === STAGE 3: per-UAV altitude refinement ============================
        # Stage 1 argued every UAV wants z*, so this should find nothing. It runs
        # anyway because the argument covers the coverage term only — the
        # movement and imbalance terms are altitude-sensitive in principle — and
        # because "the refinement changes nothing" is the empirical check on the
        # decoupling claim rather than something to assume.
        if self.n_altitudes > 1 and z_max > z_min:
            best_pos = optimize_altitudes(
                instance, fitness, best_pos,
                n_levels=self.n_altitudes,
                max_evals=max(self.max_evals - fitness.eval_count, 0),
            )
            best_fit = float(fitness(best_pos.reshape(dim)))
            convergence.append(best_fit)

        return Result(
            method=self.name,
            best_position=best_pos.reshape(dim),
            best_fitness=best_fit,
            convergence=convergence,
            n_iterations=n_iter,
            meta={
                "n_candidates": int(cands.shape[0]),
                "candidate_cap_hit": bool(cands.shape[0] >= self.max_candidates),
                "r_eff_m": r_eff,
                "altitude_m": float(best_pos[0, 2]),
            },
        )
