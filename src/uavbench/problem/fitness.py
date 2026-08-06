"""The shared scalarized objective ``F(X)``.

    F(X) = w1 * (F_cover / F_max)
         - w2 * (D_move  / D_max)
         - w3 * (L_imb   / L_max)

with ``w1=0.6, w2=0.3, w3=0.1`` and normalizers

    F_max = sum_i V_i ;  D_max = K * diag(box) ;  L_max = N^2
    D_move = sum_j || p_j - p_prev_j || ;
    L_imb  = sum_j (|A(j)| - N_assigned / K)^2 .

This :class:`Fitness` callable is the **only** scoring entry point an optimizer
may use, so every method is compared on an identical objective and an identical
greedy assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .assignment import AssignmentResult, greedy_assignment, greedy_assignment_batch
from .instance import ProblemInstance

if TYPE_CHECKING:  # pragma: no cover
    from .link import LinkModel

_EPS = 1e-9


@dataclass
class FitnessBreakdown:
    """Per-evaluation diagnostics returned by :meth:`Fitness.components`."""

    fitness: float
    f_cover: float
    f_cover_norm: float
    d_move: float
    d_move_norm: float
    l_imb: float
    l_imb_norm: float
    n_assigned: int
    assignment: AssignmentResult


class Fitness:
    """Callable objective bound to one :class:`ProblemInstance`.

    Tracks an evaluation counter so the runner can verify every metaheuristic
    spends the same budget. Maximization: higher is better.
    """

    def __init__(
        self,
        instance: ProblemInstance,
        # Paper §IV-E1 states 0.6/0.3/0.1; these defaults are instead the
        # 2026-07-20 Optuna weight search's result (scripts/tune_weights.py) —
        # coverage should dominate far more, movement penalty far less, than
        # the paper's stated split. Only affects callers that don't pass
        # explicit weights (run_full_hfl's _place_uavs); Tier-1 configs
        # (tier1_core.yaml, smoke.yaml) always pass explicit w1/w2/w3 and are
        # unaffected by this default.
        w1: float = 0.811,
        w2: float = 0.03,
        w3: float = 0.159,
        link: "LinkModel | None" = None,
    ) -> None:
        self.instance = instance
        self.w1, self.w2, self.w3 = w1, w2, w3
        # When present, coverage is gated on the radius each UAV's own altitude
        # earns under the shared air-to-ground channel instead of a flat R_comm.
        # This is what makes altitude a real decision (see uavbench.problem.link);
        # None preserves the legacy hard range gate exactly.
        self.link = link
        self.f_max = max(float(instance.value.sum()), _EPS)
        self.d_max = max(instance.K * instance.box_diagonal, _EPS)
        self.l_max = max(float(instance.N) ** 2, _EPS)
        self.eval_count = 0

    def components(self, x: np.ndarray, radii: np.ndarray | None = None) -> FitnessBreakdown:
        """Evaluate a candidate and return the full breakdown (counts one eval).

        ``radii`` optionally supplies a per-position ``(K,)`` communication
        radius overriding the scalar ``instance.R_comm`` (see
        :func:`greedy_assignment`).
        """
        self.eval_count += 1
        inst = self.instance
        positions = inst.positions_from_vector(x)

        if radii is None and self.link is not None:
            radii = self.link.slant_radius(positions[:, 2])
        res = greedy_assignment(inst, positions, radii=radii)

        d_move = float(np.sum(np.sqrt(np.sum((positions - inst.prev_positions) ** 2, axis=1))))

        mean_load = res.n_assigned / inst.K
        l_imb = float(np.sum((res.loads - mean_load) ** 2))

        f_cover_norm = res.f_cover / self.f_max
        d_move_norm = d_move / self.d_max
        l_imb_norm = l_imb / self.l_max

        fitness = self.w1 * f_cover_norm - self.w2 * d_move_norm - self.w3 * l_imb_norm
        return FitnessBreakdown(
            fitness=float(fitness),
            f_cover=res.f_cover,
            f_cover_norm=f_cover_norm,
            d_move=d_move,
            d_move_norm=d_move_norm,
            l_imb=l_imb,
            l_imb_norm=l_imb_norm,
            n_assigned=res.n_assigned,
            assignment=res,
        )

    def __call__(self, x: np.ndarray, radii: np.ndarray | None = None) -> float:
        """Return the scalar fitness of candidate ``x`` (counts one evaluation)."""
        return self.components(x, radii=radii).fitness

    def batch(self, X: np.ndarray, radii: np.ndarray | None = None) -> np.ndarray:
        """Score a whole ``(P, 3K)`` population; returns ``(P,)``, counts P evals.

        Bit-identical to ``np.array([self(X[i]) for i in range(P)])`` and 5-7x
        faster: the greedy sweep is shared across candidates (see
        :func:`greedy_assignment_batch`) while the *scalarization* stays a
        per-candidate loop over the identical expressions.

        That last part is not incidental. Vectorizing the tail as well —
        ``f_cover`` by accumulating value in descending-value order, ``d_move``
        and ``l_imb`` by summing along an added axis — changes floating-point
        summation order and shifts results by ~1 ULP. The reductions here are
        short (K terms) and run once per candidate rather than once per device,
        so keeping them scalar costs almost nothing and buys exactness.
        """
        inst = self.instance
        K = inst.K
        pos = np.asarray(X, dtype=np.float64).reshape(-1, K, 3)
        P = pos.shape[0]
        self.eval_count += P

        if radii is None and self.link is not None:
            # (P, K): candidates in a batch fly at different altitudes, so each
            # gets its own radius vector rather than one shared across the batch.
            radii = self.link.slant_radius(pos[:, :, 2])
        assignment, loads = greedy_assignment_batch(inst, pos, radii=radii)

        out = np.empty(P, dtype=np.float64)
        for p in range(P):
            assigned_mask = assignment[p] >= 0
            f_cover = float(inst.value[assigned_mask].sum())
            n_assigned = int(assigned_mask.sum())
            d_move = float(
                np.sum(np.sqrt(np.sum((pos[p] - inst.prev_positions) ** 2, axis=1)))
            )
            l_imb = float(np.sum((loads[p] - n_assigned / K) ** 2))
            out[p] = (
                self.w1 * (f_cover / self.f_max)
                - self.w2 * (d_move / self.d_max)
                - self.w3 * (l_imb / self.l_max)
            )
        return out


def fitness_components(
    instance: ProblemInstance,
    x: np.ndarray,
    w1: float = 0.6,
    w2: float = 0.3,
    w3: float = 0.1,
    radii: np.ndarray | None = None,
) -> FitnessBreakdown:
    """Convenience one-shot breakdown without persisting an eval counter."""
    return Fitness(instance, w1, w2, w3).components(x, radii=radii)
