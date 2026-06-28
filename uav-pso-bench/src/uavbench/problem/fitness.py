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

import numpy as np

from .assignment import AssignmentResult, greedy_assignment
from .instance import ProblemInstance

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
        w1: float = 0.6,
        w2: float = 0.3,
        w3: float = 0.1,
    ) -> None:
        self.instance = instance
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.f_max = max(float(instance.value.sum()), _EPS)
        self.d_max = max(instance.K * instance.box_diagonal, _EPS)
        self.l_max = max(float(instance.N) ** 2, _EPS)
        self.eval_count = 0

    def components(self, x: np.ndarray) -> FitnessBreakdown:
        """Evaluate a candidate and return the full breakdown (counts one eval)."""
        self.eval_count += 1
        inst = self.instance
        positions = inst.positions_from_vector(x)

        res = greedy_assignment(inst, positions)

        d_move = float(
            np.sum(np.sqrt(np.sum((positions - inst.prev_positions) ** 2, axis=1)))
        )

        mean_load = res.n_assigned / inst.K
        l_imb = float(np.sum((res.loads - mean_load) ** 2))

        f_cover_norm = res.f_cover / self.f_max
        d_move_norm = d_move / self.d_max
        l_imb_norm = l_imb / self.l_max

        fitness = (
            self.w1 * f_cover_norm - self.w2 * d_move_norm - self.w3 * l_imb_norm
        )
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

    def __call__(self, x: np.ndarray) -> float:
        """Return the scalar fitness of candidate ``x`` (counts one evaluation)."""
        return self.components(x).fitness


def fitness_components(
    instance: ProblemInstance, x: np.ndarray, w1: float = 0.6, w2: float = 0.3, w3: float = 0.1
) -> FitnessBreakdown:
    """Convenience one-shot breakdown without persisting an eval counter."""
    return Fitness(instance, w1, w2, w3).components(x)
