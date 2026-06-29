"""Tier-1 placement metrics derived from an optimizer Result.

Reports both the normalized fitness terms (which sum into the headline number)
and the raw operational quantities (percent covered, Joules, load variance),
because reviewers and practitioners read those differently.
"""

from __future__ import annotations

import numpy as np

from ..optimizers.base import Result
from ..problem.energy import EnergyModel
from ..problem.fitness import Fitness
from ..problem.instance import ProblemInstance


def evals_to_threshold(convergence: list[float], best: float, frac: float = 0.95) -> int:
    """Index of the first iteration reaching ``frac`` of the final best fitness.

    Returned in *iteration* units (the convergence trace is one entry per
    iteration); ``-1`` if never reached (e.g. negative best).
    """
    if best <= 0:
        return -1
    target = frac * best
    conv = np.asarray(convergence)
    hit = np.where(conv >= target)[0]
    return int(hit[0]) if hit.size else -1


# np.trapezoid is the numpy>=2.0 name; np.trapz is the <2.0 spelling.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def convergence_auc(convergence: list[float]) -> float:
    """Area under the best-fitness-vs-iteration curve (trapezoidal, normalized)."""
    conv = np.asarray(convergence, dtype=float)
    if conv.size < 2:
        return float(conv.sum())
    return float(_trapz(conv, dx=1.0) / (conv.size - 1))


def compute_metrics(
    instance: ProblemInstance,
    result: Result,
    fitness_weights: tuple[float, float, float] = (0.6, 0.3, 0.1),
    energy_model: EnergyModel | None = None,
) -> dict:
    """Return a flat dict of Tier-1 metrics for one optimizer run.

    Re-evaluates the returned best position once (on a *fresh* Fitness so the
    run's eval budget is untouched) to recover the coverage / movement / balance
    breakdown and the operational quantities.
    """
    energy_model = energy_model or EnergyModel()
    w1, w2, w3 = fitness_weights
    scorer = Fitness(instance, w1, w2, w3)
    b = scorer.components(result.best_position)

    joules, batt_frac = (
        energy_model.energy_joules(b.d_move),
        energy_model.battery_fraction(b.d_move),
    )

    return {
        "method": result.method,
        "final_fitness": result.best_fitness,
        "f_cover": b.f_cover,
        "f_cover_norm": b.f_cover_norm,
        "coverage_pct": 100.0 * b.n_assigned / instance.N,
        "n_assigned": b.n_assigned,
        "d_move_m": b.d_move,
        "movement_joules": joules,
        "movement_battery_frac": batt_frac,
        "l_imb": b.l_imb,
        "evals_to_threshold_iter": evals_to_threshold(result.convergence, result.best_fitness),
        "convergence_auc": convergence_auc(result.convergence),
        "eval_count": result.eval_count,
        "n_iterations": result.n_iterations,
        "wall_time_s": result.wall_time,
    }
