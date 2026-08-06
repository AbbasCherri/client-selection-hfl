"""Per-UAV altitude optimization, shared by every placement method.

Placement here is a **3D** deployment: each UAV gets its own ``(x, y, z)``. The
metaheuristics search all ``3K`` coordinates and so choose altitude by
construction, and the path-loss baselines derive it from their propagation model.
The constructive/clustering methods have no altitude rule of their own, and
pinning them to a constant would make them 2D placements competing in a 3D
benchmark — their ``z`` would be the author's choice rather than the method's.

This resolves that uniformly: coordinate descent over UAVs, each altitude scored
on the **shared** objective with every other UAV held fixed, so the altitude a
method flies is one the benchmark's own objective selected.

A caveat worth stating plainly, because it shapes what the altitude column can
show. Coverage is a 3D range gate against ground devices, so a UAV at altitude
``z`` projects a ground disc of radius ``sqrt(R_comm^2 - z^2)`` — altitude buys
nothing and costs radius. Under this objective a genuine 3D search will therefore
drive ``z`` toward its lower bound, and the reported altitudes are a real
optimization outcome rather than a modelling artifact. Making altitude a live
trade-off needs an altitude-dependent link model (LoS probability rising with
elevation angle, as in :mod:`uavbench.problem.path_loss`) folded into the shared
coverage test; that is a change to the objective every method is scored on, not
to any one method.
"""

from __future__ import annotations

import numpy as np

from ..problem.fitness import Fitness
from ..problem.instance import ProblemInstance


def optimize_altitudes(
    instance: ProblemInstance,
    fitness: Fitness,
    positions: np.ndarray,
    *,
    n_levels: int = 9,
    n_passes: int = 1,
    max_evals: int | None = None,
) -> np.ndarray:
    """Return ``positions`` with each UAV's ``z`` chosen on the shared objective.

    Coordinate descent: one UAV at a time, ``n_levels`` altitudes spanning the
    instance's band, others held at their current values. Never returns a layout
    worse than the input — each step keeps the incumbent unless a level strictly
    improves it — so this can be applied to any method without risk of
    degrading it.

    ``max_evals`` caps the spend for budgeted methods; the descent stops cleanly
    part-way rather than overrunning, since a partial pass is still an
    improvement.
    """
    pos = np.asarray(positions, dtype=np.float64).reshape(instance.K, 3).copy()
    z_lo, z_hi = float(instance.lower[2]), float(instance.upper[2])
    if n_levels < 2 or z_hi <= z_lo:
        return pos

    levels = np.linspace(z_lo, z_hi, n_levels)
    best = float(fitness(pos.reshape(-1)))
    spent = 1

    for _ in range(n_passes):
        changed = False
        for j in range(instance.K):
            if max_evals is not None and spent + n_levels > max_evals:
                return pos
            trials = np.repeat(pos[None, :, :], n_levels, axis=0)
            trials[:, j, 2] = levels
            scores = fitness.batch(trials)
            spent += n_levels
            b = int(np.argmax(scores))
            if scores[b] > best:
                best = float(scores[b])
                pos = trials[b].copy()
                changed = True
        if not changed:
            break
    return pos
