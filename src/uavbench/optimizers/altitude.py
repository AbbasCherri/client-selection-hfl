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


def optimal_shared_altitude(instance: ProblemInstance, link=None) -> tuple[float, float]:
    """Solve the **vertical** subproblem: the altitude every UAV should fly.

    The 3D placement problem decouples, and exactly. Altitude enters the
    objective through one channel only — it sets the ground radius ``r(z)`` — and
    the coverage term is monotonically non-decreasing in that radius, since a
    larger disc can only enlarge the family of coverable device subsets. Every
    UAV therefore wants the radius-maximizing altitude, no UAV wants any other,
    and

        max_{z, xy}  F(z, xy)   =   max_xy  F_2D( xy ; r(z*) ),
        z* = argmax_z r(z)

    so the vertical subproblem is a 1D unimodal maximization that can be solved
    once, up front, and the remainder is a pure 2D covering problem. This is the
    decoupling Mozaffari (2016) and Alzenad (2017) both assume; here it is a
    consequence of the objective rather than a modelling convenience.

    The decoupling is exact for the coverage term. The movement and imbalance
    terms are altitude-sensitive in principle and could prefer some other ``z``,
    but they carry w2=0.03 and w3=0.159 against w1=0.811 and measure ~0.04 and
    ~0.002 normalized, so they cannot overturn a coverage difference.
    :func:`optimize_altitudes` runs afterwards as a refinement and is expected to
    find nothing — that it finds nothing is the empirical check on this argument.

    Returns ``(z_star_m, ground_radius_m)``.
    """
    if link is not None:
        z = float(link.z_star_m)
        return z, float(link.radius(z))
    # Legacy hard range gate: r(z) = sqrt(R_comm^2 - z^2) is strictly decreasing,
    # so the unimodal maximum sits on the lower bound. Same argument, degenerate
    # channel — and the reason that gate makes altitude a non-decision.
    z = float(instance.lower[2])
    return z, float(np.sqrt(max(instance.R_comm ** 2 - z * z, 0.0)))


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
