"""Capacitated max-covering location (MCLP) reference for UAV placement.

Gives a near-optimal *coverage* reference the metaheuristics can be measured
against: over a candidate grid of UAV sites, choose K sites and assign covered
clients (each site serving at most ``capacity``) to maximize covered value. The
scalarized fitness is coverage-dominated (w1=0.81), so covered value is the term
that matters. Solved exactly with SciPy's bundled HiGHS MILP — no extra
dependency. Because sites are a finite grid, the MCLP optimum is a *lower bound*
on the continuous optimum; with a dense grid it closely approximates it, so
"PSO reaches X% of the MCLP coverage" is a rigorous near-optimality statement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .instance import ProblemInstance


@dataclass
class MCLPResult:
    covered_value: float          # sum of assigned client values (the F_cover term)
    covered_value_norm: float     # covered_value / total value  (== f_cover_norm scale)
    n_covered: int
    optimal: bool                 # True if HiGHS proved optimality within the time limit
    n_sites: int


def _candidate_sites(instance: ProblemInstance, grid_res: int) -> np.ndarray:
    """(M, 3) candidate UAV positions on a grid over the box at mid-altitude."""
    lo, hi = instance.lower, instance.upper
    xs = np.linspace(lo[0], hi[0], grid_res)
    ys = np.linspace(lo[1], hi[1], grid_res)
    gx, gy = np.meshgrid(xs, ys)
    z = 0.5 * (lo[2] + hi[2])
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])


def mclp_reference(
    instance: ProblemInstance,
    grid_res: int = 20,
    radii: np.ndarray | None = None,
    time_limit: float = 120.0,
    max_clients: int = 100000,
) -> MCLPResult:
    """Solve the capacitated MCLP coverage reference for ``instance``.

    ``max_clients`` subsamples clients (value-weighted-agnostic, deterministic)
    when N is large, keeping the MILP tractable; the covered fraction is reported
    on the solved subset. ``radii`` optionally overrides ``instance.R_comm`` per
    candidate site is not meaningful here (sites are a grid), so a scalar range
    is used: ``max(radii)`` if given else ``instance.R_comm``.
    """
    coords = instance.device_coords
    value = np.asarray(instance.value, dtype=np.float64)
    n = coords.shape[0]
    if n > max_clients:
        rng = np.random.default_rng(0)
        keep = rng.choice(n, size=max_clients, replace=False)
        coords, value = coords[keep], value[keep]
    n = coords.shape[0]

    sites = _candidate_sites(instance, grid_res)
    m = sites.shape[0]
    r = float(np.max(radii)) if radii is not None else float(instance.R_comm)
    cap = float(np.max(instance.capacity))

    # coverage matrix C[i, j] = 1 if client i within r of site j (2D ground range)
    d = np.sqrt(((coords[:, None, :2] - sites[None, :, :2]) ** 2).sum(axis=2))
    cover = (d <= r).astype(np.float64)

    # variables: [y_0..y_{m-1}, x flattened (n*m)]; all binary
    nx = n * m
    n_var = m + nx
    c = np.concatenate([np.zeros(m), -np.repeat(value, m)])  # minimize -covered value

    rows, cols, data, ub = [], [], [], []
    ri = 0
    # (A1) each client assigned at most once: sum_j x_ij <= 1
    for i in range(n):
        for j in range(m):
            rows.append(ri); cols.append(m + i * m + j); data.append(1.0)
        ub.append(1.0); ri += 1
    # (A2) capacity + link: sum_i x_ij - cap * y_j <= 0
    for j in range(m):
        for i in range(n):
            rows.append(ri); cols.append(m + i * m + j); data.append(1.0)
        rows.append(ri); cols.append(j); data.append(-cap)
        ub.append(0.0); ri += 1
    # (A3) at most K sites: sum_j y_j <= K
    for j in range(m):
        rows.append(ri); cols.append(j); data.append(1.0)
    ub.append(float(instance.K)); ri += 1

    from scipy.sparse import coo_matrix

    A = coo_matrix((data, (rows, cols)), shape=(ri, n_var)).tocsr()
    constraints = LinearConstraint(A, -np.inf, np.array(ub))

    # x_ij fixed to 0 where not covered
    x_ub = cover.ravel()
    bounds = Bounds(np.zeros(n_var), np.concatenate([np.ones(m), x_ub]))
    integrality = np.ones(n_var)

    res = milp(
        c, constraints=constraints, integrality=integrality, bounds=bounds,
        options={"time_limit": time_limit, "mip_rel_gap": 1e-4},
    )
    if not res.success or res.x is None:
        return MCLPResult(float("nan"), float("nan"), 0, False, m)
    x = res.x[m:].reshape(n, m)
    assigned = x.sum(axis=1) > 0.5
    covered_value = float(value[assigned].sum())
    total = float(np.asarray(instance.value).sum())
    return MCLPResult(
        covered_value=covered_value,
        covered_value_norm=covered_value / total if total > 0 else 0.0,
        n_covered=int(assigned.sum()),
        optimal=bool(res.status == 0),
        n_sites=m,
    )
