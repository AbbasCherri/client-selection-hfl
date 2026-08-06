"""Capacitated max-covering location (MCLP) reference for UAV placement.

Gives a near-optimal *coverage* reference the metaheuristics can be measured
against: over a candidate grid of UAV sites, choose K sites and assign covered
clients (each site serving at most ``capacity``) to maximize covered value. The
scalarized fitness is coverage-dominated (w1=0.81), so covered value is the term
that matters. Solved exactly with SciPy's bundled HiGHS MILP — no extra
dependency.

**What this does and does not bound.** It bounds the ``F_cover`` term only — the
movement and imbalance terms of the scalarized objective are ignored entirely.
With w1=0.811 that is a defensible proxy, but the paper must say "PSO reaches X%
of the optimal *coverage*", never "of the objective".

**Grid restriction.** Sites are a finite grid, so the MCLP optimum is a lower
bound on the continuous optimum. PSO places off-grid and can therefore exceed
100% of it — those cells are not errors, they are the grid bound being loose.
Sweep ``grid_res`` and report the convergence curve, so "X% of optimum" stops
depending on an arbitrary resolution: at grid_res=20 over a 5 km box the
spacing is 263 m against R_comm=500 m, which is coarse.
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
    mip_gap: float = float("nan")  # residual optimality gap; NaN if HiGHS reported none


def _candidate_sites(instance: ProblemInstance, grid_res: int) -> np.ndarray:
    """(M, 3) candidate UAV positions on a grid over the box at mid-altitude."""
    lo, hi = instance.lower, instance.upper
    xs = np.linspace(lo[0], hi[0], grid_res)
    ys = np.linspace(lo[1], hi[1], grid_res)
    gx, gy = np.meshgrid(xs, ys)
    z = 0.5 * (lo[2] + hi[2])
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])


def _cip_sites(instance: ProblemInstance, altitude: float, r: float) -> np.ndarray:
    """(M, 3) exact circle-intersection candidate sites at ``altitude``.

    A grid reference is only a bound on methods that also search the grid. Once
    :mod:`uavbench.optimizers.mclp_ls` places UAVs at exact intersection points,
    a 20x20 grid "optimum" can be *beaten* by the heuristic it is supposed to
    bound — a reference that reports 105% of optimum is worse than none. These
    sites are the provably sufficient set (Church 1984), dominance-pruned so the
    MILP stays closable, so the bound is genuine.
    """
    from ..optimizers.candidates import build_candidate_set, coverage_matrix, prune_dominated

    xy = build_candidate_set(
        instance.device_coords[:, :2], r, instance.lower[:2], instance.upper[:2],
        max_candidates=20000, dedupe_grid_m=1.0,
    )
    keep = prune_dominated(coverage_matrix(xy, instance.device_coords[:, :2], r))
    xy = xy[keep]
    return np.column_stack([xy, np.full(xy.shape[0], altitude)])


def mclp_reference(
    instance: ProblemInstance,
    grid_res: int = 20,
    radii: np.ndarray | None = None,
    time_limit: float = 120.0,
    max_clients: int = 100000,
    sites: str = "grid",
    altitude: float | None = None,
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

    r = float(np.max(radii)) if radii is not None else float(instance.R_comm)
    if sites == "cip":
        # Altitude eats radius under the 3D gate, so the sites are built at the
        # altitude actually flown and the ground radius derived from it.
        z = float(instance.lower[2]) if altitude is None else float(altitude)
        site_xyz = _cip_sites(instance, z, float(np.sqrt(max(r * r - z * z, 0.0))))
    elif sites == "grid":
        site_xyz = _candidate_sites(instance, grid_res)
    else:
        raise ValueError(f"sites must be 'grid' or 'cip'; got {sites!r}")
    m = site_xyz.shape[0]
    # max(), not min() or mean(), and deliberately so: candidate sites are grid
    # points, not specific UAVs, so there is no per-site capacity to use. Taking
    # the largest makes the MILP *stronger* than any real deployment, which
    # pushes the reported "PSO reaches X% of optimum" DOWN. The bound stays
    # honest; it just gets slightly harder for PSO to reach. Only matters when
    # capacities are heterogeneous, which no current config does.
    cap = float(np.max(instance.capacity))

    # Coverage matrix C[i, j] = 1 if client i is within r of site j, using the
    # 3-D slant distance so this matches ProblemInstance.distances exactly.
    # This was 2-D ground range through 2026-07, which at z~70 m / R=500 m gave
    # the MILP a ~1% larger effective footprint than the heuristics it bounds —
    # small, and conservative for PSO%, but not a like-for-like comparison.
    d = np.sqrt(((coords[:, None, :] - site_xyz[None, :, :]) ** 2).sum(axis=2))
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
        return MCLPResult(float("nan"), float("nan"), 0, False, m, float("nan"))
    x = res.x[m:].reshape(n, m)
    assigned = x.sum(axis=1) > 0.5
    covered_value = float(value[assigned].sum())
    # Normalise against the SOLVED set, not instance.value: when max_clients
    # subsamples, dividing a subset's covered value by the full-instance total
    # silently understates coverage. (No current config trips this — N=250 vs a
    # 100k cap — but the ratio would be wrong the first time one did.)
    total = float(value.sum())
    # HiGHS reports the residual optimality gap; a bare `optimal` boolean hides
    # whether the 600 s limit bound before the gap closed.
    gap = float(getattr(res, "mip_gap", float("nan")))
    return MCLPResult(
        covered_value=covered_value,
        covered_value_norm=covered_value / total if total > 0 else 0.0,
        n_covered=int(assigned.sum()),
        optimal=bool(res.status == 0),
        n_sites=m,
        mip_gap=gap,
    )
