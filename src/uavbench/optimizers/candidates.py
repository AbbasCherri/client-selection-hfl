"""Finite candidate sets for continuous planar coverage placement.

Why a *finite* set suffices
---------------------------
The placement search space is continuous (each UAV is a free point in the box),
which is why every method here is a metaheuristic. But the coverage term is not
continuous in the way that framing suggests: it only changes when a disc of
radius ``r`` crosses a device. Church, "The planar maximal covering location
problem" (J. Regional Science 24(2), 1984), makes this precise — an optimal set
of disc centres for the planar maximal covering problem can always be found
among the **circle intersection points**: the pairwise intersections of the
circles of radius ``r`` drawn around the demand points, together with the demand
points themselves. Any disc can be slid until it is "wedged" against two demand
points (or one, if it covers only one) without ever losing coverage.

So the continuous problem reduces, without loss, to selecting ``K`` points from
an ``O(N^2)`` set. That is what :mod:`.mclp_ls` exploits, and it is the reason a
deterministic local search beats a swarm on this landscape: the swarm spends its
budget rediscovering, approximately, positions that can be written down exactly.

Radius and altitude
-------------------
Coverage is gated on the **3D** distance (devices sit at ``z = 0``, UAVs at
``z > 0``), so a UAV at altitude ``z`` projects a ground disc of radius
``sqrt(R_comm^2 - z^2)``. That is strictly decreasing in ``z``: under the shared
objective the minimum feasible altitude is optimal, and every metre of altitude
is radius spent. The candidate set is therefore always built at a stated ``z``.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9

# Pulling an intersection point a hair toward the midpoint of its two generating
# devices makes both of them land strictly inside the disc, so the `<=` range
# test cannot exclude them on a floating-point tie. 1e-9 of the half-chord is far
# below any physically meaningful distance.
_WEDGE_SHRINK = 1.0 - 1e-9


def effective_radius(R_comm: float, z: float) -> float:
    """Ground-disc radius a UAV at altitude ``z`` projects under the 3D gate."""
    return float(np.sqrt(max(float(R_comm) ** 2 - float(z) ** 2, 0.0)))


def circle_intersection_points(xy: np.ndarray, r: float) -> np.ndarray:
    """The ``(M, 2)`` pairwise circle-intersection points at radius ``r``.

    For every pair of devices closer than ``2r`` the two circles of radius ``r``
    around them meet at two points; each is a position "wedged" against both.
    Pairs farther apart than ``2r`` cannot be covered by one disc and contribute
    nothing.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if r <= 0.0 or xy.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64)

    diff = xy[:, None, :] - xy[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(xy.shape[0], k=1)
    d_pair = d[iu]
    keep = (d_pair > _EPS) & (d_pair <= 2.0 * r)
    if not keep.any():
        return np.empty((0, 2), dtype=np.float64)

    i, j = iu[0][keep], iu[1][keep]
    a, b, dd = xy[i], xy[j], d_pair[keep]
    mid = 0.5 * (a + b)
    half_chord = np.sqrt(np.maximum(r * r - 0.25 * dd * dd, 0.0))
    u = (b - a) / dd[:, None]
    perp = np.column_stack([-u[:, 1], u[:, 0]])
    offset = (_WEDGE_SHRINK * half_chord)[:, None] * perp
    return np.vstack([mid + offset, mid - offset])


def build_candidate_set(
    device_xy: np.ndarray,
    r: float,
    lower_xy: np.ndarray,
    upper_xy: np.ndarray,
    *,
    max_candidates: int = 4000,
    dedupe_grid_m: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return ``(M, 2)`` candidate UAV ground positions, ``M <= max_candidates``.

    Device positions are always kept — they are the single-device wedges and the
    fallback when no pair is within ``2r`` — and the intersection points are
    subsampled only if the cap binds. Subsampling is the one place this routine
    is not exact; it is a compute bound, not a modelling choice, and is reported
    in the result meta so a run that hit the cap is visible rather than silent.
    """
    device_xy = np.asarray(device_xy, dtype=np.float64)
    cips = circle_intersection_points(device_xy, r)
    pts = np.vstack([device_xy, cips]) if cips.size else device_xy.copy()

    pts = np.clip(pts, lower_xy, upper_xy)

    # Dedupe on a grid: near-coincident wedges cover identical device sets, so
    # keeping both only inflates the screening cost.
    if dedupe_grid_m > 0.0:
        keys = np.round(pts / dedupe_grid_m).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        pts = pts[np.sort(first)]

    n_dev = device_xy.shape[0]
    if pts.shape[0] > max_candidates:
        # Devices occupy the leading rows only until the dedupe reorders nothing
        # (np.sort(first) preserves original order), so the first n_dev rows are
        # still the surviving device points.
        head = pts[:n_dev]
        tail = pts[n_dev:]
        n_take = max(max_candidates - head.shape[0], 0)
        if n_take and tail.shape[0] > n_take:
            gen = rng if rng is not None else np.random.default_rng(0)
            idx = np.sort(gen.choice(tail.shape[0], size=n_take, replace=False))
            tail = tail[idx]
        pts = np.vstack([head, tail])[:max_candidates]
    return pts


def coverage_matrix(
    candidates_xy: np.ndarray,
    device_xy: np.ndarray,
    r: float,
    *,
    chunk: int = 512,
) -> np.ndarray:
    """``(M, N)`` boolean: does candidate ``m`` cover device ``n`` at radius ``r``?

    Chunked over candidates so the ``(M, N, 2)`` difference tensor is never
    materialized — at M=4000, N=200 that temporary alone is 12.8 MB per copy and
    grows quadratically with the candidate cap.
    """
    cand = np.asarray(candidates_xy, dtype=np.float64)
    dev = np.asarray(device_xy, dtype=np.float64)
    M = cand.shape[0]
    out = np.empty((M, dev.shape[0]), dtype=bool)
    r2 = r * r
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        dx = cand[s:e, None, 0] - dev[None, :, 0]
        dy = cand[s:e, None, 1] - dev[None, :, 1]
        out[s:e] = (dx * dx + dy * dy) <= r2
    return out


def prune_dominated(cover: np.ndarray, *, chunk: int = 256) -> np.ndarray:
    """Indices of candidates whose covered device set is not contained in another's.

    Exact reduction, but **only for a formulation that allows several UAVs at one
    site**. If candidate ``i`` covers a subset of what ``j`` covers, a solution
    stationing UAVs at ``i`` can move them to ``j`` and serve the same devices —
    provided ``j`` can hold them. Under a one-UAV-per-site model this is false
    and the pruning is destructive: a capacitated instance needing two UAVs on
    one dense cluster loses that option, and the resulting "optimum" can fall
    *below* the heuristics it is meant to bound (measured: 70% of optimum
    reported against heuristics reaching 100%). See the multiplicity note in
    :func:`uavbench.problem.exact.mclp_reference`.

    Ties (identical covered sets) keep the lowest index.
    """
    cover = np.asarray(cover)
    M = cover.shape[0]
    sizes = cover.sum(axis=1).astype(np.int64)
    cf = cover.astype(np.float32)
    keep = np.ones(M, dtype=bool)
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        inter = (cf[s:e] @ cf.T).astype(np.int64)  # (chunk, M) intersection sizes
        contained = inter == sizes[s:e, None]  # i's set is inside j's
        # Strictly larger dominates; equal sets are broken by index so exactly
        # one survivor remains rather than both eliminating each other.
        strictly = sizes[None, :] > sizes[s:e, None]
        equal_later = (sizes[None, :] == sizes[s:e, None]) & (
            np.arange(M)[None, :] < np.arange(s, e)[:, None]
        )
        keep[s:e] = ~(contained & (strictly | equal_later)).any(axis=1)
    return np.flatnonzero(keep)


def capped_covered_value(
    cover_sorted: np.ndarray,
    value_sorted: np.ndarray,
    capacity: float,
) -> np.ndarray:
    """Per-candidate value of the ``capacity`` most valuable devices it covers.

    ``cover_sorted``/``value_sorted`` must already be ordered by **descending
    device value**, which makes "the top-``capacity`` covered devices" simply the
    first ``capacity`` True entries of each row — a running count and a mask,
    with no per-candidate sort. This matters because it is the screening score
    evaluated over every candidate on every local-search step.

    Returns ``(M,)``.
    """
    cnt = np.cumsum(cover_sorted, axis=1)
    take = cover_sorted & (cnt <= capacity)
    return take @ value_sorted
