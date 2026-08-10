"""Greedy value-sorted, capacity-aware device-to-position assignment.

This is the inner routine run once per fitness evaluation (PSO guide Section 5).
Hard constraints (range, capacity, battery) are enforced *implicitly*: a device
that cannot be feasibly placed simply earns no coverage credit — there is no
penalty term and no repair operator, which keeps the fitness landscape clean.

Cost: O(N log N) to sort by value + O(N*K) for the assignment sweep, with the
distance/feasibility matrix computed in vectorized NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instance import ProblemInstance


@dataclass
class AssignmentResult:
    """Outcome of a greedy assignment.

    Attributes
    ----------
    assignment:
        ``(N,)`` int array; ``assignment[i]`` is the position serving device ``i``
        or ``-1`` if uncovered.
    loads:
        ``(K,)`` int array of device counts per position.
    f_cover:
        Sum of ``V_i`` over assigned devices.
    n_assigned:
        Total number of devices assigned.
    f_cover_reachable:
        Sum of ``V_i`` over devices some UAV could serve — i.e. in range and on
        a live battery — **ignoring capacity**. Always ``>= f_cover``.

        Recorded because the two diverge in a way that matters. ``f_cover``
        saturates once ``K * capacity`` devices are assigned, so a placement
        objective built on it has no reason to reach a device it cannot serve
        this round. But selection re-runs every round and the roster rotates, so
        across a 100-round run the *reachable* population is what bounds the
        data the system can ever train on. Optimising the one-round quantity for
        a many-round objective is why the v5 placement stopped within a few
        clients of the slot budget while a literature baseline that ignored the
        objective covered 2x as many and won downstream — see
        REPORTS/preregistration_v6_method.md §2.
    n_reachable:
        Count of those devices.
    f_cover_disjoint:
        ``sum(V_i / m_i)`` over reachable devices, where ``m_i`` is how many live
        UAVs reach device ``i``. A device contributes its full value when one
        aircraft reaches it and a shared fraction when several do.

        Always ``<= f_cover_reachable``, with equality **iff** the layout is
        disjoint — so it is reachable coverage with a redundancy discount, and it
        reduces to ``f_cover_reachable`` on any tiling.

        Why it exists: ``f_cover_reachable`` counts each device once, so it never
        double-*rewards* overlap — but it never *penalises* it either, and a
        redundant layout scores identically to a tiling reaching the same set.
        Measured 2026-08-10 (`results/c3_screen`): at K=20 `moon2022` reaches 85%
        of its covered devices with exactly one aircraft and beats every
        fitness-optimising method downstream, while `mclp_ls` manages 60% and
        double-covers at multiplicity 1.53. Redundancy is not free — two aircraft
        over the same clients spend the same ``K*capacity`` slots on fewer
        distinct ones. See REPORTS/preregistration_v6_c3.md §5a.
    """

    assignment: np.ndarray
    loads: np.ndarray
    f_cover: float
    n_assigned: int
    f_cover_reachable: float = 0.0
    n_reachable: int = 0
    f_cover_disjoint: float = 0.0


def _feasible_static(
    instance: ProblemInstance,
    dist: np.ndarray,
    radii: np.ndarray | None,
) -> np.ndarray:
    """Range + battery feasibility mask, broadcast over any leading axes of ``dist``.

    ``radii`` may be ``(K,)`` — one radius per UAV, shared by every candidate —
    or ``(P, K)``, one per UAV *per candidate*. The latter is what an
    altitude-dependent link model needs: candidates in a batch fly at different
    altitudes, so they do not share a radius vector. ``dist`` is ``(N, K)`` in
    the scalar path and ``(P, N, K)`` in the batch path, so a ``(P, K)`` radius
    is reshaped to ``(P, 1, K)`` to broadcast against the device axis.
    """
    K = instance.K
    if radii is None:
        in_range = dist <= instance.R_comm
    else:
        r = np.asarray(radii, dtype=np.float64)
        if r.shape == (K,):
            in_range = dist <= r
        elif r.ndim == 2 and r.shape[1] == K and dist.ndim == 3 and r.shape[0] == dist.shape[0]:
            in_range = dist <= r[:, None, :]
        else:
            raise ValueError(
                f"radii must have shape (K,)=({K},) or (P, K) matching dist "
                f"{dist.shape}; got {r.shape}"
            )
    return in_range & (instance.battery >= instance.B_min_uav)


def greedy_assignment_batch(
    instance: ProblemInstance,
    positions: np.ndarray,
    radii: np.ndarray | None = None,
    return_reachable: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Greedy-assign a whole ``(P, K, 3)`` population at once.

    Returns ``(assignment, loads)`` of shapes ``(P, N)`` and ``(P, K)``, each row
    exactly what :func:`greedy_assignment` produces for that candidate.

    ``return_reachable`` appends two ``(P,)`` arrays: capacity-free reachable
    value and its redundancy-discounted counterpart — the batch versions of
    ``AssignmentResult.f_cover_reachable`` and ``.f_cover_disjoint``.
    Both are computed from the same feasibility mask the assignment uses, so no
    two of them can disagree, and they are **off by default** because the
    reductions are pure overhead for callers scoring the assigned-coverage
    objective. Neither touches ``assignment`` or ``loads``, so enabling them
    cannot perturb a result (the bit-identical guarantee in this module's
    history still holds).

    The device sweep cannot vectorize over *devices* — each one's feasible set
    depends on the loads left by every higher-value device before it. It can
    vectorize over *candidates*, which are independent: the outer loop still
    visits devices in the same descending-value order, but every step now
    operates on ``(P, K)`` arrays instead of ``(K,)``. Same number of Python
    iterations, ~P times fewer NumPy calls, and the per-candidate arithmetic and
    tie-breaking are untouched — ``np.lexsort(..., axis=-1)`` sorts each row
    independently with the same stable order, so row ``p`` picks the same UAV
    the scalar path picks. Measured 5.3-7.5x on the PSO/GA fitness loop.

    (Replacing the lexsort itself with a masked two-stage argmin is a known dead
    end — selection-identical but slower at K=20. This keeps it, just P-wide.)
    """
    N, K = instance.N, instance.K
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, K, 3)
    P = pos.shape[0]

    dist = instance.distances_batch(pos)  # (P, N, K)
    feasible_static = _feasible_static(instance, dist, radii)  # (P, N, K)

    capacity = instance.capacity
    loads = np.zeros((P, K), dtype=np.int64)
    loads_f = np.zeros((P, K), dtype=np.float64)
    not_full = np.ones((P, K), dtype=bool)
    assignment = np.full((P, N), -1, dtype=np.int64)

    big = np.inf
    rows = np.arange(P)

    for i in instance.value_order:
        feas = feasible_static[:, i, :] & not_full  # (P, K)
        placeable = feas.any(axis=1)
        if not placeable.any():
            continue
        load_key = np.where(feas, loads_f, big)
        dist_key = np.where(feas, dist[:, i, :], big)
        j = np.lexsort((dist_key, load_key), axis=-1)[:, 0]  # (P,)
        # Candidates with no feasible UAV are dropped, exactly as the scalar
        # path's `continue` does; `j` is meaningless for them.
        pm, jm = rows[placeable], j[placeable]
        assignment[pm, i] = jm
        loads[pm, jm] += 1
        loads_f[pm, jm] += 1.0
        not_full[pm, jm] = loads[pm, jm] < capacity[jm]

    if return_reachable:
        # (P, N) reachable mask -> (P,) value. Same mask the sweep consumed.
        f_reach = feasible_static.any(axis=2).astype(np.float64) @ instance.value
        # Redundancy-discounted coverage from the SAME mask, so the two scoring
        # paths cannot drift — the failure mode C1 had to be guarded against.
        mult = feasible_static.sum(axis=2)                       # (P, N)
        share = np.divide(
            instance.value, mult, out=np.zeros_like(mult, dtype=np.float64),
            where=mult > 0,
        )
        f_disj = share.sum(axis=1)                               # (P,)
        return assignment, loads, f_reach, f_disj
    return assignment, loads


def greedy_assignment(
    instance: ProblemInstance,
    positions: np.ndarray,
    radii: np.ndarray | None = None,
) -> AssignmentResult:
    """Assign devices to ``positions`` greedily by descending value.

    For each device (highest value first): feasible positions are those in range
    (``distance <= R_comm``), not at capacity, and with battery ``>= B_min_uav``.
    The device goes to the feasible position with the smallest current load, ties
    broken by smallest distance.

    ``radii`` optionally overrides the shared scalar ``instance.R_comm`` with a
    per-position ``(K,)`` communication radius (meters), for placement methods
    whose coverage radius is derived from each UAV's own altitude (e.g. the
    path-loss-based literature baselines). ``None`` preserves the scalar gate.
    """
    N, K = instance.N, instance.K
    dist = instance.distances(positions)  # (N, K)
    feasible_static = _feasible_static(instance, dist, radii)  # (N, K), capacity live

    capacity = instance.capacity
    loads = np.zeros(K, dtype=np.int64)
    # float64 mirror of `loads`, kept in step so the per-device lexsort key needs
    # no `astype` (that cast ran once per device per evaluation — ~10^6 times per
    # placement). `not_full` likewise replaces recomputing `loads < capacity`
    # for every device when at most one entry can have changed.
    loads_f = np.zeros(K, dtype=np.float64)
    not_full = np.ones(K, dtype=bool)
    assignment = np.full(N, -1, dtype=np.int64)

    big = np.inf
    feas = np.empty(K, dtype=bool)
    load_key = np.empty(K, dtype=np.float64)
    dist_key = np.empty(K, dtype=np.float64)

    for i in instance.value_order:  # descending value, cached on the instance
        np.logical_and(feasible_static[i], not_full, out=feas)
        if not feas.any():
            continue
        # Smallest load, ties -> smallest distance, via lexsort (last key primary).
        np.copyto(load_key, big)
        np.copyto(load_key, loads_f, where=feas)
        np.copyto(dist_key, big)
        np.copyto(dist_key, dist[i], where=feas)
        # lexsort: last key is primary -> use load primary, distance secondary.
        j = int(np.lexsort((dist_key, load_key))[0])
        assignment[i] = j
        loads[j] += 1
        loads_f[j] += 1.0
        if loads[j] >= capacity[j]:
            not_full[j] = False

    assigned_mask = assignment >= 0
    f_cover = float(instance.value[assigned_mask].sum())
    n_assigned = int(assigned_mask.sum())
    # Capacity-free reachability, from the same feasibility mask the assignment
    # used, so the two cannot drift apart. See AssignmentResult.f_cover_reachable.
    mult = feasible_static.sum(axis=1)          # (N,) UAVs reaching each device
    reachable_mask = mult > 0
    f_cover_disjoint = float(
        (instance.value[reachable_mask] / mult[reachable_mask]).sum()
    )
    return AssignmentResult(
        assignment=assignment,
        loads=loads,
        f_cover=f_cover,
        n_assigned=n_assigned,
        f_cover_reachable=float(instance.value[reachable_mask].sum()),
        n_reachable=int(reachable_mask.sum()),
        f_cover_disjoint=f_cover_disjoint,
    )
