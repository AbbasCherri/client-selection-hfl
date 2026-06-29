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
    """

    assignment: np.ndarray
    loads: np.ndarray
    f_cover: float
    n_assigned: int


def greedy_assignment(instance: ProblemInstance, positions: np.ndarray) -> AssignmentResult:
    """Assign devices to ``positions`` greedily by descending value.

    For each device (highest value first): feasible positions are those in range
    (``distance <= R_comm``), not at capacity, and with battery ``>= B_min_uav``.
    The device goes to the feasible position with the smallest current load, ties
    broken by smallest distance.
    """
    N, K = instance.N, instance.K
    dist = instance.distances(positions)  # (N, K)

    in_range = dist <= instance.R_comm
    battery_ok = instance.battery >= instance.B_min_uav  # (K,)
    feasible_static = in_range & battery_ok[None, :]      # (N, K), capacity applied live

    capacity = instance.capacity
    loads = np.zeros(K, dtype=np.int64)
    assignment = np.full(N, -1, dtype=np.int64)

    order = np.argsort(-instance.value, kind="stable")  # descending value
    big = np.inf

    for i in order:
        feas = feasible_static[i] & (loads < capacity)
        if not feas.any():
            continue
        # Smallest load, ties -> smallest distance, via lexsort (last key primary).
        load_key = np.where(feas, loads.astype(np.float64), big)
        dist_key = np.where(feas, dist[i], big)
        # lexsort: last key is primary -> use load primary, distance secondary.
        j = int(np.lexsort((dist_key, load_key))[0])
        assignment[i] = j
        loads[j] += 1

    assigned_mask = assignment >= 0
    f_cover = float(instance.value[assigned_mask].sum())
    n_assigned = int(assigned_mask.sum())
    return AssignmentResult(
        assignment=assignment,
        loads=loads,
        f_cover=f_cover,
        n_assigned=n_assigned,
    )
