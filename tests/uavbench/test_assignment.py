"""Greedy assignment respects value order, capacity, range, and battery gates."""

import numpy as np

from uavbench.problem.assignment import greedy_assignment
from uavbench.problem.instance import ProblemInstance


def _instance(value, capacity, battery, R_comm=500.0, B_min_uav=0.2):
    devices = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])
    return ProblemInstance(
        device_coords=devices,
        value=np.array(value, dtype=float),
        capacity=np.array(capacity, dtype=float),
        battery=np.array(battery, dtype=float),
        prev_positions=np.array([[0.0, 0.0, 50.0], [10.0, 0.0, 50.0]]),
        lower=np.array([0.0, 0.0, 0.0]),
        upper=np.array([1000.0, 1000.0, 120.0]),
        R_comm=R_comm,
        B_min_uav=B_min_uav,
    )


POSITIONS = np.array([[0.0, 0.0, 50.0], [10.0, 0.0, 50.0]])


def test_capacity_and_range():
    inst = _instance(value=[3, 2, 1], capacity=[1, 1], battery=[1, 1])
    res = greedy_assignment(inst, POSITIONS)
    # Device 2 is ~1000 m away -> out of R_comm -> uncovered.
    assert res.assignment[2] == -1
    # Highest-value device 0 takes the nearer position 0; device 1 takes position 1.
    assert res.assignment[0] == 0
    assert res.assignment[1] == 1
    assert res.n_assigned == 2
    assert res.f_cover == 5.0
    assert list(res.loads) == [1, 1]


def test_battery_gate_disables_position():
    # Position 0 battery below B_min_uav -> unusable.
    inst = _instance(value=[3, 2, 1], capacity=[1, 1], battery=[0.1, 1.0])
    res = greedy_assignment(inst, POSITIONS)
    assert res.assignment[0] == 1          # device 0 falls back to position 1
    assert res.assignment[1] == -1         # position 1 now full, position 0 unusable
    assert res.assignment[2] == -1
    assert res.f_cover == 3.0


def test_value_sorted_priority():
    # Single capacity-1 position in range of both near devices; highest value wins.
    devices = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    inst = ProblemInstance(
        device_coords=devices,
        value=np.array([1.0, 9.0]),
        capacity=np.array([1.0]),
        battery=np.array([1.0]),
        prev_positions=np.array([[0.0, 0.0, 0.0]]),
        lower=np.array([0.0, 0.0, 0.0]),
        upper=np.array([100.0, 100.0, 120.0]),
    )
    res = greedy_assignment(inst, np.array([[0.0, 0.0, 0.0]]))
    assert res.assignment[1] == 0          # the value-9 device is served
    assert res.assignment[0] == -1
    assert res.f_cover == 9.0
