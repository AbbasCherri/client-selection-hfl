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
    assert res.assignment[0] == 1  # device 0 falls back to position 1
    assert res.assignment[1] == -1  # position 1 now full, position 0 unusable
    assert res.assignment[2] == -1
    assert res.f_cover == 3.0


def test_equal_load_tie_broken_by_distance():
    # Exercise the lexsort's secondary key directly: one device, two empty
    # positions (load tie), one strictly nearer — the nearer must win.
    devices = np.array([[10.0, 0.0, 0.0]])
    inst = ProblemInstance(
        device_coords=devices,
        value=np.array([1.0]),
        capacity=np.array([5.0, 5.0]),
        battery=np.array([1.0, 1.0]),
        prev_positions=np.array([[0.0, 0.0, 50.0], [30.0, 0.0, 50.0]]),
        lower=np.array([0.0, 0.0, 0.0]),
        upper=np.array([1000.0, 1000.0, 120.0]),
    )
    # Device at x=10: pos0 (x=0) is 10 m away horizontally, pos1 (x=30) is
    # 20 m — loads are tied at 0, so the nearer pos0 wins.
    res = greedy_assignment(inst, np.array([[0.0, 0.0, 50.0], [30.0, 0.0, 50.0]]))
    assert res.assignment[0] == 0
    # Mirror: move pos1 closer than pos0 -> pos1 wins the same tie.
    res = greedy_assignment(inst, np.array([[0.0, 0.0, 50.0], [12.0, 0.0, 50.0]]))
    assert res.assignment[0] == 1


def test_radii_none_matches_scalar_r_comm():
    # Explicit regression pin: radii=None must reproduce the scalar-R_comm gate.
    inst = _instance(value=[3, 2, 1], capacity=[1, 1], battery=[1, 1])
    base = greedy_assignment(inst, POSITIONS)
    uniform = greedy_assignment(inst, POSITIONS, radii=np.full(2, inst.R_comm))
    assert list(base.assignment) == list(uniform.assignment)
    assert base.f_cover == uniform.f_cover


def test_per_uav_radii_changes_coverage():
    # 3D distances (devices at z=0, UAVs at z=50): device0->pos0 = 50 m,
    # device1->pos1 = 50 m, device2 ~1000 m from both. Under the scalar
    # R_comm=500 gate device 2 is uncovered; widening position 0's radius to
    # 2000 m brings it into range while position 1's 60 m radius still only
    # admits its adjacent device.
    inst = _instance(value=[3, 2, 1], capacity=[2, 2], battery=[1, 1])
    base = greedy_assignment(inst, POSITIONS)
    assert base.assignment[2] == -1
    res = greedy_assignment(inst, POSITIONS, radii=np.array([2000.0, 60.0]))
    assert res.assignment[0] == 0  # nearer position on the load tie
    assert res.assignment[1] == 1  # smaller load wins once pos0 holds device 0
    assert res.assignment[2] == 0  # only in range of the widened position 0
    assert res.n_assigned == 3


def test_radii_wrong_shape_raises():
    inst = _instance(value=[3, 2, 1], capacity=[1, 1], battery=[1, 1])
    try:
        greedy_assignment(inst, POSITIONS, radii=np.array([500.0]))
    except ValueError as e:
        assert "radii" in str(e)
    else:
        raise AssertionError("expected ValueError for (1,) radii with K=2")


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
    assert res.assignment[1] == 0  # the value-9 device is served
    assert res.assignment[0] == -1
    assert res.f_cover == 9.0
