"""Placement fitness and greedy assignment against hand-computed values.
Includes the radii=None == scalar-R_comm regression pin (manual check #3)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from _lib import check, finish

from uavbench.problem.assignment import greedy_assignment
from uavbench.problem.fitness import Fitness
from uavbench.problem.instance import ProblemInstance


def _two_device_one_position(prev_z=0.0):
    return ProblemInstance(
        device_coords=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        value=np.array([1.0, 1.0]),
        capacity=np.array([5.0]),
        battery=np.array([1.0]),
        prev_positions=np.array([[0.0, 0.0, prev_z]]),
        lower=np.array([0.0, 0.0, 0.0]),
        upper=np.array([100.0, 100.0, 100.0]),
    )


def _assignment_instance(value, capacity, battery, R_comm=500.0, B_min_uav=0.2):
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


def fitness_hand_computed():
    # Explicit w1/w2/w3 here, independent of Fitness's own defaults (which
    # get retuned — see problem/fitness.py) — this checks the *formula* is
    # right, not today's production weight values.
    inst = _two_device_one_position()
    fit = Fitness(inst, w1=0.6, w2=0.3, w3=0.1)
    b = fit.components(np.array([0.0, 0.0, 0.0]))  # coincides with prev, covers both
    assert b.f_cover == 2.0 and abs(b.f_cover_norm - 1.0) < 1e-9
    assert abs(b.d_move) < 1e-9 and abs(b.l_imb) < 1e-9
    assert abs(b.fitness - 0.6) < 1e-9  # w1 * 1.0
    # Movement penalty: 10 m up, both still covered.
    b2 = Fitness(_two_device_one_position(), w1=0.6, w2=0.3, w3=0.1).components(
        np.array([0.0, 0.0, 10.0])
    )
    diag = np.sqrt(3 * 100.0**2)
    assert abs(b2.d_move - 10.0) < 1e-9
    assert abs(b2.fitness - (0.6 - 0.3 * 10.0 / diag)) < 1e-9


def load_imbalance_hand_computed():
    inst = ProblemInstance(
        device_coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        value=np.array([1.0, 1.0]),
        capacity=np.array([5.0, 5.0]),
        battery=np.array([1.0, 1.0]),
        prev_positions=np.array([[0.0, 0.0, 0.0], [9000.0, 0.0, 0.0]]),
        lower=np.array([0.0, 0.0, 0.0]),
        upper=np.array([9000.0, 100.0, 100.0]),
    )
    b = Fitness(inst).components(np.array([0.0, 0.0, 0.0, 9000.0, 0.0, 0.0]))
    assert list(b.assignment.loads) == [2, 0]
    assert abs(b.l_imb - 2.0) < 1e-9  # (2-1)^2 + (0-1)^2


def capacity_range_battery_value():
    inst = _assignment_instance(value=[3, 2, 1], capacity=[1, 1], battery=[1, 1])
    res = greedy_assignment(inst, POSITIONS)
    assert res.assignment[2] == -1  # out of R_comm
    assert res.assignment[0] == 0 and res.assignment[1] == 1
    assert res.n_assigned == 2 and res.f_cover == 5.0
    # Battery gate disables position 0.
    res = greedy_assignment(
        _assignment_instance(value=[3, 2, 1], capacity=[1, 1], battery=[0.1, 1.0]), POSITIONS
    )
    assert res.assignment[0] == 1 and res.assignment[1] == -1 and res.f_cover == 3.0
    # Highest value wins a capacity-1 contest.
    inst2 = ProblemInstance(
        device_coords=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        value=np.array([1.0, 9.0]),
        capacity=np.array([1.0]),
        battery=np.array([1.0]),
        prev_positions=np.array([[0.0, 0.0, 0.0]]),
        lower=np.array([0.0, 0.0, 0.0]),
        upper=np.array([100.0, 100.0, 120.0]),
    )
    res2 = greedy_assignment(inst2, np.array([[0.0, 0.0, 0.0]]))
    assert res2.assignment[1] == 0 and res2.assignment[0] == -1 and res2.f_cover == 9.0


def radii_none_matches_scalar_gate():
    # radii=None must reproduce the scalar-R_comm behaviour bit-for-bit —
    # per-UAV radii support for the literature baselines must not perturb
    # PSO/GA (manual sanity check #3).
    inst = _assignment_instance(value=[3, 2, 1], capacity=[1, 1], battery=[1, 1])
    base = greedy_assignment(inst, POSITIONS)
    uniform = greedy_assignment(inst, POSITIONS, radii=np.full(2, inst.R_comm))
    assert list(base.assignment) == list(uniform.assignment)
    assert base.f_cover == uniform.f_cover


def per_uav_radii_change_coverage():
    inst = _assignment_instance(value=[3, 2, 1], capacity=[2, 2], battery=[1, 1])
    assert greedy_assignment(inst, POSITIONS).assignment[2] == -1
    res = greedy_assignment(inst, POSITIONS, radii=np.array([2000.0, 60.0]))
    assert res.assignment[2] == 0 and res.n_assigned == 3
    # Wrong radii shape must raise, not silently broadcast.
    try:
        greedy_assignment(inst, POSITIONS, radii=np.array([500.0]))
    except ValueError as e:
        assert "radii" in str(e)
    else:
        raise AssertionError("expected ValueError for (1,) radii with K=2")


check("fitness components match hand-computed values", fitness_hand_computed)
check("load-imbalance term matches hand-computed value", load_imbalance_hand_computed)
check("assignment respects capacity/range/battery/value priority", capacity_range_battery_value)
check("radii=None reproduces the scalar R_comm gate exactly", radii_none_matches_scalar_gate)
check("per-UAV radii change coverage; wrong shape raises", per_uav_radii_change_coverage)
finish()
