"""Fitness matches hand-computed values on tiny cases."""

import numpy as np
import pytest

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


def test_perfect_coverage_no_movement():
    inst = _two_device_one_position(prev_z=0.0)
    fit = Fitness(inst)
    # Position coincides with prev (no movement); both devices covered.
    x = np.array([0.0, 0.0, 0.0])
    b = fit.components(x)
    assert b.f_cover == 2.0
    assert b.f_cover_norm == pytest.approx(1.0)
    assert b.d_move == pytest.approx(0.0)
    assert b.l_imb == pytest.approx(0.0)
    assert b.fitness == pytest.approx(0.6)
    assert fit.eval_count == 1


def test_movement_penalty():
    inst = _two_device_one_position(prev_z=0.0)
    fit = Fitness(inst)
    # Move position up 10 m: both devices still in range; d_move = 10.
    x = np.array([0.0, 0.0, 10.0])
    b = fit.components(x)
    diag = np.sqrt(3 * 100.0**2)
    expected = 0.6 * 1.0 - 0.3 * (10.0 / (1 * diag))
    assert b.d_move == pytest.approx(10.0)
    assert b.fitness == pytest.approx(expected)


def test_load_imbalance_term():
    # Two positions; one covers both devices, the other is far out of range.
    inst = ProblemInstance(
        device_coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        value=np.array([1.0, 1.0]),
        capacity=np.array([5.0, 5.0]),
        battery=np.array([1.0, 1.0]),
        prev_positions=np.array([[0.0, 0.0, 0.0], [9000.0, 0.0, 0.0]]),
        lower=np.array([0.0, 0.0, 0.0]),
        upper=np.array([9000.0, 100.0, 100.0]),
    )
    fit = Fitness(inst)
    x = np.array([0.0, 0.0, 0.0, 9000.0, 0.0, 0.0])
    b = fit.components(x)
    assert list(b.assignment.loads) == [2, 0]
    # mean load = 1 -> l_imb = (2-1)^2 + (0-1)^2 = 2
    assert b.l_imb == pytest.approx(2.0)
