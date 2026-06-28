"""PSO invariants: gbest monotonic, bounds respected, sane output shape."""

import numpy as np
import pytest

from uavbench.optimizers.pso import PSO, constriction_factor
from uavbench.problem.fitness import Fitness
from uavbench.problem.instance import generate_instance

AREA = {"x": [0.0, 1000.0], "y": [0.0, 1000.0], "z": [20.0, 120.0]}


def _instance(seed=0):
    return generate_instance("uniform", N=30, K=3, area=AREA, seed=seed)


def test_constriction_factor_value():
    assert constriction_factor(4.1) == pytest.approx(0.7298, abs=1e-3)


def test_gbest_monotonic_nondecreasing():
    inst = _instance()
    fit = Fitness(inst)
    pso = PSO(P=20, G_max=30)
    res = pso.optimize(inst, fit, np.random.default_rng(1))
    conv = np.array(res.convergence)
    assert np.all(np.diff(conv) >= -1e-12)  # never worsens
    assert res.best_fitness == pytest.approx(conv[-1])


def test_best_at_least_init():
    inst = _instance()
    fit = Fitness(inst)
    pso = PSO(P=20, G_max=20)
    res = pso.optimize(inst, fit, np.random.default_rng(2))
    assert res.best_fitness >= res.convergence[0] - 1e-12


def test_bounds_respected():
    inst = _instance()
    fit = Fitness(inst)
    pso = PSO(P=20, G_max=25)
    res = pso.optimize(inst, fit, np.random.default_rng(3))
    pos = res.best_position
    lo = np.tile(inst.lower, inst.K)
    hi = np.tile(inst.upper, inst.K)
    assert np.all(pos >= lo - 1e-9)
    assert np.all(pos <= hi + 1e-9)


def test_eval_count_recorded():
    inst = _instance()
    fit = Fitness(inst)
    pso = PSO(P=20, G_max=10, use_stagnation=False)
    res = pso.optimize(inst, fit, np.random.default_rng(4))
    assert res.eval_count == fit.eval_count
    assert res.eval_count > 0
