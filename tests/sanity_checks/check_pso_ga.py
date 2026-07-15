"""PSO/GA optimizer invariants: constriction gate, monotone gbest, bounds, budget."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from _lib import check, finish

from uavbench.optimizers.ga import GA
from uavbench.optimizers.pso import PSO, constriction_factor
from uavbench.problem.fitness import Fitness
from uavbench.problem.instance import generate_instance

AREA = {"x": [0.0, 1000.0], "y": [0.0, 1000.0], "z": [20.0, 120.0]}


def _instance(seed=0):
    return generate_instance("uniform", N=30, K=3, area=AREA, seed=seed)


def constriction_gate():
    assert abs(constriction_factor(4.1) - 0.7298) < 1e-3
    for phi in (4.0, 3.55, 0.0, -1.0):
        try:
            constriction_factor(phi)
        except ValueError:
            pass
        else:
            raise AssertionError(f"phi={phi} must be rejected (Clerc & Kennedy validity gate)")


def gbest_monotone_and_bounds():
    inst = _instance()
    fit = Fitness(inst)
    res = PSO(P=20, G_max=30).optimize(inst, fit, np.random.default_rng(1))
    conv = np.array(res.convergence)
    assert np.all(np.diff(conv) >= -1e-12)  # never worsens
    assert abs(res.best_fitness - conv[-1]) < 1e-12
    lo, hi = np.tile(inst.lower, inst.K), np.tile(inst.upper, inst.K)
    assert np.all(res.best_position >= lo - 1e-9) and np.all(res.best_position <= hi + 1e-9)


def eval_budget_recorded():
    inst = _instance()
    fit = Fitness(inst)
    res = PSO(P=20, G_max=10, use_stagnation=False).optimize(inst, fit, np.random.default_rng(4))
    assert res.eval_count == fit.eval_count > 0


def ga_runs_and_improves():
    inst = _instance(seed=2)
    fit = Fitness(inst)
    res = GA(P=20, G_max=20).optimize(inst, fit, np.random.default_rng(3))
    conv = np.array(res.convergence)
    assert np.all(np.diff(conv) >= -1e-12)
    lo, hi = np.tile(inst.lower, inst.K), np.tile(inst.upper, inst.K)
    assert np.all(res.best_position >= lo - 1e-9) and np.all(res.best_position <= hi + 1e-9)


check("constriction factor value + phi<=4 rejection", constriction_gate)
check("PSO gbest monotone non-decreasing, bounds respected", gbest_monotone_and_bounds)
check("PSO eval count equals fitness eval count", eval_budget_recorded)
check("GA elitist best monotone, bounds respected", ga_runs_and_improves)
finish()
