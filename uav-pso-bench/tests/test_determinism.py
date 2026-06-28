"""Same seed -> identical results, across optimizers and instance generation."""

import numpy as np

from uavbench.optimizers.ga import GA
from uavbench.optimizers.pso import PSO
from uavbench.problem.fitness import Fitness
from uavbench.problem.instance import generate_instance

AREA = {"x": [0.0, 1000.0], "y": [0.0, 1000.0], "z": [20.0, 120.0]}


def test_instance_generation_deterministic():
    a = generate_instance("clustered", N=50, K=4, area=AREA, seed=7)
    b = generate_instance("clustered", N=50, K=4, area=AREA, seed=7)
    assert np.array_equal(a.device_coords, b.device_coords)
    assert np.array_equal(a.value, b.value)


def test_pso_deterministic():
    inst = generate_instance("uniform", N=30, K=3, area=AREA, seed=1)
    r1 = PSO(P=20, G_max=20).optimize(inst, Fitness(inst), np.random.default_rng(42))
    r2 = PSO(P=20, G_max=20).optimize(inst, Fitness(inst), np.random.default_rng(42))
    assert r1.best_fitness == r2.best_fitness
    assert np.array_equal(r1.best_position, r2.best_position)
    assert r1.convergence == r2.convergence


def test_ga_deterministic():
    inst = generate_instance("uniform", N=30, K=3, area=AREA, seed=1)
    r1 = GA(P=20, G_max=20).optimize(inst, Fitness(inst), np.random.default_rng(5))
    r2 = GA(P=20, G_max=20).optimize(inst, Fitness(inst), np.random.default_rng(5))
    assert r1.best_fitness == r2.best_fitness
    assert np.array_equal(r1.best_position, r2.best_position)
