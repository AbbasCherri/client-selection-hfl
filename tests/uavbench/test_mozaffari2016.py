"""Mozaffari 2016 circle-packing baseline: registry, shapes, radii, determinism."""

import numpy as np

from uavbench.optimizers import REGISTRY, Mozaffari2016
from uavbench.problem.fitness import Fitness
from uavbench.problem.instance import generate_instance

AREA = {"x": [0.0, 5000.0], "y": [0.0, 5000.0], "z": [20.0, 120.0]}


def _instance(seed=0, N=60, K=4):
    return generate_instance("clustered", N=N, K=K, area=AREA, seed=seed)


def test_registered():
    assert REGISTRY["mozaffari2016"] is Mozaffari2016


def test_one_shot_result_contract():
    inst = _instance()
    result = Mozaffari2016().optimize(inst, Fitness(inst), np.random.default_rng(0))
    assert result.method == "mozaffari2016"
    assert result.n_iterations == 1
    assert result.convergence == [result.best_fitness]
    assert result.eval_count == 1

    positions = inst.positions_from_vector(result.best_position)
    assert positions.shape == (inst.K, 3)
    # Centroid-of-devices centers stay inside the box; altitude in-band.
    assert np.all(positions[:, 0] >= inst.lower[0]) and np.all(positions[:, 0] <= inst.upper[0])
    assert np.all(positions[:, 1] >= inst.lower[1]) and np.all(positions[:, 1] <= inst.upper[1])
    assert np.all(positions[:, 2] >= inst.lower[2]) and np.all(positions[:, 2] <= inst.upper[2])


def test_meta_radii_shared_and_positive():
    inst = _instance()
    result = Mozaffari2016().optimize(inst, Fitness(inst), np.random.default_rng(0))
    radii = result.meta["radii"]
    assert radii.shape == (inst.K,)
    assert np.all(radii > 0.0)
    # Mozaffari packs equal discs: one shared radius across all K UAVs.
    assert np.allclose(radii, radii[0])
    assert result.meta["coverage_radius_m"] == radii[0]
    # The shared altitude matches the placement's z coordinate.
    positions = inst.positions_from_vector(result.best_position)
    assert np.allclose(positions[:, 2], result.meta["altitude_m"])


def test_deterministic_given_instance():
    # The rule is deterministic — the rng argument must not matter.
    inst = _instance(seed=3)
    r1 = Mozaffari2016().optimize(inst, Fitness(inst), np.random.default_rng(1))
    r2 = Mozaffari2016().optimize(inst, Fitness(inst), np.random.default_rng(999))
    assert np.array_equal(r1.best_position, r2.best_position)
    assert r1.best_fitness == r2.best_fitness


def test_covers_devices_on_clustered_instance():
    inst = _instance(seed=7, N=100, K=6)
    fitness = Fitness(inst)
    result = Mozaffari2016().optimize(inst, fitness, np.random.default_rng(0))
    b = fitness.components(result.best_position, radii=result.meta["radii"])
    assert b.n_assigned > 0
