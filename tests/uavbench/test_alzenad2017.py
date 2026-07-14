"""Alzenad 2017 decoupled-placement baseline: registry, shapes, per-UAV radii."""

import numpy as np

from uavbench.optimizers import REGISTRY, Alzenad2017
from uavbench.problem.fitness import Fitness
from uavbench.problem.instance import generate_instance

AREA = {"x": [0.0, 5000.0], "y": [0.0, 5000.0], "z": [20.0, 120.0]}


def _instance(seed=0, N=60, K=4):
    return generate_instance("clustered", N=N, K=K, area=AREA, seed=seed)


def test_registered():
    assert REGISTRY["alzenad2017"] is Alzenad2017


def test_one_shot_result_contract():
    inst = _instance()
    result = Alzenad2017().optimize(inst, Fitness(inst), np.random.default_rng(0))
    assert result.method == "alzenad2017"
    assert result.n_iterations == 1
    assert result.convergence == [result.best_fitness]
    assert result.eval_count == 1

    positions = inst.positions_from_vector(result.best_position)
    assert positions.shape == (inst.K, 3)
    assert np.all(positions[:, 2] >= inst.lower[2]) and np.all(positions[:, 2] <= inst.upper[2])


def test_meta_radii_per_uav():
    inst = _instance(seed=5, N=120, K=5)
    result = Alzenad2017().optimize(inst, Fitness(inst), np.random.default_rng(0))
    radii = result.meta["radii"]
    assert radii.shape == (inst.K,)
    assert np.all(radii > 0.0)
    # Decoupled per-cluster altitude optimization: altitudes recorded per UAV.
    assert len(result.meta["altitudes_m"]) == inst.K


def test_min_altitude_tracks_cluster_spread():
    # A tighter cluster needs a smaller radius, hence a lower (or equal)
    # energy-efficient altitude than a sprawling one. Build one instance of
    # each and compare the mean chosen altitude.
    tight = generate_instance(
        "epicenter_biased", N=80, K=2,
        area={"x": [0.0, 1000.0], "y": [0.0, 1000.0], "z": [20.0, 500.0]}, seed=2,
    )
    sprawl = generate_instance(
        "uniform", N=80, K=2,
        area={"x": [0.0, 20000.0], "y": [0.0, 20000.0], "z": [20.0, 500.0]}, seed=2,
    )
    opt = Alzenad2017(max_path_loss_db=110.0)
    rng = np.random.default_rng(0)
    z_tight = np.mean(opt.optimize(tight, Fitness(tight), rng).meta["altitudes_m"])
    z_sprawl = np.mean(
        opt.optimize(sprawl, Fitness(sprawl), np.random.default_rng(0)).meta["altitudes_m"]
    )
    assert z_tight <= z_sprawl


def test_covers_devices_on_clustered_instance():
    inst = _instance(seed=7, N=100, K=6)
    fitness = Fitness(inst)
    result = Alzenad2017().optimize(inst, fitness, np.random.default_rng(0))
    b = fitness.components(result.best_position, radii=result.meta["radii"])
    assert b.n_assigned > 0
