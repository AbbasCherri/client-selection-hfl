"""Mozaffari-2016 and Alzenad-2017 literature placement baselines."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from _lib import check, finish

from uavbench.optimizers import REGISTRY, Alzenad2017, Mozaffari2016
from uavbench.problem.fitness import Fitness
from uavbench.problem.instance import generate_instance

AREA = {"x": [0.0, 5000.0], "y": [0.0, 5000.0], "z": [20.0, 120.0]}


def _instance(seed=0, N=60, K=4):
    return generate_instance("clustered", N=N, K=K, area=AREA, seed=seed)


def registered():
    assert REGISTRY["mozaffari2016"] is Mozaffari2016
    assert REGISTRY["alzenad2017"] is Alzenad2017


def one_shot_contract():
    inst = _instance()
    for cls, name in ((Mozaffari2016, "mozaffari2016"), (Alzenad2017, "alzenad2017")):
        result = cls().optimize(inst, Fitness(inst), np.random.default_rng(0))
        assert result.method == name
        assert result.n_iterations == 1 and result.eval_count == 1
        assert result.convergence == [result.best_fitness]
        positions = inst.positions_from_vector(result.best_position)
        assert positions.shape == (inst.K, 3)
        assert np.all(positions[:, 2] >= inst.lower[2]) and np.all(positions[:, 2] <= inst.upper[2])


def mozaffari_shared_radius():
    inst = _instance()
    result = Mozaffari2016().optimize(inst, Fitness(inst), np.random.default_rng(0))
    radii = result.meta["radii"]
    assert radii.shape == (inst.K,) and np.all(radii > 0.0)
    assert np.allclose(radii, radii[0])  # equal-disc packing: one shared radius
    assert result.meta["coverage_radius_m"] == radii[0]
    positions = inst.positions_from_vector(result.best_position)
    assert np.allclose(positions[:, 2], result.meta["altitude_m"])


def alzenad_per_uav_radii():
    inst = _instance(seed=5, N=120, K=5)
    result = Alzenad2017().optimize(inst, Fitness(inst), np.random.default_rng(0))
    radii = result.meta["radii"]
    assert radii.shape == (inst.K,) and np.all(radii > 0.0)
    assert len(result.meta["altitudes_m"]) == inst.K


def deterministic_rules():
    inst = _instance(seed=3)
    for cls in (Mozaffari2016, Alzenad2017):
        r1 = cls().optimize(inst, Fitness(inst), np.random.default_rng(1))
        r2 = cls().optimize(inst, Fitness(inst), np.random.default_rng(999))
        assert np.array_equal(r1.best_position, r2.best_position)


def alzenad_altitude_tracks_spread():
    tight = generate_instance(
        "epicenter_biased", N=80, K=2,
        area={"x": [0.0, 1000.0], "y": [0.0, 1000.0], "z": [20.0, 500.0]}, seed=2,
    )
    sprawl = generate_instance(
        "uniform", N=80, K=2,
        area={"x": [0.0, 20000.0], "y": [0.0, 20000.0], "z": [20.0, 500.0]}, seed=2,
    )
    opt = Alzenad2017(max_path_loss_db=110.0)
    z_tight = np.mean(opt.optimize(tight, Fitness(tight), np.random.default_rng(0)).meta["altitudes_m"])
    z_sprawl = np.mean(opt.optimize(sprawl, Fitness(sprawl), np.random.default_rng(0)).meta["altitudes_m"])
    assert z_tight <= z_sprawl


def both_cover_devices():
    inst = _instance(seed=7, N=100, K=6)
    fitness = Fitness(inst)
    for cls in (Mozaffari2016, Alzenad2017):
        result = cls().optimize(inst, fitness, np.random.default_rng(0))
        assert fitness.components(result.best_position, radii=result.meta["radii"]).n_assigned > 0


check("both baselines registered under their config names", registered)
check("one-shot Result contract (1 iteration, 1 eval, in-band altitude)", one_shot_contract)
check("Mozaffari packs equal discs: one shared radius/altitude", mozaffari_shared_radius)
check("Alzenad records per-UAV radii and altitudes", alzenad_per_uav_radii)
check("both rules deterministic (rng argument irrelevant)", deterministic_rules)
check("Alzenad altitude grows with cluster spread", alzenad_altitude_tracks_spread)
check("both baselines cover devices on a clustered instance", both_cover_devices)
finish()
