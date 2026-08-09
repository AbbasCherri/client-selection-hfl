"""Mozaffari-2016 and Alzenad-2017 literature placement baselines.

Three checks here asserted a ``meta["radii"]`` array that the 2026-08-08
"Rewrite baselines from source PDFs" replaced with ``service_radius_m`` /
``packing_radius_m`` (Mozaffari, equal discs) and ``altitudes_m`` (Alzenad,
per-UAV). They have been failing since, unnoticed, which also means
``runner.py``'s ``result.meta.get("radii")`` has been silently returning None
for both baselines — the per-UAV radius path was dead. Restated against the
current contract 2026-08-09.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from _lib import check, finish

from uavbench.optimizers import REGISTRY, Alzenad2017, Mozaffari2016
from uavbench.problem.fitness import Fitness
from uavbench.problem.instance import generate_instance
from uavbench.problem.link import LinkModel

# Matches the shipped tier1_core band. The old 20-120 m box could not earn the
# 500 m R_comm at any altitude, so the baselines were being exercised outside
# the regime they are compared in.
AREA = {"x": [0.0, 5000.0], "y": [0.0, 5000.0], "z": [100.0, 400.0]}


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
    """Equal-disc packing: ONE radius and ONE altitude shared by the whole fleet."""
    inst = _instance()
    result = Mozaffari2016().optimize(inst, Fitness(inst), np.random.default_rng(0))
    r_service = float(result.meta["service_radius_m"])
    r_pack = float(result.meta["packing_radius_m"])
    assert r_service > 0.0 and r_pack > 0.0, f"non-positive radii: {r_service}, {r_pack}"
    positions = inst.positions_from_vector(result.best_position)
    assert np.allclose(positions[:, 2], result.meta["altitude_m"]), (
        "Mozaffari flies one common altitude by construction; per-UAV altitudes "
        "would mean the equal-disc packing has been broken"
    )


def alzenad_per_uav_radii():
    """Alzenad sizes each UAV to its own cluster, so altitudes must vary."""
    inst = _instance(seed=5, N=120, K=5)
    result = Alzenad2017().optimize(inst, Fitness(inst), np.random.default_rng(0))
    z = np.asarray(result.meta["altitudes_m"], dtype=float)
    assert z.shape == (inst.K,), f"expected {inst.K} altitudes, got {z.shape}"
    assert np.all(z > 0.0) and np.all(np.isfinite(z))
    assert np.all(z >= inst.lower[2] - 1e-9) and np.all(z <= inst.upper[2] + 1e-9), (
        f"altitudes {z} escape the band [{inst.lower[2]}, {inst.upper[2]}]"
    )
    positions = inst.positions_from_vector(result.best_position)
    assert np.allclose(positions[:, 2], z), "reported altitudes do not match the placement"
    assert z.std() > 0.0, (
        "every UAV got the same altitude — that is Mozaffari's equal-disc behaviour, "
        "not Alzenad's per-cluster sizing, so the two baselines are not distinct"
    )


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
    """Each baseline must cover something under BOTH gates it can be scored by.

    Tier-1 scores every method through the shared channel now, so a baseline
    that only covers devices under its own derived radius would be reported at
    zero coverage and look far worse than published — the mirror image of the
    618-vs-500 m advantage recorded in REPORTS/results_provenance.md.
    """
    inst = _instance(seed=7, N=100, K=6)
    fitness = Fitness(inst)
    link = LinkModel(r_comm_m=inst.R_comm, z_min_m=inst.lower[2], z_max_m=inst.upper[2])
    channel_fitness = Fitness(inst, link=link)
    for cls in (Mozaffari2016, Alzenad2017):
        result = cls().optimize(inst, fitness, np.random.default_rng(0))
        flat_n = fitness.components(result.best_position).n_assigned
        chan_n = channel_fitness.components(result.best_position).n_assigned
        assert flat_n > 0, f"{cls.__name__} covers nothing under the flat R_comm gate"
        assert chan_n > 0, (
            f"{cls.__name__} covers nothing under the shared channel — it would be "
            "reported at zero coverage against every other Tier-1 method"
        )


check("both baselines registered under their config names", registered)
check("one-shot Result contract (1 iteration, 1 eval, in-band altitude)", one_shot_contract)
check("Mozaffari packs equal discs: one shared radius/altitude", mozaffari_shared_radius)
check("Alzenad records per-UAV radii and altitudes", alzenad_per_uav_radii)
check("both rules deterministic (rng argument irrelevant)", deterministic_rules)
check("Alzenad altitude grows with cluster spread", alzenad_altitude_tracks_spread)
check("both baselines cover devices on a clustered instance", both_cover_devices)
finish()
