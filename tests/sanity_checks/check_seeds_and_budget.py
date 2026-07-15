"""Fair-comparison invariants: shared PSO/GA budget, disjoint seed streams,
determinism. Prints the seed lists side by side for one scenario so a human
can eyeball the pairing (manual checks #1 and #2)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from _lib import check, finish

from uavbench.fl.seeds import fullsim_method_seed, sweep_job_seed, tier2_seed
from uavbench.optimizers import build_optimizer
from uavbench.optimizers.ga import GA
from uavbench.optimizers.pso import PSO
from uavbench.problem.fitness import Fitness
from uavbench.problem.instance import generate_instance
from uavbench.runner import _instance_seed, _optimizer_rng

BUDGET = {"P": 77, "G_max": 33}
AREA = {"x": [0.0, 1000.0], "y": [0.0, 1000.0], "z": [20.0, 120.0]}


def budget_identical():
    pso = build_optimizer("pso", budget=BUDGET)
    ga = build_optimizer("ga", budget=BUDGET)
    assert (pso.P, pso.G_max) == (ga.P, ga.G_max) == (77, 33)


def budget_wins_over_params():
    # c1=2.5 keeps phi > 4, the constriction-PSO validity gate.
    pso = build_optimizer("pso", params={"P": 999, "G_max": 999, "c1": 2.5}, budget=BUDGET)
    assert (pso.P, pso.G_max) == (77, 33)
    assert pso.c1 == 2.5  # non-budget params still pass through


def unbudgeted_methods_ignore_budget():
    for method in ("centroid", "static", "random", "mozaffari2016", "alzenad2017"):
        opt = build_optimizer(method, budget=BUDGET)
        assert not hasattr(opt, "P"), method


def seed_streams_disjoint():
    # Worst case: same base for both families, seed_i=0 included — the
    # historical SeedSequence trailing-zero collision trigger.
    base = 1234
    inst = {_instance_seed(base, s, i) for s in range(4) for i in range(10)}
    opt_states = {
        int(_optimizer_rng(base, m, s, i).bit_generator.seed_seq.generate_state(1)[0])
        for m in range(6)
        for s in range(4)
        for i in range(10)
    }
    assert not (inst & opt_states)
    # The exact historical collision: instance (s, i) vs optimizer (m=s, s=i, 0).
    assert _instance_seed(999, 0, 2) != int(
        _optimizer_rng(999, 0, 2, 0).bit_generator.seed_seq.generate_state(1)[0]
    )


def instance_seed_method_independent():
    assert _instance_seed(1234, 2, 7) == _instance_seed(1234, 2, 7)
    assert _optimizer_rng(9876, 0, 2, 7).random() != _optimizer_rng(9876, 1, 2, 7).random()


def fl_seed_formulas():
    # fullsim seed folds the method in exactly once; sweep job seed does not
    # encode the method at all (paired instances across methods).
    assert fullsim_method_seed(42, "pso") != fullsim_method_seed(42, "ga")
    assert fullsim_method_seed(42, "pso") == fullsim_method_seed(42, "pso")
    assert sweep_job_seed(42, 100, 3) == sweep_job_seed(42, 100, 3)
    assert tier2_seed(42, 100, "pso") == tier2_seed(42, 100, "pso")
    assert tier2_seed(42, 100, "pso") != tier2_seed(42, 100, "ga")


def instance_and_optimizers_deterministic():
    a = generate_instance("clustered", N=50, K=4, area=AREA, seed=7)
    b = generate_instance("clustered", N=50, K=4, area=AREA, seed=7)
    assert np.array_equal(a.device_coords, b.device_coords)
    assert np.array_equal(a.value, b.value)
    inst = generate_instance("uniform", N=30, K=3, area=AREA, seed=1)
    for opt_cls, s in ((PSO, 42), (GA, 5)):
        r1 = opt_cls(P=20, G_max=20).optimize(inst, Fitness(inst), np.random.default_rng(s))
        r2 = opt_cls(P=20, G_max=20).optimize(inst, Fitness(inst), np.random.default_rng(s))
        assert r1.best_fitness == r2.best_fitness
        assert np.array_equal(r1.best_position, r2.best_position)


def print_seed_lists():
    # Human-eyeball output: one scenario, 5 seeds, 3 methods.
    base_i, base_o, scenario = 42, 42, 0
    print("    seed_i | instance_seed | optimizer_state per method (pso, ga, centroid)")
    for i in range(5):
        inst = _instance_seed(base_i, scenario, i)
        opts = [
            int(_optimizer_rng(base_o, m, scenario, i).bit_generator.seed_seq.generate_state(1)[0])
            for m in range(3)
        ]
        print(f"    {i:6d} | {inst:13d} | {opts}")


check("PSO and GA receive the identical shared budget", budget_identical)
check("budget overrides conflicting optimizer_params", budget_wins_over_params)
check("one-shot methods take no P/G_max", unbudgeted_methods_ignore_budget)
check("instance/optimizer seed streams disjoint (worst case)", seed_streams_disjoint)
check("instance seeds method-independent, optimizer streams differ", instance_seed_method_independent)
check("FL harness seed formulas (fullsim/sweep/tier2)", fl_seed_formulas)
check("instance generation + PSO/GA deterministic per seed", instance_and_optimizers_deterministic)
check("print seed lists for one scenario (eyeball the pairing)", print_seed_lists)
finish()
