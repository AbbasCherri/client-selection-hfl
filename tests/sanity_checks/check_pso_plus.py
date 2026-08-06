"""PSO+ invariants: it degenerates to PSO, and each feature does what it claims.

The governing invariant is bit-identity. Every result comparing pso_plus to pso
attributes the difference to a named feature; that attribution is only valid if
pso_plus with all features OFF is the same algorithm as pso, down to the RNG
stream. If it silently diverged, every ablation in the paper would be measuring
an incidental refactor.

The warm-start floor is the other load-bearing claim: at R_comm = 2 km plain PSO
scored 0.4005 against hfl_static's 0.4510 — it returned a layout worse than
standing still. Seeding particle 0 at prev_positions makes "never worse than
standing still" a property of the algorithm, so it gets asserted, not assumed.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _lib import check, finish  # noqa: E402

from uavbench.optimizers import REGISTRY, build_optimizer  # noqa: E402
from uavbench.optimizers.pso import PSO  # noqa: E402
from uavbench.optimizers.pso_plus import PSOPlus  # noqa: E402
from uavbench.problem.fitness import Fitness  # noqa: E402
from uavbench.problem.instance import generate_instance  # noqa: E402

BUDGET = {"P": 24, "G_max": 12}


def _instance(seed: int = 3, distribution: str = "clustered"):
    return generate_instance(
        N=60, K=4, distribution=distribution, seed=seed,
        area={"x": [0.0, 5000.0], "y": [0.0, 5000.0], "z": [20.0, 120.0]},
        R_comm=800.0, capacity=20,
    )


def _run(opt, inst, seed=11):
    fit = Fitness(inst, w1=0.6, w2=0.3, w3=0.1)
    return opt.optimize(inst, fit, np.random.default_rng(seed)), fit


def pso_plus_defaults_are_bit_identical_to_pso():
    for dist in ("clustered", "uniform"):
        for seed in (1, 2):
            inst = _instance(seed=seed, distribution=dist)
            a, fa = _run(PSO(**BUDGET), inst)
            b, fb = _run(PSOPlus(**BUDGET), inst)
            assert np.array_equal(a.best_position, b.best_position), (
                f"[{dist}/{seed}] positions differ — pso_plus defaults are NOT pso, "
                "so every ablation against pso is confounded"
            )
            assert a.best_fitness == b.best_fitness, (
                f"[{dist}/{seed}] fitness {a.best_fitness!r} != {b.best_fitness!r}"
            )
            assert a.convergence == b.convergence, f"[{dist}/{seed}] convergence curves differ"
            assert fa.eval_count == fb.eval_count, (
                f"[{dist}/{seed}] eval counts differ ({fa.eval_count} vs {fb.eval_count}) — "
                "the shared budget comparison would be void"
            )


def warm_start_never_returns_worse_than_standing_still():
    """The floor that plain PSO lacks at tight R_comm."""
    for seed in (1, 2, 3):
        inst = _instance(seed=seed)
        fit = Fitness(inst, w1=0.6, w2=0.3, w3=0.1)
        f_prev = float(fit.batch(np.asarray(inst.prev_positions).reshape(1, -1))[0])
        res, _ = _run(PSOPlus(warm_start_frac=0.25, **BUDGET), inst)
        assert res.best_fitness >= f_prev - 1e-12, (
            f"[seed {seed}] warm-started PSO returned {res.best_fitness:.6f}, worse than "
            f"the current layout {f_prev:.6f} — the floor is not holding"
        )


def warm_start_actually_changes_the_swarm():
    """Guard the guard: a no-op flag would pass the floor check trivially."""
    inst = _instance()
    plain, _ = _run(PSOPlus(**BUDGET), inst)
    warm, _ = _run(PSOPlus(warm_start_frac=0.5, **BUDGET), inst)
    assert not np.array_equal(plain.best_position, warm.best_position), (
        "warm_start_frac=0.5 produced the identical result to 0.0 — the flag is inert"
    )


def reachable_norm_changes_the_movement_penalty():
    inst = _instance()
    opt = PSOPlus(move_norm="reachable", v_max_mps=20.0, t_interval_s=300.0, **BUDGET)
    d_reach = opt._reachable_d_max(inst)
    d_box = max(inst.K * inst.box_diagonal, 1e-9)
    assert d_reach < d_box, (
        f"reachable d_max ({d_reach:.0f} m) should be tighter than the box "
        f"normaliser ({d_box:.0f} m) — otherwise the penalty stays inert"
    )
    plain, _ = _run(PSOPlus(**BUDGET), inst)
    scaled, _ = _run(opt, inst)
    assert plain.best_fitness != scaled.best_fitness, (
        "move_norm='reachable' left the objective unchanged — the flag is inert"
    )


def d_max_is_restored_after_the_run():
    """The fitness object is shared and reused; a leaked d_max would corrupt it."""
    inst = _instance()
    fit = Fitness(inst, w1=0.6, w2=0.3, w3=0.1)
    before = fit.d_max
    PSOPlus(move_norm="reachable", **BUDGET).optimize(inst, fit, np.random.default_rng(5))
    assert fit.d_max == before, (
        f"d_max leaked: {fit.d_max} != {before}. Every later evaluation through this "
        "fitness would silently use the optimizer's normaliser"
    )


def move_gate_declines_a_move_that_does_not_pay():
    """With an unreachable margin the gate must always return the current layout."""
    inst = _instance()
    fit = Fitness(inst, w1=0.6, w2=0.3, w3=0.1)
    prev = np.asarray(inst.prev_positions).reshape(-1)
    res = PSOPlus(move_gate=True, gate_margin=1e9, **BUDGET).optimize(
        inst, fit, np.random.default_rng(11)
    )
    assert res.meta.get("gate_declined") is True, "gate should have declined"
    assert np.array_equal(res.best_position, prev), "declined move did not restore prev_positions"

    # ... and with a margin of -inf it must always accept, so the flag is not
    # simply "always decline".
    fit2 = Fitness(inst, w1=0.6, w2=0.3, w3=0.1)
    res2 = PSOPlus(move_gate=True, gate_margin=-1e9, **BUDGET).optimize(
        inst, fit2, np.random.default_rng(11)
    )
    assert res2.meta.get("gate_declined") is False, "gate should have accepted"


def unimplemented_flags_raise():
    """A flag that silently does nothing is the fair_mab failure mode."""
    for kwargs in ({"analytic_seed": True}, {"class_balance_w": 0.5}):
        try:
            PSOPlus(**kwargs, **BUDGET)
        except NotImplementedError:
            continue
        raise AssertionError(f"PSOPlus({kwargs}) was accepted but does nothing")


def registered_and_budgeted():
    assert "pso_plus" in REGISTRY, "pso_plus is not in the optimizer registry"
    opt = build_optimizer("pso_plus", {"warm_start_frac": 0.2}, BUDGET)
    assert (opt.P, opt.G_max) == (BUDGET["P"], BUDGET["G_max"]), (
        "pso_plus ignored the shared eval budget — its comparison against pso "
        "would confound the new features with a bigger budget"
    )
    assert opt.warm_start_frac == 0.2, "optimizer_params did not reach the constructor"


if __name__ == "__main__":
    check("pso_plus defaults are bit-identical to pso", pso_plus_defaults_are_bit_identical_to_pso)
    check("warm start never returns worse than standing still",
          warm_start_never_returns_worse_than_standing_still)
    check("warm start actually changes the swarm", warm_start_actually_changes_the_swarm)
    check("reachable normaliser changes the movement penalty",
          reachable_norm_changes_the_movement_penalty)
    check("d_max is restored after the run", d_max_is_restored_after_the_run)
    check("move gate declines a move that does not pay",
          move_gate_declines_a_move_that_does_not_pay)
    check("unimplemented flags raise", unimplemented_flags_raise)
    check("pso_plus is registered and budgeted", registered_and_budgeted)
    finish()
