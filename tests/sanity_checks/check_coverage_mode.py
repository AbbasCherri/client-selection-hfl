"""The reachable-coverage objective must be correct in BOTH scoring paths.

`Fitness` scores through two independent implementations: `components()` (scalar,
used for the reported breakdown and the canonical re-score) and `batch()`
(vectorised, used by PSO/GA during the actual search). They duplicate the
objective arithmetic. If only one honours `coverage_mode`, the optimizer
searches one objective while the table reports another — the same silent
mismatch that had to be closed in the Tier-1 metrics path on the same day.

Also guarded here: that "reachable" is a strict generalisation (identical to
"assigned" until capacity binds), and that it is not vacuous (it really does
change what a placement optimizer does at the operating point).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np  # noqa: E402
from _lib import check, finish  # noqa: E402

from uavbench.problem.assignment import greedy_assignment  # noqa: E402
from uavbench.problem.fitness import Fitness  # noqa: E402
from uavbench.problem.instance import generate_instance  # noqa: E402

_AREA = {"x": [0.0, 5000.0], "y": [0.0, 5000.0], "z": [100.0, 400.0]}


def _inst(n=120, k=6, capacity=6, r_comm=500.0, seed=3):
    return generate_instance(
        distribution="uniform", N=n, K=k, area=_AREA, seed=seed,
        capacity=capacity, uav_battery=1.0, R_comm=r_comm,
        B_min_uav=0.2, beta_mode="pinned", t=0, T_decay=20,
    )


def _rand_positions(inst, rng, p=5):
    lo, hi = inst.lower, inst.upper
    return rng.uniform(lo, hi, size=(p, inst.K, 3)).reshape(p, -1)


def scalar_and_batch_agree_in_both_modes():
    """The load-bearing check: two implementations, one objective."""
    inst = _inst()
    rng = np.random.default_rng(0)
    X = _rand_positions(inst, rng, p=6)
    for mode in ("assigned", "reachable"):
        f = Fitness(inst, 0.811, 0.03, 0.159, coverage_mode=mode)
        batch = f.batch(X)
        scalar = np.array([f(X[i]) for i in range(X.shape[0])])
        assert np.allclose(batch, scalar, rtol=0, atol=1e-12), (
            f"coverage_mode={mode}: batch and scalar disagree by up to "
            f"{np.max(np.abs(batch - scalar)):.3e} — the optimizer would search a "
            "different objective from the one reported"
        )


def reachable_is_never_below_assigned():
    """Capacity can only remove devices from the assigned set, never add."""
    inst = _inst()
    rng = np.random.default_rng(1)
    for _ in range(20):
        pos = rng.uniform(inst.lower, inst.upper, size=(inst.K, 3))
        res = greedy_assignment(inst, pos)
        assert res.f_cover_reachable >= res.f_cover - 1e-12, (
            f"reachable {res.f_cover_reachable} < assigned {res.f_cover}"
        )
        assert res.n_reachable >= res.n_assigned


def the_two_modes_coincide_until_capacity_binds():
    """"reachable" must be a strict generalisation, not a different objective.

    With K*capacity >= N no device can ever be turned away for capacity, so the
    reachable and assigned sets are identical and the two objectives must return
    exactly the same number.
    """
    inst = _inst(n=30, k=10, capacity=6)      # 60 slots >= 30 devices
    rng = np.random.default_rng(2)
    X = _rand_positions(inst, rng, p=5)
    a = Fitness(inst, 0.811, 0.03, 0.159, coverage_mode="assigned").batch(X)
    r = Fitness(inst, 0.811, 0.03, 0.159, coverage_mode="reachable").batch(X)
    assert np.allclose(a, r, rtol=0, atol=1e-12), (
        f"modes differ by up to {np.max(np.abs(a - r)):.3e} with slack capacity — "
        "'reachable' is not reducing to 'assigned' where it must"
    )


def the_modes_differ_once_capacity_binds():
    """Guard the guard: if they never differ, the whole change is a no-op."""
    inst = _inst(n=200, k=5, capacity=6)      # 30 slots << 200 devices
    rng = np.random.default_rng(3)
    X = _rand_positions(inst, rng, p=8)
    a = Fitness(inst, 0.811, 0.03, 0.159, coverage_mode="assigned").batch(X)
    r = Fitness(inst, 0.811, 0.03, 0.159, coverage_mode="reachable").batch(X)
    assert np.max(np.abs(a - r)) > 1e-6, (
        "the two coverage modes score identically even with capacity binding "
        "hard — 'reachable' is not reaching the objective"
    )
    assert (r >= a - 1e-12).all(), "reachable scored below assigned somewhere"


def reachable_actually_changes_what_the_optimizer_does():
    """The point of the change: more clients reached at the same spend.

    This is the claim the redesign rests on, so it is asserted on optimizer
    behaviour rather than on the objective's algebra.
    """
    from uavbench.optimizers import build_optimizer

    inst = _inst(n=200, k=8, capacity=6, r_comm=800.0, seed=11)
    budget = {"P": 30, "G_max": 20}
    reached = {}
    for mode in ("assigned", "reachable"):
        f = Fitness(inst, 0.811, 0.03, 0.159, coverage_mode=mode)
        opt = build_optimizer("pso", params={}, budget=budget)
        res = opt.optimize(inst, f, np.random.default_rng(7))
        reached[mode] = greedy_assignment(
            inst, res.best_position.reshape(inst.K, 3)
        ).n_reachable

    assert reached["reachable"] > reached["assigned"], (
        f"reachable-mode placement reached {reached['reachable']} devices vs "
        f"{reached['assigned']} for assigned-mode — the new objective is not "
        "buying coverage, which is the entire premise of the redesign"
    )
    print(f"      reached: assigned={reached['assigned']}, reachable={reached['reachable']}")


def an_unknown_mode_is_rejected():
    inst = _inst()
    try:
        Fitness(inst, coverage_mode="reachible")
    except ValueError as e:
        assert "coverage_mode" in str(e)
        return
    raise AssertionError("a misspelled coverage_mode was silently accepted")


check("scalar and batch agree in both modes", scalar_and_batch_agree_in_both_modes)
check("reachable is never below assigned", reachable_is_never_below_assigned)
check("the two modes coincide until capacity binds", the_two_modes_coincide_until_capacity_binds)
check("the modes differ once capacity binds", the_modes_differ_once_capacity_binds)
check("reachable changes what the optimizer does", reachable_actually_changes_what_the_optimizer_does)
check("an unknown coverage_mode is rejected", an_unknown_mode_is_rejected)
finish()
