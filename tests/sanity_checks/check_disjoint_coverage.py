"""The redundancy-discounted coverage objective (C3), in BOTH scoring paths.

`Fitness` scores through two independent implementations — `components()`
(scalar, used for the reported breakdown and the canonical re-score) and
`batch()` (vectorised, used by PSO/GA during the actual search). They duplicate
the objective arithmetic. If only one honours `coverage_mode="disjoint"`, the
optimizer searches one objective while the table reports another. That is exactly
the mismatch check_coverage_mode.py was written for when C1 landed; C3 touches
the same two paths, so it gets the same treatment.

Also guarded: the defining property. `disjoint` must equal `reachable` on a
tiling and be strictly below it under overlap, because that difference IS the
mechanism — see REPORTS/preregistration_v6_c3.md §5a. An implementation that
merely tracked `reachable` would pass a naive smoke test and measure nothing.
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


def _inst(n=120, k=6, capacity=6, r_comm=1500.0, seed=3):
    return generate_instance(
        distribution="uniform", N=n, K=k, area=_AREA, seed=seed,
        capacity=capacity, uav_battery=1.0, R_comm=r_comm,
        B_min_uav=0.2, beta_mode="pinned", t=0, T_decay=20,
    )


def _rand_positions(inst, rng, p=6):
    lo, hi = inst.lower, inst.upper
    return rng.uniform(lo, hi, size=(p, inst.K, 3)).reshape(p, -1)


def scalar_and_batch_agree():
    inst = _inst()
    rng = np.random.default_rng(0)
    X = _rand_positions(inst, rng, p=8)
    f = Fitness(inst, coverage_mode="disjoint")
    batch = f.batch(X)
    scalar = np.array([f.components(x).fitness for x in X])
    assert np.allclose(batch, scalar, atol=1e-12), (
        f"paths disagree: max |diff| = {np.max(np.abs(batch - scalar)):.3e}"
    )


def disjoint_never_exceeds_reachable():
    inst = _inst()
    rng = np.random.default_rng(1)
    for x in _rand_positions(inst, rng, p=12):
        res = greedy_assignment(inst, inst.positions_from_vector(x))
        assert res.f_cover_disjoint <= res.f_cover_reachable + 1e-9, (
            f"{res.f_cover_disjoint} > {res.f_cover_reachable}"
        )


def they_are_equal_on_a_tiling():
    # UAVs far enough apart that no device is reached twice: multiplicity 1
    # everywhere, so the discount must vanish exactly.
    inst = _inst(n=150, k=3, r_comm=400.0, seed=9)
    pos = np.array([[200.0, 200.0, 200.0],
                    [2500.0, 2500.0, 200.0],
                    [4800.0, 4800.0, 200.0]])
    res = greedy_assignment(inst, pos)
    assert res.n_reachable > 0, "fixture reaches nobody — it proves nothing"
    assert abs(res.f_cover_disjoint - res.f_cover_reachable) < 1e-9, (
        f"disjoint {res.f_cover_disjoint} != reachable {res.f_cover_reachable} "
        "on a non-overlapping layout"
    )


def stacking_uavs_divides_the_credit():
    # K UAVs at one point: every reached device has multiplicity K, so the
    # discounted value must be exactly reachable/K.
    inst = _inst(n=150, k=4, r_comm=1500.0, seed=11)
    pos = np.repeat(np.array([[2500.0, 2500.0, 200.0]]), 4, axis=0)
    res = greedy_assignment(inst, pos)
    assert res.n_reachable > 0
    assert abs(res.f_cover_disjoint - res.f_cover_reachable / 4) < 1e-9, (
        f"{res.f_cover_disjoint} != {res.f_cover_reachable / 4}"
    )


def adding_a_redundant_uav_lowers_the_score():
    # THE mechanism. Reachable coverage is indifferent to piling a second
    # aircraft on an already-covered crowd; disjoint must penalise it.
    inst = _inst(n=150, k=2, r_comm=1200.0, seed=5)
    apart = np.array([[900.0, 900.0, 200.0], [4100.0, 4100.0, 200.0]])
    stacked = np.array([[900.0, 900.0, 200.0], [900.0, 900.0, 200.0]])
    a, s = greedy_assignment(inst, apart), greedy_assignment(inst, stacked)
    assert a.f_cover_disjoint > s.f_cover_disjoint, (
        "overlapping layout did not score below the separated one"
    )


def reaching_new_devices_raises_the_score():
    inst = _inst(n=150, k=2, r_comm=1200.0, seed=5)
    one_useful = np.array([[900.0, 900.0, 200.0], [900.0, 900.0, 200.0]])
    two_useful = np.array([[900.0, 900.0, 200.0], [4100.0, 4100.0, 200.0]])
    a = greedy_assignment(inst, one_useful)
    b = greedy_assignment(inst, two_useful)
    assert b.f_cover_disjoint > a.f_cover_disjoint


def the_mode_changes_what_the_optimizer_does():
    # If disjoint scored the same layouts as reachable, the C3 arm would be a
    # relabelled C1 and could not test anything.
    inst = _inst(n=200, k=8, r_comm=1500.0, seed=21)
    rng = np.random.default_rng(4)
    X = _rand_positions(inst, rng, p=40)
    reach = Fitness(inst, coverage_mode="reachable").batch(X)
    disj = Fitness(inst, coverage_mode="disjoint").batch(X)
    assert not np.allclose(reach, disj), "disjoint scores identically to reachable"
    # And it must reorder them, not merely shift every score by a constant.
    assert np.argmax(reach) != np.argmax(disj) or not np.allclose(
        reach - reach.mean(), disj - disj.mean()
    ), "disjoint is a constant offset of reachable — no new preference"


def the_other_modes_are_untouched():
    # C3 edited the shared reduction block; assigned/reachable must be unmoved.
    inst = _inst()
    rng = np.random.default_rng(7)
    X = _rand_positions(inst, rng, p=6)
    for mode in ("assigned", "reachable"):
        f = Fitness(inst, coverage_mode=mode)
        assert np.allclose(f.batch(X), [f.components(x).fitness for x in X], atol=1e-12), (
            f"{mode} paths disagree after the C3 edit"
        )


def an_unknown_mode_is_still_rejected():
    try:
        Fitness(_inst(), coverage_mode="disjiont")
    except ValueError as e:
        assert "coverage_mode" in str(e)
        return
    raise AssertionError("a misspelled coverage_mode was silently accepted")


check("scalar and batch agree in disjoint mode", scalar_and_batch_agree)
check("disjoint never exceeds reachable", disjoint_never_exceeds_reachable)
check("they are equal on a tiling", they_are_equal_on_a_tiling)
check("stacking K UAVs divides credit by K", stacking_uavs_divides_the_credit)
check("a redundant UAV lowers the score", adding_a_redundant_uav_lowers_the_score)
check("reaching new devices raises it", reaching_new_devices_raises_the_score)
check("the mode changes the optimizer's preference", the_mode_changes_what_the_optimizer_does)
check("assigned/reachable are untouched", the_other_modes_are_untouched)
check("an unknown mode is still rejected", an_unknown_mode_is_still_rejected)
finish()
