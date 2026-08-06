"""Client-selection rules: proposed UCB + literature baselines B1-B3 formulas."""

import sys  # noqa: I001
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from collections import Counter

import numpy as np
from _lib import check, finish

from uavbench.fl.client_selection import (
    REPCAP_GAMMA,
    ClientSelector,
)
from uavbench.fl.device_state import T_MAX_S, DeviceState


def _state(battery=0.8, snr_db=15.0, compute_time_s=100.0):
    return DeviceState(battery=battery, snr_db=snr_db, memory_ok=True, compute_time_s=compute_time_s)


def _ineligible():
    return DeviceState(battery=0.0, snr_db=0.0, memory_ok=False, compute_time_s=9999.0)


def _coords(n):
    return {i: (37.488 + i * 0.01, 137.272) for i in range(n)}


def fedcs_fastest_first_and_deadline():
    sel = ClientSelector(list(range(4)))
    covered = {i: 0 for i in range(4)}
    states = {i: _state(compute_time_s=50.0 + i * 60.0) for i in range(4)}
    rep = {i: 0.5 for i in range(4)}
    result = sel.select(covered, states, rep, _coords(4), [], 1, 2, mode="fedcs")
    assert set(result.keys()) == {0, 1}  # the two cheapest T-hat win
    # Ineligible devices never picked.
    sel2 = ClientSelector([0, 1])
    states2 = {0: _state(), 1: _ineligible()}
    r2 = sel2.select({0: 0, 1: 0}, states2, {0: 0.5, 1: 0.5}, _coords(2), [], 1, 2, mode="fedcs")
    assert 1 not in r2


def rep_cap_formula_and_static_ranking():
    # score = gamma*R + (1-gamma)*(1 - (T/T_max)^2), gamma = 0.5 (Zhao 2024).
    sel = ClientSelector([0])
    scores = sel._rep_cap_scores([0], {0: _state(compute_time_s=150.0)}, {0: 0.6})
    expected = REPCAP_GAMMA * 0.6 + (1.0 - REPCAP_GAMMA) * (1.0 - (150.0 / T_MAX_S) ** 2)
    assert abs(scores[0] - expected) < 1e-12
    # No exploration bonus: the ranking is static, starved devices stay starved.
    n = 4
    sel = ClientSelector(list(range(n)))
    covered = {i: 0 for i in range(n)}
    states = {i: _state(compute_time_s=100.0) for i in range(n)}
    rep = {0: 0.9, 1: 0.8, 2: 0.7, 3: 0.5}
    for rnd in range(1, 11):
        result = sel.select(covered, states, rep, _coords(n), [], rnd, 2, mode="rep_cap")
        assert frozenset(result.keys()) == frozenset({0, 1})
    assert sel._counts[3] == 0


def fair_mab_matches_the_published_formulation():
    """B3 must be Zhu et al.'s algorithm, not something wearing its name.

    Verified against Sensors 24(5):1599 on 2026-08-06. The previous
    implementation used 0.5/0.5 weights, a capped staleness term, battery as the
    energy proxy, and no exploration bonus — while the source is a UCB bandit
    over normalized transmission energy and unbounded participation freshness.
    Each published component is pinned here so a future "simplification" cannot
    quietly reintroduce a different algorithm.
    """
    sel = ClientSelector([0])
    st = {0: _state(snr_db=20.0)}

    # The exploration bonus must actually decay with participation, and must
    # stay finite at C_m = 0 (see the smoothing note in _fair_mab_scores: an
    # infinite tie would make round 1 select by client index).
    sel._counts[0] = 0
    unvisited = sel._fair_mab_scores([0], st, round_num=5)[0]
    sel._counts[0] = 25
    visited = sel._fair_mab_scores([0], st, round_num=5)[0]
    assert np.isfinite(unvisited), "the C_m = 0 exploration bonus is infinite; ties would sort by index"
    assert unvisited > visited, (
        "the exploration bonus did not decay with participation count; the UCB "
        "term of Eq. 16 is missing and B3 is no longer a bandit"
    )

    # Freshness (Eq. 14) is UNBOUNDED: t - a*C_m. Capping it is exactly what
    # made the old version inert, so a large t must keep growing the score.
    sel = ClientSelector([0])
    sel._counts[0] = 1
    early = sel._fair_mab_scores([0], st, round_num=10)[0]
    late = sel._fair_mab_scores([0], st, round_num=100)[0]
    # FM contributes (1 - alpha) = 0.4 of the score, so 90 rounds must move it
    # by ~0.4 * 90 = 36. A capped term would move it by O(1).
    assert late - early > 30.0, (
        f"freshness grew only {late - early:.3f} between round 10 and 100; expected "
        "~36 = (1 - alpha) * 90. The unbounded t - a*C_m term of Eq. 14 is capped again"
    )

    # A device that has participated often must lose to an equally-provisioned
    # device that has not — the whole point of the fairness enhancement.
    sel = ClientSelector([0, 1])
    sel._counts[0], sel._counts[1] = 20, 1
    scores = sel._fair_mab_scores([0, 1], {0: _state(snr_db=20.0), 1: _state(snr_db=20.0)},
                                  round_num=50)
    assert scores[1] > scores[0], (
        "a device with 20 participations outscored one with 1 at equal channel; "
        "the freshness term is not creating fairness pressure"
    )

    # Better channel => less transmission energy => higher energy term (Eq. 13).
    sel = ClientSelector([0, 1])
    sel._counts[0] = sel._counts[1] = 5
    scores = sel._fair_mab_scores([0, 1], {0: _state(snr_db=30.0), 1: _state(snr_db=3.0)},
                                  round_num=20)
    assert scores[0] > scores[1], (
        "a device on a 30 dB channel did not outscore one on 3 dB at equal "
        "participation; the normalized-energy term of Eq. 13 is inert"
    )


def fair_mab_is_not_rank_equivalent_to_battery():
    """The exact failure that went unnoticed for months.

    The old reward collapsed to battery order, so B3 was reported as a
    fairness/energy bandit while actually being "highest battery first". Battery
    is now not an input to the score at all, and this asserts it: varying only
    battery must not reorder the ranking.
    """
    sel = ClientSelector([0, 1, 2])
    for cid in (0, 1, 2):
        sel._counts[cid] = 3
    same_snr = {c: _state(snr_db=15.0) for c in (0, 1, 2)}
    base = sel._fair_mab_scores([0, 1, 2], same_snr, round_num=30)

    sel2 = ClientSelector([0, 1, 2])
    for cid in (0, 1, 2):
        sel2._counts[cid] = 3
    varied_battery = {0: _state(snr_db=15.0, battery=0.2),
                      1: _state(snr_db=15.0, battery=0.6),
                      2: _state(snr_db=15.0, battery=0.99)}
    got = sel2._fair_mab_scores([0, 1, 2], varied_battery, round_num=30)
    assert np.allclose(base, got), (
        f"battery changed the fair_mab score ({base} -> {got}). Zhu et al.'s "
        "reward is over transmission energy and freshness; battery is a "
        "stored state, and scoring by it is the degenerate rule this replaced"
    )


def fair_mab_arms_actually_vary_something():
    """Every fair_mab arm of the 0.3 sweep must change the selected set.

    The 110-job 0.3 run returned BIT-IDENTICAL means and stds for all four
    fair_mab arms — the constants reached nothing. That is the worst possible
    failure: it looks like the honest finding "this baseline is insensitive to
    its own hyperparameters" while actually measuring the same run four times.

    Both original causes are gone with the published formulation restored (the
    cap was ours and no longer exists, and the reward is no longer battery
    order), but the *class* of failure — an arm whose constants reach nothing —
    is not specific to them, so the guard stays: drive the real selection
    cadence and demand each arm's picks differ from the stock arm's.
    """
    from uavbench.fl.selection_isolation import (
        ARM_SPECS,
        apply_const_overrides,
        arm_consts,
        restore_const_overrides,
    )

    n, n_uav, cap_per_uav, t_sel = 40, 2, 5, 5
    # 10 slots for 40 eligible clients, so score ORDER decides who trains —
    # with more slots than clients every arm would trivially agree.
    covered = {i: i % n_uav for i in range(n)}
    # SNR must vary, not just battery: fair_mab's energy term is over the
    # *channel* (battery is deliberately not an input any more), so a pool of
    # identical channels makes alpha inert and the arm check vacuous.
    states = {
        i: _state(battery=0.30 + 0.7 * (i % 17) / 16.0,
                  snr_db=5.0 + 25.0 * (i % 13) / 12.0)
        for i in range(n)
    }
    reps = {i: 0.5 for i in range(n)}

    def cadence():
        sel = ClientSelector(list(range(n)))
        picks = []
        for event in range(6):
            chosen = sel.select(
                covered, states, reps, _coords(n), [], event * t_sel, cap_per_uav,
                mode="fair_mab", rng=np.random.default_rng(0),
            )
            picks.append(frozenset(chosen))
        return picks

    stock = cadence()
    assert all(len(p) < n for p in stock), (
        "every eligible client was selected — the score never binds, so this "
        "check could not detect an inert constant"
    )

    for arm, spec in ARM_SPECS.items():
        if spec.get("mode") != "fair_mab" or not arm_consts(arm):
            continue
        restores = apply_const_overrides(arm_consts(arm))
        try:
            assert cadence() != stock, (
                f"arm {arm!r} produced exactly the stock fair_mab selection — "
                f"its constants {sorted(arm_consts(arm))} changed nothing, so "
                "any result reported for it is a duplicate of the default arm"
            )
        finally:
            restore_const_overrides(restores)


def capacity_respected_by_all_modes():
    n = 9
    for mode in ("ucb", "fedcs", "rep_cap", "fair_mab", "oort", "power_of_choice"):
        sel = ClientSelector(list(range(n)))
        covered = {i: i % 3 for i in range(n)}  # 3 UAVs
        states = {i: _state() for i in range(n)}
        rep = {i: 0.5 for i in range(n)}
        kwargs = (
            {"rng": np.random.default_rng(0)}
            if mode in ("ucb", "power_of_choice")
            else {}
        )
        result = sel.select(covered, states, rep, _coords(n), [], 1, 2, mode=mode, **kwargs)
        for uav, cnt in Counter(result.values()).items():
            assert cnt <= 2, f"{mode} exceeded per-UAV capacity on UAV {uav}"


def mode_all_and_unknown_mode():
    sel = ClientSelector([0, 1])
    states = {0: _state(), 1: _ineligible()}
    # mode='all' (flat_fl) selects every ELIGIBLE covered client — the device
    # physics gate applies to every topology (2026-07-18 fix: bypassing it made
    # flat_fl immune to battery drain and the dropout/SNR stress knobs).
    r = sel.select({0: 0, 1: 0}, states, {0: 0.5, 1: 0.5}, _coords(2), [], 1, 99, mode="all")
    assert set(r.keys()) == {0}
    try:
        sel.select({0: 0}, {0: _state()}, {0: 0.5}, _coords(1), [], 1, 1, mode="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown selection mode must raise")


def oort_and_power_of_choice_loss_ranking():
    # oort: higher last-observed loss wins slots; stragglers (above the
    # eligible-pool median compute time) are penalised; never-trained clients
    # inherit the max observed loss (exploration prior).
    n = 4
    sel = ClientSelector(list(range(n)))
    sel.update_losses({0: 0.1, 1: 2.0, 2: 1.0})  # 3 never trained → prior 2.0
    states = {i: _state() for i in range(n)}
    covered = {i: 0 for i in range(n)}
    r = sel.select(covered, states, {i: 0.5 for i in range(n)}, _coords(n), [], 5, 2, mode="oort")
    assert len(r) == 2
    assert 1 in r and 0 not in r, "high-loss client must beat low-loss under oort"
    assert 3 in r or 2 in r, "untrained client competes at the max-loss prior"

    # oort straggler penalty: equal losses, the slow half loses.
    sel2 = ClientSelector(list(range(4)))
    sel2.update_losses({i: 1.0 for i in range(4)})
    states2 = {
        0: _state(),
        1: _state(),
        2: DeviceState(battery=0.8, snr_db=15.0, memory_ok=True, compute_time_s=290.0),
        3: DeviceState(battery=0.8, snr_db=15.0, memory_ok=True, compute_time_s=295.0),
    }
    r2 = sel2.select({i: 0 for i in range(4)}, states2, {i: 0.5 for i in range(4)},
                     _coords(4), [], 5, 2, mode="oort")
    assert set(r2.keys()) == {0, 1}, "stragglers above median compute must be penalised"

    # power_of_choice: with the candidate set covering the pool, top-loss wins.
    sel3 = ClientSelector(list(range(4)))
    sel3.update_losses({0: 0.1, 1: 0.2, 2: 5.0, 3: 4.0})
    r3 = sel3.select({i: 0 for i in range(4)}, {i: _state() for i in range(4)},
                     {i: 0.5 for i in range(4)}, _coords(4), [], 5, 2,
                     mode="power_of_choice", rng=np.random.default_rng(0))
    assert set(r3.keys()) == {2, 3}, "power_of_choice must keep the highest-loss candidates"


def eligibility_gate_thresholds():
    # The DeviceState gate: at-threshold passes, just-below fails.
    assert _state().eligible()
    assert not DeviceState(battery=0.0, snr_db=15.0, memory_ok=True, compute_time_s=100.0).eligible()
    assert not DeviceState(battery=0.8, snr_db=-5.0, memory_ok=True, compute_time_s=100.0).eligible()
    assert not DeviceState(battery=0.8, snr_db=15.0, memory_ok=False, compute_time_s=100.0).eligible()
    assert not DeviceState(battery=0.8, snr_db=15.0, memory_ok=True, compute_time_s=1e9).eligible()


check("FedCS: greedy fastest-first under deadline, eligibility respected", fedcs_fastest_first_and_deadline)
check("rep_cap: exact score formula (gamma=0.5), static no-exploration ranking", rep_cap_formula_and_static_ranking)
check("fair_mab matches the published formulation", fair_mab_matches_the_published_formulation)
check("fair_mab is not rank-equivalent to battery", fair_mab_is_not_rank_equivalent_to_battery)
check("fair_mab arms actually vary the selection", fair_mab_arms_actually_vary_something)
check("all modes respect per-UAV capacity", capacity_respected_by_all_modes)
check("mode 'all' is eligibility-gated; unknown mode raises", mode_all_and_unknown_mode)
check("oort/power_of_choice: loss ranking, straggler penalty, prior", oort_and_power_of_choice_loss_ranking)
check("device eligibility gate thresholds", eligibility_gate_thresholds)
finish()
