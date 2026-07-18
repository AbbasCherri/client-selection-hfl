"""Client-selection rules: proposed UCB + literature baselines B1-B3 formulas."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from collections import Counter

import numpy as np
from _lib import check, finish

from uavbench.fl.client_selection import (
    FAIRMAB_W_ENERGY,
    FAIRMAB_W_STALE,
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


def fair_mab_reward_and_fairness_pressure():
    # reward = w_e*battery + w_s*min(1, staleness/T_stale_cap) (Zhu 2024),
    # staleness cap tied to the T_sel cadence via t_stale_cap.
    sel = ClientSelector([0])
    scores = sel._fair_mab_scores([0], {0: _state(battery=0.7)}, round_num=3, t_stale_cap=5)
    assert abs(scores[0] - (FAIRMAB_W_ENERGY * 0.7 + FAIRMAB_W_STALE * 0.6)) < 1e-12
    scores = sel._fair_mab_scores([0], {0: _state(battery=0.0)}, round_num=100, t_stale_cap=5)
    assert abs(scores[0] - FAIRMAB_W_STALE * 1.0) < 1e-12  # staleness capped at 1
    # Fairness pressure: neither of two devices is starved over 7 rounds.
    sel = ClientSelector([0, 1])
    states = {0: _state(battery=0.9), 1: _state(battery=0.7)}
    winners = []
    for rnd in range(1, 8):
        result = sel.select(
            {0: 0, 1: 0}, states, {0: 0.5, 1: 0.5}, _coords(2), [], rnd, 1,
            mode="fair_mab", t_stale_cap=5,
        )
        winners.append(next(iter(result.keys())))
    assert 0 in winners and 1 in winners


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
check("fair_mab: exact reward formula, staleness cap, fairness pressure", fair_mab_reward_and_fairness_pressure)
check("all modes respect per-UAV capacity", capacity_respected_by_all_modes)
check("mode 'all' is eligibility-gated; unknown mode raises", mode_all_and_unknown_mode)
check("oort/power_of_choice: loss ranking, straggler penalty, prior", oort_and_power_of_choice_loss_ranking)
check("device eligibility gate thresholds", eligibility_gate_thresholds)
finish()
