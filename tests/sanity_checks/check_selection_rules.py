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
    for mode in ("ucb", "fedcs", "rep_cap", "fair_mab"):
        sel = ClientSelector(list(range(n)))
        covered = {i: i % 3 for i in range(n)}  # 3 UAVs
        states = {i: _state() for i in range(n)}
        rep = {i: 0.5 for i in range(n)}
        kwargs = {"rng": np.random.default_rng(0)} if mode == "ucb" else {}
        result = sel.select(covered, states, rep, _coords(n), [], 1, 2, mode=mode, **kwargs)
        for uav, cnt in Counter(result.values()).items():
            assert cnt <= 2, f"{mode} exceeded per-UAV capacity on UAV {uav}"


def mode_all_and_unknown_mode():
    sel = ClientSelector([0, 1])
    states = {0: _state(), 1: _ineligible()}
    # mode='all' (flat_fl) bypasses the eligibility gate entirely.
    r = sel.select({0: 0, 1: 0}, states, {0: 0.5, 1: 0.5}, _coords(2), [], 1, 99, mode="all")
    assert set(r.keys()) == {0, 1}
    try:
        sel.select({0: 0}, {0: _state()}, {0: 0.5}, _coords(1), [], 1, 1, mode="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown selection mode must raise")


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
check("mode 'all' bypasses gate; unknown mode raises", mode_all_and_unknown_mode)
check("device eligibility gate thresholds", eligibility_gate_thresholds)
finish()
