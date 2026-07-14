"""Tests for the literature-baseline selection modes (Algorithms B1-B3).

Covers the three baselines from REPORTS/literature_baselines.md:
  fedcs    — B1: FedCS greedy deadline selection (Nishio & Yonetani, ICC 2019)
  rep_cap  — B2: reputation-capability ranking (Zhao et al., CJA 2024)
  fair_mab — B3: fairness/energy MAB selection (Zhu et al., Sensors 2024)
"""

import tempfile
from collections import Counter

import pandas as pd
import pytest

from uavbench.fl.client_selection import (
    FAIRMAB_W_ENERGY,
    FAIRMAB_W_STALE,
    REPCAP_GAMMA,
    ClientSelector,
)
from uavbench.fl.device_state import T_MAX_S, DeviceState

# ── helpers ───────────────────────────────────────────────────────────────────

def _state(battery=0.8, snr_db=15.0, compute_time_s=100.0):
    return DeviceState(battery=battery, snr_db=snr_db, memory_ok=True,
                       compute_time_s=compute_time_s)

def _ineligible_state():
    return DeviceState(battery=0.0, snr_db=0.0, memory_ok=False, compute_time_s=9999.0)

def _coords(n):
    return {i: (37.488 + i * 0.01, 137.272) for i in range(n)}


# ── FedCS (mode="fedcs") ─────────────────────────────────────────────────────

class TestFedCS:
    def test_selects_fastest_first(self):
        """With capacity 2 of 4 candidates, the two cheapest T̂_n win."""
        sel = ClientSelector(list(range(4)))
        covered = {i: 0 for i in range(4)}
        states = {i: _state(compute_time_s=50.0 + i * 60.0) for i in range(4)}
        rep = {i: 0.5 for i in range(4)}
        result = sel.select(covered, states, rep, _coords(4), [], 1, 2, mode="fedcs")
        assert set(result.keys()) == {0, 1}

    def test_respects_per_uav_capacity(self):
        n = 9
        sel = ClientSelector(list(range(n)))
        covered = {i: i % 3 for i in range(n)}   # 3 UAVs
        states = {i: _state() for i in range(n)}
        rep = {i: 0.5 for i in range(n)}
        result = sel.select(covered, states, rep, _coords(n), [], 1, 2, mode="fedcs")
        counts = Counter(result.values())
        for uav, cnt in counts.items():
            assert cnt <= 2, f"UAV {uav} over capacity"

    def test_excludes_ineligible(self):
        sel = ClientSelector([0, 1, 2])
        covered = {i: 0 for i in range(3)}
        states = {0: _ineligible_state(), 1: _state(), 2: _state()}
        rep = {i: 0.5 for i in range(3)}
        result = sel.select(covered, states, rep, _coords(3), [], 1, 10, mode="fedcs")
        assert 0 not in result

    def test_deadline_stops_greedy_add(self):
        """_fedcs_select stops once the projected round time would exceed T_max.

        Eligible devices always satisfy T̂_n ≤ T_max, so the deadline branch is
        exercised by calling the helper directly with an over-deadline state.
        """
        sel = ClientSelector([0, 1])
        states = {
            0: _state(compute_time_s=100.0),
            1: _state(compute_time_s=T_MAX_S + 50.0),   # would blow the deadline
        }
        result = sel._fedcs_select({0: 0, 1: 0}, states, uav_capacity=10)
        assert result == {0: 0}

    def test_time_blind_to_reputation(self):
        """A slow high-reputation device loses to a fast low-reputation one."""
        sel = ClientSelector([0, 1])
        covered = {0: 0, 1: 0}
        states = {0: _state(compute_time_s=250.0), 1: _state(compute_time_s=50.0)}
        rep = {0: 1.0, 1: 0.0}
        result = sel.select(covered, states, rep, _coords(2), [], 1, 1, mode="fedcs")
        assert set(result.keys()) == {1}

    def test_increments_selection_counts(self):
        sel = ClientSelector([0, 1])
        covered = {0: 0, 1: 0}
        states = {i: _state() for i in range(2)}
        rep = {i: 0.5 for i in range(2)}
        result = sel.select(covered, states, rep, _coords(2), [], 1, 10, mode="fedcs")
        for cid in result:
            assert sel._counts[cid] == 1


# ── Reputation-capability (mode="rep_cap") ───────────────────────────────────

class TestRepCap:
    def test_high_reputation_wins_at_equal_compute(self):
        sel = ClientSelector([0, 1])
        covered = {0: 0, 1: 0}
        states = {i: _state(compute_time_s=100.0) for i in range(2)}
        rep = {0: 0.9, 1: 0.1}
        result = sel.select(covered, states, rep, _coords(2), [], 1, 1, mode="rep_cap")
        assert set(result.keys()) == {0}

    def test_score_formula(self):
        """score = γ·R + (1−γ)·(1 − (T̂/T_max)²)."""
        sel = ClientSelector([0])
        states = {0: _state(compute_time_s=150.0)}
        scores = sel._rep_cap_scores([0], states, {0: 0.6})
        l_feat = 1.0 - (150.0 / T_MAX_S) ** 2
        expected = REPCAP_GAMMA * 0.6 + (1.0 - REPCAP_GAMMA) * l_feat
        assert scores[0] == pytest.approx(expected)

    def test_no_exploration_ranking_is_static(self):
        """Repeated selection never surfaces the starved device (no UCB bonus)."""
        n = 4
        sel = ClientSelector(list(range(n)))
        covered = {i: 0 for i in range(n)}
        states = {i: _state(compute_time_s=100.0) for i in range(n)}
        rep = {0: 0.9, 1: 0.8, 2: 0.7, 3: 0.5}   # device 3: neutral, never picked
        picked_rounds = []
        for rnd in range(1, 11):
            result = sel.select(covered, states, rep, _coords(n), [], rnd, 2, mode="rep_cap")
            picked_rounds.append(frozenset(result.keys()))
        assert all(p == frozenset({0, 1}) for p in picked_rounds)
        assert sel._counts[3] == 0

    def test_respects_capacity(self):
        n = 6
        sel = ClientSelector(list(range(n)))
        covered = {i: i % 2 for i in range(n)}
        states = {i: _state() for i in range(n)}
        rep = {i: 0.5 for i in range(n)}
        result = sel.select(covered, states, rep, _coords(n), [], 1, 2, mode="rep_cap")
        counts = Counter(result.values())
        for uav, cnt in counts.items():
            assert cnt <= 2


# ── Fairness/energy MAB (mode="fair_mab") ────────────────────────────────────

class TestFairMab:
    def test_reward_formula(self):
        """reward = w_e·b + w_s·min(1, staleness/T_stale_cap)."""
        sel = ClientSelector([0])
        states = {0: _state(battery=0.7)}
        # Never selected → staleness = round_num − 0 = 3; cap 5 → 0.6
        scores = sel._fair_mab_scores([0], states, round_num=3, t_stale_cap=5)
        expected = FAIRMAB_W_ENERGY * 0.7 + FAIRMAB_W_STALE * 0.6
        assert scores[0] == pytest.approx(expected)

    def test_staleness_capped_at_one(self):
        sel = ClientSelector([0])
        states = {0: _state(battery=0.0)}
        scores = sel._fair_mab_scores([0], states, round_num=100, t_stale_cap=5)
        assert scores[0] == pytest.approx(FAIRMAB_W_STALE * 1.0)

    def test_stale_device_eventually_selected(self):
        """A starved device accrues staleness and displaces a fresh device."""
        sel = ClientSelector([0, 1])
        covered = {0: 0, 1: 0}
        # Device 0 has better battery; device 1 slightly worse.
        states = {0: _state(battery=0.9), 1: _state(battery=0.7)}
        rep = {i: 0.5 for i in range(2)}
        winners = []
        for rnd in range(1, 8):
            result = sel.select(covered, states, rep, _coords(2), [], rnd, 1,
                                mode="fair_mab", t_stale_cap=5)
            winners.append(next(iter(result.keys())))
        # Device 1 must be picked at least once — fairness pressure works.
        assert 1 in winners
        # And the schedule alternates rather than starving either device.
        assert 0 in winners

    def test_last_selected_bookkeeping(self):
        sel = ClientSelector([0, 1])
        covered = {0: 0, 1: 0}
        states = {i: _state() for i in range(2)}
        rep = {i: 0.5 for i in range(2)}
        result = sel.select(covered, states, rep, _coords(2), [], 4, 1, mode="fair_mab")
        picked = next(iter(result.keys()))
        assert sel._last_selected[picked] == 4
        unpicked = 1 - picked
        assert sel._last_selected[unpicked] == 0

    def test_respects_capacity(self):
        n = 8
        sel = ClientSelector(list(range(n)))
        covered = {i: i % 2 for i in range(n)}
        states = {i: _state() for i in range(n)}
        rep = {i: 0.5 for i in range(n)}
        result = sel.select(covered, states, rep, _coords(n), [], 1, 3, mode="fair_mab")
        counts = Counter(result.values())
        for uav, cnt in counts.items():
            assert cnt <= 3

    def test_excludes_ineligible(self):
        sel = ClientSelector([0, 1])
        covered = {0: 0, 1: 0}
        states = {0: _ineligible_state(), 1: _state()}
        rep = {i: 0.5 for i in range(2)}
        result = sel.select(covered, states, rep, _coords(2), [], 1, 10, mode="fair_mab")
        assert 0 not in result


# ── Cross-mode behaviour ─────────────────────────────────────────────────────

class TestModeDispatch:
    def test_unknown_mode_raises(self):
        sel = ClientSelector([0])
        with pytest.raises(ValueError, match="unknown selection mode"):
            sel.select({0: 0}, {0: _state()}, {0: 0.5}, _coords(1), [], 1, 5,
                       mode="bogus")

    def test_ucb_updates_last_selected(self):
        """Staleness bookkeeping is maintained under every mode, not just fair_mab."""
        sel = ClientSelector([0, 1])
        covered = {0: 0, 1: 0}
        states = {i: _state() for i in range(2)}
        rep = {i: 0.5 for i in range(2)}
        result = sel.select(covered, states, rep, _coords(2), [], 3, 10, mode="ucb")
        for cid in result:
            assert sel._last_selected[cid] == 3


# ── Full-system method wiring ────────────────────────────────────────────────

class TestMethodCfgWiring:
    def test_baselines_registered(self):
        from uavbench.fl.federated import _METHOD_CFG
        for method, mode in [("fedcs", "fedcs"), ("rep_cap", "rep_cap"),
                             ("fair_mab", "fair_mab")]:
            assert method in _METHOD_CFG
            placement, sel_mode, rep_weighted, dynamic = _METHOD_CFG[method]
            assert sel_mode == mode
            # Baselines must be pipeline-identical to proposed_hfl except for
            # the selection rule (isolates it as the experimental variable).
            assert (placement, rep_weighted, dynamic) == ("pso", True, True)

    @pytest.mark.parametrize("method", ["fedcs", "rep_cap", "fair_mab"])
    def test_full_hfl_smoke(self, method):
        """2-round synthetic end-to-end run for each literature baseline."""
        from uavbench.fl.federated import run_full_hfl
        cfg = {
            "methods": [method],
            "fl": {
                "n_rounds": 2, "n_local_epochs": 1, "n_uav_epochs": 1,
                "lr": 0.01, "uav_lr": 0.01, "batch_size": 4,
                "K": 2, "R_comm": 200_000.0, "capacity": 10,
                "T_sel": 1, "lambda_min": 0.0,
                "target_accuracy": 0.99, "seed": 42,
            },
            "budget": {"P": 5, "G_max": 3},
            "data": {"source": "synthetic", "N_clients": 12, "seed": 42},
            "optimizer_seed": 42,
        }
        with tempfile.TemporaryDirectory() as d:
            cfg["results_dir"] = d
            out = run_full_hfl(cfg)
        df = out["rounds"]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert all(df["method"] == method)
        assert df["accuracy"].between(0.0, 1.0).all()
        assert (df["n_selected"] > 0).all(), "baseline selected no clients"
