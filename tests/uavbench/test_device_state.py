"""Tests for DeviceState and DeviceStateManager (device_state.py)."""

import numpy as np
import pytest

from uavbench.fl.device_state import (
    B_MIN,
    SNR_MIN_DB,
    T_MAX_S,
    DeviceState,
    DeviceStateManager,
)


# ── DeviceState.eligible() ────────────────────────────────────────────────────

class TestDeviceStateEligible:
    def _state(self, battery=0.5, snr_db=10.0, memory_ok=True, compute_time_s=100.0):
        return DeviceState(battery, snr_db, memory_ok, compute_time_s)

    def test_fully_healthy_is_eligible(self):
        assert self._state().eligible()

    # Battery gate
    def test_battery_exactly_at_threshold_is_eligible(self):
        assert self._state(battery=B_MIN).eligible()

    def test_battery_just_below_threshold_is_ineligible(self):
        assert not self._state(battery=B_MIN - 1e-9).eligible()

    def test_battery_zero_is_ineligible(self):
        assert not self._state(battery=0.0).eligible()

    # SNR gate
    def test_snr_exactly_at_threshold_is_eligible(self):
        assert self._state(snr_db=SNR_MIN_DB).eligible()

    def test_snr_just_below_threshold_is_ineligible(self):
        assert not self._state(snr_db=SNR_MIN_DB - 1e-9).eligible()

    def test_negative_snr_is_ineligible(self):
        assert not self._state(snr_db=-5.0).eligible()

    # Memory gate
    def test_memory_failure_is_ineligible(self):
        assert not self._state(memory_ok=False).eligible()

    # Compute time gate
    def test_compute_time_exactly_at_threshold_is_eligible(self):
        assert self._state(compute_time_s=T_MAX_S).eligible()

    def test_compute_time_just_above_threshold_is_ineligible(self):
        assert not self._state(compute_time_s=T_MAX_S + 1e-9).eligible()

    def test_all_gates_failing_is_ineligible(self):
        assert not self._state(
            battery=0.0, snr_db=-1.0, memory_ok=False, compute_time_s=1000.0
        ).eligible()


# ── DeviceStateManager ────────────────────────────────────────────────────────

class TestDeviceStateManager:
    def _make(self, n=5, seed=0):
        ids = list(range(n))
        rng = np.random.default_rng(seed)
        return DeviceStateManager(ids, rng), ids

    def test_initial_battery_in_range(self):
        mgr, ids = self._make(20)
        for cid in ids:
            st = mgr.get_state(cid)
            assert 0.5 <= st.battery <= 1.0

    def test_initial_snr_in_range(self):
        mgr, ids = self._make(20)
        for cid in ids:
            # snr_base in [5, 20], noise initialised to 0
            st = mgr.get_state(cid)
            assert 5.0 <= st.snr_db <= 20.0, f"Initial SNR {st.snr_db} out of [5,20]"

    def test_initial_compute_in_range(self):
        mgr, ids = self._make(20)
        for cid in ids:
            st = mgr.get_state(cid)
            # base in [50,250], noise initialised to 0, clamped at 10
            assert 50.0 <= st.compute_time_s <= 250.0

    def test_memory_failure_rate_approx_10pct(self):
        n = 200
        mgr, ids = self._make(n, seed=7)
        failures = sum(1 for cid in ids if not mgr.get_state(cid).memory_ok)
        # Allow generous tolerance given small sample
        assert 5 <= failures <= 35, f"Memory failure count {failures} outside [5,35] for n=200"

    def test_selected_clients_discharge(self):
        mgr, ids = self._make()
        before = {cid: mgr.get_state(cid).battery for cid in ids}
        mgr.update_round(selected_ids={0})
        after = {cid: mgr.get_state(cid).battery for cid in ids}
        # Client 0 must have discharged
        assert after[0] < before[0]

    def test_unselected_clients_recharge(self):
        mgr, ids = self._make()
        # Force battery to a mid value so recharge is visible
        mgr._battery[1] = 0.6
        mgr.update_round(selected_ids={0})
        assert mgr.get_state(1).battery > 0.6 - 1e-9

    def test_battery_never_below_zero(self):
        mgr, ids = self._make()
        mgr._battery[0] = 0.01
        for _ in range(10):
            mgr.update_round(selected_ids={0})
        assert mgr.get_state(0).battery >= 0.0

    def test_battery_never_above_one(self):
        mgr, ids = self._make()
        mgr._battery[1] = 0.999
        for _ in range(10):
            mgr.update_round(selected_ids=set())
        assert mgr.get_state(1).battery <= 1.0

    def test_get_all_states_covers_all_clients(self):
        mgr, ids = self._make()
        states = mgr.get_all_states()
        assert set(states.keys()) == set(ids)
        for cid, st in states.items():
            assert isinstance(st, DeviceState)

    def test_compute_time_clamped_above_10(self):
        mgr, ids = self._make()
        # Force base compute to minimum and add large negative noise
        mgr._compute_base[0] = 50.0
        mgr._compute_noise[0] = -1000.0
        st = mgr.get_state(0)
        assert st.compute_time_s >= 10.0

    def test_update_round_different_seeds_differ(self):
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(99)
        mgr1 = DeviceStateManager([0, 1, 2], rng1)
        mgr2 = DeviceStateManager([0, 1, 2], rng2)
        mgr1.update_round(set())
        mgr2.update_round(set())
        snr1 = mgr1.get_state(0).snr_db
        snr2 = mgr2.get_state(0).snr_db
        # Different seeds → different noise
        assert snr1 != snr2
