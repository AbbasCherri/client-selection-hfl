"""Tests for beta_schedule, compute_utility, and compute_value (hflsim/shared/value.py)."""

import numpy as np
import pytest

from hflsim.shared.value import beta_schedule, compute_utility, compute_value


# ── beta_schedule ──────────────────────────────────────────────────────────────

class TestBetaSchedule:
    def test_at_t0_returns_1(self):
        assert beta_schedule(0) == pytest.approx(1.0)

    def test_at_t_decay_returns_0(self):
        assert beta_schedule(20, T_decay=20) == pytest.approx(0.0)

    def test_past_t_decay_clamped_to_0(self):
        assert beta_schedule(30, T_decay=20) == pytest.approx(0.0)
        assert beta_schedule(1000) == pytest.approx(0.0)

    def test_halfway_returns_half(self):
        assert beta_schedule(10, T_decay=20) == pytest.approx(0.5)

    def test_decreasing_with_t(self):
        betas = [beta_schedule(t) for t in range(25)]
        for i in range(len(betas) - 1):
            assert betas[i] >= betas[i + 1]

    def test_custom_decay(self):
        assert beta_schedule(5, T_decay=10) == pytest.approx(0.5)


# ── compute_utility ───────────────────────────────────────────────────────────

def _make_inputs(N=5, K=2, seed=0):
    rng = np.random.default_rng(seed)
    device_coords = np.column_stack([
        rng.uniform(0, 1000, N),
        rng.uniform(0, 1000, N),
        np.zeros(N),
    ])
    epicenter = np.array([500.0, 500.0, 0.0])
    snr = rng.uniform(0.0, 30.0, N)
    samples = rng.integers(20, 200, N).astype(float)
    prev_positions = rng.uniform(0, 1000, (K, 3))
    prev_positions[:, 2] = 70.0
    reputation = rng.beta(2, 2, N)
    return device_coords, epicenter, snr, samples, prev_positions, reputation


class TestComputeUtility:
    def test_output_in_01(self):
        dc, epi, snr, samp, prev, _ = _make_inputs()
        u = compute_utility(dc, epi, snr, samp, prev)
        assert u.shape == (5,)
        assert np.all(u >= 0.0) and np.all(u <= 1.0)

    def test_single_device(self):
        dc = np.array([[0.0, 0.0, 0.0]])
        epi = np.array([0.0, 0.0, 0.0])
        snr = np.array([10.0])
        samp = np.array([100.0])
        prev = np.array([[0.0, 0.0, 70.0]])
        u = compute_utility(dc, epi, snr, samp, prev)
        assert u.shape == (1,)
        assert 0.0 <= float(u[0]) <= 1.0

    def test_device_at_epicenter_has_high_u_epi(self):
        # Two devices: one at epicenter, one far away
        dc = np.array([[0.0, 0.0, 0.0], [5000.0, 5000.0, 0.0]])
        epi = np.array([0.0, 0.0, 0.0])
        snr = np.array([10.0, 10.0])
        samp = np.array([100.0, 100.0])
        prev = np.array([[0.0, 0.0, 70.0]])
        u = compute_utility(dc, epi, snr, samp, prev)
        # Device 0 is at epicenter → u_epi max → overall utility higher
        assert u[0] >= u[1], "Device at epicenter should have >= utility"


class TestComputeValue:
    def test_pinned_mode_equals_utility(self):
        dc, epi, snr, samp, prev, rep = _make_inputs()
        utility = compute_utility(dc, epi, snr, samp, prev)
        value = compute_value(dc, epi, snr, samp, prev, rep, t=0, beta_mode="pinned")
        assert np.allclose(utility, value), "pinned mode should return pure utility"

    def test_scheduled_at_t0_equals_utility(self):
        dc, epi, snr, samp, prev, rep = _make_inputs()
        utility = compute_utility(dc, epi, snr, samp, prev)
        value = compute_value(dc, epi, snr, samp, prev, rep, t=0, beta_mode="scheduled")
        assert np.allclose(utility, value)

    def test_scheduled_at_t_decay_equals_reputation(self):
        dc, epi, snr, samp, prev, rep = _make_inputs()
        value = compute_value(dc, epi, snr, samp, prev, rep, t=20, T_decay=20,
                              beta_mode="scheduled")
        assert np.allclose(value, rep), "At T_decay beta=0, value should equal reputation"

    def test_output_in_01(self):
        dc, epi, snr, samp, prev, rep = _make_inputs()
        # With reputation ∈ [0,1] and utility ∈ [0,1], value ∈ [0,1]
        for mode in ("pinned", "scheduled"):
            v = compute_value(dc, epi, snr, samp, prev, rep, t=5, beta_mode=mode)
            assert np.all(v >= 0.0), f"mode={mode}: negative values"
            assert np.all(v <= 1.0), f"mode={mode}: values > 1"

    def test_invalid_beta_mode_raises(self):
        dc, epi, snr, samp, prev, rep = _make_inputs()
        with pytest.raises(ValueError, match="unknown beta_mode"):
            compute_value(dc, epi, snr, samp, prev, rep, beta_mode="bad_mode")
