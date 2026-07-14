"""Stress-test knobs: black-chip rate, dropout rate, SNR degradation, grid."""

import numpy as np

from uavbench.fl.device_state import SNR_MIN_DB, DeviceStateManager
from uavbench.fl.federated import _apply_black_chips
from uavbench.fl.stress_sweep import build_stress_grid

# ── _apply_black_chips (harness-level, data-source-agnostic) ─────────────────


class TestApplyBlackChips:
    def _features(self, n=200, seed=7):
        return np.random.default_rng(seed).standard_normal((n, 512)).astype(np.float32)

    def test_rate_zero_returns_input_unchanged(self):
        feats = self._features()
        out = _apply_black_chips(feats, 0.0, seed=7)
        assert out is feats  # identity fast-path: no copy, no RNG draw

    def test_exact_black_fraction(self):
        feats = self._features(n=200)
        out = _apply_black_chips(feats, 0.3, seed=7)
        n_black = int((np.abs(out).sum(axis=1) == 0.0).sum())
        assert n_black == round(200 * 0.3)

    def test_input_not_mutated_and_survivors_intact(self):
        # Copy semantics keep on-disk feature caches pristine, and rows not
        # selected for blacking keep their original values bit-for-bit.
        feats = self._features(n=150, seed=3)
        before = feats.copy()
        out = _apply_black_chips(feats, 0.2, seed=3)
        assert np.array_equal(feats, before)
        black_mask = np.abs(out).sum(axis=1) == 0.0
        assert np.array_equal(out[~black_mask], feats[~black_mask])

    def test_deterministic_given_seed(self):
        feats = self._features()
        a = _apply_black_chips(feats, 0.2, seed=11)
        b = _apply_black_chips(feats, 0.2, seed=11)
        assert np.array_equal(a, b)
        c = _apply_black_chips(feats, 0.2, seed=12)
        assert not np.array_equal(a, c)


# ── DeviceStateManager knobs ─────────────────────────────────────────────────


class TestDeviceStressKnobs:
    def _mgr(self, **kw):
        return DeviceStateManager(list(range(30)), np.random.default_rng(0), **kw)

    def test_defaults_reproduce_previous_behaviour(self):
        base = DeviceStateManager(list(range(20)), np.random.default_rng(5))
        knobbed = DeviceStateManager(
            list(range(20)),
            np.random.default_rng(5),
            dropout_rate=0.0,
            snr_degradation_db=0.0,
        )
        for cid in range(20):
            a, b = base.get_state(cid), knobbed.get_state(cid)
            assert (a.battery, a.snr_db, a.memory_ok, a.compute_time_s) == (
                b.battery,
                b.snr_db,
                b.memory_ok,
                b.compute_time_s,
            )

    def test_full_dropout_blocks_everyone(self):
        mgr = self._mgr(dropout_rate=1.0)
        assert all(not st.eligible() for st in mgr.get_all_states().values())

    def test_partial_dropout_reduces_eligibility(self):
        rng_states = self._mgr().get_all_states()
        n_base = sum(st.eligible() for st in rng_states.values())
        n_drop = sum(st.eligible() for st in self._mgr(dropout_rate=0.5).get_all_states().values())
        assert n_drop < n_base

    def test_snr_degradation_shifts_all_devices(self):
        base = self._mgr()
        deg = self._mgr(snr_degradation_db=6.0)
        for cid in range(30):
            assert deg.get_state(cid).snr_db == base.get_state(cid).snr_db - 6.0

    def test_huge_snr_degradation_fails_gate(self):
        # Base SNR is uniform [5, 20] dB + N(0, 2) noise; -100 dB sinks all
        # devices far below SNR_MIN_DB.
        mgr = self._mgr(snr_degradation_db=100.0)
        for st in mgr.get_all_states().values():
            assert st.snr_db < SNR_MIN_DB
            assert not st.eligible()


# ── stress grid construction ─────────────────────────────────────────────────


class TestStressGrid:
    CFG = {
        "dropout_rates": [0.0, 0.1, 0.2],
        "snr_degradations_db": [0.0, 3.0],
        "black_chip_rates": [0.0, 0.05, 0.10],
    }

    def test_one_axis_at_a_time_default(self):
        cells = build_stress_grid(self.CFG)
        # baseline + 2 dropout + 1 snr + 2 chip = 6 cells
        assert len(cells) == 6
        assert cells[0] == (0.0, 0.0, 0.0)
        # Every non-baseline cell deviates on exactly one axis.
        for d, s, c in cells[1:]:
            assert sum(v != 0.0 for v in (d, s, c)) == 1

    def test_full_grid(self):
        cells = build_stress_grid({**self.CFG, "full_grid": True})
        assert len(cells) == 3 * 2 * 3
