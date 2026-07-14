"""Stress-test knobs: black-chip rate, dropout rate, SNR degradation, grid."""

import numpy as np

from uavbench.fl.dataset import SyntheticClientData
from uavbench.fl.device_state import SNR_MIN_DB, DeviceStateManager
from uavbench.fl.stress_sweep import build_stress_grid

# ── SyntheticClientData.black_chip_rate ──────────────────────────────────────

class TestBlackChipRate:
    def test_default_zero_is_bit_identical_to_before(self):
        base = SyntheticClientData(N=100, K=5, seed=7).build()
        again = SyntheticClientData(N=100, K=5, seed=7, black_chip_rate=0.0).build()
        assert np.array_equal(base["img_features"], again["img_features"])
        assert base["client_train_indices"] == again["client_train_indices"]

    def test_exact_black_fraction(self):
        raw = SyntheticClientData(N=200, K=5, seed=7, black_chip_rate=0.3).build()
        n_black = int((np.abs(raw["img_features"]).sum(axis=1) == 0.0).sum())
        assert n_black == round(200 * 0.3)

    def test_partition_unchanged_across_rates(self):
        # Separate chip RNG stream: varying the rate must not reshuffle clients.
        a = SyntheticClientData(N=150, K=6, seed=3, black_chip_rate=0.0).build()
        b = SyntheticClientData(N=150, K=6, seed=3, black_chip_rate=0.2).build()
        assert a["client_train_indices"] == b["client_train_indices"]
        assert a["client_coords"] == b["client_coords"]
        # Non-black rows keep their original features.
        black_mask = np.abs(b["img_features"]).sum(axis=1) == 0.0
        assert np.array_equal(a["img_features"][~black_mask], b["img_features"][~black_mask])


# ── DeviceStateManager knobs ─────────────────────────────────────────────────

class TestDeviceStressKnobs:
    def _mgr(self, **kw):
        return DeviceStateManager(list(range(30)), np.random.default_rng(0), **kw)

    def test_defaults_reproduce_previous_behaviour(self):
        base = DeviceStateManager(list(range(20)), np.random.default_rng(5))
        knobbed = DeviceStateManager(
            list(range(20)), np.random.default_rng(5),
            dropout_rate=0.0, snr_degradation_db=0.0,
        )
        for cid in range(20):
            a, b = base.get_state(cid), knobbed.get_state(cid)
            assert (a.battery, a.snr_db, a.memory_ok, a.compute_time_s) == (
                b.battery, b.snr_db, b.memory_ok, b.compute_time_s
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
