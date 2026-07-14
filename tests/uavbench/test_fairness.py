"""Jain's fairness index (shared fairness.py module)."""

import numpy as np
import pytest

from uavbench.fl.fairness import jain_index


class TestJainIndex:
    def test_uniform_counts_is_one(self):
        assert jain_index(np.array([5.0, 5.0, 5.0])) == pytest.approx(1.0)

    def test_single_active_client_is_1_over_n(self):
        assert jain_index(np.array([10.0, 0.0, 0.0, 0.0])) == pytest.approx(0.25)

    def test_all_zero_returns_one(self):
        assert jain_index(np.zeros(5)) == pytest.approx(1.0)

    def test_bounded_between_1_over_n_and_1(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            counts = rng.integers(0, 50, size=8).astype(float)
            j = jain_index(counts)
            assert 1.0 / 8 - 1e-12 <= j <= 1.0 + 1e-12

    def test_selection_isolation_alias_points_here(self):
        from uavbench.fl.selection_isolation import _jain_index
        assert _jain_index is jain_index
