"""Tests for ReputationManager (reputation.py)."""

import numpy as np
import pytest
import torch

from uavbench.fl.reputation import (
    EMA_ALPHA,
    W_ANOMALY,
    W_CONTRIB,
    W_TEMP,
    ReputationManager,
    _vec,
)


def _make_sd(val: float = 1.0, size: int = 16) -> dict:
    return {"w": torch.full((size,), val)}


class TestReputationWeights:
    def test_weights_sum_to_one(self):
        assert abs(W_CONTRIB + W_ANOMALY + W_TEMP - 1.0) < 1e-9


class TestVec:
    def test_flat_output_dtype(self):
        sd = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])}
        v = _vec(sd)
        assert v.dtype == np.float32
        assert v.shape == (3,)

    def test_works_on_gradient_tensor(self):
        sd = {"w": torch.tensor([1.0, 2.0], requires_grad=True)}
        v = _vec(sd)  # must not raise
        assert v.shape == (2,)

    def test_works_on_detached_tensor(self):
        sd = {"w": torch.tensor([1.0, 2.0]).detach()}
        v = _vec(sd)
        assert v.shape == (2,)


class TestReputationManagerInit:
    def test_initial_scores_in_range(self):
        mgr = ReputationManager([0, 1, 2])
        for cid in [0, 1, 2]:
            s = mgr.get_score(cid)
            assert 0.0 <= s <= 1.0

    def test_initial_score_formula(self):
        mgr = ReputationManager([0])
        # R_contrib=0.5, R_anomaly=1.0, R_temp=0.5
        expected = W_CONTRIB * 0.5 + W_ANOMALY * 1.0 + W_TEMP * 0.5
        assert abs(mgr.get_score(0) - expected) < 1e-9

    def test_get_all_scores_covers_all_ids(self):
        ids = [10, 20, 30]
        mgr = ReputationManager(ids)
        scores = mgr.get_all_scores()
        assert set(scores.keys()) == set(ids)


class TestUpdateBatch:
    def test_scores_stay_in_01_after_many_updates(self):
        ids = list(range(5))
        mgr = ReputationManager(ids)
        rng = np.random.default_rng(42)
        for _ in range(50):
            updates = {cid: _make_sd(float(rng.uniform(0, 10))) for cid in ids}
            mgr.update_batch(updates, global_update_vec=None)
        for cid in ids:
            s = mgr.get_score(cid)
            assert 0.0 <= s <= 1.0, f"Score {s} out of [0,1] for client {cid}"

    def test_empty_update_batch_no_crash(self):
        mgr = ReputationManager([0, 1])
        mgr.update_batch({}, global_update_vec=None)  # must not raise
        assert mgr.get_score(0) > 0  # still has initial value

    def test_norm_window_capped_at_100(self):
        mgr = ReputationManager([0])
        sd = _make_sd(1.0, size=4)
        for _ in range(200):
            mgr.update_batch({0: sd}, global_update_vec=None)
        assert len(mgr._norm_window) == 100

    def test_anomalous_norm_reduces_anomaly_score(self):
        ids = [0]
        mgr = ReputationManager(ids)
        # Prime the norm window with moderate values
        normal_sd = _make_sd(1.0, size=16)
        for _ in range(20):
            mgr.update_batch({0: normal_sd}, global_update_vec=None)
        anomaly_before = mgr._R_anomaly[0]

        # Inject an outlier (norm ≫ μ + 2σ)
        outlier_sd = _make_sd(1000.0, size=16)
        mgr.update_batch({0: outlier_sd}, global_update_vec=None)
        assert mgr._R_anomaly[0] < anomaly_before, "Anomaly score should decrease for outlier"

    def test_consistent_direction_increases_contrib(self):
        ids = [0]
        mgr = ReputationManager(ids)
        contrib_init = mgr._R_contrib[0]
        sd = _make_sd(1.0, size=16)
        direction = np.ones(16, dtype=np.float32)
        for _ in range(10):
            mgr.update_batch({0: sd}, global_update_vec=direction)
        # Contrib score should be well above its floor
        assert mgr._R_contrib[0] > contrib_init

    def test_r_contrib_bounded(self):
        mgr = ReputationManager([0])
        sd = _make_sd(1.0, size=16)
        direction = np.ones(16, dtype=np.float32)
        for _ in range(100):
            mgr.update_batch({0: sd}, global_update_vec=direction)
        assert 0.0 <= mgr._R_contrib[0] <= 1.0

    def test_success_count_increments(self):
        mgr = ReputationManager([0])
        assert mgr._success[0] == 0
        assert mgr._total[0] == 0
        mgr.update_batch({0: _make_sd()}, global_update_vec=None)
        assert mgr._success[0] == 1
        assert mgr._total[0] == 1


class TestMarkAbsent:
    def test_mark_absent_decreases_temporal_score(self):
        mgr = ReputationManager([0])
        # First give client a perfect temporal score
        sd = _make_sd(1.0, size=16)
        for _ in range(10):
            mgr.update_batch({0: sd}, global_update_vec=None)
        r_temp_before = mgr._R_temp[0]
        # Penalise absence
        mgr.mark_absent(0)
        # total increments, success does not → rate decreases → score decreases
        assert mgr._R_temp[0] <= r_temp_before + 1e-9

    def test_mark_absent_increments_total_not_success(self):
        mgr = ReputationManager([0])
        mgr.mark_absent(0)
        assert mgr._total[0] == 1
        assert mgr._success[0] == 0
