"""Reputation manager: weight simplex, adaptation cadence, score bounds."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch
from _lib import check, finish

from uavbench.fl.reputation import W_ANOMALY, W_CONTRIB, W_TEMP, ReputationManager, trimmed_mean


def _make_sd(val: float = 1.0, size: int = 16) -> dict:
    return {"w": torch.full((size,), val)}


def weights_simplex():
    assert abs(W_CONTRIB + W_ANOMALY + W_TEMP - 1.0) < 1e-9


def bayesian_adaptation_cadence():
    def run_rounds(mgr, n):
        rng = np.random.default_rng(0)
        for _ in range(n):
            updates = {cid: _make_sd(float(rng.normal(1.0, 0.3))) for cid in (0, 1, 2)}
            mgr.update_batch(updates, global_update_vec=None)

    mgr = ReputationManager([0, 1, 2])
    w0 = mgr._weights.copy()
    run_rounds(mgr, 9)
    assert np.array_equal(mgr._weights, w0)  # unchanged before round 10
    run_rounds(mgr, 1)
    assert not np.array_equal(mgr._weights, w0)  # adapts at round 10
    run_rounds(mgr, 15)  # second adaptation event
    assert abs(mgr._weights.sum() - 1.0) < 1e-9 and np.all(mgr._weights > 0.0)


def scores_bounded():
    mgr = ReputationManager([0, 1, 2])
    scores = mgr.get_all_scores()
    assert set(scores) == {0, 1, 2}
    assert all(0.0 <= s <= 1.0 for s in scores.values())
    rng = np.random.default_rng(1)
    for _ in range(50):
        updates = {cid: _make_sd(float(rng.normal(1.0, 0.5))) for cid in (0, 1, 2)}
        mgr.update_batch(updates, global_update_vec=None)
    assert all(0.0 <= s <= 1.0 for s in mgr.get_all_scores().values())
    mgr.update_batch({}, global_update_vec=None)  # empty batch: no crash


def absence_penalised():
    mgr = ReputationManager([0])
    for _ in range(10):
        mgr.update_batch({0: _make_sd(1.0)}, global_update_vec=None)
    r_temp_before = mgr._R_temp[0]
    mgr.mark_absent(0)
    # total increments, success does not -> success rate (temporal) decreases.
    assert mgr._R_temp[0] <= r_temp_before + 1e-9
    assert mgr._total[0] == 11 and mgr._success[0] == 10


def trimmed_mean_robust():
    # floor(n*0.1) values trimmed per tail: needs n >= 10 to remove the outlier.
    vals = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.03, 0.97, 50.0]
    tm = trimmed_mean(vals)
    assert 0.8 < tm < 1.3, f"trimmed mean {tm} not robust to the outlier"
    # Small-n behaviour: no trimming, plain mean (documented paper rule).
    assert abs(trimmed_mean([1.0, 2.0, 3.0]) - 2.0) < 1e-12


check("reputation weights form a simplex", weights_simplex)
check("Bayesian weight adaptation fires exactly at its cadence", bayesian_adaptation_cadence)
check("scores stay in [0,1] over many rounds; empty batch safe", scores_bounded)
check("absence reduces temporal reputation", absence_penalised)
check("trimmed mean is robust to an outlier cluster member", trimmed_mean_robust)
finish()
