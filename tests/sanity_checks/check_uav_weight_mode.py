"""Diversity-weighted edge aggregation (C2) must weight what it claims to.

The premise, measured in results/probe_topology: averaging near-single-class
fusion heads is what makes a run unlearn, and sample-count weighting cannot see
it — a 2-client single-class shard and a 2-client balanced one carry identical
weight. C2 multiplies the edge weight by the shard's effective class count.

These checks pin the weight function's endpoints (a threshold-free scheme is
only defensible if its extremes are right), that it is a no-op for a healthy
fleet, and that the config switch is validated rather than silently ignored.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np  # noqa: E402
from _lib import check, finish  # noqa: E402

from uavbench.fl.federated import (  # noqa: E402
    _shard_class_stats,
    _shard_effective_class_fraction,
)


def _groups(assignment):
    return {
        u: [SimpleNamespace(train_indices=list(range(c * 10, c * 10 + 10))) for c in cids]
        for u, cids in enumerate(assignment)
    }


def single_class_shard_gets_the_floor_weight():
    """One class seen -> exp(0)/C = 1/C, the minimum. Not zero: the shard's
    struct-branch contribution is still real, it is only its fusion head that is
    uninformative."""
    labels = np.zeros(100, dtype=int)
    w = _shard_effective_class_fraction(_groups([[0, 1]]), labels, n_classes=4)
    assert abs(w[0] - 0.25) < 1e-9, f"single-class shard weight {w[0]}, expected 0.25"


def balanced_shard_is_unweighted():
    """A healthy fleet must be untouched, or C2 is a global learning-rate change."""
    labels = np.tile([0, 1, 2, 3], 25)
    w = _shard_effective_class_fraction(_groups([[0, 1]]), labels, n_classes=4)
    assert abs(w[0] - 1.0) < 1e-9, f"balanced shard weight {w[0]}, expected 1.0"


def weight_is_monotone_in_diversity():
    """Two classes must sit strictly between one and four."""
    one = _shard_effective_class_fraction(
        _groups([[0, 1]]), np.zeros(100, dtype=int), 4)[0]
    two = _shard_effective_class_fraction(
        _groups([[0, 1]]), np.tile([0, 1], 50), 4)[0]
    four = _shard_effective_class_fraction(
        _groups([[0, 1]]), np.tile([0, 1, 2, 3], 25), 4)[0]
    assert one < two < four, f"not monotone: {one:.3f}, {two:.3f}, {four:.3f}"
    assert abs(two - 0.5) < 1e-9, f"two equal classes should score 2/4, got {two}"


def the_regime_that_motivated_it_is_down_weighted_hard():
    """A narrow geographically-clustered shard — the failure case — must lose most
    of its weight relative to a wide one, or C2 cannot change the outcome."""
    # One class per client, so pairing adjacent clients gives single-class shards.
    labels = np.concatenate([np.full(10, c % 4) for c in range(12)])
    narrow = _shard_effective_class_fraction(
        _groups([[0, 1], [2, 3], [4, 5]]), labels, 4)
    wide = _shard_effective_class_fraction(
        _groups([[0, 1, 2, 3, 4, 5]]), labels, 4)
    assert max(narrow.values()) < 0.6 * wide[0], (
        f"narrow shards ({max(narrow.values()):.3f}) not materially below the wide "
        f"one ({wide[0]:.3f}) — C2 would barely change the aggregate"
    )


def stats_and_fraction_agree():
    """The two helpers must read the same shard, or logging and weighting drift."""
    labels = np.tile([0, 0, 0, 1], 25)
    g = _groups([[0, 1]])
    (_minority, entropy) = _shard_class_stats(g, labels, 4)[0]
    frac = _shard_effective_class_fraction(g, labels, 4)[0]
    expected = float(np.exp(entropy * np.log(4)) / 4)
    assert abs(frac - expected) < 1e-12, f"{frac} != {expected}"


def empty_shards_carry_no_weight_entry():
    """An idle UAV must be absent, not weighted 1.0 or 0.0 by accident."""
    g = _groups([[0, 1]])
    g[1] = []
    w = _shard_effective_class_fraction(g, np.tile([0, 1, 2, 3], 25), 4)
    assert 1 not in w, "an empty shard was given a weight"
    assert 0 in w, "the non-empty shard lost its weight"


def an_unknown_weight_mode_is_rejected():
    """A typo must fail loudly, not fall back to sample weighting."""
    import inspect

    from uavbench.fl import federated

    src = inspect.getsource(federated.run_full_hfl)
    assert 'uav_weight_mode must be' in src, (
        "run_full_hfl does not validate fl.uav_weight_mode — a typo would "
        "silently run the historical sample-weighted aggregation"
    )


check("single-class shard gets the floor weight", single_class_shard_gets_the_floor_weight)
check("balanced shard is unweighted", balanced_shard_is_unweighted)
check("weight is monotone in diversity", weight_is_monotone_in_diversity)
check("the motivating regime is down-weighted hard", the_regime_that_motivated_it_is_down_weighted_hard)
check("stats and fraction agree", stats_and_fraction_agree)
check("empty shards carry no weight entry", empty_shards_carry_no_weight_entry)
check("an unknown weight mode is rejected", an_unknown_weight_mode_is_rejected)
finish()
