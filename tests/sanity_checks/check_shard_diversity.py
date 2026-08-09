"""Shard-diversity instrumentation must measure what it claims to measure.

This column exists to settle a question that cost a day: whether a collapsed run
had too few participants, or enough participants each pooled into shards too
narrow to train a fusion head on. Those are different failures with different
fixes, and `n_selected` cannot tell them apart — K=60/cap=2 and K=20/cap=6 both
select ~115 clients. So the metric has to be right, and it has to be sensitive
to capacity at fixed participation, or it answers nothing.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np  # noqa: E402
from _lib import check, finish  # noqa: E402

from uavbench.fl.federated import _shard_class_diversity  # noqa: E402


def _groups(assignment: list[list[int]]) -> dict[int, list]:
    """assignment[u] = list of client ids for UAV u; client c owns indices [c*10, c*10+10)."""
    return {
        u: [SimpleNamespace(train_indices=list(range(c * 10, c * 10 + 10))) for c in cids]
        for u, cids in enumerate(assignment)
    }


def single_class_shard_scores_zero():
    labels = np.zeros(100, dtype=int)
    minority, entropy = _shard_class_diversity(_groups([[0, 1]]), labels)
    assert minority == 0.0, f"single-class shard has minority share {minority}"
    assert entropy == 0.0, f"single-class shard has entropy {entropy}"


def uniform_shard_scores_one():
    labels = np.tile([0, 1, 2, 3], 25)
    minority, entropy = _shard_class_diversity(_groups([[0, 1]]), labels)
    assert abs(minority - 0.75) < 1e-9, f"uniform 4-class shard minority share {minority}"
    assert abs(entropy - 1.0) < 1e-9, f"uniform 4-class shard entropy {entropy}"


def capacity_is_detected_at_fixed_participation():
    """The load-bearing property: sensitive to shard width, not to headcount.

    Twelve clients participate either way. Clustered geography is simulated by
    giving each client a single class, so pairing adjacent clients (cap=2)
    yields near-single-class shards while pooling six (cap=6) does not. If this
    ever stops separating, the column cannot support the claim it was added for.
    """
    labels = np.concatenate([np.full(10, c % 4) for c in range(12)])
    narrow = _groups([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])  # 6 UAVs x 2
    wide = _groups([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])            # 2 UAVs x 6
    n_min, n_ent = _shard_class_diversity(narrow, labels)
    w_min, w_ent = _shard_class_diversity(wide, labels)
    assert w_ent > n_ent + 0.2, (
        f"cap=6 shards ({w_ent:.3f}) not meaningfully broader than cap=2 ({n_ent:.3f}) — "
        "the metric is insensitive to the variable it exists to isolate"
    )
    assert w_min > n_min, f"minority share did not rise with capacity: {n_min} -> {w_min}"


def empty_shards_are_skipped_not_counted_as_narrow():
    """A UAV with no clients must not be averaged in as a zero-diversity shard."""
    labels = np.tile([0, 1, 2, 3], 25)
    with_empties = _groups([[0, 1]])
    with_empties[1] = []
    with_empties[2] = []
    minority, entropy = _shard_class_diversity(with_empties, labels)
    assert abs(entropy - 1.0) < 1e-9, (
        f"empty shards dragged entropy to {entropy} — idle UAVs would be reported "
        "as single-class shards and mask a healthy fleet"
    )
    assert abs(minority - 0.75) < 1e-9, f"empty shards dragged minority share to {minority}"


def no_uav_tier_returns_nan():
    """flat_fl has no UAV tier; it must report NaN rather than a fabricated 0."""
    minority, entropy = _shard_class_diversity({}, np.zeros(10, dtype=int))
    assert np.isnan(minority) and np.isnan(entropy), (
        f"empty fleet reported ({minority}, {entropy}) instead of NaN — a 0 here "
        "would read as a maximally degenerate shard in every flat_fl row"
    )


check("single-class shard scores zero", single_class_shard_scores_zero)
check("uniform shard scores one", uniform_shard_scores_one)
check("capacity is detected at fixed participation", capacity_is_detected_at_fixed_participation)
check("empty shards are skipped, not counted as narrow", empty_shards_are_skipped_not_counted_as_narrow)
check("no UAV tier returns NaN", no_uav_tier_returns_nan)
finish()
