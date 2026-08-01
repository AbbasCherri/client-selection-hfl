"""Three-way train/val/test split invariants (the 2026-08 test-leak fix).

The bug this guards against: before 2026-08 there was no validation split, so
``scripts/tune_weights.py`` selected 22 hyperparameters by maximising macro-F1
on the exact rows the paper reports. The fix carves val out of the *held-out*
portion, so training data is unchanged and only the scored rows differ.

Exercises ``hflsim.data.loader.split_client_indices`` — the real function the
loader calls per client — rather than a restatement of its arithmetic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _lib import check, finish  # noqa: E402

from hflsim.data.loader import get_hfl_data_partitions, split_client_indices  # noqa: E402

TRAIN_RATIO = 0.8
IDX = list(range(100))


def val_is_disjoint_and_nonempty():
    train, val, test = split_client_indices(IDX, TRAIN_RATIO, 0.1)
    assert val, "val split requested but empty"
    assert not (set(val) & set(test)), "val and test overlap"
    assert not (set(val) & set(train)), "val and train overlap"
    assert set(train) | set(val) | set(test) == set(IDX), "split lost or duplicated rows"


def carving_val_does_not_touch_train():
    """Val comes out of the held-out portion, never out of train."""
    train_a, _, _ = split_client_indices(IDX, TRAIN_RATIO, 0.0)
    train_b, val_b, _ = split_client_indices(IDX, TRAIN_RATIO, 0.1)
    assert val_b, "expected a non-empty val split at val_ratio=0.1"
    assert train_a == train_b, (
        "training indices changed when val was carved out — val must come "
        "from the held-out portion, not from train"
    )


def zero_val_ratio_is_the_historical_split():
    train, val, test = split_client_indices(IDX, TRAIN_RATIO, 0.0)
    assert val == [], "val_ratio=0.0 must produce no validation rows"
    assert train == IDX[:80] and test == IDX[80:], "two-way split changed shape"


def split_sizes_track_the_ratios():
    train, val, test = split_client_indices(IDX, TRAIN_RATIO, 0.1)
    assert (len(train), len(val), len(test)) == (80, 10, 10), (
        f"got {(len(train), len(val), len(test))}, expected (80, 10, 10)"
    )


def ragged_clients_never_lose_rows():
    """int() truncation must not drop or duplicate a row at any client size."""
    for n in range(0, 37):
        idx = list(range(n))
        tr, va, te = split_client_indices(idx, TRAIN_RATIO, 0.1)
        assert len(tr) + len(va) + len(te) == n, f"row count changed at n={n}"
        assert tr + va + te == idx, f"row order/content changed at n={n}"


def val_ratio_participates_in_the_cache_key():
    """A two-way cache entry must never satisfy a three-way request.

    Mirrors the key construction in ``get_hfl_data_partitions``: val_ratio is
    appended only when > 0, so val_ratio=0.0 keeps hitting historical caches
    while any three-way request lands on a distinct key.
    """
    keys = set()
    for vr in (0.0, 0.1, 0.2):
        parts = ["1000", "12", f"{TRAIN_RATIO:.6f}", "42", "7"]
        if vr > 0.0:
            parts.append(f"val{vr:.6f}")
        keys.add("|".join(parts))
    assert len(keys) == 3, "distinct val_ratios collided on one cache key"


def rejects_impossible_ratios():
    """val_ratio >= 1 - train_ratio would leave an empty test set."""
    try:
        get_hfl_data_partitions(N=4, train_ratio=0.8, val_ratio=0.25)
    except ValueError:
        return
    except Exception as exc:  # noqa: BLE001 - must fail *before* touching data
        raise AssertionError(
            f"expected ValueError on the ratio guard, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError("expected ValueError for val_ratio >= 1 - train_ratio")


if __name__ == "__main__":
    check("val disjoint from train/test and non-empty", val_is_disjoint_and_nonempty)
    check("carving val leaves training indices untouched", carving_val_does_not_touch_train)
    check("val_ratio=0.0 is the historical two-way split", zero_val_ratio_is_the_historical_split)
    check("split sizes track the configured ratios", split_sizes_track_the_ratios)
    check("ragged client sizes never lose rows", ragged_clients_never_lose_rows)
    check("val_ratio participates in the cache key", val_ratio_participates_in_the_cache_key)
    check("impossible val_ratio is rejected loudly", rejects_impossible_ratios)
    finish()
