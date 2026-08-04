"""Guards on scripts/tune_weights.py — the hyperparameter-selection leak fixes.

Two properties matter here and neither is visible in a run's output, which is
why they get a check rather than a code comment:

1. **The objective cannot read the test column.** The original tuner maximised
   ``macro_f1`` — the reported test metric — so all 22 hyperparameters were
   chosen by looking at the answer. A silent fallback to that column would
   reintroduce the leak invisibly, so ``_score`` must RAISE when the val column
   is missing or all-NaN, and must ignore ``macro_f1`` when both are present.

2. **Stored trials replay identically.** ``--transfer-check`` re-scores stored
   parameter sets through ``optuna.trial.FixedTrial``. If replay produced
   different weights than the original trial, the transfer numbers would refer
   to a config nobody ever tuned — a silently wrong validation of a silently
   wrong config.
"""

import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _lib import check, finish  # noqa: E402

from tune_weights import (  # noqa: E402
    RESERVED_EVAL_SEEDS,
    _sample_weights,
    _score,
)


def _rounds_df(val, test) -> pd.DataFrame:
    return pd.DataFrame({
        "round": range(len(val)),
        "val_macro_f1": val,
        "macro_f1": test,
    })


def score_reads_val_not_test():
    """Same val curve, wildly different test curve -> identical score."""
    val = [0.30] * 10
    a = _score(_rounds_df(val, [0.90] * 10))
    b = _score(_rounds_df(val, [0.10] * 10))
    assert a == b, f"score changed with the TEST column only: {a} vs {b}"
    # and it is actually the val level, not some blend
    assert abs(a - 0.30) < 1e-9, f"expected the val level 0.30, got {a}"


def score_refuses_missing_val_column():
    df = pd.DataFrame({"round": range(10), "macro_f1": [0.5] * 10})
    try:
        _score(df)
    except KeyError:
        return
    raise AssertionError(
        "scored a run with no val_macro_f1 column — that is the test-set leak"
    )


def score_refuses_all_nan_val():
    df = _rounds_df([np.nan] * 10, [0.5] * 10)
    try:
        _score(df)
    except ValueError:
        return
    raise AssertionError(
        "scored a run whose val column is all-NaN (val_ratio unset) instead of "
        "refusing — this silently degrades to no validation at all"
    )


def score_penalises_instability():
    """Two runs with equal final mean; the oscillating one must score lower."""
    steady = _score(_rounds_df([0.40] * 10, [0.0] * 10))
    noisy = _score(_rounds_df([0.30, 0.50] * 5, [0.0] * 10))
    assert noisy < steady, (
        f"instability penalty inactive: noisy={noisy} !< steady={steady}"
    )


def fixed_trial_replay_is_identical():
    """The transfer check must re-run the SAME config it stored."""
    for space in ("recipe", "full"):
        study = optuna.create_study(direction="maximize")
        captured = {}

        def objective(trial, _space=space):
            captured["w"] = _sample_weights(trial, _space)
            return 0.0

        study.optimize(objective, n_trials=1)
        original = captured["w"]
        replayed = _sample_weights(
            optuna.trial.FixedTrial(study.trials[0].params), space
        )
        assert original.keys() == replayed.keys(), (
            f"[{space}] replay produced different keys: "
            f"{set(original) ^ set(replayed)}"
        )
        for k, v in original.items():
            if isinstance(v, float):
                assert abs(v - replayed[k]) < 1e-12, f"[{space}] {k}: {v} != {replayed[k]}"
            else:
                assert v == replayed[k], f"[{space}] {k}: {v} != {replayed[k]}"


def recipe_space_omits_selector_constants():
    """A baseline must not be handed proposed-method-only constants."""
    study = optuna.create_study(direction="maximize")
    got = {}

    def objective(trial):
        got["w"] = _sample_weights(trial, "recipe")
        return 0.0

    study.optimize(objective, n_trials=1)
    for k in ("sel_static_blend", "w_epi", "w_contrib", "fitness_w1"):
        assert k not in got["w"], f"recipe space leaked selector constant {k!r}"
    for k in ("lr", "lr_decay", "logit_adjust_tau"):
        assert k in got["w"], f"recipe space is missing shared recipe knob {k!r}"


def eval_seeds_are_reserved():
    """The seeds the paper evaluates on must be off-limits to tuning."""
    for s in (0, 9, 14, 19):
        assert s in RESERVED_EVAL_SEEDS, f"eval seed {s} is not reserved from tuning"
    for s in (20, 21, 22, 23, 24):
        assert s not in RESERVED_EVAL_SEEDS, f"tuning seed {s} collides with eval range"


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    check("score reads val, not test", score_reads_val_not_test)
    check("score refuses a missing val column", score_refuses_missing_val_column)
    check("score refuses an all-NaN val column", score_refuses_all_nan_val)
    check("score penalises instability", score_penalises_instability)
    check("FixedTrial replay is identical", fixed_trial_replay_is_identical)
    check("recipe space omits selector constants", recipe_space_omits_selector_constants)
    check("evaluation seeds are reserved from tuning", eval_seeds_are_reserved)
    finish()
