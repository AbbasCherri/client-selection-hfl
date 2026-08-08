"""The collapse gate must fire on the runs that actually collapsed.

A detector that never fires is worse than none — it certifies degenerate output.
So this pins it against the real numbers from 2026-08-08: three configurations
that produced complete, plausible-looking results in which the model had learned
only the majority class, and the one configuration that genuinely worked.

The closed-form baseline is verified against a brute-force confusion matrix
rather than trusted, because the whole gate is calibrated against it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np  # noqa: E402
from _lib import check, finish  # noqa: E402

from uavbench.analysis.collapse import (  # noqa: E402
    check_not_collapsed,
    constant_predictor_macro_f1,
)

# Noto training-label prior, ~82% "survived" (4 classes).
NOTO_COUNTS = np.array([10_570, 1_180, 640, 480], dtype=np.float64)

# Measured, not invented — see the module docstring in analysis/collapse.py.
#   (label, macro_f1, [f1_survived, f1_collapsed, f1_obstructed, f1_missing])
COLLAPSED_RUNS = [
    ("2km K=20 cap=6 PSO", 0.2872, [0.8995, 0.1795, 0.0660, 0.0038]),
    ("2km K=60 cap=2 PSO", 0.2661, [0.8978, 0.1217, 0.0389, 0.0062]),
    ("2km K=60 cap=2 mclp_ls", 0.2619, [0.8986, 0.1274, 0.0199, 0.0016]),
]
HEALTHY_RUN = ("20km K=20 cap=6", 0.5262, [0.8899, 0.4756, 0.3401, 0.3992])


def _brute_force_constant_macro_f1(counts: np.ndarray) -> float:
    """Macro-F1 of the always-majority predictor, computed from a confusion matrix."""
    counts = np.asarray(counts, dtype=np.float64)
    m = int(np.argmax(counts))
    f1s = []
    for c in range(counts.size):
        tp = counts[c] if c == m else 0.0
        fp = (counts.sum() - counts[m]) if c == m else 0.0
        fn = 0.0 if c == m else counts[c]
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def closed_form_baseline_matches_brute_force():
    """The gate is calibrated on this formula, so it must be right."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        counts = rng.integers(1, 5000, size=int(rng.integers(2, 7))).astype(np.float64)
        got = constant_predictor_macro_f1(counts)
        want = _brute_force_constant_macro_f1(counts)
        assert abs(got - want) < 1e-12, f"counts={counts}: {got} != {want}"


def noto_baseline_is_where_we_think():
    """Sanity-anchor the number the whole diagnosis rests on."""
    b = constant_predictor_macro_f1(NOTO_COUNTS)
    assert 0.20 < b < 0.24, f"Noto constant-predictor macro-F1 is {b:.4f}, expected ~0.225"


def every_known_collapsed_run_is_caught():
    """The three runs that wasted a day must all fail the gate."""
    for label, macro, per_class in COLLAPSED_RUNS:
        v = check_not_collapsed(macro, per_class, NOTO_COUNTS)
        assert not v.ok, f"{label}: gate PASSED a run that collapsed — {v}"
        assert v.reason, f"{label}: flagged without a reason"


def the_working_run_is_not_flagged():
    """Guard the guard: a gate that rejects everything is useless."""
    label, macro, per_class = HEALTHY_RUN
    v = check_not_collapsed(macro, per_class, NOTO_COUNTS)
    assert v.ok, f"{label}: gate rejected the run that actually learned — {v}"


def macro_f1_alone_would_not_have_caught_them():
    """Justifies the two-condition design instead of a single macro-F1 bar.

    If a margin test alone sufficed, the per-class condition would be dead
    weight. It does not: the collapsed runs clear the baseline by 0.04-0.06, so
    a margin loose enough to admit a merely-hard task would admit them too.
    """
    baseline = constant_predictor_macro_f1(NOTO_COUNTS)
    margins = [macro - baseline for _, macro, _ in COLLAPSED_RUNS]
    assert max(margins) > 0.0, "collapsed runs sit below the baseline; margin test alone would do"
    # The per-class condition separates them cleanly where the margin does not.
    worst_collapsed = max(min(pc) for _, _, pc in COLLAPSED_RUNS)
    worst_healthy = min(HEALTHY_RUN[2])
    assert worst_healthy > 10 * worst_collapsed, (
        f"per-class floor does not separate: healthy min F1 {worst_healthy:.4f} "
        f"vs worst collapsed {worst_collapsed:.4f}"
    )


def degenerate_inputs_raise():
    """Empty or zero-count inputs must fail loudly, not return a verdict."""
    for bad in (np.zeros(4), np.array([])):
        try:
            constant_predictor_macro_f1(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted degenerate class_counts {bad}")
    try:
        check_not_collapsed(0.5, [], NOTO_COUNTS)
    except ValueError:
        return
    raise AssertionError("accepted empty per_class_f1")


check("closed-form baseline matches brute force", closed_form_baseline_matches_brute_force)
check("Noto constant-predictor baseline is ~0.225", noto_baseline_is_where_we_think)
check("every known collapsed run is caught", every_known_collapsed_run_is_caught)
check("the working run is not flagged", the_working_run_is_not_flagged)
check("macro-F1 alone would not have caught them", macro_f1_alone_would_not_have_caught_them)
check("degenerate inputs raise", degenerate_inputs_raise)
finish()
