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


def sweep_gate_catches_a_single_collapsed_cell():
    """A collapsed cell must fail the sweep even when the method's mean is fine.

    This is the property that matters on a sweep and the one a per-method gate
    does not have. The coverage sweep spans radii whose bottom end is expected
    to collapse (the 800 m class-realism arm ended at macro-F1 0.207, below the
    floor); averaging that cell together with the healthy 5 km cell yields a
    comfortable mean and a green gate over a third of the grid being unusable.

    Exercised through the CLI rather than the library, because the averaging bug
    lived in the CLI's groupby, not in check_not_collapsed.
    """
    import subprocess
    import tempfile

    import pandas as pd

    repo = Path(__file__).resolve().parents[2]
    rows, conf = [], []
    # Two radii, same method: 5000 m healthy, 500 m on the floor.
    for r_comm, macro, minority in ((5000, 0.38, 0.15), (500, 0.21, 0.002)):
        for seed in range(3):
            rows.append({
                "method": "mclp_place", "R_comm": r_comm, "seed": seed, "round": 100,
                "macro_f1": macro, "accuracy": 0.80,
                "f1_survived": 0.90, "f1_collapsed": 0.20,
                "f1_obstructed": 0.15, "f1_missing": minority,
            })
    for true_label, n in zip(("survived", "collapsed", "obstructed", "missing"),
                             (10570, 1180, 640, 480)):
        conf.append({"round": 100, "true_label": true_label,
                     "pred_label": true_label, "count": n})

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        pd.DataFrame(rows).to_parquet(d / "coverage_sweep_rounds.parquet")
        pd.DataFrame(conf).to_parquet(d / "confusion.parquet")

        mean_macro = sum(r["macro_f1"] for r in rows) / len(rows)
        assert mean_macro - constant_predictor_macro_f1(NOTO_COUNTS) > 0.05, (
            f"fixture is not adversarial: the pooled mean {mean_macro:.3f} already "
            "fails the margin test, so this would pass even with per-method gating"
        )

        out = subprocess.run(
            [sys.executable, str(repo / "scripts" / "gate_collapse.py"), str(d)],
            capture_output=True, text=True,
        )
        assert out.returncode == 1, (
            "gate PASSED a sweep with a collapsed cell — per-cell gating is not "
            f"in effect.\nstdout: {out.stdout}\nstderr: {out.stderr}"
        )
        assert "R_comm=500" in out.stdout, (
            f"gate failed but did not name the collapsed cell:\n{out.stdout}"
        )
        assert "R_comm=5000" not in out.stdout.split("DEGENERATE")[-1], (
            f"the healthy 5000 m cell was reported as degenerate:\n{out.stdout}"
        )


check("closed-form baseline matches brute force", closed_form_baseline_matches_brute_force)
check("Noto constant-predictor baseline is ~0.225", noto_baseline_is_where_we_think)
check("every known collapsed run is caught", every_known_collapsed_run_is_caught)
check("the working run is not flagged", the_working_run_is_not_flagged)
check("macro-F1 alone would not have caught them", macro_f1_alone_would_not_have_caught_them)
check("degenerate inputs raise", degenerate_inputs_raise)
check("sweep gate catches a single collapsed cell", sweep_gate_catches_a_single_collapsed_cell)
finish()
