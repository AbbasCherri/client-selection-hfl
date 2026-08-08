"""Majority-class collapse detection — the gate that makes "it ran" ≠ "it worked".

Why this exists
---------------
On 2026-08-08 three consecutive configurations produced complete, well-formed,
fully-checkpointed results in which the model had learned nothing beyond the
majority class:

    2 km, K=20, cap=6, PSO       macro-F1 0.287, f1_missing 0.0040
    2 km, K=60, cap=2, PSO       macro-F1 0.266, f1_missing 0.0062
    2 km, K=60, cap=2, mclp_ls   macro-F1 0.262, f1_missing 0.0016

Nothing in the pipeline objected. Every sanity check passed, every parquet was
written, and the sweep would happily have spent three days producing 1000 more
cells exactly like them. The only reason they were caught is that someone read
`f1_missing` by hand.

The diagnosis was reached through a proxy — "participation fell below ~120
clients per round" — inferred from a *single* working run. That number is not
established: it is one observation, it confounds participation with radius, and
nothing tests it. So this module does not encode it. It gates on the outcome
that actually matters, which needs no guessed threshold:

    a run is degenerate if its macro-F1 is not meaningfully better than the
    best CONSTANT predictor, or if some class is essentially never predicted.

Both quantities are computable from the run's own outputs, and the first has a
closed form, so the bar is derived rather than chosen.

The constant-predictor baseline
-------------------------------
A classifier that always answers with the most common class ``m`` gets recall 1
and precision ``p_m`` on that class, so ``F1_m = 2*p_m / (1 + p_m)``, and F1 = 0
on every other class. Macro-F1 averages over ``C`` classes:

    macro_f1_const = 2 * p_m / ((1 + p_m) * C)

At the Noto label prior (p_m ~ 0.82, C = 4) that is ~0.225 — which is why the
three runs above, at 0.26-0.29, are barely distinguishable from predicting
"survived" every time, despite looking like plausible mid-range scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def constant_predictor_macro_f1(class_counts: np.ndarray) -> float:
    """Macro-F1 of always predicting the most frequent class.

    ``class_counts`` is the label histogram of the evaluation set. Returns the
    floor any real model must clear to have learned anything at all.
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError(f"class_counts must be a 1-D histogram; got shape {counts.shape}")
    total = counts.sum()
    if total <= 0:
        raise ValueError("class_counts sums to zero — no evaluation data")
    p_m = float(counts.max() / total)
    return 2.0 * p_m / ((1.0 + p_m) * counts.size)


@dataclass(frozen=True)
class CollapseVerdict:
    """Outcome of the gate. ``ok`` False means the run is not a usable result."""

    ok: bool
    macro_f1: float
    baseline: float
    margin: float
    min_class_f1: float
    reason: str

    def __str__(self) -> str:  # pragma: no cover - display only
        state = "OK" if self.ok else "DEGENERATE"
        return (
            f"[{state}] macro_f1={self.macro_f1:.4f} vs constant-predictor "
            f"baseline {self.baseline:.4f} (margin {self.margin:+.4f}), "
            f"min per-class F1={self.min_class_f1:.4f}"
            + ("" if self.ok else f" — {self.reason}")
        )


def check_not_collapsed(
    macro_f1: float,
    per_class_f1: dict[str, float] | np.ndarray,
    class_counts: np.ndarray,
    *,
    min_margin: float = 0.05,
    min_class_f1: float = 0.05,
) -> CollapseVerdict:
    """Gate a finished run against majority-class collapse.

    Parameters
    ----------
    min_margin:
        How far above the constant-predictor baseline macro-F1 must sit. 0.05 is
        deliberately loose — it is a floor for "learned something", not a
        quality bar. All three degenerate runs above cleared the baseline by
        0.04-0.06 and would sit at or under it.
    min_class_f1:
        No class may be essentially never predicted. This is the condition that
        separated the collapsed runs most cleanly: f1_missing ran 0.0016-0.0062
        while the healthy run was an order of magnitude higher.

    Both conditions must hold. macro-F1 alone is not enough: it averages, so a
    strong majority class can carry a run whose minority classes are dead.
    """
    f1_values = (
        np.asarray(list(per_class_f1.values()), dtype=np.float64)
        if isinstance(per_class_f1, dict)
        else np.asarray(per_class_f1, dtype=np.float64)
    )
    if f1_values.size == 0:
        raise ValueError("per_class_f1 is empty — cannot assess collapse")

    baseline = constant_predictor_macro_f1(class_counts)
    margin = float(macro_f1) - baseline
    worst = float(np.min(f1_values))

    reasons = []
    if margin < min_margin:
        reasons.append(
            f"macro-F1 is only {margin:+.4f} above the constant-predictor "
            f"baseline {baseline:.4f} (need >= {min_margin})"
        )
    if worst < min_class_f1:
        reasons.append(
            f"a class has F1 {worst:.4f} (need >= {min_class_f1}) — it is "
            "essentially never predicted"
        )
    return CollapseVerdict(
        ok=not reasons,
        macro_f1=float(macro_f1),
        baseline=baseline,
        margin=margin,
        min_class_f1=worst,
        reason="; ".join(reasons),
    )
