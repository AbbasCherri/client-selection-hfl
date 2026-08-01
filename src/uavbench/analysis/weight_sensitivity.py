"""Tier-1 fitness-weight sensitivity: is the method ranking an artifact of w?

The Tier-1 scalarization weights w=(0.811, 0.03, 0.159) came from an Optuna
study whose objective was downstream ``proposed_hfl`` macro-F1, and PSO is then
declared the winner *on that objective*. That is circular unless the ranking is
shown to be invariant to the weights. This module produces that evidence.

Two classes of method, and only one of them costs compute
---------------------------------------------------------
``Fitness`` is

    F = w1·(F_cover/F_max) − w2·(D_move/D_max) − w3·(L_imb/L_max)

and ``runs.parquet`` already stores ``f_cover_norm``, ``d_move_m`` and
``l_imb``, with both remaining normalizers determined by the config
(``D_max = K·‖diag(box)‖``, ``L_max = N²``). So F under *any* weights is
recoverable by arithmetic — **provided the placement itself did not depend on
the weights**.

* **Weight-independent** (:data:`RESCORABLE`): ``centroid``, ``static``,
  ``mozaffari2016``, ``alzenad2017``. Each constructs its placement
  geometrically and calls ``fitness()`` exactly once, to report a score. Their
  positions cannot move with w, so re-scoring is exact and free.
* **Weight-dependent** (:data:`MUST_RERUN`): ``pso``, ``ga`` — which search
  against the weighted objective — **and ``random``**, which is
  ``RandomPlacement(n_draws=20)``: it draws 20 candidates and keeps the best
  *by fitness*, so its output does move with w. Re-scoring these would silently
  report the fitness of a placement that method would never have chosen.

:func:`verify_rescoring_exact` checks the identity against the stored values
before any re-scored number is used; it must return 0.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Placement is constructed geometrically and scored once -> re-scoring is exact.
RESCORABLE: tuple[str, ...] = ("centroid", "static", "mozaffari2016", "alzenad2017")
# Placement is *selected* by fitness -> the positions themselves move with w.
MUST_RERUN: tuple[str, ...] = ("pso", "ga", "random")


def weight_grid(base: tuple[float, float, float] = (0.811, 0.03, 0.159)) -> dict[str, tuple]:
    """The sweep: w1 varied, (w2, w3) redistributed in their tuned proportion,
    plus the paper's originally stated split as an external reference point."""
    _, w2, w3 = base
    share = w2 / (w2 + w3)  # preserve the tuned movement:imbalance ratio
    grid: dict[str, tuple[float, float, float]] = {}
    for w1 in (0.6, 0.7, 0.811, 0.9):
        rest = 1.0 - w1
        grid[f"w1={w1:g}"] = (w1, rest * share, rest * (1.0 - share))
    grid["paper_0.6/0.3/0.1"] = (0.6, 0.3, 0.1)
    return grid


def _normalizers(area: dict, N: int, K: int) -> tuple[float, float]:
    """``(D_max, L_max)`` exactly as :class:`~uavbench.problem.fitness.Fitness`
    computes them (``K·box_diagonal`` and ``N²``)."""
    span = np.array(
        [area["x"][1] - area["x"][0], area["y"][1] - area["y"][0], area["z"][1] - area["z"][0]],
        dtype=np.float64,
    )
    return float(K * np.sqrt((span**2).sum())), float(N) ** 2


def rescore(
    df: pd.DataFrame, weights: tuple[float, float, float], area: dict, N: int, K: int
) -> pd.Series:
    """Fitness of each stored run under ``weights``, from its stored components."""
    w1, w2, w3 = weights
    d_max, l_max = _normalizers(area, N, K)
    return w1 * df["f_cover_norm"] - w2 * (df["d_move_m"] / d_max) - w3 * (df["l_imb"] / l_max)


def verify_rescoring_exact(
    df: pd.DataFrame, weights: tuple[float, float, float], area: dict, N: int, K: int
) -> float:
    """Max |re-scored − stored| at the weights the runs were *actually* scored
    at. Any non-zero result means the stored components no longer determine the
    stored fitness, and every re-scored number below is void."""
    return float((rescore(df, weights, area, N, K) - df["final_fitness"]).abs().max())


def sensitivity_table(
    runs: pd.DataFrame, area: dict, N: int, K: int, base: tuple[float, float, float]
) -> pd.DataFrame:
    """Long-form (weight_setting × scenario × method) mean fitness + rank.

    Covers only :data:`RESCORABLE`. The weight-dependent methods must be re-run
    per weight setting and merged in afterwards — see :data:`MUST_RERUN`.
    """
    err = verify_rescoring_exact(runs, base, area, N, K)
    if err > 0.0:
        raise ValueError(
            f"re-scoring identity does not hold (max abs error {err:.3e}); "
            "stored components no longer determine stored fitness"
        )

    sub = runs[runs["method"].isin(RESCORABLE)]
    rows = []
    for label, w in weight_grid(base).items():
        scores = rescore(sub, w, area, N, K)
        tmp = sub.assign(fitness_w=scores)
        agg = (
            tmp.groupby(["scenario", "method"])["fitness_w"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg["weight_setting"] = label
        agg["w1"], agg["w2"], agg["w3"] = w
        agg["rank_within_scenario"] = agg.groupby("scenario")["mean"].rank(
            ascending=False, method="min"
        )
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def rank_invariance(table: pd.DataFrame) -> pd.DataFrame:
    """Per (scenario, method): does its rank move across weight settings?

    ``rank_stable`` False anywhere is the honest headline — it would mean the
    Tier-1 ordering is a function of a weight choice that was itself tuned.
    """
    g = table.groupby(["scenario", "method"])["rank_within_scenario"]
    out = g.agg(rank_min="min", rank_max="max", rank_nunique="nunique").reset_index()
    out["rank_stable"] = out["rank_min"] == out["rank_max"]
    return out
