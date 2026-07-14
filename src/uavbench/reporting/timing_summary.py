"""Aggregate wall-clock reporting over already-persisted run tables.

Per-run timing is instrumented everywhere (``wall_time_s`` per optimizer run
in Tier-1's runs.parquet; ``round_time_s`` per FL round in the Tier-2 /
full-system / selection-isolation rounds tables) — this module only adds the
aggregate view the paper's runtime-disclosure section needs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# (filename, timing column) pairs recognised across the harnesses; the first
# table found in the results directory wins.
_TABLES: tuple[tuple[str, str], ...] = (
    ("runs.parquet", "wall_time_s"),
    ("tier2_rounds.parquet", "round_time_s"),
    ("fullsim_rounds.parquet", "round_time_s"),
    ("selection_rounds.parquet", "round_time_s"),
    ("sweep_rounds.parquet", "round_time_s"),
    ("paper_sweep_rounds.parquet", "round_time_s"),
    ("stress_rounds.parquet", "round_time_s"),
)


def _read_table(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    csv = path.with_suffix(".csv")
    if csv.exists():
        return pd.read_csv(csv)
    return None


def summarize_wall_clock(results_dir: Path) -> pd.DataFrame:
    """Per-method wall-clock summary from a results directory's run tables.

    Returns one row per method with mean/std/total seconds and the row count
    the aggregate is over (runs for Tier-1, rounds for the FL harnesses).
    Empty DataFrame if no recognised table is present.
    """
    results_dir = Path(results_dir)
    for fname, col in _TABLES:
        df = _read_table(results_dir / fname)
        if df is None or col not in df.columns or df.empty:
            continue
        grouped = df.groupby("method")[col]
        summary = pd.DataFrame(
            {
                "mean_s": grouped.mean(),
                "std_s": grouped.std().fillna(0.0),
                "total_s": grouped.sum(),
                "n": grouped.count(),
            }
        ).reset_index()
        summary.insert(1, "source", fname)
        return summary.round({"mean_s": 3, "std_s": 3, "total_s": 1})
    return pd.DataFrame(columns=["method", "source", "mean_s", "std_s", "total_s", "n"])
