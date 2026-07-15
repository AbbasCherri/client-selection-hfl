"""Single shared results-table writer (Parquet with loud CSV fallback).

Every harness (Tier-1 runner, Tier-2, full sim, sweeps, stress) must write
result tables through :func:`write_table` — no inline ``try/except`` around
``to_parquet`` anywhere else. The fallback exists only for environments
without pyarrow; it is never silent, and it removes the opposite-format
sibling so a stale file from a previous run can never shadow the fresh one
(readers prefer ``.parquet`` when both exist).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def write_table(df: pd.DataFrame, path: Path) -> Path:
    """Write ``df`` to ``path`` (.parquet), falling back to CSV.

    Returns the path actually written. On fallback, logs the parquet failure
    at WARNING (never swallows it silently) and deletes any stale/partial
    file at the parquet path so downstream readers that prefer parquet
    cannot pick up a previous run's data.
    """
    path = Path(path)
    csv_path = path.with_suffix(".csv")
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:
        logger.warning("Parquet write failed for %s (%s); falling back to CSV", path, exc)
        path.unlink(missing_ok=True)  # remove stale/partial parquet — must not shadow the CSV
        df.to_csv(csv_path, index=False)
        return csv_path
    csv_path.unlink(missing_ok=True)  # remove stale CSV from a previous fallback run
    return path


def read_table(path: Path) -> pd.DataFrame:
    """Read a table written by :func:`write_table` — parquet, or its CSV fallback.

    ``path`` is the ``.parquet`` path; if it is missing, the ``.csv`` sibling
    is tried (mirrors the fallback ``write_table`` may have taken). Raises
    ``FileNotFoundError`` if neither exists.
    """
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"no table at {path} or {csv_path}")
