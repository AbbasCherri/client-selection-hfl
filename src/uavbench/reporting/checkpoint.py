"""Per-job checkpoint files for resumable sweeps.

Only the Tier-1 grid (``runner.py``) needs this: its jobs return metrics +
a convergence trace with no natural on-disk artifact of their own, unlike
the Tier-2 / full-sim / selection-isolation jobs, which already persist a
per-job results table (resume there is a skip-if-exists check against that
existing file, not a new format).

A checkpoint is written exactly once, after a job finishes, via a
write-to-temp-then-rename so a process killed mid-write never leaves a
checkpoint that looks valid but is truncated — :func:`load_checkpoint`
treats anything it cannot unpickle as "not done" rather than raising, so a
half-written file is simply redone on the next run instead of poisoning it.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def save_checkpoint(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)  # atomic on POSIX — no partial file ever visible at `path`


def load_checkpoint(path: Path) -> Any | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logger.warning("Checkpoint %s unreadable (%s) — treating as incomplete, will redo", path, exc)
        return None
