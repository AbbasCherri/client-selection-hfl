"""Selection-frequency fairness metrics shared by the FL harnesses.

Jain's fairness index (R. Jain, D.-M. Chiu, W. Hawe, "A Quantitative
Measure of Fairness and Discrimination for Resource Allocation in Shared
Computer Systems," DEC TR-301, 1984) over cumulative per-client selection
counts. Distinct from the placement-side load-imbalance term: this
measures how evenly *selection frequency* is spread across clients over
rounds, not how evenly devices are assigned to UAVs within one round.
"""

from __future__ import annotations

import numpy as np


def jain_index(counts: np.ndarray) -> float:
    """Jain's fairness index over cumulative selection counts.

    ``J(x) = (Σx)² / (n·Σx²)``, bounded in ``[1/n, 1]``; 1 = perfectly
    fair. All-zero counts (no selections yet) return 1.0 by convention —
    nobody has been favoured.
    """
    total = counts.sum()
    if total <= 0:
        return 1.0
    return float(total**2 / (len(counts) * (counts**2).sum()))
