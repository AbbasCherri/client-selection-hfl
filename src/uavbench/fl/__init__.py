"""Tier-2 FL harness: placement → covered clients → FedAvg → downstream metrics."""

from .federated import run_tier2
from .sweep import run_sweep

__all__ = ["run_tier2", "run_sweep"]
