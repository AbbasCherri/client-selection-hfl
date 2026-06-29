"""Tier-2 FL harness: placement → covered clients → FedAvg → downstream metrics."""

from .federated import run_tier2

__all__ = ["run_tier2"]
