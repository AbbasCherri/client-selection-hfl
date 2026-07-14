"""Statistical analysis over persisted run tables."""

from .significance import paired_seed_test, pairwise_significance_table

__all__ = ["paired_seed_test", "pairwise_significance_table"]
