"""Reported metrics.

``placement`` (imported eagerly, numpy-only) holds Tier-1 optimizer metrics;
``fl`` holds the FL-side metrics (classification, Jain fairness, comm cost)
and is imported lazily by the FL harnesses because it pulls in torch/sklearn.
"""

from .placement import compute_metrics, convergence_auc, evals_to_threshold

__all__ = ["compute_metrics", "convergence_auc", "evals_to_threshold"]
