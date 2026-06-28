"""Placement optimizers, all behind a common :class:`Optimizer` interface."""

from .base import Optimizer, Result
from .pso import PSO
from .ga import GA
from .heuristics import Centroid, RandomPlacement, Static

# Registry keyed by the names used in configs.
REGISTRY: dict[str, type[Optimizer]] = {
    "pso": PSO,
    "ga": GA,
    "centroid": Centroid,
    "random": RandomPlacement,
    "static": Static,
}

__all__ = ["Optimizer", "Result", "PSO", "GA", "Centroid", "RandomPlacement", "Static", "REGISTRY"]
