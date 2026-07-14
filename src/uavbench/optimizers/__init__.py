"""Placement optimizers, all behind a common :class:`Optimizer` interface."""

from .alzenad2017 import Alzenad2017
from .base import Optimizer, Result
from .ga import GA
from .heuristics import Centroid, RandomPlacement, Static
from .mozaffari2016 import Mozaffari2016
from .pso import PSO

# Registry keyed by the names used in configs.
REGISTRY: dict[str, type[Optimizer]] = {
    "pso": PSO,
    "ga": GA,
    "centroid": Centroid,
    "random": RandomPlacement,
    "static": Static,
    "mozaffari2016": Mozaffari2016,
    "alzenad2017": Alzenad2017,
}

# Methods that consume the shared P/G_max evaluation budget. Heuristic and
# one-shot literature baselines take no budget, exactly like centroid/static.
_BUDGETED_METHODS = ("pso", "ga")


def build_optimizer(
    method: str,
    params: dict | None = None,
    budget: dict | None = None,
) -> Optimizer:
    """Instantiate a registered optimizer, enforcing the shared eval budget.

    ``params`` carries method-specific keyword arguments (e.g. the
    ``optimizer_params.<method>`` block of a config YAML). For budgeted
    metaheuristics, ``budget["P"]``/``budget["G_max"]`` always take precedence
    over ``params`` so no config can desync the paired PSO/GA comparison.
    This is the single construction path shared by the Tier-1 runner, the
    Tier-2/3 FL bridge, and the legacy hflsim bridge.
    """
    cls = REGISTRY[method]
    kwargs: dict = dict(params or {})
    if method in _BUDGETED_METHODS and budget is not None:
        kwargs.update(P=budget["P"], G_max=budget["G_max"])
    return cls(**kwargs)


__all__ = [
    "Optimizer",
    "Result",
    "PSO",
    "GA",
    "Centroid",
    "RandomPlacement",
    "Static",
    "Mozaffari2016",
    "Alzenad2017",
    "REGISTRY",
    "build_optimizer",
]
