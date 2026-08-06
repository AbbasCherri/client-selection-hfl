"""Placement optimizers, all behind a common :class:`Optimizer` interface."""

from .alzenad2017 import Alzenad2017
from .base import Optimizer, Result
from .ga import GA
from .heuristics import Centroid, RandomPlacement, Static
from .mozaffari2016 import Mozaffari2016
from .pso import PSO
from .pso_plus import PSOPlus

# Registry keyed by the names used in configs.
REGISTRY: dict[str, type[Optimizer]] = {
    "pso": PSO,
    "pso_plus": PSOPlus,
    "ga": GA,
    "centroid": Centroid,
    "random": RandomPlacement,
    "static": Static,
    "mozaffari2016": Mozaffari2016,
    "alzenad2017": Alzenad2017,
}

# Methods that consume the shared P/G_max evaluation budget. Heuristic and
# one-shot literature baselines take no budget, exactly like centroid/static.
# pso_plus is budgeted too: comparing it against pso at a different budget
# would confound "the new features help" with "it got more evaluations".
_BUDGETED_METHODS = ("pso", "pso_plus", "ga")


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
    opt = cls(**kwargs)
    if method in _BUDGETED_METHODS and budget is not None:
        # The paired PSO/GA comparison is void if a constructor ever stops
        # honouring the shared budget — fail here, not in a results table.
        assert (opt.P, opt.G_max) == (budget["P"], budget["G_max"]), (
            f"{method} ignored the shared eval budget: "
            f"got (P={opt.P}, G_max={opt.G_max}), expected {budget}"
        )
    return opt


__all__ = [
    "Optimizer",
    "Result",
    "PSO",
    "PSOPlus",
    "GA",
    "Centroid",
    "RandomPlacement",
    "Static",
    "Mozaffari2016",
    "Alzenad2017",
    "REGISTRY",
    "build_optimizer",
]
