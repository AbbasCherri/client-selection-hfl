"""Placement optimizers, all behind a common :class:`Optimizer` interface."""

from .alzenad2017 import Alzenad2017
from .base import Optimizer, Result
from .ga import GA
from .geometric import CapacitatedKMeans, SpiralPlacement
from .heuristics import Centroid, RandomPlacement, Static
from .mclp_ls import MCLPLocalSearch
from .mozaffari2016 import Mozaffari2016
from .pso import PSO
from .pso_plus import PSOPlus
from .recent_baselines import HierarchicalPlacement, PSOClusterPlacement
from .swarm_baselines import DifferentialEvolution, GreyWolfOptimizer

# Registry keyed by the names used in configs.
REGISTRY: dict[str, type[Optimizer]] = {
    "pso": PSO,
    "pso_plus": PSOPlus,
    "ga": GA,
    "de": DifferentialEvolution,
    "gwo": GreyWolfOptimizer,
    "mclp_ls": MCLPLocalSearch,
    "centroid": Centroid,
    "cap_kmeans": CapacitatedKMeans,
    "spiral": SpiralPlacement,
    "pso_cluster": PSOClusterPlacement,
    "ahc": HierarchicalPlacement,
    "random": RandomPlacement,
    "static": Static,
    "mozaffari2016": Mozaffari2016,
    "alzenad2017": Alzenad2017,
}

# Methods that consume the shared P/G_max evaluation budget. Heuristic and
# one-shot literature baselines take no budget, exactly like centroid/static.
# pso_plus is budgeted too: comparing it against pso at a different budget
# would confound "the new features help" with "it got more evaluations".
# mclp_ls is budgeted for the same reason in the opposite direction — it is the
# proposed method, so it must be shown to win *at* PSO's spend, not past it.
_BUDGETED_METHODS = ("pso", "pso_plus", "ga", "de", "gwo", "mclp_ls", "pso_cluster")


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
    "DifferentialEvolution",
    "GreyWolfOptimizer",
    "MCLPLocalSearch",
    "Centroid",
    "CapacitatedKMeans",
    "SpiralPlacement",
    "PSOClusterPlacement",
    "HierarchicalPlacement",
    "RandomPlacement",
    "Static",
    "Mozaffari2016",
    "Alzenad2017",
    "REGISTRY",
    "build_optimizer",
]
