# LEGACY: this package predates the live experimental path in
# ``uavbench.fl`` (run_tier2 / run_full_hfl / selection_isolation) and is
# retained for two remaining consumers only:
#   1. ``hflsim.placement`` imports :class:`UAVAggregator` (the thin
#      placement-integration bridge);
#   2. the standalone ``hflsim`` CLI (`python -m hflsim`), which still runs
#      the original single-N simulation end to end.
# Do NOT build new experiments on these classes — use ``uavbench.fl``.
# See REPORTS/master_implementation_reference.md for the migration story.

from .client import IoTClient, RandomProjection, get_flat_fusion_weights, get_fusion_params
from .coordinator import ClientSelectionCoordinator
from .orchestrator import HFLOrchestrator
from .uav import UAVAggregator

__all__ = [
    "IoTClient",
    "UAVAggregator",
    "ClientSelectionCoordinator",
    "HFLOrchestrator",
    "RandomProjection",
    "get_fusion_params",
    "get_flat_fusion_weights",
]
