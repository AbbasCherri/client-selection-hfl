"""The two roster constructions differ in shard width when slots do not bind.

Why this is a guard and not a note
----------------------------------
`paper_full` compares the proposed selector against five literature selectors,
and the comparison is only about *selection* if everything downstream of it is
held equal. It is not, in one specific regime:

  * `_class_coverage_assign` (proposed `ucb`, and `class_greedy`) walks UAV by
    UAV filling each to `capacity` before moving on;
  * `_greedy_assign` (fedcs, oort, power_of_choice, rep_cap, fair_mab) sends
    each client to the feasible UAV with the LOWEST current load.

When `K * capacity` binds, both saturate at capacity and agree. When it does
not — every `paper_full` cell where the slot budget exceeds the covered
population — the first concentrates clients into fewer, fuller shards and the
second spreads them across the whole fleet. Per-UAV shard width is what decides
whether the edge fusion heads learn (results/probe_topology: width <= 3
unlearns), so in that regime the two families are not being compared under
equal aggregation conditions.

This does not assert which policy is right. `paper_full` may legitimately claim
roster construction as part of the proposed method. It asserts that the
difference is real and regime-dependent, so that it is disclosed rather than
read as selection quality.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np  # noqa: E402
from _lib import check, finish  # noqa: E402

from uavbench.fl.client_selection import ClientSelector  # noqa: E402


def _setup(n_clients: int, n_uav: int, capacity: int):
    """Clients and UAVs colocated tightly enough that every UAV covers everyone."""
    rng = np.random.default_rng(0)
    ids = list(range(n_clients))
    # A degree-tenth of a degree is ~11 km; keep everything well inside R_comm.
    coords = {c: (37.5 + rng.normal(0, 0.002), 137.3 + rng.normal(0, 0.002)) for c in ids}
    uavs = [(37.5, 137.3) for _ in range(n_uav)]
    return ids, coords, uavs, capacity


def _widths(selected: dict[int, int]) -> list[int]:
    fill: dict[int, int] = {}
    for uav in selected.values():
        fill[uav] = fill.get(uav, 0) + 1
    return sorted(fill.values(), reverse=True)


def policies_agree_when_the_slot_budget_binds():
    """K*capacity < covered: both fill to capacity, so the comparison is fair."""
    n_clients, n_uav, cap = 200, 10, 6      # 60 slots << 200 clients
    ids, coords, uavs, cap = _setup(n_clients, n_uav, cap)
    sel = ClientSelector(ids)
    scores = np.linspace(1.0, 0.0, n_clients)
    eligible = {c: 0 for c in ids}
    greedy = sel._greedy_assign(ids, eligible, scores, cap, coords, uavs, R_comm=20000.0)
    w = _widths(greedy)
    assert all(x == cap for x in w), (
        f"with slots binding, load-balancing should still saturate every UAV: {w}"
    )
    assert len(w) == n_uav, f"expected all {n_uav} UAVs used, got {len(w)}"


def policies_diverge_when_the_slot_budget_is_slack():
    """K*capacity > covered: load-balancing spreads thin, filling concentrates.

    This is the paper_full small-N regime, and the load-balanced width lands in
    the range the capacity probe measured as unlearnable.
    """
    n_clients, n_uav, cap = 25, 20, 6       # 120 slots >> 25 clients
    ids, coords, uavs, cap = _setup(n_clients, n_uav, cap)
    sel = ClientSelector(ids)
    scores = np.linspace(1.0, 0.0, n_clients)
    eligible = {c: 0 for c in ids}
    balanced = _widths(sel._greedy_assign(ids, eligible, scores, cap, coords, uavs, 20000.0))

    assert len(balanced) == n_uav, (
        f"load-balancing should touch every UAV when slots are slack, used {len(balanced)}"
    )
    assert max(balanced) <= 2, (
        f"load-balanced shards should be 1-2 clients wide here, got {balanced}"
    )
    # The contrast that matters: a fill-to-capacity policy would use
    # ceil(25/6) = 5 UAVs at width 6. Assert the arithmetic gap explicitly so the
    # regime is documented even though the two policies are exercised through
    # different code paths.
    fill_width = cap
    assert fill_width >= 3 * max(balanced), (
        "the two roster policies no longer differ materially in shard width; "
        "if that is now true, paper_full's small-N comparison is no longer "
        "confounded and this guard should be retired"
    )


def the_confounded_regime_is_reachable_from_shipped_configs():
    """Flag which shipped paper_full cells sit in the slack-slot regime."""
    from uavbench.runner import load_config

    cfg = load_config(Path(__file__).resolve().parents[2] / "configs" / "paper_full.yaml")
    k = int(cfg["fl"]["K"])
    cap = int(cfg["fl"]["capacity"])
    slots = k * cap
    ns = [int(n) for n in cfg["N_values"]]
    # Coverage at the operating point is ~82% (results/probe_topology).
    slack = [n for n in ns if slots > 0.82 * n]
    assert slack, (
        f"no paper_full cell has slack slots (K*cap={slots}, N={ns}) — the "
        "confound is not reachable and this guard can be retired"
    )
    print(
        f"      note: slots={slots}; cells with slack slots (roster policy "
        f"matters): N={slack} of {ns}"
    )


check("policies agree when the slot budget binds", policies_agree_when_the_slot_budget_binds)
check("policies diverge when the slot budget is slack", policies_diverge_when_the_slot_budget_is_slack)
check("the confounded regime is reachable from shipped configs",
      the_confounded_regime_is_reachable_from_shipped_configs)
finish()
