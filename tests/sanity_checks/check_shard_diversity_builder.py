"""`ucb_diversity` must trade shard WIDTH-to-UAV mapping, and nothing else.

It is the third roster-construction control, alongside `hfl_balanced_roster`
(`ucb_balanced`). `_class_coverage_assign` (proposed `ucb`) walks UAV by UAV
filling each to capacity with a submodular class-coverage-maximizing greedy;
`ucb_diversity` uses the IDENTICAL selection (it delegates straight to
`_class_coverage_assign` to decide who participates) but reassigns each
selected client to whichever feasible, non-full UAV gains the most class
entropy from that client's own histogram, instead of accepting the proposed
walk order.

Without a check that the reassignment actually MOVES the entropy needle on a
fixture where an improving reassignment exists, this arm could silently be a
no-op that just calls `_class_coverage_assign` and returns its output
unchanged — a real risk given how much of the code is shared by design. That
is assertion 3 below; it is the one that matters.

Per the project's own hard-won lesson (2026-08 capacity-floor / roster
confound investigations), the synthetic fixture collapses to majority-class
degeneracy far too easily to trust downstream macro-F1 as a signal for
anything selection- or roster-related. So every assertion here is made
directly on the ASSIGNMENT STRUCTURE — who went where, and the resulting
per-UAV class histograms — never on a trained model's accuracy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np  # noqa: E402
from _lib import check, finish  # noqa: E402

from uavbench.fl.client_selection import ClientSelector  # noqa: E402
from uavbench.fl.device_state import DeviceState  # noqa: E402
from uavbench.fl.federated import _METHOD_CFG  # noqa: E402


def _world(n_clients, n_uav, seed=0):
    """Clients and UAVs colocated tightly enough that every UAV covers everyone
    (mirrors check_roster_control.py's fixture) — reachability is not the
    thing under test here, so it is kept generous and out of the way."""
    rng = np.random.default_rng(seed)
    ids = list(range(n_clients))
    coords = {c: (37.5 + rng.normal(0, 0.002), 137.3 + rng.normal(0, 0.002)) for c in ids}
    uavs = [(37.5 + rng.normal(0, 0.001), 137.3 + rng.normal(0, 0.001)) for _ in range(n_uav)]
    return ids, coords, uavs


def _run(mode, ids, coords, uavs, cap, rnd=3, class_counts=None, class_scarcity=None,
         seed=0):
    sel = ClientSelector(ids, seed=seed)
    return sel, sel.select(
        class_counts=class_counts,
        class_scarcity=class_scarcity,
        covered={c: 0 for c in ids},
        device_states={
            c: DeviceState(battery=1.0, snr_db=30.0, memory_ok=True, compute_time_s=1.0)
            for c in ids
        },
        reputation_scores={c: 0.5 for c in ids},
        client_coords=coords,
        uav_coords_latlon=uavs,
        round_num=rnd,
        uav_capacity=cap,
        mode=mode,
        rng=np.random.default_rng(0),
        R_comm=20000.0,
    )


def _single_class_counts(ids, labels, n_cls=4):
    """Every client's histogram is entirely one class — a fixture in which a
    diversity-improving reassignment provably EXISTS: any roster builder that
    ignores class identity when choosing a UAV will, with high probability,
    dump same-labelled clients on the same UAV somewhere, while a builder
    that actively spreads classes across UAVs can always do at least as well
    and generically better. ``labels`` assigns each client's single class.
    """
    counts = {}
    for cid, lab in zip(ids, labels, strict=True):
        h = np.zeros(n_cls)
        h[lab] = 40.0
        counts[cid] = h
    return counts


def _shard_entropy(selected, class_counts, n_cls=4):
    """Mean per-UAV normalised class entropy of the shards `selected` implies —
    the same quantity `_shard_class_diversity` records in the pipeline, computed
    here directly from the per-client histograms `_ucb_diversity_assign` itself
    consumes (no dependency on the dataset/labels plumbing the pipeline uses)."""
    by_uav: dict[int, np.ndarray] = {}
    for cid, uav in selected.items():
        by_uav.setdefault(uav, np.zeros(n_cls))
        by_uav[uav] = by_uav[uav] + class_counts[cid]
    entropies = []
    log_c = np.log(n_cls)
    for hist in by_uav.values():
        total = hist.sum()
        if total <= 0:
            continue
        p = hist / total
        nz = p[p > 0]
        entropies.append(float(-(nz * np.log(nz)).sum() / log_c))
    return float(np.mean(entropies)) if entropies else float("nan")


# ---------------------------------------------------------------------------


def the_method_is_registered_and_differs_only_in_selection():
    """hfl_diverse_roster must share placement, reputation and cadence with
    proposed_hfl, differing only in the selection/roster mode — the same
    invariant check_roster_control.py pins for hfl_balanced_roster."""
    a = _METHOD_CFG["proposed_hfl"]
    b = _METHOD_CFG["hfl_diverse_roster"]
    assert a[0] == b[0], f"placement differs: {a[0]} vs {b[0]}"
    assert a[2] == b[2], f"reputation weighting differs: {a[2]} vs {b[2]}"
    assert a[3] == b[3], f"repositioning cadence differs: {a[3]} vs {b[3]}"
    assert b[1] == "ucb_diversity", f"selection mode is {b[1]}"


def respects_capacity_and_reachability_like_the_other_builders():
    """No UAV may exceed uav_capacity, and every assigned UAV index must be one
    of the UAVs actually passed in (i.e. within the fixture's coverage — the
    reachability constraint the other builders enforce via R_comm)."""
    ids, coords, uavs = _world(40, 8)
    scarcity = np.array([1.0, 4.0, 3.0, 2.0])
    rng = np.random.default_rng(1)
    counts = {c: rng.multinomial(40, [0.82, 0.03, 0.06, 0.09]).astype(float) for c in ids}
    cap = 5
    _, selected = _run("ucb_diversity", ids, coords, uavs, cap,
                        class_counts=counts, class_scarcity=scarcity)

    fill: dict[int, int] = {}
    for cid, uav in selected.items():
        fill[uav] = fill.get(uav, 0) + 1
        assert 0 <= uav < len(uavs), f"client {cid} assigned to out-of-range UAV {uav}"
    assert all(n <= cap for n in fill.values()), f"a UAV exceeded capacity {cap}: {fill}"


def every_selected_client_gets_exactly_one_uav_and_nobody_else_does():
    """The output must be a clean function: selected clients each map to
    exactly one UAV (trivially true of a dict, but assert client identities
    match, not just counts), and no client outside the returned selection
    appears at all."""
    ids, coords, uavs = _world(30, 6)
    scarcity = np.array([1.0, 4.0, 3.0, 2.0])
    rng = np.random.default_rng(2)
    counts = {c: rng.multinomial(40, [0.7, 0.1, 0.1, 0.1]).astype(float) for c in ids}
    cap = 5
    sel, selected = _run("ucb_diversity", ids, coords, uavs, cap,
                          class_counts=counts, class_scarcity=scarcity)

    assert len(selected) == len(set(selected.keys())), "a client id appeared twice"
    assert set(selected.keys()).issubset(set(ids)), "an unrecognised client id was assigned"
    # Every value is a single, valid UAV index (not e.g. a list or None).
    for uav in selected.values():
        assert isinstance(uav, (int, np.integer)) and 0 <= uav < len(uavs)


def diversity_reassignment_raises_mean_shard_entropy_on_a_collapsible_fixture():
    """THE decisive check: on a fixture where a diversity-improving
    reassignment exists, `ucb_diversity` must produce a STRICTLY higher mean
    shard class entropy than the proposed system's own fill-to-capacity
    walk (`_class_coverage_assign`) — otherwise this arm is a silent no-op.

    Fixture: every client is single-class (see `_single_class_counts`), 4
    UAVs each with capacity large enough to hold a full mix. A builder that
    is blind to WHICH UAV concentrates WHICH class (the proposed walk order,
    which is driven by static/Gumbel priority and per-UAV coverage value,
    not by balancing classes across UAVs) will generically produce shards
    that are far from uniform; a builder that greedily maximises each UAV's
    own running entropy when placing each client can always do at least as
    well as ignoring class on placement, and strictly better whenever the
    naive order would have clustered same-class clients together — which,
    with clients spread round-robin across 4 classes below, it does.
    """
    n_cls = 4
    ids, coords, uavs = _world(40, 4)
    # Clients cycle through classes 0,1,2,3,0,1,2,... so any assignment that
    # is not class-aware has ample opportunity to cluster same-class
    # clients on the same UAV; a class-aware placer does not have to.
    labels = [cid % n_cls for cid in ids]
    counts = _single_class_counts(ids, labels, n_cls)
    scarcity = np.ones(n_cls)  # uniform scarcity: isolates placement, not the
    # coverage-value weighting the proposed selection step also uses.
    cap = 10  # 4 UAVs x 10 = 40 slots == 40 clients: slots bind, so both
    # builders select the SAME full roster and only the UAV-mapping differs.

    _, base = _run("ucb", ids, coords, uavs, cap,
                    class_counts=counts, class_scarcity=scarcity)
    _, div = _run("ucb_diversity", ids, coords, uavs, cap,
                   class_counts=counts, class_scarcity=scarcity)

    assert set(base.keys()) == set(div.keys()) == set(ids), (
        "the two builders selected different client sets — ucb_diversity must "
        "not change WHO is selected, only WHICH UAV they go to"
    )

    ent_base = _shard_entropy(base, counts, n_cls)
    ent_div = _shard_entropy(div, counts, n_cls)
    print(f"      mean shard entropy: _class_coverage_assign={ent_base:.4f}  "
          f"ucb_diversity={ent_div:.4f}  (higher is more diverse)")
    assert ent_div > ent_base, (
        f"ucb_diversity's mean shard entropy ({ent_div:.4f}) is not strictly "
        f"greater than the proposed walk's ({ent_base:.4f}) on a fixture built "
        "so an improving reassignment exists — the diversity arm is a no-op"
    )


def deterministic_given_the_same_inputs():
    """Same inputs, same RNG seed, twice over: identical assignment."""
    ids, coords, uavs = _world(35, 7)
    scarcity = np.array([1.0, 4.0, 3.0, 2.0])
    rng = np.random.default_rng(3)
    counts = {c: rng.multinomial(40, [0.75, 0.1, 0.1, 0.05]).astype(float) for c in ids}
    cap = 6

    _, first = _run("ucb_diversity", ids, coords, uavs, cap,
                     class_counts=counts, class_scarcity=scarcity, seed=42)
    _, second = _run("ucb_diversity", ids, coords, uavs, cap,
                      class_counts=counts, class_scarcity=scarcity, seed=42)
    assert first == second, "ucb_diversity is not deterministic given identical inputs"


def falls_back_to_class_coverage_assign_without_class_histograms():
    """No class information to diversify against => nothing for this builder
    to do differently; it must hand back exactly what `_class_coverage_assign`
    (== plain priority greedy, per that function's own fallback) produced."""
    ids, coords, uavs = _world(25, 20)
    sel = ClientSelector(ids)
    static = np.linspace(1.0, 0.0, len(ids))
    eligible = {c: 0 for c in ids}
    direct = sel._class_coverage_assign(
        ids, eligible, static, None, None, 6, coords, uavs, R_comm=20000.0,
    )
    via_diversity = sel._ucb_diversity_assign(
        ids, eligible, static, None, None, 6, coords, uavs, R_comm=20000.0,
    )
    assert via_diversity == direct, (
        "without class histograms ucb_diversity must be identical to "
        "_class_coverage_assign's own (greedy) fallback"
    )


check("the method is registered and differs only in selection",
      the_method_is_registered_and_differs_only_in_selection)
check("respects capacity and reachability like the other builders",
      respects_capacity_and_reachability_like_the_other_builders)
check("every selected client gets exactly one UAV and nobody else does",
      every_selected_client_gets_exactly_one_uav_and_nobody_else_does)
check("diversity reassignment raises mean shard entropy on a collapsible fixture",
      diversity_reassignment_raises_mean_shard_entropy_on_a_collapsible_fixture)
check("deterministic given the same inputs", deterministic_given_the_same_inputs)
check("falls back to _class_coverage_assign without class histograms",
      falls_back_to_class_coverage_assign_without_class_histograms)
finish()
