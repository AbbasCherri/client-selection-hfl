"""`ucb_balanced` must differ from `ucb` in roster construction and nothing else.

It is the control for the roster-construction confound: the proposed selector
fills each UAV to capacity while the five literature selectors send each client
to the least-loaded feasible UAV, so a proposed-vs-baseline gap in the
slack-slot regime may be a difference in shard width — which decides whether the
edge fusion heads learn — rather than in selection quality.

The control is only informative if it changes exactly one thing. These checks
pin that: same scores in, different rosters out, and identical behaviour once
the slot budget binds (where both policies saturate and the confound vanishes).
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
    rng = np.random.default_rng(seed)
    ids = list(range(n_clients))
    coords = {c: (37.5 + rng.normal(0, 0.002), 137.3 + rng.normal(0, 0.002)) for c in ids}
    uavs = [(37.5 + rng.normal(0, 0.001), 137.3 + rng.normal(0, 0.001)) for _ in range(n_uav)]
    return ids, coords, uavs


def _widths(sel):
    fill = {}
    for u in sel.values():
        fill[u] = fill.get(u, 0) + 1
    return sorted(fill.values(), reverse=True)


def _run(mode, ids, coords, uavs, cap, rnd=3, with_classes=True):
    """Every client covered and comfortably eligible, so only the roster differs.

    `with_classes` matters more than it looks: `_class_coverage_assign` falls
    back to `_greedy_assign` when no class histogram is supplied, so `ucb` and
    `ucb_balanced` are the SAME code path without one. The confound this control
    isolates therefore exists only when class-aware selection is actually on —
    which is the paper's configuration (`class_source: "true"`), and is what the
    default here reproduces.
    """
    rng = np.random.default_rng(1)
    counts = {c: rng.multinomial(40, [0.82, 0.03, 0.06, 0.09]).astype(float) for c in ids}
    scarcity = np.array([1.0, 4.0, 3.0, 2.0])
    sel = ClientSelector(ids)
    return sel.select(
        class_counts=counts if with_classes else None,
        class_scarcity=scarcity if with_classes else None,
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


def the_method_is_registered_and_differs_only_in_selection():
    """proposed_hfl and the control must share placement, reputation and cadence."""
    a = _METHOD_CFG["proposed_hfl"]
    b = _METHOD_CFG["hfl_balanced_roster"]
    assert a[0] == b[0], f"placement differs: {a[0]} vs {b[0]}"
    assert a[2] == b[2], f"reputation weighting differs: {a[2]} vs {b[2]}"
    assert a[3] == b[3], f"repositioning cadence differs: {a[3]} vs {b[3]}"
    assert b[1] == "ucb_balanced", f"selection mode is {b[1]}"
    assert a[1] == "ucb", f"proposed_hfl selection mode changed to {a[1]}"


def without_class_histograms_the_two_modes_are_one_code_path():
    """Documents the scope of the confound, so it is not overstated.

    `_class_coverage_assign` delegates to `_greedy_assign` when class_counts is
    None, so with class-aware selection off there is nothing to isolate and the
    control is expected to be identical to the proposed method.
    """
    ids, coords, uavs = _world(25, 20)
    a = _widths(_run("ucb", ids, coords, uavs, cap=6, with_classes=False))
    b = _widths(_run("ucb_balanced", ids, coords, uavs, cap=6, with_classes=False))
    assert a == b, f"expected identical rosters without class info: {a} vs {b}"


def rosters_differ_when_slots_are_slack():
    """The regime the confound lives in: few clients, many UAVs, classes on."""
    ids, coords, uavs = _world(25, 20)
    ucb = _widths(_run("ucb", ids, coords, uavs, cap=6))
    bal = _widths(_run("ucb_balanced", ids, coords, uavs, cap=6))
    assert max(bal) < max(ucb), (
        f"load-balanced widest shard {max(bal)} is not below fill-to-capacity's "
        f"{max(ucb)} — the control is not changing roster construction"
    )
    assert len(bal) > len(ucb), (
        f"load-balancing should spread over more UAVs: {len(bal)} vs {len(ucb)}"
    )


def both_saturate_when_slots_bind():
    """Where K*capacity < covered, the two policies must agree — that is why the
    N=200 comparisons are unaffected by the confound."""
    ids, coords, uavs = _world(200, 10)
    ucb = _widths(_run("ucb", ids, coords, uavs, cap=6))
    bal = _widths(_run("ucb_balanced", ids, coords, uavs, cap=6))
    assert ucb == bal == [6] * 10, (
        f"with slots binding both should fill every UAV to 6: ucb={ucb}, balanced={bal}"
    )


def the_same_clients_are_eligible_in_both():
    """Only the assignment may differ; the scored pool must not."""
    ids, coords, uavs = _world(25, 20)
    a = set(_run("ucb", ids, coords, uavs, cap=6))
    b = set(_run("ucb_balanced", ids, coords, uavs, cap=6))
    assert a == b, (
        f"different client sets selected ({len(a)} vs {len(b)}) — the control is "
        "changing who trains, not just which aircraft aggregates them"
    )


def an_unknown_mode_still_raises():
    ids, coords, uavs = _world(10, 3)
    try:
        _run("ucb_balancd", ids, coords, uavs, cap=6)
    except ValueError as e:
        assert "unknown selection mode" in str(e)
        return
    raise AssertionError("a misspelled selection mode was accepted")


check("the method is registered and differs only in selection",
      the_method_is_registered_and_differs_only_in_selection)
check("without class histograms the two modes are one code path",
      without_class_histograms_the_two_modes_are_one_code_path)
check("rosters differ when slots are slack", rosters_differ_when_slots_are_slack)
check("both saturate when slots bind", both_saturate_when_slots_bind)
check("the same clients are eligible in both", the_same_clients_are_eligible_in_both)
check("an unknown mode still raises", an_unknown_mode_still_raises)
finish()
