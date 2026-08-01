"""Oracle-degradation ladder: class-histogram sources and the class_greedy arm.

Guards the 2026-08 answer to "the selector depends on a per-client class
distribution it could never observe". The ladder replaces the ground-truth
histogram with progressively more realistic sources, and `class_greedy` gives a
bare baseline the *same* oracle so we can tell "class-awareness helps" apart
from "our UCB pipeline helps".

What must hold:
  1. every ARM_SPECS entry resolves to a real selector mode;
  2. `true` reproduces the historical np.bincount exactly;
  3. `dp` is unbiased-ish and *actually noisy* (a no-op DP rung would silently
     turn the privacy claim into a lie), and tighter epsilon means more noise;
  4. `none` yields no class information, and `class_greedy` refuses to run
     without it rather than silently degrading to a priority greedy;
  5. scarcity is derived from the same source as the counts — pairing a
     disclosure-free histogram with an oracle scarcity vector would smuggle
     the oracle back in.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from _lib import check, finish  # noqa: E402

from uavbench.fl.class_histograms import (  # noqa: E402
    N_CLASSES,
    VALID_SOURCES,
    build_class_info,
    dp_histograms,
    scarcity_from_counts,
    true_histograms,
)
from uavbench.fl.client_selection import ClientSelector  # noqa: E402
from uavbench.fl.device_state import DeviceState  # noqa: E402
from uavbench.fl.selection_isolation import ARM_SPECS, resolve_arm  # noqa: E402

LABELS = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 3, 3], dtype=torch.long)
CLIENTS = {0: [0, 1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]}


def every_arm_resolves_to_a_real_mode():
    known = {"ucb", "ucb_noclass", "class_greedy", "random", "all",
             "fedcs", "rep_cap", "fair_mab", "oort", "power_of_choice"}
    for arm in ARM_SPECS:
        mode, source, eps = resolve_arm(arm)
        assert mode in known, f"arm {arm!r} resolves to unknown mode {mode!r}"
        assert source in VALID_SOURCES, f"arm {arm!r} has bad source {source!r}"
        assert eps > 0, f"arm {arm!r} has non-positive epsilon {eps}"


def unknown_arms_pass_through_class_blind():
    """Literature baselines ignore class info by design."""
    mode, source, _ = resolve_arm("oort")
    assert (mode, source) == ("oort", "none")


def true_matches_bincount():
    got = true_histograms(LABELS, CLIENTS)
    for cid, idx in CLIENTS.items():
        want = np.bincount(LABELS[idx].numpy(), minlength=N_CLASSES).astype(float)
        assert np.array_equal(got[cid], want), f"client {cid}: {got[cid]} != {want}"


def dp_is_actually_noisy_and_epsilon_ordered():
    truth = true_histograms(LABELS, CLIENTS)
    rng = np.random.default_rng(0)
    tight = dp_histograms(truth, epsilon=0.5, rng=rng)
    loose = dp_histograms(truth, epsilon=8.0, rng=rng)

    def total_dev(noisy):
        return sum(float(np.abs(noisy[c] - truth[c]).sum()) for c in truth)

    assert total_dev(tight) > 0, "DP rung produced zero noise — privacy claim would be false"
    # Averaged over many draws, tighter epsilon must perturb more.
    dev_t = np.mean([
        total_dev(dp_histograms(truth, 0.5, np.random.default_rng(s))) for s in range(60)
    ])
    dev_l = np.mean([
        total_dev(dp_histograms(truth, 8.0, np.random.default_rng(s))) for s in range(60)
    ])
    assert dev_t > dev_l, f"eps=0.5 noise {dev_t:.2f} should exceed eps=8 noise {dev_l:.2f}"
    assert all((v >= 0).all() for v in tight.values()), "DP counts must be clamped at zero"


def dp_is_reproducible_under_a_fixed_seed():
    truth = true_histograms(LABELS, CLIENTS)
    a = dp_histograms(truth, 1.0, np.random.default_rng(7))
    b = dp_histograms(truth, 1.0, np.random.default_rng(7))
    for cid in truth:
        assert np.array_equal(a[cid], b[cid]), "DP noise is not seed-reproducible"


def none_source_yields_no_class_information():
    counts, scarcity = build_class_info("none", labels=LABELS, client_indices=CLIENTS)
    assert counts is None and scarcity is None


def scarcity_favours_rare_classes_and_is_normalised():
    counts = true_histograms(LABELS, CLIENTS)
    s = scarcity_from_counts(counts)
    assert abs(s.sum() - 1.0) < 1e-9, "scarcity must be a normalised weight vector"
    # class 2 appears once, class 3 four times -> class 2 must weigh more.
    assert s[2] > s[3], f"rarer class should carry more weight: {s}"


def _eligible_states(ids):
    """Device states that pass the four-condition gate.

    Non-negotiable for these two checks: `select` returns {} on an empty
    eligible pool *before* reaching the mode dispatch, so passing
    device_states={} would make both assertions vacuously pass.
    """
    return {
        cid: DeviceState(battery=0.9, snr_db=20.0, memory_ok=True, compute_time_s=60.0)
        for cid in ids
    }


def _select(sel, ids, **kw):
    return sel.select(
        covered=dict.fromkeys(ids, 0),
        device_states=_eligible_states(ids),
        reputation_scores=dict.fromkeys(ids, 0.5),
        client_coords={cid: (37.5, 137.3) for cid in ids},
        uav_coords_latlon=[(37.5, 137.3)],
        round_num=1,
        uav_capacity=len(ids),
        **kw,
    )


def gate_lets_these_clients_through():
    """Guard the guard: if this stops selecting anyone, the two checks below
    go vacuous and would pass no matter what the dispatch does."""
    sel = ClientSelector([0, 1], seed=0)
    got = _select(sel, [0, 1], mode="random", rng=np.random.default_rng(0))
    assert got, "fixture device states are not eligible — later checks would be vacuous"


def class_greedy_refuses_to_run_blind():
    """Without a histogram it is not a class-aware baseline — must fail loudly."""
    sel = ClientSelector([0, 1], seed=0)
    try:
        _select(sel, [0, 1], mode="class_greedy", class_counts=None, class_scarcity=None)
    except ValueError:
        return
    raise AssertionError("class_greedy accepted a call with no class information")


def unknown_mode_still_raises():
    sel = ClientSelector([0, 1], seed=0)
    try:
        _select(sel, [0, 1], mode="not_a_mode")
    except ValueError:
        return
    raise AssertionError("unknown selection mode was silently accepted")


if __name__ == "__main__":
    check("every ARM_SPECS entry resolves to a real mode", every_arm_resolves_to_a_real_mode)
    check("unknown arms pass through class-blind", unknown_arms_pass_through_class_blind)
    check("true source matches np.bincount", true_matches_bincount)
    check("dp noise is real and epsilon-ordered", dp_is_actually_noisy_and_epsilon_ordered)
    check("dp noise is seed-reproducible", dp_is_reproducible_under_a_fixed_seed)
    check("none source yields no class info", none_source_yields_no_class_information)
    check("scarcity favours rare classes", scarcity_favours_rare_classes_and_is_normalised)
    check("fixture states pass the eligibility gate", gate_lets_these_clients_through)
    check("class_greedy refuses to run blind", class_greedy_refuses_to_run_blind)
    check("unknown mode still raises", unknown_mode_still_raises)
    finish()
