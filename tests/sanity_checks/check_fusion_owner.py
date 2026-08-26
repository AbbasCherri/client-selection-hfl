"""`fl.fusion_owner` — the knob the fusion-ownership arm turns.

With `uav` the UAV tier owns img_proj + fusion and trains both on the pooled
shard of its assigned clients. With `client` the UAV keeps only img_proj and
fusion moves to the clients, where it is globally averaged. The arm in
configs/fusion_owner.yaml exists to test whether UAV-owned fusion is what makes
every hierarchical method lose to flat_fl at a coherent radius.

Guarded exactly, NOT by comparing macro-F1 curves. The first version of this
check did compare curves and failed: on the synthetic fixture every setting
collapses to the same degenerate score, so an outcome-based guard here passes
vacuously whichever way the knob is wired. `_block_ownership` is a pure
function for that reason, and the end-to-end test captures the block tuples
`_train_blocks` is actually called with instead of what the model learns.

Three silent failures are covered:

  * An unrecognised value. The ownership branch tests `== "uav"`, so ANY other
    string — a typo, a stray "server" — used to select the client branch and
    produce a clean, wrong table.
  * The knob not reaching the training call at all, which would make the arm
    score as a flat null while testing nothing.
  * flat_fl moving. The C2 contrast reuses flat_fl from results/paper_full
    unrecomputed, which is only valid because flat_fl has no UAV tier.
"""

import copy
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _fixture import build_synthetic_raw  # noqa: E402
from _lib import check, finish  # noqa: E402

from uavbench.fl import federated  # noqa: E402
from uavbench.fl.federated import _block_ownership, run_full_hfl  # noqa: E402

PREBUILT = build_synthetic_raw(N=12, K=2, seed=42)
ALL_BLOCKS = {"img_proj", "fusion", "struct_branch"}


def _cfg(results_dir, methods, fusion_owner=None):
    fl = {
        "n_rounds": 1, "n_local_epochs": 1, "n_uav_epochs": 1,
        "lr": 0.01, "uav_lr": 0.01, "batch_size": 4, "K": 2,
        "R_comm": 200_000.0, "capacity": 10, "T_sel": 1,
        "lambda_min": 0.0, "target_accuracy": 0.99, "seed": 42,
    }
    if fusion_owner is not None:
        fl["fusion_owner"] = fusion_owner
    return {
        "results_dir": results_dir, "methods": methods, "fl": fl,
        "budget": {"P": 5, "G_max": 3},
        "data": {"source": "prebuilt", "prebuilt": copy.deepcopy(PREBUILT), "seed": 42},
        "optimizer_seed": 42,
    }


def _run(methods, fusion_owner=None):
    with tempfile.TemporaryDirectory() as d:
        return run_full_hfl(_cfg(d, methods, fusion_owner))["rounds"]


def _captured_blocks(methods, fusion_owner):
    """Every `blocks` tuple _train_blocks is actually invoked with."""
    seen: list[tuple[str, ...]] = []
    original = federated._train_blocks

    def spy(*args, **kwargs):
        blocks = kwargs["blocks"] if "blocks" in kwargs else args[5]
        seen.append(tuple(blocks))
        return original(*args, **kwargs)

    federated._train_blocks = spy
    try:
        _run(methods, fusion_owner)
    finally:
        federated._train_blocks = original
    return seen


def an_unrecognised_value_is_rejected():
    for bad in ("server", "uav ", "Client!", "none", "both"):
        try:
            _run(["proposed_hfl"], bad)
        except ValueError as e:
            assert "fusion_owner" in str(e), f"wrong error for {bad!r}: {e}"
        else:
            raise AssertionError(f"fusion_owner={bad!r} was accepted silently")


def both_documented_values_are_accepted():
    for good in ("uav", "client", "client_full", "UAV", "Client", "CLIENT_FULL"):
        _run(["proposed_hfl"], good)


def ownership_is_exact_for_every_case():
    assert _block_ownership("uav", True) == (("img_proj", "fusion"), ("struct_branch",))
    assert _block_ownership("client", True) == (("img_proj",), ("struct_branch", "fusion"))
    # client_full: the UAV trains NOTHING and is a pure aggregator.
    assert _block_ownership("client_full", True) == (
        (), ("struct_branch", "img_proj", "fusion"))
    # No UAV tier: fusion_owner is irrelevant and img_proj is owned by nobody.
    for owner in ("uav", "client"):
        assert _block_ownership(owner, False) == ((), ("struct_branch", "fusion"))


def every_block_is_owned_exactly_once():
    # A block owned by both tiers would be trained twice per round; a block
    # owned by neither silently stays at initialisation. Only img_proj may be
    # unowned, and only when there is no UAV tier.
    for owner in ("uav", "client", "client_full"):
        uav, client = _block_ownership(owner, True)
        assert not set(uav) & set(client), f"{owner}: block owned by both tiers"
        assert set(uav) | set(client) == ALL_BLOCKS, f"{owner}: a block is unowned"
    uav, client = _block_ownership("uav", False)
    assert ALL_BLOCKS - (set(uav) | set(client)) == {"img_proj"}


def the_knob_reaches_the_training_call():
    uav_seen = _captured_blocks(["proposed_hfl"], "uav")
    client_seen = _captured_blocks(["proposed_hfl"], "client")
    assert ("img_proj", "fusion") in uav_seen, (
        "under fusion_owner=uav the UAV tier never trained img_proj+fusion — "
        "the knob is not reaching _train_blocks"
    )
    assert ("img_proj",) in client_seen and ("img_proj", "fusion") not in client_seen, (
        "under fusion_owner=client the UAV tier still trained fusion"
    )
    assert ("struct_branch", "fusion") in client_seen, (
        "under fusion_owner=client the clients never trained fusion"
    )


def flat_fl_never_trains_img_proj():
    # The C2 contrast pairs the new arm against flat_fl taken from
    # results/paper_full WITHOUT recomputing it. Valid only if flat_fl ignores
    # fusion_owner — it has no UAV tier, so img_proj stays at initialisation.
    for owner in ("uav", "client"):
        seen = _captured_blocks(["flat_fl"], owner)
        assert seen, "flat_fl trained nothing at all"
        assert all("img_proj" not in b for b in seen), (
            f"flat_fl trained img_proj under fusion_owner={owner}"
        )
        assert ("struct_branch", "fusion") in seen, "flat_fl did not train fusion"




def client_full_makes_the_uav_train_nothing():
    # The whole point of the mode. If the UAV still trains anything, it is not
    # a pure aggregator and the comparison against flat_fl is not clean.
    seen = _captured_blocks(["proposed_hfl"], "client_full")
    assert seen, "nothing trained at all"
    assert all(tuple(b) == ("struct_branch", "img_proj", "fusion") for b in seen), (
        f"under client_full some tier trained a partial block set: {set(seen)}"
    )


def client_full_trains_strictly_more_than_flat_fl():
    # flat_fl leaves img_proj at init. client_full must train it, otherwise it
    # cannot dominate flat_fl and the hypothesis is untestable.
    full = _captured_blocks(["proposed_hfl"], "client_full")
    flat = _captured_blocks(["flat_fl"], "client_full")
    assert any("img_proj" in b for b in full), "client_full never trained img_proj"
    assert all("img_proj" not in b for b in flat), "flat_fl trained img_proj"

check("an unrecognised value is rejected", an_unrecognised_value_is_rejected)
check("both documented values are accepted", both_documented_values_are_accepted)
check("ownership is exact for every case", ownership_is_exact_for_every_case)
check("every block is owned exactly once", every_block_is_owned_exactly_once)
check("the knob reaches the training call", the_knob_reaches_the_training_call)
check("flat_fl never trains img_proj", flat_fl_never_trains_img_proj)
check("client_full: UAV trains nothing", client_full_makes_the_uav_train_nothing)
check("client_full trains more than flat_fl", client_full_trains_strictly_more_than_flat_fl)
finish()
