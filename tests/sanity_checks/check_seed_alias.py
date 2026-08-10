"""`fl.seed_method_alias` — the switch that makes an isolating ablation isolate.

Every arm normally draws its own RNG stream, folded from a hash of its method
name. That is right for independent arms and WRONG for the roster-construction
control: `hfl_balanced_roster` differs from `proposed_hfl` only in how the roster
is built, so under separate streams the two would also differ in placement,
device values and every UCB tie-break, and the measured gap would mix the
mechanism with ordinary seed noise.

Guarded here because the failure is silent in both directions. If the alias is
not read (a typo in the key, say), the ablation quietly stops being paired and
still produces a clean-looking table. If it were applied by default, every arm
in every sweep would share one stream and the independence the seed design
exists for would be gone.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _lib import check, finish  # noqa: E402

from uavbench.fl.federated import _seed_method_for  # noqa: E402
from uavbench.fl.seeds import fullsim_method_seed, method_hash  # noqa: E402

RUN_SEED = 123456


def the_two_arms_would_otherwise_diverge():
    # The reason the switch exists. If this ever becomes false the alias is
    # unnecessary and this whole mechanism should be deleted.
    a = fullsim_method_seed(RUN_SEED, "proposed_hfl")
    b = fullsim_method_seed(RUN_SEED, "hfl_balanced_roster")
    assert a != b, "method hash no longer separates the arms — alias is pointless"


def absent_alias_uses_the_methods_own_name():
    for fl in ({}, {"seed_method_alias": None}, {"seed_method_alias": ""}):
        assert _seed_method_for(fl, "hfl_balanced_roster") == "hfl_balanced_roster", fl


def an_alias_substitutes_the_stream():
    fl = {"seed_method_alias": "proposed_hfl"}
    assert _seed_method_for(fl, "hfl_balanced_roster") == "proposed_hfl"
    assert (
        fullsim_method_seed(RUN_SEED, _seed_method_for(fl, "hfl_balanced_roster"))
        == fullsim_method_seed(RUN_SEED, "proposed_hfl")
    ), "aliased arm does not land on the aliased method's stream"


def aliasing_to_itself_is_a_no_op():
    fl = {"seed_method_alias": "proposed_hfl"}
    assert _seed_method_for(fl, "proposed_hfl") == "proposed_hfl"


def the_alias_does_not_leak_to_other_methods():
    # It is read per run from that run's own fl block, so a config that aliases
    # one arm must not move any other arm's stream.
    plain = {}
    assert _seed_method_for(plain, "fedcs") == "fedcs"
    assert _seed_method_for(plain, "oort") == "oort"


def default_behaviour_is_unchanged_for_every_shipped_method():
    # Bit-identical guarantee: without the key, every method resolves exactly as
    # it did before the option existed.
    for m in ("proposed_hfl", "fedcs", "oort", "power_of_choice", "rep_cap",
              "fair_mab", "flat_fl", "hfl_static", "mclp_place", "moon2022"):
        assert _seed_method_for({}, m) == m
        assert fullsim_method_seed(RUN_SEED, m) == (
            RUN_SEED ^ method_hash(m, 16)
        ) % (2**31), f"seed derivation changed for {m}"


check("the two arms would otherwise diverge", the_two_arms_would_otherwise_diverge)
check("no alias -> the method's own name", absent_alias_uses_the_methods_own_name)
check("an alias substitutes the stream", an_alias_substitutes_the_stream)
check("aliasing to itself is a no-op", aliasing_to_itself_is_a_no_op)
check("the alias does not leak to other methods", the_alias_does_not_leak_to_other_methods)
check("default behaviour unchanged for every method", default_behaviour_is_unchanged_for_every_shipped_method)
finish()
