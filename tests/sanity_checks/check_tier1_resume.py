"""Tier-1 checkpoints must not survive a change to what they computed.

The failure this guards against is recorded rather than hypothetical. On
2026-08-10 the Tier-1 rebuild — whose whole purpose was to rescore placement
through the Al-Hourani channel on a corrected 100-400 m band — printed

    Resuming: 840/840 jobs already checkpointed, running the remaining 0

and finished in eleven seconds, re-serving results computed under the old flat
range gate and the old 20-120 m band. The checkpoint key was
(method, scenario, seed) and knew nothing about the physics. The altitude gate
caught it, but only because that change happened to have a gate; a fitness-weight
change would have produced a clean, wrong, plausible table.

So: the signature must move when the physics moves, and must NOT move for
cosmetic edits, or every rerun recomputes everything and resume becomes useless.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from copy import deepcopy  # noqa: E402

from _lib import check, finish  # noqa: E402

from uavbench.runner import _checkpoint_path, _config_signature  # noqa: E402

BASE = {
    "name": "tier1_core",
    "results_dir": "results/tier1_core",
    "n_workers": 8,
    "n_seeds": 30,
    "instance_seed": 1234,
    "optimizer_seed": 9876,
    "area": {"x": [0.0, 5000.0], "y": [0.0, 5000.0], "z": [100.0, 400.0]},
    "problem": {"R_comm": 500.0, "link_model": "path_loss", "capacity": 15},
    "fitness": {"w1": 0.811, "w2": 0.03, "w3": 0.159},
    "budget": {"P": 100, "G_max": 200},
    "value": {"beta_mode": "scheduled"},
    "scenarios": [{"distribution": "clustered", "N": 250, "K": 10}],
    "optimizer_params": {"pso": {"c1": 2.05}},
    "methods": ["pso", "ga"],
}


def _sig(**overrides):
    cfg = deepcopy(BASE)
    for path, val in overrides.items():
        keys = path.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = val
    return _config_signature(cfg)


def identical_configs_give_the_same_signature():
    assert _sig() == _sig(), "signature is not stable across calls"
    assert _config_signature(deepcopy(BASE)) == _config_signature(deepcopy(BASE))


def the_altitude_band_changes_the_signature():
    # The exact 2026-08-10 failure: 20-120 m results reused for a 100-400 m run.
    assert _sig(**{"area.z": [20.0, 120.0]}) != _sig()


def the_link_model_changes_the_signature():
    assert _sig(**{"problem.link_model": "range_gate"}) != _sig()


def the_radius_changes_the_signature():
    assert _sig(**{"problem.R_comm": 2000.0}) != _sig()


def the_fitness_weights_change_the_signature():
    # The change that would NOT have been caught by the altitude gate.
    assert _sig(**{"fitness.w1": 1.0}) != _sig()


def the_budget_changes_the_signature():
    assert _sig(**{"budget.G_max": 400}) != _sig()


def the_scenarios_change_the_signature():
    assert _sig(**{"scenarios": [{"distribution": "uniform", "N": 250, "K": 10}]}) != _sig()


def the_seeds_change_the_signature():
    assert _sig(**{"instance_seed": 4321}) != _sig()
    assert _sig(**{"optimizer_seed": 1111}) != _sig()


def optimizer_params_change_the_signature():
    assert _sig(**{"optimizer_params": {"pso": {"c1": 1.5}}}) != _sig()


def cosmetic_edits_do_not_change_the_signature():
    # If these invalidated, resume would be worthless: a worker-count tweak or a
    # rename would force a full recompute of a multi-hour grid.
    for path, val in (("n_workers", 12), ("name", "tier1_core_rerun"),
                      ("results_dir", "results/elsewhere")):
        assert _sig(**{path: val}) == _sig(), f"{path} should not invalidate checkpoints"


def the_signature_is_in_the_checkpoint_filename():
    p = _checkpoint_path(Path("results/x"), "pso", 0, 3, "abc1234567")
    assert "abc1234567" in p.name, p.name
    # A pre-signature checkpoint file must not match a signed path, or the old
    # ones would still be picked up.
    assert p.name != "pso__s0__seed3.pkl"


def a_missing_section_does_not_crash():
    cfg = deepcopy(BASE)
    del cfg["fitness"]
    assert isinstance(_config_signature(cfg), str)


check("identical configs give the same signature", identical_configs_give_the_same_signature)
check("the altitude band changes it", the_altitude_band_changes_the_signature)
check("the link model changes it", the_link_model_changes_the_signature)
check("R_comm changes it", the_radius_changes_the_signature)
check("the fitness weights change it", the_fitness_weights_change_the_signature)
check("the budget changes it", the_budget_changes_the_signature)
check("the scenarios change it", the_scenarios_change_the_signature)
check("the seeds change it", the_seeds_change_the_signature)
check("optimizer_params change it", optimizer_params_change_the_signature)
check("cosmetic edits do NOT change it", cosmetic_edits_do_not_change_the_signature)
check("the signature is in the filename", the_signature_is_in_the_checkpoint_filename)
check("a missing section does not crash", a_missing_section_does_not_crash)
finish()
