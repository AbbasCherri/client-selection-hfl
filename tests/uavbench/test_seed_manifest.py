"""Seed manifest: exact agreement with the harnesses' own seed derivations."""

import hashlib

import pytest

from uavbench.fl.seeds import fullsim_method_seed, method_hash, sweep_job_seed, tier2_seed
from uavbench.reporting import build_seed_manifest
from uavbench.runner import _instance_seed

# ── formula regression pins (frozen historical behaviour) ────────────────────

class TestSeedFormulas:
    def test_method_hash_matches_md5_fold(self):
        for method in ("proposed_hfl", "fedcs", "mozaffari2016"):
            raw = int(hashlib.md5(method.encode()).hexdigest(), 16)
            assert method_hash(method, 31) == raw % (2 ** 31)
            assert method_hash(method, 16) == raw % (2 ** 16)

    def test_tier2_seed_formula(self):
        # Pin the exact pre-refactor run_tier2 expression.
        raw = int(hashlib.md5(b"pso").hexdigest(), 16) % (2 ** 31)
        assert tier2_seed(9876, 40, "pso") == (9876 + 40 * 7919 + raw) % (2 ** 31)

    def test_fullsim_seed_formula(self):
        raw = int(hashlib.md5(b"proposed_hfl").hexdigest(), 16) % (2 ** 16)
        assert fullsim_method_seed(1234, "proposed_hfl") == (1234 ^ raw) % (2 ** 31)

    def test_sweep_job_seed_formula(self):
        assert sweep_job_seed(9876, 2, 100) == 9876 + 2 * 7919 + 100 * 31


# ── manifest construction ─────────────────────────────────────────────────────

def _tier1_cfg():
    return {
        "instance_seed": 1234,
        "optimizer_seed": 9876,
        "n_seeds": 2,
        "scenarios": [{"distribution": "uniform", "N": 40, "K": 4}],
        "methods": ["pso", "ga"],
    }


class TestBuildSeedManifest:
    def test_tier1_matches_runner_instance_seed(self):
        m = build_seed_manifest(_tier1_cfg(), "tier1")
        assert len(m) == 1 * 2 * 2  # scenarios x methods x seeds
        for _, row in m.iterrows():
            assert row["instance_seed"] == _instance_seed(1234, 0, row["seed_idx"])
        # Instance seed is method-independent (paired comparison property).
        by_seed = m.groupby("seed_idx")["instance_seed"].nunique()
        assert (by_seed == 1).all()

    def test_paper_sweep_grid_and_seeds(self):
        cfg = {
            "optimizer_seed": 9876,
            "n_seeds": 3,
            "N_values": [30, 50],
            "methods": ["proposed_hfl", "fedcs"],
        }
        m = build_seed_manifest(cfg, "paper_sweep")
        assert len(m) == 2 * 2 * 3
        row = m[(m["N"] == 50) & (m["method"] == "fedcs") & (m["seed_idx"] == 1)].iloc[0]
        job = sweep_job_seed(9876, 1, 50)
        assert row["job_seed"] == job
        assert row["seed"] == fullsim_method_seed(job, "fedcs")

    def test_selection_sweep_seed_shared_across_modes(self):
        cfg = {
            "optimizer_seed": 9876,
            "n_seeds": 2,
            "N_values": [30],
            "modes": ["ucb", "random", "fedcs"],
        }
        m = build_seed_manifest(cfg, "selection_sweep")
        # The deliberate design: identical seed for every mode at a given
        # (N, seed_idx), isolating the selection rule.
        per_cell = m.groupby(["N", "seed_idx"])["seed"].nunique()
        assert (per_cell == 1).all()

    def test_tier2_seed_uses_configured_n(self):
        cfg = {"optimizer_seed": 9876, "data": {"N_clients": 40}, "methods": ["pso"]}
        m = build_seed_manifest(cfg, "tier2")
        assert m.iloc[0]["seed"] == tier2_seed(9876, 40, "pso")
        assert "note" in m.columns  # the n_clients caveat travels with the data

    def test_unknown_harness_raises(self):
        with pytest.raises(ValueError, match="unknown harness"):
            build_seed_manifest({}, "nope")


def test_manifest_matches_live_run_tier2_seed():
    """End-to-end pin: manifest seed equals the rng seed run_tier2 derives.

    run_tier2 folds len(clients) into its seed; on synthetic data every
    client keeps a non-empty shard, so len(clients) == N_clients and the
    manifest's assumption holds exactly.
    """
    from uavbench.fl.dataset import SyntheticClientData

    N = 24
    raw = SyntheticClientData(N=200, K=N, seed=42).build()
    n_loaded = sum(1 for cid in raw["client_coords"] if raw["client_train_indices"].get(cid))
    assert n_loaded == N  # synthetic mode never drops clients

    cfg = {"optimizer_seed": 9876, "data": {"N_clients": N}, "methods": ["pso"]}
    m = build_seed_manifest(cfg, "tier2")
    assert m.iloc[0]["seed"] == tier2_seed(9876, n_loaded, "pso")
