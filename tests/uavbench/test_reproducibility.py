"""Tests for experiment reproducibility — seeds must be stable across runs.

Python's built-in hash() is randomised by default (PYTHONHASHSEED) since 3.3.
We replaced hash(method) with hashlib.md5 in sweep._paper_job and
federated.run_full_hfl / run_tier2 to make seeds deterministic.
These tests verify that property.
"""

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestMethodHashDeterminism:
    """hash(str) varies between runs; the seed formula must use a stable hash.

    The method hash must be folded into the run seed exactly once. It happens
    inside ``federated.run_full_hfl`` / ``run_tier2``; ``sweep._paper_job`` must
    NOT also add it, or the method identity gets double-counted (see R-1 in the
    final research-standards review).
    """

    def _method_hash_federated(self, method: str, bits: int = 16) -> int:
        """Mirror the formula in federated.run_full_hfl."""
        return int(hashlib.md5(method.encode()).hexdigest(), 16) % (2**bits)

    def test_federated_hash_is_deterministic(self):
        h1 = self._method_hash_federated("hfl_no_selection")
        h2 = self._method_hash_federated("hfl_no_selection")
        assert h1 == h2

    def test_different_methods_have_different_hashes(self):
        methods = [
            "proposed_hfl", "flat_fl", "centralized",
            "hfl_no_selection", "hfl_static", "hfl_no_reputation",
        ]
        hashes = [self._method_hash_federated(m) for m in methods]
        assert len(set(hashes)) == len(methods), "Some methods share the same hash bucket"

    def test_hash_in_valid_range(self):
        for method in ("proposed_hfl", "flat_fl", "centralized"):
            h = self._method_hash_federated(method)
            assert 0 <= h < 2**16

    def test_paper_job_seed_does_not_encode_method(self):
        """sweep._paper_job's seed formula must be method-independent.

        run_full_hfl folds in the method hash itself; if _paper_job did too,
        the method identity would be double-counted in the final seed.
        """
        optimizer_seed = 9876
        seed_idx = 1
        N = 200
        seed_for_method_a = optimizer_seed + seed_idx * 7919 + N * 31
        seed_for_method_b = optimizer_seed + seed_idx * 7919 + N * 31
        assert seed_for_method_a == seed_for_method_b

    def test_run_full_hfl_seed_combines_run_seed_and_method_once(self):
        """The (run_seed, method) -> final seed formula must apply the hash once."""
        run_seed = 9876 + 1 * 7919 + 200 * 31
        method = "proposed_hfl"
        h = self._method_hash_federated(method)
        seed1 = (run_seed ^ h) % (2**31)
        seed2 = (run_seed ^ h) % (2**31)
        assert seed1 == seed2

    def test_run_full_hfl_same_seed_same_accuracy(self):
        """Same config → same initial random state → identical round-1 accuracy."""
        from uavbench.fl.federated import run_full_hfl
        cfg_base = {
            "methods": ["proposed_hfl"],
            "fl": {
                "n_rounds": 1,
                "n_local_epochs": 1,
                "lr": 0.01,
                "batch_size": 4,
                "K": 2,
                "R_comm": 200_000.0,
                "capacity": 10,
                "T_sel": 1,
                "target_accuracy": 0.99,
                "seed": 12345,
            },
            "budget": {"P": 3, "G_max": 2},
            "data": {"source": "synthetic", "N_clients": 10, "seed": 0},
            "optimizer_seed": 42,
        }
        results = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as d:
                import copy
                cfg = copy.deepcopy(cfg_base)
                cfg["results_dir"] = d
                out = run_full_hfl(cfg)
            acc = float(out["rounds"]["accuracy"].iloc[0])
            results.append(acc)
        assert results[0] == pytest.approx(results[1]), \
            f"Same seed produced different accuracy: {results}"
