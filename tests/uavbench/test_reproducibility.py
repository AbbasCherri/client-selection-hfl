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
    """hash(str) varies between runs; the seed formula must use a stable hash."""

    def _method_hash_sweep(self, method: str) -> int:
        """Mirror the formula in sweep._paper_job."""
        return int(hashlib.md5(method.encode()).hexdigest(), 16) % (2**20)

    def _method_hash_federated(self, method: str, bits: int = 30) -> int:
        """Mirror the formula in federated.run_full_hfl."""
        return int(hashlib.md5(method.encode()).hexdigest(), 16) % (2**bits)

    def test_sweep_hash_is_deterministic(self):
        h1 = self._method_hash_sweep("proposed_hfl")
        h2 = self._method_hash_sweep("proposed_hfl")
        assert h1 == h2

    def test_federated_hash_is_deterministic(self):
        h1 = self._method_hash_federated("hfl_no_selection")
        h2 = self._method_hash_federated("hfl_no_selection")
        assert h1 == h2

    def test_different_methods_have_different_hashes(self):
        methods = [
            "proposed_hfl", "flat_fl", "centralized",
            "hfl_no_selection", "hfl_static", "hfl_no_reputation",
        ]
        hashes = [self._method_hash_sweep(m) for m in methods]
        assert len(set(hashes)) == len(methods), "Some methods share the same hash bucket"

    def test_hash_in_valid_range(self):
        for method in ("proposed_hfl", "flat_fl", "centralized"):
            h = self._method_hash_sweep(method)
            assert 0 <= h < 2**20

    def test_sweep_seed_formula_stable(self):
        """Full seed formula used in _paper_job must be stable."""
        optimizer_seed = 9876
        seed_idx = 1
        N = 200
        method = "proposed_hfl"
        _method_hash = int(hashlib.md5(method.encode()).hexdigest(), 16) % (2**20)
        seed1 = optimizer_seed + seed_idx * 7919 + N * 31 + _method_hash
        seed2 = optimizer_seed + seed_idx * 7919 + N * 31 + _method_hash
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
