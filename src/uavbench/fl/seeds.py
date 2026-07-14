"""Seed-derivation formulas shared by the FL harnesses and the seed manifest.

Each harness's formula lives here — and only here — so the seed manifest
(:mod:`uavbench.reporting.seed_manifest`) can enumerate the exact seeds a
run will use by calling the same functions the harness calls, with no risk
of a reimplementation drifting.

The formulas are frozen: they reproduce the historical behaviour of
``run_tier2`` / ``run_full_hfl`` / the sweep jobs bit-for-bit (pinned by
tests/uavbench/test_reproducibility.py). Note the two hash widths are
intentionally different (31-bit additive fold for tier-2; 16-bit XOR fold
for the full sim) — do not "unify" them, doing so changes every seed.
"""

from __future__ import annotations

import hashlib


def method_hash(method: str, bits: int) -> int:
    """Stable ``bits``-wide hash of a method name (md5 fold)."""
    return int(hashlib.md5(method.encode()).hexdigest(), 16) % (2 ** bits)


def tier2_seed(optimizer_seed: int, n_clients: int, method: str) -> int:
    """Per-method seed for ``run_tier2``.

    ``n_clients`` is the number of clients actually loaded (clients with
    empty shards are dropped, so it can be below the configured N on real
    data).
    """
    return (optimizer_seed + n_clients * 7919 + method_hash(method, 31)) % (2 ** 31)


def fullsim_method_seed(run_seed: int, method: str) -> int:
    """Per-method seed for ``run_full_hfl``.

    The method identity is folded in exactly once, here; sweep callers must
    NOT pre-encode the method into ``run_seed`` (double-counting silently
    shifts every per-method RNG draw).
    """
    return (run_seed ^ method_hash(method, 16)) % (2 ** 31)


def sweep_job_seed(optimizer_seed: int, seed_idx: int, N: int) -> int:
    """Per-(N, seed_idx) run seed used by the paper and selection sweeps.

    Deliberately method-free: the selection-isolation benchmark requires
    every mode to see the identical problem instance per (N, seed), and the
    paper sweep folds the method in later via :func:`fullsim_method_seed`.
    """
    return optimizer_seed + seed_idx * 7919 + N * 31
