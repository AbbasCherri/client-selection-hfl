"""Sweep resume gate + job failure isolation.

Regression coverage for the 2026-07-16 paper run: (1) stale checkpoints from
an earlier run with different settings (or an older pipeline version) were
silently reused as results; (2) one job's exception aborted the whole sweep,
discarding hours of sibling-job compute.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import yaml
from _lib import check, finish

from uavbench.fl.sweep import (
    PIPELINE_VERSION,
    _collect_or_raise,
    _isolated,
    _resume_signature,
    _stale_checkpoint_reason,
)


def _cfg(**overrides) -> dict:
    base = {
        "_pipeline_version": PIPELINE_VERSION,
        "data": {"source": "real", "subsample": 1.0, "seed": 42, "N_clients": 30},
        "fl": {"n_rounds": 100, "seed": 123, "K": 20},
        "budget": {"P": 50, "G_max": 30},
        "methods": ["proposed_hfl"],
        "optimizer_params": {},
        "epicentre": [37.488, 137.272],
        "results_dir": "results/x",
    }
    base.update(overrides)
    return base


def _write_ckpt(d: str, cfg: dict) -> Path:
    p = Path(d) / "config.fullsim.resolved.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f)
    return p


def signature_normalizes_and_ignores_volatile_keys():
    a = _cfg()
    b = _cfg()
    b["epicentre"] = (37.488, 137.272)  # tuple vs list must compare equal
    b["data"] = {**b["data"], "hf_token": "secret", "feature_cache_path": "/tmp/f.npy"}
    b["results_dir"] = "somewhere/else"  # location, not semantics
    assert _resume_signature(a) == _resume_signature(b)
    c = _cfg()
    c["data"] = {**c["data"], "subsample": 0.05}
    assert _resume_signature(a) != _resume_signature(c)


def stale_checkpoint_detection():
    with tempfile.TemporaryDirectory() as d:
        # Matching config → resume allowed.
        assert _stale_checkpoint_reason(_write_ckpt(d, _cfg()), _cfg()) is None
        # Pre-versioning checkpoint (the 2026-07-14 leftovers) → stale.
        old = _cfg()
        del old["_pipeline_version"]
        reason = _stale_checkpoint_reason(_write_ckpt(d, old), _cfg())
        assert reason is not None and "_pipeline_version" in reason
        # Different data settings → stale, and the reason names the section.
        changed = _cfg()
        changed["data"] = {**changed["data"], "subsample": 0.05}
        reason = _stale_checkpoint_reason(_write_ckpt(d, changed), _cfg())
        assert reason is not None and "data" in reason
        # Different seed → stale (seed streams changed 2026-07: old Tier-1
        # results are not seed-comparable).
        seeded = _cfg()
        seeded["fl"] = {**seeded["fl"], "seed": 999}
        assert _stale_checkpoint_reason(_write_ckpt(d, seeded), _cfg()) is not None
        # Unreadable checkpoint → stale, not a crash.
        bad = Path(d) / "config.fullsim.resolved.yaml"
        bad.write_text("]: not yaml [")
        assert _stale_checkpoint_reason(bad, _cfg()) is not None


def _ok_job(tag_df_value: int) -> pd.DataFrame:
    return pd.DataFrame({"x": [tag_df_value]})


def _boom_job(_: int) -> pd.DataFrame:
    raise RuntimeError("synthetic job failure")


def one_failure_does_not_abort_and_is_not_silent():
    results = [
        _isolated(_ok_job, "job-a", 1),
        _isolated(_boom_job, "job-b", 2),
        _isolated(_ok_job, "job-c", 3),
    ]
    # Every job ran; the failure is a record, not an abort.
    assert [tag for tag, _, _ in results] == ["job-a", "job-b", "job-c"]
    assert results[1][1] is None and "synthetic job failure" in results[1][2]
    # Aggregation refuses to pretend the sweep succeeded (no silent failures).
    try:
        _collect_or_raise(results)
    except RuntimeError as e:
        assert "job-b" in str(e) and "1/3" in str(e)
    else:
        raise AssertionError("failed jobs must raise after collection")
    # All-success path concatenates in job order.
    ok = [_isolated(_ok_job, t, v) for t, v in [("a", 1), ("b", 2)]]
    df = _collect_or_raise(ok)
    assert df["x"].tolist() == [1, 2]


check("resume signature: normalized, volatile keys ignored, semantics compared", signature_normalizes_and_ignores_volatile_keys)
check("stale checkpoints (old version / changed config / corrupt) are refused", stale_checkpoint_detection)
check("one failing job neither aborts the sweep nor fails silently", one_failure_does_not_abort_and_is_not_silent)
finish()
