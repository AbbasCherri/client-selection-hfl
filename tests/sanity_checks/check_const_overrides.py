"""Environment/baseline constant overrides actually take effect (Phase 6, §0.3).

Phase 6 screens the simulator's hand-picked environment constants by patching
module globals per job. The failure mode this guards is silent and expensive:
if the patch did not reach the joblib WORKER process, every screening cell
would run at the stock constants, all 12 cells would agree, and the result
would read as the reassuring "the conclusion is insensitive to these
constants" — the exact claim the screen is supposed to earn rather than
assume. Nothing in a run's output would look wrong.

So this exercises the real mechanism end to end: patch through
``apply_const_overrides`` inside a loky worker, and confirm both the constant
and the DOWNSTREAM DECISION it feeds (``DeviceState.eligible``) changed.
"""

import sys
from pathlib import Path

from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
from _lib import check, finish  # noqa: E402

import uavbench.fl.device_state as ds  # noqa: E402
from uavbench.fl.selection_isolation import (  # noqa: E402
    apply_const_overrides,
    restore_const_overrides,
)

T_PATH = "uavbench.fl.device_state.T_MAX_S"


def _worker_probe(value: float) -> tuple[float, bool]:
    """Patch T_MAX_S inside a worker; report the constant and a decision it drives.

    Module-level (not a closure) because loky pickles by reference.
    """
    import uavbench.fl.device_state as w_ds
    from uavbench.fl.selection_isolation import apply_const_overrides

    apply_const_overrides({T_PATH: value})
    # compute_time_s=200 sits between the two probe values, so eligibility
    # flips iff the override truly reached this process.
    state = w_ds.DeviceState(battery=1.0, snr_db=30.0, memory_ok=True,
                             compute_time_s=200.0)
    return w_ds.T_MAX_S, state.eligible()


def override_reaches_the_joblib_worker():
    got = Parallel(n_jobs=2)(delayed(_worker_probe)(v) for v in (150.0, 450.0))
    assert got[0][0] == 150.0 and got[1][0] == 450.0, (
        f"constant did not change inside workers: {got}"
    )
    assert got[0][1] is False, "T_MAX_S=150 should exclude a 200 s device"
    assert got[1][1] is True, "T_MAX_S=450 should admit a 200 s device"


def override_changes_the_eligibility_decision():
    """In-process version of the same claim, so a failure localises."""
    device = ds.DeviceState(battery=1.0, snr_db=30.0, memory_ok=True,
                            compute_time_s=200.0)
    restores = apply_const_overrides({T_PATH: 150.0})
    try:
        assert not device.eligible(), "deadline 150 s admitted a 200 s device"
    finally:
        restore_const_overrides(restores)
    assert device.eligible(), "restore did not put the stock deadline back"


def restore_returns_the_original_value():
    original = ds.T_MAX_S
    restores = apply_const_overrides({T_PATH: 42.0})
    assert ds.T_MAX_S == 42.0
    restore_const_overrides(restores)
    assert ds.T_MAX_S == original, (
        f"restore left {ds.T_MAX_S} instead of {original} — worker processes are "
        "reused, so a leaked value would contaminate the next cell"
    )


def typo_in_the_path_raises():
    """A misspelled constant must not screen nothing at all, silently."""
    try:
        apply_const_overrides({"uavbench.fl.device_state.T_MAX_SECONDS": 1.0})
    except AttributeError:
        return
    raise AssertionError("a nonexistent constant path was accepted")


def non_numeric_value_raises():
    """The Phase 6 shell wrapper once passed the constant's PATH as its value.

    It set T_MAX_S to a string, which surfaced hundreds of lines later as a
    TypeError in the energy model, after a garbage results directory had
    already been written.
    """
    try:
        apply_const_overrides({T_PATH: T_PATH})
    except TypeError:
        assert ds.T_MAX_S != T_PATH, "rejected the value but assigned it anyway"
        return
    raise AssertionError("a string was accepted for a numeric constant")


if __name__ == "__main__":
    check("override reaches the joblib worker", override_reaches_the_joblib_worker)
    check("override changes the eligibility decision", override_changes_the_eligibility_decision)
    check("restore returns the original value", restore_returns_the_original_value)
    check("a typo in the constant path raises", typo_in_the_path_raises)
    check("a non-numeric value raises", non_numeric_value_raises)
    finish()
