"""Tiny pass/fail harness for the manual sanity-check scripts.

Not pytest. Each check_*.py script runs standalone:

    python tests/sanity_checks/check_whatever.py

and exits 0 (all PASS) or 1 (any FAIL, with the failure printed). Run the
whole list with run_all.py before trusting a batch of results.
"""

from __future__ import annotations

import sys
import traceback

_failures: list[str] = []
_passes = 0


def check(name: str, fn) -> None:
    """Run ``fn`` (raises on failure); print one PASS/FAIL line."""
    global _passes
    try:
        fn()
    except Exception:
        _failures.append(name)
        print(f"FAIL  {name}")
        traceback.print_exc()
    else:
        _passes += 1
        print(f"pass  {name}")


def finish() -> None:
    print(f"\n{_passes} passed, {len(_failures)} failed")
    if _failures:
        for f in _failures:
            print(f"  FAILED: {f}")
        sys.exit(1)
