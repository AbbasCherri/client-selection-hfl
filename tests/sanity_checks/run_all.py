"""Run every sanity check before trusting a batch of results.

    python tests/sanity_checks/run_all.py

Exits non-zero if any script fails. This replaces the old pytest suite —
run it by hand before generating paper numbers, not on every commit.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SCRIPTS = sorted(p for p in HERE.glob("check_*.py"))

failed = []
for script in SCRIPTS:
    print(f"\n=== {script.name} ===")
    rc = subprocess.run([sys.executable, str(script)]).returncode
    if rc != 0:
        failed.append(script.name)

print("\n" + "=" * 50)
if failed:
    print(f"{len(failed)}/{len(SCRIPTS)} scripts FAILED: {failed}")
    sys.exit(1)
print(f"all {len(SCRIPTS)} sanity-check scripts passed")
