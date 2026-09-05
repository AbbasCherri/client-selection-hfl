"""Every per-method job builder must apply `fl.per_method`.

The failure this guards is silent and was live for the entire project until
2026-09-05: only `_paper_job` applied the per-method recipes. Every other
harness — the radius sweep, the fleet sweep, the selection-isolation sweep —
ran EVERY method on the BASE recipe, which is `proposed_hfl`'s own. That is the
single-arm tuning defect the main table was explicitly fixed to remove, still
live everywhere else, and it silently favoured or penalised baselines depending
on how far their own recipe sat from the proposed method's.

It cannot be caught by comparing outputs: each sweep looks internally
consistent. It is only visible by cross-checking the SAME cell between two
harnesses, which is how it was found (proposed - fedcs at N=200/R=5000 read
+0.051 in the main table and -0.024 in the radius sweep).

Source-level rather than behavioural because these builders run a complete FL
job; there is no cheap way to invoke one.
"""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _lib import check, finish  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FILES = ("src/uavbench/fl/sweep.py", "src/uavbench/fl/selection_isolation.py")


def _builders():
    """Every module-level function taking a `method` param and building job_cfg."""
    out = []
    for rel in FILES:
        path = ROOT / rel
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            args = [a.arg for a in node.args.args]
            if "method" not in args or "cfg" not in args:
                continue
            src = ast.get_source_segment(path.read_text(), node) or ""
            if "job_cfg" not in src:
                continue
            out.append((rel, node.name, src))
    return out


def every_job_builder_is_discovered():
    b = _builders()
    names = {n for _, n, _ in b}
    assert "_paper_job" in names, f"discovery broken; found {names}"
    assert len(b) >= 4, f"expected >=4 job builders, found {sorted(names)}"


def every_job_builder_applies_per_method():
    missing = []
    for rel, name, src in _builders():
        if 'pop("per_method"' not in src or "_per_method.get(method" not in src:
            missing.append(f"{rel}::{name}")
    assert not missing, (
        "these job builders do NOT apply fl.per_method, so every method in "
        "their sweep runs proposed_hfl's base recipe: " + ", ".join(missing)
    )


def per_method_is_popped_not_read():
    # Popping keeps it out of the resume signature; a plain .get() would make
    # editing one method's numbers invalidate every other method's checkpoints.
    for rel, name, src in _builders():
        assert 'get("per_method"' not in src.replace('pop("per_method"', ""), (
            f"{rel}::{name} reads per_method without popping it"
        )


check("every job builder is discovered", every_job_builder_is_discovered)
check("every job builder applies per_method", every_job_builder_applies_per_method)
check("per_method is popped, not read", per_method_is_popped_not_read)
finish()
