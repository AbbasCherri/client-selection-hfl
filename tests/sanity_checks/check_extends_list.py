#!/usr/bin/env python
"""Guard: `extends:` accepts a list of parents, merged left to right, child last.

Also asserts the property that motivated the change — a config inheriting both a
domain base and a recipe file gets the recipe file's tuned values, not the
domain base's — because silently picking up the WRONG recipe is the failure this
mechanism exists to prevent, and it is invisible in the output.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uavbench.runner import load_config  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    print("check_extends_list")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "domain.yaml").write_text(
            "name: domain\nn_seeds: 10\nfl:\n  lr: 0.001\n  rounds: 100\n")
        (d / "recipes.yaml").write_text(
            "fl:\n  lr: 0.049\n  per_method:\n    flat_fl:\n      lr: 0.049\n")
        (d / "child.yaml").write_text(
            "extends: [domain.yaml, recipes.yaml]\nname: child\n"
            "fl:\n  fusion_owner: client_full\n")
        (d / "single.yaml").write_text("extends: domain.yaml\nname: single\n")
        (d / "order.yaml").write_text(
            "extends: [recipes.yaml, domain.yaml]\nname: order\n")

        cfg = load_config(d / "child.yaml")
        check("list form loads", cfg.get("name") == "child")
        check("domain keys inherited", cfg.get("n_seeds") == 10)
        check("domain nested keys survive", cfg["fl"].get("rounds") == 100)
        check("LATER parent wins on conflict", cfg["fl"]["lr"] == 0.049,
              f"lr={cfg['fl']['lr']} (recipes 0.049 must beat domain 0.001)")
        check("recipe subtrees merge", cfg["fl"]["per_method"]["flat_fl"]["lr"] == 0.049)
        check("child overrides both", cfg["fl"]["fusion_owner"] == "client_full")

        single = load_config(d / "single.yaml")
        check("string form still works", single.get("n_seeds") == 10)
        check("string form unaffected by change", single["fl"]["lr"] == 0.001)

        order = load_config(d / "order.yaml")
        check("order is left-to-right", order["fl"]["lr"] == 0.001,
              f"lr={order['fl']['lr']} (domain listed last must win here)")

    print()
    if FAILS:
        print(f"FAILED {len(FAILS)}: {', '.join(FAILS)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
