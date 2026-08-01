"""One-at-a-time sensitivity screening of the simulator's environment constants.

The problem this addresses: the device/energy model is full of hand-picked
numbers — ``T_MAX_S=300``, ``B_MIN=0.20``, battery discharge 0.02, compute time
``U[50,250]``, and so on. None is measured, none is cited, and a reviewer is
entitled to ask whether the paper's conclusions are artifacts of those choices.

The right response is NOT to tune them. Tuning environment constants is how you
end up with a simulator that flatters the proposed method. The response is to
show the **sign of the proposed-vs-baseline gap is invariant** across a wide
perturbation of each one, and to report openly any constant where it is not.

Constants are classified in REPORTS/rigor_plan_2026-08.md §Phase 6:
  Class E   — environment/physics: screened here, never tuned.
  Class M   — a method's own knobs: tuned, with equal budget per method.
  Class T   — shared training recipe: tuned per method.

``T_MAX_S`` is the headline case and the reason this module exists. At 300 s
against a compute-time distribution of ``U[50,250] + N(0,30)``, the deadline
excludes 0.28% of devices on the raw gate and 8.4% once the adaptive margin
applies — so **FedCS's entire distinguishing mechanism (deadline-constrained
greedy) barely fires**, and it degenerates toward cheapest-first capacity fill.
Screening it is not diligence, it is what makes FedCS a fair baseline.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Class-E constants, with the module path they live at and a screening range.
# Ranges are ±50% of the current value unless the constant has a natural bound
# (a probability, a fraction) or a value whose *point* is a specific regime.
ENV_PARAMS: dict[str, dict] = {
    "T_MAX_S": {
        "module": "uavbench.fl.device_state", "attr": "T_MAX_S",
        "values": [150.0, 225.0, 300.0],
        "note": "round deadline; at 300 s it excludes 0.28% of devices raw / "
                "8.4% with the margin, leaving FedCS's rule nearly inert. "
                "Nishio & Yonetani sweep this rather than fixing it.",
    },
    "B_MIN": {
        "module": "uavbench.fl.device_state", "attr": "B_MIN",
        "values": [0.10, 0.20, 0.30],
        "note": "battery eligibility floor",
    },
    "SNR_MIN_DB": {
        "module": "uavbench.fl.device_state", "attr": "SNR_MIN_DB",
        "values": [1.5, 3.0, 6.0],
        "note": "SNR eligibility floor",
    },
    "UCB_C": {
        "module": "uavbench.fl.client_selection", "attr": "UCB_C",
        "values": [0.707, 1.414, 2.828],
        "note": "UCB exploration constant. Class-M strictly, but it is the "
                "theoretical sqrt(2) and was never searched — screening it is "
                "cheaper than defending 'we used the textbook value'.",
    },
}


@dataclass
class ScreenResult:
    param: str
    value: float
    method: str
    macro_f1: float
    accuracy: float
    seed: int


def _patch(module_path: str, attr: str, value):
    """Set a module-level constant, returning the previous value.

    The harnesses read these at call time (not import time), so patching the
    module attribute is enough — the same mechanism scripts/tune_weights.py
    already uses for the Class-M weights.
    """
    import importlib

    mod = importlib.import_module(module_path)
    old = getattr(mod, attr)
    setattr(mod, attr, value)
    return mod, old


def screen_one(
    param: str, value: float, base_cfg: dict, methods: list[str], seed: int
) -> list[ScreenResult]:
    """Run ``methods`` at one perturbed value of one environment constant."""
    from .federated import run_full_hfl

    spec = ENV_PARAMS[param]
    mod, old = _patch(spec["module"], spec["attr"], value)
    try:
        cfg = copy.deepcopy(base_cfg)
        cfg["methods"] = list(methods)
        cfg["fl"]["seed"] = seed
        out = run_full_hfl(cfg)
        rounds = out["rounds"]
        final = rounds[rounds["round"] == rounds["round"].max()]
        return [
            ScreenResult(
                param=param, value=value, method=str(r["method"]),
                macro_f1=float(r["macro_f1"]), accuracy=float(r["accuracy"]), seed=seed,
            )
            for _, r in final.iterrows()
        ]
    finally:
        setattr(mod, spec["attr"], old)  # never leak a patch into the next cell


def gap_table(results: list[ScreenResult], reference: str) -> pd.DataFrame:
    """Per (param, value): the reference method's margin over each other arm.

    The screening question is NOT "does accuracy move" — of course it does, the
    environment changed. It is "does the *sign* of the gap survive". A row with
    ``sign_flipped`` True is a finding, and belongs in the limitations section.
    """
    df = pd.DataFrame([r.__dict__ for r in results])
    if df.empty:
        return df
    rows = []
    for (param, value, seed), sub in df.groupby(["param", "value", "seed"]):
        ref = sub[sub["method"] == reference]
        if ref.empty:
            continue
        ref_f1 = float(ref["macro_f1"].iloc[0])
        for _, r in sub[sub["method"] != reference].iterrows():
            rows.append({
                "param": param, "value": value, "seed": seed,
                "vs": r["method"], "gap_macro_f1": ref_f1 - float(r["macro_f1"]),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    agg = out.groupby(["param", "vs"])["gap_macro_f1"].agg(["mean", "min", "max"])
    agg["sign_flipped"] = (agg["min"] < 0) != (agg["max"] < 0)
    return agg.reset_index()
