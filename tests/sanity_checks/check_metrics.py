"""Reported metrics: classification, Jain fairness, comm-cost accounting,
convergence metrics. Payload constants are pinned against the live model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch
from _lib import check, finish
from torch.utils.data import DataLoader, TensorDataset

from uavbench.fl.model import CachedFusionModel
from uavbench.metrics.fl import (
    CLASS_NAMES,
    IOT_MODEL_SIZE_MB,
    UAV_MODEL_SIZE_MB,
    confusion_rows,
    evaluate_loader,
    jain_index,
    round_comm_mb,
)
from uavbench.metrics.placement import convergence_auc, evals_to_threshold


class _EchoModel(torch.nn.Module):
    def forward(self, img_feat, struct):
        return torch.nn.functional.one_hot(struct[:, 0].long(), num_classes=4).float()


class _ConstantModel(torch.nn.Module):
    def forward(self, img_feat, struct):
        logits = torch.zeros(struct.shape[0], 4)
        logits[:, 0] = 1.0
        return logits


def _loader(labels: np.ndarray) -> DataLoader:
    n = len(labels)
    struct = torch.zeros(n, 9)
    struct[:, 0] = torch.from_numpy(labels).float()
    return DataLoader(
        TensorDataset(torch.zeros(n, 512), struct, torch.from_numpy(labels).long()), batch_size=8
    )


def confusion_and_f1():
    labels = np.array([0, 0, 1, 1, 2, 3, 3, 3])
    m = evaluate_loader(_EchoModel(), _loader(labels))
    cm = m["confusion_matrix"]
    assert cm.shape == (4, 4) and cm.sum() == len(labels)
    assert np.array_equal(np.diag(cm), np.bincount(labels, minlength=4))
    assert m["accuracy"] == 1.0 and abs(m["macro_f1"] - 1.0) < 1e-12
    assert list(m["f1_per_class"].keys()) == CLASS_NAMES
    # Majority collapse fills the first column — the failure signature the
    # black-chip diagnostic exists to explain.
    cm2 = evaluate_loader(_ConstantModel(), _loader(np.array([0, 1, 2, 3, 1, 2])))["confusion_matrix"]
    assert np.array_equal(cm2[:, 0], np.bincount(np.array([0, 1, 2, 3, 1, 2]), minlength=4))
    assert cm2[:, 1:].sum() == 0


def confusion_long_form():
    cm = np.arange(16).reshape(4, 4)
    rows = confusion_rows("proposed_hfl", 7, cm)
    assert len(rows) == 16
    assert sum(r["count"] for r in rows) == cm.sum()
    lookup = {(r["true_label"], r["pred_label"]): r["count"] for r in rows}
    assert lookup[("collapsed", "missing")] == cm[1, 3]


def jain_bounds():
    assert jain_index(np.array([3.0, 3.0, 3.0])) == 1.0  # uniform
    assert abs(jain_index(np.array([5.0, 0.0, 0.0, 0.0])) - 0.25) < 1e-12  # 1/n
    assert jain_index(np.zeros(4)) == 1.0  # no selections yet, by convention
    rng = np.random.default_rng(0)
    for _ in range(20):
        c = rng.integers(0, 50, size=8).astype(float)
        j = jain_index(c)
        assert 1.0 / 8 - 1e-12 <= j <= 1.0 + 1e-12


def payload_constants_match_live_model():
    # The MB constants must track the actual parameter counts, or the comm
    # numbers in the paper silently drift from the model.
    model = CachedFusionModel()
    iot_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if name.startswith(("struct_branch", "fusion"))
    )
    uav_params = sum(p.numel() for p in model.parameters())
    assert abs(IOT_MODEL_SIZE_MB - iot_params * 4 / 1e6) < 1e-9, (IOT_MODEL_SIZE_MB, iot_params)
    assert abs(UAV_MODEL_SIZE_MB - uav_params * 4 / 1e6) < 1e-9, (UAV_MODEL_SIZE_MB, uav_params)


def comm_accounting_hand_computed():
    # Hierarchical: 2*12*0.270608 + 2*3*0.533264 = 9.694176 MB.
    assert abs(round_comm_mb(12, n_active_uavs=3) - 9.694176) < 1e-9
    assert abs(round_comm_mb(12) - 2.0 * 12 * IOT_MODEL_SIZE_MB) < 1e-12  # flat path
    assert round_comm_mb(0, n_active_uavs=0) == 0.0


def convergence_metrics():
    conv = [1.0, 2.0, 4.0, 4.0, 4.0]
    assert evals_to_threshold(conv, best=4.0, frac=0.95) == 2
    assert evals_to_threshold(conv, best=-1.0) == -1
    # AUC normalized by shared G_max; early-stopped traces are flat-extended,
    # so a converged-early run is not inflated by a smaller denominator.
    full = convergence_auc([4.0] * 11, G_max=10)
    early = convergence_auc([4.0, 4.0], G_max=10)
    assert abs(full - early) < 1e-12
    assert convergence_auc([], G_max=10) == 0.0


check("classification metrics: diagonal, majority collapse, macro-F1", confusion_and_f1)
check("confusion long-form rows self-describing", confusion_long_form)
check("Jain index bounds and conventions", jain_bounds)
check("payload MB constants match the live model's parameter counts", payload_constants_match_live_model)
check("comm-cost accounting matches hand-computed values", comm_accounting_hand_computed)
check("evals-to-threshold and flat-extended convergence AUC", convergence_metrics)
finish()
