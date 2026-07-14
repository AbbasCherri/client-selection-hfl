"""Confusion-matrix reporting: _evaluate_loader output and long-form rows."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from uavbench.fl.federated import CLASS_NAMES, _confusion_rows, _evaluate_loader


class _EchoModel(torch.nn.Module):
    """Predicts the class encoded in the first structured feature (perfect model)."""

    def forward(self, img_feat, struct):
        labels = struct[:, 0].long()
        return torch.nn.functional.one_hot(labels, num_classes=4).float()


class _ConstantModel(torch.nn.Module):
    """Always predicts class 0 (majority-collapse model)."""

    def forward(self, img_feat, struct):
        logits = torch.zeros(struct.shape[0], 4)
        logits[:, 0] = 1.0
        return logits


def _loader(labels: np.ndarray) -> DataLoader:
    n = len(labels)
    img = torch.zeros(n, 512)
    struct = torch.zeros(n, 9)
    struct[:, 0] = torch.from_numpy(labels).float()
    return DataLoader(
        TensorDataset(img, struct, torch.from_numpy(labels).long()), batch_size=8
    )


def test_perfect_prediction_gives_diagonal():
    labels = np.array([0, 0, 1, 1, 2, 3, 3, 3])
    metrics = _evaluate_loader(_EchoModel(), _loader(labels))
    cm = metrics["confusion_matrix"]
    assert cm.shape == (4, 4)
    assert np.array_equal(np.diag(cm), np.bincount(labels, minlength=4))
    assert cm.sum() == len(labels)
    assert (cm - np.diag(np.diag(cm))).sum() == 0
    assert metrics["accuracy"] == 1.0


def test_majority_collapse_fills_first_column():
    labels = np.array([0, 1, 2, 3, 1, 2])
    cm = _evaluate_loader(_ConstantModel(), _loader(labels))["confusion_matrix"]
    assert np.array_equal(cm[:, 0], np.bincount(labels, minlength=4))
    assert cm[:, 1:].sum() == 0


def test_row_sums_equal_per_class_counts():
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 4, size=40)
    cm = _evaluate_loader(_EchoModel(), _loader(labels))["confusion_matrix"]
    assert np.array_equal(cm.sum(axis=1), np.bincount(labels, minlength=4))


def test_confusion_rows_long_form():
    cm = np.arange(16).reshape(4, 4)
    rows = _confusion_rows("proposed_hfl", 7, cm)
    assert len(rows) == 16
    assert all(r["method"] == "proposed_hfl" and r["round"] == 7 for r in rows)
    assert sum(r["count"] for r in rows) == cm.sum()
    # Label names, not indices, so the parquet is self-describing.
    assert {r["true_label"] for r in rows} == set(CLASS_NAMES)
    lookup = {(r["true_label"], r["pred_label"]): r["count"] for r in rows}
    assert lookup[("collapsed", "missing")] == cm[1, 3]
