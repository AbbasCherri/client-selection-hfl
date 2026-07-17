"""FL-side reported metrics, consolidated in one module.

Everything a results table reports about a federated run lives here —
classification quality (accuracy, macro-F1, per-class F1, confusion
matrix), selection fairness (Jain's index), and communication-cost
accounting — so every harness (Tier-2, full sim, selection isolation)
shares one implementation and the comparison can't drift per-method.

Placement-optimizer metrics (coverage, convergence AUC, movement energy)
stay in :mod:`uavbench.metrics.placement`, which is numpy-only so the
Tier-1 harness never imports torch. The physical energy model likewise
stays in :mod:`uavbench.problem.energy`: it is a simulation input, not a
post-hoc reporting metric.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset

# Damage classes in label order 0-3 (shared by confusion reporting/plots).
CLASS_NAMES = ["survived", "collapsed", "obstructed", "missing"]

# rounds_to_target: a single-round accuracy crossing can be a transient — on
# the imbalanced test set, near-majority-class predictions in the first rounds
# reach ~0.70 before the model has learned the minority classes (observed in
# the 2026-07-16 run: rounds_to_target=1 with final accuracy far below the
# target). Every harness therefore counts the target as reached only when the
# accuracy holds for this many consecutive evaluations.
TARGET_CONSEC_ROUNDS = 2

# Communication payload sizes (float32 parameter counts x 4 bytes):
# IoT payload:  struct_branch (17,216) + fusion (50,436)         = 67,652 params ~ 0.271 MB
# UAV payload:  img_proj (65,664) + struct_branch + fusion       = 133,316 params ~ 0.533 MB
IOT_MODEL_SIZE_MB: float = 67_652 * 4 / 1_000_000
UAV_MODEL_SIZE_MB: float = 133_316 * 4 / 1_000_000


def round_comm_mb(n_selected: int, n_active_uavs: int | None = None) -> float:
    """Per-round communication cost in MB (up + down, hence the factor 2).

    ``n_active_uavs=None`` is the flat/no-hierarchy case: only IoT payloads
    move. With a UAV tier, each active UAV additionally exchanges the larger
    UAV payload with the server. Single accounting rule for every method —
    do not reimplement this per-harness.
    """
    mb = 2.0 * n_selected * IOT_MODEL_SIZE_MB
    if n_active_uavs is not None:
        mb += 2.0 * n_active_uavs * UAV_MODEL_SIZE_MB
    return mb


def _classification_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    """Accuracy, macro-F1, per-class F1, confusion matrix from label/pred arrays."""
    acc = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0, labels=[0, 1, 2, 3]))
    per_class = f1_score(labels, preds, average=None, zero_division=0, labels=[0, 1, 2, 3])
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "f1_per_class": dict(zip(CLASS_NAMES, per_class.tolist())),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1, 2, 3]),
    }


def evaluate_loader(model, loader: DataLoader) -> dict:
    """Accuracy, macro-F1, per-class F1, and confusion matrix on a test loader."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for img_feat, struct, labels in loader:
            preds = model(img_feat, struct).argmax(dim=1)
            all_preds.append(preds.numpy())
            all_labels.append(labels.numpy())

    return _classification_metrics(np.concatenate(all_labels), np.concatenate(all_preds))


def evaluate_subset(model, dataset, indices: list[int], batch_size: int = 512) -> dict:
    """Sequential, unweighted evaluation on ``dataset[indices]``; zero metrics if empty.

    Datasets exposing ``eval_tensors()`` (CachedDataset) are evaluated by
    slicing the in-memory tensors directly — order-preserving and unshuffled,
    exactly like the DataLoader fallback (shuffle=False), but without the
    per-item ``__getitem__`` + collate overhead. This runs every round, so it
    is the hottest evaluation path in every FL harness.
    """
    if not indices:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "f1_per_class": {},
            "confusion_matrix": np.zeros((4, 4), dtype=int),
        }

    eval_tensors = getattr(dataset, "eval_tensors", None)
    if eval_tensors is not None:
        img, struct, labels = eval_tensors()
        idx = torch.as_tensor(indices, dtype=torch.long)
        model.eval()
        pred_chunks = []
        with torch.inference_mode():
            for b in idx.split(batch_size):
                pred_chunks.append(model(img[b], struct[b]).argmax(dim=1))
        preds = torch.cat(pred_chunks).numpy()
        return _classification_metrics(labels[idx].numpy(), preds)

    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    return evaluate_loader(model, loader)


def confusion_rows(method: str, rnd: int, cm: np.ndarray) -> list[dict]:
    """Flatten a (4,4) confusion matrix into long-form rows for parquet.

    The 4x4 matrix doesn't belong in the flat per-round row; long form keeps
    the rounds table schema stable and the matrix queryable per (method, round).
    """
    return [
        {
            "method": method,
            "round": rnd,
            "true_label": CLASS_NAMES[t],
            "pred_label": CLASS_NAMES[p],
            "count": int(cm[t, p]),
        }
        for t in range(4)
        for p in range(4)
    ]


def jain_index(counts: np.ndarray) -> float:
    """Jain's fairness index over cumulative selection counts.

    ``J(x) = (Σx)² / (n·Σx²)`` (R. Jain, D.-M. Chiu, W. Hawe, DEC TR-301,
    1984), bounded in ``[1/n, 1]``; 1 = perfectly fair. Measures how evenly
    *selection frequency* spreads across clients over rounds — distinct from
    the placement-side load-imbalance term (within-round device-to-UAV
    balance). All-zero counts (no selections yet) return 1.0 by convention —
    nobody has been favoured.
    """
    total = counts.sum()
    if total <= 0:
        return 1.0
    return float(total**2 / (len(counts) * (counts**2).sum()))
