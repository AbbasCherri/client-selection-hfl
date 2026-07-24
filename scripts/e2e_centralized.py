#!/usr/bin/env python
"""Centralized end-to-end validation of the frozen ResNet-18 feature simplification.

The FL pipeline trains on cached, frozen ResNet-18 features (no image
forward/backward in the loop). A reviewer will ask whether that simplification
throws away signal. This script answers it directly: on the SAME pooled training
set and held-out test set, it trains

  (a) the frozen-feature model (CachedFusionModel on the cache) — the oracle used
      as `centralized` in paper_full, and
  (b) the same head with a *trainable* ResNet-18 backbone on raw images
      (EndToEndFusionModel),

for the same number of epochs and the same logit-adjusted loss, and reports
"frozen features retain X% of the end-to-end macro-F1". A retention near 100%
justifies the cached-feature design.

Usage (on the VM, .venv active, HF_TOKEN exported):
    python scripts/e2e_centralized.py --subsample 0.2 --n 200 --epochs 15 \
        --out results/e2e_centralized
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from uavbench.fl.dataset import CachedDataset
from uavbench.fl.federated import _load_data, _make_optimizer, _run_centralized
from uavbench.fl.model import EndToEndFusionModel, make_loss_fn
from uavbench.metrics.fl import _classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("e2e")


def _log_prior(labels: torch.Tensor, idx: list[int], tau: float):
    counts = torch.bincount(labels[torch.as_tensor(idx)].long(), minlength=4).double()
    prior = counts / counts.sum().clamp_min(1.0)
    return torch.log(prior.clamp_min(1e-8)).float() if tau > 0 else None


def _eval_raw(model, dataset, indices: list[int], batch_size: int = 128) -> dict:
    """Evaluate a raw-image model over a subset of the base dataset."""
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    model.eval()
    ys, ps = [], []
    with torch.inference_mode():
        for img, struct, label in loader:
            preds = model(img, struct).argmax(1)
            ys.append(label.numpy())
            ps.append(preds.numpy())
    return _classification_metrics(np.concatenate(ys), np.concatenate(ps))


def _train_e2e(model, dataset, idx, test_idx, epochs, lr, batch_size, loss_fn, opt_cfg):
    loader = DataLoader(Subset(dataset, idx), batch_size=batch_size, shuffle=True)
    opt = _make_optimizer([p for p in model.parameters() if p.requires_grad], opt_cfg, lr)
    best = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        for img, struct, label in loader:
            opt.zero_grad()
            loss_fn(model(img, struct), label).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        m = _eval_raw(model, dataset, test_idx)
        best = max(best, m["macro_f1"])
        logger.info("[e2e] epoch %d/%d  acc=%.3f  macro-F1=%.3f", ep, epochs, m["accuracy"], m["macro_f1"])
    return best, m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    ap.add_argument("--logit-tau", type=float, default=0.601, dest="logit_tau")
    ap.add_argument("--out", default="results/e2e_centralized")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    cfg = {
        "data": {"source": "real", "subsample": args.subsample, "seed": args.seed,
                 "data_dir": "./data", "feature_batch_size": 32, "N_clients": args.n},
        "fl": {"K": 1, "R_comm": 20000.0, "capacity": args.n},
    }
    full_dataset, client_train, test_idx, _coords, img_features = _load_data(cfg, out)
    cached = CachedDataset(full_dataset, img_features)
    all_train = [i for cid in client_train for i in client_train[cid]]
    opt_cfg = {"name": "sgd", "momentum": 0.9}
    loss_fn = make_loss_fn(_log_prior(cached.labels, all_train, args.logit_tau), tau=args.logit_tau)

    # (a) frozen-feature centralized oracle (reuses the exact paper recipe)
    from uavbench.fl.model import CachedFusionModel

    frozen = CachedFusionModel()
    frozen.unfreeze_img_proj()
    rows, _ = _run_centralized(frozen, cached, all_train, test_idx, args.epochs, 1,
                               args.lr, args.batch_size, loss_fn, opt_cfg, balanced=False)
    frozen_f1 = max(r["macro_f1"] for r in rows)
    frozen_acc = rows[-1]["accuracy"]
    logger.info("[frozen] best macro-F1=%.3f", frozen_f1)

    # (b) trainable-backbone end-to-end
    e2e = EndToEndFusionModel()
    e2e_f1, e2e_last = _train_e2e(e2e, full_dataset, all_train, test_idx, args.epochs,
                                  args.lr, args.batch_size, loss_fn, opt_cfg)

    retention = 100.0 * frozen_f1 / e2e_f1 if e2e_f1 > 0 else float("nan")
    df = pd.DataFrame([
        {"model": "frozen_features", "macro_f1": frozen_f1, "accuracy": frozen_acc},
        {"model": "end_to_end", "macro_f1": e2e_f1, "accuracy": e2e_last["accuracy"]},
        {"model": "frozen_retention_pct", "macro_f1": retention, "accuracy": np.nan},
    ])
    df.to_csv(out / "e2e_comparison.csv", index=False)
    logger.info("Frozen features retain %.1f%% of end-to-end macro-F1 "
                "(frozen %.3f vs end-to-end %.3f) → %s", retention, frozen_f1, e2e_f1,
                out / "e2e_comparison.csv")


if __name__ == "__main__":
    main()
