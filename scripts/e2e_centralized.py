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

Three things were wrong with this script through 2026-07 and are fixed here:
  1. the reported epoch was chosen by ``max(test macro-F1)`` — model selection
     on the reported set. Epochs are now selected on a validation split;
  2. macro-F1 came from the best epoch while accuracy came from the *last* one,
     so the two reported numbers described different models;
  3. it ran a single seed, which cannot support an interval on the retention
     ratio, and both arms shared one learning rate (1e-3) while the paper's
     frozen recipe is 1.775e-2 — neither arm was at its own optimum.

Usage (on the VM, .venv active, HF_TOKEN exported):
    python scripts/e2e_centralized.py --seeds 0 1 2 3 4 5 6 7 8 9 \
        --subsample 1.0 --n 200 --epochs 15 --out results/e2e_centralized
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


def _train_e2e(
    model, dataset, idx, val_idx, test_idx, epochs, lr, batch_size, loss_fn, opt_cfg,
    patience: int = 3,
):
    """Train end-to-end, selecting the epoch on VALIDATION and reporting on test.

    Through 2026-07 this took ``max(test macro-F1)`` over all epochs and paired
    it with the *last* epoch's accuracy — model selection on the reported set,
    and two metrics from two different epochs. Both are fixed here: the epoch is
    chosen by val macro-F1, and the returned metrics all come from that one
    epoch's test evaluation.

    Early stopping on val also roughly halves the epochs actually run, which is
    where most of this script's cost sits (it is the only image-gradient pass in
    the project).
    """
    loader = DataLoader(Subset(dataset, idx), batch_size=batch_size, shuffle=True)
    opt = _make_optimizer([p for p in model.parameters() if p.requires_grad], opt_cfg, lr)
    best_val, best_test, best_epoch, since_improved = -1.0, None, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        for img, struct, label in loader:
            opt.zero_grad()
            loss_fn(model(img, struct), label).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = _eval_raw(model, dataset, val_idx)
        t = _eval_raw(model, dataset, test_idx)
        logger.info(
            "[e2e] epoch %d/%d  val-F1=%.3f  test-F1=%.3f  test-acc=%.3f",
            ep, epochs, v["macro_f1"], t["macro_f1"], t["accuracy"],
        )
        if v["macro_f1"] > best_val:
            best_val, best_test, best_epoch, since_improved = v["macro_f1"], t, ep, 0
        else:
            since_improved += 1
            if since_improved >= patience:
                logger.info("[e2e] early stop at epoch %d (best val epoch %d)", ep, best_epoch)
                break
    return best_val, best_test, best_epoch


def _run_one_seed(seed: int, args, out: Path) -> dict:
    """One paired (frozen, end-to-end) comparison on identical data."""
    torch.manual_seed(seed)
    cfg = {
        "data": {"source": "real", "subsample": args.subsample, "seed": args.seed_data,
                 "partition_seed": seed, "val_ratio": 0.1,
                 "data_dir": "./data", "feature_batch_size": 32, "N_clients": args.n},
        "fl": {"K": 1, "R_comm": 20000.0, "capacity": args.n},
    }
    full_dataset, client_train, test_idx, _coords, img_features, val_idx = _load_data(cfg, out)
    if not val_idx:
        raise RuntimeError(
            "no validation split — epoch selection would fall back to the test "
            "set, which is the bug this script was rewritten to fix"
        )
    cached = CachedDataset(full_dataset, img_features)
    all_train = [i for cid in client_train for i in client_train[cid]]
    opt_cfg = {"name": "sgd", "momentum": 0.9}
    loss_fn = make_loss_fn(_log_prior(cached.labels, all_train, args.logit_tau), tau=args.logit_tau)

    from uavbench.fl.model import CachedFusionModel

    # Per-arm LR chosen on validation. A linear head on cached features and a
    # trainable ResNet-18 backbone do not want the same step size; through
    # 2026-07 both ran at 1e-3 while the paper's frozen recipe is 1.775e-2, so
    # neither arm was at its own optimum and the retention ratio compared two
    # mistuned models.
    best = {}
    for lr in args.lr_grid:
        frozen = CachedFusionModel()
        frozen.unfreeze_img_proj()
        rows, _ = _run_centralized(frozen, cached, all_train, val_idx, args.epochs, 1,
                                   lr, args.batch_size, loss_fn, opt_cfg, balanced=False)
        val_f1 = max(r["macro_f1"] for r in rows)  # val, so selecting on it is legitimate
        if val_f1 > best.get("val", -1.0):
            best = {"val": val_f1, "lr": lr}
    frozen_lr = best["lr"]

    # Re-fit at the chosen LR, then pick the epoch on val and report on test.
    frozen = CachedFusionModel()
    frozen.unfreeze_img_proj()
    val_rows, _ = _run_centralized(frozen, cached, all_train, val_idx, args.epochs, 1,
                                   frozen_lr, args.batch_size, loss_fn, opt_cfg, balanced=False)
    best_ep = int(np.argmax([r["macro_f1"] for r in val_rows]))
    frozen2 = CachedFusionModel()
    frozen2.unfreeze_img_proj()
    test_rows, _ = _run_centralized(frozen2, cached, all_train, test_idx, best_ep + 1, 1,
                                    frozen_lr, args.batch_size, loss_fn, opt_cfg, balanced=False)
    # Both metrics from the SAME epoch — the 2026-07 version paired best-epoch
    # macro-F1 with last-epoch accuracy.
    frozen_f1 = test_rows[-1]["macro_f1"]
    frozen_acc = test_rows[-1]["accuracy"]

    e2e = EndToEndFusionModel()
    _e2e_val, e2e_test, e2e_ep = _train_e2e(
        e2e, full_dataset, all_train, val_idx, test_idx, args.epochs,
        args.e2e_lr, args.batch_size, loss_fn, opt_cfg,
    )

    retention = 100.0 * frozen_f1 / e2e_test["macro_f1"] if e2e_test["macro_f1"] > 0 else float("nan")
    logger.info(
        "[seed %d] frozen F1=%.3f (lr=%g, ep=%d) | e2e F1=%.3f (ep=%d) | retention %.1f%%",
        seed, frozen_f1, frozen_lr, best_ep + 1, e2e_test["macro_f1"], e2e_ep, retention,
    )
    return {
        "seed": seed, "frozen_lr": frozen_lr,
        "frozen_macro_f1": frozen_f1, "frozen_accuracy": frozen_acc,
        "e2e_macro_f1": e2e_test["macro_f1"], "e2e_accuracy": e2e_test["accuracy"],
        "retention_pct": retention,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=float, default=1.0,
                    help="paper regime by default; retention may depend on data volume")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                    help="partition seeds; n>=10 makes the retention ratio testable")
    ap.add_argument("--seed-data", type=int, default=42, dest="seed_data",
                    help="row-subsample seed, held fixed so the feature cache is shared")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr-grid", type=float, nargs="+", default=[1e-3, 5e-3, 1.775e-2],
                    dest="lr_grid", help="frozen-arm LR candidates, selected on val")
    ap.add_argument("--e2e-lr", type=float, default=1e-4, dest="e2e_lr",
                    help="trainable-backbone LR (fine-tuning wants a smaller step)")
    ap.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    ap.add_argument("--logit-tau", type=float, default=0.601, dest="logit_tau")
    ap.add_argument("--out", default="results/e2e_centralized")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [_run_one_seed(s, args, out) for s in args.seeds]
    df = pd.DataFrame(rows)
    df.to_csv(out / "e2e_per_seed.csv", index=False)

    # Interval on the per-seed retention: n=1 (the 2026-07 default) could not
    # support any interval, let alone a significance claim. Reuses the same
    # percentile bootstrap the significance tables use, so the CI here means
    # exactly what it means everywhere else in the paper.
    from uavbench.analysis.significance import _bootstrap_ci

    ci = _bootstrap_ci(df["retention_pct"].to_numpy())
    summary = pd.DataFrame([{
        "frozen_macro_f1_mean": df["frozen_macro_f1"].mean(),
        "e2e_macro_f1_mean": df["e2e_macro_f1"].mean(),
        "retention_pct_mean": df["retention_pct"].mean(),
        "retention_ci_lo": ci[0], "retention_ci_hi": ci[1],
        "n_seeds": len(df),
    }])
    summary.to_csv(out / "e2e_comparison.csv", index=False)
    logger.info(
        "Frozen features retain %.1f%% (95%% CI [%.1f, %.1f]) of end-to-end macro-F1 "
        "over %d seeds → %s",
        df["retention_pct"].mean(), ci[0], ci[1], len(df), out / "e2e_comparison.csv",
    )


if __name__ == "__main__":
    main()
