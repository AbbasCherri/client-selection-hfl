"""Data pipeline: label remap, split pooling, balanced train sampling,
black-chip diagnostic + hard gate, stress knobs."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
import torch
from _fixture import SyntheticTorchDataset, build_synthetic_raw
from _lib import check, finish
from PIL import Image

from hflsim.data.loader import FEATURE_COLS, MultiModalDataset, _remap_damage_labels
from uavbench.fl.dataset import CachedDataset, make_client_loader
from uavbench.fl.federated import _apply_black_chips, _report_black_chip_rate


def label_remap_keeps_all_four_classes():
    # Raw damage classes are {0, 1, 9, 99}; an isin([0..3]) filter would drop
    # ~16% of rows and collapse the task to 2 classes (the historical bug).
    df = pd.DataFrame({"damage_val": [0, 1, 9, 99, 0, 9, 99, 1]})
    out = _remap_damage_labels(df)
    assert len(out) == 8, "remap must not drop rows"
    assert set(out["damage_val"].unique()) == {0, 1, 2, 3}
    # 9 -> obstructed (2), 99 -> missing (3)
    assert out["damage_val"].tolist() == [0, 1, 2, 3, 0, 2, 3, 1]


def splits_disjoint_and_pooled():
    raw = build_synthetic_raw(N=50, K=5, seed=42)
    all_test = set(raw["global_test_indices"])
    for cid in raw["client_train_indices"]:
        train = set(raw["client_train_indices"][cid])
        test = set(raw["client_test_indices"][cid])
        assert train.isdisjoint(test), f"client {cid} train/test overlap"
        assert test <= all_test  # pooled global test covers every client's test
    assert len(raw["client_coords"]) == 5


def cached_dataset_passthrough():
    feats = np.random.randn(5, 9).astype(np.float32)
    labels = np.zeros(5, dtype=np.int64)
    img_feats = np.random.randn(5, 512).astype(np.float16)  # f16 cache on disk
    ds = CachedDataset(SyntheticTorchDataset(feats, labels), img_feats)
    img_feat, struct, label = ds[0]
    assert img_feat.dtype == torch.float32 and img_feat.shape == (512,)
    assert torch.allclose(img_feat, torch.from_numpy(img_feats[0].astype(np.float32)))
    assert struct.shape == (9,)


def train_loader_balanced_eval_loader_not():
    # Inverse-class-frequency sampling on the *training* loader only;
    # evaluation must see the true imbalanced distribution or macro-F1 lies.
    raw = build_synthetic_raw(N=60, K=3, seed=1)
    ds = CachedDataset(raw["full_dataset"], raw["img_features"])
    indices = list(range(30))
    loader = make_client_loader(ds, indices, batch_size=8)
    # Per-sample weights are exactly 1/(class count within the shard).
    shard_labels = ds.labels[torch.as_tensor(indices)].to(torch.long)
    counts = torch.bincount(shard_labels, minlength=4).to(torch.float64)
    assert torch.allclose(loader.weights, 1.0 / (counts[shard_labels] + 1e-6))
    assert loader.batch_size == 8
    # batch size capped at shard size
    assert make_client_loader(ds, [0, 1, 2], batch_size=64).batch_size == 3
    # One epoch draws exactly len(indices) samples, deterministically under
    # torch.manual_seed, re-drawn per epoch (two iterations differ).
    torch.manual_seed(0)
    epoch1 = [labels for _, _, labels in loader]
    epoch2 = [labels for _, _, labels in loader]
    torch.manual_seed(0)
    epoch1_again = [labels for _, _, labels in loader]
    assert sum(len(b) for b in epoch1) == 30
    assert all(torch.equal(a, b) for a, b in zip(epoch1, epoch1_again))
    assert not all(torch.equal(a, b) for a, b in zip(epoch1, epoch2))
    # The evaluation path is sequential and unweighted: fast tensor path and
    # DataLoader fallback must agree exactly, on the true distribution.
    from torch.utils.data import DataLoader, Subset

    from uavbench.fl.model import CachedFusionModel
    from uavbench.metrics.fl import evaluate_loader, evaluate_subset

    torch.manual_seed(3)
    model = CachedFusionModel()
    test_idx = list(range(30, 60))
    fast = evaluate_subset(model, ds, test_idx)
    slow = evaluate_loader(model, DataLoader(Subset(ds, test_idx), batch_size=64, shuffle=False))
    assert fast["accuracy"] == slow["accuracy"] and fast["macro_f1"] == slow["macro_f1"]
    assert np.array_equal(fast["confusion_matrix"], slow["confusion_matrix"])
    # The confusion matrix reflects the shard's true label counts (no resampling).
    true_counts = np.bincount(ds.labels[torch.as_tensor(test_idx)].numpy(), minlength=4)
    assert np.array_equal(fast["confusion_matrix"].sum(axis=1), true_counts)


def mismatched_feature_cache_fails_fast():
    # Regression for the 2026-07-16 crash: a feature cache built for a
    # different subsample must be rejected at construction with an actionable
    # error, not fail with IndexError hours into a training epoch.
    feats = np.random.randn(10, 9).astype(np.float32)
    labels = np.zeros(10, dtype=np.int64)
    stale_cache = np.random.randn(4, 512).astype(np.float16)  # wrong row count
    try:
        CachedDataset(SyntheticTorchDataset(feats, labels), stale_cache)
    except ValueError as e:
        assert "stale" in str(e) or "subsample" in str(e)
    else:
        raise AssertionError("mismatched img_features must raise at construction")


def stale_feature_cache_recomputed():
    # compute_feature_cache must detect a cache whose row count disagrees with
    # the dataset and rebuild it, and must NOT touch the backbone when the
    # cache is valid.
    import tempfile

    import torch.nn as nn

    from uavbench.fl import features as F

    class TinyBackbone(nn.Module):
        def forward(self, x):  # (B, 3, H, W) -> (B, 512), deterministic
            return x.mean(dim=(2, 3)).repeat(1, 171)[:, :512]

    ds = SyntheticTorchDataset(np.random.randn(7, 9).astype(np.float32), np.zeros(7, np.int64))
    orig = F._frozen_resnet18
    try:
        with tempfile.TemporaryDirectory() as d:
            cache = str(Path(d) / "img_features.npy")
            np.save(cache, np.zeros((3, 512), dtype=np.float16))  # stale: 3 != 7 rows
            F._frozen_resnet18 = TinyBackbone
            arr = F.compute_feature_cache(ds, cache, batch_size=4)
            assert arr.shape == (7, 512), "stale cache must be rebuilt to dataset size"
            assert np.load(cache).shape == (7, 512), "rebuilt cache must be persisted"

            def _boom():
                raise AssertionError("valid cache must be loaded without a backbone pass")

            F._frozen_resnet18 = _boom
            arr2 = F.compute_feature_cache(ds, cache, batch_size=4)
            assert arr2.shape == (7, 512)
    finally:
        F._frozen_resnet18 = orig


def black_chip_rate_exact():
    def df_for(chip_paths):
        n = len(chip_paths)
        data = {col: np.linspace(0.1, 0.9, n) for col in FEATURE_COLS}
        data["damage_val"] = np.zeros(n, dtype=np.int64)
        data["chip_path"] = chip_paths
        data["latitude"] = np.full(n, 0.0)  # outside GSI box
        data["longitude"] = np.full(n, 0.0)
        return pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as d:
        ds = MultiModalDataset(df_for(["missing.jpg"]), data_dir=d, use_gsi=False)
        assert ds.black_chip_rate() == 0.0  # zero-loads guard
        good = []
        for i in range(3):
            p = str(Path(d) / f"chip{i}.jpg")
            Image.new("RGB", (8, 8), (120, 60, 30)).save(p)
            good.append(p)
        ds = MultiModalDataset(df_for(good + ["m1.jpg", "m2.jpg"]), data_dir=d, use_gsi=False)
        for i in range(5):
            ds[i]
        assert ds.total_image_loads == 5 and ds.black_chip_count == 2
        assert abs(ds.black_chip_rate() - 0.4) < 1e-12


def black_chip_hard_gate():
    # Above data.max_black_chip_rate the run must CRASH, not report a
    # majority-collapse accuracy that looks valid. The measured rate is
    # persisted to _diagnostics before the raise so the evidence survives.
    class FakeDs:
        def black_chip_rate(self):
            return 0.9

    cfg: dict = {}
    try:
        _report_black_chip_rate(FakeDs(), cfg)
    except RuntimeError as e:
        assert "black-chip" in str(e).lower() or "Black-chip" in str(e)
        assert cfg["_diagnostics"]["black_chip_rate"] == 0.9
    else:
        raise AssertionError("90% black-chip rate must abort the run")

    class OkDs:
        def black_chip_rate(self):
            return 0.1

    cfg2: dict = {}
    _report_black_chip_rate(OkDs(), cfg2)  # low rate: no raise
    assert cfg2["_diagnostics"]["black_chip_rate"] == 0.1
    # Explicit stress-run override is honoured.
    cfg3: dict = {"data": {"max_black_chip_rate": 0.95}}
    _report_black_chip_rate(FakeDs(), cfg3)
    # Datasets without the counter (prebuilt fixtures): no-op.
    cfg4: dict = {}
    _report_black_chip_rate(object(), cfg4)
    assert cfg4 == {}


def stress_black_chip_knob():
    img = np.random.default_rng(0).standard_normal((100, 512)).astype(np.float32)
    assert _apply_black_chips(img, 0.0, seed=1) is img  # rate 0: untouched
    out = _apply_black_chips(img, 0.25, seed=1)
    n_black = int((out == 0).all(axis=1).sum())
    assert n_black == 25  # exact fraction
    assert not (img == 0).all(axis=1).any()  # input not mutated
    out2 = _apply_black_chips(img, 0.25, seed=1)
    assert np.array_equal(out, out2)  # deterministic given seed


check("label remap keeps all four classes {0,1,9,99}->{0,1,2,3}", label_remap_keeps_all_four_classes)
check("per-client train/test disjoint; global test set is the pool", splits_disjoint_and_pooled)
check("CachedDataset serves f32 cached features, not raw images", cached_dataset_passthrough)
check("weighted sampling on train loader ONLY; eval sees true distribution", train_loader_balanced_eval_loader_not)
check("mismatched feature cache fails fast at CachedDataset construction", mismatched_feature_cache_fails_fast)
check("stale feature cache detected and rebuilt; valid cache loaded as-is", stale_feature_cache_recomputed)
check("black-chip rate counted exactly on a known fixture", black_chip_rate_exact)
check("black-chip hard gate: crash above threshold, diagnostic persisted", black_chip_hard_gate)
check("stress black-chip knob: exact, pure, deterministic", stress_black_chip_knob)
finish()
