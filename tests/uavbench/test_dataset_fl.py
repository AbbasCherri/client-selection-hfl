"""Tests for CachedDataset, SyntheticClientData, and make_client_loader (dataset.py)."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from uavbench.fl.dataset import (
    CachedDataset,
    ClientData,
    SyntheticClientData,
    SyntheticTorchDataset,
    make_client_loader,
)


# ── SyntheticTorchDataset ─────────────────────────────────────────────────────

class TestSyntheticTorchDataset:
    def _make(self, n=20):
        feats = np.random.randn(n, 9).astype(np.float32)
        labels = np.array([i % 4 for i in range(n)], dtype=np.int64)
        return SyntheticTorchDataset(feats, labels)

    def test_len(self):
        ds = self._make(15)
        assert len(ds) == 15

    def test_getitem_shapes(self):
        ds = self._make()
        img, feat, label = ds[0]
        assert img.shape == (3, 128, 128)
        assert feat.shape == (9,)
        assert label.shape == ()

    def test_labels_attribute_exists(self):
        ds = self._make()
        assert hasattr(ds, "labels")
        assert len(ds.labels) == 20

    def test_img_tensor_is_zeros(self):
        ds = self._make()
        img, _, _ = ds[0]
        assert torch.all(img == 0.0)


# ── CachedDataset ─────────────────────────────────────────────────────────────

class TestCachedDataset:
    def _make(self, n=20, feat_dim=512):
        feats = np.random.randn(n, 9).astype(np.float32)
        labels = np.array([i % 4 for i in range(n)], dtype=np.int64)
        base = SyntheticTorchDataset(feats, labels)
        img_feats = np.random.randn(n, feat_dim).astype(np.float32)
        return CachedDataset(base, img_feats), img_feats

    def test_len_matches_base(self):
        ds, _ = self._make(10)
        assert len(ds) == 10

    def test_getitem_returns_feature_not_image(self):
        ds, img_feats = self._make(5, feat_dim=512)
        img_feat, struct, label = ds[0]
        assert img_feat.shape == (512,)
        # Must be the cached feature, not a raw image
        assert torch.allclose(img_feat, torch.from_numpy(img_feats[0]))

    def test_getitem_struct_shape(self):
        ds, _ = self._make()
        _, struct, _ = ds[0]
        assert struct.shape == (9,)

    def test_getitem_label_is_valid_class(self):
        ds, _ = self._make()
        _, _, label = ds[0]
        assert int(label.item()) in [0, 1, 2, 3]

    def test_img_features_stored_as_float32(self):
        feats = np.random.randn(5, 9).astype(np.float32)
        labels = np.zeros(5, dtype=np.int64)
        base = SyntheticTorchDataset(feats, labels)
        # Pass float16 cache — should be converted to float32
        img_feats_f16 = np.random.randn(5, 512).astype(np.float16)
        ds = CachedDataset(base, img_feats_f16)
        img_feat, _, _ = ds[0]
        assert img_feat.dtype == torch.float32


# ── make_client_loader ────────────────────────────────────────────────────────

class TestMakeClientLoader:
    def _make_ds(self, n=40):
        feats = np.random.randn(n, 9).astype(np.float32)
        labels = np.array([i % 4 for i in range(n)], dtype=np.int64)
        base = SyntheticTorchDataset(feats, labels)
        img_feats = np.random.randn(n, 512).astype(np.float32)
        return CachedDataset(base, img_feats)

    def test_returns_dataloader(self):
        ds = self._make_ds()
        loader = make_client_loader(ds, list(range(20)), batch_size=8)
        assert isinstance(loader, DataLoader)

    def test_batch_size_capped_at_subset_size(self):
        ds = self._make_ds()
        small_indices = [0, 1, 2]
        loader = make_client_loader(ds, small_indices, batch_size=64)
        # batch_size = min(64, 3) = 3
        assert loader.batch_size == 3

    def test_yields_correct_tensor_shapes(self):
        ds = self._make_ds(40)
        loader = make_client_loader(ds, list(range(16)), batch_size=4)
        img_feat, struct, label = next(iter(loader))
        assert img_feat.shape == (4, 512)
        assert struct.shape == (4, 9)
        assert label.shape == (4,)

    def test_labels_in_valid_range(self):
        ds = self._make_ds(40)
        loader = make_client_loader(ds, list(range(40)), batch_size=16)
        for _, _, labels in loader:
            assert torch.all(labels >= 0) and torch.all(labels <= 3)


# ── SyntheticClientData ───────────────────────────────────────────────────────

class TestSyntheticClientData:
    def _build(self, N=50, K=5, seed=42):
        return SyntheticClientData(N=N, K=K, seed=seed).build()

    def test_required_keys_present(self):
        raw = self._build()
        for key in ("full_dataset", "client_train_indices", "client_test_indices",
                    "global_test_indices", "client_coords", "img_features"):
            assert key in raw, f"Missing key: {key}"

    def test_client_coords_count_matches_K(self):
        raw = self._build(N=50, K=5)
        assert len(raw["client_coords"]) == 5

    def test_all_train_indices_valid(self):
        raw = self._build(N=50, K=5)
        for cid, indices in raw["client_train_indices"].items():
            for idx in indices:
                assert 0 <= idx < 50

    def test_no_train_test_overlap_per_client(self):
        raw = self._build(N=50, K=5)
        for cid in raw["client_train_indices"]:
            train = set(raw["client_train_indices"][cid])
            test = set(raw["client_test_indices"][cid])
            assert train.isdisjoint(test), f"Client {cid} has train/test overlap"

    def test_global_test_covers_all_client_test(self):
        raw = self._build(N=50, K=5)
        all_test = set(raw["global_test_indices"])
        for cid, indices in raw["client_test_indices"].items():
            for idx in indices:
                assert idx in all_test

    def test_img_features_shape(self):
        raw = self._build(N=50)
        assert raw["img_features"].shape == (50, 512)

    def test_deterministic_with_same_seed(self):
        r1 = self._build(seed=7)
        r2 = self._build(seed=7)
        assert np.array_equal(r1["img_features"], r2["img_features"])
        assert r1["client_train_indices"] == r2["client_train_indices"]

    def test_different_seeds_differ(self):
        r1 = self._build(seed=0)
        r2 = self._build(seed=1)
        assert not np.array_equal(r1["img_features"], r2["img_features"])

    def test_lat_lon_in_noto_range(self):
        raw = self._build(N=100, K=5)
        for cid, (lat, lon) in raw["client_coords"].items():
            assert 37.0 <= lat <= 37.8, f"Lat {lat} out of Noto range"
            assert 136.8 <= lon <= 137.5, f"Lon {lon} out of Noto range"

    def test_full_dataset_labels_in_0123(self):
        raw = self._build(N=30)
        ds = raw["full_dataset"]
        for i in range(len(ds)):
            _, _, label = ds[i]
            assert int(label.item()) in [0, 1, 2, 3]


# ── ClientData ────────────────────────────────────────────────────────────────

class TestClientData:
    def test_n_samples_set_from_train_indices(self):
        c = ClientData(client_id=0, coords=(37.0, 137.0),
                       train_indices=[0, 1, 2, 3], test_indices=[])
        assert c.n_samples == 4

    def test_empty_train_indices(self):
        c = ClientData(client_id=0, coords=(37.0, 137.0),
                       train_indices=[], test_indices=[])
        assert c.n_samples == 0
