"""Deterministic offline data fixture for the test suite ONLY.

The experimental pipeline is real-data only (`AbbasABC/HFL-Dataset`); no
synthetic data feeds any config, sweep, or reported result. This module
exists so the test suite and CI can exercise the FL harnesses offline in
seconds, injected through the harnesses' ``data.source: prebuilt`` seam
(``cfg["data"]["prebuilt"] = build_synthetic_raw(...)``).

Moved out of ``uavbench.fl.dataset`` on 2026-07-14 so the shipped library
contains no synthetic-data generation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticTorchDataset(Dataset):
    """Minimal stand-in matching MultiModalDataset's item signature.

    Items: ``(img_tensor(3,128,128), struct(9,), label)`` so ``CachedDataset``
    wraps it unchanged.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.img_tensors = torch.zeros(len(features), 3, 128, 128)
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.img_tensors[idx], self.features[idx], self.labels[idx]


def build_synthetic_raw(N: int, K: int, seed: int = 42) -> dict:
    """Deterministic fake-client raw dict, same shape as the real pipeline.

    ``N`` building samples partitioned across ``K`` clients in the Noto
    Peninsula coordinate box, with the empirical class imbalance and random
    seismic/image features. Returned dict plugs directly into
    ``cfg["data"]["prebuilt"]``.
    """
    rng = np.random.default_rng(seed)

    raw_lat = rng.uniform(37.0, 37.8, size=N)
    raw_lon = rng.uniform(136.8, 137.5, size=N)

    lat_n = (raw_lat - 37.0) / 0.8
    lon_n = (raw_lon - 136.8) / 0.7
    seismic = rng.normal(0, 1, size=(N, 7)).astype(np.float32)
    features = np.column_stack([lat_n, lon_n, seismic]).astype(np.float32)

    labels = rng.choice([0, 1, 2, 3], size=N, p=[0.60, 0.20, 0.10, 0.10]).astype(np.int64)
    img_features = rng.standard_normal((N, 512)).astype(np.float32)

    client_coords: dict[int, tuple[float, float]] = {}
    client_train_indices: dict[int, list[int]] = {}
    client_test_indices: dict[int, list[int]] = {}

    all_idx = list(range(N))
    rng.shuffle(all_idx)
    chunk = N // K
    for k in range(K):
        start = k * chunk
        end = (k + 1) * chunk if k < K - 1 else N
        shard = all_idx[start:end]
        split = max(1, int(len(shard) * 0.8))
        client_train_indices[k] = shard[:split]
        client_test_indices[k] = shard[split:]
        client_coords[k] = (float(np.mean(raw_lat[shard])), float(np.mean(raw_lon[shard])))

    global_test = [i for sub in client_test_indices.values() for i in sub]

    return {
        "full_dataset": SyntheticTorchDataset(features, labels),
        "client_train_indices": client_train_indices,
        "client_test_indices": client_test_indices,
        "global_test_indices": global_test,
        "client_coords": client_coords,
        "img_features": img_features,
        "raw_lat": raw_lat,
        "raw_lon": raw_lon,
    }


def prebuilt_data_cfg(N: int, K: int, seed: int = 42, **extra) -> dict:
    """A ready-to-use ``cfg["data"]`` block injecting the fixture."""
    return {"source": "prebuilt", "prebuilt": build_synthetic_raw(N, K, seed), **extra}
