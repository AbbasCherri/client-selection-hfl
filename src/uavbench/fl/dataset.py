"""Dataset adapters for Tier-2: cached-feature dataset and synthetic fallback.

The ``CachedDataset`` wraps a ``MultiModalDataset`` and replaces the image
tensor with a row from the precomputed ResNet-18 feature cache, so the FL
training loop never touches the image backbone.

When ``data_source: synthetic`` is set in the config (no HF token needed),
``SyntheticClientData`` generates deterministic fake clients for offline CI
and smoke testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

logger = logging.getLogger("uavbench.fl.dataset")

# Expected number of structured features (must match hflsim FEATURE_COLS).
STRUCT_DIM = 9


@dataclass
class ClientData:
    """Everything the FL loop needs to know about one client."""

    client_id: int
    coords: tuple[float, float]          # (lat, lon) in degrees
    train_indices: list[int]
    test_indices: list[int]
    n_samples: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_samples = len(self.train_indices)


class CachedDataset(Dataset):
    """Wraps MultiModalDataset, swapping the image tensor for a cached feature vector.

    ``base[idx]`` → ``(img_tensor(3,128,128), struct(9,), label)``
    ``CachedDataset[idx]`` → ``(img_feat(512,), struct(9,), label)``
    """

    def __init__(self, base_dataset: Dataset, img_features: np.ndarray) -> None:
        self.base = base_dataset
        self.img_features = torch.from_numpy(img_features.astype(np.float32))

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, struct, label = self.base[idx]
        return self.img_features[idx], struct, label


def make_client_loader(
    dataset: CachedDataset,
    indices: list[int],
    batch_size: int = 16,
) -> DataLoader:
    """DataLoader for one client's shard with value-balanced sampling."""
    subset = Subset(dataset, indices)
    labels = [int(dataset.base.labels[i].item()) for i in indices]  # type: ignore[attr-defined]
    n_classes = 4
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    weights = [1.0 / (counts[l] + 1e-6) for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(subset, batch_size=min(batch_size, len(indices)), sampler=sampler)


# ---------------------------------------------------------------------------
# Synthetic data (offline / no HF token)
# ---------------------------------------------------------------------------


class SyntheticTorchDataset(Dataset):
    """Fake MultiModalDataset mimic for synthetic runs.

    Items: ``(img_tensor(3,128,128), struct(9,), label)`` — same signature as
    the real dataset so ``CachedDataset`` and ``compute_feature_cache`` work
    unchanged.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.img_tensors = torch.zeros(len(features), 3, 128, 128)
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.img_tensors[idx], self.features[idx], self.labels[idx]


@dataclass
class SyntheticClientData:
    """Fully synthetic dataset for offline Tier-2 testing (no HF token required).

    Generates N building samples across K clients in the Noto Peninsula coordinate
    range, with balanced-ish damage labels and random seismic features.
    """

    N: int
    K: int
    seed: int = 42

    def build(self) -> dict:
        """Return the same dict shape as the real data pipeline produces."""
        rng = np.random.default_rng(self.seed)
        N, K = self.N, self.K

        # Geographic coordinates: Noto Peninsula region.
        raw_lat = rng.uniform(37.0, 37.8, size=N)
        raw_lon = rng.uniform(136.8, 137.5, size=N)

        # Normalize lat/lon to [0,1]; generate 7 synthetic seismic features (z-scored).
        lat_n = (raw_lat - 37.0) / 0.8
        lon_n = (raw_lon - 136.8) / 0.7
        seismic = rng.normal(0, 1, size=(N, 7)).astype(np.float32)
        features = np.column_stack([lat_n, lon_n, seismic]).astype(np.float32)

        # Damage labels: heavily imbalanced toward Survived (class 0), reflecting reality.
        labels = rng.choice([0, 1, 2, 3], size=N, p=[0.60, 0.20, 0.10, 0.10]).astype(np.int64)

        # Pre-generated image features (skip full ResNet pass for synthetic mode).
        img_features = rng.standard_normal((N, 512)).astype(np.float32)

        # Simple even client partition.
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
            # Centroid of the shard's samples, not just the first one — a UAV
            # covering this point should plausibly cover the client's data.
            client_coords[k] = (
                float(np.mean(raw_lat[shard])), float(np.mean(raw_lon[shard])),
            )

        global_test = [i for sub in client_test_indices.values() for i in sub]

        torch_dataset = SyntheticTorchDataset(features, labels)
        return {
            "full_dataset": torch_dataset,
            "client_train_indices": client_train_indices,
            "client_test_indices": client_test_indices,
            "global_test_indices": global_test,
            "client_coords": client_coords,
            "img_features": img_features,      # skip feature cache for synthetic mode
            "raw_lat": raw_lat,
            "raw_lon": raw_lon,
        }
