"""Dataset adapters for Tier-2: cached-feature dataset and synthetic fallback.

The ``CachedDataset`` wraps a ``MultiModalDataset`` and replaces the image
tensor with a row from the precomputed ResNet-18 feature cache, so the FL
training loop never touches the image backbone.

The experimental pipeline is real-data only. The test suite injects
deterministic offline fixtures through the harnesses' ``data.source:
prebuilt`` seam (see tests/uavbench/synthetic_fixture.py) — no synthetic
generation lives in the library.
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
    coords: tuple[float, float]  # (lat, lon) in degrees
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
