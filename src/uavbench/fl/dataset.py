"""Dataset adapters for Tier-2: cached-feature dataset and shard loaders.

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
from torch.utils.data import Dataset

logger = logging.getLogger("uavbench.fl.dataset")

# Expected number of structured features (must match hflsim FEATURE_COLS).
STRUCT_DIM = 9

# Damage classes 0-3 (see uavbench.metrics.fl.CLASS_NAMES).
N_CLASSES = 4


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
        if len(img_features) != len(base_dataset):  # type: ignore[arg-type]
            # A mismatched feature cache indexes out of bounds mid-epoch (hours
            # into a sweep). Refuse it here, at construction, with the cause.
            raise ValueError(
                f"img_features has {len(img_features)} rows but the dataset has "
                f"{len(base_dataset)} samples. The feature cache was built for a "  # type: ignore[arg-type]
                "different data configuration (e.g. data.subsample changed) — "
                "delete the stale img_features.npy or let compute_feature_cache "
                "rebuild it."
            )
        self.base = base_dataset
        self.img_features = torch.from_numpy(img_features.astype(np.float32))
        # In-memory views used for tensor-sliced batching (both the real
        # MultiModalDataset and the prebuilt test fixture store these as
        # torch tensors).
        self.struct_features: torch.Tensor = self.base.features  # type: ignore[attr-defined]
        self.labels: torch.Tensor = self.base.labels  # type: ignore[attr-defined]

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Never call base.__getitem__ here: it decodes the raw image chip from
        # disk, which this wrapper immediately discards. Struct features and
        # labels are already in-memory tensors on the base dataset.
        return self.img_features[idx], self.struct_features[idx], self.labels[idx]

    def eval_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The full (img_features, struct_features, labels) tensors for
        vectorized evaluation (see metrics.fl.evaluate_subset fast path)."""
        return self.img_features, self.struct_features, self.labels


class BalancedShardLoader:
    """Class-balanced batch iterator over one client/UAV shard.

    Sampling semantics match the DataLoader + WeightedRandomSampler it
    replaces: each epoch draws ``len(indices)`` samples with replacement,
    weighted by inverse class frequency within the shard, from the global
    torch RNG (deterministic under ``torch.manual_seed``). Batches are built
    by slicing the in-memory feature tensors instead of per-item
    ``__getitem__`` + collate, which dominated training wall-time.
    """

    def __init__(self, dataset: CachedDataset, indices: list[int], batch_size: int) -> None:
        if not indices:
            raise ValueError("BalancedShardLoader needs a non-empty shard")
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        shard_labels = dataset.labels[self.indices].to(torch.long)
        counts = torch.bincount(shard_labels, minlength=N_CLASSES).to(torch.float64)
        self.weights = 1.0 / (counts[shard_labels] + 1e-6)
        self.batch_size = min(batch_size, len(indices))
        self._img = dataset.img_features
        self._struct = dataset.struct_features
        self._labels = dataset.labels

    def __len__(self) -> int:
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = torch.multinomial(self.weights, len(self.indices), replacement=True)
        sel = self.indices[order]
        for start in range(0, len(sel), self.batch_size):
            b = sel[start : start + self.batch_size]
            yield self._img[b], self._struct[b], self._labels[b]


def make_client_loader(
    dataset: CachedDataset,
    indices: list[int],
    batch_size: int = 16,
) -> BalancedShardLoader:
    """Loader for one client's shard with value-balanced sampling."""
    return BalancedShardLoader(dataset, indices, batch_size)
