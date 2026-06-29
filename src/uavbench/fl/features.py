"""Frozen ResNet-18 vision feature cache.

Precomputed once over the full dataset and stored as float16 .npy. Per-round
FL training loads only this cache, so there are no image forward passes during
the FL loop — the dominant CPU-feasibility measure for Tier-2.

Cache size: N samples × 512 dims × 2 bytes = ~5 MB for N=5000. Well within
the 30 GB disk budget.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

logger = logging.getLogger("uavbench.fl.features")

FEAT_DIM = 512
_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _frozen_resnet18() -> nn.Module:
    """Return a pretrained ResNet-18 with the classification head replaced by Identity."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def compute_feature_cache(
    dataset,
    cache_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    force: bool = False,
) -> np.ndarray:
    """Compute (or load) float32 ResNet-18 image features for every sample.

    Parameters
    ----------
    dataset:
        A ``MultiModalDataset`` (or any Dataset whose __getitem__ returns
        ``(img_tensor(3,H,W), struct_feat, label)``).
    cache_path:
        ``.npy`` file to save/load float16 features.
    batch_size:
        Batch size for the one-time forward pass. 32 is safe on 30 GB RAM.
    num_workers:
        DataLoader workers. 0 = main process (safer on GCP VMs).
    force:
        Recompute even if the cache file already exists.

    Returns
    -------
    np.ndarray
        ``(N, 512)`` float32 feature array ready to index by sample position.
    """
    if os.path.exists(cache_path) and not force:
        logger.info("Loading cached ResNet-18 features from %s", cache_path)
        arr = np.load(cache_path)
        return arr.astype(np.float32)

    logger.info(
        "Computing ResNet-18 features for %d samples (one-time pass, CPU)…", len(dataset)
    )
    backbone = _frozen_resnet18()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for imgs, _, _ in loader:
            imgs = (imgs - _IMG_MEAN) / _IMG_STD
            feats = backbone(imgs)  # (B, 512)
            chunks.append(feats.numpy().astype(np.float16))

    features_f16 = np.vstack(chunks)  # (N, 512) float16

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.save(cache_path, features_f16)
    size_mb = os.path.getsize(cache_path) / 1e6
    logger.info("Feature cache saved: %.2f MB → %s", size_mb, cache_path)
    return features_f16.astype(np.float32)


def synthetic_feature_cache(N: int, seed: int = 0) -> np.ndarray:
    """Return (N, 512) random float32 features for offline / synthetic runs."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, 512)).astype(np.float32)
