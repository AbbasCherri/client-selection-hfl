"""Tier-2 fusion model that consumes cached image features.

Architecture mirrors ``hflsim.models.MultiModalFusionModel`` but the image
branch is replaced by a lightweight linear projection from the precomputed
ResNet-18 (512-dim) cache, so no image forward pass occurs during FL training.

Only ``struct_branch`` + ``fusion`` parameters are communicated between clients
and the server. The ``img_proj`` layer is initialized once globally and frozen
for the lifetime of the experiment — it acts as a fixed embedding adapter.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ImageProjection(nn.Module):
    """Project cached ResNet-18 (512) features into the shared embedding space."""

    def __init__(self, in_dim: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class StructuredBranch(nn.Module):
    """MLP on the 9-dim seismic / geographic feature vector."""

    def __init__(self, input_dim: int = 9, embedding_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FusionHead(nn.Module):
    """Classify the concatenated image + structured embedding."""

    def __init__(self, in_dim: int = 192, num_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CachedFusionModel(nn.Module):
    """Full Tier-2 model: cached image features + seismic MLP + fusion head.

    Parameters
    ----------
    img_feat_dim:
        Dimensionality of the precomputed feature cache (512 for ResNet-18).
    struct_dim:
        Number of structured / seismic input features (9 in the real dataset).
    img_embed:
        Output dim of the frozen image projection (matches the hflsim default).
    struct_embed:
        Output dim of the trainable structured branch.
    num_classes:
        Damage categories (4: Survived / Collapsed / Obstructed / Missing).
    """

    def __init__(
        self,
        img_feat_dim: int = 512,
        struct_dim: int = 9,
        img_embed: int = 128,
        struct_embed: int = 64,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.img_proj = ImageProjection(img_feat_dim, img_embed)
        # Freeze the image projection — only struct_branch + fusion are trained.
        for p in self.img_proj.parameters():
            p.requires_grad_(False)
        self.struct_branch = StructuredBranch(struct_dim, struct_embed)
        self.fusion = FusionHead(img_embed + struct_embed, num_classes)

    def forward(self, img_feat: torch.Tensor, struct: torch.Tensor) -> torch.Tensor:
        img_emb = self.img_proj(img_feat)
        struct_emb = self.struct_branch(struct)
        return self.fusion(torch.cat([img_emb, struct_emb], dim=1))

    # --- Parameter communication helpers ---------------------------------

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only the parameters that are communicated in FedAvg."""
        return {
            **{f"struct_branch.{k}": v.clone() for k, v in self.struct_branch.state_dict().items()},
            **{f"fusion.{k}": v.clone() for k, v in self.fusion.state_dict().items()},
        }

    def load_trainable_state_dict(self, d: dict[str, torch.Tensor]) -> None:
        """Load aggregated parameters back into the model."""
        sb = {k[len("struct_branch."):]: v for k, v in d.items() if k.startswith("struct_branch.")}
        fh = {k[len("fusion."):]: v for k, v in d.items() if k.startswith("fusion.")}
        if sb:
            self.struct_branch.load_state_dict(sb, strict=True)
        if fh:
            self.fusion.load_state_dict(fh, strict=True)


def fedavg(updates: list[tuple[dict, int]]) -> dict[str, torch.Tensor]:
    """Sample-weighted FedAvg of (state_dict, n_samples) pairs."""
    total = sum(n for _, n in updates)
    if total == 0:
        return {k: v.clone() for k, v in updates[0][0].items()}
    agg: dict[str, torch.Tensor] = {}
    for sd, n in updates:
        w = n / total
        for k, v in sd.items():
            agg[k] = agg.get(k, torch.zeros_like(v)) + w * v.float()
    return agg


def reputation_fedavg(
    updates: list[tuple[dict[str, torch.Tensor], int, float]],
) -> dict[str, torch.Tensor]:
    """Reputation-and-sample-weighted FedAvg (paper §IV-D).

    Weight for client n  =  reputation_n × n_samples_n.
    Falls back to uniform sample-count weighting if all reputations collapse to zero.
    """
    weights = [max(rep, 0.0) * n for _, n, rep in updates]
    total_w = sum(weights)
    if total_w < 1e-10:
        return fedavg([(sd, n) for sd, n, _ in updates])
    agg: dict[str, torch.Tensor] = {}
    for (sd, _n, _rep), w in zip(updates, weights):
        w_norm = w / total_w
        for k, v in sd.items():
            agg[k] = agg.get(k, torch.zeros_like(v)) + w_norm * v.float()
    return agg


def clone_model(model: CachedFusionModel) -> CachedFusionModel:
    """Return a deep copy of the model (same architecture, same weights)."""
    return copy.deepcopy(model)
