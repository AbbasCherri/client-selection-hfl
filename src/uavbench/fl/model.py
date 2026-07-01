"""Tier-2 fusion model that consumes cached image features.

Architecture mirrors ``hflsim.models.MultiModalFusionModel`` but the image
branch is replaced by a lightweight linear projection from the precomputed
ResNet-18 (512-dim) cache, so no image forward pass occurs during FL training.

Asymmetric training (paper §IV-B):
- UAVs train the full model including ``img_proj`` (unfreeze_img_proj()).
- IoT devices train only ``struct_branch`` + ``fusion`` (img_proj frozen via
  freeze_img_proj(), called after model creation in the FL harness).

``trainable_state_dict`` / ``load_trainable_state_dict`` — IoT-level comms
``full_trainable_state_dict`` / ``load_full_trainable_state_dict`` — UAV-level
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
        Output dim of the image projection.
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
        # img_proj starts trainable; the FL harness calls freeze_img_proj()
        # after construction so IoT clients cannot update it.  UAV training
        # calls unfreeze_img_proj() on its own local copy before each round.
        self.struct_branch = StructuredBranch(struct_dim, struct_embed)
        self.fusion = FusionHead(img_embed + struct_embed, num_classes)

    def forward(self, img_feat: torch.Tensor, struct: torch.Tensor) -> torch.Tensor:
        img_emb = self.img_proj(img_feat)
        struct_emb = self.struct_branch(struct)
        return self.fusion(torch.cat([img_emb, struct_emb], dim=1))

    # --- img_proj freeze control (paper §IV-B asymmetric training) --------

    def freeze_img_proj(self) -> None:
        """Freeze img_proj — used for the global model so IoT clients cannot update it."""
        for p in self.img_proj.parameters():
            p.requires_grad_(False)

    def unfreeze_img_proj(self) -> None:
        """Unfreeze img_proj — called on the UAV's local clone before training."""
        for p in self.img_proj.parameters():
            p.requires_grad_(True)

    # --- IoT-level parameter communication (struct_branch + fusion only) --

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Return struct_branch + fusion parameters (IoT-level FedAvg payload)."""
        return {
            **{f"struct_branch.{k}": v.clone() for k, v in self.struct_branch.state_dict().items()},
            **{f"fusion.{k}": v.clone() for k, v in self.fusion.state_dict().items()},
        }

    def load_trainable_state_dict(self, d: dict[str, torch.Tensor]) -> None:
        """Load aggregated IoT-level parameters (struct_branch + fusion)."""
        sb = {k[len("struct_branch."):]: v for k, v in d.items() if k.startswith("struct_branch.")}
        fh = {k[len("fusion."):]: v for k, v in d.items() if k.startswith("fusion.")}
        if sb:
            self.struct_branch.load_state_dict(sb, strict=True)
        if fh:
            self.fusion.load_state_dict(fh, strict=True)

    # --- UAV-level parameter communication (img_proj + struct_branch + fusion)

    def full_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Return img_proj + struct_branch + fusion parameters (UAV-level payload)."""
        return {
            **{f"img_proj.{k}": v.clone() for k, v in self.img_proj.state_dict().items()},
            **self.trainable_state_dict(),
        }

    def load_full_trainable_state_dict(self, d: dict[str, torch.Tensor]) -> None:
        """Load aggregated UAV-level parameters; img_proj keys are optional.

        If img_proj keys are absent (e.g. flat_fl server aggregation), only
        struct_branch and fusion are updated — img_proj stays unchanged.
        """
        ip = {k[len("img_proj."):]: v for k, v in d.items() if k.startswith("img_proj.")}
        if ip:
            self.img_proj.load_state_dict(ip, strict=True)
        self.load_trainable_state_dict(d)


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


def mixed_fedavg(
    uav_update: tuple[dict[str, torch.Tensor], int],
    iot_updates: list[tuple[dict[str, torch.Tensor], int]],
) -> dict[str, torch.Tensor]:
    """Paper §IV-A Step 6: w̃_u = (n_img·w_img + Σ n_i·w_i) / (n_img + Σ n_i).

    img_proj keys come from the UAV only (n_img weight, no IoT contribution).
    struct_branch + fusion keys are FedAvg of UAV + all IoT updates.
    """
    uav_sd, n_img = uav_update
    total_n = n_img + sum(n for _, n in iot_updates)
    if total_n == 0:
        return {k: v.clone() for k, v in uav_sd.items()}
    agg: dict[str, torch.Tensor] = {}
    for k, v in uav_sd.items():
        if k.startswith("img_proj."):
            agg[k] = v.clone()                         # UAV owns img_proj entirely
        else:
            agg[k] = (n_img / total_n) * v.float()     # UAV's weighted share of struct+fusion
    for sd, n in iot_updates:
        w = n / total_n
        for k, v in sd.items():                        # IoT keys: struct_branch.* / fusion.*
            agg[k] = agg[k] + w * v.float()
    return agg


def mixed_reputation_fedavg(
    uav_update: tuple[dict[str, torch.Tensor], int, float],
    iot_updates: list[tuple[dict[str, torch.Tensor], int, float]],
) -> dict[str, torch.Tensor]:
    """Reputation-weighted variant of mixed_fedavg (proposed_hfl / hfl_no_selection).

    UAV reputation is treated as 1.0 (trusted aggregator; paper §IV-C7).
    img_proj comes from UAV only regardless of reputation weighting.
    """
    uav_sd, n_img, uav_rep = uav_update
    weights_iot = [max(r, 0.0) * n for _, n, r in iot_updates]
    w_uav = max(uav_rep, 0.0) * n_img
    total_w = w_uav + sum(weights_iot)
    if total_w < 1e-10:
        return mixed_fedavg((uav_sd, n_img), [(sd, n) for sd, n, _ in iot_updates])
    agg: dict[str, torch.Tensor] = {}
    for k, v in uav_sd.items():
        if k.startswith("img_proj."):
            agg[k] = v.clone()
        else:
            agg[k] = (w_uav / total_w) * v.float()
    for (sd, _n, _r), w in zip(iot_updates, weights_iot):
        wn = w / total_w
        for k, v in sd.items():
            agg[k] = agg[k] + wn * v.float()
    return agg


def clone_model(model: CachedFusionModel) -> CachedFusionModel:
    """Return a deep copy of the model (same architecture, same weights)."""
    return copy.deepcopy(model)
