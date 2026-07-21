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

    # --- block freeze control (paper §IV-B asymmetric / modality-aligned training) --

    #: canonical block names, in state-dict prefix order
    BLOCKS = ("img_proj", "struct_branch", "fusion")

    def _block(self, name: str) -> nn.Module:
        return getattr(self, name)

    def set_trainable_blocks(self, blocks: set[str] | tuple[str, ...]) -> None:
        """Enable gradients on exactly the named blocks; freeze the rest.

        Modality-aligned training (Tier B) assigns one owner per block: the UAV
        tier trains ``img_proj``+``fusion`` (both modalities co-located) with
        ``struct_branch`` frozen; IoT clients train ``struct_branch`` alone with
        the other two frozen. Each block is thus optimized against the exact
        frozen value of its counterpart, rather than two tiers drifting apart
        in parallel and being glued together at aggregation time.
        """
        want = set(blocks)
        for name in self.BLOCKS:
            flag = name in want
            for p in self._block(name).parameters():
                p.requires_grad_(flag)

    def freeze_img_proj(self) -> None:
        """Freeze img_proj — used for the global model so IoT clients cannot update it."""
        for p in self.img_proj.parameters():
            p.requires_grad_(False)

    def unfreeze_img_proj(self) -> None:
        """Unfreeze img_proj — called on the UAV's local clone before training."""
        for p in self.img_proj.parameters():
            p.requires_grad_(True)

    # --- generic per-block parameter communication -----------------------

    def block_state_dict(self, blocks: set[str] | tuple[str, ...]) -> dict[str, torch.Tensor]:
        """Prefixed clone of the parameters in the named blocks."""
        out: dict[str, torch.Tensor] = {}
        for name in self.BLOCKS:
            if name in blocks:
                out.update({f"{name}.{k}": v.clone() for k, v in self._block(name).state_dict().items()})
        return out

    def load_block_state_dict(self, d: dict[str, torch.Tensor]) -> None:
        """Load whichever ``<block>.<param>`` prefixes are present; ignore the rest."""
        for name in self.BLOCKS:
            prefix = f"{name}."
            sub = {k[len(prefix) :]: v for k, v in d.items() if k.startswith(prefix)}
            if sub:
                self._block(name).load_state_dict(sub, strict=True)

    # --- IoT-level parameter communication (struct_branch + fusion only) --

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Return struct_branch + fusion parameters (legacy IoT-level payload)."""
        return self.block_state_dict(("struct_branch", "fusion"))

    def load_trainable_state_dict(self, d: dict[str, torch.Tensor]) -> None:
        """Load aggregated struct_branch + fusion (whichever prefixes are present)."""
        self.load_block_state_dict(d)

    # --- UAV-level parameter communication (img_proj + struct_branch + fusion)

    def full_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Return img_proj + struct_branch + fusion parameters (UAV-level payload)."""
        return self.block_state_dict(self.BLOCKS)

    def load_full_trainable_state_dict(self, d: dict[str, torch.Tensor]) -> None:
        """Load aggregated parameters; any subset of block prefixes is accepted.

        Missing prefixes leave that block unchanged (e.g. flat_fl server
        aggregation carries no img_proj keys → img_proj stays put).
        """
        self.load_block_state_dict(d)


def make_loss_fn(log_prior: torch.Tensor | None = None, tau: float = 1.0):
    """Logit-adjusted cross-entropy (Menon et al., ICLR 2021) if a class
    log-prior is supplied, else plain cross-entropy.

    Training shifts the logits by ``tau·log_prior`` so the long-tailed prior is
    corrected inside the loss — every client optimizes the *same* objective on
    its raw shard, instead of each resampling its shard to a different effective
    distribution (the per-shard ``BalancedShardLoader`` behaviour this replaces,
    which manufactured client drift). Inference uses the raw logits (no shift),
    so only the training gradient changes.
    """
    ce = nn.CrossEntropyLoss()
    if log_prior is None:
        return ce
    shift = tau * log_prior

    def _loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ce(logits + shift, target)

    return _loss


def fedavg(updates: list[tuple[dict, int]]) -> dict[str, torch.Tensor]:
    """Sample-weighted FedAvg of (state_dict, n_samples) pairs.

    Accumulates in place (first contribution seeds the accumulator, the rest
    ``+=``) instead of allocating a ``zeros_like`` plus a fresh sum tensor per
    key per update. Numerically identical to the out-of-place form: in-place
    and out-of-place float add invoke the same kernel, and the first update's
    ``w·v`` is exactly what ``0 + w·v`` produced before.
    """
    total = sum(n for _, n in updates)
    if total == 0:
        return {k: v.clone() for k, v in updates[0][0].items()}
    agg: dict[str, torch.Tensor] = {}
    for sd, n in updates:
        w = n / total
        for k, v in sd.items():
            contrib = v.float() * w
            if k in agg:
                agg[k] += contrib
            else:
                agg[k] = contrib
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
            contrib = v.float() * w_norm
            if k in agg:
                agg[k] += contrib
            else:
                agg[k] = contrib
    return agg


# NOTE: mixed_fedavg / mixed_reputation_fedavg (paper §IV-A Step 6 mixed
# aggregation) were removed 2026-07-14: the live round loop aggregates the
# UAV image-branch and IoT struct/fusion contributions separately (see
# run_full_hfl), and neither function had a remaining call site. Recover
# from git history if the mixed formulation is ever revisited.


def clone_model(model: CachedFusionModel) -> CachedFusionModel:
    """Return an independent deep copy of the model (same weights, same
    per-parameter ``requires_grad``, same training mode).

    Bit-identical to ``copy.deepcopy(model)`` for every model this codebase
    builds (verified in check_integration_hfl), but avoids deepcopy's generic
    per-object protocol overhead (~1.5x faster on the hot per-client/per-UAV
    clone). Two invariants make the fast path exact:

    * **RNG stream is preserved.** Constructing a fresh module runs the layer
      initializers, which draw from torch's global RNG; we snapshot and restore
      the RNG state around construction so cloning consumes zero random numbers
      (as deepcopy does). This is load-bearing for reproducibility — a shifted
      RNG stream would change every downstream draw.
    * **Uniform training flag.** ``clone.train(model.training)`` reproduces
      deepcopy's per-submodule flags only because this codebase toggles
      train/eval at whole-model granularity exclusively (no per-submodule
      calls). The model also registers no buffers (no BatchNorm/LayerNorm), so
      the parameter copy is a complete state copy.
    """
    rng_state = torch.get_rng_state()
    clone = model.__class__()  # every clone target is a default-constructed CachedFusionModel
    torch.set_rng_state(rng_state)
    with torch.no_grad():
        for src, dst in zip(model.parameters(), clone.parameters()):
            dst.copy_(src)
            dst.requires_grad_(src.requires_grad)
        for src, dst in zip(model.buffers(), clone.buffers()):
            dst.copy_(src)
    clone.train(model.training)
    return clone
