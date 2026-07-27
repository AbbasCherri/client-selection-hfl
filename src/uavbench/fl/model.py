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
import torch.nn.functional as F


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
        self._bind_leaves()

    def _bind_leaves(self) -> None:
        """Cache direct references to the leaf layers used by ``forward``.

        A plain list attribute holding Modules is *not* registered as a child
        (that is what ``nn.ModuleList`` is for), so ``state_dict`` keys,
        ``parameters()`` order, and ``block_state_dict`` are all unchanged —
        these are the same objects already reachable through img_proj /
        struct_branch / fusion, just pre-resolved so ``forward`` need not walk
        ``nn.Sequential.__getitem__`` on every call.
        """
        self._leaves = [
            self.img_proj.proj[0],        # Linear(512, 128)
            self.struct_branch.mlp[0],    # Linear(9, 64)
            self.struct_branch.mlp[2],    # Linear(64, 128)
            self.struct_branch.mlp[4],    # Dropout(0.2)
            self.struct_branch.mlp[5],    # Linear(128, 64)
            self.fusion.net[0],           # Linear(192, 256)
            self.fusion.net[2],           # Dropout(0.3)
            self.fusion.net[3],           # Linear(256, 4)
        ]

    def forward(self, img_feat: torch.Tensor, struct: torch.Tensor) -> torch.Tensor:
        """Flattened functional forward — identical ops, identical order.

        Calling ``F.*`` on the leaf parameters directly rather than invoking the
        three sub-modules (each wrapping an ``nn.Sequential``, each layer going
        through ``nn.Module._call_impl``) removes ~11 Python frames per forward.
        Measured: 495 us -> 441 us per batch-32 forward, i.e. 11% of the forward
        and ~4% of a full training step, which runs ~10^7 times in a paper grid.
        Bit-identical (including the dropout RNG draws, which ``nn.Dropout``
        makes through this exact ``F.dropout`` call) — pinned in
        check_integration_hfl.
        """
        ip, s0, s2, sdrop, s5, f0, fdrop, f3 = self._leaves
        img_emb = F.relu(F.linear(img_feat, ip.weight, ip.bias))
        x = F.relu(F.linear(struct, s0.weight, s0.bias))
        x = F.relu(F.linear(x, s2.weight, s2.bias))
        x = F.dropout(x, sdrop.p, sdrop.training)
        struct_emb = F.linear(x, s5.weight, s5.bias)
        h = F.relu(F.linear(torch.cat([img_emb, struct_emb], dim=1), f0.weight, f0.bias))
        h = F.dropout(h, fdrop.p, fdrop.training)
        return F.linear(h, f3.weight, f3.bias)

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


def clip_grad_norm_(params, max_norm: float) -> None:
    """``torch.nn.utils.clip_grad_norm_`` with the dispatch layers stripped out.

    Same op sequence as torch's implementation — stack the per-tensor 2-norms,
    take their 2-norm, clamp ``max_norm / (total + 1e-6)`` at 1, scale the grads
    in place with ``_foreach_mul_`` — so the result is bit-identical. What is
    skipped is ~15 Python frames per call: ``_no_grad_wrapper``, the
    ``@_compile`` shim, ``_group_tensors_by_device_and_dtype``, and the
    foreach-support probing, none of which can vary for this model (one device,
    one dtype, no sparse grads). Measured 83 us -> 46 us per call; it runs once
    per optimizer step, i.e. ~10^7 times in a paper grid.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    total_norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads, 2.0)), 2.0)
    torch._foreach_mul_(grads, torch.clamp(max_norm / (total_norm + 1e-6), max=1.0))


class MomentumSGD:
    """``torch.optim.SGD(momentum=m)`` for the plain case, without the wrappers.

    Restricted on purpose to ``weight_decay=0, dampening=0, nesterov=False,
    maximize=False, momentum != 0`` — exactly what ``fl.local_optimizer``
    configures — because that is the case whose update reduces to

        buf = clone(grad)                       (first step)
        buf = buf*momentum + grad               (later steps)
        param = param - lr*buf

    which ``torch._foreach_*`` reproduces element-for-element (these are
    element-wise kernels; there is no reduction whose order could change). The
    win is not the arithmetic but the ~10 Python frames torch.optim wraps every
    ``step()`` in — the ``_use_grad`` context, the profiler ``record_function``
    hook, ``_init_group``, and the per-tensor ``_single_tensor_sgd`` loop that
    torch's CPU path selects because foreach is only auto-enabled on CUDA.
    Measured 92 us -> 41 us per step. ``_make_optimizer`` falls back to
    ``torch.optim`` for every configuration outside the restriction.

    ``momentum=0`` is excluded rather than special-cased: torch skips the buffer
    entirely there, and emulating it as ``0*buf + grad`` would differ on a
    non-finite buffer (``0*inf = nan``).
    """

    __slots__ = ("params", "lr", "momentum", "_buf")

    def __init__(self, params, lr: float, momentum: float) -> None:
        self.params = list(params)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self._buf: list[torch.Tensor] | None = None

    @torch.no_grad()
    def step(self) -> None:
        grads = [p.grad for p in self.params]
        if any(g is None for g in grads):
            # torch skips such params (and leaves their buffer un-advanced);
            # the batched form below cannot, and silently diverging is worse
            # than refusing. No call site produces this — every parameter
            # handed to this optimizer is on the loss path.
            raise RuntimeError(
                "MomentumSGD requires a gradient on every parameter; got None. "
                "Build the optimizer with torch.optim.SGD for this model instead."
            )
        if self._buf is None:
            self._buf = [torch.clone(g).detach() for g in grads]
        else:
            torch._foreach_mul_(self._buf, self.momentum)
            torch._foreach_add_(self._buf, grads)
        torch._foreach_add_(self.params, self._buf, alpha=-self.lr)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


def _weighted_accumulate(
    sds: list[dict[str, torch.Tensor]], weights: list[float]
) -> dict[str, torch.Tensor]:
    """agg[k] = Σ_i weights[i]·sds[i][k], batched with ``torch._foreach_*``.

    One mul + one in-place add call per *update* (instead of one small torch op
    per key per update — per-op dispatch overhead dominated the aggregation
    cost at ~14 keys × tens of updates). Per-element math and accumulation
    order are unchanged: each element still receives w₀·v₀ then += wᵢ·vᵢ in
    update order, via the same mul/add kernels, so the result is bit-identical
    to the per-key loop (pinned in check_integration_hfl). Falls back to that
    loop if the updates do not share one key set (no current caller does, but
    the function contract never required homogeneity).
    """
    keys = list(sds[0].keys())
    if any(len(sd) != len(keys) or any(k not in sd for k in keys) for sd in sds[1:]):
        agg: dict[str, torch.Tensor] = {}
        for sd, w in zip(sds, weights):
            for k, v in sd.items():
                contrib = v.float() * w
                if k in agg:
                    agg[k] += contrib
                else:
                    agg[k] = contrib
        return agg
    agg_list = torch._foreach_mul([sds[0][k].float() for k in keys], weights[0])
    for sd, w in zip(sds[1:], weights[1:]):
        torch._foreach_add_(agg_list, torch._foreach_mul([sd[k].float() for k in keys], w))
    return dict(zip(keys, agg_list))


def fedavg(updates: list[tuple[dict, int]]) -> dict[str, torch.Tensor]:
    """Sample-weighted FedAvg of (state_dict, n_samples) pairs."""
    total = sum(n for _, n in updates)
    if total == 0:
        return {k: v.clone() for k, v in updates[0][0].items()}
    return _weighted_accumulate([sd for sd, _ in updates], [n / total for _, n in updates])


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
    return _weighted_accumulate([sd for sd, _n, _rep in updates], [w / total_w for w in weights])


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

    The clone's ``_leaves`` cache is rebuilt by its own ``__init__`` and points
    at the clone's own sub-modules, so the functional forward stays correct.
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


class EndToEndFusionModel(nn.Module):
    """Same head as :class:`CachedFusionModel`, but the 512-dim image feature is
    produced by a *trainable* ResNet-18 backbone from raw images instead of the
    frozen cache. Used only by the centralized end-to-end validation
    (scripts/e2e_centralized.py) to check the frozen-feature simplification.
    """

    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(
        self,
        struct_dim: int = 9,
        img_embed: int = 128,
        struct_embed: int = 64,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()  # -> 512-dim features, fully trainable
        self.img_proj = ImageProjection(512, img_embed)
        self.struct_branch = StructuredBranch(struct_dim, struct_embed)
        self.fusion = FusionHead(img_embed + struct_embed, num_classes)

    def forward(self, img: torch.Tensor, struct: torch.Tensor) -> torch.Tensor:
        img = (img - self._MEAN.to(img.device)) / self._STD.to(img.device)
        feat = self.backbone(img)
        return self.fusion(torch.cat([self.img_proj(feat), self.struct_branch(struct)], dim=1))
