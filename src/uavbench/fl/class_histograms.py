"""Where the per-client class histogram comes from — the oracle-degradation ladder.

The proposed selector ranks clients partly by a 4-bin histogram of their local
labels. Through 2026-07 that histogram was ``np.bincount`` of the **ground-truth
labels**, read straight out of the simulator: a statistic no participant
discloses and the server cannot compute. A reviewer is right to call that
artificial, so this module makes the histogram's *source* an explicit
experimental variable rather than a hard-coded oracle.

Sources
-------
``true``    Ground-truth ``np.bincount`` — the historical behaviour. Kept as the
            upper bound, not as the default claim.
``pseudo``  Histogram of the **current global model's predictions** on the
            client's own data. Requires no labels to leave the device and no
            disclosure at all — the server could compute it from a model it
            already has. If the result survives this, the realism objection
            dissolves entirely.
``dp``      Ground-truth counts plus Laplace noise at a stated ``epsilon``,
            clamped at zero. A one-shot release: the partition is static, so
            each client releases its histogram **once**, not per round — the
            sensitivity is 1 (one row moves one bin) and the privacy budget is
            not multiplied by the round count.
``none``    No histogram; the selector falls back to its priority ordering.

Feasibility note for the paper: supervised FL already requires local labels to
compute a loss, so a client that can train can compute ``true``. The real cost
is disclosure, not implementability — which is why ``pseudo`` (zero disclosure)
and ``dp`` (bounded disclosure) are the interesting rungs.
"""

from __future__ import annotations

import numpy as np
import torch

N_CLASSES = 4
VALID_SOURCES = ("true", "pseudo", "dp", "none")


def true_histograms(labels: torch.Tensor, client_indices: dict[int, list[int]]) -> dict[int, np.ndarray]:
    """Exact per-client label histogram (the oracle)."""
    out: dict[int, np.ndarray] = {}
    for cid, idx in client_indices.items():
        lab = labels[torch.as_tensor(idx, dtype=torch.long)].numpy().astype(int)
        out[cid] = np.bincount(lab, minlength=N_CLASSES).astype(np.float64)
    return out


def dp_histograms(
    true_counts: dict[int, np.ndarray], epsilon: float, rng: np.random.Generator
) -> dict[int, np.ndarray]:
    """Laplace-noised histograms at ``epsilon`` (one-shot release, sensitivity 1).

    Counts are clamped at zero afterwards. No renormalisation to the true total
    is applied: rescaling back to a known row count would leak the exact total
    and quietly undo part of the noise.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    scale = 1.0 / float(epsilon)  # sensitivity 1 / epsilon
    return {
        cid: np.maximum(0.0, c + rng.laplace(0.0, scale, size=c.shape))
        for cid, c in true_counts.items()
    }


def pseudo_histograms(
    model,
    cached_dataset,
    client_indices: dict[int, list[int]],
    batch_size: int = 512,
) -> dict[int, np.ndarray]:
    """Histogram of the global model's *predictions* on each client's own data.

    Zero label disclosure: the server already holds the model, and the client
    reveals only a distribution over its own predicted classes. Recomputed by
    the caller whenever it wants the estimate refreshed — early rounds give a
    poor histogram, which is precisely the cost this rung is meant to expose.
    """
    out: dict[int, np.ndarray] = {}
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for cid, idx in client_indices.items():
                if not idx:
                    out[cid] = np.zeros(N_CLASSES, dtype=np.float64)
                    continue
                preds: list[np.ndarray] = []
                idx_t = torch.as_tensor(idx, dtype=torch.long)
                for start in range(0, len(idx), batch_size):
                    chunk = idx_t[start : start + batch_size]
                    preds.append(
                        model(
                            cached_dataset.img_features[chunk],
                            cached_dataset.struct_features[chunk],
                        )
                        .argmax(1)
                        .numpy()
                    )
                out[cid] = np.bincount(
                    np.concatenate(preds).astype(int), minlength=N_CLASSES
                ).astype(np.float64)
    finally:
        if was_training:
            model.train()
    return out


def scarcity_from_counts(counts: dict[int, np.ndarray]) -> np.ndarray:
    """Inverse-prior class weights derived from the *same* source as ``counts``.

    Deliberately not computed from the global ground-truth prior: pairing a
    disclosure-free histogram with an oracle scarcity vector would smuggle the
    oracle back in through the weights.
    """
    total = np.sum(list(counts.values()), axis=0) if counts else np.zeros(N_CLASSES)
    prior = total / max(total.sum(), 1e-8)
    inv = 1.0 / np.clip(prior, 1e-8, None)
    return inv / inv.sum()


def build_class_info(
    source: str,
    *,
    labels: torch.Tensor,
    client_indices: dict[int, list[int]],
    global_prior: np.ndarray | None = None,
    model=None,
    cached_dataset=None,
    epsilon: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[dict[int, np.ndarray] | None, np.ndarray | None]:
    """Return ``(class_counts, class_scarcity)`` for the requested ``source``.

    ``none`` returns ``(None, None)``, which every selector treats as "no class
    information" — the fallback path, not an error.
    """
    if source not in VALID_SOURCES:
        raise ValueError(f"class_source must be one of {VALID_SOURCES}, got {source!r}")

    if source == "none":
        return None, None

    if source == "true":
        counts = true_histograms(labels, client_indices)
        if global_prior is not None:
            inv = 1.0 / np.clip(global_prior, 1e-8, None)
            return counts, inv / inv.sum()
        return counts, scarcity_from_counts(counts)

    if source == "dp":
        if rng is None:
            raise ValueError("class_source='dp' requires an rng for reproducible noise")
        counts = dp_histograms(true_histograms(labels, client_indices), epsilon, rng)
        # Scarcity from the noised counts too — the server never sees the clean
        # prior under this release model.
        return counts, scarcity_from_counts(counts)

    if model is None or cached_dataset is None:
        raise ValueError("class_source='pseudo' requires model and cached_dataset")
    counts = pseudo_histograms(model, cached_dataset, client_indices)
    return counts, scarcity_from_counts(counts)
