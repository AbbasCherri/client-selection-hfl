"""Device value score V_i(t) — single source of truth for both hflsim and uavbench.

V_i(t) = beta(t) * U_i(t) + (1 - beta(t)) * R_i(t)
beta(t) = max(0, 1 - t / T_decay),  T_decay = 20

Utility U_i is a weighted sum of four normalized features
(weights 0.4 / 0.3 / 0.2 / 0.1):

    * epicenter proximity   (closer to epicenter -> higher; capped at d95)
    * SNR                   (min-max over the cluster)
    * sample density        (proxy for building density -- no building
                             inventory exists for the study area, so local
                             sample count stands in, min-max normalized
                             per disaster area)
    * nearest-UAV proximity (U_prox = max(0, 1 - d_min / R_comm))
"""

from __future__ import annotations

import numpy as np

_W_EPI, _W_SNR, _W_DENS, _W_PROX = 0.4, 0.3, 0.2, 0.1
_EPS = 1e-9


def beta_schedule(t: int, T_decay: int = 20) -> float:
    """Return the utility/reputation blend factor beta(t)."""
    return max(0.0, 1.0 - t / float(T_decay))


def _minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo + _EPS)


def compute_utility(
    device_coords: np.ndarray,
    epicenter: np.ndarray,
    snr: np.ndarray,
    samples: np.ndarray,
    prev_positions: np.ndarray,
    R_comm: float = 500.0,
) -> np.ndarray:
    """Compute the per-device utility score U_i (no reputation)."""
    device_coords = np.asarray(device_coords, dtype=np.float64)
    epicenter = np.asarray(epicenter, dtype=np.float64)

    d_epi = np.sqrt(np.sum((device_coords - epicenter) ** 2, axis=1))
    d95 = np.percentile(d_epi, 95)
    u_epi = np.maximum(0.0, (d95 - np.minimum(d_epi, d95)) / (d95 + _EPS))

    u_snr = _minmax(np.asarray(snr, dtype=np.float64))

    # Building-density proxy: min-max normalized sample count per device.
    u_dens = _minmax(np.asarray(samples, dtype=np.float64))

    prev_positions = np.asarray(prev_positions, dtype=np.float64)
    diff = device_coords[:, None, :] - prev_positions[None, :, :]
    d_uav = np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1)
    u_prox = np.clip(1.0 - d_uav / max(R_comm, _EPS), 0.0, 1.0)

    return _W_EPI * u_epi + _W_SNR * u_snr + _W_DENS * u_dens + _W_PROX * u_prox


def compute_value(
    device_coords: np.ndarray,
    epicenter: np.ndarray,
    snr: np.ndarray,
    samples: np.ndarray,
    prev_positions: np.ndarray,
    reputation: np.ndarray,
    *,
    t: int = 0,
    T_decay: int = 20,
    beta_mode: str = "scheduled",
    R_comm: float = 500.0,
) -> np.ndarray:
    """Compute the fixed per-device value vector V_i(t).

    beta_mode:
        * "scheduled" — use beta(t) (history-aware blend).
        * "pinned"    — beta = 1 (utility-only, history-free benchmark).
    """
    utility = compute_utility(device_coords, epicenter, snr, samples, prev_positions, R_comm)
    if beta_mode == "pinned":
        beta = 1.0
    elif beta_mode == "scheduled":
        beta = beta_schedule(t, T_decay)
    else:
        raise ValueError(f"unknown beta_mode: {beta_mode!r}")
    return beta * utility + (1.0 - beta) * np.asarray(reputation, dtype=np.float64)
