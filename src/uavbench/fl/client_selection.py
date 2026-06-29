"""Client selection — Algorithms 1-4 from paper (§IV-C).

Pipeline
--------
1. Eligibility gates    : battery ≥ B_min, SNR ≥ SNR_min, memory OK, time ≤ T_max
2. Priority score       : P_n = w_b·b̃ + w_ℓ·r̃ + w_U·Ũ
3. UCB exploration      : UCB_n(t) = P_n(t) + C·√(ln t / (N_n(t)+1))
4. Greedy assignment    : sort by UCB, fill UAVs up to capacity C_u

Selection modes
---------------
"ucb"    — full pipeline (proposed system)
"random" — eligibility filter only, then random draw per UAV (hfl_no_selection)
"all"    — skip all filters; every covered client participates (flat_fl / centralized)
"""

from __future__ import annotations

import math

import numpy as np

from .device_state import DeviceState

# Paper §IV-C priority weights
W_BATTERY  = 0.35
W_LEARNING = 0.30
W_UTILITY  = 0.35

# Utility sub-weights (§IV-C)
W_EPI  = 0.4
W_SNR  = 0.3
W_DENS = 0.2
W_PROX = 0.1

UCB_C = math.sqrt(2)   # exploration constant from paper

# Noto Peninsula 2024 epicentre (default; override via cfg)
DEFAULT_EPICENTRE = (37.488, 137.272)   # (lat °N, lon °E)


def _minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-10:
        return np.full_like(v, 0.5, dtype=float)
    return (v - lo) / (hi - lo)


def _xy_metres(coords: list[tuple[float, float]]) -> np.ndarray:
    """Cheap linear projection to metres for intra-area distances (error <0.1% at 50 km)."""
    arr = np.array(coords, dtype=np.float64)
    lat0 = arr[:, 0].mean()
    R = 6_371_000.0
    x = (arr[:, 1] - arr[:, 1].mean()) * R * math.pi / 180.0 * math.cos(math.radians(lat0))
    y = (arr[:, 0] - lat0) * R * math.pi / 180.0
    return np.column_stack([x, y])


def _compute_utility(
    eligible_ids: list[int],
    device_states: dict[int, DeviceState],
    client_coords: dict[int, tuple[float, float]],
    uav_coords_latlon: list[tuple[float, float]],
    epicentre: tuple[float, float],
) -> dict[int, float]:
    """Return Û_n = 0.4·U_epi + 0.3·U_SNR + 0.2·U_dens + 0.1·U_prox for each eligible client."""
    n = len(eligible_ids)
    if n == 0:
        return {}

    coords = [client_coords[cid] for cid in eligible_ids]

    # U_epi — proximity to earthquake epicentre (closer → higher).
    # Project epicentre + clients together so they share the same reference frame.
    epi_client_xy = _xy_metres([epicentre] + coords)
    epi_xy  = epi_client_xy[0]          # (2,)
    client_xy = epi_client_xy[1:]        # (N, 2)
    epi_dists = np.linalg.norm(client_xy - epi_xy, axis=1)
    u_epi = 1.0 - _minmax(epi_dists)

    # U_SNR — normalised SNR score
    snr = np.clip([device_states[cid].snr_db for cid in eligible_ids], 0.0, 30.0)
    u_snr = _minmax(snr)

    # U_dens — number of eligible clients within 5 km radius (vectorised O(N²)).
    # Use the same client projection (client_xy already in a consistent frame).
    if n > 1:
        diff = client_xy[:, None, :] - client_xy[None, :, :]   # (N, N, 2)
        sq_dists = (diff ** 2).sum(axis=2)                       # (N, N)
        density = (sq_dists < 5_000.0 ** 2).sum(axis=1) - 1.0   # exclude self
    else:
        density = np.zeros(n)
    u_dens = _minmax(density)

    # U_prox — proximity to nearest UAV (closer → higher).
    # Project UAVs + clients together so they share the same reference frame.
    if uav_coords_latlon:
        K_uav = len(uav_coords_latlon)
        uav_client_xy = _xy_metres(uav_coords_latlon + coords)
        uav_xy      = uav_client_xy[:K_uav]    # (K, 2)
        client_xy2  = uav_client_xy[K_uav:]    # (N, 2)
        # Vectorised (N, K) distance matrix → min over K
        diff2 = client_xy2[:, None, :] - uav_xy[None, :, :]   # (N, K, 2)
        prox_dists = np.sqrt((diff2 ** 2).sum(axis=2)).min(axis=1)  # (N,)
        u_prox = 1.0 - _minmax(prox_dists)
    else:
        u_prox = np.full(n, 0.5)

    utility = W_EPI * u_epi + W_SNR * u_snr + W_DENS * u_dens + W_PROX * u_prox
    return {cid: float(utility[i]) for i, cid in enumerate(eligible_ids)}


class ClientSelector:
    """Stateful client selector: tracks per-client selection counts for UCB."""

    def __init__(
        self,
        client_ids: list[int],
        epicentre: tuple[float, float] | None = None,
    ) -> None:
        self._counts: dict[int, int] = {cid: 0 for cid in client_ids}
        self._epicentre = epicentre or DEFAULT_EPICENTRE

    def select(
        self,
        covered: dict[int, int],                        # {client_id: uav_idx}
        device_states: dict[int, DeviceState],
        reputation_scores: dict[int, float],
        client_coords: dict[int, tuple[float, float]],
        uav_coords_latlon: list[tuple[float, float]],
        round_num: int,
        uav_capacity: int,
        mode: str = "ucb",                              # "ucb" | "random" | "all"
        rng: np.random.Generator | None = None,
    ) -> dict[int, int]:
        """Return {client_id: uav_idx} for the clients selected this round."""
        if mode == "all":
            return dict(covered)

        # ── Eligibility gate ────────────────────────────────────────────
        eligible: dict[int, int] = {}
        for cid, uav_idx in covered.items():
            st = device_states.get(cid)
            if st is not None and st.eligible():
                eligible[cid] = uav_idx

        if not eligible:
            return {}

        if mode == "random":
            return self._random_select(eligible, uav_capacity, round_num, rng=rng)

        # ── UCB pipeline ────────────────────────────────────────────────
        eligible_ids = list(eligible.keys())

        utility = _compute_utility(
            eligible_ids, device_states, client_coords, uav_coords_latlon, self._epicentre
        )

        batteries    = np.array([device_states[cid].battery            for cid in eligible_ids])
        reputations  = np.array([reputation_scores.get(cid, 0.5)       for cid in eligible_ids])
        utilities    = np.array([utility.get(cid, 0.5)                 for cid in eligible_ids])

        priority = (
            W_BATTERY  * _minmax(batteries)
            + W_LEARNING * _minmax(reputations)
            + W_UTILITY  * _minmax(utilities)
        )

        t = max(round_num, 1)
        sel_cnts = np.array([self._counts[cid] for cid in eligible_ids], dtype=float)
        ucb = priority + UCB_C * np.sqrt(math.log(t) / (sel_cnts + 1.0))

        return self._greedy_assign(eligible_ids, eligible, ucb, uav_capacity)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _greedy_assign(
        self,
        eligible_ids: list[int],
        eligible: dict[int, int],
        scores: np.ndarray,
        uav_capacity: int,
    ) -> dict[int, int]:
        """Assign highest-scoring clients to UAVs, respecting capacity."""
        order = np.argsort(-scores)
        fill: dict[int, int] = {}
        selected: dict[int, int] = {}
        for idx in order:
            cid = eligible_ids[idx]
            uav = eligible[cid]
            if fill.get(uav, 0) < uav_capacity:
                selected[cid] = uav
                fill[uav] = fill.get(uav, 0) + 1
                self._counts[cid] += 1
        return selected

    def _random_select(
        self,
        eligible: dict[int, int],
        uav_capacity: int,
        round_num: int,
        rng: np.random.Generator | None = None,
    ) -> dict[int, int]:
        """Random selection respecting UAV capacity.

        Uses the caller's RNG when provided (required for correct multi-seed
        sweeps). Falls back to a round-derived seed only for legacy callers.
        """
        uav_buckets: dict[int, list[int]] = {}
        for cid, uav in eligible.items():
            uav_buckets.setdefault(uav, []).append(cid)
        _rng = rng if rng is not None else np.random.default_rng(round_num * 7919)
        selected: dict[int, int] = {}
        for uav, cids in uav_buckets.items():
            n = min(uav_capacity, len(cids))
            for cid in _rng.choice(cids, size=n, replace=False):
                selected[int(cid)] = uav
        return selected
