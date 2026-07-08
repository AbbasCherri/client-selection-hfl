"""Client selection — Algorithms 1-4 from paper (§IV-C).

Pipeline
--------
1. Eligibility gates    : battery ≥ B_min, SNR ≥ SNR_min, memory OK, time ≤ T_max − ε
2. Priority score       : P_n = w_b·b_n + w_ℓ·(1 − (T̂_n/T_max)²) + w_U·Ũ_n
                          with Ũ_n = β(t)·Û_n + (1 − β(t))·R_n
3. UCB exploration      : UCB_n(t) = P_n(t) + C·√(ln t / (N_n(t)+1))
4. Greedy assignment    : sort by UCB; each client goes to the feasible UAV
                          (in range, under capacity) with the lowest current
                          load, ties broken by smallest distance; skipped if
                          no UAV is feasible (paper Algorithm 4)

Selection modes
---------------
"ucb"      — full pipeline (proposed system)
"random"   — eligibility filter only, then random draw per UAV (hfl_no_selection)
"all"      — skip all filters; every covered client participates (flat_fl / centralized)

Literature baselines (Algorithms B1-B3, REPORTS/literature_baselines.md)
------------------------------------------------------------------------
"fedcs"    — B1: FedCS greedy deadline selection per UAV, purely time-driven
             (Nishio & Yonetani, ICC 2019)
"rep_cap"  — B2: γ·R_n + (1−γ)·ℓ̃_n reputation-capability ranking, no
             exploration, no utility (Zhao et al., Chin. J. Aeronaut. 2024)
"fair_mab" — B3: w_e·b_n + w_s·staleness_n fairness/energy MAB reward,
             no reputation, no utility (Zhu et al., Sensors 2024)

All three run behind the same eligibility gate as "ucb"; "rep_cap" and
"fair_mab" rank candidates and feed the identical greedy UAV assignment
(Algorithm 4), so the selection rule is the only experimental variable.
"""

from __future__ import annotations

import math

import numpy as np

from hflsim.shared.coords import haversine
from hflsim.shared.value import beta_schedule

from .device_state import DeviceState, T_MAX_S

# Paper §IV-C priority weights
W_BATTERY  = 0.35
W_LEARNING = 0.30
W_UTILITY  = 0.35

# Utility sub-weights (§IV-C3)
W_EPI  = 0.4
W_SNR  = 0.3
W_DENS = 0.2
W_PROX = 0.1

UCB_C = math.sqrt(2)   # exploration constant from paper

# Baseline B2 (Zhao et al. 2024): trust/capability split γ·R_n + (1−γ)·ℓ̃_n.
# γ = 0.5 is the symmetric default documented in the adaptation note.
REPCAP_GAMMA = 0.5

# Baseline B3 (Zhu et al. 2024): reward = w_energy·b_n + w_stale·staleness_n,
# weights sum to 1 (0.5/0.5 default per the source's balanced setting).
FAIRMAB_W_ENERGY = 0.5
FAIRMAB_W_STALE = 0.5
# Staleness normalization cap in rounds; T_sel (the reselection interval) is
# the documented default so staleness saturates on the same cadence the
# proposed system reconsiders devices. Callers pass fl.T_sel via select().
DEFAULT_T_STALE_CAP = 5

# Noto Peninsula 2024 epicentre (default; override via cfg)
DEFAULT_EPICENTRE = (37.488, 137.272)   # (lat °N, lon °E)

# Fallback UAV-IoT communication range (paper Table II) when the harness does
# not pass its own R_comm.
DEFAULT_R_COMM_M = 500.0


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
    R_comm: float = DEFAULT_R_COMM_M,
) -> dict[int, float]:
    """Return Û_n = 0.4·U_epi + 0.3·U_SNR + 0.2·U_dens + 0.1·U_prox for each eligible client."""
    n = len(eligible_ids)
    if n == 0:
        return {}

    coords = [client_coords[cid] for cid in eligible_ids]

    # U_epi — Haversine distance to the epicentre, capped at the 95th percentile
    # across the currently eligible devices (paper Algorithm 2):
    #   U_epi = (d95 − min(d_n, d95)) / d95
    epi_dists = np.array([haversine(c, epicentre) for c in coords])
    d95 = float(np.percentile(epi_dists, 95))
    if d95 > 1e-9:
        u_epi = (d95 - np.minimum(epi_dists, d95)) / d95
    else:
        u_epi = np.ones(n)

    # U_SNR — min-max normalised SNR score
    snr = np.clip([device_states[cid].snr_db for cid in eligible_ids], 0.0, 30.0)
    u_snr = _minmax(snr)

    # U_dens — building-density proxy: eligible clients within 5 km radius
    # (no building inventory exists for the study area), normalised per
    # disaster area via min-max. Vectorised O(N²) in a local metre frame.
    client_xy = _xy_metres(coords)
    if n > 1:
        diff = client_xy[:, None, :] - client_xy[None, :, :]   # (N, N, 2)
        sq_dists = (diff ** 2).sum(axis=2)                       # (N, N)
        density = (sq_dists < 5_000.0 ** 2).sum(axis=1) - 1.0   # exclude self
    else:
        density = np.zeros(n)
    u_dens = _minmax(density)

    # U_prox — proximity to nearest UAV (paper: max(0, 1 − d_min / R_comm)).
    if uav_coords_latlon:
        prox_dists = np.array([
            min(haversine(c, ul) for ul in uav_coords_latlon) for c in coords
        ])
        u_prox = np.clip(1.0 - prox_dists / max(R_comm, 1e-9), 0.0, 1.0)
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
        # Round of each client's most recent selection (0 = never selected).
        # Maintained for every mode; consumed by the fair_mab staleness term.
        self._last_selected: dict[int, int] = {cid: 0 for cid in client_ids}
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
        mode: str = "ucb",       # "ucb" | "random" | "all" | "fedcs" | "rep_cap" | "fair_mab"
        rng: np.random.Generator | None = None,
        R_comm: float = DEFAULT_R_COMM_M,
        t_stale_cap: int = DEFAULT_T_STALE_CAP,
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
            selected = self._random_select(eligible, uav_capacity, round_num, rng=rng)
            return self._record_selection(selected, round_num)

        eligible_ids = list(eligible.keys())

        # ── Literature baselines (Algorithms B1-B3) ─────────────────────
        if mode == "fedcs":
            selected = self._fedcs_select(eligible, device_states, uav_capacity)
            return self._record_selection(selected, round_num)

        if mode in ("rep_cap", "fair_mab"):
            if mode == "rep_cap":
                scores = self._rep_cap_scores(eligible_ids, device_states, reputation_scores)
            else:
                scores = self._fair_mab_scores(
                    eligible_ids, device_states, round_num, t_stale_cap
                )
            selected = self._greedy_assign(
                eligible_ids, eligible, scores, uav_capacity,
                client_coords, uav_coords_latlon, R_comm,
            )
            return self._record_selection(selected, round_num)

        if mode != "ucb":
            raise ValueError(f"unknown selection mode: {mode!r}")

        # ── UCB pipeline ────────────────────────────────────────────────

        utility = _compute_utility(
            eligible_ids, device_states, client_coords, uav_coords_latlon,
            self._epicentre, R_comm,
        )

        batteries    = np.array([device_states[cid].battery            for cid in eligible_ids])
        compute_s    = np.array([device_states[cid].compute_time_s     for cid in eligible_ids])
        reputations  = np.array([reputation_scores.get(cid, 0.5)       for cid in eligible_ids])
        utilities    = np.array([utility.get(cid, 0.5)                 for cid in eligible_ids])

        # Paper §IV-C2: b̃ = b_n, ℓ̃ = 1 − (T̂_n/T_max)², Ũ = β·Û + (1−β)·R.
        l_feat = np.clip(1.0 - (compute_s / T_MAX_S) ** 2, 0.0, 1.0)
        beta = beta_schedule(round_num)
        u_tilde = beta * utilities + (1.0 - beta) * reputations

        priority = W_BATTERY * batteries + W_LEARNING * l_feat + W_UTILITY * u_tilde

        t = max(round_num, 1)
        sel_cnts = np.array([self._counts[cid] for cid in eligible_ids], dtype=float)
        ucb = priority + UCB_C * np.sqrt(math.log(t) / (sel_cnts + 1.0))

        selected = self._greedy_assign(
            eligible_ids, eligible, ucb, uav_capacity,
            client_coords, uav_coords_latlon, R_comm,
        )
        return self._record_selection(selected, round_num)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_selection(self, selected: dict[int, int], round_num: int) -> dict[int, int]:
        """Update last-selected bookkeeping (fair_mab staleness) and pass through."""
        for cid in selected:
            self._last_selected[cid] = round_num
        return selected

    def _fedcs_select(
        self,
        eligible: dict[int, int],
        device_states: dict[int, DeviceState],
        uav_capacity: int,
    ) -> dict[int, int]:
        """Baseline B1 — FedCS greedy deadline selection (Nishio & Yonetani 2019).

        Per UAV, cheapest-first by predicted completion time T̂_n; a candidate
        is added while the projected round time stays within T_max and the UAV
        is under capacity. Purely time-driven: no reputation, no utility.
        """
        by_uav: dict[int, list[int]] = {}
        for cid, uav in eligible.items():
            by_uav.setdefault(uav, []).append(cid)

        selected: dict[int, int] = {}
        for uav, cids in by_uav.items():
            cids.sort(key=lambda c: device_states[c].compute_time_s)
            t_proj = 0.0
            n_sel = 0
            for cid in cids:
                t_hat = device_states[cid].compute_time_s
                t_inc = max(t_hat - t_proj, 0.0)
                if t_proj + t_inc <= T_MAX_S and n_sel < uav_capacity:
                    selected[cid] = uav
                    t_proj = max(t_proj, t_hat)
                    n_sel += 1
                    self._counts[cid] += 1
                else:
                    break   # deadline would be exceeded; stop adding candidates
            # (capacity full also stops the loop; ascending order makes any
            #  later candidate at least as expensive — matches Algorithm B1)
        return selected

    def _rep_cap_scores(
        self,
        eligible_ids: list[int],
        device_states: dict[int, DeviceState],
        reputation_scores: dict[int, float],
    ) -> np.ndarray:
        """Baseline B2 — reputation-capability score (Zhao et al. 2024).

        score_n = γ·R_n + (1−γ)·ℓ̃_n with ℓ̃_n = 1 − (T̂_n/T_max)². Reuses the
        system's own reputation (Algorithm 3) so the selection rule — not the
        reputation estimator — is the experimental variable. No exploration
        term, no utility term (that omission is the point of the baseline).
        """
        reputations = np.array([reputation_scores.get(cid, 0.5) for cid in eligible_ids])
        compute_s = np.array([device_states[cid].compute_time_s for cid in eligible_ids])
        l_feat = np.clip(1.0 - (compute_s / T_MAX_S) ** 2, 0.0, 1.0)
        return REPCAP_GAMMA * reputations + (1.0 - REPCAP_GAMMA) * l_feat

    def _fair_mab_scores(
        self,
        eligible_ids: list[int],
        device_states: dict[int, DeviceState],
        round_num: int,
        t_stale_cap: int,
    ) -> np.ndarray:
        """Baseline B3 — fairness/energy MAB reward (Zhu et al. 2024).

        reward_n = w_energy·b_n + w_stale·min(1, staleness_n/T_stale_cap) with
        staleness_n = rounds since last selection. No reputation, no utility —
        the bandit explores over fairness/energy rather than data value.
        """
        batteries = np.array([device_states[cid].battery for cid in eligible_ids])
        cap = max(t_stale_cap, 1)
        staleness = np.array([
            min(1.0, (round_num - self._last_selected.get(cid, 0)) / cap)
            for cid in eligible_ids
        ])
        return FAIRMAB_W_ENERGY * batteries + FAIRMAB_W_STALE * staleness

    def _greedy_assign(
        self,
        eligible_ids: list[int],
        eligible: dict[int, int],
        scores: np.ndarray,
        uav_capacity: int,
        client_coords: dict[int, tuple[float, float]],
        uav_coords_latlon: list[tuple[float, float]],
        R_comm: float,
    ) -> dict[int, int]:
        """Paper Algorithm 4: highest-score first; each client goes to the
        feasible UAV (in range, under capacity) with the lowest current load,
        ties broken by smallest distance; skipped when no UAV is feasible.

        Without UAV positions (legacy callers / flat topologies) the client's
        pre-computed covering UAV from ``eligible`` is used instead.
        """
        order = np.argsort(-scores)
        fill: dict[int, int] = {}
        selected: dict[int, int] = {}

        # Pre-compute the (N, K) client-UAV Haversine distance matrix once.
        dist: np.ndarray | None = None
        if uav_coords_latlon:
            dist = np.array([
                [haversine(client_coords[cid], ul) for ul in uav_coords_latlon]
                for cid in eligible_ids
            ])

        for idx in order:
            cid = eligible_ids[idx]
            if dist is None:
                # Fallback: fixed covering UAV, capacity-gated.
                uav = eligible[cid]
                if fill.get(uav, 0) < uav_capacity:
                    selected[cid] = uav
                    fill[uav] = fill.get(uav, 0) + 1
                    self._counts[cid] += 1
                continue

            feasible = [
                j for j in range(len(uav_coords_latlon))
                if dist[idx, j] <= R_comm and fill.get(j, 0) < uav_capacity
            ]
            if not feasible:
                continue   # skip client — no feasible UAV (Algorithm 4)
            j_star = min(feasible, key=lambda j: (fill.get(j, 0), dist[idx, j]))
            selected[cid] = j_star
            fill[j_star] = fill.get(j_star, 0) + 1
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
