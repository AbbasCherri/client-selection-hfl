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
"class_greedy" — **the decisive 2026-08 baseline**: the identical per-client
             class histogram the proposed selector receives, consumed by a
             plain submodular class-coverage greedy and nothing else (no
             utility, no reputation, no beta blend, no UCB bonus, no Gumbel
             roster sampling, no static-priority blend). Implemented as
             ``_class_coverage_assign`` with ``static_blend=gumbel_scale=0``,
             so it cannot drift from the proposed roster builder. Separates
             "class-awareness helps" (prior art) from "our pipeline helps".
"ucb_noclass" — the full priority/UCB pipeline with the class histogram
             withheld; the lower anchor of the oracle-degradation ladder.
"random"   — eligibility filter only, then random draw per UAV (hfl_no_selection)
"all"      — every ELIGIBLE covered client participates (flat_fl). Eligibility
             still applies: a device with a dead battery, no memory, or a
             below-threshold channel cannot deliver an update regardless of
             topology, so exempting flat_fl from the device physics would
             hand it a free upper bound (and made it invariant to the
             dropout/SNR stress knobs — the 2026-07-18 fix).

Literature baselines (Algorithms B1-B5, REPORTS/master_implementation_reference.md Appendix C)
------------------------------------------------------------------------
"fedcs"    — B1: FedCS greedy deadline selection per UAV, purely time-driven
             (Nishio & Yonetani, ICC 2019)
"rep_cap"  — B2: γ·R_n + (1−γ)·ℓ̃_n reputation-capability ranking, no
             exploration, no utility (Zhao et al., Chin. J. Aeronaut. 2024)
"fair_mab" — B3: w_e·b_n + w_s·staleness_n fairness/energy MAB reward,
             no reputation, no utility (Zhu et al., Sensors 2024)
"oort"     — B4: Oort guided participant selection (Lai et al., OSDI 2021):
             statistical utility (last-observed local training loss) ×
             system-speed penalty (T_max/T̂_n)^α for stragglers. Losses are
             the last-observed values, exactly as in Oort's own stale-utility
             design; never-trained clients get the current max utility so
             they are explored first.
"power_of_choice" — B5: Power-of-Choice (Cho et al., 2020 / AISTATS 2022):
             draw d = 2·(slots) candidates uniformly from the eligible pool,
             keep the highest last-observed local loss. The canonical rule
             evaluates the current global loss on the candidate set; the
             simulation uses last-observed local losses (the standard
             cached-loss variant in FL benchmarks) so no extra forward
             passes are billed to the baseline.

All five run behind the same eligibility gate as "ucb"; all except fedcs
rank candidates and feed the identical greedy UAV assignment (Algorithm 4),
so the selection rule is the only experimental variable.
"""

from __future__ import annotations

import math

import numpy as np

from hflsim.shared.coords import haversine_matrix
from hflsim.shared.value import beta_schedule

from .device_state import T_MAX_S, DeviceState

# Paper §IV-C priority weights
W_BATTERY = 0.35
W_LEARNING = 0.30
W_UTILITY = 0.35

# Proposed (ucb) priority with battery removed from the *score*. Battery stays
# a hard eligibility gate; scoring by it made UCB drain the highest-charge
# cohort in lockstep and then swap wholesale to the rested cohort every T_sel —
# the non-IID ping-pong (highest volatility of all methods, worse than random
# on macro-F1). Learning-capability + class-coverage utility carry the
# reweighted mass (summing to 1) — values from the 2026-07-20 Optuna weight
# search (scripts/tune_weights.py, 120 trials, real data; +22% over the prior
# hand-picked 0.4/0.6 split on the search's objective). Not re-derived from
# the paper; see that script's docstring for the search protocol.
W_LEARNING_NB = 0.702
W_UTILITY_NB = 0.298

# Class-coverage roster construction (proposed system, Tier C). Roster building
# is a submodular class-coverage maximization: value(S) = Σ_c scarcity_c·√(Σ_{i∈S}
# count_c(i)), whose √ gives diminishing returns per class (majority saturates
# fast, scarce classes keep paying), so greedy is near-optimal. The static
# priority and a Gumbel perturbation are blended in (Gumbel-top-k = sampling
# clients ∝ score without replacement), replacing the deterministic argsort that
# locked the roster in place. Values from the 2026-07-20 Optuna search (see
# W_LEARNING_NB above) — the search pushed SEL_GUMBEL_SCALE near the top of its
# searched range, i.e. more roster randomness helps more than intuition suggested.
SEL_STATIC_BLEND = 0.435   # weight of normalized static priority vs coverage gain
SEL_GUMBEL_SCALE = 1.475   # stochasticity of the roster (0 → deterministic greedy)
SEL_MIN_MARGINAL = 0.0   # stop a UAV early below this normalized gain (0 → fill capacity)

# Utility sub-weights (§IV-C3 states 0.4/0.3/0.2/0.1; these are the 2026-07-20
# Optuna-searched values instead — see W_LEARNING_NB above. Deliberately
# deviates from the paper text: the search found UAV-proximity dominant and
# epicentre-distance/SNR nearly irrelevant, the opposite emphasis from the
# paper's stated split.
W_EPI = 0.043
W_SNR = 0.078
W_DENS = 0.295
W_PROX = 0.584

UCB_C = math.sqrt(2)  # exploration constant from paper

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

# Multiplier on T_sel for that cap. **1 makes the staleness term inert**, and 1
# is what every result before 2026-08-04 used.
#
# Selection happens every T_sel rounds, so by the next selection event a client
# picked at the previous one already has staleness (T_sel)/(T_sel) = 1, and the
# min(1, ·) clamp pins a never-picked client to 1 as well. Every client scores
# exactly 1.0, the term becomes a constant offset, and since argsort ignores a
# positive rescaling the reward collapses to battery order — B3 degenerates to
# "highest battery first" and is invariant to w_energy/w_stale. Measured: the
# four fair_mab_* arms of the 0.3 sweep returned bit-identical means AND stds
# across 10 seeds.
#
# Left at 1 so historical results stay reproducible; the 0.3 sweep varies it and
# reports the baseline at its best setting (REPORTS/rigor_plan_2026-08.md §0.3).
# A cap spanning several reselection intervals is what makes the fairness half
# of the reward discriminate at all.
FAIRMAB_STALE_CAP_MULT = 1

# Baseline B4 (Oort, Lai et al. OSDI 2021): system-utility penalty exponent.
# util_n = stat_n · (T_pref/T̂_n)^α for stragglers (T̂_n > T_pref), stat_n
# otherwise. Oort leaves the developer-preferred round duration T_pref open;
# here it is the median compute time of the currently eligible pool (T_max
# would never fire — eligibility already caps T̂_n ≤ T_max − ε). α = 2 is the
# paper's default.
OORT_ALPHA = 2.0
# T_pref quantile of the eligible pool's compute times. Oort leaves T_pref to
# the developer (Appendix C.4 note 1), so unlike alpha this is OUR choice and
# is swept in the 0.3 provenance block rather than asserted.
OORT_TPREF_Q = 0.5  # median

# Power-of-Choice candidate-set multiplier: d = POC_D_MULT x (total slots).
# Cho et al. sweep d rather than fixing it, so sweeping it is the faithful
# reproduction; 2 was previously hard-coded and described as "standard".
POC_D_MULT = 2

# Noto Peninsula 2024 epicentre (default; override via cfg)
DEFAULT_EPICENTRE = (37.488, 137.272)  # (lat °N, lon °E)

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
    epi_dists = haversine_matrix(np.asarray(coords), np.asarray([epicentre]))[:, 0]
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
        diff = client_xy[:, None, :] - client_xy[None, :, :]  # (N, N, 2)
        sq_dists = (diff**2).sum(axis=2)  # (N, N)
        density = (sq_dists < 5_000.0**2).sum(axis=1) - 1.0  # exclude self
    else:
        density = np.zeros(n)
    u_dens = _minmax(density)

    # U_prox — proximity to nearest UAV (paper: max(0, 1 − d_min / R_comm)).
    if uav_coords_latlon:
        prox_dists = haversine_matrix(np.asarray(coords), np.asarray(uav_coords_latlon)).min(axis=1)
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
        seed: int = 0,
    ) -> None:
        # Dedicated stochastic-roster RNG (Gumbel-top-k), seeded from the run's
        # per-method seed. Kept separate from the shared placement/device RNG so
        # the selection sampler does not perturb the environment simulation
        # (battery/SNR trajectories) — decoupling that also keeps every non-ucb
        # baseline's device physics byte-identical to a deterministic-greedy run.
        self._sel_rng = np.random.default_rng(seed)
        self._counts: dict[int, int] = {cid: 0 for cid in client_ids}
        # Round of each client's most recent selection (0 = never selected).
        # Maintained for every mode; consumed by the fair_mab staleness term.
        self._last_selected: dict[int, int] = {cid: 0 for cid in client_ids}
        # Last-observed local training loss per client (oort/power_of_choice
        # statistical utility). Harnesses push these via update_losses() after
        # each training round; None = never trained (gets the exploration
        # prior: current max observed loss).
        self._last_loss: dict[int, float] = {}
        self._epicentre = epicentre or DEFAULT_EPICENTRE

    def update_losses(self, losses: dict[int, float]) -> None:
        """Record the mean local training loss of the clients that trained
        this round (consumed by the oort / power_of_choice baselines)."""
        for cid, loss in losses.items():
            self._last_loss[cid] = float(loss)

    def select(
        self,
        covered: dict[int, int],  # {client_id: uav_idx}
        device_states: dict[int, DeviceState],
        reputation_scores: dict[int, float],
        client_coords: dict[int, tuple[float, float]],
        uav_coords_latlon: list[tuple[float, float]],
        round_num: int,
        uav_capacity: int,
        mode: str = "ucb",  # "ucb" | "random" | "all" | "fedcs" | "rep_cap" | "fair_mab"
        rng: np.random.Generator | None = None,
        R_comm: float = DEFAULT_R_COMM_M,
        t_stale_cap: int = DEFAULT_T_STALE_CAP,
        class_counts: dict[int, np.ndarray] | None = None,
        class_scarcity: np.ndarray | None = None,
    ) -> dict[int, int]:
        """Return {client_id: uav_idx} for the clients selected this round.

        ``class_counts`` (per-client 4-bin label histogram) and
        ``class_scarcity`` (inverse-prior class weights) drive the proposed
        ``ucb`` mode's class-coverage roster; every other mode ignores them, so
        the selection rule stays the only experimental variable.
        """
        # ── Eligibility gate (every mode — see module docstring on "all") ──
        eligible: dict[int, int] = {}
        for cid, uav_idx in covered.items():
            st = device_states.get(cid)
            if st is not None and st.eligible():
                eligible[cid] = uav_idx

        if not eligible:
            return {}

        if mode == "all":
            return self._record_selection(dict(eligible), round_num)

        if mode == "random":
            selected = self._random_select(eligible, uav_capacity, round_num, rng=rng)
            return self._record_selection(selected, round_num)

        eligible_ids = list(eligible.keys())

        # ── Literature baselines (Algorithms B1-B3) ─────────────────────
        if mode == "fedcs":
            selected = self._fedcs_select(eligible, device_states, uav_capacity)
            return self._record_selection(selected, round_num)

        if mode == "power_of_choice":
            # B5: uniform candidate draw, then rank by loss. The candidate
            # subset d = 2·(total slots) approximated as 2·capacity per
            # covering UAV, bounded by the pool size.
            _rng = rng if rng is not None else np.random.default_rng(round_num * 6271)
            n_uavs = len(set(eligible.values()))
            d = min(len(eligible_ids), max(1, POC_D_MULT * uav_capacity * max(n_uavs, 1)))
            candidate_ids = [
                eligible_ids[i]
                for i in _rng.choice(len(eligible_ids), size=d, replace=False)
            ]
            scores = self._loss_scores(candidate_ids)
            selected = self._greedy_assign(
                candidate_ids,
                eligible,
                scores,
                uav_capacity,
                client_coords,
                uav_coords_latlon,
                R_comm,
            )
            return self._record_selection(selected, round_num)

        if mode in ("rep_cap", "fair_mab", "oort"):
            if mode == "rep_cap":
                scores = self._rep_cap_scores(eligible_ids, device_states, reputation_scores)
            elif mode == "oort":
                scores = self._oort_scores(eligible_ids, device_states)
            else:
                scores = self._fair_mab_scores(eligible_ids, device_states, round_num, t_stale_cap)
            selected = self._greedy_assign(
                eligible_ids,
                eligible,
                scores,
                uav_capacity,
                client_coords,
                uav_coords_latlon,
                R_comm,
            )
            return self._record_selection(selected, round_num)

        if mode == "class_greedy":
            # The decisive baseline (2026-08). Receives the *identical*
            # class_counts the proposed selector gets, and nothing else: no
            # utility, no reputation, no beta blend, no UCB bonus, no Gumbel
            # roster sampling, no static-priority blend. If `ucb` cannot beat
            # this, the contribution is class-awareness — which is prior art —
            # and not the UCB pipeline built on top of it.
            if class_counts is None or class_scarcity is None:
                raise ValueError(
                    "mode='class_greedy' requires class_counts and class_scarcity; "
                    "without them it is not a class-aware baseline at all"
                )
            selected = self._class_coverage_assign(
                eligible_ids,
                eligible,
                np.zeros(len(eligible_ids)),  # no priority signal whatsoever
                class_counts,
                class_scarcity,
                uav_capacity,
                client_coords,
                uav_coords_latlon,
                R_comm,
                static_blend=0.0,
                gumbel_scale=0.0,
            )
            return self._record_selection(selected, round_num)

        if mode not in ("ucb", "ucb_noclass"):
            raise ValueError(f"unknown selection mode: {mode!r}")

        # ── UCB pipeline ────────────────────────────────────────────────

        utility = _compute_utility(
            eligible_ids,
            device_states,
            client_coords,
            uav_coords_latlon,
            self._epicentre,
            R_comm,
        )

        compute_s = np.array([device_states[cid].compute_time_s for cid in eligible_ids])
        reputations = np.array([reputation_scores.get(cid, 0.5) for cid in eligible_ids])
        utilities = np.array([utility.get(cid, 0.5) for cid in eligible_ids])

        # Paper §IV-C2 with battery dropped from the score (eligibility gate only):
        # ℓ̃ = 1 − (T̂_n/T_max)², Ũ = β·Û + (1−β)·R.
        l_feat = np.clip(1.0 - (compute_s / T_MAX_S) ** 2, 0.0, 1.0)
        beta = beta_schedule(round_num)
        u_tilde = beta * utilities + (1.0 - beta) * reputations

        priority = W_LEARNING_NB * l_feat + W_UTILITY_NB * u_tilde

        t = max(round_num, 1)
        sel_cnts = np.array([self._counts[cid] for cid in eligible_ids], dtype=float)
        static = priority + UCB_C * np.sqrt(math.log(t) / (sel_cnts + 1.0))

        selected = self._class_coverage_assign(
            eligible_ids,
            eligible,
            static,
            # ucb_noclass: the full priority/UCB pipeline with the class
            # histogram withheld — the lower anchor of the oracle-degradation
            # ladder. Falls through to the plain priority greedy.
            None if mode == "ucb_noclass" else class_counts,
            None if mode == "ucb_noclass" else class_scarcity,
            uav_capacity,
            client_coords,
            uav_coords_latlon,
            R_comm,
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
                    break  # deadline would be exceeded; stop adding candidates
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
        staleness = np.array(
            [min(1.0, (round_num - self._last_selected.get(cid, 0)) / cap) for cid in eligible_ids]
        )
        return FAIRMAB_W_ENERGY * batteries + FAIRMAB_W_STALE * staleness

    def _loss_scores(self, ids: list[int]) -> np.ndarray:
        """Last-observed local losses; never-trained clients get the current
        max observed loss (exploration prior — matches Oort's init-to-max)."""
        observed = [self._last_loss[c] for c in ids if c in self._last_loss]
        prior = max(observed) if observed else 1.0
        return np.array([self._last_loss.get(c, prior) for c in ids])

    def _oort_scores(
        self,
        eligible_ids: list[int],
        device_states: dict[int, DeviceState],
    ) -> np.ndarray:
        """Baseline B4 — Oort utility (Lai et al., OSDI 2021).

        util_n = stat_n · (T_pref/T̂_n)^α for stragglers, stat_n otherwise,
        with stat_n the last-observed local loss and T_pref the median
        compute time of the eligible pool (see OORT_ALPHA note).
        """
        stat = self._loss_scores(eligible_ids)
        compute_s = np.array([device_states[cid].compute_time_s for cid in eligible_ids])
        t_pref = float(np.quantile(compute_s, OORT_TPREF_Q))
        penalty = np.where(
            compute_s > t_pref,
            (t_pref / np.maximum(compute_s, 1e-9)) ** OORT_ALPHA,
            1.0,
        )
        return stat * penalty

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
            dist = haversine_matrix(
                np.asarray([client_coords[cid] for cid in eligible_ids]),
                np.asarray(uav_coords_latlon),
            )

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
                j
                for j in range(len(uav_coords_latlon))
                if dist[idx, j] <= R_comm and fill.get(j, 0) < uav_capacity
            ]
            if not feasible:
                continue  # skip client — no feasible UAV (Algorithm 4)
            j_star = min(feasible, key=lambda j: (fill.get(j, 0), dist[idx, j]))
            selected[cid] = j_star
            fill[j_star] = fill.get(j_star, 0) + 1
            self._counts[cid] += 1
        return selected

    def _class_coverage_assign(
        self,
        eligible_ids: list[int],
        eligible: dict[int, int],
        static: np.ndarray,
        class_counts: dict[int, np.ndarray] | None,
        class_scarcity: np.ndarray | None,
        uav_capacity: int,
        client_coords: dict[int, tuple[float, float]],
        uav_coords_latlon: list[tuple[float, float]],
        R_comm: float,
        static_blend: float | None = None,
        gumbel_scale: float | None = None,
    ) -> dict[int, int]:
        """Proposed roster construction: per-UAV submodular class-coverage greedy.

        For each UAV, fill slots by repeatedly adding the feasible unassigned
        client that maximizes

            marginal_coverage_gain / gain₀  +  blend·statiĉ  +  scale·Gumbel

        where the coverage value of a roster is Σ_c scarcity_c·√(Σ count_c) (√ →
        diminishing returns per class → submodular → greedy near-optimal),
        ``gain₀`` normalizes it to the first-pick scale, ``statiĉ`` is the
        min-max-normalized learning/utility/UCB priority, and the per-client
        Gumbel term makes the roster a sample rather than a fixed argsort. Falls
        back to the plain priority greedy when class information is absent.

        ``static_blend`` / ``gumbel_scale`` default to the module constants
        (the proposed system). Setting **both to 0.0** strips the priority and
        stochastic terms, leaving a pure submodular class-coverage greedy —
        that is exactly the ``class_greedy`` baseline, so the ablation differs
        from the proposed selector only in these two coefficients rather than
        being a separate implementation that could drift.
        """
        if class_counts is None or class_scarcity is None:
            return self._greedy_assign(
                eligible_ids, eligible, static, uav_capacity,
                client_coords, uav_coords_latlon, R_comm,
            )

        blend = SEL_STATIC_BLEND if static_blend is None else float(static_blend)
        g_scale = SEL_GUMBEL_SCALE if gumbel_scale is None else float(gumbel_scale)

        _rng = self._sel_rng  # decoupled from the shared placement/device RNG
        n = len(eligible_ids)
        scarcity = np.asarray(class_scarcity, dtype=np.float64)
        n_cls = scarcity.shape[0]
        counts = np.array(
            [np.asarray(class_counts.get(cid, np.zeros(n_cls)), dtype=np.float64) for cid in eligible_ids]
        )

        static_n = _minmax(static)
        # The Gumbel draw is taken even when g_scale == 0 so that every mode
        # consumes the selection RNG identically — otherwise class_greedy would
        # desynchronise the stream and stop being seed-comparable to ucb.
        gumbel = -np.log(-np.log(_rng.uniform(1e-12, 1.0, size=n)))
        perturbed_static = blend * static_n + g_scale * gumbel

        dist: np.ndarray | None = None
        if uav_coords_latlon:
            dist = haversine_matrix(
                np.asarray([client_coords[cid] for cid in eligible_ids]),
                np.asarray(uav_coords_latlon),
            )
            n_uav = len(uav_coords_latlon)
        else:
            n_uav = (max(eligible.values()) + 1) if eligible else 0

        def cov(acc: np.ndarray) -> float:
            return float(np.dot(scarcity, np.sqrt(acc)))

        def cov_rows(rows: np.ndarray) -> np.ndarray:
            """Per-row ``cov`` for a ``(m, n_cls)`` block, in one pass.

            ``(x * scarcity).sum(axis=1)`` reproduces ``np.dot(scarcity, x)``
            element-for-element at these widths — unlike ``rows @ scarcity``,
            which routes through BLAS and reassociates the sum. Verified
            bit-identical over 300 randomized trials; ~7x faster than calling
            ``cov`` once per candidate, which this loop did O(K · capacity · N)
            times per round.
            """
            return (np.sqrt(rows) * scarcity).sum(axis=1)

        selected: dict[int, int] = {}
        assigned: set[int] = set()
        for j in range(n_uav):
            if dist is not None:
                cand = [i for i in range(n) if i not in assigned and dist[i, j] <= R_comm]
            else:
                cand = [
                    i for i in range(n)
                    if i not in assigned and eligible[eligible_ids[i]] == j
                ]
            if not cand:
                continue
            acc = np.zeros(n_cls)
            base = cov(acc)
            cand_arr = np.fromiter(cand, dtype=np.intp, count=len(cand))
            gain0 = float((cov_rows(counts[cand_arr]) - base).max()) or 1.0
            for slot in range(uav_capacity):
                if not cand:
                    break
                cand_arr = np.fromiter(cand, dtype=np.intp, count=len(cand))
                mg_all = (cov_rows(acc + counts[cand_arr]) - base) / gain0
                # argmax takes the first maximum, matching the strict `>` scan
                # this replaces (which kept the earliest candidate on a tie).
                k = int((mg_all + perturbed_static[cand_arr]).argmax())
                best_i, best_mg = int(cand_arr[k]), float(mg_all[k])
                if slot > 0 and best_mg < SEL_MIN_MARGINAL:
                    break  # only redundant coverage remains → stop (energy-aware)
                cid = eligible_ids[best_i]
                selected[cid] = j
                assigned.add(best_i)
                cand.remove(best_i)
                acc = acc + counts[best_i]
                base = cov(acc)
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
