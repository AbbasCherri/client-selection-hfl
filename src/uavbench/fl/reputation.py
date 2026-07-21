"""Reputation tracking system — Algorithm 3 from paper (§IV-C4).

R_n = w_contrib * R_contrib  +  w_anomaly * R_anomaly  +  w_temp * R_temp

Components (paper formulas)
---------------------------
R_contrib  (0.4) — contribution quality: the client's update-delta EMA is
                   advanced as  Δw̄_n(t) = 0.7·Δw_n(t) + 0.3·Δw̄_n(t−1)  and
                   R_contrib = (1 + cos(Δw̄_n(t), Δw̄_n(t−1))) / 2.
R_anomaly  (0.3) — Mahalanobis anomaly: per-parameter mean/variance maintained
                   globally across all clients' updates (diagonal covariance);
                   R_anomaly = 1 if d ≤ 2 else exp(−0.5·(d − 2)).
                   ``d`` is the RMS standardized deviation (Mahalanobis distance
                   divided by √J for J parameters) so the paper's threshold of 2
                   is dimension-independent.
R_temp     (0.3) — temporal reliability: 0.5·success_rate + 0.5/(1 + σ_RT),
                   where σ_RT is the variance of the client's recent response
                   times (seconds).

The component weights start at the prior below and are adapted every 10 rounds
via a Dirichlet-posterior update (paper §IV-C4 "Bayesian inference"): each
component's score on a delivering client counts as evidence that the component
predicted reliability correctly; (1 − score) counts as evidence on an absent
client. Weights are the normalized posterior concentrations.

The starting prior is the 2026-07-20 Optuna weight search's result (see
scripts/tune_weights.py), not the paper's stated (0.4, 0.3, 0.3) — the search
found temporal reliability a much stronger prior than contribution quality or
anomaly detection. Deliberately deviates from §IV-C4's stated split.
"""

from __future__ import annotations

import numpy as np

W_CONTRIB: float = 0.091
W_ANOMALY: float = 0.134
W_TEMP: float = 0.775
EMA_ALPHA: float = 0.30  # legacy score-smoothing constant (kept for back-compat)

# EMA on the update *vector* (paper Algorithm 3: 0.7 new / 0.3 old).
_VEC_EMA_NEW: float = 0.7
_VEC_EMA_OLD: float = 0.3
# EMA rate for the global per-parameter mean/variance statistics.
_STATS_ALPHA: float = 0.1
# Weight adaptation cadence (rounds).
_ADAPT_EVERY: int = 10
# Dirichlet prior concentration (pseudo-counts backing the initial weights).
_PRIOR_STRENGTH: float = 20.0

_EPS = 1e-12


def trimmed_mean(values: list[float] | np.ndarray, trim: float = 0.1) -> float:
    """Mean after removing exactly ``floor(n*trim)`` values from each tail.

    Matches paper §IV-C7: trim 10% from each end; for small n where
    ``floor(n*trim) == 0`` no trimming occurs (plain mean).
    """
    arr = np.sort(np.asarray(values, dtype=np.float64))
    n = arr.shape[0]
    if n == 0:
        return 0.0
    k = int(np.floor(n * trim))
    if n - 2 * k >= 1 and k > 0:
        arr = arr[k : n - k]
    return float(arr.mean())


def _vec(state_dict: dict) -> np.ndarray:
    """Flatten a trainable state dict to a 1-D float32 numpy vector."""
    return np.concatenate(
        [v.detach().cpu().numpy().ravel().astype(np.float32) for v in state_dict.values()]
    )


class ReputationManager:
    """Maintains per-client reputation scores updated after every FL round.

    ``update_batch`` expects update *deltas* (local minus global trainable
    weights), not absolute weights — cosine similarity between absolute weight
    vectors is ≈ 1 for every client because the shared global initialization
    dominates.
    """

    def __init__(self, client_ids: list[int], window_size: int | None = None) -> None:
        self._R_contrib: dict[int, float] = {cid: 0.5 for cid in client_ids}
        self._R_anomaly: dict[int, float] = {cid: 1.0 for cid in client_ids}
        self._R_temp: dict[int, float] = {cid: 0.5 for cid in client_ids}

        self._total: dict[int, int] = {cid: 0 for cid in client_ids}
        self._success: dict[int, int] = {cid: 0 for cid in client_ids}

        # Rolling window of update ℓ2-norms (diagnostics / back-compat). Sized to
        # hold several rounds' worth of updates so it isn't fully replaced by
        # a single round when the client pool is large (default: 10 rounds).
        self._window_size = (
            window_size if window_size is not None else max(100, 10 * len(client_ids))
        )
        self._norm_window: list[float] = []
        # Per-client norm history (diagnostics)
        self._norm_history: dict[int, list[float]] = {cid: [] for cid in client_ids}

        # Per-client EMA of the update delta (paper Δw̄_n), and the cached
        # float64 norm of that stored EMA (recomputing it next round on the
        # unchanged array returns the identical value — cache is bit-exact).
        self._update_ema: dict[int, np.ndarray] = {}
        self._update_ema_norm: dict[int, float] = {}
        # Global per-parameter statistics across all clients' updates.
        self._param_mean: np.ndarray | None = None
        self._param_var: np.ndarray | None = None
        # Per-client response-time history (seconds, last 10 rounds).
        self._rt_history: dict[int, list[float]] = {cid: [] for cid in client_ids}

        # Adaptive component weights (Dirichlet posterior over (contrib, anomaly, temp)).
        self._weights = np.array([W_CONTRIB, W_ANOMALY, W_TEMP], dtype=np.float64)
        self._prior = self._weights * _PRIOR_STRENGTH
        self._evidence = np.zeros(3, dtype=np.float64)
        self._round_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_batch(
        self,
        updates: dict[int, dict],  # client_id → trainable update-delta state dict
        global_update_vec: np.ndarray | None,
        response_times: dict[int, float] | None = None,
    ) -> None:
        """Update reputation for every client that submitted an update this round.

        ``global_update_vec`` is accepted for back-compat but unused: the paper
        scores contribution against the client's *own* update history.
        """
        if not updates:
            return

        # One float32 norm and one float64 conversion per client, computed
        # up front and reused — the previous version recomputed the same
        # norm twice and ran .astype(float64) three times per client.
        vecs: dict[int, np.ndarray] = {cid: _vec(sd) for cid, sd in updates.items()}
        norms32: dict[int, float] = {cid: float(np.linalg.norm(v)) for cid, v in vecs.items()}
        v64s: dict[int, np.ndarray] = {cid: v.astype(np.float64) for cid, v in vecs.items()}

        # Keep the global norm window (capped at self._window_size entries).
        self._norm_window.extend(norms32.values())
        self._norm_window = self._norm_window[-self._window_size :]

        # Global per-parameter mean/variance across all clients' updates (EMA).
        for v64 in v64s.values():
            if self._param_mean is None:
                self._param_mean = v64.copy()
                self._param_var = np.zeros_like(v64)
            else:
                diff = v64 - self._param_mean
                self._param_mean += _STATS_ALPHA * diff
                self._param_var = (1.0 - _STATS_ALPHA) * (
                    self._param_var + _STATS_ALPHA * diff * diff
                )

        # _param_mean/_param_var are final for this round from here on, so the
        # per-parameter std is a loop invariant — hoist the full-length sqrt
        # out of the per-client loop.
        std = np.sqrt(self._param_var + _EPS)

        for cid, vec in vecs.items():
            norm = norms32[cid]
            v64 = v64s[cid]

            # --- R_contrib: cos(Δw̄_n(t), Δw̄_n(t−1)) with 0.7/0.3 vector EMA ---
            prev_ema = self._update_ema.get(cid)
            if prev_ema is None:
                self._update_ema[cid] = v64.copy()
                self._update_ema_norm.pop(cid, None)  # lazily recomputed next round
                r_c = 0.5  # cold start: no history to compare against
            else:
                new_ema = _VEC_EMA_NEW * v64 + _VEC_EMA_OLD * prev_ema
                new_norm = np.linalg.norm(new_ema)
                prev_norm = self._update_ema_norm.get(cid)
                if prev_norm is None:
                    prev_norm = np.linalg.norm(prev_ema)
                denom = new_norm * prev_norm
                if denom > _EPS:
                    cos = float(np.dot(new_ema, prev_ema) / denom)
                    r_c = (np.clip(cos, -1.0, 1.0) + 1.0) / 2.0
                else:
                    r_c = 0.5
                self._update_ema[cid] = new_ema
                self._update_ema_norm[cid] = float(new_norm)
            self._R_contrib[cid] = float(r_c)

            # --- R_anomaly: diagonal Mahalanobis vs global per-parameter stats ---
            z = (v64 - self._param_mean) / std
            d = float(np.sqrt(np.mean(z * z)))  # Mahalanobis / sqrt(J)
            r_a = 1.0 if d <= 2.0 else float(np.exp(-0.5 * (d - 2.0)))
            self._R_anomaly[cid] = r_a

            # --- R_temp: 0.5·success_rate + 0.5/(1 + σ_RT) ---
            self._total[cid] += 1
            self._success[cid] += 1
            if response_times is not None and cid in response_times:
                self._rt_history[cid].append(float(response_times[cid]))
                self._rt_history[cid] = self._rt_history[cid][-10:]
            self._R_temp[cid] = self._temporal_score(cid)

            self._norm_history[cid].append(norm)
            self._norm_history[cid] = self._norm_history[cid][-10:]

        # --- Bayesian weight adaptation every 10 rounds ---
        self._round_count += 1
        for cid in vecs:
            self._evidence += np.array(
                [self._R_contrib[cid], self._R_anomaly[cid], self._R_temp[cid]]
            )
        if self._round_count % _ADAPT_EVERY == 0:
            posterior = self._prior + self._evidence
            self._weights = posterior / posterior.sum()

    def mark_absent(self, client_id: int) -> None:
        """Call when an eligible client failed to return an update (straggler/dropout)."""
        self._total[client_id] += 1  # success unchanged → rate decreases
        self._R_temp[client_id] = self._temporal_score(client_id)
        # Absence is evidence *against* components that scored this client highly.
        self._evidence += 1.0 - np.array(
            [self._R_contrib[client_id], self._R_anomaly[client_id], self._R_temp[client_id]]
        )

    def get_score(self, client_id: int) -> float:
        """Aggregate reputation R_n ∈ [0, 1]."""
        w = self._weights
        return float(
            w[0] * self._R_contrib[client_id]
            + w[1] * self._R_anomaly[client_id]
            + w[2] * self._R_temp[client_id]
        )

    def get_all_scores(self) -> dict[int, float]:
        return {cid: self.get_score(cid) for cid in self._R_contrib}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _temporal_score(self, cid: int) -> float:
        success_rate = self._success[cid] / max(self._total[cid], 1)
        rts = self._rt_history.get(cid, [])
        sigma_rt = float(np.var(rts)) if len(rts) >= 2 else 0.0
        return 0.5 * success_rate + 0.5 / (1.0 + sigma_rt)
