"""Reputation tracking system — Algorithm 3 from paper (§IV-D).

R_n = W_CONTRIB * R_contrib  +  W_ANOMALY * R_anomaly  +  W_TEMP * R_temp

Components
----------
R_contrib  (0.4) — contribution quality: EMA of cosine similarity between
                   this client's update vector and the round's mean update direction.
R_anomaly  (0.3) — anomaly score: 1 if update ℓ2-norm is within 2σ of the
                   current distribution, else 0.  EMA-smoothed.
R_temp     (0.3) — temporal reliability: EMA of (success_rate × consistency),
                   where consistency = 1 / (1 + update-norm CV).

All scores live in [0, 1]; initialised at 0.5 / 1.0 / 0.5 for cold starts.
"""

from __future__ import annotations

import numpy as np

W_CONTRIB: float = 0.4
W_ANOMALY: float = 0.3
W_TEMP: float = 0.3
EMA_ALPHA: float = 0.30   # recency weight


def _vec(state_dict: dict) -> np.ndarray:
    """Flatten a trainable state dict to a 1-D float32 numpy vector."""
    return np.concatenate([v.detach().cpu().numpy().ravel().astype(np.float32) for v in state_dict.values()])


class ReputationManager:
    """Maintains per-client reputation scores updated after every FL round."""

    def __init__(self, client_ids: list[int], window_size: int | None = None) -> None:
        self._R_contrib: dict[int, float] = {cid: 0.5 for cid in client_ids}
        self._R_anomaly: dict[int, float] = {cid: 1.0 for cid in client_ids}
        self._R_temp:    dict[int, float] = {cid: 0.5 for cid in client_ids}

        self._total:   dict[int, int] = {cid: 0 for cid in client_ids}
        self._success: dict[int, int] = {cid: 0 for cid in client_ids}

        # Rolling window of update ℓ2-norms (for anomaly detection). Sized to
        # hold several rounds' worth of updates so it isn't fully replaced by
        # a single round when the client pool is large (default: 10 rounds).
        self._window_size = window_size if window_size is not None else max(100, 10 * len(client_ids))
        self._norm_window: list[float] = []
        # Per-client norm history (for consistency)
        self._norm_history: dict[int, list[float]] = {cid: [] for cid in client_ids}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_batch(
        self,
        updates: dict[int, dict],          # client_id → trainable state dict
        global_update_vec: np.ndarray | None,
    ) -> None:
        """Update reputation for every client that submitted an update this round."""
        if not updates:
            return

        vecs: dict[int, np.ndarray] = {cid: _vec(sd) for cid, sd in updates.items()}
        norms = [float(np.linalg.norm(v)) for v in vecs.values()]

        # Extend the global norm window (capped at self._window_size entries)
        self._norm_window.extend(norms)
        self._norm_window = self._norm_window[-self._window_size:]
        mu = float(np.mean(self._norm_window))
        sigma = float(np.std(self._norm_window)) + 1e-8

        # Mean update direction (used when no explicit global update is passed)
        all_vecs = np.array(list(vecs.values()))
        mean_direction = all_vecs.mean(axis=0)
        direction = global_update_vec if global_update_vec is not None else mean_direction

        for cid, vec in vecs.items():
            norm = float(np.linalg.norm(vec))

            # --- R_contrib ---
            if np.linalg.norm(direction) > 1e-10 and norm > 1e-10:
                cos = float(np.dot(vec, direction) / (norm * np.linalg.norm(direction) + 1e-10))
                r_c = (cos + 1.0) / 2.0   # [−1,1] → [0,1]
            else:
                r_c = 0.5
            self._R_contrib[cid] = (1 - EMA_ALPHA) * self._R_contrib[cid] + EMA_ALPHA * r_c

            # --- R_anomaly ---
            r_a = 1.0 if abs(norm - mu) <= 2.0 * sigma else 0.0
            self._R_anomaly[cid] = (1 - EMA_ALPHA) * self._R_anomaly[cid] + EMA_ALPHA * r_a

            # --- R_temp ---
            self._total[cid] += 1
            self._success[cid] += 1
            success_rate = self._success[cid] / max(self._total[cid], 1)
            self._norm_history[cid].append(norm)
            recent_norms = self._norm_history[cid][-10:]
            if len(recent_norms) >= 3:
                cv = float(np.std(recent_norms)) / (float(np.mean(recent_norms)) + 1e-8)
                consistency = 1.0 / (1.0 + cv)
            else:
                consistency = 0.5
            r_t = success_rate * consistency
            self._R_temp[cid] = (1 - EMA_ALPHA) * self._R_temp[cid] + EMA_ALPHA * r_t

    def mark_absent(self, client_id: int) -> None:
        """Call when an eligible client failed to return an update (straggler/dropout)."""
        self._total[client_id] += 1   # success unchanged → rate decreases
        success_rate = self._success[client_id] / max(self._total[client_id], 1)
        r_t = success_rate * 0.5      # penalise consistency
        self._R_temp[client_id] = (1 - EMA_ALPHA) * self._R_temp[client_id] + EMA_ALPHA * r_t

    def get_score(self, client_id: int) -> float:
        """Aggregate reputation R_n ∈ [0, 1]."""
        return (
            W_CONTRIB * self._R_contrib[client_id]
            + W_ANOMALY * self._R_anomaly[client_id]
            + W_TEMP    * self._R_temp[client_id]
        )

    def get_all_scores(self) -> dict[int, float]:
        return {cid: self.get_score(cid) for cid in self._R_contrib}
