"""IoT device heterogeneity simulation for HFL client selection (paper §IV-B).

Each IoT device carries a per-round state:
  battery     — [0,1]; decays when selected, slowly recharges otherwise
  snr_db      — signal-to-noise ratio in dB; fluctuates each round
  memory_ok   — bool; whether the device has enough RAM to hold the local model
  compute_time_s — estimated local training time in seconds (straggler model)

Eligibility constants match paper Table II.
"""

from __future__ import annotations

import numpy as np

# Paper Table II eligibility thresholds
B_MIN: float = 0.20  # minimum battery fraction
SNR_MIN_DB: float = 3.0  # minimum SNR (dB)
T_MAX_S: float = 300.0  # maximum compute time (s)


class DeviceState:
    __slots__ = ("battery", "snr_db", "memory_ok", "compute_time_s", "margin_s")

    def __init__(
        self,
        battery: float,
        snr_db: float,
        memory_ok: bool,
        compute_time_s: float,
        margin_s: float = 0.0,
    ) -> None:
        self.battery = battery
        self.snr_db = snr_db
        self.memory_ok = memory_ok
        self.compute_time_s = compute_time_s
        # Adaptive safety margin ε_n(t) from historical completion-time
        # variance (paper §IV-C1: T̂_n ≤ T_max − ε_n).
        self.margin_s = margin_s

    def eligible(self) -> bool:
        return (
            self.battery >= B_MIN
            and self.snr_db >= SNR_MIN_DB
            and self.memory_ok
            and self.compute_time_s <= T_MAX_S - self.margin_s
        )


class DeviceStateManager:
    """Simulate per-round IoT device state for N heterogeneous clients.

    Initial conditions are drawn once at construction; per-round noise is
    applied via ``update_round(selected_ids)`` at the end of each FL round.

    Stress-test knobs (both default to 0.0 = exact historical behaviour):

    ``dropout_rate``
        Per-``get_state``-call probability of forcing ``memory_ok=False``,
        modelling transient connectivity loss through the existing
        four-condition eligibility gate rather than a fifth dimension.
        Callers snapshot states once per round via ``get_all_states``, so
        this reads as an i.i.d. per-(device, round) dropout draw.
    ``snr_degradation_db``
        Uniform dB subtraction from every device's SNR — an area-wide
        aftershock-triggered channel degradation, not a per-device effect.
    """

    def __init__(
        self,
        client_ids: list[int],
        rng: np.random.Generator,
        dropout_rate: float = 0.0,
        snr_degradation_db: float = 0.0,
    ) -> None:
        self._ids = list(client_ids)
        self._rng = rng
        self._dropout_rate = float(dropout_rate)
        self._snr_degradation_db = float(snr_degradation_db)

        # Initial batteries: uniform [0.5, 1.0]
        self._battery: dict[int, float] = {cid: float(rng.uniform(0.5, 1.0)) for cid in client_ids}
        # Base SNR: uniform [5, 20] dB — device-specific channel quality
        self._snr_base: dict[int, float] = {
            cid: float(rng.uniform(5.0, 20.0)) for cid in client_ids
        }
        # 10% of devices have insufficient memory (permanent constraint)
        self._memory_ok: dict[int, bool] = {cid: bool(rng.random() > 0.10) for cid in client_ids}
        # Base compute time: uniform [50, 250] s — hardware heterogeneity
        self._compute_base: dict[int, float] = {
            cid: float(rng.uniform(50.0, 250.0)) for cid in client_ids
        }
        # Per-round noise (updated each call to update_round)
        self._snr_noise: dict[int, float] = {cid: 0.0 for cid in client_ids}
        self._compute_noise: dict[int, float] = {cid: 0.0 for cid in client_ids}
        # Recent observed completion times (last 10 rounds) for the adaptive
        # eligibility margin ε_n(t) = 1.96·std (paper §IV-C1).
        self._compute_history: dict[int, list[float]] = {cid: [] for cid in client_ids}

    def update_round(self, selected_ids: set[int]) -> None:
        """Advance device states by one FL round.

        Selected devices discharge their battery; all devices experience
        channel fluctuation and straggler variance.
        """
        for cid in self._ids:
            if cid in selected_ids:
                # Active discharge: -0.02 per round (paper §IV-B)
                self._battery[cid] = max(0.0, self._battery[cid] - 0.02)
            else:
                # Passive recharge at half the discharge rate. The
                # discharge:recharge ratio sets the sustainable participating
                # fraction f via f·discharge = (1−f)·recharge → f = 1/3 of the
                # fleet, so the eligible pool rotates instead of collapsing.
                # (The pre-2026-07-18 value 0.005 gave f = 1/5: with 100-round
                # runs the fleet drained to ~20 permanently-cycling devices and
                # global accuracy decayed with the shrinking aggregate.)
                self._battery[cid] = min(1.0, self._battery[cid] + 0.01)
            self._snr_noise[cid] = float(self._rng.normal(0.0, 2.0))
            self._compute_noise[cid] = float(self._rng.normal(0.0, 30.0))
            if cid in selected_ids:
                observed = max(10.0, self._compute_base[cid] + self._compute_noise[cid])
                self._compute_history[cid] = (self._compute_history[cid] + [observed])[-10:]

    def get_state(self, client_id: int) -> DeviceState:
        history = self._compute_history.get(client_id, [])
        margin = 1.96 * float(np.std(history)) if len(history) >= 3 else 0.0
        memory_ok = self._memory_ok[client_id]
        if self._dropout_rate > 0 and self._rng.random() < self._dropout_rate:
            memory_ok = False  # transient dropout via the existing gate
        return DeviceState(
            battery=self._battery[client_id],
            snr_db=self._snr_base[client_id]
            + self._snr_noise[client_id]
            - self._snr_degradation_db,
            memory_ok=memory_ok,
            compute_time_s=max(
                10.0, self._compute_base[client_id] + self._compute_noise[client_id]
            ),
            margin_s=margin,
        )

    def get_all_states(self) -> dict[int, DeviceState]:
        return {cid: self.get_state(cid) for cid in self._ids}
