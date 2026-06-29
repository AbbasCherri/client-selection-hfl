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
B_MIN: float = 0.20       # minimum battery fraction
SNR_MIN_DB: float = 3.0   # minimum SNR (dB)
T_MAX_S: float = 300.0    # maximum compute time (s)


class DeviceState:
    __slots__ = ("battery", "snr_db", "memory_ok", "compute_time_s")

    def __init__(
        self,
        battery: float,
        snr_db: float,
        memory_ok: bool,
        compute_time_s: float,
    ) -> None:
        self.battery = battery
        self.snr_db = snr_db
        self.memory_ok = memory_ok
        self.compute_time_s = compute_time_s

    def eligible(self) -> bool:
        return (
            self.battery >= B_MIN
            and self.snr_db >= SNR_MIN_DB
            and self.memory_ok
            and self.compute_time_s <= T_MAX_S
        )


class DeviceStateManager:
    """Simulate per-round IoT device state for N heterogeneous clients.

    Initial conditions are drawn once at construction; per-round noise is
    applied via ``update_round(selected_ids)`` at the end of each FL round.
    """

    def __init__(self, client_ids: list[int], rng: np.random.Generator) -> None:
        self._ids = list(client_ids)
        self._rng = rng

        # Initial batteries: uniform [0.5, 1.0]
        self._battery: dict[int, float] = {
            cid: float(rng.uniform(0.5, 1.0)) for cid in client_ids
        }
        # Base SNR: uniform [5, 20] dB — device-specific channel quality
        self._snr_base: dict[int, float] = {
            cid: float(rng.uniform(5.0, 20.0)) for cid in client_ids
        }
        # 10% of devices have insufficient memory (permanent constraint)
        self._memory_ok: dict[int, bool] = {
            cid: bool(rng.random() > 0.10) for cid in client_ids
        }
        # Base compute time: uniform [50, 250] s — hardware heterogeneity
        self._compute_base: dict[int, float] = {
            cid: float(rng.uniform(50.0, 250.0)) for cid in client_ids
        }
        # Per-round noise (updated each call to update_round)
        self._snr_noise: dict[int, float] = {cid: 0.0 for cid in client_ids}
        self._compute_noise: dict[int, float] = {cid: 0.0 for cid in client_ids}

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
                # Slow passive recharge
                self._battery[cid] = min(1.0, self._battery[cid] + 0.005)
            self._snr_noise[cid] = float(self._rng.normal(0.0, 2.0))
            self._compute_noise[cid] = float(self._rng.normal(0.0, 30.0))

    def get_state(self, client_id: int) -> DeviceState:
        return DeviceState(
            battery=self._battery[client_id],
            snr_db=self._snr_base[client_id] + self._snr_noise[client_id],
            memory_ok=self._memory_ok[client_id],
            compute_time_s=max(10.0, self._compute_base[client_id] + self._compute_noise[client_id]),
        )

    def get_all_states(self) -> dict[int, DeviceState]:
        return {cid: self.get_state(cid) for cid in self._ids}
