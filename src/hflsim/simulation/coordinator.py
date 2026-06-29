import numpy as np

from hflsim.shared.coords import haversine


class ClientSelectionCoordinator:
    """Coordinates eligibility gating, priority ranking, and greedy UAV assignment."""

    def __init__(
        self,
        epicenter,
        clients,
        uavs,
        R_comm=500.0,
        B_min_iot=0.2,
        B_min_uav=0.3,
        T_max=300.0,
        SNR_min=3.0,
    ):
        self.epicenter = epicenter  # (lat, lon)
        self.clients = clients
        self.uavs = uavs
        self.R_comm = R_comm
        self.B_min_iot = B_min_iot
        self.B_min_uav = B_min_uav
        self.T_max = T_max
        self.SNR_min = SNR_min

        self.distances = [haversine(c.coords, epicenter) for c in clients]
        self.d95 = np.percentile(self.distances, 95)
        self.max_samples = max(c.num_samples for c in clients)

    def perform_selection(
        self,
        round_num,
        selection_method="proposed",
        w_b=0.35,
        w_l=0.30,
        w_u=0.35,
        c_exploration=1.414,
    ):
        """Executes client selection and UAV assignment under the specified policy."""
        # 1. Eligibility gating
        eligible_clients = []
        for client in self.clients:
            if selection_method == "random":
                if client.battery >= self.B_min_iot and client.snr >= self.SNR_min:
                    eligible_clients.append(client)
                continue

            pred_time = client.get_predicted_latency()
            margin = client.get_safety_margin()

            if (
                client.battery >= self.B_min_iot
                and client.snr >= self.SNR_min
                and pred_time <= (self.T_max - margin)
            ):
                eligible_clients.append(client)

        if not eligible_clients:
            ranked_clients = sorted(
                self.clients,
                key=lambda c: (c.battery, c.snr, c.num_samples),
                reverse=True,
            )
            eligible_clients = ranked_clients[: max(1, min(len(ranked_clients), len(self.uavs) or 1))]

        # 2. Priority scoring and UCB exploration
        scores = {}
        for client in eligible_clients:
            if selection_method == "random":
                scores[client] = np.random.rand()
                continue
            elif selection_method == "battery_only":
                scores[client] = client.battery
                continue

            d_epi = haversine(client.coords, self.epicenter)
            u_epi = max(0.0, (self.d95 - min(d_epi, self.d95)) / self.d95)

            all_snrs = [c.snr for c in eligible_clients]
            max_snr, min_snr = max(all_snrs), min(all_snrs)
            u_snr = (client.snr - min_snr) / (max_snr - min_snr + 1e-5)

            u_dens = min(1.0, client.num_samples / (self.max_samples * 0.5))

            u_score = 0.4 * u_epi + 0.3 * u_snr + 0.3 * u_dens

            beta = max(0.0, 1.0 - round_num / 20.0)
            u_rep = beta * u_score + (1.0 - beta) * client.reputation

            tilde_l = 1.0 - (client.get_predicted_latency() / self.T_max) ** 2

            priority = w_b * client.battery + w_l * tilde_l + w_u * u_rep

            if selection_method == "utility_only":
                scores[client] = priority
            else:
                ucb_bonus = c_exploration * np.sqrt(
                    np.log(round_num + 1) / (client.selection_count + 1)
                )
                scores[client] = priority + ucb_bonus

        sorted_clients = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

        # 3. Greedy UAV assignment
        selected_clients = []
        assignment = {}

        for uav in self.uavs:
            uav.assigned_clients = []

        for client in sorted_clients:
            feasible_uavs = []
            for uav in self.uavs:
                dist = haversine(client.coords, uav.coords)
                if (
                    uav.battery >= self.B_min_uav
                    and len(uav.assigned_clients) < uav.capacity
                    and dist <= self.R_comm
                ):
                    feasible_uavs.append(uav)

            if feasible_uavs:
                if selection_method == "fedcs":
                    feasible_uavs.sort(key=lambda u: len(u.assigned_clients))
                else:
                    feasible_uavs.sort(
                        key=lambda u: (len(u.assigned_clients), haversine(client.coords, u.coords))
                    )
                target_uav = feasible_uavs[0]
                target_uav.assigned_clients.append(client)
                selected_clients.append(client)
                assignment[client.client_id] = target_uav.uav_id
                client.selection_count += 1

        if not selected_clients and sorted_clients and self.uavs:
            # Emergency fallback: assign the top-ranked client to the nearest UAV
            # even if the communication gate was too strict. This keeps the
            # simulation from stalling with zero-participation rounds.
            client = sorted_clients[0]
            target_uav = min(self.uavs, key=lambda u: haversine(client.coords, u.coords))
            target_uav.assigned_clients.append(client)
            selected_clients.append(client)
            assignment[client.client_id] = target_uav.uav_id
            client.selection_count += 1

        return selected_clients, assignment
