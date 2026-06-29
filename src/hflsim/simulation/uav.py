import copy

import numpy as np
import torch


class UAVAggregator:
    """Simulates a UAV acting as a Tier-2 edge aggregator.
    Tracks UAV status, battery, assigned clients, and performs local edge aggregation.
    """

    def __init__(self, uav_id, coords, capacity=20):
        self.uav_id = uav_id
        self.coords = coords  # (latitude, longitude)
        self.capacity = capacity
        self.battery = 1.0
        self.assigned_clients = []
        self.reputation = 0.5

    def update_state(self):
        """Simulates UAV battery consumption during the round."""
        if len(self.assigned_clients) > 0:
            self.battery -= np.random.uniform(0.01, 0.02)
        else:
            self.battery -= np.random.uniform(0.002, 0.005)
        self.battery = max(0.0, self.battery)

    def edge_aggregate(self, global_model):
        """FedAvg-style aggregation of assigned clients' model states."""
        success_clients = [c for c in self.assigned_clients if hasattr(c, 'local_model_state')]
        if len(success_clients) == 0:
            return None

        aggregated_state = copy.deepcopy(global_model.state_dict())

        # Zero out only floating-point tensors. Integer buffers (for example,
        # BatchNorm counters) should be preserved as-is to avoid dtype errors.
        float_keys = []
        for key, tensor in aggregated_state.items():
            if torch.is_floating_point(tensor):
                aggregated_state[key] = torch.zeros_like(tensor)
                float_keys.append(key)

        total_samples = sum(c.num_samples for c in success_clients)

        for client in success_clients:
            weight = client.num_samples / total_samples
            client_state = client.local_model_state
            for key in float_keys:
                aggregated_state[key] += (
                    client_state[key].to(aggregated_state[key].device) * weight
                )

        return aggregated_state, total_samples
