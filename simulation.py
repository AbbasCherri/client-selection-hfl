import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import parameters_to_vector

def get_fusion_params(model):
    """
    Returns the fusion-head parameters in model order for fast vectorization.
    """
    return [param for name, param in model.named_parameters() if 'fusion_fc' in name]


def get_flat_fusion_weights(model, fusion_params=None):
    """
    Extracts and flattens only the fusion head parameters of the model
    to use for client weight anomaly/similarity analysis.
    This saves CPU memory and accelerates operations.
    """
    with torch.no_grad():
        if fusion_params is None:
            fusion_params = get_fusion_params(model)
        return parameters_to_vector(fusion_params).detach().cpu().numpy()

class RandomProjection:
    """
    Projects high-dimensional weight updates into a low-dimensional space
    to compute the Mahalanobis distance without memory bottlenecks.
    """
    def __init__(self, input_dim, proj_dim=10, seed=42):
        np.random.seed(seed)
        self.proj_matrix = np.random.normal(0, 1.0 / np.sqrt(proj_dim), (input_dim, proj_dim)).astype(np.float32)

    def project(self, vector):
        return np.dot(vector, self.proj_matrix)

class IoTClient:
    """
    Simulates a ground IoT client sensor in the Hierarchical Federated Learning network.
    Manages local hardware status, battery consumption, network delay, and local training.
    """
    def __init__(self, client_id, coords, dataset, indices, device="cpu"):
        self.client_id = client_id
        self.coords = coords  # (latitude, longitude)
        self.device = device
        self.loader_workers = int(os.getenv("HFL_DATALOADER_WORKERS", "0"))
        
        self.dataset = dataset
        self.indices = indices
        self.num_samples = len(indices)
        
        loader_kwargs = {
            "batch_size": 64,
            "shuffle": True,
            "drop_last": False,
            "num_workers": self.loader_workers,
            "pin_memory": self.device == "cuda",
        }
        if self.loader_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

        self.train_loader = DataLoader(Subset(dataset, indices), **loader_kwargs)

        self.battery = np.random.uniform(0.3, 1.0)
        self.memory = np.random.choice([2.0, 4.0, 8.0])  # GB
        self.snr = np.random.uniform(10.0, 25.0)  # dB
        self.base_compute_time = np.random.uniform(15.0, 150.0)
        
        self.selection_count = 0
        self.reputation = 0.5
        self.latency_history = [self.base_compute_time]
        self.update_ema = None
        self.is_active = False

    def update_hardware_state(self, is_selected):
        if is_selected:
            self.battery -= np.random.uniform(0.015, 0.025)
        else:
            self.battery -= np.random.uniform(0.0005, 0.0015)
            
        if np.random.rand() < 0.05:
            self.battery = min(1.0, self.battery + np.random.uniform(0.05, 0.15))
            
        self.battery = max(0.0, self.battery)
        self.snr += np.random.normal(0, 1.5)
        self.snr = max(1.0, min(30.0, self.snr))

    def get_predicted_latency(self):
        return np.mean(self.latency_history)

    def get_safety_margin(self):
        if len(self.latency_history) >= 3:
            return 1.96 * np.std(self.latency_history)
        return 5.0

    def sample_actual_latency(self):
        noise = np.random.normal(0, 5.0)
        actual_latency = self.base_compute_time + noise
        return max(5.0, actual_latency)

    def train_local(self, global_model, loss_fn, lr=2e-4, epochs=1):
        """
        Performs local model training with strict type-safety checks.
        """
        if not hasattr(self, '_local_model'):
            self._local_model = copy.deepcopy(global_model)
        self._local_model.load_state_dict(global_model.state_dict())
        model = self._local_model.to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if not hasattr(self, '_fusion_params'):
            self._fusion_params = get_fusion_params(model)

        initial_weights = get_flat_fusion_weights(model, self._fusion_params)

        for epoch in range(epochs):
            for imgs, features, labels in self.train_loader:
                # ── CRITICAL FIX: Explicit Type Casting to match PyTorch layers ──
                imgs = imgs.to(self.device).float()
                features = features.to(self.device).float()
                labels = labels.to(self.device).long()
                
                # ── CRITICAL FIX: Handle variable image channel structures safely ──
                if imgs.shape[1] == 4:
                    imgs = imgs[:, :3, :, :]  # Strip Alpha Channel
                elif imgs.shape[1] == 1:
                    imgs = imgs.repeat(1, 3, 1, 1)  # Expand Grayscale to RGB
                
                optimizer.zero_grad(set_to_none=True)
                outputs = model(imgs, features)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

        trained_weights = get_flat_fusion_weights(model, self._fusion_params)
        delta_w = trained_weights - initial_weights
        
        actual_time = self.sample_actual_latency()
        self.latency_history.append(actual_time)
        if len(self.latency_history) > 10:
            self.latency_history.pop(0)

        self.local_model_state = model.state_dict()
        return delta_w, actual_time

class UAVAggregator:
    def __init__(self, uav_id, coords, capacity=20):
        self.uav_id = uav_id
        self.coords = coords
        self.capacity = capacity
        self.battery = 1.0
        self.assigned_clients = []
        self.reputation = 0.5

    def update_state(self):
        if len(self.assigned_clients) > 0:
            self.battery -= np.random.uniform(0.01, 0.02)
        else:
            self.battery -= np.random.uniform(0.002, 0.005)
        self.battery = max(0.0, self.battery)

    def edge_aggregate(self, global_model):
        success_clients = [c for c in self.assigned_clients if hasattr(c, 'local_model_state')]
        if len(success_clients) == 0:
            return None
            
        aggregated_state = copy.deepcopy(global_model.state_dict())
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
                aggregated_state[key] += client_state[key].to(aggregated_state[key].device) * weight
                
        return aggregated_state, total_samples

class ClientSelectionCoordinator:
    def __init__(self, epicenter, clients, uavs, R_comm=500.0, B_min_iot=0.2, B_min_uav=0.3, T_max=300.0, SNR_min=3.0):
        self.epicenter = epicenter
        self.clients = clients
        self.uavs = uavs
        self.R_comm = R_comm
        self.B_min_iot = B_min_iot
        self.B_min_uav = B_min_uav
        self.T_max = T_max
        self.SNR_min = SNR_min
        
        self.distances = [self.haversine(c.coords, epicenter) for c in clients]
        self.d95 = np.percentile(self.distances, 95) if self.distances else 1.0
        self.max_samples = max(c.num_samples for c in clients) if clients else 1

    @staticmethod
    def haversine(coord1, coord2):
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        R = 6371000.0
        
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        
        a = np.sin(delta_phi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2.0)**2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return R * c

    def perform_selection(self, round_num, selection_method="proposed", w_b=0.35, w_l=0.30, w_u=0.35, c_exploration=1.414):
        eligible_clients = []
        is_fallback = False

        for client in self.clients:
            if selection_method == "random":
                if client.battery >= 0.05:
                    eligible_clients.append(client)
                continue
                
            pred_time = client.get_predicted_latency()
            margin = client.get_safety_margin()
            
            if (client.battery >= self.B_min_iot and 
                client.snr >= self.SNR_min and 
                pred_time <= (self.T_max - margin)):
                eligible_clients.append(client)

        if not eligible_clients:
            is_fallback = True
            ranked_clients = sorted(self.clients, key=lambda c: c.battery, reverse=True)
            eligible_clients = ranked_clients[:max(5, len(self.uavs) * 2)]

        scores = {}
        for client in eligible_clients:
            if selection_method == "random" or is_fallback:
                scores[client] = np.random.uniform(0, 1)
                continue
            elif selection_method == "battery_only":
                scores[client] = client.battery
                continue
                
            d_epi = self.haversine(client.coords, self.epicenter)
            u_epi = max(0.0, (self.d95 - min(d_epi, self.d95)) / self.d95)
            
            all_snrs = [c.snr for c in eligible_clients]
            max_snr, min_snr = max(all_snrs), min(all_snrs)
            u_snr = (client.snr - min_snr) / (max_snr - min_snr + 1e-5)
            u_dens = min(1.0, client.num_samples / (self.max_samples * 0.5))
            
            u_score = 0.4 * u_epi + 0.3 * u_snr + 0.3 * u_dens
            beta = max(0.0, 1.0 - round_num / 20.0)
            u_rep = beta * u_score + (1.0 - beta) * client.reputation
            
            tilde_l = 1.0 - (client.get_predicted_latency() / self.T_max)**2
            priority = w_b * client.battery + w_l * tilde_l + w_u * u_rep
            
            if selection_method == "utility_only":
                scores[client] = priority
            else:
                ucb_bonus = c_exploration * np.sqrt(np.log(round_num + 1) / (client.selection_count + 1))
                scores[client] = priority + ucb_bonus

        sorted_clients = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        selected_clients = []
        
        for uav in self.uavs:
            uav.assigned_clients = []
            
        for client in sorted_clients:
            feasible_uavs = []
            for uav in self.uavs:
                dist = self.haversine(client.coords, uav.coords)
                if is_fallback or (uav.battery >= self.B_min_uav and 
                                  len(uav.assigned_clients) < uav.capacity and 
                                  dist <= self.R_comm):
                    feasible_uavs.append(uav)
                    
            if feasible_uavs:
                feasible_uavs.sort(key=lambda u: (len(u.assigned_clients), self.haversine(client.coords, u.coords)))
                target_uav = feasible_uavs[0]
                
                target_uav.assigned_clients.append(client)
                selected_clients.append(client)
                client.selection_count += 1

        if not selected_clients and sorted_clients and self.uavs:
            client = sorted_clients[0]
            target_uav = min(self.uavs, key=lambda u: self.haversine(client.coords, u.coords))
            target_uav.assigned_clients.append(client)
            selected_clients.append(client)
            client.selection_count += 1
                
        return selected_clients, None

class HFLOrchestrator:
    def __init__(self, global_model, clients, uavs, selection_coordinator, loss_fn, test_loader, device='cpu'):
        self.global_model = global_model
        self.clients = clients
        self.uavs = uavs
        self.selection_coordinator = selection_coordinator
        self.loss_fn = loss_fn
        self.test_loader = test_loader
        self.device = device
        
        self.fusion_params = get_fusion_params(global_model)
        self.flat_dim = len(get_flat_fusion_weights(global_model, self.fusion_params))
        self.proj_dim = 10
        self.projector = RandomProjection(self.flat_dim, self.proj_dim)
        self.total_comm_cost = 0.0

    def evaluate(self):
        self.global_model.eval()
        self.global_model.to(self.device)
        
        all_preds = []
        all_targets = []
        
        with torch.inference_mode():
            for imgs, features, labels in self.test_loader:
                # ── CRITICAL FIX: Add safe casting step to global test runner ──
                imgs = imgs.to(self.device).float()
                features = features.to(self.device).float()
                
                if imgs.shape[1] == 4:
                    imgs = imgs[:, :3, :, :]
                elif imgs.shape[1] == 1:
                    imgs = imgs.repeat(1, 3, 1, 1)

                outputs = self.global_model(imgs, features)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.numpy())
                
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        if len(all_targets) == 0:
            return 0.0, 0.0
            
        accuracy = np.mean(all_preds == all_targets)
        
        f1_scores = []
        for c in range(4):
            tp = np.sum((all_preds == c) & (all_targets == c))
            fp = np.sum((all_preds == c) & (all_targets != c))
            fn = np.sum((all_preds != c) & (all_targets == c))
            
            precision = tp / (tp + fp + 1e-5)
            recall = tp / (tp + fn + 1e-5)
            f1 = 2.0 * (precision * recall) / (precision + recall + 1e-5)
            f1_scores.append(f1)
            
        macro_f1 = np.mean(f1_scores)
        return accuracy, macro_f1

    def get_jains_fairness(self):
        counts = np.array([c.selection_count for c in self.clients])
        sum_counts = np.sum(counts)
        if sum_counts == 0:
            return 1.0
        sum_sq_counts = np.sum(counts ** 2)
        N = len(self.clients)
        return (sum_counts ** 2) / (N * sum_sq_counts + 1e-8)

    def simulate_round(self, round_num, selection_method="proposed"):
        selected_clients, _ = self.selection_coordinator.perform_selection(
            round_num=round_num,
            selection_method=selection_method
        )
        
        selected_client_ids = {client.client_id for client in selected_clients}
        for client in self.clients:
            client.is_active = (client.client_id in selected_client_ids)
            if hasattr(client, 'local_model_state'):
                delattr(client, 'local_model_state')
                
        successful_clients = []
        round_updates = []
        
        for client in selected_clients:
            if round_num > 2 and np.random.rand() < 0.03:
                continue
                
            try:
                delta_w, actual_time = client.train_local(
                    global_model=self.global_model,
                    loss_fn=self.loss_fn,
                    epochs=1
                )
                
                if actual_time <= self.selection_coordinator.T_max or round_num <= 5:
                    successful_clients.append(client)
                    round_updates.append(delta_w)
                    self.total_comm_cost += 0.02
            except Exception as e:
                # ── CRITICAL FIX: Expose hidden exceptions to reveal API/Network issues ──
                print(f"  [DEBUG LOG] Client {client.client_id} execution crashed: {e}")
                continue

        successful_client_ids = {client.client_id for client in successful_clients}
                
        if len(successful_clients) > 0:
            projected_updates = np.array([self.projector.project(up) for up in round_updates])
            mean_update = np.mean(projected_updates, axis=0)
            if len(successful_clients) > 1:
                cov_matrix = np.cov(projected_updates, rowvar=False)
                cov_matrix += np.eye(self.proj_dim) * 1e-3
                inv_cov = np.linalg.inv(cov_matrix)
            else:
                inv_cov = np.eye(self.proj_dim)
                
            for idx, client in enumerate(successful_clients):
                delta_w = round_updates[idx]
                proj_w = projected_updates[idx]
                
                if client.update_ema is None:
                    client.update_ema = delta_w
                    r_contrib = 0.5
                else:
                    client.update_ema = 0.9 * client.update_ema + 0.1 * delta_w
                    cos_sim = np.dot(delta_w, client.update_ema) / (
                        np.linalg.norm(delta_w) * np.linalg.norm(client.update_ema) + 1e-8
                    )
                    r_contrib = (cos_sim + 1.0) / 2.0
                    
                diff = proj_w - mean_update
                d_m = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
                r_anomaly = 1.0 if d_m <= 2.0 else np.exp(-(d_m - 2.0))
                    
                var_lat = np.var(client.latency_history) if len(client.latency_history) >= 2 else 0.0
                if not hasattr(client, 'success_history'):
                    client.success_history = []
                client.success_history.append(1)
                if len(client.success_history) > 10:
                    client.success_history.pop(0)
                    
                success_rate = np.mean(client.success_history)
                r_temp = 0.5 * success_rate + 0.5 * (100.0 / (100.0 + var_lat + 1e-5))
                
                new_rep = (r_contrib + r_anomaly + r_temp) / 3.0
                client.reputation = 0.8 * client.reputation + 0.2 * new_rep
                
        for client in selected_clients:
            if client.client_id not in successful_client_ids:
                if not hasattr(client, 'success_history'):
                    client.success_history = []
                client.success_history.append(0)
                if len(client.success_history) > 10:
                    client.success_history.pop(0)
                    
                success_rate = np.mean(client.success_history)
                var_lat = np.var(client.latency_history) if len(client.latency_history) >= 2 else 0.0
                r_temp = 0.5 * success_rate + 0.5 * (100.0 / (100.0 + var_lat + 1e-5))
                client.reputation = 0.8 * client.reputation + 0.2 * (r_temp / 3.0)
                
        uav_updates = {}
        for uav in self.uavs:
            agg_result = uav.edge_aggregate(self.global_model)
            if agg_result is not None:
                edge_state, total_samples = agg_result
                uav_updates[uav] = (edge_state, total_samples)
                
                assigned_reps = [c.reputation for c in uav.assigned_clients if hasattr(c, 'local_model_state')]
                if len(assigned_reps) >= 3:
                    assigned_reps.sort()
                    trim_idx = max(1, int(len(assigned_reps) * 0.1))
                    trimmed_reps = assigned_reps[trim_idx:-trim_idx] if len(assigned_reps) - 2 * trim_idx > 0 else assigned_reps
                    uav.reputation = np.mean(trimmed_reps)
                elif len(assigned_reps) > 0:
                    uav.reputation = np.mean(assigned_reps)
                else:
                    uav.reputation = 0.5
            else:
                uav.reputation = 0.0
                
        active_uavs = [u for u in uav_updates.keys() if u.reputation >= 0.1 or round_num <= 5]
        if len(active_uavs) > 0:
            total_weighted_samples = sum((u.reputation + 1e-2) * uav_updates[u][1] for u in active_uavs)
            global_state = self.global_model.state_dict()
            
            float_keys = []
            for key, tensor in global_state.items():
                if torch.is_floating_point(tensor):
                    global_state[key] = torch.zeros_like(tensor)
                    float_keys.append(key)
                
            for uav in active_uavs:
                edge_state, total_samples = uav_updates[uav]
                weight = ((uav.reputation + 1e-2) * total_samples) / (total_weighted_samples + 1e-8)
                for key in float_keys:
                    global_state[key] += edge_state[key].to(global_state[key].device) * weight
                    
            self.global_model.load_state_dict(global_state)
            
        for client in self.clients:
            client.update_hardware_state(is_selected=(client.client_id in selected_client_ids))
            
        for uav in self.uavs:
            uav.update_state()