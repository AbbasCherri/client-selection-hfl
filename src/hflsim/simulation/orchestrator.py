import numpy as np
import torch

from .client import get_fusion_params, get_flat_fusion_weights, RandomProjection


class HFLOrchestrator:
    """Orchestrates the hierarchical federated learning simulation.
    Performs rounds of client selection, local training, reputation updating,
    edge aggregation, global server aggregation, and model evaluation.
    """

    def __init__(
        self,
        global_model,
        clients,
        uavs,
        selection_coordinator,
        loss_fn,
        test_loader,
        device='cpu',
    ):
        self.global_model = global_model
        self.clients = clients
        self.uavs = uavs
        self.selection_coordinator = selection_coordinator
        self.loss_fn = loss_fn
        self.test_loader = test_loader
        self.device = device

        # Number of output classes, read from the model head rather than hard-coded.
        try:
            self.num_classes = global_model.fusion_fc[-1].out_features
        except (AttributeError, IndexError):
            self.num_classes = 4
        # Populated by evaluate(): distribution of predicted classes on the test set.
        self.last_pred_distribution = None

        self.fusion_params = get_fusion_params(global_model)
        self.flat_dim = len(get_flat_fusion_weights(global_model, self.fusion_params))
        self.proj_dim = 10
        self.projector = RandomProjection(self.flat_dim, self.proj_dim)
        self.total_comm_cost = 0.0

    def evaluate(self):
        """Evaluates the global model; returns (accuracy, macro_f1)."""
        self.global_model.eval()
        self.global_model.to(self.device)

        all_preds = []
        all_targets = []

        with torch.inference_mode():
            for imgs, features, labels in self.test_loader:
                imgs = imgs.to(self.device)
                features = features.to(self.device)
                outputs = self.global_model(imgs, features)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        accuracy = np.mean(all_preds == all_targets)

        self.last_pred_distribution = np.bincount(
            all_preds, minlength=self.num_classes
        ) / max(1, len(all_preds))

        # Macro F1 computed only over classes that actually appear in the test
        # targets. Averaging over always-absent classes silently halves the reported macro-F1.
        present_classes = np.unique(all_targets)
        f1_scores = []
        for c in present_classes:
            tp = np.sum((all_preds == c) & (all_targets == c))
            fp = np.sum((all_preds == c) & (all_targets != c))
            fn = np.sum((all_preds != c) & (all_targets == c))
            precision = tp / (tp + fp + 1e-5)
            recall = tp / (tp + fn + 1e-5)
            f1 = 2.0 * (precision * recall) / (precision + recall + 1e-5)
            f1_scores.append(f1)

        macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
        return accuracy, macro_f1

    def get_jains_fairness(self):
        """Computes Jain's fairness index across all clients."""
        counts = np.array([c.selection_count for c in self.clients])
        sum_counts = np.sum(counts)
        if sum_counts == 0:
            return 1.0
        sum_sq_counts = np.sum(counts ** 2)
        N = len(self.clients)
        return (sum_counts ** 2) / (N * sum_sq_counts + 1e-8)

    def simulate_round(self, round_num, selection_method="proposed", epochs=3, lr=3e-4):
        """Executes one full global round of Hierarchical Federated Learning."""
        # 1. Client selection & UAV assignment
        selected_clients, assignment = self.selection_coordinator.perform_selection(
            round_num=round_num, selection_method=selection_method
        )

        selected_client_ids = {client.client_id for client in selected_clients}
        for client in self.clients:
            client.is_active = (client.client_id in selected_client_ids)
            if hasattr(client, 'local_model_state'):
                delattr(client, 'local_model_state')

        # 2. Local training with straggler simulation
        successful_clients = []
        round_updates = []
        client_latencies = {}

        for client in selected_clients:
            if np.random.rand() < 0.03:
                continue

            if client.num_samples < 2:
                print(
                    f"  [WARN] Client {client.client_id} has only {client.num_samples} "
                    f"sample(s) — skipping to avoid BatchNorm crash."
                )
                continue

            try:
                delta_w, actual_time = client.train_local(
                    global_model=self.global_model, loss_fn=self.loss_fn, lr=lr, epochs=epochs
                )
            except Exception as e:
                print(
                    f"  [ERROR] Client {client.client_id} train_local failed: "
                    f"{type(e).__name__}: {e}"
                )
                continue

            if actual_time <= self.selection_coordinator.T_max:
                successful_clients.append(client)
                round_updates.append(delta_w)
                client_latencies[client.client_id] = actual_time
                self.total_comm_cost += 0.02

        successful_client_ids = {client.client_id for client in successful_clients}

        # 3. Update client reputations
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

                # A. Contribution score (cosine similarity with EMA update history)
                if client.update_ema is None:
                    client.update_ema = delta_w
                    r_contrib = 0.5
                else:
                    client.update_ema = 0.9 * client.update_ema + 0.1 * delta_w
                    cos_sim = np.dot(delta_w, client.update_ema) / (
                        np.linalg.norm(delta_w) * np.linalg.norm(client.update_ema) + 1e-8
                    )
                    r_contrib = (cos_sim + 1.0) / 2.0

                # B. Anomaly score (Mahalanobis distance)
                diff = proj_w - mean_update
                d_m = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
                r_anomaly = 1.0 if d_m <= 2.0 else np.exp(-(d_m - 2.0))

                # C. Temporal score
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

        # 4. UAV edge aggregation
        uav_updates = {}
        for uav in self.uavs:
            agg_result = uav.edge_aggregate(self.global_model)
            if agg_result is not None:
                edge_state, total_samples = agg_result
                uav_updates[uav] = (edge_state, total_samples)

                assigned_reps = [
                    c.reputation for c in uav.assigned_clients if hasattr(c, 'local_model_state')
                ]
                if len(assigned_reps) >= 3:
                    assigned_reps.sort()
                    trim_idx = max(1, int(len(assigned_reps) * 0.1))
                    if len(assigned_reps) - 2 * trim_idx > 0:
                        trimmed_reps = assigned_reps[trim_idx:-trim_idx]
                    else:
                        trimmed_reps = assigned_reps
                    uav.reputation = np.mean(trimmed_reps)
                elif len(assigned_reps) > 0:
                    uav.reputation = np.mean(assigned_reps)
                else:
                    uav.reputation = 0.5
            else:
                uav.reputation = 0.0

        # 5. Server global aggregation
        active_uavs = [u for u in uav_updates.keys() if u.reputation >= 0.3]
        if len(active_uavs) > 0:
            total_weighted_samples = sum(
                u.reputation * uav_updates[u][1] for u in active_uavs
            )
            global_state = self.global_model.state_dict()
            # Reset only floating-point tensors. Integer buffers must stay as
            # integer tensors to avoid casting errors during aggregation.
            float_keys = []
            for key, tensor in global_state.items():
                if torch.is_floating_point(tensor):
                    global_state[key] = torch.zeros_like(tensor)
                    float_keys.append(key)

            for uav in active_uavs:
                edge_state, total_samples = uav_updates[uav]
                weight = (uav.reputation * total_samples) / (total_weighted_samples + 1e-8)
                for key in float_keys:
                    global_state[key] += edge_state[key].to(global_state[key].device) * weight

            self.global_model.load_state_dict(global_state)

        # 6. Update client and UAV state for next round
        for client in self.clients:
            client.update_hardware_state(is_selected=(client.client_id in selected_client_ids))

        for uav in self.uavs:
            uav.update_state()
