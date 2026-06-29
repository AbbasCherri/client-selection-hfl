import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, Subset


def get_fusion_params(model):
    return [param for name, param in model.named_parameters() if 'fusion_fc' in name]


def get_flat_fusion_weights(model, fusion_params=None):
    with torch.no_grad():
        if fusion_params is None:
            fusion_params = get_fusion_params(model)
        return parameters_to_vector(fusion_params).detach().cpu().numpy()


class RandomProjection:
    """Projects high-dimensional weight updates into a low-dimensional space
    to compute the Mahalanobis distance without memory bottlenecks."""

    def __init__(self, input_dim, proj_dim=10, seed=42):
        np.random.seed(seed)
        self.proj_matrix = np.random.normal(
            0, 1.0 / np.sqrt(proj_dim), (input_dim, proj_dim)
        ).astype(np.float32)

    def project(self, vector):
        return np.dot(vector, self.proj_matrix)


class IoTClient:
    """Simulates a ground IoT client sensor in the Hierarchical Federated Learning network.
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

        # BALANCED SAMPLING — the root cause of the frozen metrics.
        #
        # With ~97% class-0 data and non-IID geographic sharding, most client
        # shards are almost entirely class 0. A standard DataLoader shows the
        # model class-0 batches on every step → loss is already near-minimal →
        # gradients are near-zero → model never moves → metrics freeze.
        #
        # WeightedRandomSampler assigns each sample a weight inversely proportional
        # to its class count within this shard, so every sampled batch is balanced
        # across whichever classes the shard contains, guaranteeing minority-class
        # gradient signal in every optimizer step.
        #
        # Adaptive batch size: EfficientNet BatchNorm requires >= 2 samples per
        # batch in train() mode. Clamp to [2, 32] and drop the last partial batch.
        effective_batch = max(2, min(32, len(indices) // 2)) if len(indices) >= 2 else 2

        subset = Subset(dataset, indices)

        # Per-sample weights from inverse class frequency within this shard.
        # np.where evaluates BOTH branches before selecting, so naively writing
        # np.where(counts > 0, 1.0 / counts, 0.0) still performs the division
        # for zero-count classes and emits a RuntimeWarning.  Use np.errstate to
        # suppress the spurious warning; the zero-count slots are replaced by 0.0
        # before any weight is ever used.
        shard_labels = dataset.labels[indices].numpy()
        class_counts = np.bincount(shard_labels, minlength=4)
        with np.errstate(divide='ignore', invalid='ignore'):
            class_weights_arr = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
        sample_weights = torch.tensor(
            [class_weights_arr[int(lbl)] for lbl in shard_labels], dtype=torch.float32
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        loader_kwargs = {
            "batch_size": effective_batch,
            "sampler": sampler,
            "drop_last": len(indices) > effective_batch,
            "num_workers": self.loader_workers,
            "pin_memory": self.device == "cuda",
        }
        if self.loader_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

        self.train_loader = DataLoader(subset, **loader_kwargs)

        # Hardware and channel characteristics (Heterogeneous & Dynamic)
        self.battery = np.random.uniform(0.3, 1.0)
        self.memory = np.random.choice([2.0, 4.0, 8.0])
        self.snr = np.random.uniform(10.0, 25.0)
        self.base_compute_time = np.random.uniform(50.0, 400.0)

        # Historical metrics
        self.selection_count = 0
        self.reputation = 0.5
        self.latency_history = [self.base_compute_time]
        self.update_ema = None
        self.is_active = False

    def update_hardware_state(self, is_selected):
        """Simulates battery decay, sporadic solar replenishment, and SNR random walk."""
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
        return 15.0

    def sample_actual_latency(self):
        noise = np.random.normal(0, 10.0)
        return max(10.0, self.base_compute_time + noise)

    def train_local(self, global_model, loss_fn, lr=3e-4, epochs=3):
        """Performs local model training with entropy balancing."""
        if not hasattr(self, '_local_model'):
            self._local_model = copy.deepcopy(global_model)
        self._local_model.load_state_dict(global_model.state_dict())
        model = self._local_model.to(self.device)

        model.train()
        if hasattr(model, 'image_branch') and hasattr(model.image_branch, 'backbone'):
            model.image_branch.backbone.eval()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        self._fusion_params = get_fusion_params(model)
        initial_weights = get_flat_fusion_weights(model, self._fusion_params)

        for epoch in range(epochs):
            for imgs, features, labels in self.train_loader:
                imgs = imgs.to(self.device)
                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(imgs, features)
                base_loss = loss_fn(outputs, labels)

                # Entropy regularization: gently discourage overconfident majority-class
                # predictions. Coefficient 0.01 is intentionally small — the focal loss
                # already handles class imbalance; a large entropy bonus (0.1) fights the
                # focal loss and destabilises training on single-class client shards.
                probs = F.softmax(outputs, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                loss = base_loss - 0.01 * entropy

                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

        trained_weights = get_flat_fusion_weights(model, self._fusion_params)
        delta_w = trained_weights - initial_weights

        actual_time = self.sample_actual_latency()
        self.latency_history.append(actual_time)
        if len(self.latency_history) > 10:
            self.latency_history.pop(0)

        self.local_model_state = model.state_dict()
        return delta_w, actual_time
