import unittest
import torch
import numpy as np
import pandas as pd

from models import MultiModalFusionModel, FocalLoss
from torch.utils.data import DataLoader, Dataset
from data_loader import MultiModalDataset
from simulation import IoTClient, UAVAggregator, ClientSelectionCoordinator, HFLOrchestrator

class TestHFLSanity(unittest.TestCase):
    def test_model_forward_pass(self):
        """
        Verifies the MultiModalFusionModel forward pass with dummy images and structured features.
        """
        print("\n--- Testing Model Forward Pass ---")
        model = MultiModalFusionModel(num_classes=4, pretrained=False)
        model.eval()
        
        # Batch of 4 images (3, 128, 128) and 4 structured vectors (9 features)
        dummy_imgs = torch.randn(4, 3, 128, 128)
        dummy_features = torch.randn(4, 9)
        
        with torch.no_grad():
            outputs = model(dummy_imgs, dummy_features)
            
        print(f"Output shape: {outputs.shape}")
        self.assertEqual(outputs.shape, (4, 4))
        print("Model forward pass verified.")

    def test_focal_loss(self):
        """
        Verifies Focal Loss works and output is a scalar tensor.
        """
        print("\n--- Testing Focal Loss ---")
        alpha = torch.tensor([1.0, 1.0, 1.0, 1.0])
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
        
        inputs = torch.randn(4, 4)
        targets = torch.tensor([0, 2, 1, 3], dtype=torch.long)
        
        loss = loss_fn(inputs, targets)
        print(f"Computed loss value: {loss.item():.4f}")
        self.assertTrue(loss.item() >= 0.0)
        print("Focal Loss computation verified.")

    def test_orchestration_sanity(self):
        """
        Simulates 1 mock HFL round with 4 clients and 2 UAVs on mock dataset.
        """
        print("\n--- Testing HFL Orchestration Sanity ---")
        # 1. Create a dummy metadata dataframe
        df_dummy = pd.DataFrame({
            'latitude': np.random.uniform(37.3, 37.5, 40),
            'longitude': np.random.uniform(136.8, 137.3, 40),
            'MMI_original': np.random.uniform(5.0, 8.0, 40),
            'MMI_shape': np.random.uniform(5.0, 8.0, 40),
            'PGA': np.random.uniform(0.1, 1.5, 40),
            'PGV': np.random.uniform(10.0, 100.0, 40),
            'SA_0_3': np.random.uniform(0.1, 2.0, 40),
            'SA_1_0': np.random.uniform(0.1, 2.0, 40),
            'SA_3_0': np.random.uniform(0.1, 2.0, 40),
            'damage_val': np.random.choice([0, 1, 2, 3], 40),
            'chip_path': ['../Images/dummy.tif'] * 40
        })

        # 2. Mock MultiModalDataset subclass to bypass disk loading for unit test
        class MockDataset(MultiModalDataset):
            def __getitem__(self, idx):
                img_tensor = torch.randn(3, 128, 128)
                features_tensor = torch.tensor(self.features[idx])
                label_tensor = torch.tensor(self.labels[idx])
                return img_tensor, features_tensor, label_tensor

        full_dataset = MockDataset(df_dummy, data_dir="")
        
        # 3. Setup client indices
        clients = []
        for i in range(4):
            indices = list(range(i * 10, (i + 1) * 10))
            client_coords = (df_dummy.iloc[indices]['latitude'].mean(), df_dummy.iloc[indices]['longitude'].mean())
            client = IoTClient(
                client_id=i,
                coords=client_coords,
                dataset=full_dataset,
                indices=indices,
                device="cpu"
            )
            # Give high battery to pass eligibility gate in unit test
            client.battery = 0.9
            client.snr = 20.0
            client.base_compute_time = 100.0
            clients.append(client)

        # 4. Setup UAVs
        uavs = [
            UAVAggregator(uav_id=0, coords=(37.35, 137.00), capacity=5),
            UAVAggregator(uav_id=1, coords=(37.45, 137.15), capacity=5)
        ]

        # 5. Setup selection coordinator
        epicenter = (37.50, 137.27)
        coordinator = ClientSelectionCoordinator(
            epicenter=epicenter,
            clients=clients,
            uavs=uavs,
            R_comm=50000.0,  # Expand range for mock test so all are covered
            B_min_iot=0.2,
            B_min_uav=0.3,
            T_max=300.0,
            SNR_min=3.0
        )

        global_model = MultiModalFusionModel(num_classes=4, pretrained=False)
        loss_fn = FocalLoss()
        
        # Mock test loader
        test_loader = DataLoader(full_dataset, batch_size=4, shuffle=False)

        # Create HFLOrchestrator
        orchestrator = HFLOrchestrator(
            global_model=global_model,
            clients=clients,
            uavs=uavs,
            selection_coordinator=coordinator,
            loss_fn=loss_fn,
            test_loader=test_loader,
            device="cpu"
        )

        # Run 1 mock round of HFL
        print("Simulating HFL round...")
        orchestrator.simulate_round(round_num=1, selection_method="proposed")
        
        # Validate selection counts
        selected_count = sum(1 for c in clients if c.selection_count > 0)
        print(f"Selected clients in mock round: {selected_count}")
        
        # Evaluate
        accuracy, macro_f1 = orchestrator.evaluate()
        print(f"Mock global accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}")
        
        self.assertTrue(accuracy >= 0.0)
        self.assertTrue(selected_count > 0)
        print("Orchestration round execution sanity verified.")

if __name__ == '__main__':
    unittest.main()
