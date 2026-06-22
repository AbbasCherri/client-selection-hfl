"""
test_sanity.py
Unit / integration sanity checks for the HFL streaming-based pipeline.
Run with:  python test_sanity.py   (or via unittest discover)
"""

import os
import sys
import unittest
from pathlib import Path


def _bootstrap_venv():
    repo_root  = Path(__file__).resolve().parent
    venv_root  = repo_root / ".venv"
    if not venv_root.exists():
        return
    for sp in [
        venv_root / "lib"   / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        venv_root / "lib64" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    ]:
        if sp.exists() and str(sp) not in sys.path:
            sys.path.insert(0, str(sp))


_bootstrap_venv()

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models     import MultiModalFusionModel, FocalLoss
from data_loader import MultiModalDataset, FEATURE_COLS
from simulation  import IoTClient, UAVAggregator, ClientSelectionCoordinator, HFLOrchestrator


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_dummy_df(n: int = 40) -> pd.DataFrame:
    """Minimal DataFrame that satisfies MultiModalDataset requirements."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "latitude":     rng.uniform(37.3, 37.5, n),
        "longitude":    rng.uniform(136.8, 137.3, n),
        "MMI_original": rng.uniform(5.0, 8.0, n),
        "MMI_shape":    rng.uniform(5.0, 8.0, n),
        "PGA":          rng.uniform(0.1, 1.5, n),
        "PGV":          rng.uniform(10.0, 100.0, n),
        "SA_0_3":       rng.uniform(0.1, 2.0, n),
        "SA_1_0":       rng.uniform(0.1, 2.0, n),
        "SA_3_0":       rng.uniform(0.1, 2.0, n),
        "damage_val":   rng.choice([0, 1, 2, 3], n),
        "chip_path":    [""] * n,          # No local chips; images come from GSI
    })


class MockDataset(MultiModalDataset):
    """
    Overrides _load_image to return a random tensor, bypassing
    both local-file and GSI-tile I/O so tests stay fully offline.
    """
    def _load_image(self, idx: int):
        from PIL import Image
        arr = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
        return Image.fromarray(arr, "RGB")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModelForwardPass(unittest.TestCase):
    def test_forward(self):
        print("\n--- Model forward pass ---")
        model      = MultiModalFusionModel(num_classes=4, pretrained=False)
        model.eval()
        imgs       = torch.randn(4, 3, 128, 128)
        feats      = torch.randn(4, 9)
        with torch.no_grad():
            out = model(imgs, feats)
        self.assertEqual(out.shape, (4, 4))
        print(f"  Output shape: {out.shape}  ✓")


class TestFocalLoss(unittest.TestCase):
    def test_loss(self):
        print("\n--- Focal loss ---")
        alpha   = torch.ones(4)
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
        logits  = torch.randn(4, 4)
        targets = torch.tensor([0, 2, 1, 3])
        loss    = loss_fn(logits, targets)
        self.assertGreaterEqual(loss.item(), 0.0)
        print(f"  Loss value: {loss.item():.4f}  ✓")


class TestStreamingDataset(unittest.TestCase):
    """Verifies that MockDataset (GSI-image bypass) loads correctly."""
    def test_dataset_getitem(self):
        print("\n--- Streaming dataset getitem ---")
        df      = _make_dummy_df(8)
        dataset = MockDataset(df, data_dir="", use_gsi=False)
        img, feat, label = dataset[0]
        self.assertEqual(img.shape,  (3, 128, 128))
        self.assertEqual(feat.shape, (9,))
        self.assertIn(label.item(), [0, 1, 2, 3])
        print(f"  img={img.shape}  feat={feat.shape}  label={label.item()}  ✓")


class TestHFLOrchestration(unittest.TestCase):
    """Simulates a single HFL round on a tiny mock dataset."""

    def test_one_round(self):
        print("\n--- HFL orchestration (1 round, mock data) ---")
        df      = _make_dummy_df(40)
        dataset = MockDataset(df, data_dir="", use_gsi=False)

        # Build 4 clients, 10 samples each
        clients = []
        for i in range(4):
            indices = list(range(i * 10, (i + 1) * 10))
            lat_mean = float(df.iloc[indices]["latitude"].mean())
            lon_mean = float(df.iloc[indices]["longitude"].mean())
            c = IoTClient(
                client_id=i,
                coords=(lat_mean, lon_mean),
                dataset=dataset,
                indices=indices,
                device="cpu",
            )
            # Ensure every client passes eligibility gates in the unit test
            c.battery            = 0.9
            c.snr                = 20.0
            c.base_compute_time  = 100.0
            c.latency_history    = [100.0]
            clients.append(c)

        uavs = [
            UAVAggregator(uav_id=0, coords=(37.35, 137.00), capacity=5),
            UAVAggregator(uav_id=1, coords=(37.45, 137.15), capacity=5),
        ]

        coordinator = ClientSelectionCoordinator(
            epicenter = (37.50, 137.27),
            clients   = clients,
            uavs      = uavs,
            R_comm    = 50_000.0,    # Wide range so all clients are reachable
            B_min_iot = 0.2,
            B_min_uav = 0.3,
            T_max     = 300.0,
            SNR_min   = 3.0,
        )

        global_model = MultiModalFusionModel(num_classes=4, pretrained=False)
        loss_fn      = FocalLoss()
        test_loader  = DataLoader(dataset, batch_size=4, shuffle=False)

        orchestrator = HFLOrchestrator(
            global_model          = global_model,
            clients               = clients,
            uavs                  = uavs,
            selection_coordinator = coordinator,
            loss_fn               = loss_fn,
            test_loader           = test_loader,
            device                = "cpu",
        )

        orchestrator.simulate_round(round_num=1, selection_method="proposed")
        selected_count = sum(1 for c in clients if c.selection_count > 0)
        accuracy, macro_f1 = orchestrator.evaluate()

        print(f"  Selected clients : {selected_count}")
        print(f"  Accuracy         : {accuracy:.4f}")
        print(f"  Macro F1         : {macro_f1:.4f}")

        self.assertGreater(selected_count, 0)
        self.assertGreaterEqual(accuracy, 0.0)
        print("  Orchestration round sanity verified  ✓")


if __name__ == "__main__":
    unittest.main(verbosity=2)
