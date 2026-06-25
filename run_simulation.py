"""
run_simulation.py – HFL simulation runner with streaming data support.

Usage (streaming from HuggingFace, full dataset, 100 rounds — matches paper):
    python run_simulation.py --N 70

Usage (quick smoke-test with 5% subsample):
    python run_simulation.py --N 70 --rounds 30 --subsample 0.05

Usage (local CSV if already downloaded):
    python run_simulation.py --csv_path ./data/Final_Dataset/training_dataset_with_city.csv

The HF_TOKEN environment variable must be set when the HuggingFace repo is private:
    export HF_TOKEN=hf_xxxxxx
"""

import os
import copy
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")            # headless backend – safe on GCP VMs
import matplotlib.pyplot as plt

from data_loader import get_hfl_data_partitions, MultiModalDataset
from models      import MultiModalFusionModel, FocalLoss
from simulation  import IoTClient, UAVAggregator, ClientSelectionCoordinator, HFLOrchestrator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="HFL Client-Selection Simulation Runner")

    # ── Data source ──────────────────────────────────────────────────────── #
    p.add_argument("--csv_path",    type=str,   default=None,
                   help="Path to local CSV metadata file. "
                        "Omit to stream from HuggingFace.")
    p.add_argument("--data_dir",    type=str,   default="./data",
                   help="Root directory for local image chips (optional).")
    p.add_argument("--subsample",   type=float, default=1.0,
                   help="Fraction of the dataset to use (0 < x ≤ 1). "
                        "Default 1.0 → full 128k-building dataset (matches paper).")
    p.add_argument("--hf_token",    type=str,   default=None,
                   help="HuggingFace access token. "
                        "Falls back to the HF_TOKEN environment variable.")

    # ── HFL topology ─────────────────────────────────────────────────────── #
    p.add_argument("--N",           type=int,   default=70,
                   help="Number of IoT clients (paper: 14/35/70/140).")
    p.add_argument("--U",           type=int,   default=3,
                   help="Number of UAV edge aggregators.")
    p.add_argument("--rounds",      type=int,   default=100,
                   help="Global communication rounds (paper: 100).")
    p.add_argument("--epochs",      type=int,   default=3,
                   help="Local training epochs per client per round (Optimized to fix majority trap).")
    p.add_argument("--lr",          type=float, default=3e-4,
                   help="Adam learning rate.")
    p.add_argument("--seed",        type=int,   default=42)

    # ── Output ───────────────────────────────────────────────────────────── #
    p.add_argument("--output_dir",  type=str,   default="./results")
    p.add_argument("--plot_dir",    type=str,   default="./plots")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def initialize_uavs(U: int, seed: int = 42):
    """
    Place U UAVs across the Noto Peninsula building area
    (Lat 37.3–37.5, Lon 136.8–137.3).
    """
    np.random.seed(seed)
    base_locations = [
        (37.38, 137.05),
        (37.40, 137.15),
        (37.35, 137.22),
    ]
    uavs = []
    for i in range(U):
        base = base_locations[i % len(base_locations)]
        loc  = (
            base[0] + np.random.normal(0, 0.01),
            base[1] + np.random.normal(0, 0.01),
        )
        uavs.append(UAVAggregator(uav_id=i, coords=loc, capacity=20))
    return uavs


def save_comparison_plots(df_results: pd.DataFrame, plot_dir: str, N: int, rounds: int):
    """Generate and save the four comparison plots described in the paper."""
    methods = df_results["method"].unique()

    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    colors = {
        "proposed":    "#2ca02c",
        "random":      "#7f7f7f",
        "battery_only":"#d62728",
        "utility_only":"#ff7f0e",
        "fedcs":       "#1f77b4",
    }
    labels = {
        "proposed":    "Proposed (UCB + Rep + Res)",
        "random":      "Random Selection",
        "battery_only":"Battery-Only",
        "utility_only":"Utility-Only",
        "fedcs":       "FedCS Baseline",
    }

    def _save(fig_name, xlabel, ylabel, title, col, ylim=None, legend_loc="lower right"):
        plt.figure(figsize=(8, 5))
        for m in methods:
            sub = df_results[df_results["method"] == m]
            plt.plot(sub["round"], sub[col],
                     label=labels.get(m, m),
                     color=colors.get(m),
                     linewidth=2.0)
        plt.title(title, fontsize=12, fontweight="bold")
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        if ylim:
            plt.ylim(*ylim)
        plt.legend(loc=legend_loc, frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{fig_name}_N{N}.pdf"))
        plt.savefig(os.path.join(plot_dir, f"{fig_name}_N{N}.png"), dpi=150)
        plt.close()

    _save("accuracy",  "Global Communication Round", "Test Accuracy",
          f"Global Model Accuracy Convergence (IoT Clients N={N})", "accuracy")
    _save("f1",        "Global Communication Round", "Macro F1 Score",
          f"Global Model Macro F1 Convergence (IoT Clients N={N})",  "macro_f1")
    _save("fairness",  "Global Communication Round", "Jain's Fairness Index",
          f"Jain's Fairness Index Evolution (IoT Clients N={N})",    "fairness",
          ylim=(0.5, 1.05), legend_loc="lower left")

    # Accuracy vs Communication Cost
    plt.figure(figsize=(8, 5))
    for m in methods:
        sub = df_results[df_results["method"] == m]
        plt.plot(sub["comm_cost"], sub["accuracy"],
                 label=labels.get(m, m),
                 color=colors.get(m),
                 linewidth=2.0)
    plt.title(f"Accuracy vs. Communication Cost (IoT Clients N={N})", fontsize=12, fontweight="bold")
    plt.xlabel("Total Communication Load (MB)", fontsize=10)
    plt.ylabel("Test Accuracy", fontsize=10)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"efficiency_N{N}.pdf"))
    plt.savefig(os.path.join(plot_dir, f"efficiency_N{N}.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve relative paths to absolute (important when running via shell scripts)
    def _abs(p):
        return os.path.abspath(p) if p else p

    args.data_dir   = _abs(args.data_dir)
    args.output_dir = _abs(args.output_dir)
    args.plot_dir   = _abs(args.plot_dir)
    if args.csv_path:
        args.csv_path = _abs(args.csv_path)

    setup_dirs(args.output_dir, args.plot_dir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_simulation] Device: {device}")

    # ── Load / stream dataset ──────────────────────────────────────────── #
    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    print("[run_simulation] Loading data partitions …")
    full_dataset, client_train_indices, client_test_indices, \
        global_test_indices, client_coords = get_hfl_data_partitions(
            csv_path   = args.csv_path,   # None → stream
            data_dir   = args.data_dir,
            N          = args.N,
            train_ratio= 0.8,
            random_seed= args.seed,
            subsample  = args.subsample,
            hf_token   = hf_token,
        )

    # ── Global test loader ─────────────────────────────────────────────── #
    test_subset = Subset(full_dataset, global_test_indices)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False,
                             num_workers=0, pin_memory=(device.type == "cuda"))
    print(f"[run_simulation] Global test set: {len(test_subset)} samples.")

    # ── Epicentre (Noto Peninsula 2024 earthquake) ─────────────────────── #
    epicenter = (37.50, 137.27)

    # ── Determine the actual number of classes present in the data ─────── #
    # The dataset only contains a subset of the nominal 4 damage labels (in the
    # released data, only classes 0 and 1 appear). Building a 4-output model and
    # averaging macro-F1 over the always-empty classes both wastes capacity and
    # understates F1, so size the head to the classes that actually occur.
    present_labels = np.unique(full_dataset.labels.numpy())
    num_classes = int(present_labels.max()) + 1
    print(f"[run_simulation] Damage classes present: {present_labels.tolist()} "
          f"→ model output dim {num_classes}")

    # ── Base model (cloned identically for every method) ──────────────── #
    # Set pretrained=True to utilize ImageNet features and break the guessing trap
    master_model = MultiModalFusionModel(num_classes=num_classes, pretrained=True)

    # ── Focal loss with class weights ─────────────────────────────────── #
    # Derive class counts from the training split of the full dataset
    train_labels = full_dataset.labels[
        [i for idxs in client_train_indices.values() for i in idxs]
    ].numpy()
    class_counts  = np.bincount(train_labels, minlength=num_classes)
    total_samples = len(train_labels)

    # Statistical smoothing + Clamping to avoid gradient explosion while prioritizing minority classes
    smoothed_counts = class_counts + 10
    raw_weights = total_samples / (num_classes * smoothed_counts)
    normalized_weights = raw_weights / np.mean(raw_weights)
    clamped_weights = np.clip(normalized_weights, 0.2, 5.0)
    
    class_weights = torch.tensor(clamped_weights, dtype=torch.float32)
    loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
    print(f"[run_simulation] Class weights: {class_weights.tolist()}")

    # ── Client specifications (shared across methods) ──────────────────── #
    client_specs = [
        (cid, client_coords[cid], client_train_indices[cid])
        for cid in range(args.N)
    ]

    # ── Selection methods to benchmark ────────────────────────────────── #
    methods = ["proposed", "random", "battery_only", "utility_only", "fedcs"]
    all_records = []

    for method in methods:
        print(f"\n{'='*60}")
        print(f"  Method: {method.upper()}")
        print(f"{'='*60}")

        # Reset seed so every method starts from identical client states
        np.random.seed(args.seed)

        # Build fresh clients
        clients = [
            IoTClient(
                client_id=cid,
                coords=coords,
                dataset=full_dataset,
                indices=indices,
                device=device,
            )
            for cid, coords, indices in client_specs
        ]

        uavs = initialize_uavs(args.U, seed=args.seed)

        coordinator = ClientSelectionCoordinator(
            epicenter  = epicenter,
            clients    = clients,
            uavs       = uavs,
            R_comm     = 50000.0,      # metres – 50 km covers Noto Peninsula area
            B_min_iot  = 0.2,
            B_min_uav  = 0.3,
            T_max      = 300.0,
            SNR_min    = 3.0,
        )

        global_model = copy.deepcopy(master_model).to(device)

        orchestrator = HFLOrchestrator(
            global_model        = global_model,
            clients             = clients,
            uavs                = uavs,
            selection_coordinator = coordinator,
            loss_fn             = loss_fn,
            test_loader         = test_loader,
            device              = device,
        )

        for r in range(1, args.rounds + 1):
            orchestrator.simulate_round(round_num=r, selection_method=method)
            accuracy, macro_f1 = orchestrator.evaluate()
            fairness            = orchestrator.get_jains_fairness()

            # Predicted-class distribution makes the majority-class collapse
            # visible: if one entry is ~1.0 the model predicts a single class for
            # every test sample, which is exactly why accuracy/F1 stay frozen.
            pred_dist = orchestrator.last_pred_distribution
            pred_str = "[" + " ".join(f"{p:.2f}" for p in pred_dist) + "]"

            print(
                f"  Round {r:02d} | "
                f"Acc {accuracy:.4f} | F1 {macro_f1:.4f} | "
                f"Fair {fairness:.4f} | "
                f"Comm {orchestrator.total_comm_cost:.2f} MB | "
                f"Pred {pred_str}"
            )

            # One-time warnings on the first round of the first method: surface
            # the two root causes of frozen metrics so they are not silent.
            if r == 1 and method == methods[0]:
                black_rate = full_dataset.black_chip_rate()
                if black_rate > 0.5:
                    print(
                        f"  [WARN] {black_rate*100:.0f}% of aerial chips loaded as "
                        f"BLACK (GSI tile fetch failing). The aerial image is the "
                        f"only discriminative modality — with blank images the model "
                        f"cannot learn and accuracy/F1 will stay frozen at the "
                        f"majority-class rate. Check the VM's outbound network access "
                        f"to {os.getenv('HFL_TILE_HOST', 'cyberjapandata.gsi.go.jp')} "
                        f"or pre-download chips into --data_dir."
                    )
                if pred_dist is not None and np.max(pred_dist) > 0.98:
                    print(
                        f"  [WARN] Model predicts a single class for "
                        f"{np.max(pred_dist)*100:.0f}% of test samples (majority-class "
                        f"collapse). Frozen accuracy/F1 are expected until the model "
                        f"receives usable signal."
                    )
            all_records.append({
                "method":    method,
                "round":     r,
                "accuracy":  accuracy,
                "macro_f1":  macro_f1,
                "fairness":  fairness,
                "comm_cost": orchestrator.total_comm_cost,
            })

    # ── Persist results ────────────────────────────────────────────────── #
    df_results  = pd.DataFrame(all_records)
    csv_out     = os.path.join(args.output_dir, f"simulation_results_N{args.N}.csv")
    df_results.to_csv(csv_out, index=False)
    print(f"\n[run_simulation] Results saved to {csv_out}")

    print("[run_simulation] Generating comparison plots …")
    save_comparison_plots(df_results, args.plot_dir, args.N, args.rounds)
    print(f"[run_simulation] Plots saved to {args.plot_dir}")


if __name__ == "__main__":
    main()