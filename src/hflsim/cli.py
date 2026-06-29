"""hflsim CLI — HFL simulation runner with streaming data support.

Usage (streaming from HuggingFace, full dataset, 100 rounds):
    python -m hflsim --N 70

Usage (quick smoke-test with 5% subsample):
    python -m hflsim --N 70 --rounds 30 --subsample 0.05

Usage (PSO-optimized UAV placement):
    python -m hflsim --N 70 --rounds 30 --subsample 0.05 --use_pso_placement

Usage (local CSV if already downloaded):
    python -m hflsim --csv_path ./data/Final_Dataset/training_dataset_with_city.csv

Set HF_TOKEN when the HuggingFace repo is private:
    export HF_TOKEN=hf_xxxxxx
"""

import copy
import os
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from hflsim.data.loader import get_hfl_data_partitions, MultiModalDataset
from hflsim.models.fusion import MultiModalFusionModel, FocalLoss
from hflsim.simulation import IoTClient, UAVAggregator, ClientSelectionCoordinator, HFLOrchestrator


def parse_args():
    p = argparse.ArgumentParser(description="HFL Client-Selection Simulation Runner")

    # ── Data source ──────────────────────────────────────────────────────── #
    p.add_argument("--csv_path",    type=str,   default=None,
                   help="Path to local CSV metadata file. Omit to stream from HuggingFace.")
    p.add_argument("--data_dir",    type=str,   default="./data")
    p.add_argument("--subsample",   type=float, default=1.0,
                   help="Fraction of the dataset to use (0 < x ≤ 1).")
    p.add_argument("--hf_token",    type=str,   default=None)

    # ── HFL topology ─────────────────────────────────────────────────────── #
    p.add_argument("--N",           type=int,   default=70,
                   help="Number of IoT clients (paper: 14/35/70/140).")
    p.add_argument("--U",           type=int,   default=3,
                   help="Number of UAV edge aggregators.")
    p.add_argument("--rounds",      type=int,   default=100)
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--seed",        type=int,   default=42)

    # ── UAV placement ─────────────────────────────────────────────────────── #
    p.add_argument("--use_pso_placement", action="store_true",
                   help="Use PSO-optimized UAV positions instead of hardcoded coordinates.")
    p.add_argument("--placement_method", type=str, default="pso",
                   choices=["pso", "ga", "centroid", "random"],
                   help="Placement optimizer to use when --use_pso_placement is set.")

    # ── Output ───────────────────────────────────────────────────────────── #
    p.add_argument("--output_dir",  type=str,   default="./results")
    p.add_argument("--plot_dir",    type=str,   default="./plots")

    return p.parse_args()


def setup_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def initialize_uavs_fixed(U: int, seed: int = 42):
    """Place U UAVs at fixed positions across the Noto Peninsula (Lat 37.3–37.5, Lon 136.8–137.3)."""
    np.random.seed(seed)
    base_locations = [
        (37.38, 137.05),
        (37.40, 137.15),
        (37.35, 137.22),
    ]
    uavs = []
    for i in range(U):
        base = base_locations[i % len(base_locations)]
        loc = (base[0] + np.random.normal(0, 0.01), base[1] + np.random.normal(0, 0.01))
        uavs.append(UAVAggregator(uav_id=i, coords=loc, capacity=20))
    return uavs


def initialize_uavs_pso(client_coords: dict, U: int, seed: int = 42, method: str = "pso"):
    """Place U UAVs using PSO optimization on real client coordinates."""
    from hflsim.placement import pso_place_uavs
    print(f"[hflsim] Running {method.upper()} placement for {U} UAVs …")
    uavs = pso_place_uavs(client_coords, K=U, seed=seed, method=method)
    print(f"[hflsim] PSO placement complete. UAV coords: {[u.coords for u in uavs]}")
    return uavs


def save_comparison_plots(df_results: pd.DataFrame, plot_dir: str, N: int, rounds: int):
    """Generate and save the four comparison plots."""
    methods = df_results["method"].unique()
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    colors = {
        "proposed":     "#2ca02c",
        "random":       "#7f7f7f",
        "battery_only": "#d62728",
        "utility_only": "#ff7f0e",
        "fedcs":        "#1f77b4",
    }
    labels = {
        "proposed":     "Proposed (UCB + Rep + Res)",
        "random":       "Random Selection",
        "battery_only": "Battery-Only",
        "utility_only": "Utility-Only",
        "fedcs":        "FedCS Baseline",
    }

    def _save(fig_name, xlabel, ylabel, title, col, ylim=None, legend_loc="lower right"):
        plt.figure(figsize=(8, 5))
        for m in methods:
            sub = df_results[df_results["method"] == m]
            plt.plot(sub["round"], sub[col],
                     label=labels.get(m, m), color=colors.get(m), linewidth=2.0)
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

    _save("accuracy", "Global Communication Round", "Test Accuracy",
          f"Global Model Accuracy Convergence (N={N})", "accuracy")
    _save("f1",       "Global Communication Round", "Macro F1 Score",
          f"Global Model Macro F1 Convergence (N={N})", "macro_f1")
    _save("fairness", "Global Communication Round", "Jain's Fairness Index",
          f"Jain's Fairness Index Evolution (N={N})", "fairness",
          ylim=(0.5, 1.05), legend_loc="lower left")

    plt.figure(figsize=(8, 5))
    for m in methods:
        sub = df_results[df_results["method"] == m]
        plt.plot(sub["comm_cost"], sub["accuracy"],
                 label=labels.get(m, m), color=colors.get(m), linewidth=2.0)
    plt.title(f"Accuracy vs. Communication Cost (N={N})", fontsize=12, fontweight="bold")
    plt.xlabel("Total Communication Load (MB)", fontsize=10)
    plt.ylabel("Test Accuracy", fontsize=10)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"efficiency_N{N}.pdf"))
    plt.savefig(os.path.join(plot_dir, f"efficiency_N{N}.png"), dpi=150)
    plt.close()


def main():
    args = parse_args()

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
    print(f"[hflsim] Device: {device}")

    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    print("[hflsim] Loading data partitions …")
    full_dataset, client_train_indices, client_test_indices, \
        global_test_indices, client_coords = get_hfl_data_partitions(
            csv_path    = args.csv_path,
            data_dir    = args.data_dir,
            N           = args.N,
            train_ratio = 0.8,
            random_seed = args.seed,
            subsample   = args.subsample,
            hf_token    = hf_token,
        )

    test_subset = Subset(full_dataset, global_test_indices)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False,
                             num_workers=0, pin_memory=(device.type == "cuda"))
    print(f"[hflsim] Global test set: {len(test_subset)} samples.")

    epicenter = (37.50, 137.27)

    present_labels = np.unique(full_dataset.labels.numpy())
    num_classes = int(present_labels.max()) + 1
    print(f"[hflsim] Damage classes present: {present_labels.tolist()} → model output dim {num_classes}")

    master_model = MultiModalFusionModel(num_classes=num_classes, pretrained=True)

    train_labels = full_dataset.labels[
        [i for idxs in client_train_indices.values() for i in idxs]
    ].numpy()
    class_counts  = np.bincount(train_labels, minlength=num_classes)
    total_samples = len(train_labels)
    smoothed_counts    = class_counts + 10
    raw_weights        = total_samples / (num_classes * smoothed_counts)
    normalized_weights = raw_weights / np.mean(raw_weights)
    clamped_weights    = np.clip(normalized_weights, 0.2, 5.0)
    class_weights = torch.tensor(clamped_weights, dtype=torch.float32)
    loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
    print(f"[hflsim] Class weights: {class_weights.tolist()}")

    client_specs = [
        (cid, client_coords[cid], client_train_indices[cid])
        for cid in range(args.N)
    ]

    methods = ["proposed", "random", "battery_only", "utility_only", "fedcs"]
    all_records = []

    for method in methods:
        print(f"\n{'='*60}\n  Method: {method.upper()}\n{'='*60}")

        np.random.seed(args.seed)

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

        if args.use_pso_placement:
            uavs = initialize_uavs_pso(
                client_coords, args.U, seed=args.seed, method=args.placement_method
            )
        else:
            uavs = initialize_uavs_fixed(args.U, seed=args.seed)

        coordinator = ClientSelectionCoordinator(
            epicenter  = epicenter,
            clients    = clients,
            uavs       = uavs,
            R_comm     = 50000.0,
            B_min_iot  = 0.2,
            B_min_uav  = 0.3,
            T_max      = 300.0,
            SNR_min    = 3.0,
        )

        global_model = copy.deepcopy(master_model).to(device)

        orchestrator = HFLOrchestrator(
            global_model          = global_model,
            clients               = clients,
            uavs                  = uavs,
            selection_coordinator = coordinator,
            loss_fn               = loss_fn,
            test_loader           = test_loader,
            device                = device,
        )

        for r in range(1, args.rounds + 1):
            orchestrator.simulate_round(round_num=r, selection_method=method)
            accuracy, macro_f1 = orchestrator.evaluate()
            fairness = orchestrator.get_jains_fairness()

            pred_dist = orchestrator.last_pred_distribution
            pred_str  = "[" + " ".join(f"{p:.2f}" for p in pred_dist) + "]"

            print(
                f"  Round {r:02d} | "
                f"Acc {accuracy:.4f} | F1 {macro_f1:.4f} | "
                f"Fair {fairness:.4f} | "
                f"Comm {orchestrator.total_comm_cost:.2f} MB | "
                f"Pred {pred_str}"
            )

            if r == 1 and method == methods[0]:
                black_rate = full_dataset.black_chip_rate()
                if black_rate > 0.5:
                    print(
                        f"  [WARN] {black_rate*100:.0f}% of aerial chips loaded as BLACK "
                        f"(GSI tile fetch failing). Check outbound network access to "
                        f"{os.getenv('HFL_TILE_HOST', 'cyberjapandata.gsi.go.jp')} "
                        f"or pre-download chips into --data_dir."
                    )
                if pred_dist is not None and np.max(pred_dist) > 0.98:
                    print(
                        f"  [WARN] Model predicts a single class for "
                        f"{np.max(pred_dist)*100:.0f}% of test samples (majority-class collapse)."
                    )

            all_records.append({
                "method":    method,
                "round":     r,
                "accuracy":  accuracy,
                "macro_f1":  macro_f1,
                "fairness":  fairness,
                "comm_cost": orchestrator.total_comm_cost,
            })

    df_results = pd.DataFrame(all_records)
    csv_out    = os.path.join(args.output_dir, f"simulation_results_N{args.N}.csv")
    df_results.to_csv(csv_out, index=False)
    print(f"\n[hflsim] Results saved to {csv_out}")

    print("[hflsim] Generating comparison plots …")
    save_comparison_plots(df_results, args.plot_dir, args.N, args.rounds)
    print(f"[hflsim] Plots saved to {args.plot_dir}")


if __name__ == "__main__":
    main()
