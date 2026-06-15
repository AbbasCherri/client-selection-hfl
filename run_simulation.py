import os
import argparse
import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# Import components from our modules
from data_loader import get_hfl_data_partitions, MultiModalDataset
from models import MultiModalFusionModel, FocalLoss
from simulation import IoTClient, UAVAggregator, ClientSelectionCoordinator, HFLOrchestrator

def parse_args():
    parser = argparse.ArgumentParser(description="HFL Client Selection Simulation Runner")
    parser.add_argument("--csv_path", type=str, default="./data/Final_Dataset/training_dataset_with_city.csv", help="Path to CSV metadata file")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory where image files reside")
    parser.add_argument("--N", type=int, default=70, help="Number of IoT clients (default 70)")
    parser.add_argument("--U", type=int, default=3, help="Number of UAV edge aggregators (default 3)")
    parser.add_argument("--rounds", type=int, default=30, help="Number of HFL rounds (default 30)")
    parser.add_argument("--subsample", type=float, default=0.05, help="Subsample fraction of the 128k buildings for faster CPU simulation")
    parser.add_argument("--epochs", type=int, default=1, help="Local training epochs per client per round")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for local optimizer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save CSV results")
    parser.add_argument("--plot_dir", type=str, default="./plots", help="Directory to save comparison plots")
    return parser.parse_args()

def setup_directories(output_dir, plot_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

def initialize_uavs(U, seed=42):
    """
    Spreads U UAVs across the geographic layout.
    """
    np.random.seed(seed)
    # The buildings range around 37.3 - 37.5 Latitude and 136.8 - 137.3 Longitude.
    # We place UAVs near the center of building densities.
    # UAV coordinates (latitude, longitude)
    uavs = []
    uav_locations = [
        (37.38, 137.05), # Central/West area
        (37.40, 137.15), # North-East area
        (37.35, 137.22)  # South-East area
    ]
    for i in range(U):
        loc = uav_locations[i % len(uav_locations)]
        # Add a tiny random offset
        loc = (loc[0] + np.random.normal(0, 0.01), loc[1] + np.random.normal(0, 0.01))
        uavs.append(UAVAggregator(uav_id=i, coords=loc, capacity=20))
    return uavs

def save_comparison_plots(df_results, plot_dir, N, rounds):
    """
    Generates and saves visual plots comparing the 5 selection methods.
    """
    methods = df_results['method'].unique()
    
    # Set up styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    colors = {
        'proposed': '#2ca02c',     # Green
        'random': '#7f7f7f',       # Gray
        'battery_only': '#d62728', # Red
        'utility_only': '#ff7f0e', # Orange
        'fedcs': '#1f77b4'         # Blue
    }
    labels = {
        'proposed': 'Proposed (UCB + Rep + Res)',
        'random': 'Random Selection',
        'battery_only': 'Battery-Only',
        'utility_only': 'Utility-Only',
        'fedcs': 'FedCS Baseline'
    }

    # Plot 1: Accuracy Convergence
    plt.figure(figsize=(8, 5))
    for method in methods:
        sub_df = df_results[df_results['method'] == method]
        plt.plot(sub_df['round'], sub_df['accuracy'], label=labels.get(method, method), 
                 color=colors.get(method, None), linewidth=2.0)
    plt.title(f"Global Model Accuracy Convergence (IoT Clients N={N})", fontsize=12, fontweight='bold')
    plt.xlabel("Global Communication Round", fontsize=10)
    plt.ylabel("Test Accuracy", fontsize=10)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"accuracy_N{N}.pdf"))
    plt.savefig(os.path.join(plot_dir, f"accuracy_N{N}.png"), dpi=150)
    plt.close()

    # Plot 2: Macro F1 Convergence
    plt.figure(figsize=(8, 5))
    for method in methods:
        sub_df = df_results[df_results['method'] == method]
        plt.plot(sub_df['round'], sub_df['macro_f1'], label=labels.get(method, method), 
                 color=colors.get(method, None), linewidth=2.0)
    plt.title(f"Global Model Macro F1 Convergence (IoT Clients N={N})", fontsize=12, fontweight='bold')
    plt.xlabel("Global Communication Round", fontsize=10)
    plt.ylabel("Macro F1 Score", fontsize=10)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"f1_N{N}.pdf"))
    plt.savefig(os.path.join(plot_dir, f"f1_N{N}.png"), dpi=150)
    plt.close()

    # Plot 3: Fairness Evolution
    plt.figure(figsize=(8, 5))
    for method in methods:
        sub_df = df_results[df_results['method'] == method]
        plt.plot(sub_df['round'], sub_df['fairness'], label=labels.get(method, method), 
                 color=colors.get(method, None), linewidth=2.0)
    plt.title(f"Jain's Fairness Index Evolution (IoT Clients N={N})", fontsize=12, fontweight='bold')
    plt.xlabel("Global Communication Round", fontsize=10)
    plt.ylabel("Jain's Fairness Index", fontsize=10)
    plt.ylim(0.5, 1.05)
    plt.legend(loc='lower left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"fairness_N{N}.pdf"))
    plt.savefig(os.path.join(plot_dir, f"fairness_N{N}.png"), dpi=150)
    plt.close()

    # Plot 4: Accuracy vs Communication Cost
    plt.figure(figsize=(8, 5))
    for method in methods:
        sub_df = df_results[df_results['method'] == method]
        plt.plot(sub_df['comm_cost'], sub_df['accuracy'], label=labels.get(method, method), 
                 color=colors.get(method, None), linewidth=2.0)
    plt.title(f"Accuracy vs. Communication Cost (IoT Clients N={N})", fontsize=12, fontweight='bold')
    plt.xlabel("Total Communication Load (MB)", fontsize=10)
    plt.ylabel("Test Accuracy", fontsize=10)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"efficiency_N{N}.pdf"))
    plt.savefig(os.path.join(plot_dir, f"efficiency_N{N}.png"), dpi=150)
    plt.close()

def main():
    args = parse_args()
    setup_directories(args.output_dir, args.plot_dir)
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running simulation on device: {device}")

    # Check for data presence, trigger download if not exists
    if not os.path.exists(args.csv_path):
        print("Dataset CSV not found locally. Downloading from Hugging Face...")
        from download_data import download_dataset
        download_dataset()

    # Load dataset
    print(f"Loading metadata from {args.csv_path}...")
    df_meta = pd.read_csv(args.csv_path)
    
    # Subsampling for CPU efficiency
    if args.subsample < 1.0:
        sub_size = int(len(df_meta) * args.subsample)
        print(f"Subsampling dataset from {len(df_meta)} down to {sub_size} buildings ({args.subsample*100}%)...")
        df_meta = df_meta.sample(n=sub_size, random_state=args.seed).reset_index(drop=True)
        # Re-save subsampled CSV temporarily or pass directly
        sub_csv_path = "./data/subsampled_metadata.csv"
        df_meta.to_csv(sub_csv_path, index=False)
        csv_path_to_use = sub_csv_path
    else:
        csv_path_to_use = args.csv_path

    # Partition dataset into train/test and clients
    full_dataset, client_train_indices, client_test_indices, global_test_indices = get_hfl_data_partitions(
        csv_path=csv_path_to_use,
        data_dir=args.data_dir,
        N=args.N,
        train_ratio=0.8,
        random_seed=args.seed
    )
    
    # Setup global test loader
    test_dataset = Subset(full_dataset, global_test_indices)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"Global Test Set Size: {len(test_dataset)} samples.")

    # Epicenter coordinates from Noto Peninsula earthquake
    epicenter = (37.50, 137.27)

    # Initialize a master global model to clone from (for identical starting point)
    master_model = MultiModalFusionModel(num_classes=4, pretrained=False)
    
    # Setup class weights for Focal Loss
    labels = df_meta['damage_val'].values
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = torch.tensor([total_samples / (4.0 * count + 1e-5) for count in class_counts], dtype=torch.float32)
    loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)

    # Methods to evaluate
    methods = ["proposed", "random", "battery_only", "utility_only", "fedcs"]
    all_round_records = []

    # Run simulation for each selection scheme
    for method in methods:
        print(f"\n==================================================")
        print(f"Starting Simulation: {method.upper()}")
        print(f"==================================================")
        
        # Reset seed for identical client initialization
        np.random.seed(args.seed)
        
        # Initialize clients with the exact same initial state for each method
        clients = []
        for client_id in range(args.N):
            indices = client_train_indices[client_id]
            # Locate client coordinate as mean building coordinate in cluster
            client_df = df_meta.iloc[indices]
            coords = (client_df['latitude'].mean(), client_df['longitude'].mean())
            
            client = IoTClient(
                client_id=client_id,
                coords=coords,
                dataset=full_dataset,
                indices=indices,
                device=device
            )
            clients.append(client)
            
        # Initialize UAVs
        uavs = initialize_uavs(args.U, seed=args.seed)
        
        # Initialize Client Selection Coordinator
        coordinator = ClientSelectionCoordinator(
            epicenter=epicenter,
            clients=clients,
            uavs=uavs,
            R_comm=500.0,
            B_min_iot=0.2,
            B_min_uav=0.3,
            T_max=300.0,
            SNR_min=3.0
        )
        
        # Copy global model from master to start identical
        global_model = copy.deepcopy(master_model).to(device)
        
        # Create orchestrator
        orchestrator = HFLOrchestrator(
            global_model=global_model,
            clients=clients,
            uavs=uavs,
            selection_coordinator=coordinator,
            loss_fn=loss_fn,
            test_loader=test_loader,
            device=device
        )
        
        # Run communication rounds
        for r in range(1, args.rounds + 1):
            orchestrator.simulate_round(round_num=r, selection_method=method)
            
            # Evaluate model
            accuracy, macro_f1 = orchestrator.evaluate()
            fairness = orchestrator.get_jains_fairness()
            
            print(f"Round {r:02d} | Accuracy: {accuracy:.4f} | Macro F1: {macro_f1:.4f} | Fairness: {fairness:.4f} | Comm Cost: {orchestrator.total_comm_cost:.2f} MB")
            
            all_round_records.append({
                "method": method,
                "round": r,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "fairness": fairness,
                "comm_cost": orchestrator.total_comm_cost
            })

    # Save results to CSV
    df_results = pd.DataFrame(all_round_records)
    csv_out_path = os.path.join(args.output_dir, f"simulation_results_N{args.N}.csv")
    df_results.to_csv(csv_out_path, index=False)
    print(f"\nSimulation completed. Results written to {csv_out_path}")

    # Generate and save comparison plots
    print("Generating comparison graphs...")
    save_comparison_plots(df_results, args.plot_dir, args.N, args.rounds)
    print(f"Plots saved to {args.plot_dir}")

if __name__ == "__main__":
    main()
