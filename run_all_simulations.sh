#!/bin/bash
set -e

echo "=== Running Hierarchical Federated Learning Simulations for All Network Sizes ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CSV_PATH="${CSV_PATH:-./data/Final_Dataset/training_dataset_with_city.csv}"
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
PLOT_DIR="${PLOT_DIR:-./plots}"
ROUNDS="${ROUNDS:-30}"
SUBSAMPLE="${SUBSAMPLE:-0.05}"

if [ ! -f "$CSV_PATH" ]; then
    echo "Dataset CSV not found at $CSV_PATH"
    echo "Place the dataset locally before running simulations."
    exit 1
fi

# Array of client counts to simulate (as specified in the paper)
N_values=(14 35 70 140)

# Run simulation for each client size
for N in "${N_values[@]}"; do
    echo ""
    echo "========================================================================"
    echo " Starting HFL Simulation with N = $N IoT clients"
    echo "========================================================================"
    .venv/bin/python run_simulation.py \
        --csv_path "$CSV_PATH" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --plot_dir "$PLOT_DIR" \
        --N "$N" \
        --rounds "$ROUNDS" \
        --subsample "$SUBSAMPLE"
done

echo ""
echo "=== All simulations completed successfully! ==="
echo "CSV results are in './results/' and comparison plots are in './plots/'."
