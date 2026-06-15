#!/bin/bash
set -e

echo "=== Running Hierarchical Federated Learning Simulations for All Network Sizes ==="

# Array of client counts to simulate (as specified in the paper)
N_values=(14 35 70 140)

# Run simulation for each client size
for N in "${N_values[@]}"; do
    echo ""
    echo "========================================================================"
    echo " Starting HFL Simulation with N = $N IoT clients"
    echo "========================================================================"
    .venv/bin/python run_simulation.py --N "$N" --rounds 30 --subsample 0.05
done

echo ""
echo "=== All simulations completed successfully! ==="
echo "CSV results are in './results/' and comparison plots are in './plots/'."
