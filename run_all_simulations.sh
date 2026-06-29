#!/bin/bash
# run_all_simulations.sh
# Runs HFL simulations for all four IoT network sizes described in the paper.
# Data is streamed from HuggingFace – no local dataset download required.
#
# Required environment variable (if the HF repo is private):
#   export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
#
# Optional overrides:
#   CSV_PATH   – path to a locally-downloaded CSV (skips streaming)
#   DATA_DIR   – root directory for local image chips / tile cache
#   OUTPUT_DIR – directory for CSV result files   (default: ./results)
#   PLOT_DIR   – directory for comparison plots   (default: ./plots)
#   ROUNDS     – global communication rounds       (default: 30)
#   SUBSAMPLE  – fraction of dataset rows to use  (default: 0.05)

set -e

echo "=== Hierarchical Federated Learning Simulations (all network sizes) ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
PLOT_DIR="${PLOT_DIR:-./plots}"
ROUNDS="${ROUNDS:-30}"
SUBSAMPLE="${SUBSAMPLE:-0.05}"

# Determine Python interpreter (prefer .venv if present)
if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

echo "Python interpreter : $PYTHON"
echo "Data directory     : $DATA_DIR"
echo "Output directory   : $OUTPUT_DIR"
echo "Plot directory     : $PLOT_DIR"
echo "Rounds             : $ROUNDS"
echo "Subsample fraction : $SUBSAMPLE"
echo ""

# Client counts from the paper (Section V-A)
N_values=(14 35 70 140)

# Build the base argument list; append --csv_path only when a local file exists.
BASE_ARGS=(
    --data_dir   "$DATA_DIR"
    --output_dir "$OUTPUT_DIR"
    --plot_dir   "$PLOT_DIR"
    --rounds     "$ROUNDS"
    --subsample  "$SUBSAMPLE"
)

if [ -n "$CSV_PATH" ] && [ -f "$CSV_PATH" ]; then
    echo "Using local CSV: $CSV_PATH"
    BASE_ARGS+=(--csv_path "$CSV_PATH")
else
    echo "No local CSV found – streaming from HuggingFace."
    if [ -z "$HF_TOKEN" ]; then
        echo "WARNING: HF_TOKEN is not set. Streaming may fail for private repos."
    fi
fi

for N in "${N_values[@]}"; do
    echo ""
    echo "========================================================================"
    echo "  Starting HFL Simulation with N = $N IoT clients"
    echo "========================================================================"
    "$PYTHON" -m hflsim "${BASE_ARGS[@]}" --N "$N"
done

echo ""
echo "=== All simulations completed! ==="
echo "Results : $OUTPUT_DIR"
echo "Plots   : $PLOT_DIR"
