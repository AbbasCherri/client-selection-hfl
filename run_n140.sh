#!/bin/bash
# run_n140.sh
# Runs a single HFL simulation with N = 140 IoT clients.
# This is the only paper size where the number of clients exceeds total UAV
# capacity (default 3 UAVs x 20 = 60 slots), so client selection actually binds
# and the proposed algorithm can differ from the baselines.
#
# Required environment variable (if the HF repo is private):
#   export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
#
# Optional overrides:
#   CSV_PATH   - path to a locally-downloaded CSV (skips streaming)
#   DATA_DIR   - root directory for local image chips / tile cache (default: ./data)
#   OUTPUT_DIR - directory for CSV result files   (default: ./results)
#   PLOT_DIR   - directory for comparison plots   (default: ./plots)
#   ROUNDS     - global communication rounds       (default: 30)
#   SUBSAMPLE  - fraction of dataset rows to use   (default: 0.05)
#   U          - number of UAV aggregators         (default: 3)

set -e

echo "=== Hierarchical Federated Learning Simulation (N = 140) ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
PLOT_DIR="${PLOT_DIR:-./plots}"
ROUNDS="${ROUNDS:-30}"
SUBSAMPLE="${SUBSAMPLE:-0.05}"
U="${U:-3}"

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
echo "UAVs               : $U"
echo ""

ARGS=(
    --N          140
    --U          "$U"
    --data_dir   "$DATA_DIR"
    --output_dir "$OUTPUT_DIR"
    --plot_dir   "$PLOT_DIR"
    --rounds     "$ROUNDS"
    --subsample  "$SUBSAMPLE"
)

if [ -n "$CSV_PATH" ] && [ -f "$CSV_PATH" ]; then
    echo "Using local CSV: $CSV_PATH"
    ARGS+=(--csv_path "$CSV_PATH")
else
    echo "No local CSV found - streaming from HuggingFace."
    if [ -z "$HF_TOKEN" ]; then
        echo "WARNING: HF_TOKEN is not set. Streaming may fail for private repos."
    fi
fi

"$PYTHON" -m hflsim "${ARGS[@]}"

echo ""
echo "=== Simulation complete (N = 140) ==="
echo "Results : $OUTPUT_DIR/simulation_results_N140.csv"
echo "Plots   : $PLOT_DIR"
