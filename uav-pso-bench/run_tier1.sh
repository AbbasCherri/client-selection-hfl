#!/bin/bash
# run_tier1.sh
# Runs the uav-pso-bench Tier-1 placement benchmark end-to-end in a single pass:
#   run (the method x scenario x seed grid) -> analyze (summary table) -> plot
#   (convergence figures). Results land in the config's results_dir.
#
# Optional overrides:
#   CONFIG   - experiment config to run   (default: configs/tier1_core.yaml)
#   WORKERS  - joblib worker count; if set, overrides n_workers in the config
#
# Examples:
#   ./run_tier1.sh                          # full tier1_core
#   CONFIG=configs/smoke.yaml ./run_tier1.sh
#   WORKERS=4 ./run_tier1.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${CONFIG:-configs/tier1_core.yaml}"

# Determine Python interpreter (prefer a local .venv, then the parent repo's .venv).
if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -x "../.venv/bin/python" ]; then
    PYTHON="../.venv/bin/python"
else
    PYTHON="python3"
fi

# Make the src-layout package importable without requiring `pip install -e .`.
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

# Optional worker-count override (per-machine tuning).
if [ -n "$WORKERS" ]; then
    export UAVBENCH_N_WORKERS="$WORKERS"
fi

echo "=== uav-pso-bench Tier-1 benchmark ==="
echo "Python interpreter : $PYTHON"
echo "Config             : $CONFIG"
[ -n "$WORKERS" ] && echo "Workers (override) : $WORKERS"
echo ""

echo "--- [1/3] run ---"
"$PYTHON" -m uavbench run --config "$CONFIG"

echo ""
echo "--- [2/3] analyze ---"
"$PYTHON" -m uavbench analyze --config "$CONFIG"

echo ""
echo "--- [3/3] plot ---"
"$PYTHON" -m uavbench plot --config "$CONFIG"

echo ""
echo "=== Tier-1 benchmark complete ==="
RESULTS_DIR="$("$PYTHON" -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['results_dir'])")"
echo "Results : $RESULTS_DIR (runs.parquet, summary.parquet, convergence_*.png)"
