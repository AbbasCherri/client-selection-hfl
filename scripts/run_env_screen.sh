#!/usr/bin/env bash
# Phase 6 — one-at-a-time screening of the simulator's ENVIRONMENT constants.
#
# The device/energy model is full of hand-picked numbers. None is measured, none
# is cited. The defensible response is NOT to tune them — tuning environment
# constants is how you get a simulator that flatters the proposed method. It is
# to show the SIGN of the proposed-vs-baseline gap survives a wide perturbation
# of each, and to report openly any constant where it does not.
#
# T_MAX_S is the headline. At 300 s against compute times of U[50,250]+N(0,30),
# the deadline excludes 0.28% of devices on the raw gate and 8.4% once the
# adaptive margin applies — so FedCS's entire distinguishing mechanism
# (deadline-constrained greedy) barely fires and it degenerates toward
# cheapest-first capacity fill. Nishio & Yonetani sweep the deadline rather than
# fixing it, so screening it is what makes FedCS a fair baseline, not a courtesy.
#
# Arms: ucb (proposed), fair_mab (strongest baseline on accuracy), random
# (floor). N=500, where the selection effect is decisive.
#
# ~4 h on a 12-vCPU VM.  Usage:  nohup ./scripts/run_env_screen.sh &
set -euo pipefail
cd "$(dirname "$0")/.."

# Self-activate the venv. The other phase scripts assume the caller already did,
# which holds for an interactive shell but NOT under `sudo` — and the results
# tree on the VM is root-owned, so sudo is how this has to run. Without this,
# `python` resolves to nothing and the whole screen dies on cell 1.
if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

LOG=results/env_screen.log
mkdir -p results
exec > >(tee -a "$LOG") 2>&1

# tag | dotted constant path | values (baseline value included so every cell is
# scored under the identical harness rather than against a remembered number)
#
# Pipe-delimited on purpose. The first version of this loop aligned the columns
# with runs of spaces and split them back apart with awk/cut, which put the
# CONSTANT PATH into the value list — the run died on `T_MAX_S = "uavbench.fl.
# device_state.T_MAX_S"` after producing one garbage cell directory. Three
# unambiguous fields beat pretty alignment.
SETTINGS=(
  "tmax|uavbench.fl.device_state.T_MAX_S|150 225 300"
  "bmin|uavbench.fl.device_state.B_MIN|0.10 0.20 0.30"
  "snrmin|uavbench.fl.device_state.SNR_MIN_DB|1.5 3.0 6.0"
  "ucbc|uavbench.fl.client_selection.UCB_C|0.707 1.414 2.828"
)

for s in "${SETTINGS[@]}"; do
    IFS='|' read -r tag path values <<< "$s"
    for v in $values; do
        out="results/env_screen/${tag}_${v}"
        if [[ -f "$out/config.selection_sweep.resolved.yaml" ]]; then
            echo "--- [$(date -Is)] $tag=$v already done, skipping ---"
            continue
        fi
        echo
        echo "--- [$(date -Is)] screening $tag = $v ($path) ---"
        cat > "/tmp/env_${tag}_${v}.yaml" <<EOF
extends: $(pwd)/configs/env_screen_base.yaml
name: env_${tag}_${v}
results_dir: $out
fl:
  const_overrides:
    $path: $v
EOF
        python -m uavbench run_selection_sim --config "/tmp/env_${tag}_${v}.yaml"
    done
done

echo
echo "--- [$(date -Is)] environment screening complete ---"
