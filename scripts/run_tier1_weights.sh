#!/usr/bin/env bash
# Phase 3 of the August 2026 rigor plan: Tier-1 fitness-weight sensitivity.
#
# Answers the circularity objection — the weights w=(0.811, 0.03, 0.159) were
# tuned on downstream proposed_hfl accuracy, then used to crown PSO. If the
# method ranking is invariant across the weight sweep, that objection is dead.
# If it is NOT invariant, that is the honest headline and the weights become a
# stated limitation. Either outcome is reportable; do not re-run to taste.
#
# Only pso/ga/random are re-run (their placement depends on w). The other four
# Tier-1 methods are re-scored offline at zero cost — see
# uavbench.analysis.weight_sensitivity and configs/tier1_weight_sweep.yaml.
#
# ~2 h on a 12-vCPU VM.  Usage:  nohup ./scripts/run_tier1_weights.sh &
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=results/tier1_weights.log
mkdir -p results
exec > >(tee -a "$LOG") 2>&1

# w1 swept; (w2, w3) redistributed in the tuned 0.1587:0.8413 proportion, plus
# the paper's originally stated split as an external reference point.
# Keep in sync with uavbench.analysis.weight_sensitivity.weight_grid().
SETTINGS=(
    "w1_0.6      0.6    0.0635 0.3365"
    "w1_0.7      0.7    0.0476 0.2524"
    "w1_0.9      0.9    0.0159 0.0841"
    "paper       0.6    0.3    0.1"
)

for s in "${SETTINGS[@]}"; do
    read -r tag w1 w2 w3 <<< "$s"
    out="results/tier1_weight_sweep/$tag"
    echo
    echo "--- [$(date -Is)] weight setting $tag: w=($w1, $w2, $w3) ---"
    # Written as an override file rather than sed-editing the config: patching
    # config files in place on the VM is how the stress grid silently drifted.
    cat > "/tmp/tier1_w_$tag.yaml" <<EOF
extends: $(pwd)/configs/tier1_weight_sweep.yaml
name: tier1_w_$tag
results_dir: $out
fitness:
  w1: $w1
  w2: $w2
  w3: $w3
EOF
    python -m uavbench run --config "/tmp/tier1_w_$tag.yaml"
done

echo
echo "--- [$(date -Is)] w=0.811 baseline already exists in results/tier1_core ---"
echo "--- [$(date -Is)] Tier-1 weight sweep complete ---"
