#!/usr/bin/env bash
# Full rebuild at PIPELINE_VERSION 5 — everything, in dependency order.
#
# Why every result is being recomputed
# ------------------------------------
# The UAV altitude band was 20-120 m (the 400 ft small-UAS ceiling) while R_comm
# ran up to 20 km. Those are physically incompatible under the Al-Hourani
# channel this benchmark is built on:
#
#   * the radius-vs-altitude curve peaks at a fixed elevation angle (20.34 deg
#     suburban), so the coherent ground radius is z/tan(theta_opt) = 54-324 m
#     under that band. Every R_comm above ~324 m pinned the optimum at the
#     ceiling, making the vertical decision degenerate — the exact failure
#     link.py was written to remove;
#   * at 20 km the elevation angle is 0.34 deg, giving P(LoS) ~ 3%. The link was
#     97% NLoS: an aerial base station with its line-of-sight advantage switched
#     off. LinkModel never objected because it back-solves whatever path-loss
#     budget the configured R_comm needs.
#
# So placement was being evaluated almost entirely outside the regime where the
# model has physics, and the null results it produced cannot be trusted.
#
# The band is now 100-2000 m with R_comm = 5 km, inside the coherent interval
# (270 m - 5.4 km), guarded by tests/sanity_checks/check_altitude_band.py. The
# ceiling is the reach — z/tan(theta_opt) — so 2000 m rather than 1000 m is what
# makes a modest fleet viable: 78.5 km^2 of ground per aircraft against 12.6.
# At 2 km even 60 UAVs put too few clients in range and the task collapsed onto
# the majority class in three successive attempts.
#
# Operating point K=60 / capacity=2, set from measured participation
# (results/probe_participation): 110 clients selected per round against the 120
# of the last configuration that learned properly, 82.5% coverage, and 120 slots
# against ~165 covered so the selection rule still binds.
#
# Order and why
# -------------
#   1. class realism   (~3 h)  cheapest; re-confirms the +0.065 selection result
#   2. fleet sweep     (~12 h) K is the coverage knob; sizes the operating point
#   3. coverage sweep  (~12 h) R axis, rebuilt grid, new results dir
#   4. paper_full      (~40 h) biggest — run last, once the regime is settled
#
# ~3 days total. Every job is checkpointed and PIPELINE_VERSION 5 refuses v4
# checkpoints, so this is safe to interrupt and relaunch: finished v5 jobs are
# skipped, v4 leftovers are recomputed.
#
# SELF-STOPS ONLY after the last block succeeds. A failing block is logged and
# the run continues to the next one (they are independent experiments), but the
# VM is left up so the failure can be inspected.
#
# Usage:  nohup ./scripts/run_rebuild_v5.sh &
set -uo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.hf_env" ]]; then
    HF_TOKEN=$(grep -o 'hf_[A-Za-z0-9]*' "$HOME/.hf_env" | head -1)
    export HF_TOKEN
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN missing — the real-data pipeline cannot run"; exit 1
fi

LOG=results/rebuild_v5.log
mkdir -p results
exec > >(tee -a "$LOG") 2>&1
say() { echo; echo "[$(date -Is)] $*"; }

FAILED=""
commit_block() {
    git add -A -- results || true
    if git diff --cached --quiet; then
        say "  nothing new to commit"
    else
        git -c user.email=vm@local -c user.name=vm commit -q -m "Add $1 results $(date -Is)" || true
        say "  committed $(git rev-parse --short HEAD)"
    fi
}
run_block() {                     # run_block <name> <results_dir> <cmd...>
    local name="$1"; local rdir="$2"; shift 2
    say "--- $name ---"
    if "$@"; then
        # A block that finishes is not a block that worked. Three configurations
        # on 2026-08-08 produced complete, well-formed results in which the model
        # had learned only the majority class, and the pipeline was perfectly
        # happy with all of them. Gate on the outcome before banking it.
        if [[ -n "$rdir" ]] && ! python scripts/gate_collapse.py "$rdir"; then
            say "  !! $name COLLAPSED — results are degenerate, not a usable result"
            FAILED="$FAILED $name(collapsed)"
        fi
        commit_block "$name"
    else
        say "  !! $name FAILED (exit $?) — continuing to the next block"
        FAILED="$FAILED $name"
        commit_block "$name-partial"
    fi
}

say "===== v5 rebuild: altitude 100-2000 m, R_comm 5 km, K=60/cap=2 ====="

# The altitude gate is the whole reason this rebuild exists: if the band and the
# configured radii ever drift back out of the coherent interval, three days of
# compute would reproduce the degenerate regime and look entirely normal doing it.
for chk in check_altitude_band check_collapse_guard check_placement_methods \
           check_uav_sweep check_class_histograms check_mclp; do
    python "tests/sanity_checks/${chk}.py" || { say "${chk} FAILED — aborting"; exit 1; }
done

# ---- 1. class realism (6 arms) -------------------------------------------
for arm in class_realism class_realism_pseudo class_realism_noplace \
           class_realism_none class_realism_bind class_realism_bind_noplace; do
    rdir=$(python -c "import sys,yaml;sys.path.insert(0,'src');from uavbench.runner import load_config;print(load_config('configs/${arm}.yaml')['results_dir'])")
    run_block "$arm" "$rdir" python -m uavbench run_paper_sim --config "configs/${arm}.yaml"
    # Every arm shares K, capacity and R_comm — they differ only in class_source
    # and placement_class_aware. So if the first one collapses, all six will, and
    # the two sweeps after them will too. Stop here rather than spend three days
    # confirming it six more times.
    if [[ "$FAILED" == *"(collapsed)"* ]]; then
        say "FIRST ARM COLLAPSED at the current operating point — aborting the"
        say "whole rebuild. Re-measure participation before relaunching; do not"
        say "just rerun this script."
        exit 1
    fi
done
for grp in main bind; do
    python scripts/merge_class_realism.py --group "$grp" || FAILED="$FAILED merge-$grp"
done
for metric in macro_f1 accuracy f1_collapsed f1_missing f1_obstructed f1_survived; do
    python -m uavbench significance --config results/class_realism_main_merged \
        --metric "$metric" --reference true_placeaware --correction-scope group || true
    python -m uavbench significance --config results/class_realism_bind_merged \
        --metric "$metric" --reference bind_placeaware --correction-scope group || true
done
commit_block "class-realism-v5-significance"

# ---- 2. fleet-size sweep --------------------------------------------------
run_block "uav-count" results/paper_uav_count python -m uavbench run_uav_sweep --config configs/paper_uav_count.yaml
for metric in macro_f1 coverage_pct accuracy; do
    python -m uavbench significance --config results/paper_uav_count \
        --metric "$metric" --reference mclp_place --correction-scope group || true
done
commit_block "uav-count-significance"

# ---- 3. coverage sweep (rebuilt grid, new dir) ----------------------------
run_block "coverage-v5" results/paper_coverage_v5 python -m uavbench run_coverage_sweep --config configs/paper_coverage.yaml
for metric in macro_f1 coverage_pct accuracy; do
    python -m uavbench significance --config results/paper_coverage_v5 \
        --metric "$metric" --reference mclp_place --correction-scope group || true
done
commit_block "coverage-v5-significance"

# ---- 4. paper_full (headline FL table) ------------------------------------
run_block "paper-full-v5" results/paper_full python -m uavbench run_paper_sim --config configs/paper_full.yaml
for metric in macro_f1 accuracy f1_collapsed f1_missing f1_obstructed f1_survived; do
    python -m uavbench significance --config results/paper_full \
        --metric "$metric" --reference proposed_hfl --correction-scope group || true
done
commit_block "paper-full-v5-significance"

say "===== rebuild complete ====="
du -sh results/class_realism_* results/paper_uav_count results/paper_coverage_v5 \
       results/paper_full 2>/dev/null || true

if [[ -n "$FAILED" ]]; then
    say "BLOCKS FAILED:$FAILED — VM left RUNNING for inspection"
    exit 1
fi

say "all blocks succeeded — stopping the VM"
sudo shutdown -h +2 "v5 rebuild complete"
