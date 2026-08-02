#!/usr/bin/env bash
# Unattended supervisor for the VM #1 batch of the August 2026 rigor plan.
#
# Guarantees the instance is never idle while nobody is watching, and that a
# single crash does not waste a night:
#
#   * Phase 1 (class-awareness + oracle ladder, ~30 h) is retried on failure.
#     Every (N, arm, seed) job is checkpointed AND the resume gate now compares
#     the stored config (444d99a4), so a resume cannot silently reuse a
#     checkpoint from a different split — the 2026-08-01 contamination bug.
#   * When Phase 1 finishes, the cheap blocks run in ascending cost order:
#     Phase 3 Tier-1 weight sensitivity (~2 h), then Phase 4 MCLP (~5 h).
#   * Only after all of it does the VM stop, to conserve credits.
#
# Retries are capped: a deterministic failure should surface as an idle VM with
# a log, not an infinite relaunch loop burning the remaining credits.
#
# Usage:  STOP_VM=1 setsid nohup ./scripts/run_overnight.sh > /tmp/overnight.log 2>&1 < /dev/null &
set -uo pipefail          # NOT -e: a failing phase must be caught, not abort the supervisor
cd "$(dirname "$0")/.."

STOP_VM="${STOP_VM:-0}"
MAX_RETRIES="${MAX_RETRIES:-5}"
LOG=results/overnight.log
mkdir -p results
exec > >(tee -a "$LOG") 2>&1

say() { echo "[$(date -Is)] $*"; }

if [[ -z "${HF_TOKEN:-}" ]]; then
    say "HF_TOKEN missing — the real-data pipeline cannot run"; exit 1
fi

# Phase 1 owns no shutdown here; the supervisor decides when the VM stops.
run_phase1() {
    local attempt=1
    while (( attempt <= MAX_RETRIES )); do
        say "Phase 1 attempt $attempt/$MAX_RETRIES"
        if STOP_VM=0 ./scripts/run_phase1.sh; then
            say "Phase 1 completed"
            return 0
        fi
        say "Phase 1 attempt $attempt failed — sweeping orphaned workers before retry"
        # Killing a joblib run leaves loky workers reparented to init, each
        # pinning a core (14 of them survived the 2026-08-01 kill and starved
        # the relaunch for 4.5 h). Sweep anything older than an hour.
        ps -eo pid,etimes,args \
            | awk '/client-selection-hfl\/\.venv\/bin\/python/ && $2 > 3600 {print $1}' \
            | xargs -r sudo kill -9 || true
        sleep 60
        (( attempt++ ))
    done
    say "Phase 1 exhausted $MAX_RETRIES attempts — stopping here, VM stays up"
    return 1
}

if [[ -f results/phase1.done ]]; then
    say "Phase 1 already complete (marker present) — skipping"
else
    run_phase1 || { say "ABORTING supervisor: Phase 1 unrecoverable"; exit 1; }
fi

say "Phase 3: Tier-1 fitness-weight sensitivity (~2 h)"
./scripts/run_tier1_weights.sh || say "Phase 3 FAILED (continuing — it is independent)"

say "Phase 4: MCLP near-optimality, 30 seeds x grid {20,30,45} (~5 h)"
python -m uavbench mclp --config configs/tier1_core.yaml \
    --n-seeds 30 --grid-res 20 30 45 --grid-seeds 10 --time-limit 600 \
    || say "Phase 4 FAILED (continuing)"

say "Committing all results"
git add -A -- results || true
if git diff --cached --quiet; then
    say "no new results to commit"
else
    git commit -m "Add VM1 overnight results $(date -Is)" || true
    say "committed $(git rev-parse --short HEAD) — retrieve via git bundle (VM cannot push)"
fi

say "===== overnight batch complete ====="
if [[ "$STOP_VM" == "1" ]]; then
    say "stopping instance to conserve credits"
    sudo shutdown -h now || true
fi
