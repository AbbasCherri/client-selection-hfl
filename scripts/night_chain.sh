#!/usr/bin/env bash
# Overnight chain: wait out the v5 rebuild, then run what is queued behind it.
#
# Why this exists
# ---------------
# Every long job here is already detached and checkpointed, so a dropped SSH
# session or a laptop going to sleep cannot hurt a run. What it CAN do is waste
# the night: the rebuild finishes at 04:00 and the VM then sits idle for eight
# hours because the next command lives on a laptop that is offline.
#
# This runs entirely on the VM. Once launched it needs no network at all.
#
# Order and why
#   1. wait for run_rebuild_v5.sh (blocks 3-4) to exit
#   2. git pull — deliberately HERE, not earlier. The v6 code (C1/C2) defaults
#      to v5 behaviour, but pulling while a block's loky workers are respawning
#      means some jobs in one experiment import different code from others, and
#      "the defaults are no-ops" is a claim resting on my own guards. Between
#      experiments it costs nothing to be sure.
#   3. sanity checks — abort the chain if any fail; a broken tree must not be
#      spent on five hours of compute
#   4. v6 method evaluation (run_v6.sh — control first, self-aborting)
#   5. Tier-1 rebuild under the link model (run_tier1_v5.sh)
#
# NEVER shuts the VM down. Results live on this disk and nothing here is worth
# risking an unattended power-off for.
#
# Usage (from the laptop, detached so it outlives the SSH session):
#   setsid nohup /home/dan/night_chain.sh > /home/dan/night_chain.log 2>&1 &
set -uo pipefail

REPO=/home/dan/client-selection-hfl
LOCK=/tmp/night_chain.lock
LOG=/home/dan/night_chain.log

# Single instance. Re-running this after a dropped connection is the obvious
# thing to try, and two chains racing on the same results dir would interleave
# git commits and confuse every checkpoint.
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "[$(date -Is)] another night_chain is already running (lock: $LOCK) — exiting"
    exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

say() { echo; echo "[$(date -Is)] $*"; }
cd "$REPO" || { say "no repo at $REPO"; exit 1; }

say "===== night chain armed ====="

# ---- 1. wait for the rebuild ------------------------------------------------
if pgrep -f run_rebuild_v5 >/dev/null; then
    say "waiting for run_rebuild_v5.sh to finish…"
    while pgrep -f run_rebuild_v5 >/dev/null; do sleep 60; done
    say "rebuild process has exited"
else
    say "rebuild is not running — proceeding straight to the queue"
fi
sleep 30   # let its final commit land

# ---- 2. pull ----------------------------------------------------------------
say "pulling latest code"
git -c user.email=vm@local -c user.name=vm pull --no-rebase --no-edit -q 2>&1 | tail -3
say "now at $(git rev-parse --short HEAD)"

# ---- 3. sanity checks -------------------------------------------------------
say "running sanity checks"
source .venv/bin/activate 2>/dev/null || true
FAILED_CHK=""
for chk in check_coverage_mode check_uav_weight_mode check_roster_control \
           check_shard_diversity check_collapse_guard check_roster_width \
           check_altitude_band check_tier1_link; do
    if [[ -f "tests/sanity_checks/${chk}.py" ]]; then
        python "tests/sanity_checks/${chk}.py" >/dev/null 2>&1 \
            || FAILED_CHK="$FAILED_CHK $chk"
    fi
done
if [[ -n "$FAILED_CHK" ]]; then
    say "SANITY CHECKS FAILED:$FAILED_CHK — refusing to spend the night on a broken tree"
    exit 1
fi
say "sanity checks pass"

# ---- 4. v6 ------------------------------------------------------------------
if [[ -x scripts/run_v6.sh ]]; then
    say "--- starting v6 method evaluation ---"
    ./scripts/run_v6.sh || say "v6 exited non-zero — see results/v6.log (continuing)"
else
    say "scripts/run_v6.sh missing — skipping"
fi

# ---- 5. Tier-1 --------------------------------------------------------------
if [[ -x scripts/run_tier1_v5.sh ]]; then
    say "--- starting Tier-1 rebuild ---"
    ./scripts/run_tier1_v5.sh || say "Tier-1 exited non-zero — see results/tier1_v5.log"
else
    say "scripts/run_tier1_v5.sh missing — skipping"
fi

say "===== night chain complete — VM deliberately left running ====="
