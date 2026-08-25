#!/usr/bin/env bash
# Resumable driver for the pre-registered programme in PREREGISTRATION.md.
#
# Design intent: the PROGRAMME must advance whether or not any interactive
# session is alive. Wifi drops, token limits and closed laptops must cost
# nothing but latency. So orchestration lives here, on the machine that owns
# the compute, and the interactive side only ever READS results/programme_state.txt.
#
# Safe to re-run at any time. Idempotent in both directions:
#   * an arm whose verdict file exists is never re-run
#   * an arm that is half-done resumes from its own job checkpoints
#   * a second driver refuses to start while one holds the lock
#
# It runs arms and records numbers. It does NOT make scientific decisions:
# every criterion it applies was fixed in the arm's own config and committed
# before that arm's first seed existed. Arms needing new code or a judgement
# call are reported BLOCKED and left alone.
set -uo pipefail
cd "$(dirname "$0")/.."

LOCK=results/.programme.lock
STATE=results/programme_state.txt
LOG=results/programme_driver.log
mkdir -p results

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "another driver holds $LOCK — exiting"; exit 0
fi

exec > >(tee -a "$LOG") 2>&1
say() { echo; echo "[$(date -Is)] $*"; }

if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.hf_env" ]]; then
    HF_TOKEN=$(grep -o 'hf_[A-Za-z0-9]*' "$HOME/.hf_env" | head -1); export HF_TOKEN
fi
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN missing"; exit 1; }

# id | results-dir name | config | human label
READY_ARMS=(
    "H1|fusion_owner|configs/fusion_owner.yaml|fusion ownership"
    "H3|capacity12|configs/capacity12.yaml|capacity above tested ceiling"
)
# Arms that must not be auto-run: they need code or a judgement call.
BLOCKED_ARMS=(
    "H2|shard_diversity|NEEDS_IMPL|shard-diversity assignment builder"
    "P1|retune_5km|NEEDS_DECISION|re-tune every method at 5 km (runs LAST, under the final architecture)"
)

write_state() {
    {
        echo "UPDATED=$(date -Is)"
        echo "DRIVER_PID=$$"
        for entry in "${READY_ARMS[@]}"; do
            IFS='|' read -r id name cfg label <<<"$entry"
            local_status="QUEUED"
            if [[ -f "results/${name}_verdict.txt" ]]; then
                if grep -q "PASSES its pre-registered criteria" "results/${name}_verdict.txt" 2>/dev/null; then
                    local_status="DONE_PASS"
                else
                    local_status="DONE_FAIL"
                fi
            elif pgrep -f "config ${cfg}" >/dev/null 2>&1 || [[ "${RUNNING_ARM:-}" == "$name" ]]; then
                local_status="RUNNING"
            fi
            jobs=$(find "results/${name}" -name "fullsim_rounds.parquet" 2>/dev/null | wc -l)
            echo "${id} ${name} ${local_status} jobs=${jobs}/40 -- ${label}"
        done
        for entry in "${BLOCKED_ARMS[@]}"; do
            IFS='|' read -r id name why label <<<"$entry"
            echo "${id} ${name} ${why} -- ${label}"
        done
    } > "$STATE"
}

commit_results() {
    git add -A -- results PREREGISTRATION.md 2>/dev/null || true
    git diff --cached --quiet 2>/dev/null || \
        git -c user.email=vm@local -c user.name=vm commit -q -m "$1" || true
}

# If an arm is already running under its own launcher, wait it out rather than
# starting a competing copy. 12 vCPU is fully subscribed by one sweep.
wait_for_running_sweep() {
    local waited=0
    while pgrep -f "run_paper_sim" >/dev/null 2>&1; do
        [[ $waited -eq 0 ]] && say "a sweep is already running — waiting for it"
        write_state; sleep 120; waited=$((waited + 120))
    done
    [[ $waited -gt 0 ]] && say "previous sweep finished after ${waited}s"
    return 0
}

say "===== programme driver up ====="
write_state
wait_for_running_sweep

for entry in "${READY_ARMS[@]}"; do
    IFS='|' read -r id name cfg label <<<"$entry"

    if [[ -f "results/${name}_verdict.txt" ]]; then
        say "$id ($name) already has a verdict — skipping"
        continue
    fi

    say "$id ($name) — $label"
    export RUNNING_ARM="$name"; write_state

    if python -m uavbench run_paper_sim --config "$cfg"; then
        python scripts/gate_collapse.py "results/${name}" \
            || say "  !! degenerate cells in $name — see gate output"
        python scripts/score_arm.py "$name" | tee "results/${name}_verdict.txt"
    else
        say "  !! $id ($name) FAILED — leaving no verdict so it retries next pass"
    fi

    unset RUNNING_ARM
    write_state
    commit_results "Add ${name} results $(date -Is)"
done

write_state
say "===== ready queue exhausted ====="
say "Blocked arms remain — see $STATE. Per PREREGISTRATION.md §5 the programme"
say "does NOT stop on a win and does NOT continue hunting after the queue is"
say "empty: STOP-EXHAUSTED is a pre-committed outcome."
commit_results "Update programme state $(date -Is)"
