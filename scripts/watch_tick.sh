#!/usr/bin/env bash
# One poll of every live log. Prints only what is NEW since the previous call.
#
# The read position lives here, on the VM, in $STATE — not in the poller. That
# is the whole point: a laptop watching this over SSH will lose connectivity,
# and if it tracked offsets itself a dropped poll would either replay old lines
# or skip new ones depending on how the reconnect went. Keeping the cursor on
# the machine that owns the logs makes a missed poll cost exactly nothing — the
# next successful call returns everything since the last successful call, however
# long ago that was.
#
# Idempotent per line: a line is emitted once and only once.
#
# Usage:  /home/dan/watch_tick.sh
set -uo pipefail

STATE=/home/dan/.watch_offsets
REPO=/home/dan/client-selection-hfl
touch "$STATE"

emit() {                          # emit <file> <tag> <grep-ERE>
    local f="$1" tag="$2" pat="$3"
    [[ -f "$f" ]] || return 0
    local key off total
    key=$(printf '%s' "$f" | md5sum | cut -c1-10)
    off=$(awk -v k="$key" '$1==k {print $2}' "$STATE" | tail -1)
    off=${off:-0}
    total=$(wc -l < "$f")
    # Truncated or rotated file: start over rather than skip everything.
    (( total < off )) && off=0
    if (( total > off )); then
        awk -v s="$off" 'NR>s' "$f" | grep -aE "$pat" | sed "s|^|[$tag] |"
    fi
    { awk -v k="$key" '$1!=k' "$STATE"; echo "$key $total"; } > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
}

emit "$REPO/results/rebuild_v5.log" rebuild \
     '\[gate\] (all|DEGENERATE)|COLLAPSED|!! |^\[2.*--- |===== |Traceback|BLOCKS FAILED|rebuild complete|NOTE: a non-first|MemoryError|Killed'
emit /home/dan/night_chain.log chain \
     '^\[2[0-9]'
emit "$REPO/results/v6.log" v6 \
     '\[gate\] (all|DEGENERATE)|\[v6-control\]|VERDICT|^\[2.*--- |===== |Traceback|FAILED|\[1\]|\[2\]|\[3\]|\[4\]'
emit "$REPO/results/tier1_v5.log" tier1 \
     '\[altitude\]|^\[2.*--- |===== |Traceback|FAILED|DEGENERATE'

# Liveness. Reported every tick so silence in the poller means "no connection",
# never "nothing happening" — the two are otherwise indistinguishable.
alive=""
for p in run_rebuild_v5 night_chain run_v6 run_tier1_v5 uavbench; do
    pgrep -f "$p" >/dev/null 2>&1 && alive="$alive $p"
done
if [[ -z "$alive" ]]; then
    echo "__IDLE"
else
    echo "__ALIVE:$alive"
fi
