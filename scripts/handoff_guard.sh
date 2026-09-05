#!/usr/bin/env bash
# Handoff guard for the two-stream retune.
#
# Stream A's method list still ends with fair_mab and power_of_choice, which
# stream B is already tuning. If stream A ever reaches them, those two methods
# would receive 50 trials against everyone else's 25 — an asymmetric search
# budget, which is precisely the confound this whole retune exists to remove.
#
# So: the instant stream A logs "TUNE fair_mab", kill stream A and its python
# child. By then it has finished fedcs, oort, proposed_hfl, hfl_no_selection and
# rep_cap, which is exactly its share.
#
# Polls every 20 s. Exits on its own once stream A is gone.
set -uo pipefail
cd "$HOME/client-selection-hfl"
exec >> results/handoff_guard.log 2>&1
say() { echo "[$(date -Is)] $*"; }

say "guard armed — will kill stream A when it reaches fair_mab"
while true; do
  if ! pgrep -f "[r]etune60.sh" >/dev/null 2>&1; then
    say "stream A exited on its own; guard standing down"
    exit 0
  fi
  if grep -q "TUNE fair_mab" results/retune60.log 2>/dev/null; then
    say "stream A reached fair_mab — killing it now"
    pkill -f "[r]etune60.sh"
    sleep 2
    # its python child is the tuner for the method it just started
    pkill -f "tune_weights.py --method fair_mab .*weights_retune60.db"
    sleep 3
    if pgrep -f "[r]etune60.sh" >/dev/null 2>&1; then
      say "  !! stream A still alive after pkill — escalating"
      pkill -9 -f "[r]etune60.sh"
    fi
    say "stream A stopped; stream B owns fair_mab and power_of_choice"
    # remove any trials stream A managed to write before dying
    python - <<'PY' 2>&1 || true
import sqlite3, pathlib
db = pathlib.Path("results/hpo_5km/weights_retune60.db")
if db.exists():
    c = sqlite3.connect(db)
    r = c.execute(
        "select study_id from studies where study_name='cf60_fair_mab'").fetchall()
    if r:
        print(f"  stream A had created study cf60_fair_mab (id {r[0][0]}) "
              f"in its own db; stream B uses weights_retune60b.db so there is "
              f"no cross-contamination. Leaving it in place as a record.")
    else:
        print("  stream A wrote no cf60_fair_mab study — clean handoff.")
    c.close()
PY
    exit 0
  fi
  sleep 20
done
