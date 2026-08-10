#!/usr/bin/env bash
# Tier-1 placement benchmark, rebuilt under the air-to-ground channel.
#
# Why every Tier-1 result is being recomputed
# -------------------------------------------
# Tier-1 is the paper's headline placement table and until 2026-08-09 it scored
# coverage through a flat `slant_distance <= R_comm` sphere. Under that gate
# altitude is a pure penalty — it only ever adds slant distance — so every
# optimizer drove z to the floor and the advertised 3D search had a closed-form
# third dimension. The band made it worse: 20-120 m against R_comm = 500 m is a
# coherent reach of 54-324 m, i.e. a radius no altitude in the band could earn.
#
# Now: band 100-400 m, R_comm 500 m (coherent interval 270-1080 m, so the
# optimum ~185 m is interior), and `problem.link_model: path_loss`, which
# derives each UAV's radius from its own altitude through the shared
# Al-Hourani channel. Guarded by tests/sanity_checks/check_tier1_link.py and
# check_altitude_band.py.
#
# Side effect worth stating in the paper: the literature baselines no longer
# carry a private coverage radius. Every method is gated by the same function of
# the altitude it chose, which is the structural fix for the unequal-radius
# comparison recorded in REPORTS/results_provenance.md.
#
# SEPARATE from scripts/run_rebuild_v5.sh on purpose. bash reads a script
# lazily from disk, so appending to a running one can corrupt its execution
# mid-flight; the rebuild is a multi-day job. Run this only once the rebuild has
# finished, or on a machine that is not running it.
#
# Usage:  nohup ./scripts/run_tier1_v5.sh &
set -uo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

LOG=results/tier1_v5.log
mkdir -p results
exec > >(tee -a "$LOG") 2>&1
say() { echo; echo "[$(date -Is)] $*"; }

# tier1_core carries the real-coordinate scenario, which needs the cached Noto
# metadata; the others inherit from it.
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.hf_env" ]]; then
    HF_TOKEN=$(grep -o 'hf_[A-Za-z0-9]*' "$HOME/.hf_env" | head -1)
    export HF_TOKEN
fi

FAILED=""

say "===== Tier-1 v5: 100-400 m band, path-loss link, R_comm 500 m ====="

# check_tier1_resume is not optional here. The first attempt at this rebuild
# (2026-08-10) resumed all 840 jobs from results computed under the old band and
# the old flat range gate, and finished in eleven seconds. Checkpoints are now
# keyed by a config signature; this asserts the key actually moves when the
# physics does.
for chk in check_tier1_resume check_altitude_band check_tier1_link \
           check_placement_baselines check_placement_methods check_mclp; do
    python "tests/sanity_checks/${chk}.py" || { say "${chk} FAILED — aborting"; exit 1; }
done

run_cfg() {                       # run_cfg <config-stem>
    local stem="$1"
    local cfg="configs/${stem}.yaml"
    local rdir
    rdir=$(python -c "import sys;sys.path.insert(0,'src');from uavbench.runner import load_config;print(load_config('${cfg}')['results_dir'])")

    say "--- ${stem} -> ${rdir} ---"
    if python -m uavbench run --config "$cfg" && python -m uavbench analyze --config "$cfg"; then
        # A finished run is not a correct run. If the link failed to reach the
        # scorer, every method pins to a bound and the tables look normal.
        if ! python scripts/gate_altitude.py "$rdir" --config "$cfg"; then
            say "  !! ${stem} has a DEGENERATE vertical search — not a usable result"
            FAILED="$FAILED ${stem}(degenerate)"
        fi
    else
        say "  !! ${stem} FAILED (exit $?)"
        FAILED="$FAILED ${stem}"
    fi

    git add -A -- results || true
    if ! git diff --cached --quiet; then
        git -c user.email=vm@local -c user.name=vm commit -q -m "Add ${stem} v5 results $(date -Is)" || true
        say "  committed $(git rev-parse --short HEAD)"
    fi
}

# Headline first: if the channel is not reaching the scorer, it shows here and
# there is no point spending the ablations on it.
run_cfg tier1_core
if [[ "$FAILED" == *"(degenerate)"* ]]; then
    say "HEADLINE Tier-1 RUN IS DEGENERATE — aborting. Every ablation inherits"
    say "tier1_core's band and link, so all three would fail the same way."
    exit 1
fi

for stem in tier1_equal_radius tier1_regime_hetero tier1_warmprev; do
    run_cfg "$stem"
done

say "===== Tier-1 v5 complete ====="
if [[ -n "$FAILED" ]]; then
    say "FAILED:$FAILED"
    exit 1
fi
say "all Tier-1 configs succeeded"
