#!/usr/bin/env bash
# Bring the VM's results commits home and onto GitHub. Run from the LAPTOP.
#
# Why this is not just "git pull"
# ------------------------------
# VM2 was set up with a plain clone of a public repo, so it has no GitHub
# credential and cannot push. Its results therefore accumulate as local commits
# on one disk — which is exactly how the VM1 data ended up stranded. Every block
# that finishes should be brought home the same day.
#
# Two failure modes this encodes, both hit for real on 2026-08-10:
#
#  1. `git bundle create <file> origin/main..HEAD` writes the tip under the ref
#     name HEAD, not `main`, so the fetch refspec must say `HEAD:`. Fetching
#     `main:` fails with "couldn't find remote ref main".
#
#  2. Pushing the whole range in one go sends a single pack of tens of MB of
#     binary parquet and dies on an HTTP 408 before the server finishes reading.
#     Chunking straight onto `main` does NOT help: the VM's results commits were
#     made before it pulled the laptop's work, so none of them has the current
#     remote tip as an ancestor and every chunk is rejected non-fast-forward.
#     Only the final merge commit fast-forwards. So the objects go up on a
#     throwaway branch, where each successive commit IS a fast-forward, in small
#     packs; `push origin main` afterwards has nothing left to send.
#
# Safe to re-run: staging is idempotent and a partial transfer resumes.
#
# Usage:  ./scripts/pull_vm_results.sh
set -uo pipefail

VM_USER=dan
VM_HOST=instance-20260715-133652
VM_ZONE=europe-west1-b
VM_CFG=vm2
VM_REPO=/home/dan/client-selection-hfl
REPO=/home/ody/Projects/client-selection-hfl
WORK=$(mktemp -d)
STAGE=results-staging
CHUNK=3           # commits per push; smaller if the connection is worse

export PATH="$HOME/google-cloud-sdk/bin:$PATH"
vm() { gcloud --configuration="$VM_CFG" compute ssh "$VM_USER@$VM_HOST" --zone="$VM_ZONE" --command="$1"; }

cd "$REPO" || exit 1
git fetch origin -q

echo "== VM state =="
vm "cd $VM_REPO && git log --oneline -1 && echo ahead=\$(git rev-list --count origin/main..HEAD)"

echo "== bundling on the VM =="
vm "cd $VM_REPO && git bundle create /tmp/vm_results.bundle origin/main..HEAD >/dev/null 2>&1; ls -lh /tmp/vm_results.bundle"

echo "== fetching bundle =="
gcloud --configuration="$VM_CFG" compute scp \
    "$VM_USER@$VM_HOST:/tmp/vm_results.bundle" "$WORK/vm.bundle" --zone="$VM_ZONE" || exit 1

# NOTE: HEAD:, not main: — see (1) above.
git fetch "$WORK/vm.bundle" HEAD:refs/vmresults || exit 1
echo "fetched $(git rev-list --count HEAD..refs/vmresults) commits"

echo "== scanning for secrets before anything reaches a PUBLIC repo =="
hits=$(git diff HEAD..refs/vmresults --text -U0 2>/dev/null | grep -aoE \
    '(hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)' \
    | sort -u)
if [[ -n "$hits" ]]; then
    echo "!! SECRET-SHAPED STRINGS IN THE INCOMING COMMITS — refusing to push:"
    echo "$hits" | sed 's/^/    /'
    echo "   (commits are fetched as refs/vmresults; inspect, fix, then push by hand)"
    exit 1
fi
echo "clean"

git merge --ff-only refs/vmresults || { echo "not a fast-forward — resolve by hand"; exit 1; }
git update-ref -d refs/vmresults

git config http.postBuffer 524288000
git config http.lowSpeedLimit 1000
git config http.lowSpeedTime 600

mapfile -t COMMITS < <(git rev-list --reverse origin/main..HEAD)
echo "== staging ${#COMMITS[@]} commits in packs of $CHUNK =="
i=0
for c in "${COMMITS[@]}"; do
    i=$((i + 1))
    (( i % CHUNK != 0 )) && (( i != ${#COMMITS[@]} )) && continue
    for attempt in 1 2 3 4; do
        echo "[$(date +%T)] stage $i/${#COMMITS[@]} ${c:0:9} (try $attempt)"
        git push origin "$c:refs/heads/$STAGE" >/dev/null 2>&1 && break
        sleep 20
    done
done

echo "== fast-forwarding main =="
for attempt in 1 2 3; do
    git push origin main && break
    sleep 20
done
git push origin --delete "$STAGE" >/dev/null 2>&1

rm -rf "$WORK"
git fetch origin -q
echo "DONE origin/main=$(git rev-parse --short origin/main) local=$(git rev-parse --short HEAD) unpushed=$(git rev-list --count origin/main..HEAD)"
