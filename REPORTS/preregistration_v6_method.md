# Pre-registration — v6 method redesign

Written **2026-08-09, before any v6 code exists and before any v6 result is
seen.** Its purpose is to fix the success criteria in advance so that a win is
evidence rather than the product of searching until something won. If the
criteria below are not met, that is the result, and it gets reported.

The v5 rebuild established that the current method loses. This document says
what would count as fixing that.

---

## 1. What v5 measured (the problem being solved)

Fleet sweep, `results/paper_uav_count`, N=200, capacity 6, R_comm 5 km,
100 rounds, 10 seeds, macro-F1 as mean of the last 10 rounds:

| method | K=5 | K=10 | K=15 | K=20 | K=30 |
|---|---|---|---|---|---|
| `flat_fl` (no UAVs) | 0.3929 | 0.3929 | 0.3929 | 0.3929 | 0.3929 |
| `moon2022` (literature) | 0.3859 | **0.4122** | **0.4181** | **0.4089** | **0.4061** |
| `mclp_place` (proposed) | 0.3472 | 0.3910 | 0.3850 | 0.3739 | 0.3667 |
| `proposed_hfl` (PSO) | 0.3859 | 0.3854 | 0.3686 | 0.3552 | 0.3438 |

Two failures: the proposed placement loses to a 2022 baseline at every fleet
size, and the whole UAV system loses to flat FL.

## 2. Diagnosis (measured, not assumed)

`src/uavbench/problem/assignment.py`:
`f_cover = instance.value[assigned_mask].sum()` — the placement objective sums
value over **assigned** devices, and assignment is capped at `capacity` per UAV.
So the objective saturates at `B = K·capacity` and further coverage earns
nothing.

Confirmed by what the optimizers actually do (`scratchpad/why_undercover.py`,
single instance, seed 0):

| method | covered, K=10 (B=60) | covered, K=20 (B=120) |
|---|---|---|
| `mclp_ls` | 64 | 134 |
| `pso` | 74 | 135 |
| `moon2022` | 138 | 187 |

Both fitness-optimizing methods stop within a few clients of the slot budget.
`moon2022` does not optimize this objective and covers far more.

Altitude is **not** the mechanism: `moon2022`, `pso_cluster`, `mogoa` and
`spiral` all fly at exactly z\* = 1851.6 m with identical 5332 m radius, yet
cover 69% / 57.5% / 36% / 41%. The difference is horizontal.

Why covered-but-unassigned clients still matter: selection re-runs every round
(`reselect_every: 1`) and the roster rotates, so across 100 rounds the reachable
population — not the per-round assignment — bounds the data the system can train
on. Placement is optimizing a one-round quantity for a many-round objective.

## 3. The two changes

**C1 — reachability-maximizing placement.** `f_cover` becomes the value summed
over *covered* devices rather than assigned ones. Reduces exactly to the current
objective when `|covered| <= B`, so it is a strict generalization, not a
different method in disguise.

**C2 — diversity-gated edge aggregation.** A UAV whose pooled shard is
near-single-class contributes its fusion-block update with reduced weight (or
not at all). Motivated by `results/probe_topology`: shards of <= 3 clients make
the run *unlearn*, and the mechanism is single-class fusion heads entering the
average. Gates on measured shard diversity rather than on a capacity threshold,
because the threshold was one measurement at one N and one radius.

Each is evaluated **separately and together** (2x2). A combined-only result
would not show which part works.

## 4. Success criteria — fixed now

Primary endpoint: **macro-F1**, mean of the last 10 rounds per seed, 10 seeds,
paired Wilcoxon vs the named comparator, Holm-corrected within each fleet size.
Every arm must first clear `scripts/gate_collapse.py`.

The method is declared an improvement only if **all** of:

1. **It beats `moon2022`** — the strongest v5 placement baseline — at K=10, 15
   and 20, Holm-significant at at least two of the three.
2. **It beats the v5 proposed method** (`mclp_place`) at every K in 5-30.
3. **It does not lose to `flat_fl`** at the operating point (K=20): either
   significantly above, or within a difference whose 95% CI contains zero.
4. **No cell it wins in is degenerate** — the collapse gate passes for every
   (method, K) cell reported.

Criterion 3 is deliberately weak. `flat_fl` presumes surviving terrestrial
infrastructure, which the disaster premise excludes, so it is an infeasible
reference rather than a competitor; parity with it is a strong result for a
system that assumes no ground network. Claiming to *beat* it would be the
dishonest framing.

**Failure is a reportable outcome.** If C1+C2 do not meet these criteria, the
paper reports the negative result and reframes around the honest finding —
that a capacity-capped placement objective is mis-specified for multi-round FL
— which stands on its own regardless of whether the fix wins.

## 5. What is NOT allowed

* **No re-tuning of `w1/w2/w3`, `lr`, or any shared training constant to make
  v6 win.** The existing `configs/tuned_weights.yaml` recipe applies unchanged
  to every arm. If re-tuning is judged necessary it must be run for *every*
  method at equal budget, and disclosed — see
  `REPORTS/rigor_plan_2026-08.md` on tuning asymmetry.
* **No selecting the operating point after seeing v6 results.** K=20 / cap=6 is
  fixed by `results/probe_topology`, which predates this document.
* **No reporting a subset of K.** All five fleet sizes are reported.
* **No switching the primary endpoint.** macro-F1, decided here.
* **Any tuning reads `val_*` columns only.** The test columns are for the final
  reported numbers.

## 6. Confound to control

`roster-construction-confound`: the proposed selector fills UAVs to capacity
while the literature selectors load-balance, which changes shard width whenever
slots are slack (N=30/50/100 at the operating point; **not** N=200, where slots
bind). Since the fleet sweep and this evaluation run at N=200, the confound is
inactive here. It must be handled separately for the `paper_full` N-sweep, and
C2 interacts with it directly — both concern shard composition — so the
`paper_full` comparison needs the isolating arm before it can be read.

## 7. Compute plan

Block 4 of the v5 rebuild is **not** wasted by this work: `paper_full`
checkpoints per `(N, seed, method)`, so adding v6 as new methods computes only
the new arms and reuses every existing one.
