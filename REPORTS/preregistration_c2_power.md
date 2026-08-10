# Pre-registration — C2 at higher power

Written **2026-08-11, before any of these seeds have been computed and before any
new C2 number has been seen.** Fixes the design, the analysis and the criteria in
advance, so that a win is evidence rather than the product of running until
something crossed 0.05.

---

## 1. Why this run exists

`scripts/power_analysis.py` shows the v6 evaluation was underpowered for the
effects it was measuring. **Median minimum detectable effect at n=10 is 0.0340
macro-F1** (alpha 0.05, power 0.80), and most of the differences in play are
smaller than that.

C2 — diversity-weighted edge aggregation — is the single arm where that matters
for a *positive* claim:

| K | C2 − mclp_place | sd | p | MDE@80% | observed/MDE |
|---|---|---|---|---|---|
| 5 | +0.0015 | 0.0107 | 1.000 | 0.0106 | 0.14 |
| 10 | **+0.0330** | 0.0408 | 0.037 | 0.0406 | 0.81 |
| 15 | +0.0217 | 0.0315 | 0.084 | 0.0314 | 0.69 |
| 20 | +0.0117 | 0.0209 | 0.106 | 0.0208 | 0.56 |
| 30 | +0.0160 | 0.0327 | 0.160 | 0.0325 | 0.49 |

Positive at all five fleet sizes, Holm-significant at none, and sitting just
under the detection threshold at K=10. That is the signature of an
under-powered comparison rather than of no effect — but it is equally the
signature of noise, which is why this is a pre-registered test and not a
narrative.

## 2. What is honest about this and what is not

**Disclosed selection.** C2 was chosen for extension *after* seeing the v6
results. Nothing else about the design is post-hoc, but that choice is, and it
must be stated wherever this run is reported.

**Fixed stopping rule.** n = 25 seeds, decided here. No optional stopping, no
peeking, no extension if the result lands just short. If 25 seeds do not
resolve it, that is the answer.

**This does not reopen the v6 verdict.** v6's criteria were about beating
`moon2022`, and C2 failed them; `REPORTS/preregistration_v6_method.md` stands as
written. This run asks a narrower question — does C2 improve on the proposed
placement, `mclp_place` — and a win here must not be reported as v6 passing.

## 3. Design

Arms: `mclp_place` under C2 (`fl.uav_weight_mode: diversity`) and `mclp_place`
under the v5 default (`samples`), **both computed in the same run** at the same
25 seeds, so the pairing is exact. Capacity 6, R_comm 5 km, 100 rounds.

Two grids, because power is not the only thing wrong with the existing evidence:

**(a) Fleet replication — K ∈ {5, 10, 15, 20, 30} at N=200.** Identical to the
fleet sweep, so it tests whether the observed K-pattern reproduces at power.
250 jobs.

**(b) Client-count generalisation — N ∈ {50, 100} at K ∈ {10, 20}.** 100 jobs.

Grid (b) exists because **every C2 number to date sits at a single client
count**, N=200, and that is not an incidental gap. At K=20/cap=6 there are 120
slots, so **N=200 binds and N=50/100 are slack** — and C2 acts on shard
composition, which is exactly what the slack/binding regime changes. Testing
only at N=200 would test the one regime most favourable to a shard-composition
fix, and would leave "validated at one client count" as an open reviewer
objection against what may be the paper's only positive result. K=10 and K=20
are chosen as the two most informative fleet sizes — C2's largest observed
effect, and the operating point — not because they scored best.

The slack regime is also where the roster-construction confound is active
(`configs/roster_control.yaml`), so grid (b) must be read alongside that
control's outcome.

* Fresh results directory. The existing 10-seed results are **not** reused and
  checkpoints are **not** copied between directories: `run_uav_sweep` rebuilds
  its summary parquet from only the jobs in the current run, so pointing a
  reduced method list at an existing directory would overwrite a 17-method
  summary with a 1-method one, and copying job checkpoints across directories is
  the stale-reuse failure this project has now hit twice in one day.
* Seeds 0-9 of grid (a) reproduce the discovery sample exactly
  (`sweep_job_seed` is deterministic in `seed_idx`); seeds 10-24 are new.
* ~350 jobs total, ≈6.5 h on 12 vCPU.
* Fresh results directory. The existing 10-seed results are **not** reused and
  checkpoints are **not** copied between directories: `run_uav_sweep` rebuilds
  its summary parquet from only the jobs in the current run, so pointing a
  reduced method list at an existing directory would overwrite a 17-method
  summary with a 1-method one, and copying job checkpoints across directories is
  the stale-reuse failure this project has now hit twice in one day.
* Seeds 0-9 reproduce the discovery sample exactly (`sweep_job_seed` is
  deterministic in `seed_idx`); seeds 10-24 are new.
* ~250 jobs, ≈5 h on 12 vCPU.

## 4. Analysis plan — fixed now

Primary endpoint: **macro-F1, mean of the last 10 rounds per seed**, paired
Wilcoxon against `mclp_place`, **Holm-corrected within each fleet size**. Every
cell must clear `scripts/gate_collapse.py`.

**C2 is declared a real effect only if ALL of:**

1. **Holm-significant wins over `mclp_place` at ≥ 2 of the 5 fleet sizes** in
   grid (a), and
2. **positive sign at all 5** in grid (a) — i.e. the pattern that motivated this
   run replicates, rather than one cell surviving on its own, and
3. **positive sign in at least 3 of the 4 cells of grid (b)** — the effect is
   not confined to the single client count it was discovered at.

Criterion 3 is deliberately weaker than 1 and 2: grid (b) has four cells and is
there to catch an effect that exists *only* where slots bind, not to demand
significance at every client count. If C2 passes 1 and 2 but fails 3, the honest
report is "improves the proposed placement at N=200; does not generalise across
client count on the evidence available" — and the paper says exactly that.

Reported alongside, and **not** decisive either way:

* the independent subset, seeds 10-24 alone (n=15). It is underpowered on its
  own (MDE ≈ 0.040 under Holm) and is reported for transparency, not as the
  test. Do not quote it as confirmation or as refutation.
* the per-K effect sizes and bootstrap CIs, which are what a reader needs
  regardless of the verdict.

## 5. What is NOT allowed

* **No extending past 25 seeds** because the result landed close.
* **No switching the primary endpoint**, and no moving to accuracy.
* **No re-tuning** `lr`, the fitness weights, or any shared constant.
* **No dropping a fleet size or a client count.** All five K and all four
  grid-(b) cells are reported, whatever they show.
* **No promoting a single K or N** to the headline because it won.
* Any tuning reads `val_*` only.

## 6. What a win would and would not mean

A win establishes that diversity-weighted edge aggregation improves on the
proposed placement's own numbers. It would be a genuine positive contribution
and the only one currently available.

It would **not**: beat `moon2022` (C2 was negative there at K=15/20/30), rescue
any selection claim (`paper_full` has the proposed selector Holm-losing to FedCS
and Oort), or change the coverage-causality result. The paper's spine stays the
negative and methodological findings either way; C2 would be a component result
inside it.

**Failure is a reportable outcome** and gets written up as one.
