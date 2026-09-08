# Pre-registered programme: does the proposed method win at a coherent radius?

Written 2026-08-25, **before** any arm below had produced a seed. Committed
before its first run. Arms are appended only under the rule in §6.

## 1. The commitment that makes this worth reading

This programme runs unattended and works a queue. Its stopping rule is **not**
"stop when the method wins". A queue that stops on a win is optional stopping:
it produces a positive result eventually with probability 1, and a reviewer is
right to discard it.

**"No arm passes" is a pre-committed, acceptable, publishable outcome** (§5).
That possibility staying live is the only thing that makes a pass meaningful.

This project already has the demonstration. C2's discovery cell K=10 read
+0.0330, p=0.037 at n=10 — a reportable-looking win. On 15 fresh seeds alone it
was **-0.0045**. Without the pre-registration the paper would have carried it.

## 2. Standard of evidence — three gates, all required

An arm is a WIN only if it clears all three:

1. **Its own criteria**, fixed in its config before its first seed exists,
   evaluated on the discovery seeds (0-9).
2. **Replication** on fresh seeds (10-24, n=15), never used for discovery,
   with the same sign and Holm significance.
3. **Holm survival** within the final reported comparison family, against every
   literature baseline at every N — not against a subset chosen afterwards.

Gate 2 is deliberately replication rather than an alpha correction. With 4
hypotheses at alpha 0.05 the uncorrected family-wise error is ~18.5%;
replication on unseen seeds is both stronger and the thing that actually caught
C2. No arm is promoted on gate 1 alone, and **no cell may be selected after the
fact** — the winning N or K cannot be picked from the grid post hoc.

## 3. Fairness prerequisites — not hypotheses, and they consume no alpha

These are not tests. They are conditions without which no comparison is
reportable in either direction.

**P1 — re-tune at the operating radius.** Every method's recipe in
`configs/tuned_weights.yaml` was searched at **R_comm = 20 km**
(`results/hpo/`, 2026-08-05) and is currently applied at **5 km**. Every arm in
the v5 table therefore runs hyperparameters fit for a different physical
regime. It is symmetric, so it does not explain the direction of any result,
but it must be closed before the table ships.

**P1 must be run for EVERY method, not only the proposed one.** Tuning one arm
and imposing its recipe on the others is precisely the defect that produced the
inflated v4 result. Repeating it here in the other direction would be the same
error wearing a different hat.

**P2 — any architecture change forces a full-table rerun.** If an arm below
changes the proposed system's architecture, the comparative table must be
regenerated for *every* baseline under that architecture, then re-tuned under
P1, before any "we beat X" sentence is written.

## 4. The queue

Ordered by mechanism strength, not by how likely each is to flatter the method.
Every arm attacks a *measured* mediator.

### H1 — fusion ownership — RUNNING (2026-08-25)
Config: `configs/fusion_owner.yaml` (criteria and contrasts written there).
Mechanism: the UAV tier owns `img_proj + fusion` and trains both on the pooled
shard of its `capacity`=6 assigned clients, all inside one 5 km disc. Noto
damage classes are strongly spatially clustered, so that shard is near
single-class and the fusion head, averaged across K=20 UAVs, forgets.
Evidence it is the right target: `flat_fl` never trains `img_proj` at all and
beats every hierarchical arm. Cost ~3 h.

### H2 — shard-diversity assignment
Mechanism: *which UAV a client reports to* sets shard class composition. Both
existing builders (`_class_coverage_assign` fill-to-capacity,
`_greedy_assign` load-balanced) ignore class entirely. A builder maximising
within-shard class entropy subject to reachability attacks the mediator
directly. Distinct from class-aware *selection*, which chooses who participates,
not who they report to. Mediator already recorded per round as
`shard_class_entropy` / `shard_minority_share`, so the mechanism is checkable
independently of the outcome. Requires implementation. Cost ~3 h + build.

### H3 — capacity above the tested ceiling
Mechanism: shard width. The capacity ladder in `results/probe_topology` held
slots = K x capacity = 120 fixed, so cap > 6 at K = 20 was **never measured** —
every rung traded capacity against fleet size. If width is what matters,
K=20/cap=12 (240 slots) should help and the ladder cannot say otherwise.
Cheap. Cost ~2 h.

### H4 — UAV-tier step control
Mechanism: if the UAV tier forgets, its effective step per round is too large.
Search `n_uav_epochs` and `uav_lr` jointly. **Runs only if P1's search space
excludes `n_uav_epochs`**; otherwise it is subsumed and must not be counted as
a separate test.

## 5. Stopping rules

* **STOP-WIN** — an arm clears all three gates in §2. This triggers P2 and P1,
  not a claim. A comparative sentence may be written only after the full table
  has been regenerated and re-tuned under the winning architecture.
* **STOP-EXHAUSTED** — queue empty, no arm clears the gates. **Pre-committed
  outcome:** the paper reports the crossover characterisation — hierarchical FL
  with class-aware selection pays off only when UAV reach exceeds the spatial
  correlation length of the label field; below it, geographic sharding destroys
  class diversity faster than selection can recover it. The radius sweep
  (`results/paper_coverage_v5`: -0.165 at 500 m rising monotonically to -0.019
  at 5000 m) is the quantitative backbone, and it explains every negative result
  in this programme with one mechanism. This is a publishable finding, not a
  failure.
* **STOP-BUDGET** — VM credit exhausted. Report the queue state as it stands.

## 6. Rule for adding arms

An arm may be appended after seeing another arm's results **only if** its
mechanism is derived from a *measured mediator* (a recorded column that moved),
and it must be written into this file, with its criteria, **before it runs**.

Arms may not be added because a previous arm failed and something else might
work. That is the search this file exists to prevent.

## 7. Log

| arm | launched | verdict | gates cleared |
|---|---|---|---|
| H1 fusion ownership | 2026-08-25 21:19 | pending | — |

### H1 verdict — FAILS (2026-08-25)

`results/fusion_owner_verdict.txt`. Criteria fixed in `configs/fusion_owner.yaml`
at commit `c2b463bef`, before the first seed existed. n=10, no optional stopping.

**Both criteria fail 0/4. Rule 3 fires: UAV-owned fusion is NOT the mechanism.**

C1 (client-fusion minus uav-fusion): -0.098 / -0.085 / -0.156 / -0.154 at
N=30/50/100/200, every one Holm-significant in the WRONG direction.

**The arm COLLAPSED — the collapse gate fired in 4/4 cells.** At N=200,
macro-F1 0.2245 against a constant-predictor baseline of 0.2243 (margin
+0.0001, need >= 0.05) with at least one class never predicted. So C1 and C2
are not graded effect sizes; they are comparisons against a degenerate arm and
must be reported that way. **The scorer's printed line "UAV-trained img_proj is
worse than flat_fl's frozen random projection" is NOT supportable** — it is
confounded by the collapse, and is retracted here.

What this does establish: co-locating img_proj + fusion at the UAV tier is
**load-bearing and correct as designed**. Moving fusion to the client tier does
not merely cost accuracy, it destroys learning.

Mechanism, consistent with an already-established result rather than invented
after the fact: a UAV trains fusion on the POOLED shard of its `capacity`
clients, whereas a client trains it on its own shard alone — the narrowest,
most single-class pool available. The capacity ladder in `results/probe_topology`
already showed narrow pools unlearn (cap<=3 collapses below the floor). H1 moved
fusion to the narrowest pool in the system and got the same signature.

This **strengthens** the shard-width and shard-diversity hypotheses (H3, H2)
rather than motivating a new arm. No arm is added under §6.

### H3 verdict — FAILS, and the intervention was INERT (2026-08-25)

`results/capacity12_verdict.txt`. Criteria fixed in `configs/capacity12.yaml`
at `2e9de421b`, before the first seed. All 4 cells CLEARED the collapse gate,
so unlike H1 this is a clean comparison.

Both criteria fail. C1 (cap12 - cap6): -0.017 / -0.001 / -0.013 / +0.014,
none Holm-significant, all below MDE. Gap closed: **-10% / -1% / -25%** at
N=30/50/100 — it moved the wrong way.

**The manipulation did not manipulate.** `mean_shard_clients` at capacity 6 is
already only **1.18 / 1.48 / 2.32 / 3.79** at N=30/50/100/200. Nothing was
reaching the cap of 6, so raising it to 12 changed shard width by +0.0000 at
N=30 and -0.0018 at N=50. Shard class entropy moved by <0.016 everywhere.

So H3 is **not** evidence that shard width does not matter. It is a failed
intervention on a non-binding constraint, and it must not be reported as a null
about width. The design error was mine and was avoidable: the roster-control
result already established that capacity never binds here, because participation
sits near 25% and the selector fills only a fraction of K x capacity slots.

**What it did measure, and this is the substantive finding:** with K=20 at a
5 km coherent radius, the average UAV pools **one to four clients**. At N=30 it
is 1.18. A one-client shard means the UAV tier performs no pooling at all — it
is a lossy extra aggregation layer between a client and the server. That is a
sufficient explanation for why `flat_fl`, which has no such layer, beats every
hierarchical arm, and it is independent of both fusion ownership (H1) and class
composition (H2).

The lever this identifies is **clients pooled per UAV**, which is set by K and
by participation, NOT by capacity. Whether that is already answered by the
existing fleet sweep is checked before any new arm is written (§6).

### H5 — selector separation at K=10 (added 2026-08-25 under §6)

Added under the §6 rule: derived from a MEASURED mediator that moved.
`mean_shard_clients` in results/paper_uav_count runs 4.52 -> 2.58 as K goes
5 -> 30, and proposed_hfl's macro-F1 falls monotonically with it (0.3859 ->
0.3438) while coverage rises (22% -> 73%). flat_fl is 0.3929 at every K.

Criteria are in `configs/selectors_k10.yaml`, committed before the first seed.

**Why this is an operating-point choice and not post-hoc K-shopping:** the fleet
sweep that identifies K contains NO selector baselines — its methods are
placement variants plus flat_fl and proposed_hfl. It is structurally incapable
of answering whether proposed_hfl beats fedcs or oort, which is the question H5
asks. K is selected on shard width, a mediator, in data that cannot see the
outcome being tested.

One lever: K 20 -> 10 with capacity 6 -> 12, holding K x capacity = 120 slots
exactly as in paper_full. Capacity moves only so it cannot bind (H3 measured
cap=6 as non-binding, but 60 slots at K=10 WOULD bind at N=200).

All three criteria required: beat fedcs at >=3 of 4 N, beat oort at >=3 of 4 N,
and not be significantly worse than flat_fl. Criterion 3 exists because a
selector that wins its own comparison while losing to no-hierarchy-at-all has
not earned a hierarchical framing.

If criteria 1-2 fail, STOP-EXHAUSTED applies. **No further K values may be
tried** — the sweep already spans 5 to 30, and searching operating points until
one passes is exactly what §1 forbids.

### H2 — built, and its premise is PARTLY FALSIFIED (2026-08-25)

Builder implemented and committed at `32520b8d5` as selection mode
`ucb_diversity` / method `hfl_diverse_roster`. It delegates SELECTION to the
same `_class_coverage_assign` call the proposed method uses (so the selected
client set is identical and cannot drift), then reassigns each selected client
to the feasible non-full UAV whose class histogram gains the most entropy.
Six sanity checks pass, and `check_roster_control`, `check_roster_width` and
`check_selection_rules` all still pass, so nothing existing was broken. The
change is purely additive: one registry entry plus one widened mode-validity
tuple.

**CORRECTION to §4.** The H2 entry above states that both existing builders
"ignore class entirely". **That is wrong.** `_class_coverage_assign` already
carries a submodular class-coverage objective with diminishing (sqrt) returns,
which discourages same-class stacking, and its per-slot pick order is
deliberately randomised by a large `SEL_GUMBEL_SCALE`. So the proposed builder
is ALREADY partly diversity-aware, and H2 is a marginal refinement of an
existing mechanism, not the introduction of a missing one. I wrote the §4 claim
without checking the builder, and it inflated H2's prior.

Measured consequence, reported by the implementation: on a deliberately
collapsible fixture the new builder raises mean shard class entropy from 0.9543
to 0.9699, **+0.0156** — and across ~10 other seeds/fixtures the margin
"occasionally flips sign", because it is competing with an objective that
already does much of this work.

**H2 is therefore NOT queued for a sweep yet.** H3 already cost 40 jobs and
produced a misleading null because its knob turned out to be inert, and the
signature here is the same: a small manipulation on a quantity that barely
moves. Before H2 may run, it must clear a mediator pre-check, fixed here:

> At the paper_full operating point, on >= 3 seeds, `hfl_diverse_roster` must
> raise mean `shard_class_entropy` over `proposed_hfl` by **>= +0.05**.

The threshold is derived, not chosen: H3 moved shard entropy by <= +0.016 and
produced no accuracy effect whatsoever, so a manipulation smaller than roughly
three times that cannot be expected to be informative. If the pre-check fails,
H2 is recorded as **INERT — not run**, which is an honest outcome and NOT a
null about class diversity.

The pre-check runs after H5, which currently owns all 12 vCPUs.

### H5 verdict — FAILS decisively (2026-08-26)

`results/selectors_k10_verdict.txt`. Criteria fixed in
`configs/selectors_k10.yaml` at `db67b5ba1`, before the first seed.
**All 16 method/cell combinations cleared the collapse gate**, so unlike H1 this
is a clean, valid comparison. And unlike H3 the manipulation WORKED:
`mean_shard_clients` rose from 1.18/1.48/2.32/3.79 at K=20 to
**1.34/1.84/3.07/5.57** at K=10.

All three criteria fail, 0/4, 0/4, 1/4.

proposed_hfl minus fedcs: -0.0775 / -0.0756 / -0.0524 / -0.0357
proposed_hfl minus oort:  -0.0513 / -0.0553 / -0.0726 / -0.0494

**Every one of those eight cells is a Holm-significant LOSS.** The proposed
selection rule does not separate from the literature selectors, and widening the
UAV shards — the lever the whole mechanism argument pointed at — did not change
that. It also remains worse than flat_fl at 3 of 4 N.

Per the arm's config: **no further K values may be tried.** The existing fleet
sweep already spans K=5..30 and searching operating points until one passes is
the optional stopping §1 forbids.

**Programme status: H1 FAIL, H3 FAIL (inert), H5 FAIL (clean and decisive).**
The only items outstanding are H2, which is gated behind a mediator pre-check,
and P1, the re-tune prerequisite. No arm has changed the architecture, so the
final architecture for P1 is settled as the paper_full operating point
(K=20, capacity 6, fusion_owner uav, R_comm 5 km).

### H2 — INERT, NOT RUN (2026-08-26)

`results/h2_precheck/`, 3 seeds, N=100, 20 rounds, seed-aliased so the roster
builder is the only moving part. Threshold fixed beforehand: >= +0.05 mean
`shard_class_entropy`.

  proposed_hfl        0.4303
  hfl_diverse_roster  0.4307
  **delta +0.0004**, per-seed -0.0030 / +0.0054 / -0.0012

**125x below the threshold, and the sign flips across seeds.** The builder is
correct — six sanity checks pass and it demonstrably raises entropy on a
collapsible fixture — but at the real operating point it does nothing, because
`_class_coverage_assign` already carries a submodular class-coverage objective
that does the same work.

**H2 is recorded as INERT and is NOT RUN.** This is not a null result about
class diversity: the intervention was verified not to intervene, so a sweep
could only have produced an uninformative null at a cost of 40 jobs. The gate
existed because H3 had already cost exactly that.

The `hfl_diverse_roster` code is kept (commit `32520b8d5`) as a documented
negative engineering result: an explicit entropy-maximising assignment adds
nothing over the proposed builder's existing coverage objective.

### P1 — DECIDED and launched (2026-08-26)

Unblocked: no arm changed the architecture (H1 collapsed, H3 inert, H5 clean
fail, H2 inert), so the final architecture is settled as the paper_full
operating point — K=20, capacity 6, fusion_owner uav, R_comm 5 km. P1 therefore
runs there.

`scripts/tune_weights.py` had `R_comm = 20000.0` HARDCODED in its job builder.
That hardcoding IS the defect: it is why every recipe in
`configs/tuned_weights.yaml` was fitted at 20 km and applied at 5 km. Now a CLI
option (`--r-comm`, `--k-uavs`, `--capacity`); defaults reproduce the old
behaviour exactly, so nothing already run changes meaning.

Re-tuning proposed_hfl (50 trials, full space) and fedcs, oort, flat_fl
(30 each, recipe space) at R_comm=5000. Objective `val_macro_f1` on tuning
seeds 20-22, disjoint from evaluation seeds 0-19; the test metric is never read.

**flat_fl is included deliberately.** It currently has no recipe of its own and
inherits proposed_hfl's, making it the one under-tuned arm — and it is beating
proposed_hfl anyway. Tuning it can only strengthen the negative result, which is
why fairness requires it.

**Adoption is NOT automatic.** P1 produces leaderboards. Adopting winners into
`configs/tuned_weights.yaml` and rerunning the comparison is a separate,
deliberate step. P1 changes no reported number by itself.

### P1 first pass — 3 of 4 methods tuned; proposed_hfl hit a LATENT BUG (2026-08-26)

Best `val_macro_f1` at 5 km (tuning seeds 20-22, N=30/50, subsample 0.2):

    oort          0.3497   (30 trials)
    fedcs         0.3467   (30 trials)
    flat_fl       0.3425   (30 trials)
    proposed_hfl  -inf     (50 trials, ALL FAILED)

`-inf` is not a result. All 50 trials raised
`TypeError: <lambda>() got an unexpected keyword argument 'link'`.

**Root cause, and it predates this programme.** `scripts/tune_weights.py`
monkeypatches `fed.Fitness` with a lambda accepting only `instance` when the
search space contains `fitness_w*`. `Fitness.__init__` gained a `link`
parameter in the **2026-08-06 placement redesign**, and `run_full_hfl` calls it
as `Fitness(instance, link=..., coverage_mode=..., **w)`. The last HPO ran
**2026-08-05**, one day earlier. So from the redesign onward, ANY attempt to
tune the proposed method's fitness weights silently scored -inf, and nothing
caught it because proposed_hfl is the only method whose space contains
`fitness_w*` and it had not been retuned since.

**Consequence for the paper:** the placement weights in
`configs/tuned_weights.yaml` were fitted under the OLD placement physics
(20 km, range-gate link) and have been unrevisable ever since. Every result in
this programme ran them.

Fixed at `ab7fc884b`: the patch now forwards `link`/`coverage_mode` untouched
and lets the tuned triple win. Verified — trials return finite values (0.2550,
0.1276) instead of -inf. proposed_hfl is being retuned as study
`w5k_proposed_hfl_fix`, 50 trials.

**To verify at adoption time, BEFORE adopting anything:** `Fitness`'s w1/w2/w3
are the SHARED placement objective, not a proposed-method-only constant, and
every method uses pso placement. So a tuned triple must be applied to ALL arms
or to NONE. Tuning it inside proposed_hfl's full space and applying it only
there would hand the proposed method a placement advantage the baselines never
got — the same single-arm tuning defect that inflated v4.

### R1 verdict — FAILS. STOP-EXHAUSTED. (2026-08-26)

`results/paper_full_5km_verdict.txt`. Criteria fixed in
`configs/paper_full_5km.yaml` at `4378e5379` with ZERO result files in
existence. **All 16 method/cell combinations cleared the collapse gate.**
Config merge verified before scoring: R_comm 5000, K 20, capacity 6, 100
rounds, 10 seeds, and the stale 20 km recipes confirmed overridden
(oort lr 0.0073 -> 0.0347, a 4.8x change).

**0/4 on all three criteria.**

    proposed_hfl - fedcs    -0.0700 / -0.1241 / -0.0677 / -0.0898  (all Holm-sig losses)
    proposed_hfl - oort     -0.0498 / -0.0631 / -0.0442 / -0.0593  (2/4 Holm-sig losses)
    proposed_hfl - flat_fl  -0.1907 / -0.1961 / -0.1502 / -0.1389  (all Holm-sig losses)

The margins are LARGER than H5's, because flat_fl received its own tuned recipe
for the first time (lr 0.0382 against the 0.0230 it used to inherit). Tuning the
one under-tuned arm made the negative result stronger, which is exactly what was
predicted when it was included.

**The val proxy did not transfer.** Pooled val had proposed_hfl first at 0.3533.
On the real protocol it is last of four. A 20-round, 0.2-subsample proxy does not
predict 100-round full-data performance here, and that is worth stating as a
methodological finding in its own right.

#### Observation, NOT a pre-registered test: an early-round advantage that reverses

From R1's own round trajectories (post-hoc analysis of already-collected data,
reported with that caveat):

    proposed_hfl - oort   r<=20        r<=50        r91-100
      N=100             +0.0260**    -0.0002      -0.0442*
      N=200             +0.0505**    +0.0078      -0.0593*

proposed_hfl converges FASTER than oort in early rounds at large client counts,
reaches parity by round 50, and is overtaken by round 100. This explains the
proxy discrepancy exactly, since the proxy horizon was 20 rounds.

**It does not rescue the method, and must not be reported as if it does.**
Against flat_fl, proposed_hfl loses at EVERY round budget (-0.064 / -0.077 /
-0.050 / -0.003 at r<=20). **There is no round budget at which the proposed
method leads the field.** The early advantage is relative to one baseline only.
Promoting it to a claim would require pre-registration and fresh seeds; it is
recorded here as an observation and nothing more.

## FINAL: STOP-EXHAUSTED

Per §5, the pre-committed outcome. Every hypothesis failed and the prerequisite
is closed:

| arm | verdict |
|---|---|
| H1 fusion ownership | FAIL — collapsed 4/4 |
| H3 capacity | FAIL — manipulation inert |
| H5 selectors at K=10 | FAIL — clean, all 8 cells Holm-sig losses |
| H2 shard diversity | INERT — verified not to intervene, not run |
| P1 re-tune at 5 km | done; exposed and fixed a latent tuner bug |
| R1 fair rerun | FAIL — 0/4 on all three criteria |

**The proposed client-selection method does not beat the literature selectors,
and the hierarchy does not beat flat FL, at a physically coherent radius.** This
was tested under conditions deliberately favourable to it: its own tuned recipe,
a 19-parameter search against the baselines' ~5, 50 trials against their 30, and
a fill-to-capacity roster builder already shown to be the better one.

No further arm is licensed. §6 permits additions only from a measured mediator,
and the two mediators this programme identified — shard width and shard class
entropy — were each manipulated and each failed to move accuracy.

**What the paper reports** is the crossover characterisation: hierarchical FL
with class-aware selection pays off only when UAV reach exceeds the spatial
correlation length of the label field. Below it, geographic sharding leaves the
average UAV pooling 1-4 clients, making the UAV tier a lossy aggregation hop
rather than a pooling layer. The radius sweep (-0.165 at 500 m rising
monotonically to -0.019 at 5000 m) is the quantitative backbone, and that single
mechanism explains every negative result recorded above.

## 8. Reopened: `client_full` ownership, and the decision rule for R2

STOP-EXHAUSTED was declared at §7 on the basis that no arm could close the
deficit. That conclusion was reached under ONE architecture, `fusion_owner: uav`,
in which **clients train only `struct_branch`** and the entire image pathway is
trained on a UAV shard pooling 1.2-3.8 clients. R1 itself showed `flat_fl` beats
that while leaving `img_proj` at random initialisation — a frozen random
projection outperforming a trained pathway. That points at the OWNERSHIP SPLIT,
not at the hierarchy, and the split had never been varied except by H1, which
moved fusion alone and collapsed.

`client_full` (commit `11dba5865`) gives clients the whole model and demotes the
UAV tier to a pure aggregator, still doing placement and selection: textbook
hierarchical FedAvg. Guarded by `check_fusion_owner.py`, 8 checks, including two
asserting the UAV trains nothing and that clients train `img_proj` where
`flat_fl` does not.

### Development evidence (NOT a result)

`results/dev_client_full`, seeds **20-24**, deliberately disjoint from the
evaluation seeds 0-9, which remain untouched. Variants tried: **1**.

    proposed_hfl - fedcs    +0.0434 / +0.0509   (N=50 / N=100)
    proposed_hfl - oort     +0.0327 / +0.0271
    proposed_hfl - flat_fl  -0.0204 / -0.0216

The flat_fl deficit falls from R1's -0.196/-0.150 to -0.020/-0.022. At n=5 the
minimum two-sided Wilcoxon p is 0.0625, so p=0.062 means all five seeds agreed —
the strongest signal available at that size, and NOT significance. Dev seeds are
also easier in absolute terms (flat_fl 0.471 here vs 0.393 elsewhere), so only
within-run contrasts are meaningful.

### R2 — the confirmatory arm

Recipes re-tuned under `client_full` by P1b for proposed_hfl, fedcs and oort
(flat_fl is unaffected by the mode). Adoption is recipe-params only, symmetric,
per the `tuned_weights.yaml` rule. Evaluation seeds 0-9, 100 rounds, full data,
4 N values.

**Criteria, unchanged from H5 and R1 — the bar does not move with the
architecture:**

  1. proposed_hfl Holm-beats fedcs at >= 3 of 4 N
  2. proposed_hfl Holm-beats oort  at >= 3 of 4 N
  3. proposed_hfl NOT significantly worse than flat_fl at >= 3 of 4 N

### The decision rule, fixed BEFORE R2 runs

**If R2 PASSES — Path A.** This is still only gate 1 of §2. It then requires:
replication on seeds **10-19** (untouched by development, which used 20-24, and
by R2, which uses 0-9); expansion to the full method set for the paper table;
and regeneration of `paper_coverage_v5` (radius sweep) and `paper_uav_count`
(fleet sweep) under `client_full`. Those two are NOT optional: both measure
quantities that depend on what the UAV tier trains, and reporting a main table
under `client_full` beside sweeps under `uav` would make the paper internally
inconsistent. Estimated ~20 h VM.

**If R2 FAILS — Path B.** STOP-EXHAUSTED at §7 stands as written. The existing
`uav` corpus is complete and internally consistent, and the crossover
characterisation is the paper. `client_full` is then reported honestly as a
development variant that looked promising on 5 dev seeds and did not confirm —
which is precisely the outcome the dev/evaluation split exists to detect, and
the second time this project has caught one (see the C2 K=10 collapse).

**Superseded either way:** `roster_control`, `capacity12`, `selectors_k10` and
`fusion_owner` probe a UAV tier that does not train under `client_full`. Under
Path A they describe the old architecture and must not be cited as evidence
about the new one. The four `tier1_*` runs and the MCLP reference are pure
placement and survive under both paths.

### §8 AMENDMENT, recorded BEFORE R2 runs: R2 does NOT adopt P1b's recipes

§8 above said R2 would use the recipes P1b tuned under `client_full`. That is
being changed, and the reason is recorded here before any R2 seed exists.

**P1b's proxy cannot discriminate under this architecture.** Best pooled val:

    fedcs 0.3437   oort 0.3444   proposed_hfl 0.3436     spread 0.0008
    (under `uav` the same proxy spread was 0.009: 0.3467 / 0.3497 / 0.3555)

A 0.0008 spread across three methods is noise. The proxy is 20 rounds at 0.2
subsample; `client_full` trains far more parameters at the client tier and so
converges later, which the proxy scores as worse. It is measuring convergence
SPEED, not final quality. Adopting near-randomly-selected recipes would inject
noise into the confirmation without making it fairer.

**The stronger reason: a confirmatory run must replicate the development
configuration on held-out data.** The dev result was produced with the recipes
inherited from `tuned_weights.yaml`, applied symmetrically to every arm
including flat_fl. Changing the recipes between development and confirmation
would mean R2 tests something other than what dev found, and a null could not
be attributed.

**R2 is therefore an exact replication of `configs/dev_client_full.yaml` on
evaluation seeds 0-9 with the full N grid.** Same recipes, same symmetry, new
data. Criteria unchanged.

**Consequence for Path A, stated now:** if R2 passes, the SHIPPED table still
requires a proper re-tune under `client_full` (§3 P1 is not waived). That tune
must use a horizon long enough to discriminate — the 20-round proxy is
demonstrably unfit for this architecture — which is itself a finding worth
reporting. Confirm the effect first, then perfect the tuning; not the reverse.

### R2 verdict — FAILS the criteria, and FALSIFIES §7's stated conclusion (2026-08-28)

`results/r2_client_full_verdict.txt`. Criteria fixed in
`configs/r2_client_full.yaml` at `d2a30fc7c` with zero result files existing;
the amendment explaining the recipe choice at `068f1068e`, also beforehand.
Evaluation seeds 0-9, untouched by development (which used 20-24).
**All 16 method/cell combinations cleared the collapse gate.**

    proposed_hfl - fedcs    -0.0122 / +0.0217 / +0.0481 / +0.0512   -> 3/4 PASS
    proposed_hfl - oort     +0.0082 / +0.0147 / +0.0361 / +0.0512   -> 2/4 FAIL
    proposed_hfl - flat_fl  -0.0962 / -0.0225 / -0.0055 / +0.0124   -> 3/4 PASS

**VERDICT: FAIL.** All three criteria were required. Criterion 2 needed
Holm-significant wins over oort at >= 3 of 4 N and delivered 2. A marginal
result is a failure under the bar as written, and is recorded as one.

**No Path A.** No regeneration of the sweeps, no expansion of the method set,
no claim of a win. The dev signal (n=5, seeds 20-24) did not fully confirm.

#### But §7's conclusion is now FALSE and is retracted here

§7 stated: "The proposed client-selection method does not beat the literature
selectors, and the hierarchy does not beat flat FL, at a physically coherent
radius." R2 falsifies both halves.

* It beats fedcs at 3 of 4 N, Holm-significant.
* It is positive against oort at **all four** N (Holm-significant at 2).
* It is statistically indistinguishable from flat_fl at 3 of 4 N, having been
  -0.14 to -0.20 against it under `uav`.

Across the three families, 8 of 12 cells are positive and 5 are
Holm-significant positive, against 1 Holm-significant negative.

**The mechanism behind every negative result in §7 was the BLOCK-OWNERSHIP
SPLIT, not the hierarchy.** Under `uav`, clients trained only `struct_branch`
while the whole image pathway was trained on a UAV shard pooling 1.2-3.8
clients. That is why `flat_fl` — which never trains `img_proj` at all — beat
every hierarchical arm. Give clients the full model and the deficit closes.

H1/H3/H5/R1 were therefore all measured inside a design defect. They remain
valid statements ABOUT THAT DESIGN and must be reported as such, but they do
not support the general claim §7 drew from them.

#### What R2 does and does not license

DOES: reporting that the ownership split, not the hierarchy, caused the deficit;
that the proposed selector beats fedcs at scale; that its advantage GROWS
monotonically with N in every family (-0.012 -> +0.051 vs fedcs;
+0.008 -> +0.051 vs oort), tracking `mean_shard_clients` 1.18 -> 3.79 and
replicating the documented selection-isolation scaling result.

DOES NOT: any claim of beating the literature selectors generally. Criterion 2
failed. The shortfall is concentrated at N=30, the thinnest-shard cell, where
proposed_hfl also takes its only Holm-significant loss to flat_fl (-0.096).

**Seeds 10-19 remain untouched** by development and by R2, so a §2 gate-2
replication is still available if a NEW pre-registration is written. None is
written here: continuing to iterate after a pre-committed stop, without a new
registration, is the optional stopping §1 forbids.

### R3 verdict — REPLICATION CONFIRMED (2026-08-28)

`results/r3_replication`. Criteria fixed in `configs/r3_replication.yaml` at
`9106b6540` with zero result files existing. Seeds **10-19**, verified before
launch as untouched by development (20-24) and by R2 (0-9). Identical to R2 in
every respect except the seed window. **All 16 cells cleared the collapse gate.**

    R-1 sign agreement >= 3/4 in every family    PASS  (11 of 12 cells agree)
    R-2 beats fedcs >= 3/4, Holm                 PASS  (3/4)
    R-3 not worse than flat_fl >= 3/4            PASS  (3/4)
    R-4 monotone N=30 -> N=200 (fedcs & oort)    PASS

                         R2 (0-9)              R3 (10-19)
    minus fedcs    -0.012/+0.022/+0.048/+0.051  +0.007/+0.034*/+0.053*/+0.062*
    minus oort     +0.008/+0.015/+0.036/+0.051  +0.018/+0.031*/+0.044*/+0.056*
    minus flat_fl  -0.096/-0.023/-0.006/+0.012  -0.088*/-0.014/-0.006/+0.009
                                                 (* Holm-significant in R3)

#### R2's FAIL STANDS. It is not converted by R3.

R3 gives 3/4 Holm-significant wins over oort where R2 gave 2/4. **Combining
them to declare the composite bar passed would be optional stopping**, and
`configs/r3_replication.yaml` excluded the oort criterion from R3's primary
set precisely so this could not happen. R2 failed its bar; that verdict is
final. What R3 establishes is that R2's EFFECTS are real, not that R2 passed.

Both statements are true and neither cancels the other.

#### What is now supported, with independent replication

1. **The block-ownership split, not the hierarchy, caused the deficit.** Under
   `uav` clients trained only `struct_branch` while the whole image pathway was
   trained on a UAV shard pooling 1.2-3.8 clients. `client_full` closes a
   -0.14/-0.20 gap against flat_fl to indistinguishability at N>=50, twice, on
   independent seeds.
2. **The proposed selector beats fedcs and oort at N >= 50**, Holm-significant
   in R3 at 3/4 for both, positive in sign in 7 of 8 cells across both runs.
3. **The advantage grows monotonically with N** in both runs, tracking
   `mean_shard_clients` 1.18 -> 3.79, replicating
   the documented selection-isolation scaling result.
4. **N=30 is a genuine failure case** and must be reported: the only
   Holm-significant loss to flat_fl in either run (-0.096, -0.088), and the
   thinnest-shard cell.

#### What is NOT supported

Any claim of beating the baselines across all N. The composite pre-registered
bar was failed. The honest sentence is: *at N >= 50 the method beats both
literature selectors with independent replication, and at N = 30 it does not
and is beaten by flat FL.*

#### Consequence for the corpus — now unavoidable

The paper's system is `client_full`. Every other FL result
(`paper_full`, `paper_coverage_v5` radius sweep, `paper_uav_count` fleet sweep,
all ablations) was produced under `uav` and therefore describes the defective
design. They remain valid AS the design-defect evidence, but the main table,
the sweeps and the ablations must be regenerated under `client_full` before the
paper ships, or it will report a headline system that differs between tables.
STOP applies to ARMS, not to this: no new hypothesis is licensed, and
regeneration adds no test.

### Main table under `client_full` — complete, and it reframes the claim (2026-08-28)

`results/r2_client_full` (4 methods) + `results/paper_full_cf` (9 methods),
seeds 0-9, identical seed streams so the trees concatenate exactly.
**All 36 method/cell combinations cleared the collapse gate.**

proposed_hfl minus X, Holm within each opponent row across the 4 N:

    fedcs             -0.0122   +0.0217*  +0.0481*  +0.0512*
    oort              +0.0082   +0.0147   +0.0361*  +0.0512*
    rep_cap           +0.0380*  +0.0093   +0.0275*  +0.0266*
    fair_mab          +0.0453*  +0.0427*  +0.0508*  +0.0432*
    power_of_choice   +0.0193   +0.0291*  +0.0398*  +0.0395*
    mozaffari2016     +0.0681*  +0.0657*  +0.0759*  +0.0576*
    alzenad2017       +0.0895*  +0.0947*  +0.0798*  +0.0724*
    ---------------------------------------------------------
    flat_fl           -0.0962*  -0.0225   -0.0055   +0.0124
    hfl_no_selection  -0.0798*  -0.0428*  -0.0351*  -0.0201*
    hfl_static        +0.0074   -0.0026   -0.0036   -0.0164
    hfl_no_reputation +0.0046   -0.0129   +0.0038   +0.0012
    centralized       -0.1808*  -0.1729*  -0.1700*  -0.1907*

#### The finding that changes the claim

**`hfl_no_selection` — the ablation that REMOVES the selection rule — beats
`proposed_hfl` at ALL FOUR N, Holm-significant.** Selecting at all costs
accuracy; using every reachable client is better. The proposed rule loses less
accuracy than the other selectors do, but it does not beat not-selecting.

**`hfl_static` and `hfl_no_reputation` are statistically indistinguishable from
`proposed_hfl` at every N.** Repositioning contributes nothing measurable
(consistent with the coverage-sweep negative result) and neither does
reputation. Of the three components, only selection has a measurable effect,
and its effect on accuracy is NEGATIVE relative to no selection.

#### What the paper may therefore claim, and what it may not

MAY: among methods operating under a participation constraint, the proposed
rule beats **all five literature selectors** — fedcs, oort, rep_cap, fair_mab,
power_of_choice — Holm-significant at 3-4 of 4 N for each, and beats both
placement baselines at every N. Replicated for fedcs/oort on independent seeds
(R3). And it attains the **best accuracy-per-MB of any method at every N**
(0.117/0.063/0.031/0.016 vs hfl_no_selection's 0.104/0.057/0.030/0.015), at
5-26% less communication per round.

MAY NOT: that it improves accuracy over the state of the art in absolute terms.
It does not beat its own no-selection ablation, and it does not beat flat FL at
N=30. Selection is a CONSTRAINT imposed by bandwidth and energy, not an
accuracy improvement, and the paper must say so.

MAY NOT: claim repositioning or reputation contribute. Both ablations are null.
Reporting them as components of a working system would be unsupported.

**Accuracy-per-MB is a DERIVED metric constructed after seeing the accuracy
result.** It is reported as secondary and clearly labelled post-hoc. The primary
accuracy comparison stands unmodified; the ratio does not rescue it.

#### Remaining corpus gap

`paper_coverage_v5` (radius sweep) and `paper_uav_count` (fleet sweep) are still
under `uav`. The radius sweep's crossover was measured inside the ownership
defect and is expected to flatten under `client_full`; it cannot be cited until
re-measured.

### CORPUS DEFECT found 2026-09-05: sweeps never applied per-method recipes

Found by cross-checking the SAME cell between two harnesses. At N=200, R=5000,
`client_full`, `mclp_ls` placement, `proposed_hfl - fedcs` read **+0.051** in the
main table and **-0.024** in the radius sweep. `proposed_hfl` itself reproduced
almost exactly (0.4046 vs 0.4049); `fedcs` did not (0.3534 vs 0.4289).

**Cause.** `fl.per_method` — the per-method tuned recipes — was applied in
`_paper_job` ONLY. `_job`, `_coverage_job`, `_uav_job` and `_selection_job` all
ignored it, so every method in those sweeps ran the BASE recipe, which is
`proposed_hfl`'s own. That is exactly the single-arm tuning defect the main
table was fixed to remove ([[baseline-lr-tuning-fairness]]), still live in every
other harness for the whole life of the project.

It is invisible within a sweep — each looks internally consistent. Only a
cross-harness comparison of the same cell exposes it.

**Fixed 2026-09-05** in all four builders, guarded by
`tests/sanity_checks/check_permethod.py` (3 checks, AST-level: these builders
run a full FL job, so there is no cheap behavioural test).

#### Which results this invalidates, and which survive

`flat_fl` has **no** `per_method` entry in `configs/tuned_weights.yaml`, so it
inherits the base recipe in EVERY harness. Any `proposed_hfl` vs `flat_fl`
comparison is therefore unaffected.

SURVIVES:
* the radius-sweep crossover (proposed vs flat_fl), all versions
* the fleet-sweep K / shard-width analysis (proposed vs flat_fl)
* the entire main table, R2 and R3 — all produced by `_paper_job`

INVALID and must not be cited:
* any radius- or fleet-sweep comparison of `proposed_hfl` against `fedcs`,
  `oort`, `rep_cap`, `fair_mab` or `power_of_choice` — i.e. the method columns
  of `paper_coverage_v5`, `paper_coverage_cf`, `paper_coverage_cf_mclp`,
  `paper_uav_count`, and every `selection_isolation*` / `selection_scaling*` run
* `probe_topology` and `probe_coverage_vs_k` cross-method readings

#### The crossover, re-measured under the citable configuration

`proposed_hfl - flat_fl` across R_comm (valid, flat_fl has no per-method entry):

    v5   uav + pso           -0.1651 -0.1376 -0.0890 -0.0573 -0.0466 -0.0189
    client_full + pso        +0.0061 +0.0139 +0.0500*+0.0405*+0.0329*+0.0342*
    client_full + mclp_ls    -0.1017*-0.0415*+0.0133 +0.0219 +0.0062 +0.0121

Under the configuration that matches the main table (`mclp_ls`), the crossover
SURVIVES but weakens sharply and now actually crosses: negative and significant
below ~1 km, positive from ~2 km. Under `uav` it never crossed at all.

So the honest statement is narrower than either earlier version: the ownership
split accounted for most of the deficit, and a genuine reach-dependence remains
at small radii. Placement method also interacts strongly with radius — pso and
mclp_ls disagree by 0.11 at R=500 — which is itself worth reporting and was
never visible while the two harnesses used different placement.

### Tuning horizon, derived from data (2026-09-05)

Why P1b could not discriminate, measured rather than guessed. Spearman
correlation between the 7-method ranking in a given round window and the final
ranking (r91-100), averaged over N=30/50/100/200, from the client_full main
table:

    rounds  1-10   rho = -0.54    ANTI-predictive
    rounds 11-20   rho = +0.22    <- the horizon P1/P1b tuned at
    rounds 21-30   rho = +0.60
    rounds 31-40   rho = +0.74
    rounds 41-50   rho = +0.85
    rounds 51-60   rho = +0.97
    rounds 81-90   rho = +1.00

proposed_hfl ranks 4th-5th of 7 in rounds 1-10 and 1st-3rd from round 21 on.
A short-horizon proxy does not merely add noise here, it measures a different
quantity: early-round convergence speed, which this method trades away.

**Consequence:** any future tuning must use >= 60 rounds. Both the 2026-08-05
HPO and P1/P1b used 20 and are therefore unreliable for ranking, though they
remain symmetric across methods and so do not bias the DIRECTION of any
reported comparison.

### Final radius sweep — verdict (2026-09-05, `results/paper_coverage_final`)

`client_full` + `mclp_ls` + per-method recipes actually applied. 30 000 rows,
6 radii x 5 methods x 10 seeds x 100 rounds, 0 errors, all 30 cells cleared the
collapse gate. Reduction = mean macro-F1 over the last 10 rounds; paired
Wilcoxon over seeds; Holm within each comparator family of 6 radii.

    proposed_hfl - flat_fl            -0.1017* -0.0415* +0.0133  +0.0219  +0.0062  +0.0121
    proposed_hfl - hfl_no_selection   -0.0470* -0.0531* -0.0255* -0.0272* -0.0370* -0.0206*
    proposed_hfl - fedcs              -0.0356* +0.0065  +0.0531* +0.0630* +0.0472* +0.0521*
    proposed_hfl - oort               -0.0204  +0.0123  +0.0499* +0.0632* +0.0488* +0.0520*
                                       R=500    1000     2000     3000     4000     5000

Mechanism, proposed_hfl, mean over seeds and last 10 rounds:

    R_comm             500    1000    2000    3000    4000    5000
    coverage_pct     12.30   23.35   43.25   56.05   69.70   80.50
    mean_shard_clients 1.038  1.184   1.646   1.966   2.357   2.824

The crossover against `flat_fl` survives and now actually crosses: significant
losses below 1 km, non-negative from 2 km, never a significant win. The wins
against `fedcs` and `oort` switch on at the same radius. Both transitions sit at
`mean_shard_clients` ~ 1.65, the same mediator and the same threshold as the
N-scaling result (1.18/1.48/2.31/3.79 at N=30/50/100/200). Two independent axes
acting through one measured quantity.

`hfl_no_selection` Holm-beats `proposed_hfl` at ALL SIX radii. This replicates
the main table, where it beats proposed at all four N. It is the most robust
negative in the programme and must be reported as such.

### The recipe-regime confound — the headline result is not currently defensible

Scoring `paper_coverage_final` against `paper_coverage_cf_mclp` isolates one
variable: whether each method runs its own 2026-08-05 recipe (per-method) or
every method runs `proposed_hfl`'s (shared). Nothing else differs.

Per-method delta relative to the shared regime, by method:

    proposed_hfl      0.0000 at every radius   (bit-identical)
    flat_fl           0.0000 at every radius   (bit-identical)
    hfl_no_selection  0.0000 at every radius   (bit-identical)
    fedcs            -0.0119 -0.0642 -0.0818 -0.0897 -0.0812 -0.0761
    oort             +0.0153 -0.0101 -0.0488 -0.0553 -0.0538 -0.0550

The three methods with no per-method entry reproduce exactly, which is a clean
determinism check. The two that have one are made WORSE by up to 0.09 by being
given their own tuned recipe.

Consequently the headline comparison inverts with the regime:

    vs fedcs   per-method: -0.0356* +0.0065  +0.0531* +0.0630* +0.0472* +0.0521*
    vs fedcs   shared    : -0.0475* -0.0577* -0.0287* -0.0267* -0.0339* -0.0239*
    vs oort    per-method: -0.0204  +0.0123  +0.0499* +0.0632* +0.0488* +0.0520*
    vs oort    shared    : -0.0051  +0.0022  +0.0011  +0.0079  -0.0049  -0.0031

Under the shared regime `proposed_hfl` LOSES to fedcs at every radius, all
Holm-significant, and ties oort everywhere. Under the per-method regime it wins
both by ~0.05 from 2 km up. **The sign of the paper's main claim is decided by
which of two tuning regimes is used, not by the method.**

This is not "the baselines got lucky." The shared recipe is `proposed_hfl`'s own
tuned recipe; if it biased anything it should bias toward `proposed_hfl`. It
does not.

Root cause: every recipe in `configs/tuned_weights.yaml` comes from the
2026-08-05 search, which ran at **20 rounds under `fusion_owner: uav`**. The
horizon analysis above measured rank correlation with final ordering at
rho = +0.22 for rounds 11-20. So the per-method recipes were selected by a proxy
that barely correlates with the quantity being compared, under an architecture
the paper no longer uses. For fedcs and oort that proxy picked recipes that are
actively worse at 100 rounds than no per-method tuning at all.

`results/r2_client_full`, `results/r3_replication` and `results/paper_full_cf`
all carry `per_method` in their resolved configs with the same fedcs entry
(`lr: 0.019217`, `lr_decay: sqrt`). **The entire main table has this defect.**
The wins over fedcs, oort, rep_cap, fair_mab and power_of_choice cannot be
defended against the first reviewer who asks how the baselines were tuned.

Neither regime is publishable. The shared regime imposes one method's recipe on
all; the per-method regime uses recipes from an invalid horizon. The comparison
is unanchored either way.

**Required remedy, and it is not optional:** re-tune every compared method at
>= 60 rounds under `client_full`, symmetrically, on the tuning seeds, then
regenerate the main table. This is the work deprioritised on 2026-09-05 as
"full 7-method re-tune, ~7 h, low value". That judgement was wrong and this
sweep is the evidence. Scope: `proposed_hfl`, `fedcs`, `oort`, `flat_fl`
(already running), `hfl_no_selection`, `rep_cap`, `fair_mab`,
`power_of_choice`. The ablations `hfl_static` and `hfl_no_reputation` continue
to inherit `proposed_hfl`'s recipe by design — they ablate that system, and
tuning them separately would ablate the recipe too.

No criteria attach to the regeneration itself; it tests nothing and consumes no
alpha. What it does is make the existing pre-registered criteria answerable.

### Recipe-regime triangulation (2026-09-05, `scripts/score_recipe_regimes.py`)

Three independent recipe sets, same architecture (`client_full`), same
reduction. A and B sweep R_comm at N=200 (n=10); C sweeps N at R=5000 (n=5,
seeds 20-24, where the minimum two-sided Wilcoxon p is 0.0625 and no cell can
reach significance — sign check only).

    A shared      every method on proposed_hfl's recipe   results/paper_coverage_cf_mclp
    B per-method  each on its own 2026-08-05 recipe       results/paper_coverage_final
    C alt         P1's 5 km recipes                       results/robustness_alt_recipes

`proposed_hfl` minus comparator (* = raw Wilcoxon p < 0.05):

    A  fedcs             -0.0475* -0.0577* -0.0287* -0.0267* -0.0339* -0.0239*
    B  fedcs             -0.0356* +0.0065  +0.0531* +0.0630* +0.0472* +0.0521*
    A  oort              -0.0051  +0.0022  +0.0011  +0.0079  -0.0049  -0.0031
    B  oort              -0.0204* +0.0123* +0.0499* +0.0632* +0.0488* +0.0520*
    A  flat_fl           -0.1017* -0.0415* +0.0133  +0.0219* +0.0062  +0.0121
    B  flat_fl           -0.1017* -0.0415* +0.0133  +0.0219* +0.0062  +0.0121
    A  hfl_no_selection  -0.0470* -0.0531* -0.0255* -0.0272* -0.0370* -0.0206*
    B  hfl_no_selection  -0.0470* -0.0531* -0.0255* -0.0272* -0.0370* -0.0206*
                          R=500    1000     2000     3000     4000     5000

Sign agreement A vs B, on identical cells:

    vs fedcs             1/6   ORDERING IS RECIPE-DEPENDENT
    vs oort              4/6   ORDERING IS RECIPE-DEPENDENT
    vs flat_fl           6/6
    vs hfl_no_selection  6/6

Regime C, ranked by mean macro-F1 over the last 10 rounds:

    N=50    flat_fl 0.5232  fedcs 0.4909  proposed_hfl 0.4662  oort 0.4471
    N=100   flat_fl 0.5064  fedcs 0.4396  oort 0.4275  proposed_hfl 0.4227

    proposed_hfl - fedcs    -0.0247  -0.0169
    proposed_hfl - oort     +0.0191  -0.0048
    proposed_hfl - flat_fl  -0.0570  -0.0838

#### What triangulates and what does not

**Regime B is the outlier.** `proposed_hfl` beats `fedcs` and `oort` in exactly
one of three recipe regimes — the one built from the 20-round proxy measured at
rho = +0.22 against the final ordering, and the one that demonstrably degrades
fedcs by up to 0.09 and oort by up to 0.055 relative to no per-method tuning at
all. In regime A it loses to fedcs at all six radii and ties oort. In regime C
it ranks 3rd of 4 at N=50 and 4th of 4 at N=100.

The main table (`r2_client_full`, `r3_replication`, `paper_full_cf`) was
produced under regime B. **The selector claim rests on the least defensible of
the three regimes and does not survive either alternative.**

**Recipe-invariant, and therefore reportable now:**

* `hfl_no_selection` beats `proposed_hfl` in 6/6 cells under both A and B, and
  at all four N in the main table. Using every reachable client beats selecting
  among them. This is the most robust finding in the programme and it is
  negative for the selection rule.
* The `flat_fl` curve is bit-identical across A and B (neither method has a
  per-method entry). The crossover at ~2 km and the shard-width threshold at
  `mean_shard_clients` ~ 1.65 are recipe-independent.
* The block-ownership effect (0.14-0.20, `uav` -> `client_full`) is an order of
  magnitude larger than any recipe difference observed here and is replicated on
  independent seeds (R3). It is not at risk from this confound.

The symmetric 60-round retune (`scripts/retune60.sh`, launched 15:30 UTC) is the
decisive test. Regime C is the closest existing approximation to a fair regime
and `proposed_hfl` loses there, so the prior going in should be that the
selector claim does not survive. That prediction is recorded here BEFORE the
retune completes, so the outcome cannot be re-narrated afterwards.

## VERDICT — the decisive table (2026-09-07, `results/paper_full_cf60`)

Symmetric 60-round recipes, `client_full`, evaluation seeds 0-9, 200 jobs,
0 errors, all 20 method/cell combinations cleared the collapse gate.

`proposed_hfl` minus comparator, mean macro-F1 over the last 10 rounds, paired
Wilcoxon over seeds 0-9, Holm within each family of 4 N (* = Holm-significant).
The previous table's row, produced under the 20-round `uav` recipes, is printed
beneath each for comparison.

    vs fedcs             -0.0612* -0.0336* -0.0084* -0.0085*
      previous           -0.0122  +0.0217* +0.0481* +0.0512*

    vs oort              +0.0031  -0.0047  +0.0097  +0.0045
      previous           +0.0082  +0.0147  +0.0361* +0.0512*

    vs flat_fl           -0.1350* -0.0755* -0.0586* -0.0430*
      previous           -0.0962* -0.0225  -0.0055  +0.0124

    vs hfl_no_selection  -0.0790* -0.0542* -0.0338* -0.0309*
                          N=30     N=50     N=100    N=200

### Pre-registered criteria

    1. Holm-beats fedcs at >= 3 of 4 N          0/4   FAIL
    2. Holm-beats oort  at >= 3 of 4 N          0/4   FAIL
    3. Not sig. worse than flat_fl at >= 3 of 4 0/4   FAIL

    COMPOSITE: FAIL, 0 of 3.

Ranking by mean macro-F1 over the last 10 rounds:

    N=30    flat_fl .5366  hfl_no_selection .4806  fedcs .4628  proposed_hfl .4016  oort .3985
    N=50    flat_fl .4773  hfl_no_selection .4561  fedcs .4355  oort .4066  proposed_hfl .4018
    N=100   flat_fl .4519  hfl_no_selection .4270  fedcs .4016  proposed_hfl .3932  oort .3836
    N=200   flat_fl .4035  hfl_no_selection .3913  fedcs .3690  proposed_hfl .3604  oort .3559

`proposed_hfl` ranks 4th or 5th of 5 at every N.

### This confirms the prediction recorded before the run

The entry of 2026-09-05 stated: "the prior going in should be that the selector
claim does not survive. That prediction is recorded here BEFORE the retune
completes, so the outcome cannot be re-narrated afterwards." It did not survive.

The apparent wins over fedcs (+0.022/+0.048/+0.051) and oort (+0.036/+0.051)
in `r2_client_full`, `r3_replication` and `paper_full_cf` were artefacts of
baselines handicapped by recipes searched at a 20-round horizon under a
different architecture. Given equal search budget, equal space, equal horizon
and equal seeds, `proposed_hfl` Holm-LOSES to fedcs at all four N and is
statistically indistinguishable from oort at all four.

**R3's replication does not rescue this.** R3 replicated R2 faithfully — same
seeds discipline, same criteria, independent seeds — and both are invalidated by
the same defect, because both used the same handicapped baseline recipes. A
replication inherits its comparison's confounds. That is the lesson, and it is
worth more than the result it retracts.

### STOP

§5's stopping rules apply. The composite failed on fresh evaluation seeds under
the fairest configuration the project can construct. No further variant of the
selection rule is licensed without a new pre-registration, and no such
pre-registration is contemplated: the mechanism evidence below explains WHY the
rule cannot win here, so further search would be search against a known cause.

### What survives, and it is not nothing

1. **Block ownership is the dominant design variable.** Moving `img_proj` and
   `fusion` from the UAV tier to the clients is worth 0.14-0.20 macro-F1, an
   order of magnitude more than any selection rule, recipe, or placement effect
   measured in this project. Replicated on independent seeds (R3). The contrast
   holds recipes constant across the two ownerships, so the retune does not
   disturb it.

2. **The deficit is governed by shard width, and it is monotone.** Against
   `flat_fl` the gap closes steadily as clients per shard rises:
   -0.1350 / -0.0755 / -0.0586 / -0.0430 at N=30/50/100/200, tracking
   `mean_shard_clients` 1.18/1.48/2.31/3.79. The radius sweep gives the same
   mechanism on an independent axis. It never crosses zero within the tested
   range, so the honest statement is a rate and a direction, not a crossover.

3. **Client selection does not pay for itself.** `hfl_no_selection` — identical
   hierarchy, no selector — Holm-beats `proposed_hfl` at all four N here, at all
   six radii in `paper_coverage_final`, and under every recipe regime tested.
   Using every reachable client beats choosing among them. This is the most
   robust finding in the programme and it is negative.

4. **Tuning horizon is a first-order confound in FL benchmarking.** A 20-round
   proxy correlates with the 100-round ordering at rho = +0.22; 60 rounds
   reaches +0.97. Recipes adopted at the short horizon made fedcs worse by up to
   0.09 than no per-method tuning at all, and flipped the sign of the headline
   comparison. Any FL paper that tunes at a short horizon and reports at a long
   one is exposed to exactly this. That is a methodological result worth
   reporting on its own.

The paper this data supports is a mechanism-and-negative-result paper, not a
new-selector paper. The selector claim is retracted.
