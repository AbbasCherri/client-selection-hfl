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
