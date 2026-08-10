# Reframing the paper — what the corrected evidence supports

Written 2026-08-10, after the v5 rebuild's four FL blocks and the v6 C1 arm.
Every claim below points at a row in `REPORTS/results_provenance.md`; anything
that does not is marked as pending.

---

## 1. What died

Three claims carried the previous draft. None survives the corrected regime.

| Claim | Status | Evidence |
|---|---|---|
| The proposed selector beats FedCS / Oort / PoC / rep-cap / fair-MAB | **Dead — reversed** | `paper_full` v5: Holm-*loses* to fedcs, oort and flat_fl; Holm-beats no literature selector at any N |
| The hierarchy beats flat FL by ~0.19 | **Dead — reversed** | `flat_fl` 0.393-0.510 beats every hierarchical arm at N=30/50/100; 1 of 102 coverage-sweep cells beats it, and that cell is a literature method |
| Class-aware *selection* is the contribution | **Dead — inverted** | Class-realism v5: selection +0.017 (p=0.375, null); *placement* +0.039 is the only Holm survivor |

The cause is common to all three and is not a bug in the method: the earlier
results ran at R_comm = 20 km with a 20-120 m altitude band. Under the
Al-Hourani channel the coherent ground radius is `z / tan(θ_opt)`, so that band
supports 54-324 m, and 20 km implies an elevation angle of 0.34° — a 97% NLoS
link, an aerial base station with its defining advantage switched off. Coverage
saturated at 99.95%, which handed the hierarchy free reach that flat FL could
not have and left class-aware placement nothing to steer.

**The honest one-line summary: the previous results measured the radius, not the
method.**

## 2. What survives, and it is not nothing

### C1. A corrected evaluation regime for UAV-assisted FL

The altitude/radius coherence condition is not new physics — it is Al-Hourani's
own result — but it is routinely violated in the UAV-FL literature, and we can
show what violating it costs, because **our own results invert when it is
fixed**. That is a stronger demonstration than criticising others' numbers: the
before/after is matched on every other variable, on the same data and code.

Deliverables: the coherence condition as a design constraint, the band/radius
pairing procedure, and `tests/sanity_checks/check_altitude_band.py` /
`scripts/gate_altitude.py` as the mechanical check.

### C2. An outcome-based degeneracy gate

macro-F1 must clear the closed-form constant-predictor baseline
`2·p_m / ((1+p_m)·C)` by 0.05 **and** every per-class F1 must clear 0.05. Both
thresholds derived, not chosen.

This matters because the runs it catches are *invisible*: complete,
fully-checkpointed, plausible-looking output with every sanity check passing.
The calibration set is in the provenance (three runs at macro-F1 0.287 / 0.266 /
0.262 against a 0.2245 floor, caught only by reading `f1_missing` by hand). In
the corrected regime the gate flags 47 of 102 coverage-sweep cells and 6 of 52
`paper_full` cells — and it separates two failure modes that a single accuracy
number conflates: majority-class collapse, and *never learning a rare class*
while the headline metric looks acceptable.

### C3. Placement geometry predicts accuracy and does not cause it — shown by two interventions

This is the strongest result in the paper, and as of 2026-08-10 it rests on two
matched, pre-registered interventions rather than one.

**Both interventions moved their target variable as designed. Neither produced
the predicted accuracy gain.**

| | target | did it move? | accuracy |
|---|---|---|---|
| **C1** reachable coverage | coverage_pct | yes, +8-11 pp mid-grid | **34%** of the observational slope's prediction, sign inconsistent across K |
| **C3** disjoint coverage | overlap | yes, *overshot* — more tiled than `moon2022` (unique-cover 0.913 vs 0.845 at K=20) | **worse** — −0.009 vs `mclp_place` at K=20 and K=30 |

C3's K=20 cell is the cleanest evidence in the project: coverage *higher* than
`mclp_place` (+2.3 pp) **and** overlap far lower — both target variables moving
favourably — and macro-F1 *down* 0.009.

`moon2022` beats every placement method at every fleet size, and by more than its
coverage explains. Its two measurable geometric signatures — more reach, less
overlap, each differing from `mclp_place` at p=0.002 — were each isolated and
each failed to transfer. **The mechanism is unidentified, and saying so is the
result.**

The supporting arithmetic for C1, which is the quantitative core:

*Observationally*, across 16 placement methods at fixed fleet size,
`coverage_pct` is the best available predictor of macro-F1: within-K r = **0.70**
(0.493 / 0.803 / 0.838 / 0.798 / 0.577 at K = 5/10/15/20/30), positive at every
fleet size, ahead of every other recorded mediator.

*Interventionally*, the v6 C1 arm changes only the placement objective — from
capacity-capped assigned coverage to reachable coverage — and moves coverage
+1.7 / +8.2 / +11.0 / +8.4 / +1.6 pp. The observational slope predicts +0.056
macro-F1 summed over K. It delivers **+0.019, or 34%**, with per-K realised
fractions of −0.31 / 0.88 / −0.36 / 0.56 / 2.38 — inconsistent in sign, not
merely attenuated.

So a placement benchmark that optimises coverage is optimising a variable whose
association with the end task is largely confounded. This indicts a standard
practice, it indicts our own earlier framing, and it is demonstrated by
intervention rather than by another correlation.

**Why two interventions is much stronger than one.** A single failed intervention
invites the reply "you changed the wrong variable". Two, targeting the two
different geometric properties that actually distinguish the winning method,
both firing mechanically and both failing to transfer, closes that door — and
the second one *overshot* its target, so "the effect was too small" does not
apply either.

### C4. A capacity floor with a distinct failure signature

Per-UAV shard width ≤ 3 makes the system *unlearn*: it rises to ~0.34 macro-F1
by round 10, then decays monotonically to the constant-predictor floor, while
width ≥ 4 recovers and climbs. A final-round number cannot distinguish this from
never having learned. Mechanism: with `fusion_owner: uav` each UAV trains the
fusion head on its pooled shard, and geographically clustered damage classes
make a narrow shard near-single-class.

**Scope, stated honestly:** this is a floor effect measured within one placement
method, where capacity was the only moving part. Across methods it does *not*
generalise — r(width, macro-F1) = 0.117 over 80 cells, and the widest-sharded
method in the grid is nearly the worst. Report it as a floor, never as an
explanatory variable.

## 3. Proposed narrative

Lead with the protocol, not with the failure. The paper is:

> **An evaluation protocol for UAV-assisted hierarchical FL, and what it
> overturns.**
>
> 1. UAV-FL evaluations are routinely run outside the coherent
>    altitude-radius regime of the air-to-ground channel. We give the
>    condition and a mechanical check.
> 2. Federated runs on clustered geospatial data fail in two distinct,
>    metric-invisible ways. We give a derived, outcome-based gate.
> 3. Under the corrected protocol, on real Noto 2024 damage data: placement
>    method does not drive end-task accuracy; coverage predicts it but does not
>    cause it (shown by intervention); and hierarchical FL does not beat flat FL
>    at most operating points.
> 4. Consequently, placement's value in this setting should be reported through
>    operational metrics — reach, movement energy, communication cost — and not
>    through accuracy.

Point 4 is the constructive close and is already supported by
`operational_summary.csv`.

**Do not** write this as "our method failed". Written as above, the negative
findings are the *result of* the protocol, which is the contribution.

## 3a. The power problem — read before writing any null

`scripts/power_analysis.py`, 2026-08-11. **Median minimum detectable effect at
n=10 is 0.0340 macro-F1**, at alpha=0.05 and 80% power. Most of the effects this
project argues about are smaller than that.

| comparison | observed | MDE | observed/MDE |
|---|---|---|---|
| class-realism **selection** | +0.0173 | 0.0597 | **0.29** |
| class-realism placement | +0.0391 | 0.0398 | 0.98 |
| C2 best cell (K=10) | +0.0330 | 0.0406 | 0.81 |
| C1 vs mclp, K=20 | +0.0051 | 0.0263 | 0.19 |

**So most individual per-cell nulls are uninformative and must not be written as
"no effect".** The correct sentence is "not detected; this design could not have
detected an effect below X", with X quoted.

What survives this unchanged, because the effects are large relative to MDE:

* `paper_full`'s losses to fedcs / oort / flat_fl (−0.035 … −0.167).
* C1 losing to `moon2022` at K=15 (obs/MDE 1.32); C3 at K=20 (1.14).
* **The coverage-causality result — because it rests on the AGGREGATE across
  K (34% of the predicted gain, summed), not on any single cell.** Write it that
  way. Never defend it by pointing at one K.
* C3's mechanism overshoot, which is a geometry measurement, not a null.

Class-realism placement at 0.98 is exactly on the threshold: report it as
marginally powered, not as a clean detection.

**Design implication for any future run: n=10 is too small for this problem.**
MDE scales as 1/√n, so resolving a 0.02 effect needs roughly n=30, and 0.015
needs n≈55. Any claim the paper wants to rest on a single cell needs that.

## 4. Risk, stated plainly

This is an evaluation/negative-result paper. Some Q1 venues take these readily
(empirical-rigour and benchmark tracks); others want a method that wins. The
mitigations, in order of value:

1. **The intervention (C3).** A refuted causal claim with a matched intervention
   is a genuine finding, not an absence of one. Lead with it.
2. **The self-inversion.** We are overturning our *own* published-quality
   results with the same code and data. That is unusually clean evidence and
   should be foregrounded, not buried.
3. ~~**C3-the-experiment** could still supply a positive mechanism.~~
   **Resolved 2026-08-10: it did not.** C3 fails all three substantive criteria
   and moves *away* from `moon2022`. There is no positive mechanism to add, and
   the paper should stop looking for one without new evidence — two candidate
   mechanisms are now refuted as causal and a third guess costs a grid at no
   better prior.

## 5. What is still pending

* v6 arms C2 and C1+C2, and `score_v6.py`'s formal verdict — running.
* Tier-1 rebuild under the shared path-loss channel — queued.
* C3 screen — queued behind Tier-1; both hypotheses can be rejected, which is a
  reportable outcome.
* `hfl_balanced_roster`, to close the roster-construction confound on the
  `paper_full` N-sweep. Lower stakes now: the confound was hypothesised to
  *inflate* an advantage that no longer exists, but the N-sweep should not be
  reported with a known uncontrolled difference.
* Every figure and table regenerated from v5/v6 results; the current
  `results/*/paper_*.png` predate the corrected regime for anything not listed
  in the provenance as v5.
