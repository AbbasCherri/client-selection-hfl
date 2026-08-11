# Paper draft

Working draft against `REPORTS/paper_reframe_2026-08-10.md`. Every number here
traces to a row in `REPORTS/results_provenance.md`; nothing is quoted that does
not. Bracketed `[TABLE n]` / `[FIG n]` markers map to
`REPORTS/paper_data_manifest.md`.

Status: §1-§7 drafted 2026-08-11. Two experiments still in flight (roster
control, C2 at n=25) — their placeholders are marked **[PENDING]** and are the
only gaps.

---

## Title

**Coverage Predicts but Does Not Cause: A Corrected-Physics Evaluation of UAV
Placement and Client Selection in Hierarchical Federated Learning**

Alternative, if the venue prefers the protocol framing: *"An Evaluation Protocol
for UAV-Assisted Hierarchical Federated Learning, and What It Overturns."*

## Abstract

UAV-assisted hierarchical federated learning (HFL) is usually evaluated by
configuring an altitude band and a communication radius independently. We show
this is not innocuous. Under the standard Al-Hourani air-to-ground model the
served ground radius is unimodal in altitude with a peak at a fixed elevation
angle, so a band and a radius chosen apart can specify a link no altitude in the
band can deliver: a 20-120 m band supports 54-324 m of ground radius, while a
20 km radius requires an altitude of 7.4 km. We give the coherence condition,
a mechanical check for it, and an outcome-based degeneracy gate that catches
runs which complete normally while having collapsed to a constant predictor.

Re-running a full UAV-HFL evaluation on real Noto 2024 earthquake damage data
inside the corrected regime **reverses our own previously-obtained results**.
The hierarchical system no longer beats flat federated learning; a proposed
class-aware selector that appeared to beat five literature selectors instead
loses to two of them; and the contribution attributed to class-aware selection
moves to class-aware placement.

We then test the one relationship that survives description. Across 16 placement
methods at fixed fleet size, coverage predicts macro-F1 with within-fleet
r = 0.70. Two pre-registered interventions — one maximising reachable coverage,
one penalising redundant overlap — each move their target geometric variable as
designed, one of them past the level of the best-performing baseline, and
**neither delivers the predicted accuracy**: the first realises 34% of the
slope's prediction with inconsistent sign, the second is negative. We conclude
that placement geometry which correlates strongly with downstream accuracy in
this setting does not cause it, and that placement's value should be reported
through operational metrics rather than end-task accuracy.

## 1. Introduction

After a major earthquake, the terrestrial network is among the first things to
fail, and the imagery needed to triage the response is spread across exactly the
devices that can no longer reach a base station. Uncrewed aerial vehicles are an
appealing answer: fly aggregation capacity to the data, let devices train
locally, and combine their updates at the aircraft. This pairing of UAV
placement with hierarchical federated learning (HFL) now has a substantial
literature, and its two halves are usually optimised against different
objectives — placement against a geometric coverage criterion, selection against
a learning criterion — on the assumption that improving the first improves the
second.

This paper argues that the assumption is untested, that the standard way of
configuring these evaluations can make them physically incoherent, and that when
both problems are fixed the usual conclusions do not survive. We know the last
part in an unusually direct way: **the results reported here reverse our own
earlier findings, produced by the same code on the same data, with one
configuration defect corrected.**

**The protocol defect.** UAV-FL evaluations typically fix an altitude band and a
communication radius as independent configuration values. Under the standard
Al-Hourani air-to-ground model they are not independent. The served ground
radius is unimodal in altitude — too low and buildings obstruct the link, too
high and free-space loss dominates — with its maximum at a fixed elevation angle
determined by the environment. A band and a radius chosen apart can therefore
specify a link that no altitude in the band can deliver. Concretely, a 20-120 m
band supports a ground radius of 54-324 m, while a 20 km radius requires flying
at 7.4 km. An evaluation configured at 20 km with a 120 m ceiling is operating
at an elevation angle of 0.34°, where the probability of line-of-sight is about
3% — an aerial base station with the property that motivates aerial base
stations switched off. Nothing in such a run announces the problem: the
simulator solves whatever budget it is handed, and every output looks normal.

**A second, quieter failure.** On heavily imbalanced data a federated run can
converge to something indistinguishable from a constant predictor while
producing complete, well-formed, plausible metrics. In our data the majority
class holds 81% of samples, so a model that predicts it always scores 0.75
accuracy and 0.22 macro-F1. Runs of this kind passed every check we had until we
began comparing macro-F1 against its closed-form constant-predictor value and
inspecting per-class F1 by hand. We formalise that comparison as a gate.

**What changes when both are fixed.** Re-running the full evaluation inside a
coherent regime on real Noto 2024 earthquake damage data reverses three findings
we had previously obtained: the hierarchical system no longer beats flat
federated learning; a class-aware selector that had appeared to beat five
literature selectors instead loses to two of them; and the measured contribution
moves from class-aware selection to class-aware placement. The explanation is
uncomfortable but simple — at 20 km coverage saturates at 99.95%, which hands
the hierarchy reach that flat FL cannot have and leaves class-aware placement
nothing to steer.

**The relationship that survives, and does not hold up.** One regularity is
robust to all of this: across 16 placement methods at a fixed fleet size,
coverage predicts downstream macro-F1 with within-fleet r = 0.70, positive at
every fleet size and ahead of every other quantity we record. It is exactly the
regularity that justifies optimising placement for coverage. We test it
interventionally rather than descriptively, with two pre-registered changes to
the placement objective — one maximising reachable coverage, one discounting
redundant overlap. Each moves its target geometric variable as designed, the
second past the level achieved by the best-performing baseline. **Neither
produces the predicted accuracy.** The first realises 34% of the slope's
prediction with inconsistent sign across fleet sizes; the second is negative. An
association of that strength should not behave that way, and we conclude it is
largely confounded.

**Contributions.**

1. A coherence condition relating altitude band to communication radius, with a
   mechanical check, and a demonstration of what violating it costs (§4.1, §6.3).
2. An outcome-based degeneracy gate with derived rather than chosen thresholds,
   which separates majority-class collapse from never learning a rare class
   (§4.2).
3. A corrected re-evaluation on real disaster data that reverses three of our
   own prior findings (§6.1-6.3).
4. An interventional test showing that the coverage-accuracy association, though
   strong, is largely non-causal — and that the advantage of the best baseline
   is explained by neither of its measurable geometric signatures (§6.5).
5. A reporting standard for negative results: every null is accompanied by the
   minimum effect the design could have detected (§6.7), and every candidate
   improvement is confirmed against criteria fixed before it is run — a
   discipline that, in this paper, caught one of our own apparent wins
   evaporating on fresh seeds (§6.8).

Contribution 3 is a self-inversion, and we state it plainly rather than burying
it. It is the paper's strongest piece of evidence about the protocol, because
the before and after are matched on every variable except the one corrected.

A note on the comparison to flat federated learning, which recurs throughout.
Flat FL presumes a surviving base station that the disaster premise excludes, so
it is an infeasible reference rather than a competitor, and parity with it is a
strong result for a system that assumes no ground network. We nonetheless report
it at every operating point, because a hierarchy that cannot approach it has not
earned its aircraft.

## 2. Related work

*Structure:* UAV placement (Mozaffari 2016, Al-Zenad 2017, Lyu 2017, Sawalmeh
2021, Moon 2022, Almaameri 2026); client selection in FL (FedCS, Oort,
Power-of-Choice, reputation- and MAB-based); hierarchical FL; air-to-ground
channel models (Al-Hourani). The gap: none of the placement work is evaluated
against a downstream learning objective under a coherent link budget, and the
selection work is evaluated without a placement layer.

## 3. System model

3.1 Air-to-ground channel — Al-Hourani LoS/NLoS, suburban preset, 2 GHz.
3.2 Coverage: each UAV's served radius is a function of its own altitude through
one shared channel. **No method carries a private radius** — this is the
structural fix for an unequal-radius comparison, not a patch.
3.3 Hierarchical FL: clients → UAV edge aggregation → global. `fusion_owner:
uav`, so each UAV trains the image projection and fusion head on the pooled
shard of its assigned clients. This is why per-UAV capacity is a shard width and
matters (§6.4).
3.4 Device value `V_i(t) = β(t)·U_i + (1−β(t))·R_i`.
3.5 Placement objective and its three terms; capacity-capped assignment.

## 4. Evaluation protocol

### 4.1 The coherence condition [FIG 1]

Served ground radius is unimodal in altitude with its peak at θ\* = 20.339°
(suburban), so the coherent radius is `z/tan θ\*`.

| band | coherent ground radius |
|---|---|
| 20-120 m (as commonly configured) | 54-324 m |
| 100-2000 m (this paper, FL) | 270-5395 m |
| 100-400 m (this paper, Tier-1) | 270-1079 m |

Required altitude for coherence: R = 500 m → 185 m; R = 5 km → 1853 m;
**R = 20 km → 7414 m.** At 20 km with a 120 m ceiling the elevation angle is
0.34° and P(LoS) ≈ 3% — an aerial base station with its defining advantage
switched off. Guarded by `check_altitude_band.py` / `gate_altitude.py`.

### 4.2 The degeneracy gate

A run is degenerate unless macro-F1 clears the closed-form constant-predictor
baseline `2·p_m/((1+p_m)·C)` by 0.05 **and** every per-class F1 clears 0.05.
Both thresholds derived, not chosen.

The runs this catches are invisible: complete, checkpointed, plausible, with
every sanity check passing. Calibration set: three runs at macro-F1 0.287 /
0.266 / 0.262 against a floor of 0.2245, identified only by reading `f1_missing`
by hand. In the corrected regime the gate flags 47 of 102 coverage-sweep cells
and 6 of 52 main-table cells — and it separates two failure modes a single
accuracy number conflates: majority-class collapse, and never learning a rare
class while the headline metric looks acceptable.

### 4.3 Reporting standard

Paired Wilcoxon, Holm within block, effect size and bootstrap CI on every
comparison, and **the minimum detectable effect quoted beside every null**
(§6.7).

## 5. Experimental setup [TABLE 1]

Noto 2024 earthquake damage, 128,843 samples, 200 clients partitioned
geographically.

| | |
|---|---|
| classes | survived 0.8143, collapsed 0.0264, obstructed 0.0649, missing 0.0943 |
| imbalance | 30.8:1; constant-predictor macro-F1 floor 0.2244 |
| per-client samples | 21-2678 (median 393.5), Gini 0.4146 |
| per-client class entropy | 0.3676 (normalised) |
| class presence | survived 100% of clients, missing 98.0%, obstructed 66.5%, **collapsed 50.5%** |
| extent | ~58 × 60 km |
| splits | 102,991 train / 12,793 val / 13,059 test |

The `collapsed` row is load-bearing: the rarest class is absent from half the
clients, so a narrow UAV shard is likely to contain none of it. This is the
mechanism behind the capacity floor (§6.4) and behind `f1_collapsed` being the
specific per-class failure that trips the gate.

Frozen ResNet-18 features are used throughout; §6.8 validates that choice.
Operating point K=20, capacity 6, R_comm 5 km, band 100-2000 m, 100 rounds,
10 seeds unless stated.

## 6. Results

### 6.1 Placement benchmark [TABLE 2, FIG 2]

7 methods × 4 scenarios (clustered, epicentre-biased, uniform, real Noto
coordinates) × 30 seeds. PSO Holm-beats every baseline on clustered,
epicentre-biased and real geometry in the capacity-binding configuration
(23/24 comparisons significant).

**It does not win everywhere, and the exceptions are systematic.**
`alzenad2017` Holm-beats PSO on `uniform` (−0.023), and on **both `real` and
`uniform`** in the saturating-capacity (−0.042, −0.032) and warm-start (−0.046,
−0.031) configurations. `tier1_core` binds capacity (150 slots < 250 clients)
while those two saturate it (300 slots). **PSO's advantage is capacity-awareness,
so it wins where capacity binds and loses to a capacity-blind physics baseline
where it does not.** PSO ties GA on `uniform` throughout.

**Caveat carried into the table:** `mozaffari2016` falls to 0.075-0.294 fitness
at 11.8-34.9% coverage. This is not a placement-quality result — under the shared
channel it no longer carries a private 618 m radius, and its altitude rule sends
it to the 400 m ceiling where z\* ≈ 185 m, so it earns a small radius by its own
choice. Report as a mis-calibrated altitude rule meeting a shared channel.

Near-optimality: a capacitated max-covering MILP solved to proven optimality on
all 12 instances puts PSO at 104.8% / 102.3% / 101.1% of the grid-restricted
optimum at 20×20 / 30×30 / 45×45. PSO exceeds it because it places continuously,
so **the MILP is a lower bound on the continuous optimum**; the convergence
curve is the honest form of the claim.

### 6.2 Federated learning main results [TABLE 3, FIG 3]

13 methods × N ∈ {30,50,100,200} × 10 seeds.

**The proposed method Holm-beats no literature selector anywhere, and loses to
three.**

| N | Holm wins | Holm losses |
|---|---|---|
| 30 | — | fedcs −0.079, flat_fl −0.168 |
| 50 | — | oort −0.105, fedcs −0.111, flat_fl −0.139 |
| 100 | alzenad2017 +0.099, rep_cap +0.082, mozaffari2016 +0.065, hfl_no_selection +0.059 | fedcs −0.035, oort −0.050, flat_fl −0.052 |
| 200 | alzenad2017 +0.090, mozaffari2016 +0.048 | oort −0.052 |

Every win is over a *placement* baseline or its own ablation; every loss is to a
*selection* baseline or to no-UAV flat FL. Best federated arm: `flat_fl` at
N=30/50/100, `oort` at N=200; `centralized` 0.595-0.599 throughout.

Selection is isolated within this table by construction — every selector runs on
identical placement.

### 6.3 The inversion

The prior version of this evaluation, at R=20 km with a 20-120 m band, reported
the proposed method beating all five literature selectors and beating flat FL by
0.19, with class-aware selection carrying the contribution. All three reverse in
the corrected regime. Coverage saturates at 99.95% at 20 km, which hands the
hierarchy reach that flat FL cannot have and leaves class-aware placement nothing
to steer.

Class-realism ablation: placement +0.0391 (p=0.0137, the only Holm survivor);
**selection +0.0173, not detected — and see §6.7, the design could not have
detected an effect below 0.06.**

### 6.4 Capacity is a shard width [FIG 4]

Capacity ≤ 3 makes the system *unlearn*: it rises to ~0.34 macro-F1 by round 10
then decays monotonically to the floor, while capacity ≥ 4 recovers and climbs.
A final-round number cannot distinguish this from never having learned.

**Scope, stated:** this is a floor effect measured within one placement method
where capacity was the only moving part. Across methods it does not generalise —
r(shard width, macro-F1) = 0.117 over 80 cells, and the widest-sharded method in
the grid is nearly the worst. Report as a floor, never as an explanatory
variable.

### 6.5 Coverage predicts accuracy and does not cause it [TABLE 4, FIG 5]

*Observational.* Across 16 methods at fixed fleet size, coverage is the strongest
mediator of macro-F1: within-K r = 0.70 (0.493/0.803/0.838/0.798/0.577 at
K=5/10/15/20/30), positive at every fleet size, ahead of active UAVs 0.63, shard
class entropy 0.42, shard width 0.39, fairness 0.37, unique clients 0.35 — the
last three flip sign at some K.

*Intervention 1 (reachable coverage).* Changes only the placement objective;
lifts coverage +1.7/+8.2/+11.0/+8.4/+1.6 pp. Predicted gain from the slope
+0.056 summed over K; **realised +0.019 = 34%**, with per-K realised fractions
−0.31/0.88/−0.36/0.56/2.38 — inconsistent in sign, not merely attenuated.

*Intervention 2 (redundancy-discounted coverage).* The best baseline,
`moon2022`, tiles: 85% of its covered devices are reached by exactly one
aircraft (multiplicity 1.16) against the proposed method's 60% (1.53), p=0.002.
Discounting redundant coverage produces layouts **more tiled than `moon2022`'s**
(unique-cover 0.913 vs 0.845 at K=20) — the objective overshot its target — and
macro-F1 *falls* 0.009. At K=20 the arm has **both** more coverage (+2.3 pp)
**and** less overlap than the method it replaces, and is still worse.

A prior screening step rejected the competing explanation that the tuned fitness
weights cause the under-coverage: removing the energy and imbalance terms
entirely closes only 7% / 14% of the coverage gap.

**Conclusion.** `moon2022`'s advantage is reproduced by neither of its two
measurable geometric signatures. The mechanism is **unidentified**, and an
association of r = 0.70 that behaves this way under direct intervention should
not be used to justify a placement objective.

### 6.6 Operational metrics [TABLE 5]

If placement does not buy accuracy (§6.5), the natural fallback is that it buys
operational quality — reach per joule, communication cost. The data does not
support even that for the optimisation-based methods.

Repositioning energy at K=20 spans **three orders of magnitude**, and it is not
repaid:

| method | coverage % | repositioning energy (J) | MJ per coverage point | macro-F1 |
|---|---|---|---|---|
| `flat_fl` (no UAVs) | 100.0 | 0 | 0 | 0.395 |
| `hfl_static` (place once) | 59.0 | 0 | 0 | 0.347 |
| `ahc_place` | **83.4** | 2.41e5 | **0.0029** | 0.366 |
| `spiral_place` | 78.2 | 2.28e5 | 0.0029 | 0.365 |
| `mclp_place` (proposed) | 80.4 | 8.05e6 | 0.100 | 0.380 |
| `moon2022` | **93.4** | 6.35e7 | 0.679 | **0.411** |
| `pso_cluster_place` | 75.8 | 1.93e8 | 2.54 | 0.370 |
| `proposed_hfl` (PSO) | 60.5 | 1.87e8 | 3.10 | 0.354 |
| `ga_place` | 46.7 | 1.90e8 | 4.08 | 0.379 |
| `random_place` | 32.0 | 1.94e8 | 6.08 | 0.327 |

Three things follow, and none is the expected one.

**The metaheuristic family is dominated.** PSO, GA, DE, GWO, MOGOA and
`cap_kmeans` all burn ~1.9e8 J to reach 47-76% of the population. `moon2022`
reaches 93.4% for a third of that, and `ahc_place` reaches 83.4% for **1/800th**
of it. Spending on repositioning does not buy reach here.

**The proposed placement is not the operational winner either.** `ahc_place`
achieves *higher* coverage than `mclp_place` (83.4% vs 80.4%) at 1/33 the
energy, giving up 0.014 macro-F1 — well inside the 0.034 MDE (§6.7), so that
difference is not resolved by this design.

**The honest positive statement is narrower than "placement pays
operationally":** among methods that reposition at all, the cheap deterministic
rules are on the efficient frontier, and the optimisation-based ones are not.
A deployment optimising for reach per joule should not be running a
metaheuristic.

*Caveat.* `cumulative_energy_j` counts **movement** energy only; hover and
communication have no simulated-time model here, so methods that place once
legitimately report zero and the column must be labelled as repositioning
energy, not total energy. Comparisons among repositioning methods are
unaffected.

### 6.7 Power [TABLE 6]

**Median minimum detectable effect at n=10 is 0.0340 macro-F1** (α=0.05,
power=0.80) — larger than most effects discussed. Per-comparison observed/MDE:
class-realism selection 0.29, pseudo-vs-none 0.17, intervention-1 vs proposed
0.05-0.51.

**Most individual per-cell nulls are therefore uninformative and are reported as
"not detected, and this design could not have detected an effect below X".**
What is well-powered and stands: the main-table losses (−0.035…−0.167), both
significant intervention losses (obs/MDE 1.32 and 1.14), and the
coverage-causality result — **which rests on the aggregate across K, not on any
single cell.**

### 6.8 Controls

- *Refactor no-op:* with both intervention switches off, the pipeline reproduces
  the prior numbers to **diff = 0.0** on every seed, so the arms measure the
  method and not the refactor.
- *Frozen features:* frozen ResNet-18 retains **82.3%** of end-to-end macro-F1
  (0.4113 vs 0.5000) at ~1/25 the compute. Report the ratio; disclose the 17.7%
  headroom.
- *Roster construction:* the proposed selector fills each aircraft to capacity
  before moving on; every literature selector load-balances. Handing the
  proposed **scoring** to the load-balanced **builder**, with seeds aliased so
  the builder is the only moving part, costs 0.021 macro-F1 at N=50 (p=0.0098),
  0.012 at N=100 and 0.035 at N=200. **The proposed method's builder is the
  better one, so the main table gave it a systematic advantage over every
  literature selector — and it lost anyway.** Corrected for the builder the gap
  to FedCS and Oort widens.

  This control also falsified its own design premise, which we report because it
  bears on how the main table should be read. We had expected N=200 to saturate
  capacity (120 slots, 200 clients) and so serve as a cell where the two
  builders must agree. Capacity never binds: participation holds at 22-26%, so
  the selector fills 5.5% / 10.4% / 21.7% / 40.9% of available slots at
  N=30/50/100/200. **There is no binding cell, and the confound is therefore
  active in every cell of the main table**, in the direction stated above.
- *Higher-powered replication, and a replication failure worth reporting.* One
  aggregation variant (diversity-weighted edge aggregation) showed a positive
  effect at all five fleet sizes in the first evaluation, largest at K=10
  (+0.033, p=0.037), significant at none after correction. We pre-registered a
  25-seed replication with criteria fixed in advance and no optional stopping.

  **It failed, and the cell that motivated it failed hardest.** K=10 fell to
  +0.0105 at n=25 and is **−0.0045 across the 15 fresh seeds alone**. K=20
  instead strengthened to +0.0185 (Holm-significant; +0.0230 on fresh seeds
  alone), and a client-count generalisation at N ∈ {50,100} came out positive in
  4 of 4 cells. The pre-registered bar — Holm-significant at ≥2 of 5 fleet sizes
  and positive at all 5 — is not met.

  We report this as a failure rather than promoting K=20, and we draw the
  methodological point explicitly: without the discovery/confirmation split, the
  reportable result from the first run would have been "wins at K=10, +0.033,
  p=0.037". At n=10 with an MDE of 0.041 (§6.7), that was a coin-flip dressed as
  a finding.

## 7. Discussion and limitations

- Placement geometry is not the lever it is assumed to be *for end-task
  accuracy* in this setting; it remains the lever for operational cost.
- The hierarchy does not beat flat FL at most operating points. Flat FL presumes
  surviving ground infrastructure the disaster premise excludes, so it is an
  infeasible reference rather than a competitor — but parity with it, not
  victory over it, is the honest bar, and the system does not clear it
  everywhere.
- **Limitations:** single dataset and single disaster geometry; frozen features
  (17.7% headroom quantified); n=10 for most cells with MDE stated; no
  perturbation-robustness grid in the corrected regime; the mechanism behind the
  strongest baseline is unidentified.
- **On the value of pre-registration in systems work.** Three of the four
  candidate improvements evaluated here had their success criteria fixed in
  writing before the code existed, and all three failed them. One had already
  produced a publishable-looking cell (+0.033, p=0.037) that vanished on fresh
  seeds. We would have reported it. The cost of the discipline was a few hours
  of writing; the cost of skipping it would have been a false claim in print.
- **Threats to validity we actively checked:** stale checkpoints surviving a
  physics change (now signature-keyed), derived artifacts surviving a rerun (now
  deleted before recompute), and unequal per-method coverage radii (now one
  shared channel).

## 8. Conclusion

An evaluation protocol for UAV-assisted HFL — a coherence condition on the
altitude band and radius, and a derived degeneracy gate — overturns three of our
own previously-published-quality findings on the same code and data. Under the
corrected protocol, coverage predicts downstream accuracy strongly across
placement methods and does not cause it, as shown by two pre-registered
interventions that each moved their target variable and neither of which moved
accuracy. Placement should be reported through operational metrics; claims that
a placement objective improves federated accuracy require an interventional
test, not a correlation.
