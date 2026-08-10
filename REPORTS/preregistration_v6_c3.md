# Pre-registration — C3, the coverage-independent residual

Written **2026-08-10, before any C3 code exists and before any C3 result is
seen.** Same purpose as `preregistration_v6_method.md`: fix what would count as
an answer, in advance, so that a win is evidence rather than the product of
searching until something won.

**This hypothesis was generated from data**, by inspecting residuals off a fitted
line after C1 had already failed. That is exploratory work, and exploratory work
produces hypotheses, not findings. This document exists precisely because the
hypothesis came from the data: everything below is the confirmatory test, and it
includes a screening step that can kill the idea before any FL compute is spent.

---

## 1. What is established

**C1 failed.** The capacity-capped placement objective *was* mis-specified —
replacing it lifted coverage 8-11 pp through the middle of the fleet grid,
exactly as predicted — and it bought no accuracy: significant at no K, and it
loses to `moon2022` at K=15 and K=20.

**Coverage is mostly not causal.** Across the 16 non-`flat_fl` methods at fixed
K, coverage_pct predicts macro-F1 with within-K r = 0.70 (0.493/0.803/0.838/
0.798/0.577 at K=5/10/15/20/30). C1 intervened on that variable and realised
**34%** of the slope's prediction (+0.019 vs +0.056 summed over K), with per-K
realised fractions of −0.31 / 0.88 / −0.36 / 0.56 / 2.38 — inconsistent in sign.

**`moon2022` beats `mclp_place` by more than its coverage explains**, and the
excess grows with fleet size. Residual off the per-K coverage line:

| K | 5 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|
| `moon2022` residual | +0.019 | −0.005 | +0.016 | **+0.022** | **+0.042** |
| rank among 16 methods | 6 | 11 | 5 | **2** | **1** |

At K=30 both methods sit at 94-100% coverage, so the +0.039 raw gap there cannot
be a coverage effect. **K=30 and K=20 are therefore the identification points**,
and they were chosen for that reason before any C3 arm was run.

## 2. Two candidate mechanisms

The first hypothesis — that `moon2022`'s greedy-over-residual construction wins
by placing *disjoint* discs — is **already weakened** and is recorded here so it
is not quietly revived. `f_cover_reachable` counts each device once, so C1 was
already a unique-coverage objective; if de-duplication were the mechanism, C1
would have captured it. It did not.

What survives:

**H-A — spatial dispersion.** `moon2022` places over the residual device set, so
consecutive UAVs are pushed apart by construction. The fitness optimisers are
free to cluster. Dispersion is not the same thing as de-duplicated coverage: two
placements can reach identical device sets with very different geometry, and
geometry determines *which* UAV each device is assigned to, hence shard
composition.

**H-B — the fitness weights.** `moon2022` ignores this benchmark's objective
entirely. `mclp_ls` maximises `w1·coverage − w2·energy − w3·imbalance` with
`w = 0.811 / 0.03 / 0.159` from `configs/tuned_weights.yaml`. Those weights were
fitted by Optuna on downstream `proposed_hfl` macro-F1 **in the pre-2026-08-08
regime** — 20 km radius, 20-120 m altitude band — which is void under Defect 1.
A method that pays for movement energy out of its coverage budget, under an
exchange rate calibrated in a regime that no longer exists, will under-cover
relative to a method that does not pay at all. This predicts exactly what is
observed: `moon2022` out-covers even C1 (93.4% vs 88.8% at K=20).

H-B is the more likely of the two and the more embarrassing, which is not a
reason to rank it lower.

## 3. Screening step — cheap, and able to kill this

Placement is fast; only FL training is slow. So before any FL arm runs,
recompute placements only (no training) for K ∈ {20, 30} × 10 seeds × three
configurations: `mclp_ls` at the shipped weights, `mclp_ls` at `w = (1, 0, 0)`,
and `moon2022`. Record per placement:

* reachable coverage (value and device count)
* mean coverage multiplicity over covered devices, and the fraction of covered
  devices reachable by exactly one UAV
* mean pairwise UAV separation, and mean distance to fleet centroid
* all three fitness components, evaluated identically for all three

**Falsification conditions, fixed now:**

* If `mclp_ls` at `w = (1, 0, 0)` does **not** materially close the coverage gap
  to `moon2022` (< 1/3 of it at both K), **H-B is rejected** and no weight-based
  C3 arm is run.
* If `moon2022`'s dispersion statistics are **within the spread of the fitness
  optimisers'** at both K, **H-A is rejected** and no dispersion-based C3 arm is
  run.
* If **both** are rejected, C3 is not run at all and the mechanism is reported as
  unidentified. That is a legitimate outcome and it gets written up.

## 4. The intervention, if screening survives

Whichever hypothesis survives is tested as one FL arm on the **full** fleet grid
(K ∈ {5,10,15,20,30} × 10 seeds, N=200, cap 6, R_comm 5 km, 100 rounds), with
baselines reused from `results/paper_uav_count` so the pairing stays valid.

Primary endpoint unchanged: **macro-F1, mean of the last 10 rounds per seed**,
paired Wilcoxon, Holm within fleet size. Every arm clears
`scripts/gate_collapse.py` first.

C3 is declared to have identified the mechanism only if **all** of:

1. It closes **at least half** the `moon2022` − `mclp_place` macro-F1 gap at
   **both** K=20 and K=30 (the coverage-saturated identification points): gaps
   are +0.035 and +0.039, so it must reach ≥ +0.018 and ≥ +0.020 over
   `mclp_place`.
2. It is **not negative** against `mclp_place` at any K in 5-30.
3. It does **not lose** to `flat_fl` at K=20 — significantly above, or a
   difference whose 95% CI contains zero. (Same reasoning as before: `flat_fl`
   presumes surviving ground infrastructure that the disaster premise excludes,
   so parity is the honest bar.)
4. No cell it wins in is degenerate.

Partial success — say, criterion 1 at K=30 but not K=20 — is reported as partial.
It is not rounded up into a claim.

## 5. What is NOT allowed

* **No search over `w`.** H-B is tested at exactly one alternative setting,
  `w = (1, 0, 0)`, fixed here, and chosen because it removes the trade entirely
  rather than because it performed well. Searching for a better `w` would be
  re-tuning the proposed method against baselines that were never re-tuned —
  the asymmetry already documented in `REPORTS/rigor_plan_2026-08.md` — and if
  it is ever done it must be done for every method at equal budget and
  disclosed.
* **No re-tuning of `lr` or any shared training constant.**
* **No moving the identification points.** K=20 and K=30 are named above and
  were chosen from the residual table, which predates this document.
* **No reporting a subset of K.** All five are reported.
* **No switching the primary endpoint.**
* Any tuning reads `val_*` columns only.

## 5a. Addendum, 2026-08-10 — screen outcome and the exact intervention

Written after the screen and **before any C3 arm code exists**. §4's criteria are
unchanged and are not reopened here; this only fixes the operationalisation that
§4 left as "the corresponding C3 arm".

### The screen's result

**H-B is REJECTED.** `mclp_ls` at `w = (1, 0, 0)` closes **7%** of the coverage
gap to `moon2022` at K=20 and **14%** at K=30, against the 1/3 threshold fixed
above. The Optuna weights are not why the optimisers under-cover. This was the
hypothesis ranked *more likely*, and the screen cost minutes.

**H-A SURVIVES**, on `unique_cover_frac` and `cover_multiplicity_mean`, at both
K, against both `mclp` variants, at p = 0.002 throughout:

| statistic | K | moon2022 | mclp shipped | mclp w100 |
|---|---|---|---|---|
| unique-cover fraction | 20 | 0.845 | 0.597 | 0.557 |
| unique-cover fraction | 30 | 0.652 | 0.460 | 0.384 |
| cover multiplicity | 20 | 1.157 | 1.525 | 1.570 |
| cover multiplicity | 30 | 1.398 | 1.758 | 1.906 |

`moon2022` tiles; the fitness optimisers double-cover. Its aircraft are also
slightly *closer together* (−0.8 to −1.4 km mean pairwise separation), so the
effect is not "spread out more" — it is "stop covering the same devices twice".

### A correction to §2 of this document

§2 claimed H-A was "already weakened" because `f_cover_reachable` counts each
device once, so C1 would have captured disjointness if it were the mechanism.
**That reasoning was wrong and the screen shows it.** De-duplicating the
*objective* is not the same as producing a *disjoint layout*: C1 declines to
double-*reward* overlap but never *penalises* it, so a redundant layout and a
tiling that reach the same device set score identically. There is no gradient
toward disjointness. And redundancy is not free — it consumes the slot budget,
since two aircraft over the same clients chase fewer distinct clients with the
same `K·capacity`. That also resolves why `moon2022` out-covers even C1.

### C3, named exactly

**`fl.coverage_mode: "disjoint"`.** The coverage term becomes

    f_cover = Σ_{d : m_d ≥ 1}  value_d / m_d

where `m_d` is the number of live UAVs whose radius reaches device `d`. Each
device contributes its full value when reached by one aircraft and a shared
fraction when reached by several.

Properties that make this the right operationalisation rather than one of many:

* **Parameter-free.** No penalty coefficient, so there is nothing to tune and
  §5's ban on searching is not strained. A `f_cover − λ·overlap` form was
  considered and rejected for exactly this reason.
* `f_cover ≤ f_cover_reachable`, with equality **iff** the layout is disjoint —
  so it is reachable coverage with a redundancy discount, and it reduces to C1
  on any tiling.
* Adding an aircraft over unreached devices always increases it; adding one over
  already-covered devices does not. That is the pressure C1 lacks.

Judged against §4's four criteria, unchanged, on the full fleet grid
(K ∈ {5,10,15,20,30} × 10 seeds), baselines reused from `results/paper_uav_count`.

**One arm.** A C2+C3 combination is *not* the confirmatory test and will not be
reported as one: C2 was chosen after seeing v6, so pairing it with C3 is
exploratory by construction. If C3 lands, C2+C3 may be run and must be labelled
exploratory.

## 6. What this can and cannot rescue

Even a complete success here does **not** restore the paper's original claims.
Class-aware selection is null (block 1) and the proposed method loses outright to
FedCS and Oort (block 4); no placement result changes either. The most C3 can
deliver is a mechanism for a placement effect, on top of a paper whose spine is
now the negative and methodological findings.

Stated here so that a C3 win is not later inflated into a rescue of the method.
