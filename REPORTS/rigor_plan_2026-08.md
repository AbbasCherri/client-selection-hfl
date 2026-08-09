# Rigor remediation plan — August 2026

> **Status update 2026-08-09 — two defects found that this plan did not
> anticipate, and they outrank most of what follows.**
>
> **F4 — Incoherent flight geometry.** A 20-120 m altitude band against radii up
> to 20 km puts every configured radius outside the channel's coherent interval
> (`z/tan θ_opt` = 54-324 m for that band), pinning the vertical optimum at a
> bound; at 20 km the link is 97% NLoS. Every result written before 2026-08-08
> was produced outside the regime where the channel model has physics —
> including the Tier-1 placement headline, which additionally ran on a flat
> range gate where altitude is a pure penalty. This invalidates more than F1-F3
> do, and it invalidates them *first*: re-running anything on the old geometry
> would waste the compute.
>
> **F5 — Per-UAV capacity below the learning floor.** Capacity ≤ 3 makes the
> run *unlearn* — it peaks near 0.34 macro-F1 by round 10 and decays to the
> constant-predictor floor, because each UAV trains the fusion head on a shard
> of that many geographically-adjacent (hence near-single-class) clients.
> Capacity orders the outcome; coverage does not (`results/probe_topology`).
>
> Consequences for this plan:
> * every compute phase below re-runs at K=20 / cap=6 / R_comm 5 km /
>   band 100-2000 m, PIPELINE_VERSION 5;
> * **Phase 3 (Tier-1) is no longer optional hardening** — it is a correctness
>   fix, since the flat gate made its third dimension decorative. Done in code
>   (`scripts/run_tier1_v5.sh`, queued);
> * the coverage-sweep negative result and the class-realism +0.065 both need
>   re-establishing before they can be cited, and the absolute macro-F1 level
>   drops from ~0.53 to ~0.38, so neither should be assumed to survive;
> * **"the task needs ~120 participants per round" is retired** — one
>   observation, confounded with radius, and falsified by the cap=1 cell, which
>   meets it and is the worst configuration measured. Replaced by the
>   outcome-based gate `scripts/gate_collapse.py`.
>
> The section below is preserved as written on 2026-08-01. Read it against this
> note, not on its own.

Target: a results set no Q1 reviewer can reasonably call bad-faith. Written
against code commit `9aa933ed` / results commit `e1894dc7` (the 2026-07-29 run).

Everything below is grouped so that **code changes come first and compute
comes second**, and so that the experiments that could *change the paper's
claims* run before the experiments that merely *harden* them.

---

## 0. What invalidates what

Three findings force a full re-run rather than an incremental top-up.

**F1 — Hyperparameters were selected on the reported test set.**
`hflsim/data/loader.py:530` splits each client 80/20 into train/test and
`global_test_indices` is the union of the per-client test indices. There is no
validation split anywhere in the pipeline. `scripts/tune_weights.py::_score`
maximises `last5_mean(macro_f1) − 0.5·last10_std(macro_f1)` — computed on that
same test set. So 22 hyperparameters were chosen by looking at test
performance. The FL loop itself is clean (significance consumes the
**final** round, not a best round — `analysis/significance.py:10`), so this is
a tuning leak, not a training leak, but it is the single most serious
objection on the list.

**F2 — The proposed method's knobs are tuned; the baselines' knobs are not.**
`W_LEARNING_NB/W_UTILITY_NB`, `W_EPI/W_SNR/W_DENS/W_PROX`,
`SEL_STATIC_BLEND`, `SEL_GUMBEL_SCALE`, the reputation weights and the fitness
weights all come from a 120-trial Optuna study whose objective is
`proposed_hfl` macro-F1 alone (`tune_weights.py:152`). The baselines' own
constants — `REPCAP_GAMMA=0.5`, `FAIRMAB_W_ENERGY/W_STALE=0.5/0.5`,
`OORT_ALPHA=2.0`, `DEFAULT_T_STALE_CAP=5`, PoC's `d=2·slots` — sit at
hand-picked literature defaults with zero search budget.

**F3 — Tuning ran in a different regime from evaluation.**
The search used `subsample: 0.2`, `n_rounds: 20`, `N ∈ {30,50}`,
`seeds {0,1}` (`tune_weights.py:215-216`). Evaluation uses `subsample: 1.0`,
`n_rounds: 100`, `N ∈ {30,50,100,200}`, `seeds 0-9`. Both a distribution shift
and a seed overlap.

Consequence: **all FL results are regenerated at 20 fresh seeds under a
three-way split.** Tier-1 (placement-only, no learned model) is unaffected by
F1 and can be extended rather than replaced.

---

## Phase 0 — Code changes (no compute)

**0.1 Three-way split.** Add `val_ratio` to `get_hfl_data_partitions`, splitting
the existing held-out 20% into 10% val / 10% test, and return
`global_val_indices` alongside `global_test_indices`. Route through
`federated.py::_load_data`, `selection_isolation.py`, `sweep.py`,
`stress_sweep.py`. **All** model/hyperparameter selection reads val; every
reported number reads test. Both are logged per round so the val→test gap is
itself reportable (a reviewer-pleasing diagnostic).

**0.2 Retune against val — proxy search + transfer check.**
Point `tune_weights.py::_score` at the val stream, and move the search seeds to
`{20, 21}` — **disjoint from the 0-19 evaluation seeds**.

Searching directly in the full evaluation regime (`subsample 1.0`,
`n_rounds 100`) costs ~134 h across 8 methods and is not affordable. Instead,
two stages, which together fix F3 *better* than a single full-regime search:

1. **Search** at `subsample 0.5`, `n_rounds 50`, `N ∈ {100, 200}`,
   `seeds {20, 21}` — ~0.56 core-h/trial.
2. **Transfer check**: take the **top-3 configs per method** and re-evaluate all
   three at the full regime (`subsample 1.0`, `n_rounds 100`, 3 val seeds),
   selecting the winner on val.

Stage 2 is the point: it *demonstrates* that the proxy regime ranked configs
correctly instead of assuming it. The current pipeline assumes it and never
checks — that is what F3 actually is. If the top-3 re-rank at full scale, that
is a reportable finding and the search regime moves closer to evaluation.

**0.3 Baseline constants — provenance-driven, not blanket tuning.**

Appendix C (§C.1–C.5) already records where each baseline constant came from,
and the provenance is uneven. The rule is three-way, by provenance:

| baseline | constant | provenance per Appendix C | treatment |
|---|---|---|---|
| `oort` | `OORT_ALPHA = 2.0` | "α = 2, **the paper's default**" (C.4 note 4) — a specific claim, but still *our* restatement | keep if confirmed; verify against Lai et al. |
| `oort` | `T_pref` | "left open by the source" (C.4 note 1) | our adaptation → search |
| `fedcs` | deadline `T_MAX_S` | source treats the deadline as a swept **system** parameter | Class E — sweep (see Phase 6) |
| `power_of_choice` | `d = 2×slots` | called "the standard oversample by 2× setting" (C.5 note 1); Cho et al. **sweep** `d` | sweep as the source does, report at its best |
| `rep_cap` | `γ = 0.5` | code cites "the adaptation note"; the note just states `# γ = 0.5` — **circular, unattributed** | verify against Zhao et al.; if absent, search |
| `fair_mab` | `w_e/w_s = 0.5/0.5` | stated in the C.3 pseudocode without attribution | verify against Zhu et al.; if absent, search |
| `fair_mab` | `T_stale_cap` | "a hyperparameter **we introduce**" (C.3) | our adaptation → search |

1. **Source specifies it** → use it, cite section/page. Standard practice.
2. **Source sweeps it** → sweep it as they did and report the baseline at its
   best setting. Pinning one value is the *unfaithful* choice here.
3. **Constant exists only because of our adaptation** → equal-budget search.

**Open action, cannot be resolved from this repo:** every "source default"
claim above is *our own restatement* — the repo holds no copy of the source
papers. Open Zhao et al., Zhu et al. **and Lai et al.** and check whether
`γ=0.5`, `w_e/w_s=0.5/0.5` and `α=2` actually appear. Where yes → cite section
or page and keep. Where no → **correct the code comments**, which currently
assert a source default that may not exist. This is a citation-integrity issue
independent of any fairness question, and `α=2` is included not because the
claim is weak — it is the most specific of the three — but because the same
standard has to apply to all of them.

**Caveat that survives all of the above:** a published constant was tuned by
its authors on *their* environment (vehicular networks, emergency comms) —
using it here is out-of-domain for the baseline while our constants are
in-domain. Defensible if disclosed; the unattackable version reports each
baseline **both** at its published default and at equal-budget-tuned, and
claims the win only if it survives the second.

**0.3b Shared training recipe — per-method tuning (unaffected by 0.3).**
`lr`, `uav_lr`, `logit_adjust_tau`, `server_momentum`, `ema_decay`,
`lr_decay` were tuned on `proposed_hfl` and imposed on all 13 methods. No
source paper specifies a learning rate for our model on our dataset, so there
is nothing to import — this half of F2 needs actual per-method search. Add
`--method` to `tune_weights.py`; **equal trial budget per method** (60 trials
each) — that equality is the claim and must be stated in the paper. Results
land in `configs/tuned_weights_<method>.yaml`, consumed via `extends:` (never
copied — see the stress-grid `lr` drift incident).

**0.4 New selection modes** in `client_selection.py`, all behind the existing
eligibility gate and the identical greedy assignment:

- `class_greedy` — **the decisive baseline.** Same exact `class_counts`
  histogram the proposed selector gets, consumed by a plain submodular
  class-coverage greedy: no UCB term, no reputation, no utility, no Gumbel
  perturbation, no static-priority blend. Isolates "class-awareness" from
  "our UCB pipeline".
- `ucb_pseudo` — the proposed selector with histograms built from
  **global-model pseudo-labels** on each client's own data instead of ground
  truth. Zero label disclosure. If the result survives this, the realism
  objection dissolves entirely.
- `ucb_dp` — true histograms + Laplace noise at ε ∈ {0.5, 1.0, 4.0}, clamped
  non-negative and renormalised. One-shot release (the partition is static, so
  it is a single query, not a per-round leak) — state that in the privacy
  accounting.
- ~~`ucb_stale`~~ — **dropped, proven a no-op.** `client_class_counts` is built
  once at `federated.py:1078`, *outside* the round loop, from the fixed
  `train_indices`, and the same object is passed to `select()` every round.
  The histogram never changes, so "round-0, never refreshed" is bit-identical
  to `ucb`. (This also confirms the privacy accounting: a genuine one-time
  release, not a per-round leak.)
- `ucb_noclass` — the full pipeline with `class_counts=None`, falling back to
  the pre-Tier-C deterministic priority ordering. The lower anchor.

**0.5 Placement class-awareness.** — **implemented 2026-08-07** (`configs/class_realism*.yaml`,
`scripts/run_class_realism.sh`). `fl.placement_class_aware` biases placement
coverage by `_client_class_value`, derived from the same histogram as
selection. Applied globally, so it is **not** a fairness asymmetry between
methods — but it is the oracle-realism question, and it is a *sharper* one for
placement than for selection:

> Selection only ranks clients some UAV already covers, so their histograms are
> observable by construction. Placement decides **who becomes reachable**, so
> conditioning it on per-client histograms needs histograms from clients no UAV
> covers yet — circular unless the report path is separate from the data path.

The assumption the benchmark makes is now stated explicitly at
`federated.py`'s `placement_class_aware`: the placer already receives every
client's coordinates, exactly as every placement baseline does (a demand map is
the standard input to maximal-covering placement), and a 4-bin histogram is a
few bytes over the same low-rate control channel while `R_comm` is sized for
sustained model-update throughput. That is an increment on an assumption
already made, not a new class of oracle — but it is still an assumption, hence
the ablation rather than the default.

Rather than a separate `placement_class_source` knob, placement inherits
`fl.class_source` (the two must degrade together, or a "no class information"
run would still be steering UAVs with class information — guarded by
`check_class_histograms.py`). Two things had to be fixed first:

- **`class_source: pseudo` was unreachable from `run_full_hfl`** — it called
  `build_class_info` without `model`/`cached_dataset`, so the rung raised on
  every run and the realism answer existed only in the selection-isolation
  harness. It now starts at `None` and refreshes from the current global model
  on the reselection cadence, ahead of placement so placement and selection
  within a round see the same histogram. `check_integration_hfl.py` exercises
  all four rungs end-to-end: a rung nobody can run is not a rung.
- The histogram is now a **per-method** copy, so a pseudo refresh under one
  method cannot leak into the next.

Four paired arms at N=200 (`class_realism{,_pseudo,_noplace,_none}`):
`(1 vs 2)` is the cost of removing the oracle, `(1 vs 3)` is what class-aware
*placement* is worth at all, `(3 vs 4)` is what class-aware *selection* is
worth once placement is blind. **Pre-registered reading:** the coverage sweep
already found placement does not move macro-F1 at any `R_comm`, so `(1 vs 3)`
is expected null — in which case class-aware placement buys nothing for an
information assumption and comes out of the headline claim, leaving the
placement contribution to stand on its optimality guarantee and operational
metrics, which assume no class information at all.

**OUTCOME (ran 2026-08-08, `results/class_realism_*`).** Means 0.5262 / 0.5166 /
0.5250 / 0.4538. Paired Wilcoxon vs the oracle arm, Holm-corrected:

* `(3 vs 4)` **class-awareness is worth +0.0653 macro-F1, p=0.0059**,
  rank-biserial 0.93, CI [0.034, 0.097]. It is doing real work.
* `(1 vs 2)` removing the oracle costs +0.0100, **p=0.131, not significant**.
  Pseudo retains ≈87% of the benefit with zero label disclosure. Report
  `pseudo` as the operating point and `true` as an upper bound; the realism
  objection to class-aware *selection* is answered, not argued around.
* `(1 vs 3)` +0.0035, p=0.77.

**The pre-registered reading of `(1 vs 3)` was wrong and must not be used.**
That contrast is *powerless*, not null: `proposed_hfl` already covers 99.95% of
clients at R_comm=20 km, so class-aware placement has nothing left to steer. The
error was inheriting `paper_full`'s R_comm without checking `coverage_pct`, and
it was caught only because arm 1 printed it. The valid test is the binding-radius
pair (`configs/class_realism_bind{,_noplace}.yaml`, R_comm=8 km, coverage 94.6%),
run separately and merged separately — pooling the two radii would pair 20 km
runs against 8 km runs, since `_SIG_TABLES` groups `paper_sweep_rounds` by `N`
alone and `N`=200 in both.

Independent of that, one operational finding already stands: at saturated
coverage class-aware placement burns **41% more movement energy** (1.129e8 vs
7.997e7 J) for +0.0012 macro-F1 and slightly *worse* accuracy — it chases
rare-class clients that were already inside coverage.

**0.6 Reporting.** `reporting/tables.py`: emit accuracy, macro-F1 and all four
per-class F1s side by side in every headline table; run `significance` over
`accuracy`, `macro_f1`, `f1_collapsed`, `f1_missing`, `f1_obstructed`,
`f1_survived`. The columns already exist in the parquets — this is a reporting
change, and it applies retroactively to the 2026-07-29 data too.

**0.7 e2e fixes** (`scripts/e2e_centralized.py`):
- epoch selection moves to val; `frozen_f1` and `frozen_acc` come from the
  **same** epoch (currently line 113 takes best-macro-F1 and last-epoch
  accuracy — different epochs);
- `--seeds 0..9` instead of a single `--seed 42`, with paired Wilcoxon on the
  retention ratio;
- per-arm LR selected on val from a small grid. Right now both arms run at
  `lr=1e-3` while the paper's frozen recipe is `1.775e-2` — neither arm is at
  its own optimum and the frozen arm is 17.75× off the tuned value;
- `--subsample 1.0` to match the paper regime (the docstring's usage example
  runs 0.2).

**0.8 MCLP fixes** (`problem/exact.py`):
- coverage uses 2-D ground range (`exact.py:71`) while the actual assignment
  uses 3-D slant distance (`instance.py:122`). At z≈70 m, R=500 m that gives
  MCLP a ~1% radius advantage — small, and it makes PSO's percentage
  *conservative*, but it must be matched, not explained away;
- `cap = np.max(instance.capacity)` — use the per-site capacity actually in
  force;
- report the HiGHS **MIP gap** and time-limit status per instance, not just a
  boolean `optimal`;
- add `grid_res` as a swept parameter, and report PSO% against each.

---

## Phase 1 — The decisive experiment (items A + B)

The question this answers: *does the proposed selector earn its complexity, or
is it just class-awareness plus an oracle?*

Harness: `selection_isolation` (static elbow K-means UAVs, so the selection
rule is the only variable).

- **10 arms at N ∈ {100, 200, 500} × 10 seeds**: the 7 existing +
  `class_greedy` + `ucb_pseudo` + `ucb_noclass`.
- **`ucb_dp` at ε ∈ {0.5, 1.0, 4.0}, N = 500 only.** The ε ladder is a
  sensitivity curve whose job is to show the *shape* of the degradation; it is
  reported at the single operating point where the effect is largest and
  cleanest. Running it at all three N triples its cost for no additional claim.

N ∈ {100, 200, 500} is the regime where the effect exists at all — it is null
at N=30, and we say so.

**Known limitation to state, not to fix by spending:** at 200 rounds these runs
are **not converged** — mean macro-F1 at N=500 rises 0.332 (r80) → 0.345 (r120)
→ 0.367 (r200) and is still climbing. So this is a **budget-matched**
comparison at a fixed 200-round budget, not a converged-performance
comparison. That is legitimate and common in FL, but it must be said in those
words, and the claim "the advantage grows with N" must be scoped to the fixed
budget. Running to convergence would cost several times Phase 1's budget and
is out of scope; flag it as a limitation instead.

Pre-registered reading of the outcome, written down **before** the run:

| outcome | what the paper says |
|---|---|
| `ucb` > `class_greedy` significantly | the pipeline earns its keep; keep the current framing |
| `ucb` ≈ `class_greedy` | the contribution is class-aware selection, not UCB; reframe the contribution and cite the class-aware FL selection literature as the true baseline |
| `ucb_pseudo` ≈ `ucb` | the oracle objection dissolves; make pseudo-label histograms the **default** method and report the ground-truth variant as an upper bound |
| `ucb_pseudo` ≪ `ucb` | the method depends on label disclosure; say so plainly, report the DP curve, and scope the claim to deployments where a one-time 4-bin histogram release is acceptable |

Reported on macro-F1 **and** accuracy **and** per-class F1. Finding #2 from the
audit — that the macro-F1 win does not transfer to accuracy — goes in the
paper regardless of how Phase 1 lands.

**Cost:** (10 arms × 3 N + 3 DP arms × 1 N) × 10 seeds = 330 jobs × ~1.1
core-h ≈ 363 core-h ≈ **30 h** wall on 12 vCPU.

---

## Phase 2 — paper_full re-run (items 1 + 2)

### Shortcomings of the current `paper_full` run

Beyond F1–F3, a full audit of `configs/paper_full.yaml` and the FL loop:

1. **Test-set-selected hyperparameters** (F1) — fatal, fixed by 0.1/0.2.
2. **Asymmetric tuning budget** (F2) — fixed by 0.3.
3. **Tuning/evaluation regime mismatch and seed overlap** (F3) — fixed by 0.2.
4. **Seed count.** n=10 puts the paired-Wilcoxon floor at p=0.00195; several
   Holm-corrected comparisons sit *at* that floor, i.e. the test is saturated
   and cannot distinguish "just significant" from "overwhelming". n=20 lowers
   the floor to ~1.9e-6.
5. **`R_comm = 20 km` saturates coverage.** Every client is in range of some
   UAV, so placement cannot bind — which is exactly why the coverage sweep
   came back negative. Report `paper_full` explicitly as the
   *coverage-saturated* operating point and stop implying placement drives
   accuracy there.
6. **`capacity=6 × K=20 = 120` slots** binds only at N=200 (and barely at
   N=100). At N=30/50 every eligible client is selected, so the four N values
   are not four samples of the same phenomenon — the selection rule is close
   to inert at the bottom of the grid. Either report only N≥100 as
   selection-relevant, or add N=350/500 so most of the grid is in the binding
   regime.
7. **`ema_decay=0.9` was never searched** (documented in the config header) and
   is applied to all 13 methods. Fold it into the per-method search (0.3).
8. **`target_value=0.45`** is a hand-picked floor and rounds-to-target is
   sensitive to it. Report the full curve or a rounds-to-target *curve* across
   thresholds, not one number.
9. **Placement baselines run at `max_path_loss_db=145`** in `paper_full` vs
   `95` in Tier-1. The 145 dB figure is justified in the header as
   scale-matching, but the two configs scoring the same baselines at radically
   different radii needs one explicit sentence, or a reviewer will read it as
   convenient.
10. **`centralized` trains on pooled data with the same epoch budget** as the
    federated arms get rounds — the comparison is "oracle", so this is fine,
    but the compute asymmetry should be stated.
11. **No per-class significance** — fixed by 0.6.
12. **`optimizer_seed=9876` is shared across all methods** by design (so the
    placement stream is common). Correct, but must be documented as a
    deliberate variance-reduction choice, not an oversight.

### The re-run

13 methods × N ∈ {30,50,100,200} × **15 seeds** (0–14), 100 rounds,
`subsample 1.0`, three-way split, per-method tuned configs.

**Why 15 and not 20.** The two reasons to exceed n=10 were the paired-Wilcoxon
p-floor and CI width. n=15 puts the floor at 2/2¹⁵ ≈ 6.1e-5 — orders of
magnitude below any Holm-corrected threshold, so the test is no longer
saturated — while CI width (∝1/√n) improves 18% over n=10. Going to n=20 buys
a further 13% CI narrowing for 12 more hours. 15 paired seeds is already well
above the field norm of 3–5.

**Cost:** 780 jobs × ~0.56 core-h ≈ 437 core-h ≈ **36 h**.

Shortcoming #6 (selection barely binds at N=30/50) is addressed by
**reporting** those cells as the non-binding regime, not by extending the grid.

---

## Phase 3 — Tier-1 (item 3)

Tier-1 is placement-only: no learned model, so F1 does not apply and the
existing 30-seed results stay valid. What it needs is the weight-sensitivity
sweep plus answers to its own points of contention.

### Points of contention in Tier-1

1. **The fitness weights are circular.** `w=(0.811, 0.03, 0.159)` was chosen by
   an Optuna study whose objective is downstream `proposed_hfl` macro-F1, and
   PSO is then declared the winner *on that objective*. The weights were picked
   in a way that is not independent of which method wins.
2. **`w2 = 0.03` all but deletes the movement-energy term** — one of the three
   stated objectives contributes ~3% of the scalarization. A reviewer will ask
   why energy is in the objective at all.
3. **Scalarization hides the trade-off.** Reporting a single `final_fitness`
   cannot show whether PSO wins on coverage while losing on energy.
4. **Equal-radius fix is right but under-reported.** The 95 dB / r*≈499 m
   calibration is a genuine fairness correction; it needs to be in the paper
   body, not only the config comment, with the 618 m "before" number shown.
5. **Capacity=15 × K=10 = 150 < N=250 was chosen to make the regime
   informative.** Legitimate, but it is a design choice that favours methods
   that ration capacity well. Report the saturating variant
   (`tier1_equal_radius`) alongside as the contrast, which is what it exists for.
6. **PSO ≥ GA is not established on unstructured geometry** — `real` (p=0.035,
   fails Holm) and `uniform` (p=0.096). Claim "PSO ≥ GA, strictly beats every
   non-optimiser" and never more.
7. **`prev_mode: stale`** sets the movement-penalty reference; with w2=0.03 it
   barely matters, but it interacts with #2 and should be swept alongside it.

### Work

- **Weight-sensitivity sweep:** w1 ∈ {0.6, 0.7, 0.811, 0.9} with (w2, w3)
  redistributed proportionally, plus the paper's original (0.6, 0.3, 0.1).
  Deliverable: a table showing the **method ranking is invariant** to the
  weights — which is what defuses the circularity in #1. If the ranking is
  *not* invariant, that is the honest headline and the weights become a stated
  limitation.
- **Pareto reporting** for #3: per-component table + a coverage-vs-energy
  scatter, already computable from `results/tier1_core/runs.parquet`.
- **`prev_mode` ablation** for #7, folded into the same sweep.

### Compute optimization: exact re-scoring for the non-optimizers

`runs.parquet` stores the three fitness components (`f_cover_norm`, `d_move_m`,
`l_imb`), and the normalizers are deterministic from the config
(`D_max = K·‖diag(box)‖`, `L_max = N²`). So

```
F = w1·f_cover_norm − w2·(d_move_m/D_max) − w3·(l_imb/L_max)
```

**Verified exact:** reconstructing `final_fitness` at w=(0.811, 0.03, 0.159)
from the stored columns reproduces the stored value with
`max|error| = 0.000e+00` across all 840 runs.

**Weight-independent — free re-scoring (4 methods).** `centroid`
(`heuristics.py:31-42`), `static` (`heuristics.py:84-90`), `mozaffari2016`
(`mozaffari2016.py:111`) and `alzenad2017` (`alzenad2017.py:102`) each
construct their placement geometrically and call `fitness(...)` exactly **once**
to report a score. Their positions cannot depend on the weights, so their
scores under any new weight setting are pure arithmetic on the existing
parquet, at zero compute cost.

**Weight-dependent — must re-run (3 methods).** `pso`, `ga`, **and `random`**.
`random` is `RandomPlacement(n_draws=20)` (`heuristics.py:53-70`): it draws 20
candidate placements and keeps the **best by fitness**, so its output does move
with the weights. An earlier draft of this plan listed `random` as
weight-independent — that was wrong.

**Cost:** 3 methods × 4 scenarios × 30 seeds × 4 new weight settings = 1440
runs, but `random` averages 0.1 s against PSO/GA's 83 s, so the true cost is
22.1 core-h ≈ **2 h** wall — unchanged by the correction.

(Measured, not estimated: the existing 840 Tier-1 runs consumed 5.56 core-h in
total. The earlier 8–12 h figure in this plan was wrong by 20×.)

**Validity of the re-scoring path is verified, not assumed:** reconstructing
`final_fitness` from the stored components at the *current* weights reproduces
the stored value with `max|error| = 0.000e+00` across all 840 rows — including
`mozaffari2016`/`alzenad2017`, which pass a `radii` override into `fitness()`,
confirming the identity holds for the per-UAV-radius path too.

---

## Phase 4 — MCLP (item 4)

### Points of contention

1. **3 seeds** (`reproduce_paper.sh:53`, `--n-seeds 3`) → 12 instances. No CI.
2. **`grid_res=20`** over a 5000 m box = 263 m spacing against R_comm=500 m.
   PSO places off-grid, which is why some cells exceed 100%. The claim
   "99.3% of optimum" is therefore "99.3% of a grid-restricted lower bound".
3. **2-D vs 3-D distance mismatch** (0.8).
4. **MCLP bounds `F_cover` only**, not the scalarized `F` — it ignores the
   movement and imbalance terms entirely. With w1=0.811 that is defensible,
   but the sentence must say "coverage term" not "objective".
5. **`np.max(capacity)`** gives MCLP the most generous capacity if capacities
   are heterogeneous.
6. **No MIP gap reported** — `optimal` is a boolean and the 120 s time limit is
   silent when it binds.

### Work

Time limit 600 s; report covered value, MIP gap, solve time and PSO%. The
grid-resolution curve is the deliverable: it shows the bound *converging*, so
"99.x% of optimum" stops depending on an arbitrary grid.

**Compute optimization:** the seed count and the grid-resolution curve answer
*different* questions. The PSO% CI needs seeds; grid convergence is a property
of the instance geometry and needs resolution, not repetitions. So:

- `grid_res ∈ {20, 30}` × **30 seeds** × 4 scenarios — the reported PSO% + CI;
- `grid_res = 45` × **10 seeds** × 4 scenarios — the convergence curve;
- `grid_res = 60` only if 45 has not visibly converged.

240 + 40 = **280 solves instead of 480**. **Cost: ~5 h.**

---

## Phase 5 — e2e (item 5)

Fixes in 0.7, then 10 seeds × 2 arms at `subsample 1.0`, paired Wilcoxon on
the per-seed retention ratio with a bootstrap CI. Reported as
"frozen features retain X% (95% CI [a, b]) of end-to-end macro-F1".

**Compute optimization:** the end-to-end arm is the only image-gradient pass in
the project and dominates the cost; the frozen arm reads the existing feature
cache and is near-free. Two savings, both of which *improve* the method:

- **Early stopping on val** (patience 3) instead of the current fixed 15
  epochs. This is required by 0.7 anyway — the current code runs all 15 epochs
  and then takes the best *test* macro-F1. Proper val-based early stopping is
  both correct and roughly half the epochs.
- Reuse the cached features for the frozen arm rather than recomputing.

**Cost: ~12 h** (from ~20 h). Keep `subsample 1.0`: retention could plausibly
depend on data volume — more data gives the trainable backbone more to exploit
— so measuring it in the paper's own regime is the point of the experiment.

**Item 6 (small federated end-to-end loop) is dropped** at your direction.

---

## Phase 6 — Arbitrary constants (the T_delay question)

The right response is not "tune them" — tuning environment constants is how
you get a simulator that flatters the proposed method. It is to **classify
every constant and treat each class differently.**

### Class E — environment / simulator physics
Never tuned. Each is either *derived from a citable source or a measurement*,
or *shown not to matter* by sensitivity analysis.

`device_state.py`: `T_MAX_S=300`, `B_MIN=0.20`, `SNR_MIN_DB=3.0`,
battery init `U[0.5,1.0]`, discharge `0.02`, recharge `0.01`, SNR base
`U[5,20]`, SNR noise `σ=2.0`, compute base `U[50,250]`, compute noise `σ=30`,
compute floor `10.0`, memory-fail rate `0.10`, margin `1.96·std`, history
window `10`.
`energy.py`: `p_fly=250 W`, `p_hover=200 W`, `cruise=15 m/s`, `t_serve=60 s`,
`battery=200 kJ`.
`value.py`: `beta_schedule` `T_decay=20`.
`reputation.py`: `_VEC_EMA_NEW/_OLD=0.7/0.3`, `_STATS_ALPHA=0.1`,
`_ADAPT_EVERY=10`, `_PRIOR_STRENGTH=20`.
Config-level: `R_comm`, `K`, `capacity`, `n_rounds`, `T_sel`, `lambda_min`,
`R_min`, `target_value`.

**`T_MAX_S = 300` is the worst offender and it has a concrete consequence.**
With compute time `U[50,250] + N(0,30)`, the raw deadline excludes **0.28%** of
devices; even with the adaptive margin it excludes **8.4%**. So the deadline
essentially never binds — which means **FedCS's entire distinguishing
mechanism (deadline-constrained greedy) is inert in our simulator**, and it
degenerates to cheapest-first capacity fill. A reviewer who checks this will
say we configured a baseline into irrelevance. Fix: derive `T_MAX_S` from the
round-time budget actually implied by the FL schedule (2 local epochs on a
client shard) rather than picking it, and sweep `T_MAX_S ∈ {150, 200, 300}` so
FedCS is evaluated in a regime where its rule does something.

Method: **Morris one-at-a-time screening** at ±50% on ~12 Class-E constants,
at N=200, 3 seeds, 60 rounds, on 3 arms (`proposed_hfl`, the strongest
baseline, `random`). Deliverable: a table showing the **sign of the
proposed-vs-baseline gap is invariant** across the screened range. Any
parameter whose perturbation flips a sign gets a dedicated full sweep and goes
in the limitations section.
**Cost:** ~12 params × 2 levels × 3 arms × 3 seeds ≈ 216 jobs × 0.33 core-h ≈
**6 h**.

### Class M-proposed — the proposed method's own knobs
`W_LEARNING_NB`, `W_UTILITY_NB`, `W_EPI/W_SNR/W_DENS/W_PROX`,
`SEL_STATIC_BLEND`, `SEL_GUMBEL_SCALE`, `SEL_MIN_MARGINAL`, `UCB_C=√2`,
reputation `W_CONTRIB/W_ANOMALY/W_TEMP`, fitness `w1/w2/w3`.
Tuned — legitimately, provided Class M-baseline gets the same budget.
Note `UCB_C=√2` is the *theoretical* constant and was **not** searched; either
search it or state that the theoretical value is used deliberately.

### Class M-baseline — the baselines' own knobs
`REPCAP_GAMMA`, `FAIRMAB_W_ENERGY/W_STALE`, `T_STALE_CAP`, `OORT_ALPHA`,
PoC `d`, Oort's `T_pref`. **Not** a blanket-tuning problem — handled by the
three-way provenance rule in 0.3 (published → cite; source-swept → sweep;
our adaptation → equal-budget search). Only `OORT_ALPHA` is genuinely a
published value today.

Note that FedCS's deadline is **not** in this class: the source treats it as a
swept system parameter, so it belongs to Class E above — and Class E is where
the `T_MAX_S=300` inertness finding bites. The two issues are the same issue.

### Class T — shared training recipe
`lr`, `uav_lr`, `logit_adjust_tau`, `server_momentum`, `ema_decay`,
`lr_decay`, `batch_size`, `n_local_epochs`, `n_uav_epochs`, `server_lr`.
Tuned on `proposed_hfl` and imposed on all 13 methods. Fixed by 0.3
(per-method) — report both the shared-recipe and per-method-tuned tables so
the effect of the fix is visible rather than silently absorbed.

---

## Compute budget

| Phase | Block | Before | **After** |
|---|---|---|---|
| 0 | code changes | 0 | 0 (dev only) |
| 0.3 | baseline-constant sweeps | 6 h | **6 h** |
| 0.3b | per-method HPO: proxy search + transfer check | 35 h¹ | **25 h** |
| 1 | **selection isolation** (13 arms, DP at one N) | 39 h | **30 h** |
| 2 | paper_full, 13 × 4 N × 15 seeds | 48 h | **36 h** |
| 3 | Tier-1: re-score non-optimizers, re-run PSO/GA only | 8–12 h | **2 h** |
| 4 | MCLP: 30 seeds at 2 grids + 10 seeds for convergence | 10 h | **5 h** |
| 5 | e2e with val early stopping | 20 h | **12 h** |
| 6 | Morris screening of Class-E constants | 6 h | **6 h** |
| | **total** | ~176 h | **~122 h ≈ 5 days** |

¹ the 35 h figure was itself an error — carried over from the *old* cheap
search settings (`subsample 0.2`, 20 rounds). Searching at the full evaluation
regime, as an earlier draft of 0.2 specified, would have been ~134 h. The
proxy-search + transfer-check design in 0.2 is what brings it to 25 h.

**Where the ~52 h came from** — every saving is validity-neutral or
validity-positive, not a reduction in scope. (Phase 2b's 24 h is separate: it
was already excluded from the 176 h total and is now deleted outright.)

| saving | h | why it costs nothing |
|---|---|---|
| paper_full 20→15 seeds | 12 | p-floor 6.1e-5, CI within 13% of n=20 |
| HPO proxy + transfer check | 10 | *adds* a verification stage the current pipeline lacks entirely |
| Tier-1 exact re-scoring | 8 | proven bit-exact (`max err = 0.0`); 4 of 7 methods place geometrically, so their positions are weight-independent |
| e2e val early stopping | 8 | replaces "15 fixed epochs, best-on-test" — strictly more correct |
| DP ladder at one N | 6 | an ε sensitivity curve is a shape claim at one operating point |
| MCLP seeds ⊥ grid resolution | 5 | seeds buy CI, resolution buys convergence — different questions |
| drop `ucb_stale` | 3 | proven bit-identical to `ucb` |
| shared feature cache across harnesses | ? | the cache depends only on `(N, data.seed, subsample)`, not on `partition_seed` — so the N=100/200 caches are common to Phase 1 and Phase 2 and must not be built twice. Unquantified (the TIF-decode prefetch is serial and was never timed separately), but strictly free: point both harnesses at one cache dir. |

**Rejected as invalid** — checked and turned down rather than assumed:

- *Truncating selection-isolation below 200 rounds.* Measured: macro-F1 is
  still climbing at r200 (N=500: 0.332@r80 → 0.367@r200). Would have saved
  ~12 h and corrupted the central result.
- *Mixed seed counts per method* (10 for the placement nulls and the oracle,
  15 for the rest). Statistically fine — pair on the shared seeds — but saves
  only 2.4 h for a permanent footnote on every table. Not worth it.
- *Dropping N=50 from paper_full.* Saves 9 h but thins the scalability curve
  to three points.

**Free, no VM required, runnable now:** 0.6 (accuracy + per-class F1 tables and
significance over the existing 2026-07-29 parquets), the Tier-1 Pareto tables,
and the entire non-optimizer half of the Phase-3 weight sweep.

## VM budget split (added 2026-08-01)

VM #1 (`instance-20260715-110613`) has **~60 h of credits left**; the plan needs
~122 h. A second VM on another account takes over afterwards — **tell the user
when VM #1 runs out**, the switchover is manual.

Everything that can *change a paper claim* goes on VM #1:

| VM | blocks | h |
|---|---|---|
| **#1 (~49 h + 11 h buffer)** | Phase 1 (30) · Phase 3 (2) · Phase 4 (5) · Phase 6 (6) · 0.3 (6) | 49 |
| **#2 (~73 h)** | 0.3b per-method HPO (25) · Phase 2 paper_full (36) · Phase 5 e2e (12) | 73 |

This split is safe because Phase 1's decisive comparison (`ucb` vs
`class_greedy` vs `ucb_pseudo` vs `ucb_noclass`) is between arms that all share
one training recipe — it does **not** depend on the per-method HPO in 0.3b. The
literature baselines inside Phase 1 do inherit untuned constants, so their rows
must be re-read after VM #2's HPO lands.

## Sequencing

1. Phase 0 code + sanity checks, smoke-validated (`reproduce_paper.sh --smoke`).
2. **Phase 1 first.** It is the only block that can change what the paper
   claims; everything after it is hardening. Do not spend 130 h re-running
   `paper_full` under a framing Phase 1 might invalidate.
3. Read Phase 1 against the pre-registered table above, settle the framing.
4. Phases 2–6 in one VM run, checkpointed, self-stopping.
5. Provenance rows + memory updates; retire the 2026-07-29 rows as superseded
   for everything Phase 2 regenerates (Tier-1 rows survive).

## Risks

- **Phase 1 may cost the headline claim.** That is the intended risk. The
  fallback framing — "class-aware selection improves minority-class triage
  performance at no accuracy cost, under a one-time histogram disclosure" — is
  publishable and honest, but it is narrower than the current draft.
- **Per-method tuning will shrink the margins.** Expect some of the 71/84 and
  the six Holm-significant selection wins to fall away. Report the before/after.
- **~5 days of VM time** on `instance-20260715-110613`. If that must shrink
  further, the honest order to cut is Phase 5 (e2e, 12 h — the frozen-feature
  justification is a secondary claim) then Phase 6 (Morris, 6 h — replaceable
  by a narrower sweep of `T_MAX_S` alone, which is the constant that actually
  bites). Phases 1–3 are load-bearing and should not be cut.
- **Phase 1's runs are not converged at 200 rounds** (measured above). The
  comparison is budget-matched, and the paper must say so. A reviewer who
  wants converged numbers is asking for several times this compute budget.
