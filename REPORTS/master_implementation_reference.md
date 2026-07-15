# Master Implementation Reference — Why & How

**Purpose.** Single consolidated reference for writing the research paper:
every mechanism in the system, the design rationale behind it, the exact
constants, and the statements the paper must make explicitly. Sections
1–18 are the map and the "why"; Appendices A–E hold the line-level
algorithm walkthroughs, literature-baseline pseudocode/citations, and
data-pipeline internals that used to live in separate companion documents.
Which config produced which result file is tracked separately in
`results_provenance.md`.

Updated 2026-07-15. State of the codebase: real-data-only experimental
pipeline; correctness is verified by manually run sanity-check scripts
(consolidated under `tests/sanity_checks/`; run `python
tests/sanity_checks/run_all.py` before trusting a batch of results) — there
is no CI or lint
gate.

---

## 1. Research problem and system overview

**Problem.** Post-earthquake building-damage classification where ground
IoT devices hold multi-modal, geographically partitioned data that cannot
be centralized (bandwidth, privacy, infrastructure loss). UAVs act as
mobile aggregators forming a **two-tier hierarchical federated learning
(HFL)** topology: IoT → UAV → server.

**The two jointly studied decisions:**

1. **Where to place K UAVs each reconfiguration interval** — a 3-D
   placement problem over device positions, solved by metaheuristics
   (PSO/GA) and compared against published placement literature.
2. **Which covered clients to select each round** — an eligibility-gated,
   utility/reputation-driven UCB bandit, compared against published
   selection literature.

**Why hierarchical:** the communication model (§5) shows the two-tier
payload split (IoT devices train/ship only the small structured+fusion
sub-model; UAVs train/ship the image branch) — the source of the
communication-efficiency claim, made auditable by identical accounting
across all methods.

**Why one shared evaluation pipeline:** every claim of superiority is a
*controlled comparison*. All placement methods score through one shared
fitness/assignment path; all selection methods run inside the
otherwise-identical FL pipeline; instance randomness is method-independent
(§11). The paper's comparison-fairness argument is encoded in invariant
checks that are run manually before results are trusted, not prose.

---

## 2. Data: real-data-only policy

**Policy (state in the paper):** every config, sweep, and reported number
derives from the real dataset. The library contains **no synthetic data
generation**; the offline sanity checks inject a deterministic fixture
through a documented seam (`data.source: "prebuilt"`) that never touches
results, so they run without an HF token.
`data.source` defaults to `"real"`; a config requesting `"synthetic"`
fails loudly (`fl/federated.py::_load_data`).

**Dataset.** HuggingFace `AbbasABC/HFL-Dataset`, pinned revision
`6cf97c900445e080e61cb45e1aa72515d3ff1de8` (env-overridable via
`HF_DATASET_REVISION`; identical pin in every runner script). Per-building
fusion of three sources for the **2024 Noto Peninsula earthquake**:

| Modality | Content | Role |
|---|---|---|
| Damage survey | raw codes {0 Survived, 1 Collapsed, 9 Obstructed view, 99 Missing/inconsistent} remapped to contiguous {0..3} | 4-class label |
| USGS ShakeMap | MMI (original+shape), PGA, PGV, SA 0.3/1.0/3.0 s + lat/lon = 9 structured features | structured branch input |
| GSI aerial imagery | on-demand 128×128 chips from zoom-18 XYZ tiles (~0.6 m/px), disk-cached under `data/tile_cache/` (`HFL_TILE_CACHE`) | image branch input |

**Subsample fractions (per config):** `configs/paper_full.yaml` uses
`subsample: 1.0` (~128k rows); the sweep/selection/stress configs use 0.1
(~12.8k rows); `tier2_fl.yaml` uses 0.05. Subsampling is deterministic
(`df.sample(..., random_state=seed)`) and applied after streaming.

**Preprocessing:** lat/lon min-max normalized to [0,1] (raw degrees kept
separately for tile lookups — z-scored coordinates would request
impossible tiles); the 7 seismic features are z-scored (StandardScaler).
Label remap note: an earlier version silently dropped classes 9/99 (~16%
of rows) — fixed; the paper reports 4 classes.

**Partitioning & split.** Clients = K-means geographic clusters of
buildings (N configured per experiment). Per client, an 80/20 train/test
split (`train_ratio=0.8`); every client's test indices are pooled into one
**global held-out test set**, and *every reported accuracy/F1 is computed
on it, every round* — no training-set numbers anywhere.

**Black-chip diagnostic (report in the paper).** A GSI fetch failure
yields an all-black chip carrying zero image signal; a high rate collapses
the model to the majority class while non-accuracy metrics keep moving.
The measured rate is logged and persisted per run under
`_diagnostics.black_chip_rate` in the resolved-config YAML — quote it next
to the accuracy tables as evidence the image modality was informative.

**Licensing / availability (data availability statement).** GSI aerial
tiles are used under the **Government of Japan Standard Terms of Use
v2.0**; cached tiles are **never redistributed** with this repository
(data/ is gitignored) — they are re-fetched on demand from the public GSI
service. Any use of the HF dataset must respect the upstream terms of the
fused sources; the pinned revision hash above makes the exact evaluated
snapshot citable and re-fetchable. `HF_TOKEN` is needed on first run for
streaming; the metadata/partition/tile caches under `data/` are derived
artifacts, regenerated automatically, and make later runs
offline-capable.

**Single-event limitation (mandatory paper statement).** All real-data
results derive from one seismic event. No second real, geo-located,
multi-modal damage dataset with ShakeMap-equivalent parameters was
obtainable; claims are scoped to "on this event", with the real-data
stress sweep (§14) as the controlled robustness complement.

---

## 3. Model architecture and CPU feasibility

`CachedFusionModel` (`fl/model.py`) — deliberately small, CPU-trainable:

| Module | Structure | Params | Trained by |
|---|---|---|---|
| `img_proj` | Linear(512→128)+ReLU over **cached** ResNet-18 features | 65,664 | UAVs only |
| `struct_branch` | 9→64→128→embedding MLP (ReLU, Dropout 0.2) | 17,216 | IoT + UAV |
| `fusion` | concat(128+128)→256→4 head (Dropout 0.3) | 50,436 | IoT + UAV |

**Why a feature cache:** ResNet-18 runs **once** per dataset
(`fl/features.py::compute_feature_cache` → `img_features.npy`, shared
across sweep workers via `data.feature_cache_path`); FL training never
touches the backbone. This is the load-bearing mechanism behind the
CPU-feasibility claim — back it with measured wall-clock (§12), not
architecture prose.

**Freeze discipline (paper §IV-A/§IV-B):** the global model freezes
`img_proj`; IoT clients clone it and train only struct+fusion (their
payload). Each UAV unfreezes `img_proj` on its own clone and trains the
full model on its cluster's pooled shard — the image branch learns at the
UAV tier, where compute exists. Class imbalance is handled at the loader
level with inverse-frequency `WeightedRandomSampler`.

---

## 4. Device heterogeneity model

`fl/device_state.py` — per-round IoT state, all constants from paper
Table II:

- battery ∈ [0,1]: init U(0.5,1.0); −0.02/round when selected, +0.005
  passive recharge; **eligibility gate ≥ 0.20**
- SNR: base U(5,20) dB + per-round N(0,2) noise; **gate ≥ 3 dB**
- memory: 10% of devices permanently memory-constrained; **gate =
  memory_ok**
- compute time: base U(50,250) s + N(0,30) noise; **gate T̂ ≤ 300 s −
  ε_n**, with adaptive margin ε_n = 1.96·std of the last 10 observed
  completion times (needs ≥3 observations) — the straggler-safety margin
  of §IV-C1.

**Stress knobs (default 0.0 = exact baseline behaviour):**
`dropout_rate` — per-(device, round) probability of forcing
`memory_ok=False`, i.e. transient loss modeled through the *existing*
four-condition gate rather than a fifth condition (why: no new eligibility
semantics to justify); `snr_degradation_db` — uniform dB subtraction, an
area-wide aftershock channel effect, not per-device.

---

## 5. Communication and energy models

**Communication (per round, uplink+downlink on both tiers):**

```
IoT payload  = struct+fusion            =  67,652 params ≈ 0.271 MB (float32)
UAV payload  = img_proj+struct+fusion   = 133,316 params ≈ 0.533 MB
hierarchical: comm = 2·n_selected·0.271 + 2·n_active_uavs·0.533   MB
flat_fl:      comm = 2·n_selected·0.271                            MB
```

Identical accounting for every method (`comm_mb_round` column); the exact
constants and the two-tier formula are pinned by manually run sanity
checks so the efficiency claim is auditable.

**Energy (`problem/energy.py`, reporting-only by design):**
`E(d) = P_fly·(d/v) + P_hover·t_serve` with P_fly=250 W, P_hover=200 W,
v=15 m/s, t_serve=60 s, battery 200 kJ. Why reporting-only: optimizers
minimize the *normalized movement term* inside the fitness; the energy
model translates that into Joules/battery-fraction for interpretability
without ever feeding back into the objective (no hidden coupling).

---

## 6. UAV placement: problem formulation

`problem/instance.py` — all coordinates in a projected metric frame
(equirectangular about a reference; matches Haversine to <0.1% at study
scale). A `ProblemInstance` holds device coords (z=0), per-device value
V_i, per-position capacity and battery, previous UAV positions, search
bounds, and the range gate `R_comm`.

**Fitness (`problem/fitness.py`) — the single scoring entry point:**

```
F(X) = 0.6·(F_cover/F_max) − 0.3·(D_move/D_max) − 0.1·(L_imb/L_max)
F_max = Σ V_i;  D_max = K·diag(box);  L_max = N²
```

Every optimizer scores **only** through this callable (one objective, one
greedy assignment, one budget counter) — the central decoupling invariant.

**Greedy assignment (`problem/assignment.py`, Algorithm 4 style):**
devices in descending value; feasible positions are in-range, under
capacity, battery ≥ B_min; winner = lowest current load, ties by smallest
distance (lexsort). Hard constraints are implicit (no coverage credit),
keeping the fitness landscape penalty-free.

**Per-UAV radius extension (2026-07 architecture change).** The
path-loss literature baselines derive a *per-UAV* coverage radius from
each UAV's own altitude, which the scalar `R_comm` gate cannot express.
Design: `radii: (K,) | None` is a **call-time parameter** threaded
`greedy_assignment → Fitness → compute_metrics → _covered_clients`, not a
`ProblemInstance` field (the radius isn't known at instance-generation
time — it's a property of the candidate placement). Optimizers publish it
via `result.meta["radii"]`; `None` reproduces the scalar gate
bit-for-bit, so PSO/GA/heuristics are untouched (regression-pinned).

**Value model (`hflsim/shared/value.py`):**
V_i(t) = β(t)·Û_i + (1−β(t))·R_i with β(t) = max(0, 1 − t/T_decay),
T_decay=20 — placement weights coverage by utility early (no history) and
by earned reputation later. Tier-1 benchmarks pin β=1 (history-free).
Tier-1 raw feature distributions (synthesized in `generate_instance`):
SNR ~ U(0,30) dB, samples ~ U_int(20,200), reputation R_i ~ Beta(2,2)
held fixed within one instance.

**Scenario generation (`generate_instance`):** device distributions
`uniform` (uniform over the box), `clustered` (√N/2 Gaussian clusters,
σ=0.06·span), `epicenter_biased` (single Gaussian at the epicenter,
σ=0.12·span). `prev_mode` sets the movement-penalty baseline:
**`"stale"`** (default) fits the previous layout to a *shifted* epicenter
(offset N(0, 0.25·span)) — modelling the situation that triggers a
reconfiguration, so `static` is a genuine floor; **`"warm"`** puts
previous positions near current device sub-centroids (already
near-optimal) to study the conservative regime.

---

## 7. Placement optimizers

All behind one interface (`optimizers/base.py`: implement `_run`,
`optimize()` times it and records the eval count) and one registry
(`optimizers/__init__.py::REGISTRY` + `build_optimizer` — the single
construction path used by the Tier-1 runner, the FL bridge, and the legacy
bridge; budget keys P/G_max always override config for budgeted methods so
no YAML can desync the paired PSO/GA comparison).

**PSO (proposed; `optimizers/pso.py`).** Constriction-factor PSO (Clerc &
Kennedy 2002; χ from φ=c1+c2=4.10, guard φ>4 raises), lbest ring topology
(k=2), per-dimension velocity clamp (0.2·range), absorbing walls,
value-weighted k-means++ seeding, stagnation reinit (Δ<1e-4 for 20 iters →
re-scatter ρ=0.2 of swarm), turbulence (p=0.1, jitter 10 m), early stop at
95% plateau. Every design choice is a config toggle → ablations are YAML
edits. Full update equations, enhancement rationale, and the default
hyperparameter table: Appendix A.

**GA (internal head-to-head).** Real-coded, tournament (size 3), SBX
crossover (η_c=15, p=0.9), polynomial mutation (η_m=20), 2 elites — same
fitness, same P·G_max budget as PSO by construction. Operator formulas:
Appendix A.

**Heuristic floors:** `centroid` (value-weighted k-means),
`random` (best of 20 uniform draws), `static` (no repositioning — the
ablation that prices dynamic placement).

**Literature baselines (published methods, faithful path-loss radius):**

*Shared channel model* (`problem/path_loss.py`): Al-Hourani et al. 2014
probabilistic LoS — P(LoS) = 1/(1+a·exp(−b(θ−a))), mean loss = FSPL(d_3d,f)
+ P_LoS·η_LoS + (1−P_LoS)·η_NLoS; `coverage_radius(h)` by bisection
(monotonicity checked); environment presets (suburban a=4.88, b=0.43,
η=0.1/21 dB, plus urban/dense/high-rise). Radius-vs-altitude is unimodal
(rises while LoS gain dominates, falls when FSPL dominates) — verified
numerically.

- **`mozaffari2016`** (IEEE Comm. Lett. 2016): compute the
  radius-maximizing altitude (h*, r*) once, then place K equal discs by
  **greedy maximal covering discretized at device locations** (each disc
  centers on the device maximizing not-yet-covered value within r*, then
  removes covered devices). *Adaptation note for the paper:* the source
  derives the single-UAV (h*, r*); this K-disc greedy is our
  generalization (a residual-centroid rule degenerates on clustered
  layouts — centroid farther than r* from every cluster covers nothing).
- **`alzenad2017`** (IEEE WCL 2017): decoupled 2-D + altitude. Value-
  weighted k-means partition into K clusters; per cluster, required radius
  = max member distance to center; altitude = **minimum** h whose
  path-loss radius reaches it (energy-efficient choice) → genuinely
  non-uniform per-UAV radii. *Adaptation notes:* source is single-UAV; the
  2-D step uses the value-weighted centroid rather than the paper's
  smallest-enclosing-circle center.

*Both:* altitude search restricted to the intersection of configured
[h_min, h_max] and the instance z-bounds (no post-hoc clipping → altitude
and radius always mutually consistent); one-shot deterministic; scored
once through the shared fitness with `radii` applied.

**Link-budget calibration (state explicitly in the paper):**
`max_path_loss_db` is deployment-specific. Tier-1 (R_comm=500 m, z≤120 m)
uses 100 dB @ 2 GHz → radii ~200–630 m, same order as the gate the other
methods face. Full-sim Noto scale (R_comm=20 km) uses 145 dB → radii on
the 20 km order. The calibration makes the *placement rule* the compared
variable, not an arbitrarily tighter radio.

**`# VERIFY AGAINST PAPER` markers (resolve before quoting absolute
numbers):** exact (a,b,η) preset values; 3-D vs horizontal FSPL
convention; Mozaffari's closed-form optimum (grid search is the stand-in);
Alzenad's exact altitude rule and SEC-vs-centroid 2-D step; each paper's
own multi-UAV arrangement.

---

## 8. Client selection

All modes behind one `ClientSelector.select()` and the same **eligibility
gate** (§4). Modes:

**Proposed (`ucb`) — the full pipeline:**
1. Utility Û = 0.4·U_epi + 0.3·U_SNR + 0.2·U_dens + 0.1·U_prox
   (epicentre distance capped at the eligible-set 95th percentile;
   min-max SNR; 5 km-radius density proxy — no building inventory exists
   for the area; proximity 1 − d_min/R_comm).
2. Priority P = 0.35·battery + 0.30·(1 − (T̂/T_max)²) + 0.35·Ũ, with
   Ũ = β(t)·Û + (1−β)·R (same β schedule as placement — utility early,
   reputation later).
3. UCB score = P + √2·√(ln t / (N_n+1)) — the exploration bonus is the
   *anti-starvation mechanism*; the fairness claim it implies is measured
   with Jain's index (§12), not asserted.
4. Greedy per-UAV assignment (highest score first, lowest-load feasible
   UAV, distance tie-break, capacity-gated) — the (N,K) distance matrix is
   vectorized (`haversine_matrix`).

Selection runs every `T_sel` rounds (default 5) **or** on the
early-reselection trigger: eligible < λ_min·min(K·capacity, N) (λ_min=0.5)
— §IV-E6; between reselections the roster persists.

**Literature baselines (selection is the only variable; identical PSO
placement, reputation FedAvg, T_sel cadence as `proposed_hfl`):**

- **B1 `fedcs`** (Nishio & Yonetani, ICC 2019): greedy fastest-first under
  a per-round deadline; reputation/fairness-blind by design.
- **B2 `rep_cap`** (Zhao et al., Chin. J. Aeronaut. 2024): static ranking
  γ·R + (1−γ)·(1−(T̂/T_max)²), γ=0.5; no exploration — starvation is
  expected and measured.
- **B3 `fair_mab`** (Zhu et al., Sensors 2024): reward = 0.5·battery +
  0.5·min(1, staleness/T_sel); staleness cap tied to the reselection
  cadence.

*Adaptation constants (γ, w's) are documented inline in
`client_selection.py` — cite them as our instantiation choices.* Full
citations, faithful pseudocode (Algorithms B1–B3), and per-baseline
fidelity/adaptation notes: Appendix C. Stage-by-stage selection formulas
and the constants table: Appendix B.

---

## 9. Reputation and aggregation

**Reputation (`fl/reputation.py`), three components per client**
(exact formulas in Appendix B §B.3):
contribution (cosine similarity between successive EMAs of the client's
update-*delta* vector — absolute weight vectors are ≈1-cosine-similar for
everyone under shared init), anomaly (diagonal Mahalanobis distance of
the update vs a global EMA-tracked per-parameter mean/variance, divided
by √J so the fixed threshold d≤2 is dimension-independent), temporal
reliability (success rate + response-time stability; `mark_absent` on
selected-but-silent clients). Aggregate score
R_n = w·[R_contrib, R_anomaly, R_temp], initial w = (0.4, 0.3, 0.3).

**Bayesian weight adaptation:** w is a Dirichlet posterior updated every
10 rounds — prior = 20·w_init pseudo-counts, evidence accumulates the
per-round component values; posterior renormalized to the simplex.
Absence counts as evidence *against* components that scored the client
highly. (Covered by sanity checks including the round-10 adaptation
firing.)

**Aggregation (per round):**
- UAV tier: sample-weighted FedAvg within each UAV's roster.
- Server tier: **reputation-weighted FedAvg** — weight = reputation ×
  n_samples (uniform fallback if all reputations collapse); each UAV
  cluster's reputation = 10%-trimmed mean of member reputations, and
  clusters below **R_min = 0.3** are excluded that round (§IV-D poisoning
  guard).
- UAV image branches (img_proj) aggregate separately from the IoT
  struct+fusion contributions (the mixed single-formula variant was
  removed as dead code; the live loop aggregates the two payload types
  separately).

---

## 10. Experimental harnesses and the method table

| Harness (CLI) | What it isolates | Output |
|---|---|---|
| Tier-1 `run` (`tier1_core.yaml`) | placement optimizer quality on generated instances (uniform/clustered/epicenter-biased; 3 scenarios × 7 methods × 30 seeds) | `runs.parquet`, `convergence.parquet` |
| Tier-2 `run_tier2` / `smoke_tier2` | placement inside a real FL loop (coverage = participation; no selection layer) | `tier2_rounds.parquet` |
| Full sim `run_paper_sim` (`paper_full.yaml`) | the complete system + ablations + literature baselines, (N × method × seed) | `fullsim_rounds.parquet` (consolidated `paper_sweep_rounds.parquet`) |
| Selection isolation `run_selection_sim` | selection rule **only**: static elbow-K-means UAVs, identical layout/seed across modes | `selection_rounds.parquet` |
| Stress sweep `run_stress_sweep` (`stress_test.yaml`) | robustness under dropout / SNR / black-chip degradation (§14) | `stress_rounds.parquet` |

**`_METHOD_CFG`** (`fl/federated.py`) maps each full-sim method to
(placement, selection, reputation-weighted, dynamic):
`proposed_hfl` = (pso, ucb, T, T); ablations `flat_fl`, `centralized`
(oracle), `hfl_no_selection` (random), `hfl_static` (no repositioning),
`hfl_no_reputation` (uniform FedAvg); selection literature `fedcs`,
`rep_cap`, `fair_mab` (pso + own selection); placement literature
`mozaffari2016`, `alzenad2017` (own placement + ucb). Report ablations and
literature baselines in **separate tables** — they answer different
questions. `fl.placement_method` swaps the authoritative optimizer for
the proposed system's ablations but **exempts** the placement-literature
entries (their placement *is* the variable).

---

## 11. Seeding, pairing, and why paired tests are valid

**Tier-1 (`runner.py`):** two SeedSequence streams with structural
tags — instance stream `[base, 0, scenario, seed]` (method-independent →
every method sees identical instances) and optimizer stream
`[base, 1, method, scenario, seed]`. The tags fix a real hazard: numpy
SeedSequence treats trailing-zero entropy as equivalent
(`[b,s,i] ≡ [b,s,i,0]`), so without tags an optimizer stream at seed 0
collides with an instance stream whenever the two bases are set equal.
Found by, and now pinned in, the manually run invariant checks.

**FL harnesses (`fl/seeds.py`, formulas frozen & regression-pinned):**
- `tier2_seed = (opt_seed + n_clients·7919 + md5(method) mod 2³¹) mod 2³¹`
- `fullsim_method_seed = (run_seed XOR md5(method) mod 2¹⁶) mod 2³¹` —
  method identity folded exactly once (sweep callers must not pre-fold).
- `sweep_job_seed = opt_seed + seed_idx·7919 + N·31` — deliberately
  method-free: paper sweep folds method later; **selection isolation
  shares it across modes** so the selection rule is the only cross-mode
  difference; the **stress sweep keeps it knob-independent** so every
  stress cell sees the identical base problem per seed.

**Seed manifests:** `build_seed_manifest(cfg, harness)` enumerates every
resolved seed by calling these same functions (no reimplementation → no
drift) and every run CLI writes `seed_manifest.csv` *before* the run
starts. Every number in the paper traces to an exact rerunnable seed.

---

## 12. Metrics: what is reported, from where, and why

| Metric | Source | Why |
|---|---|---|
| Accuracy | held-out global test set, every round | comparability with prior damage-classification work |
| Macro-F1 | same | class imbalance (~60/20/10/10) makes accuracy alone misleading under majority collapse |
| Per-class F1 (`f1_survived…f1_missing`) | same | shows rare classes are learned, not masked by macro-F1 |
| Confusion matrix | `confusion.parquet` (long form per method×round) + heatmaps | the direct evidence for/against majority-class collapse |
| Communication MB | pinned two-tier accounting (§5) | the efficiency claim, identical rules for all methods |
| Energy (J, battery frac) | `EnergyModel`, reporting-only | translates the movement term into physical units |
| Coverage %, F_cover | shared fitness breakdown | primary placement quality |
| Movement (m) | fitness breakdown | shows coverage isn't bought by unconstrained repositioning |
| Load imbalance | fitness breakdown | per-round *assignment* balance — distinct from selection fairness; label both clearly |
| Jain's fairness index | cumulative selection counts, `fl/fairness.py` (Jain et al., DEC TR-301, 1984) | the instrument for the UCB anti-starvation claim; reported by all FL harnesses |
| `evals_to_threshold` | iteration reaching 95% of final best | convergence speed under an identical eval budget |
| `convergence_auc` | trapezoidal AUC normalized by shared G_max, flat plateau extension | trajectory summary that doesn't reward early stopping with a smaller denominator |
| Wall-clock | `wall_time_s` per optimizer run; `round_time_s` per FL round; per-method aggregate via `summarize_wall_clock` | the CPU-feasibility claim needs measured numbers |
| Black-chip rate | `_diagnostics` in resolved config | data-quality evidence behind the accuracy numbers |
| `rounds_to_target` | first round ≥ target accuracy | deployment-facing convergence measure |

Rule: **never report a metric without a traceable computation in the
codebase** (each row above names its source).

---

## 13. Statistical methodology

- **Paired tests by design:** the seed architecture (§11) guarantees every
  method sees the identical problem instance per seed, so per-seed metric
  values are paired samples. Use the **Wilcoxon signed-rank** test
  (default; no normality assumption at modest seed counts) or paired t —
  `uavbench significance` / `analysis/significance.py`. *State the
  pairing property as the justification in the paper.*
- **Multiplicity:** Holm–Bonferroni step-down over the whole family of
  (pair × group) comparisons; raw p-values reported alongside the
  corrected decisions.
- **Guard rails:** the test refuses mismatched or duplicated seed sets
  (pairing would be invalid); round tables are reduced to
  final-round-per-seed first; grouping is per scenario (Tier-1), per N
  (sweeps), or per stress cell.
- **Seed counts:** Tier-1 ships 30 seeds/cell; the full sim and stress
  sweep are configured for ≥3–5 — raise to ~10 for headline claims if the
  budget allows, and report mean ± sd or CIs, never single-run numbers.

---

## 14. Robustness stress sweep (real data)

Purpose: the controlled complement to the single-event scope — degradation
axes the recorded event cannot exhibit on demand, applied to the **real
dataset** (`configs/stress_test.yaml`, subsample 0.1 to keep the 55-cell
grid CPU-tractable):

- `dropout_rate` ∈ {0, .1, .2, .3, .4} — transient per-(device, round)
  loss through the existing eligibility gate;
- `snr_degradation_db` ∈ {0, 3, 6, 10} — area-wide aftershock channel
  degradation;
- `black_chip_rate` ∈ {0, .05, .10, .20} — **additional** unusable-imagery
  fraction: `_apply_black_chips` deterministically zeroes real cached
  image-feature rows (copy semantics — disk caches stay pristine;
  dedicated RNG stream seed+977 so nothing else shifts with the rate), on
  top of the measured natural fetch-failure rate. *State this
  additional-degradation semantics in the paper.*

Grid: one-axis-at-a-time (baseline = first value) for the paper body;
`full_grid: true` Cartesian product for an appendix. Seeds are
knob-independent (§11) → along-axis comparisons are paired. Methods:
proposed_hfl, flat_fl, hfl_no_selection, fair_mab. Sanity signature
(verified): dropout collapses the selected-set size for gated methods
while `flat_fl`'s "all" mode is untouched; black-chip degrades accuracy.

---

## 15. Reproducibility and artifacts

- **Environment:** Python 3.13.14; `requirements-lock.txt` = exact
  `pip freeze` of the results-producing machine (floor pins in
  `pyproject.toml` for installability).
- **One command:** `scripts/reproduce_paper.sh [--smoke]` chains Tier-1 →
  analyze/plot → paper sim → selection isolation → N-sweep → stress sweep
  → significance → artifact staging, logging to
  `results/reproduce_paper.log`.
- **Per-run artifacts:** resolved config YAML (prebuilt payloads elided),
  `seed_manifest.csv` (written before the run), rounds/runs parquet,
  `confusion.parquet`, figures.
- **Sanity checks (manual):** offline checks run without a token via the
  fixture seam; they are run manually before results are trusted (the
  suite is being consolidated into `tests/sanity_checks/` scripts).
  Invariant checks encode the fairness-of-comparison claims: PSO/GA
  always share P/G_max regardless of config; instance/optimizer seed
  streams are disjoint even under equal bases.
- **Hardware:** see §18 for the full machine/parallelism/runtime
  disclosure; fill its wall-clock table from the final grid run.

---

## 16. Engineering quality notes (methods-section adjacent)

- **Single scoring path invariant:** no optimizer implements its own
  objective; no selection mode bypasses the eligibility gate (except the
  deliberate `all` upper bound). One `build_optimizer`, one `_load_data`,
  one `_dump_resolved_cfg`.
- **Vectorization:** `haversine_matrix` (exact vectorized twin of the
  scalar, equivalence-pinned) replaced Python double loops in coverage
  checks and selection scoring — ~17× on `_covered_clients` at N=500,
  K=20 with byte-identical output.

---

## 17. Mandatory paper statements — checklist

1. **Single-event scope** stated as a limitation; stress sweep presented
   as the controlled robustness complement (§2, §14).
2. **Real-data-only pipeline**; test fixture never touches results (§2).
3. **Black-chip rate** quoted next to accuracy results (§2, §12).
4. **Link-budget calibration** of the placement-literature baselines
   (100 dB Tier-1 / 145 dB Noto scale) and why (§7).
5. **Adaptation notes** for Mozaffari (K-disc greedy maximal covering)
   and Alzenad (per-cluster decoupling, centroid vs SEC) — and for the
   selection baselines' instantiation constants (§7, §8).
6. **Resolve all `# VERIFY AGAINST PAPER` markers** in
   `problem/path_loss.py`, `optimizers/mozaffari2016.py`,
   `optimizers/alzenad2017.py` before quoting absolute radii (§7).
7. **Paired-test justification** via the shared-instance seed design;
   Holm correction; seed counts and manifests (§11, §13).
8. **Separate tables** for own-ablations vs literature baselines (§10).
9. **Load imbalance ≠ selection fairness** — both reported, labeled as
   assignment balance vs selection-frequency fairness (§12).
10. **Wall-clock numbers** behind the CPU-feasibility claim; hardware
    disclosure (§15, §18).
11. Data availability + GSI licensing statement (§2).

---

## 18. Hardware and runtime disclosure

### Machines

| Role | Machine | CPU | RAM | Notes |
|---|---|---|---|---|
| Development / smoke runs | Dell Latitude 5540 | Intel i7-1355U (10c/12t) | 32 GB | CPU-only; all harnesses runnable |
| Full experimental grids | GCP `n1-standard-12` | 12 vCPU | 45 GB | via `scripts/run_gcp.sh` / `scripts/run_paper_sim.sh` / `scripts/run_selection_gcp.sh` (self-terminating) |

No GPU is used anywhere: the CPU-feasibility claim is backed by measured
wall-clock numbers, not an architectural argument.

### Parallelism per harness

Each sweep worker pins `torch.set_num_threads(1)` so total active threads
= `n_workers` × 1 (no BLAS thrash). `n_workers` per checked-in config:

| Config | Harness | n_workers |
|---|---|---|
| `configs/tier1_core.yaml` | Tier-1 placement grid | 8 |
| `configs/paper_full.yaml` | full paper sim (N × method × seed) | 12 |
| `configs/selection_isolation.yaml` | selection isolation | (see config) |
| `configs/tier2_sweep.yaml` | N-scalability sweep | (see config) |
| `configs/stress_test.yaml` | stress-test sweep | 8 |

`UAVBENCH_N_WORKERS` overrides the Tier-1 worker count per machine.

### Where the timing numbers come from

Wall-clock is instrumented at two levels and persisted with every run:

- **Per optimizer run** — `wall_time_s` column in Tier-1 `runs.parquet`
  (timed around `Optimizer.optimize`).
- **Per FL round** — `round_time_s` column in every rounds table
  (`tier2_rounds.parquet`, `fullsim_rounds.parquet`,
  `selection_rounds.parquet`, `stress_rounds.parquet`).

The per-method aggregate (mean/std/total seconds) is printed by
`uavbench analyze` / `run_tier2` / `run_paper_sim` / `run_stress_sweep`
via `uavbench.reporting.summarize_wall_clock`, and should be quoted in
the paper's runtime disclosure.

### Fill in after the final grid run

Record the totals from `results/reproduce_paper.log` and the wall-clock
summaries here before submission:

| Harness | Grid size | Total wall-clock |
|---|---|---|
| Tier-1 core (`tier1_core.yaml`) | 7 methods × 3 scenarios × 30 seeds | _TBD_ |
| Paper full sim (`paper_full.yaml`) | 11 methods × 3 N × 3 seeds | _TBD (previous 9-method run: ~4–7 h on n1-standard-12)_ |
| Selection isolation | modes × N × seeds | _TBD_ |
| N-scalability sweep | methods × 6 N | _TBD_ |
| Stress-test sweep | 11 cells × 4 methods × 5 seeds | _TBD_ |

---

# Appendix A — Optimizer mechanics in detail

Line-level reference for the placement optimizers. Everything here is
read directly from `src/uavbench/optimizers/` and `src/uavbench/problem/`.

## A.1 Encoding and fitness internals

Each particle/individual is a flat vector **X ∈ ℝ³ᴷ** — K UAV positions
concatenated as (x₁,y₁,z₁,…,x_K,y_K,z_K); per-dimension bounds are the
area box tiled K times; `positions_from_vector` reshapes back to (K,3).
A nuance of the imbalance term: `L_imb` uses `n_assigned/K` (the mean
load *among served devices*), not `N/K`, so it penalizes imbalance among
served devices only. The theoretical fitness ceiling is `F ≤ w1 = 0.6`
(full coverage, zero movement, zero imbalance); PSO and GA both
early-stop at `0.95·w1 = 0.57`. `Fitness` tracks an `eval_count` so the
runner can verify every optimizer spends the identical budget — no
optimizer may implement its own scoring. Assignment cost per evaluation:
`O(N log N)` sort + `O(N·K)` sweep, with the `(N,K)` distance matrix
vectorized.

## A.2 PSO (`optimizers/pso.py`)

**Constriction factor**, derived (never hardcoded) from `c1, c2`:

```
phi = c1 + c2                      # must be > 4; raises ValueError otherwise
chi = 2 / |2 − phi − sqrt(phi² − 4·phi)|
```
Defaults `c1 = c2 = 2.05` → `phi = 4.1` → `chi ≈ 0.7298`. Deriving χ at
construction means an ablation that changes the acceleration
coefficients cannot silently break the convergence guarantee.

**Velocity update** (constriction mode, default):
```
V ← chi · (V + c1·r1⊙(pbest − X) + c2·r2⊙(nbest − X))
```
**Inertia-mode fallback** (`use_constriction=False`, ablation only):
```
w = inertia_max − (inertia_max − inertia_min) · (tau / G_max)   # 0.9 → 0.4
V ← w·V + c1·r1⊙(pbest − X) + c2·r2⊙(nbest − X)
```

**Ring topology** (default): each particle's neighborhood best is the
best `pbest` among `2·ring_k+1` particles on a ring (`ring_k=2` →
5-particle neighborhoods), fully vectorized; information about a good
region spreads gradually, preserving diversity on a multimodal coverage
landscape. `topology="gbest"` shares one global best instead.

**Initialization** (`seeding="value_kmeans"`, default): half the swarm
seeded by value-weighted k-means++ centers over device (x,y), jittered
N(0, jitter_m=10 m), uniform random altitude; the other half uniform.
`"plain_kmeans"` drops value weighting; `"uniform"` seeds all uniformly.
Initial velocity `0.5·U(−vmax, vmax)`. Rationale: a good placement
almost certainly hovers UAVs over high-value clusters, so the swarm
starts with diverse, already-plausible layouts.

**Safeguards:** per-dimension velocity clamp
`vmax[d] = vmax_frac·(hi[d]−lo[d])` (default 0.2); **absorbing walls**
(out-of-bound positions clamped, violating velocity component zeroed —
no bounce); **turbulence** (each iteration a random `p_turb=0.1`
fraction of particles gets a kick ~U(−0.1·vmax, 0.1·vmax) before
clamping — prevents micro-stagnation); **stagnation reinit** (gbest
improvement < `delta_stag=1e-4` for `G_stag=20` consecutive generations
→ worst `floor(rho·P)` particles replaced with fresh uniform samples,
velocities/pbests reset, gbest re-checked; incumbents untouched).
`gbest` updates only on strict improvement. Early stop once
`gbest ≥ early_stop_frac·w1`.

**Full loop:**
```
1. Bounds ← tile(lower, upper) K times;  vmax ← 0.2·(hi−lo)
2. Init: P/2 value-weighted k-means++ + P/2 uniform;  Vel ~ 0.5·U(−vmax,vmax)
3. Evaluate all; set pbest, gbest
4. For tau = 1..G_max:
   a. nbest ← ring-neighborhood best
   b. V ← chi·(V + cognitive + social); turbulence kicks; clamp to ±vmax
   c. X ← X+V; absorbing walls
   d. Evaluate; update pbest, gbest (monotonic)
   e. Stagnation counter → reinit worst 20% at G_stag=20
   f. Early stop if gbest ≥ 0.95·w1
5. Return gbest position/fitness/convergence/eval_count/chi/phi
```

**Default hyperparameters:**

| Parameter | Default | Meaning |
|---|---|---|
| `P` | 100 | swarm size |
| `G_max` | 200 | max generations |
| `c1`, `c2` | 2.05, 2.05 | cognitive / social coefficients |
| `vmax_frac` | 0.2 | per-dim velocity clamp fraction |
| `ring_k` | 2 | ring half-width (neighborhood = 5) |
| `delta_stag` | 1e-4 | stagnation improvement threshold |
| `G_stag` | 20 | stagnation window (iterations) |
| `rho` | 0.2 | fraction of worst particles reinitialized |
| `p_turb` | 0.1 | turbulence probability per particle |
| `early_stop_frac` | 0.95 | fraction of theoretical max for early exit |
| `jitter_m` | 10.0 | seeding jitter std-dev (m) |
| `inertia_max/min` | 0.9 / 0.4 | inertia range (inertia mode only) |

## A.3 GA (`optimizers/ga.py`)

Real-coded GA (SBX + polynomial mutation, Deb & Agrawal 1999),
uniform-random init (no warm start, unlike PSO), binary tournament
(`tournament_size=3`), elitism (`n_elite=2` copied straight into the
next generation), same `0.95·w1` early stop, same P/G_max budget as PSO
by construction.

**Crossover (SBX)**, probability `crossover_prob=0.9`, per-gene gate
`u ≤ 0.5`, `eta_c=15`:
```
beta = (2u)^(1/(eta_c+1))            if u ≤ 0.5
     = (1/(2(1−u)))^(1/(eta_c+1))    otherwise
child1 = 0.5·((1+beta)p1 + (1−beta)p2)
child2 = 0.5·((1−beta)p1 + (1+beta)p2)      # clipped to [lo,hi]
```
**Mutation (bounded polynomial)**, per-gene probability `1/dim`,
`eta_m=20`:
```
delta = (2u)^(1/(eta_m+1)) − 1              if u<0.5   (range [−1,0])
      = 1 − (2(1−u))^(1/(eta_m+1))          otherwise  (range [0,1])
x_new = x + delta·(x−lo)   if u<0.5   else   x + delta·(hi−x)
```
Guaranteed in-bounds without clipping (scales toward the nearer bound).

## A.4 Heuristics and clustering utilities

- **`Centroid`** — value-weighted k-means centroids (`weighted_kmeans`),
  fixed altitude `lower.z + altitude_frac·range` (default 0.5); one
  evaluation; `value_weighted=False` toggles unweighted centroids.
- **`RandomPlacement`** — best of `n_draws=20` uniform candidates.
- **`Static`** — stays at `instance.prev_positions`; one evaluation.
- **`kmeanspp_centers(rng, points, K, weights)`** — k-means++ with
  optional value-proportional sampling (first center ∝ weights,
  subsequent ∝ D²·weights; uniform fallback on zero/non-finite weight).
- **`weighted_kmeans(...)`** — Lloyd's algorithm seeded by the above,
  weighted centroid updates, ≤25 iterations, early convergence stop.

---

# Appendix B — Client selection and reputation in detail

Line-level reference for `src/uavbench/fl/client_selection.py`,
`device_state.py`, `reputation.py`, and their wiring in `federated.py`.
Corresponds to paper Algorithms 1–4 (§IV-C).

## B.1 The four-stage pipeline

```
1. Eligibility gate    : battery ≥ B_min, SNR ≥ SNR_min, memory OK, time ≤ T_max − ε
2. Priority score      : P_n = w_b·b_n + w_ℓ·(1 − (T̂_n/T_max)²) + w_U·Ũ_n
                         with Ũ_n = β(t)·Û_n + (1 − β(t))·R_n
3. UCB exploration     : UCB_n(t) = P_n(t) + C·√(ln t / (N_n(t)+1)),  C = √2
4. Greedy assignment   : sort by UCB; each client → feasible UAV (in range,
                         under capacity) with lowest current load, ties by
                         smallest distance; skipped if none feasible
```

**Utility details:** U_epi caps epicentre distance at the eligible-set
95th percentile (one outlier can't squash everyone); U_SNR is min-max
over snr_db clipped to [0,30] dB; U_dens counts other eligible clients
within 5 km (vectorized O(n²), local equirectangular frame), min-max
normalized; U_prox = clip(1 − d_min/R_comm, 0, 1), defaulting to 0.5
when no UAV coordinates are supplied. `_minmax` has a degenerate-input
guard: if all values are within 1e-10, everyone gets exactly 0.5.
Squaring T̂/T_max in the speed term makes the penalty grow superlinearly
near the deadline. Reputation defaults to 0.5 for unscored clients; `t`
is floored at 1 so ln(t) never sees 0. UCB selection counts persist in
the `ClientSelector` instance across the whole run; the greedy
assignment increments them, feeding the next round's bonus.

**`"random"` mode** (`hfl_no_selection`): eligibility gate, then per-UAV
uniform draw of `min(capacity, |bucket|)` without replacement, using the
caller-supplied rng (required for correct multi-seed sweeps; a
round-derived `round_num·7919` seed exists only for legacy callers).
Does not increment UCB counts. **`"all"` mode** (`flat_fl`,
`centralized`): returns `covered` verbatim — no filtering, scoring, or
capacity limit.

## B.2 Device state (`device_state.py`)

Init per client: battery ~ U(0.5,1.0); snr_base ~ U(5,20) dB;
memory_ok ~ Bernoulli(0.90); compute_base ~ U(50,250) s. Per round
(end-of-round, given the actually-selected set): selected −0.02 battery,
unselected +0.005 (clipped [0,1]); fresh N(0,2) dB SNR noise and
N(0,30) s compute noise for everyone; selected clients' observed compute
time (`max(10, base+noise)`) appended to a rolling 10-round history.
Adaptive margin ε_n = 1.96·std(last 10 observed times) once ≥3
observations (else 0) — a 95%-confidence-style buffer giving volatile
clients a stricter effective deadline. Eligibility = all four hard
gates: battery ≥ 0.20, snr ≥ 3.0 dB, memory_ok,
compute_time ≤ 300 − ε_n.

## B.3 Reputation formulas (`reputation.py`)

- **R_contrib** — per-client EMA of the update-delta vector,
  `Δw̄(t) = 0.7·Δw(t) + 0.3·Δw̄(t−1)`;
  `R_contrib = (1 + cos(Δw̄(t), Δw̄(t−1))) / 2`; cold start or
  near-zero-norm EMA (denom ≤ 1e-12) → 0.5.
- **R_anomaly** — global per-parameter mean/variance EMA
  (`_STATS_ALPHA=0.1`) across all clients; per update vector v (dim J):
  `z = (v − mean)/sqrt(var + eps)`, `d = sqrt(mean(z²))` (Mahalanobis/√J,
  dimension-independent); `R_anomaly = 1` if `d ≤ 2`, else
  `exp(−0.5·(d−2))`.
- **R_temp** — `0.5·success_rate + 0.5/(1+σ_RT)` with σ_RT the variance
  of the last 10 response times (0 if <2 samples); `mark_absent()`
  increments attempts without successes.
- **Bayesian adaptation** — every `_ADAPT_EVERY=10` rounds:
  `posterior = prior + evidence`, `weights = posterior/sum(posterior)`,
  prior = `_PRIOR_STRENGTH=20` × (0.4,0.3,0.3). Delivering clients add
  their component scores as evidence; absent clients add the complement
  (evidence *against* components that scored them highly).
- **`trimmed_mean`** — a UAV's own reputation is the 10%-per-tail
  trimmed mean of its selected clients' reputations (plain mean for
  small clusters where the trim floor is 0); feeds the `R_min` server
  gate.

## B.4 Round-loop wiring (`run_full_hfl`)

Per round, in order: refresh device states + reputation scores → check
`low_eligible` → placement (if due; per-device V_i(t) from live
SNR/reputation) → selection (if `reselect = (rnd−1) % T_sel == 0 or
low_eligible or not selected`) → per-UAV groups → UAV image training +
IoT structured training on selected clients only → `mark_absent` for
selected clients that produced no update → `update_batch` on delivered
deltas → UAV-level then reputation-gated server-level aggregation →
`device_mgr.update_round(selected)` → evaluate global model, log row.

## B.5 Constants at a glance

| Constant | Value | Meaning |
|---|---|---|
| `B_MIN` | 0.20 | minimum battery fraction to be eligible |
| `SNR_MIN_DB` | 3.0 dB | minimum SNR to be eligible |
| `T_MAX_S` | 300 s | nominal max local-training time |
| `W_BATTERY / W_LEARNING / W_UTILITY` | 0.35 / 0.30 / 0.35 | priority weights |
| `W_EPI, W_SNR, W_DENS, W_PROX` | 0.4 / 0.3 / 0.2 / 0.1 | utility sub-weights |
| `UCB_C` | √2 | UCB1 exploration constant |
| `T_decay` (β schedule) | 20 rounds | utility→reputation decay, shared with placement |
| Battery discharge/recharge | −0.02 / +0.005 per round | selected vs unselected |
| SNR / compute noise | N(0,2) dB / N(0,30) s | per-round fluctuation |
| Adaptive margin ε_n | 1.96·σ(last 10 compute times) | dynamic eligibility buffer |
| `T_sel` / `lambda_min` / `R_min` | 5 rounds / 0.5 / 0.3 | cadence, early-reselect trigger, aggregation gate |
| Reputation init weights | 0.4 / 0.3 / 0.3 | Dirichlet-adapted every 10 rounds |
| `_VEC_EMA_NEW/OLD`, `_STATS_ALPHA` | 0.7/0.3, 0.1 | EMAs |
| `_PRIOR_STRENGTH` | 20.0 | Dirichlet prior concentration |
| Mahalanobis threshold | d ≤ 2 → R_anomaly = 1 | dimension-independent (/√J) |
| `trimmed_mean` trim | 10% each tail | UAV cluster reputation |

---

# Appendix C — Selection-literature baselines: citations, pseudocode, fidelity notes

All three baselines are implemented in
`src/uavbench/fl/client_selection.py` as selection modes `"fedcs"`,
`"rep_cap"`, `"fair_mab"`, exposed as full-system methods in
`_METHOD_CFG`. They share the identical PSO placement, reputation
FedAvg aggregation, and T_sel cadence as `proposed_hfl`, so the
selection rule is the only experimental variable. Notation matches the
paper's Algorithms 1–4.

## C.1 Baseline B1 — FedCS

**Citation:** T. Nishio and R. Yonetani, "Client selection for federated
learning with heterogeneous resources in mobile edge," in *2019 IEEE
International Conference on Communications (ICC)*, 2019, pp. 1–7.

**Core idea (source):** a fixed per-round deadline `Tmax`; greedily add
the candidate that increases the round's projected completion time the
least, stopping when the deadline would be exceeded. No data-value or
reputation signal anywhere — selection is purely a function of estimated
time.

**Adaptation note (state explicitly in the paper):** the original
assumes a flat server↔client topology. We adapt it to the hierarchy by
(i) running the identical greedy-marginal-time rule independently per
UAV over devices already gated in by our eligibility filter (so FedCS
benefits from the same eligibility floor everyone else gets), and
(ii) using each device's T̂_n(t) — already computed for our own priority
score — as FedCS's time estimate, so both methods see the same
measurement and only the selection *rule* differs.

```
Algorithm B1: FedCS-style Greedy Deadline Selection (per UAV u)
1: S_u(t) ← ∅;  T_proj ← 0
2: candidates ← E_u(t) sorted by ascending T̂n(t)     # cheapest first
3: for each n in candidates do
4:     T_inc ← max(T̂n(t) − T_proj, 0)                # marginal time increase
5:     if T_proj + T_inc ≤ Tmax and |S_u(t)| < Cu then
6:         S_u(t) ← S_u(t) ∪ {n};  T_proj ← max(T_proj, T̂n(t))
7:     else break
8: return S_u(t)
```

**Expected contrast:** resource-efficient but value-blind — cannot
distinguish a fast device with poor seismic SNR from a fast device near
the epicenter.

## C.2 Baseline B2 — Reputation-capability selection

**Citation:** H. Zhao, L. Geng, W. Feng, and C. Zhou, "Client selection
and resource scheduling in reliable federated learning for uav-assisted
vehicular networks," *Chinese Journal of Aeronautics*, vol. 37, no. 9,
pp. 328–346, Jun. 2024.

**Core idea (source):** reputation-based mechanism integrating data
quality and computation capability to select reliable, high-performance
nodes — no exploration term, no geospatial utility, no starvation guard.

**Adaptation note:** we reuse our own `R_contrib`/`R_anomaly`/`R_temp`
sub-scores as the "data quality/reliability" half (recomputing a
different reputation estimator would conflate "different reputation
model" with "different selection rule"), and our existing
ℓ̃_n = 1 − (T̂_n/Tmax)² as the "computation capability" half. What this
baseline *omits* relative to ours: utility Û_n, the β(t) blend, and the
UCB bonus — so it systematically starves seismically valuable devices
with middling reputation/compute, exactly the failure mode the UCB term
prevents; its starvation is expected and measured (Jain's index).

```
Algorithm B2: Reputation-Capability Selection (per round t)
1: for each n in E(t):  score_n ← γ·Rn(t) + (1 − γ)·ℓ̃n(t)     # γ = 0.5
2: sort E(t) by descending score_n → L
3: return L            # feeds Algorithm 4 (greedy UAV assignment),
                       # replacing UCB_n with score_n
```

## C.3 Baseline B3 — Fairness-enhanced MAB scheduling

**Citation:** C. Zhu, Y. Shi, H. Zhao, K. Chen, T. Zhang, and C. Bao,
"A fairness-enhanced federated learning scheduling mechanism for
UAV-assisted emergency communication," *Sensors*, vol. 24, no. 5,
p. 1599, 2024.

**Core idea (source):** a multi-armed-bandit scheduler whose reward
weights model freshness (staleness since last contribution) and energy,
explicitly to enforce participation fairness in an emergency-
communication UAV setting — the same disaster context as this paper,
making it a strong head-to-head.

**Adaptation note:** their "energy" maps onto our b_n(t) (battery);
their "staleness" maps onto our selection bookkeeping (their reward
*directly rewards* being stale; our UCB *bonuses* rarely-selected
devices — structurally similar exploration pressure, different reward
shape). No seismic/geospatial utility and no reputation/trust component
at all. `T_stale_cap` (the staleness normalizer) is set to `T_sel` — a
hyperparameter we introduce to bound their reward to [0,1] for fair
comparison; state this choice explicitly. Present it as "our UCB
explores over *value*, theirs explores over *fairness/energy*".

```
Algorithm B3: Fairness/Energy MAB Selection (per round t)
1: for each n in E(t):
2:     reward_n ← w_energy·bn(t) + w_stale·min(1, staleness_n(t)/T_stale_cap)
                                                    # weights 0.5 / 0.5
3: sort E(t) by descending reward_n → L
4: return L                                         # feeds Algorithm 4
5: after the round: last_selected_n ← t for each selected n
```

## C.4 Presenting all three in the paper

- **Signal table** (alongside Table I): one row per baseline with
  columns *Selection signal(s)*, *Exploration mechanism*,
  *Reputation/trust modeled?*, *Data-value/utility modeled?* — makes the
  "why these three" argument visually obvious: FedCS (resource only),
  Zhao et al. (reputation+capability, no exploration, no utility), Zhu
  et al. (fairness/energy bandit, no reputation, no utility), Proposed
  (all four).
- **Results:** keep literature baselines in a *separate* table/figure
  from our own ablations — two tables, two questions (§10).
- **Related Work:** for each of the three citations add one sentence
  noting "we implement this as a baseline in §V", connecting related
  work to results.

---

# Appendix D — Data pipeline internals

Line-level reference for `src/hflsim/data/loader.py` and the
PSO→HFL bridge.

- **Streaming loader:** rows are streamed (not snapshot-downloaded) from
  `AbbasABC/HFL-Dataset` at the pinned revision via the `datasets`
  `IterableDataset` API, keeping only the columns needed for
  partitioning + training. The outer streaming call retries up to
  5 times with exponential backoff (30s, 60s, 120s, 240s, capped) so a
  mid-stream HF gateway failure or tree-listing 504 under concurrent
  load doesn't crash a whole GCP job. Deterministic subsampling
  (`df.sample(n=..., random_state=seed)`) is applied after streaming.
- **GSI tile fetch:** each building chip is composited from a 2×2 mosaic
  of zoom-18 XYZ tiles and cropped to 128×128 RGB centered on the
  building's (lat,lon); tiles cached under `HFL_TILE_CACHE` (default
  `./data/tile_cache`). Network fetch retries 3 times (1s, 2s backoff)
  before returning a black chip, so a flaky connection never stalls
  training; the black-chip rate diagnostic (§2) exists because the
  seismic features are near-constant across one disaster area — the
  image is the only modality with real discriminative signal, and a high
  black-chip rate silently collapses the model to majority-class
  prediction.
- **Feature scaling order:** raw lat/lon are saved *before* any
  transform (needed unscaled for tile lookups), then normalized to [0,1]
  for model input; the 7 seismic columns (`MMI_original, MMI_shape,
  PGA, PGV, SA_0_3, SA_1_0, SA_3_0`) are z-scored (StandardScaler).
- **Partitioning:** `KMeans(n_clusters=N, n_init=10)` over
  (longitude, latitude); per client, indices shuffled with a per-client
  seed (`random_seed + cid` — independent across clients, reproducible
  from one base seed) and split 80/20; a client's reported coordinate is
  the mean lat/lon of its rows (dataset mean if a cluster is empty).
- **Caches:** the streamed metadata DataFrame is written to
  `<data_dir>/.metadata_df_cache_sub<pct>_seed<seed>.parquet` via
  temp-file + atomic `os.replace` (a partial write can never corrupt the
  next run); K-means assignments + train/test index lists are pickled to
  `<data_dir>/.partition_cache/partitions_<sha256[:16]>.pkl`, keyed by a
  hash of (row_count, N, train_ratio, seed) — the N-sweep needs each N's
  partition exactly once.
- **Feature cache:** `compute_feature_cache` runs a frozen pretrained
  ResNet-18 (head replaced by Identity, ImageNet normalization) once
  over the whole dataset and saves (N,512) float16 features (~5 MB at
  N=5000); FL training never runs the backbone.
- **PSO→HFL bridge (`hflsim/placement.py`):** converts a
  `{client_id: (lat,lon)}` dict to a `ProblemInstance` in projected
  metric space, runs the requested optimizer via
  `uavbench.optimizers.REGISTRY`, and converts the result back to
  `UAVAggregator` objects at optimized lat/lon. When `prev_positions`
  isn't supplied it defaults to an even linspace spread across the
  bounding box at mid-altitude, avoiding a placement bias toward the
  projection origin. `UAVAggregator` (legacy `hflsim.simulation`) is
  used only by this bridge; the live round loop aggregates via
  `uavbench/fl/model.py` (§9).

---

# Appendix E — Reference constants tables

### Placement objective & PSO/GA

| Parameter | Value |
|---|---|
| `w1, w2, w3` (fitness weights) | 0.6 / 0.3 / 0.1 |
| Theoretical fitness ceiling | `w1 = 0.6` |
| Early-stop threshold (PSO & GA) | `0.95 × w1 = 0.57` |
| `R_comm` (placement default) | 500 m |
| `B_min_uav` | 0.2 |
| PSO `P`, `G_max` | 100, 200 |
| PSO `c1=c2` (→ `phi`, `chi`) | 2.05 (→ 4.1, ≈0.7298) |
| PSO `ring_k`, `vmax_frac` | 2, 0.2 |
| PSO stagnation `(delta, G_stag, rho)` | 1e-4, 20, 0.2 |
| PSO turbulence `p_turb` | 0.1 |
| GA `crossover_prob`, `eta_c`, `eta_m` | 0.9, 15.0, 20.0 |
| GA `tournament_size`, `n_elite` | 3, 2 |
| Energy model `p_fly/p_hover/cruise/t_serve/capacity` | 250 W / 200 W / 15 m/s / 60 s / 200,000 J |

### Model & deployment

| Parameter | Value |
|---|---|
| Image feature dim (ResNet-18) | 512 |
| Structured feature dim | 9 |
| Fusion embed dims (img/struct) | 128 / 64 |
| Damage classes | 4 (Survived/Collapsed/Obstructed/Missing) |
| IoT payload size | ≈0.271 MB (67,652 params) |
| UAV payload size | ≈0.533 MB (133,316 params) |
| GSI tile zoom / chip size | 18 (~0.6 m/px) / 128×128 |
| Raw damage code remap | `{0→0, 1→1, 9→2, 99→3}` |
| Client partitioning | K-means over (lon, lat), `n_init=10` |
| Default `train_ratio` | 0.8 |
| HF stream retry backoff | 30s, 60s, 120s, 240s (capped) |

(Client-selection and reputation constants: Appendix B §B.5.)
