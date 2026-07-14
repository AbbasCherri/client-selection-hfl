# Master Implementation Reference — Why & How

**Purpose.** Single consolidated reference for writing the research paper:
every mechanism in the system, the design rationale behind it, the exact
constants, and the statements the paper must make explicitly. Deeper
line-level walkthroughs live in the companion documents
(`full_system_implementation_details.md`, `deployment_implem_details.md`,
`client_selection_algorithm_explained.md`, `pso_algorithm_explained.md`,
`literature_baselines.md`, `data_availability.md`,
`hardware_and_runtime.md`); this file is the map and the "why".

Updated 2026-07-14. State of the codebase: 342 offline tests passing,
ruff/black clean, real-data-only experimental pipeline.

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
(§11). The paper's comparison-fairness argument is enforced by CI
invariant tests, not prose (`tests/uavbench/test_invariants.py`).

---

## 2. Data: real-data-only policy

**Policy (state in the paper):** every config, sweep, and reported number
derives from the real dataset. The library contains **no synthetic data
generation**; the offline test suite injects a deterministic fixture
through a documented seam (`data.source: "prebuilt"`,
`tests/uavbench/synthetic_fixture.py`) that never touches results.
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
| GSI aerial imagery | on-demand 128×128 chips from zoom-18 XYZ tiles (~0.6 m/px), disk-cached | image branch input |

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

**Licensing / availability:** GSI tiles under the Government of Japan
Standard Terms of Use v2.0, cached locally and **never redistributed**
(data/ is gitignored); reruns re-fetch. `HF_TOKEN` needed on first run;
metadata/partition/tile caches make later runs offline-capable. Full
statement: `data_availability.md`.

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
constants and the two-tier formula are pinned by tests
(`test_comm_cost.py`) so the efficiency claim is auditable.

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
edits.

**GA (internal head-to-head).** Real-coded, tournament (size 3), SBX
crossover (η_c=15, p=0.9), polynomial mutation (η_m=20), 2 elites — same
fitness, same P·G_max budget as PSO by construction.

**Heuristic floors:** `centroid` (value-weighted k-means),
`random` (best of 20 uniform draws), `static` (no repositioning — the
ablation that prices dynamic placement).

**Literature baselines (published methods, faithful path-loss radius):**

*Shared channel model* (`problem/path_loss.py`): Al-Hourani et al. 2014
probabilistic LoS — P(LoS) = 1/(1+a·exp(−b(θ−a))), mean loss = FSPL(d_3d,f)
+ P_LoS·η_LoS + (1−P_LoS)·η_NLoS; `coverage_radius(h)` by bisection
(monotonicity tested); environment presets (suburban a=4.88, b=0.43,
η=0.1/21 dB, plus urban/dense/high-rise). Radius-vs-altitude is unimodal
(rises while LoS gain dominates, falls when FSPL dominates) — verified
numerically and by test.

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
`client_selection.py` — cite them as our instantiation choices.*

---

## 9. Reputation and aggregation

**Reputation (`fl/reputation.py`), three components per client:**
contribution (does the update delta improve alignment), anomaly
(ℓ2-norm outlier score vs the cohort), temporal reliability (success
rate; `mark_absent` on selected-but-silent clients). Aggregate score
R_n = w·[R_contrib, R_anomaly, R_temp], initial w = (0.4, 0.3, 0.3).

**Bayesian weight adaptation:** w is a Dirichlet posterior updated every
10 rounds — prior = 20·w_init pseudo-counts, evidence accumulates the
per-round component values; posterior renormalized to the simplex.
Absence counts as evidence *against* components that scored the client
highly. (Covered by tests including the round-10 adaptation firing.)

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
Found by, and now pinned in, the CI invariant tests.

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
- **CI (`.github/workflows/ci.yml`):** pytest (342 offline tests — no
  token; the fixture seam) + ruff + black + advisory mypy. Invariant tests
  encode the fairness-of-comparison claims: PSO/GA always share P/G_max
  regardless of config; instance/optimizer seed streams are disjoint even
  under equal bases.
- **Hardware:** Dell Latitude 5540 (i7-1355U, 32 GB) for development; GCP
  n1-standard-12 for grids; workers pin `torch.set_num_threads(1)`; HF
  streaming is prefetched sequentially before any parallel phase (rate
  limits), features computed once per N. Fill the wall-clock table in
  `hardware_and_runtime.md` from the final grid run.

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
- **Dead code policy:** unused aggregation variants deleted (git history
  preserves them); the pre-`uavbench` simulator is explicitly marked
  LEGACY and excluded from lint/type gates.
- **Repo hygiene:** no data, logs, or caches tracked in git (GSI tiles
  must not be redistributed); results of the published baseline remain
  tracked.

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
    disclosure (§15).
11. Data availability + GSI licensing statement (§2;
    `data_availability.md`).
