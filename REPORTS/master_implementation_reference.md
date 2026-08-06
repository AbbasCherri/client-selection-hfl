# Master Implementation Reference — Why & How

**Purpose.** Single consolidated reference for writing the research paper:
every mechanism in the system, the design rationale behind it, the exact
constants, and the statements the paper must make explicitly.

**If you are drafting the paper, start at §0.5** — the claims ledger,
quotable numbers, notation/algorithm maps, consolidated limitations, and
the "do not write" list. §0 explains which constants deviate from the
paper text and why. §§1–21 are the mechanism and the "why"; Appendices A–E
hold the line-level algorithm walkthroughs, literature-baseline
pseudocode/citations, and data-pipeline internals. Which config produced
which result file is tracked separately in `results_provenance.md`.

Updated **2026-07-28** (previous revision 2026-07-15; ~4,150 lines of code
changed across 52 files in between). State of the codebase: real-data-only
experimental pipeline; correctness is verified by manually run
sanity-check scripts (**81 checks across 14 scripts** under
`tests/sanity_checks/`; run `python tests/sanity_checks/run_all.py` before
trusting a batch of results) — there is no CI or lint gate.

**Read §0 first.** A large number of constants no longer match the values
stated in the paper text, deliberately. §0 is the complete list and the
disclosure the paper owes the reader.

---

## 0. Deliberate deviations from the paper text (read first)

Between 2026-07-18 and 2026-07-22 the proposed system underperformed its
own literature baselines. The diagnosis was that essentially every
weight in the system had been *chosen by reasoning* (paper defaults or
engineering judgment) and never tested. `scripts/tune_weights.py` (an
Optuna study, 2026-07-20, real data at reduced scale, storage + leaderboard
under `results/hpo/`) searched them jointly against a downstream objective
and moved most of them a long way. **Those searched values are now the
committed defaults.** They must be disclosed, not quietly substituted.

**The objective the weights were fit to:**
`mean(last-5-round macro_f1) − 0.5·std(last-10-round macro_f1)` of
`proposed_hfl`, averaged over a small (N, seed) grid. The stability
penalty is deliberate — the Tier-A changes (§6) specifically targeted
round-to-round oscillation, so a config that wins by luck on one volatile
round must not outscore one that is consistently good. Net gain on that
objective: **+22%** over the prior hand-picked recipe.

| Quantity | Paper text | Committed value | Where | Why it moved |
|---|---|---|---|---|
| Fitness weights `w1,w2,w3` | 0.6 / 0.3 / 0.1 | **0.811 / 0.03 / 0.159** | `problem/fitness.py` defaults; `tier1_core.yaml` | Coverage should dominate far more, movement penalty far less. Coverage correlates **+0.78** with downstream accuracy — an uncovered client cannot participate at all, so coverage is nearly the whole story. |
| Priority weights `w_b/w_ℓ/w_U` | 0.35 / 0.30 / 0.35 | **battery removed**; 0 / **0.702** / **0.298** | `client_selection.py` `W_LEARNING_NB`, `W_UTILITY_NB` (the old `W_BATTERY`/`W_LEARNING`/`W_UTILITY` are still *defined* in the module but referenced nowhere — dead constants, do not read them as live) | Scoring by battery made UCB drain the highest-charge cohort in lockstep and then swap wholesale to the rested cohort every `T_sel` — a non-IID ping-pong that gave the proposed method the **highest round-to-round volatility of all methods** and a worse macro-F1 than random selection. Battery stays a **hard eligibility gate** (§4); it is simply no longer a score. |
| Utility sub-weights `W_EPI/W_SNR/W_DENS/W_PROX` | 0.4 / 0.3 / 0.2 / 0.1 | **0.043 / 0.078 / 0.295 / 0.584** | `client_selection.py` | The search found UAV-proximity dominant and epicentre-distance/SNR nearly irrelevant — the *opposite* emphasis from the paper's stated split. Report this as an empirical finding, not a typo. |
| Reputation prior `w_contrib/w_anomaly/w_temp` | 0.4 / 0.3 / 0.3 | **0.091 / 0.134 / 0.775** | `reputation.py` | Temporal reliability is a far stronger prior than contribution quality or anomaly detection. This is the *Dirichlet prior* only — it still adapts every 10 rounds (§9), so the search tuned the starting point, not the mechanism. |
| Client LR | 1.0e-3 | **1.775e-2** | `configs/tuned_weights.yaml` | ~18× larger; the old value never left the flat region of the loss surface within 100 rounds. |
| Logit-adjust τ | (new) | **0.601** | `configs/tuned_weights.yaml` | See §6 Tier A1. |
| Server momentum | (new) | **0.528** | `configs/tuned_weights.yaml` | See §6 Tier A3. |
| Roster blend / Gumbel | (new) | **0.435 / 1.475** | `client_selection.py` | See §8. The search pushed the Gumbel scale near the top of its searched range — more roster randomness helps more than intuition suggested. |

**One value was searched, not adopted, and then separately validated:**
`ema_decay`. The search's 20-round window structurally penalizes heavy EMA
smoothing (a decay-0.9 EMA has not caught up to the live model within 20
rounds regardless of its merit at 100-round scale), so its `ema_decay ≈
0.17` result was not trusted. `scripts/validate_ema_decay.py` re-ran it
alone at **80 rounds** with every other weight pinned at the adopted
defaults — **this has been done**
(`results/hpo/ema_decay_validation.csv`):

| `ema_decay` | 0.95 | **0.9** | 0.3 | 0.7 | 0.0 | 0.17 | 0.5 | 0.1 | 0.99 |
|---|---|---|---|---|---|---|---|---|---|
| score | .4936 | **.4913** | .4910 | .4910 | .4909 | .4909 | .4909 | .4909 | .4623 |

**Verdict: the committed 0.9 is validated and the item is closed.** Three
things to report. (i) 0.9 is within 0.0023 of the best value (0.95) — well
inside seed noise — so it is a defensible pick and not a lucky one.
(ii) The search's 0.17 scores 0.4909, i.e. *identical to no EMA at all*
(0.0), which confirms the 20-round window was simply blind to the
mechanism — the original reason for distrusting it was correct.
(iii) Everything in [0, 0.95] lies within 0.003; only **0.99 collapses**
(−0.029). So the honest claim is not "EMA evaluation helps a lot" but
"EMA evaluation is insensitive across a wide band and only fails when the
decay is so slow the average never tracks the model". Do not oversell A4.

**Single-source-of-truth convention (do not violate).** The tuned values
live in exactly one file, `configs/tuned_weights.yaml`, and every
experiment config inherits them via `extends: tuned_weights.yaml`
(`runner.py::load_config`, recursive with cycle detection and deep merge).
They are never copied. This convention exists because it was violated:
`configs/stress_test.yaml` held its own `lr: 1.0e-3` — 17.75× below the
tuned value — through the entire 2026-07-22 full run, producing 440 jobs
of undertrained, uniformly non-significant robustness results (`flat_fl`
collapsed to macro-F1 ≈ 0.07). A duplicated constant silently drifts.

**Honest framing for the paper.** The tuned weights are fit to a
*downstream* objective (proposed-system macro-F1), which couples the
placement objective to the evaluation metric. Two mitigations make this
defensible and both must be stated: (i) the searched weights are applied
**identically to every method** — Tier-1 scores PSO, GA, and all baselines
through the same `w=(0.811, 0.03, 0.159)`; (ii) the weights are strongly
identified rather than knife-edge (coverage correlation +0.78). State the
objective coupling explicitly rather than presenting the weights as
derived.

---

## 0.5 Drafting guide — claims ledger, quotable numbers, notation

§§1–21 answer *"how does it work and is it right?"*. This section answers
*"what sentence can I write, and what backs it?"* Everything below is read
from the committed result files, re-derived on 2026-07-28.

### 0.5.1 Paper skeleton → where the material lives

| Paper section | Draw from | Notes |
|---|---|---|
| Introduction | §1 | two jointly-studied decisions; frame the contribution per 0.5.3 |
| Related Work | Appendix A.5–A.8, C.1–C.6 | 7 implemented baselines, each with a "we implement this in §V" sentence |
| System Model | §2 (data), §3 (model), §4 (devices), §5 (comm/energy), §7 (placement formulation) | |
| Proposed Method | §6 (recipe), §7 (PSO), §8 (selection), §9 (reputation/aggregation), Appendices A.1–A.4, B | Algorithms map in 0.5.5 |
| Experimental Setup | §10, §11 (seeds), §13 (stats), §15 (repro), §18 (hardware) | |
| Results | 0.5.2 + 0.5.3 | |
| Limitations | 0.5.6 | |
| Reproducibility stmt | §15, §18 | |

### 0.5.2 Quotable numbers (all verified against committed files)

| Number | Value | Source |
|---|---|---|
| PSO share of MILP-optimal coverage | **99.3%** (96.4–101.1, all 12 solves proved optimal) | `tier1_core/mclp_reference.csv` |
| PSO vs GA, epicenter-biased | **+0.067**, p=3.7e-9 | `tier1_core/significance_final_fitness.csv` |
| PSO vs GA, clustered | **+0.029**, p=1.3e-7 | same |
| PSO vs GA, uniform / real | **tied** (p=0.096 / 0.035 n.s. after Holm) | same |
| Frozen-feature macro-F1 retention | **80.3%** (0.410 vs 0.511) | `e2e_centralized/e2e_comparison.csv` |
| proposed_hfl macro-F1 (N=30→200) | 0.582 / 0.581 / 0.560 / 0.533 | `paper_full/paper_sweep_rounds.parquet` |
| centralized oracle macro-F1 | 0.612 / 0.614 / 0.610 / 0.609 | same |
| flat_fl macro-F1 | 0.504 / 0.427 / 0.402 / 0.344 | same |
| proposed vs flat_fl | **+0.079 → +0.190**, significant at every N | `paper_full/significance_macro_f1.csv` |
| proposed vs oort | +0.144 / +0.115 / +0.053 sig; **n.s. at N=200** | same |
| Total measured compute | **≈657 core-hours**, no GPU | §18 |
| Test-set majority class | ~0.77–0.79 accuracy floor | accuracy column vs macro-F1 |

**Always report macro-F1, not accuracy.** Accuracy spans only 0.764–0.786
across all 13 methods (§0.5.2) — it cannot separate them, because the
majority class dominates. Macro-F1 spans 0.344–0.612 on the same runs.

### 0.5.3 Claims ledger — what the evidence supports

Status: 🟢 supported · 🟡 tied (no significant difference) · 🔴 contradicted
· ⏳ pending a run · ⛔ void (§19).

| Candidate claim | Status | Evidence |
|---|---|---|
| PSO places near-optimally | 🟢 | 99.3% of MILP optimum, all scenarios |
| PSO beats non-searching placement (mozaffari, alzenad, centroid, static, random) | 🟢 | significant in **all 4** Tier-1 scenarios |
| PSO beats GA | 🟡 | wins clustered + epicenter; **ties uniform and real** |
| Proposed selection beats the selection literature (fedcs, rep_cap, fair_mab, oort, PoC) | 🟢 at N≤100 | +0.047…+0.144, all significant |
| …also at N=200 | 🟡 | only `power_of_choice` stays significant after Holm |
| Hierarchy beats flat FL | 🟢 | +0.079→+0.190, significant at every N, and 11/11 stress cells |
| Proposed beats the *placement* literature end-to-end | 🟡 | **tied with alzenad2017 and mozaffari2016 at every N**, and in all 11 stress cells |
| Dynamic repositioning helps (vs `hfl_static`) | 🟡 | **tied at every N** (p=0.11–0.77); also tied across the coverage sweep |
| Reputation weighting helps (vs `hfl_no_reputation`) | 🟡 | **tied at every N** (p=0.56–0.77) |
| Proposed degrades more gracefully than placement baselines | 🟡 | **0 significant wins over alzenad/mozaffari in 22 stress cells**; also 0 losses |
| Proposed never degrades worse than any comparator | 🟢 | **0 significant losses** in either stress grid |
| Placement quality matters as coverage binds | 🟡 | holds strongly vs `mozaffari2016` (5/6 radii) and `flat_fl`; **not** vs `alzenad2017` (1/6) or `hfl_static` (0/6) |
| Selection scales to large N | 🔴 (starved) / ⏳ (matched) | starved: `ucb` falls below `random` at N=500; matched run in flight (§10.1) |
| UCB prevents starvation | 🟡 | Jain is reported, but at N=500 starved `ucb` had the **highest** Jain and near-worst macro-F1 — uniform participation was harmful |

**Honest contribution framing.** The two claims the evidence carries
cleanly are **(a) the placement optimizer is near-optimal and beats every
non-searching placement rule** (Tier-1 + MCLP), and **(b) the selection
mechanism beats five published selection rules inside an identical
pipeline** (paper_full at N≤100, selection isolation). The end-to-end
comparison against *placement* literature is a tie, and two ablations
(repositioning cadence, reputation weighting) show no significant
macro-F1 effect in the FL harness. Write the paper around (a) and (b);
report the ties as ties. Claiming end-to-end superiority over
alzenad2017 is not supportable from this data.

**Why the ablation ties are defensible rather than damaging:** at
`R_comm = 20 km` every client is covered (coverage_pct = 100 for all
methods), so placement *cannot* matter at that operating point — which is
exactly what the coverage sweep was built to expose. Report the ablations
against the coverage-constrained grid, not against `paper_full`, and say
why. Note also the crossover: at `R_comm = 2 km` coverage falls to ~27%
and `flat_fl` (0.344) edges every UAV method — a real operating boundary
worth reporting rather than hiding.

### 0.5.4 Notation map (paper symbol ↔ code)

| Symbol | Meaning | Code |
|---|---|---|
| N, K | clients, UAVs | `data.N_clients`, `fl.K` |
| C_u | per-UAV capacity | `fl.capacity` |
| R_comm | coverage radius gate | `fl.R_comm` / `instance.R_comm` |
| V_i(t) | device value for placement | `hflsim/shared/value.py::compute_value` |
| β(t) | utility→reputation schedule | `beta_schedule`, `T_decay=20` |
| Û_n | geospatial utility | `_utility` (§8 step 1) |
| P_n | priority score | §8 step 2 |
| R_n | reputation | `reputation.py` |
| T̂_n, T_max, ε_n | compute estimate, deadline, adaptive margin | `device_state.py` |
| T_sel | placement cadence | `fl.T_sel` |
| λ_min, R_min | early-reselect trigger, aggregation gate | `fl.lambda_min`, `fl.R_min` |
| F(X), w1..w3 | placement fitness + weights | `problem/fitness.py` |
| F_cover, D_move, L_imb | fitness components | `AssignmentResult` / `components()` |

### 0.5.5 Algorithm map

| Paper Algorithm | Implementation |
|---|---|
| 1–3 (utility, priority, UCB) | `client_selection.py::_ucb_select` (§8 steps 1–3) |
| 4 (greedy UAV assignment) | `_greedy_assign` / `_class_coverage_assign` (§8 step 4) |
| 5 (placement search) | `optimizers/pso.py` (Appendix A.2) |
| 6 (device→UAV assignment) | `problem/assignment.py::greedy_assignment` |
| B1–B5 (baselines) | Appendix C.1–C.5 |
| A6–A7 (placement baselines) | Appendix A.6–A.7 |

### 0.5.6 Limitations — consolidated (write these, don't wait to be asked)

1. **Single seismic event** (§2) — stress sweeps are the controlled
   complement, not a second dataset.
2. **Tuned constants deviate from the stated design** (§0), fit to a
   downstream objective, applied identically to all methods.
3. **Frozen features cost ~20% macro-F1** (§3) — a feasibility trade-off
   that caps every federated number reported.
4. **Placement ablations are inconclusive at the main operating point**
   (0.5.3) because coverage saturates at 20 km.
5. **Selection degrades under data starvation** (§2) — reportable
   operating envelope with an identified mechanism (4-bin histograms go
   uninformative below ~30 samples/client).
6. **Communication cost is billed at a conservative upper bound** (§5).
7. **Simulated device heterogeneity** — battery/SNR/compute are modelled
   (§4), not measured from hardware.

### 0.5.7 Do not write

- Any number from §19's void list.
- "+0.041 vs Oort at N=500" or any matched-data isolation figure — no
  committed artifact yet (§10.1).
- Results for `centroid_place` / `random_place` — not yet run.
- "PSO beats GA" without the uniform/real ties.
- "Proposed beats alzenad2017" — it is a tie.
- Any accuracy-based ranking (0.5.2).
- The stress `significance_*.csv` files as they stand: they are **stale
  all-pairs tables** with no `reference`/`correction_scope` columns,
  generated before reference mode existed. The in-flight run regenerates
  them with `--reference proposed_hfl --correction-scope group`, which
  shrinks the family from 308 to 7 per cell and will change which cells
  clear Holm. Re-read them before quoting.

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
   (PSO/GA), compared against published placement literature and against
   a **MILP coverage optimum** (§7, Appendix A.9).
2. **Which covered clients to select each round** — an eligibility-gated,
   utility/reputation-driven UCB bandit with a submodular class-coverage
   roster, compared against **five** published selection methods.

**Why hierarchical:** the communication model (§5) shows the two-tier
payload split — the source of the communication-efficiency claim, made
auditable by identical accounting across all methods.

**Why one shared evaluation pipeline:** every claim of superiority is a
*controlled comparison*. All placement methods score through one shared
fitness/assignment path; all selection methods run inside the
otherwise-identical FL pipeline; instance randomness is method-independent
(§11). The comparison-fairness argument is encoded in invariant checks run
manually before results are trusted, plus runtime asserts at the
enforcement points, not prose.

---

## 2. Data: real-data-only policy

**Policy (state in the paper):** every config, sweep, and reported number
derives from the real dataset. The library contains **no synthetic data
generation**; the offline sanity checks inject a deterministic fixture
through a documented seam (`data.source: "prebuilt"`) that never touches
results, so they run without an HF token. `data.source` defaults to
`"real"`; a config requesting `"synthetic"` fails loudly
(`fl/federated.py::_load_data`).

**Dataset.** HuggingFace `AbbasABC/HFL-Dataset`, pinned revision
`6cf97c900445e080e61cb45e1aa72515d3ff1de8` (env-overridable via
`HF_DATASET_REVISION`; identical pin in every runner script). Per-building
fusion of three sources for the **2024 Noto Peninsula earthquake**:

| Modality | Content | Role |
|---|---|---|
| Damage survey | raw codes {0 Survived, 1 Collapsed, 9 Obstructed view, 99 Missing/inconsistent} remapped to contiguous {0..3} | 4-class label |
| USGS ShakeMap | MMI (original+shape), PGA, PGV, SA 0.3/1.0/3.0 s + lat/lon = 9 structured features | structured branch input |
| GSI aerial imagery | on-demand 128×128 chips from zoom-18 XYZ tiles (~0.6 m/px), disk-cached under `data/tile_cache/` (`HFL_TILE_CACHE`) | image branch input |

**Subsample fractions (per config).** `paper_full.yaml`, `paper_coverage.yaml`
and `selection_isolation.yaml` use `subsample: 1.0` (~128k rows);
`paper_smoke.yaml` uses 0.2; the stress configs use 0.1; `tier2_fl.yaml`
uses 0.05. Subsampling is deterministic (`df.sample(..., random_state=seed)`)
and applied after streaming.

**Selection isolation moved from 0.1 to 1.0 — a substantive correction,
not a cost decision.** At `subsample: 0.1` the training pool was held
fixed (~10.3k rows) while N grew, so samples-per-client collapsed:
N=30→344, N=50→206, N=100→103, N=200→52, N=350→29, **N=500→21**. That is
not "more clients", it is the same small pool sliced 500 ways. Under it
the proposed selector *lost* significantly to Oort/Power-of-Choice at
N=350/500 and fell below random. The mechanism is specific and worth
reporting: the proposed rule's class-coverage utility is built on
per-client 4-bin label histograms, and at ~21 samples with an ~81%
majority class most clients hold **zero** rare-class samples, so the
scarcity-weighted coverage term chases noise — while loss-based selectors
still carry signal. Selection pressure does **not** explain it:
`paper_full` N=200 and isolation N=500 have the same servable fraction
(ρ = 0.60) and opposite outcomes. At `subsample: 1.0` (N=500 → 206
samples/client, the density of the old N=50 cell) the proposed selector
wins every large-N cell significantly (+0.041 vs Oort, +0.030 vs PoC at
N=500). A real deployment with more clients has more total data, not
thinner slices, so 1.0 is the faithful setting. **The starved regime is
preserved and must be reported** as a sensitivity/limitation analysis:
`configs/selection_isolation_starved.yaml` (full grid at 0.1) and
`configs/selection_scaling{,_full}.yaml` (the N=350/500 cells only, starved
vs matched, as the decisive paired control).

**Preprocessing:** lat/lon min-max normalized to [0,1] (raw degrees kept
separately for tile lookups); the 7 seismic features are z-scored
(StandardScaler). The paper reports 4 classes.

**Partitioning & split.** Clients = K-means geographic clusters of
buildings. Per client, an 80/20 train/test split (`train_ratio=0.8`);
every client's test indices are pooled into one **global held-out test
set**, and *every reported accuracy/F1 is computed on it, every round*.

**Partition seed is now separate from the data seed (2026-07-18).**
`get_hfl_data_partitions(..., partition_seed=...)`: `random_seed` controls
the subsample row draw (and therefore which feature caches apply);
`partition_seed` controls **only** the K-means partition and the
per-client train/test splits. Every sweep sets
`partition_seed = partition_seed_for(seed_idx) = 5309 + seed_idx` (§11).
Why: with a fixed partition, one unlucky geographic layout at a single N
distorts the whole scaling curve (that is what the pre-2026-07-18 N=100
dip was), and the partition/test-split variance never entered the
confidence intervals at all. Because it depends only on `seed_idx` — never
on method, mode, or stress knob — every method still sees the identical
partition per seed, so the Wilcoxon pairing stays valid. It is folded into
the partition cache key, and the cache-file temp name is now PID-unique
(`.tmp<pid>`) so parallel workers writing the same key cannot corrupt each
other.

**Feature cache staleness is now a hard failure (2026-07-16 regression).**
A cache built for a different `data.subsample` used to index out of bounds
*deep inside a training epoch*, hours into a sweep. Two guards:
`compute_feature_cache` recomputes when the stored shape ≠
`(len(dataset), 512)`, and `CachedDataset.__init__` refuses a mismatched
array with an actionable message. Both pinned in `check_dataset_pipeline`.

**Black-chip diagnostic + hard gate (report in the paper).** A GSI fetch
failure yields an all-black chip carrying zero image signal; a high rate
collapses the model to the majority class while non-accuracy metrics keep
moving. The measured rate is logged and persisted per run under
`_diagnostics.black_chip_rate` in the resolved-config YAML — quote it next
to the accuracy tables. Beyond reporting, it is a **hard gate**: above
`data.max_black_chip_rate` (default **0.5**) the run raises `RuntimeError`
rather than producing a completed-looking majority-collapse result (the
diagnostic is persisted *before* the raise). Deliberate stress runs that
push the injected knob past the gate must raise the threshold explicitly.

**Licensing / availability (data availability statement).** GSI aerial
tiles are used under the **Government of Japan Standard Terms of Use
v2.0**; cached tiles are **never redistributed** with this repository
(`data/` is gitignored) — they are re-fetched on demand from the public
GSI service. Any use of the HF dataset must respect the upstream terms of
the fused sources; the pinned revision hash makes the exact evaluated
snapshot citable and re-fetchable. `HF_TOKEN` is needed on first run for
streaming; the metadata/partition/tile caches under `data/` are derived
artifacts, regenerated automatically, and make later runs offline-capable.

**Single-event limitation (mandatory paper statement).** All real-data
results derive from one seismic event. No second real, geo-located,
multi-modal damage dataset with ShakeMap-equivalent parameters was
obtainable; claims are scoped to "on this event", with the real-data
stress sweeps (§14) as the controlled robustness complement.

---

## 3. Model architecture and CPU feasibility

`CachedFusionModel` (`fl/model.py`) — deliberately small, CPU-trainable:

| Block | Structure | Params | MB (fp32) |
|---|---|---|---|
| `img_proj` | Linear(512→128)+ReLU over **cached** ResNet-18 features | 65,664 | 0.263 |
| `struct_branch` | 9→64→128→64 MLP (ReLU, Dropout 0.2) | 17,216 | 0.069 |
| `fusion` | concat(128+64)→256→4 head (Dropout 0.3) | 50,436 | 0.202 |
| **total** | | **133,316** | **0.533** |

**Why a feature cache:** ResNet-18 runs **once** per dataset
(`fl/features.py::compute_feature_cache` → `img_features.npy`, shared
across sweep workers via `data.feature_cache_path`); FL training never
touches the backbone. This is the load-bearing mechanism behind the
CPU-feasibility claim — back it with measured wall-clock (§12).

**The frozen-feature simplification is now measured — and it costs 20%
(`scripts/e2e_centralized.py`, new).** A reviewer will ask whether
freezing the backbone throws away signal. The script answers it directly:
on the *same* pooled training set and the *same* held-out test set, with
the *same* epochs and the *same* logit-adjusted loss, it trains (a)
`CachedFusionModel` on the cache and (b) `EndToEndFusionModel` — the
identical head on a **trainable** ResNet-18 over raw images. Wired into
`reproduce_paper.sh` (full run: `--subsample 0.2 --n 200 --epochs 15`; a
1-epoch tiny version runs in smoke).

**Measured result** (`results/e2e_centralized/e2e_comparison.csv`, already
run):

| Model | Macro-F1 | Accuracy |
|---|---|---|
| frozen features (`CachedFusionModel`) | 0.4101 | 0.7560 |
| end-to-end (trainable ResNet-18) | 0.5110 | 0.7787 |
| **frozen retention** | **80.27%** | 97.1% |

This is **not** the "retention near 100%" the design argument would have
liked, and the paper must not claim it is. The honest framing: freezing
the backbone costs ~20% of achievable macro-F1 and ~3% of accuracy, and is
justified on **feasibility** grounds (it is what makes CPU-only federated
training tractable at all — §12 wall-clock), not on the grounds that it is
free. State the number and the trade-off explicitly; a reviewer who runs
this check will find it. It also bounds the whole study: every federated
macro-F1 in the paper sits under a 0.51 ceiling that is itself imposed by
the caching decision, so the federated-vs-centralized gap should be read
against 0.410 (the same-architecture oracle), not 0.511.

**Block ownership replaces the freeze/unfreeze pair (Tier B, §6).** The
model now exposes generic per-block control instead of an img_proj-only
special case:

- `BLOCKS = ("img_proj", "struct_branch", "fusion")` — canonical order,
  matching state-dict prefixes.
- `set_trainable_blocks(blocks)` — gradients on exactly those blocks,
  everything else frozen.
- `block_state_dict(blocks)` / `load_block_state_dict(d)` — prefixed
  clone / prefix-tolerant load. `load_*` ignores absent prefixes, so a
  partial aggregate (e.g. `flat_fl`'s server aggregate, which carries no
  `img_proj` keys) leaves the missing block untouched.
- `trainable_state_dict` / `full_trainable_state_dict` are retained as
  thin wrappers over these, so the legacy Tier-2 and selection-isolation
  harnesses are unchanged.

Class imbalance is no longer handled by per-shard resampling; it is
corrected in the loss (§6 Tier A1).

---

## 4. Device heterogeneity model

`fl/device_state.py` — per-round IoT state:

- battery ∈ [0,1]: init U(0.5,1.0); −0.02/round when selected, **+0.01**
  passive recharge; **eligibility gate ≥ 0.20**
- SNR: base U(5,20) dB + per-round N(0,2) noise; **gate ≥ 3 dB**
- memory: 10% of devices permanently memory-constrained; **gate =
  memory_ok**
- compute time: base U(50,250) s + N(0,30) noise; **gate T̂ ≤ 300 s −
  ε_n**, with adaptive margin ε_n = 1.96·std of the last 10 observed
  completion times (needs ≥3 observations) — the straggler-safety margin
  of §IV-C1.

**Recharge changed 0.005 → 0.01 (2026-07-18) — derive it in the paper,
don't just state it.** The discharge:recharge ratio *is* the sustainable
participating fraction: at steady state `f·discharge = (1−f)·recharge`, so
`f = recharge/(discharge+recharge)`. The old 0.005 gives **f = 1/5**; over
a 100-round run the fleet drained until only ~20% of devices were ever
eligible, the eligible pool collapsed to a small permanently-cycling set,
and global accuracy *decayed* with the shrinking aggregate. 0.01 gives
**f = 1/3**, so the pool rotates instead of collapsing. This is a
modelling parameter with a closed-form consequence — present it that way.

**Stress knobs (default 0.0 = exact baseline behaviour):** `dropout_rate`
— per-(device, round) probability of forcing `memory_ok=False`, i.e.
transient loss modeled through the *existing* four-condition gate rather
than a fifth condition (no new eligibility semantics to justify);
`snr_degradation_db` — uniform dB subtraction, an area-wide aftershock
channel effect, not per-device.

**These knobs now bind on `flat_fl` too.** Selection mode `"all"` used to
return the covered set verbatim, bypassing the eligibility gate entirely —
which handed `flat_fl` a free upper bound and made it **invariant to the
dropout and SNR stress axes**, i.e. its stress curves were flat by
construction. Fixed 2026-07-18: `"all"` now means *every eligible covered
client*. A device with a dead battery, no memory, or a below-threshold
channel cannot deliver an update regardless of topology; exempting one
method from device physics is not an upper bound, it is a different
simulation.

---

## 5. Communication and energy models

**Communication (per round, uplink+downlink on both tiers):**

```
IoT payload  = struct+fusion            =  67,652 params ≈ 0.271 MB (float32)
UAV payload  = img_proj+struct+fusion   = 133,316 params ≈ 0.533 MB
hierarchical: comm = 2·n_selected·0.271 + 2·n_active_uavs·0.533   MB
flat_fl:      comm = 2·n_selected·0.271                            MB
```

Identical accounting for every method (`comm_mb_round` column): the single
rule is `uavbench/metrics/fl.py::round_comm_mb`, called by every harness.
The payload constants are pinned against the **live model's parameter
counts** by `check_metrics.py` (not just against themselves), so the
efficiency claim cannot silently drift from the architecture.

> **OPEN ITEM — the comm accounting has not been updated for block
> ownership, and the paper must not quote it without this caveat.**
> `round_comm_mb` still bills the IoT payload as `struct_branch + fusion`
> (0.271 MB). Under the committed default `fusion_owner: uav` (§6 Tier B),
> a hierarchical client actually ships `struct_branch` **only** — 17,216
> params, **0.069 MB** — and a UAV ships `img_proj + fusion` = 116,100
> params, **0.464 MB**. So for every hierarchical method the reported
> per-round cost overstates the IoT uplink by **3.9×** and the UAV uplink
> by 1.15×. `flat_fl`'s clients genuinely own `struct+fusion`, so its
> 0.271 MB is correct.
>
> The bias therefore runs **against** the proposed system: the hierarchy's
> true communication advantage over `flat_fl` is larger than reported, so
> no published claim is inflated. But the absolute MB figures are wrong.
> Either (a) make `round_comm_mb` block-aware and rerun the reporting
> (round tables only — this does not touch training, so it does not
> invalidate any checkpoint), or (b) state in the paper that comm cost is
> billed at the full struct+fusion IoT payload as a conservative upper
> bound. (a) is preferable and cheap.

**Energy (`problem/energy.py`, reporting-only by design):**
`E(d) = P_fly·(d/v) + P_hover·t_serve` with P_fly=250 W, P_hover=200 W,
v=15 m/s, t_serve=60 s, battery 200 kJ. Why reporting-only: optimizers
minimize the *normalized movement term* inside the fitness; the energy
model translates that into Joules/battery-fraction for interpretability
without ever feeding back into the objective (no hidden coupling).

**Per-UAV energy fix (2026-07-23) — a reported quantity was physically
impossible.** `metrics/placement.py::compute_metrics` charged the
*fleet-summed* movement distance `b.d_move` to a *single* battery, so
`movement_battery_frac` exceeded 2.0 — ten drones' movement billed to one
drone's battery. Now per-UAV distances are computed from the placement,
their energies summed for the fleet total (hover counted once per UAV),
and the reported `movement_battery_frac` is the **mean per-UAV** battery
fraction, which is the physical [0,1] per-drone quantity. Any Tier-1
energy number from before this commit is wrong by roughly a factor of K.

**Label the FL harnesses' `cumulative_energy_j` as movement energy.** It
counts repositioning only; there is no simulated-time model for hover or
communication energy in the round loop, so non-repositioning methods
(`hfl_static`, `centralized`, `flat_fl`) legitimately report 0 J. Say
"movement energy" in every figure axis and table header.

---

## 6. The training recipe (Tiers A / B / C) — new, and load-bearing

This section is new. It exists because on 2026-07-18 the proposed system
lost to its own baselines, and the cause was the *shared training recipe*,
not the selection or placement rule. Three tiers of fixes were introduced
(`0d3a2f59`, `92166776`, `f7ccd1b8`, `ba2e49f9`), every one of them a
config knob so each is an ablation. `PIPELINE_VERSION` was bumped to **3**
(§15), which invalidates every older checkpoint.

### Tier A — training dynamics

**A1. Logit-adjusted loss replaces per-shard balanced resampling.**
`make_loss_fn(log_prior, tau)` shifts logits by `τ·log_prior` during
training (Menon et al., ICLR 2021); inference uses raw logits, so only the
gradient changes. The prior is the **global** training-label distribution.
*Why this and not the old `WeightedRandomSampler`:* per-shard inverse
frequency resampling made every client optimize a **different** effective
objective — each shard was reweighted to its own local balance — which
manufactured client drift on top of the genuine non-IID split. Logit
adjustment corrects the imbalance once, globally, so every client
optimizes the *same* objective on its raw shard. `balanced_sampling: false`
(uniform permutation) is now the default; `true` restores the old sampler
for the ablation. τ = 0.601 (§0); `logit_adjust_tau: 0` gives plain CE.

**A2. Momentum SGD replaces Adam for local training.** `_make_optimizer`
reads `fl.local_optimizer` (default `{name: sgd, momentum: 0.9}`).
*Why:* Adam's moment estimates reset on every fresh per-round clone and
never amortize, so every round paid an unwarmed-up optimizer's bias;
momentum SGD carries no such per-round warm-up penalty and is the standard
FedAvg choice. `name: adam` restores the old behaviour.

**A3. Server momentum (FedAvgM) + across-round LR decay.**
`_apply_server_momentum` treats `(global − aggregate)` as the server
pseudo-gradient, applies heavy-ball momentum, and writes back only the
keys present in the aggregate; velocity persists across rounds. Reduces to
plain FedAvg at `server_momentum=0, server_lr=1`. `_lr_scale` applies a
`cosine` (default) or `sqrt` multiplier to both client and UAV LR.
*Why:* a fixed client LR under non-IID FedAvg orbits the optimum
indefinitely — the plateau-plus-oscillation the diagnostics showed —
and momentum damps the round-to-round direction reversal that non-IID
cohort rotation induces. `server_momentum = 0.528` (§0).

**A4. EMA-of-global evaluation.** With ±0.05/round oscillation, the
round-100 snapshot is a lottery ticket. An EMA of the global weights
(`ema_decay = 0.9`, blended in place each round) is what gets evaluated;
`ema_decay: 0` evaluates the raw global model. This changes what is
*reported*, not what is *trained* — state that explicitly, and note the
paired significance test additionally averages the last K rounds (§13),
so the reported statistic is doubly smoothed by design.

### Tier B — modality-aligned block ownership

**One owner per block, no parallel drift.** Hierarchical methods:

| Tier | Owns (default `fusion_owner: uav`) | Frozen during its step |
|---|---|---|
| UAV | `img_proj`, `fusion` | `struct_branch` |
| IoT client | `struct_branch` | `img_proj`, `fusion` |

`fusion_owner: client` moves `fusion` to the client tier (the ablation).
`flat_fl` has no UAV tier, so its clients necessarily own
`struct_branch + fusion` and `img_proj` **stays at initialization** — it
cannot use imagery at all, which is precisely the limitation the hierarchy
removes and the cleanest statement of what the UAV tier buys.

*Why:* previously both tiers trained overlapping parameters in parallel
from the same starting point and were glued together at aggregation. Now
each block is optimized against the **exact frozen value** of its
counterpart — vision and fusion learn against the structured
representation they will actually deploy with. Frozen blocks still
participate in the forward pass at their current global weights.

**Aggregation follows ownership** (§9): client-owned blocks flow
client→UAV as data-size FedAvg within each coverage zone; UAV-owned blocks
originate at the UAV; both are assembled per zone and combined at the
server with reputation weighting. A zone missing either contribution falls
back to the current global weights for those blocks — no double counting.

### Tier C — class-aware placement and selection

**C1. Class-scarcity placement weighting** (`placement_class_aware: true`).
Per-client minority-information weight `Σ_c scarcity_c · count_c(client)`,
normalized to mean 1 so it re-weights the placement value `V_i` without
changing its scale. UAV capacity is steered toward rare-class-rich zones.

**C2. Class-coverage stochastic roster** — see §8.

**C3. Selection cadence decoupled from placement cadence**
(`reselect_every: 1`, placement still every `T_sel = 5`). Holding the whole
roster for 5 rounds and then swapping it wholesale is the non-IID ping-pong
that made the frozen-roster method the most volatile of all; reselecting
every round lets the roster rotate smoothly under the UCB count bonus.
Placement stays on the expensive cadence.

**Privacy note the paper should make:** the class-aware machinery requires
each client to disclose only its **4-bin label histogram** — a minimal
statistic, and a natural insertion point for differential-privacy noise as
a stated future step.

**Validation protocol.** `configs/paper_smoke.yaml` (real data, subsample
0.2, 30 rounds) exists specifically to check these before committing the
VM to the ~45 h grid, with three pre-registered falsifiable signatures:
(1) round-to-round |Δmacro_f1| collapses below ~0.01 after ~round 10
[Tier A]; (2) the HFL family's macro-F1 climbs above `flat_fl` within ~30
rounds [Tier B]; (3) `proposed_hfl ≥ hfl_no_selection` on last-10 macro-F1
[Tier C]. `tier2_reduced.yaml` cannot do this — it drives the legacy
`run_tier2` harness, which none of these changes touch.

---

## 7. UAV placement: problem formulation and optimizers

`problem/instance.py` — all coordinates in a projected metric frame
(equirectangular about a reference; matches Haversine to <0.1% at study
scale). A `ProblemInstance` holds device coords (z=0), per-device value
V_i, per-position capacity and battery, previous UAV positions, search
bounds, and the range gate `R_comm`.

**Fitness (`problem/fitness.py`) — the single scoring entry point:**

```
F(X) = w1·(F_cover/F_max) − w2·(D_move/D_max) − w3·(L_imb/L_max)
w = (0.811, 0.03, 0.159)   ← searched; see §0
F_max = Σ V_i;  D_max = K·diag(box);  L_max = N²
```

Every optimizer scores **only** through this callable — the central
decoupling invariant. Tier-1 configs always pass `w1/w2/w3` explicitly, so
they are unaffected by the module default; `run_full_hfl`'s `_place_uavs`
uses the default.

**Batched scoring (`Fitness.batch`, 2026-07-27).** PSO and GA now score a
whole `(P, 3K)` population in one call. The greedy sweep is shared across
candidates (`greedy_assignment_batch`) while the *scalarization* stays a
per-candidate loop over the identical expressions — deliberately, because
vectorizing the tail too would change floating-point summation order and
shift results by ~1 ULP. Measured **5.3–7.5×** on the fitness loop,
bit-identical. `eval_count` still increments by P, so the shared-budget
assert is unaffected.

**Greedy assignment (`problem/assignment.py`, Algorithm 4 style):**
devices in descending value; feasible positions are in-range, under
capacity, battery ≥ B_min; winner = lowest current load, ties by smallest
distance (lexsort). Hard constraints are implicit (no coverage credit),
keeping the landscape penalty-free. The descending-value order is now
cached on the instance (`value_order`) rather than re-sorted on every
evaluation.

**Per-UAV radius extension.** `radii: (K,) | None` is a **call-time
parameter** threaded `greedy_assignment → Fitness → compute_metrics →
_covered_clients`, not a `ProblemInstance` field (the radius is a property
of the candidate placement, unknown at instance-generation time).
Optimizers publish it via `result.meta["radii"]`; `None` reproduces the
scalar gate bit-for-bit.

**Equal-radius fairness guard (new).** `compute_metrics` now *warns* when
a method is scored at `max(radii) > 1.02·R_comm` — a method scored at a
larger coverage radius gets more coverage per UAV for free. A warning, not
an error, so a deliberate future design that varies radius stays possible.
This is the automated tripwire for the bug described next.

**Heterogeneous fleets (new).** `generate_instance(capacity_cv=,
battery_cv=)` draws per-UAV capacity ~ `U(mean·(1∓cv))` and battery
likewise. Placement must then co-optimize *which* UAV (large vs small)
serves *which* region — a joint problem the value-blind physics baselines
cannot see. Both default to 0.0, and the draw happens **after every other
RNG draw**, so a `cv=0` instance is byte-identical to before and a
heterogeneous instance shares the exact device layout of its homogeneous
twin at the same seed. Used by `tier1_regime_hetero.yaml`
(`capacity_cv=0.5` → caps in [7,22] around 15; `battery_cv=0.3`).

**Real-geometry scenario (new).** `distribution: "real"` K-means-clusters
the actual cached Noto coordinates into N client centroids (the same
partition convention the FL harness uses), projects to metres, and min-max
normalizes into the benchmark box, so `R_comm` and K stay comparable to the
synthetic scenarios — **only the spatial structure is real**. The epicentre
is projected in the same frame. For `prev_mode: stale` the prior layout is
a displaced epicentre-biased cluster (real coords are not re-sampleable);
synthetic scenarios are byte-identical to before. Cached per (N, seed) as
`.tier1_real_centers_N*_seed*.npy`. This ties the Tier-1 benchmark to the
deployment and is now the 4th scenario in `tier1_core.yaml`.

**Value model (`hflsim/shared/value.py`):**
V_i(t) = β(t)·Û_i + (1−β(t))·R_i with β(t) = max(0, 1 − t/T_decay),
T_decay=20. Tier-1 benchmarks pin β=1 (history-free). Tier-1 raw feature
distributions: SNR ~ U(0,30) dB, samples ~ U_int(20,200), reputation
R_i ~ Beta(2,2) fixed within an instance. In `run_full_hfl` the live V_i
is additionally multiplied by the class-scarcity weight (§6 C1).

**Scenario generation:** `uniform`, `clustered` (√N/2 Gaussian clusters,
σ=0.06·span), `epicenter_biased` (single Gaussian, σ=0.12·span), `real`.
`prev_mode`: **`"stale"`** (default) fits the previous layout to a shifted
epicentre (offset N(0, 0.25·span)) — the situation that triggers a
reconfiguration, so `static` is a genuine floor; **`"warm"`** puts previous
positions near current device sub-centroids (the conservative,
already-deployed regime; `tier1_warmprev.yaml`).

### Optimizers

All behind one interface (`optimizers/base.py`) and one registry
(`build_optimizer`; budget keys P/G_max always override config for budgeted
methods, so no YAML can desync the paired PSO/GA comparison).

**PSO (proposed).** Constriction-factor PSO (Clerc & Kennedy 2002; χ from
φ=c1+c2=4.10, guard φ>4 raises), lbest ring topology (k=2), per-dimension
velocity clamp (0.2·range), absorbing walls, value-weighted k-means++
seeding, stagnation reinit (Δ<1e-4 for 20 iters → re-scatter ρ=0.2),
turbulence (p=0.1, jitter 10 m), early stop at 95% plateau. Every design
choice is a config toggle. Full equations and hyperparameters: Appendix A.

**GA (internal head-to-head).** Real-coded, tournament (size 3), SBX
(η_c=15, p=0.9), polynomial mutation (η_m=20), 2 elites — same fitness,
same P·G_max budget by construction.

**Heuristic floors:** `centroid` (value-weighted k-means), `random` (best
of 20 uniform draws), `static` (no repositioning).

**Literature baselines (published methods, path-loss radius).** Shared
channel model in `problem/path_loss.py`: probabilistic LoS, mean loss =
FSPL(d_3d,f) + P_LoS·η_LoS + (1−P_LoS)·η_NLoS; `coverage_radius(h)` by
bisection; environment presets (suburban a=4.88, b=0.43, η=0.1/21 dB, plus
urban/dense/high-rise). Sourcing confirmed against **two** Al-Hourani
papers (Appendix A.5).

- **`mozaffari2016`** (IEEE Comm. Lett. 2016): radius-maximizing altitude
  (h*, r*) once, then K equal discs by greedy maximal covering
  discretized at device locations. Adaptation note: Appendix A.6.
- **`alzenad2017`** (IEEE WCL 2017): value-weighted k-means into K
  clusters; per cluster, required radius = max member distance; altitude =
  **minimum** h whose path-loss radius reaches it → genuinely non-uniform
  per-UAV radii. Adaptation note: Appendix A.7.

**Naive placement arms (new: `centroid_place`, `random_place`).** These
exist because **`hfl_static` was being read as a bad-placement baseline
and is not one** — it runs PSO placement, just *once* instead of every
`T_sel` rounds, so with static ground clients it ties the proposed system
and isolates **repositioning cadence**, not placement quality. The two new
arms supply the missing contrast: identical UCB selection, reputation
FedAvg and cadence, but a naive placement rule (k-means centroids /
uniform-random positions). Both are in `_PLACEMENT_BASELINES` (§10).

### Link-budget calibration — two distinct mechanisms

**(1) Fixed calibration per deployment scale.** `max_path_loss_db` is
deployment-specific and must make the *placement rule* the compared
variable, not the radio.

- **Tier-1: 95 dB @ 2 GHz → r* ≈ 499 m, matching `R_comm = 500 m`.** This
  is a **correction of a scoring asymmetry, not a tuning knob.** The
  previous 100 dB gave the baselines r* = 618 m — a **1.53× ground-area
  advantage** — which is the entire reason the earlier Tier-1 tables showed
  the physics baselines beating PSO on the uniform scenario. Re-scored at
  equal radius (`results/tier1_core/significance_final_fitness.csv`, 4
  scenarios × 30 seeds, Holm-corrected), PSO beats **every non-optimizing
  baseline** (mozaffari, alzenad, centroid, static, random) in **all four**
  scenarios. Against GA it wins `epicenter_biased` (+0.067, p=3.7e-9) and
  `clustered` (+0.029, p=1.3e-7) and **ties** on `uniform` (GA +0.006,
  p=0.096) and `real` (PSO +0.008, raw p=0.035, not significant after
  Holm). Report the GA ties — PSO's advantage is over the *non-searching*
  methods, not over another metaheuristic on easy geometry.
- **Full-sim Noto scale: 145 dB @ 2 GHz** → radii on the 20 km order,
  matching `R_comm = 20 km`.

**(2) Per-job re-calibration for the coverage sweep (new).**
`path_loss_db_for_radius` inverts `optimal_altitude_mozaffari` by bisection
(monotone in the link budget). The coverage sweep varies `R_comm`, so a
baseline tuned for a 20 km radius but gated at 2 km spreads its UAVs far
too thin and loses on the **mismatch** rather than on its placement rule —
the sweep's first run showed exactly that artifact (mozaffari covered 7% at
R=2 km vs 27% for the others). `_coverage_job` therefore re-derives
`max_path_loss_db` per swept R_comm. It calibrates **only the method that
job actually runs**: `optimizer_params` is part of the resume signature
(§15), so editing all entries would mark every finished job stale — measured
300/300 invalidated instead of 120.

**(3) Coverage-radius uniformity switch (new).** `fl.uniform_coverage_radius:
true` gates *every* method's coverage at the scalar `R_comm`, ignoring any
path-loss-derived per-UAV radius. This is the Tier-1 equal-radius rule
carried into the FL harnesses; used by `paper_coverage.yaml` so a baseline's
own radius cannot bypass a tight `R_comm`. (This flag was referenced but
never defined in `run_tier2` from `e79701f5` until 2026-07-26, so every
placement method in that harness raised `NameError` on its first placement —
fixed; any Tier-2 result from that window is void.)

**`# VERIFY AGAINST PAPER` markers — status:** the `(a,b,η)` presets and
the 3-D-vs-horizontal FSPL convention are **confirmed** (Appendix A.5) and
closed. What remains open *by design, as stated adaptation choices*:
Mozaffari's closed-form altitude optimum (grid search used as a deliberate
numerical stand-in); Alzenad's exact altitude rule and SEC-vs-centroid 2-D
step; each paper's own multi-UAV packing arrangement.

---

## 8. Client selection

All modes behind one `ClientSelector.select()` and the same **eligibility
gate** (§4 — including mode `"all"`, since 2026-07-18).

**Proposed (`ucb`) — the full pipeline:**

1. **Utility** Û = 0.043·U_epi + 0.078·U_SNR + 0.295·U_dens + 0.584·U_prox
   (searched weights, §0; epicentre distance capped at the eligible-set
   95th percentile; min-max SNR; 5 km-radius density proxy — no building
   inventory exists for the area; proximity 1 − d_min/R_comm).
2. **Priority** P = 0.702·(1 − (T̂/T_max)²) + 0.298·Ũ, with
   Ũ = β(t)·Û + (1−β)·R. **Battery is no longer in the score** — it
   remains a hard gate (§0 explains why).
3. **UCB score** = P + √2·√(ln t / (N_n+1)) — the anti-starvation
   mechanism; the fairness claim it implies is measured with Jain's index
   (§12), not asserted.
4. **Class-coverage roster construction** — replaces the deterministic
   greedy argsort. Per UAV, slots are filled by repeatedly adding the
   feasible unassigned client maximizing

   ```
   marginal_coverage_gain / gain₀  +  0.435·static̃  +  1.475·Gumbel
   ```

   where a roster's coverage value is `Σ_c scarcity_c·√(Σ_{i∈S} count_c(i))`.
   The **√ gives diminishing returns per class** (the majority class
   saturates fast; scarce classes keep paying), which makes the objective
   **submodular** and therefore greedy near-optimal — that is the
   justification for using greedy here, and it should be stated. `gain₀`
   normalizes to the first-pick scale; `static̃` is the min-max-normalized
   priority+UCB score; the per-client **Gumbel** perturbation makes the
   roster a *sample* rather than a fixed argsort (Gumbel-top-k ≡ sampling
   ∝ score without replacement). `SEL_MIN_MARGINAL = 0.0` (fill capacity;
   raise it for an energy-aware early stop when only redundant coverage
   remains). Falls back to plain priority greedy when class information is
   absent.

   **The roster sampler has its own RNG** (`_sel_rng`, seeded from the
   run's per-method seed), kept separate from the shared placement/device
   RNG so the sampler cannot perturb the environment simulation
   (battery/SNR trajectories) — which also keeps every non-`ucb` baseline's
   device physics byte-identical to a deterministic run.

Selection runs every `reselect_every` rounds (default **1**, decoupled from
placement — §6 C3) **or** on the early-reselection trigger:
eligible < λ_min·min(K·capacity, N), λ_min = 0.5.

**Literature baselines (selection is the only variable; identical PSO
placement, reputation FedAvg, cadence as `proposed_hfl`):**

- **B1 `fedcs`** (Nishio & Yonetani, ICC 2019): greedy fastest-first under
  a per-round deadline; reputation/fairness-blind by design.
- **B2 `rep_cap`** (Zhao et al., Chin. J. Aeronaut. 2024): static ranking
  γ·R + (1−γ)·(1−(T̂/T_max)²), γ=0.5; no exploration.
- **B3 `fair_mab`** (Zhu et al., Sensors 2024): reward = 0.5·battery +
  0.5·min(1, staleness/T_sel).
- **B4 `oort`** (Lai et al., OSDI 2021) — **new**: statistical utility
  (last-observed local training loss) × system-speed penalty
  `(T_pref/T̂_n)^α` for stragglers, α = 2 (the paper's default). Oort
  leaves the developer-preferred round duration open; here `T_pref` is the
  **median compute time of the currently eligible pool** — `T_max` would
  never fire, since eligibility already caps `T̂_n ≤ T_max − ε`. Losses are
  the last-observed values, exactly as in Oort's own stale-utility design;
  never-trained clients get the current max observed loss so they are
  explored first.
- **B5 `power_of_choice`** (Cho et al., 2020 / AISTATS 2022) — **new**:
  draw `d = 2 × (total slots)` candidates uniformly from the eligible pool,
  keep the highest last-observed local loss. The canonical rule evaluates
  the *current global* loss on the candidate set; the simulation uses
  cached last-observed local losses (the standard variant in FL benchmarks)
  **so no extra forward passes are billed to the baseline** — state this,
  it is a fairness choice in the baseline's favour.

B4/B5 required a new plumbing path: `_train_blocks` returns the
sample-weighted mean local training loss, and the round loop pushes it via
`selector.update_losses()` — a no-op for every other mode.

*Adaptation constants are documented inline in `client_selection.py` —
cite them as our instantiation choices.* Full citations, pseudocode
(B1–B5), and per-baseline fidelity notes: Appendix C. Stage-by-stage
formulas and the constants table: Appendix B.

**Bug that invalidated a published-looking result — report the corrected
numbers only.** `run_selection_isolation` never passed
`class_counts`/`class_scarcity` to `select()`, so the proposed `ucb` rule
**silently fell back to plain priority greedy** — the entire class-coverage
component was disabled while every baseline ran fully featured. Under that
bug `ucb` lost significantly to Oort/PoC at N=350 (−0.036/−0.041) and N=500
(−0.064/−0.061). Fixed 2026-07-26 (`f71659fb`); the harness now builds the
histograms exactly as `run_full_hfl` does. Any selection-isolation number
predating that commit is void.

---

## 9. Reputation and aggregation

**Reputation (`fl/reputation.py`), three components per client** (exact
formulas in Appendix B §B.3): contribution (cosine similarity between
successive EMAs of the client's update-*delta* vector), anomaly (diagonal
Mahalanobis distance vs a global EMA-tracked per-parameter mean/variance,
divided by √J so the fixed threshold d≤2 is dimension-independent),
temporal reliability (success rate + response-time stability; `mark_absent`
on selected-but-silent clients). Aggregate R_n = w·[R_contrib, R_anomaly,
R_temp], with the **prior now w = (0.091, 0.134, 0.775)** — searched, §0.

**Bayesian weight adaptation:** w is a Dirichlet posterior updated every 10
rounds — prior = 20·w_init pseudo-counts, evidence accumulates the
per-round component values; posterior renormalized to the simplex. Absence
counts as evidence *against* components that scored the client highly.

**Aggregation (per round), now block-structured (§6 Tier B):**

1. **Client → UAV.** For each coverage zone, the client-owned blocks are
   combined by sample-weighted FedAvg over that zone's delivering clients.
   A zone with UAV training but no client deliveries falls back to the
   current global weights for those blocks.
2. **UAV contribution.** The UAV-owned blocks come from the UAV's own
   training on its pooled shard; a zone without UAV training falls back to
   the global weights for those blocks. Assembling per zone this way is
   what prevents double counting — the UAV trains on the *pooled shards of
   its own clients* (no separate aerial dataset exists), so mixing its
   struct update in would count every sample twice.
3. **UAV → server.** **Reputation-weighted FedAvg** — weight = reputation ×
   n_samples (uniform fallback if all reputations collapse); each zone's
   reputation = 10%-trimmed mean of member reputations, and zones below
   **R_min = 0.3** are excluded that round (§IV-D poisoning guard).
4. **Server momentum** (§6 A3) is applied to the resulting aggregate, then
   the EMA evaluation model is advanced.

---

## 10. Experimental harnesses and the method table

| Harness (CLI) | What it isolates | Output |
|---|---|---|
| Tier-1 `run` (`tier1_core.yaml`) | placement optimizer quality on generated instances (**4 scenarios** incl. real geometry × 7 methods × 30 seeds) | `runs.parquet`, `convergence.parquet`, `component_summary.*`, `pareto_*.png` |
| Tier-1 `mclp` (new) | PSO's **% of MILP-optimal coverage** | `mclp_reference.csv` |
| Tier-2 `run_tier2` / `smoke_tier2` | placement inside a real FL loop (coverage = participation; no selection layer) | `tier2_rounds.parquet` |
| Full sim `run_paper_sim` (`paper_full.yaml`) | the complete system + ablations + literature baselines (13 methods × 4 N × 10 seeds = 520 jobs) | `fullsim_rounds.parquet` → `paper_sweep_rounds.parquet`, `operational_summary.csv`, `per_class_f1.csv` |
| **Coverage sweep `run_coverage_sweep`** (`paper_coverage.yaml`, **new**) | does placement quality matter for FL accuracy? R_comm × method × seed at fixed N (6 × 7 × 10 = 420) | `coverage_sweep_rounds.parquet`, `coverage_*.png` |
| Selection isolation `run_selection_sim` | selection rule **only**: static elbow-K-means UAVs, identical layout/seed across modes (6 N × 7 modes × 10 seeds = 420) | `selection_sweep_rounds.parquet` |
| Stress sweep `run_stress_sweep` (`stress_test.yaml`) | robustness under dropout / SNR / black-chip degradation (§14) | `stress_rounds.parquet`, `stress_*.png` |
| **Selection-pressure stress** (`stress_selection.yaml`, **new**) | the same grid with slots < N so selection binds (§14) | `stress_rounds.parquet` |
| `scripts/e2e_centralized.py` (new) | frozen-feature simplification vs trainable ResNet-18 (§3) | `results/e2e_centralized/` |

### 10.1 What is configured vs what has actually been run

The table above describes **configured** grids. What exists in `results/`
is not the same thing, and the difference decides what can be written up
today. Verified against the on-disk parquet/resolved-config files on
2026-07-28.

> **A full pipeline is in flight — check it before acting on this table.**
> As of 2026-07-28, GCP `instance-20260715-110613` (europe-west4-a) is
> running `launch_final.sh` → `scripts/run_gcp.sh` → `reproduce_paper.sh`
> (non-smoke), started **2026-07-27 06:42 UTC**. The VM's git HEAD is
> **`9aa933ed`, identical to local HEAD** — no code drift, `PIPELINE_VERSION
> = 3` on both sides, so its checkpoints and ours are mutually valid.
>
> Because the Tier-1 and paper-sim configs were unchanged, those steps
> **resumed in ~30 seconds each** from checkpoints (the resume gate working
> as designed, §15). The genuinely new work is:
>
> - **Selection isolation at `subsample: 1.0`** — running now, 420 jobs, at
>   ~177 complete. The subsample change altered the resume signature, so all
>   420 jobs correctly rerun rather than reusing the 0.1 results. **This
>   closes open item 1.** At the observed rate (~0.126 jobs/min aggregate on
>   12 workers) the step alone is ~56 h, finishing ≈ 2026-07-29 14:30 UTC.
> - **Coverage sweep with all 7 methods** — queued. The VM's
>   `paper_coverage.yaml` already contains `centroid_place` and
>   `random_place`, and the resume gate will skip the 300 finished jobs and
>   compute only the 120 new ones (~5 h). **This closes open item 2.**
> - `results/selection_scaling/` is **already complete on the VM disk**
>   (N350 + N500, 10 seeds, figures, resolved config, seed manifest) but has
>   never been committed. `run_gcp.sh` ends with `git add -A -- results` +
>   push, so it will arrive with everything else.
>
> **Not covered by this run:** the Tier-1 supplements (open item 3).
> `reproduce_paper.sh` runs `tier1_core.yaml` only, so
> `tier1_equal_radius` / `_regime_hetero` / `_warmprev` will still be
> 3-scenario when the VM stops. They can be closed independently in ~25
> minutes on any machine — Tier-1 `.checkpoints/` are git-tracked, so only
> the `real` jobs recompute (item 3).
>
> Estimated total remaining ≈ 45 h → pipeline ends ≈ **2026-07-30**, after
> which the VM commits, pushes, and self-terminates.

| Config | Configured | Actually on disk | Gap |
|---|---|---|---|
| `tier1_core` | 7 methods × **4** scenarios × 30 seeds | ✅ all four incl. `real`; `mclp_reference.csv`, `component_summary`, `pareto_*.png` present | — |
| `tier1_equal_radius` / `_regime_hetero` / `_warmprev` | inherit 4 scenarios | ⚠️ **3 scenarios only** (uniform/clustered/epicenter) | predate the `real` scenario; re-run if the supplements are to match the headline |
| `paper_full` | 13 methods × 4 N × 10 seeds | ✅ 52,000 rows = 13 × 4 × 10 × 100 | — |
| `paper_coverage` | **7** methods × 6 R_comm × 10 seeds | ⚠️ **5 methods** (proposed_hfl, hfl_static, mozaffari, alzenad, flat_fl); resolved config confirms it | **`centroid_place` and `random_place` have never been run.** The naive-placement contrast (§7) currently exists only as code + config |
| `selection_isolation` (matched, subsample 1.0) | 7 modes × 6 N × 10 seeds | ❌ **not run** — no `results/selection_isolation/` | `46a8b85f` renamed the old directory to `_starved`; the matched run was never executed |
| `selection_isolation_starved` (0.1) | same grid | ✅ 84,000 rows, 7 modes × 6 N × 10 seeds | this is the *old* isolation run, re-labelled |
| `selection_scaling` / `selection_scaling_full` | 4 modes × 2 N × 10 seeds | ❌ **not run** | config-only |
| `stress_test` | 8 methods × 11 cells × 16 seeds | ✅ 42,240 rows = 8 × 11 × 16 × 30 | — |
| `stress_selection` | same | ✅ 42,240 rows | — |
| `tier2_sweep` | 6 N (30…250) | ⚠️ **N=100/150/200/250 only** | N=30, N=50 missing |
| `e2e_centralized` | 1 comparison | ✅ `e2e_comparison.csv` (§3) | — |
| `hpo` | weight search | ✅ `weights.db`, leaderboard, `ema_decay_validation.csv` | the ema_decay validation *has* been run — read it before closing open item 3 |

**Consequence for §2's selection-at-scale argument.** The matched-data
numbers quoted in `configs/selection_isolation.yaml` and
`selection_scaling_full.yaml` — "+0.041 vs Oort, +0.030 vs PoC at N=500 at
subsample 1.0" — **have no committed results artifact.** They appear only
as config commentary. Either re-run the matched grid and add a
`results_provenance.md` row, or drop the claim; it cannot go in the paper
on the strength of a YAML comment. The *starved* result (proposed selector
loses at N≥350) **is** backed by committed data, so as of today the only
defensible statement is the negative one plus the mechanism.

**Why the coverage sweep exists (state this in the paper).** At the
`paper_full` R_comm = 20 km every client is covered, so the placement
method barely moves macro-F1 (proposed ≈ mozaffari ≈ alzenad) — which
reads as "placement doesn't matter". Varying R_comm at fixed N makes
coverage **bind**, which is the honest way to test it.

**What the committed sweep actually shows** (mean macro-F1, last 10
rounds, 10 seeds; `coverage_sweep_rounds.parquet`) — this partially
confirms the hypothesis and partially refutes it, and the doc used to
overstate it:

| R_comm (km) | 2 | 4 | 6 | 8 | 12 | 20 |
|---|---|---|---|---|---|---|
| proposed_hfl | 0.337 | 0.409 | 0.435 | 0.454 | **0.504** | 0.533 |
| alzenad2017 | 0.342 | 0.415 | 0.439 | 0.455 | 0.486 | 0.537 |
| hfl_static | **0.352** | 0.399 | 0.430 | 0.450 | 0.496 | 0.531 |
| mozaffari2016 | 0.288 | 0.327 | 0.332 | 0.359 | 0.392 | 0.522 |
| flat_fl | 0.344 | 0.344 | 0.344 | 0.344 | 0.344 | 0.344 |

- ✅ **The fan-out is real against `mozaffari2016`** (significant at 5 of 6
  radii, +0.049…+0.112) **and against `flat_fl`** (significant at 5 of 6).
  Mozaffari's equal-disc packing collapses to **7% coverage at 2 km**,
  which is the mechanism.
- ❌ **It is not real against `alzenad2017`** — one significant cell out of
  six (R=12 km, +0.018); the two track each other within ±0.006 elsewhere.
- ❌ **It is not real against `hfl_static`** — zero significant cells, so
  repositioning cadence remains unproven even where coverage binds.
- ⚠️ **There is a crossover at 2 km**: coverage falls to ~27% and `flat_fl`
  (0.344) edges every UAV method. Report this operating boundary — it is a
  genuine limit of the hierarchy, and hiding it invites exactly the
  question it answers.

So this figure connects Tier-1 placement quality to the end task **against
the naive and equal-disc rules**, not against every baseline. Frame it
that way (0.5.3).

**`_METHOD_CFG`** (`fl/federated.py`) registers **15** methods, mapping
each to (placement, selection, reputation-weighted, dynamic).
`paper_full.yaml` runs 13 of them; the two naive placement arms are used
only by `paper_coverage.yaml` (and have not yet been run — §10.1):

- **proposed:** `proposed_hfl` = (pso, ucb, T, T)
- **ablations:** `flat_fl`, `centralized` (oracle), `hfl_no_selection`
  (random), `hfl_static` (PSO once — *cadence*, not placement quality),
  `hfl_no_reputation` (uniform FedAvg)
- **selection literature:** `fedcs`, `rep_cap`, `fair_mab`, **`oort`**,
  **`power_of_choice`** (pso + own selection)
- **placement literature:** `mozaffari2016`, `alzenad2017` (own placement
  + ucb)
- **naive placement (new):** `centroid_place`, `random_place` (own
  placement + ucb)

Report ablations, selection literature, and placement literature in
**three separate tables** — they answer different questions.

**`fl.placement_method` override — bug fixed.** The override swaps the
authoritative optimizer for the proposed system's ablations but must
**exempt** methods whose placement *is* the variable. The exemption test
used to be `placement_method == method` (a string coincidence); it is now
explicit membership in
`_PLACEMENT_BASELINES = {mozaffari2016, alzenad2017, centroid_place,
random_place}`. Under the old test the two new naive arms would have been
silently swapped to PSO — i.e. three identical methods.

---

## 11. Seeding, pairing, and why paired tests are valid

**Tier-1 (`runner.py`):** two SeedSequence streams with structural tags —
instance stream `[base, 0, scenario, seed]` (method-independent → every
method sees identical instances) and optimizer stream
`[base, 1, method, scenario, seed]`. The tags fix a real hazard: numpy
SeedSequence treats trailing-zero entropy as equivalent
(`[b,s,i] ≡ [b,s,i,0]`), so without tags an optimizer stream at seed 0
collides with an instance stream whenever the two bases are set equal.
`_run_one` asserts the streams did not collide — a violation crashes the
run.

**FL harnesses (`fl/seeds.py`, formulas frozen & regression-pinned):**

- `tier2_seed = (opt_seed + n_clients·7919 + md5(method) mod 2³¹) mod 2³¹`
- `fullsim_method_seed = (run_seed XOR md5(method) mod 2¹⁶) mod 2³¹` —
  method identity folded exactly once (sweep callers must not pre-fold).
- `sweep_job_seed = opt_seed + seed_idx·7919 + N·31` — deliberately
  method-free: the paper sweep folds method later; **selection isolation
  shares it across modes** so the selection rule is the only cross-mode
  difference; the **stress sweeps keep it knob-independent** so every cell
  sees the identical base problem per seed; the **coverage sweep keeps it
  R_comm-independent**, since R_comm is a within-seed condition, not a seed
  input.
- **`partition_seed_for(seed_idx) = 5309 + seed_idx`** (new) — the
  data-partition seed for every sweep. Depends only on `seed_idx`, never
  on method/mode/knob, so paired comparisons still see the identical
  problem instance per seed while the partition/test-split variance
  enters the CIs. The offset 5309 is deliberately distinct from the
  historical data seed 42, to make the change visible in manifests.

**Seed manifests:** `build_seed_manifest(cfg, harness)` enumerates every
resolved seed by calling these same functions (no reimplementation → no
drift), now including `partition_seed` on every FL row and a new
`coverage` harness. Every run CLI writes `seed_manifest.csv` *before* the
run starts. Every number in the paper traces to an exact rerunnable seed.

---

## 12. Metrics: what is reported, from where, and why

| Metric | Source | Why |
|---|---|---|
| Accuracy | held-out global test set, every round | comparability with prior damage-classification work |
| **Macro-F1 (primary)** | same | class imbalance (~80/…) makes accuracy alone misleading; this is now `fl.target_metric` |
| Per-class F1 (`f1_survived…f1_missing`) | same; aggregated by `analyze_fl_reporting` → `per_class_f1.csv` | shows rare classes are learned, not masked by macro-F1 |
| Confusion matrix | `confusion.parquet` (long form per method×round) + heatmaps | direct evidence for/against majority-class collapse. **`centralized` now emits it too** — it was silently dropped before 2026-07-18, leaving that column empty for the oracle |
| Communication MB | pinned two-tier accounting (§5) | efficiency claim — **read the OPEN ITEM in §5 before quoting** |
| Energy (J, mean per-UAV battery frac) | `EnergyModel`, reporting-only, per-UAV since 2026-07-23 | translates the movement term into physical units |
| Coverage %, F_cover | shared fitness breakdown | primary placement quality |
| Movement (m) | fitness breakdown | shows coverage isn't bought by unconstrained repositioning |
| Load imbalance | fitness breakdown | per-round *assignment* balance — distinct from selection fairness; label both clearly |
| Jain's fairness index | cumulative selection counts, `metrics/fl.py::jain_index` (Jain et al., DEC TR-301, 1984) | instrument for the anti-starvation claim (NaN for `centralized`) |
| `n_unique_selected` | cumulative counts | complements Jain: how much of the fleet was ever used |
| `evals_to_threshold` | iteration reaching 95% of final best | convergence speed under an identical eval budget |
| `convergence_auc` | trapezoidal AUC normalized by shared G_max, flat plateau extension | trajectory summary that doesn't reward early stopping with a smaller denominator |
| Wall-clock | `wall_time_s` per optimizer run; `round_time_s` per FL round; `summarize_wall_clock` per method | the CPU-feasibility claim needs measured numbers |
| Black-chip rate | `_diagnostics` in resolved config | data-quality evidence behind the accuracy numbers |
| `rounds_to_target` | first round after which the target holds for **2 consecutive** evaluations | deployment-facing convergence measure |
| **`pct_of_optimal`** (new) | `mclp_reference.csv` | PSO's coverage as a % of the MILP optimum (§7, Appendix A.9) |

**`rounds_to_target` was measuring noise (fixed 2026-07-18).** Two changes,
with **different scopes — do not conflate them**:

1. `TARGET_CONSEC_ROUNDS = 2` — a single crossing can be a transient; on
   this imbalanced test set near-majority-class predictions reach ~0.70 in
   the first rounds, and the 2026-07-16 run reported `rounds_to_target = 1`
   for runs whose *final* accuracy was far below the target. **All three FL
   harnesses** (`run_tier2`, `run_full_hfl`, `run_selection_isolation`) now
   apply the consecutive-round rule.
2. The target *metric* changed to `fl.target_metric = macro_f1` at
   `target_value = 0.45` — but **only in `run_full_hfl`**. `run_tier2` and
   `run_selection_isolation` still gate on **accuracy** against
   `fl.target_accuracy` (0.70 default; 0.60 in
   `selection_isolation.yaml`). So `rounds_to_target` is *not* comparable
   across harnesses. Say which metric a reported rounds-to-target uses.

*Cosmetic mismatch worth knowing when reading logs:* `run_full_hfl`'s
per-method completion line still prints `target_accuracy * 100` ("Rounds to
70%") while the value it reports was gated on macro-F1 ≥ 0.45. The parquet
column is correct; the log string is misleading.

**New reporting artifacts.**
`analyze_fl_reporting` writes `operational_summary.csv` (coverage %,
participation %, comm MB, movement energy, round time, placement fitness,
Jain, rounds-to-target) and `per_class_f1.csv`. `plot_tier1_pareto` writes
`component_summary.{parquet,csv}` and per-scenario **coverage-vs-movement-
energy Pareto scatters**. Both exist for the same reason: the scalarized
fitness and the headline accuracy hide what the system actually costs. The
operational table shows placement/selection *cost* even where accuracy
saturates — that is the honest framing of the placement contribution, and
it is what the coverage sweep (§10) then turns into an accuracy story.

Rule: **never report a metric without a traceable computation in the
codebase** (each row above names its source).

---

## 13. Statistical methodology

- **Paired tests by design:** the seed architecture (§11) guarantees every
  method sees the identical problem instance *and* the identical data
  partition per seed, so per-seed values are paired samples. Use the
  **Wilcoxon signed-rank** test (default) or paired t — `uavbench
  significance` / `analysis/significance.py`. *State the pairing property
  as the justification.*
- **The reported statistic is the mean of the last K rounds, K = 10
  (`--last-k`).** Under ±0.05/round oscillation a final-round snapshot is a
  lottery draw, so a paired test on it mostly measures snapshot noise.
  `--last-k 1` restores the old behaviour. Disclose K in the paper.
- **Effect sizes and CIs are now reported alongside every p-value (new).**
  `effect_size` = matched-pairs **rank-biserial correlation** (Kerby 2014),
  the natural Wilcoxon effect size: rank the nonzero |differences|,
  `r = (W⁺ − W⁻)/(W⁺ + W⁻)`, range [−1,1], sign matching `mean_diff`.
  `ci_low`/`ci_high` = 95% **percentile bootstrap** on the mean paired
  difference (10,000 resamples of the *difference vector*, so the pairing
  is preserved; deterministic given the seed). A significance table with no
  effect size is not publishable.
- **Multiplicity:** Holm–Bonferroni step-down; raw p-values reported
  alongside corrected decisions. Two new controls:
  - `--reference <method>` tests only reference-vs-each-other (M−1
    comparisons) instead of all M(M−1)/2 pairs;
  - `--correction-scope {table,group}` — `table` corrects over every row
    (most conservative), `group` corrects within each group (per N / per
    R_comm / per scenario), appropriate when each group is a separate
    experiment or figure panel.

**The power floor — this is why those controls exist, and it must be in
the paper's methods section.** A two-sided paired Wilcoxon on *n* seeds has
a **hard minimum p-value of 2/2ⁿ**, independent of effect size. If the Holm
family is large enough to push the first threshold below that floor, *every*
comparison is structurally non-significant no matter how large the effect.
This is not hypothetical — it silently produced two entire rounds of
"no significant results":

| Family | Comparisons | Holm threshold | Seeds needed | What happened |
|---|---|---|---|---|
| `paper_full` all-pairs, 13 methods × 4 N | 312 | 1.6e-4 | n ≥ 13 | at n=10 (floor 1.95e-3) nothing could ever be significant |
| `paper_full` reference-only, within group | 12 | 4.2e-3 | n = 10 suffices | the design now used |
| `stress_test` all-pairs, 8 methods × 11 cells | 308 | 1.62e-4 | **n ≥ 14** | the 2026-07-22 and 2026-07-23 runs both returned **0** significant results at any effect size |

Consequences now baked in: `n_seeds = 10` for `paper_full`,
`selection_isolation`, `paper_coverage`; **`n_seeds = 16`** for the stress
sweeps (n≥14 required, 16 leaves margin); and `reproduce_paper.sh` runs
every **FL** significance step with `--reference proposed_hfl` (or
`--reference ucb` for selection isolation) and `--correction-scope group`.
The CLI also **emits an `UNDERPOWERED` warning** when the floor exceeds the
threshold, so the failure can never again be silent.

**Tier-1 deliberately stays all-pairs and uncorrected-scope**, and that is
correct rather than an oversight: at n=30 the Wilcoxon floor is 2/2³⁰ ≈
1.9e-9, far below the all-pairs Holm threshold, so the full 7-method ×
4-scenario family is comfortably powered — and all-pairs is the right
question for a placement benchmark, where GA-vs-centroid matters as much as
PSO-vs-anything. The observed p-values (down to 3.7e-9) confirm the floor
is not binding there.

- **Guard rails:** the test refuses mismatched or duplicated seed sets;
  round tables are reduced per (method, group, seed) first; grouping is per
  scenario (Tier-1), per N (sweeps), per R_comm (coverage), or per stress
  cell.
- **Output naming:** non-default metrics write `significance_<metric>.csv`
  so a macro-F1 pass no longer overwrites the accuracy table (that
  collision is why `macro_f1` results kept disappearing).
- **Tier-1 seed count:** 30 per cell; supplements (`tier1_regime_hetero`,
  `tier1_warmprev`) use 20. Report mean ± sd or CIs, never single-run
  numbers.

---

## 14. Robustness stress sweeps (real data)

Purpose: the controlled complement to the single-event scope — degradation
axes the recorded event cannot exhibit on demand, applied to the **real
dataset** (`configs/stress_test.yaml`, subsample 0.1, N=60, K=6):

- `dropout_rate` ∈ {0, .1, .2, .3, .4} — transient per-(device, round)
  loss through the existing eligibility gate;
- `snr_degradation_db` ∈ {0, 3, 6, 10} — area-wide aftershock channel
  degradation;
- `black_chip_rate` ∈ {0, .05, .10, .20} — **additional** unusable-imagery
  fraction: `_apply_black_chips` deterministically zeroes real cached
  image-feature rows (copy semantics — disk caches stay pristine; dedicated
  RNG stream seed+977 so nothing else shifts with the rate), on top of the
  measured natural fetch-failure rate. *State this additional-degradation
  semantics in the paper.*

Grid: one-axis-at-a-time (baseline = first value) → **11 cells** for the
paper body; `full_grid: true` gives the 80-cell Cartesian product for an
appendix. Seeds and partitions are knob-independent (§11) → along-axis
comparisons are paired.

**Method list expanded to 8 (from 4), for a stated reason.** On clean data
`alzenad2017` edges `proposed_hfl` at every N, so the paper's robustness
claim rests on showing the proposed system **degrades more gracefully**
than the placement baselines do. Without them in the grid there was no
evidence either way. Methods: `proposed_hfl`, `flat_fl`,
`hfl_no_selection`, `alzenad2017`, `mozaffari2016`, `fair_mab`, `oort`,
`fedcs`.

**`stress_selection.yaml` (new) — because the main stress grid could not
test selection at all.** `stress_test.yaml` has K·capacity = 6·10 = 60 =
N_clients, so *every* client fits in a slot and the selection rule never
has to choose whom to drop — its entire purpose. Under that config
`proposed_hfl` was statistically tied with every selection baseline simply
because selection was inert. The new config inherits everything and changes
one number — `capacity: 6` → 36 slots for 60 clients (60% servable, the
same binding ratio as `paper_full`'s N=200 cell) — so results are directly
comparable cell-for-cell while the selection mechanism is actually
exercised.

**Other corrections in this harness:** `n_seeds` 10 → 16 (§13);
`n_local_epochs`/`n_uav_epochs` 1 → 2 to match the paper recipe; `lr`
un-pinned so the tuned value is inherited (§0); `n_workers` 8 → 12 to match
the VM; `macro_f1` added to the significance step (through the 2026-07-22
run the robustness claims rested on accuracy alone, which barely separates
methods under an ~0.8 majority class); and `plot_stress` now emits
one-axis-at-a-time degradation curves with ±1 sd seed bands, sliced exactly
the way the grid was generated.

Sanity signature (verified): dropout collapses the selected-set size for
gated methods; black-chip degrades accuracy. **`flat_fl` is no longer
immune** to the dropout/SNR axes (§4).

---

## 15. Reproducibility and artifacts

- **Environment:** Python 3.13.14; `requirements.txt` = exact `pip freeze`
  closure of the results-producing machine (now including **Optuna** for
  the weight search), dev/test tooling excluded; floor pins in
  `pyproject.toml`.
- **Config inheritance:** `extends: <file>` with recursive resolution,
  deep merge (override always wins; nested mappings merge key-by-key), and
  cycle detection. This is what makes `tuned_weights.yaml` a single source
  of truth (§0) and what lets the ablation configs
  (`tier1_equal_radius`, `tier1_warmprev`, `tier1_regime_hetero`,
  `selection_isolation_starved`, `selection_scaling*`, `stress_selection`,
  `coverage_smoke`) each be a 3-line file that changes exactly one variable
  — which is also what makes them *readable as controlled comparisons*.
- **One command:** `scripts/reproduce_paper.sh [--smoke]` now chains
  Tier-1 → analyze/plot → **Tier-1 MCLP reference** → paper sim
  (+ accuracy **and** macro-F1 significance) → selection isolation
  (+ significance) → N-sweep → **coverage sweep** (+ significance) →
  **end-to-end centralized validation** → stress sweep → **selection-
  pressure stress sweep** → artifact staging, logging to
  `results/reproduce_paper.log`.
- **Resume gate — now verified, not just existence-checked.** A job used
  to be skipped whenever its resolved-config file existed. That is how
  2026-07-14 smoke leftovers ended up standing in for full paper runs, and
  it would have caused the 2026-07-23 stress `lr` fix to skip all 440 jobs
  and return the undertrained numbers unchanged. `_stale_checkpoint_reason`
  now compares a **resume signature** — the job-defining config subset
  (`data`, `fl`, `budget`, `methods`, `optimizer_params`, `epicentre`, plus
  `_pipeline_version`), normalized through a YAML round-trip so stored and
  freshly-built signatures compare on plain types, with locations and
  credentials (`hf_token`, `prebuilt`, `feature_cache_path`) excluded as
  non-semantic. A mismatch logs *which key* differs and reruns.
  **`PIPELINE_VERSION = 3`** is bumped whenever a code change alters RNG
  draw order or numerics, invalidating all older checkpoints wholesale.
  Applies to every sweep (`_job`, `_paper_job`, `_coverage_job`, stress
  `_job`); pinned by `check_sweep_resume.py`.
- **Tier-1 resumes differently, and its checkpoints are committed.**
  `runner.py` checkpoints per `{method}__s{scenario_idx}__seed{seed}.pkl`
  under `results_dir/.checkpoints/`, and those pickles are **tracked in
  git** (840 for `tier1_core`). The upside is real: the entire Tier-1 grid
  replays in ~24 s on a fresh machine — the 2026-07-27 VM run recomputed
  none of it — which makes the placement benchmark genuinely reproducible
  by anyone who clones the repo, without 5.6 core-hours. The asymmetry to
  know about: this gate is **existence-only**. It carries no config
  signature and no `PIPELINE_VERSION`, so replaying checkpoints across a
  code change is safe only to the extent that the change was verified
  bit-identical (§16). Two further properties follow from the key being
  *positional* in `scenario_idx`: appending a scenario (as `real` was)
  preserves existing keys, but **reordering or removing one silently
  remaps** every downstream index onto the wrong stored result. Append
  only.
- **Job failure isolation.** joblib's fail-fast aborted an entire sweep on
  the first job error, discarding hours of sibling compute (the 2026-07-16
  run lost ~2.5 h to one stale-cache `IndexError`). `_isolated` converts any
  job exception into a returned record; `_collect_or_raise` concatenates
  the survivors and then raises **once**, listing every failed tag with its
  full traceback. Completed jobs keep their checkpoints, so the rerun
  resumes rather than restarts. This is a deliberate trade: the sweep still
  fails loudly, just after banking the work.
- **Per-run artifacts:** resolved config YAML (prebuilt payloads elided),
  `seed_manifest.csv` (written before the run, now carrying
  `partition_seed`), rounds/runs parquet, `confusion.parquet`,
  `operational_summary.csv`, `per_class_f1.csv`, figures.
- **Sanity checks (manual):** **81 checks across 14 scripts** under
  `tests/sanity_checks/` (`run_all.py` runs them all), offline via the
  prebuilt-fixture seam, no token needed. Not pytest — a person reads the
  printed PASS/FAIL output before trusting a batch of results. New coverage
  this cycle: `check_sweep_resume.py` (stale-checkpoint gate + failure
  isolation); `clone_model` ≡ `deepcopy` **including RNG-stream
  preservation**; `fedavg`/`reputation_fedavg` in-place accumulation ≡ the
  naive formula; confusion-derived F1 ≡ sklearn exactly over 500 randomized
  trials; feature-cache staleness detection; `oort`/`power_of_choice`
  ranking, straggler penalty and exploration prior; `"all"` mode
  eligibility gating; the tensor-sliced eval path ≡ the DataLoader path.
  Fairness invariants are additionally enforced by **runtime asserts**:
  `build_optimizer` asserts the shared P/G_max budget, and Tier-1's
  `_run_one` asserts the seed streams did not collide — a violation crashes
  the run itself.
- **GCP:** `scripts/run_gcp.sh` is the only entry point; it wraps
  `reproduce_paper.sh`, then **commits and pushes `results/` to GitHub**
  before the VM self-stops. Feature caches and logs are gitignored so
  nothing staged can hit GitHub's 100 MB file limit; on any git failure it
  exits non-zero, the shutdown trap still stops the VM, the results stay on
  disk, and the log says exactly what to push manually. Active instance:
  `instance-20260715-110613` in `europe-west4-a`.

---

## 16. Engineering quality notes (methods-section adjacent)

- **Single scoring path invariant:** no optimizer implements its own
  objective; no selection mode bypasses the eligibility gate (mode `"all"`
  no longer exempt). One `build_optimizer`, one `_load_data`, one
  `_dump_resolved_cfg`, one `load_config`.
- **Single metrics home:** every reported FL metric lives in
  `uavbench/metrics/fl.py`; Tier-1 placement metrics stay numpy-only in
  `metrics/placement.py`.

  ⚠️ **The "Tier-1 never imports torch" claim is false as of 2026-07-28 —
  do not repeat it.** The *intent* holds at module level and was verified:
  `problem.fitness`, `optimizers`, and `metrics.placement` each import
  torch-free, and `metrics/__init__.py` deliberately imports only
  `placement` eagerly. But `uavbench.runner` pulls torch in anyway through
  `reporting.tables` → `uavbench.reporting.__init__` → `seed_manifest` →
  `uavbench.fl.seeds` → `uavbench/fl/__init__.py` → `federated` → torch. A
  package `__init__` re-export defeats the layering. Consequences are mild
  (import cost and OpenMP setup in each of the 8 Tier-1 workers, which do
  not call `torch.set_num_threads(1)` — only the FL sweeps do), but the
  claim as stated is wrong. Fixing it is a one-line change: import
  `uavbench.fl.seeds` lazily inside `build_seed_manifest`.
- **No silent failures in the results path:** every harness writes tables
  through `reporting/tables.py::write_table` — the pyarrow-missing CSV
  fallback logs at WARNING and deletes the stale/partial parquet. The
  black-chip gate (§2), the feature-cache shape check (§2), the resume
  signature (§15), and the equal-radius warning (§7) all apply the same
  rule: a state that would corrupt a reported number stops or flags itself
  rather than continuing quietly. Plot generation is the one tolerated soft
  failure (figures are regenerable from the saved parquet).
- **Bit-identical optimization discipline.** Two performance passes
  (`8391c208`+`431a869f`, then `9aa933ed`) rewrote the hottest paths —
  functional forward, `clone_model`, `MomentumSGD`, `clip_grad_norm_`,
  `_weighted_accumulate`, the EMA/server-momentum blends, tensor-sliced
  batching and evaluation, `distances`/`distances_batch`,
  `greedy_assignment{,_batch}`, confusion-derived F1 — **every one verified
  bit-identical** to the code it replaced, with the verification pinned as
  a sanity check. Three rules were followed and are worth stating as
  methodology, because they are what make "we optimized the code" a
  non-event for reproducibility:
  1. *Never reassociate a reduction.* `Fitness.batch` deliberately keeps
     its scalarization tail as a per-candidate scalar loop; vectorizing it
     would shift results by ~1 ULP. `cov_rows` uses `(x*scarcity).sum(1)`
     rather than `rows @ scarcity` for the same reason (BLAS reassociates).
  2. *Never consume RNG you didn't consume before.* `clone_model`
     snapshots and restores torch's global RNG around module construction;
     the reusable scratch models are built **before** any per-method
     `torch.manual_seed`; the roster sampler has its own generator.
  3. *Refuse rather than diverge.* `MomentumSGD` is restricted to the exact
     configuration it can reproduce exactly and raises on a missing
     gradient instead of silently differing; `_make_optimizer` falls back
     to `torch.optim` for everything else.
- **The 25× `CachedDataset` bug is fixed and must stay fixed:**
  `__getitem__` must never call `base[idx]`, which decodes a TIF chip from
  disk that the wrapper immediately discards.
- **Vectorization:** `haversine_matrix` (exact vectorized twin of the
  scalar, equivalence-pinned) replaced Python double loops — ~17× on
  `_covered_clients` at N=500, K=20 with byte-identical output.

---

## 17. Mandatory paper statements — checklist

1. **Tuned constants disclosure (§0)** — which values deviate from the
   paper text, the search objective they were fit to, the objective
   coupling, and that the weights are applied identically to all methods.
   *This is new and non-optional.*
2. **Single-event scope** as a limitation; the two stress sweeps as the
   controlled robustness complement (§2, §14).
3. **Real-data-only pipeline**; the test fixture never touches results (§2).
4. **Black-chip rate** quoted next to accuracy results (§2, §12).
5. **Frozen-feature cost** — the measured **80.3% macro-F1 retention**
   from `e2e_centralized.py`, framed as a feasibility trade-off, not as a
   free simplification, and used to set the ceiling against which the
   federated numbers are read (§3).
6. **Equal-radius scoring** for the placement-literature baselines: the
   Tier-1 correction from 100 dB (r*=618 m) to 95 dB (r*≈499 m ≈ R_comm),
   the 145 dB Noto-scale calibration, the per-job re-calibration in the
   coverage sweep, and `uniform_coverage_radius` (§7). State that the
   earlier 618-vs-500 m asymmetry, not the placement rule, produced the
   baselines' apparent uniform-scenario win.
7. **Adaptation notes** for Mozaffari (K-disc greedy maximal covering),
   Alzenad (per-cluster decoupling, centroid vs SEC), and each selection
   baseline's instantiation constants — including **Oort's `T_pref` =
   eligible-pool median** and **Power-of-Choice's cached-loss variant**
   (§7, §8, Appendix A.6–A.7, Appendix C).
8. **`hfl_static` is a cadence ablation, not a bad-placement baseline** —
   and `centroid_place`/`random_place` are what supply that contrast (§7,
   §10).
9. **Paired-test justification** via the shared-instance *and shared-
   partition* seed design; Holm correction; the **Wilcoxon p-floor 2/2ⁿ**
   and the resulting seed counts (10 / 16) and `--reference` +
   `--correction-scope group` design; the `--last-k 10` reporting statistic
   (§11, §13).
10. **Effect sizes (rank-biserial) and bootstrap CIs** reported with every
    p-value (§13).
11. **Separate tables** for own-ablations vs selection literature vs
    placement literature (§10).
12. **Load imbalance ≠ selection fairness** — both reported, labeled as
    assignment balance vs selection-frequency fairness (§12).
13. **Selection-at-scale sensitivity**: the matched-data (subsample 1.0)
    headline *and* the starved (0.1) regime, with the samples-per-client
    table and the ρ=0.60 control showing selection pressure does not
    explain the flip (§2).
14. **Communication accounting caveat** (§5 OPEN ITEM) — or fix
    `round_comm_mb` and drop this item.
15. **MCLP near-optimality**: PSO recovers **99.3%** of the MILP-optimal
    covered value (all reference solves proved optimal), with the ≤1.1%
    overshoot explained as grid-discretization slack (§7, Appendix A.9).
16. **PSO ties GA** on the `uniform` and `real` Tier-1 scenarios — state it;
    the placement claim is over non-searching baselines (§7).
17. **Wall-clock numbers** behind the CPU-feasibility claim; hardware
    disclosure (§15, §18).
18. **Data availability + GSI licensing statement** (§2).

---

## 18. Hardware and runtime disclosure

### Machines

| Role | Machine | CPU | RAM | Notes |
|---|---|---|---|---|
| Development / smoke runs | Dell Latitude 5540 | Intel i7-1355U (10c/12t) | 32 GB | CPU-only; all harnesses runnable |
| Full experimental grids | GCP `n1-standard-12` (`instance-20260715-110613`, `europe-west4-a`) | 12 vCPU | 45 GB | via `scripts/run_gcp.sh` (self-terminating; wraps `reproduce_paper.sh`; pushes results before stopping) |

No GPU is used anywhere: the CPU-feasibility claim is backed by measured
wall-clock numbers, not an architectural argument.

### Parallelism per harness

Each sweep worker pins `torch.set_num_threads(1)` so total active threads
= `n_workers` × 1 (no BLAS thrash). One exception is deliberate:
`data.feature_num_workers` (default 0, set to **8** in the paper configs)
parallelizes TIF decode for the **one-time** ResNet feature pass, which
runs in the *sequential prefetch phase* before the sweep workers fork — so
it never multiplies processes.

| Config | Harness | n_workers |
|---|---|---|
| `configs/tier1_core.yaml` | Tier-1 placement grid | 8 |
| `configs/paper_full.yaml` | full paper sim (13 × 4 N × 10 seeds) | 12 |
| `configs/paper_coverage.yaml` | coverage-constrained sweep | 12 |
| `configs/selection_isolation.yaml` | selection isolation | 12 |
| `configs/tier2_sweep.yaml` | N-scalability sweep (6 N: 30…250) | 8 |
| `configs/stress_test.yaml`, `stress_selection.yaml` | stress sweeps | 12 |
| `configs/paper_smoke.yaml` | recipe smoke | 6 |

`UAVBENCH_N_WORKERS` overrides the Tier-1 worker count per machine.

### Where the timing numbers come from

- **Per optimizer run** — `wall_time_s` in Tier-1 `runs.parquet` (timed
  around `Optimizer.optimize`), also surfaced in `component_summary.csv`.
- **Per FL round** — `round_time_s` in every rounds table.

The per-method aggregate (mean/std/total seconds) is printed by
`uavbench analyze` / `run_tier2` / `run_paper_sim` / `run_stress_sweep`
via `uavbench.reporting.summarize_wall_clock`, and should be quoted in the
paper's runtime disclosure.

### Measured compute (from the committed result tables)

These are **core-hours** — the sum of the instrumented per-unit timings
(`round_time_s` for FL harnesses, `wall_time_s` for Tier-1) across all
jobs. They are *not* wall-clock: the sweeps run `n_workers` jobs
concurrently, so wall-clock ≈ core-hours / `n_workers` plus the sequential
prefetch phase. Quote core-hours as the honest measure of compute consumed
and derive wall-clock only where `results/reproduce_paper.log` confirms it.

| Harness | Grid actually run | Core-hours | ≈ wall-clock at its `n_workers` |
|---|---|---|---|
| Tier-1 core | 7 methods × 4 scenarios × 30 seeds = 840 runs | **5.6** | ~0.7 h (8 workers) |
| Paper full sim | 13 methods × 4 N × 10 seeds = 520 jobs | **295.3** | ~24.6 h (12) |
| Coverage sweep | 5 methods × 6 R_comm × 10 seeds = 300 jobs | **141.7** | ~11.8 h (12) |
| Selection isolation (starved) | 7 modes × 6 N × 10 seeds = 420 jobs | **65.9** | ~5.5 h (12) |
| Stress-test sweep | 8 methods × 11 cells × 16 seeds = 1408 jobs | **63.5** | ~5.3 h (12) |
| Selection-pressure stress | same grid | **85.4** | ~7.1 h (12) |
| **Total (committed runs)** | | **≈657 core-h** | |

Two things this table is good for beyond disclosure. First, **it is the
CPU-feasibility evidence**: a 520-job, 13-method, full-dataset federated
grid at 100 rounds costs ~295 core-hours on commodity vCPUs with no GPU
anywhere — that is the number to quote, not an architectural argument.
Second, it prices the outstanding runs (§20): the matched selection
isolation is ~66 core-h by analogy with the starved grid, and the two
missing coverage arms are 120 of 420 jobs ≈ 57 core-h.

Still to fill from `results/reproduce_paper.log` once a full pipeline runs
end to end: true wall-clock including the sequential data/feature prefetch,
which none of the above captures.

---

## 19. Known-void results (do not quote)

Every entry here is a result that *looked* complete and was wrong. Keep the
list; it is also the honest answer if a reviewer asks how the pipeline was
validated.

| Window | What is void | Cause |
|---|---|---|
| before 2026-07-24 | Tier-1 tables where physics baselines beat PSO on uniform | 618 m vs 500 m coverage radius (§7) |
| before 2026-07-23 | any `movement_battery_frac` / Tier-1 energy figure | fleet-summed distance charged to one battery (§5) |
| the 2026-07-22 stress grid (440 jobs) | all stress robustness numbers | `lr` pinned 17.75× below the tuned value (§0) |
| 2026-07-22 & 2026-07-23 stress significance | "no significant differences" | family of 308 vs a 10-seed p-floor — mathematically impossible (§13) |
| before 2026-07-26 | all selection-isolation results | proposed selector ran with class-coverage disabled (§8) |
| `e79701f5` → 2026-07-26 | any `run_tier2` placement result | `uniform_coverage_radius` referenced but undefined → `NameError` (§7) |
| before 2026-07-18 | `centralized` confusion matrices; `rounds_to_target` anywhere | silently dropped; single-round crossing (§12) |
| before 2026-07-18 | `flat_fl` stress curves on dropout/SNR | mode `"all"` bypassed the eligibility gate (§4) |
| any checkpoint below `PIPELINE_VERSION = 3` | everything | RNG/numerics changed; the resume gate now refuses them (§15) |

---

## 20. Open items

Ordered by what blocks the paper. **Items 1 and 2 are already running on
the VM** (§10.1) — they need no action, only patience and a provenance row
when they land. Item 3 is the only missing run nobody has launched.

> ⛔ **Do not land items 4 or 6 while the VM run is in flight.** The resume
> gate keys on the *config* signature plus `PIPELINE_VERSION` — it does not
> hash the source. A change to `round_comm_mb` (item 4) or to the selector's
> RNG seeding (item 6) that is pushed mid-run would be picked up by the
> VM's next `git pull --rebase` and produce jobs whose numbers disagree with
> their already-completed siblings *inside the same parquet*, with nothing
> flagging it. That is precisely the failure class §15 was built to prevent,
> and it would be self-inflicted. Either wait for the VM to self-stop, or
> bump `PIPELINE_VERSION` with the change and accept a full recompute.
> Items 7 (log string) and 8 (provenance rows) touch no computed value and
> are safe to land now.

1. 🟡 **Matched-data selection isolation — IN FLIGHT** (§10.1). Running now
   at `subsample: 1.0`, ~177/420 jobs, ETA ≈ 2026-07-29 14:30 UTC. Until it
   lands, the only committed evidence is the *starved* grid, so the
   defensible statement today remains the negative one (the proposed rule
   degrades under data starvation, with the class-histogram mechanism).
   Every "+0.041 vs Oort at N=500" claim stays a YAML comment until this
   parquet exists — **do not write it up early.**
2. 🟡 **`centroid_place` / `random_place` — IN FLIGHT** (§10.1). Queued
   behind item 1 in the same pipeline; the VM's `paper_coverage.yaml`
   already lists all 7 methods and the resume gate will compute only the
   120 new jobs. Until then, `hfl_static` has no naive-placement foil and
   the "cadence, not placement quality" argument rests on reasoning alone.
3. 🟢 **The Tier-1 supplements predate the `real` scenario — and closing it
   costs ~25 minutes on a laptop.** Verified uncovered four ways:
   (a) `TIER1_CFG=configs/tier1_core.yaml` is a hardcoded assignment in
   `reproduce_paper.sh` (not `${VAR:-default}`) and no script anywhere
   references the supplement configs; (b) local supplement `runs.parquet`
   carry 3 scenarios; (c) the VM's carry 3 scenarios with mtime
   2026-07-24 12:02, untouched by the 2026-07-27 run; (d) their
   `.checkpoints/` hold exactly 7×3×n_seeds entries — no `real` jobs exist.

   **But this is cheap, because Tier-1 `.checkpoints/` are tracked in git**
   (840 pickles for `tier1_core` alone). Two consequences worth knowing:
   the VM's 24-second Tier-1 step *never recomputed anything* — it replayed
   git-pulled checkpoints; and a supplement rerun reuses the committed
   3-scenario jobs and computes only the `real` ones — 7×(30+20+20) = **490
   runs ≈ 3.3 core-h ≈ 25 min on 8 workers**. The checkpoint key is
   `{method}__s{scenario_idx}__seed{seed}`, positional in `scenario_idx`,
   and `real` is listed **last** in `tier1_core.yaml` (index 3), which the
   supplements inherit via `extends` — so the existing s0/s1/s2 keys stay
   valid and nothing is silently remapped.

   It needs no VM and does not disturb the in-flight run (Tier-1 is
   numpy-only in its compute path and writes to its own results dirs; it
   does transitively import torch via `reporting` — see §16 — but never
   uses it, so it neither competes for BLAS threads nor touches FL state).
   The `real` scenario's data dependency is satisfied locally: the metadata
   cache it selects is `.metadata_df_cache_sub10000_seed42.parquet`, the
   same full-dataset file the VM used.

   *Caveat to record if the supplements are rerun:* the Tier-1 checkpoint
   gate is **existence-only** — unlike the FL sweeps it has no
   config-signature or `PIPELINE_VERSION` check (§15). Reusing 2026-07-24
   checkpoints under 2026-07-27 code is therefore safe *only because* the
   intervening perf passes were verified bit-identical (§16). That is a
   real dependency on that verification discipline, not a free lunch.
4. **`round_comm_mb` is not block-aware** (§5 OPEN ITEM). Fix it and
   regenerate the round tables (training is untouched, so no checkpoint is
   invalidated), or state the conservative-upper-bound caveat.
5. ~~**`ema_decay` unvalidated**~~ — **CLOSED 2026-07-28.** The 80-round
   validation had already been run; 0.9 is within noise of the optimum and
   only 0.99 collapses (§0). Needs a `results_provenance.md` row (item 8),
   nothing more.
6. **`run_selection_isolation` constructs its `ClientSelector` without a
   seed**, so the Gumbel roster sampler runs at the default `seed=0` there
   while `run_full_hfl` seeds it per method. Deterministic and identical
   across modes, so it does not break the isolation logic — but it should
   be seeded from the job seed, and the discrepancy should not be
   discovered by a reviewer.
7. **`run_full_hfl`'s completion log prints the wrong target** (§12) — a
   log-string bug only; the parquet column is right.
8. **`results_provenance.md` stops at 2026-07-24.** Missing rows: the
   coverage sweep, both 16-seed stress grids, the corrected/renamed
   selection isolation, the MCLP reference, and the end-to-end validation.
   No number from those runs goes in the paper until its row exists.
9. ~~**Wall-clock table unfilled**~~ — **CLOSED 2026-07-28.** Filled from
   the committed result tables: ≈657 core-hours total, 295 of them the
   paper sim (§18). Only true end-to-end wall-clock including the
   sequential prefetch is still missing, and that arrives with the next
   full pipeline run.

---

## 21. Legacy code: the second simulator (`hflsim` CLI)

`src/hflsim/simulation/__init__.py` says *"See
REPORTS/master_implementation_reference.md for the migration story."* Until
2026-07-28 this document contained no such story. It does now.

**What exists.** Alongside `uavbench`, the repository contains a complete
**second, older simulator** — ~1,100 lines, ~10% of `src/`:

| Module | Lines | What it is |
|---|---|---|
| `hflsim/cli.py` | 363 | standalone single-N HFL runner with its own argparse surface |
| `hflsim/models/fusion.py` | 129 | `MultiModalFusionModel` + `FocalLoss` — a **different architecture** from `CachedFusionModel` (§3) |
| `hflsim/simulation/{orchestrator,client,coordinator,uav}.py` | 646 | its own round loop, client, UAV aggregator, and selection coordinator |

**It is an installed entry point.** `pyproject.toml` declares
`hflsim = "hflsim.cli:main"` next to `uavbench`, so `hflsim --N 70` and
`python -m hflsim` both work after `pip install -e .`, including a
`--use_pso_placement` flag.

**It shares nothing with the experimental pipeline.** Verified: no module
under `hflsim/cli.py` or `hflsim/simulation/` imports `uavbench` at all. It
therefore bypasses every invariant this document describes — no shared
`Fitness`, no eligibility gate, no `round_comm_mb` accounting (0
references), no `seed_manifest.csv` (0), no job checkpointing (0), no
resolved-config dump (0), no black-chip hard gate. It writes
`simulation_results_N{N}.csv` plus its own PDF/PNG figures.

**Why this matters, stated plainly.**

1. **§16's "single scoring path" invariant is scoped to `uavbench`, not to
   the repository.** Say so when claiming it. Within `uavbench` the
   invariant is real and enforced; `hflsim` is simply outside it.
2. **`results/simulation_results_N14.csv` and `_N35.csv` came from this
   simulator** (`results_provenance.md` records them as legacy, produced by
   the pre-`uavbench` code). They are *not* comparable to any current
   number — different model, different loop, different accounting — and
   nothing in the paper should cite them.
3. **It is a live footgun.** Because it is installed and runnable, a
   collaborator can produce plausible-looking `simulation_results_*.csv`
   files that satisfy none of the reproducibility guarantees in §15 and
   carry no provenance. The `simulation/__init__.py` header already warns
   "Do NOT build new experiments on these classes"; the CLI itself carries
   no such warning.

**Recommendation.** Keep `hflsim.data` (the loader **is** live — §2,
Appendix D), `hflsim.shared` (coords/value, live), and `hflsim.placement`
(the bridge). For the rest, either delete `hflsim/cli.py`,
`hflsim/models/`, and `hflsim/simulation/` (retaining only `UAVAggregator`
for the bridge), or at minimum drop the `hflsim` console-script entry from
`pyproject.toml` so the legacy path cannot be invoked by accident. Deleting
is cleaner and costs nothing: nothing in `uavbench`, `scripts/`, or
`tests/` depends on it.

---

# Appendix A — Optimizer mechanics in detail

Line-level reference for the placement optimizers, read directly from
`src/uavbench/optimizers/` and `src/uavbench/problem/`.

## A.1 Encoding and fitness internals

Each particle/individual is a flat vector **X ∈ ℝ³ᴷ** — K UAV positions
concatenated as (x₁,y₁,z₁,…,x_K,y_K,z_K); per-dimension bounds are the area
box tiled K times; `positions_from_vector` reshapes back to (K,3). A nuance
of the imbalance term: `L_imb` uses `n_assigned/K` (the mean load *among
served devices*), not `N/K`, so it penalizes imbalance among served devices
only. The theoretical fitness ceiling is `F ≤ w1` (full coverage, zero
movement, zero imbalance) — **0.811 under the searched weights, 0.6 under
the paper's**; PSO and GA both early-stop at `0.95·w1`. `Fitness` tracks an
`eval_count` (incremented by P in `batch`) so the runner can verify every
optimizer spends the identical budget. Assignment cost per evaluation:
`O(N log N)` sort (now hoisted to `instance.value_order`) + `O(N·K)` sweep,
with the `(N,K)` distance matrix vectorized.

## A.2 PSO (`optimizers/pso.py`)

**Constriction factor**, derived (never hardcoded) from `c1, c2`:

```
phi = c1 + c2                      # must be > 4; raises ValueError otherwise
chi = 2 / |2 − phi − sqrt(phi² − 4·phi)|
```

Defaults `c1 = c2 = 2.05` → `phi = 4.1` → `chi ≈ 0.7298`. Deriving χ at
construction means an ablation that changes the acceleration coefficients
cannot silently break the convergence guarantee.

**Velocity update** (constriction mode, default):
```
V ← chi · (V + c1·r1⊙(pbest − X) + c2·r2⊙(nbest − X))
```
**Inertia-mode fallback** (`use_constriction=False`, ablation only):
```
w = inertia_max − (inertia_max − inertia_min) · (tau / G_max)   # 0.9 → 0.4
V ← w·V + c1·r1⊙(pbest − X) + c2·r2⊙(nbest − X)
```

**Ring topology** (default): each particle's neighborhood best is the best
`pbest` among `2·ring_k+1` particles on a ring (`ring_k=2` → 5-particle
neighborhoods), fully vectorized; information about a good region spreads
gradually, preserving diversity on a multimodal coverage landscape.
`topology="gbest"` shares one global best instead.

**Initialization** (`seeding="value_kmeans"`, default): half the swarm
seeded by value-weighted k-means++ centers over device (x,y), jittered
N(0, jitter_m=10 m), uniform random altitude; the other half uniform.
`"plain_kmeans"` drops value weighting; `"uniform"` seeds all uniformly.
Initial velocity `0.5·U(−vmax, vmax)`.

**Safeguards:** per-dimension velocity clamp `vmax[d] =
vmax_frac·(hi[d]−lo[d])` (default 0.2); **absorbing walls**;
**turbulence** (`p_turb=0.1` of particles get a kick ~U(−0.1·vmax,
0.1·vmax) before clamping); **stagnation reinit** (gbest improvement <
`delta_stag=1e-4` for `G_stag=20` consecutive generations → worst
`floor(rho·P)` particles replaced with fresh uniform samples, velocities
and pbests reset, gbest re-checked). `gbest` updates only on strict
improvement. Early stop once `gbest ≥ early_stop_frac·w1`.

**Full loop** (all population scoring now via `fitness.batch`, §7):
```
1. Bounds ← tile(lower, upper) K times;  vmax ← 0.2·(hi−lo)
2. Init: P/2 value-weighted k-means++ + P/2 uniform;  Vel ~ 0.5·U(−vmax,vmax)
3. pbest_fit ← fitness.batch(X); set pbest, gbest
4. For tau = 1..G_max:
   a. nbest ← ring-neighborhood best
   b. V ← chi·(V + cognitive + social); turbulence kicks; clamp to ±vmax
   c. X ← X+V; absorbing walls
   d. fit ← fitness.batch(X); update pbest, gbest (monotonic)
   e. Stagnation counter → reinit worst 20% at G_stag=20 (re-scored by batch)
   f. Early stop if gbest ≥ 0.95·w1
5. Return gbest position/fitness/convergence/eval_count/chi/phi
```

**Default hyperparameters:**

| Parameter | Default | Meaning |
|---|---|---|
| `P` | 100 (Tier-1); 50 (full sim) | swarm size |
| `G_max` | 200 (Tier-1); 30 (full sim) | max generations |
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
(`tournament_size=3`), elitism (`n_elite=2`), same `0.95·w1` early stop,
same P/G_max budget as PSO by construction. Population scoring via
`fitness.batch`.

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
  evaluation; `value_weighted=False` toggles unweighted centroids. Exposed
  as the full-system arm `centroid_place`.
- **`RandomPlacement`** — best of `n_draws=20` uniform candidates. Exposed
  as `random_place`.
- **`Static`** — stays at `instance.prev_positions`; one evaluation.
- **`kmeanspp_centers(rng, points, K, weights)`** — k-means++ with optional
  value-proportional sampling (first center ∝ weights, subsequent ∝
  D²·weights; uniform fallback on zero/non-finite weight).
- **`weighted_kmeans(...)`** — Lloyd's algorithm seeded by the above,
  weighted centroid updates, ≤25 iterations, early convergence stop.

## A.5 Placement literature baselines: shared channel-model sourcing

Both A.6 and A.7 build on the probabilistic air-to-ground channel in
`problem/path_loss.py`, which draws on **two** distinct Al-Hourani et al.
papers cited separately (not a single merged "2014" source):

- Al-Hourani, Kandeepan & Jamalipour, "Modeling Air-to-Ground Path Loss for
  Low Altitude Platforms in Urban Environments," IEEE GLOBECOM 2014 —
  source of the environment shape constants and, confirmed exactly, the
  `eta_los_db`/`eta_nlos_db` excess-loss values (that paper's `mu_1`/`mu_2`,
  Table II, 2000 MHz row: Suburban 0.1/21, Urban 1.0/20, Dense Urban
  1.6/23, Highrise 2.3/34 dB).
- Al-Hourani, Kandeepan & Lardner, "Optimal LAP Altitude for Maximum
  Coverage," IEEE WCL 2014 — source of the S-curve LoS-probability fit and
  the altitude/coverage-radius optimization; its own free-space term is
  evaluated at the 3-D slant distance `d = sqrt(h² + r²)`, which
  `average_path_loss`'s `d3d = hypot(distance_ground_m, altitude_m)`
  matches exactly (confirmed, not the horizontal-only distance).

The `(a, b)` LoS shape constants are the standard literature-cited
Al-Hourani values, used as-is rather than re-derived from the WCL letter's
own bivariate surface fit — a stated simplification. Radius-vs-altitude is
unimodal (rises while LoS gain dominates, falls when FSPL dominates),
verified numerically; `path_loss_db_for_radius` (§7) inverts the
radius-maximizing map by bisection, exploiting monotonicity in the link
budget.

## A.6 Placement baseline — Mozaffari2016

Implemented in `optimizers/mozaffari2016.py`; full-system method
`mozaffari2016` (own placement + `ucb` selection).

**Citation:** M. Mozaffari, W. Saad, M. Bennis, and M. Debbah, "Efficient
Deployment of Multiple Unmanned Aerial Vehicles for Optimal Wireless
Coverage," *IEEE Communications Letters*, vol. 20, no. 8, pp. 1647–1650,
Aug. 2016.

**Core idea (source):** derive the single-UAV altitude/radius pair (h*, r*)
that maximizes ground coverage radius under the air-to-ground path-loss
model, then deploy multiple UAVs as equal-radius discs.

**Adaptation note (state explicitly):** the source's core result is the
single-UAV (h*, r*) derivation; its multi-UAV deployment packs equal discs.
Here that packing is realized as **greedy maximal covering discretized at
device locations** — a deterministic K-UAV generalization aligned with this
benchmark's value-weighted coverage objective, not a verbatim reproduction
of the paper's own packing. A residual-centroid rule was rejected: it
degenerates on clustered layouts, where the centroid can sit farther than
r* from every cluster and cover nothing.

```
Algorithm A6: Mozaffari-style Greedy Equal-Disc Coverage
1: (h*, r*) ← optimal_altitude_mozaffari(max_path_loss_db, freq, env)   # grid search
2: remaining ← all devices;  centers ← []
3: for k = 1..K do
4:     if remaining is empty:
5:         centers.append(last center, or device centroid if none yet)   # co-locate spare
6:         continue
7:     j* ← argmax_j  sum_{i in remaining} value_i · [dist(i, j) <= r*]  # candidate = device j
8:     centers.append(position(j*))
9:     remaining ← remaining \ { i : dist(i, j*) <= r* }
10: positions ← [(centers[k], h*) for k in 1..K];  radii ← [r*]*K
11: return positions, radii   # scored once through shared Fitness with `radii` applied
```

**Expected contrast:** a physically grounded, closed-form-altitude
baseline, but altitude is uniform and radius fixed regardless of local
device density — it cannot trade a smaller radius for tighter clustering
the way value-weighted metaheuristic search can, and it is blind to the
heterogeneous-fleet capacity structure (§7).

## A.7 Placement baseline — Alzenad2017

Implemented in `optimizers/alzenad2017.py`; full-system method
`alzenad2017` (own placement + `ucb` selection).

**Citation:** A. Alzenad, A. El-Keyi, F. Lagum, and H. Yanikomeroglu, "3-D
Placement of an Unmanned Aerial Vehicle Base Station (UAV-BS) for
Energy-Efficient Maximal Coverage," *IEEE Wireless Communications Letters*,
vol. 6, no. 4, pp. 434–437, Aug. 2017.

**Core idea (source):** decouple 3-D placement into a 2-D horizontal step
and an altitude step (the *minimum* altitude whose path-loss-derived radius
still reaches the required coverage circle — the energy-efficient choice).

**Adaptation note (state explicitly):** the source places a single UAV-BS;
here devices are partitioned into K value-weighted k-means clusters and the
decoupled rule is applied independently per cluster, yielding genuinely
non-uniform per-UAV radii. The 2-D step uses the value-weighted cluster
centroid rather than the paper's smallest-enclosing-circle (SEC) center;
whether an exact SEC center (Welzl) would materially tighten the required
radius is not evaluated (a stated, not verified, deviation).

```
Algorithm A7: Alzenad-style Per-Cluster Decoupled 2-D + Altitude Placement
1: centers ← weighted_kmeans(devices, K, value)              # value-weighted k-means
2: assign each device to its nearest center → K clusters
3: for k = 1..K do
4:     req_r_k ← max_{i in cluster k} dist(i, centers[k])     # required coverage radius
5:     (h_k, r_k) ← min_altitude_for_radius(req_r_k, max_path_loss_db, freq, env)
6: positions ← [(centers[k], h_k) for k in 1..K];  radii ← [r_k for k in 1..K]
7: return positions, radii   # scored once through shared Fitness with per-UAV `radii`
```

**Expected contrast:** genuinely non-uniform per-UAV radii and altitude
tuned to each cluster's footprint, but still centroid-based rather than
value-weighted-optimized — it cannot trade off movement penalty or load
balance the way PSO/GA can.

## A.8 Presenting placement baselines in the paper

- **Signal table:** one row per placement arm with columns *Altitude rule*,
  *Radius uniformity*, *2-D placement rule*, *Repositioning/movement-
  aware?*, *Capacity/value-aware?* — Mozaffari (uniform radius, greedy
  covering, no repositioning-awareness), Alzenad (non-uniform per-cluster
  radius, centroid-based), `centroid_place` (value-weighted centroids, no
  physics), `random_place` (floor), `hfl_static` (**PSO, one-shot — a
  cadence arm**), Proposed PSO/GA (value-, movement-, load- and
  class-aware search).
- **Results:** placement literature in a *separate* table/figure from the
  own-system ablations (§10), and the coverage sweep as the figure that
  shows when placement quality actually moves macro-F1.
- **Related Work:** for each citation add one sentence noting "we implement
  this as a baseline in §V".

## A.9 MCLP coverage optimality reference (`problem/exact.py`, new)

**What it is.** A capacitated maximum-covering location problem solved
exactly with SciPy's bundled HiGHS MILP (no extra dependency): over a
candidate grid of UAV sites, choose K sites and assign covered clients
(each site serving at most `capacity`) to maximize covered value.

**Why it belongs in the paper.** Every other placement number is relative —
"PSO beats GA", "PSO beats Mozaffari". This one is absolute: *how close to
optimal is PSO?* Because the fitness is coverage-dominated (w1 = 0.811),
covered value is the term that matters, and because the candidate sites are
a finite grid, the MCLP optimum is a **lower bound** on the continuous
optimum. With a dense grid it closely approximates it, so **"PSO reaches
X% of the MCLP coverage" is a rigorous near-optimality statement** — and
the bounding direction is the conservative one (the true optimum is at
least this high, so the reported % is not flattering).

**Formulation.** Binary `y_j` (site j opened), binary `x_ij` (client i
served by j); minimize `−Σ_i value_i · Σ_j x_ij` subject to
(A1) `Σ_j x_ij ≤ 1` per client, (A2) `Σ_i x_ij − cap·y_j ≤ 0` per site
(capacity + open-site link), (A3) `Σ_j y_j ≤ K`; `x_ij` fixed to 0 outside
the coverage radius via its upper bound. Defaults: `grid_res = 20` (400
sites), `time_limit = 120 s`, `mip_rel_gap = 1e-4`, `max_clients = 100000`
(deterministic subsample above that, with the covered fraction reported on
the solved subset). Radius: `max(radii)` if given, else `instance.R_comm`.
`MCLPResult.optimal` records whether HiGHS *proved* optimality within the
time limit — report it, and do not claim optimality for rows where it is
False.

**CLI:** `uavbench mclp --config configs/tier1_core.yaml --n-seeds 3` →
`mclp_reference.csv` with `mclp_cover_norm`, `pso_cover_norm`,
`pct_of_optimal`, `mclp_optimal`, `n_sites`. Wired into
`reproduce_paper.sh`. Three seeds per scenario is deliberate — it is a
reference point, not a distribution.

**Measured result** (`results/tier1_core/mclp_reference.csv`, 4 scenarios ×
3 seeds, 400 candidate sites, **all 12 rows proved optimal by HiGHS**):

| Scenario | PSO coverage as % of MCLP optimum |
|---|---|
| uniform | 99.35 |
| clustered | 98.79 |
| epicenter_biased | 100.01 |
| real | 99.18 |
| **overall** | **99.33** (range 96.44 – 101.10) |

**PSO recovers ~99% of the MILP-optimal covered value.** Two things to say
about the values above 100%: they are not an error and they are not PSO
beating the optimum. The MCLP is restricted to a 20×20 grid of candidate
sites, so its optimum is a *lower bound* on the continuous optimum; PSO
places off-grid and can therefore exceed it. Empirically that overshoot is
≤1.1%, which is itself the useful finding — **it bounds the grid
discretization error**, and so confirms that "99.3% of the grid optimum" is
a tight statement about the continuous problem rather than a loose one.

---

# Appendix B — Client selection and reputation in detail

Line-level reference for `src/uavbench/fl/client_selection.py`,
`device_state.py`, `reputation.py`, and their wiring in `federated.py`.

## B.1 The proposed pipeline

```
1. Eligibility gate    : battery ≥ B_min, SNR ≥ SNR_min, memory OK, time ≤ T_max − ε
                         (EVERY mode, including "all")
2. Priority score      : P_n = w_ℓ·(1 − (T̂_n/T_max)²) + w_U·Ũ_n
                         with Ũ_n = β(t)·Û_n + (1 − β(t))·R_n
                         w_ℓ = 0.702, w_U = 0.298   (battery is gate-only)
3. UCB exploration     : static_n = P_n(t) + C·√(ln t / (N_n(t)+1)),  C = √2
4. Class-coverage roster (per UAV, capacity slots):
   value(S) = Σ_c scarcity_c · √(Σ_{i∈S} count_c(i))          # submodular
   pick argmax over feasible unassigned i of
       (value(S∪{i}) − value(S))/gain₀  +  0.435·minmax(static)_i  +  1.475·G_i
   with G_i ~ Gumbel(0,1) from the selector's own RNG
   (feasible = in range of that UAV; ties → first index, matching the
    strict `>` scan it replaced)
```

**Utility details:** U_epi caps epicentre distance at the eligible-set 95th
percentile (one outlier can't squash everyone); U_SNR is min-max over
snr_db clipped to [0,30] dB; U_dens counts other eligible clients within
5 km (vectorized, local equirectangular frame), min-max normalized;
U_prox = clip(1 − d_min/R_comm, 0, 1), defaulting to 0.5 when no UAV
coordinates are supplied. `_minmax` has a degenerate-input guard: if all
values are within 1e-10, everyone gets exactly 0.5. Squaring T̂/T_max makes
the speed penalty grow superlinearly near the deadline. Reputation defaults
to 0.5 for unscored clients; `t` is floored at 1 so ln(t) never sees 0. UCB
selection counts persist in the `ClientSelector` across the run; the roster
builder increments them.

**`"random"` mode** (`hfl_no_selection`): eligibility gate, then per-UAV
uniform draw of `min(capacity, |bucket|)` without replacement, using the
caller-supplied rng. Does not increment UCB counts. **`"all"` mode**
(`flat_fl`, `centralized`): every **eligible** covered client (§4).

## B.2 Device state (`device_state.py`)

Init per client: battery ~ U(0.5,1.0); snr_base ~ U(5,20) dB; memory_ok ~
Bernoulli(0.90); compute_base ~ U(50,250) s. Per round (end-of-round, given
the actually-selected set): selected −0.02 battery, unselected **+0.01**
(clipped [0,1]); fresh N(0,2) dB SNR noise and N(0,30) s compute noise for
everyone; selected clients' observed compute time (`max(10, base+noise)`)
appended to a rolling 10-round history. Adaptive margin ε_n = 1.96·std(last
10 observed times) once ≥3 observations (else 0). Eligibility = all four
hard gates: battery ≥ 0.20, snr ≥ 3.0 dB, memory_ok, compute ≤ 300 − ε_n.

## B.3 Reputation formulas (`reputation.py`)

- **R_contrib** — per-client EMA of the update-delta vector,
  `Δw̄(t) = 0.7·Δw(t) + 0.3·Δw̄(t−1)`;
  `R_contrib = (1 + cos(Δw̄(t), Δw̄(t−1))) / 2`; cold start or near-zero-norm
  EMA (denom ≤ 1e-12) → 0.5. (The EMA's norm is cached between rounds; the
  cached value is bit-exact, since recomputing it on the unchanged array
  returns the identical float.)
- **R_anomaly** — global per-parameter mean/variance EMA (`_STATS_ALPHA=0.1`)
  across all clients; per update vector v (dim J): `z = (v − mean)/sqrt(var
  + eps)`, `d = sqrt(mean(z²))` (Mahalanobis/√J, dimension-independent);
  `R_anomaly = 1` if `d ≤ 2`, else `exp(−0.5·(d−2))`.
- **R_temp** — `0.5·success_rate + 0.5/(1+σ_RT)` with σ_RT the variance of
  the last 10 response times (0 if <2 samples); `mark_absent()` increments
  attempts without successes.
- **Bayesian adaptation** — every `_ADAPT_EVERY=10` rounds:
  `posterior = prior + evidence`, `weights = posterior/sum(posterior)`,
  prior = `_PRIOR_STRENGTH=20` × **(0.091, 0.134, 0.775)**. Delivering
  clients add their component scores as evidence; absent clients add the
  complement.
- **`trimmed_mean`** — a UAV's reputation is the 10%-per-tail trimmed mean
  of its selected clients' reputations (plain mean for small clusters);
  feeds the `R_min` server gate.

Note: reputation scores the client's **owned-block** delta (§6 Tier B), so
under `fusion_owner: uav` the delta vector is `struct_branch` only. This is
consistent — it is the client's actual contribution — but it means the
vector dimension J changed with the Tier-B rework, so anomaly distances are
not comparable across that boundary.

## B.4 Round-loop wiring (`run_full_hfl`)

Per round, in order: refresh device states + reputation scores → check
`low_eligible` → placement (if due; per-device V_i(t) from live
SNR/reputation, **× class-scarcity weight**) → selection (if
`(rnd−1) % reselect_every == 0 or low_eligible or not selected`) → per-UAV
groups → **LR decay** → UAV-tier training on its owned blocks → client-tier
training on its owned blocks (returning mean local loss) →
`selector.update_losses(...)` → `mark_absent` for selected clients that
produced no update → `update_batch` on delivered deltas → per-zone assembly
→ reputation-gated server FedAvg → **server momentum** → **EMA update** →
`device_mgr.update_round(selected)` → evaluate the **EMA model**, log row.

## B.5 Constants at a glance

| Constant | Value | Meaning |
|---|---|---|
| `B_MIN` | 0.20 | minimum battery fraction to be eligible |
| `SNR_MIN_DB` | 3.0 dB | minimum SNR to be eligible |
| `T_MAX_S` | 300 s | nominal max local-training time |
| `W_LEARNING_NB / W_UTILITY_NB` | 0.702 / 0.298 | priority weights (**battery removed**) |
| `W_EPI, W_SNR, W_DENS, W_PROX` | 0.043 / 0.078 / 0.295 / 0.584 | utility sub-weights (searched) |
| `UCB_C` | √2 | UCB1 exploration constant |
| `SEL_STATIC_BLEND` | 0.435 | static-priority weight in the roster score |
| `SEL_GUMBEL_SCALE` | 1.475 | roster stochasticity (0 → deterministic greedy) |
| `SEL_MIN_MARGINAL` | 0.0 | early-stop threshold on normalized coverage gain |
| `OORT_ALPHA` | 2.0 | Oort straggler penalty exponent (B4) |
| `FEDCS` / `REPCAP_GAMMA` / `FAIRMAB_W_*` | deadline / 0.5 / 0.5, 0.5 | baseline constants |
| `DEFAULT_T_STALE_CAP` | 5 (= `T_sel`) | fair_mab staleness normalizer |
| `T_decay` (β schedule) | 20 rounds | utility→reputation decay, shared with placement |
| Battery discharge/recharge | −0.02 / **+0.01** per round | ⇒ sustainable participating fraction 1/3 |
| SNR / compute noise | N(0,2) dB / N(0,30) s | per-round fluctuation |
| Adaptive margin ε_n | 1.96·σ(last 10 compute times) | dynamic eligibility buffer |
| `T_sel` / `reselect_every` | 5 / **1** | placement cadence / selection cadence |
| `lambda_min` / `R_min` | 0.5 / 0.3 | early-reselect trigger, aggregation gate |
| Reputation prior weights | **0.091 / 0.134 / 0.775** | Dirichlet-adapted every 10 rounds |
| `_VEC_EMA_NEW/OLD`, `_STATS_ALPHA` | 0.7/0.3, 0.1 | EMAs |
| `_PRIOR_STRENGTH` | 20.0 | Dirichlet prior concentration |
| Mahalanobis threshold | d ≤ 2 → R_anomaly = 1 | dimension-independent (/√J) |
| `trimmed_mean` trim | 10% each tail | UAV cluster reputation |
| `TARGET_CONSEC_ROUNDS` | 2 | consecutive hits required for rounds-to-target |

---

# Appendix C — Selection-literature baselines: citations, pseudocode, fidelity notes

All five baselines are implemented in `src/uavbench/fl/client_selection.py`
as selection modes `"fedcs"`, `"rep_cap"`, `"fair_mab"`, `"oort"`,
`"power_of_choice"`, exposed as full-system methods in `_METHOD_CFG`. They
share the identical PSO placement, reputation FedAvg aggregation, training
recipe, and cadence as `proposed_hfl`, so the selection rule is the only
experimental variable. All five sit behind the same eligibility gate; all
except `fedcs` produce a ranking that feeds the identical greedy UAV
assignment (Algorithm 4).

## C.1 Baseline B1 — FedCS

**Citation:** T. Nishio and R. Yonetani, "Client selection for federated
learning with heterogeneous resources in mobile edge," *2019 IEEE ICC*,
pp. 1–7.

**Core idea (source):** a fixed per-round deadline `Tmax`; greedily add the
candidate that increases the round's projected completion time the least,
stopping when the deadline would be exceeded. No data-value or reputation
signal — selection is purely a function of estimated time.

**Adaptation note:** the original assumes a flat server↔client topology. We
adapt it to the hierarchy by (i) running the identical greedy-marginal-time
rule independently per UAV over devices already gated in by our eligibility
filter (so FedCS gets the same eligibility floor everyone else gets), and
(ii) using each device's T̂_n(t) — already computed for our own priority
score — as FedCS's time estimate, so both methods see the same measurement
and only the selection *rule* differs.

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
distinguish a fast device with poor seismic SNR from a fast device near the
epicenter.

## C.2 Baseline B2 — Reputation-capability selection

**Citation:** H. Zhao, L. Geng, W. Feng, and C. Zhou, "Client selection and
resource scheduling in reliable federated learning for UAV-assisted
vehicular networks," *Chinese Journal of Aeronautics*, vol. 37, no. 9,
pp. 328–346, Jun. 2024.

**Core idea (source):** reputation-based mechanism integrating data quality
and computation capability — no exploration term, no geospatial utility, no
starvation guard.

**Adaptation note:** we reuse our own `R_contrib`/`R_anomaly`/`R_temp` as
the "data quality/reliability" half (recomputing a different reputation
estimator would conflate "different reputation model" with "different
selection rule"), and our existing ℓ̃_n = 1 − (T̂_n/Tmax)² as the
"computation capability" half. What it *omits* relative to ours: utility
Û_n, the β(t) blend, the UCB bonus, and the class-coverage roster — so it
systematically starves seismically valuable devices with middling
reputation/compute. Its starvation is expected and measured (Jain's index).

```
Algorithm B2: Reputation-Capability Selection (per round t)
1: for each n in E(t):  score_n ← γ·Rn(t) + (1 − γ)·ℓ̃n(t)     # γ = 0.5
2: sort E(t) by descending score_n → L
3: return L            # feeds Algorithm 4 (greedy UAV assignment)
```

## C.3 Baseline B3 — Fairness-enhanced MAB scheduling

**Citation:** C. Zhu, Y. Shi, H. Zhao, K. Chen, T. Zhang, and C. Bao, "A
fairness-enhanced federated learning scheduling mechanism for UAV-assisted
emergency communication," *Sensors*, vol. 24, no. 5, p. 1599, 2024.

**Core idea (source):** a multi-armed-bandit scheduler whose reward weights
model freshness (staleness since last contribution) and energy, explicitly
to enforce participation fairness in an emergency-communication UAV setting
— the same disaster context as this paper, making it a strong head-to-head.

**Adaptation note:** their "energy" maps onto our b_n(t) (battery — note
that this baseline therefore *keeps* a battery term in its score while the
proposed system has moved it to the gate, §0, which is exactly the contrast
worth drawing); their "staleness" maps onto our selection bookkeeping.
`T_stale_cap` is set to `T_sel` — a hyperparameter we introduce to bound
their reward to [0,1]; state this choice explicitly. Present it as "our
exploration is over *value*, theirs over *fairness/energy*".

```
Algorithm B3: Fairness/Energy MAB Selection (per round t)
1: for each n in E(t):
2:     reward_n ← w_energy·bn(t) + w_stale·min(1, staleness_n(t)/T_stale_cap)
                                                    # weights 0.5 / 0.5
3: sort E(t) by descending reward_n → L
4: return L                                         # feeds Algorithm 4
5: after the round: last_selected_n ← t for each selected n
```

## C.4 Baseline B4 — Oort (new)

**Citation:** F. Lai, X. Zhu, H. V. Madhyastha, and M. Chowdhury, "Oort:
Efficient Federated Learning via Guided Participant Selection," *15th
USENIX OSDI*, 2021, pp. 19–35.

**Core idea (source):** score each client by *statistical utility* — how
much its data would move the model, proxied by its local training loss —
multiplied by a *system utility* penalty for clients slower than a
developer-specified target round duration. It is the standard strong
baseline for "utility-guided" FL selection, and the closest published
competitor to our priority score, which makes it the most informative
head-to-head in the set.

**Adaptation notes (state explicitly):**
1. `T_pref` (the developer-preferred round duration) is left open by the
   source. Here it is the **median compute time of the currently eligible
   pool**. Using `T_max` instead would make the penalty inert, because our
   eligibility gate already enforces `T̂_n ≤ T_max − ε` — every eligible
   client would be "on time" and Oort would reduce to pure loss ranking.
   The median keeps the straggler mechanism live and is the least arbitrary
   pool-relative choice.
2. Losses are **last-observed**, exactly as in Oort's own stale-utility
   design (Oort explicitly reuses the utility from a client's last
   participation).
3. Never-trained clients receive the current **max** observed loss, which
   reproduces Oort's initialize-to-max exploration behaviour — so the
   baseline explores rather than being trapped by its own cold start.
4. α = 2, the paper's default.

```
Algorithm B4: Oort Guided Participant Selection (per round t)
1: T_pref ← median{ T̂n(t) : n ∈ E(t) }
2: for each n in E(t):
3:     stat_n ← last observed local training loss (max observed if never trained)
4:     penalty_n ← (T_pref / T̂n(t))^α  if T̂n(t) > T_pref  else  1
5:     util_n ← stat_n · penalty_n
6: sort E(t) by descending util_n → L
7: return L                                         # feeds Algorithm 4
```

**Expected contrast:** loss-guided and straggler-aware, but with **no
reputation/trust model, no geospatial or seismic utility, and no
class-coverage structure** — it optimizes "who is currently hardest to fit",
which under heavy class imbalance can concentrate on the same
hard-but-uninformative clients. Note the operating-envelope finding in §2:
under extreme data starvation Oort *beats* our rule, because loss magnitude
still carries signal at 21 samples/client while a 4-bin class histogram does
not. Report that honestly — it is a genuine limitation, not a nuisance.

## C.5 Baseline B5 — Power-of-Choice (new)

**Citation:** Y. J. Cho, J. Wang, and G. Joshi, "Client Selection in
Federated Learning: Convergence Analysis and Power-of-Choice Selection
Strategies," arXiv:2010.01243, 2020 (AISTATS 2022).

**Core idea (source):** sample a candidate subset of size `d` uniformly at
random from the available clients, evaluate the current global loss on each
candidate, and keep the highest-loss clients. It is the canonical
theory-backed biased-sampling baseline, with a convergence analysis — the
right foil for an empirically tuned selector.

**Adaptation notes (state explicitly):**
1. `d = 2 × (total slots)`, approximated as `2 · capacity · (number of
   covering UAVs)`, bounded by the eligible pool size — the standard
   "oversample by 2×" setting.
2. The canonical rule evaluates the **current global loss** on the
   candidate set, which costs an extra forward pass per candidate per
   round. The simulation instead uses **last-observed local losses** (the
   standard cached-loss variant used in FL benchmarks). This is a choice
   *in the baseline's favour on cost*: it is not billed compute that our
   method does not pay. Say so — it forecloses the "you handicapped the
   baseline" objection.
3. The uniform candidate draw uses the caller-supplied RNG, so it is
   reproducible and shares the harness's seed stream.

```
Algorithm B5: Power-of-Choice Selection (per round t)
1: d ← min(|E(t)|, 2 · capacity · #UAVs)
2: C ← d clients drawn uniformly WITHOUT replacement from E(t)
3: for each n in C:  score_n ← last observed local loss (max observed if never trained)
4: sort C by descending score_n → L
5: return L                                         # feeds Algorithm 4
```

**Expected contrast:** the uniform pre-draw gives it a built-in fairness
floor that pure greedy selection lacks, but it has no reputation, no
utility, no straggler awareness, and no class structure — and its
exploration is undirected (uniform) rather than count-driven (UCB).

## C.6 Presenting all five in the paper

**Signal table** (alongside Table I), one row per baseline:

| | Selection signal(s) | Exploration | Reputation/trust | Data-value/utility | Class-aware |
|---|---|---|---|---|---|
| B1 FedCS | compute time | none | ✗ | ✗ | ✗ |
| B2 Zhao et al. | reputation + capability | none | ✓ | ✗ | ✗ |
| B3 Zhu et al. | battery + staleness | staleness reward | ✗ | ✗ | ✗ |
| B4 Oort | local loss + straggler penalty | init-to-max | ✗ | loss proxy | ✗ |
| B5 Power-of-Choice | local loss | uniform pre-draw | ✗ | loss proxy | ✗ |
| **Proposed** | geospatial utility + capability + reputation | **UCB + Gumbel roster** | ✓ | ✓ | **✓** |

- **Results:** literature baselines in a *separate* table/figure from our
  own ablations — two tables, two questions (§10).
- **Related Work:** for each of the five citations add one sentence noting
  "we implement this as a baseline in §V".

---

# Appendix D — Data pipeline internals

Line-level reference for `src/hflsim/data/loader.py`, the FL dataset
adapters, and the PSO→HFL bridge.

- **Streaming loader:** rows are streamed (not snapshot-downloaded) from
  `AbbasABC/HFL-Dataset` at the pinned revision via the `datasets`
  `IterableDataset` API, keeping only the columns needed for partitioning +
  training. The outer streaming call retries up to 5 times with exponential
  backoff (30s, 60s, 120s, 240s, capped) so a mid-stream HF gateway failure
  or a tree-listing 504 under concurrent load doesn't crash a whole GCP
  job. Deterministic subsampling (`df.sample(n=..., random_state=seed)`) is
  applied after streaming.
- **GSI tile fetch:** each building chip is composited from a 2×2 mosaic of
  zoom-18 XYZ tiles and cropped to 128×128 RGB centered on the building's
  (lat,lon); tiles cached under `HFL_TILE_CACHE` (default
  `./data/tile_cache`). Network fetch retries 3 times (1s, 2s backoff)
  before returning a black chip, so a flaky connection never stalls
  training; the black-chip rate diagnostic (§2) exists because the seismic
  features are near-constant across one disaster area — the image is the
  only modality with real discriminative signal. The per-chip fallback is
  tolerated *because* the run-level hard gate bounds its aggregate effect.
- **Feature scaling order:** raw lat/lon are saved *before* any transform
  (needed unscaled for tile lookups), then normalized to [0,1] for model
  input; the 7 seismic columns (`MMI_original, MMI_shape, PGA, PGV, SA_0_3,
  SA_1_0, SA_3_0`) are z-scored (StandardScaler).
- **Partitioning:** `KMeans(n_clusters=N, n_init=10, random_state=pseed)`
  over (longitude, latitude); per client, indices shuffled with a
  per-client seed (`pseed + cid`) and split 80/20; a client's reported
  coordinate is the mean lat/lon of its rows (dataset mean if a cluster is
  empty). `pseed = partition_seed if given else random_seed` (§2, §11).
- **Caches:** the streamed metadata DataFrame is written to
  `<data_dir>/.metadata_df_cache_sub<pct>_seed<seed>.parquet` via temp-file
  + atomic `os.replace`; K-means assignments + train/test index lists are
  pickled to `<data_dir>/.partition_cache/partitions_<sha256[:16]>.pkl`,
  keyed by a hash of (row_count, N, train_ratio, **random_seed, pseed**).
  The temp file is now PID-suffixed so concurrent workers computing the
  same key cannot clobber each other's partial write.
- **Feature cache:** `compute_feature_cache` runs a frozen pretrained
  ResNet-18 (head replaced by Identity, ImageNet normalization) once over
  the whole dataset under `torch.inference_mode()` and saves (N,512)
  float16 features; FL training never runs the backbone. Shape-validated on
  load (§2). `feature_num_workers` parallelizes the decode during the
  sequential prefetch phase only.
- **`CachedDataset`** exposes `struct_features` and `labels` as in-memory
  tensor views and `eval_tensors()` for the vectorized evaluation path;
  `__getitem__` **never** touches `base[idx]` (that decoded and discarded a
  TIF per sample). The float32 cast is `copy=False` — the cache is already
  float32, and the unconditional copy duplicated the whole (N,512) array
  (262 MB and 130 ms per job at paper scale, × 12 workers).
- **`BalancedShardLoader`** replaces the `DataLoader` + `Subset` +
  `WeightedRandomSampler` stack. `balanced=True` reproduces the old
  inverse-class-frequency draw with replacement exactly; `balanced=False`
  (the new default, §6 A1) draws a uniform permutation. Batches slice the
  in-memory tensors, gathering 64 batches at a time and handing out
  contiguous slices — same rows in the same order as the per-batch gather,
  bit-identical, with the block cap keeping extra live memory ~4 MB even
  when the shard is the entire training set.
- **PSO→HFL bridge (`hflsim/placement.py`):** converts a
  `{client_id: (lat,lon)}` dict to a `ProblemInstance` in projected metric
  space, runs the requested optimizer via `uavbench.optimizers.REGISTRY`,
  and converts the result back to `UAVAggregator` objects at optimized
  lat/lon. When `prev_positions` isn't supplied it defaults to an even
  linspace spread across the bounding box at mid-altitude, avoiding a
  placement bias toward the projection origin. `UAVAggregator` (legacy
  `hflsim.simulation`) has **two** consumers — this bridge and the
  standalone `hflsim` CLI (§21) — not one; the live round loop aggregates
  via `uavbench/fl/model.py` (§9).

---

# Appendix E — Reference constants tables

### Placement objective & PSO/GA

| Parameter | Value |
|---|---|
| `w1, w2, w3` (fitness weights, **searched** — §0) | 0.811 / 0.03 / 0.159 |
| (paper-stated values, for the deviation table) | 0.6 / 0.3 / 0.1 |
| Theoretical fitness ceiling | `w1` = 0.811 |
| Early-stop threshold (PSO & GA) | `0.95 × w1` |
| `R_comm` (Tier-1 / full sim) | 500 m / 20,000 m |
| `B_min_uav` | 0.2 |
| Tier-1 `capacity`, `K`, `N` | 15, 10, 250 (binding: 150 slots < 250) |
| Tier-1 saturating ablation `capacity` | 30 (300 slots > 250) |
| Heterogeneous fleet `capacity_cv`, `battery_cv` | 0.5, 0.3 (`tier1_regime_hetero`) |
| PSO `P`, `G_max` (Tier-1 / full sim) | 100, 200 / 50, 30 |
| PSO `c1=c2` (→ `phi`, `chi`) | 2.05 (→ 4.1, ≈0.7298) |
| PSO `ring_k`, `vmax_frac` | 2, 0.2 |
| PSO stagnation `(delta, G_stag, rho)` | 1e-4, 20, 0.2 |
| PSO turbulence `p_turb` | 0.1 |
| GA `crossover_prob`, `eta_c`, `eta_m` | 0.9, 15.0, 20.0 |
| GA `tournament_size`, `n_elite` | 3, 2 |
| Energy model `p_fly/p_hover/cruise/t_serve/capacity` | 250 W / 200 W / 15 m/s / 60 s / 200,000 J |
| Path-loss budget (Tier-1 / Noto scale) | 95 dB (r*≈499 m) / 145 dB |
| MCLP reference `grid_res`, `time_limit`, `mip_rel_gap` | 20 (400 sites), 120 s, 1e-4 |

### Training recipe (Tier A/B/C)

| Parameter | Value | Source |
|---|---|---|
| `lr`, `uav_lr` | 1.775e-2 | `tuned_weights.yaml` |
| `logit_adjust_tau` | 0.601 | `tuned_weights.yaml` |
| `server_momentum`, `server_lr` | 0.528, 1.0 | `tuned_weights.yaml` / config |
| `lr_decay` | `cosine` | config |
| `ema_decay` | 0.9 (**not** from the search) | config |
| `balanced_sampling` | false (uniform sampling) | config |
| `local_optimizer` | SGD, momentum 0.9 | config |
| `fusion_owner` | `uav` (UAV owns img_proj+fusion) | config |
| `placement_class_aware` | true | config |
| `reselect_every` / `T_sel` | 1 / 5 | config |
| `target_metric`, `target_value` | `macro_f1`, 0.45 | config |
| `n_local_epochs`, `n_uav_epochs`, `batch_size` | 2, 2, 32 | config |

### Model & deployment

| Parameter | Value |
|---|---|
| Image feature dim (ResNet-18) | 512 |
| Structured feature dim | 9 |
| Fusion embed dims (img/struct) | 128 / 64 |
| Damage classes | 4 (Survived/Collapsed/Obstructed/Missing) |
| `img_proj` / `struct_branch` / `fusion` params | 65,664 / 17,216 / 50,436 |
| IoT payload as billed by `round_comm_mb` | ≈0.271 MB (67,652 params) — **see §5 OPEN ITEM** |
| UAV payload as billed | ≈0.533 MB (133,316 params) |
| Actual client payload under `fusion_owner: uav` | ≈0.069 MB (17,216 params) |
| Actual UAV payload under `fusion_owner: uav` | ≈0.464 MB (116,100 params) |
| GSI tile zoom / chip size | 18 (~0.6 m/px) / 128×128 |
| Raw damage code remap | `{0→0, 1→1, 9→2, 99→3}` |
| Client partitioning | K-means over (lon, lat), `n_init=10`, `random_state=partition_seed` |
| Default `train_ratio` | 0.8 |
| HF stream retry backoff | 30s, 60s, 120s, 240s (capped) |

### Experiment design

| Parameter | Value |
|---|---|
| `PIPELINE_VERSION` | 3 |
| `partition_seed_for(i)` | 5309 + i |
| `n_seeds` (paper_full / coverage / selection isolation) | 10 |
| `n_seeds` (stress sweeps) | 16 (n ≥ 14 required — §13) |
| `n_seeds` (Tier-1 core / supplements) | 30 / 20 |
| Significance defaults | Wilcoxon, Holm, α=0.05, `--last-k 10` |
| Bootstrap CI | 10,000 resamples, 95% percentile |
| Wilcoxon p-floor | 2/2ⁿ (n=10 → 1.95e-3; n=16 → 3.05e-5) |

(Client-selection and reputation constants: Appendix B §B.5.)
