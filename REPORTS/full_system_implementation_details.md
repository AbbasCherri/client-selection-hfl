# Full System Implementation Reference — Placement, Client Selection, and Deployment

This is the single, comprehensive reference for the entire system: the PSO/GA
placement optimizers and their shared objective, the UCB client-selection
pipeline and reputation system, the federated model and aggregation rules,
the real-world data-loading/deployment pipeline, and the experiment
orchestration that ties it all together. It supersedes and merges
`pso_algorithm_explained.md`, `client_selection_algorithm_explained.md`, and
`deployment_implem_details.md` into one document. Every claim is derived
directly from the code, with file/line references throughout.

---

## Table of contents

- [Part 0 — System architecture map](#part-0--system-architecture-map)
- [Part I — UAV placement optimization](#part-i--uav-placement-optimization)
  1. [The placement problem & encoding](#1-the-placement-problem--encoding)
  2. [The shared fitness objective](#2-the-shared-fitness-objective)
  3. [Greedy device→UAV assignment](#3-greedy-devicesuav-assignment)
  4. [The device value model V_i(t)](#4-the-device-value-model-vit)
  5. [ProblemInstance & scenario generation](#5-probleminstance--scenario-generation)
  6. [PSO — the proposed optimizer](#6-pso--the-proposed-optimizer)
  7. [GA — the head-to-head baseline](#7-ga--the-head-to-head-baseline)
  8. [Heuristic baselines](#8-heuristic-baselines)
  9. [Seeding & clustering utilities](#9-seeding--clustering-utilities)
  10. [Energy model (reporting only)](#10-energy-model-reporting-only)
- [Part II — Client selection & reputation](#part-ii--client-selection--reputation)
  11. [Overview: the four-stage pipeline](#11-overview-the-four-stage-pipeline)
  12. [Per-device state model](#12-per-device-state-model)
  13. [Stage 1 — eligibility gate](#13-stage-1--eligibility-gate)
  14. [Stage 2 — utility Û_n](#14-stage-2--utility-ûn)
  15. [Stage 2 continued — priority score P_n](#15-stage-2-continued--priority-score-pn)
  16. [Stage 3 — UCB exploration bonus](#16-stage-3--ucb-exploration-bonus)
  17. [Stage 4 — greedy UAV assignment](#17-stage-4--greedy-uav-assignment)
  18. ["random" and "all" ablation modes](#18-random-and-all-ablation-modes)
  19. [The reputation system](#19-the-reputation-system)
- [Part III — Federated model & aggregation](#part-iii--federated-model--aggregation)
  20. [CachedFusionModel architecture](#20-cachedfusionmodel-architecture)
  21. [Aggregation rules (fedavg family)](#21-aggregation-rules-fedavg-family)
  22. [Dataset adapters & synthetic fallback](#22-dataset-adapters--synthetic-fallback)
  23. [Image feature caching](#23-image-feature-caching)
- [Part IV — Deployment / real-data pipeline](#part-iv--deployment--real-data-pipeline)
  24. [HuggingFace streaming loader](#24-huggingface-streaming-loader)
  25. [On-demand aerial imagery (GSI tiles)](#25-on-demand-aerial-imagery-gsi-tiles)
  26. [Label remapping & feature scaling](#26-label-remapping--feature-scaling)
  27. [K-means geographic client partitioning](#27-k-means-geographic-client-partitioning)
  28. [Partition & metadata caching](#28-partition--metadata-caching)
  29. [The PSO→HFL integration bridge](#29-the-psohfl-integration-bridge)
- [Part V — Orchestration & experiment infrastructure](#part-v--orchestration--experiment-infrastructure)
  30. [Tier-1 runner (placement-only benchmark)](#30-tier-1-runner-placement-only-benchmark)
  31. [Tier-1 post-hoc metrics](#31-tier-1-post-hoc-metrics)
  32. [Tier-2 / full-system round loop](#32-tier-2--full-system-round-loop)
  33. [Method configuration table](#33-method-configuration-table)
  34. [N-scalability sweep](#34-n-scalability-sweep)
  35. [Reproducibility — RNG stream discipline](#35-reproducibility--rng-stream-discipline)
- [Part VI — Reference tables](#part-vi--reference-tables)

---

## Part 0 — System architecture map

Two Python packages under `src/` cooperate:

- **`uavbench`** — the placement-optimization benchmark (Tier-1: PSO/GA/heuristics
  vs. a shared fitness objective) *and* the Tier-2/full-system federated
  learning harness (`uavbench/fl/`) that consumes placement as a subroutine.
- **`hflsim`** — shared coordinate/value utilities (`hflsim/shared/`), the
  real-world data loading pipeline (`hflsim/data/loader.py`), a thin
  PSO-integration bridge (`hflsim/placement.py`), and legacy simulation
  scaffolding (`hflsim/simulation/`) that predates `uavbench/fl/` and is no
  longer on the main execution path (only `simulation/uav.py`'s
  `UAVAggregator` is still used, by the bridge in `hflsim/placement.py`).

Execution flow for a full run (`run_full_hfl` in
[`federated.py`](../src/uavbench/fl/federated.py)):

```
Data loading (real HF stream or synthetic)
        │
        ▼
For each method, each round:
  Placement (PSO/GA, every T_sel rounds) ──► Coverage (R_comm gate)
        │
        ▼
  Client selection (UCB / random / all) ──► reads DeviceState + Reputation
        │
        ▼
  Local training (IoT struct+fusion; UAV full model incl. img_proj)
        │
        ▼
  Reputation update (contrib / anomaly / temporal, Bayesian-adapted weights)
        │
        ▼
  UAV-level FedAvg ──► Server-level (reputation-gated) FedAvg
        │
        ▼
  Evaluate global model; advance device state; log round
```

---

# Part I — UAV placement optimization

## 1. The placement problem & encoding

The algorithm places **K UAV aggregators in 3D space** over a disaster area so
they can serve **N ground IoT devices**. Each device sits at ground level
(z = 0); each UAV hovers at (x, y, z) with z bounded to an altitude band. A
placement is scored on coverage, movement cost, and load balance (§2). This is
a continuous, non-convex, multimodal optimization problem in **3K
dimensions** ([pso.py](../src/uavbench/optimizers/pso.py)).

Each particle/individual is a flat vector **X ∈ ℝ³ᴷ**: K UAV positions
concatenated as (x₁, y₁, z₁, …, x_K, y_K, z_K). Per-dimension bounds are the
area box tiled K times
([base.py:63-68](../src/uavbench/optimizers/base.py#L63-L68)); a flat vector
reshapes back to `(K, 3)` via
[`positions_from_vector`](../src/uavbench/problem/instance.py#L99-L101).

## 2. The shared fitness objective

Every optimizer scores candidates through **one** shared callable
([fitness.py](../src/uavbench/problem/fitness.py)), so all methods are
compared on an identical objective and evaluation budget:

```
F(X) = w1 · (F_cover / F_max)  −  w2 · (D_move / D_max)  −  w3 · (L_imb / L_max)
w1 = 0.6,  w2 = 0.3,  w3 = 0.1                     (maximization)
```

| Term | Definition | Normalizer |
|---|---|---|
| `F_cover` | `Σ Vᵢ` over assigned devices (value-weighted, not a raw count) | `F_max = Σᵢ Vᵢ` |
| `D_move` | `Σⱼ ‖pⱼ − pⱼ_prev‖` — total 3D reposition distance | `D_max = K · box_diagonal` |
| `L_imb` | `Σⱼ (loadsⱼ − n_assigned/K)²` — load variance **among served devices only** | `L_max = N²` |

The theoretical ceiling is `F ≤ w1 = 0.6` (full coverage, zero movement, zero
imbalance); PSO and GA both early-stop at `0.95 × w1 = 0.57`.
([fitness.py:1-16](../src/uavbench/problem/fitness.py#L1-L16),
[fitness.py:85-87](../src/uavbench/problem/fitness.py#L85-L87))

`Fitness` tracks an `eval_count` so the runner can verify every optimizer
spends the identical evaluation budget — no optimizer may implement its own
scoring. ([base.py](../src/uavbench/optimizers/base.py))

## 3. Greedy device→UAV assignment

Every fitness evaluation runs a deterministic greedy assignment
([assignment.py](../src/uavbench/problem/assignment.py)):

1. Sort devices by **descending value** (stable sort, `O(N log N)`).
2. For each device (highest value first), collect feasible positions —
   **range** (3D Euclidean ≤ `R_comm`, default 500 m) **AND** **battery**
   (`battery[j] ≥ B_min_uav`, default 0.2) **AND** **capacity**
   (`loads[j] < capacity[j]`).
3. Assign to the feasible position with the **smallest current load**, ties
   broken by **smallest distance** (`np.lexsort`).
4. A device with no feasible position is left unserved — **no penalty term,
   no repair operator**. This keeps the fitness landscape clean; hard
   constraints are enforced implicitly by zero coverage credit rather than by
   a penalty coefficient that would need tuning.

Cost: `O(N log N)` sort + `O(N·K)` sweep, with the `(N, K)` distance matrix
computed once in vectorized NumPy.
([assignment.py:44-86](../src/uavbench/problem/assignment.py#L44-L86))

## 4. The device value model V_i(t)

Single source of truth: [`hflsim/shared/value.py`](../src/hflsim/shared/value.py)
(re-exported by [`uavbench/problem/value.py`](../src/uavbench/problem/value.py)).

```
V_i(t) = β(t)·U_i(t) + (1 − β(t))·R_i(t)
β(t) = max(0, 1 − t / T_decay),  T_decay = 20
```

`beta_mode="pinned"` forces `β = 1` (utility-only, history-free benchmark);
`"scheduled"` uses the decaying `β(t)` above. This same schedule is reused
verbatim by client selection's `Ũ_n` blend (§15) — one shared function,
two consumers.

**Utility U_i — four weighted features (sum to 1.0):**

| Feature | Weight | Formula |
|---|---|---|
| Epicenter proximity | 0.4 | `max(0, (d95 − min(d_epi, d95)) / d95)`; `d95` = 95th-pct distance to epicenter |
| SNR | 0.3 | Min-max normalized over the cluster |
| Sample density | 0.2 | Min-max normalized local sample count (proxy for building density — no inventory exists for the study area) |
| Nearest-UAV proximity | 0.1 | `max(0, 1 − d_min/R_comm)` |

**Raw feature distributions** (synthesized in `generate_instance`): SNR ~
`Uniform(0, 30)` dB; samples ~ `Uniform_int(20, 200)`; reputation `R_i` ~
`Beta(2, 2)`, held fixed within one instance.
([value.py](../src/hflsim/shared/value.py))

## 5. ProblemInstance & scenario generation

`ProblemInstance` ([instance.py](../src/uavbench/problem/instance.py)) fields:

| Field | Shape | Description |
|---|---|---|
| `device_coords` | `(N,3)` | ground positions, z=0 |
| `value` | `(N,)` | fixed `V_i(t)` |
| `capacity` | `(K,)` | max devices/UAV |
| `battery` | `(K,)` | UAV battery fraction |
| `prev_positions` | `(K,3)` | previous layout (movement baseline) |
| `lower/upper` | `(3,)` | per-dim bounds |
| `R_comm`, `B_min_uav` | scalar | range gate, battery gate |

`generate_instance` distributions: `"uniform"` (uniform over the box),
`"clustered"` (`√N/2` Gaussian clusters, σ = 0.06·span), `"epicenter_biased"`
(single Gaussian at the epicenter, σ = 0.12·span).

`prev_mode` controls the movement-penalty baseline:
- **`"stale"`** (default) — previous layout fitted to a *shifted* epicenter
  (offset `N(0, 0.25·span)`); models the situation that triggers a
  reconfiguration, so `static` is a genuine floor.
- **`"warm"`** — previous positions near current device sub-centroids
  (already near-optimal); studies the conservative regime.

All coordinates are projected meters (equirectangular about a reference
point); 3D Euclidean distance is used throughout.
([instance.py:138-234](../src/uavbench/problem/instance.py#L138-L234))

## 6. PSO — the proposed optimizer

**Family:** Constriction-factor PSO with an lbest ring topology (Clerc &
Kennedy 2002), plus four safeguards layered on top.
([pso.py](../src/uavbench/optimizers/pso.py))

**Constriction factor**, derived (never hardcoded) from `c1, c2`:
```
phi = c1 + c2                      # must be > 4; raises ValueError otherwise
chi = 2 / |2 − phi − sqrt(phi² − 4·phi)|
```
Defaults `c1 = c2 = 2.05` → `phi = 4.1` → `chi ≈ 0.7298`.
([pso.py:19-26](../src/uavbench/optimizers/pso.py#L19-L26),
[pso.py:62-65](../src/uavbench/optimizers/pso.py#L62-L65))

**Velocity update** (constriction mode, default):
```
V ← chi · (V + c1·r1⊙(pbest − X) + c2·r2⊙(nbest − X))
```
**Inertia-mode fallback** (`use_constriction=False`, ablation only):
```
w = inertia_max − (inertia_max − inertia_min) · (tau / G_max)   # linear decay 0.9→0.4
V ← w·V + c1·r1⊙(pbest − X) + c2·r2⊙(nbest − X)
```

**Ring topology** (default, `topology="ring"`): each particle's
neighborhood-best is the best `pbest` among `2·ring_k + 1` particles on a
ring (`ring_k=2` → 5-particle neighborhoods), computed fully vectorized.
`topology="gbest"` shares one global best across the swarm instead.
([pso.py:108-128](../src/uavbench/optimizers/pso.py#L108-L128))

**Initialization** (`seeding="value_kmeans"`, default): half the swarm seeded
by value-weighted k-means++ centers over device (x,y), jittered with
`N(0, jitter_m=10m)` and a uniform random altitude; the other half uniform
random. `"plain_kmeans"` drops the value weighting; `"uniform"` seeds the
whole swarm uniformly. Initial velocity: `0.5·Uniform(−vmax, vmax)`.
([pso.py:82-104](../src/uavbench/optimizers/pso.py#L82-L104))

**Velocity clamp:** `vmax[d] = vmax_frac·(hi[d]−lo[d])`, default `vmax_frac=0.2`.

**Boundary handling — absorbing walls:** out-of-bound positions clamped to
`[lo,hi]`; the velocity component that caused the violation is zeroed (no
bounce, no wraparound). ([pso.py:180-183](../src/uavbench/optimizers/pso.py#L180-L183))

**Turbulence** (`use_turbulence=True`, default): each iteration, a random
`p_turb=0.1` fraction of particles get a kick `~Uniform(−0.1·vmax, 0.1·vmax)`
added to velocity before clamping — prevents micro-stagnation without
disrupting convergence. ([pso.py:169-173](../src/uavbench/optimizers/pso.py#L169-L173))

**Stagnation reinitialization** (`use_stagnation=True`, default): if gbest
improves by less than `delta_stag=1e-4` for `G_stag=20` consecutive
generations, the worst `floor(rho·P)` particles (default `rho=0.2` → 20 of
100) are replaced with fresh uniform samples, velocities and personal bests
reset, and gbest is re-checked immediately. Best particles are untouched — the
incumbent is never lost. ([pso.py:202-215](../src/uavbench/optimizers/pso.py#L202-L215))

**Global best:** updated only on strict improvement — never overwritten by an
equal or worse value.

**Early stopping:** loop exits once `gbest_fit ≥ early_stop_frac·w1` (default
`0.95 × 0.6 = 0.57`), exploiting the fact that `w1` is a hard fitness ceiling.
([pso.py:148-149](../src/uavbench/optimizers/pso.py#L148-L149))

**Every design choice is an ablation toggle** — `use_constriction`,
`topology`, `use_clamp`, `use_stagnation`, `use_turbulence`, `seeding` — so
each enhancement's contribution can be isolated in one config line.

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
| `jitter_m` | 10.0 | seeding jitter std-dev (meters) |
| `inertia_max/min` | 0.9 / 0.4 | inertia range (inertia mode only) |

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

## 7. GA — the head-to-head baseline

**File:** [ga.py](../src/uavbench/optimizers/ga.py). Real-coded GA with SBX
crossover, polynomial mutation, tournament selection, and elitism — run at
the **same P/G_max budget as PSO** for a fair evaluation-count comparison.

- **Initialization:** uniform random (no warm-start, unlike PSO).
- **Selection:** binary tournament (`tournament_size=3` random contenders;
  fitness argmax wins).
- **Crossover (SBX)**, probability `crossover_prob=0.9`; per-gene gate
  `u ≤ 0.5`:
  ```
  beta = (2u)^(1/(eta_c+1))            if u ≤ 0.5
       = (1/(2(1−u)))^(1/(eta_c+1))    otherwise
  child1 = 0.5·((1+beta)p1 + (1−beta)p2)
  child2 = 0.5·((1−beta)p1 + (1+beta)p2)
  ```
  clipped to `[lo,hi]`. `eta_c=15.0`.
- **Mutation (bounded polynomial)**, per-gene probability `1/dim` by default,
  `eta_m=20.0`:
  ```
  delta = (2u)^(1/(eta_m+1)) − 1              if u<0.5   (range [−1,0])
        = 1 − (2(1−u))^(1/(eta_m+1))          otherwise  (range [0,1])
  x_new = x + delta·(x−lo)   if u<0.5   else   x + delta·(hi−x)
  ```
  Guaranteed in-bounds without clipping (scales toward the nearer boundary).
- **Elitism:** top `n_elite=2` individuals copied directly into the next
  generation before the tournament/crossover/mutation loop fills the rest.
- **Early stop:** same `0.95·w1` threshold as PSO.

| Parameter | Default |
|---|---|
| `P` | 100 |
| `G_max` | 200 |
| `crossover_prob` | 0.9 |
| `eta_c` | 15.0 |
| `eta_m` | 20.0 |
| `mutation_prob` | `1/dim` |
| `tournament_size` | 3 |
| `n_elite` | 2 |
| `early_stop_frac` | 0.95 |

## 8. Heuristic baselines

**File:** [heuristics.py](../src/uavbench/optimizers/heuristics.py). Three
deterministic/trivial competitors bracketing the metaheuristics:

- **`Centroid`** — UAVs placed at value-weighted k-means centroids
  (`weighted_kmeans`), fixed altitude `= lower.z + altitude_frac·(range)`
  (default 0.5). Single evaluation, deterministic given the RNG seed. Fast
  and value-aware but blind to capacity saturation and movement cost.
  `value_weighted=False` toggles unweighted centroids.
- **`RandomPlacement`** — best of `n_draws=20` uniform random candidates
  (one evaluation each). The no-intelligence floor.
- **`Static`** — UAVs stay exactly at `instance.prev_positions`; one
  evaluation. Measures the marginal value of *any* repositioning — the
  ablation-style lower bound for dynamic placement.

## 9. Seeding & clustering utilities

**File:** [seeding.py](../src/uavbench/optimizers/seeding.py), shared by PSO
and `Centroid`.

- **`kmeanspp_centers(rng, points, K, weights=None)`** — k-means++ with
  optional value-proportional sampling: first center ∝ `weights` (uniform if
  `None`); subsequent centers ∝ `D²(x)·weights` where `D²` is squared
  distance to the nearest chosen center. Falls back to a uniform index draw
  if total weight is zero/non-finite.
- **`weighted_kmeans(rng, points, K, weights=None, n_iter=25)`** — Lloyd's
  algorithm seeded by the above; weighted centroid updates
  `c_k = Σ(w_i·x_i)/Σw_i`; stops early on convergence (`np.allclose`).

## 10. Energy model (reporting only)

**File:** [energy.py](../src/uavbench/problem/energy.py). Converts the
abstract movement distance the optimizer minimizes into physical units —
**never called inside the fitness function**, so the optimizer's objective
stays identical across methods; it only feeds reporting metrics and the FL
harness's cumulative-energy column.

```
E_move = p_fly · (d / cruise_speed) + p_hover · t_serve
battery_fraction = E_move / battery_capacity_j
```

| Constant | Default | Meaning |
|---|---|---|
| `p_fly` | 250 W | cruise power draw |
| `p_hover` | 200 W | hover power draw |
| `cruise_speed` | 15 m/s | horizontal cruise speed |
| `t_serve` | 60 s | hover service time per round |
| `battery_capacity_j` | 200,000 J | usable battery energy |

---

# Part II — Client selection & reputation

## 11. Overview: the four-stage pipeline

Corresponds to **paper Algorithms 1–4 (§IV-C)**.
([client_selection.py:1-19](../src/uavbench/fl/client_selection.py#L1-L19))

```
1. Eligibility gate    : battery ≥ B_min, SNR ≥ SNR_min, memory OK, time ≤ T_max − ε
2. Priority score      : P_n = w_b·b_n + w_ℓ·(1 − (T̂_n/T_max)²) + w_U·Ũ_n
                         with Ũ_n = β(t)·Û_n + (1 − β(t))·R_n
3. UCB exploration     : UCB_n(t) = P_n(t) + C·√(ln t / (N_n(t)+1))
4. Greedy assignment   : sort by UCB; each client goes to the feasible UAV
                         (in range, under capacity) with the lowest current
                         load, ties broken by smallest distance; skipped if
                         no UAV is feasible
```

Three selection **modes** share this file:

| Mode | Behavior | Used by |
|---|---|---|
| `"ucb"` | Full 4-stage pipeline | `proposed_hfl`, `hfl_static`, `hfl_no_reputation` |
| `"random"` | Eligibility filter, then uniform random draw per UAV | `hfl_no_selection` |
| `"all"` | Skip every filter — all covered clients participate | `flat_fl`, `centralized` |

## 12. Per-device state model

**File:** [device_state.py](../src/uavbench/fl/device_state.py), paper §IV-B.

```python
class DeviceState:
    battery: float          # [0,1]
    snr_db: float           # dB
    memory_ok: bool         # fixed per device
    compute_time_s: float   # straggler model
    margin_s: float         # adaptive eligibility margin ε_n(t)
```

**Initialization** (once per client): `battery ~ Uniform(0.5,1.0)`;
`snr_base ~ Uniform(5,20)` dB; `memory_ok ~ Bernoulli(0.90)` (10% permanently
memory-constrained); `compute_base ~ Uniform(50,250)` s.
([device_state.py:57-76](../src/uavbench/fl/device_state.py#L57-L76))

**Per-round update** (`update_round(selected_ids)`, called end-of-round):
selected clients discharge battery by **−0.02**; unselected recharge by
**+0.005** (clipped [0,1]). Every device gets fresh `N(0,2)` dB SNR noise and
`N(0,30)` s compute noise. Selected clients' observed compute time is
appended to a rolling 10-round history.
([device_state.py:84-101](../src/uavbench/fl/device_state.py#L84-L101))

**Adaptive margin ε_n(t)** (paper §IV-C1):
```
margin_s = 1.96 · std(last 10 observed compute times)   if ≥ 3 observations, else 0
```
A device with historically volatile completion times gets a **stricter**
effective deadline (`T_max − margin_s`).
([device_state.py:103-112](../src/uavbench/fl/device_state.py#L103-L112))

**Eligibility predicate** (all four gates hard):
```python
eligible = (battery >= 0.20) and (snr_db >= 3.0) and memory_ok and (compute_time_s <= 300 − margin_s)
```
(`B_MIN=0.20, SNR_MIN_DB=3.0, T_MAX_S=300` — paper Table II)

## 13. Stage 1 — eligibility gate

```python
eligible = {cid: uav_idx for cid, uav_idx in covered.items()
            if device_states.get(cid) is not None and device_states[cid].eligible()}
```
If nothing survives, selection returns `{}`. `"all"` mode skips this and
everything after; `"random"` mode applies only this gate before a uniform
draw. ([client_selection.py:152-160](../src/uavbench/fl/client_selection.py#L152-L160))

## 14. Stage 2 — utility Û_n

`_compute_utility()` recomputes fresh each selection round over the
currently-eligible set (paper §IV-C3):

```
Û_n = 0.4·U_epi + 0.3·U_SNR + 0.2·U_dens + 0.1·U_prox
```

- **U_epi** — capped at the **95th percentile** distance across the eligible
  set: `U_epi = (d95 − min(d_n,d95))/d95` (prevents one outlier device from
  squashing everyone else's score).
- **U_SNR** — min-max of `snr_db` clipped to [0,30] dB.
- **U_dens** — count of other eligible clients within 5 km (local
  equirectangular frame, vectorized O(n²)), min-max normalized.
- **U_prox** — `clip(1 − min_j(haversine(client,uav_j))/R_comm, 0, 1)`;
  defaults to `0.5` for every client if no UAV coordinates are supplied.

`_minmax` degenerate-input guard: if all values are within `1e-10`, every
client gets exactly `0.5` rather than dividing by a near-zero range.
([client_selection.py:53-121](../src/uavbench/fl/client_selection.py#L53-L121))

## 15. Stage 2 continued — priority score P_n

```python
l_feat  = clip(1 − (compute_time_s/T_MAX_S)^2, 0, 1)
beta    = beta_schedule(round_num)              # same β(t) as the placement value model
u_tilde = beta·utility + (1−beta)·reputation
priority = 0.35·battery + 0.30·l_feat + 0.35·u_tilde
```
`W_BATTERY=0.35, W_LEARNING=0.30, W_UTILITY=0.35`. Squaring the compute-time
ratio makes the speed penalty grow superlinearly as a client's estimated time
approaches the deadline. Reputation defaults to `0.5` if the reputation
manager hasn't scored a client yet.
([client_selection.py:173-183](../src/uavbench/fl/client_selection.py#L173-L183))

## 16. Stage 3 — UCB exploration bonus

```python
t = max(round_num, 1)
ucb_n = priority_n + sqrt(2) · sqrt(ln(t) / (N_n(t) + 1))
```
`UCB_C = √2`; `N_n(t)` is the client's cumulative selection count, tracked
statefully inside `ClientSelector` across the whole run. Classic UCB1 —
rarely-selected clients get a larger bonus, preventing permanent starvation of
low-priority-but-plausible clients.
([client_selection.py:43](../src/uavbench/fl/client_selection.py#L43),
[client_selection.py:185-187](../src/uavbench/fl/client_selection.py#L185-L187))

## 17. Stage 4 — greedy UAV assignment

`_greedy_assign()` processes clients in **descending UCB score order**; for
each, feasibility requires in-range (Haversine ≤ `R_comm`) **and** under
capacity at that moment. Among feasible UAVs: smallest current load, ties by
smallest distance. No feasible UAV → client is skipped this round entirely
(no error, no penalty) — mirroring the placement-side assignment's
philosophy. The full `(N,K)` Haversine matrix is precomputed once per call.
A legacy fallback path (no UAV coordinates supplied) assigns to the client's
pre-computed covering UAV if not yet full. Every successful assignment
increments `self._counts[cid]`, feeding next round's UCB bonus.
([client_selection.py:198-247](../src/uavbench/fl/client_selection.py#L198-L247))

## 18. "random" and "all" ablation modes

**`"random"`** (used by `hfl_no_selection`): after the eligibility gate,
clients are bucketed by pre-assigned UAV; each bucket draws
`min(capacity, len(bucket))` uniformly without replacement. Uses the
caller-supplied `rng` when given (required for correct multi-seed sweeps);
falls back to `round_num*7919` only for legacy callers. Does **not**
increment UCB counts. ([client_selection.py:250-271](../src/uavbench/fl/client_selection.py#L250-L271))

**`"all"`** (used by `flat_fl`, `centralized`): returns `dict(covered)`
unchanged — no filtering, scoring, or capacity limit.

## 19. The reputation system

**File:** [reputation.py](../src/uavbench/fl/reputation.py), paper Algorithm 3
(§IV-C4). Maintains `R_n ∈ [0,1]` per client:

```
R_n = w_contrib·R_contrib + w_anomaly·R_anomaly + w_temp·R_temp
```
Initial weights `(0.4, 0.3, 0.3)`, Bayesian-adapted every 10 rounds (§19.4).

**19.1 R_contrib (contribution quality)** — cosine similarity between
successive EMAs of the client's **update-delta vector** (not absolute
weights, which are ≈1-cosine-similar for everyone due to shared init):
```
Δw̄_n(t) = 0.7·Δw_n(t) + 0.3·Δw̄_n(t−1)
R_contrib = (1 + cos(Δw̄_n(t), Δw̄_n(t−1))) / 2
```
Cold start or near-zero-norm EMA → `0.5`.
([reputation.py:145-162](../src/uavbench/fl/reputation.py#L145-L162))

**19.2 R_anomaly (diagonal Mahalanobis anomaly)** — a **global**
per-parameter mean/variance is EMA-tracked (`_STATS_ALPHA=0.1`) across every
client's updates:
```
z = (v − param_mean) / sqrt(param_var + eps)
d = sqrt(mean(z²))                     # Mahalanobis / √J — dimension-independent
R_anomaly = 1                if d ≤ 2
          = exp(−0.5·(d−2))  otherwise
```
The `/√J` normalization makes the fixed threshold of 2 meaningful regardless
of parameter-vector dimensionality.
([reputation.py:134-168](../src/uavbench/fl/reputation.py#L134-L168))

**19.3 R_temp (temporal reliability)**:
```
success_rate = total_successes / total_attempts
sigma_RT = variance of last 10 response times (0 if <2 samples)
R_temp = 0.5·success_rate + 0.5/(1+sigma_RT)
```
`mark_absent()` (called for clients selected but unable to deliver an
update) increments `total` without incrementing `success`, dropping the rate.

**19.4 Bayesian weight adaptation (every 10 rounds).** Each round, every
delivering client's three component scores are added as evidence; an absent
client instead contributes `(1 − score)` as evidence against components that
scored it highly. Every `_ADAPT_EVERY=10` rounds:
```
posterior = prior + evidence          # prior = (0.4,0.3,0.3) * _PRIOR_STRENGTH(20.0)
weights = posterior / sum(posterior)
```
The relative importance of contribution/anomaly/temporal drifts over a run
based on which signal actually correlated with follow-through.
([reputation.py:181-198](../src/uavbench/fl/reputation.py#L181-L198))

**19.5 `trimmed_mean`** — not per-client, but used one layer up: a UAV's own
reputation is the **10%-trimmed mean** (per tail) of its assigned clients'
reputations; no trimming for small clusters where the trim floor is 0.
([reputation.py:49-62](../src/uavbench/fl/reputation.py#L49-L62)) This feeds
the server-aggregation gate in §21.

---

# Part III — Federated model & aggregation

## 20. CachedFusionModel architecture

**File:** [model.py](../src/uavbench/fl/model.py). Mirrors
`hflsim.models.MultiModalFusionModel` but replaces the image branch with a
lightweight projection over precomputed ResNet-18 features (no backbone
forward pass during FL training — the dominant CPU-feasibility measure).

```
ImageProjection    : Linear(512→128) + ReLU        (img_proj)
StructuredBranch    : MLP(9→64→128→64), Dropout 0.2  (struct_branch)
FusionHead          : Linear(192→256)+ReLU+Dropout(0.3)+Linear(256→4)  (fusion)
```
4 output classes: Survived / Collapsed / Obstructed / Missing.

**Asymmetric training (paper §IV-B):** the global model's `img_proj` is
frozen (`freeze_img_proj()`) so IoT clients cannot update it — IoT clients
train only `struct_branch + fusion`. UAVs clone the model and call
`unfreeze_img_proj()` to train the full model on their pooled aerial-imagery
shard. Two payload sizes follow from this split:

| Payload | Params | Size |
|---|---|---|
| IoT (struct+fusion) | 67,652 | ≈0.271 MB |
| UAV (img_proj+struct+fusion) | 133,316 | ≈0.533 MB |

`trainable_state_dict()`/`load_trainable_state_dict()` — IoT-level comms
(struct+fusion only). `full_trainable_state_dict()`/
`load_full_trainable_state_dict()` — UAV-level comms (adds img_proj; missing
img_proj keys leave it unchanged, e.g. for `flat_fl`).
([model.py:1-156](../src/uavbench/fl/model.py#L1-L156))

## 21. Aggregation rules (fedavg family)

**File:** same, [model.py](../src/uavbench/fl/model.py). Several weighting
schemes, used at different points in the hierarchy:

- **`fedavg(updates)`** — plain sample-count-weighted average:
  `w_n = n_samples_n / Σn_samples`. Used for UAV-level struct+fusion
  aggregation and for the server level when reputation weighting is off.
- **`reputation_fedavg(updates)`** — weight `= reputation_n × n_samples_n`;
  falls back to plain `fedavg` if all reputations collapse to ≈0. Used at the
  server level when reputation weighting is on (§IV-D).
- **`mixed_fedavg` / `mixed_reputation_fedavg`** — combine one UAV's own
  image-training update with its IoT clients' updates in a single call:
  `img_proj` keys come from the UAV alone (its unique contribution — no
  IoT client can train it); `struct_branch`/`fusion` keys are FedAvg'd (or
  reputation-FedAvg'd, with UAV reputation pinned to 1.0 as trusted
  aggregator) across the UAV plus its IoT clients. Present in the module but
  the live round loop (`federated.py`) instead computes the UAV image update
  and IoT struct+fusion update **separately** and interleaves them explicitly
  (§32) — the UAV's own struct+fusion contribution is deliberately **not**
  mixed in there, to avoid double-counting samples already covered by IoT
  training on the same pooled data.
- **`clone_model`** — deep-copies the model for a client-local or UAV-local
  training pass.

## 22. Dataset adapters & synthetic fallback

> **Superseded (2026-07-14):** the synthetic fallback described below was
> removed from the library — the experimental pipeline is real-data only.
> The offline generator survives solely as a test fixture
> (`tests/uavbench/synthetic_fixture.py`) injected via the harnesses'
> `data.source: prebuilt` seam. Kept for historical reference.

**File:** [dataset.py](../src/uavbench/fl/dataset.py).

- **`ClientData`** — one client's `client_id`, `(lat,lon)`, `train_indices`,
  `test_indices`, and derived `n_samples`.
- **`CachedDataset`** — wraps a `MultiModalDataset`, swapping its
  `(3,128,128)` image tensor for the corresponding row of the precomputed
  512-dim feature cache; item signature `(img_feat(512,), struct(9,), label)`.
- **`make_client_loader`** — builds a client's `DataLoader` with a
  `WeightedRandomSampler` weighted by `1/(class_count+eps)` — inverse-frequency
  balancing so rare damage classes aren't starved within a client's own shard.
- **`SyntheticClientData`** — fully synthetic offline mode (no HF token
  needed): N points inside the Noto Peninsula lat/lon box, 7 synthetic
  z-scored seismic features, damage labels drawn `[0.60, 0.20, 0.10, 0.10]`
  over the 4 classes (heavily imbalanced toward "Survived", reflecting
  reality), random `(N,512)` image features, and an even N/K client
  partition (each client's coordinate = centroid of its shard, not just its
  first point — so a UAV covering that centroid plausibly covers the whole
  shard). ([dataset.py:100-167](../src/uavbench/fl/dataset.py#L100-L167))

## 23. Image feature caching

**File:** [features.py](../src/uavbench/fl/features.py).
`compute_feature_cache` runs a **one-time** frozen, pretrained
`ResNet18(weights=DEFAULT)` (classification head replaced by `Identity`, all
params frozen, ImageNet normalization applied) over the whole dataset, saves
`(N,512)` features as **float16** `.npy` (~5 MB for N=5000, well within a
30 GB disk budget), and every subsequent FL round loads only this cache — no
backbone forward pass occurs during training. `synthetic_feature_cache`
returns random `(N,512)` features for the offline mode.
([features.py](../src/uavbench/fl/features.py))

---

# Part IV — Deployment / real-data pipeline

## 24. HuggingFace streaming loader

**File:** [hflsim/data/loader.py](../src/hflsim/data/loader.py). Instead of a
full snapshot download, rows are **streamed** from
`AbbasABC/HFL-Dataset` (pinned to a specific revision hash) via the
`datasets` library's `IterableDataset` API, keeping only the columns needed
for partitioning + training (`FEATURE_COLS` + `damage_val` + `chip_path`).

**Resilience:** the outer streaming call retries up to 5 times with
exponential backoff (30s, 60s, 120s, 240s, capped) if HF's API gateway
fails mid-stream or the tree-listing 504s under concurrent load — sized to
survive within typical GCP preemption horizons rather than crash a whole job.
([loader.py:200-247](../src/hflsim/data/loader.py#L200-L247))

Deterministic subsampling (`df.sample(n=..., random_state=random_seed)`) is
applied after streaming when `subsample < 1.0`.

## 25. On-demand aerial imagery (GSI tiles)

Images are **not** bundled with the metadata stream — each building chip is
fetched lazily from the Japan GSI XYZ aerial-photo tile API
(`GSI_ZOOM=18`, ~0.6 m/px), composited from a 2×2 tile mosaic and cropped to
a 128×128 RGB chip centered on the device's (lat,lon). Tiles are cached to a
local directory (`HFL_TILE_CACHE`, default `./data/tile_cache`) so repeat runs
don't re-fetch. Network fetch retries 3 times (1s, 2s backoff) before
returning a **black chip** so a flaky connection never stalls the training
loop. ([loader.py:84-133](../src/hflsim/data/loader.py#L84-L133))

`MultiModalDataset` tracks a **black-chip rate** diagnostic
(`black_chip_count / total_image_loads`): since the seismic/structured
features are near-constant across one disaster area, the aerial image is the
*only* modality with real discriminative signal — a high black-chip rate
silently collapses the model to majority-class prediction (the documented
"accuracy/F1 frozen" failure mode), so this counter exists specifically to let
the runner surface that condition instead of failing silently.
([loader.py:302-373](../src/hflsim/data/loader.py#L302-L373))

## 26. Label remapping & feature scaling

Raw `damage_val` codes are `{0, 1, 9, 99}`, **not** contiguous `{0,1,2,3}`:
`9` = "Obstructed view", `99` = "Missing/inconsistent" — real classes, not
sentinels. `_remap_damage_labels` maps `{0→0, 1→1, 9→2, 99→3}` and drops any
row outside that set. (An earlier `isin([0,1,2,3])` filter silently discarded
~16% of rows by treating 9/99 as invalid, collapsing the stated 4-class task
to 2 classes — the current mapping fixes that.)
([loader.py:140-160](../src/hflsim/data/loader.py#L140-L160))

Feature scaling: raw `latitude`/`longitude` are saved **before** any
transform (needed unscaled for GSI tile lookups), then separately normalized
to `[0,1]` for the model input; the 7 seismic columns
(`MMI_original, MMI_shape, PGA, PGV, SA_0_3, SA_1_0, SA_3_0`) are
z-scored via `sklearn.StandardScaler`.
([loader.py:456-478](../src/hflsim/data/loader.py#L456-L478))

## 27. K-means geographic client partitioning

`get_hfl_data_partitions` partitions the full row set into **N clients** via
`KMeans(n_clusters=N, n_init=10)` over `(longitude, latitude)`. Per client,
indices are shuffled with a **per-client seed** (`random_seed + cid`, so
shuffles are independent across clients but still globally reproducible from
one base seed) and split `train_ratio` (default 0.8) train / test. A client's
reported coordinate is the mean lat/lon of its assigned rows (or the dataset
mean, if a cluster ends up empty).
([loader.py:499-533](../src/hflsim/data/loader.py#L499-L533))

## 28. Partition & metadata caching

Two independent caches avoid repeating expensive work across runs:

- **Metadata cache** — the streamed HF DataFrame is written to
  `<data_dir>/.metadata_df_cache_sub<pct>_seed<seed>.parquet` via a
  temp-file + atomic `os.replace` (crash-safe: a partial write never leaves a
  corrupt cache that would break the next run's `read_parquet`).
- **Partition cache** — K-means cluster assignments + train/test index lists
  are pickled to `<data_dir>/.partition_cache/partitions_<sha256[:16]>.pkl`,
  keyed by a hash of `(row_count, N, train_ratio, seed)` — running K-means on
  ~100k+ rows is slow, and the N-sweep (§34) needs it once per N value.

## 29. The PSO→HFL integration bridge

**File:** [hflsim/placement.py](../src/hflsim/placement.py), the "Tier-2
integration point": converts a `{client_id: (lat,lon)}` dict into a
`ProblemInstance` in projected metric space, runs the requested optimizer
(`pso`/`ga`/`centroid`/`random`/`static` via `uavbench.optimizers.REGISTRY`),
and converts the result back into a list of
[`UAVAggregator`](../src/hflsim/simulation/uav.py) objects at the optimized
lat/lon positions. When `prev_positions` isn't supplied, it defaults to an
even linspace spread across the bounding box at mid-altitude rather than the
coordinate-projection origin, avoiding a placement bias toward `(0,0)`.
([placement.py:25-116](../src/hflsim/placement.py#L25-L116))

`UAVAggregator` itself (legacy `hflsim.simulation` scaffolding, still used
only by this bridge) tracks `battery`, `assigned_clients`, and `reputation`,
and exposes an `edge_aggregate()` FedAvg helper — superseded in the live
round loop by the `uavbench/fl/model.py` aggregation functions (§21), which
are what `federated.py` actually calls.

---

# Part V — Orchestration & experiment infrastructure

## 30. Tier-1 runner (placement-only benchmark)

**File:** [runner.py](../src/uavbench/runner.py). A *run* is one
`(method, scenario, seed)` triple. `run_experiment` builds the full grid
(`methods × scenarios × n_seeds`) and executes it via `joblib.Parallel`
(worker count overridable via the `UAVBENCH_N_WORKERS` env var).

`_build_optimizer(method, budget, method_params)` instantiates the optimizer
from `REGISTRY`; for `pso`/`ga`, the shared `budget.P`/`budget.G_max`
**always override** any `optimizer_params.<method>` values from the config,
guaranteeing every optimizer spends an identical evaluation budget regardless
of what ablation parameters are being tested.
([runner.py:35-47](../src/uavbench/runner.py#L35-L47))

Per run: `generate_instance` (deterministic from the instance seed) →
`Fitness` → optimizer `.optimize()` → `compute_metrics`. Results are written
to `runs.parquet` (metrics) and `convergence.parquet` (best-so-far per
iteration per run), with `config.resolved.yaml` persisted alongside for
exact reproducibility.

## 31. Tier-1 post-hoc metrics

**File:** [metrics/placement.py](../src/uavbench/metrics/placement.py).
Re-evaluates the optimizer's returned best position **once**, on a *fresh*
`Fitness` instance (so the run's own eval-count budget is untouched), to
recover the coverage/movement/balance breakdown plus:

- **`evals_to_threshold`** — first iteration index reaching `95%` of the
  final best fitness (`-1` if never, e.g. non-positive best).
- **`convergence_auc`** — trapezoidal area under the best-fitness-vs-iteration
  curve, normalized by `G_max` (the *shared* budget), **not** by the trace's
  own length — early-stopped methods have shorter traces, and normalizing by
  `len-1` would inflate their AUC relative to methods that ran the full
  budget. The trace is extended flat at its final value out to `G_max` before
  integrating (same plateau-extension convention used for convergence-plot
  confidence bands). ([metrics/placement.py:36-58](../src/uavbench/metrics/placement.py#L36-L58))
- Movement is also converted to **Joules and battery fraction** via
  `EnergyModel`, purely for reporting (§10).

## 32. Tier-2 / full-system round loop

**File:** [federated.py](../src/uavbench/fl/federated.py). Two entry points:

- **`run_tier2`** — simpler placement→coverage→plain-FedAvg loop (no client
  selection, no reputation): every covered client participates every round
  it's placed. Used for the pure placement-method comparison inside a real FL
  setting.
- **`run_full_hfl`** — the complete paper system (§0 diagram), described in
  detail in Part II/III above. Per round: refresh device/reputation state →
  check the early-reselection trigger → placement (if due) → selection (if
  due) → UAV image training + IoT structured-data training on selected
  clients only → mark absent clients → reputation update → UAV-level then
  reputation-gated server-level aggregation → device-state advance →
  evaluate.

**Early-reselection trigger** (paper §IV-E6):
```python
n_eligible = count of currently-eligible devices
low_eligible = n_eligible < lambda_min * min(K*capacity, len(clients))
```
`lambda_min` defaults to 0.5. The `min(K*capacity, len(clients))` cap handles
configs where total UAV capacity exceeds the client population (the paper
assumes N≫ΣC_u, which not every config satisfies) — without the cap the
threshold would be trivially always-true.
([federated.py:788-793](../src/uavbench/fl/federated.py#L788-L793))
`low_eligible` forces **both** an early placement re-run and an early
reselection, since a sudden eligibility drop likely means the layout is
stale.

**Reselection cadence:** `reselect = (rnd-1) % T_sel == 0 or low_eligible or not selected`
— selection is **not** recomputed every round; the `ClientSelector` instance
persists across the whole run so UCB counts accumulate correctly, and between
reselections the previous roster is reused as-is.
([federated.py:848-850](../src/uavbench/fl/federated.py#L848-L850))

**Server-aggregation reputation gate (§IV-D):** when reputation weighting is
on, UAVs whose cluster's trimmed-mean reputation falls below `R_min` (default
0.3) are excluded entirely from that round's server aggregation:
```python
active = [u for u in uav_updates if u.reputation >= R_min]
server_agg = reputation_fedavg(active) if active else <model unchanged>
```
([federated.py:979-987](../src/uavbench/fl/federated.py#L979-L987))

**Communication accounting:** `flat_fl` counts only IoT↔server payload
(struct+fusion, `_IOT_MODEL_SIZE_MB`); hierarchical methods count
IoT↔UAV (`_IOT_MODEL_SIZE_MB`) **plus** UAV↔server (`_UAV_MODEL_SIZE_MB`,
img_proj+struct+fusion), each counted uplink+downlink (`2.0×`).

## 33. Method configuration table

```python
_METHOD_CFG = {
    "proposed_hfl":      ("pso", "ucb",    True,  True),
    "flat_fl":           (None,  "all",    False, False),
    "centralized":       (None,  "all",    False, False),   # handled specially
    "hfl_no_selection":  ("pso", "random", True,  True),
    "hfl_static":        ("pso", "ucb",    True,  False),
    "hfl_no_reputation": ("pso", "ucb",    False, True),
}
# tuple = (placement_method, selection_mode, reputation_weighted, dynamic)
```
`placement_method=None` → `flat_fl`/`centralized`: no UAV hierarchy, all
clients always "covered" (static, no dropouts). `dynamic=False` (`hfl_static`)
places once at round 1 and never repositions. Every ablation ("drop UCB?",
"drop reputation weighting?", "static placement?") is a one-line
configuration change reusing the identical code paths.
`centralized` bypasses the federated path entirely via `_run_centralized` —
an oracle that trains the whole model (img_proj unfrozen) on all pooled data
at one node, the upper-bound reference.
([federated.py:533-540](../src/uavbench/fl/federated.py#L533-L540),
[federated.py:601-648](../src/uavbench/fl/federated.py#L601-L648))

## 34. N-scalability sweep

**File:** [sweep.py](../src/uavbench/fl/sweep.py). Runs the full
`(N_values × methods)` grid in parallel across the machine's vCPUs.

- **Sequential HF pre-fetch first:** `_prefetch_all_N` streams and caches
  every N value's dataset (metadata + partitions + ResNet feature `.npy`)
  **before any parallel worker starts**, so parallel workers only ever touch
  local disk — zero concurrent HF API calls, avoiding rate-limit failures
  under parallel load.
- **Thread budget:** each worker process calls `torch.set_num_threads(1)` so
  MKL/OpenBLAS doesn't oversubscribe; total active threads = `n_workers × 1`,
  matched to the vCPU budget.
- Each `(N, method)` job gets its own `results_dir/N{N}/` subdirectory.

## 35. Reproducibility — RNG stream discipline

Two independent RNG streams are used throughout, so **every method sees the
identical problem instance** (paired comparison) while each optimizer's own
stochasticity is independently reproducible:

- **Instance RNG:** `SeedSequence([base, scenario_idx, seed_i])` → governs
  only device/scenario generation.
- **Optimizer RNG:** `SeedSequence([base, method_idx, scenario_idx, seed_i])`
  → governs only optimizer stochasticity, keyed additionally by method so
  different methods don't share a stream.
  ([runner.py:50-56](../src/uavbench/runner.py#L50-L56))

In the full-system harness, per-method seeds combine the run-level seed with
a stable MD5 hash of the method name (`_seed = (run_seed ^ method_hash) %
2**31`), applied **exactly once** — a comment in the code explicitly warns
callers (e.g. paper-reproduction jobs) not to pre-encode the method identity
into `run_seed` themselves, to avoid double-hashing.
([federated.py:729-735](../src/uavbench/fl/federated.py#L729-L735))
`torch.manual_seed(_seed)` is also set per method so model initialization is
deterministic across runs/ablations.

---

# Part VI — Reference tables

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
| Energy model `p_fly/p_hover/cruise/t_serve/capacity` | 250W / 200W / 15m/s / 60s / 200,000 J |

### Client selection & reputation

| Parameter | Value |
|---|---|
| `B_MIN, SNR_MIN_DB, T_MAX_S` | 0.20, 3.0 dB, 300 s |
| Priority weights `W_BATTERY/LEARNING/UTILITY` | 0.35 / 0.30 / 0.35 |
| Utility sub-weights `EPI/SNR/DENS/PROX` | 0.4 / 0.3 / 0.2 / 0.1 |
| `UCB_C` | √2 |
| `T_decay` (shared β schedule) | 20 rounds |
| Battery discharge/recharge per round | −0.02 / +0.005 |
| SNR / compute noise per round | N(0,2) dB / N(0,30) s |
| Adaptive margin | 1.96·σ(last 10 compute times) |
| `T_sel`, `lambda_min`, `R_min` | 5 rounds, 0.5, 0.3 |
| Reputation init weights (contrib/anomaly/temporal) | 0.4 / 0.3 / 0.3, Dirichlet-adapted /10 rounds |
| Reputation EMA `_VEC_EMA_NEW/OLD`, `_STATS_ALPHA` | 0.7/0.3, 0.1 |
| `_PRIOR_STRENGTH` | 20.0 |
| Mahalanobis anomaly threshold | d ≤ 2 → R_anomaly = 1 |
| `trimmed_mean` trim | 10% each tail |

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

---

**In one sentence:** the system is a constriction-factor lbest-ring PSO
placing UAVs against a value-weighted, movement- and balance-penalized
objective; a UCB1-driven, reputation-informed client-selection pipeline that
gates who trains each round and who is trusted enough to be aggregated; an
asymmetric-training fusion model bridging cached vision features with
per-round structured seismic data; and a real-world deployment pipeline that
streams a HuggingFace dataset, fetches aerial imagery on demand from Japan's
GSI tile service, and geographically partitions it into clients — all wired
together by a reproducible, two-RNG-stream experiment harness where every
ablation is a one-line configuration change.
