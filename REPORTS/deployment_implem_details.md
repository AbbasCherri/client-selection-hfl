# Optimizers and Positioning Algorithms — Implementation Reference

This document is the authoritative summary of all positioning optimizers in this
repository (PSO, GA, and heuristic baselines), the shared objective, device-value
model, assignment routine, seeding utilities, energy model, and how the runner
configures and invokes each method. All claims are derived directly from the code.

---

## PSO (Particle Swarm Optimization)

**File:** [src/uavbench/optimizers/pso.py](src/uavbench/optimizers/pso.py)

### Algorithm family
Constriction-factor PSO (Clerc & Kennedy 2002). Optionally falls back to
linearly-decayed inertia weight instead of constriction.

### Constriction factor
```
phi = c1 + c2          # must be > 4 for the formula to be real
chi = 2 / |2 - phi - sqrt(phi^2 - 4*phi)|
```
With defaults `c1 = c2 = 2.05`, `phi = 4.1`, giving `chi ≈ 0.729`. The code
raises `ValueError` if `phi <= 4`.

### Velocity update (constriction mode, default)
```
Vel ← chi * (Vel + c1*r1*(pbest - X) + c2*r2*(nbest - X))
```
### Velocity update (inertia mode, `use_constriction=False`)
```
w = inertia_max - (inertia_max - inertia_min) * (tau / G_max)   # linear decay
Vel ← w * Vel + c1*r1*(pbest - X) + c2*r2*(nbest - X)
```
Default `inertia_max=0.9`, `inertia_min=0.4`.

### Topology
- **Ring (default, `topology="ring"`)**: local-best ring with symmetric
  neighbourhood of size `2*ring_k + 1`. Default `ring_k=2` → 5-particle
  neighbourhood per particle. `ring_k=1` gives the classic 3-particle lbest ring.
- **Global best (`topology="gbest"`)**: all particles share the single global best.

### Initialization
- **`seeding="value_kmeans"` (default)**: 50% of the swarm (`P//2` particles)
  seeded by running `kmeanspp_centers` with device-value weights, jittered with
  `N(0, jitter_m)` noise (default `jitter_m=10.0 m`); 50% drawn uniformly.
- **`seeding="plain_kmeans"`**: same as above but weights=None (unweighted k-means++).
- **`seeding="uniform"`**: all particles drawn uniformly.

Initial velocity: `0.5 * Uniform(-vmax, vmax)` per dimension.

### Per-dimension velocity bound
```
vmax[d] = vmax_frac * (hi[d] - lo[d])     # default vmax_frac=0.2
```

### Boundary handling
Absorbing walls: positions clamped to `[lo, hi]`; velocity components that
caused a violation are zeroed.

### Turbulence (`use_turbulence=True`)
Each iteration a random fraction `p_turb` (default 0.1) of particles receive a
velocity kick drawn from `Uniform(-0.1*vmax, 0.1*vmax)`, applied before clamping.

### Stagnation reinitialization (`use_stagnation=True`)
If the global-best improvement over consecutive iterations stays below
`delta_stag=1e-4` for `G_stag=20` generations, the worst `floor(rho*P)` particles
(default `rho=0.2` → 20 particles) are replaced with fresh uniform samples and
their personal bests reset. The global best is re-evaluated immediately after.

### Global best
`gbest` is updated only when a strictly better value is found (monotonically
non-decreasing). It is never overwritten with an equal or worse value.

### Early stopping
If `gbest_fit >= early_stop_frac * w1` (default threshold 0.95 × 0.6 = 0.57),
the loop terminates before `G_max` iterations.

### Default hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `P` | 100 | Swarm size |
| `G_max` | 200 | Max generations |
| `c1`, `c2` | 2.05, 2.05 | Cognitive / social coefficients |
| `vmax_frac` | 0.2 | Per-dim velocity clamp fraction |
| `ring_k` | 2 | Ring half-width (neighbourhood = 5) |
| `delta_stag` | 1e-4 | Stagnation improvement threshold |
| `G_stag` | 20 | Stagnation window (iterations) |
| `rho` | 0.2 | Fraction of worst particles reinitialized |
| `p_turb` | 0.1 | Turbulence probability per particle |
| `early_stop_frac` | 0.95 | Fraction of theoretical max for early exit |
| `jitter_m` | 10.0 | Seeding jitter std-dev (meters) |
| `inertia_max/min` | 0.9 / 0.4 | Inertia range (inertia mode only) |

---

## GA (Genetic Algorithm)

**File:** [src/uavbench/optimizers/ga.py](src/uavbench/optimizers/ga.py)

### Algorithm family
Real-coded genetic algorithm with SBX (simulated binary crossover) and polynomial
mutation (Deb & Agrawal 1999). Run at the same `P` / `G_max` budget as PSO for
a fair head-to-head evaluation count.

### Initialization
Uniform random over `[lo, hi]` (no warm-start seeding, unlike PSO).

### Selection
Binary tournament selection: `tournament_size` random indices drawn per parent;
the individual with the highest fitness wins.

### Crossover (SBX)
Applied with probability `crossover_prob` (default 0.9). Per gene, crossover
occurs with probability 0.5 (the `u <= 0.5` gate inside `_sbx`). The spread
factor `beta` is drawn such that offspring cluster near parents for large `eta_c`:
```
beta = (2u)^(1/(eta_c+1))          if u <= 0.5
beta = (1/(2(1-u)))^(1/(eta_c+1))  otherwise
child1 = 0.5*((1+beta)*p1 + (1-beta)*p2)
child2 = 0.5*((1-beta)*p1 + (1+beta)*p2)
```
Children are clipped to `[lo, hi]`.

### Mutation (bounded polynomial)
Per gene with probability `mutation_prob` (default `1/dim`):
```
delta = (2u)^(1/(eta_m+1)) - 1           if u < 0.5   → range [-1, 0]
delta = 1 - (2(1-u))^(1/(eta_m+1))       otherwise     → range [0,  1]
x_new = x + delta*(x - lo)   if u < 0.5
x_new = x + delta*(hi - x)   otherwise
```
Guarantees offspring lies in `[lo, hi]` without clipping.

### Elitism
The top `n_elite` individuals from the current generation are copied directly
into the next generation before the tournament/crossover/mutation loop fills the
remainder.

### Early stopping
Same threshold as PSO: `best_fit >= early_stop_frac * w1` (default 0.57).

### Default hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `P` | 100 | Population size |
| `G_max` | 200 | Max generations |
| `crossover_prob` | 0.9 | Parent pair crossover probability |
| `eta_c` | 15.0 | SBX distribution index |
| `eta_m` | 20.0 | Polynomial mutation distribution index |
| `mutation_prob` | `1/dim` | Per-gene mutation probability |
| `tournament_size` | 3 | Tournament selection size |
| `n_elite` | 2 | Elites carried to next generation |
| `early_stop_frac` | 0.95 | Fraction of theoretical max for early exit |

---

## Heuristic Baselines

**File:** [src/uavbench/optimizers/heuristics.py](src/uavbench/optimizers/heuristics.py)

### Centroid
Value-weighted k-means centroids via `weighted_kmeans`. Single fitness
evaluation — deterministic given the RNG seed. `altitude_frac` (default 0.5)
sets UAV z-height as a fixed fraction of the altitude range. Toggle
`value_weighted=False` for unweighted k-means centroids.

**Role**: fast deterministic competitor; expected to lose on the joint objective
(blind to capacity saturation and the movement penalty).

### RandomPlacement
Best of `n_draws=20` uniform random candidates. Each candidate is evaluated
once; the best is returned. Serves as the no-intelligence floor.

### Static
UAVs remain at `instance.prev_positions`. One evaluation. Used to measure the
marginal value of dynamic repositioning — the ablation-style lower bound.

---

## Seeding & Clustering Utilities

**File:** [src/uavbench/optimizers/seeding.py](src/uavbench/optimizers/seeding.py)

### `kmeanspp_centers(rng, points, K, weights=None)`
k-means++ initialization with optional value-proportional sampling:
- First center sampled with probability `∝ weights` (or uniform if `weights=None`).
- Subsequent centers sampled with probability `∝ D²(x) * weights`, where `D²(x)`
  is squared distance to the nearest already-chosen center.
- Fallback to uniform index if total weight is zero or non-finite.

### `weighted_kmeans(rng, points, K, weights=None, n_iter=25)`
Lloyd's algorithm seeded by `kmeanspp_centers`. Centroid updates use weighted
averages: `c_k = sum(w_i * x_i) / sum(w_i)` for devices assigned to cluster k.
Runs up to `n_iter=25` iterations, stopping early on convergence.

---

## Objective Function, Assignment, and Constraints

**Files:** [src/uavbench/problem/fitness.py](src/uavbench/problem/fitness.py),
[src/uavbench/problem/assignment.py](src/uavbench/problem/assignment.py)

### Scalar objective (maximization)
```
F(X) = w1 * (F_cover / F_max)
      - w2 * (D_move  / D_max)
      - w3 * (L_imb   / L_max)
```

**Default weights:** `w1=0.6`, `w2=0.3`, `w3=0.1`

**Terms and normalizers:**

| Term | Definition | Normalizer |
|------|-----------|-----------|
| `F_cover` | Sum of V_i over assigned devices | `F_max = sum_i V_i` |
| `D_move` | `sum_j ‖p_j − p_prev_j‖` (total travel distance) | `D_max = K * box_diagonal` |
| `L_imb` | `sum_j (loads_j − n_assigned/K)²` (load variance) | `L_max = N²` |

`L_imb` uses `n_assigned/K` (actual mean load) not `N/K`, so it penalizes
imbalance among served devices only.

The theoretical upper bound is `F ≤ w1 = 0.6` (full coverage, no movement, zero
imbalance). PSO and GA early-stop at `early_stop_frac * w1 = 0.57`.

### Greedy assignment (per-eval)
`greedy_assignment` in [assignment.py](src/uavbench/problem/assignment.py):

1. Sort devices by descending `value` (stable sort, `O(N log N)`).
2. For each device (highest value first): find feasible positions — those
   satisfying all three hard constraints simultaneously:
   - **Range**: 3D Euclidean distance ≤ `R_comm` (default 500 m)
   - **Battery**: `battery[j] >= B_min_uav` (default 0.2)
   - **Capacity**: `loads[j] < capacity[j]`
3. Assign to the feasible position with the smallest current load; ties broken
   by smallest distance (via `np.lexsort`).
4. Devices with no feasible position are left unserved — no penalty, no repair.

Hard constraints are enforced **implicitly**: an unservable device simply
contributes zero to `F_cover`. This keeps the fitness landscape clean and
avoids penalty-coefficient tuning.

**Complexity:** `O(N log N)` sort + `O(N×K)` assignment sweep; distance matrix
`(N, K)` is computed once in vectorised NumPy before the loop.

### Fitness callable
`Fitness` tracks `eval_count` so the runner can verify every optimizer spends the
same evaluation budget. All optimizers call the same `Fitness` object — they
cannot implement their own scoring.

---

## Device Value Model

**File:** [src/hflsim/shared/value.py](src/hflsim/shared/value.py)
(re-exported via [src/uavbench/problem/value.py](src/uavbench/problem/value.py))

### Value formula
```
V_i(t) = beta(t) * U_i(t) + (1 - beta(t)) * R_i(t)
beta(t) = max(0, 1 - t / T_decay)     # T_decay=20 by default
```
- `beta=1` (early rounds): value is pure utility (responsive to current state).
- `beta=0` (after T_decay rounds): value is pure reputation (history-based).
- `beta_mode="pinned"` forces `beta=1` regardless of `t` (utility-only benchmark
  used in most experiments).

### Utility U_i — four features (weights sum to 1.0)

| Feature | Weight | Computation |
|---------|--------|-------------|
| Epicenter proximity | 0.4 | `max(0, (d95 - min(d_epi, d95)) / d95)`; d95 = 95th-percentile distance to epicenter |
| SNR | 0.3 | Min-max normalized over the cluster |
| Sample density | 0.2 | `min(1, samples / (0.5 * max_samples))`; capped at 2× the median |
| Nearest-UAV proximity | 0.1 | `1 - minmax(min_j dist(device_i, prev_j))`; high score = device already near a UAV |

### Raw feature distributions (synthesized in `generate_instance`)
- SNR: `Uniform(0, 30)` dB
- Samples: `Uniform_int(20, 200)`
- Reputation `R_i`: drawn from `Beta(2, 2)` (bell-shaped on [0,1]), held fixed
  across rounds within one instance

---

## Problem Instance

**File:** [src/uavbench/problem/instance.py](src/uavbench/problem/instance.py)

### `ProblemInstance` fields

| Field | Shape | Description |
|-------|-------|-------------|
| `device_coords` | `(N, 3)` | Ground device positions (z=0 for IoT devices) |
| `value` | `(N,)` | Fixed per-device score V_i(t) |
| `capacity` | `(K,)` | Max devices per UAV hover position |
| `battery` | `(K,)` | UAV battery fraction ∈ [0,1] |
| `prev_positions` | `(K, 3)` | Previous UAV locations (movement penalty baseline) |
| `lower / upper` | `(3,)` | Per-dimension search bounds [x,y,z] |
| `R_comm` | scalar | Communication range (m), default 500 |
| `B_min_uav` | scalar | Min battery for usable position, default 0.2 |

Search space dimension: `dim = 3K` (x,y,z per UAV, flat vector).

### `generate_instance` distributions
- **`uniform`**: devices uniform over `[x_lo, x_hi] × [y_lo, y_hi]`.
- **`clustered`**: `sqrt(N)/2` Gaussian clusters with `sigma = 0.06 * span`.
- **`epicenter_biased`**: single Gaussian centred on epicenter with `sigma = 0.12 * span`.

### `prev_mode` (movement penalty baseline)
- **`"stale"` (default)**: previous positions fitted to a *shifted* epicenter
  (offset by `N(0, 0.25*span)`). Models the situation that triggered a
  repositioning: the old layout no longer fits the current state, so `static`
  is a genuine floor.
- **`"warm"`**: previous positions near current device sub-centroids (already
  near optimal). Studies the conservative regime where holding position is nearly best.

### Coordinate system
All coordinates in projected meters (equirectangular about a reference point).
3D Euclidean distance `sqrt(dx^2 + dy^2 + dz^2)` is used throughout (range gate,
movement cost). Ground devices sit at z=0; UAVs are positioned at altitude z ∈ [z_lo, z_hi].

---

## Energy Model (reporting only)

**File:** [src/uavbench/problem/energy.py](src/uavbench/problem/energy.py)

The `EnergyModel` converts abstract reposition distance to physical energy units
for *reporting metrics only* — it is **never called inside the fitness function**,
so the optimizer objective is identical across methods.

```
E_move = P_fly * (d / v) + P_hover * t_serve
```

| Constant | Default | Meaning |
|----------|---------|---------|
| `p_fly` | 250 W | Cruise power draw |
| `p_hover` | 200 W | Hover power draw |
| `cruise_speed` | 15 m/s | Horizontal cruise speed |
| `t_serve` | 60 s | Hover service time per round |
| `battery_capacity_j` | 200 000 J | Usable battery energy |

Battery fraction consumed = `E_move / battery_capacity_j`.

---

## Runner & Configuration

**File:** [src/uavbench/runner.py](src/uavbench/runner.py)

### Optimizer instantiation
`_build_optimizer(method, budget, method_params)` looks up the optimizer class
from `REGISTRY` (defined in [src/uavbench/optimizers/\_\_init\_\_.py](src/uavbench/optimizers/__init__.py)).
For `pso` and `ga`, `P` and `G_max` from the shared `budget` block always
override any `method_params` values, ensuring equal evaluation budgets.

The `optimizer_params.<method>` section of the YAML config can pass any
additional keyword arguments for ablation studies (e.g., toggle
`use_constriction`, `topology`, `seeding`) without touching source code.

### Reproducibility — two independent RNG streams
- **Instance RNG**: `SeedSequence([base, scenario_idx, seed_i])` → deterministic
  device layout. Every method sees the **same** instance (paired comparison).
- **Optimizer RNG**: `SeedSequence([base, method_idx, scenario_idx, seed_i])` →
  independent stochasticity per (method, scenario, replicate) triple.

### Output format
Per-run metrics saved to `runs.parquet` (fallback `runs.csv`); convergence
traces (best-so-far per iteration) saved to `convergence.parquet`. The
fully-resolved config is persisted as `config.resolved.yaml` next to the
results.

### Post-hoc metrics
`compute_metrics` in [src/uavbench/metrics/placement.py](src/uavbench/metrics/placement.py)
receives the `EnergyModel` and computes energy/battery metrics from the result's
best position after the optimizer returns — these are never fed back to the optimizer.

---

## Key Files Reference

| File | Role |
|------|------|
| [src/uavbench/optimizers/pso.py](src/uavbench/optimizers/pso.py) | Constriction PSO |
| [src/uavbench/optimizers/ga.py](src/uavbench/optimizers/ga.py) | Real-coded GA |
| [src/uavbench/optimizers/heuristics.py](src/uavbench/optimizers/heuristics.py) | Centroid, Random, Static |
| [src/uavbench/optimizers/seeding.py](src/uavbench/optimizers/seeding.py) | k-means++ and Lloyd's |
| [src/uavbench/optimizers/base.py](src/uavbench/optimizers/base.py) | Optimizer ABC, Result |
| [src/uavbench/problem/fitness.py](src/uavbench/problem/fitness.py) | Scalarized objective F(X) |
| [src/uavbench/problem/assignment.py](src/uavbench/problem/assignment.py) | Greedy assignment |
| [src/uavbench/problem/instance.py](src/uavbench/problem/instance.py) | ProblemInstance, generate_instance |
| [src/uavbench/problem/energy.py](src/uavbench/problem/energy.py) | Rotary-wing energy model (reporting) |
| [src/hflsim/shared/value.py](src/hflsim/shared/value.py) | Device value V_i(t) |
| [src/uavbench/runner.py](src/uavbench/runner.py) | Experiment grid, parallel execution |

---
Last verified against code: 2026-07-03.
