# The Proposed PSO Algorithm for 3D UAV Placement

This document gives a full, self-contained explanation of the proposed particle
swarm optimizer for UAV placement in this repository: the problem it solves, the
objective it optimizes, the core PSO mechanics, the four enhancements layered on
top of vanilla PSO, and how it integrates into the hierarchical federated
learning (HFL) pipeline. All claims are derived directly from the code.

---

## 1. The problem being solved

The algorithm places **K UAV aggregators in 3D space** over a disaster area so
they can serve **N ground IoT devices** participating in hierarchical federated
learning. Each device sits at ground level (z = 0) inside a bounding box, and
each UAV hovers at some position (x, y, z) with z constrained to an altitude
band. A placement is good when it:

1. **Covers valuable devices** — devices within communication range `R_comm` of
   a non-full, sufficiently-charged UAV,
2. **Moves the fleet as little as possible** from the previous layout
   (repositioning costs energy and time), and
3. **Balances load** across UAVs so no single aggregator becomes a bottleneck.

This is a continuous, non-convex, multimodal optimization problem in **3K
dimensions** — exactly the regime population-based metaheuristics are designed
for ([pso.py](../src/uavbench/optimizers/pso.py)).

### Solution encoding

Each particle is a flat vector **X ∈ ℝ³ᴷ**: the K UAV positions concatenated as
(x₁, y₁, z₁, …, x_K, y_K, z_K). Per-dimension bounds are the area box tiled K
times ([base.py:63-68](../src/uavbench/optimizers/base.py#L63-L68)), and
[`positions_from_vector`](../src/uavbench/problem/instance.py#L99-L101)
reshapes it back to (K, 3) for evaluation.

---

## 2. The fitness function (what a particle is scored on)

Every optimizer in the benchmark scores candidates through one shared callable
([fitness.py](../src/uavbench/problem/fitness.py)), guaranteeing identical
objectives and evaluation budgets across methods. It is a **maximized**,
scalarized three-term objective:

```
F(X) = w1 · (F_cover / F_max)  −  w2 · (D_move / D_max)  −  w3 · (L_imb / L_max)

w1 = 0.6,  w2 = 0.3,  w3 = 0.1
```

**Coverage term** `F_cover = Σ Vᵢ` over assigned devices — not a raw device
count. Each device carries a value score **Vᵢ(t) = β(t)·Uᵢ + (1−β(t))·Rᵢ**,
blending a four-feature utility Uᵢ (epicenter proximity 0.4, SNR 0.3,
sample-density proxy 0.2, nearest-UAV proximity 0.1) with the device's FL
reputation Rᵢ. The schedule β(t) = max(0, 1 − t/20) starts fully utility-driven
and shifts toward reputation as training history accumulates
([value.py](../src/hflsim/shared/value.py)). Normalized by
`F_max = Σᵢ Vᵢ` (the value of covering everyone).

**Movement penalty** `D_move = Σⱼ ‖pⱼ − pⱼ_prev‖` — total 3D displacement from
the previous UAV layout, normalized by `K × (box diagonal)`. This is what
makes the algorithm suitable for *reconfiguration*: when a trigger fires
mid-mission, the optimizer trades off better coverage against the cost of
flying there.

**Load-imbalance penalty** `L_imb = Σⱼ (|A(j)| − N_assigned/K)²` — squared
deviation of each UAV's load from the mean, normalized by N².

### Implicit constraint handling via greedy assignment

Inside every fitness evaluation, devices are matched to UAV positions by a
deterministic greedy routine ([assignment.py](../src/uavbench/problem/assignment.py)):
devices are processed in **descending value order**, and each goes to the
feasible position (in range, under capacity, battery ≥ B_min) with the
**smallest current load**, ties broken by distance. A device with no feasible
UAV is simply left uncovered and earns no credit.

This design choice matters: the hard constraints (range, capacity, battery)
are enforced *implicitly* — there is no penalty term and no repair operator,
so the fitness landscape stays clean and the PSO never has to navigate
infeasibility cliffs. The assignment costs `O(N log N + N·K)`, keeping each
evaluation cheap.

---

## 3. The PSO core: constriction + ring topology

The proposal is a **constriction-factor PSO with an lbest ring topology**
(Clerc & Kennedy 2002), rather than vanilla inertia-weight gbest PSO. Two
decisions define it:

**Constriction factor.** With φ = c1 + c2 = 2.05 + 2.05 = 4.1, the velocity
update is

```
V ← χ · (V + c1·r1 ⊙ (pbest − X) + c2·r2 ⊙ (nbest − X))
χ = 2 / |2 − φ − √(φ² − 4φ)| ≈ 0.7298
```

χ is *derived* from c1, c2 at construction
([pso.py:62-65](../src/uavbench/optimizers/pso.py#L62-L65)) — never
hardcoded — so changing the acceleration coefficients in an ablation can't
silently break the convergence guarantee, and the code raises if φ ≤ 4 where
the formula is invalid. Constriction gives provable convergence pressure
without hand-tuning an inertia schedule (an inertia-weight fallback, w
decaying 0.9 → 0.4, exists purely as an ablation toggle).

**Ring topology (lbest) instead of gbest.** Each particle learns from the best
`pbest` in its neighborhood of **2k+1 particles on a ring** (k = 2 →
5-particle neighborhoods), computed fully vectorized in NumPy
([pso.py:108-128](../src/uavbench/optimizers/pso.py#L108-L128)). Information
about a good region spreads gradually around the ring rather than instantly
to everyone, which preserves diversity and resists premature convergence —
important here because the coverage landscape has many local optima
(different cluster-to-UAV pairings score similarly).

---

## 4. The four proposed enhancements

### 4.1 Value-weighted k-means++ warm start

Half the swarm (P/2 particles) is not initialized uniformly. Instead, each
seeded particle's K positions are drawn as **k-means++ centers over device
(x, y) coordinates, with sampling probability weighted by device value Vᵢ**
([seeding.py](../src/uavbench/optimizers/seeding.py)): the first center is
sampled ∝ value, subsequent centers ∝ (squared distance to nearest center) ×
value. Centers get Gaussian jitter (σ = 10 m) so no two particles are
identical, plus a uniform random altitude. The other half stays uniform for
exploration ([pso.py:82-104](../src/uavbench/optimizers/pso.py#L82-L104)).

The intuition: a good placement almost certainly hovers UAVs over high-value
device clusters, so the swarm starts with several diverse, already-plausible
layouts instead of burning generations rediscovering the obvious.

### 4.2 Per-dimension velocity clamping + absorbing walls

Velocities are clamped per dimension to `vmax = 0.2 × (upper − lower)`, so a
particle can't overshoot the box in one step and x/y (kilometers) and z (a
narrow altitude band) get proportionate limits. At the boundary, **absorbing
walls** apply: out-of-bound coordinates are clipped to the box and *their
velocity component zeroed*
([pso.py:180-183](../src/uavbench/optimizers/pso.py#L180-L183)), so particles
settle on the boundary rather than bouncing or repeatedly slamming into it.

### 4.3 Stagnation-triggered partial reinitialization

The algorithm tracks improvement of the global best. If gbest improves by
less than δ = 10⁻⁴ for **G_stag = 20 consecutive generations**, the **worst
ρ = 20% of particles** are replaced with fresh uniform samples (with reset
velocities and pbests), and gbest is immediately re-checked
([pso.py:202-215](../src/uavbench/optimizers/pso.py#L202-L215)). The best
particles are untouched, so the incumbent solution is never lost — this
injects fresh diversity exactly when the swarm has collapsed, at zero cost to
exploitation.

### 4.4 Turbulence

Each generation, every particle independently receives, with probability
p = 0.1, a small random velocity kick uniform in ±0.1·vmax
([pso.py:169-173](../src/uavbench/optimizers/pso.py#L169-L173)). This is a
continuous, mild counterpart to reinitialization: it prevents micro-stagnation
(particles orbiting a pbest at near-zero velocity) without disrupting
convergence.

---

## 5. The complete loop

```
1. Bounds ← tile (lower, upper) K times;  vmax ← 0.2·(hi − lo)
2. Init: P/2 value-weighted k-means++ particles + P/2 uniform;
   velocities ~ 0.5·U(−vmax, vmax)
3. Evaluate all; set pbest, gbest
4. For tau = 1 … G_max (200):
   a. nbest ← ring-neighborhood best of each particle's 5-neighborhood
   b. V ← chi*(V + cognitive + social);  turbulence kicks;  clamp to ±vmax
   c. X ← X + V;  absorbing walls (clip + zero velocity on violated dims)
   d. Evaluate; update pbest elementwise; update gbest monotonically
   e. Stagnation counter: reset if gbest improved by > delta, else increment;
      at 20, reinitialize worst 20% of the swarm
   f. Early stop if gbest >= 0.95*w1   (95% of the theoretical maximum:
      full coverage with zero movement and imbalance would score exactly w1)
5. Return gbest position, fitness, convergence curve, eval count, chi and phi
```

The early-stop threshold at
[pso.py:148-149](../src/uavbench/optimizers/pso.py#L148-L149) exploits the
objective's structure: since the penalty terms are subtractive and coverage is
normalized, w1 = 0.6 is a hard ceiling, so 0.95·w1 means "a near-perfect layout
was found — stop spending budget."

---

## 6. Rigor and integration

Three engineering properties are worth calling out as part of the proposal:

- **Fair benchmarking by construction.** The `Fitness` object counts every
  evaluation, and the base `Optimizer` class forces all methods (PSO, GA,
  centroid, random, static baselines) through the same objective and greedy
  assignment ([base.py](../src/uavbench/optimizers/base.py)), so comparisons
  are on identical budgets.
- **Determinism.** The instance-generation seed and the optimizer RNG stream
  are separated, so every method sees literally the same problem, and runs are
  exactly reproducible.
- **Every design choice is an ablation toggle.** `use_constriction`,
  `topology` (ring/gbest), `use_clamp`, `use_stagnation`, `use_turbulence`, and
  `seeding` (value-kmeans / plain-kmeans / uniform) are constructor flags, so
  each enhancement's contribution can be isolated in one line.
- **HFL integration.**
  [`pso_place_uavs`](../src/hflsim/placement.py#L25) bridges the optimizer
  into the real federated pipeline: it projects client lat/lon to metric
  coordinates, builds a `ProblemInstance` (optionally with FL-derived
  per-client values and the actual previous UAV layout), runs the optimizer,
  and converts the result back into `UAVAggregator` objects — making
  placement a live component of the hierarchical FL loop rather than an
  offline preprocessing step.

---

## 7. Default parameters at a glance

| Parameter | Value | Role |
|---|---|---|
| P | 100 | swarm size |
| G_max | 200 | max generations |
| c1 = c2 | 2.05 | cognitive/social acceleration (phi = 4.1) |
| chi | ≈ 0.7298 | constriction factor, derived from phi |
| ring_k | 2 | ring neighborhood = 5 particles |
| vmax_frac | 0.2 | velocity clamp fraction of range |
| delta_stag / G_stag / rho | 1e-4 / 20 / 0.2 | stagnation threshold / patience / reinit fraction |
| p_turb | 0.1 | turbulence probability (kick <= 0.1*vmax) |
| jitter | 10 m | seed-particle diversification |
| early stop | 0.95*w1 | near-optimal termination |
| w1, w2, w3 | 0.6 / 0.3 / 0.1 | coverage / movement / balance weights |

**In one sentence:** the proposal is a constriction-factor lbest-ring PSO over
the flat 3K UAV-position vector, warm-started with value-weighted k-means++
seeds, safeguarded against stagnation by turbulence and partial
reinitialization, and driven by a value-weighted coverage objective whose hard
constraints are absorbed into a deterministic greedy assignment — making it a
drop-in, reproducible placement engine for the hierarchical federated learning
simulation.
