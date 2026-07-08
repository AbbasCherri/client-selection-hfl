# The Client Selection Algorithm — Full Implementation Reference

This document explains, in complete detail, how IoT client selection works in
this repository: the per-device state model, the eligibility gate, the
priority score, the UCB exploration bonus, the greedy UAV assignment, the
reputation system that feeds into it, and exactly how all of it is wired into
the federated learning round loop. Nothing in this document is inferred —
every formula and constant is read directly from the code, with file/line
references throughout. This corresponds to **Algorithms 1–4 from the paper
(§IV-C)**.

Relevant files:
- [`src/uavbench/fl/client_selection.py`](../src/uavbench/fl/client_selection.py) — selector itself
- [`src/uavbench/fl/device_state.py`](../src/uavbench/fl/device_state.py) — per-device simulated state
- [`src/uavbench/fl/reputation.py`](../src/uavbench/fl/reputation.py) — reputation manager (Algorithm 3)
- [`src/uavbench/fl/federated.py`](../src/uavbench/fl/federated.py) — round loop that drives everything
- [`src/hflsim/shared/coords.py`](../src/hflsim/shared/coords.py) — Haversine distance
- [`src/hflsim/shared/value.py`](../src/hflsim/shared/value.py) — shared β(t) schedule

---

## 1. Overview: the four-stage pipeline

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
([client_selection.py:1-19](../src/uavbench/fl/client_selection.py#L1-L19))

Three selection **modes** share this file:

| Mode | Behavior | Used by |
|---|---|---|
| `"ucb"` | Full 4-stage pipeline described above | `proposed_hfl`, `hfl_static`, `hfl_no_reputation` |
| `"random"` | Eligibility filter only, then uniform random draw per UAV up to capacity | `hfl_no_selection` |
| `"all"` | Skip every filter — every currently covered client participates | `flat_fl`, `centralized` |

([client_selection.py:14-18](../src/uavbench/fl/client_selection.py#L14-L18), [federated.py:533-540](../src/uavbench/fl/federated.py#L533-L540))

---

## 2. Per-device state model (paper §IV-B)

Each IoT device carries a simulated per-round state
([device_state.py](../src/uavbench/fl/device_state.py)):

```python
class DeviceState:
    battery: float          # [0, 1]
    snr_db: float           # signal-to-noise ratio, dB
    memory_ok: bool         # RAM sufficiency (fixed per device)
    compute_time_s: float   # estimated local-training time (straggler model)
    margin_s: float         # adaptive safety margin ε_n(t)
```

### 2.1 Initialization (`DeviceStateManager.__init__`)

Drawn once per client at construction, all via the run's RNG:

| Field | Distribution | Notes |
|---|---|---|
| `battery` | `Uniform(0.5, 1.0)` | initial charge |
| `snr_base` | `Uniform(5, 20)` dB | fixed device-specific channel quality |
| `memory_ok` | `Bernoulli(p=0.90)` i.e. `rng.random() > 0.10` | 10% of devices are permanently memory-constrained |
| `compute_base` | `Uniform(50, 250)` s | hardware heterogeneity |

([device_state.py:57-76](../src/uavbench/fl/device_state.py#L57-L76))

### 2.2 Per-round update (`update_round(selected_ids)`)

Called once at the **end** of every FL round with the set of clients that
were actually selected that round:

- **Battery**: selected clients discharge by a flat **−0.02**; unselected
  clients passively recharge by **+0.005** (clipped to [0, 1]).
- **SNR noise**: fresh `Normal(0, 2)` dB added to `snr_base` every round for
  every device (channel fluctuation), regardless of selection.
- **Compute noise**: fresh `Normal(0, 30)` s added to `compute_base` every
  round for every device (straggler variance).
- If selected, the *observed* compute time (`max(10, compute_base +
  compute_noise)`) is appended to a rolling 10-round history used for the
  adaptive margin below.

([device_state.py:84-101](../src/uavbench/fl/device_state.py#L84-L101))

### 2.3 Adaptive eligibility margin ε_n(t) (paper §IV-C1)

`get_state()` recomputes the *live* `DeviceState` on demand:

```
margin_s = 1.96 * std(last 10 observed compute times)   if ≥ 3 observations
         = 0                                              otherwise (cold start)
```

This is a 95%-confidence-style buffer: a client with historically volatile
completion times gets a **stricter** effective deadline
(`T_max − margin_s`), preventing it from being deemed eligible on the basis of
a lucky single measurement. ([device_state.py:103-112](../src/uavbench/fl/device_state.py#L103-L112))

### 2.4 Eligibility predicate

```python
def eligible(self) -> bool:
    return (
        self.battery >= B_MIN            # 0.20
        and self.snr_db >= SNR_MIN_DB    # 3.0 dB
        and self.memory_ok
        and self.compute_time_s <= T_MAX_S - self.margin_s   # T_MAX_S = 300 s
    )
```
([device_state.py:16-19](../src/uavbench/fl/device_state.py#L16-L19), [device_state.py:41-47](../src/uavbench/fl/device_state.py#L41-L47))

All four gates are **hard** — failing any one makes the device ineligible
regardless of how good its other metrics are. Constants match paper Table II.

---

## 3. Stage 1 — Eligibility gate (in `ClientSelector.select`)

Before any scoring, the selector restricts to clients that are both
**covered** (within `R_comm` of some UAV, computed upstream by the placement
step) **and** eligible:

```python
eligible = {cid: uav_idx for cid, uav_idx in covered.items()
            if device_states.get(cid) is not None and device_states[cid].eligible()}
```
([client_selection.py:152-160](../src/uavbench/fl/client_selection.py#L152-L160))

If nothing survives, selection returns `{}` for the round. In `"all"` mode
this stage (and everything after) is skipped entirely — `covered` is returned
verbatim. In `"random"` mode, only this gate applies before a uniform draw.

---

## 4. Stage 2 — Utility Û_n (paper §IV-C3)

Computed by `_compute_utility()`, a **cross-sectional** score recomputed
fresh each selection round over the currently-eligible set:

```
Û_n = 0.4·U_epi + 0.3·U_SNR + 0.2·U_dens + 0.1·U_prox
```
(`W_EPI=0.4, W_SNR=0.3, W_DENS=0.2, W_PROX=0.1`, [client_selection.py:37-41](../src/uavbench/fl/client_selection.py#L37-L41))

- **U_epi** — epicenter proximity, capped at the **95th percentile** distance
  across the eligible set (not a fixed radius):
  ```
  d95 = percentile(haversine(client, epicentre), 95)
  U_epi = (d95 − min(d_n, d95)) / d95        [= 1 if d95 ≈ 0]
  ```
  Closer to the epicenter → higher score; the cap prevents one extreme outlier
  device from squashing everyone else's score toward zero.
  ([client_selection.py:85-93](../src/uavbench/fl/client_selection.py#L85-L93))

- **U_SNR** — plain min-max normalization of `snr_db` (clipped to [0, 30] dB
  first) across the eligible set. ([client_selection.py:96-97](../src/uavbench/fl/client_selection.py#L96-L97))

- **U_dens** — building-density proxy. Since no building inventory exists for
  the study area, it counts **other eligible clients within a 5 km radius**
  (Euclidean, in a local equirectangular metre frame via `_xy_metres`),
  excluding self, then min-max normalizes the count. Computed as a vectorized
  O(n²) pairwise distance check. ([client_selection.py:99-109](../src/uavbench/fl/client_selection.py#L99-L109))

- **U_prox** — proximity to the **nearest** UAV:
  ```
  U_prox = clip(1 − min_j(haversine(client, uav_j)) / R_comm, 0, 1)
  ```
  If no UAV coordinates are supplied (legacy/flat callers), defaults to a
  neutral `0.5` for every client. ([client_selection.py:111-118](../src/uavbench/fl/client_selection.py#L111-L118))

`_minmax` itself has a degenerate-input guard: if all values in the batch are
within `1e-10` of each other, every client gets exactly `0.5` rather than
dividing by a near-zero range. ([client_selection.py:53-57](../src/uavbench/fl/client_selection.py#L53-L57))

---

## 5. Stage 2 (continued) — Priority score P_n (paper §IV-C2)

```python
l_feat  = clip(1 − (compute_time_s / T_MAX_S)^2, 0, 1)
beta    = beta_schedule(round_num)                     # shared with placement value model
u_tilde = beta * utility + (1 − beta) * reputation
priority = W_BATTERY*battery + W_LEARNING*l_feat + W_UTILITY*u_tilde
```
Weights: `W_BATTERY = 0.35, W_LEARNING = 0.30, W_UTILITY = 0.35`
([client_selection.py:33-35](../src/uavbench/fl/client_selection.py#L33-L35), [client_selection.py:173-183](../src/uavbench/fl/client_selection.py#L173-L183))

- **b̃ (battery term)** is the raw battery fraction — devices with more
  charge are preferred directly.
- **ℓ̃ (learning-speed term)** rewards *faster* clients: squaring
  `compute_time_s / T_MAX_S` makes the penalty grow superlinearly as a
  client's estimated time approaches the deadline, then it's clipped to
  `[0, 1]` so times beyond `T_MAX_S` don't go negative.
- **Ũ (blended utility/reputation)** reuses the *exact same* β(t) decay
  schedule as the placement value model
  (`beta_schedule(t) = max(0, 1 − t/T_decay)`, `T_decay = 20`,
  [value.py:26-28](../src/hflsim/shared/value.py#L26-L28)): early rounds trust the
  geometric/channel utility (no track record yet), later rounds shift weight
  onto accumulated reputation. Reputation defaults to `0.5` for any client the
  reputation manager hasn't scored yet (`reputation_scores.get(cid, 0.5)`,
  [client_selection.py:175](../src/uavbench/fl/client_selection.py#L175)).

---

## 6. Stage 3 — UCB exploration bonus

```python
t = max(round_num, 1)
ucb_n = priority_n + C * sqrt(ln(t) / (N_n(t) + 1))
```
with **C = √2** (`UCB_C = math.sqrt(2)`, [client_selection.py:43](../src/uavbench/fl/client_selection.py#L43)) and `N_n(t)` the number of times client
n has been selected so far (`self._counts[cid]`, tracked statefully inside the
`ClientSelector` instance across the whole run).

This is the classic UCB1 exploration term: clients selected rarely (`N_n`
small) get a larger bonus, ensuring the algorithm doesn't permanently starve
low-priority-but-plausible clients in favor of always picking the same
high-priority ones — important for reputation and eligibility statistics to
stay fresh across the whole population. `t` is floored at 1 so `ln(t)` never
sees `ln(0)`. ([client_selection.py:185-187](../src/uavbench/fl/client_selection.py#L185-L187))

---

## 7. Stage 4 — Greedy UAV assignment (paper Algorithm 4)

`_greedy_assign()` processes clients in **descending UCB score order**
(`np.argsort(-scores)`), and for each one finds the best UAV it can currently
join:

```python
feasible = [j for j in range(K)
            if dist[client, j] <= R_comm and fill[j] < uav_capacity]
if not feasible:
    continue   # client is skipped this round — no feasible UAV
j_star = min(feasible, key=lambda j: (fill[j], dist[client, j]))
```

- Feasibility requires both **in-range** (Haversine distance ≤ `R_comm`) and
  **under capacity** at the moment of assignment (capacity fills up as
  higher-scored clients are placed first).
- Among feasible UAVs, the client goes to the one with the **lowest current
  load**, ties broken by **smallest distance** — this is the same
  load-balancing tie-break rule used by the PSO placement's own greedy
  assignment (`assignment.py`), so both algorithms share a philosophy: spread
  load evenly, prefer proximity only as a tiebreaker.
- A client with no feasible UAV (e.g. every in-range UAV is already full) is
  simply **not selected this round** — no error, no penalty, no repair.
- The full `(N, K)` client–UAV Haversine distance matrix is precomputed once
  per call, not per client. ([client_selection.py:198-247](../src/uavbench/fl/client_selection.py#L198-L247))

There is also a **fallback path** for legacy callers that don't supply UAV
coordinates: the client is assigned to its pre-computed covering UAV
(`eligible[cid]`, i.e. whatever `covered` said upstream) as long as that UAV
isn't yet at capacity. ([client_selection.py:227-236](../src/uavbench/fl/client_selection.py#L227-L236))

Every successful assignment increments that client's selection counter
`self._counts[cid]`, feeding back into next round's UCB bonus.

---

## 8. `"random"` mode — the ablation baseline

Used by `hfl_no_selection` to isolate the effect of the UCB pipeline. After
the same eligibility gate, clients are bucketed by their pre-assigned
covering UAV, and for each UAV a uniform-without-replacement sample of size
`min(capacity, len(bucket))` is drawn:

```python
n = min(uav_capacity, len(cids))
selected = rng.choice(cids, size=n, replace=False)   # per UAV
```

Uses the caller-supplied `rng` when given (required for correct multi-seed
sweeps — reusing the shared run RNG keeps the whole run reproducible from one
seed); falls back to a round-derived seed (`round_num * 7919`) only for
legacy callers that don't pass one. ([client_selection.py:250-271](../src/uavbench/fl/client_selection.py#L250-L271))
Note: this mode does **not** increment `self._counts`, since there's no UCB
term depending on it.

---

## 9. `"all"` mode — no selection at all

Simply returns `dict(covered)` unchanged — every currently-covered client
participates, no eligibility filtering, no scoring, no capacity limit.
Used by `flat_fl` (no UAV hierarchy — clients go straight to the server) and
`centralized` (oracle: all data pooled at one node, selection is meaningless).
([client_selection.py:149-150](../src/uavbench/fl/client_selection.py#L149-L150))

---

## 10. The reputation system feeding Ũ_n (paper Algorithm 3, §IV-C4)

`ReputationManager` ([reputation.py](../src/uavbench/fl/reputation.py)) maintains
`R_n ∈ [0, 1]` per client, consumed by client selection's `Ũ_n` blend and by
server-side aggregation gating (§IV-D, see §12 below).

```
R_n = w_contrib · R_contrib + w_anomaly · R_anomaly + w_temp · R_temp
```
Initial weights `(0.4, 0.3, 0.3)`, adaptive thereafter (see §10.4).

### 10.1 R_contrib — contribution quality (weight 0.4 nominal)

Tracks a **per-client EMA of the update delta vector** (not absolute
weights — cosine similarity between absolute weight vectors is ≈1 for every
client because the shared global initialization dominates, per the module's
own docstring):

```
Δw̄_n(t) = 0.7 · Δw_n(t) + 0.3 · Δw̄_n(t−1)          (_VEC_EMA_NEW=0.7, _VEC_EMA_OLD=0.3)
R_contrib = (1 + cos(Δw̄_n(t), Δw̄_n(t−1))) / 2
```
Cold start (no prior EMA for this client): `R_contrib = 0.5`. If either EMA
vector has near-zero norm (`denom ≤ 1e-12`), also falls back to `0.5`.
([reputation.py:145-162](../src/uavbench/fl/reputation.py#L145-L162))

### 10.2 R_anomaly — diagonal Mahalanobis anomaly (weight 0.3 nominal)

A **global** per-parameter mean/variance is maintained via EMA
(`_STATS_ALPHA = 0.1`) across *every* client's update vectors seen so far:

```
param_mean ← param_mean + 0.1·(v − param_mean)
param_var  ← 0.9·(param_var + 0.1·(v − param_mean)²)
```
Then for each client's update vector `v` (dimension J):
```
z = (v − param_mean) / sqrt(param_var + eps)
d = sqrt(mean(z²))                    # Mahalanobis distance / √J — dimension-independent
R_anomaly = 1                          if d ≤ 2
          = exp(−0.5·(d − 2))          otherwise
```
The `/√J` normalization is deliberate: it makes the paper's fixed threshold
of 2 meaningful regardless of how many parameters the update vector has.
([reputation.py:134-168](../src/uavbench/fl/reputation.py#L134-L168))

### 10.3 R_temp — temporal reliability (weight 0.3 nominal)

```
success_rate = total_successes / total_attempts
σ_RT = variance of last 10 response times (0 if < 2 samples)
R_temp = 0.5 · success_rate + 0.5 / (1 + σ_RT)
```
`update_batch()` increments both `total` and `success` for every client that
delivers an update this round (success_rate only drops via `mark_absent()`,
called from the FL loop for clients that were *selected* but failed to return
an update — e.g. an empty training shard). Response times are capped to the
last 10 rounds. ([reputation.py:170-176](../src/uavbench/fl/reputation.py#L170-L176), [reputation.py:216-220](../src/uavbench/fl/reputation.py#L216-L220), [federated.py:912-916](../src/uavbench/fl/federated.py#L912-L916))

### 10.4 Bayesian weight adaptation (every 10 rounds)

The three component weights are **not fixed** — every round, each
component's score for every delivering client is accumulated as evidence:

```
evidence += (R_contrib, R_anomaly, R_temp)     # per delivering client, every round
```
On `mark_absent`, the *complement* is added instead — an absent client is
evidence *against* whichever components scored it highly (a component that
gave a high score to a client who then disappeared was wrong):
```
evidence += (1 − R_contrib, 1 − R_anomaly, 1 − R_temp)
```
Every `_ADAPT_EVERY = 10` rounds, weights are recomputed as a **Dirichlet
posterior**, seeded with a prior concentration of `_PRIOR_STRENGTH = 20.0`
scaled by the initial `(0.4, 0.3, 0.3)` weights:
```
posterior = prior + evidence
weights = posterior / sum(posterior)
```
([reputation.py:181-189](../src/uavbench/fl/reputation.py#L181-L189), [reputation.py:191-198](../src/uavbench/fl/reputation.py#L191-L198))

This means the relative importance of contribution-quality vs. anomaly vs.
temporal-reliability drifts over the course of a run based on which signal
actually correlated with clients following through.

### 10.5 `trimmed_mean` — used for UAV-level reputation, not client-level

Not part of per-client `R_n`, but used one layer up: a UAV's own reputation
is the **10%-trimmed mean** of its assigned clients' reputations (floor of
`n*0.1` values dropped from each tail; no trimming if that floor is 0, i.e.
small clusters use the plain mean). ([reputation.py:49-62](../src/uavbench/fl/reputation.py#L49-L62)) This feeds the
server-aggregation gate in §12.

---

## 11. Wiring into the FL round loop (`run_full_hfl`, `federated.py`)

### 11.1 Selection cadence — not every round

Selection is **not** recomputed every round. A `ClientSelector` instance
persists across the whole run (so UCB counts accumulate correctly), and
`reselect` is only triggered when:

```python
reselect = (rnd - 1) % T_sel == 0 or low_eligible or not selected
```
`T_sel` defaults to 5 rounds (`fl.T_sel`). Between reselections, the previous
round's roster (`selected`) is reused as-is. ([federated.py:848-850](../src/uavbench/fl/federated.py#L848-L850))

### 11.2 Early-reselection trigger (paper §IV-E6)

```python
n_eligible = count of currently-eligible devices
low_eligible = n_eligible < lambda_min * min(K * capacity, len(clients))
```
`lambda_min` defaults to `0.5`. The `min(K*capacity, len(clients))` cap
handles the case where total UAV capacity exceeds the client population (the
paper assumes N ≫ ΣC_u, which doesn't hold for every config) — without the
cap the threshold would be meaningless in that regime.
([federated.py:788-793](../src/uavbench/fl/federated.py#L788-L793)) This same
`low_eligible` flag also forces an early **placement** re-run, since a sudden
drop in eligible devices likely means the current UAV layout is stale.

### 11.3 Per-round sequence for the federated path

For every round, in order:

1. Get live `device_states` (via `DeviceStateManager.get_all_states()`) and
   `rep_scores` (via `ReputationManager.get_all_scores()`) — both computed
   fresh each round.
2. Check `low_eligible`.
3. **Placement** (if due): compute per-device value `V_i(t)` from live
   SNR/reputation, run the PSO/GA optimizer, recompute `covered_all`.
4. **Selection** (if `reselect`): call `selector.select(...)` with the fresh
   `covered_all`, `device_states`, `rep_scores`, current UAV lat/lon, and the
   configured `selection_mode`.
5. Build per-UAV client groups from the `selected` mapping.
6. UAV-level image training + IoT-level structured-data local training on
   only the `selected` clients.
7. Any `selected` client that produced no update (e.g. empty shard) is marked
   `mark_absent` in the reputation manager.
8. `ReputationManager.update_batch()` on the delta vectors of clients that
   *did* deliver.
9. UAV-level and server-level aggregation (§12).
10. `device_mgr.update_round(set(selected.keys()))` — battery/SNR/compute
    noise advance based on who was actually selected this round.
11. Evaluate global model; log/record row.

([federated.py:781-1028](../src/uavbench/fl/federated.py#L781-L1028))

### 11.4 Method configuration table

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
([federated.py:533-540](../src/uavbench/fl/federated.py#L533-L540))

This makes every ablation ("what if we drop UCB?", "what if we drop
reputation weighting?", "what if placement is static?") a one-line
configuration change reusing the exact same selection code path.

---

## 12. How selection output feeds aggregation (context, §IV-D)

Not part of client *selection* itself, but the immediate consumer of its
output, so included for completeness:

- **UAV-level**: plain sample-count-weighted FedAvg over the `struct+fusion`
  updates of clients selected into that UAV's group (`fedavg`), overlaid with
  the UAV's own `img_proj` update from its aerial-image training pass.
- **UAV reputation**: `trimmed_mean` of the reputations of the clients
  actually selected into that UAV's cluster (§10.5).
- **Server-level gate (§IV-D)**: when `reputation_weighted` is true (all
  methods except `hfl_no_reputation`), UAVs whose trimmed-mean cluster
  reputation falls **below `R_min` (default 0.3)** are excluded entirely from
  the server aggregation round:
  ```python
  active = [u for u in uav_updates if u.reputation >= R_min]
  server_agg = reputation_fedavg(active) if active else <no update>
  ```
  ([federated.py:979-987](../src/uavbench/fl/federated.py#L979-L987)) This is
  the mechanism by which a UAV whose selected clients are consistently
  low-reputation gets silently dropped from a round, rather than poisoning
  the global model.

---

## 13. Constants at a glance

| Constant | Value | Meaning |
|---|---|---|
| `B_MIN` | 0.20 | minimum battery fraction to be eligible |
| `SNR_MIN_DB` | 3.0 dB | minimum SNR to be eligible |
| `T_MAX_S` | 300 s | nominal max local-training time |
| `W_BATTERY` | 0.35 | priority weight on battery |
| `W_LEARNING` | 0.30 | priority weight on speed term ℓ̃ |
| `W_UTILITY` | 0.35 | priority weight on blended utility/reputation Ũ |
| `W_EPI, W_SNR, W_DENS, W_PROX` | 0.4 / 0.3 / 0.2 / 0.1 | utility sub-weights |
| `UCB_C` | √2 | UCB1 exploration constant |
| `T_decay` (β schedule) | 20 rounds | utility→reputation blend decay, shared with placement |
| Battery discharge/recharge | −0.02 / +0.005 per round | selected vs. unselected |
| SNR / compute noise | N(0,2) dB / N(0,30) s | per-round channel & straggler fluctuation |
| Adaptive margin ε_n(t) | 1.96·σ(last 10 compute times) | dynamic eligibility buffer |
| `T_sel` | 5 rounds (config `fl.T_sel`) | selection (and placement) cadence |
| `lambda_min` | 0.5 (config `fl.lambda_min`) | early-reselection eligibility-ratio trigger |
| `R_min` | 0.3 (config `fl.R_min`) | min UAV cluster reputation to be aggregated |
| Reputation weights (init) | 0.4 / 0.3 / 0.3 | contrib / anomaly / temporal, Dirichlet-adapted every 10 rounds |
| `_VEC_EMA_NEW / OLD` | 0.7 / 0.3 | update-delta EMA for R_contrib |
| `_STATS_ALPHA` | 0.1 | global per-parameter mean/var EMA rate |
| `_PRIOR_STRENGTH` | 20.0 | Dirichlet prior concentration |
| Mahalanobis threshold | d ≤ 2 → R_anomaly = 1 | dimension-independent (divided by √J) |
| `trimmed_mean` trim | 10% each tail | UAV cluster reputation aggregation |

---

**In one sentence:** client selection is a stateful, per-round pipeline that
filters IoT devices through hard battery/SNR/memory/deadline gates, scores
survivors on a battery/speed/utility-or-reputation priority blended by a
time-decaying schedule shared with the placement objective, adds a UCB1
exploration bonus keyed on each client's historical selection count, and
greedily assigns the highest-scoring clients to their best-fit UAV under
range and capacity constraints — with reputation itself computed by a
separate Bayesian-adapted three-component (contribution / anomaly / temporal)
manager that also gates which UAV clusters are trusted enough to reach the
server aggregation step.
