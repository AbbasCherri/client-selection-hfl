# Literature Baseline Algorithms — Faithful Pseudocode

Written to slot directly after your existing Algorithms 1–4 (eligibility
gating, priority score, UCB, greedy assignment), using the same notation:
`N(t)` = candidate/eligible IoT devices this round, `n` = a device,
`t` = round index, `U` = UAV set, `Cu`/`Rcomm`/`B_min^UAV` = UAV capacity/
range/battery threshold, `Tmax` = round deadline. Each baseline includes
(a) a one-paragraph fidelity note on what's faithful vs. adapted for your
setting, since reviewers will check this, and (b) the exact citation to
use.

> **Implementation:** all three baselines are implemented in
> `src/uavbench/fl/client_selection.py` as selection modes `"fedcs"`,
> `"rep_cap"`, and `"fair_mab"`, and exposed as full-system methods
> `fedcs` / `rep_cap` / `fair_mab` in `src/uavbench/fl/federated.py`
> (`_METHOD_CFG`). They share the identical PSO placement, reputation
> FedAvg aggregation, and T_sel cadence as `proposed_hfl` so the
> selection rule is the only experimental variable.

---

## Baseline 1 — FedCS (Nishio & Yonetani, ICC 2019)

**Citation:** T. Nishio and R. Yonetani, "Client selection for federated
learning with heterogeneous resources in mobile edge," in *2019 IEEE
International Conference on Communications (ICC)*, 2019, pp. 1–7.

**Core idea (from the source):** FedCS sets a fixed per-round deadline
`Tmax`. Starting from an empty selected set, it **greedily adds the
candidate that increases the round's projected completion time the
least**, stopping when either no candidate can be added without blowing
the deadline, or a target participation count is reached. There is no
data-value or reputation signal anywhere in the original — selection is
purely a function of estimated download + local-update + upload time.

**Adaptation note for your setting (state explicitly in the paper):**
The original FedCS assumes a flat server↔client topology with a single
resource-request round. You adapt it to your hierarchical topology by
(i) running the identical greedy-marginal-time rule independently per
UAV over the devices already gated in by your eligibility filter (§IV-C1)
— so FedCS still benefits from the same eligibility floor everyone else
gets — and (ii) using each device's `T̂n(t)` (already computed for your
own priority score) as FedCS's per-client time estimate, so both methods
see the same underlying measurement, isolating the *selection rule*
as the only variable. This is the standard adaptation pattern used when
FedCS is ported to non-flat topologies in the literature (e.g., the
"Stochastic Client Selection for Federated Learning with Volatile
Clients" baseline appendix does the same kind of minimal adaptation and
documents it explicitly — you should too).

```
Algorithm B1: FedCS-style Greedy Deadline Selection (per UAV u)
Require: eligible candidates E_u(t) assigned/reachable by UAV u,
         predicted completion times {T̂n(t)}, round deadline Tmax,
         UAV capacity Cu
Ensure: selected set S_u(t) ⊆ E_u(t)

1: S_u(t) ← ∅
2: T_proj ← 0                      # projected round completion time so far
3: candidates ← E_u(t) sorted by ascending T̂n(t)   # cheapest first
4: for each n in candidates do
5:     T_inc ← max(T̂n(t) − T_proj, 0)   # marginal time increase from adding n
                                          # (FedCS Eq.: T_inc(S,k) = time to
                                          #  bring the round up to serve k too)
6:     if T_proj + T_inc ≤ Tmax and |S_u(t)| < Cu then
7:         S_u(t) ← S_u(t) ∪ {n}
8:         T_proj ← max(T_proj, T̂n(t))
9:     else
10:        break     # deadline would be exceeded; stop adding candidates
11:    end if
12: end for
13: return S_u(t)
```

**What to report alongside it:** since FedCS has no reputation or utility
notion, expect it to plateau at a resource-efficient-but-value-blind
selection — this is exactly the contrast you want in the discussion
("FedCS achieves comparable straggler-avoidance but lower classification
utility because it cannot distinguish a fast device with poor seismic
SNR from a fast device near the epicenter").

---

## Baseline 2 — Reputation-Based UAV-Vehicular Selection (Zhao, Geng, Feng & Zhou, *Chinese Journal of Aeronautics*, 2024)

**Citation:** H. Zhao, L. Geng, W. Feng, and C. Zhou, "Client selection
and resource scheduling in reliable federated learning for uav-assisted
vehicular networks," *Chinese Journal of Aeronautics*, vol. 37, no. 9,
pp. 328–346, Jun. 2024.

**Core idea:** a reputation-based mechanism integrating **data quality**
and **computation capability** metrics to select "reliable, high-
performance" nodes — no exploration term, no epicenter/geospatial utility
notion (their setting is vehicular, not seismic), and no UCB-style
starvation guard. Reputation there is built from delivered-update quality
signals analogous in spirit to your `R_contrib`/`R_temp`, but combined
multiplicatively with a device-capability score rather than blended via
a decaying β(t) schedule.

**Adaptation note:** you reuse your own `R_contrib`, `R_anomaly`, `R_temp`
sub-scores (Algorithm 3) as the "data quality/reliability" half, since
recomputing an entirely different reputation estimator would conflate
"different reputation model" with "different selection rule" — the thing
you actually want to isolate. The "computation capability" half is your
existing `˜ℓn` (compute/connectivity feature). The key structural
difference from your method is what this baseline **omits**: no utility
term (`Û_n`), no β(t) blending schedule, and no UCB exploration bonus —
so it will systematically starve devices that are seismically valuable
but have middling reputation/compute, which is precisely the failure
mode your UCB term is designed to prevent. That contrast is the point of
including it.

```
Algorithm B2: Reputation-Capability Selection (per round t)
Require: eligible set E(t), reputation Rn(t) (Algorithm 3, unchanged),
         compute/connectivity feature ˜ℓn(t) = 1 − (T̂n(t)/Tmax)^2,
         weight γ ∈ (0,1) (default 0.5, symmetric trust/capability split)
Ensure: ranked candidate list for greedy UAV assignment

1: for each n in E(t) do
2:     score_n ← γ · Rn(t) + (1 − γ) · ˜ℓn(t)
3: end for
4: Sort E(t) by descending score_n → list L
5: return L      # feed directly into your existing Algorithm 4
                  # (greedy UAV assignment), replacing UCB_n with score_n
```

**Note the missing exploration term explicitly in text**: a
newly-added or long-dormant device with `Rn(0) = 0.5` (neutral) and
mediocre `˜ℓn` will never surface here regardless of how valuable its
seismic data becomes later — that's the exact starvation scenario your
Introduction motivates the UCB term against, so this baseline's
convergence-plateau/fairness-collapse behavior (analogous to your current
Battery-only and Utility-only curves in Fig. 4) is the expected and
citable result.

---

## Baseline 3 — Fairness-Enhanced MAB Scheduling (Zhu, Shi, Zhao, Chen, Zhang & Bao, *Sensors*, 2024)

**Citation:** C. Zhu, Y. Shi, H. Zhao, K. Chen, T. Zhang, and C. Bao, "A
fairness-enhanced federated learning scheduling mechanism for UAV-assisted
emergency communication," *Sensors*, vol. 24, no. 5, p. 1599, 2024.

**Core idea:** a multi-armed-bandit scheduler whose reward function
weights **model freshness** (how long since a device last contributed,
i.e. staleness) and **energy consumption**, explicitly to enforce
selection fairness across terrestrial devices in an emergency-
communication setting — the same disaster/UAV context as your paper,
which is why this is a stronger head-to-head than a generic bandit
baseline would be.

**Adaptation note:** their reward has two terms — staleness and energy —
where "energy" in their vehicular/terrestrial-device setting maps
directly onto your `bn(t)` (battery), and "staleness" maps onto your own
selection-count bookkeeping `Nn(t)` (used differently: their reward
*directly rewards* being stale, your UCB *bonuses* rarely-selected
devices — structurally similar exploration pressure, different reward
shape, which is the right level of "different but comparable"). Unlike
your method, this baseline has **no seismic/geospatial utility term and
no reputation/trust component at all** — it optimizes purely for
fairness of participation and energy sustainability, not for the value
of the contributed data. This is a UCB-*family* baseline, distinct from
your own UCB term in what feeds the bandit's reward, so pair it with your
method in the discussion as "our UCB explores over *value*, theirs
explores over *fairness/energy*" rather than treating both as
interchangeable UCB variants.

```
Algorithm B3: Fairness/Energy MAB Selection (per round t)
Require: eligible set E(t), battery bn(t), rounds since last selection
         staleness_n(t) = t − last_selected_n, reward weights
         (w_energy, w_stale) with w_energy + w_stale = 1 (default 0.5/0.5)
Ensure: ranked candidate list for greedy UAV assignment

1: for each n in E(t) do
2:     energy_term ← bn(t)                         # normalized [0,1]
3:     stale_term  ← min(1, staleness_n(t) / T_stale_cap)  # normalized,
                                                            # capped at 1
4:     reward_n ← w_energy · energy_term + w_stale · stale_term
5: end for
6: Sort E(t) by descending reward_n → list L
7: return L      # feed directly into your existing Algorithm 4

# Bandit bookkeeping (updated after each round, mirrors your Nn(t) update):
8: for each n selected this round do
9:     last_selected_n ← t
10: end for
```

`T_stale_cap` is a normalization constant (e.g. `T_sel`, your existing
reselection interval, is a reasonable default so staleness saturates on
the same cadence your own system reconsiders devices) — state this choice
explicitly since it's a hyperparameter you're introducing to adapt their
reward to a bounded [0,1] scale for fair comparison with your own
[0,1]-normalized priority score.

---

## How to present all three in the paper

- **Table** (new, sits alongside your existing Table I): one row per
  baseline with columns *Selection signal(s) used*, *Exploration
  mechanism*, *Reputation/trust modeled?*, *Data-value/utility modeled?*
  — this makes the "why these three" argument visually obvious: FedCS
  (resource only), Zhao et al. (reputation+capability, no exploration,
  no utility), Zhu et al. (fairness/energy bandit, no reputation, no
  utility), Proposed (all four: resource, reputation, utility, UCB
  exploration).
- **Results**: keep this as a *separate* table/figure from your own
  ablations (no-reputation, no-UCB/random, no-selection/all) — per the
  earlier audit note, conflating "vs. literature" with "vs. our own
  ablations" is what made the current baseline set look thin. Two
  tables, two questions answered.
- **Related Work**: you already cite all three — add one sentence to
  each existing citation noting "we implement this as a baseline in
  §V" so the related-work section and the results section are visibly
  connected rather than the current pattern of citing-without-comparing.
