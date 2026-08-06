"""PSO+ — the deployment-aware PSO variant, as a SEPARATE algorithm.

``pso`` is left untouched so every published number remains reproducible and
every addition here is ablatable against it. The governing invariant, asserted
by ``tests/sanity_checks/check_pso_plus.py``:

    PSOPlus with all features off  ==  PSO, bit for bit.

That is what makes each flag's effect attributable to the flag rather than to
an incidental refactor, so every feature is off by default and each is a single
independent switch.

Why this exists (measured, results/paper_coverage at 2026-08-06)
---------------------------------------------------------------
At the deployed R_comm = 20 km every method covers 100% of clients, so
placement is unfalsifiable there. At R_comm = 2 km, where coverage actually
binds, plain PSO scores **0.4005** on placement fitness against mozaffari2016's
**0.5365** — and against ``hfl_static``'s **0.4510**. It loses on the objective
it optimises, to *not optimising at all*. Two causes, addressed by A and B:

A. ``warm_start_frac`` — PSO's swarm is 50% value-weighted k-means++ seeds and
   50% uniform, and never contains the fleet's CURRENT layout. So the optimiser
   has no floor: it can, and at tight radius does, return a worse layout than
   the one already flown. Seeding particle 0 at ``prev_positions`` makes
   ``gbest >= f(prev_positions)`` a guarantee of the algorithm, not a hope.

B. ``move_norm="reachable"`` — the movement penalty is normalised by
   ``d_max = K * box_diagonal``, i.e. against every UAV crossing the whole
   arena. Measured in Tier-1: d_move ~26.4 km against d_max ~70.7 km, so at the
   tuned ``w2=0.03`` the penalty contributes ~0.011 against a coverage term of
   ~0.60 — under 2%. Normalising instead by what a UAV can physically fly
   between repositionings makes ``w2`` mean something. (The paper specifies
   w2=0.3; the weight search cut it to 0.03 because movement energy does not
   help the objective it was maximising.)

C. ``move_gate`` — placement fires every T_sel rounds unconditionally; nothing
   asks whether moving is worth it. ``hfl_static`` spends zero movement energy
   and is statistically indistinguishable on macro-F1 at every N, so the
   ability to decline the move is a real strategy, not a degenerate one.
"""

from __future__ import annotations

import numpy as np

from ..problem.fitness import Fitness
from ..problem.instance import ProblemInstance
from .base import Result
from .pso import PSO

_EPS = 1e-9


class PSOPlus(PSO):
    """PSO with deployment-aware warm starting, movement scaling, and a move gate."""

    name = "pso_plus"

    def __init__(
        self,
        # --- A: warm start from the current fleet layout ---
        warm_start_frac: float = 0.0,
        # --- B: movement normalisation ---
        move_norm: str = "box",  # "box" (legacy) | "reachable"
        v_max_mps: float = 20.0,  # UAV cruise speed
        t_interval_s: float = 300.0,  # seconds between repositionings
        # --- C: cost-benefit gate on executing the move ---
        move_gate: bool = False,
        gate_margin: float = 0.0,
        # --- staged, not yet implemented: fail loudly rather than silently ---
        analytic_seed: bool = False,
        class_balance_w: float = 0.0,
        **kw,
    ) -> None:
        super().__init__(**kw)
        if not 0.0 <= warm_start_frac <= 1.0:
            raise ValueError(f"warm_start_frac must be in [0,1]; got {warm_start_frac}")
        if move_norm not in ("box", "reachable"):
            raise ValueError(f"move_norm must be 'box' or 'reachable'; got {move_norm!r}")
        # A flag that silently does nothing is how fair_mab's staleness term
        # produced four bit-identical "ablations". Refuse instead.
        if analytic_seed:
            raise NotImplementedError("analytic_seed lands in stage 3; do not enable it yet")
        if class_balance_w:
            raise NotImplementedError("class_balance_w lands in stage 4; do not enable it yet")

        self.warm_start_frac = warm_start_frac
        self.move_norm = move_norm
        self.v_max_mps = v_max_mps
        self.t_interval_s = t_interval_s
        self.move_gate = move_gate
        self.gate_margin = gate_margin

    # -- feature A -------------------------------------------------------

    def _init_positions(
        self, instance: ProblemInstance, lo: np.ndarray, hi: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Base swarm, then overwrite the first particles with the current layout.

        The base call runs first and unmodified, so with ``warm_start_frac=0``
        this consumes the identical RNG stream and returns the identical array
        as :class:`PSO` — which is what the bit-identity check relies on.

        Particle 0 is the current layout EXACTLY (zero movement, zero jitter).
        Because ``pbest`` is initialised from the evaluated swarm, that single
        particle is what turns "PSO never returns worse than standing still"
        into a property of the algorithm.
        """
        X = super()._init_positions(instance, lo, hi, rng)
        if self.warm_start_frac <= 0.0:
            return X

        prev = np.asarray(instance.prev_positions, dtype=np.float64).reshape(-1)
        if prev.shape[0] != X.shape[1]:
            raise ValueError(
                f"prev_positions has {prev.shape[0]} genes but the swarm has "
                f"{X.shape[1]}; cannot warm start"
            )
        n_warm = min(self.P, max(1, int(round(self.warm_start_frac * self.P))))
        X[0] = np.clip(prev, lo, hi)
        for p in range(1, n_warm):
            X[p] = np.clip(prev + rng.normal(0.0, self.jitter_m, size=prev.shape), lo, hi)
        return X

    # -- feature B -------------------------------------------------------

    def _reachable_d_max(self, instance: ProblemInstance) -> float:
        """Furthest the fleet can actually fly between two repositionings."""
        return max(instance.K * self.v_max_mps * self.t_interval_s, _EPS)

    # -- main ------------------------------------------------------------

    def _run(self, instance: ProblemInstance, fitness: Fitness, rng: np.random.Generator) -> Result:
        # `Optimizer.optimize` reads eval_count back off the SAME fitness object
        # it passed in, so the normaliser is mutated in place and restored —
        # handing the swarm a copy would silently zero the reported eval count
        # and void the shared-budget assertion in build_optimizer.
        original_d_max = fitness.d_max
        if self.move_norm == "reachable":
            fitness.d_max = self._reachable_d_max(instance)
        try:
            result = super()._run(instance, fitness, rng)
        finally:
            fitness.d_max = original_d_max

        if not self.move_gate:
            return result

        # -- feature C: decline the move unless it pays for itself ----------
        prev = np.asarray(instance.prev_positions, dtype=np.float64).reshape(1, -1)
        f_prev = float(fitness.batch(prev)[0])
        gain = result.best_fitness - f_prev
        result.meta = {**result.meta, "gate_gain": gain, "gate_f_prev": f_prev}
        if gain <= self.gate_margin:
            result.meta["gate_declined"] = True
            result.best_position = prev.reshape(-1).copy()
            result.best_fitness = f_prev
            return result
        result.meta["gate_declined"] = False
        return result
