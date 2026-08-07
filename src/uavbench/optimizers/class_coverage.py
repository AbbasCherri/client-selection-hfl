"""Submodular class-coverage objective for placement.

The idea, and why it is not the objective every other placement method uses
-----------------------------------------------------------------------------
Every placement paper surveyed here — Mozaffari 2016, Alzenad 2017, Lyu 2017,
Sawalmeh 2021, Moon 2022, MOGOA 2026 — maximizes *how much demand is covered*:
users, area, or users weighted by a priority or QoS label. That is the right
objective when the UAVs are serving traffic.

These UAVs are not serving traffic. They are recruiting training data for a
federated model, and the metric that model is judged on is **macro-F1**, which
averages recall over classes and therefore weights a rare class exactly as much
as a common one. Total covered demand is the wrong proxy for that: a placement
covering 200 samples of one class scores the same as one covering 100 samples of
each of two classes, while macro-F1 strongly prefers the second.

So the objective here is over the *label composition* of what gets covered:

    F(S) = sum_c  w_c * min( n_c(S), tau_c )

with ``n_c(S)`` the number of class-``c`` samples on devices covered by the sites
in ``S``, ``w_c`` a per-class weight (scarcity), and ``tau_c`` a saturation
quota. The truncation is what makes it class-*diverse* rather than
class-weighted: once class ``c`` is represented up to ``tau_c``, further samples
of it earn nothing and the greedy is pushed toward classes still missing.

Why the form is exactly this, and not a nicer-looking one
---------------------------------------------------------
The guarantee below needs ``F`` to be monotone submodular, and that is fragile:

* ``n_c(S)`` is a weighted **coverage** function (modular measure over a union of
  discs), hence monotone submodular — but *not* modular.
* "Concave of modular is submodular" is the familiar rule, and it does **not**
  extend to concave-of-submodular. So the natural-looking ``sqrt(n_c(S))`` is
  *not* guaranteed submodular here, and using it would void the guarantee while
  looking perfectly reasonable.
* Truncation *does* preserve monotone submodularity: ``min(f, tau)`` is monotone
  submodular whenever ``f`` is. Non-negative weighted sums preserve it too.

Hence ``sum_c w_c min(n_c, tau_c)`` — provably monotone submodular, where the
concave-composition version is not.

The guarantee, and what makes it stronger than the usual one
-------------------------------------------------------------
Greedy on a monotone submodular function under a cardinality constraint
(``|S| = K`` UAVs) returns at least ``(1 - 1/e)`` of the optimum over the ground
set. Submodular coverage placement with that bound is established work; what is
usually left implicit is that the ground set is a **grid**, so the guarantee is
against the best grid placement, not the best placement.

Here the ground set is the circle-intersection candidate set, which by Church
(1984) provably *contains* an optimal placement in the continuous plane (see
:mod:`.candidates`). The bound therefore holds against the **continuous**
optimum:

    F(S_greedy) >= (1 - 1/e) * max over all placements in R^2 of F

``tau_c = rho * total_c`` makes the flag exactly inert at ``rho = 1``: no set can
cover more of class ``c`` than exists, so the truncation never binds and ``F``
collapses to plain weighted coverage — the objective the method already had.
Any behaviour change at ``rho < 1`` is therefore attributable to saturation and
nothing else.

Scope, stated honestly: the guarantee is for the uncapacitated class-coverage
objective. Per-UAV capacity is enforced downstream by the shared greedy
assignment and is not part of ``F``; at the radii where class coverage matters
it is measurably slack (max load 3 of 10 at R_comm = 250 m), so it does not bind
on the placements this objective produces.
"""

from __future__ import annotations

import numpy as np


class ClassCoverage:
    """Monotone submodular class-coverage objective over a candidate set.

    Parameters
    ----------
    class_hist:
        ``(N, C)`` per-device label counts.
    scarcity:
        ``(C,)`` per-class weights; ``None`` weights all classes equally, which
        is what macro-F1 itself does.
    quota_frac:
        ``rho`` in ``tau_c = rho * total_c``. ``1.0`` disables saturation and
        reduces ``F`` to weighted coverage exactly.
    """

    def __init__(
        self,
        class_hist: np.ndarray,
        scarcity: np.ndarray | None = None,
        quota_frac: float = 0.35,
    ) -> None:
        self.hist = np.asarray(class_hist, dtype=np.float64)
        if self.hist.ndim != 2:
            raise ValueError(f"class_hist must be (N, C); got shape {self.hist.shape}")
        n_cls = self.hist.shape[1]
        self.w = (
            np.ones(n_cls)
            if scarcity is None
            else np.asarray(scarcity, dtype=np.float64).reshape(n_cls)
        )
        if not 0.0 < quota_frac <= 1.0:
            raise ValueError(f"quota_frac must be in (0, 1]; got {quota_frac}")
        self.quota_frac = float(quota_frac)
        self.total = self.hist.sum(axis=0)  # (C,) how much of each class exists
        self.tau = self.quota_frac * self.total
        # Normaliser: the largest F any placement could reach.
        self.f_max = max(float(np.dot(self.w, self.tau)), 1e-12)

    def value(self, covered_mask: np.ndarray) -> float:
        """``F(S)`` for the devices covered by ``S``, normalized to [0, 1]."""
        n_c = covered_mask.astype(np.float64) @ self.hist
        return float(np.dot(self.w, np.minimum(n_c, self.tau)) / self.f_max)

    def marginal(
        self,
        cover: np.ndarray,
        covered_mask: np.ndarray,
    ) -> np.ndarray:
        """``(M,)`` normalized gain ``F(S + m) - F(S)`` for every candidate.

        Vectorized over candidates: only devices *not already covered* can
        contribute, which is exactly what makes the gains diminish as ``S``
        grows — the submodularity this objective is built on.
        """
        n_now = covered_mask.astype(np.float64) @ self.hist  # (C,)
        base = np.dot(self.w, np.minimum(n_now, self.tau))
        fresh = np.asarray(cover, dtype=bool) & ~covered_mask[None, :]
        delta = fresh.astype(np.float64) @ self.hist  # (M, C)
        gained = np.minimum(n_now[None, :] + delta, self.tau[None, :])
        return (gained @ self.w - base) / self.f_max
