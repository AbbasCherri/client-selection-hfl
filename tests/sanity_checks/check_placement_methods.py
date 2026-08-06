"""Correctness gates for the candidate-set placement method and its competitors.

Three classes of claim are asserted here, in increasing order of how much of the
paper rests on them:

1. **Bookkeeping.** Every optimizer's reported ``best_fitness`` is the fitness of
   the position it returned, and no method outspends the shared evaluation
   budget. A method that reports a score it did not achieve, or that wins by
   evaluating more, invalidates every table it appears in.

2. **The candidate-set reduction.** :mod:`uavbench.optimizers.candidates` claims
   the continuous placement problem can be restricted to circle intersection
   points without loss (Church 1984). That is the entire justification for
   replacing the swarm, so it is checked against brute force on a fine grid
   rather than cited.

3. **The floor.** ``mclp_ls`` must never return a layout worse than standing
   still — the specific failure plain PSO showed at tight ``R_comm``.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _lib import check, finish  # noqa: E402

from uavbench.optimizers import REGISTRY, build_optimizer  # noqa: E402
from uavbench.optimizers.candidates import (  # noqa: E402
    build_candidate_set,
    capped_covered_value,
    circle_intersection_points,
    coverage_matrix,
    effective_radius,
)
from uavbench.problem.fitness import Fitness  # noqa: E402
from uavbench.problem.instance import generate_instance  # noqa: E402

BUDGET = {"P": 30, "G_max": 15}
W = dict(w1=0.811, w2=0.03, w3=0.159)

# Tight enough that coverage actually binds — at the deployed 20 km every method
# covers everything and none of this is falsifiable.
AREA = {"x": [0.0, 6000.0], "y": [0.0, 6000.0], "z": [20.0, 120.0]}


def _instance(seed=1, distribution="clustered", R_comm=600.0, N=80, K=6, capacity=12):
    return generate_instance(
        N=N, K=K, distribution=distribution, seed=seed, area=AREA,
        R_comm=R_comm, capacity=capacity,
    )


def _fit(inst):
    return Fitness(inst, **W)


# --- 1. bookkeeping ------------------------------------------------------


def reported_fitness_matches_returned_position():
    """A method may not report a score its returned layout does not achieve."""
    for name in REGISTRY:
        inst = _instance()
        fit = _fit(inst)
        res = build_optimizer(name, {}, BUDGET).optimize(inst, fit, np.random.default_rng(4))
        # Score on a fresh scorer at the method's own radii, so this checks
        # bookkeeping only — cross-method comparability is a separate concern
        # handled by re-scoring in _place_uavs (see equal_radius check below).
        recomputed = _fit(inst)(res.best_position, radii=res.meta.get("radii"))
        assert abs(recomputed - res.best_fitness) < 1e-9, (
            f"{name}: reported best_fitness {res.best_fitness!r} != fitness of the "
            f"returned position {recomputed!r}"
        )


def no_method_outspends_the_shared_budget():
    max_evals = BUDGET["P"] * (BUDGET["G_max"] + 1)
    for name in REGISTRY:
        inst = _instance()
        fit = _fit(inst)
        build_optimizer(name, {}, BUDGET).optimize(inst, fit, np.random.default_rng(4))
        assert fit.eval_count <= max_evals, (
            f"{name} spent {fit.eval_count} evaluations against a shared budget of "
            f"{max_evals} — any comparison against it is confounded by spend"
        )


def every_method_returns_k_positions_in_bounds():
    for name in REGISTRY:
        inst = _instance()
        res = build_optimizer(name, {}, BUDGET).optimize(inst, _fit(inst), np.random.default_rng(4))
        pos = res.best_position.reshape(-1, 3)
        assert pos.shape == (inst.K, 3), f"{name} returned {pos.shape}, expected ({inst.K}, 3)"
        assert np.isfinite(pos).all(), f"{name} returned non-finite coordinates"
        assert (pos >= inst.lower - 1e-6).all() and (pos <= inst.upper + 1e-6).all(), (
            f"{name} returned positions outside the search box"
        )


def every_method_deploys_in_three_dimensions():
    """Altitude must be a decision the method makes, not a constant it inherits.

    Several constructive methods originally pinned ``z`` (mid-band for centroid,
    ``z_min`` for the clustering baselines), which made them planar placements
    competing in a 3D benchmark — their altitude was the implementer's choice
    standing in for the method's. Every method except ``static`` (whose defining
    behaviour is to not move) now selects altitude, and this asserts the degree
    of freedom is genuinely exercised: changing the altitude band must change
    where the method flies.
    """
    tall = {"x": [0.0, 6000.0], "y": [0.0, 6000.0], "z": [20.0, 120.0]}
    lifted = {"x": [0.0, 6000.0], "y": [0.0, 6000.0], "z": [300.0, 400.0]}
    for name in REGISTRY:
        if name in ("static", "random"):
            continue  # static holds position by definition; random is uniform by definition
        zs = []
        for area in (tall, lifted):
            inst = generate_instance(
                N=80, K=6, distribution="clustered", seed=1, area=area,
                R_comm=600.0, capacity=12,
            )
            res = build_optimizer(name, {}, BUDGET).optimize(
                inst, _fit(inst), np.random.default_rng(4)
            )
            pos = res.best_position.reshape(inst.K, 3)
            assert (pos[:, 2] >= area["z"][0] - 1e-6).all() and (
                pos[:, 2] <= area["z"][1] + 1e-6
            ).all(), f"{name} flew outside the altitude band {area['z']}"
            zs.append(float(pos[:, 2].mean()))
        assert abs(zs[1] - zs[0]) > 1.0, (
            f"{name} flew at {zs[0]:.1f} m and {zs[1]:.1f} m under altitude bands "
            f"{tall['z']} and {lifted['z']} — its z is a hard-coded constant, not a "
            "placement decision, so it is a 2D method in a 3D benchmark"
        )


def altitude_optimization_never_makes_a_layout_worse():
    from uavbench.optimizers.altitude import optimize_altitudes

    for seed in (0, 1, 2):
        inst = _instance(seed=seed)
        fit = _fit(inst)
        start = np.column_stack([
            inst.device_coords[: inst.K, :2],
            np.full(inst.K, 0.5 * (inst.lower[2] + inst.upper[2])),
        ])
        f0 = float(_fit(inst)(start.reshape(-1)))
        out = optimize_altitudes(inst, fit, start)
        f1 = float(_fit(inst)(out.reshape(-1)))
        assert f1 >= f0 - 1e-12, (
            f"[seed {seed}] altitude descent lowered fitness {f0:.6f} -> {f1:.6f}; it is "
            "applied to every constructive method and must be safe to apply blind"
        )


def path_loss_baselines_are_flagged_as_unequal_radius():
    """The 618-vs-500 m artifact must stay visible, not be silently reintroduced.

    mozaffari2016/alzenad2017 legitimately derive their own coverage radius. That
    makes their self-reported fitness incomparable with a method scored at the
    system gate, which is why the FL path re-scores. This asserts the condition
    that makes re-scoring necessary is detectable from the result alone.
    """
    inst = _instance(R_comm=500.0)
    for name in ("mozaffari2016", "alzenad2017"):
        res = build_optimizer(name, {}, BUDGET).optimize(inst, _fit(inst), np.random.default_rng(4))
        radii = res.meta.get("radii")
        assert radii is not None, (
            f"{name} no longer publishes result.meta['radii'] — the equal-radius "
            "re-score in _place_uavs and the Tier-1 guard both key off it"
        )
        assert np.asarray(radii).shape == (inst.K,), f"{name} radii has wrong shape"


def equal_radius_rescore_changes_the_ranking_metric():
    """Guard the fix: re-scoring at R_comm must actually differ from self-scoring."""
    inst = _instance(R_comm=500.0)
    res = build_optimizer("mozaffari2016", {}, BUDGET).optimize(
        inst, _fit(inst), np.random.default_rng(4)
    )
    at_own = _fit(inst)(res.best_position, radii=res.meta["radii"])
    at_gate = _fit(inst)(res.best_position)
    assert abs(at_own - at_gate) > 1e-9, (
        "mozaffari2016 scores identically at its own radius and at R_comm on this "
        "instance — the equal-radius test has gone inert and would not catch the "
        "handicap it exists to catch"
    )


# --- 2. the candidate-set reduction --------------------------------------


def intersection_points_are_wedged_against_both_devices():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0.0, 1000.0, size=(30, 2))
    r = 250.0
    cips = circle_intersection_points(xy, r)
    assert cips.shape[0] > 0, "no intersection points generated on a dense layout"
    d = np.sqrt(((cips[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
    # Every intersection point must cover at least its two generating devices,
    # under the same `<=` test the assignment uses.
    assert (np.sum(d <= r, axis=1) >= 2).all(), (
        "an intersection point covers fewer than the two devices that generated "
        "it — the boundary nudge is not making the `<=` test robust"
    )


def candidate_set_matches_brute_force_best_disc():
    """The load-bearing claim: restricting to candidates loses no coverage.

    Brute-forces the best single-disc position on a fine grid and requires the
    best candidate to match or beat it. If the reduction were unsound this is
    where it shows.
    """
    rng = np.random.default_rng(7)
    N = 25
    xy = rng.uniform(200.0, 1800.0, size=(N, 2))
    value = rng.uniform(0.2, 1.0, size=N)
    r = 300.0
    lo, hi = np.array([0.0, 0.0]), np.array([2000.0, 2000.0])

    cands = build_candidate_set(xy, r, lo, hi, max_candidates=100_000, dedupe_grid_m=0.0)
    cand_best = float((coverage_matrix(cands, xy, r) @ value).max())

    g = np.linspace(0.0, 2000.0, 401)  # 5 m grid
    gx, gy = np.meshgrid(g, g)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    grid_best = float((coverage_matrix(grid, xy, r) @ value).max())

    assert cand_best >= grid_best - 1e-9, (
        f"best candidate covers {cand_best:.6f} of value but a brute-force 5 m grid "
        f"found {grid_best:.6f} — the circle-intersection reduction is unsound, and "
        "with it the justification for mclp_ls"
    )


def capped_covered_value_matches_the_naive_computation():
    rng = np.random.default_rng(3)
    M, N, cap = 40, 30, 7
    cover = rng.random((M, N)) < 0.3
    value = np.sort(rng.uniform(0.1, 1.0, size=N))[::-1].copy()  # descending, as required
    fast = capped_covered_value(cover, value, cap)
    naive = np.array([value[np.flatnonzero(cover[m])[:cap]].sum() for m in range(M)])
    assert np.allclose(fast, naive), "capped_covered_value disagrees with the naive top-cap sum"


def effective_radius_shrinks_with_altitude():
    assert effective_radius(500.0, 0.0) == 500.0
    assert effective_radius(500.0, 120.0) < effective_radius(500.0, 20.0)
    assert effective_radius(100.0, 200.0) == 0.0, "an unreachable altitude must give radius 0"


def candidate_cap_is_respected_and_reported():
    inst = _instance(N=120, R_comm=1500.0)
    cands = build_candidate_set(
        inst.device_coords[:, :2], 1500.0, inst.lower[:2], inst.upper[:2],
        max_candidates=300, rng=np.random.default_rng(0),
    )
    assert cands.shape[0] <= 300, f"candidate cap ignored: got {cands.shape[0]}"


# --- 3. mclp_ls behaviour -------------------------------------------------


def mclp_ls_never_returns_worse_than_standing_still():
    for seed in (1, 2, 3, 4):
        inst = _instance(seed=seed)
        fit = _fit(inst)
        f_prev = float(_fit(inst)(inst.prev_positions.reshape(-1)))
        res = build_optimizer("mclp_ls", {}, BUDGET).optimize(
            inst, fit, np.random.default_rng(9)
        )
        assert res.best_fitness >= f_prev - 1e-12, (
            f"[seed {seed}] mclp_ls returned {res.best_fitness:.6f}, worse than the "
            f"current layout {f_prev:.6f} — the incumbent floor is not holding"
        )


def mclp_ls_polish_never_drops_a_served_device():
    from uavbench.optimizers.mclp_ls import MCLPLocalSearch

    inst = _instance(seed=5)
    fit = _fit(inst)
    opt = MCLPLocalSearch(**BUDGET)
    res = opt.optimize(inst, fit, np.random.default_rng(2))
    pos = res.best_position.reshape(inst.K, 3)
    comp = _fit(inst).components(res.best_position)
    assignment = comp.assignment.assignment
    polished = opt._polish_toward_prev(inst, pos, assignment)
    for j in range(inst.K):
        served = inst.device_coords[assignment == j]
        if served.shape[0] == 0:
            continue
        d = np.sqrt(((served - polished[j]) ** 2).sum(axis=1))
        assert d.max() <= inst.R_comm + 1e-6, (
            f"UAV {j} slid {d.max() - inst.R_comm:.3f} m past the range gate during "
            "the movement polish — it dropped a device it was serving"
        )


def mclp_ls_beats_the_clustering_baselines_where_coverage_binds():
    """The claim the method exists to make, guarded against silent regression."""
    wins = 0
    seeds = range(6)
    for seed in seeds:
        inst = _instance(seed=seed, R_comm=500.0)
        scores = {}
        for name in ("mclp_ls", "centroid", "cap_kmeans", "pso"):
            scores[name] = build_optimizer(name, {}, BUDGET).optimize(
                inst, _fit(inst), np.random.default_rng(9)
            ).best_fitness
        if scores["mclp_ls"] >= max(scores[m] for m in ("centroid", "cap_kmeans", "pso")):
            wins += 1
    assert wins >= 5, (
        f"mclp_ls led on only {wins}/{len(list(seeds))} seeds at R_comm=500 m. It is "
        "the proposed placement method; if it no longer dominates the baselines where "
        "coverage binds, the placement chapter's headline claim has regressed"
    )


def mclp_ls_reports_whether_the_candidate_cap_bound():
    inst = _instance()
    res = build_optimizer("mclp_ls", {"max_candidates": 50}, BUDGET).optimize(
        inst, _fit(inst), np.random.default_rng(1)
    )
    assert res.meta["candidate_cap_hit"] is True, (
        "a run capped at 50 candidates did not flag candidate_cap_hit — a truncated "
        "candidate set would be indistinguishable from an exact one in the results"
    )


# --- 4. metaheuristic controls -------------------------------------------


def de_and_gwo_search_operators_are_live():
    """An inert control proves nothing, so assert the operators actually search.

    Deliberately initialized *uniformly*. From the shared value-weighted k-means++
    seeding both DE and GWO measurably fail to improve at all at small budget
    (0/4 seeds, gain exactly 0.0, while PSO gains +0.037) — the seed is already
    better than their operators can reach. That is a finding about the landscape,
    not a bug, and it is reported as one; testing for improvement from the seeded
    population would encode the landscape's difficulty as a correctness bug.
    From a bad start they gain +0.186 and +0.294, which is what "the operator
    works" actually means here.
    """
    for name in ("de", "gwo"):
        inst = _instance(seed=2)
        res = build_optimizer(name, {"seeding": "uniform"}, BUDGET).optimize(
            inst, _fit(inst), np.random.default_rng(6)
        )
        assert res.convergence[-1] > res.convergence[0] + 1e-3, (
            f"{name} barely moved off a uniform random start "
            f"({res.convergence[0]:.6f} -> {res.convergence[-1]:.6f}) — the search "
            "operator is inert and it cannot serve as a control"
        )


def recombination_operators_cannot_beat_the_constructive_seed():
    """Documents *why* candidate-set construction replaces the swarm.

    Placement is permutation-symmetric — relabelling the UAVs gives the same
    layout — so operators that recombine coordinates across two solutions (DE's
    difference vector, GWO's leader averaging) blend layouts that use different
    labellings and destroy structure. Both therefore stall on a good seed while
    PSO, whose per-particle memory anchors each label, still improves.

    If this ever stops holding, the argument for :mod:`.mclp_ls` weakens and the
    paper's framing needs revisiting — so it is asserted rather than assumed.
    """
    inst = _instance(seed=2)
    gains = {}
    for name in ("de", "gwo", "pso"):
        res = build_optimizer(name, {}, BUDGET).optimize(inst, _fit(inst), np.random.default_rng(6))
        gains[name] = res.convergence[-1] - res.convergence[0]
    assert gains["pso"] > max(gains["de"], gains["gwo"]), (
        f"the recombination handicap has reversed: gains {gains} — PSO no longer "
        "leads DE/GWO from the shared constructive seed"
    )


def de_and_gwo_convergence_is_monotone():
    for name in ("de", "gwo"):
        inst = _instance(seed=2)
        res = build_optimizer(name, {}, BUDGET).optimize(inst, _fit(inst), np.random.default_rng(6))
        c = np.asarray(res.convergence)
        assert (np.diff(c) >= -1e-12).all(), (
            f"{name} best-so-far curve decreases — the incumbent is being overwritten "
            "by a worse solution"
        )


def spiral_and_cap_kmeans_use_the_shared_radius():
    """Equal-radius by construction: neither may publish its own radii override."""
    inst = _instance()
    for name in ("spiral", "cap_kmeans"):
        res = build_optimizer(name, {}, BUDGET).optimize(inst, _fit(inst), np.random.default_rng(0))
        assert res.meta.get("radii") is None, (
            f"{name} published a radii override; it is specified to be scored at the "
            "shared R_comm gate, and a private radius would reintroduce the handicap"
        )


if __name__ == "__main__":
    check("reported fitness matches the returned position", reported_fitness_matches_returned_position)
    check("no method outspends the shared budget", no_method_outspends_the_shared_budget)
    check("every method returns K in-bounds positions", every_method_returns_k_positions_in_bounds)
    check("every method deploys in three dimensions", every_method_deploys_in_three_dimensions)
    check("altitude descent never worsens a layout", altitude_optimization_never_makes_a_layout_worse)
    check("path-loss baselines still publish radii", path_loss_baselines_are_flagged_as_unequal_radius)
    check("equal-radius re-score is not inert", equal_radius_rescore_changes_the_ranking_metric)
    check("intersection points cover both generators", intersection_points_are_wedged_against_both_devices)
    check("candidate set matches brute-force best disc", candidate_set_matches_brute_force_best_disc)
    check("capped_covered_value matches naive top-cap", capped_covered_value_matches_the_naive_computation)
    check("effective radius shrinks with altitude", effective_radius_shrinks_with_altitude)
    check("candidate cap is respected", candidate_cap_is_respected_and_reported)
    check("mclp_ls never worse than standing still", mclp_ls_never_returns_worse_than_standing_still)
    check("mclp_ls polish never drops a served device", mclp_ls_polish_never_drops_a_served_device)
    check("mclp_ls leads where coverage binds", mclp_ls_beats_the_clustering_baselines_where_coverage_binds)
    check("mclp_ls flags a bound candidate cap", mclp_ls_reports_whether_the_candidate_cap_bound)
    check("de/gwo search operators are live", de_and_gwo_search_operators_are_live)
    check("recombination stalls on the constructive seed",
          recombination_operators_cannot_beat_the_constructive_seed)
    check("de/gwo convergence is monotone", de_and_gwo_convergence_is_monotone)
    check("spiral/cap_kmeans use the shared radius", spiral_and_cap_kmeans_use_the_shared_radius)
    finish()
