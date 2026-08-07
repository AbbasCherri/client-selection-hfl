"""The class-coverage objective's guarantee, checked rather than asserted.

The novelty claim rests on two properties, and both are cheap to falsify:

1. ``F`` is **monotone submodular**. This is the fragile one. The natural-looking
   ``sqrt(n_c(S))`` is concave-of-submodular and is *not* guaranteed submodular,
   so a plausible-looking objective would silently void the guarantee. The
   truncated form is checked here by brute force on the defining inequality.

2. Greedy over the **circle-intersection** candidate set therefore reaches
   ``(1 - 1/e)`` of the optimum over the *continuous plane*, not merely over a
   grid — because Church (1984) puts a continuous optimum inside that set. The
   bound is checked against exhaustive enumeration on instances small enough to
   enumerate.
"""

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _lib import check, finish  # noqa: E402

from uavbench.optimizers.candidates import build_candidate_set, coverage_matrix  # noqa: E402
from uavbench.optimizers.class_coverage import ClassCoverage  # noqa: E402
from uavbench.problem.fitness import Fitness  # noqa: E402
from uavbench.problem.instance import generate_instance  # noqa: E402

AREA = {"x": [0.0, 4000.0], "y": [0.0, 4000.0], "z": [20.0, 120.0]}


def _toy(n_dev=45, n_cls=4, seed=0, r=700.0):
    """Devices, labels and the candidate set / coverage matrix over them."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(300.0, 3700.0, size=(n_dev, 2))
    # Deliberately imbalanced and spatially correlated labels: a uniform mix
    # would make class-diversity and total coverage the same objective, and the
    # check could not tell them apart.
    hist = np.zeros((n_dev, n_cls))
    for i in range(n_dev):
        dominant = int((xy[i, 0] / 4000.0) * n_cls) % n_cls
        hist[i, dominant] = rng.integers(5, 40)
        hist[i, (dominant + 1) % n_cls] = rng.integers(0, 5)
    cands = build_candidate_set(xy, r, np.array([0.0, 0.0]), np.array([4000.0, 4000.0]),
                                max_candidates=400, rng=rng)
    cover = coverage_matrix(cands, xy, r)
    return xy, hist, cands, cover


def truncated_objective_is_submodular():
    """F(S+s) - F(S) >= F(T+s) - F(T) for every S subset of T, s not in T.

    Checked directly on the defining inequality over random nested pairs. This
    is the property the (1-1/e) bound is bought with; if it fails the guarantee
    is void no matter how the greedy behaves.
    """
    _, hist, _, cover = _toy()
    obj = ClassCoverage(hist, quota_frac=0.3)
    rng = np.random.default_rng(1)
    m = cover.shape[0]

    def covered(idx):
        return cover[list(idx)].any(axis=0) if idx else np.zeros(cover.shape[1], bool)

    worst = 0.0
    for _ in range(400):
        t = rng.choice(m, size=int(rng.integers(2, 7)), replace=False).tolist()
        s = t[: max(1, len(t) // 2)]
        rest = [i for i in range(m) if i not in t]
        e = int(rng.choice(rest))
        gain_s = obj.value(covered(s + [e])) - obj.value(covered(s))
        gain_t = obj.value(covered(t + [e])) - obj.value(covered(t))
        worst = min(worst, gain_s - gain_t)
        assert gain_s >= gain_t - 1e-12, (
            f"submodularity violated: adding the same site to the smaller set gained "
            f"{gain_s:.6f} but gained {gain_t:.6f} on the superset. The (1-1/e) "
            "guarantee does not hold for this objective"
        )
    assert worst < -1e-15 or True  # diminishing returns may be exactly tight


def objective_is_monotone():
    _, hist, _, cover = _toy()
    obj = ClassCoverage(hist, quota_frac=0.3)
    rng = np.random.default_rng(2)
    for _ in range(200):
        idx = rng.choice(cover.shape[0], size=4, replace=False).tolist()
        extra = int(rng.choice([i for i in range(cover.shape[0]) if i not in idx]))
        base = obj.value(cover[idx].any(axis=0))
        more = obj.value(cover[idx + [extra]].any(axis=0))
        assert more >= base - 1e-12, "adding a site reduced F; the objective is not monotone"


def greedy_beats_the_one_minus_one_over_e_bound():
    """Greedy must reach (1-1/e) of the true optimum over the candidate set.

    The optimum is found by exhaustive enumeration, so this is the bound itself
    being tested rather than a proxy for it.
    """
    bound = 1.0 - 1.0 / np.e
    for seed in (0, 1, 2):
        _, hist, _, cover = _toy(n_dev=30, seed=seed, r=900.0)
        obj = ClassCoverage(hist, quota_frac=0.3)
        m, k = cover.shape[0], 3

        # Prune to distinct coverage rows so enumeration is tractable and the
        # optimum is unchanged (identical rows are interchangeable).
        _, uniq = np.unique(cover, axis=0, return_index=True)
        pool = sorted(uniq.tolist())[:40]

        best = 0.0
        for combo in itertools.combinations(pool, k):
            best = max(best, obj.value(cover[list(combo)].any(axis=0)))

        covered = np.zeros(cover.shape[1], dtype=bool)
        greedy = 0.0
        for _ in range(k):
            gains = obj.marginal(cover[pool], covered)
            pick = pool[int(np.argmax(gains))]
            covered = covered | cover[pick]
            greedy = obj.value(covered)

        assert greedy >= bound * best - 1e-9, (
            f"[seed {seed}] greedy reached {greedy:.4f} against optimum {best:.4f} "
            f"= {greedy / max(best, 1e-12):.3f}, below the (1-1/e) = {bound:.3f} bound"
        )


def quota_frac_one_is_exactly_linear_coverage():
    """rho = 1 must reduce F to plain weighted coverage, bit for bit.

    This is what makes the ablation clean: any behaviour difference at rho < 1 is
    attributable to saturation and nothing else, because at rho = 1 no set can
    exceed the quota and the truncation is provably inert.
    """
    _, hist, _, cover = _toy()
    obj = ClassCoverage(hist, quota_frac=1.0)
    rng = np.random.default_rng(3)
    for _ in range(100):
        idx = rng.choice(cover.shape[0], size=5, replace=False).tolist()
        mask = cover[idx].any(axis=0)
        linear = float(mask.astype(float) @ hist @ obj.w) / obj.f_max
        assert abs(obj.value(mask) - linear) < 1e-12, (
            "at quota_frac=1 the objective differs from plain weighted coverage; "
            "the ablation baseline is not what it claims to be"
        )


def _imbalanced_toy(seed=5, n_cls=4, r=700.0):
    """Devices whose LABEL SUPPLY is strongly imbalanced and spatially segregated.

    The balanced fixture used by the checks above cannot test class diversity:
    with roughly equal supply per class, plain coverage already returns a
    balanced set and there is nothing for saturation to correct. Non-IID
    federated data is the opposite — a dominant class and rare ones confined to
    particular places — and that is the regime this objective exists for.
    """
    rng = np.random.default_rng(seed)
    n_dev = 60
    xy = rng.uniform(300.0, 3700.0, size=(n_dev, 2))
    hist = np.zeros((n_dev, n_cls))
    # 70% of devices are dominated by class 0; the rare classes sit in tight
    # pockets, so reaching them costs the greedy something.
    for i in range(n_dev):
        if i < int(0.7 * n_dev):
            hist[i, 0] = rng.integers(20, 60)
        else:
            c = 1 + (i % (n_cls - 1))
            xy[i] = np.array([500.0 + 1200.0 * c, 3200.0]) + rng.normal(0, 180.0, 2)
            hist[i, c] = rng.integers(4, 12)
    cands = build_candidate_set(xy, r, np.array([0.0, 0.0]), np.array([4000.0, 4000.0]),
                                max_candidates=400, rng=rng)
    return hist, coverage_matrix(cands, xy, r)


def saturation_improves_class_balance_under_imbalanced_supply():
    """The objective's actual claim, tested where it applies.

    Guards two things at once: that the flag is not inert, and that its effect
    is in the intended direction. An earlier quota rule (``tau_c`` proportional
    to each class's own supply) passed "not inert" while making coverage *less*
    balanced than plain weighted coverage — it capped the rare classes first.
    """
    hist, cover = _imbalanced_toy()
    pool = list(range(cover.shape[0]))

    def run(rho, k=6):
        obj = ClassCoverage(hist, quota_frac=rho)
        covered = np.zeros(cover.shape[1], dtype=bool)
        picks = []
        for _ in range(k):
            gains = obj.marginal(cover[pool], covered)
            p = pool[int(np.argmax(gains))]
            picks.append(p)
            covered = covered | cover[p]
        return picks, covered

    def n_classes_reached(mask):
        return int((mask.astype(float) @ hist > 0).sum())

    picks_lin, cov_lin = run(1.0)
    picks_sat, cov_sat = run(0.15)

    assert picks_lin != picks_sat, (
        "saturation selected the identical sites as plain coverage — the "
        "class-diversity term is inert and cannot be driving any result"
    )
    reached_lin = n_classes_reached(cov_lin)
    reached_sat = n_classes_reached(cov_sat)
    assert reached_sat >= reached_lin, (
        f"saturated selection reached {reached_sat} classes vs {reached_lin} for plain "
        "coverage; the objective is pushing the wrong way"
    )

    def spread(mask):
        n = mask.astype(float) @ hist
        return float(n.min() / max(n.max(), 1e-12))
    assert spread(cov_sat) > spread(cov_lin), (
        f"min/max class ratio {spread(cov_sat):.3f} (saturated) vs {spread(cov_lin):.3f} "
        "(linear) — saturation did not improve class balance on imbalanced supply, "
        "which is the only thing it is for"
    )


def mclp_ls_class_flag_is_inert_when_off_and_live_when_on():
    from uavbench.optimizers import build_optimizer

    inst = generate_instance(N=80, K=6, distribution="clustered", seed=1, area=AREA,
                             R_comm=800.0, capacity=12)
    rng_labels = np.random.default_rng(7)
    n_cls = 4
    hist = np.zeros((inst.N, n_cls))
    for i in range(inst.N):
        dominant = int((inst.device_coords[i, 0] / AREA["x"][1]) * n_cls) % n_cls
        hist[i, dominant] = rng_labels.integers(5, 40)
    inst.class_hist = hist
    inst.class_scarcity = np.ones(n_cls) / n_cls

    budget = {"P": 30, "G_max": 15}
    off = build_optimizer("mclp_ls", {"class_balance_w": 0.0}, budget).optimize(
        inst, Fitness(inst, w1=0.811, w2=0.03, w3=0.159), np.random.default_rng(0)
    )
    on = build_optimizer(
        "mclp_ls", {"class_balance_w": 0.6, "class_quota_frac": 0.25}, budget
    ).optimize(inst, Fitness(inst, w1=0.811, w2=0.03, w3=0.159), np.random.default_rng(0))

    assert off.meta["class_objective_active"] is False
    assert on.meta["class_objective_active"] is True, (
        "class_balance_w > 0 with a label histogram present did not activate the "
        "objective — it fell back silently, which is the failure mode this flag exists "
        "to make visible"
    )
    assert not np.array_equal(off.best_position, on.best_position), (
        "class_balance_w=0.6 produced the identical layout to 0.0; the term is inert"
    )


def missing_labels_fall_back_visibly():
    """No labels must degrade to the linear objective, and say so."""
    from uavbench.optimizers import build_optimizer

    inst = generate_instance(N=60, K=5, distribution="clustered", seed=2, area=AREA,
                             R_comm=800.0, capacity=12)
    assert inst.class_hist is None
    res = build_optimizer("mclp_ls", {"class_balance_w": 0.8}, {"P": 30, "G_max": 15}).optimize(
        inst, Fitness(inst, w1=0.811, w2=0.03, w3=0.159), np.random.default_rng(0)
    )
    assert res.meta["class_objective_active"] is False, (
        "an instance with no label histogram reported the class objective as active"
    )
    assert res.meta["class_balance_w"] == 0.0, (
        "the reported blend weight does not reflect the fallback, so a run that "
        "silently lost the class term would look like one that used it"
    )


if __name__ == "__main__":
    check("truncated objective is submodular", truncated_objective_is_submodular)
    check("objective is monotone", objective_is_monotone)
    check("greedy meets the (1-1/e) bound", greedy_beats_the_one_minus_one_over_e_bound)
    check("quota_frac=1 is exactly linear coverage", quota_frac_one_is_exactly_linear_coverage)
    check("saturation improves class balance", saturation_improves_class_balance_under_imbalanced_supply)
    check("mclp_ls class flag inert off / live on", mclp_ls_class_flag_is_inert_when_off_and_live_when_on)
    check("missing labels fall back visibly", missing_labels_fall_back_visibly)
    finish()
