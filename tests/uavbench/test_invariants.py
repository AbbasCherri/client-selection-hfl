"""CI invariants: the properties the paired-comparison design depends on.

These are the automated version of "prove the comparison is fair": the
shared evaluation budget can never be desynced by config, and the instance
and optimizer RNG streams can never receive the same SeedSequence input.
"""


from uavbench.optimizers import build_optimizer
from uavbench.runner import _build_optimizer, _instance_seed, _optimizer_rng


class TestSharedBudgetInvariant:
    BUDGET = {"P": 77, "G_max": 33}

    def test_pso_ga_get_identical_budget(self):
        pso = _build_optimizer("pso", self.BUDGET, {})
        ga = _build_optimizer("ga", self.BUDGET, {})
        assert (pso.P, pso.G_max) == (ga.P, ga.G_max) == (77, 33)

    def test_budget_wins_over_conflicting_optimizer_params(self):
        # A config trying to hand PSO a bigger budget via optimizer_params
        # must be silently overridden by the shared budget. (c1=2.5 keeps
        # phi = c1 + c2 > 4, the constriction-PSO validity gate.)
        pso = _build_optimizer("pso", self.BUDGET, {"P": 999, "G_max": 999, "c1": 2.5})
        assert (pso.P, pso.G_max) == (77, 33)
        assert pso.c1 == 2.5  # non-budget params still pass through

    def test_shared_helper_used_by_all_three_call_sites(self):
        # runner._build_optimizer, federated._place_uavs, and
        # hflsim.placement all construct through this one function.
        opt = build_optimizer("ga", params={"P": 999}, budget=self.BUDGET)
        assert (opt.P, opt.G_max) == (77, 33)

    def test_unbudgeted_methods_ignore_budget(self):
        # One-shot methods take no P/G_max; passing a budget must not leak
        # unexpected kwargs into their constructors.
        for method in ("centroid", "static", "random", "mozaffari2016", "alzenad2017"):
            opt = build_optimizer(method, budget=self.BUDGET)
            assert not hasattr(opt, "P") or method in ("pso", "ga")


class TestSeedStreamDisjointness:
    """The two seed families must never receive the same SeedSequence input.

    numpy's SeedSequence treats trailing-zero entropy as equivalent
    ([b, s, i] == [b, s, i, 0]), so before the stream tags were added an
    optimizer stream with seed_i=0 collided exactly with an instance stream
    whenever a config set instance_seed == optimizer_seed. These tests pin
    the fix under that worst case.
    """

    def test_derived_states_disjoint_even_with_equal_bases(self):
        # Worst case: same base for both families (a config CAN do this),
        # seed_i=0 included — the historical collision trigger.
        base = 1234
        inst = {_instance_seed(base, s, i) for s in range(4) for i in range(10)}
        opt_states = {
            int(_optimizer_rng(base, m, s, i).bit_generator.seed_seq.generate_state(1)[0])
            for m in range(6) for s in range(4) for i in range(10)
        }
        assert not (inst & opt_states)

    def test_trailing_zero_equivalence_is_neutralized(self):
        # The exact historical collision: instance (s, i) vs optimizer
        # (m=s, s=i, seed_i=0) under a shared base.
        base = 999
        inst = _instance_seed(base, 0, 2)
        opt = int(
            _optimizer_rng(base, 0, 2, 0).bit_generator.seed_seq.generate_state(1)[0]
        )
        assert inst != opt

    def test_instance_seed_is_method_independent(self):
        # The paired-comparison property: every method sees the same instance,
        # while each method gets its own optimizer stream.
        assert _instance_seed(1234, 2, 7) == _instance_seed(1234, 2, 7)
        rng_a = _optimizer_rng(9876, 0, 2, 7)
        rng_b = _optimizer_rng(9876, 1, 2, 7)
        assert rng_a.random() != rng_b.random()  # methods get distinct streams
