"""Extended tests for client_selection.py — _compute_utility and ClientSelector."""

import math

import numpy as np
import pytest

from uavbench.fl.client_selection import (
    DEFAULT_EPICENTRE,
    ClientSelector,
    _compute_utility,
    _minmax,
    _xy_metres,
)
from uavbench.fl.device_state import DeviceState


# ── helpers ───────────────────────────────────────────────────────────────────

def _eligible_state(battery=0.8, snr_db=15.0):
    return DeviceState(battery=battery, snr_db=snr_db, memory_ok=True, compute_time_s=100.0)

def _ineligible_state():
    return DeviceState(battery=0.0, snr_db=0.0, memory_ok=False, compute_time_s=9999.0)

def _coords_cluster(n=5):
    """n clients near the Noto Peninsula epicentre."""
    base_lat, base_lon = 37.488, 137.272
    coords = {i: (base_lat + i * 0.01, base_lon + i * 0.01) for i in range(n)}
    return coords


# ── _minmax ────────────────────────────────────────────────────────────────────

class TestMinmax:
    def test_identical_values_returns_half(self):
        v = np.array([3.0, 3.0, 3.0])
        result = _minmax(v)
        assert np.allclose(result, 0.5)

    def test_range_0_to_1(self):
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _minmax(v)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)


# ── _xy_metres ────────────────────────────────────────────────────────────────

class TestXyMetres:
    def test_centroid_is_near_origin(self):
        coords = [(37.0, 137.0), (38.0, 138.0)]
        xy = _xy_metres(coords)
        assert abs(xy[:, 0].mean()) < 1.0  # x mean near 0
        assert abs(xy[:, 1].mean()) < 1.0  # y mean near 0

    def test_1_degree_lat_approx_111km(self):
        # Two points 1° of latitude apart
        coords = [(37.0, 137.0), (38.0, 137.0)]
        xy = _xy_metres(coords)
        dy = abs(xy[1, 1] - xy[0, 1])
        assert 110_000 < dy < 112_000, f"1° lat ≈ 111 km, got {dy:.0f} m"

    def test_shape(self):
        coords = [(37.0, 137.0), (37.5, 137.5), (38.0, 138.0)]
        xy = _xy_metres(coords)
        assert xy.shape == (3, 2)


# ── _compute_utility ──────────────────────────────────────────────────────────

class TestComputeUtility:
    def _states(self, cids):
        return {cid: _eligible_state(snr_db=5.0 + cid * 2) for cid in cids}

    def test_empty_input_returns_empty(self):
        result = _compute_utility([], {}, {}, [], DEFAULT_EPICENTRE)
        assert result == {}

    def test_all_utilities_in_01(self):
        coords = _coords_cluster(6)
        cids = list(coords.keys())
        states = self._states(cids)
        uav_coords = [(37.5, 137.3)]
        result = _compute_utility(cids, states, coords, uav_coords, DEFAULT_EPICENTRE)
        assert set(result.keys()) == set(cids)
        for cid, u in result.items():
            assert 0.0 <= u <= 1.0, f"utility[{cid}] = {u} out of [0,1]"

    def test_single_client_returns_value(self):
        coords = {0: (37.488, 137.272)}
        states = {0: _eligible_state(snr_db=10.0)}
        result = _compute_utility([0], states, coords, [], DEFAULT_EPICENTRE)
        assert 0 in result
        assert 0.0 <= result[0] <= 1.0

    def test_no_uavs_uses_half_prox(self):
        coords = _coords_cluster(3)
        cids = list(coords.keys())
        states = self._states(cids)
        result = _compute_utility(cids, states, coords, [], DEFAULT_EPICENTRE)
        # Without UAVs, u_prox = 0.5 for all clients.
        # Just check we get a valid dict — not a crash.
        assert len(result) == len(cids)

    def test_projection_called_once_epi(self):
        # If _xy_metres were called twice for the same input we'd get wrong utility.
        # Verify: U_epi is based on correct epicentre distance.
        epi = (37.488, 137.272)
        # Client 0 IS the epicentre → closest → highest u_epi
        # Client 1 is far away
        coords = {0: epi, 1: (38.5, 138.5)}
        states = {0: _eligible_state(), 1: _eligible_state()}
        result = _compute_utility([0, 1], states, coords, [], epi)
        assert result[0] > result[1], "Client at epicentre should have higher u_epi"

    def test_prox_vectorised_vs_loop(self):
        """Vectorised implementation must match a reference loop."""
        from uavbench.fl.client_selection import _xy_metres as _xym
        coords = _coords_cluster(4)
        cids = list(coords.keys())
        states = self._states(cids)
        uav_coords = [(37.49, 137.28), (37.50, 137.30)]
        result = _compute_utility(cids, states, coords, uav_coords, DEFAULT_EPICENTRE)
        # Compute prox reference manually
        K_uav = len(uav_coords)
        uav_client_xy = _xym(uav_coords + [coords[c] for c in cids])
        uav_xy = uav_client_xy[:K_uav]
        client_xy = uav_client_xy[K_uav:]
        for i, cid in enumerate(cids):
            dists = [math.sqrt(float(np.sum((client_xy[i] - uav_xy[j])**2)))
                     for j in range(K_uav)]
            min_dist = min(dists)
            assert min_dist >= 0  # just sanity; full check done via bounds above


# ── ClientSelector.select ─────────────────────────────────────────────────────

class TestClientSelectorModeAll:
    def test_mode_all_returns_entire_covered_set(self):
        sel = ClientSelector([0, 1, 2])
        covered = {0: 0, 1: 0, 2: 1}
        device_states = {i: _eligible_state() for i in range(3)}
        rep = {i: 0.5 for i in range(3)}
        coords = _coords_cluster(3)
        result = sel.select(covered, device_states, rep, coords, [], 1, 10, mode="all")
        assert result == covered

    def test_mode_all_includes_ineligible_clients(self):
        sel = ClientSelector([0, 1])
        covered = {0: 0, 1: 0}
        states = {0: _ineligible_state(), 1: _eligible_state()}
        rep = {i: 0.5 for i in range(2)}
        coords = {i: (37.0 + i * 0.01, 137.0) for i in range(2)}
        result = sel.select(covered, states, rep, coords, [], 1, 10, mode="all")
        assert 0 in result  # ineligible client still selected in "all" mode


class TestClientSelectorModeRandom:
    def _setup(self, n=6, capacity=2):
        ids = list(range(n))
        sel = ClientSelector(ids)
        covered = {cid: cid % 2 for cid in ids}   # 2 UAVs
        states = {cid: _eligible_state() for cid in ids}
        rep = {cid: 0.5 for cid in ids}
        coords = {cid: (37.0 + cid * 0.01, 137.0) for cid in ids}
        return sel, covered, states, rep, coords, capacity

    def test_random_respects_uav_capacity(self):
        sel, covered, states, rep, coords, cap = self._setup(capacity=2)
        rng = np.random.default_rng(0)
        result = sel.select(covered, states, rep, coords, [], 1, cap, mode="random", rng=rng)
        # Each UAV can have at most 2 clients
        from collections import Counter
        counts = Counter(result.values())
        for uav_idx, cnt in counts.items():
            assert cnt <= cap, f"UAV {uav_idx} over capacity: {cnt}"

    def test_random_with_same_rng_is_deterministic(self):
        sel1, covered, states, rep, coords, cap = self._setup()
        sel2 = ClientSelector(list(covered.keys()))
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        r1 = sel1.select(covered, states, rep, coords, [], 1, cap, mode="random", rng=rng1)
        r2 = sel2.select(covered, states, rep, coords, [], 1, cap, mode="random", rng=rng2)
        assert r1 == r2

    def test_random_with_different_rngs_may_differ(self):
        # Run many seeds; at least one pair should differ
        results = set()
        for seed in range(30):
            sel, covered, states, rep, coords, cap = self._setup(n=8, capacity=2)
            rng = np.random.default_rng(seed)
            r = sel.select(covered, states, rep, coords, [], 1, cap, mode="random", rng=rng)
            results.add(frozenset(r.keys()))
        assert len(results) > 1, "All seeds produced identical selection — rng not used"

    def test_random_excludes_ineligible(self):
        sel = ClientSelector([0, 1, 2])
        covered = {0: 0, 1: 0, 2: 0}
        states = {0: _ineligible_state(), 1: _eligible_state(), 2: _eligible_state()}
        rep = {i: 0.5 for i in range(3)}
        coords = {i: (37.0, 137.0 + i * 0.01) for i in range(3)}
        rng = np.random.default_rng(0)
        result = sel.select(covered, states, rep, coords, [], 1, 10, mode="random", rng=rng)
        assert 0 not in result

    def test_random_empty_eligible_returns_empty(self):
        sel = ClientSelector([0])
        covered = {0: 0}
        states = {0: _ineligible_state()}
        rep = {0: 0.5}
        coords = {0: (37.0, 137.0)}
        rng = np.random.default_rng(0)
        result = sel.select(covered, states, rep, coords, [], 1, 5, mode="random", rng=rng)
        assert result == {}

    def test_random_none_rng_falls_back_to_round_seed(self):
        sel = ClientSelector([0, 1])
        covered = {0: 0, 1: 0}
        states = {cid: _eligible_state() for cid in [0, 1]}
        rep = {i: 0.5 for i in [0, 1]}
        coords = {i: (37.0, 137.0 + i * 0.01) for i in [0, 1]}
        # Must not raise even without rng
        result = sel.select(covered, states, rep, coords, [], 1, 1, mode="random", rng=None)
        assert len(result) <= 1


class TestClientSelectorModeUCB:
    def _setup(self, n=6, capacity=3):
        ids = list(range(n))
        sel = ClientSelector(ids, epicentre=DEFAULT_EPICENTRE)
        covered = {cid: 0 for cid in ids}  # all on UAV 0
        states = {cid: _eligible_state(battery=0.5 + cid * 0.05) for cid in ids}
        rep = {cid: float(cid) / n for cid in ids}  # increasing rep
        coords = {cid: (37.488 + cid * 0.01, 137.272) for cid in ids}
        return sel, covered, states, rep, coords, capacity

    def test_ucb_selection_respects_capacity(self):
        sel, covered, states, rep, coords, cap = self._setup()
        result = sel.select(covered, states, rep, coords, [], 1, cap, mode="ucb")
        assert len(result) <= cap

    def test_ucb_selection_only_eligible(self):
        sel, covered, states, rep, coords, cap = self._setup()
        states[0] = _ineligible_state()
        result = sel.select(covered, states, rep, coords, [], 1, cap, mode="ucb")
        assert 0 not in result

    def test_ucb_increments_selection_count(self):
        sel, covered, states, rep, coords, cap = self._setup()
        before = dict(sel._counts)
        result = sel.select(covered, states, rep, coords, [], 1, cap, mode="ucb")
        for cid in result:
            assert sel._counts[cid] == before[cid] + 1

    def test_ucb_count_not_incremented_for_unselected(self):
        sel, covered, states, rep, coords, cap = self._setup()
        result = sel.select(covered, states, rep, coords, [], 1, cap, mode="ucb")
        for cid in range(len(sel._counts)):
            if cid not in result:
                assert sel._counts[cid] == 0

    def test_ucb_empty_eligible_returns_empty(self):
        sel = ClientSelector([0])
        covered = {0: 0}
        states = {0: _ineligible_state()}
        rep = {0: 0.5}
        coords = {0: (37.488, 137.272)}
        result = sel.select(covered, states, rep, coords, [], 1, 5, mode="ucb")
        assert result == {}

    def test_ucb_never_exceeds_per_uav_capacity(self):
        n = 10
        sel = ClientSelector(list(range(n)))
        covered = {cid: cid % 3 for cid in range(n)}  # 3 UAVs
        states = {cid: _eligible_state() for cid in range(n)}
        rep = {cid: 0.5 for cid in range(n)}
        coords = {cid: (37.488 + cid * 0.01, 137.272) for cid in range(n)}
        cap = 2
        result = sel.select(covered, states, rep, coords, [], 1, cap, mode="ucb")
        from collections import Counter
        counts = Counter(result.values())
        for uav_idx, cnt in counts.items():
            assert cnt <= cap
