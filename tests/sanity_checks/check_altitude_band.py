"""The altitude band must leave the vertical decision non-degenerate.

`link.py` exists to make altitude a real decision variable: the Al-Hourani
radius-versus-altitude curve is unimodal, so a correct 3D optimizer should find
an *interior* optimum instead of pinning every UAV to a bound. That only holds
if the configured band actually contains the peak.

Under the old 20-120 m band it did not. The channel-optimal ground radius is
``z / tan(theta_opt)`` with ``theta_opt = 20.34 deg`` for suburban, so a 120 m
ceiling caps the coherent radius at ~324 m: every ``R_comm`` above that pinned
``z_star`` at the ceiling and "3D placement" silently collapsed back to a planar
placement carrying a height column. At ``R_comm = 20 km`` it was worse than
degenerate — an elevation angle of 0.34 deg means P(LoS) ~ 3%, so the
line-of-sight advantage that motivates an aerial base station was switched off,
and the radius was only reachable because ``LinkModel`` back-solves whatever
path-loss budget the configured ``R_comm`` requires.

These checks are the standing guard on that. A future config that tightens the
band or raises ``R_comm`` past the band's reach fails here rather than producing
a plausible-looking sweep with no physics in it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np  # noqa: E402
from _lib import check, finish  # noqa: E402

from uavbench.fl.federated import Z_MAX_M_DEFAULT, Z_MIN_M_DEFAULT  # noqa: E402
from uavbench.problem.link import LinkModel  # noqa: E402
from uavbench.problem.path_loss import ENV_PRESETS, los_probability  # noqa: E402
from uavbench.runner import load_config  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
# Configs whose placement runs must sit in a coherent regime, with the R_comm
# each one actually deploys at.
_PAPER_CONFIGS = ("paper_full.yaml", "paper_uav_count.yaml")


def _z_star(r_comm, z_min=Z_MIN_M_DEFAULT, z_max=Z_MAX_M_DEFAULT):
    return LinkModel(r_comm_m=r_comm, z_min_m=z_min, z_max_m=z_max).z_star_m


def optimum_is_interior_in_the_default_band():
    """z_star must be strictly inside the band, or altitude is not a decision.

    Valid only inside the band's reach. The coherent ground-radius interval is
    ``z/tan(theta_opt)`` = 270 m .. 2700 m for a 100-1000 m band; outside it the
    optimum correctly sits on a bound, which is a property of the geometry, not
    a bug. `coherent_interval_edges_are_where_expected` pins those edges.
    """
    for r_comm in (500.0, 1000.0, 2000.0, 2500.0):
        z = _z_star(r_comm)
        assert Z_MIN_M_DEFAULT < z < Z_MAX_M_DEFAULT, (
            f"R_comm={r_comm}: z*={z} is pinned at a bound of "
            f"[{Z_MIN_M_DEFAULT}, {Z_MAX_M_DEFAULT}] — vertical decision degenerate"
        )


def coherent_interval_edges_are_where_expected():
    """Radii outside z/tan(theta_opt) must pin — that is the constraint's shape.

    Documents the usable window so a future config change that walks past it
    fails loudly here instead of quietly reproducing the degenerate regime this
    whole change was made to escape.
    """
    assert _z_star(4000.0) >= Z_MAX_M_DEFAULT - 1e-6, (
        "R_comm=4 km should exceed the band's reach and pin at the ceiling; "
        "if it no longer does, the coherent interval has moved and the shipped "
        "R_comm values need rechecking"
    )
    assert _z_star(150.0) <= Z_MIN_M_DEFAULT + 1e-6, (
        "R_comm=150 m should sit below the band's reach and pin at the floor"
    )


def the_old_band_is_shown_to_be_degenerate():
    """Guard the guard: the check must actually be able to fail.

    If this stops failing under the retired 20-120 m band, the assertion above
    has gone vacuous and would pass for any band at all.
    """
    pinned = [r for r in (500.0, 2000.0, 20000.0) if _z_star(r, 20.0, 120.0) >= 119.0]
    assert pinned, (
        "the old 20-120 m band no longer pins z* at its ceiling — "
        "optimum_is_interior_in_the_default_band is no longer a real test"
    )


def altitude_actually_changes_the_radius():
    """A flat radius curve would make the vertical search meaningless."""
    link = LinkModel(r_comm_m=2000.0, z_min_m=Z_MIN_M_DEFAULT, z_max_m=Z_MAX_M_DEFAULT)
    z = np.linspace(Z_MIN_M_DEFAULT, Z_MAX_M_DEFAULT, 25)
    r = link.radius(z)
    assert r.max() > 1.10 * r.min(), (
        f"radius varies only {r.min():.0f}-{r.max():.0f} m across the band — "
        "altitude barely matters, so 3D placement is decorative"
    )
    # Unimodal with an interior peak: rises then falls, never monotone.
    assert np.argmax(r) not in (0, len(r) - 1), "radius peaks at a band edge"


def operating_point_is_los_dominated():
    """The UAV's whole advantage is LoS — the deployment must actually have it."""
    env = ENV_PRESETS["suburban"]
    for r_comm in (1000.0, 2000.0):
        z = _z_star(r_comm)
        theta_deg = np.degrees(np.arctan2(z, r_comm))
        p_los = los_probability(theta_deg, env["a"], env["b"])
        assert p_los > 0.25, (
            f"R_comm={r_comm}: at z*={z:.0f} m the elevation angle is "
            f"{theta_deg:.2f} deg giving P(LoS)={p_los:.3f} — the link is NLoS-"
            "dominated, so this is a ground mast, not an aerial base station"
        )
    # The retired operating point must fail that bar, or the bar is meaningless.
    theta_20km = np.degrees(np.arctan2(120.0, 20000.0))
    assert los_probability(theta_20km, env["a"], env["b"]) < 0.10, (
        "the old 20 km / 120 m point is no longer NLoS-dominated — "
        "the P(LoS) threshold above is not discriminating"
    )


def shipped_configs_stay_in_the_coherent_band():
    """Every paper config must deploy at an R_comm its altitude band can serve."""
    for name in _PAPER_CONFIGS:
        path = _REPO / "configs" / name
        if not path.exists():
            continue
        cfg = load_config(path)
        fl = cfg["fl"]
        z_min = float(fl.get("z_min_m", Z_MIN_M_DEFAULT))
        z_max = float(fl.get("z_max_m", Z_MAX_M_DEFAULT))
        r_comm = float(fl["R_comm"])
        # Every radius the config actually deploys at, including swept grids —
        # checking only fl.R_comm would miss a sweep whose cells leave the band.
        radii = [r_comm] + [float(r) for r in cfg.get("R_comm_values", [])]
        for r in radii:
            z = _z_star(r, z_min, z_max)
            assert z_min < z < z_max, (
                f"{name}: R_comm={r} with band [{z_min}, {z_max}] pins z*={z} "
                "at a bound — placement would be run outside the model's physics"
            )


def swept_radius_grids_stay_in_the_band():
    """The coverage sweep's whole grid must be coherent, not just its default."""
    path = _REPO / "configs" / "paper_coverage.yaml"
    if not path.exists():
        return
    cfg = load_config(path)
    fl = cfg["fl"]
    z_min = float(fl.get("z_min_m", Z_MIN_M_DEFAULT))
    z_max = float(fl.get("z_max_m", Z_MAX_M_DEFAULT))
    grid = [float(r) for r in cfg["R_comm_values"]]
    assert grid, "paper_coverage has no R_comm_values"
    for r in grid:
        z = _z_star(r, z_min, z_max)
        assert z_min < z < z_max, (
            f"paper_coverage R_comm={r} pins z*={z} in band [{z_min}, {z_max}] — "
            "that cell would repeat the degenerate-altitude regime"
        )


check("optimum is interior in the default band", optimum_is_interior_in_the_default_band)
check("coherent interval edges are where expected", coherent_interval_edges_are_where_expected)
check("the retired 20-120 m band is demonstrably degenerate", the_old_band_is_shown_to_be_degenerate)
check("altitude materially changes the radius", altitude_actually_changes_the_radius)
check("operating point is LoS-dominated", operating_point_is_los_dominated)
check("shipped configs stay in the coherent band", shipped_configs_stay_in_the_coherent_band)
check("swept radius grids stay in the band", swept_radius_grids_stay_in_the_band)
finish()
