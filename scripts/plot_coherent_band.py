#!/usr/bin/env python
"""The coherent altitude-radius figure — the paper's lead methodological claim.

Under the Al-Hourani air-to-ground model the served ground radius is unimodal in
altitude: too low and buildings block the link, too high and free-space loss
dominates. The peak sits at a fixed elevation angle (20.34 deg suburban), so for
a given path-loss budget there is one best altitude and, symmetrically, for a
target radius there is a band of altitudes that can actually deliver it.

Evaluations that fix an altitude band and a communication radius independently
can therefore ask for a radius no altitude in the band can serve. That is
Defect 1: a 20-120 m band supports roughly 54-324 m of ground radius, while the
experiments were configured with R_comm up to 20 km, where the elevation angle
is 0.34 deg and P(LoS) is about 3% — an aerial base station with its defining
advantage switched off.

This draws the curve, marks the void band against the corrected one, and shows
where each configured R_comm falls. Analytic — no simulation data needed.

Usage:  python scripts/plot_coherent_band.py [--out results/protocol_figs]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uavbench.problem.path_loss import (  # noqa: E402
    ENV_PRESETS,
    coverage_radius,
    optimal_elevation_angle,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/protocol_figs")
    ap.add_argument("--env", default="suburban")
    ap.add_argument("--freq-ghz", type=float, default=2.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = ENV_PRESETS[args.env]
    freq_hz = args.freq_ghz * 1e9
    theta = optimal_elevation_angle(**env)
    tan_t = float(np.tan(np.radians(theta)))

    z = np.logspace(np.log10(10), np.log10(5000), 400)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # One curve per path-loss budget: each is the radius achievable at altitude z.
    for pl, style in ((120.0, "-"), (130.0, "--"), (140.0, ":")):
        r = np.array([coverage_radius(float(zz), pl, freq_hz, **env) for zz in z])
        ax.plot(z, r / 1000.0, style, lw=1.8, label=f"path-loss budget {pl:.0f} dB")

    # The locus of optima: r = z / tan(theta_opt) is where every curve peaks.
    ax.plot(z, (z / tan_t) / 1000.0, color="0.35", lw=1.0, alpha=0.9,
            label=rf"coherent optimum $r=z/\tan\theta^*$ ($\theta^*$={theta:.2f}$\degree$)")

    ax.axvspan(20, 120, color="tab:red", alpha=0.13)
    ax.axvspan(100, 2000, color="tab:green", alpha=0.11)
    ax.text(48, 12, "void band\n20-120 m", color="tab:red", fontsize=8.5,
            ha="center", va="top")
    ax.text(560, 12, "corrected band\n100-2000 m", color="tab:green", fontsize=8.5,
            ha="center", va="top")

    for r_km, lbl in ((20.0, "R=20 km (v4)"), (5.0, "R=5 km (v5 FL)"),
                      (0.5, "R=500 m (Tier-1)")):
        ax.axhline(r_km, color="0.5", lw=0.8, ls="-.")
        ax.text(11, r_km * 1.06, lbl, fontsize=8, color="0.3")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("UAV altitude z (m, log)")
    ax.set_ylabel("served ground radius (km, log)")
    ax.set_title("Coherent altitude-radius regime (Al-Hourani, suburban)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    p = out / "coherent_band.png"
    fig.savefig(p, dpi=200)
    fig.savefig(out / "coherent_band.pdf")
    print(f"wrote {p} (+ .pdf)")

    # The numbers the caption quotes, so the text cannot drift from the figure.
    print(f"\ntheta_opt = {theta:.4f} deg,  tan = {tan_t:.5f}")
    for lo, hi, tag in ((20.0, 120.0, "void band"), (100.0, 2000.0, "corrected band"),
                        (100.0, 400.0, "Tier-1 band")):
        print(f"{tag:<16} z in [{lo:>6.0f}, {hi:>6.0f}] m -> coherent ground radius "
              f"[{lo / tan_t:>8.0f}, {hi / tan_t:>8.0f}] m")
    for r in (500.0, 5000.0, 20000.0):
        print(f"R_comm {r:>7.0f} m needs z* = {r * tan_t:>8.0f} m for coherence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
