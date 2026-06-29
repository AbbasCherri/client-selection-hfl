"""Command-line entry point: run / analyze / plot / smoke / clean."""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .plotting import analyze_dir, plot_dir
from .runner import load_config, run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("uavbench")

# Repo root (uav-pso-bench/) relative to this file: src/uavbench/cli.py -> ../../..
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_config(path_str: str) -> Path:
    """Resolve a config path relative to cwd or the repo root."""
    p = Path(path_str)
    if p.exists():
        return p
    alt = _REPO_ROOT / path_str
    if alt.exists():
        return alt
    raise FileNotFoundError(f"config not found: {path_str}")


def _print_headline(summary: pd.DataFrame) -> None:
    cols = [c for c in ["scenario", "method", "final_fitness_mean", "final_fitness_ci95",
                        "coverage_pct_mean", "wall_time_s_mean"] if c in summary.columns]
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print("\n=== Tier-1 headline (mean over seeds) ===")
        print(summary[cols].to_string(index=False))


def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_config(_find_config(args.config))
    run_experiment(cfg)


def cmd_analyze(args: argparse.Namespace) -> None:
    cfg = load_config(_find_config(args.config))
    summary = analyze_dir(Path(cfg["results_dir"]))
    _print_headline(summary)


def cmd_plot(args: argparse.Namespace) -> None:
    cfg = load_config(_find_config(args.config))
    paths = plot_dir(Path(cfg["results_dir"]))
    for p in paths:
        logger.info("Wrote figure %s", p)


def cmd_clean(args: argparse.Namespace) -> None:
    if args.config:
        cfg = load_config(_find_config(args.config))
        target = Path(cfg["results_dir"])
    else:
        target = Path("results")
    if target.exists():
        shutil.rmtree(target)
        logger.info("Removed %s", target)
    else:
        logger.info("Nothing to clean at %s", target)


def cmd_smoke(args: argparse.Namespace) -> None:
    cfg = load_config(_find_config("configs/smoke.yaml"))
    start = time.perf_counter()
    out = run_experiment(cfg)
    elapsed = time.perf_counter() - start

    runs = out["runs"]
    summary = analyze_dir(out["results_dir"])
    figs = plot_dir(out["results_dir"])
    _print_headline(summary)

    # Evals/sec micro-benchmark from the metaheuristic runs, and a projection.
    meta = runs[runs["method"].isin(["pso", "ga"])]
    total_evals = float(meta["eval_count"].sum())
    total_time = float(meta["wall_time_s"].sum())
    eps = total_evals / total_time if total_time > 0 else float("nan")
    logger.info("Smoke finished in %.1fs; %d figures written", elapsed, len(figs))
    logger.info("Metaheuristic throughput: %.0f fitness evals/sec (single-core)", eps)

    # Project a full tier1_core run (3 scenarios x 30 seeds x {pso,ga} at P*G_max).
    proj_budget = 100 * 200
    proj_runs = 3 * 30 * 2
    proj_evals = proj_runs * proj_budget
    proj_sec = proj_evals / eps / max(1, cfg["n_workers"]) if eps == eps else float("nan")
    logger.info(
        "Projected tier1_core metaheuristic time: ~%.1f min on %d workers",
        proj_sec / 60.0, cfg["n_workers"],
    )
    print(f"\nDisk footprint: {out['size_mb']:.2f} MB at {out['results_dir']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uavbench", description="PSO UAV-placement benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run an experiment grid from a config")
    p_run.add_argument("--config", required=True)
    p_run.set_defaults(func=cmd_run)

    p_an = sub.add_parser("analyze", help="summarize saved runs into a table")
    p_an.add_argument("--config", required=True)
    p_an.set_defaults(func=cmd_analyze)

    p_pl = sub.add_parser("plot", help="generate convergence figures from saved traces")
    p_pl.add_argument("--config", required=True)
    p_pl.set_defaults(func=cmd_plot)

    p_sm = sub.add_parser("smoke", help="fast end-to-end run (table + figure + projection)")
    p_sm.set_defaults(func=cmd_smoke)

    p_cl = sub.add_parser("clean", help="remove results (of a config, or all)")
    p_cl.add_argument("--config", default=None)
    p_cl.set_defaults(func=cmd_clean)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
