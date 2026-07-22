"""Command-line entry point: run / analyze / plot / smoke / clean."""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

import pandas as pd

from .fl.federated import run_tier2
from .fl.selection_isolation import run_selection_sweep
from .fl.sweep import run_paper_sweep, run_sweep
from .plotting import _last_round, analyze_dir, plot_dir, plot_sweep, plot_tier2
from .reporting import build_seed_manifest, summarize_wall_clock
from .runner import load_config, run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("uavbench")

# Repo root relative to this file: src/uavbench/cli.py -> ../..
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
    cols = [
        c
        for c in [
            "scenario",
            "method",
            "final_fitness_mean",
            "final_fitness_ci95",
            "coverage_pct_mean",
            "wall_time_s_mean",
        ]
        if c in summary.columns
    ]
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print("\n=== Tier-1 headline (mean over seeds) ===")
        print(summary[cols].to_string(index=False))


def _write_seed_manifest(cfg: dict, harness: str) -> None:
    """Persist the exact seeds this run will use, before it starts.

    Written up front so a crashed run still leaves an auditable record of
    what it was going to execute.
    """
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_seed_manifest(cfg, harness)
    manifest.to_csv(results_dir / "seed_manifest.csv", index=False)
    logger.info("Seed manifest (%d rows) at %s", len(manifest), results_dir / "seed_manifest.csv")


def _print_timing(results_dir: Path) -> None:
    """Per-method wall-clock aggregate for the paper's runtime disclosure."""
    timing = summarize_wall_clock(results_dir)
    if timing.empty:
        return
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print("\n=== Wall-clock summary (per method) ===")
        print(timing.to_string(index=False))


def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_config(_find_config(args.config))
    _write_seed_manifest(cfg, "tier1")
    run_experiment(cfg)


def cmd_analyze(args: argparse.Namespace) -> None:
    cfg = load_config(_find_config(args.config))
    summary = analyze_dir(Path(cfg["results_dir"]))
    _print_headline(summary)
    _print_timing(Path(cfg["results_dir"]))


def cmd_plot(args: argparse.Namespace) -> None:
    cfg = load_config(_find_config(args.config))
    paths = plot_dir(Path(cfg["results_dir"]))
    for p in paths:
        logger.info("Wrote figure %s", p)


# Tables cmd_significance recognises, in probe order, with their group
# columns (tests are paired within each group; () → single pooled group).
# Tier-1 runs.parquet is already one row per (method, scenario, seed);
# round tables are reduced to final-round-per-seed first.
_SIG_TABLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("runs.parquet", ("scenario",)),
    ("paper_sweep_rounds.parquet", ("N",)),
    ("stress_rounds.parquet", ("dropout_rate", "snr_degradation_db", "black_chip_rate")),
    ("selection_sweep_rounds.parquet", ("N",)),
    ("sweep_rounds.parquet", ("N",)),
    ("fullsim_rounds.parquet", ()),
    ("tier2_rounds.parquet", ()),
)


def cmd_significance(args: argparse.Namespace) -> None:
    """Paired multi-seed significance tests over an existing results directory."""
    from .analysis import pairwise_significance_table

    target = Path(args.config)
    results_dir = (
        target if target.is_dir() else Path(load_config(_find_config(args.config))["results_dir"])
    )

    df = None
    for fname, group_spec in _SIG_TABLES:
        path = results_dir / fname
        if path.exists():
            df = pd.read_parquet(path)
        elif path.with_suffix(".csv").exists():
            df = pd.read_csv(path.with_suffix(".csv"))
        if df is not None:
            break
    if df is None:
        raise FileNotFoundError(f"no recognised results table in {results_dir}")

    if "seed" not in df.columns:
        raise ValueError(
            f"{fname} has no 'seed' column — significance testing needs a "
            "multi-seed run (rerun with n_seeds >= 2)"
        )

    group_cols = [c for c in group_spec if c in df.columns]

    # Round tables → per (method, group, seed) summary. The mean of the last K
    # rounds is the reported statistic (not the final-round snapshot): under
    # ±0.05/round oscillation a single round is a lottery draw, so the paired
    # test would mostly measure snapshot noise. --last-k 1 restores the old
    # final-round behaviour.
    if "round" in df.columns:
        keys = ["method", "seed"] + group_cols
        last_k = max(int(getattr(args, "last_k", 10)), 1)
        ordered = df.sort_values(keys + ["round"])
        if last_k == 1:
            df = ordered.groupby(keys, as_index=False).last()
        else:
            tail = ordered.groupby(keys, group_keys=False).tail(last_k)
            df = tail.groupby(keys, as_index=False)[args.metric].mean()

    if not group_cols:
        df = df.assign(_all="all")
        group_cols = ["_all"]

    methods = args.methods.split(",") if args.methods else sorted(df["method"].unique())
    table = pairwise_significance_table(
        df,
        metric=args.metric,
        methods=methods,
        group_cols=group_cols,
        test=args.test,
        alpha=args.alpha,
    )
    # Metric-suffixed filename for non-default metrics so a macro_f1 pass
    # doesn't overwrite the accuracy table (plain name kept for back-compat).
    out_name = "significance.csv" if args.metric == "accuracy" else f"significance_{args.metric}.csv"
    out_path = results_dir / out_name
    table.to_csv(out_path, index=False)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(f"\n=== Paired {args.test} on '{args.metric}' (source: {fname}, Holm-corrected) ===")
        print(table.round(5).to_string(index=False))
    logger.info("Wrote %s", out_path)


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

    meta = runs[runs["method"].isin(["pso", "ga"])]
    total_evals = float(meta["eval_count"].sum())
    total_time = float(meta["wall_time_s"].sum())
    eps = total_evals / total_time if total_time > 0 else float("nan")
    logger.info("Smoke finished in %.1fs; %d figures written", elapsed, len(figs))
    logger.info("Metaheuristic throughput: %.0f fitness evals/sec (single-core)", eps)

    proj_budget = 100 * 200
    proj_runs = 3 * 30 * 2
    proj_evals = proj_runs * proj_budget
    proj_sec = proj_evals / eps / max(1, cfg["n_workers"]) if eps == eps else float("nan")
    logger.info(
        "Projected tier1_core metaheuristic time: ~%.1f min on %d workers",
        proj_sec / 60.0,
        cfg["n_workers"],
    )
    print(f"\nDisk footprint: {out['size_mb']:.2f} MB at {out['results_dir']}")


def cmd_run_tier2(args: argparse.Namespace) -> None:
    cfg = load_config(_find_config(args.config))
    _write_seed_manifest(cfg, "tier2")
    out = run_tier2(cfg)
    df = out["rounds"]
    print("\n=== Tier-2 summary (final round per method) ===")
    last = _last_round(df, ["method"])[
        ["method", "accuracy", "macro_f1", "coverage_pct", "cumulative_energy_j"]
    ].set_index("method")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(last.to_string())
    _print_timing(out["results_dir"])
    print(f"\nDisk footprint: {out['size_mb']:.2f} MB at {out['results_dir']}")


def cmd_smoke_tier2(args: argparse.Namespace) -> None:
    cfg = load_config(_find_config("configs/tier2_reduced.yaml"))
    start = time.perf_counter()
    out = run_tier2(cfg)
    elapsed = time.perf_counter() - start

    df = out["rounds"]
    last = _last_round(df, ["method"])[
        ["method", "accuracy", "macro_f1", "coverage_pct", "n_covered"]
    ].set_index("method")
    print("\n=== Tier-2 smoke (real data, reduced subsample; final round per method) ===")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(last.to_string())

    try:
        figs = plot_tier2(out["results_dir"])
        logger.info("Tier-2 smoke finished in %.1fs; %d figures written", elapsed, len(figs))
    except Exception as exc:
        logger.warning("Tier-2 plotting skipped: %s", exc)
        logger.info("Tier-2 smoke finished in %.1fs", elapsed)
    print(f"\nDisk footprint: {out['size_mb']:.2f} MB at {out['results_dir']}")


def cmd_run_paper_sim(args: argparse.Namespace) -> None:
    cfg = load_config(_find_config(args.config))
    _write_seed_manifest(cfg, "paper_sweep")
    out = run_paper_sweep(cfg)
    df = out["rounds"]

    print("\n=== Paper simulation summary (final round, mean across seeds) ===")
    summary = (
        _last_round(df, ["method", "N", "seed"])
        .groupby(["method", "N"])[
            ["accuracy", "macro_f1", "coverage_pct", "comm_mb_round", "cumulative_energy_j"]
        ]
        .mean()
        .reset_index()
    )
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(summary.round(4).to_string(index=False))

    try:
        from .plotting import plot_paper_sim

        figs = plot_paper_sim(out["results_dir"])
        logger.info("%d paper-sim figures written", len(figs))
    except Exception as exc:
        logger.warning("Paper-sim plotting skipped: %s", exc)

    _print_timing(out["results_dir"])
    print(f"\nDisk footprint: {out['size_mb']:.2f} MB at {out['results_dir']}")


def cmd_run_selection_sim(args: argparse.Namespace) -> None:
    cfg = load_config(_find_config(args.config))
    _write_seed_manifest(cfg, "selection_sweep")
    out = run_selection_sweep(cfg)
    df = out["rounds"]

    print("\n=== Selection isolation summary (final round, mean across seeds) ===")
    cols = [
        c
        for c in ("accuracy", "macro_f1", "jain_fairness", "n_unique_selected", "K_uav")
        if c in df.columns
    ]
    summary = (
        _last_round(df, ["method", "N", "seed"]).groupby(["method", "N"])[cols].mean().reset_index()
    )
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(summary.round(4).to_string(index=False))

    try:
        from .plotting import plot_selection_sim

        figs = plot_selection_sim(out["results_dir"])
        logger.info("%d selection-isolation figures written", len(figs))
    except Exception as exc:
        logger.warning("Selection-isolation plotting skipped: %s", exc)

    print(f"\nDisk footprint: {out['size_mb']:.2f} MB at {out['results_dir']}")


def cmd_run_stress_sweep(args: argparse.Namespace) -> None:
    from .fl.stress_sweep import run_stress_sweep

    cfg = load_config(_find_config(args.config))
    _write_seed_manifest(cfg, "stress_sweep")
    out = run_stress_sweep(cfg)
    df = out["rounds"]

    print("\n=== Stress-test summary (final round, mean across seeds) ===")
    summary = (
        _last_round(df, ["method", "dropout_rate", "snr_degradation_db", "black_chip_rate", "seed"])
        .groupby(["method", "dropout_rate", "snr_degradation_db", "black_chip_rate"])[
            ["accuracy", "macro_f1", "coverage_pct"]
        ]
        .mean()
        .reset_index()
    )
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(summary.round(4).to_string(index=False))

    _print_timing(out["results_dir"])
    print(f"\nDisk footprint: {out['size_mb']:.2f} MB at {out['results_dir']}")


def cmd_run_sweep(args: argparse.Namespace) -> None:
    cfg = load_config(_find_config(args.config))
    _write_seed_manifest(cfg, "sweep")
    out = run_sweep(cfg)
    df = out["rounds"]

    print("\n=== Sweep summary (final round, mean across N) ===")
    summary = (
        df.groupby(["method", "N"])
        .last()[["accuracy", "macro_f1", "coverage_pct"]]
        .reset_index()
        .pivot_table(index="method", columns="N", values="accuracy")
    )
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(summary.round(3).to_string())

    try:
        figs = plot_sweep(out["results_dir"])
        logger.info("%d sweep figures written", len(figs))
    except Exception as exc:
        logger.warning("Sweep plotting skipped: %s", exc)

    print(f"\nDisk footprint: {out['size_mb']:.2f} MB at {out['results_dir']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uavbench", description="PSO UAV-placement benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a Tier-1 experiment grid from a config")
    p_run.add_argument("--config", required=True)
    p_run.set_defaults(func=cmd_run)

    p_an = sub.add_parser("analyze", help="summarize saved Tier-1 runs into a table")
    p_an.add_argument("--config", required=True)
    p_an.set_defaults(func=cmd_analyze)

    p_pl = sub.add_parser("plot", help="generate Tier-1 convergence figures")
    p_pl.add_argument("--config", required=True)
    p_pl.set_defaults(func=cmd_plot)

    p_sm = sub.add_parser("smoke", help="fast Tier-1 end-to-end run (table + figure + projection)")
    p_sm.set_defaults(func=cmd_smoke)

    p_t2 = sub.add_parser("run_tier2", help="run Tier-2 FL benchmark from a config")
    p_t2.add_argument("--config", required=True)
    p_t2.set_defaults(func=cmd_run_tier2)

    p_s2 = sub.add_parser(
        "smoke_tier2",
        help="fast Tier-2 smoke run on real data at reduced subsample "
        "(HF_TOKEN needed on first run; warm ./data caches after that)",
    )
    p_s2.set_defaults(func=cmd_smoke_tier2)

    p_ps = sub.add_parser(
        "run_paper_sim",
        help="Full paper system simulation: proposed HFL vs baselines (N × method × seed, 8-core parallel)",
    )
    p_ps.add_argument("--config", default="configs/paper_full.yaml")
    p_ps.set_defaults(func=cmd_run_paper_sim)

    p_si = sub.add_parser(
        "run_selection_sim",
        help="Selection-isolation benchmark: selection rules head-to-head with static elbow K-means UAVs (N × mode × seed, parallel)",
    )
    p_si.add_argument("--config", default="configs/selection_isolation.yaml")
    p_si.set_defaults(func=cmd_run_selection_sim)

    p_sw = sub.add_parser(
        "run_sweep", help="N-scalability sweep (N=30..250, all methods, 8-core parallel)"
    )
    p_sw.add_argument("--config", default="configs/tier2_sweep.yaml")
    p_sw.set_defaults(func=cmd_run_sweep)

    p_st = sub.add_parser(
        "run_stress_sweep",
        help="real-data stress-test sweep: dropout / SNR degradation / black-chip rate (robustness evidence for the single-event scope)",
    )
    p_st.add_argument("--config", default="configs/stress_test.yaml")
    p_st.set_defaults(func=cmd_run_stress_sweep)

    p_sig = sub.add_parser(
        "significance",
        help="paired multi-seed significance tests (Wilcoxon/t-test, Holm-corrected) over saved results",
    )
    p_sig.add_argument("--config", required=True, help="results directory or config YAML")
    p_sig.add_argument("--metric", default="accuracy")
    p_sig.add_argument("--methods", default=None, help="comma-separated; default = all in table")
    p_sig.add_argument("--test", default="wilcoxon", choices=["wilcoxon", "ttest_rel"])
    p_sig.add_argument("--alpha", type=float, default=0.05)
    p_sig.add_argument(
        "--last-k",
        type=int,
        default=10,
        dest="last_k",
        help="reduce round tables to the mean of the last K rounds per seed "
        "(default 10; 1 = final-round snapshot). Averaging tames the "
        "±0.05/round oscillation that made a single round a lottery draw.",
    )
    p_sig.set_defaults(func=cmd_significance)

    p_cl = sub.add_parser("clean", help="remove results (of a config, or all)")
    p_cl.add_argument("--config", default=None)
    p_cl.set_defaults(func=cmd_clean)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
