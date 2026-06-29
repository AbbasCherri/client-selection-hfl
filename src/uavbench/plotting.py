"""Plotting and summary-table helpers (regenerable from saved raw metrics)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / no-display CPU box
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _read_table(path: Path) -> pd.DataFrame:
    p = path if path.exists() else path.with_suffix(".csv")
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def summarize(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std and 95% CI of key metrics per (scenario, method)."""
    metrics = ["final_fitness", "coverage_pct", "f_cover_norm", "movement_joules",
               "l_imb", "wall_time_s", "eval_count"]
    metrics = [m for m in metrics if m in runs_df.columns]
    g = runs_df.groupby(["scenario", "method"])
    out = g[metrics].agg(["mean", "std", "count"])
    # Flatten and add 95% CI half-width for the headline metric.
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    ci = 1.96 * out["final_fitness_std"] / np.sqrt(out["final_fitness_count"].clip(lower=1))
    out["final_fitness_ci95"] = ci
    return out.reset_index()


def plot_convergence(conv_df: pd.DataFrame, out_path: Path, scenario: str | None = None) -> Path:
    """Averaged best-fitness-vs-iteration curve per method with 95% CI bands."""
    if scenario is None:
        scenario = sorted(conv_df["scenario"].unique())[0]
    sub = conv_df[conv_df["scenario"] == scenario]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in sorted(sub["method"].unique()):
        m = sub[sub["method"] == method]
        pivot = m.pivot_table(index="iteration", columns="seed", values="best_fitness")
        pivot = pivot.ffill()  # carry final value for early-stopped runs
        mean = pivot.mean(axis=1)
        n = pivot.count(axis=1).clip(lower=1)
        ci = 1.96 * pivot.std(axis=1) / np.sqrt(n)
        ax.plot(mean.index, mean.values, label=method, linewidth=1.8)
        ax.fill_between(mean.index, (mean - ci).values, (mean + ci).values, alpha=0.15)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness (mean +/- 95% CI)")
    ax.set_title(f"Convergence — {scenario}")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def analyze_dir(results_dir: Path) -> pd.DataFrame:
    """Load runs.parquet, compute the summary table, and write it next to it."""
    runs = _read_table(results_dir / "runs.parquet")
    summary = summarize(runs)
    out = results_dir / "summary.parquet"
    try:
        summary.to_parquet(out, index=False)
    except Exception:
        out = results_dir / "summary.csv"
        summary.to_csv(out, index=False)
    return summary


def plot_dir(results_dir: Path) -> list[Path]:
    """Generate one convergence figure per scenario from saved traces."""
    conv = _read_table(results_dir / "convergence.parquet")
    paths = []
    for scenario in sorted(conv["scenario"].unique()):
        out = results_dir / f"convergence_{scenario}.png"
        paths.append(plot_convergence(conv, out, scenario))
    return paths


def plot_tier2(results_dir: Path) -> list[Path]:
    """Generate Tier-2 accuracy, macro-F1, and coverage curves per placement method."""
    df = _read_table(results_dir / "tier2_rounds.parquet")
    paths: list[Path] = []

    for metric, ylabel in [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
        ("coverage_pct", "Coverage (%)"),
    ]:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method in sorted(df["method"].unique()):
            sub = df[df["method"] == method]
            ax.plot(sub["round"], sub[metric], label=method, linewidth=1.8, marker="o", markersize=3)
        ax.set_xlabel("FL Round")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Tier-2: {ylabel} vs Round")
        ax.legend(frameon=False)
        fig.tight_layout()
        out = results_dir / f"tier2_{metric}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        paths.append(out)

    return paths


def plot_sweep(results_dir: Path) -> list[Path]:
    """Generate scalability sweep figures: accuracy/macro-F1 vs N, per method."""
    df = _read_table(results_dir / "sweep_rounds.parquet")
    # Use the final FL round per (N, method) as the headline value.
    final = df.groupby(["N", "method"]).last().reset_index()
    paths: list[Path] = []

    for metric, ylabel in [
        ("accuracy", "Final Accuracy"),
        ("macro_f1", "Final Macro F1"),
        ("coverage_pct", "Final Coverage (%)"),
    ]:
        if metric not in final.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in sorted(final["method"].unique()):
            sub = final[final["method"] == method].sort_values("N")
            style = "--" if method == "no_uav" else "-"
            ax.plot(sub["N"], sub[metric], label=method, linewidth=1.8,
                    marker="o", markersize=5, linestyle=style)
        ax.set_xlabel("Number of Clients (N)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Scalability Sweep: {ylabel} vs N")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        out = results_dir / f"sweep_{metric}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        paths.append(out)

    # Heatmap: accuracy[method × N]
    try:
        pivot = final.pivot(index="method", columns="N", values="accuracy")
        fig, ax = plt.subplots(figsize=(9, 4))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("N (clients)")
        ax.set_title("Final Accuracy — method × N")
        plt.colorbar(im, ax=ax, label="Accuracy")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
        fig.tight_layout()
        out = results_dir / "sweep_heatmap_accuracy.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        paths.append(out)
    except Exception:
        pass

    return paths
