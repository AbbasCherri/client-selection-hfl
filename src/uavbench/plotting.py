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


def _last_round(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Row-order-independent 'final FL round' selection: max `round` per group.

    Using ``.groupby(...).last()`` silently depends on on-disk row order; this
    instead picks the row with the maximum ``round`` value explicitly.
    """
    idx = df.groupby(group_cols)["round"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def summarize(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std and 95% CI of key metrics per (scenario, method)."""
    metrics = [
        "final_fitness",
        "coverage_pct",
        "f_cover_norm",
        "movement_joules",
        "l_imb",
        "wall_time_s",
        "eval_count",
    ]
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
        # Std/CI are computed on the *raw* (un-filled) values: ffill-ing first
        # would carry a converged seed's plateau value into both the variance
        # estimate and the sample count, artificially shrinking the CI band
        # right when fewer seeds are still actively contributing new data.
        n = pivot.count(axis=1).clip(lower=1)
        ci = 1.96 * pivot.std(axis=1) / np.sqrt(n)
        filled = pivot.ffill()  # carry final value forward, mean line only
        mean = filled.mean(axis=1)
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
            ax.plot(
                sub["round"], sub[metric], label=method, linewidth=1.8, marker="o", markersize=3
            )
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


def plot_confusion_matrix(results_dir: Path) -> list[Path]:
    """Final-round confusion-matrix heatmap per method from confusion.parquet.

    Reads the long-form (method, round, true_label, pred_label, count) table
    written by run_tier2 / run_full_hfl / run_selection_isolation and renders
    one row-normalized 4x4 heatmap per method for its last recorded round —
    the per-class evidence that rare classes are learned, not ignored.
    """
    path = results_dir / "confusion.parquet"
    if not path.exists() and not path.with_suffix(".csv").exists():
        return []
    df = _read_table(path)
    if df.empty:
        return []

    class_names = ["survived", "collapsed", "obstructed", "missing"]
    paths: list[Path] = []
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method]
        final = sub[sub["round"] == sub["round"].max()]
        cm = (
            final.pivot(index="true_label", columns="pred_label", values="count")
            .reindex(index=class_names, columns=class_names)
            .fillna(0.0)
            .to_numpy()
        )
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(4), class_names, rotation=30, ha="right")
        ax.set_yticks(range(4), class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion (final round): {method}")
        for t in range(4):
            for p in range(4):
                ax.text(
                    p,
                    t,
                    f"{int(cm[t, p])}",
                    ha="center",
                    va="center",
                    color="white" if cm_norm[t, p] > 0.5 else "black",
                    fontsize=9,
                )
        fig.colorbar(im, ax=ax, label="Row fraction")
        fig.tight_layout()
        out = results_dir / f"confusion_{method}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        paths.append(out)
    return paths


def plot_paper_sim(results_dir: Path) -> list[Path]:
    """Generate the paper's §V figures from paper_sweep_rounds.parquet.

    Figures produced
    ----------------
    paper_accuracy_vs_rounds_N{N}.png   — per-N accuracy curves (mean ± 95% CI)
    paper_f1_vs_rounds_N{N}.png         — per-N macro-F1 curves (mean ± 95% CI)
    paper_scalability.png               — final accuracy vs N (all methods)
    paper_comm_energy.png               — comm cost + energy bar charts at N=200
    paper_ablation_table.png            — final accuracy/F1 heat-table at N=200
    """
    p = results_dir if isinstance(results_dir, Path) else Path(results_dir)
    df = _read_table(p / "paper_sweep_rounds.parquet")
    paths: list[Path] = []

    METHOD_ORDER = [
        "proposed_hfl",
        "flat_fl",
        "centralized",
        "hfl_no_selection",
        "hfl_static",
        "hfl_no_reputation",
        "fedcs",
        "rep_cap",
        "fair_mab",
    ]
    METHOD_LABELS = {
        "proposed_hfl": "Proposed HFL",
        "flat_fl": "Flat FL",
        "centralized": "Centralized",
        "hfl_no_selection": "No Selection",
        "hfl_static": "Static UAVs",
        "hfl_no_reputation": "No Reputation",
        # Literature baselines (Algorithms B1-B3)
        "fedcs": "FedCS (Nishio & Yonetani '19)",
        "rep_cap": "Rep-Capability (Zhao et al. '24)",
        "fair_mab": "Fairness MAB (Zhu et al. '24)",
    }
    COLORS = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#17becf",
    ]
    METHOD_COLOR = {m: COLORS[i % len(COLORS)] for i, m in enumerate(METHOD_ORDER)}

    N_values = sorted(df["N"].unique())

    # ── Figure 1 & 2: Accuracy and Macro-F1 vs rounds per N ─────────────
    for metric, ylabel, suffix in [
        ("accuracy", "Accuracy", "accuracy"),
        ("macro_f1", "Macro F1", "f1"),
    ]:
        if metric not in df.columns:
            continue
        for N in N_values:
            sub = df[df["N"] == N]
            fig, ax = plt.subplots(figsize=(7, 4.5))
            methods_present = [m for m in METHOD_ORDER if m in sub["method"].unique()]
            for method in methods_present:
                m_df = sub[sub["method"] == method]
                pivot = m_df.pivot_table(index="round", columns="seed", values=metric)
                pivot = pivot.ffill()
                mean = pivot.mean(axis=1)
                n_s = pivot.count(axis=1).clip(lower=1)
                ci = 1.96 * pivot.std(axis=1, ddof=1).fillna(0) / np.sqrt(n_s)
                color = METHOD_COLOR.get(method, None)
                label = METHOD_LABELS.get(method, method)
                ax.plot(mean.index, mean.values, label=label, linewidth=1.8, color=color)
                ax.fill_between(
                    mean.index, (mean - ci).values, (mean + ci).values, alpha=0.15, color=color
                )
            ax.set_xlabel("FL Round")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} vs Round  (N={N})")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            out = p / f"paper_{suffix}_vs_rounds_N{N}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            paths.append(out)

    # ── Figure 3: Scalability — final accuracy vs N ──────────────────────
    if "accuracy" in df.columns:
        last = (
            _last_round(df, ["N", "method", "seed"])
            .groupby(["N", "method"])["accuracy"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        last["ci95"] = 1.96 * last["std"] / np.sqrt(last["count"].clip(lower=1))

        fig, ax = plt.subplots(figsize=(8, 5))
        for method in [m for m in METHOD_ORDER if m in last["method"].unique()]:
            s = last[last["method"] == method].sort_values("N")
            color = METHOD_COLOR.get(method, None)
            label = METHOD_LABELS.get(method, method)
            ax.errorbar(
                s["N"],
                s["mean"],
                yerr=s["ci95"],
                label=label,
                marker="o",
                linewidth=1.8,
                color=color,
                capsize=4,
                markersize=5,
            )
        ax.set_xlabel("Number of Clients (N)")
        ax.set_ylabel("Final Accuracy (mean ± 95% CI)")
        ax.set_title("Scalability: Final Accuracy vs N")
        ax.legend(frameon=False, fontsize=8)
        ax.set_xticks(N_values)
        fig.tight_layout()
        out = p / "paper_scalability.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        paths.append(out)

    # ── Figure 4: Communication and energy bar charts (N_mid or N=200) ───
    N_ref = 200 if 200 in N_values else N_values[len(N_values) // 2]
    sub_ref = df[df["N"] == N_ref]
    if not sub_ref.empty:
        last_ref = _last_round(sub_ref, ["method", "seed"])
        agg_ref = last_ref.groupby("method")[["comm_mb_round", "cumulative_energy_j"]].mean()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, col, ylabel, scale, unit in [
            (axes[0], "comm_mb_round", "Comm. Cost / Round (MB)", 1.0, ""),
            (axes[1], "cumulative_energy_j", "Cumulative Energy (kJ)", 1e-3, ""),
        ]:
            if col not in agg_ref.columns:
                continue
            methods_bar = [m for m in METHOD_ORDER if m in agg_ref.index]
            vals = [agg_ref.loc[m, col] * scale if m in agg_ref.index else 0.0 for m in methods_bar]
            colors = [METHOD_COLOR.get(m, "grey") for m in methods_bar]
            labels = [METHOD_LABELS.get(m, m) for m in methods_bar]
            ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_ylabel(ylabel)
            ax.set_title(f"N={N_ref}")
            ax.tick_params(axis="x", rotation=30)
        fig.suptitle("Communication Cost & Energy Comparison", fontsize=11)
        fig.tight_layout()
        out = p / "paper_comm_energy.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        paths.append(out)

    # ── Figure 5: Ablation heat-table (accuracy + F1 at N_ref) ──────────
    if not sub_ref.empty and "accuracy" in df.columns and "macro_f1" in df.columns:
        abl = (
            _last_round(sub_ref, ["method", "seed"])
            .groupby("method")[["accuracy", "macro_f1"]]
            .mean()
            .reindex([m for m in METHOD_ORDER if m in sub_ref["method"].unique()])
        )
        fig, ax = plt.subplots(figsize=(6, 3.5))
        data = abl.values
        im = ax.imshow(data.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(abl.index)))
        ax.set_xticklabels(
            [METHOD_LABELS.get(m, m) for m in abl.index], rotation=30, ha="right", fontsize=8
        )
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Accuracy", "Macro F1"], fontsize=9)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    i, j, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=9, color="black"
                )
        plt.colorbar(im, ax=ax, label="Score")
        ax.set_title(f"Ablation: Final Accuracy & F1  (N={N_ref})", fontsize=10)
        fig.tight_layout()
        out = p / "paper_ablation_table.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        paths.append(out)

    return paths


def plot_sweep(results_dir: Path) -> list[Path]:
    """Generate scalability sweep figures: accuracy/macro-F1 vs N, per method."""
    df = _read_table(results_dir / "sweep_rounds.parquet")
    # Use the final FL round per (N, method) as the headline value.
    final = _last_round(df, ["N", "method"])
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
            ax.plot(
                sub["N"],
                sub[metric],
                label=method,
                linewidth=1.8,
                marker="o",
                markersize=5,
                linestyle=style,
            )
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
                ax.text(
                    j,
                    i,
                    f"{pivot.values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
        fig.tight_layout()
        out = results_dir / "sweep_heatmap_accuracy.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        paths.append(out)
    except Exception:
        pass

    return paths


def plot_selection_sim(results_dir: Path) -> list[Path]:
    """Figures for the selection-isolation benchmark (selection_sweep_rounds.parquet).

    Figures produced
    ----------------
    selection_accuracy_vs_rounds_N{N}.png — per-N accuracy curves (mean ± 95% CI)
    selection_f1_vs_rounds_N{N}.png       — per-N macro-F1 curves (mean ± 95% CI)
    selection_fairness_vs_rounds_N{N}.png — per-N Jain fairness index curves
    selection_scalability.png             — final accuracy vs N (all modes)
    selection_summary_table.png           — accuracy/F1/fairness heat-table at N_ref
    """
    p = results_dir if isinstance(results_dir, Path) else Path(results_dir)
    df = _read_table(p / "selection_sweep_rounds.parquet")
    paths: list[Path] = []

    MODE_ORDER = ["ucb", "random", "fedcs", "rep_cap", "fair_mab", "all"]
    MODE_LABELS = {
        "ucb": "Proposed (UCB)",
        "random": "Random",
        "fedcs": "FedCS (Nishio & Yonetani '19)",
        "rep_cap": "Rep-Capability (Zhao et al. '24)",
        "fair_mab": "Fairness MAB (Zhu et al. '24)",
        "all": "All Covered",
    }
    COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    MODE_COLOR = {m: COLORS[i % len(COLORS)] for i, m in enumerate(MODE_ORDER)}

    if "seed" not in df.columns:
        df = df.assign(seed=0)
    N_values = sorted(df["N"].unique())

    # ── Per-N curves: accuracy, macro-F1, Jain fairness ──────────────────
    for metric, ylabel, suffix in [
        ("accuracy", "Accuracy", "accuracy"),
        ("macro_f1", "Macro F1", "f1"),
        ("jain_fairness", "Jain Fairness Index", "fairness"),
    ]:
        if metric not in df.columns:
            continue
        for N in N_values:
            sub = df[df["N"] == N]
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for mode in [m for m in MODE_ORDER if m in sub["method"].unique()]:
                m_df = sub[sub["method"] == mode]
                pivot = m_df.pivot_table(index="round", columns="seed", values=metric)
                pivot = pivot.ffill()
                mean = pivot.mean(axis=1)
                n_s = pivot.count(axis=1).clip(lower=1)
                ci = 1.96 * pivot.std(axis=1, ddof=1).fillna(0) / np.sqrt(n_s)
                color = MODE_COLOR.get(mode, None)
                ax.plot(
                    mean.index,
                    mean.values,
                    label=MODE_LABELS.get(mode, mode),
                    linewidth=1.8,
                    color=color,
                )
                ax.fill_between(
                    mean.index, (mean - ci).values, (mean + ci).values, alpha=0.15, color=color
                )
            ax.set_xlabel("FL Round")
            ax.set_ylabel(ylabel)
            K = int(sub["K_uav"].iloc[0]) if "K_uav" in sub.columns and len(sub) else 0
            ax.set_title(f"{ylabel} vs Round  (N={N}, K={K} static UAVs)")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            out = p / f"selection_{suffix}_vs_rounds_N{N}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            paths.append(out)

    # ── Scalability: final accuracy vs N ─────────────────────────────────
    if "accuracy" in df.columns:
        last = (
            _last_round(df, ["N", "method", "seed"])
            .groupby(["N", "method"])["accuracy"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        last["ci95"] = 1.96 * last["std"] / np.sqrt(last["count"].clip(lower=1))
        fig, ax = plt.subplots(figsize=(8, 5))
        for mode in [m for m in MODE_ORDER if m in last["method"].unique()]:
            s = last[last["method"] == mode].sort_values("N")
            ax.errorbar(
                s["N"],
                s["mean"],
                yerr=s["ci95"],
                label=MODE_LABELS.get(mode, mode),
                marker="o",
                linewidth=1.8,
                color=MODE_COLOR.get(mode, None),
                capsize=4,
                markersize=5,
            )
        ax.set_xlabel("Number of Clients (N)")
        ax.set_ylabel("Final Accuracy (mean ± 95% CI)")
        ax.set_title("Selection Isolation: Final Accuracy vs N (static UAVs)")
        ax.legend(frameon=False, fontsize=8)
        ax.set_xticks(N_values)
        fig.tight_layout()
        out = p / "selection_scalability.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        paths.append(out)

    # ── Summary heat-table at N_ref ──────────────────────────────────────
    N_ref = max(N_values)
    sub_ref = df[df["N"] == N_ref]
    metrics_cols = [c for c in ("accuracy", "macro_f1", "jain_fairness") if c in df.columns]
    if not sub_ref.empty and metrics_cols:
        summary = (
            _last_round(sub_ref, ["method", "seed"])
            .groupby("method")[metrics_cols]
            .mean()
            .reindex([m for m in MODE_ORDER if m in sub_ref["method"].unique()])
        )
        fig, ax = plt.subplots(figsize=(7, 3.5))
        data = summary.values
        im = ax.imshow(data.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(summary.index)))
        ax.set_xticklabels(
            [MODE_LABELS.get(m, m) for m in summary.index], rotation=30, ha="right", fontsize=8
        )
        ax.set_yticks(range(len(metrics_cols)))
        ax.set_yticklabels(
            ["Accuracy", "Macro F1", "Jain Fairness"][: len(metrics_cols)], fontsize=9
        )
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    i, j, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=9, color="black"
                )
        plt.colorbar(im, ax=ax, label="Score")
        ax.set_title(f"Selection Rules: Final Metrics  (N={N_ref})", fontsize=10)
        fig.tight_layout()
        out = p / "selection_summary_table.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        paths.append(out)

    return paths
