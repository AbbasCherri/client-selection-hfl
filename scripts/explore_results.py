"""Quick exploration of paper_full parquet results."""
import glob
import sys
from pathlib import Path

import pandas as pd

RESULTS = Path("results/paper_full")


def load_all() -> pd.DataFrame:
    """Load the consolidated sweep parquet if present, else concat per-seed files."""
    consolidated = RESULTS / "paper_sweep_rounds.parquet"
    if consolidated.exists():
        df = pd.read_parquet(consolidated)
        print(f"Loaded consolidated: {consolidated} ({len(df):,} rows)")
        return df

    paths = sorted(RESULTS.rglob("fullsim_rounds.parquet"))
    if not paths:
        sys.exit("No parquet files found under results/paper_full/")

    dfs = []
    for p in paths:
        chunk = pd.read_parquet(p)
        # Inject N and seed from path if not already columns
        parts = p.parts
        for part in parts:
            if part.startswith("N") and part[1:].isdigit() and "N" not in chunk.columns:
                chunk.insert(0, "N", int(part[1:]))
            if part.startswith("seed") and part[4:].isdigit() and "seed" not in chunk.columns:
                chunk.insert(1, "seed", int(part[4:]))
        dfs.append(chunk)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(paths)} per-seed files → {len(df):,} rows")
    return df


def summarize(df: pd.DataFrame) -> None:
    print("\n── Columns ──────────────────────────────────────")
    print(df.dtypes.to_string())

    print("\n── Shape / nulls ────────────────────────────────")
    print(f"  rows: {len(df):,}   cols: {df.shape[1]}")
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(null_counts[null_counts > 0].to_string())
    else:
        print("  no nulls")

    group_cols = [c for c in ("N", "method", "seed") if c in df.columns]
    if not group_cols:
        print("\n── Raw describe ─────────────────────────────────")
        print(df.describe().to_string())
        return

    print(f"\n── Rounds per group ({' × '.join(group_cols)}) ──────────────")
    print(df.groupby(group_cols)["round"].max().unstack() if "method" in group_cols else
          df.groupby(group_cols)["round"].max().to_string())

    metric_cols = [c for c in ("accuracy", "macro_f1", "loss") if c in df.columns]
    if not metric_cols:
        return

    print("\n── Final-round metrics (mean ± std across seeds) ──")
    last = df.loc[df.groupby(group_cols)["round"].idxmax()]
    agg = (
        last.groupby([c for c in group_cols if c != "seed"])[metric_cols]
        .agg(["mean", "std"])
    )
    pd.options.display.float_format = "{:.4f}".format
    print(agg.to_string())

    if "method" in df.columns and "accuracy" in df.columns:
        print("\n── Method ranking by final accuracy ─────────────")
        rank = (
            last.groupby("method")["accuracy"].mean()
            .sort_values(ascending=False)
            .rename("mean_acc")
        )
        print(rank.to_string())


def convergence_summary(df: pd.DataFrame) -> None:
    if "round" not in df.columns or "accuracy" not in df.columns or "method" not in df.columns:
        return
    print("\n── Accuracy by round (method mean across N & seeds) ─")
    tbl = df.groupby(["method", "round"])["accuracy"].mean().unstack("round")
    # show first, middle, last 3 rounds
    rounds = tbl.columns.tolist()
    cols = sorted(set(rounds[:3] + rounds[len(rounds)//2 - 1:len(rounds)//2 + 2] + rounds[-3:]))
    print(tbl[cols].to_string())


if __name__ == "__main__":
    df = load_all()
    summarize(df)
    convergence_summary(df)
