#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T00:47:38Z
"""Plot detailed 5-minute pre-entry context for selected trade_rows.

Usage examples:
- Plot top 30 outliers from an entry_outliers.csv:
    python3 scripts/plot_oracle_entry_outlier_examples_*.py \
      --entry-precontext data/oracle_precontext/oracle_daily_ETH-USD_pre_entry_dyn5m_<ts>.parquet \
      --outliers-csv data/analysis_oracle_entry_outliers_<ts>/entry_outliers.csv \
      --top 30

Outputs
- A folder with one PNG per trade_row.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURE_COLS_DEFAULT = [
    "ret_1m_pct",
    "mom_3m_pct",
    "mom_5m_pct",
    "vol_std_5m",
    "range_5m",
    "range_norm_5m",
    "macd",
    "vwap_dev_5m",
]


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot detailed entry precontext for selected outlier trade_rows")
    ap.add_argument(
        "--entry-precontext",
        required=True,
        help="Entry precontext parquet (pre_entry) with trade_row + rel_min + OHLCV + features",
    )
    ap.add_argument("--outliers-csv", required=True, help="CSV containing trade_row column (e.g., entry_outliers.csv)")
    ap.add_argument("--top", type=int, default=30, help="Top N rows from outliers CSV")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: sibling of outliers csv)")

    args = ap.parse_args()

    pre_p = Path(args.entry_precontext)
    out_csv = Path(args.outliers_csv)
    if not pre_p.exists():
        raise SystemExit(f"missing: {pre_p}")
    if not out_csv.exists():
        raise SystemExit(f"missing: {out_csv}")

    outlier_df = pd.read_csv(out_csv)
    if "trade_row" not in outlier_df.columns:
        raise SystemExit("outliers CSV must contain trade_row")

    rows = outlier_df["trade_row"].astype(int).head(int(args.top)).tolist()

    # Output folder
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = out_csv.parent / f"outlier_examples_{_now_ts()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read precontext rows for selected trades only.
    df = pd.read_parquet(pre_p)
    df = df[df["trade_row"].isin(rows)].copy()
    if df.empty:
        raise SystemExit("No matching trade_rows found in precontext parquet")

    # Normalize
    df["rel_min"] = df["rel_min"].astype(int)
    for tc in ["anchor_time", "trade_entry_time", "trade_exit_time"]:
        if tc in df.columns:
            df[tc] = pd.to_datetime(df[tc], utc=True, errors="coerce")

    feature_cols = [c for c in FEATURE_COLS_DEFAULT if c in df.columns]

    for trade_row in rows:
        g = df[df["trade_row"] == int(trade_row)].sort_values("rel_min")
        if g.empty:
            continue

        # x axis is rel_min
        x = g["rel_min"].to_numpy(np.int64)

        # Price normalization by last close
        close = pd.to_numeric(g["close"], errors="coerce").to_numpy(np.float64)
        close_last = close[-1] if np.isfinite(close[-1]) and close[-1] != 0 else np.nan
        close_norm = (close / close_last - 1.0) * 100.0 if np.isfinite(close_last) else close * np.nan

        # metadata
        side = str(g["trade_side"].iloc[-1]) if "trade_side" in g.columns else "?"
        ret = float(g["trade_net_return_pct"].iloc[-1]) if "trade_net_return_pct" in g.columns else float("nan")
        et = g["trade_entry_time"].iloc[-1] if "trade_entry_time" in g.columns else None
        xt = g["trade_exit_time"].iloc[-1] if "trade_exit_time" in g.columns else None

        n_plots = 1 + len(feature_cols) + 1  # price + features + volume
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.2 * n_plots), sharex=True)

        ax0 = axes[0]
        ax0.plot(x, close_norm, marker="o")
        ax0.axhline(0.0, color="#888", linewidth=0.8, alpha=0.7)
        ax0.set_ylabel("close norm %")
        ax0.grid(True, alpha=0.2)
        ax0.set_title(f"trade_row={trade_row} side={side} net_ret={ret:.4f}%  entry={et}  exit={xt}")

        # Volume
        vol = pd.to_numeric(g.get("volume"), errors="coerce").to_numpy(np.float64)
        axv = axes[1]
        axv.plot(x, np.log1p(np.maximum(0.0, vol)), marker="o", color="#2ca02c")
        axv.set_ylabel("log1p(vol)")
        axv.grid(True, alpha=0.2)

        for i, col in enumerate(feature_cols):
            ax = axes[2 + i]
            y = pd.to_numeric(g[col], errors="coerce").to_numpy(np.float64)
            ax.plot(x, y, marker="o")
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.2)

        axes[-1].set_xlabel("rel_min (pre-entry)")
        fig.tight_layout()
        fig.savefig(out_dir / f"trade_row_{int(trade_row):07d}.png", dpi=170)
        plt.close(fig)

    print("Wrote outlier example plots to", out_dir)


if __name__ == "__main__":
    main()
