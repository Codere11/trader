#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# Reuse feature computation from crossref utilities
import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))
from binance_adapter.crossref_oracle_vs_agent_patterns import compute_feature_frame

FEATURES = [
    "ret_1m_pct",
    "mom_3m_pct",
    "mom_5m_pct",
    "vol_std_5m",
    "range_5m",
    "range_norm_5m",
    "slope_ols_5m",
    "rsi_14",
    "macd",
    "macd_hist",
    "vwap_dev_5m",
    "last3_same_sign",
]


def load_market(csv_path: Path, start: str | None, end: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["ts_min"] = pd.to_datetime(df["timestamp"]).dt.floor("min")
    df = (
        df.groupby("ts_min")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )
    if start:
        df = df[df["ts_min"] >= pd.to_datetime(start)]
    if end:
        df = df[df["ts_min"] <= pd.to_datetime(end)]
    df = df.sort_values("ts_min").reset_index(drop=True)
    return df


def map_ts_to_index(ts_sorted: np.ndarray, query: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(ts_sorted, query)
    ok = (pos < ts_sorted.size) & (ts_sorted[pos] == query)
    return np.where(ok, pos, -1).astype(np.int64)


def build_training_tables(trades: pd.DataFrame, feat: pd.DataFrame, win_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts_sorted = feat["ts_min"].to_numpy(dtype="datetime64[ns]")

    entry_idx = map_ts_to_index(ts_sorted, trades["entry_ts"].to_numpy(dtype="datetime64[ns]"))
    exit_idx = map_ts_to_index(ts_sorted, trades["exit_ts"].to_numpy(dtype="datetime64[ns]"))

    ok = (entry_idx >= 0) & (exit_idx >= 0)
    trades = trades.loc[ok].reset_index(drop=True)
    entry_idx = entry_idx[ok]
    exit_idx = exit_idx[ok]

    labels = (trades["net_return_flat_pct"] > win_threshold).astype(int)

    # Entry snapshot
    entry_feat = feat.iloc[entry_idx].reset_index(drop=True)
    entry = pd.concat([trades[["entry_ts", "exit_ts", "duration_min", "net_return_flat_pct"]].reset_index(drop=True), entry_feat[FEATURES]], axis=1)
    entry["label_win"] = labels
    entry["entry_date"] = pd.to_datetime(entry["entry_ts"]).dt.date

    # Exit snapshot (state at exit)
    exit_feat = feat.iloc[exit_idx].reset_index(drop=True)
    exit_df = pd.concat([trades[["entry_ts", "exit_ts", "duration_min", "net_return_flat_pct"]].reset_index(drop=True), exit_feat[FEATURES]], axis=1)
    exit_df["elapsed_min"] = trades["duration_min"].to_numpy()
    exit_df["label_win"] = labels
    exit_df["exit_date"] = pd.to_datetime(exit_df["exit_ts"]).dt.date

    return entry, exit_df


def read_trades(paths: Sequence[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        frames.append(pd.read_parquet(p))
    if not frames:
        raise SystemExit("No trades parquet files found")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build entry/exit training tables from rule trades")
    ap.add_argument("--trades-parquet", required=True, nargs="+", help="One or more parquet files (globs ok)")
    ap.add_argument("--market-csv", required=True, help="Aggregated 1m market CSV with timestamp, open, high, low, close, volume")
    ap.add_argument("--start", default=None, help="Optional start date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="Optional end date (YYYY-MM-DD)")
    ap.add_argument("--win-threshold", type=float, default=0.0, help="Return threshold (pct) to label a win")
    ap.add_argument("--out-dir", default="data/ranker_training", help="Output directory for training parquets")
    ap.add_argument("--stem", default="rule_best", help="Filename stem for outputs")
    args = ap.parse_args()

    trade_paths: list[str] = []
    for pat in args.trades_parquet:
        trade_paths.extend(glob.glob(pat))
    if not trade_paths:
        raise SystemExit(f"No parquet files matched: {args.trades_parquet}")

    trades = read_trades(trade_paths)
    market = load_market(Path(args.market_csv), args.start, args.end)
    feat = compute_feature_frame(market)

    entry, exit_df = build_training_tables(trades, feat, win_threshold=args.win_threshold)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    entry_path = out_dir / f"{args.stem}__entry_train.parquet"
    exit_path = out_dir / f"{args.stem}__exit_train.parquet"
    entry.to_parquet(entry_path, index=False)
    exit_df.to_parquet(exit_path, index=False)

    print(f"Wrote entry training: {entry_path} (n={len(entry):,})")
    print(f"Wrote exit training:  {exit_path} (n={len(exit_df):,})")


if __name__ == "__main__":
    main()