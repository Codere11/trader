#!/usr/bin/env python3
"""
Analyze patterns of why trades succeed and fail by joining executed trades
with pre-entry features and simple context.

Outputs:
- data/analysis/trade_pattern_summary_<ts>.csv
- data/analysis/trade_pattern_by_hour_<ts>.csv
- data/analysis/trade_pattern_by_feature_bucket_<ts>.csv
- Prints a concise text summary to stdout.
"""
from __future__ import annotations

import argparse
from datetime import timezone, datetime
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import compute_feature_frame  # type: ignore

ENTRY_FEATURES = [
    "ret_1m_pct","mom_3m_pct","mom_5m_pct","vol_std_5m","range_5m","range_norm_5m",
    "slope_ols_5m","rsi_14","macd","macd_hist","vwap_dev_5m","last3_same_sign",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades-csv", required=True)
    ap.add_argument("--market-csv", default="data/btc_profitability_analysis_filtered.csv")
    ap.add_argument("--min-profit-thr", type=float, default=0.0, help="Trade success defined as trade_return_pct > this threshold")
    return ap.parse_args()


def map_ts_to_index(ts_sorted: np.ndarray, query: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(ts_sorted, query)
    ok = (pos < ts_sorted.size) & (ts_sorted[pos] == query)
    return np.where(ok, pos, -1).astype(np.int64)


def main() -> None:
    args = parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    trades = pd.read_csv(args.trades_csv, parse_dates=["entry_time","exit_time"]) if Path(args.trades_csv).exists() else None
    if trades is None or trades.empty:
        raise SystemExit("No trades to analyze.")

    # Success flag
    trades["success"] = trades["trade_return_pct"] > float(args.min_profit_thr)

    # Load market and compute features to attach entry context
    mkt_raw = pd.read_csv(args.market_csv, parse_dates=["timestamp"]) 
    mkt_raw["ts_min"] = pd.to_datetime(mkt_raw["timestamp"]).dt.floor("min")
    mkt = (
        mkt_raw.groupby("ts_min")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .sort_values("ts_min")
        .reset_index(drop=True)
    )
    mkt_feat = compute_feature_frame(mkt)

    # Join by floored minute timestamp
    trades["ts_min"] = trades["entry_time"].dt.floor("min")
    feat_ts = mkt_feat["ts_min"].to_numpy(dtype="datetime64[ns]")
    q = trades["ts_min"].to_numpy(dtype="datetime64[ns]")
    idx = map_ts_to_index(feat_ts, q)
    ok = idx >= 0
    joined = trades.loc[ok].reset_index(drop=True).copy()
    feat_cols = mkt_feat.iloc[idx[ok]][ENTRY_FEATURES].reset_index(drop=True)
    joined = pd.concat([joined, feat_cols], axis=1)
    if joined.empty:
        raise SystemExit("Failed to align trades with features; timestamps may not match market file.")

    # Bucketing helpers
    def bucket(series: pd.Series, bins: list[float], labels: list[str] | None = None) -> pd.Series:
        b = pd.cut(series, bins=bins, include_lowest=True, labels=labels)
        return b.astype(str)

    # Basic aggregates
    summary = {
        "n_trades": int(len(joined)),
        "win_rate": float(joined["success"].mean() if len(joined) else 0.0),
        "median_trade_pct": float(joined["trade_return_pct"].median() if len(joined) else 0.0),
        "mean_trade_pct": float(joined["trade_return_pct"].mean() if len(joined) else 0.0),
        "liq_rate": float(joined["liquidated"].mean() if "liquidated" in joined else 0.0),
        "median_duration_min": float(joined["duration_min"].median() if "duration_min" in joined else 0.0),
    }

    # Hour-of-day performance
    joined["hour"] = joined["entry_time"].dt.hour
    by_hour = (
        joined.groupby("hour").agg(
            n=("success","size"),
            win_rate=("success","mean"),
            median_pct=("trade_return_pct","median"),
            mean_pct=("trade_return_pct","mean"),
            liq_rate=("liquidated","mean") if "liquidated" in joined else ("success","size"),
        ).reset_index()
    )

    # Feature buckets and performance
    buckets = {
        "rsi_14": [-np.inf, 30, 40, 50, 60, 70, np.inf],
        "mom_5m_pct": [-np.inf, -0.5, 0, 0.5, 1.0, np.inf],
        "vol_std_5m": [0, 5, 10, 20, 40, np.inf],
        "range_norm_5m": [0, 0.001, 0.002, 0.004, 0.008, np.inf],
        "vwap_dev_5m": [-np.inf, -0.5, -0.2, 0.0, 0.2, 0.5, np.inf],
    }
    rows = []
    # Drop rows with NaNs in any key feature to avoid NaN buckets bias
    clean = joined.dropna(subset=list(buckets.keys()), how="any").copy()
    for feat, bins in buckets.items():
        if feat not in clean:
            continue
        lab = [f"({bins[i]},{bins[i+1]}]" for i in range(len(bins)-1)]
        clean[f"bucket__{feat}"] = bucket(clean[feat], bins=bins, labels=lab)
        g = (
            clean.groupby(f"bucket__{feat}")
            .agg(n=("success","size"), win_rate=("success","mean"), median_pct=("trade_return_pct","median"), mean_pct=("trade_return_pct","mean"))
            .reset_index()
        )
        g = g[g["n"] >= 30]  # only stable buckets
        g.insert(0, "feature", feat)
        rows.append(g)
    by_feat_bucket = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # Binary flags
    if "last3_same_sign" in joined:
        by_last3 = (
            joined.groupby("last3_same_sign").agg(n=("success","size"), win_rate=("success","mean"), median_pct=("trade_return_pct","median"), mean_pct=("trade_return_pct","mean")).reset_index()
        )
    else:
        by_last3 = pd.DataFrame()

    # Save
    out_dir = REPO_ROOT / "data" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"trade_pattern_summary_{ts}.csv").write_text(pd.DataFrame([summary]).to_csv(index=False))
    by_hour.to_csv(out_dir / f"trade_pattern_by_hour_{ts}.csv", index=False)
    if len(by_feat_bucket):
        by_feat_bucket.to_csv(out_dir / f"trade_pattern_by_feature_bucket_{ts}.csv", index=False)
    if len(by_last3):
        by_last3.to_csv(out_dir / f"trade_pattern_by_last3_{ts}.csv", index=False)

    # Human-readable stdout summary
    print("Summary:", summary)
    top_hours = by_hour.sort_values("win_rate", ascending=False).head(5)
    worst_hours = by_hour.sort_values("win_rate", ascending=True).head(5)
    print("Top hours by win_rate:\n", top_hours)
    print("Worst hours by win_rate:\n", worst_hours)
    if len(by_feat_bucket):
        top = by_feat_bucket.sort_values(["win_rate","n"], ascending=[False, False]).head(10)
        bot = by_feat_bucket.sort_values(["win_rate","n"], ascending=[True, False]).head(10)
        print("Feature buckets with highest win_rate (n>=30):\n", top)
        print("Feature buckets with lowest win_rate (n>=30):\n", bot)


if __name__ == "__main__":
    main()
