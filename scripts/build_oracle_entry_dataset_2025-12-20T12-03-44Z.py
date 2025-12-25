#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import compute_feature_frame

FEATURES_LEFT = [
    "ret_1m_pct",
    "mom_3m_pct",
    "mom_5m_pct",
    "vol_std_5m",
    "range_5m",
    "range_norm_5m",
    "macd",
    "vwap_dev_5m",
]

_MKT_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def minute_bars(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        csv_path,
        parse_dates=["timestamp"],
        usecols=_MKT_COLS,
        dtype={"open": "float64", "high": "float64", "low": "float64", "close": "float64", "volume": "float64"},
    )

    # Ensure chronological ordering so open/close are computed correctly.
    if not raw["timestamp"].is_monotonic_increasing:
        raw = raw.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    raw["ts_min"] = raw["timestamp"].dt.floor("min")

    g = (
        raw.groupby("ts_min")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .rename(columns={"ts_min": "timestamp"})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return g


def _compute_oracle_forward_slow(mkt: pd.DataFrame, hold_min: int, fee: float) -> pd.DataFrame:
    px = mkt["close"].to_numpy(dtype=np.float64, copy=False)
    n = int(px.size)
    best_ret = np.full(n, np.nan, dtype=np.float64)
    best_exit_idx = np.full(n, -1, dtype=np.int64)

    for i in range(n):
        j2 = min(n, i + int(hold_min) + 1)
        if i + 1 >= j2:
            continue
        entry_cost = px[i] * (1.0 + fee)
        exits = px[i + 1 : j2] * (1.0 - fee)
        rets = exits / entry_cost - 1.0
        k = int(np.argmax(rets))
        best_ret[i] = float(rets[k])
        best_exit_idx[i] = int(i + 1 + k)

    ts_arr = mkt["timestamp"].to_numpy(dtype="datetime64[ns]")
    best_exit_ts = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    mask = best_exit_idx >= 0
    best_exit_ts[mask] = ts_arr[best_exit_idx[mask]]

    horizon = int(hold_min)
    out = mkt[["timestamp"]].copy()
    out[f"oracle_best_ret_pct_{horizon}m"] = best_ret * 100.0
    out[f"oracle_best_exit_ts_{horizon}m"] = best_exit_ts
    out[f"oracle_has_profit_{horizon}m"] = out[f"oracle_best_ret_pct_{horizon}m"] > 0.0
    return out


def compute_oracle_forward(mkt: pd.DataFrame, hold_min: int, fee: float) -> pd.DataFrame:
    """Fast oracle label computation.

    For each minute i, considers exits at i+1..i+hold_min and chooses the exit with maximum net return
    (net of per-side fee).

    Complexity: O(n * hold_min) but fully vectorized over n with only hold_min small loops.
    Memory: O(n).
    """

    hold_min = int(hold_min)
    fee = float(fee)
    if hold_min < 1:
        raise ValueError("hold_min must be >= 1")
    if fee < 0:
        raise ValueError("fee must be >= 0")

    px = mkt["close"].to_numpy(dtype=np.float64, copy=False)
    ts_arr = mkt["timestamp"].to_numpy(dtype="datetime64[ns]")
    n = int(px.size)

    fee_in = 1.0 + fee
    fee_out = 1.0 - fee

    entry_cost = px * fee_in
    best_ret = np.full(n, -np.inf, dtype=np.float64)
    best_exit_idx = np.full(n, -1, dtype=np.int64)

    for k in range(1, hold_min + 1):
        if n <= k:
            break
        # returns for entries 0..n-k-1, exiting exactly at i+k
        ret_k = (px[k:] * fee_out) / entry_cost[:-k] - 1.0
        better = ret_k > best_ret[:-k]
        if not np.any(better):
            continue
        idx = np.flatnonzero(better)
        best_ret[idx] = ret_k[idx]
        best_exit_idx[idx] = idx + k

    valid = best_exit_idx >= 0
    best_ret_out = np.full(n, np.nan, dtype=np.float64)
    best_ret_out[valid] = best_ret[valid]

    best_exit_ts = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    best_exit_ts[valid] = ts_arr[best_exit_idx[valid]]

    horizon = int(hold_min)
    out = mkt[["timestamp"]].copy()
    out[f"oracle_best_ret_pct_{horizon}m"] = best_ret_out * 100.0
    out[f"oracle_best_exit_ts_{horizon}m"] = best_exit_ts
    out[f"oracle_has_profit_{horizon}m"] = out[f"oracle_best_ret_pct_{horizon}m"] > 0.0
    return out


def _assert_datetime64_equal(a: np.ndarray, b: np.ndarray) -> None:
    if a.dtype != b.dtype:
        raise AssertionError(f"dtype mismatch: {a.dtype} vs {b.dtype}")
    same = (a == b) | (np.isnat(a) & np.isnat(b))
    if not bool(np.all(same)):
        bad = int(np.flatnonzero(~same)[0])
        raise AssertionError(f"datetime mismatch at idx={bad}: {a[bad]} vs {b[bad]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build training dataset for entry model assuming oracle exits within next 5 minutes.")
    ap.add_argument("--market-csv", default="data/btc_profitability_analysis_filtered.csv")
    ap.add_argument("--fee", type=float, default=0.001, help="Per-side fee (e.g., 0.001 = 0.1%)")
    ap.add_argument("--hold-min", type=int, default=5, help="Oracle exit lookahead in minutes (1..hold-min)")
    ap.add_argument("--out-dir", default="data/ranker_training")
    ap.add_argument("--out-stem", default="oracle_entry_5m")
    ap.add_argument("--self-check", action="store_true", help="Validate fast oracle labels vs slow reference on a small slice")
    ap.add_argument("--self-check-n", type=int, default=5000, help="Number of rows to use for --self-check")
    args = ap.parse_args()

    mkt = minute_bars(Path(args.market_csv))

    # Compute per-minute features (remaining subset)
    feats = compute_feature_frame(mkt.rename(columns={"timestamp": "ts_min"}))
    feats = feats[["ts_min"] + FEATURES_LEFT].copy()

    # Oracle forward labels for hold window (net of fees)
    oracle = compute_oracle_forward(mkt, hold_min=int(args.hold_min), fee=float(args.fee))

    if args.self_check:
        n_check = max(0, int(args.self_check_n))
        if n_check:
            mkt_s = mkt.iloc[:n_check].reset_index(drop=True)
            fast = compute_oracle_forward(mkt_s, hold_min=int(args.hold_min), fee=float(args.fee))
            slow = _compute_oracle_forward_slow(mkt_s, hold_min=int(args.hold_min), fee=float(args.fee))
            h = int(args.hold_min)
            col_ret = f"oracle_best_ret_pct_{h}m"
            col_ts = f"oracle_best_exit_ts_{h}m"
            col_pos = f"oracle_has_profit_{h}m"
            np.testing.assert_allclose(
                fast[col_ret].to_numpy(dtype=np.float64),
                slow[col_ret].to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )
            _assert_datetime64_equal(
                fast[col_ts].to_numpy(dtype="datetime64[ns]"),
                slow[col_ts].to_numpy(dtype="datetime64[ns]"),
            )
            if not bool(np.array_equal(fast[col_pos].to_numpy(dtype=bool), slow[col_pos].to_numpy(dtype=bool))):
                raise AssertionError("oracle_has_profit mismatch")
            print(f"Self-check OK on n={n_check} rows")

    # Join features+labels (fast path: same market-derived timestamps => concat)
    h = int(args.hold_min)
    label_cols = [
        f"oracle_best_ret_pct_{h}m",
        f"oracle_best_exit_ts_{h}m",
        f"oracle_has_profit_{h}m",
    ]

    ts_feat = feats["ts_min"].to_numpy(dtype="datetime64[ns]")
    ts_orc = oracle["timestamp"].to_numpy(dtype="datetime64[ns]")
    if len(feats) == len(oracle) and bool(np.all((ts_feat == ts_orc) | (np.isnat(ts_feat) & np.isnat(ts_orc)))):
        df = pd.concat(
            [
                feats.rename(columns={"ts_min": "timestamp"}).reset_index(drop=True),
                oracle[label_cols].reset_index(drop=True),
            ],
            axis=1,
        )
    else:
        df = feats.merge(oracle, left_on="ts_min", right_on="timestamp", how="inner")
        df = df.drop(columns=["timestamp"]).rename(columns={"ts_min": "timestamp"})

    # Save artifacts
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    out_parquet = out_dir / f"{args.out_stem}_{ts}.parquet"
    out_csv = out_dir / f"{args.out_stem}_{ts}.csv"

    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    # Quick summary
    pos = float(df[f"oracle_has_profit_{h}m"].mean() * 100.0)
    print("Rows:", len(df))
    print(f"Positive (oracle_has_profit_{h}m) %:", round(pos, 2))
    print("Wrote:", out_parquet)
    print("Also CSV:", out_csv)


if __name__ == "__main__":
    main()
