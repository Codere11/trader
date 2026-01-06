#!/usr/bin/env python3
from __future__ import annotations

"""
Build symmetric oracle labels (long and short) for multiple horizons.

Outputs a per-minute frame with, for each horizon H in --hold-mins:
- oracle_long_best_ret_pct_{H}m
- oracle_long_best_exit_ts_{H}m
- oracle_long_best_delay_min_{H}m
- oracle_long_has_profit_{H}m
- oracle_short_best_ret_pct_{H}m
- oracle_short_best_exit_ts_{H}m
- oracle_short_best_delay_min_{H}m
- oracle_short_has_profit_{H}m

Also writes CSV and Parquet artifacts with a UTC timestamp in the filename.
"""

import argparse
import datetime as dt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

_MKT_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def minute_bars(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        csv_path,
        parse_dates=["timestamp"],
        usecols=_MKT_COLS,
        dtype={"open": "float64", "high": "float64", "low": "float64", "close": "float64", "volume": "float64"},
    )
    if not raw["timestamp"].is_monotonic_increasing:
        raw = raw.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    raw["ts_min"] = pd.to_datetime(raw["timestamp"]).dt.floor("min")
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


def _oracle_forward_long(mkt: pd.DataFrame, hold_min: int, fee: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized best forward return for LONG entries within [1, hold_min] minutes.
    Returns (best_ret_pct, best_exit_idx, best_delay_min)."""
    px = mkt["close"].to_numpy(np.float64, copy=False)
    n = int(px.size)
    fee_in = 1.0 + float(fee)
    fee_out = 1.0 - float(fee)

    entry_cost = px * fee_in
    best_ret = np.full(n, -np.inf, dtype=np.float64)
    best_exit_idx = np.full(n, -1, dtype=np.int64)
    best_delay = np.full(n, np.nan, dtype=np.float64)

    H = int(hold_min)
    for k in range(1, H + 1):
        if n <= k:
            break
        ret_k = (px[k:] * fee_out) / entry_cost[:-k] - 1.0
        better = ret_k > best_ret[:-k]
        if np.any(better):
            idx = np.flatnonzero(better)
            best_ret[idx] = ret_k[idx]
            best_exit_idx[idx] = idx + k
            best_delay[idx] = float(k)

    out_ret = np.full(n, np.nan, dtype=np.float64)
    sel = best_exit_idx >= 0
    out_ret[sel] = best_ret[sel] * 100.0
    out_delay = best_delay
    return out_ret, best_exit_idx, out_delay


def _oracle_forward_short(mkt: pd.DataFrame, hold_min: int, fee: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized best forward return for SHORT entries within [1, hold_min] minutes.
    Returns (best_ret_pct, best_exit_idx, best_delay_min).

    Short round-trip net multiplier for exact-k exit:
        mult_k = (entry_px*(1-fee)) / (exit_px*(1+fee))
        ret_k = mult_k - 1
    """
    px = mkt["close"].to_numpy(np.float64, copy=False)
    n = int(px.size)
    fee_in = 1.0 - float(fee)   # proceeds received when selling
    fee_out = 1.0 + float(fee)  # buying back to cover

    entry_proceeds = px * fee_in
    best_ret = np.full(n, -np.inf, dtype=np.float64)
    best_exit_idx = np.full(n, -1, dtype=np.int64)
    best_delay = np.full(n, np.nan, dtype=np.float64)

    H = int(hold_min)
    for k in range(1, H + 1):
        if n <= k:
            break
        # ret_k defined for entries 0..n-k-1 covering at i+k
        ret_k = (entry_proceeds[:-k] / (px[k:] * fee_out)) - 1.0
        better = ret_k > best_ret[:-k]
        if np.any(better):
            idx = np.flatnonzero(better)
            best_ret[idx] = ret_k[idx]
            best_exit_idx[idx] = idx + k
            best_delay[idx] = float(k)

    out_ret = np.full(n, np.nan, dtype=np.float64)
    sel = best_exit_idx >= 0
    out_ret[sel] = best_ret[sel] * 100.0
    out_delay = best_delay
    return out_ret, best_exit_idx, out_delay


def build_labels(mkt: pd.DataFrame, hold_mins: Iterable[int], fee: float) -> pd.DataFrame:
    ts = mkt[["timestamp"]].copy()
    ts_arr = mkt["timestamp"].to_numpy(dtype="datetime64[ns]")

    out = ts.copy()
    for H in hold_mins:
        H = int(H)
        # Long side
        l_ret, l_exit_idx, l_delay = _oracle_forward_long(mkt, H, fee)
        l_exit_ts = np.full(l_exit_idx.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
        ok_l = l_exit_idx >= 0
        l_exit_ts[ok_l] = ts_arr[l_exit_idx[ok_l]]

        out[f"oracle_long_best_ret_pct_{H}m"] = l_ret
        out[f"oracle_long_best_exit_ts_{H}m"] = l_exit_ts
        out[f"oracle_long_best_delay_min_{H}m"] = l_delay
        out[f"oracle_long_has_profit_{H}m"] = (l_ret > 0.0)

        # Short side
        s_ret, s_exit_idx, s_delay = _oracle_forward_short(mkt, H, fee)
        s_exit_ts = np.full(s_exit_idx.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
        ok_s = s_exit_idx >= 0
        s_exit_ts[ok_s] = ts_arr[s_exit_idx[ok_s]]

        out[f"oracle_short_best_ret_pct_{H}m"] = s_ret
        out[f"oracle_short_best_exit_ts_{H}m"] = s_exit_ts
        out[f"oracle_short_best_delay_min_{H}m"] = s_delay
        out[f"oracle_short_has_profit_{H}m"] = (s_ret > 0.0)

    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build symmetric oracle labels (long/short) over multiple horizons")
    ap.add_argument("--market-csv", required=True, help="CSV with timestamp, open, high, low, close, volume")
    ap.add_argument("--fee", type=float, default=0.001, help="Per-side fee (e.g. 0.001 = 0.1%)")
    ap.add_argument(
        "--hold-mins",
        default="5,15,30",
        help="Comma-separated horizons in minutes, e.g. 5,15,30",
    )
    ap.add_argument("--out-dir", default="data/oracle_labels", help="Output directory")
    ap.add_argument("--out-stem", default="oracle_labels_sym", help="Filename stem")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    hold_mins = [int(x) for x in str(args.hold_mins).split(",") if str(x).strip()]

    mkt = minute_bars(Path(args.market_csv))

    labels = build_labels(mkt, hold_mins=hold_mins, fee=float(args.fee))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    out_parquet = out_dir / f"{args.out_stem}_{ts}.parquet"
    out_csv = out_dir / f"{args.out_stem}_{ts}.csv"

    labels.to_parquet(out_parquet, index=False)
    labels.to_csv(out_csv, index=False)

    # Quick stats for the smallest horizon as a sanity check
    H0 = hold_mins[0]
    lp = float(pd.Series(labels[f"oracle_long_has_profit_{H0}m"]).mean() * 100.0)
    sp = float(pd.Series(labels[f"oracle_short_has_profit_{H0}m"]).mean() * 100.0)
    print(f"Rows: {len(labels):,}")
    print(f"Long>0% (H={H0}m): {lp:.2f}%  Short>0% (H={H0}m): {sp:.2f}%")
    print("Wrote:", out_parquet)
    print("Also CSV:", out_csv)


if __name__ == "__main__":
    main()
