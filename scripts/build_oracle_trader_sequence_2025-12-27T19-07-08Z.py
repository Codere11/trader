#!/usr/bin/env python3
from __future__ import annotations
"""
Oracle trader sequence (both BUY and SELL), one trade at a time, maximizing total net return
over the entire history subject to a maximum holding window.

- Reads a minute-level market CSV (timestamp, open, high, low, close, volume)
- Uses TOTAL round-trip fee (e.g., 0.001 for 0.1% total, applied symmetrically as 0.0005 per side)
- For each minute i, precomputes the best long and best short exit within [i+1, i+H]
- Runs a dynamic program to choose an optimal non-overlapping sequence of trades that
  maximizes the cumulative capital multiplier
- Emits detailed trades CSV + a summary CSV/printout

Complexity: O(n * H) time, O(n) memory. Choose H reasonably (e.g., 30, 60, 120, 240).
"""
import argparse
import datetime as dt
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

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


def best_forward_long_short(px: np.ndarray, per_side_fee: float, hold_min: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized best forward returns for both long and short from each index i.
    per_side_fee is half of the total round-trip fee.
    Returns: long_best_ret_pct, long_best_exit_idx, short_best_ret_pct, short_best_exit_idx
    """
    n = int(px.size)
    fee_in = 1.0 + float(per_side_fee)
    fee_out = 1.0 - float(per_side_fee)

    # Long
    entry_cost_long = px * fee_in
    best_long = np.full(n, -np.inf, dtype=np.float64)
    best_long_exit = np.full(n, -1, dtype=np.int64)

    # Short
    entry_proceeds_short = px * (1.0 - float(per_side_fee))
    best_short = np.full(n, -np.inf, dtype=np.float64)
    best_short_exit = np.full(n, -1, dtype=np.int64)

    H = int(hold_min)
    for k in range(1, H + 1):
        if n <= k:
            break
        # Long: exit at i+k
        ret_long_k = (px[k:] * fee_out) / entry_cost_long[:-k] - 1.0
        better_l = ret_long_k > best_long[:-k]
        if np.any(better_l):
            idx = np.flatnonzero(better_l)
            best_long[idx] = ret_long_k[idx]
            best_long_exit[idx] = idx + k
        # Short: exit at i+k
        ret_short_k = (entry_proceeds_short[:-k] / (px[k:] * (1.0 + float(per_side_fee)))) - 1.0
        better_s = ret_short_k > best_short[:-k]
        if np.any(better_s):
            idx = np.flatnonzero(better_s)
            best_short[idx] = ret_short_k[idx]
            best_short_exit[idx] = idx + k

    out_long_pct = np.full(n, np.nan, dtype=np.float64)
    out_short_pct = np.full(n, np.nan, dtype=np.float64)
    mask_l = best_long_exit >= 0
    mask_s = best_short_exit >= 0
    out_long_pct[mask_l] = best_long[mask_l] * 100.0
    out_short_pct[mask_s] = best_short[mask_s] * 100.0
    return out_long_pct, best_long_exit, out_short_pct, best_short_exit


def dp_optimal_sequence(long_ret: np.ndarray, long_exit: np.ndarray, short_ret: np.ndarray, short_exit: np.ndarray) -> tuple[list[dict], float]:
    """Dynamic program over time to choose an optimal non-overlapping trade sequence.
    Uses multiplicative capital; returns trade list and final multiplier.
    """
    n = int(long_ret.size)
    M = np.ones(n + 1, dtype=np.float64)  # best multiplier from i..end
    choice = np.zeros(n, dtype=np.int8)    # 0=skip, 1=long, 2=short
    exit_idx = np.full(n, -1, dtype=np.int64)
    ret_pct = np.zeros(n, dtype=np.float64)

    for i in range(n - 1, -1, -1):
        # Option: skip
        best_mult = M[i + 1]
        best_c = 0
        best_e = -1
        best_r = 0.0

        # Option: take long from i
        j = int(long_exit[i])
        if j >= 0 and np.isfinite(long_ret[i]):
            mult = (1.0 + float(long_ret[i]) / 100.0) * M[j + 1]
            if mult > best_mult:
                best_mult = mult; best_c = 1; best_e = j; best_r = float(long_ret[i])

        # Option: take short from i
        j = int(short_exit[i])
        if j >= 0 and np.isfinite(short_ret[i]):
            mult = (1.0 + float(short_ret[i]) / 100.0) * M[j + 1]
            if mult > best_mult:
                best_mult = mult; best_c = 2; best_e = j; best_r = float(short_ret[i])

        M[i] = best_mult
        choice[i] = best_c
        exit_idx[i] = best_e
        ret_pct[i] = best_r

    # Reconstruct
    trades: list[dict] = []
    i = 0
    cum_mult = 1.0
    while i < n:
        c = int(choice[i])
        if c == 0:
            i += 1
            continue
        j = int(exit_idx[i])
        r = float(ret_pct[i])
        side: Literal['BUY','SELL'] = 'BUY' if c == 1 else 'SELL'
        trades.append(dict(entry_idx=i, exit_idx=j, net_return_pct=r, side=side))
        cum_mult *= (1.0 + r / 100.0)
        i = j + 1

    return trades, float(cum_mult)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Oracle trader sequence: optimal non-overlapping BUY/SELL trades over history")
    ap.add_argument("--market-csv", required=True, help="CSV with timestamp, open, high, low, close, volume")
    ap.add_argument("--fee-total", type=float, default=0.001, help="TOTAL round-trip fee (e.g. 0.001 = 0.1% total)")
    ap.add_argument("--hold-min", type=int, default=60, help="Max holding minutes (window for best-exit search)")
    ap.add_argument("--min-ret-pct", type=float, default=0.0, help="Optional floor to drop trades with net_return_pct < this (after DP)")
    ap.add_argument("--out-dir", default="data/oracle_trader", help="Output directory for trades and summary")
    ap.add_argument("--out-stem", default="oracle_trader_sequence", help="Filename stem")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    mkt = minute_bars(Path(args.market_csv))
    px = mkt["close"].to_numpy(np.float64, copy=False)

    per_side_fee = float(args.fee_total) / 2.0
    long_ret, long_exit, short_ret, short_exit = best_forward_long_short(px, per_side_fee, int(args.hold_min))

    trades, final_mult = dp_optimal_sequence(long_ret, long_exit, short_ret, short_exit)

    # Materialize with timestamps and durations
    ts = mkt["timestamp"].to_numpy(dtype="datetime64[ns]")
    rows = []
    for t in trades:
        ei = int(t["entry_idx"]); xj = int(t["exit_idx"]) ; r = float(t["net_return_pct"]) ; side = str(t["side"]) 
        entry_ts = ts[ei]; exit_ts = ts[xj]
        dur_min = int((exit_ts - entry_ts) / np.timedelta64(1, 'm'))
        rows.append(dict(
            entry_time=pd.Timestamp(entry_ts),
            exit_time=pd.Timestamp(exit_ts),
            duration_min=dur_min,
            side=side,
            net_return_pct=r,
        ))

    df = pd.DataFrame(rows)
    if float(args.min_ret_pct) > 0.0:
        df = df[df["net_return_pct"] >= float(args.min_ret_pct)].reset_index(drop=True)

    # Cumulative multiplier over kept trades
    cum_mult = float(np.prod(1.0 + df["net_return_pct"].to_numpy(np.float64, copy=False) / 100.0)) if len(df) else 1.0

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    now_ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    out_trades = out_dir / f"{args.out_stem}_trades_{now_ts}.csv"
    out_summary = out_dir / f"{args.out_stem}_summary_{now_ts}.csv"

    df.to_csv(out_trades, index=False)

    summary = pd.DataFrame([dict(
        n_trades=int(len(df)),
        hold_min=int(args.hold_min),
        fee_total=float(args.fee_total),
        fee_per_side=per_side_fee,
        final_multiplier=cum_mult,
        final_return_pct=(cum_mult - 1.0) * 100.0,
        total_minutes=int((mkt["timestamp"].iloc[-1] - mkt["timestamp"].iloc[0]).total_seconds() // 60),
    )])
    summary.to_csv(out_summary, index=False)

    print(f"Trades: {len(df):,}")
    print(f"Final multiplier: {cum_mult:.6f}  Return %: {(cum_mult-1.0)*100.0:.2f}")
    print("Wrote:", out_trades)
    print("Summary:", out_summary)


if __name__ == "__main__":
    main()
