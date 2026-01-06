#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-05T18:02:27Z
"""Label ETH-USD regimes (CHOP vs TREND) in a clean, reproducible way.

Core idea
- Use a gap-safe 60-minute "efficiency ratio" (ER):
    ER60 = abs(60m_return) / sum_{last 60m} abs(1m_return)
  * High ER => trending (net progress is large relative to back-and-forth)
  * Low ER  => choppy (lots of movement but little net progress)

- Also require a minimum 60m normalized range to avoid labeling totally dead/flat periods as "chop".

Outputs
- data/backtests/ethusd_regimes_<ts>.parquet : per-minute regime labels + metrics
- data/backtests/ethusd_regimes_daily_<ts>.csv : per-day summary stats and fractions

Notes
- Handles missing minutes by invalidating windows that cross timestamp gaps > 1 minute.
- Labels are quantile-based, so they are stable and interpretable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def compute_gap_safe_absret_1m(ts: np.ndarray, close: np.ndarray) -> np.ndarray:
    """abs 1m return aligned to row i (uses close[i]/close[i-1]-1), invalid across gaps."""
    close = np.asarray(close, dtype=np.float64)
    ts = np.asarray(ts)
    out = np.full(close.shape, np.nan, dtype=np.float64)
    if close.size < 2:
        return out

    # row-adjacent return
    r = close[1:] / np.maximum(1e-12, close[:-1]) - 1.0

    # invalidate across gaps > 1 minute
    dt_min = (ts[1:] - ts[:-1]) / np.timedelta64(1, "m")
    ok = np.isclose(dt_min.astype(np.float64), 1.0, atol=1e-9)

    out[1:] = np.where(ok, np.abs(r), np.nan)
    return out


def rolling_sum_valid(x: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rolling sum treating NaN as 0, and rolling count of finite values."""
    x = np.asarray(x, dtype=np.float64)
    w = int(w)
    n = int(x.size)
    s = np.nan_to_num(x, nan=0.0)
    v = np.isfinite(x).astype(np.int32)

    cs = np.cumsum(s)
    cv = np.cumsum(v)

    out_sum = np.full(n, np.nan, dtype=np.float64)
    out_n = np.full(n, np.nan, dtype=np.float64)

    if n >= w:
        out_sum[w - 1 :] = cs[w - 1 :] - np.concatenate(([0.0], cs[: n - w]))
        out_n[w - 1 :] = cv[w - 1 :] - np.concatenate(([0.0], cv[: n - w]))

    return out_sum, out_n


def main() -> None:
    ap = argparse.ArgumentParser(description="Label ETH-USD regimes: CHOP vs TREND")
    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="Parquet with at least timestamp/open/high/low/close.",
    )
    ap.add_argument("--window-min", type=int, default=60, help="Rolling window length in minutes.")

    ap.add_argument("--er-low-q", type=float, default=0.35, help="ER quantile below which we label CHOP")
    ap.add_argument("--er-high-q", type=float, default=0.65, help="ER quantile above which we label TREND")
    ap.add_argument("--rng-min-q", type=float, default=0.20, help="range_norm quantile floor required to label CHOP/TREND")

    ap.add_argument(
        "--out-dir",
        default="data/backtests",
        help="Where to write outputs.",
    )
    ap.add_argument("--start-iso", default=None)
    ap.add_argument("--end-iso", default=None)
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    ts_now = now_ts()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load bars
    cols = ["timestamp", "open", "high", "low", "close"]
    df = pd.read_parquet(Path(args.bars), columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if args.start_iso:
        start_dt = pd.to_datetime(str(args.start_iso), utc=True, errors="coerce")
        if pd.isna(start_dt):
            raise SystemExit(f"Invalid --start-iso: {args.start_iso!r}")
        df = df[df["timestamp"] >= start_dt].copy()
    if args.end_iso:
        end_dt = pd.to_datetime(str(args.end_iso), utc=True, errors="coerce")
        if pd.isna(end_dt):
            raise SystemExit(f"Invalid --end-iso: {args.end_iso!r}")
        df = df[df["timestamp"] < end_dt].copy()

    df = df.reset_index(drop=True)

    ts = df["timestamp"].to_numpy(dtype="datetime64[ns]")
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64)
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(np.float64)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(np.float64)

    W = int(args.window_min)

    # Abs 1m return (gap-safe) and rolling sum
    absret1 = compute_gap_safe_absret_1m(ts, close)
    abs_sum, abs_n = rolling_sum_valid(absret1, W)

    # 60m return aligned to row i: close[i]/close[i-W] - 1
    retW = np.full_like(close, np.nan)
    if close.size > W:
        retW[W:] = close[W:] / np.maximum(1e-12, close[:-W]) - 1.0

    # Require contiguity: in a true 60m window you have W-1 valid 1m diffs
    full = np.isfinite(abs_n) & (abs_n >= float(W - 1))
    abs_sum = np.where(full, abs_sum, np.nan)
    retW = np.where(full, retW, np.nan)

    # Range norm over W rows ending at i
    rngW = (
        pd.Series(high).rolling(W, min_periods=W).max().to_numpy(np.float64)
        - pd.Series(low).rolling(W, min_periods=W).min().to_numpy(np.float64)
    )
    rng_norm = rngW / np.maximum(1e-12, close)
    rng_norm = np.where(full, rng_norm, np.nan)

    # Efficiency ratio (0..1-ish), and chop index (>=1)
    er = np.abs(retW) / np.maximum(1e-12, abs_sum)
    chop = abs_sum / np.maximum(1e-12, np.abs(retW))
    er = np.where(np.isfinite(er), er, np.nan)
    chop = np.where(np.isfinite(chop), chop, np.nan)

    # Quantile thresholds computed on valid minutes
    er_valid = er[np.isfinite(er)]
    rng_valid = rng_norm[np.isfinite(rng_norm)]

    if er_valid.size < 10_000:
        raise SystemExit(f"Not enough valid regime rows (er_valid={er_valid.size})")

    er_lo = float(np.quantile(er_valid, float(args.er_low_q)))
    er_hi = float(np.quantile(er_valid, float(args.er_high_q)))
    rng_min = float(np.quantile(rng_valid, float(args.rng_min_q)))

    # Labels
    label = np.full((len(df),), "UNKNOWN", dtype=object)
    label[np.isfinite(er) & np.isfinite(rng_norm)] = "NEUTRAL"

    chop_mask = np.isfinite(er) & np.isfinite(rng_norm) & (rng_norm >= rng_min) & (er <= er_lo)
    trend_mask = np.isfinite(er) & np.isfinite(rng_norm) & (rng_norm >= rng_min) & (er >= er_hi)

    label[chop_mask] = "CHOP"
    label[trend_mask] = "TREND"

    # attach outputs
    out = df[["timestamp", "open", "high", "low", "close"]].copy()
    out["retW_pct"] = retW * 100.0
    out["abs_sumW"] = abs_sum
    out["erW"] = er
    out["chopW"] = chop
    out["range_normW"] = rng_norm
    out["regime"] = label

    # daily summary
    out["day"] = out["timestamp"].dt.date
    g = out.groupby("day")
    daily = pd.DataFrame(
        {
            "day": g.size().index.astype(str),
            "n_rows": g.size().to_numpy(np.int64),
            "frac_chop": g["regime"].apply(lambda s: float((s == "CHOP").mean())).to_numpy(np.float64),
            "frac_trend": g["regime"].apply(lambda s: float((s == "TREND").mean())).to_numpy(np.float64),
            "frac_neutral": g["regime"].apply(lambda s: float((s == "NEUTRAL").mean())).to_numpy(np.float64),
            "median_chopW": g["chopW"].median(numeric_only=True).to_numpy(np.float64),
            "median_erW": g["erW"].median(numeric_only=True).to_numpy(np.float64),
            "median_range_normW": g["range_normW"].median(numeric_only=True).to_numpy(np.float64),
            "median_abs_retW_pct": g["retW_pct"].apply(lambda s: float(np.nanmedian(np.abs(s.to_numpy(np.float64))))).to_numpy(np.float64),
        }
    )

    out_path = out_dir / f"ethusd_regimes_{ts_now}.parquet"
    daily_path = out_dir / f"ethusd_regimes_daily_{ts_now}.csv"

    out.to_parquet(out_path, index=False)
    daily.to_csv(daily_path, index=False)

    if args.verbose:
        print('Thresholds:')
        print('  er_lo', er_lo, 'er_hi', er_hi, 'rng_min', rng_min)
        vc = pd.Series(label).value_counts(dropna=False)
        print('Regime counts:', vc.to_dict())

    print('Wrote:')
    print(' ', out_path)
    print(' ', daily_path)


if __name__ == "__main__":
    main()
