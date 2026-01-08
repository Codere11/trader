#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-08T14:38:15Z
"""Compute simple daily regime features for ETH-USD.

Intent
- Produce day-level aggregates aligned with what you described: daily return, daily range/volatility,
  daily volume, and derived measures like trend efficiency and close position.
- Also compute early-window (first 60m / 120m) variants so you can estimate regime during the day.

Input
- A 1-minute bars file (parquet or CSV) with at least:
    timestamp, open, high, low, close, volume

Outputs
- data/regimes/ethusd_daily_regime_simple_<ts>.csv
- data/regimes/ethusd_daily_regime_simple_<ts>.parquet (if parquet engine available)

Notes
- All timestamps are treated as UTC.
- Return/volatility metrics are computed within-day (no cross-day returns).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _read_bars(path: Path) -> pd.DataFrame:
    p = Path(path)
    cols = ["timestamp", "open", "high", "low", "close", "volume"]

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p, columns=cols)
    else:
        df = pd.read_csv(p, parse_dates=["timestamp"], usecols=cols)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    df["volume"] = df["volume"].fillna(0.0)

    return df


def _day_start(ts: pd.Series) -> pd.Series:
    # Keep tz-aware UTC day boundaries.
    return pd.to_datetime(ts, utc=True).dt.floor("D")


def _compute_intraday_ret_1m_pct(df: pd.DataFrame) -> pd.Series:
    # pct_change within day.
    day = df["day"]
    close = df["close"]
    return close.groupby(day).pct_change(1) * 100.0


def _daily_agg(df: pd.DataFrame) -> pd.DataFrame:
    # df must contain: day, timestamp, open/high/low/close/volume, ret_1m_pct
    g = df.groupby("day", as_index=False)

    daily = g.agg(
        open_day=("open", "first"),
        high_day=("high", "max"),
        low_day=("low", "min"),
        close_day=("close", "last"),
        volume_day=("volume", "sum"),
        n_bars=("timestamp", "count"),
    )

    daily["usd_volume_day"] = (
        df.assign(usd_vol=df["volume"] * df["close"])  # proxy
        .groupby("day")["usd_vol"]
        .sum()
        .to_numpy(np.float64)
    )

    # returns + range
    daily["day_ret_pct"] = (daily["close_day"] / daily["open_day"] - 1.0) * 100.0
    daily["day_range_pct"] = (daily["high_day"] - daily["low_day"]) / daily["open_day"].replace(0, np.nan) * 100.0

    # close position within day range (0..1)
    denom = (daily["high_day"] - daily["low_day"]).replace(0, np.nan)
    daily["close_position"] = (daily["close_day"] - daily["low_day"]) / denom

    # realized vol within day
    rv = df.groupby("day")["ret_1m_pct"].std(ddof=0)
    daily["realized_vol_day"] = rv.to_numpy(np.float64)

    # asymmetry
    daily["max_up_1m_pct"] = df.groupby("day")["ret_1m_pct"].max().to_numpy(np.float64)
    daily["max_down_1m_pct"] = df.groupby("day")["ret_1m_pct"].min().to_numpy(np.float64)

    daily["sum_pos_ret_1m_pct"] = (
        df.assign(pos=np.where(df["ret_1m_pct"] > 0, df["ret_1m_pct"], 0.0)).groupby("day")["pos"].sum().to_numpy(np.float64)
    )
    daily["sum_neg_ret_1m_pct"] = (
        df.assign(neg=np.where(df["ret_1m_pct"] < 0, df["ret_1m_pct"], 0.0)).groupby("day")["neg"].sum().to_numpy(np.float64)
    )

    # trend efficiency (simple)
    daily["trend_efficiency"] = (daily["day_ret_pct"].abs() / daily["day_range_pct"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # rolling volume z (14d median, using prior days only)
    vol_med14 = daily["volume_day"].shift(1).rolling(14, min_periods=5).median()
    daily["volume_z_14d"] = (daily["volume_day"] / vol_med14.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # prev-day convenience columns
    for c in [
        "day_ret_pct",
        "day_range_pct",
        "realized_vol_day",
        "volume_day",
        "volume_z_14d",
        "trend_efficiency",
        "close_position",
        "max_up_1m_pct",
        "max_down_1m_pct",
    ]:
        daily[f"prev_{c}"] = daily[c].shift(1)

    return daily


def _early_window_features(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    # For each day, compute aggregates on [day_start, day_start + minutes).
    mins = int(minutes)
    if mins <= 0:
        raise ValueError("minutes must be > 0")

    out_rows = []
    for d, g in df.groupby("day"):
        day0 = pd.to_datetime(d, utc=True)
        end = day0 + timedelta(minutes=mins)
        w = g[(g["timestamp"] >= day0) & (g["timestamp"] < end)].copy()
        if w.empty:
            out_rows.append({"day": d, f"n_first_{mins}m": 0})
            continue

        open0 = float(w["open"].iloc[0])
        close_last = float(w["close"].iloc[-1])
        high = float(w["high"].max())
        low = float(w["low"].min())
        vol = float(w["volume"].sum())

        ret = (close_last / open0 - 1.0) * 100.0 if open0 != 0 else np.nan
        rng = (high - low) / open0 * 100.0 if open0 != 0 else np.nan
        rv = float(w["ret_1m_pct"].std(ddof=0))

        out_rows.append(
            {
                "day": d,
                f"n_first_{mins}m": int(len(w)),
                f"ret_first_{mins}m_pct": float(ret),
                f"range_first_{mins}m_pct": float(rng),
                f"realized_vol_first_{mins}m": float(rv),
                f"volume_first_{mins}m": float(vol),
                f"max_up_1m_first_{mins}m_pct": float(w["ret_1m_pct"].max()),
                f"max_down_1m_first_{mins}m_pct": float(w["ret_1m_pct"].min()),
            }
        )

    return pd.DataFrame(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute ETH-USD daily regime features (simple daily aggregates + early windows)")
    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="Parquet/CSV with timestamp, open, high, low, close, volume.",
    )
    ap.add_argument("--out-dir", default="data/regimes")
    ap.add_argument("--first-windows", default="60,120", help="Comma-separated early windows in minutes")

    args = ap.parse_args()

    bars_path = Path(args.bars)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_bars(bars_path)
    df["day"] = _day_start(df["timestamp"])
    df["ret_1m_pct"] = _compute_intraday_ret_1m_pct(df)

    daily = _daily_agg(df)

    windows = [int(x) for x in str(args.first_windows).split(",") if str(x).strip()]
    for w in windows:
        wdf = _early_window_features(df, int(w))
        daily = daily.merge(wdf, on="day", how="left")

    ts = _now_ts()
    out_csv = out_dir / f"ethusd_daily_regime_simple_{ts}.csv"
    out_parq = out_dir / f"ethusd_daily_regime_simple_{ts}.parquet"

    daily = daily.sort_values("day").reset_index(drop=True)
    daily.to_csv(out_csv, index=False)

    try:
        daily.to_parquet(out_parq, index=False)
        wrote_parquet = True
    except Exception as e:
        wrote_parquet = False
        print(f"NOTE: Parquet write skipped ({type(e).__name__}: {e}).")

    print("Wrote:")
    print(" ", out_csv)
    if wrote_parquet:
        print(" ", out_parq)

    # quick status
    if not daily.empty:
        last = daily.iloc[-1]
        print("\nLatest day:", str(last["day"]))
        print("  day_ret_pct:", float(last.get("day_ret_pct", float("nan"))))
        print("  day_range_pct:", float(last.get("day_range_pct", float("nan"))))
        print("  volume_day:", float(last.get("volume_day", float("nan"))))
        print("  realized_vol_day:", float(last.get("realized_vol_day", float("nan"))))


if __name__ == "__main__":
    main()
