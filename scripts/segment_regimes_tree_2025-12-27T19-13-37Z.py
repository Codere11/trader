#!/usr/bin/env python3
from __future__ import annotations
"""
Segment daily data into 3 main market phases and 8 subtypes using tree-style rules
based on your criteria: clean-opportunity density, BUY/SELL prevalence, and
expected duration, plus volatility context.

Inputs:
- --daily-features: parquet/csv produced by aggregate_regime_features_daily_*.py

Outputs:
- data/regimes/segmented_k3_k8_<TS>.parquet/csv with columns:
  date, main_type, subtype, thresholds used, and core metrics per day
- prints summary counts per class and threshold values
"""
import argparse
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

MAIN_TYPES = [
    "TREND_UP",
    "TREND_DOWN",
    "CHOP",
]

SUBTYPES = [
    # Trend Up (4)
    "TREND_UP__LV_SD",
    "TREND_UP__LV_LD",
    "TREND_UP__HV_SD",
    "TREND_UP__HV_LD",
    # Trend Down (4)
    "TREND_DOWN__LV_SD",
    "TREND_DOWN__LV_LD",
    "TREND_DOWN__HV_SD",
    "TREND_DOWN__HV_LD",
    # Chop (2) — will only use the first two names
    "CHOP__LV",
    "CHOP__HV",
]


def read_daily(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["date"])  # date may be parsed to datetime
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])
    return df


def compute_thresholds(df: pd.DataFrame) -> dict:
    dens = df["clean_density_total_per_1kmin"].to_numpy()
    prev = df["prevalence"].to_numpy()
    dur = df["expected_delay_med_pooled"].to_numpy()
    vol = df["vol_std_5m_med"].to_numpy()

    # Density tertiles: low/mid/high
    d_lo = float(np.nanpercentile(dens, 33))
    d_hi = float(np.nanpercentile(dens, 67))

    # Prevalence threshold for trend detection: abs(prev) >= p_thr
    p_thr = float(np.nanpercentile(np.abs(prev), 60))  # moderately strong bias
    # Duration threshold: split at median of valid
    dur_thr = float(np.nanmedian(dur[~np.isnan(dur)])) if np.isfinite(dur).any() else 10.0
    # Volatility threshold: split near median
    vol_thr = float(np.nanmedian(vol[~np.isnan(vol)])) if np.isfinite(vol).any() else 10.0

    return dict(density_lo=d_lo, density_hi=d_hi, prevalence_thr=p_thr, duration_thr=dur_thr, vol_thr=vol_thr)


def assign_main(df: pd.DataFrame, thr: dict) -> pd.Series:
    dens = df["clean_density_total_per_1kmin"].fillna(0.0).to_numpy()
    prev = df["prevalence"].fillna(0.0).to_numpy()
    out = np.array(["CHOP"] * len(df), dtype=object)

    strong_trend = np.abs(prev) >= thr["prevalence_thr"]
    high_density = dens >= thr["density_hi"]
    # Prefer strong bias OR high density with some bias
    trend_mask = strong_trend | (high_density & (np.abs(prev) >= thr["prevalence_thr"] * 0.5))

    up = (prev > 0) & trend_mask
    down = (prev < 0) & trend_mask
    out[up] = "TREND_UP"
    out[down] = "TREND_DOWN"
    # Remaining days default to CHOP
    return pd.Series(out, index=df.index, name="main_type")


def assign_subtype(df: pd.DataFrame, main: pd.Series, thr: dict) -> pd.Series:
    vol = df["vol_std_5m_med"].fillna(thr["vol_thr"]).to_numpy()
    dur = df["expected_delay_med_pooled"].fillna(thr["duration_thr"]).to_numpy()
    sub = np.empty(len(df), dtype=object)

    is_low_vol = vol < thr["vol_thr"]
    is_short = dur <= thr["duration_thr"]

    for i, m in enumerate(main.to_numpy()):
        if m == "TREND_UP":
            if is_low_vol[i] and is_short[i]: sub[i] = "TREND_UP__LV_SD"
            elif is_low_vol[i] and not is_short[i]: sub[i] = "TREND_UP__LV_LD"
            elif not is_low_vol[i] and is_short[i]: sub[i] = "TREND_UP__HV_SD"
            else: sub[i] = "TREND_UP__HV_LD"
        elif m == "TREND_DOWN":
            if is_low_vol[i] and is_short[i]: sub[i] = "TREND_DOWN__LV_SD"
            elif is_low_vol[i] and not is_short[i]: sub[i] = "TREND_DOWN__LV_LD"
            elif not is_low_vol[i] and is_short[i]: sub[i] = "TREND_DOWN__HV_SD"
            else: sub[i] = "TREND_DOWN__HV_LD"
        else:
            sub[i] = "CHOP__LV" if is_low_vol[i] else "CHOP__HV"
    return pd.Series(sub, index=df.index, name="subtype")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Segment daily regime features into 3 main and 8 subtypes via tree rules")
    ap.add_argument("--daily-features", required=True)
    ap.add_argument("--out-dir", default="data/regimes")
    ap.add_argument("--out-stem", default="segmented_tree")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    daily = read_daily(Path(args.daily_features))

    thr = compute_thresholds(daily)
    main = assign_main(daily, thr)
    sub = assign_subtype(daily, main, thr)

    out = daily.copy()
    out["main_type"] = main
    out["subtype"] = sub
    for k, v in thr.items():
        out[f"thr__{k}"] = v

    ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / f"{args.out_stem}_{ts}.parquet"
    out_csv = out_dir / f"{args.out_stem}_{ts}.csv"
    out.to_parquet(out_parquet, index=False)
    out.to_csv(out_csv, index=False)

    # Print summary
    print("Thresholds:", thr)
    print("Main type counts:\n", out["main_type"].value_counts())
    print("Subtype counts:\n", out["subtype"].value_counts())
    print("Wrote:", out_parquet)
    print("Also CSV:", out_csv)


if __name__ == "__main__":
    main()
