#!/usr/bin/env python3
from __future__ import annotations
"""
Segment daily data (noise-based cleanliness) into 3 main regimes and 8 total subtypes
using exactly the three characteristics:
- Amount of daily clean opportunities (density)
- BUY/SELL prevalence
- Expected duration (pooled)

Inputs:
- --daily-features: parquet/csv produced by aggregate_regime_features_daily_noise_*.py

Outputs:
- data/regimes/segmented_tree_noise_<TS>.{parquet,csv}
"""
import argparse
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd


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

    # Density tertiles for chop vs active
    d_lo = float(np.nanpercentile(dens, 33))
    d_hi = float(np.nanpercentile(dens, 67))

    # Trend bias threshold from prevalence magnitude
    p_thr = float(np.nanpercentile(np.abs(prev), 60))

    # Duration median split
    dur_thr = float(np.nanmedian(dur[~np.isnan(dur)])) if np.isfinite(dur).any() else 15.0

    return dict(density_lo=d_lo, density_hi=d_hi, prevalence_thr=p_thr, duration_thr=dur_thr)


def assign_main(df: pd.DataFrame, thr: dict) -> pd.Series:
    dens = df["clean_density_total_per_1kmin"].fillna(0.0).to_numpy()
    prev = df["prevalence"].fillna(0.0).to_numpy()
    out = np.array(["CHOP"] * len(df), dtype=object)

    strong_trend = np.abs(prev) >= thr["prevalence_thr"]
    active = dens >= thr["density_hi"]
    trend_mask = strong_trend | (active & (np.abs(prev) >= thr["prevalence_thr"] * 0.5))

    up = (prev > 0) & trend_mask
    down = (prev < 0) & trend_mask
    out[up] = "TREND_UP"
    out[down] = "TREND_DOWN"
    return pd.Series(out, index=df.index, name="main_type")


def assign_subtype(df: pd.DataFrame, main: pd.Series, thr: dict) -> pd.Series:
    # Only duration is used for subtype within each main type; density already encoded in main, and prevalence sets direction
    dur = df["expected_delay_med_pooled"].fillna(thr["duration_thr"]).to_numpy()
    dens = df["clean_density_total_per_1kmin"].fillna(0.0).to_numpy()

    is_short = dur <= thr["duration_thr"]
    is_active = dens >= thr["density_hi"]  # proxy for higher-activity subtypes inside trends

    sub = np.empty(len(df), dtype=object)
    for i, m in enumerate(main.to_numpy()):
        if m == "TREND_UP":
            # 4 subtypes: active/passive × short/long duration
            if is_active[i] and is_short[i]: sub[i] = "TREND_UP__ACTIVE_SD"
            elif is_active[i] and not is_short[i]: sub[i] = "TREND_UP__ACTIVE_LD"
            elif (not is_active[i]) and is_short[i]: sub[i] = "TREND_UP__QUIET_SD"
            else: sub[i] = "TREND_UP__QUIET_LD"
        elif m == "TREND_DOWN":
            if is_active[i] and is_short[i]: sub[i] = "TREND_DOWN__ACTIVE_SD"
            elif is_active[i] and not is_short[i]: sub[i] = "TREND_DOWN__ACTIVE_LD"
            elif (not is_active[i]) and is_short[i]: sub[i] = "TREND_DOWN__QUIET_SD"
            else: sub[i] = "TREND_DOWN__QUIET_LD"
        else:
            # CHOP split into low vs high density only (2 subtypes)
            sub[i] = "CHOP__LOW_DENS" if dens[i] < thr["density_lo"] else "CHOP__HIGH_DENS"
    return pd.Series(sub, index=df.index, name="subtype")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Segment noise-based daily features into 3 main and 8 subtypes via tree rules")
    ap.add_argument("--daily-features", required=True)
    ap.add_argument("--out-dir", default="data/regimes")
    ap.add_argument("--out-stem", default="segmented_tree_noise")
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

    print("Thresholds:", thr)
    print("Main counts:\n", out["main_type"].value_counts())
    print("Subtype counts:\n", out["subtype"].value_counts())
    print("Wrote:", out_parquet)
    print("Also CSV:", out_csv)


if __name__ == "__main__":
    main()
