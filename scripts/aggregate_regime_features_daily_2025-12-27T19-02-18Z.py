#!/usr/bin/env python3
from __future__ import annotations
"""
Aggregate per-minute symmetric oracle labels and market features into daily regime
features used for clustering into market phases.

Inputs:
- --oracle-labels: parquet/csv from build_oracle_labels_sym_*.py
- --market-csv: same raw minute file used to compute features

Outputs (Parquet + CSV):
- One row per UTC day with fields like:
  - clean_long_{H}m, clean_short_{H}m, clean_total_{H}m
  - clean_density_total (per 1k minutes), prevalence (-1..1)
  - expected_delay_med_{H}m_{long,short,pooled}
  - vol_std_5m_med, range_norm_5m_med, mom_5m_pct_med
"""

import argparse
import datetime as dt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
from binance_adapter.crossref_oracle_vs_agent_patterns import compute_feature_frame


H_DEFAULTS = [5, 15, 30]


def read_labels(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["timestamp"])  # timestamp is datetime
    if "timestamp" not in df:
        raise SystemExit("oracle labels must include 'timestamp'")
    # ensure datetime and date key
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # if parquet preserved
    df["date"] = df["timestamp"].dt.date
    return df


def minute_bars(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, parse_dates=["timestamp"], usecols=["timestamp","open","high","low","close","volume"])
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    raw["ts_min"] = pd.to_datetime(raw["timestamp"]).dt.floor("min")
    mkt = (
        raw.groupby("ts_min")
        .agg(open=("open","first"), high=("high","max"), low=("low","min"), close=("close","last"), volume=("volume","sum"))
        .reset_index()
        .rename(columns={"ts_min":"timestamp"})
        .sort_values("timestamp").reset_index(drop=True)
    )
    return mkt


def daily_context_features(mkt: pd.DataFrame) -> pd.DataFrame:
    feats = compute_feature_frame(mkt.rename(columns={"timestamp":"ts_min"}))
    feats = feats.rename(columns={"ts_min":"timestamp"})
    feats["date"] = pd.to_datetime(feats["timestamp"]).dt.date
    grp = feats.groupby("date").agg(
        vol_std_5m_med=("vol_std_5m","median"),
        range_norm_5m_med=("range_norm_5m","median"),
        mom_5m_pct_med=("mom_5m_pct","median"),
    ).reset_index()
    return grp


def aggregate_daily(labels: pd.DataFrame, horizons: Iterable[int], thr_pct: float) -> pd.DataFrame:
    rows = []
    for d, g in labels.groupby("date"):
        row: dict[str, float | int | dt.date] = {"date": d}
        n_minutes = len(g)
        row["minutes"] = int(n_minutes)

        total_clean = 0
        delays_pooled = []

        for H in horizons:
            l_ret = g[f"oracle_long_best_ret_pct_{H}m"].to_numpy(np.float64, copy=False)
            s_ret = g[f"oracle_short_best_ret_pct_{H}m"].to_numpy(np.float64, copy=False)
            l_has = g[f"oracle_long_has_profit_{H}m"].to_numpy(bool, copy=False)
            s_has = g[f"oracle_short_has_profit_{H}m"].to_numpy(bool, copy=False)
            l_delay = g[f"oracle_long_best_delay_min_{H}m"].to_numpy(np.float64, copy=False)
            s_delay = g[f"oracle_short_best_delay_min_{H}m"].to_numpy(np.float64, copy=False)

            clean_l = (l_has) & np.isfinite(l_ret) & (l_ret >= thr_pct)
            clean_s = (s_has) & np.isfinite(s_ret) & (s_ret >= thr_pct)

            n_l = int(np.count_nonzero(clean_l))
            n_s = int(np.count_nonzero(clean_s))
            row[f"clean_long_{H}m"] = n_l
            row[f"clean_short_{H}m"] = n_s
            row[f"clean_total_{H}m"] = n_l + n_s

            # durations
            ld = l_delay[clean_l]
            sd = s_delay[clean_s]
            row[f"expected_delay_med_{H}m_long"] = float(np.nanmedian(ld)) if ld.size else np.nan
            row[f"expected_delay_med_{H}m_short"] = float(np.nanmedian(sd)) if sd.size else np.nan

            delays_pooled.extend(ld.tolist())
            delays_pooled.extend(sd.tolist())

            total_clean += (n_l + n_s)

        # pooled delay
        row["expected_delay_med_pooled"] = float(np.nanmedian(np.array(delays_pooled))) if len(delays_pooled) else np.nan

        # density per 1k minutes to normalize days with gaps
        row["clean_density_total_per_1kmin"] = (total_clean / max(1, n_minutes)) * 1000.0

        # buy/sell prevalence using the broadest horizon (last one) or all summed
        n_l_all = sum(int(row.get(f"clean_long_{H}m", 0)) for H in horizons)
        n_s_all = sum(int(row.get(f"clean_short_{H}m", 0)) for H in horizons)
        denom = n_l_all + n_s_all
        row["prevalence"] = float((n_l_all - n_s_all) / denom) if denom > 0 else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate daily regime features from symmetric oracle labels and market features")
    ap.add_argument("--oracle-labels", required=True, help="Path to oracle_labels_sym_*.parquet or .csv")
    ap.add_argument("--market-csv", required=True, help="Raw market CSV with timestamp,open,high,low,close,volume")
    ap.add_argument("--thr-pct", type=float, default=0.10, help="Threshold (in pct) for 'clean' opportunity, default 0.10%")
    ap.add_argument("--hold-mins", default="5,15,30", help="Horizons to aggregate, e.g. 5,15,30")
    ap.add_argument("--out-dir", default="data/regimes", help="Output directory for daily features")
    ap.add_argument("--out-stem", default="daily_regime_features", help="Filename stem")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    horizons = [int(x) for x in str(args.hold_mins).split(",") if str(x).strip()]

    labels = read_labels(Path(args.oracle_labels))

    # Ensure we have all required columns for horizons
    needed = []
    for H in horizons:
        needed += [
            f"oracle_long_best_ret_pct_{H}m",
            f"oracle_long_best_delay_min_{H}m",
            f"oracle_long_has_profit_{H}m",
            f"oracle_short_best_ret_pct_{H}m",
            f"oracle_short_best_delay_min_{H}m",
            f"oracle_short_has_profit_{H}m",
        ]
    missing = [c for c in needed if c not in labels.columns]
    if missing:
        raise SystemExit(f"Missing columns in oracle labels: {missing}")

    daily = aggregate_daily(labels, horizons=horizons, thr_pct=float(args.thr_pct))

    # Attach context medians from market features
    mkt = minute_bars(Path(args.market_csv))
    ctx = daily_context_features(mkt)
    out = daily.merge(ctx, on="date", how="left")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    out_parquet = out_dir / f"{args.out_stem}_{ts}.parquet"
    out_csv = out_dir / f"{args.out_stem}_{ts}.csv"
    out.to_parquet(out_parquet, index=False)
    out.to_csv(out_csv, index=False)

    print(f"Days: {len(out)}")
    if len(out):
        print("Preview:\n", out.head(3))
    print("Wrote:", out_parquet)
    print("Also CSV:", out_csv)


if __name__ == "__main__":
    main()
