#!/usr/bin/env python3
from __future__ import annotations
"""
Aggregate daily regime features using a noise-based cleanliness definition:
"lack of noise in the 5-minute window before entry".

A minute qualifies as a clean opportunity for side S and horizon H iff:
  (a) Oracle indicates a profitable exit within [1,H] (net of fees) for side S, and
  (b) The prior 5-minute window (t-5..t-1) is "quiet" by several metrics:
      - range_norm_5m <= max_range_norm_5m
      - vol_std_5m   <= max_vol_std_5m
      - sign_flips_5 <= max_sign_flips_5 (sign changes of ret_1m over last 5)
      - abs(vwap_dev_5m) <= max_vwap_dev_abs_5m
      - wick_ratio_5m <= max_wick_ratio_5m (median over last 5 of (high-low)/max(1e-6, abs(close-open)))

These thresholds are tunable via CLI flags.

Inputs:
- --oracle-labels: parquet/csv from build_oracle_labels_sym_*.py
- --market-csv: raw market CSV

Outputs:
- data/regimes/daily_regime_features_noise_<TS>.{parquet,csv}
"""
import argparse
import datetime as dt
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
from binance_adapter.crossref_oracle_vs_agent_patterns import compute_feature_frame


def read_labels(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["timestamp"])  # timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
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


def add_noise_metrics(mkt: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    df = mkt.copy()
    df["ret_1m_pct"] = df["close"].pct_change(1) * 100.0
    # sign flips in last 5 returns (t-5..t-1)
    sgn = np.sign(df["ret_1m_pct"].to_numpy(np.float64))
    flips = np.zeros(len(df), dtype=np.int16)
    for i in range(5, len(df)):
        window = sgn[i-5:i]
        cnt = 0
        for k in range(1, 5):
            if window[k] == 0 or window[k-1] == 0:
                continue
            if window[k] != window[k-1]:
                cnt += 1
        flips[i] = cnt
    df["sign_flips_5"] = flips

    # wick ratio per bar, then median over last 5 bars
    oc = (df["close"] - df["open"]).abs() + 1e-9
    wick = (df["high"] - df["low"]) / oc
    df["wick_ratio_5m"] = wick.rolling(5, min_periods=5).median()

    # join features for range_norm_5m, vol_std_5m, vwap_dev_5m
    feats2 = feats.rename(columns={"ts_min":"timestamp"})[["timestamp","range_norm_5m","vol_std_5m","vwap_dev_5m"]]
    out = df.merge(feats2, on="timestamp", how="left")
    return out


def aggregate_daily_noise(labels: pd.DataFrame, noise: pd.DataFrame, horizons: list[int],
                           max_range_norm_5m: float, max_vol_std_5m: float, max_sign_flips_5: int,
                           max_vwap_dev_abs_5m: float, max_wick_ratio_5m: float) -> pd.DataFrame:
    # Align on timestamp
    labels = labels.sort_values("timestamp").reset_index(drop=True)
    noise = noise.sort_values("timestamp").reset_index(drop=True)
    # Inner merge to ensure alignment
    df = labels.merge(noise[["timestamp","range_norm_5m","vol_std_5m","vwap_dev_5m","sign_flips_5","wick_ratio_5m"]], on="timestamp", how="inner")
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    # noise cleanliness mask (prior 5 ensures NaNs early -> False)
    clean_noise = (
        (df["range_norm_5m"] <= max_range_norm_5m) &
        (df["vol_std_5m"]   <= max_vol_std_5m) &
        (df["sign_flips_5"] <= int(max_sign_flips_5)) &
        (df["vwap_dev_5m"].abs() <= max_vwap_dev_abs_5m) &
        (df["wick_ratio_5m"] <= max_wick_ratio_5m)
    )
    df["clean_noise"] = clean_noise.astype(bool)

    rows = []
    for d, g in df.groupby("date"):
        row: dict[str, float | int | object] = {"date": d, "minutes": int(len(g))}
        total_clean = 0
        delays = []
        for H in horizons:
            # side-specific labels
            l_has = g.get(f"oracle_long_has_profit_{H}m").to_numpy(bool, copy=False)
            s_has = g.get(f"oracle_short_has_profit_{H}m").to_numpy(bool, copy=False)
            l_delay = g.get(f"oracle_long_best_delay_min_{H}m").to_numpy(np.float64, copy=False)
            s_delay = g.get(f"oracle_short_best_delay_min_{H}m").to_numpy(np.float64, copy=False)

            cn = g["clean_noise"].to_numpy()
            ok_l = l_has & cn
            ok_s = s_has & cn

            n_l = int(np.count_nonzero(ok_l))
            n_s = int(np.count_nonzero(ok_s))
            row[f"clean_long_{H}m"] = n_l
            row[f"clean_short_{H}m"] = n_s
            row[f"clean_total_{H}m"] = n_l + n_s

            delays.extend(l_delay[ok_l].tolist())
            delays.extend(s_delay[ok_s].tolist())

            total_clean += (n_l + n_s)

        row["expected_delay_med_pooled"] = float(np.nanmedian(np.array(delays))) if delays else np.nan
        row["clean_density_total_per_1kmin"] = (total_clean / max(1, int(len(g)))) * 1000.0
        n_l_all = sum(int(row.get(f"clean_long_{H}m", 0)) for H in horizons)
        n_s_all = sum(int(row.get(f"clean_short_{H}m", 0)) for H in horizons)
        denom = n_l_all + n_s_all
        row["prevalence"] = float((n_l_all - n_s_all) / denom) if denom > 0 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Daily aggregation using noise-based cleanliness (5m pre-entry quietness)")
    ap.add_argument("--oracle-labels", required=True)
    ap.add_argument("--market-csv", required=True)
    ap.add_argument("--hold-mins", default="5,15,30")
    # Cleanliness thresholds
    ap.add_argument("--max-range-norm-5m", type=float, default=0.003)
    ap.add_argument("--max-vol-std-5m", type=float, default=12.0)
    ap.add_argument("--max-sign-flips-5", type=int, default=1)
    ap.add_argument("--max-vwap-dev-abs-5m", type=float, default=0.20)
    ap.add_argument("--max-wick-ratio-5m", type=float, default=3.0)
    ap.add_argument("--out-dir", default="data/regimes")
    ap.add_argument("--out-stem", default="daily_regime_features_noise")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    horizons = [int(x) for x in str(args.hold_mins).split(',') if str(x).strip()]

    labels = read_labels(Path(args.oracle_labels))

    mkt = minute_bars(Path(args.market_csv))
    feats = compute_feature_frame(mkt.rename(columns={"timestamp":"ts_min"}))
    noise = add_noise_metrics(mkt, feats)

    out = aggregate_daily_noise(
        labels,
        noise,
        horizons,
        float(args.max_range_norm_5m),
        float(args.max_vol_std_5m),
        int(args.max_sign_flips_5),
        float(args.max_vwap_dev_abs_5m),
        float(args.max_wick_ratio_5m),
    )

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    out_parquet = out_dir / f"{args.out_stem}_{ts}.parquet"
    out_csv = out_dir / f"{args.out_stem}_{ts}.csv"
    out.to_parquet(out_parquet, index=False)
    out.to_csv(out_csv, index=False)

    print(f"Days: {len(out)}\nWrote: {out_parquet}\nAlso CSV: {out_csv}")


if __name__ == "__main__":
    main()
