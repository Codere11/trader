#!/usr/bin/env python3
"""
Build an oracle-exit labeled dataset for training a new exit timing model.
Labels: for each entry, the minute t (>=3 and <= hold_min) that maximizes since_entry_return_pct
        is labeled is_oracle_exit=1; all other minutes are 0.
Features: EXIT_FEATURE_COLS at minute t, plus the same features 1m and 3m before t.
Outputs (timestamped):
- data/exit_oracle_training/exit_oracle_dataset_<ts>.parquet
- data/exit_oracle_training/exit_oracle_dataset_sample_<ts>.csv (small sample)

No future leakage in features (they are computed at/behind time t). Labels use future info (oracle) for training only.
"""
from __future__ import annotations

import argparse
from datetime import timezone, datetime
from pathlib import Path
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Reuse constants from evaluator
EXIT_FEATURE_COLS = [
    "elapsed_min",
    "since_entry_return_pct",
    "prob_enter_at_entry",
    "daily_pred_high_profit",
    "momentum_3m",
    "momentum_5m",
    "momentum_10m",
    "volatility_3m",
    "volatility_5m",
    "volatility_10m",
    "drawdown_from_max_pct",
    "since_entry_cummax",
    "mins_since_peak",
]

ENTRY_FEATURES = [
    "ret_1m_pct","mom_3m_pct","mom_5m_pct","vol_std_5m","range_5m","range_norm_5m",
    "slope_ols_5m","rsi_14","macd","macd_hist","vwap_dev_5m","last3_same_sign",
]

# Import compute_feature_frame for entry features
from binance_adapter.crossref_oracle_vs_agent_patterns import compute_feature_frame  # type: ignore


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market-csv", default="data/btc_profitability_analysis_filtered.csv")
    ap.add_argument("--rule-trades", default="data/pattern_runs/rule_best/rule_rule__rn0.0015__vw0.000__mn1__m30.050__vwmin0.020__sl5.0__rsi55.0.parquet")
    ap.add_argument("--entry-models-dir", default="models")
    ap.add_argument("--entry-stem", default="rule_best_ranker")
    ap.add_argument("--start-date", type=str, default="2022-01-01")
    ap.add_argument("--end-date", type=str, default="2025-12-18")
    ap.add_argument("--entry-threshold", type=float, default=0.52)
    ap.add_argument("--k-per-day", type=int, default=3)
    ap.add_argument("--hold-min", type=int, default=8)
    ap.add_argument("--fee-rate", type=float, default=0.001)
    # Optional entry filters to align with production regime
    ap.add_argument("--min-vol-std-5m", type=float, default=10.0)
    ap.add_argument("--min-range-norm-5m", type=float, default=0.002)
    ap.add_argument("--rsi-min", type=float, default=40.0)
    ap.add_argument("--rsi-max", type=float, default=60.0)
    ap.add_argument("--max-vwap-dev-abs", type=float, default=0.20)
    ap.add_argument("--allow-hours", type=str, default="0,3,9,10,19")
    return ap.parse_args()


def load_market(csv_path: Path, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["ts_min"] = pd.to_datetime(df["timestamp"]).dt.floor("min")
    agg = (
        df.groupby("ts_min")
        .agg(open=("open","first"), high=("high","max"), low=("low","min"), close=("close","last"), volume=("volume","sum"))
        .reset_index()
    )
    if start is not None:
        agg = agg[agg["ts_min"] >= start]
    if end is not None:
        agg = agg[agg["ts_min"] <= end]
    return agg.sort_values("ts_min").reset_index(drop=True)


def map_ts_to_index(ts_sorted: np.ndarray, query: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(ts_sorted, query)
    ok = (pos < ts_sorted.size) & (ts_sorted[pos] == query)
    return np.where(ok, pos, -1).astype(np.int64)


def attach_entry_features(trades: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    ts_sorted = feat["ts_min"].to_numpy(dtype="datetime64[ns]")
    idx = map_ts_to_index(ts_sorted, trades["entry_ts"].to_numpy(dtype="datetime64[ns]"))
    ok = idx >= 0
    trades = trades.loc[ok].reset_index(drop=True)
    idx = idx[ok]
    feat_cols = feat.iloc[idx][ENTRY_FEATURES].reset_index(drop=True)
    trades = pd.concat([trades.reset_index(drop=True), feat_cols], axis=1)
    trades["entry_date"] = trades["entry_ts"].dt.date
    return trades


def load_entry_models(models_dir: Path, stem: str):
    try:
        import lightgbm as lgb
        import xgboost as xgb
    except Exception as e:
        raise SystemExit("LightGBM and XGBoost are required") from e
    lgb_path = models_dir / f"{stem}__entry_ranker_lgb.txt"
    xgb_path = models_dir / f"{stem}__entry_ranker_xgb.json"
    lgb_model = lgb.Booster(model_file=str(lgb_path)) if lgb_path.exists() else None
    xgb_model = None
    if xgb_path.exists():
        xgb_model = xgb.Booster(); xgb_model.load_model(str(xgb_path))
    return lgb_model, xgb_model


def score_entries(df: pd.DataFrame, lgb_model, xgb_model) -> pd.DataFrame:
    feats = df[ENTRY_FEATURES].fillna(0.0).astype(np.float32)
    scores = []
    if lgb_model is not None:
        cols = list(lgb_model.feature_name())
        Xl = feats.reindex(columns=cols, fill_value=0.0).to_numpy(dtype=float)
        scores.append(lgb_model.predict(Xl))
    if xgb_model is not None:
        Xx = feats[ENTRY_FEATURES].to_numpy(dtype=float)
        import xgboost as xgb
        dmat = xgb.DMatrix(Xx, feature_names=ENTRY_FEATURES)
        scores.append(xgb_model.predict(dmat))
    if not scores:
        raise SystemExit("No entry models found for scoring")
    out = df.copy(); out["score"] = np.mean(np.vstack(scores), axis=0)
    return out


def select_entries(df: pd.DataFrame, thr: float, k_per_day: int, hold_min: int, allow_hours: Optional[set[int]], filters: dict) -> pd.DataFrame:
    rows = []
    hold_delta = pd.Timedelta(minutes=hold_min)
    gdf = df.copy()
    mv = filters.get("min_vol_std_5m"); mrn = filters.get("min_range_norm_5m"); rmin = filters.get("rsi_min"); rmax = filters.get("rsi_max"); mvd = filters.get("max_vwap_dev_abs")
    if mv is not None: gdf = gdf[gdf["vol_std_5m"] >= float(mv)]
    if mrn is not None: gdf = gdf[gdf["range_norm_5m"] >= float(mrn)]
    if rmin is not None: gdf = gdf[gdf["rsi_14"] >= float(rmin)]
    if rmax is not None: gdf = gdf[gdf["rsi_14"] <= float(rmax)]
    if mvd is not None: gdf = gdf[gdf["vwap_dev_5m"].abs() <= float(mvd)]
    for date, g in gdf.groupby("entry_date"):
        g = g[g["score"] >= thr].sort_values("score", ascending=False)
        last_exit_ts: Optional[pd.Timestamp] = None
        chosen = []
        for _, row in g.iterrows():
            if allow_hours is not None and int(pd.to_datetime(row["entry_ts"]).hour) not in allow_hours:
                continue
            if last_exit_ts is not None and row["entry_ts"] < last_exit_ts:
                continue
            chosen.append(row)
            last_exit_ts = row["entry_ts"] + pd.Timedelta(minutes=hold_min)
            if len(chosen) >= k_per_day:
                break
        if chosen:
            rows.append(pd.DataFrame(chosen))
    if not rows:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(rows, ignore_index=True)


def build_exit_features(day_slice: pd.DataFrame, entry_idx: int, hold_min: int, fee_rate: float) -> pd.DataFrame:
    start_row = day_slice.iloc[entry_idx]
    entry_time = start_row["timestamp"]
    entry_price = float(start_row["close"])
    end_idx = min(len(day_slice) - 1, entry_idx + hold_min)
    seq = day_slice.iloc[entry_idx : end_idx + 1].copy().reset_index(drop=True)
    seq["elapsed_min"] = (seq["timestamp"] - entry_time).dt.total_seconds().div(60).astype(int)
    entry_cost = entry_price * (1 + fee_rate)
    exit_value = seq["close"] * (1 - fee_rate)
    seq["since_entry_return_pct"] = (exit_value / entry_cost - 1.0) * 100.0
    seq["prob_enter_at_entry"] = 0.5
    seq["daily_pred_high_profit"] = 1.0
    for w in [3, 5, 10]:
        seq[f"momentum_{w}m"] = seq["close"].pct_change(w) * 100.0
        seq[f"volatility_{w}m"] = seq["close"].rolling(w).std()
    seq["since_entry_cummax"] = seq["since_entry_return_pct"].cummax()
    seq["drawdown_from_max_pct"] = seq["since_entry_return_pct"] - seq["since_entry_cummax"]
    best = -1e9; best_i = 0; msp = []
    for i, v in enumerate(seq["since_entry_return_pct"].tolist()):
        if v > best: best = v; best_i = i
        msp.append(i - best_i)
    seq["mins_since_peak"] = msp
    return seq[["timestamp"] + EXIT_FEATURE_COLS].copy()


def main() -> None:
    args = parse_args()
    ts_out = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    start = pd.Timestamp(args.start_date) if args.start_date else None
    end = pd.Timestamp(args.end_date) if args.end_date else None

    market = load_market(Path(args.market_csv), start, end)
    market["date"] = market["ts_min"].dt.date
    day_lookup = {d: g.sort_values("ts_min").rename(columns={"ts_min":"timestamp"}).reset_index(drop=True) for d, g in market.groupby("date")}

    feat = compute_feature_frame(market)

    # Build candidate entries using rule trades + entry models
    trades = pd.read_parquet(args.rule_trades)
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"])
    trades["exit_ts"] = pd.to_datetime(trades["exit_ts"])
    if start is not None:
        trades = trades[trades["entry_ts"] >= start]
    if end is not None:
        trades = trades[trades["entry_ts"] <= end]

    trades = attach_entry_features(trades, feat)
    lgb_model, xgb_model = load_entry_models(Path(args.entry_models_dir), args.entry_stem)
    scored = score_entries(trades, lgb_model, xgb_model)

    allow_hours: Optional[set[int]] = None
    if args.allow_hours:
        allow_hours = {int(h) for h in str(args.allow_hours).split(',') if h.strip()}
    filters = dict(
        min_vol_std_5m=args.min_vol_std_5m,
        min_range_norm_5m=args.min_range_norm_5m,
        rsi_min=args.rsi_min,
        rsi_max=args.rsi_max,
        max_vwap_dev_abs=args.max_vwap_dev_abs,
    )
    selected = select_entries(scored, float(args.entry_threshold), int(args.k_per_day), int(args.hold_min), allow_hours, filters)

    # For each selected entry, build per-minute rows and label oracle exit
    rows = []
    for row in selected.itertuples(index=False):
        day = day_lookup.get(row.entry_date)
        if day is None or day.empty:
            continue
        ts_arr = day["timestamp"].to_numpy(dtype="datetime64[ns]")
        pos = int(np.searchsorted(ts_arr, np.datetime64(row.entry_ts)))
        if pos >= len(day):
            continue
        seq = build_exit_features(day, pos, int(args.hold_min), float(args.fee_rate))
        if seq.empty:
            continue
        # Oracle exit index within [3, hold_min]
        mask = (seq["elapsed_min"] >= 3) & (seq["elapsed_min"] <= int(args.hold_min))
        cand = seq.loc[mask]
        if cand.empty:
            continue
        oracle_rel_idx = int(cand["since_entry_return_pct"].values.argmax())
        oracle_abs_idx = int(cand.index[oracle_rel_idx])
        seq["is_oracle_exit"] = 0
        seq.loc[oracle_abs_idx, "is_oracle_exit"] = 1
        # Add 1m and 3m lagged features (within sequence) to capture pre-exit context
        for lag in (1, 3):
            lagged = seq[EXIT_FEATURE_COLS].shift(lag)
            lagged.columns = [f"{c}_m{lag}" for c in lagged.columns]
            seq = pd.concat([seq, lagged], axis=1)
        seq["entry_time"] = row.entry_ts
        seq["date"] = pd.to_datetime(row.entry_ts).date()
        rows.append(seq)

    if not rows:
        raise SystemExit("No rows produced for oracle exit dataset.")

    data = pd.concat(rows, ignore_index=True).dropna(subset=["since_entry_return_pct"]).reset_index(drop=True)

    out_dir = REPO_ROOT / "data" / "exit_oracle_training"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / f"exit_oracle_dataset_{ts_out}.parquet"
    out_csv = out_dir / f"exit_oracle_dataset_sample_{ts_out}.csv"
    data.to_parquet(out_parquet, index=False)
    data.sample(n=min(2000, len(data)), random_state=42).to_csv(out_csv, index=False)
    print("Saved:", out_parquet)
    print("Sample:", out_csv)


if __name__ == "__main__":
    main()
