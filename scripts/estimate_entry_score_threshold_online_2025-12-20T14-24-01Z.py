#!/usr/bin/env python3
"""
Estimate a causal per-day entry score threshold using only prior days.
Goal: choose threshold so expected selection rate ≈ target_frac (default 0.001 => 0.1%).
Outputs:
- data/exit_regression/entry_score_thresholds_online_<ts>.csv with columns: date, threshold, n_minutes_prior, est_rate

Procedure:
- Score all minutes with entry regressor.
- For each day D in chronological order, compute threshold = quantile(1 - target_frac) of scores from all days < D.
- For the first warmup_days, use a fixed percentile from the earliest available chunk.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import FEATURES, compute_feature_frame

_MKT_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def minute_bars(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, parse_dates=["timestamp"], usecols=_MKT_COLS)
    if not raw["timestamp"].is_monotonic_increasing:
        raw = raw.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    raw["ts_min"] = raw["timestamp"].dt.floor("min")
    g = (
        raw.groupby("ts_min")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"))
        .reset_index()
        .rename(columns={"ts_min": "timestamp"})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate online per-day score thresholds from prior days only")
    ap.add_argument("--market-csv", default="data/btc_profitability_analysis_filtered.csv")
    ap.add_argument("--entry-model", default=str(REPO_ROOT / "models" / "entry_regressor_btcusdt_2025-12-20T13-47-55Z.joblib"))
    ap.add_argument("--target-frac", type=float, default=0.001)
    ap.add_argument("--warmup-days", type=int, default=7)
    ap.add_argument("--out-dir", default="data/exit_regression")
    args = ap.parse_args()

    ts = _now_ts()
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mkt = minute_bars(Path(args.market_csv))
    feats = compute_feature_frame(mkt.rename(columns={"timestamp": "ts_min"}))

    payload = joblib.load(args.entry_model)
    model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    feat_names = None
    pre_min = None
    if isinstance(payload, dict) and "features" in payload:
        feat_names = list(payload["features"])
        ctx = payload.get("context", {})
        pre_min = int(ctx.get("pre_min", 5))
    else:
        feat_names = list(FEATURES)
        pre_min = 5

    base = feats[[c for c in FEATURES if c in feats.columns]]
    ctx_mean = base.rolling(pre_min, min_periods=pre_min).mean().add_suffix(f"_mean_{pre_min}m")
    full = pd.concat([base, ctx_mean], axis=1)
    for c in feat_names:
        if c not in full.columns:
            full[c] = np.nan
    X = full[feat_names].to_numpy(dtype=np.float32)

    score = model.predict(X) if hasattr(model, "predict") else model.predict_proba(X)[:, 1]

    df = pd.DataFrame({
        "timestamp": feats["ts_min"].to_numpy(),
        "score": score,
    })
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    days = sorted(df["date"].unique())
    res = []
    prior_scores = []

    for i, d in enumerate(days):
        g_prior = df[df["date"] < d]
        if len(g_prior) == 0:
            res.append({"date": d, "threshold": np.nan, "n_minutes_prior": 0, "est_rate": np.nan})
            continue
        q = float(np.quantile(g_prior["score"].to_numpy(), 1.0 - float(args.target_frac)))
        res.append({"date": d, "threshold": q, "n_minutes_prior": int(len(g_prior)), "est_rate": float(args.target_frac)})

    out = pd.DataFrame(res)
    out_path = out_dir / f"entry_score_thresholds_online_{ts}.csv"
    out.to_csv(out_path, index=False)
    print("Online thresholds:", out_path)


if __name__ == "__main__":
    main()
