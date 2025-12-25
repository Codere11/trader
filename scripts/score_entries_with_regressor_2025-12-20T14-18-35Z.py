#!/usr/bin/env python3
"""
Score historical minute bars with the current entry regressor and select top 0.1% entries per day.
Outputs:
- data/exit_regression/entries_scored_<ts>.csv
- data/exit_regression/entries_selected_top0p1pct_<ts>.csv
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception as e:
    raise SystemExit("tqdm is required: pip install tqdm") from e

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
    ap = argparse.ArgumentParser(description="Score entries with entry regressor and select top 0.1% per day")
    ap.add_argument("--market-csv", default="data/btc_profitability_analysis_filtered.csv")
    ap.add_argument("--entry-model", default=str(REPO_ROOT / "models" / "entry_regressor_btcusdt_2025-12-20T13-47-55Z.joblib"))
    ap.add_argument("--out-dir", default="data/exit_regression")
    ap.add_argument("--frac-per-day", type=float, default=0.001, help="Top fraction per day to keep (0.001 = 0.1%)")
    args = ap.parse_args()

    ts = _now_ts()

    mkt = minute_bars(Path(args.market_csv))

    # Compute features (past-only) aligned to minutes
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

    # Reconstruct context features if present in feature list
    base = feats[[c for c in FEATURES if c in feats.columns]]
    ctx_mean = base.rolling(pre_min, min_periods=pre_min).mean().add_suffix(f"_mean_{pre_min}m")
    full = pd.concat([base, ctx_mean], axis=1)

    # Ensure all required columns exist
    for c in feat_names:
        if c not in full.columns:
            full[c] = np.nan
    X = full[feat_names].to_numpy(dtype=np.float32)

    # Predict scores
    if hasattr(model, "predict"):
        score = model.predict(X)
    elif hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]
    else:
        raise SystemExit("Unsupported model type for scoring")

    df = pd.DataFrame({
        "timestamp": feats["ts_min"].to_numpy(),
        "score": score,
    })
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scored_path = out_dir / f"entries_scored_{ts}.csv"
    df.to_csv(scored_path, index=False)

    # Top fraction per day selection
    def top_frac(g: pd.DataFrame) -> pd.DataFrame:
        n = len(g)
        k = max(1, int(round(n * float(args.frac_per_day))))
        return g.nlargest(k, columns="score")

    selected = df.groupby("date", group_keys=False).apply(top_frac).reset_index(drop=True)
    selected.rename(columns={"timestamp": "entry_time"}, inplace=True)

    sel_path = out_dir / f"entries_selected_top0p1pct_{ts}.csv"
    selected.to_csv(sel_path, index=False)

    print("Scored entries:", scored_path)
    print("Selected entries (per-day top 0.1%):", sel_path)


if __name__ == "__main__":
    main()
