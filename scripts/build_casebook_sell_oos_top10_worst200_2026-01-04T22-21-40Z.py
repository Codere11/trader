#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T22:21:40Z
"""Build a casebook for OOS top-10% SELL selection failures.

Inputs
- OOS analysis output dir (from analyze_sell_oos_goodbad_breakdown_*):
  - oos_selected_meta_*.parquet (timestamp/pred/y_sell/label/row_idx/sel_pos)
  - oos_selected_features_*.parquet (rows aligned with meta via sel_pos)
- Bars parquet for OHLCV extraction

What it does
- Load OOS top-10% selected set.
- Take the worst N entries by y_sell (default 200).
- For each entry, extract an OHLCV window:
  [t - pre_minutes, ..., t + post_minutes] (default 120 before, 30 after).
- Write:
  - worst_entries_<ts>.csv (entry snapshot: timestamp/pred/y_sell/label + key features)
  - worst_entries_features_<ts>.parquet (all model input features for the worst N)
  - ohlcv_windows_<ts>.parquet (long-form windowed OHLCV with trade_id and rel_min)
  - summary_<ts>.json

Notes
- This is a diagnostic artifact; it does not simulate execution.
- Window extraction uses the dataset index (1-minute bars).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build casebook for OOS top-10% worst trades")
    ap.add_argument(
        "--oos-dir",
        default="data/analysis_sell_oos_breakdown_2026-01-04T22-11-31Z",
        help="Directory produced by analyze_sell_oos_goodbad_breakdown_*",
    )
    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="Bars parquet (needs at least timestamp, open, high, low, close, volume)",
    )
    ap.add_argument("--worst-n", type=int, default=200)
    ap.add_argument("--pre-minutes", type=int, default=120)
    ap.add_argument("--post-minutes", type=int, default=30)

    ap.add_argument("--out-dir", default="data")

    args = ap.parse_args()

    oos_dir = Path(args.oos_dir)
    if not oos_dir.exists():
        raise SystemExit(f"oos dir not found: {oos_dir}")

    meta_files = sorted(oos_dir.glob("oos_selected_meta_*.parquet"))
    feat_files = sorted(oos_dir.glob("oos_selected_features_*.parquet"))
    if not meta_files or not feat_files:
        raise SystemExit(f"Missing oos_selected_meta_*.parquet or oos_selected_features_*.parquet in {oos_dir}")

    meta_path = meta_files[-1]
    feat_path = feat_files[-1]

    print("Loading meta:", meta_path, flush=True)
    M = pd.read_parquet(meta_path)
    print("Loading features:", feat_path, flush=True)
    X = pd.read_parquet(feat_path)

    if len(M) != len(X):
        raise SystemExit(f"meta/features length mismatch: {len(M)} vs {len(X)}")

    # sanitize
    M = M.copy()
    M["timestamp"] = pd.to_datetime(M["timestamp"], utc=True, errors="coerce")
    M["y_sell"] = pd.to_numeric(M["y_sell"], errors="coerce")
    M["pred"] = pd.to_numeric(M["pred"], errors="coerce")
    M["row_idx"] = pd.to_numeric(M.get("row_idx"), errors="coerce").astype("Int64")

    worst_n = int(args.worst_n)
    if worst_n <= 0:
        raise SystemExit("--worst-n must be > 0")

    Mw = M.nsmallest(worst_n, "y_sell").reset_index(drop=True)
    # align feature rows by selection position: original outputs are row-aligned, and Mw rows contain sel_pos.
    if "sel_pos" not in Mw.columns:
        raise SystemExit("Expected sel_pos in meta parquet")

    sel_pos = pd.to_numeric(Mw["sel_pos"], errors="coerce").astype(int).to_numpy()
    Xw = X.iloc[sel_pos].reset_index(drop=True)

    # Load OHLCV bars
    bars_path = Path(args.bars)
    if not bars_path.exists():
        raise SystemExit(f"bars not found: {bars_path}")

    need_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    print("Loading bars:", bars_path, flush=True)
    bars = pd.read_parquet(bars_path, columns=need_cols)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    if not bars["timestamp"].is_monotonic_increasing:
        bars = bars.sort_values("timestamp").reset_index(drop=True)

    n_bars = int(len(bars))

    pre = int(args.pre_minutes)
    post = int(args.post_minutes)

    # choose key features for the CSV snapshot
    key_feats = [
        # compression / microstructure
        "range_norm_5m__last",
        "range_norm_5m__mean5",
        "range_norm_5m__min5",
        "range_norm_5m__max5",
        "range_5m__last",
        "ret_1m_pct__std5",
        "px_close_norm_pct__std5",
        # regime structure
        "range_60m__std5",
        "range_120m__mean5",
        "vol_std_60m__mean5",
        "vol_std_120m__mean5",
        "mom_60m_pct__mean5",
        "vwap_dev_60m__mean5",
        # state
        "close_pos_in_120m_range__last",
        "close_dd_from_120m_max_pct__last",
        "close_bounce_from_120m_min_pct__last",
        # features known to have stability issues in OOS
        "vol_log1p__range5",
        "mom_5m_pct__mean5",
        "macd__mean5",
    ]
    key_feats = [c for c in key_feats if c in Xw.columns]

    entry_snapshot = pd.concat(
        [
            Mw[["sel_pos", "row_idx", "timestamp", "pred", "y_sell", "label"]].reset_index(drop=True),
            Xw[key_feats].reset_index(drop=True),
        ],
        axis=1,
    )

    # Build windows
    print(f"Extracting OHLCV windows for worst {len(Mw)} entries (pre={pre} post={post})...", flush=True)
    rows = []
    for i in range(len(Mw)):
        ridx = int(Mw.loc[i, "row_idx"]) if pd.notna(Mw.loc[i, "row_idx"]) else None
        if ridx is None:
            continue
        start = max(0, ridx - pre)
        end = min(n_bars - 1, ridx + post)
        w = bars.iloc[start : end + 1].copy()
        w["trade_id"] = i
        w["entry_row_idx"] = ridx
        w["entry_timestamp"] = Mw.loc[i, "timestamp"]
        w["entry_pred"] = Mw.loc[i, "pred"]
        w["entry_y_sell"] = Mw.loc[i, "y_sell"]
        w["entry_label"] = Mw.loc[i, "label"]
        # rel minute (0 at entry)
        w["rel_min"] = (w.index.to_numpy() - ridx).astype(int)
        rows.append(w)

    windows = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()

    ts = now_ts()
    out_root = Path(args.out_dir) / f"casebook_sell_oos_top10_worst{len(Mw)}_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    (out_root / f"worst_entries_{ts}.csv").write_text(entry_snapshot.to_csv(index=False), encoding="utf-8")
    Xw.to_parquet(out_root / f"worst_entries_features_{ts}.parquet", index=False)
    if not windows.empty:
        windows.to_parquet(out_root / f"ohlcv_windows_{ts}.parquet", index=False)

    summary = {
        "created_utc": ts,
        "oos_dir": str(oos_dir),
        "meta_file": meta_path.name,
        "features_file": feat_path.name,
        "bars": str(bars_path),
        "worst_n": int(len(Mw)),
        "pre_minutes": pre,
        "post_minutes": post,
        "windows_rows": int(len(windows)),
        "windows_trades": int(windows["trade_id"].nunique()) if not windows.empty and "trade_id" in windows.columns else 0,
        "entry_y_sell_min": float(Mw["y_sell"].min()),
        "entry_y_sell_p10": float(Mw["y_sell"].quantile(0.10)),
        "entry_y_sell_median": float(Mw["y_sell"].median()),
        "entry_y_sell_max": float(Mw["y_sell"].max()),
        "entry_pred_min": float(Mw["pred"].min()),
        "entry_pred_p90": float(Mw["pred"].quantile(0.90)),
        "entry_pred_max": float(Mw["pred"].max()),
        "labels": Mw["label"].value_counts().to_dict(),
        "key_feature_cols_in_csv": key_feats,
    }
    (out_root / f"summary_{ts}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Wrote", out_root, flush=True)


if __name__ == "__main__":
    main()
