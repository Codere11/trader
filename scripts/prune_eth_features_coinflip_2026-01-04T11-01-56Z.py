#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T11:01:56Z
"""Prune ETH datasets: drop features that are same or worse than coin flip.

Important: "coin flip" only makes sense relative to a prediction task.
Here we define a concrete task using the ETH entry-trade dataset:
  y = 1{ trade_net_return_pct >= 0.2 }
Scoring: per-feature ROC AUC on a time split (train first 80%, test last 20%).
Pruning rule: DROP feature if AUC_test <= 0.5.

This script:
- Computes per-side (BUY/SELL) keep-lists for entry-precontext descriptor features.
- Writes pruned versions of:
  - top-10%-outliers-stripped entry datasets (with mixed returns)
  - top-10%-outliers-stripped AND ret>=0.2 datasets (apply same keep-list)
- Prunes the full ETH 1m features dataset by scoring its engineered columns (excluding OHLCV)
  at trade entry timestamps; kept only if AUC_test > 0.5 for BOTH BUY and SELL.

Outputs (timestamped)
- data/eth_pruned_coinflip_<ts>/
  - keep_drop_report_<ts>.json
  - entry_keep_features_BUY_<ts>.json
  - entry_keep_features_SELL_<ts>.json
  - entry_feature_scores_BUY_<ts>.csv
  - entry_feature_scores_SELL_<ts>.csv
  - BUY_entry_inliers_pruned_<ts>.parquet
  - SELL_entry_inliers_pruned_<ts>.parquet
  - BUY_entry_profitable_pruned_<ts>.parquet
  - SELL_entry_profitable_pruned_<ts>.parquet
  - dydx_ETH-USD_1MIN_features_full_pruned_<ts>.parquet
  - bars_feature_scores_BUY_<ts>.csv
  - bars_feature_scores_SELL_<ts>.csv
  - bars_keep_features_intersection_<ts>.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def time_split_idx(n: int, test_frac: float) -> int:
    return int(n * (1.0 - float(test_frac)))


def numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    # ML feature columns = numeric, non-metadata.
    meta = {
        "trade_row",
        "trade_date",
        "trade_entry_time",
        "trade_exit_time",
        "trade_duration_min",
        "trade_side",
        "trade_net_return_pct",
        "outlier_score",
        "nn_dist",
        "missing_any",
        "missing_close_n",
        "zone_id",
        "is_outlier",
    }
    cols = []
    for c in df.columns:
        if c in meta:
            continue
        if c.startswith("trade_"):
            continue
        if df[c].dtype.kind not in "fc":
            continue
        cols.append(c)
    return cols


def meta_cols_present(df: pd.DataFrame) -> list[str]:
    want = [
        "trade_row",
        "trade_date",
        "trade_entry_time",
        "trade_exit_time",
        "trade_duration_min",
        "trade_side",
        "trade_net_return_pct",
    ]
    return [c for c in want if c in df.columns]


def score_univariate_auc(
    *,
    df: pd.DataFrame,
    time_col: str,
    y_col: str,
    feat_cols: list[str],
    test_frac: float,
) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)

    y_raw = pd.to_numeric(df[y_col], errors="coerce").to_numpy(np.float64)
    # y is expected already binary here

    n = len(df)
    split = time_split_idx(n, test_frac)

    y_tr, y_te = y_raw[:split], y_raw[split:]

    rows = []

    for c in feat_cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)
        x_tr, x_te = x[:split], x[split:]

        # NaN -> train median
        med = float(np.nanmedian(x_tr)) if np.isfinite(np.nanmedian(x_tr)) else 0.0
        x_tr = np.where(np.isfinite(x_tr), x_tr, med)
        x_te = np.where(np.isfinite(x_te), x_te, med)

        # AUC undefined if test y has single class
        auc = float("nan")
        if len(np.unique(y_te)) == 2:
            try:
                auc = float(roc_auc_score(y_te, x_te))
            except Exception:
                auc = float("nan")

        rows.append(
            {
                "feature": c,
                "auc_test": auc,
            }
        )

    out = pd.DataFrame(rows).sort_values("auc_test", ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Prune ETH datasets by coin-flip feature performance")
    ap.add_argument(
        "--entry-inliers-root",
        default="data/entry_precontext_clean_top10pct_2026-01-04T10-46-32Z",
        help="Root with BUY_clean_top10pct_*.parquet and SELL_clean_top10pct_*.parquet (mixed returns)",
    )
    ap.add_argument(
        "--entry-prof-root",
        default="data/entry_precontext_clean_top10pct_retge0p2_2026-01-04T10-47-43Z",
        help="Root with BUY_clean_top10pct_retge0p2_*.parquet and SELL... (profitable-only)",
    )
    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="Full ETH 1m candles+features parquet",
    )
    ap.add_argument("--ret-thresh", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--min-auc", type=float, default=0.5, help="Keep only features with AUC_test > min_auc")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    ts = now_ts()
    out_root = Path(args.out_dir) / f"eth_pruned_coinflip_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "created_utc": ts,
        "entry_inliers_root": str(args.entry_inliers_root),
        "entry_prof_root": str(args.entry_prof_root),
        "bars": str(args.bars),
        "ret_thresh": float(args.ret_thresh),
        "test_frac": float(args.test_frac),
        "min_auc": float(args.min_auc),
        "notes": "AUC computed on y=(trade_net_return_pct>=ret_thresh) using top-10%-outliers-stripped entry dataset; features kept iff AUC_test > min_auc (no sign-flipping).",
    }

    # --- Entry datasets (BUY/SELL) ---
    keep_entry: dict[str, list[str]] = {}
    scores_entry: dict[str, pd.DataFrame] = {}

    for side in ["BUY", "SELL"]:
        in_root = Path(args.entry_inliers_root)
        in_files = sorted(in_root.glob(f"{side}_clean_top10pct_*.parquet"))
        if not in_files:
            raise SystemExit(f"Missing entry inliers parquet for {side} under {in_root}")
        in_path = in_files[-1]

        df = pd.read_parquet(in_path)
        df["trade_side"] = df["trade_side"].astype(str).str.upper()
        df = df[df["trade_side"] == side].copy()
        df["trade_entry_time"] = pd.to_datetime(df["trade_entry_time"], utc=True, errors="coerce")
        df = df.sort_values("trade_entry_time").reset_index(drop=True)

        # Build binary label
        y = (pd.to_numeric(df["trade_net_return_pct"], errors="coerce") >= float(args.ret_thresh)).astype(int)
        if y.nunique() < 2:
            raise SystemExit(f"{side}: label has <2 classes in inliers dataset; cannot prune by coin flip")
        df["_y_bin"] = y

        feat_cols = numeric_feature_cols(df)
        sc = score_univariate_auc(
            df=df,
            time_col="trade_entry_time",
            y_col="_y_bin",
            feat_cols=feat_cols,
            test_frac=float(args.test_frac),
        )

        keep = sc[(pd.to_numeric(sc["auc_test"], errors="coerce") > float(args.min_auc))]["feature"].tolist()

        keep_entry[side] = keep
        scores_entry[side] = sc

        report[f"entry_{side}"] = {
            "inliers_path": str(in_path),
            "n_rows": int(len(df)),
            "n_feat_numeric": int(len(feat_cols)),
            "n_keep": int(len(keep)),
            "n_drop": int(len(feat_cols) - len(keep)),
        }

        # save scores + keep list
        sc.to_csv(out_root / f"entry_feature_scores_{side}_{ts}.csv", index=False)
        (out_root / f"entry_keep_features_{side}_{ts}.json").write_text(json.dumps(keep, indent=2), encoding="utf-8")

        # write pruned inliers dataset
        meta_cols = meta_cols_present(df)
        pruned = df[meta_cols + keep].copy()
        pruned.to_parquet(out_root / f"{side}_entry_inliers_pruned_{ts}.parquet", index=False)

        # write pruned profitable dataset by applying the same keep list
        prof_root = Path(args.entry_prof_root)
        prof_files = sorted(prof_root.glob(f"{side}_clean_top10pct_retge0p2_*.parquet"))
        if not prof_files:
            raise SystemExit(f"Missing entry profitable parquet for {side} under {prof_root}")
        prof_path = prof_files[-1]
        dfp = pd.read_parquet(prof_path)
        dfp["trade_side"] = dfp["trade_side"].astype(str).str.upper()
        dfp = dfp[dfp["trade_side"] == side].copy()

        meta_cols_p = meta_cols_present(dfp)
        keep_here = [c for c in keep if c in dfp.columns]
        pruned_p = dfp[meta_cols_p + keep_here].copy()
        pruned_p.to_parquet(out_root / f"{side}_entry_profitable_pruned_{ts}.parquet", index=False)

    # --- Bars dataset pruning (engineered cols only) ---
    bars = pd.read_parquet(args.bars)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    bars = bars.sort_values("timestamp").set_index("timestamp")

    always_keep = [c for c in ["open", "high", "low", "close", "volume"] if c in bars.columns]
    candidate = [c for c in bars.columns if c not in set(always_keep)]

    bars_scores = {}
    bars_keep_sets = {}

    for side in ["BUY", "SELL"]:
        # use the corresponding inliers dataset as scoring source
        in_root = Path(args.entry_inliers_root)
        in_path = sorted(in_root.glob(f"{side}_clean_top10pct_*.parquet"))[-1]
        tdf = pd.read_parquet(in_path)
        tdf["trade_side"] = tdf["trade_side"].astype(str).str.upper()
        tdf = tdf[tdf["trade_side"] == side].copy()
        tdf["trade_entry_time"] = pd.to_datetime(tdf["trade_entry_time"], utc=True, errors="coerce")
        tdf = tdf.sort_values("trade_entry_time").set_index("trade_entry_time")
        y = (pd.to_numeric(tdf["trade_net_return_pct"], errors="coerce") >= float(args.ret_thresh)).astype(int)

        joined = pd.DataFrame({"_y_bin": y}).join(bars[candidate], how="left")

        feat_cols = [c for c in candidate if joined[c].dtype.kind in "fc"]
        sc = score_univariate_auc(
            df=joined.reset_index().rename(columns={"index": "trade_entry_time"}),
            time_col="trade_entry_time",
            y_col="_y_bin",
            feat_cols=feat_cols,
            test_frac=float(args.test_frac),
        )
        bars_scores[side] = sc
        keep = sc[(pd.to_numeric(sc["auc_test"], errors="coerce") > float(args.min_auc))]["feature"].tolist()
        bars_keep_sets[side] = set(keep)

        sc.to_csv(out_root / f"bars_feature_scores_{side}_{ts}.csv", index=False)

    # Keep intersection (must beat coin flip on BOTH sides)
    keep_bars = sorted(list(bars_keep_sets["BUY"].intersection(bars_keep_sets["SELL"])))
    (out_root / f"bars_keep_features_intersection_{ts}.json").write_text(json.dumps(keep_bars, indent=2), encoding="utf-8")

    # Write pruned bars parquet
    out_bars = bars.reset_index()[["timestamp"] + always_keep + keep_bars].copy()
    out_bars.to_parquet(out_root / f"dydx_ETH-USD_1MIN_features_full_pruned_{ts}.parquet", index=False)

    report["bars"] = {
        "always_keep": always_keep,
        "candidate_scored": int(len(candidate)),
        "keep_intersection": int(len(keep_bars)),
        "dropped": int(len(candidate) - len(keep_bars)),
    }

    (out_root / f"keep_drop_report_{ts}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Done.")
    print("Outputs:", out_root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
