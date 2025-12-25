#!/usr/bin/env python3
"""
Train a new exit timing classifier from the oracle-exit dataset.
- Time-based split: train on rows where entry_time < holdout_start (from data/splits.json if available),
  else use --train-end-date.
- Model: LightGBM LGBMClassifier.
Outputs (timestamped):
- models/exit_timing_model_oracle_<ts>.joblib (dict with model, features)
- data/exit_oracle_training/exit_model_training_metrics_<ts>.csv
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

try:
    from lightgbm import LGBMClassifier
except Exception as e:
    raise SystemExit("LightGBM is required: pip install lightgbm") from e

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits-file", default="data/splits.json")
    ap.add_argument("--train-end-date", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    df = pd.read_parquet(args.dataset)
    # Determine holdout boundary
    holdout_start = None
    sp = Path(args.splits_file)
    if sp.exists():
        try:
            with open(sp, "r") as f:
                spl = json.load(f)
            if spl.get("holdout_start"):
                holdout_start = pd.Timestamp(spl["holdout_start"])
        except Exception:
            holdout_start = None
    if args.train_end_date:
        holdout_start = pd.Timestamp(args.train_end_date)

    # Time-based split
    if holdout_start is not None and "entry_time" in df:
        train_df = df[pd.to_datetime(df["entry_time"]) < holdout_start].copy()
        test_df = df[pd.to_datetime(df["entry_time"]) >= holdout_start].copy()
        if len(test_df) == 0:
            # Fallback to time-based 80/20 split by timestamp order to keep temporal ordering
            dsorted = df.sort_values("entry_time").reset_index(drop=True)
            cut = int(0.8 * len(dsorted))
            train_df, test_df = dsorted.iloc[:cut].copy(), dsorted.iloc[cut:].copy()
    else:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=args.seed, shuffle=True)

    # Features: all columns derived from EXIT_FEATURE_COLS plus their lags, excluding timestamp-like and label
    drop_cols = {"timestamp","entry_time","date","is_oracle_exit"}
    X_cols = [c for c in train_df.columns if c not in drop_cols]

    Xtr = train_df[X_cols].fillna(0.0).to_numpy(dtype=float)
    ytr = train_df["is_oracle_exit"].astype(int).to_numpy()
    Xte = test_df[X_cols].fillna(0.0).to_numpy(dtype=float)
    yte = test_df["is_oracle_exit"].astype(int).to_numpy()

    clf = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=args.seed,
        min_child_samples=40,
        n_jobs=-1,
    )
    clf.fit(Xtr, ytr)

    pr_tr = clf.predict_proba(Xtr)[:,1]
    pr_te = clf.predict_proba(Xte)[:,1]
    auc_tr = roc_auc_score(ytr, pr_tr)
    ap_tr = average_precision_score(ytr, pr_tr)
    auc_te = roc_auc_score(yte, pr_te) if len(np.unique(yte))>1 else float("nan")
    ap_te = average_precision_score(yte, pr_te) if len(np.unique(yte))>1 else float("nan")

    out_dir = REPO_ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"exit_timing_model_oracle_{ts}.joblib"
    payload = {"model": clf, "features": X_cols}
    joblib.dump(payload, model_path)

    met = pd.DataFrame([{
        "train_rows": len(train_df), "test_rows": len(test_df),
        "auc_tr": float(auc_tr), "ap_tr": float(ap_tr),
        "auc_te": float(auc_te), "ap_te": float(ap_te),
        "model_path": str(model_path)
    }])
    mpath = REPO_ROOT / "data" / "exit_oracle_training" / f"exit_model_training_metrics_{ts}.csv"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    met.to_csv(mpath, index=False)
    print("Saved model:", model_path)
    print("Metrics:", mpath)


if __name__ == "__main__":
    main()
