#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-02T21:43:17Z
"""
Train entry regressor on oracle entries (BTC-USD perp):
- Input: oracle entry dataset with 5-minute pre-entry context
- Label: oracle_best_ret_pct (how profitable the opportunity is)
- Features: All pre-entry features (pre1-pre5) + entry features
- Model: LightGBM regressor
- NO LOOKAHEAD: Only uses features from BEFORE entry time

Output: Trained model that predicts expected return if entering at this minute.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train entry regressor on oracle entries")
    ap.add_argument("--dataset", required=True, help="Oracle entry training data (parquet)")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    ap.add_argument("--out-dir", default="models")
    
    args = ap.parse_args()
    
    # Load dataset
    print("Loading oracle entry dataset...")
    df = pd.read_parquet(args.dataset)
    print(f"Loaded {len(df)} oracle entries")
    
    # Extract features and labels
    # Features: all pre-entry context (pre1-pre5) + entry minute features
    feature_cols = [c for c in df.columns if c.startswith("pre") or c.startswith("entry_")]
    feature_cols = [c for c in feature_cols if c not in ["entry_time", "entry_open", "entry_close"]]
    
    label_col = "oracle_best_ret_pct"
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Label: {label_col}")
    
    # Prepare data
    X = df[feature_cols].copy()
    y = df[label_col].copy()
    
    # Drop rows with NaN (from early minutes without full pre-context)
    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid]
    
    print(f"\nAfter dropping NaN: {len(X)} samples")
    
    # Chronological split (not random - no future leakage)
    split_idx = int(len(X) * (1 - args.test_size))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"\nTrain: {len(X_train)} samples")
    print(f"Test:  {len(X_test)} samples")
    print(f"Train mean label: {y_train.mean():.4f}%")
    print(f"Test mean label:  {y_test.mean():.4f}%")
    
    # Train LightGBM regressor
    print("\nTraining LightGBM regressor...")
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=64,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
    )
    
    # Evaluate
    print("\n=== Evaluation ===")
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
    test_rmse = np.sqrt(np.mean((test_pred - y_test) ** 2))
    
    train_mae = np.mean(np.abs(train_pred - y_train))
    test_mae = np.mean(np.abs(test_pred - y_test))
    
    print(f"Train RMSE: {train_rmse:.4f}%")
    print(f"Test RMSE:  {test_rmse:.4f}%")
    print(f"Train MAE:  {train_mae:.4f}%")
    print(f"Test MAE:   {test_mae:.4f}%")
    
    # Correlation
    train_corr = np.corrcoef(train_pred, y_train)[0, 1]
    test_corr = np.corrcoef(test_pred, y_test)[0, 1]
    print(f"\nTrain correlation: {train_corr:.4f}")
    print(f"Test correlation:  {test_corr:.4f}")
    
    # Feature importance
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print(f"\nTop 15 features:")
    print(importances.head(15).to_string(index=False))
    
    # Save model
    ts = _now_ts()
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    artifact = {
        "model": model,
        "features": feature_cols,
        "label": label_col,
        "context": {
            "pre_min": 5,
            "dataset": str(args.dataset),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "train_rmse": float(train_rmse),
            "test_rmse": float(test_rmse),
            "train_corr": float(train_corr),
            "test_corr": float(test_corr),
        },
        "created_utc": ts,
    }
    
    model_path = out_dir / f"entry_regressor_dydx_oracle_{ts}.joblib"
    joblib.dump(artifact, model_path)
    
    print(f"\n=== Model Saved ===")
    print(f"{model_path}")
    print(f"\nTo use this model:")
    print(f"  - Predicts expected return if entering at this minute")
    print(f"  - Enter when prediction > some threshold (e.g. > 0.2%)")
    print(f"  - Model sees ONLY past 5 minutes of features (no future leakage)")


if __name__ == "__main__":
    main()
