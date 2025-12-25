#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover
    raise SystemExit("lightgbm is required. pip install lightgbm") from exc

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover
    raise SystemExit("xgboost is required. pip install xgboost") from exc

FEATURES = [
    "ret_1m_pct",
    "mom_3m_pct",
    "mom_5m_pct",
    "vol_std_5m",
    "range_5m",
    "range_norm_5m",
    "slope_ols_5m",
    "rsi_14",
    "macd",
    "macd_hist",
    "vwap_dev_5m",
    "last3_same_sign",
]
EXIT_FEATURES = FEATURES + ["elapsed_min"]


def _time_split(df: pd.DataFrame, date_col: str, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    dates = sorted(d for d in df[date_col].dropna().unique())
    if not dates:
        raise ValueError(f"No valid dates in {date_col}")
    split_idx = max(1, int(len(dates) * (1.0 - val_frac)))
    train_dates = set(dates[:split_idx])
    train_df = df[df[date_col].isin(train_dates)].copy()
    valid_df = df[~df[date_col].isin(train_dates)].copy()
    return train_df, valid_df


def _train_lgb(train_df: pd.DataFrame, valid_df: pd.DataFrame, features: list[str], label_col: str, out_path: Path, desc: str) -> None:
    print(f"[lgb] {desc}: train={len(train_df):,} valid={len(valid_df):,} feats={len(features)}")
    Xtr = train_df[features].astype(np.float32)
    ytr = train_df[label_col].astype(int)
    Xva = valid_df[features].astype(np.float32)
    yva = valid_df[label_col].astype(int)

    train_set = lgb.Dataset(Xtr, label=ytr, feature_name=features)
    valid_set = lgb.Dataset(Xva, label=yva, feature_name=features)

    params: dict[str, object] = dict(
        objective="binary",
        metric=["binary_logloss", "auc"],
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=200,
        feature_pre_filter=False,
        random_state=42,
        verbosity=-1,
    )

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(200, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_path))
    print(f"[lgb] saved {out_path}")


def _train_xgb(train_df: pd.DataFrame, valid_df: pd.DataFrame, features: list[str], label_col: str, out_path: Path, desc: str) -> None:
    print(f"[xgb] {desc}: train={len(train_df):,} valid={len(valid_df):,} feats={len(features)}")
    Xtr = train_df[features].astype(np.float32)
    ytr = train_df[label_col].astype(int)
    Xva = valid_df[features].astype(np.float32)
    yva = valid_df[label_col].astype(int)

    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=features)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=features)

    params: dict[str, object] = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": 42,
    }

    booster = xgb.train(
        params,
        dtr,
        num_boost_round=2000,
        evals=[(dva, "valid")],
        early_stopping_rounds=200,
        verbose_eval=200,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_path))
    print(f"[xgb] saved {out_path}")


def train_models(entry_path: Path, exit_path: Path, models_dir: Path, stem: str, val_frac: float) -> None:
    if not entry_path.exists() or not exit_path.exists():
        raise SystemExit(f"Missing training data: {entry_path} or {exit_path}")

    print(f"Loading entry training: {entry_path}")
    entry_df = pd.read_parquet(entry_path)
    entry_df = entry_df.dropna(subset=["label_win"] + FEATURES)

    print(f"Loading exit training:  {exit_path}")
    exit_df = pd.read_parquet(exit_path)
    exit_df = exit_df.dropna(subset=["label_win"] + EXIT_FEATURES)

    # Entry
    tr_e, va_e = _time_split(entry_df, "entry_ts", val_frac)
    _train_lgb(tr_e, va_e, FEATURES, "label_win", models_dir / f"{stem}__entry_ranker_lgb.txt", "entry")
    _train_xgb(tr_e, va_e, FEATURES, "label_win", models_dir / f"{stem}__entry_ranker_xgb.json", "entry")

    # Exit
    tr_x, va_x = _time_split(exit_df, "exit_ts", val_frac)
    _train_lgb(tr_x, va_x, EXIT_FEATURES, "label_win", models_dir / f"{stem}__exit_ranker_lgb.txt", "exit")
    _train_xgb(tr_x, va_x, EXIT_FEATURES, "label_win", models_dir / f"{stem}__exit_ranker_xgb.json", "exit")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train entry/exit rankers from rule trades")
    ap.add_argument("--entry-train", required=True)
    ap.add_argument("--exit-train", required=True)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--stem", default="rule_best_ranker")
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    train_models(Path(args.entry_train), Path(args.exit_train), Path(args.models_dir), args.stem, args.val_frac)


if __name__ == "__main__":
    main()