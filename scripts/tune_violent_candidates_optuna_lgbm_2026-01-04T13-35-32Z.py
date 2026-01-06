#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T13:35:32Z
"""Optuna tuning on violent-candidates dataset (BUY) for FAST/SLOW/NO labels.

Dataset
- data/violent_candidates_oracle15m_*/violent_candidates.parquet
  Required columns: timestamp, y_cls
  y_cls: 0=NO_HIT, 1=SLOW_HIT, 2=FAST_HIT

Goal
- Tune a LightGBM multiclass classifier on the violent slice.
- Objective focuses on what matters for the entry system:
  - AUC_hit15_vs_no: predict HIT within 15m (SLOW+FAST) vs NO
  - AUC_fast_vs_rest: predict FAST vs (SLOW+NO)
  Score = 0.5*AUC_hit15_vs_no + 0.5*AUC_fast_vs_rest

Notes
- Time split: train first (1-test_frac), validate last test_frac by timestamp.
- Uses ONLY precontext-derived features from the dataset. Excludes:
  timestamp, entry_i, best_ret_15m, t_hit, y_cls, pred

Outputs (timestamped)
- data/tune_violent_candidates_optuna_<ts>/
  - study_best.json
  - trials.csv
  - metrics_best.json
  - model_best.joblib
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd

from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def pick_latest_violent_dataset() -> Path:
    roots = sorted(Path("data").glob("violent_candidates_oracle15m_*/violent_candidates.parquet"))
    if not roots:
        raise SystemExit("No violent_candidates.parquet found under data/violent_candidates_oracle15m_*/")
    return roots[-1]


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {
        "timestamp",
        "entry_i",
        "pred",
        "best_ret_15m",
        "t_hit",
        "y_cls",
    }
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if df[c].dtype.kind not in "fc":
            continue
        cols.append(c)
    return cols


def _fill_na_train_median(X_tr: np.ndarray, X_va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_va = np.asarray(X_va, dtype=np.float64)
    med = np.nanmedian(X_tr, axis=0)

    bad_tr = ~np.isfinite(X_tr)
    if bad_tr.any():
        X_tr[bad_tr] = med[np.where(bad_tr)[1]]

    bad_va = ~np.isfinite(X_va)
    if bad_va.any():
        X_va[bad_va] = med[np.where(bad_va)[1]]

    return X_tr, X_va


def main() -> None:
    ap = argparse.ArgumentParser(description="Optuna tune LightGBM on violent candidates (3-class)")
    ap.add_argument("--data", default=str(pick_latest_violent_dataset()), help="violent_candidates.parquet")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--early-stop", type=int, default=200)
    ap.add_argument("--log-period", type=int, default=0, help="LightGBM eval log period; 0 disables")
    args = ap.parse_args()

    ts = now_ts()
    out_root = Path(args.out_dir) / f"tune_violent_candidates_optuna_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.data}", flush=True)
    df = pd.read_parquet(args.data)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    y = pd.to_numeric(df["y_cls"], errors="coerce").astype(np.int64).to_numpy()
    X_cols = feature_columns(df)
    if not X_cols:
        raise SystemExit("No numeric feature columns found")

    # time split
    n = len(df)
    split = int(n * (1.0 - float(args.test_frac)))
    if split < 2000 or (n - split) < 2000:
        raise SystemExit(f"Too small for split: n={n} split={split}")

    X = df[X_cols].to_numpy(np.float32, copy=False)

    X_tr = X[:split]
    y_tr = y[:split]
    X_va = X[split:]
    y_va = y[split:]

    X_tr, X_va = _fill_na_train_median(X_tr, X_va)

    y_hit15 = (y_va > 0).astype(int)  # SLOW+FAST vs NO
    y_fast = (y_va == 2).astype(int)  # FAST vs rest

    print(f"Rows: n={n:,} train={split:,} val={n-split:,} n_features={len(X_cols):,}", flush=True)
    print("Class balance (all):", dict(zip(*np.unique(y, return_counts=True))), flush=True)

    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)

    def objective(trial: optuna.Trial) -> float:
        # Parameter search space
        params = {
            "n_estimators": 5000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 512, log=True),
            "max_depth": trial.suggest_int("max_depth", -1, 16),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 2000, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "random_state": int(args.seed),
            "n_jobs": -1,
            "objective": "multiclass",
            "num_class": 3,
            "verbosity": -1,
        }

        clf = LGBMClassifier(**params)

        callbacks = [early_stopping(stopping_rounds=int(args.early_stop), verbose=False)]
        if int(args.log_period) > 0:
            callbacks.append(log_evaluation(period=int(args.log_period)))

        clf.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="multi_logloss",
            callbacks=callbacks,
        )

        proba = clf.predict_proba(X_va)
        p_hit15 = proba[:, 1] + proba[:, 2]
        p_fast = proba[:, 2]

        # AUCs
        auc_hit15 = roc_auc_score(y_hit15, p_hit15) if len(np.unique(y_hit15)) == 2 else float("nan")
        auc_fast = roc_auc_score(y_fast, p_fast) if len(np.unique(y_fast)) == 2 else float("nan")

        score = 0.5 * float(auc_hit15) + 0.5 * float(auc_fast)

        trial.set_user_attr("auc_hit15", float(auc_hit15))
        trial.set_user_attr("auc_fast", float(auc_fast))
        trial.set_user_attr("best_iteration", int(getattr(clf, "best_iteration_", params["n_estimators"])))

        return float(score)

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=int(args.n_trials), show_progress_bar=True)

    best = study.best_trial
    best_params = dict(best.params)
    best_info = {
        "created_utc": ts,
        "data": str(args.data),
        "n_trials": int(args.n_trials),
        "score": float(best.value),
        "params": best_params,
        "auc_hit15": float(best.user_attrs.get("auc_hit15", float("nan"))),
        "auc_fast": float(best.user_attrs.get("auc_fast", float("nan"))),
        "best_iteration": int(best.user_attrs.get("best_iteration", -1)),
        "n_features": int(len(X_cols)),
        "features": X_cols,
        "split": {"train": int(split), "val": int(n - split)},
    }

    (out_root / "study_best.json").write_text(json.dumps(best_info, indent=2), encoding="utf-8")

    # Save trials
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials_df.to_csv(out_root / "trials.csv", index=False)

    # Train a final model with best params on train, early-stop on val
    final_params = {
        "n_estimators": 5000,
        "learning_rate": float(best_params["learning_rate"]),
        "num_leaves": int(best_params["num_leaves"]),
        "max_depth": int(best_params["max_depth"]),
        "min_child_samples": int(best_params["min_child_samples"]),
        "subsample": float(best_params["subsample"]),
        "subsample_freq": 1,
        "colsample_bytree": float(best_params["colsample_bytree"]),
        "reg_alpha": float(best_params["reg_alpha"]),
        "reg_lambda": float(best_params["reg_lambda"]),
        "min_split_gain": float(best_params["min_split_gain"]),
        "random_state": int(args.seed),
        "n_jobs": -1,
        "objective": "multiclass",
        "num_class": 3,
        "verbosity": -1,
    }

    clf = LGBMClassifier(**final_params)
    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="multi_logloss",
        callbacks=[early_stopping(stopping_rounds=int(args.early_stop), verbose=False)],
    )

    proba = clf.predict_proba(X_va)
    p_hit15 = proba[:, 1] + proba[:, 2]
    p_fast = proba[:, 2]

    auc_hit15 = float(roc_auc_score(y_hit15, p_hit15))
    auc_fast = float(roc_auc_score(y_fast, p_fast))

    metrics = {
        "score": 0.5 * auc_hit15 + 0.5 * auc_fast,
        "auc_hit15": auc_hit15,
        "auc_fast": auc_fast,
        "best_iteration": int(getattr(clf, "best_iteration_", -1)),
    }
    (out_root / "metrics_best.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    artifact = {
        "created_utc": ts,
        "data": str(args.data),
        "features": X_cols,
        "split": {"train": int(split), "val": int(n - split)},
        "params": final_params,
        "metrics": metrics,
        "model": clf,
    }
    joblib.dump(artifact, out_root / "model_best.joblib")

    print("Done.")
    print("Outputs:", out_root)
    print(json.dumps(best_info, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
