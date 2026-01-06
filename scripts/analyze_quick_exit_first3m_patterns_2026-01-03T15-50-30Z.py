#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-03T15:50:30Z
"""Analyze patterns where exiting within the first 3 minutes is preferable.

Goal
- Identify the strongest *causal* signs (available by decision time) that a new trade should
  likely be exited quickly (within minutes 1..3).

Dataset
- Expects an oracle-exit dataset parquet produced by:
  scripts/train_exit_oracle_classifier_hold15_*.py
  where each row is one minute-in-trade (k_rel=1..hold_min).

Outputs
- Writes a report dir under data/analysis_quick_exit_first3m_*/ with:
  - report.json (label rates, top features by AUC for k=1/2/3, logistic model coefficients)
  - top_features_k{1,2,3}.csv

No future leakage
- This script only uses per-row features that already exist at that decision minute.
- Labels are derived from realized returns within the same trade (oracle-style evaluation).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        y_true = np.asarray(y_true, dtype=np.int64)
        y_score = np.asarray(y_score, dtype=np.float64)
        ok = np.isfinite(y_score)
        if ok.sum() < 50:
            return float("nan")
        if len(np.unique(y_true[ok])) < 2:
            return float("nan")
        return float(roc_auc_score(y_true[ok], y_score[ok]))
    except Exception:
        return float("nan")


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score

        y_true = np.asarray(y_true, dtype=np.int64)
        y_score = np.asarray(y_score, dtype=np.float64)
        ok = np.isfinite(y_score)
        if ok.sum() < 50:
            return float("nan")
        if len(np.unique(y_true[ok])) < 2:
            return float("nan")
        return float(average_precision_score(y_true[ok], y_score[ok]))
    except Exception:
        return float("nan")


def infer_feature_cols(df: pd.DataFrame) -> List[str]:
    # Exclude ids/metadata AND any derived labels/summary fields created by this analysis.
    # Anything derived from the full trade path (e.g. hold_ret, gap3_vs_hold) is NOT a causal feature.
    drop_cols = {
        "trade_id",
        "signal_i",
        "entry_idx",
        "entry_time",
        "entry_px",
        "oracle_k",
        "oracle_ret_pct",
        "k_rel",
        "decision_idx",
        "decision_time",
        "y_oracle_exit",
        "ret_if_exit_now_pct",
        "created_utc",
        # analysis-derived trade summary fields
        "hold_ret",
        "best3_ret",
        "best3_k",
        "gap3_vs_hold",
        # analysis-derived labels
        "y_leave3_better",
        "y_leave3_saves_loss",
    }
    cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    return [str(c) for c in cols]


@dataclass(frozen=True)
class TradeSummary:
    trade_id: int
    hold_ret: float
    best3_ret: float
    best3_k: int
    gap3_vs_hold: float


def build_trade_summary(df: pd.DataFrame, hold_min: int) -> pd.DataFrame:
    # Expect full sequences; caller should have filtered.
    w = df[["trade_id", "k_rel", "ret_if_exit_now_pct"]].copy()

    hold = w[w["k_rel"] == int(hold_min)].set_index("trade_id")["ret_if_exit_now_pct"].rename("hold_ret")

    first3 = w[w["k_rel"] <= 3].copy()
    best3 = first3.groupby("trade_id")["ret_if_exit_now_pct"].max().rename("best3_ret")

    # argmax k within 1..3
    first3_sorted = first3.sort_values(["trade_id", "ret_if_exit_now_pct", "k_rel"], ascending=[True, False, True])
    best3_k = first3_sorted.groupby("trade_id").first()["k_rel"].astype(int).rename("best3_k")

    out = pd.concat([hold, best3, best3_k], axis=1).reset_index()
    out["gap3_vs_hold"] = out["best3_ret"] - out["hold_ret"]
    return out


def time_ordered_trade_split(df: pd.DataFrame, test_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    # Unique trades ordered by entry_time if available; else by trade_id.
    if "entry_time" in df.columns:
        order = df[["trade_id", "entry_time"]].drop_duplicates("trade_id").sort_values("entry_time")
    else:
        order = df[["trade_id"]].drop_duplicates("trade_id").sort_values("trade_id")

    tids = order["trade_id"].to_numpy()
    n = len(tids)
    n_test = max(1, int(n * float(test_frac)))
    return tids[: n - n_test], tids[n - n_test :]


def rank_features_auc(dk: pd.DataFrame, y: np.ndarray, feat_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for c in feat_cols:
        s = pd.to_numeric(dk[c], errors="coerce").to_numpy(np.float64)
        auc = safe_auc(y, s)
        ap = safe_ap(y, s)
        rows.append({"feature": str(c), "auc": float(auc), "ap": float(ap)})

    out = pd.DataFrame(rows)
    # sort by how far AUC is from 0.5
    out["auc_edge"] = np.abs(out["auc"] - 0.5)
    out = out.sort_values(["auc_edge", "ap"], ascending=[False, False]).reset_index(drop=True)
    return out


def fit_logistic(dk: pd.DataFrame, y: np.ndarray, feat_cols: List[str], train_ids: np.ndarray, test_ids: np.ndarray) -> Dict[str, object]:
    from sklearn.linear_model import LogisticRegression

    # one row per trade for fixed k
    dk = dk[dk["trade_id"].isin(np.concatenate([train_ids, test_ids]))].copy()

    X = dk[feat_cols].to_numpy(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y, dtype=np.int64)

    tr_mask = dk["trade_id"].isin(train_ids).to_numpy()
    te_mask = dk["trade_id"].isin(test_ids).to_numpy()

    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_te, y_te = X[te_mask], y[te_mask]

    # standardize
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)

    X_trz = (X_tr - mu) / sd
    X_tez = (X_te - mu) / sd

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=200,
        class_weight="balanced",
    )
    clf.fit(X_trz, y_tr)

    p_te = clf.predict_proba(X_tez)[:, 1]

    out: Dict[str, object] = {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "pos_rate_train": float(np.mean(y_tr)),
        "pos_rate_test": float(np.mean(y_te)),
        "auc_test": float(safe_auc(y_te, p_te)),
        "ap_test": float(safe_ap(y_te, p_te)),
        "feature_cols": list(feat_cols),
        "standardize": {"mean": mu.tolist(), "std": sd.tolist()},
        "coef": clf.coef_.reshape(-1).tolist(),
        "intercept": float(clf.intercept_[0]),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze quick-exit-preferred patterns for the first 3 minutes")
    ap.add_argument(
        "--dataset-parquet",
        default="data/exit_oracle/exit_oracle_dataset_hold15_frac0p01_2026-01-03T15-35-31Z.parquet",
        help="Oracle-exit dataset parquet",
    )
    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--gap-thresh-pp", type=float, default=0.25, help="Gap threshold in percentage points")
    ap.add_argument("--test-trade-frac", type=float, default=0.2)
    ap.add_argument("--topn", type=int, default=30)
    ap.add_argument("--logit-topk", type=int, default=12, help="Use top-K univariate features for logistic fit")
    ap.add_argument("--out-dir", default="data")

    args = ap.parse_args()

    ds_path = Path(args.dataset_parquet)
    if not ds_path.exists():
        raise SystemExit(f"dataset not found: {ds_path}")

    hold_min = int(args.hold_min)
    gap = float(args.gap_thresh_pp)

    print(f"Loading: {ds_path}", flush=True)
    df = pd.read_parquet(ds_path)

    # basic filtering
    need = {"trade_id", "k_rel", "ret_if_exit_now_pct"}
    if not need.issubset(df.columns):
        raise SystemExit(f"dataset missing required columns: {sorted(list(need - set(df.columns)))}")

    df = df[(df["k_rel"] >= 1) & (df["k_rel"] <= hold_min)].copy()
    df = df.sort_values(["trade_id", "k_rel"]).reset_index(drop=True)

    sizes = df.groupby("trade_id")["k_rel"].count()
    full_ids = sizes[sizes == hold_min].index.to_numpy()
    if len(full_ids) == 0:
        raise SystemExit("no full-length trades found")

    df = df[df["trade_id"].isin(full_ids)].copy().reset_index(drop=True)

    # trade-level summary + labels
    summ = build_trade_summary(df, hold_min=hold_min)

    # Label: exiting within first 3 minutes is materially better than holding to horizon.
    summ["y_leave3_better"] = (summ["gap3_vs_hold"] >= gap).astype(int)
    summ["y_leave3_saves_loss"] = ((summ["hold_ret"] < 0.0) & (summ["gap3_vs_hold"] >= gap)).astype(int)

    # Merge labels back to rows
    df = df.merge(summ[["trade_id", "hold_ret", "best3_ret", "best3_k", "gap3_vs_hold", "y_leave3_better", "y_leave3_saves_loss"]], on="trade_id", how="left")

    # Feature universe
    feat_cols = infer_feature_cols(df)

    out_root = Path(args.out_dir)
    out_dir = out_root / f"analysis_quick_exit_first3m_{now_ts()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split
    train_ids, test_ids = time_ordered_trade_split(df, test_frac=float(args.test_trade_frac))

    report: Dict[str, object] = {
        "dataset": str(ds_path),
        "hold_min": int(hold_min),
        "gap_thresh_pp": float(gap),
        "n_trades": int(summ.shape[0]),
        "label_rates": {
            "leave3_better": float(summ["y_leave3_better"].mean()),
            "leave3_saves_loss": float(summ["y_leave3_saves_loss"].mean()),
        },
        "splits": {"n_train_trades": int(len(train_ids)), "n_test_trades": int(len(test_ids))},
        "per_k": {},
    }

    # Analyze k=1..3
    for k in [1, 2, 3]:
        dk = df[df["k_rel"] == int(k)].copy()
        y = dk["y_leave3_saves_loss"].to_numpy(np.int64)

        ranked = rank_features_auc(dk, y=y, feat_cols=feat_cols)
        topn = ranked.head(int(args.topn)).copy()
        topn.to_csv(out_dir / f"top_features_k{k}.csv", index=False)

        # Fit logistic on top-K
        use = topn["feature"].head(int(args.logit_topk)).tolist()
        logit = fit_logistic(dk, y=y, feat_cols=use, train_ids=train_ids, test_ids=test_ids)

        report["per_k"][str(k)] = {
            "label": "y_leave3_saves_loss",
            "n_rows": int(len(dk)),
            "pos_rate": float(np.mean(y)),
            "top_features": topn.to_dict(orient="records"),
            "logistic": logit,
        }

        print(f"k={k}  pos_rate={float(np.mean(y)):.4f}  logistic_auc={logit['auc_test']:.4f}  logistic_ap={logit['ap_test']:.4f}")
        print("Top AUC features:")
        print(topn.head(12)[["feature", "auc", "ap"]].to_string(index=False))
        print("", flush=True)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWrote: {out_dir}")


if __name__ == "__main__":
    main()
