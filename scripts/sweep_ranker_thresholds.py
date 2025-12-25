#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

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


def load_models(models_dir: Path, stem: str):
    lgb_path = models_dir / f"{stem}__entry_ranker_lgb.txt"
    xgb_path = models_dir / f"{stem}__entry_ranker_xgb.json"
    lgb_model = lgb.Booster(model_file=str(lgb_path)) if lgb_path.exists() else None
    xgb_model = xgb.Booster() if xgb_path.exists() else None
    if xgb_path.exists():
        xgb_model.load_model(str(xgb_path))
    return lgb_model, xgb_model


def score_rows(df: pd.DataFrame, lgb_model, xgb_model) -> np.ndarray:
    X = df[FEATURES].astype(np.float32)
    scores = []
    if lgb_model is not None:
        names = list(lgb_model.feature_name())
        Xl = X.reindex(columns=names, fill_value=0.0).values.astype(float)
        scores.append(lgb_model.predict(Xl))
    if xgb_model is not None:
        names = list(lgb_model.feature_name()) if (lgb_model is not None) else list(X.columns)
        Xx = X.reindex(columns=names, fill_value=0.0).values.astype(float)
        dm = xgb.DMatrix(Xx, feature_names=list(X.columns))
        scores.append(xgb_model.predict(dm))
    if not scores:
        return np.zeros(len(df), dtype=float)
    return np.mean(np.vstack(scores), axis=0)


def sweep(df: pd.DataFrame, scores: np.ndarray, thresholds: Iterable[float]) -> pd.DataFrame:
    rows = []
    ret = df["net_return_flat_pct"].to_numpy(np.float64)
    for thr in thresholds:
        mask = scores >= thr
        n = int(mask.sum())
        if n == 0:
            rows.append({"threshold": thr, "n": 0, "win_rate": np.nan, "mean_ret": np.nan, "median_ret": np.nan, "cum_ret_pct": 0.0})
            continue
        r = ret[mask]
        wr = float((r > 0).mean())
        mean_ret = float(r.mean())
        med_ret = float(np.median(r))
        cum = float((r / 100.0 + 1.0).prod() - 1.0) * 100.0
        rows.append({"threshold": thr, "n": n, "win_rate": wr, "mean_ret": mean_ret, "median_ret": med_ret, "cum_ret_pct": cum})
    return pd.DataFrame(rows).sort_values("threshold")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep entry-ranker score thresholds for profitability")
    ap.add_argument("--entry-parquet", required=True, help="Entry training parquet with net_return_flat_pct and features")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--stem", default="rule_best_ranker")
    ap.add_argument("--thresholds", nargs="*", type=float, default=None, help="List of thresholds; if omitted, use 0.30..0.70 step 0.02")
    ap.add_argument("--out-csv", default="data/sweeps/ranker_threshold_sweep.csv")
    args = ap.parse_args()

    df = pd.read_parquet(args.entry_parquet)
    lgb_model, xgb_model = load_models(Path(args.models_dir), args.stem)

    thr_list = args.thresholds if args.thresholds else [round(x, 2) for x in np.arange(0.30, 0.71, 0.02)]
    scores = score_rows(df, lgb_model, xgb_model)
    res = sweep(df, scores, thr_list)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_path, index=False)
    print(res.to_string(index=False))
    best = res.loc[res['cum_ret_pct'].idxmax()]
    print("\nBest by cumulative return:")
    print(best.to_dict())


if __name__ == "__main__":
    main()