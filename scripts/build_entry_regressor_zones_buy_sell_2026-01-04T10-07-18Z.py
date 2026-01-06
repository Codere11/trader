#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T10:07:18Z
"""Build BUY/SELL entry training datasets with 'generalization zones' and stripped pattern-outliers.

You asked for:
- Split into BUY vs SELL.
- Identify 'pattern-outliers' where the 5m precontext differs a lot from the dense/typical mass.
- Strip those outliers.
- Produce clean 'generalization zones' (clusters) and train an 800-tree regressor per side.

Inputs
- Side-split aggregated pre-entry descriptor datasets (from earlier analysis):
    data/analysis_oracle_entry_outliers_by_side_*/BUY/entry_context_agg_scored.parquet
    data/analysis_oracle_entry_outliers_by_side_*/SELL/entry_context_agg_scored.parquet

Outputs (timestamped)
- data/entry_precontext_zones_<ts>/
  - BUY_cleaned_<ts>.parquet
  - SELL_cleaned_<ts>.parquet
  - BUY_cluster_summary_<ts>.csv
  - SELL_cluster_summary_<ts>.csv
  - BUY_outliers_<ts>.csv
  - SELL_outliers_<ts>.csv
  - BUY_zone_meta_<ts>.json
  - SELL_zone_meta_<ts>.json
  - run_meta_<ts>.json
- models/ (only if --train)
  - entry_regressor_buy_zones_<ts>.joblib
  - entry_regressor_sell_zones_<ts>.joblib

Notes
- Outlier definition is intentionally simple and controllable:
    outlier if nn_dist >= q_nn OR outlier_score >= q_iso
  where q_nn and q_iso are quantiles computed within-side on complete contexts.
- Zones are KMeans clusters fit on inlier-only standardized descriptor features.
- Label is trade_net_return_pct (net return percent of the oracle trade).

This is meant to be fast and to produce clean training datasets, not to be a final research pipeline.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def pick_input_root(default: str) -> Path:
    """Pick most recent analysis_oracle_entry_outliers_by_side_* directory if default missing."""
    p = Path(default)
    if p.exists():
        return p
    # fallback: find latest by lexicographic timestamp suffix
    candidates = sorted(Path("data").glob("analysis_oracle_entry_outliers_by_side_*") )
    if not candidates:
        raise SystemExit(f"No analysis_oracle_entry_outliers_by_side_* found and default missing: {p}")
    return candidates[-1]


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop_exact = {
        "trade_row",
        "trade_net_return_pct",
        "outlier_score",
        "nn_dist",
        "missing_any",
        "missing_close_n",
    }
    cols: list[str] = []
    for c in df.columns:
        if c in drop_exact:
            continue
        if c.startswith("trade_"):
            continue
        if df[c].dtype.kind not in "fc":
            continue
        cols.append(c)
    return cols


def load_side(input_root: Path, side: str) -> pd.DataFrame:
    p = input_root / side / "entry_context_agg_scored.parquet"
    if not p.exists():
        raise SystemExit(f"Missing: {p}")
    df = pd.read_parquet(p)
    df["trade_side"] = df["trade_side"].astype(str).str.upper()
    df = df[df["trade_side"] == side].copy()
    # keep complete contexts only
    if "missing_any" in df.columns:
        df = df[df["missing_any"].astype(float) == 0.0]
    df["trade_entry_time"] = pd.to_datetime(df["trade_entry_time"], utc=True, errors="coerce")
    df = df.sort_values("trade_entry_time").reset_index(drop=True)
    return df


def compute_outlier_thresholds(df: pd.DataFrame, *, q_nn: float, q_iso: float) -> tuple[float, float]:
    if "nn_dist" not in df.columns or "outlier_score" not in df.columns:
        raise SystemExit("Expected nn_dist and outlier_score columns in input scored dataset")

    nn = pd.to_numeric(df["nn_dist"], errors="coerce").to_numpy(np.float64)
    iso = pd.to_numeric(df["outlier_score"], errors="coerce").to_numpy(np.float64)

    nn_thr = float(np.nanquantile(nn, float(q_nn)))
    iso_thr = float(np.nanquantile(iso, float(q_iso)))
    return nn_thr, iso_thr


def add_zone_labels(df: pd.DataFrame, *, X_cols: list[str], n_clusters: int, random_state: int) -> tuple[pd.DataFrame, dict]:
    X = df[X_cols].to_numpy(np.float64)
    med = np.nanmedian(X, axis=0)
    bad = ~np.isfinite(X)
    if np.any(bad):
        X[bad] = med[np.where(bad)[1]]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=int(n_clusters), random_state=int(random_state), n_init=10)
    zone = km.fit_predict(Xs)

    out = df.copy()
    out["zone_id"] = zone.astype(np.int32)

    meta = {
        "n_clusters": int(n_clusters),
        "random_state": int(random_state),
        "feature_cols": list(X_cols),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "kmeans_centers": km.cluster_centers_.tolist(),
    }
    return out, meta


def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("zone_id", as_index=False)
    out = g.agg(
        n=("zone_id", "size"),
        ret_mean=("trade_net_return_pct", "mean"),
        ret_median=("trade_net_return_pct", "median"),
        ret_p10=("trade_net_return_pct", lambda s: float(np.nanquantile(pd.to_numeric(s, errors="coerce"), 0.10))),
        ret_p90=("trade_net_return_pct", lambda s: float(np.nanquantile(pd.to_numeric(s, errors="coerce"), 0.90))),
        nn_mean=("nn_dist", "mean"),
    )
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def train_regressor(df: pd.DataFrame, *, X_cols: list[str], n_estimators: int) -> tuple[dict, dict]:
    """Train 800-tree regressor; return (artifact, metrics)."""
    y = pd.to_numeric(df["trade_net_return_pct"], errors="coerce").to_numpy(np.float64)
    X = df[X_cols].to_numpy(np.float64)

    med = np.nanmedian(X, axis=0)
    bad = ~np.isfinite(X)
    if np.any(bad):
        X[bad] = med[np.where(bad)[1]]

    # time split
    n = len(df)
    split = int(n * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    model = LGBMRegressor(
        n_estimators=int(n_estimators),
        learning_rate=0.05,
        num_leaves=128,
        max_depth=-1,
        min_child_samples=200,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(X_tr, y_tr)

    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)

    def rmse(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def corr(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.size < 2:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    metrics = {
        "n": int(n),
        "split": int(split),
        "rmse_train": rmse(pred_tr, y_tr),
        "rmse_test": rmse(pred_te, y_te),
        "corr_train": corr(pred_tr, y_tr),
        "corr_test": corr(pred_te, y_te),
        "y_mean": float(np.nanmean(y)),
        "y_median": float(np.nanmedian(y)),
        "y_p10": float(np.nanquantile(y, 0.10)),
        "y_p90": float(np.nanquantile(y, 0.90)),
    }

    artifact = {
        "model": model,
        "feature_cols": list(X_cols),
        "label": "trade_net_return_pct",
        "created_utc": now_ts(),
        "metrics": metrics,
    }

    return artifact, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Strip precontext outliers, assign zones, train BUY/SELL regressors")
    ap.add_argument(
        "--input-root",
        default="data/analysis_oracle_entry_outliers_by_side_2026-01-04T00-55-44Z",
        help="Root with BUY/SELL entry_context_agg_scored.parquet",
    )
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--n-clusters", type=int, default=12, help="Generalization zones per side")
    ap.add_argument("--q-nn", type=float, default=0.99, help="Outlier nn_dist quantile (e.g. 0.99 keeps 99% by nn_dist)")
    ap.add_argument("--q-iso", type=float, default=0.99, help="Outlier outlier_score quantile")
    ap.add_argument("--n-estimators", type=int, default=800, help="Trees for regressor (only used with --train)")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--train", action="store_true", help="Train and save LightGBM regressors (otherwise just write cleaned datasets)")
    args = ap.parse_args()

    input_root = pick_input_root(str(args.input_root))
    ts = now_ts()

    out_root = Path(args.out_dir) / f"entry_precontext_zones_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "input_root": str(input_root),
        "created_utc": ts,
        "n_clusters": int(args.n_clusters),
        "q_nn": float(args.q_nn),
        "q_iso": float(args.q_iso),
        "n_estimators": int(args.n_estimators),
        "random_state": int(args.random_state),
    }

    print("Config:")
    print(json.dumps(run_meta, indent=2))

    models_out = Path("models")
    models_out.mkdir(parents=True, exist_ok=True)

    for side in ["BUY", "SELL"]:
        df = load_side(input_root, side)
        print(f"\n[{side}] loaded complete rows: {len(df):,}")

        X_cols = feature_columns(df)
        if not X_cols:
            raise SystemExit(f"[{side}] no numeric feature columns found")

        nn_thr, iso_thr = compute_outlier_thresholds(df, q_nn=float(args.q_nn), q_iso=float(args.q_iso))
        nn = pd.to_numeric(df["nn_dist"], errors="coerce").to_numpy(np.float64)
        iso = pd.to_numeric(df["outlier_score"], errors="coerce").to_numpy(np.float64)
        is_outlier = (nn >= nn_thr) | (iso >= iso_thr)

        df2 = df.copy()
        df2["is_outlier"] = is_outlier.astype(np.int8)
        n_out = int(df2["is_outlier"].sum())
        print(f"[{side}] outlier thresholds: nn_dist>= {nn_thr:.6f}  outlier_score>= {iso_thr:.6f}")
        print(f"[{side}] outliers: {n_out:,}/{len(df2):,} ({(n_out/len(df2))*100.0:.2f}%)")

        outliers_csv = out_root / f"{side}_outliers_{ts}.csv"
        df2[df2["is_outlier"] == 1][
            [
                "trade_row",
                "trade_entry_time",
                "trade_side",
                "trade_net_return_pct",
                "nn_dist",
                "outlier_score",
            ]
        ].to_csv(outliers_csv, index=False)

        inliers = df2[df2["is_outlier"] == 0].reset_index(drop=True)

        # Zones (clusters) on inliers
        inliers_z, zone_meta = add_zone_labels(inliers, X_cols=X_cols, n_clusters=int(args.n_clusters), random_state=int(args.random_state))

        # Persist cleaned dataset
        cleaned = inliers_z.copy()
        cleaned_path = out_root / f"{side}_cleaned_{ts}.parquet"
        cleaned.to_parquet(cleaned_path, index=False)
        print(f"[{side}] wrote cleaned dataset: {cleaned_path} (rows={len(cleaned):,})")

        # Cluster summary
        summ = cluster_summary(cleaned)
        summ_path = out_root / f"{side}_cluster_summary_{ts}.csv"
        summ.to_csv(summ_path, index=False)
        print(f"[{side}] wrote cluster summary: {summ_path}")

        if args.train:
            # Train regressor on cleaned (inlier) data
            art, metrics = train_regressor(cleaned, X_cols=X_cols + ["zone_id"], n_estimators=int(args.n_estimators))
            model_path = models_out / f"entry_regressor_{side.lower()}_zones_{ts}.joblib"
            joblib.dump(art, model_path)
            print(f"[{side}] model saved: {model_path}")
            print(f"[{side}] metrics: rmse_test={metrics['rmse_test']:.6f}  corr_test={metrics['corr_test']:.4f}  n={metrics['n']:,}")
        else:
            print(f"[{side}] skipping training (pass --train to enable)")

        # Save zone meta
        (out_root / f"{side}_zone_meta_{ts}.json").write_text(json.dumps(zone_meta, indent=2), encoding="utf-8")

    (out_root / f"run_meta_{ts}.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print("\nDone.")
    print("Outputs:", out_root)


if __name__ == "__main__":
    main()
