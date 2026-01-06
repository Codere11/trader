#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T16:41:00Z
"""Select a *trainable* subset of oracle entry trades by trimming pattern outliers.

Goal
- Start from oracle trader 5-minute *pre_entry* contexts.
- Restrict to trades whose oracle net return is >= return_min (e.g. 0.2% at 1x).
- Compute a within-side outlier score on aggregated 5m descriptors.
- Drop the top `drop_outlier_frac` most unusual contexts ("confusing" tail).

This is intended to mirror the early BUY-side workflow: focus training on the most
self-consistent profitable contexts, while sacrificing the most idiosyncratic 10%.

Inputs
- data/oracle_precontext/oracle_daily_<SYMBOL>_pre_entry_dyn5m_<ts>.parquet

Outputs
- data/analysis_entry_inliers_outliertrim_<ts>/<SIDE>/
  - entry_context_agg_scored.parquet
  - inliers.csv / outliers.csv (trade_row lists + key stats)
  - retention_curve.csv
  - feature_return_spearman_inliers.csv
  - plots/*.png

Notes
- Outlier detection is done ONLY on complete contexts (no missing precontext minutes).
- Outlier scoring excludes trade_* columns (including the return label).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


FEATURE_COLS_DEFAULT = [
    "ret_1m_pct",
    "mom_3m_pct",
    "mom_5m_pct",
    "vol_std_5m",
    "range_5m",
    "range_norm_5m",
    "macd",
    "vwap_dev_5m",
]
CORE_COLS = ["open", "high", "low", "close", "volume"]


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _slope(y: np.ndarray) -> np.ndarray:
    """Vectorized slope for y over x=[-4,-3,-2,-1,0] (5 points). y shape (n,5)."""
    xc = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    denom = float(np.sum(xc * xc))  # 10.0
    return np.nansum(y * xc[None, :], axis=1) / denom


def _safe_nanstd(x: np.ndarray) -> np.ndarray:
    import warnings

    out = np.full(x.shape[0], np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m = np.nanmean(x, axis=1)
        ok = np.isfinite(m)
        if np.any(ok):
            out[ok] = np.nanmean((x[ok] - m[ok, None]) ** 2, axis=1) ** 0.5
    return out


def _agg_5(x: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean5 = np.nanmean(x, axis=1)
        min5 = np.nanmin(x, axis=1)
        max5 = np.nanmax(x, axis=1)

    return {
        f"{prefix}__last": x[:, -1],
        f"{prefix}__mean5": mean5,
        f"{prefix}__std5": _safe_nanstd(x),
        f"{prefix}__min5": min5,
        f"{prefix}__max5": max5,
        f"{prefix}__range5": max5 - min5,
        f"{prefix}__slope5": _slope(x),
    }


def _ensure_5x(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("trade_row")["rel_min"].size()
    bad = counts[counts != 5]
    if not bad.empty:
        raise SystemExit(f"Unexpected precontext row count per trade (expected 5). Bad trades: {len(bad)}")
    return df


def _spearman_top(df: pd.DataFrame, *, y_col: str, top_n: int = 60) -> pd.DataFrame:
    import warnings

    y = pd.to_numeric(df[y_col], errors="coerce")
    rows = []
    for c in df.columns:
        if c == y_col:
            continue
        if c.startswith("trade_"):
            continue
        if df[c].dtype.kind not in "fc":
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = s.corr(y, method="spearman")
        if pd.isna(corr):
            continue
        rows.append({"feature": c, "spearman": float(corr), "abs": float(abs(corr))})
    return pd.DataFrame(rows).sort_values("abs", ascending=False).head(int(top_n)).reset_index(drop=True)


def _build_entry_context_agg(in_path: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq

    # Load only needed columns.
    cols = [
        "trade_row",
        "rel_min",
        "anchor_time",
        "trade_date",
        "trade_entry_time",
        "trade_exit_time",
        "trade_duration_min",
        "trade_side",
        "trade_net_return_pct",
    ] + CORE_COLS + [c for c in FEATURE_COLS_DEFAULT]

    schema_cols = set(pq.read_schema(str(in_path)).names)
    cols = [c for c in cols if c in schema_cols]

    df = pd.read_parquet(in_path, columns=cols)

    df["trade_row"] = df["trade_row"].astype(np.int64)
    df["rel_min"] = df["rel_min"].astype(np.int64)
    for tcol in ["anchor_time", "trade_entry_time", "trade_exit_time"]:
        if tcol in df.columns:
            df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")

    df = df.sort_values(["trade_row", "rel_min"]).reset_index(drop=True)
    df = _ensure_5x(df)

    trade_rows = df["trade_row"].to_numpy(np.int64)
    uniq_trades = np.unique(trade_rows)
    n_trades = int(uniq_trades.size)

    rel = df["rel_min"].to_numpy(np.int64).reshape(n_trades, 5)
    expected = np.asarray([-5, -4, -3, -2, -1], dtype=np.int64)
    if not np.all(rel == expected[None, :]):
        raise SystemExit("Unexpected rel_min ordering/values; expected -5..-1 for each trade_row")

    meta_cols = [
        "trade_row",
        "trade_date",
        "trade_entry_time",
        "trade_exit_time",
        "trade_duration_min",
        "trade_side",
        "trade_net_return_pct",
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]
    meta = df.loc[df["rel_min"] == -1, meta_cols].copy().reset_index(drop=True)

    arrays: dict[str, np.ndarray] = {}
    for c in CORE_COLS + [c for c in FEATURE_COLS_DEFAULT if c in df.columns]:
        arrays[c] = df[c].to_numpy(np.float64, copy=False).reshape(n_trades, 5)

    close = arrays.get("close")
    if close is None:
        raise SystemExit("Expected 'close' in input")

    missing_close = np.isnan(close)
    missing_n = missing_close.sum(axis=1).astype(np.int64)

    close_last = close[:, -1]
    close_norm = (close / np.maximum(1e-12, close_last[:, None]) - 1.0) * 100.0

    vol = arrays.get("volume")
    vol_log = np.log1p(np.maximum(0.0, vol)) if vol is not None else None

    feats: dict[str, np.ndarray] = {}

    feats.update(_agg_5(close_norm, "px_close_norm_pct"))
    feats["px_close_norm_pct__ret5m"] = close_norm[:, -1] - close_norm[:, 0]
    feats["px_close_norm_pct__absret5m"] = np.abs(feats["px_close_norm_pct__ret5m"])

    if vol_log is not None:
        feats.update(_agg_5(vol_log, "vol_log1p"))

    for f in FEATURE_COLS_DEFAULT:
        if f not in arrays:
            continue
        feats.update(_agg_5(arrays[f], f))

    feats["missing_close_n"] = missing_n.astype(np.float64)
    feats["missing_any"] = (missing_n > 0).astype(np.float64)

    out = meta.copy()
    for k, v in feats.items():
        out[k] = v

    out["px_close_norm_pct__m5"] = close_norm[:, 0]
    out["px_close_norm_pct__m4"] = close_norm[:, 1]
    out["px_close_norm_pct__m3"] = close_norm[:, 2]
    out["px_close_norm_pct__m2"] = close_norm[:, 3]
    out["px_close_norm_pct__m1"] = close_norm[:, 4]

    return out


def _score_outliers_within_slice(df: pd.DataFrame, *, iso_n_estimators: int, nn_k: int, seed: int, iso_contamination: float) -> tuple[pd.DataFrame, list[str]]:
    """Return df_scored (same rows) and list of feature columns used for scoring."""
    out = df.copy().reset_index(drop=True)

    # Feature matrix
    drop_cols = {"trade_row"}
    X_cols = []
    for c in out.columns:
        if c in drop_cols:
            continue
        if c.startswith("trade_"):
            continue
        if out[c].dtype.kind not in "fc":
            continue
        X_cols.append(c)

    if not X_cols:
        raise SystemExit("No numeric feature columns found for outlier scoring")

    X = out[X_cols].to_numpy(np.float64)
    col_med = np.nanmedian(X, axis=0)
    inds = np.where(~np.isfinite(X))
    if inds[0].size:
        X[inds] = col_med[inds[1]]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=int(iso_n_estimators),
        contamination=float(iso_contamination),
        random_state=int(seed),
        n_jobs=-1,
    )
    iso.fit(Xs)
    outlier_score = -iso.decision_function(Xs)
    out["outlier_score"] = outlier_score

    nn_k = max(2, int(nn_k))
    nn = NearestNeighbors(n_neighbors=nn_k, metric="euclidean")
    nn.fit(Xs)
    dists, _ = nn.kneighbors(Xs, return_distance=True)
    out["nn_dist"] = dists[:, 1]

    return out, X_cols


def main() -> None:
    ap = argparse.ArgumentParser(description="Trim oracle entry precontext outliers within profitable slice")
    ap.add_argument(
        "--entry-precontext",
        default="data/oracle_precontext/oracle_daily_ETH-USD_pre_entry_dyn5m_2026-01-04T00-27-03Z.parquet",
        help="Entry precontext Parquet",
    )
    ap.add_argument("--out-dir", default="data", help="Root output dir")

    ap.add_argument("--side", default="SELL", choices=["BUY", "SELL", "buy", "sell"], help="Which side to analyze")
    ap.add_argument("--return-min", type=float, default=0.2, help="Keep only trades with trade_net_return_pct >= this")
    ap.add_argument(
        "--drop-outlier-frac",
        type=float,
        default=0.10,
        help="Drop this top fraction of highest outlier_score within the slice",
    )

    ap.add_argument("--iso-n-estimators", type=int, default=500)
    ap.add_argument("--iso-contamination", type=float, default=None, help="IsolationForest contamination (default: drop-outlier-frac)")
    ap.add_argument("--nn-k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--curve-max-drop", type=float, default=0.20, help="Retention curve up to this drop fraction")
    ap.add_argument("--curve-step", type=float, default=0.01)

    args = ap.parse_args()

    in_path = Path(args.entry_precontext)
    if not in_path.exists():
        raise SystemExit(f"Missing entry precontext parquet: {in_path}")

    side = str(args.side).upper()
    return_min = float(args.return_min)
    drop_frac = float(args.drop_outlier_frac)
    if not (0.0 <= drop_frac < 1.0):
        raise SystemExit("--drop-outlier-frac must be in [0,1)")

    iso_cont = float(args.iso_contamination) if args.iso_contamination is not None else drop_frac

    out_root = Path(args.out_dir)
    ts = _now_ts()
    out_dir = out_root / f"analysis_entry_inliers_outliertrim_{ts}" / side
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Build per-trade aggregated context features
    agg = _build_entry_context_agg(in_path)
    if "trade_side" not in agg.columns:
        raise SystemExit("Expected trade_side in aggregated dataset")

    agg = agg[agg["trade_side"].astype(str).str.upper() == side].copy().reset_index(drop=True)
    if agg.empty:
        raise SystemExit(f"No rows for side={side}")

    # Only complete contexts
    agg["missing_any"] = pd.to_numeric(agg.get("missing_any"), errors="coerce").fillna(1.0)
    agg_complete = agg[agg["missing_any"] == 0.0].copy().reset_index(drop=True)

    # Profitable slice
    agg_complete["trade_net_return_pct"] = pd.to_numeric(agg_complete["trade_net_return_pct"], errors="coerce")
    sl = agg_complete[agg_complete["trade_net_return_pct"] >= return_min].copy().reset_index(drop=True)

    if sl.empty:
        raise SystemExit(f"No trades meet side={side} and trade_net_return_pct >= {return_min}")

    # Score outliers within slice
    scored, X_cols = _score_outliers_within_slice(
        sl,
        iso_n_estimators=int(args.iso_n_estimators),
        nn_k=int(args.nn_k),
        seed=int(args.seed),
        iso_contamination=float(iso_cont),
    )

    # Determine cutoff by quantile within slice
    cutoff = float(np.nanquantile(scored["outlier_score"].to_numpy(np.float64), 1.0 - drop_frac)) if drop_frac > 0 else float("inf")
    scored["is_outlier_dropped"] = (scored["outlier_score"] > cutoff).astype(np.int64)

    inliers = scored[scored["is_outlier_dropped"] == 0].copy().reset_index(drop=True)
    outliers = scored[scored["is_outlier_dropped"] == 1].copy().reset_index(drop=True)

    # Save tables
    scored.to_parquet(out_dir / "entry_context_agg_scored.parquet", index=False)
    scored.sample(n=min(20000, len(scored)), random_state=42).to_csv(out_dir / "entry_context_agg_scored.sample.csv", index=False)

    keep_cols = [
        "trade_row",
        "trade_entry_time",
        "trade_exit_time",
        "trade_duration_min",
        "trade_net_return_pct",
        "outlier_score",
        "nn_dist",
    ]
    keep_cols = [c for c in keep_cols if c in scored.columns]

    inliers[keep_cols].sort_values("outlier_score").to_csv(out_dir / "inliers.csv", index=False)
    outliers[keep_cols].sort_values("outlier_score", ascending=False).to_csv(out_dir / "outliers.csv", index=False)

    # Diagnostics: retention curve
    curve = []
    max_drop = float(args.curve_max_drop)
    step = float(args.curve_step)
    drop_grid = np.arange(0.0, max_drop + 1e-12, step)
    scores = scored["outlier_score"].to_numpy(np.float64)
    rets = scored["trade_net_return_pct"].to_numpy(np.float64)

    for d in drop_grid:
        if d <= 0:
            m = np.full(len(scores), True, dtype=bool)
            c = float("inf")
        else:
            c = float(np.nanquantile(scores, 1.0 - d))
            m = scores <= c
        sub = rets[m]
        curve.append(
            {
                "drop_frac": float(d),
                "keep_n": int(np.isfinite(sub).sum()),
                "keep_frac": float(np.isfinite(sub).sum() / len(rets)),
                "ret_median": float(np.nanmedian(sub)) if sub.size else float("nan"),
                "ret_p90": float(np.nanquantile(sub, 0.9)) if sub.size else float("nan"),
                "ret_p99": float(np.nanquantile(sub, 0.99)) if sub.size else float("nan"),
                "cutoff_outlier_score": float(c),
            }
        )

    curve_df = pd.DataFrame(curve)
    curve_df.to_csv(out_dir / "retention_curve.csv", index=False)

    # Correlations within inliers
    corr = _spearman_top(inliers, y_col="trade_net_return_pct", top_n=80)
    corr.to_csv(out_dir / "feature_return_spearman_inliers.csv", index=False)

    # Plots
    plt.figure(figsize=(10, 5))
    plt.hist(scored["outlier_score"].to_numpy(np.float64), bins=80, alpha=0.85)
    plt.title(f"{side}: outlier_score (slice ret>={return_min}%)")
    plt.xlabel("outlier_score")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(plots_dir / "outlier_score_hist.png", dpi=170)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(scored["outlier_score"], scored["trade_net_return_pct"], s=4, alpha=0.18)
    plt.axvline(cutoff, color="#d62728", linewidth=1.2, alpha=0.9, label=f"drop cutoff (top {drop_frac:.0%})")
    plt.title(f"{side}: outlier_score vs trade_net_return_pct (slice ret>={return_min}%)")
    plt.xlabel("outlier_score")
    plt.ylabel("trade_net_return_pct")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plots_dir / "outlier_score_vs_return.png", dpi=170)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(curve_df["drop_frac"], curve_df["keep_n"], marker="o")
    plt.title(f"{side}: retention curve (slice ret>={return_min}%)")
    plt.xlabel("drop_frac")
    plt.ylabel("kept trades")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots_dir / "retention_curve_keep_n.png", dpi=170)
    plt.close()

    # Summary
    summary = {
        "entry_precontext": str(in_path),
        "side": side,
        "return_min": return_min,
        "drop_outlier_frac": drop_frac,
        "iso_contamination": iso_cont,
        "n_side_total": int(len(agg)),
        "n_side_complete": int(len(agg_complete)),
        "n_slice": int(len(scored)),
        "n_inliers": int(len(inliers)),
        "n_outliers_dropped": int(len(outliers)),
        "cutoff_outlier_score": cutoff,
        "features_used_n": int(len(X_cols)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("Wrote", out_dir)


if __name__ == "__main__":
    main()
