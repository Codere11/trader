#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T00:35:01Z
"""Deep analysis of oracle trader *entry* 5-minute precontext data.

Focus
- How spread / diverse the 5-minute pre-entry contexts are (similarity, clusters).
- Find unusual/outlier entry contexts for inspection.
- Do everything split by trade side (BUY vs SELL), because pre-entry dynamics are not symmetric.

Inputs
- Entry precontext Parquet produced by oracle trader runner:
    data/oracle_precontext/oracle_daily_ETH-USD_pre_entry_dyn5m_<ts>.parquet

Outputs
- data/analysis_oracle_entry_outliers_by_side_<ts>/
  - BUY/...
  - SELL/...
Each side folder contains:
  - entry_context_agg.parquet
  - entry_context_agg_scored.parquet (adds outlier_score + nn_dist)
  - entry_outliers.csv
  - feature_return_spearman.csv (top correlations vs trade_net_return_pct)
  - plots/*.png

Notes
- We summarize each 5-minute precontext sequence into per-trade descriptors.
- Outlier detection + nearest-neighbor distances are computed *within-side*.
- Primary outlier detection is done on complete contexts (no missing minutes).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
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
    # x mean is -2, centered x = [-2,-1,0,1,2]
    xc = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    denom = float(np.sum(xc * xc))  # 10.0
    # slope = sum(xc*y) / sum(xc^2)
    return np.nansum(y * xc[None, :], axis=1) / denom


def _safe_nanstd(x: np.ndarray) -> np.ndarray:
    # np.nanstd emits warnings for all-nan slices; suppress by manual.
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
        # When a trade has missing precontext minutes (NaNs), some slices are all-NaN;
        # we keep the resulting NaNs, but we don't want noisy RuntimeWarnings.
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
    # Expect exactly 5 rows per trade, rel_min in [-5..-1].
    counts = df.groupby("trade_row")["rel_min"].size()
    bad = counts[counts != 5]
    if not bad.empty:
        raise SystemExit(f"Unexpected precontext row count per trade (expected 5). Bad trades: {len(bad)}")
    return df


def _side_dir(root: Path, side: str) -> Path:
    d = root / str(side).upper()
    d.mkdir(parents=True, exist_ok=True)
    (d / "plots").mkdir(parents=True, exist_ok=True)
    return d


def _spearman_top(df: pd.DataFrame, *, y_col: str, top_n: int = 40) -> pd.DataFrame:
    # Use pandas' Spearman to avoid extra deps.
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
            # When a feature is constant over the slice, Spearman is undefined; ignore that warning.
            warnings.simplefilter("ignore")
            corr = s.corr(y, method="spearman")
        if pd.isna(corr):
            continue
        rows.append({"feature": c, "spearman": float(corr), "abs": float(abs(corr))})
    out = pd.DataFrame(rows).sort_values("abs", ascending=False).head(int(top_n)).reset_index(drop=True)
    return out


def _plot_close_path_templates(out_scored: pd.DataFrame, plots_dir: Path) -> None:
    # Compare typical vs top-winner paths within a side.
    cols = [
        "px_close_norm_pct__m5",
        "px_close_norm_pct__m4",
        "px_close_norm_pct__m3",
        "px_close_norm_pct__m2",
        "px_close_norm_pct__m1",
    ]
    if not all(c in out_scored.columns for c in cols):
        return

    r = pd.to_numeric(out_scored["trade_net_return_pct"], errors="coerce")
    if r.notna().sum() < 1000:
        return

    q99 = float(r.quantile(0.99))
    all_df = out_scored
    top_df = out_scored[r >= q99]

    x = np.array([-5, -4, -3, -2, -1], dtype=np.int64)

    def _band(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mat = df[cols].to_numpy(np.float64)
        med = np.nanmedian(mat, axis=0)
        p25 = np.nanpercentile(mat, 25, axis=0)
        p75 = np.nanpercentile(mat, 75, axis=0)
        return med, p25, p75

    med_all, p25_all, p75_all = _band(all_df)
    med_top, p25_top, p75_top = _band(top_df)

    plt.figure(figsize=(10, 5))
    plt.fill_between(x, p25_all, p75_all, alpha=0.18, color="#1f77b4", label="all p25-p75")
    plt.plot(x, med_all, marker="o", color="#1f77b4", label="all median")
    plt.fill_between(x, p25_top, p75_top, alpha=0.18, color="#d62728", label="top1% p25-p75")
    plt.plot(x, med_top, marker="o", color="#d62728", label="top1% median")
    plt.axhline(0.0, color="#888", linewidth=0.8)
    plt.title(f"Normalized close path templates (all vs top1% winners; q99={q99:.4f}%)")
    plt.xlabel("rel_min")
    plt.ylabel("close norm %")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plots_dir / "close_path_templates_all_vs_top1pct.png", dpi=170)
    plt.close()


def _analyze_side(out_all: pd.DataFrame, *, side: str, root: Path, sample_n: int, outliers_top: int, iso_contamination: float, nn_k: int) -> None:
    side = str(side).upper()
    side_dir = _side_dir(root, side)
    plots_dir = side_dir / "plots"

    out = out_all[out_all["trade_side"].astype(str).str.upper() == side].copy()
    if out.empty:
        print(f"Side {side}: no rows")
        return

    # Save aggregated
    out_parq = side_dir / "entry_context_agg.parquet"
    out_sample_csv = side_dir / "entry_context_agg.sample.csv"
    out.to_parquet(out_parq, index=False)
    out.sample(n=min(20000, len(out)), random_state=42).to_csv(out_sample_csv, index=False)

    # Return distributions
    ret = pd.to_numeric(out.get("trade_net_return_pct"), errors="coerce").to_numpy(np.float64)
    plt.figure(figsize=(10, 5))
    plt.hist(ret[np.isfinite(ret)], bins=80, color="#1f77b4", alpha=0.85)
    plt.title(f"{side}: trade net_return_pct distribution")
    plt.xlabel("net_return_pct")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(plots_dir / "returns_hist.png", dpi=160)
    plt.close()

    q99 = float(np.nanquantile(ret, 0.99))
    plt.figure(figsize=(10, 5))
    plt.hist(ret[(np.isfinite(ret)) & (ret >= q99)], bins=60, color="#ff7f0e", alpha=0.85)
    plt.title(f"{side}: net_return_pct top 1% tail (>= {q99:.4f}%)")
    plt.xlabel("net_return_pct")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(plots_dir / "returns_tail_top1pct_hist.png", dpi=160)
    plt.close()

    # Outlier detection (complete contexts only)
    mask_complete = out["missing_any"].to_numpy(np.float64) == 0.0
    out_complete = out.loc[mask_complete].reset_index(drop=True)

    # Feature matrix
    drop_cols = {"trade_row"}
    X_cols = []
    for c in out_complete.columns:
        if c in drop_cols:
            continue
        if c.startswith("trade_"):
            continue
        if out_complete[c].dtype.kind not in "fc":
            continue
        X_cols.append(c)

    X = out_complete[X_cols].to_numpy(np.float64)
    col_med = np.nanmedian(X, axis=0)
    inds = np.where(~np.isfinite(X))
    if inds[0].size:
        X[inds] = col_med[inds[1]]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=300,
        contamination=float(iso_contamination),
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(Xs)
    out_score = -iso.decision_function(Xs)
    out_complete["outlier_score"] = out_score

    nn_k = max(2, int(nn_k))
    nn = NearestNeighbors(n_neighbors=nn_k, metric="euclidean")
    nn.fit(Xs)
    dists, _ = nn.kneighbors(Xs, return_distance=True)
    out_complete["nn_dist"] = dists[:, 1]

    # Attach scores back
    scores = out_complete[["trade_row", "outlier_score", "nn_dist"]].copy()
    out_scored = out.merge(scores, on="trade_row", how="left")

    out_scored_parq = side_dir / "entry_context_agg_scored.parquet"
    out_scored_sample_csv = side_dir / "entry_context_agg_scored.sample.csv"
    out_scored.to_parquet(out_scored_parq, index=False)
    out_scored.sample(n=min(20000, len(out_scored)), random_state=42).to_csv(out_scored_sample_csv, index=False)

    # PCA visualization
    if sample_n > 0 and len(out_complete) > sample_n:
        samp = out_complete.sample(n=sample_n, random_state=42).reset_index(drop=True)
        Xs_s = scaler.transform(samp[X_cols].to_numpy(np.float64))
    else:
        samp = out_complete
        Xs_s = Xs

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs_s)
    samp["pca0"] = Z[:, 0]
    samp["pca1"] = Z[:, 1]

    r = pd.to_numeric(samp["trade_net_return_pct"], errors="coerce").to_numpy(np.float64)
    q50 = float(np.nanquantile(r, 0.50))
    q90 = float(np.nanquantile(r, 0.90))
    q99s = float(np.nanquantile(r, 0.99))
    bucket = np.full(len(r), 0, dtype=np.int64)
    bucket[(r >= q50) & (r < q90)] = 1
    bucket[(r >= q90) & (r < q99s)] = 2
    bucket[r >= q99s] = 3

    colors = np.array(["#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728"], dtype=object)
    plt.figure(figsize=(10, 7))
    plt.scatter(samp["pca0"], samp["pca1"], s=6, c=colors[bucket], alpha=0.35, linewidths=0)
    plt.title(f"{side}: entry precontext descriptors PCA (color by return bucket)")
    plt.xlabel("PC0")
    plt.ylabel("PC1")
    plt.tight_layout()
    plt.savefig(plots_dir / "pca_scatter_return_buckets.png", dpi=170)
    plt.close()

    # Score vs return
    plt.figure(figsize=(10, 6))
    plt.scatter(out_complete["outlier_score"], out_complete["trade_net_return_pct"], s=4, alpha=0.2)
    plt.title(f"{side}: outlier_score vs net_return_pct")
    plt.xlabel("outlier_score (higher=more unusual)")
    plt.ylabel("net_return_pct")
    plt.tight_layout()
    plt.savefig(plots_dir / "outlier_score_vs_return.png", dpi=170)
    plt.close()

    # NN dist histogram
    plt.figure(figsize=(10, 5))
    plt.hist(out_complete["nn_dist"].to_numpy(np.float64), bins=80, color="#2ca02c", alpha=0.85)
    plt.title(f"{side}: nearest-neighbor distance (spread/novelty)")
    plt.xlabel("nn_dist")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(plots_dir / "nn_dist_hist.png", dpi=170)
    plt.close()

    # Outliers table
    outliers = out_complete.sort_values("outlier_score", ascending=False).head(int(outliers_top)).copy()
    out_csv = side_dir / "entry_outliers.csv"
    outliers.to_csv(out_csv, index=False)

    # Feature-return correlations
    corr = _spearman_top(out_scored, y_col="trade_net_return_pct", top_n=50)
    corr.to_csv(side_dir / "feature_return_spearman.csv", index=False)

    # Template paths
    _plot_close_path_templates(out_scored, plots_dir)

    q_nn = np.nanquantile(out_complete["nn_dist"].to_numpy(np.float64), [0.5, 0.9, 0.99])
    q_out = np.nanquantile(out_complete["outlier_score"].to_numpy(np.float64), [0.5, 0.9, 0.99])
    print(f"{side}: trades={len(out):,} complete={len(out_complete):,} q99_ret={q99:.4f}%")
    print(f"{side}: nn_dist p50/p90/p99 = {[float(x) for x in q_nn]}")
    print(f"{side}: outlier_score p50/p90/p99 = {[float(x) for x in q_out]}")
    print(f"{side}: wrote {side_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze oracle entry precontext spread + outliers (BUY vs SELL split)")
    ap.add_argument(
        "--entry-precontext",
        default="data/oracle_precontext/oracle_daily_ETH-USD_pre_entry_dyn5m_2026-01-04T00-27-03Z.parquet",
        help="Entry precontext Parquet",
    )
    ap.add_argument("--out-dir", default="data", help="Root output dir")
    ap.add_argument("--symbol", default="ETH-USD")

    ap.add_argument("--sample-n", type=int, default=30000, help="Sample size for PCA/outlier plots per side (0=use all)")
    ap.add_argument("--outliers-top", type=int, default=300)

    ap.add_argument("--iso-contamination", type=float, default=0.01)
    ap.add_argument("--nn-k", type=int, default=2)

    ap.add_argument("--side", default="both", choices=["both", "BUY", "SELL", "buy", "sell"], help="Analyze one side or both")

    args = ap.parse_args()

    in_path = Path(args.entry_precontext)
    if not in_path.exists():
        raise SystemExit(f"Missing entry precontext parquet: {in_path}")

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

    # Some files may not have all columns (older runs); drop missing.
    import pyarrow.parquet as pq

    schema_cols = set(pq.read_schema(str(in_path)).names)
    cols = [c for c in cols if c in schema_cols]

    df = pd.read_parquet(in_path, columns=cols)

    # Normalize types
    df["trade_row"] = df["trade_row"].astype(np.int64)
    df["rel_min"] = df["rel_min"].astype(np.int64)

    for tcol in ["anchor_time", "trade_entry_time", "trade_exit_time"]:
        if tcol in df.columns:
            df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")

    # Ensure consistent ordering, then reshape.
    df = df.sort_values(["trade_row", "rel_min"]).reset_index(drop=True)
    df = _ensure_5x(df)

    trade_rows = df["trade_row"].to_numpy(np.int64)
    uniq_trades = np.unique(trade_rows)
    n_trades = int(uniq_trades.size)

    # Verify rel_min pattern is the expected [-5,-4,-3,-2,-1] for each trade.
    rel = df["rel_min"].to_numpy(np.int64).reshape(n_trades, 5)
    expected = np.asarray([-5, -4, -3, -2, -1], dtype=np.int64)
    if not np.all(rel == expected[None, :]):
        # If this triggers, downstream aggregation is unsafe.
        raise SystemExit("Unexpected rel_min ordering/values; expected -5..-1 for each trade_row")

    # Base metadata (one row per trade)
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

    # Extract 5x arrays
    arrays: dict[str, np.ndarray] = {}
    for c in CORE_COLS + [c for c in FEATURE_COLS_DEFAULT if c in df.columns]:
        arrays[c] = df[c].to_numpy(np.float64, copy=False).reshape(n_trades, 5)

    # Missingness (price)
    close = arrays.get("close")
    if close is None:
        raise SystemExit("Expected 'close' in input")

    missing_close = np.isnan(close)
    missing_n = missing_close.sum(axis=1).astype(np.int64)

    # Normalize close path by last close (rel=-1)
    close_last = close[:, -1]
    close_norm = (close / np.maximum(1e-12, close_last[:, None]) - 1.0) * 100.0

    # Normalize OHLC similarly (optional)
    ohlc_norm = {}
    for c in ["open", "high", "low"]:
        if c in arrays:
            ohlc_norm[c] = (arrays[c] / np.maximum(1e-12, close_last[:, None]) - 1.0) * 100.0

    # Volume transform
    vol = arrays.get("volume")
    if vol is not None:
        vol_log = np.log1p(np.maximum(0.0, vol))
    else:
        vol_log = None

    feats: dict[str, np.ndarray] = {}

    # Price shape descriptors
    feats.update(_agg_5(close_norm, "px_close_norm_pct"))
    feats["px_close_norm_pct__ret5m"] = close_norm[:, -1] - close_norm[:, 0]
    feats["px_close_norm_pct__absret5m"] = np.abs(feats["px_close_norm_pct__ret5m"])

    if vol_log is not None:
        feats.update(_agg_5(vol_log, "vol_log1p"))

    # Feature descriptors
    for f in FEATURE_COLS_DEFAULT:
        if f not in arrays:
            continue
        feats.update(_agg_5(arrays[f], f))

    # Add missingness
    feats["missing_close_n"] = missing_n.astype(np.float64)
    feats["missing_any"] = (missing_n > 0).astype(np.float64)

    # Build aggregated dataset
    out = meta.copy()
    for k, v in feats.items():
        out[k] = v

    # Add raw normalized path columns for inspection (5 dims)
    out["px_close_norm_pct__m5"] = close_norm[:, 0]
    out["px_close_norm_pct__m4"] = close_norm[:, 1]
    out["px_close_norm_pct__m3"] = close_norm[:, 2]
    out["px_close_norm_pct__m2"] = close_norm[:, 3]
    out["px_close_norm_pct__m1"] = close_norm[:, 4]

    # Output directory (by side)
    out_root = Path(args.out_dir)
    ts = _now_ts()
    out_dir = out_root / f"analysis_oracle_entry_outliers_by_side_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "trade_side" not in out.columns:
        raise SystemExit("Expected trade_side in aggregated dataset. Re-run oracle precontext generation with trade metadata.")

    side_arg = str(args.side).upper()
    sides = ["BUY", "SELL"] if side_arg == "BOTH" else [side_arg]

    for side in sides:
        _analyze_side(
            out,
            side=side,
            root=out_dir,
            sample_n=int(args.sample_n),
            outliers_top=int(args.outliers_top),
            iso_contamination=float(args.iso_contamination),
            nn_k=int(args.nn_k),
        )

    print("Wrote side-split analysis to", out_dir)


if __name__ == "__main__":
    main()
