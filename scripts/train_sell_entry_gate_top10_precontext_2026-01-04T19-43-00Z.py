#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T19:43:00Z
"""Train a clean good-vs-bad gate on the baseline SELL entry model's top-10% selection.

Why
- The inlier trade dataset is (by design) mostly/all profitable, so it doesn't contain "bad" examples.
- To "cleanly separate" good vs bad without sacrificing trade count, we train a classifier on the
  *actual top-10% selected minutes*, then use it as a re-ranker.

Definitions (within top-10% selection)
- y_sell[t] = net_ret_pct_sell(close[t], min(close[t+1..t+15]), fee_side)
- good = y_sell >= ret_thresh (default 0.2)
- bad  = y_sell < bad_thresh (default 0.0)
- neutrals excluded from training

Features (ONLY 5-minute precontext)
- The 79-feature precontext descriptor set:
  px_close_norm_pct*, vol_log1p*, and BASE_FEATS_DEFAULT aggregated over t-5..t-1.

Outputs (timestamped)
- data/entry_gate_sell_top10_precontext_<ts>/
  - summary_<ts>.json
  - gate_metrics_<ts>.json
  - gate_feature_importance_<ts>.csv
  - gate_retention_curve_<ts>.csv
  - models/entry_gate_sell_top10_precontext_<ts>.joblib

Model usage
- At inference, score every valid minute with this gate model (using the same 5m precontext features).
- Select top 10% by gate probability to keep trade count constant.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier


BASE_FEATS_DEFAULT = [
    "ret_1m_pct",
    "mom_3m_pct",
    "mom_5m_pct",
    "vol_std_5m",
    "range_5m",
    "range_norm_5m",
    "macd",
    "vwap_dev_5m",
]


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def net_ret_pct_sell(entry_px: np.ndarray, exit_px: np.ndarray, f: float) -> np.ndarray:
    entry_px = np.asarray(entry_px, dtype=np.float64)
    exit_px = np.asarray(exit_px, dtype=np.float64)
    mult = (entry_px * (1.0 - f)) / (exit_px * (1.0 + f))
    return (mult - 1.0) * 100.0


def _future_extrema_excl_current(x: np.ndarray, W: int, *, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    b = x[::-1]

    ans_rev = np.full(n, np.nan, dtype=np.float64)
    dq: deque[int] = deque()

    def better(a: float, b: float) -> bool:
        return (a >= b) if mode == "max" else (a <= b)

    for j in range(n):
        while dq and dq[0] < j - W:
            dq.popleft()

        ans_rev[j] = b[dq[0]] if dq else np.nan

        v = b[j]
        if np.isfinite(v):
            while dq and np.isfinite(b[dq[-1]]) and better(v, b[dq[-1]]):
                dq.pop()
            dq.append(j)
        else:
            dq.append(j)

    return ans_rev[::-1]


def _safe_nanstd_rows(x: np.ndarray) -> np.ndarray:
    import warnings

    out = np.full(x.shape[0], np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m = np.nanmean(x, axis=1)
        ok = np.isfinite(m)
        if np.any(ok):
            out[ok] = np.nanmean((x[ok] - m[ok, None]) ** 2, axis=1) ** 0.5
    return out


def _slope_5_rows(y: np.ndarray) -> np.ndarray:
    xc = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    return np.nansum(y * xc[None, :], axis=1) / 10.0


def _agg5_rows(x: np.ndarray) -> dict[str, np.ndarray]:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean5 = np.nanmean(x, axis=1)
        min5 = np.nanmin(x, axis=1)
        max5 = np.nanmax(x, axis=1)

    return {
        "__last": x[:, -1],
        "__mean5": mean5,
        "__std5": _safe_nanstd_rows(x),
        "__min5": min5,
        "__max5": max5,
        "__range5": max5 - min5,
        "__slope5": _slope_5_rows(x),
    }


def compute_precontext_features_for_indices(bars: pd.DataFrame, idx: np.ndarray, *, pre_min: int) -> pd.DataFrame:
    if int(pre_min) != 5:
        raise SystemExit("This script currently supports --pre-min=5 only")

    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(np.float64, copy=False)
    volume = pd.to_numeric(bars["volume"], errors="coerce").to_numpy(np.float64, copy=False)
    base_arrs = {c: pd.to_numeric(bars[c], errors="coerce").to_numpy(np.float64, copy=False) for c in BASE_FEATS_DEFAULT}

    ii = np.asarray(idx, dtype=np.int64)
    i0, i1, i2, i3, i4 = ii - 5, ii - 4, ii - 3, ii - 2, ii - 1

    out: dict[str, np.ndarray] = {}

    close_w = np.stack([close[i0], close[i1], close[i2], close[i3], close[i4]], axis=1)
    close_last = close_w[:, -1]
    close_last = np.where((np.isfinite(close_last)) & (close_last > 0.0), close_last, np.nan)
    close_norm = (close_w / close_last[:, None] - 1.0) * 100.0

    px = _agg5_rows(close_norm)
    for suf, arr in px.items():
        out[f"px_close_norm_pct{suf}"] = arr
    px_ret5m = close_norm[:, -1] - close_norm[:, 0]
    out["px_close_norm_pct__ret5m"] = px_ret5m
    out["px_close_norm_pct__absret5m"] = np.abs(px_ret5m)
    out["px_close_norm_pct__m5"] = close_norm[:, 0]
    out["px_close_norm_pct__m4"] = close_norm[:, 1]
    out["px_close_norm_pct__m3"] = close_norm[:, 2]
    out["px_close_norm_pct__m2"] = close_norm[:, 3]
    out["px_close_norm_pct__m1"] = close_norm[:, 4]

    miss = ~np.isfinite(close_w)
    miss_n = miss.sum(axis=1).astype(np.float64)
    out["missing_close_n"] = miss_n
    out["missing_any"] = (miss_n > 0).astype(np.float64)

    vol_w = np.stack([volume[i0], volume[i1], volume[i2], volume[i3], volume[i4]], axis=1)
    vol_log = np.log1p(np.maximum(0.0, vol_w))
    vv = _agg5_rows(vol_log)
    for suf, arr in vv.items():
        out[f"vol_log1p{suf}"] = arr

    for name, arr0 in base_arrs.items():
        w = np.stack([arr0[i0], arr0[i1], arr0[i2], arr0[i3], arr0[i4]], axis=1)
        a = _agg5_rows(w)
        for suf, v in a.items():
            out[f"{name}{suf}"] = v

    df = pd.DataFrame(out)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Train SELL gate on baseline top-10% selection (5m precontext only)")
    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="ETH 1m OHLCV+features parquet",
    )
    ap.add_argument(
        "--baseline-entry-model",
        default="data/entry_regressor_inliers_sell_ctx120_2026-01-04T18-55-10Z/models/entry_regressor_sell_inliers_ctx120_2026-01-04T18-55-10Z.joblib",
        help="Baseline ctx120 SELL entry model used to define top-10% selection",
    )
    ap.add_argument("--fee-total", type=float, default=0.001)
    ap.add_argument("--horizon-min", type=int, default=15)
    ap.add_argument("--pre-min", type=int, default=5)

    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--ret-thresh", type=float, default=0.2)
    ap.add_argument("--bad-thresh", type=float, default=0.0)

    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--n-estimators", type=int, default=1200)

    ap.add_argument("--chunk", type=int, default=120_000)
    ap.add_argument("--out-dir", default="data")

    args = ap.parse_args()

    bars_path = Path(args.bars)
    if not bars_path.exists():
        raise SystemExit(f"bars not found: {bars_path}")

    base_art = joblib.load(args.baseline_entry_model)
    if not isinstance(base_art, dict) or "model" not in base_art or "feature_cols" not in base_art:
        raise SystemExit("baseline model must be a dict with keys: model, feature_cols")

    base_model = base_art["model"]
    base_feature_cols = list(base_art["feature_cols"])

    fee_side = float(args.fee_total) / 2.0
    horizon = int(args.horizon_min)
    pre_min = int(args.pre_min)
    top_frac = float(args.top_frac)
    ret_thresh = float(args.ret_thresh)
    bad_thresh = float(args.bad_thresh)

    need_cols = ["timestamp", "open", "high", "low", "close", "volume"] + list(BASE_FEATS_DEFAULT)

    print(f"Loading bars: {bars_path}", flush=True)
    bars = pd.read_parquet(bars_path, columns=need_cols)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    if not bars["timestamp"].is_monotonic_increasing:
        bars = bars.sort_values("timestamp").reset_index(drop=True)

    n = int(len(bars))
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(np.float64, copy=False)

    print(f"Computing y_sell within {horizon}m...", flush=True)
    fut_min = _future_extrema_excl_current(close, horizon, mode="min")
    y_sell = net_ret_pct_sell(close, fut_min, fee_side)

    valid = (np.arange(n) >= pre_min) & np.isfinite(y_sell) & np.isfinite(close)
    idxs = np.where(valid)[0]

    # Score baseline model on all valid minutes using the same chunked feature builder used in backtest
    print("Scoring baseline model to define top-10% selection...", flush=True)

    # Reuse the chunked scoring approach from backtest script: build Xc only for required feature cols
    col_to_j = {c: j for j, c in enumerate(base_feature_cols)}

    # Determine series prefixes needed
    prefixes = set()
    for c in base_feature_cols:
        if "__" in c:
            prefixes.add(c.split("__", 1)[0])
    prefixes.discard("px_close_norm_pct")
    prefixes.discard("vol_log1p")

    base_arrs = {c: pd.to_numeric(bars[c], errors="coerce").to_numpy(np.float64, copy=False) for c in BASE_FEATS_DEFAULT}

    series_map: dict[str, np.ndarray] = {}
    for p in prefixes:
        if p in base_arrs:
            series_map[p] = base_arrs[p]

    volume = pd.to_numeric(bars["volume"], errors="coerce").to_numpy(np.float64, copy=False)

    pred = np.full(n, np.nan, dtype=np.float32)

    def _set(X: np.ndarray, name: str, arr: np.ndarray) -> None:
        j = col_to_j.get(name)
        if j is None:
            return
        X[:, j] = np.asarray(arr, dtype=np.float32)

    chunk = int(args.chunk)
    for s in range(0, idxs.size, chunk):
        e = min(idxs.size, s + chunk)
        ii = idxs[s:e]
        m = ii.size
        Xc = np.full((m, len(base_feature_cols)), np.nan, dtype=np.float32)

        i0, i1, i2, i3, i4 = ii - 5, ii - 4, ii - 3, ii - 2, ii - 1

        close_w = np.stack([close[i0], close[i1], close[i2], close[i3], close[i4]], axis=1)
        close_last = close_w[:, -1]
        close_last = np.where((np.isfinite(close_last)) & (close_last > 0.0), close_last, np.nan)
        close_norm = (close_w / close_last[:, None] - 1.0) * 100.0

        px = _agg5_rows(close_norm)
        for suf, arr in px.items():
            _set(Xc, f"px_close_norm_pct{suf}", arr)
        px_ret5m = close_norm[:, -1] - close_norm[:, 0]
        _set(Xc, "px_close_norm_pct__ret5m", px_ret5m)
        _set(Xc, "px_close_norm_pct__absret5m", np.abs(px_ret5m))

        vol_w = np.stack([volume[i0], volume[i1], volume[i2], volume[i3], volume[i4]], axis=1)
        vol_log = np.log1p(np.maximum(0.0, vol_w))
        vv = _agg5_rows(vol_log)
        for suf, arr in vv.items():
            _set(Xc, f"vol_log1p{suf}", arr)

        if "missing_close_n" in col_to_j or "missing_any" in col_to_j:
            miss = ~np.isfinite(close_w)
            miss_n = miss.sum(axis=1).astype(np.float64)
            _set(Xc, "missing_close_n", miss_n)
            _set(Xc, "missing_any", (miss_n > 0).astype(np.float64))

        for name, arr0 in series_map.items():
            w = np.stack([arr0[i0], arr0[i1], arr0[i2], arr0[i3], arr0[i4]], axis=1)
            a = _agg5_rows(w)
            for suf, v in a.items():
                _set(Xc, f"{name}{suf}", v)

        pred[ii] = base_model.predict(Xc).astype(np.float32)

        if (s // chunk) % 2 == 0:
            print(f"  scored {e:,}/{idxs.size:,}", flush=True)

    pv = pred[valid].astype(np.float64)
    thr_top = float(np.nanquantile(pv, 1.0 - top_frac))
    sel_mask = valid & np.isfinite(pred) & (pred >= thr_top)
    sel_idx = np.where(sel_mask)[0]

    y_sel = y_sell[sel_idx]
    good = y_sel >= ret_thresh
    bad = y_sel < bad_thresh
    keep = good | bad

    idx_train_all = sel_idx[keep]
    y_train_all = good[keep].astype(np.int32)  # 1 for good, 0 for bad

    # Build 5m precontext features for selected rows
    print(f"Building 5m precontext features for gate dataset: n={idx_train_all.size:,} ...", flush=True)
    X_df = compute_precontext_features_for_indices(bars, idx_train_all, pre_min=pre_min)
    X = X_df.to_numpy(np.float32, copy=False)

    # chrono split by timestamp of idx
    ts_sel = bars.loc[idx_train_all, "timestamp"].to_numpy(dtype="datetime64[ns]")
    order = np.argsort(ts_sel)
    X = X[order]
    y = y_train_all[order]

    n_gate = int(len(y))
    split = int(n_gate * (1.0 - float(args.test_frac)))
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    clf = LGBMClassifier(
        n_estimators=int(args.n_estimators),
        learning_rate=0.03,
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
        class_weight="balanced",
    )

    clf.fit(X_tr, y_tr)
    proba_te = clf.predict_proba(X_te)[:, 1].astype(np.float64)

    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y_te, proba_te)) if len(np.unique(y_te)) == 2 else float("nan")
    except Exception:
        auc = float("nan")

    # retention curve on test: keep top q by proba
    y_sell_te = y_sel[keep][order][split:]
    qs = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    curve = []
    for q in qs:
        thr = float(np.quantile(proba_te, 1.0 - q))
        m = proba_te >= thr
        sel_y = y_sell_te[m]
        curve.append(
            {
                "keep_top_frac": float(q),
                "threshold": float(thr),
                "selected": int(m.sum()),
                "good_rate_y>=ret": float(np.mean(sel_y >= ret_thresh)) if sel_y.size else float("nan"),
                "bad_rate_y<bad": float(np.mean(sel_y < bad_thresh)) if sel_y.size else float("nan"),
                "median_y": float(np.median(sel_y)) if sel_y.size else float("nan"),
            }
        )

    ts = now_ts()
    out_root = Path(args.out_dir) / f"entry_gate_sell_top10_precontext_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    models_dir = out_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "created_utc": ts,
        "bars": str(bars_path),
        "baseline_entry_model": str(Path(args.baseline_entry_model)),
        "fee_total": float(args.fee_total),
        "fee_side": float(fee_side),
        "horizon_min": int(horizon),
        "pre_min": int(pre_min),
        "top_frac": float(top_frac),
        "top_pred_threshold": float(thr_top),
        "selected_n": int(sel_idx.size),
        "selected_good_n": int(good.sum()),
        "selected_bad_n": int(bad.sum()),
        "selected_neutral_n": int((~(good | bad)).sum()),
        "gate_dataset_n": int(n_gate),
        "gate_test_auc": float(auc),
        "ret_thresh": float(ret_thresh),
        "bad_thresh": float(bad_thresh),
        "n_gate_features": int(X_df.shape[1]),
    }

    (out_root / f"summary_{ts}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_root / f"gate_metrics_{ts}.json").write_text(json.dumps({"auc_test": float(auc)}, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(curve).to_csv(out_root / f"gate_retention_curve_{ts}.csv", index=False)

    # importance
    imp = pd.DataFrame(
        {
            "feature": list(X_df.columns),
            "importance_gain": clf.booster_.feature_importance(importance_type="gain"),
            "importance_split": clf.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    imp.to_csv(out_root / f"gate_feature_importance_{ts}.csv", index=False)

    artifact = {
        "created_utc": ts,
        "side": "SELL",
        "pre_min": int(pre_min),
        "label": "good_vs_bad_in_top10",
        "label_def": {"good": float(ret_thresh), "bad": float(bad_thresh)},
        "feature_cols": list(X_df.columns),
        "model": clf,
        "baseline_top_threshold": float(thr_top),
        "metrics": summary,
    }

    model_path = models_dir / f"entry_gate_sell_top10_precontext_{ts}.joblib"
    joblib.dump(artifact, model_path)

    print(json.dumps(summary, indent=2))
    print("Saved gate model:", model_path)
    print("Wrote", out_root)


if __name__ == "__main__":
    main()
