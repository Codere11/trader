#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T19:51:10Z
"""Train SELL entry regressor on full ETH 1m dataset with ctx120 features + weighting.

You chose approach A:
- Train a regressor for y_sell (15m oracle best-exit net return).
- Keep trade count/coverage the same; improve ranking by emphasizing the precontext patterns
  that separate good vs bad inside the *top-10% selection*.

How weighting works
- We load a feature shift file computed inside the top-10% selection:
  feature_shifts_good_vs_bad_*.csv
- Take the top-K features by abs(robust_med_shift_good_minus_bad).
- For each selected feature f, define direction = sign(med_good - med_bad) and midpoint = (med_good+med_bad)/2.
- For each row, compute z_f = direction * (x_f - midpoint) / robust_scale(f).
- Convert to [0,1] via sigmoid and average across features.
- Final sample_weight = 1 + alpha * avg_sigmoid.

This does NOT drop data; it only reweights training.

Outputs (timestamped)
- data/entry_regressor_oracle15m_sell_ctx120_weighted_<ts>/
  - run_meta_<ts>.json
  - metrics_<ts>.json
  - feature_importance_<ts>.csv
  - models/entry_regressor_sell_oracle15m_ctx120_weighted_<ts>.joblib
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

from lightgbm import LGBMRegressor
from lightgbm.callback import log_evaluation


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


def _rolling_prev_matrix(x: np.ndarray, L: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    mat = np.full((n, L), np.nan, dtype=np.float64)
    for k in range(1, L + 1):
        mat[k:, L - k] = x[: n - k]
    return mat


def _slope_5(y: np.ndarray) -> np.ndarray:
    xc = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    return np.nansum(y * xc[None, :], axis=1) / 10.0


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
        f"{prefix}__slope5": _slope_5(x),
    }


def _rolling_sum_nan_prefix(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    cs = np.cumsum(x)
    out[w - 1 :] = cs[w - 1 :] - np.concatenate(([0.0], cs[: n - w]))
    return out


def _compute_ctx_series(df_bars: pd.DataFrame, windows: list[int]) -> dict[str, np.ndarray]:
    """Compute causal context series at each minute using only t-1 and earlier."""
    close_prev = pd.to_numeric(df_bars["close"], errors="coerce").shift(1)
    high_prev = pd.to_numeric(df_bars["high"], errors="coerce").shift(1)
    low_prev = pd.to_numeric(df_bars["low"], errors="coerce").shift(1)
    vol_prev = pd.to_numeric(df_bars["volume"], errors="coerce").shift(1)

    out: dict[str, np.ndarray] = {}

    for w in windows:
        w = int(w)
        out[f"mom_{w}m_pct"] = (close_prev.pct_change(w) * 100.0).to_numpy(np.float64)
        out[f"vol_std_{w}m"] = close_prev.rolling(w, min_periods=w).std(ddof=0).to_numpy(np.float64)

        rng = (high_prev.rolling(w, min_periods=w).max() - low_prev.rolling(w, min_periods=w).min()).to_numpy(np.float64)
        out[f"range_{w}m"] = rng
        out[f"range_norm_{w}m"] = rng / close_prev.replace(0, np.nan).to_numpy(np.float64)

        cmax = close_prev.rolling(w, min_periods=w).max().to_numpy(np.float64)
        cmin = close_prev.rolling(w, min_periods=w).min().to_numpy(np.float64)
        crng = cmax - cmin
        eps = 1e-9

        cp = close_prev.to_numpy(np.float64)
        out[f"close_dd_from_{w}m_max_pct"] = (cmax / np.maximum(1e-9, cp) - 1.0) * 100.0
        out[f"close_bounce_from_{w}m_min_pct"] = (cp / np.maximum(1e-9, cmin) - 1.0) * 100.0
        out[f"close_pos_in_{w}m_range"] = (cp - cmin) / (crng + eps)

        v = vol_prev.to_numpy(np.float64, copy=False)
        sum_v = _rolling_sum_nan_prefix(np.nan_to_num(v, nan=0.0), w)
        sum_pv = _rolling_sum_nan_prefix(np.nan_to_num(v, nan=0.0) * np.nan_to_num(cp, nan=0.0), w)
        vwap = sum_pv / np.maximum(1e-9, sum_v)
        out[f"vwap_dev_{w}m"] = np.where(sum_v > 0.0, ((cp - vwap) / np.maximum(1e-9, vwap)) * 100.0, 0.0)

    return out


def _past_5m_features_with_ctx(
    df: pd.DataFrame,
    *,
    L: int,
    base_feat_cols: list[str],
    ctx_series: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, list[str]]:
    """Build 5-minute precontext features for:

    - px_close_norm_pct (close normalized)
    - vol_log1p (volume)
    - base features (ret/mom/vol/macd/vwapdev)
    - ctx series (mom_30/60/120, range_*, vol_std_*, close_dd/bounce/pos, vwap_dev_*)

    All computed over t-5..t-1.
    """
    close = df["close"].to_numpy(np.float64, copy=False)
    vol = df["volume"].to_numpy(np.float64, copy=False)

    close_prev = _rolling_prev_matrix(close, L)
    close_last = close_prev[:, -1]
    close_last = np.where((np.isfinite(close_last)) & (close_last > 0.0), close_last, np.nan)

    close_norm = (close_prev / close_last[:, None] - 1.0) * 100.0

    out: dict[str, np.ndarray] = {}

    out.update(_agg_5(close_norm, "px_close_norm_pct"))
    out["px_close_norm_pct__ret5m"] = close_norm[:, -1] - close_norm[:, 0]
    out["px_close_norm_pct__absret5m"] = np.abs(out["px_close_norm_pct__ret5m"])
    out["px_close_norm_pct__m5"] = close_norm[:, 0]
    out["px_close_norm_pct__m4"] = close_norm[:, 1]
    out["px_close_norm_pct__m3"] = close_norm[:, 2]
    out["px_close_norm_pct__m2"] = close_norm[:, 3]
    out["px_close_norm_pct__m1"] = close_norm[:, 4]

    vol_prev = _rolling_prev_matrix(vol, L)
    vol_log = np.log1p(np.maximum(0.0, vol_prev))
    out.update(_agg_5(vol_log, "vol_log1p"))

    for c in base_feat_cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64, copy=False)
        x_prev = _rolling_prev_matrix(x, L)
        out.update(_agg_5(x_prev, c))

    # ctx series
    for name, arr in ctx_series.items():
        x_prev = _rolling_prev_matrix(arr, L)
        out.update(_agg_5(x_prev, name))

    miss = np.isnan(close_prev)
    miss_n = miss.sum(axis=1).astype(np.float64)
    out["missing_close_n"] = miss_n
    out["missing_any"] = (miss_n > 0).astype(np.float64)

    feat_df = pd.DataFrame(out)
    for c in feat_df.columns:
        feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce").astype(np.float32)

    return feat_df, list(feat_df.columns)


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


def _robust_scale_1d(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad > 1e-12:
        return mad
    sd = float(np.std(x))
    return sd if sd > 1e-12 else 1.0


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = np.clip(z, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train weighted SELL entry regressor (oracle15m) with ctx120")
    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="ETH 1m dataset parquet",
    )
    ap.add_argument("--fee-total", type=float, default=0.001)
    ap.add_argument("--horizon-min", type=int, default=15)
    ap.add_argument("--pre-min", type=int, default=5)

    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--max-train-rows", type=int, default=0)

    ap.add_argument("--ctx-windows", default="30,60,120")

    ap.add_argument(
        "--shift-csv",
        default="data/analysis_sell_top10_goodbad_precontext_2026-01-04T19-36-48Z/feature_shifts_good_vs_bad_2026-01-04T19-36-48Z.csv",
        help="CSV with columns {feature, median_good, median_bad, robust_med_shift_good_minus_bad, abs}",
    )
    ap.add_argument("--shift-top-k", type=int, default=25)
    ap.add_argument("--weight-alpha", type=float, default=3.0)

    ap.add_argument("--lgb-log-period", type=int, default=50)

    ap.add_argument("--out-dir", default="data")

    args = ap.parse_args()

    ts = now_ts()
    out_root = Path(args.out_dir) / f"entry_regressor_oracle15m_sell_ctx120_weighted_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    models_dir = out_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading bars: {args.bars}", flush=True)
    df = pd.read_parquet(args.bars)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    pre_L = int(args.pre_min)
    horizon = int(args.horizon_min)
    fee_side = float(args.fee_total) / 2.0

    ctx_windows = [int(x.strip()) for x in str(args.ctx_windows).split(",") if x.strip()]

    print(f"[2/6] Computing ctx series (windows={ctx_windows}) ...", flush=True)
    ctx_series = _compute_ctx_series(df, ctx_windows)

    feat_cols_base = [c for c in BASE_FEATS_DEFAULT if c in df.columns]

    print(f"[3/6] Building {pre_L}-minute precontext feature matrix (base+ctx) ...", flush=True)
    X_df, X_names = _past_5m_features_with_ctx(df, L=pre_L, base_feat_cols=feat_cols_base, ctx_series=ctx_series)

    print(f"[3/6] Features built: n_features={len(X_names):,}", flush=True)

    print(f"[4/6] Building oracle {horizon}m SELL labels ...", flush=True)
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64, copy=False)
    fut_min = _future_extrema_excl_current(close, horizon, mode="min")
    y_sell = net_ret_pct_sell(close, fut_min, fee_side)

    valid = (np.arange(len(df)) >= pre_L) & np.isfinite(y_sell) & np.isfinite(close)

    # split indices chronologically
    n = int(df.shape[0])
    split_test = int(n * (1.0 - float(args.test_frac)))

    idx_all = np.arange(n)
    train_mask = valid & (idx_all < split_test)
    test_mask = valid & (idx_all >= split_test)

    train_idx = np.where(train_mask)[0]
    if int(args.max_train_rows) > 0 and train_idx.size > int(args.max_train_rows):
        train_idx = train_idx[-int(args.max_train_rows) :]

    test_idx = np.where(test_mask)[0]

    # validation from tail of training
    n_tr = int(train_idx.size)
    n_val = int(n_tr * float(args.val_frac))
    if n_val < 1000:
        n_val = min(5000, n_tr // 5) if n_tr >= 5000 else max(0, n_tr // 10)
    if n_val <= 0 or n_tr - n_val <= 0:
        raise SystemExit("Not enough training data for validation split")

    tr_idx = train_idx[: n_tr - n_val]
    va_idx = train_idx[n_tr - n_val :]

    # Build sample weights from shift CSV on ALL valid rows, then apply to train/val
    shift_path = Path(args.shift_csv)
    if not shift_path.exists():
        raise SystemExit(f"shift csv not found: {shift_path}")

    shifts = pd.read_csv(shift_path)
    shifts = shifts[shifts["feature"].isin(X_names)].copy()
    shifts = shifts.sort_values("abs", ascending=False).head(int(args.shift_top_k)).reset_index(drop=True)
    if shifts.empty:
        raise SystemExit("No overlap between shift features and current feature set")

    # compute per-feature scales on training slice (robust)
    scales = {}
    mid = {}
    direction = {}
    for _, r in shifts.iterrows():
        f = str(r["feature"])
        mg = float(r["median_good"])
        mb = float(r["median_bad"])
        mid[f] = 0.5 * (mg + mb)
        direction[f] = 1.0 if (mg - mb) >= 0 else -1.0
        scales[f] = _robust_scale_1d(pd.to_numeric(X_df.loc[train_idx, f], errors="coerce").to_numpy(np.float64))

    # Build weight for each row index (only valid rows are meaningful)
    w = np.ones(n, dtype=np.float64)
    # accumulate sigmoid scores
    ssum = np.zeros(n, dtype=np.float64)
    scount = 0
    for f in list(shifts["feature"]):
        x = pd.to_numeric(X_df[f], errors="coerce").to_numpy(np.float64)
        s = float(scales.get(f, 1.0))
        if not np.isfinite(s) or s <= 0:
            s = 1.0
        z = direction[f] * (x - mid[f]) / s
        ssum += _sigmoid(z)
        scount += 1

    if scount > 0:
        avg = ssum / float(scount)
    else:
        avg = np.zeros(n, dtype=np.float64)

    alpha = float(args.weight_alpha)
    w = 1.0 + alpha * avg

    # Only use weights on train/val; test weights not used for scoring
    X = X_df.to_numpy(np.float32, copy=False)

    X_tr, y_tr, w_tr = X[tr_idx], y_sell[tr_idx], w[tr_idx]
    X_va, y_va, w_va = X[va_idx], y_sell[va_idx], w[va_idx]

    print(
        f"[5/6] Training weighted SELL regressor: train={len(tr_idx):,} val={len(va_idx):,} test={len(test_idx):,} n_features={len(X_names):,}",
        flush=True,
    )

    model = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=256,
        max_depth=-1,
        min_child_samples=300,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    callbacks = []
    if int(args.lgb_log_period) > 0:
        callbacks.append(log_evaluation(period=int(args.lgb_log_period)))

    model.fit(
        X_tr,
        y_tr,
        sample_weight=w_tr,
        eval_set=[(X_va, y_va)],
        eval_sample_weight=[w_va],
        eval_metric="l2",
        callbacks=callbacks,
    )

    pred_all = np.full(n, np.nan, dtype=np.float64)
    pred_all[valid] = model.predict(X[valid])

    # Evaluate on test: win AUC and selection quality
    try:
        from sklearn.metrics import roc_auc_score

        y_test = y_sell[test_idx]
        p_test = pred_all[test_idx]
        win_test = (y_test >= 0.2).astype(int)
        auc_test = float(roc_auc_score(win_test, p_test)) if len(np.unique(win_test)) == 2 else float("nan")
    except Exception:
        auc_test = float("nan")

    run_meta = {
        "created_utc": ts,
        "bars": str(args.bars),
        "fee_total": float(args.fee_total),
        "fee_side": float(fee_side),
        "horizon_min": int(horizon),
        "pre_min": int(pre_L),
        "ctx_windows": ctx_windows,
        "test_frac": float(args.test_frac),
        "val_frac": float(args.val_frac),
        "max_train_rows": int(args.max_train_rows),
        "shift_csv": str(shift_path),
        "shift_top_k": int(args.shift_top_k),
        "weight_alpha": float(alpha),
        "n_total": int(n),
        "n_valid": int(valid.sum()),
        "n_train": int(tr_idx.size),
        "n_val": int(va_idx.size),
        "n_test": int(test_idx.size),
        "n_features": int(len(X_names)),
    }
    (out_root / f"run_meta_{ts}.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")

    metrics = {
        "SELL": {
            "auc_test_win_y>=0.2": float(auc_test),
            "pred_test_mean": float(np.nanmean(pred_all[test_idx])),
            "pred_test_p90": float(np.nanquantile(pred_all[test_idx], 0.9)),
            "y_test_mean": float(np.nanmean(y_sell[test_idx])),
            "y_test_p90": float(np.nanquantile(y_sell[test_idx], 0.9)),
        }
    }
    (out_root / f"metrics_{ts}.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    imp = pd.DataFrame(
        {
            "feature": X_names,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    imp.to_csv(out_root / f"feature_importance_{ts}.csv", index=False)

    artifact = {
        "created_utc": ts,
        "side": "SELL",
        "horizon_min": int(horizon),
        "pre_min": int(pre_L),
        "fee_total": float(args.fee_total),
        "label": "y_sell_oracle15m",
        "feature_cols": list(X_names),
        "model": model,
        "metrics": metrics,
        "weighting": {
            "shift_csv": str(shift_path),
            "shift_top_k": int(args.shift_top_k),
            "weight_alpha": float(alpha),
        },
    }

    model_path = models_dir / f"entry_regressor_sell_oracle15m_ctx120_weighted_{ts}.joblib"
    joblib.dump(artifact, model_path)

    print("[6/6] Done.", flush=True)
    print("Saved model:", model_path, flush=True)
    print("Outputs:", out_root, flush=True)


if __name__ == "__main__":
    main()
