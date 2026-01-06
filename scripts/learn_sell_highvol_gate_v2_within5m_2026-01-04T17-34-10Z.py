#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T17:34:10Z
"""High-vol regime split for SELL entries, using *within-5m derived features*.

This is the v2 of the high-vol gating attempt:
- still uses ONLY the last 5 minutes (t-5..t-1)
- but adds extra derived features from inside that 5-minute window (shape/reversal metrics)

Pipeline
1) Compute y_sell[t] = SELL oracle best-exit net return within next horizon minutes (default 15m).
2) Score the trained SELL entry model on all valid minutes.
3) Select top-10% by score.
4) Define high-vol regime within that top-10% selection using proxy thresholds.
5) Inside high-vol:
   - good = y_sell >= ret_thresh (default 0.2)
   - bad  = y_sell < bad_thresh (default 0.0)
   - neutrals excluded from gate training
6) Train a 500-tree LightGBM classifier on high-vol only, using:
   - base 5m descriptor features (same as entry model feature_cols)
   - PLUS derived-within-5m features from raw 5-point windows (ret/mom/macd/vwapdev/close)

Outputs (timestamped)
- data/highvol_sell_gate_v2_<ts>/
  - summary_<ts>.json
  - hv_proxy_thresholds_<ts>.json
  - gate_metrics_<ts>.json
  - gate_threshold_curve_<ts>.csv
  - gate_feature_importance_<ts>.csv
  - feature_shifts_good_vs_bad_<ts>.csv

Note
- This is selection diagnostics; it does not simulate non-overlap execution.
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


def build_pre5m_feature_matrix(
    df: pd.DataFrame,
    *,
    pre_min: int,
    base_feat_cols: list[str],
    out_feature_cols: list[str],
) -> np.ndarray:
    L = int(pre_min)
    if L != 5:
        raise SystemExit("This script currently supports --pre-min=5 only")

    n = len(df)
    col_to_j = {c: j for j, c in enumerate(out_feature_cols)}
    X = np.full((n, len(out_feature_cols)), np.nan, dtype=np.float32)

    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64, copy=False)
    vol = pd.to_numeric(df["volume"], errors="coerce").to_numpy(np.float64, copy=False)

    def _set(name: str, arr: np.ndarray) -> None:
        j = col_to_j.get(name)
        if j is None:
            return
        X[:, j] = np.asarray(arr, dtype=np.float32)

    # close-normalized window
    close_prev = _rolling_prev_matrix(close, L)
    close_last = close_prev[:, -1]
    close_last = np.where((np.isfinite(close_last)) & (close_last > 0.0), close_last, np.nan)
    close_norm = (close_prev / close_last[:, None] - 1.0) * 100.0

    px = _agg_5(close_norm, "px_close_norm_pct")
    px_ret5m = close_norm[:, -1] - close_norm[:, 0]
    px_absret5m = np.abs(px_ret5m)

    for k, v in px.items():
        _set(k, v)
    _set("px_close_norm_pct__ret5m", px_ret5m)
    _set("px_close_norm_pct__absret5m", px_absret5m)
    _set("px_close_norm_pct__m5", close_norm[:, 0])
    _set("px_close_norm_pct__m4", close_norm[:, 1])
    _set("px_close_norm_pct__m3", close_norm[:, 2])
    _set("px_close_norm_pct__m2", close_norm[:, 3])
    _set("px_close_norm_pct__m1", close_norm[:, 4])

    miss = np.isnan(close_prev)
    miss_n = miss.sum(axis=1).astype(np.float64)
    _set("missing_close_n", miss_n)
    _set("missing_any", (miss_n > 0).astype(np.float64))

    # volume log1p window
    vol_prev = _rolling_prev_matrix(vol, L)
    vol_log = np.log1p(np.maximum(0.0, vol_prev))
    vv = _agg_5(vol_log, "vol_log1p")
    for k, v in vv.items():
        _set(k, v)

    # base feature windows
    for c in base_feat_cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64, copy=False)
        x_prev = _rolling_prev_matrix(x, L)
        a = _agg_5(x_prev, c)
        for k, v in a.items():
            _set(k, v)

    return X


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


def _win_matrix_for_idxs(x: np.ndarray, idx: np.ndarray, L: int) -> np.ndarray:
    """Return window matrix shape (m, L) for idx (all idx must be >= L)."""
    idx = np.asarray(idx, dtype=np.int64)
    mats = []
    for k in range(L, 0, -1):
        mats.append(x[idx - k])
    return np.stack(mats, axis=1)


def _sign_changes(w: np.ndarray) -> np.ndarray:
    """Count sign changes across adjacent elements (ignore zeros). w shape (m,L)."""
    s = np.sign(w)
    s[s == 0] = np.nan
    # sign change where both finite and differ
    changes = (np.isfinite(s[:, 1:]) & np.isfinite(s[:, :-1]) & (s[:, 1:] != s[:, :-1])).sum(axis=1)
    return changes.astype(np.float64)


def _add_within5m_features(
    *,
    base_df: pd.DataFrame,
    raw: pd.DataFrame,
    idx: np.ndarray,
    pre_min: int,
) -> pd.DataFrame:
    """Add additional derived-within-5m features (computed only on the selected indices)."""
    L = int(pre_min)
    out = base_df.copy()

    def safe_argmin_rows(w: np.ndarray) -> np.ndarray:
        all_nan = ~np.isfinite(w).any(axis=1)
        w_fill = np.where(np.isfinite(w), w, np.inf)
        a = np.argmin(w_fill, axis=1).astype(np.float32)
        a[all_nan] = np.nan
        return a

    def safe_argmax_rows(w: np.ndarray) -> np.ndarray:
        all_nan = ~np.isfinite(w).any(axis=1)
        w_fill = np.where(np.isfinite(w), w, -np.inf)
        a = np.argmax(w_fill, axis=1).astype(np.float32)
        a[all_nan] = np.nan
        return a

    close = pd.to_numeric(raw["close"], errors="coerce").to_numpy(np.float64)
    close_w = _win_matrix_for_idxs(close, idx, L)
    close_last = close_w[:, -1]
    close_last = np.where((np.isfinite(close_last)) & (close_last > 0), close_last, np.nan)
    close_norm = (close_w / close_last[:, None] - 1.0) * 100.0  # last element is 0 (when close_last finite)

    # shape features from normalized close path
    cmin = np.nanmin(close_norm, axis=1)
    cmax = np.nanmax(close_norm, axis=1)
    crng = cmax - cmin
    eps = 1e-9
    out["px5__min_pos"] = safe_argmin_rows(close_norm)
    out["px5__max_pos"] = safe_argmax_rows(close_norm)
    out["px5__range"] = crng.astype(np.float32)
    out["px5__bounce_from_min"] = (0.0 - cmin).astype(np.float32)
    out["px5__drawdown_from_max"] = (cmax - 0.0).astype(np.float32)
    out["px5__pos_in_range"] = ((0.0 - cmin) / (crng + eps)).astype(np.float32)
    # curvature proxy: last step - first step
    out["px5__curvature"] = ((close_norm[:, -1] - close_norm[:, -2]) - (close_norm[:, 1] - close_norm[:, 0])).astype(np.float32)

    def add_series(name: str) -> None:
        if name not in raw.columns:
            return
        x = pd.to_numeric(raw[name], errors="coerce").to_numpy(np.float64)
        w = _win_matrix_for_idxs(x, idx, L)
        out[f"{name}__m5"] = w[:, 0].astype(np.float32)
        out[f"{name}__m4"] = w[:, 1].astype(np.float32)
        out[f"{name}__m3"] = w[:, 2].astype(np.float32)
        out[f"{name}__m2"] = w[:, 3].astype(np.float32)
        out[f"{name}__m1"] = w[:, 4].astype(np.float32)
        out[f"{name}__last2_delta"] = (w[:, 4] - w[:, 3]).astype(np.float32)
        out[f"{name}__last3_delta"] = (w[:, 4] - w[:, 2]).astype(np.float32)
        out[f"{name}__sign_changes"] = _sign_changes(w).astype(np.float32)
        out[f"{name}__pos_count"] = (w > 0).sum(axis=1).astype(np.float32)
        out[f"{name}__neg_count"] = (w < 0).sum(axis=1).astype(np.float32)
        out[f"{name}__min_pos"] = safe_argmin_rows(w)
        out[f"{name}__max_pos"] = safe_argmax_rows(w)

    for s in ["ret_1m_pct", "mom_3m_pct", "mom_5m_pct", "vwap_dev_5m", "macd"]:
        add_series(s)

    # volume window derived features
    vol = pd.to_numeric(raw["volume"], errors="coerce").to_numpy(np.float64)
    vol_w = _win_matrix_for_idxs(vol, idx, L)
    vol_log = np.log1p(np.maximum(0.0, vol_w))
    out["vol5__log_median"] = np.nanmedian(vol_log, axis=1).astype(np.float32)
    out["vol5__log_max"] = np.nanmax(vol_log, axis=1).astype(np.float32)
    out["vol5__log_last2_delta"] = (vol_log[:, -1] - vol_log[:, -2]).astype(np.float32)

    return out


def _robust_scale(x: np.ndarray) -> float:
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


def feature_shifts(df: pd.DataFrame, *, good_mask: np.ndarray, bad_mask: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    g = good_mask.astype(bool)
    b = bad_mask.astype(bool)
    rows = []
    for c in feature_cols:
        xg = pd.to_numeric(df.loc[g, c], errors="coerce").to_numpy(np.float64)
        xb = pd.to_numeric(df.loc[b, c], errors="coerce").to_numpy(np.float64)
        if np.isfinite(xg).sum() < 200 or np.isfinite(xb).sum() < 200:
            continue
        med_g = float(np.nanmedian(xg))
        med_b = float(np.nanmedian(xb))
        scale = _robust_scale(pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64))
        z = (med_g - med_b) / scale if np.isfinite(scale) and scale > 0 else float("nan")
        rows.append({"feature": c, "median_good": med_g, "median_bad": med_b, "robust_med_shift_good_minus_bad": float(z), "abs": float(abs(z))})
    out = pd.DataFrame(rows).sort_values("abs", ascending=False).reset_index(drop=True)
    return out


def guardrail_sweep_1d(
    hv: pd.DataFrame,
    *,
    feature_cols: list[str],
    ret_thresh: float,
    bad_thresh: float,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """Brute 1D threshold sweep for simple guardrails on the *high-vol* set.

    We score by (good_rate - bad_rate) * sqrt(coverage) to avoid tiny slices.
    """
    if quantiles is None:
        quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    y = pd.to_numeric(hv["y_sell"], errors="coerce").to_numpy(np.float64)
    good = y >= ret_thresh
    bad = y < bad_thresh

    rows = []
    for c in feature_cols:
        x = pd.to_numeric(hv[c], errors="coerce").to_numpy(np.float64)
        if np.isfinite(x).sum() < 500:
            continue
        qs = np.nanquantile(x[np.isfinite(x)], quantiles)
        qs = np.unique(qs)
        for thr in qs:
            for op in ("<=", ">="):
                m = np.isfinite(x) & ((x <= thr) if op == "<=" else (x >= thr))
                cov = float(np.mean(m))
                if cov < 0.02 or cov > 0.98:
                    continue
                ys = y[m]
                if ys.size < 500:
                    continue
                gr = float(np.mean(ys >= ret_thresh))
                br = float(np.mean(ys < bad_thresh))
                score = (gr - br) * (cov**0.5)
                rows.append(
                    {
                        "feature": c,
                        "op": op,
                        "threshold": float(thr),
                        "coverage": cov,
                        "good_rate": gr,
                        "bad_rate": br,
                        "neutral_rate": float(np.mean((ys >= bad_thresh) & (ys < ret_thresh))),
                        "median_y": float(np.median(ys)),
                        "p10_y": float(np.quantile(ys, 0.10)),
                        "p90_y": float(np.quantile(ys, 0.90)),
                        "score": float(score),
                        "selected_n": int(ys.size),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["score", "coverage"], ascending=[False, False]).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="High-vol SELL gate v2 with within-5m derived features")
    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="ETH 1m OHLCV+features parquet",
    )
    ap.add_argument(
        "--entry-model",
        default="data/entry_regressor_inliers_sell_2026-01-04T16-49-43Z/models/entry_regressor_sell_inliers_2026-01-04T16-49-43Z.joblib",
        help="SELL entry regressor artifact",
    )

    ap.add_argument("--fee-total", type=float, default=0.001)
    ap.add_argument("--horizon-min", type=int, default=15)
    ap.add_argument("--pre-min", type=int, default=5)

    ap.add_argument("--top-frac", type=float, default=0.10)

    ap.add_argument("--hv-q", type=float, default=0.90)
    ap.add_argument("--hv-min-count", type=int, default=2)

    ap.add_argument("--ret-thresh", type=float, default=0.2)
    ap.add_argument("--bad-thresh", type=float, default=0.0)

    ap.add_argument("--gate-n-estimators", type=int, default=500)
    ap.add_argument("--gate-test-frac", type=float, default=0.20)

    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--out-dir", default="data")

    ap.add_argument(
        "--save-datasets",
        action="store_true",
        help="If set, save high-vol and gate datasets (small ~10k rows) for offline inspection",
    )

    args = ap.parse_args()

    bars_path = Path(args.bars)
    if not bars_path.exists():
        raise SystemExit(f"bars not found: {bars_path}")

    art = joblib.load(args.entry_model)
    entry_model = art["model"]
    feat_cols = list(art["feature_cols"])

    pre_min = int(args.pre_min)
    horizon = int(args.horizon_min)
    fee_side = float(args.fee_total) / 2.0

    need_cols = ["timestamp", "close", "volume"] + list(BASE_FEATS_DEFAULT)
    print(f"Loading bars: {bars_path}", flush=True)
    df = pd.read_parquet(bars_path, columns=need_cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if not df["timestamp"].is_monotonic_increasing:
        df = df.sort_values("timestamp").reset_index(drop=True)

    n = int(len(df))
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64, copy=False)

    print(f"Computing y_sell within {horizon}m...", flush=True)
    fut_min = _future_extrema_excl_current(close, horizon, mode="min")
    y_sell = net_ret_pct_sell(close, fut_min, fee_side)

    valid = (np.arange(n) >= pre_min) & np.isfinite(y_sell) & np.isfinite(close)
    idxs = np.where(valid)[0]

    print(f"Building base 5m descriptor features for entry model (cols={len(feat_cols)})...", flush=True)
    X_all = build_pre5m_feature_matrix(df, pre_min=pre_min, base_feat_cols=list(BASE_FEATS_DEFAULT), out_feature_cols=feat_cols)

    print("Scoring entry model...", flush=True)
    pred = np.full(n, np.nan, dtype=np.float32)
    chunk = int(args.chunk)
    for s in range(0, idxs.size, chunk):
        e = min(idxs.size, s + chunk)
        ii = idxs[s:e]
        pred[ii] = entry_model.predict(X_all[ii]).astype(np.float32)
        if (s // chunk) % 10 == 0:
            print(f"  scored {e:,}/{idxs.size:,}", flush=True)

    pv = pred[valid].astype(np.float64)
    pv = pv[np.isfinite(pv)]
    if pv.size == 0:
        raise SystemExit("No finite predictions")

    top_frac = float(args.top_frac)
    thr_top = float(np.quantile(pv, 1.0 - top_frac))
    sel_mask = valid & np.isfinite(pred) & (pred >= thr_top)
    sel_idx = np.where(sel_mask)[0]

    # Base selected dataframe with the base features
    sel_base = pd.DataFrame(
        {
            "idx": sel_idx.astype(np.int64),
            "timestamp": df.loc[sel_idx, "timestamp"].to_numpy(),
            "pred": pred[sel_idx].astype(np.float64),
            "y_sell": y_sell[sel_idx].astype(np.float64),
        }
    )
    sel_feat = pd.DataFrame(X_all[sel_idx], columns=feat_cols)
    sel = pd.concat([sel_base, sel_feat], axis=1)

    # Define high-vol regime within selection
    proxies = [
        "vol_std_5m__last",
        "range_norm_5m__max5",
        "vol_log1p__last",
        "range_5m__max5",
    ]
    proxies = [p for p in proxies if p in sel.columns]
    hv_q = float(args.hv_q)
    hv_thr = {p: float(np.nanquantile(pd.to_numeric(sel[p], errors="coerce"), hv_q)) for p in proxies}

    hv_hits = np.zeros(len(sel), dtype=np.int32)
    for p in proxies:
        hv_hits += (pd.to_numeric(sel[p], errors="coerce").to_numpy(np.float64) >= hv_thr[p]).astype(np.int32)

    sel["hv_hits"] = hv_hits
    sel["is_high_vol"] = (hv_hits >= int(args.hv_min_count)).astype(np.int32)

    hv = sel[sel["is_high_vol"] == 1].copy().reset_index(drop=True)

    # Add derived-within-5m features
    hv_idx = hv["idx"].to_numpy(np.int64)
    hv = _add_within5m_features(base_df=hv, raw=df, idx=hv_idx, pre_min=pre_min)

    # labels
    ret_thresh = float(args.ret_thresh)
    bad_thresh = float(args.bad_thresh)
    hv["label_good"] = (hv["y_sell"] >= ret_thresh).astype(np.int32)
    hv["label_bad"] = (hv["y_sell"] < bad_thresh).astype(np.int32)
    hv["label_neutral"] = ((hv["y_sell"] >= bad_thresh) & (hv["y_sell"] < ret_thresh)).astype(np.int32)

    gate_df = hv[hv["label_neutral"] == 0].copy().reset_index(drop=True)
    if gate_df.empty:
        raise SystemExit("Empty gate dataset after removing neutrals")

    gate_df["timestamp"] = pd.to_datetime(gate_df["timestamp"], utc=True, errors="coerce")
    gate_df = gate_df.sort_values("timestamp").reset_index(drop=True)

    # Use all numeric columns except obvious non-features
    drop = {"idx", "timestamp", "pred", "y_sell", "label_good", "label_bad", "label_neutral"}
    feat_all = [c for c in gate_df.columns if c not in drop and gate_df[c].dtype.kind in "fc"]

    X = gate_df[feat_all].to_numpy(np.float32, copy=False)
    y = gate_df["label_good"].to_numpy(np.int32)

    n_gate = int(len(gate_df))
    split = int(n_gate * (1.0 - float(args.gate_test_frac)))
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    clf = LGBMClassifier(
        n_estimators=int(args.gate_n_estimators),
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
        class_weight="balanced",
    )
    clf.fit(X_tr, y_tr)

    proba_te = clf.predict_proba(X_te)[:, 1].astype(np.float64)

    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y_te, proba_te)) if len(np.unique(y_te)) == 2 else float("nan")
    except Exception:
        auc = float("nan")

    # retention curve
    y_sell_te = gate_df["y_sell"].to_numpy(np.float64)[split:]
    qs = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80]
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
                "win_rate_y>=ret": float(np.mean(sel_y >= ret_thresh)) if sel_y.size else float("nan"),
                "sel_median": float(np.median(sel_y)) if sel_y.size else float("nan"),
                "sel_p90": float(np.quantile(sel_y, 0.9)) if sel_y.size else float("nan"),
            }
        )

    # feature shifts good vs bad (within gate dataset)
    good_mask = gate_df["label_good"].to_numpy(np.int32) == 1
    bad_mask = gate_df["label_bad"].to_numpy(np.int32) == 1
    shifts = feature_shifts(gate_df, good_mask=good_mask, bad_mask=bad_mask, feature_cols=feat_all)

    ts = now_ts()
    out_root = Path(args.out_dir) / f"highvol_sell_gate_v2_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "created_utc": ts,
        "bars": str(bars_path),
        "entry_model": str(Path(args.entry_model)),
        "fee_total": float(args.fee_total),
        "fee_side": float(fee_side),
        "horizon_min": int(horizon),
        "pre_min": int(pre_min),
        "top_frac": float(top_frac),
        "top_pred_threshold": float(thr_top),
        "selected_n": int(sel.shape[0]),
        "hv_q": float(hv_q),
        "hv_min_count": int(args.hv_min_count),
        "high_vol_n": int(hv.shape[0]),
        "high_vol_good_rate_y>=ret": float(np.mean(hv["y_sell"] >= ret_thresh)) if hv.shape[0] else float("nan"),
        "high_vol_bad_rate_y<bad": float(np.mean(hv["y_sell"] < bad_thresh)) if hv.shape[0] else float("nan"),
        "gate_dataset_n": int(n_gate),
        "gate_test_auc": float(auc),
        "ret_thresh": float(ret_thresh),
        "bad_thresh": float(bad_thresh),
        "n_gate_features": int(len(feat_all)),
    }

    (out_root / f"summary_{ts}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_root / f"hv_proxy_thresholds_{ts}.json").write_text(json.dumps(hv_thr, indent=2) + "\n", encoding="utf-8")

    gate_metrics = {
        "auc_test": float(auc),
        "pos_rate_train": float(np.mean(y_tr)) if y_tr.size else float("nan"),
        "pos_rate_test": float(np.mean(y_te)) if y_te.size else float("nan"),
    }
    (out_root / f"gate_metrics_{ts}.json").write_text(json.dumps(gate_metrics, indent=2) + "\n", encoding="utf-8")

    pd.DataFrame(curve).to_csv(out_root / f"gate_threshold_curve_{ts}.csv", index=False)

    imp = pd.DataFrame(
        {
            "feature": feat_all,
            "importance_gain": clf.booster_.feature_importance(importance_type="gain"),
            "importance_split": clf.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    imp.to_csv(out_root / f"gate_feature_importance_{ts}.csv", index=False)

    shifts.to_csv(out_root / f"feature_shifts_good_vs_bad_{ts}.csv", index=False)

    # guardrail sweep (1D)
    # candidate columns: shifts top + some explicitly reversal-ish within-5m features
    cand = []
    if not shifts.empty:
        cand.extend(shifts.head(60)["feature"].tolist())
    cand.extend(
        [
            "ret_1m_pct__m1",
            "ret_1m_pct__m2",
            "ret_1m_pct__last2_delta",
            "ret_1m_pct__sign_changes",
            "mom_3m_pct__sign_changes",
            "mom_5m_pct__sign_changes",
            "vwap_dev_5m__last2_delta",
            "macd__last2_delta",
            "px5__pos_in_range",
            "px5__bounce_from_min",
            "px5__curvature",
        ]
    )
    cand = [c for c in dict.fromkeys(cand) if c in hv.columns and hv[c].dtype.kind in "fc"]

    sweep = guardrail_sweep_1d(hv, feature_cols=cand, ret_thresh=ret_thresh, bad_thresh=bad_thresh)
    if not sweep.empty:
        sweep.to_csv(out_root / f"guardrail_sweep_1d_{ts}.csv", index=False)

    # Optionally save datasets
    if args.save_datasets:
        hv.to_parquet(out_root / f"highvol_dataset_{ts}.parquet", index=False)
        gate_df.to_parquet(out_root / f"gate_dataset_{ts}.parquet", index=False)

    print(json.dumps(summary, indent=2))
    print("High-vol proxy thresholds:")
    print(json.dumps(hv_thr, indent=2))
    if not sweep.empty:
        print("Top guardrails (1D, high-vol slice):")
        print(sweep.head(8)[["feature", "op", "threshold", "coverage", "good_rate", "bad_rate", "median_y", "p10_y", "p90_y", "score"]].to_string(index=False))
    print("Wrote", out_root)


if __name__ == "__main__":
    main()