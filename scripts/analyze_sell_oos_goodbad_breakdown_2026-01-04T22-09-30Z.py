#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T22:09:30Z
"""Analyze why SELL entry ranking degrades out-of-sample (OOS).

We focus on OOS = last `--test-frac` portion of time (chronological split).

What this script produces
1) OOS selection set
- Score a given entry regressor (joblib artifact {model, feature_cols}) on the full dataset.
- Restrict to OOS indices.
- Select top-q% within OOS.
- Label each selected minute:
  - good: y_sell >= ret_thresh (default 0.2)
  - bad:  y_sell < bad_thresh (default 0.0)
  - neutral otherwise

2) OOS good-vs-bad separations
- Compute robust median shifts (good - bad) for model input features (full feature_cols).

3) Drift train -> OOS
- Compute robust median drift between a train sample and OOS sample for each feature.

4) Separator stability
- Load the precomputed separator CSV (feature_shifts_good_vs_bad_*.csv) that was used for weighting.
- Compare its direction/magnitude to the OOS good-vs-bad shifts for the same features.
- Highlight sign flips and weakened separators.

Outputs (timestamped)
- data/analysis_sell_oos_breakdown_<ts>/
  - summary_<ts>.json
  - oos_selected_sample_<ts>.csv
  - oos_selected_features_<ts>.parquet
  - feature_shifts_oos_good_vs_bad_<ts>.csv
  - feature_drift_train_vs_oos_<ts>.csv
  - separator_stability_vs_ref_<ts>.csv

Notes
- This is a *ranking diagnostic* (no position overlap constraint).
- Feature computation matches the backtest/training intent:
  entry at t uses ONLY t-5..t-1; ctx series are causal (use t-1 and earlier).
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
    """For each i, return extreme over x[i+1:i+W+1]."""
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


def _rolling_sum_nan_prefix(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    cs = np.cumsum(x)
    out[w - 1 :] = cs[w - 1 :] - np.concatenate(([0.0], cs[: n - w]))
    return out


def compute_ctx_series(bars: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute causal context series at each minute using only t-1 and earlier."""
    close_prev = pd.to_numeric(bars["close"], errors="coerce").shift(1)
    high_prev = pd.to_numeric(bars["high"], errors="coerce").shift(1)
    low_prev = pd.to_numeric(bars["low"], errors="coerce").shift(1)
    vol_prev = pd.to_numeric(bars["volume"], errors="coerce").shift(1)

    out = pd.DataFrame(index=bars.index)

    for w in windows:
        w = int(w)
        out[f"mom_{w}m_pct"] = close_prev.pct_change(w) * 100.0
        out[f"vol_std_{w}m"] = close_prev.rolling(w, min_periods=w).std(ddof=0)

        rng = high_prev.rolling(w, min_periods=w).max() - low_prev.rolling(w, min_periods=w).min()
        out[f"range_{w}m"] = rng
        out[f"range_norm_{w}m"] = rng / close_prev.replace(0, np.nan)

        cmax = close_prev.rolling(w, min_periods=w).max()
        cmin = close_prev.rolling(w, min_periods=w).min()
        crng = cmax - cmin
        eps = 1e-9

        out[f"close_dd_from_{w}m_max_pct"] = (cmax / close_prev.replace(0, np.nan) - 1.0) * 100.0
        out[f"close_bounce_from_{w}m_min_pct"] = (close_prev / cmin.replace(0, np.nan) - 1.0) * 100.0
        out[f"close_pos_in_{w}m_range"] = (close_prev - cmin) / (crng + eps)

        # VWAP dev
        v = vol_prev.to_numpy(np.float64, copy=False)
        c = close_prev.to_numpy(np.float64, copy=False)
        sum_v = _rolling_sum_nan_prefix(np.nan_to_num(v, nan=0.0), w)
        sum_pv = _rolling_sum_nan_prefix(np.nan_to_num(v, nan=0.0) * np.nan_to_num(c, nan=0.0), w)
        vwap = sum_pv / np.maximum(1e-9, sum_v)
        vwap_dev = np.where(sum_v > 0.0, ((c - vwap) / np.maximum(1e-9, vwap)) * 100.0, 0.0)
        out[f"vwap_dev_{w}m"] = vwap_dev

    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float32)

    return out


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


def build_model_features_for_indices(
    *,
    bars: pd.DataFrame,
    idx: np.ndarray,
    feature_cols: list[str],
    ctx_windows: list[int],
) -> pd.DataFrame:
    """Build the full model feature matrix (columns=feature_cols) for the given indices."""
    ii = np.asarray(idx, dtype=np.int64)

    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(np.float64, copy=False)
    volume = pd.to_numeric(bars["volume"], errors="coerce").to_numpy(np.float64, copy=False)

    base_arrs = {c: pd.to_numeric(bars[c], errors="coerce").to_numpy(np.float64, copy=False) for c in BASE_FEATS_DEFAULT}
    ctx_df = compute_ctx_series(bars, ctx_windows)
    ctx_arrs = {c: pd.to_numeric(ctx_df[c], errors="coerce").to_numpy(np.float64, copy=False) for c in ctx_df.columns}

    col_to_j = {c: j for j, c in enumerate(feature_cols)}

    # Which series prefixes are needed (excluding px_close_norm_pct, vol_log1p)
    prefixes = set()
    for c in feature_cols:
        if "__" in c:
            prefixes.add(c.split("__", 1)[0])
    prefixes.discard("px_close_norm_pct")
    prefixes.discard("vol_log1p")

    series_map: dict[str, np.ndarray] = {}
    for p in prefixes:
        if p in base_arrs:
            series_map[p] = base_arrs[p]
        elif p in ctx_arrs:
            series_map[p] = ctx_arrs[p]

    m = int(ii.size)
    X = np.full((m, len(feature_cols)), np.nan, dtype=np.float32)

    def _set(name: str, arr: np.ndarray) -> None:
        j = col_to_j.get(name)
        if j is None:
            return
        X[:, j] = np.asarray(arr, dtype=np.float32)

    # indices for t-5..t-1
    i0 = ii - 5
    i1 = ii - 4
    i2 = ii - 3
    i3 = ii - 2
    i4 = ii - 1

    close_w = np.stack([close[i0], close[i1], close[i2], close[i3], close[i4]], axis=1)
    close_last = close_w[:, -1]
    close_last = np.where((np.isfinite(close_last)) & (close_last > 0.0), close_last, np.nan)
    close_norm = (close_w / close_last[:, None] - 1.0) * 100.0

    px = _agg5_rows(close_norm)
    for suf, arr in px.items():
        _set(f"px_close_norm_pct{suf}", arr)

    px_ret5m = close_norm[:, -1] - close_norm[:, 0]
    _set("px_close_norm_pct__ret5m", px_ret5m)
    _set("px_close_norm_pct__absret5m", np.abs(px_ret5m))

    # These exist in some variants
    _set("px_close_norm_pct__m5", close_norm[:, 0])
    _set("px_close_norm_pct__m4", close_norm[:, 1])
    _set("px_close_norm_pct__m3", close_norm[:, 2])
    _set("px_close_norm_pct__m2", close_norm[:, 3])
    _set("px_close_norm_pct__m1", close_norm[:, 4])

    # volume log1p
    vol_w = np.stack([volume[i0], volume[i1], volume[i2], volume[i3], volume[i4]], axis=1)
    vol_log = np.log1p(np.maximum(0.0, vol_w))
    vv = _agg5_rows(vol_log)
    for suf, arr in vv.items():
        _set(f"vol_log1p{suf}", arr)

    # missing flags
    if "missing_close_n" in col_to_j or "missing_any" in col_to_j:
        miss = ~np.isfinite(close_w)
        miss_n = miss.sum(axis=1).astype(np.float64)
        _set("missing_close_n", miss_n)
        _set("missing_any", (miss_n > 0).astype(np.float64))

    # aggregated windows for each required series
    for name, arr0 in series_map.items():
        w = np.stack([arr0[i0], arr0[i1], arr0[i2], arr0[i3], arr0[i4]], axis=1)
        a = _agg5_rows(w)
        for suf, v in a.items():
            _set(f"{name}{suf}", v)

    df = pd.DataFrame(X, columns=feature_cols)
    return df


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


def feature_shifts_good_bad(df: pd.DataFrame, *, good_mask: np.ndarray, bad_mask: np.ndarray, min_n: int) -> pd.DataFrame:
    g = good_mask.astype(bool)
    b = bad_mask.astype(bool)
    rows = []

    for c in df.columns:
        xg = pd.to_numeric(df.loc[g, c], errors="coerce").to_numpy(np.float64)
        xb = pd.to_numeric(df.loc[b, c], errors="coerce").to_numpy(np.float64)
        if np.isfinite(xg).sum() < min_n or np.isfinite(xb).sum() < min_n:
            continue
        med_g = float(np.nanmedian(xg))
        med_b = float(np.nanmedian(xb))
        scale = _robust_scale_1d(pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64))
        z = (med_g - med_b) / scale if np.isfinite(scale) and scale > 0 else float("nan")
        rows.append(
            {
                "feature": c,
                "median_good": med_g,
                "median_bad": med_b,
                "robust_med_shift_good_minus_bad": float(z),
                "abs": float(abs(z)),
            }
        )

    return pd.DataFrame(rows).sort_values("abs", ascending=False).reset_index(drop=True)


def feature_drift(a: pd.DataFrame, b: pd.DataFrame, *, min_n: int) -> pd.DataFrame:
    """Robust median drift from a -> b, scaled by MAD on a."""
    rows = []
    for c in a.columns:
        xa = pd.to_numeric(a[c], errors="coerce").to_numpy(np.float64)
        xb = pd.to_numeric(b[c], errors="coerce").to_numpy(np.float64)
        if np.isfinite(xa).sum() < min_n or np.isfinite(xb).sum() < min_n:
            continue
        med_a = float(np.nanmedian(xa))
        med_b = float(np.nanmedian(xb))
        scale = _robust_scale_1d(xa)
        z = (med_b - med_a) / scale if np.isfinite(scale) and scale > 0 else float("nan")
        rows.append(
            {
                "feature": c,
                "median_train": med_a,
                "median_oos": med_b,
                "robust_med_shift_oos_minus_train": float(z),
                "abs": float(abs(z)),
            }
        )
    return pd.DataFrame(rows).sort_values("abs", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze OOS good vs bad for SELL selection")
    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="ETH 1m OHLCV+features parquet",
    )
    ap.add_argument(
        "--entry-model",
        default="data/entry_regressor_oracle15m_sell_ctx120_weighted_2026-01-04T19-54-19Z/models/entry_regressor_sell_oracle15m_ctx120_weighted_2026-01-04T19-54-19Z.joblib",
        help="Entry model artifact (joblib dict: {model, feature_cols, ...})",
    )
    ap.add_argument("--fee-total", type=float, default=0.001)
    ap.add_argument("--horizon-min", type=int, default=15)
    ap.add_argument("--pre-min", type=int, default=5)
    ap.add_argument("--ctx-windows", default="30,60,120")

    ap.add_argument("--test-frac", type=float, default=0.20, help="OOS fraction at end of timeline")

    ap.add_argument("--top-frac-oos", type=float, default=0.10, help="Select top-q% within OOS")
    ap.add_argument("--ret-thresh", type=float, default=0.2)
    ap.add_argument("--bad-thresh", type=float, default=0.0)

    ap.add_argument(
        "--ref-shifts",
        default="data/analysis_sell_top10_goodbad_precontext_2026-01-04T19-36-48Z/feature_shifts_good_vs_bad_2026-01-04T19-36-48Z.csv",
        help="Reference shifts CSV used to derive weighting signal",
    )
    ap.add_argument("--ref-top-k", type=int, default=50)

    ap.add_argument("--train-sample", type=int, default=80_000)
    ap.add_argument("--min-n", type=int, default=800)

    ap.add_argument("--out-dir", default="data")

    args = ap.parse_args()

    bars_path = Path(args.bars)
    if not bars_path.exists():
        raise SystemExit(f"bars not found: {bars_path}")

    art = joblib.load(args.entry_model)
    if not isinstance(art, dict) or "model" not in art or "feature_cols" not in art:
        raise SystemExit("entry model artifact must be a dict with keys: model, feature_cols")

    model = art["model"]
    feature_cols = list(art["feature_cols"])

    need_cols = ["timestamp", "open", "high", "low", "close", "volume"] + list(BASE_FEATS_DEFAULT)

    print(f"Loading bars: {bars_path}", flush=True)
    bars = pd.read_parquet(bars_path, columns=need_cols)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    if not bars["timestamp"].is_monotonic_increasing:
        bars = bars.sort_values("timestamp").reset_index(drop=True)

    n = int(len(bars))
    pre_min = int(args.pre_min)
    horizon = int(args.horizon_min)
    fee_side = float(args.fee_total) / 2.0

    ctx_windows = [int(x.strip()) for x in str(args.ctx_windows).split(",") if x.strip()]

    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(np.float64, copy=False)
    fut_min = _future_extrema_excl_current(close, horizon, mode="min")
    y_sell = net_ret_pct_sell(close, fut_min, fee_side)

    idx_all = np.arange(n)
    valid = (idx_all >= pre_min) & np.isfinite(y_sell) & np.isfinite(close)

    split_test = int(n * (1.0 - float(args.test_frac)))
    train_mask = valid & (idx_all < split_test)
    oos_mask = valid & (idx_all >= split_test)

    oos_idx = np.where(oos_mask)[0]
    print(f"Valid minutes: {int(valid.sum()):,}  | train: {int(train_mask.sum()):,}  | oos: {int(oos_mask.sum()):,}", flush=True)

    # Score model on OOS only (build X for all OOS indices in chunks by using feature builder per selection below)
    # For efficiency: we first compute features for all OOS indices, predict, then select top-frac.
    print("Building model features for all OOS minutes (this is the heavy step)...", flush=True)
    X_oos = build_model_features_for_indices(bars=bars, idx=oos_idx, feature_cols=feature_cols, ctx_windows=ctx_windows)
    pred_oos = model.predict(X_oos.to_numpy(np.float32, copy=False)).astype(np.float64)

    top_frac = float(args.top_frac_oos)
    thr = float(np.nanquantile(pred_oos, 1.0 - top_frac))
    sel = np.isfinite(pred_oos) & (pred_oos >= thr)

    sel_idx = oos_idx[sel]
    sel_pred = pred_oos[sel]
    sel_y = y_sell[sel_idx].astype(np.float64)

    ret_thresh = float(args.ret_thresh)
    bad_thresh = float(args.bad_thresh)

    good = sel_y >= ret_thresh
    bad = sel_y < bad_thresh
    neutral = ~(good | bad)

    print(
        f"OOS top-{top_frac*100:.2f}%: selected {sel_idx.size:,} (thr={thr:.6g}) | good {int(good.sum()):,} bad {int(bad.sum()):,} neutral {int(neutral.sum()):,}",
        flush=True,
    )

    # Feature DF for selected rows
    X_sel = X_oos.loc[sel].reset_index(drop=True)

    # OOS good-bad shifts inside selection
    keep = good | bad
    shifts_oos = feature_shifts_good_bad(X_sel.loc[keep].reset_index(drop=True), good_mask=good[keep], bad_mask=bad[keep], min_n=int(args.min_n))

    # Drift: sample train vs sample OOS (unconditional, not top selection)
    rng = np.random.default_rng(42)
    tr_idx = np.where(train_mask)[0]
    if tr_idx.size == 0:
        raise SystemExit("No train indices")

    tr_take = min(int(args.train_sample), int(tr_idx.size))
    oos_take = min(int(args.train_sample), int(oos_idx.size))

    tr_samp = rng.choice(tr_idx, size=tr_take, replace=False)
    oos_samp = rng.choice(oos_idx, size=oos_take, replace=False)
    tr_samp.sort()
    oos_samp.sort()

    print(f"Building features for drift samples: train={tr_take:,} oos={oos_take:,}", flush=True)
    X_tr_s = build_model_features_for_indices(bars=bars, idx=tr_samp, feature_cols=feature_cols, ctx_windows=ctx_windows)
    X_oos_s = build_model_features_for_indices(bars=bars, idx=oos_samp, feature_cols=feature_cols, ctx_windows=ctx_windows)

    drift = feature_drift(X_tr_s, X_oos_s, min_n=int(args.min_n))

    # Separator stability vs reference file
    ref_path = Path(args.ref_shifts)
    ref = None
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        ref = ref[ref["feature"].isin(shifts_oos["feature"])].copy()
        ref = ref.sort_values("abs", ascending=False).head(int(args.ref_top_k)).reset_index(drop=True)

    stab = None
    if ref is not None and not ref.empty and not shifts_oos.empty:
        oos_map = shifts_oos.set_index("feature")
        rows = []
        for _, r in ref.iterrows():
            f = str(r["feature"])
            if f not in oos_map.index:
                continue
            ref_shift = float(r["robust_med_shift_good_minus_bad"])
            oos_shift = float(oos_map.loc[f, "robust_med_shift_good_minus_bad"])
            rows.append(
                {
                    "feature": f,
                    "ref_shift": ref_shift,
                    "oos_shift": oos_shift,
                    "ref_abs": float(abs(ref_shift)),
                    "oos_abs": float(abs(oos_shift)),
                    "sign_flip": bool(np.sign(ref_shift) != np.sign(oos_shift) and np.isfinite(ref_shift) and np.isfinite(oos_shift)),
                    "oos_over_ref_abs": float(abs(oos_shift) / abs(ref_shift)) if abs(ref_shift) > 1e-12 else float("nan"),
                }
            )
        stab = pd.DataFrame(rows).sort_values(["sign_flip", "oos_abs"], ascending=[False, False]).reset_index(drop=True)

    ts = now_ts()
    out_root = Path(args.out_dir) / f"analysis_sell_oos_breakdown_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Full selected metadata (index-aligned with X_sel parquet rows)
    sel_meta = pd.DataFrame(
        {
            "sel_pos": np.arange(sel_idx.size, dtype=np.int64),
            "row_idx": sel_idx.astype(np.int64),
            "timestamp": bars.loc[sel_idx, "timestamp"].to_numpy(),
            "pred": sel_pred.astype(np.float64),
            "y_sell": sel_y.astype(np.float64),
            "label": np.where(good, "good", np.where(bad, "bad", "neutral")),
        }
    )

    # sample rows for inspection
    take = min(20000, int(sel_idx.size))
    pick = rng.choice(np.arange(sel_idx.size), size=take, replace=False)
    pick.sort()

    sample = sel_meta.loc[pick].reset_index(drop=True)

    summary = {
        "created_utc": ts,
        "bars": str(bars_path),
        "entry_model": str(Path(args.entry_model)),
        "horizon_min": int(horizon),
        "pre_min": int(pre_min),
        "fee_side": float(fee_side),
        "split_test_index": int(split_test),
        "oos_start_ts": str(bars.loc[split_test, "timestamp"]) if split_test < n else None,
        "oos_end_ts": str(bars.loc[n - 1, "timestamp"]) if n else None,
        "n_total": int(n),
        "n_valid": int(valid.sum()),
        "n_train_valid": int(train_mask.sum()),
        "n_oos_valid": int(oos_mask.sum()),
        "oos_base_win_rate_y>=ret": float(np.mean(y_sell[oos_mask] >= ret_thresh)),
        "oos_top_frac": float(top_frac),
        "oos_top_threshold": float(thr),
        "oos_selected_n": int(sel_idx.size),
        "oos_selected_win_rate_y>=ret": float(np.mean(sel_y >= ret_thresh)) if sel_y.size else float("nan"),
        "oos_selected_bad_rate_y<0": float(np.mean(sel_y < 0.0)) if sel_y.size else float("nan"),
        "oos_selected_good_n": int(good.sum()),
        "oos_selected_bad_n": int(bad.sum()),
        "oos_selected_neutral_n": int(neutral.sum()),
        "shifts_oos_rows": int(len(shifts_oos)),
        "drift_rows": int(len(drift)),
        "ref_shifts_loaded": bool(ref is not None),
        "ref_shifts_top_k": int(args.ref_top_k),
        "separator_stability_rows": int(len(stab)) if stab is not None else 0,
    }

    (out_root / f"summary_{ts}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Sample + full metadata for joining with feature parquet (via sel_pos)
    sample.to_csv(out_root / f"oos_selected_sample_{ts}.csv", index=False)
    sel_meta.to_parquet(out_root / f"oos_selected_meta_{ts}.parquet", index=False)
    sel_meta.to_csv(out_root / f"oos_selected_meta_{ts}.csv", index=False)

    # Feature rows are index-aligned with sel_meta (same ordering)
    X_sel.to_parquet(out_root / f"oos_selected_features_{ts}.parquet", index=False)

    shifts_oos.to_csv(out_root / f"feature_shifts_oos_good_vs_bad_{ts}.csv", index=False)
    drift.to_csv(out_root / f"feature_drift_train_vs_oos_{ts}.csv", index=False)
    if stab is not None:
        stab.to_csv(out_root / f"separator_stability_vs_ref_{ts}.csv", index=False)

    print(json.dumps(summary, indent=2), flush=True)
    print("\nTop 20 OOS good-vs-bad separators:")
    if not shifts_oos.empty:
        print(shifts_oos.head(20).to_string(index=False))

    print("\nTop 20 train->OOS drift features:")
    if not drift.empty:
        print(drift.head(20).to_string(index=False))

    if stab is not None and not stab.empty:
        print("\nTop stability issues vs ref (sign flips first):")
        print(stab.head(30).to_string(index=False))

    print("Wrote", out_root, flush=True)


if __name__ == "__main__":
    main()
