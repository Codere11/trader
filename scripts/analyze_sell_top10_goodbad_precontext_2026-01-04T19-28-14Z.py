#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T19:28:14Z
"""Analyze top-10% SELL selection: biggest 5m-precontext separations between good vs bad.

We:
- Load full ETH 1m bars.
- Compute y_sell[t] = SELL oracle best-exit net return within next horizon minutes.
- Score the ctx120 SELL entry model over all valid minutes.
- Select the top 10% by prediction (over valid minutes).
- Inside that selection:
  - good = y_sell >= ret_thresh (default 0.2%)
  - bad  = y_sell < bad_thresh (default 0.0%)
  - neutrals excluded from good/bad comparison
- Compute 5-minute precontext features ONLY (t-5..t-1):
  px_close_norm_pct, vol_log1p, and base series (ret_1m_pct, mom_*, vol_std_5m, range_*, macd, vwap_dev_5m).
- Report robust median-shift (good - bad) scaled by MAD to identify biggest separations.

Outputs (timestamped)
- data/analysis_sell_top10_goodbad_precontext_<ts>/
  - summary_<ts>.json
  - feature_shifts_good_vs_bad_<ts>.csv
  - selection_sample_<ts>.csv

Note
- This is selection diagnostics; it does not enforce non-overlap trading.
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


def _rolling_sum_nan_prefix(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    cs = np.cumsum(x)
    out[w - 1 :] = cs[w - 1 :] - np.concatenate(([0.0], cs[: n - w]))
    return out


def compute_ctx_series(bars: pd.DataFrame, windows: list[int]) -> dict[str, np.ndarray]:
    """Compute causal 30/60/120m context series at each minute using only t-1 and earlier."""
    close_prev = pd.to_numeric(bars["close"], errors="coerce").shift(1)
    high_prev = pd.to_numeric(bars["high"], errors="coerce").shift(1)
    low_prev = pd.to_numeric(bars["low"], errors="coerce").shift(1)
    vol_prev = pd.to_numeric(bars["volume"], errors="coerce").shift(1)

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


def compute_pred_all(
    *,
    bars: pd.DataFrame,
    model,
    feature_cols: list[str],
    ctx_windows: list[int],
    pre_min: int,
    horizon: int,
    fee_side: float,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pred[n], y_sell[n], valid[n]) for full dataset."""
    n = int(len(bars))

    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(np.float64, copy=False)
    volume = pd.to_numeric(bars["volume"], errors="coerce").to_numpy(np.float64, copy=False)

    base_arrs = {c: pd.to_numeric(bars[c], errors="coerce").to_numpy(np.float64, copy=False) for c in BASE_FEATS_DEFAULT}
    ctx_arrs = compute_ctx_series(bars, ctx_windows)

    fut_min = _future_extrema_excl_current(close, horizon, mode="min")
    y_sell = net_ret_pct_sell(close, fut_min, fee_side)

    valid = (np.arange(n) >= pre_min) & np.isfinite(y_sell) & np.isfinite(close)

    # feature column mapping
    col_to_j = {c: j for j, c in enumerate(feature_cols)}

    # which series prefixes are needed
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

    idxs = np.where(valid)[0]

    pred = np.full(n, np.nan, dtype=np.float32)

    def _set(X: np.ndarray, name: str, arr: np.ndarray) -> None:
        j = col_to_j.get(name)
        if j is None:
            return
        X[:, j] = np.asarray(arr, dtype=np.float32)

    for s in range(0, idxs.size, chunk):
        e = min(idxs.size, s + chunk)
        ii = idxs[s:e]

        m = ii.size
        Xc = np.full((m, len(feature_cols)), np.nan, dtype=np.float32)

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
            _set(Xc, f"px_close_norm_pct{suf}", arr)
        px_ret5m = close_norm[:, -1] - close_norm[:, 0]
        _set(Xc, "px_close_norm_pct__ret5m", px_ret5m)
        _set(Xc, "px_close_norm_pct__absret5m", np.abs(px_ret5m))

        # volume log1p
        vol_w = np.stack([volume[i0], volume[i1], volume[i2], volume[i3], volume[i4]], axis=1)
        vol_log = np.log1p(np.maximum(0.0, vol_w))
        vv = _agg5_rows(vol_log)
        for suf, arr in vv.items():
            _set(Xc, f"vol_log1p{suf}", arr)

        # base missing flags (some models may include them)
        if "missing_close_n" in col_to_j or "missing_any" in col_to_j:
            miss = ~np.isfinite(close_w)
            miss_n = miss.sum(axis=1).astype(np.float64)
            _set(Xc, "missing_close_n", miss_n)
            _set(Xc, "missing_any", (miss_n > 0).astype(np.float64))

        for name, arr in series_map.items():
            w = np.stack([arr[i0], arr[i1], arr[i2], arr[i3], arr[i4]], axis=1)
            a = _agg5_rows(w)
            for suf, vv2 in a.items():
                _set(Xc, f"{name}{suf}", vv2)

        pred[ii] = model.predict(Xc).astype(np.float32)

        if (s // chunk) % 2 == 0:
            print(f"  scored {e:,}/{idxs.size:,}", flush=True)

    return pred, y_sell, valid


def compute_precontext_features_for_indices(
    *,
    bars: pd.DataFrame,
    idx: np.ndarray,
    pre_min: int,
) -> pd.DataFrame:
    """Compute 5m precontext feature set (79 cols) for given indices."""
    if int(pre_min) != 5:
        raise SystemExit("This script currently supports --pre-min=5 only")

    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(np.float64, copy=False)
    volume = pd.to_numeric(bars["volume"], errors="coerce").to_numpy(np.float64, copy=False)
    base_arrs = {c: pd.to_numeric(bars[c], errors="coerce").to_numpy(np.float64, copy=False) for c in BASE_FEATS_DEFAULT}

    ii = np.asarray(idx, dtype=np.int64)
    i0 = ii - 5
    i1 = ii - 4
    i2 = ii - 3
    i3 = ii - 2
    i4 = ii - 1

    out: dict[str, np.ndarray] = {}

    # px_close_norm_pct
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

    # volume log1p
    vol_w = np.stack([volume[i0], volume[i1], volume[i2], volume[i3], volume[i4]], axis=1)
    vol_log = np.log1p(np.maximum(0.0, vol_w))
    vv = _agg5_rows(vol_log)
    for suf, arr in vv.items():
        out[f"vol_log1p{suf}"] = arr

    # base series windows
    for name, arr0 in base_arrs.items():
        w = np.stack([arr0[i0], arr0[i1], arr0[i2], arr0[i3], arr0[i4]], axis=1)
        a = _agg5_rows(w)
        for suf, v in a.items():
            out[f"{name}{suf}"] = v

    df = pd.DataFrame(out)
    # downcast
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    return df


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


def feature_shifts_good_bad(pre: pd.DataFrame, *, good_mask: np.ndarray, bad_mask: np.ndarray) -> pd.DataFrame:
    g = good_mask.astype(bool)
    b = bad_mask.astype(bool)
    rows = []
    for c in pre.columns:
        if pre[c].dtype.kind not in "fc":
            continue
        xg = pd.to_numeric(pre.loc[g, c], errors="coerce").to_numpy(np.float64)
        xb = pd.to_numeric(pre.loc[b, c], errors="coerce").to_numpy(np.float64)
        if np.isfinite(xg).sum() < 500 or np.isfinite(xb).sum() < 500:
            continue
        med_g = float(np.nanmedian(xg))
        med_b = float(np.nanmedian(xb))
        scale = _robust_scale(pd.to_numeric(pre[c], errors="coerce").to_numpy(np.float64))
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
    out = pd.DataFrame(rows).sort_values("abs", ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze top-10% SELL selection: good vs bad separations (5m precontext only)")
    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="ETH 1m OHLCV+features parquet",
    )
    ap.add_argument(
        "--entry-model",
        default="data/entry_regressor_inliers_sell_ctx120_2026-01-04T18-55-10Z/models/entry_regressor_sell_inliers_ctx120_2026-01-04T18-55-10Z.joblib",
        help="SELL entry regressor (ctx120 inliers) joblib artifact",
    )
    ap.add_argument("--fee-total", type=float, default=0.001)
    ap.add_argument("--horizon-min", type=int, default=15)
    ap.add_argument("--pre-min", type=int, default=5)

    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--ret-thresh", type=float, default=0.2)
    ap.add_argument("--bad-thresh", type=float, default=0.0)

    ap.add_argument("--ctx-windows", default="30,60,120")
    ap.add_argument("--chunk", type=int, default=120_000)

    ap.add_argument("--out-dir", default="data")

    args = ap.parse_args()

    bars_path = Path(args.bars)
    if not bars_path.exists():
        raise SystemExit(f"bars not found: {bars_path}")

    art = joblib.load(args.entry_model)
    if not isinstance(art, dict) or "model" not in art or "feature_cols" not in art:
        raise SystemExit("entry model artifact must be a dict with keys: model, feature_cols")

    model = art["model"]
    model_feature_cols = list(art["feature_cols"])

    need_cols = ["timestamp", "open", "high", "low", "close", "volume"] + list(BASE_FEATS_DEFAULT)

    print(f"Loading bars: {bars_path}", flush=True)
    bars = pd.read_parquet(bars_path, columns=need_cols)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    if not bars["timestamp"].is_monotonic_increasing:
        bars = bars.sort_values("timestamp").reset_index(drop=True)

    fee_side = float(args.fee_total) / 2.0
    horizon = int(args.horizon_min)
    pre_min = int(args.pre_min)
    top_frac = float(args.top_frac)
    ret_thresh = float(args.ret_thresh)
    bad_thresh = float(args.bad_thresh)

    ctx_windows = [int(x.strip()) for x in str(args.ctx_windows).split(",") if x.strip()]

    print("Scoring ctx120 model on full dataset...", flush=True)
    pred, y_sell, valid = compute_pred_all(
        bars=bars,
        model=model,
        feature_cols=model_feature_cols,
        ctx_windows=ctx_windows,
        pre_min=pre_min,
        horizon=horizon,
        fee_side=fee_side,
        chunk=int(args.chunk),
    )

    pv = pred[valid].astype(np.float64)
    thr = float(np.nanquantile(pv, 1.0 - top_frac))

    sel_mask = valid & np.isfinite(pred) & (pred >= thr)
    sel_idx = np.where(sel_mask)[0]

    y_sel = y_sell[sel_idx].astype(np.float64)

    good = y_sel >= ret_thresh
    bad = y_sel < bad_thresh

    # exclude neutrals from separation
    keep = good | bad
    sel_idx2 = sel_idx[keep]
    y2 = y_sel[keep]
    good2 = y2 >= ret_thresh
    bad2 = y2 < bad_thresh

    print(f"Top-{top_frac*100:.2f}% selection: {sel_idx.size:,} rows (thr={thr:.6g})", flush=True)
    print(f"Good: {int(good.sum()):,}  Bad: {int(bad.sum()):,}  Neutral: {int((~(good|bad)).sum()):,}", flush=True)

    print("Computing 5m precontext features for selected rows...", flush=True)
    pre = compute_precontext_features_for_indices(bars=bars, idx=sel_idx2, pre_min=pre_min)

    shifts = feature_shifts_good_bad(pre, good_mask=good2, bad_mask=bad2)

    ts = now_ts()
    out_root = Path(args.out_dir) / f"analysis_sell_top10_goodbad_precontext_{ts}"
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
        "top_pred_threshold": float(thr),
        "selected_n": int(sel_idx.size),
        "selected_good_n": int(good.sum()),
        "selected_bad_n": int(bad.sum()),
        "selected_neutral_n": int((~(good | bad)).sum()),
        "selected_good_rate": float(np.mean(good)) if sel_idx.size else float("nan"),
        "selected_bad_rate": float(np.mean(bad)) if sel_idx.size else float("nan"),
        "shifts_rows": int(len(shifts)),
    }

    (out_root / f"summary_{ts}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    shifts.to_csv(out_root / f"feature_shifts_good_vs_bad_{ts}.csv", index=False)

    # small sample for inspection
    rng = np.random.default_rng(42)
    take = min(20000, int(sel_idx.size))
    pick = rng.choice(sel_idx, size=take, replace=False)
    pick.sort()
    sample = pd.DataFrame(
        {
            "timestamp": bars.loc[pick, "timestamp"].to_numpy(),
            "pred": pred[pick].astype(np.float64),
            "y_sell": y_sell[pick].astype(np.float64),
        }
    )
    sample.to_csv(out_root / f"selection_sample_{ts}.csv", index=False)

    print(json.dumps(summary, indent=2))
    print("Top 20 separations (robust median shift):")
    print(shifts.head(20).to_string(index=False))
    print("Wrote", out_root)


if __name__ == "__main__":
    main()
