#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T19:15:27Z
"""Backtest SELL entry model (ctx120 inliers) vs 15m oracle exit on the full dataset.

Meaning here
- We score every valid minute as a candidate SELL entry.
- Oracle exit label: y_sell[t] = net_ret_pct_sell(close[t], min(close[t+1..t+15]), fee_side).
- We report selection quality for:
  - absolute pred thresholds
  - top-q% by pred

Important
- No non-overlap constraint (selection diagnostic, not execution simulation).
- Feature computation matches training intent:
  - entry at t uses ONLY prior 5 minutes (t-5..t-1)
  - 30/60/120m context series are computed *causally* using only t-1 and earlier

Outputs (timestamped)
- data/backtest_sell_entry_ctx120_oracle15m_<ts>/
  - summary_<ts>.json
  - selection_rules_<ts>.csv
  - preds_sample_<ts>.csv
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
    """For each i, return extreme over x[i+1:i+W+1]. mode in {'max','min'}.

    O(n) via deque on reversed series.
    """
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
    """Row-wise pop std with NaN-safe behavior; x shape (m,5)."""
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


def backtest(
    *,
    bars: pd.DataFrame,
    model,
    feature_cols: list[str],
    fee_side: float,
    horizon: int,
    pre_min: int,
    ret_thresh: float,
    ctx_windows: list[int],
    chunk: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    n = int(len(bars))

    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(np.float64, copy=False)
    volume = pd.to_numeric(bars["volume"], errors="coerce").to_numpy(np.float64, copy=False)

    # base feature arrays (already computed in parquet)
    base_arrs = {}
    for c in BASE_FEATS_DEFAULT:
        base_arrs[c] = pd.to_numeric(bars[c], errors="coerce").to_numpy(np.float64, copy=False)

    # compute context series and attach arrays
    ctx_df = compute_ctx_series(bars, ctx_windows)
    ctx_arrs = {c: pd.to_numeric(ctx_df[c], errors="coerce").to_numpy(np.float64, copy=False) for c in ctx_df.columns}

    # oracle labels
    fut_min = _future_extrema_excl_current(close, horizon, mode="min")
    y_sell = net_ret_pct_sell(close, fut_min, fee_side)

    valid = (np.arange(n) >= pre_min) & np.isfinite(y_sell) & np.isfinite(close)

    # feature column mapping
    col_to_j = {c: j for j, c in enumerate(feature_cols)}

    # Determine which series prefixes are needed (excluding px_close_norm_pct, vol_log1p)
    prefixes = set()
    for c in feature_cols:
        if "__" in c:
            prefixes.add(c.split("__", 1)[0])
    prefixes.discard("px_close_norm_pct")
    prefixes.discard("vol_log1p")

    # Build source series lookup for those prefixes
    series_map: dict[str, np.ndarray] = {}
    for p in prefixes:
        if p in base_arrs:
            series_map[p] = base_arrs[p]
        elif p in ctx_arrs:
            series_map[p] = ctx_arrs[p]
        elif p in ("range_5m", "range_norm_5m"):
            # already in base_arrs, but keep safe
            series_map[p] = base_arrs[p]
        else:
            # Not expected for this model; ignore.
            pass

    idxs = np.where(valid)[0]

    pred = np.full(n, np.nan, dtype=np.float32)

    def _set(X: np.ndarray, name: str, arr: np.ndarray) -> None:
        j = col_to_j.get(name)
        if j is None:
            return
        X[:, j] = np.asarray(arr, dtype=np.float32)

    print(f"Valid minutes: {idxs.size:,}/{n:,}")
    print(f"Scoring model on valid minutes in chunks of {chunk:,} ...")

    for s in range(0, idxs.size, chunk):
        e = min(idxs.size, s + chunk)
        ii = idxs[s:e]

        m = ii.size
        Xc = np.full((m, len(feature_cols)), np.nan, dtype=np.float32)

        # indices for t-5..t-1
        i0 = ii - 5
        i1 = ii - 4
        i2 = ii - 3
        i3 = ii - 2
        i4 = ii - 1

        # close normalized path (relative to close[t-1])
        close_w = np.stack([close[i0], close[i1], close[i2], close[i3], close[i4]], axis=1).astype(np.float64)
        close_last = close_w[:, -1]
        close_last = np.where((np.isfinite(close_last)) & (close_last > 0.0), close_last, np.nan)
        close_norm = (close_w / close_last[:, None] - 1.0) * 100.0

        px = _agg5_rows(close_norm)
        for suf, arr in px.items():
            _set(Xc, f"px_close_norm_pct{suf}", arr)
        # extra px features
        px_ret5m = close_norm[:, -1] - close_norm[:, 0]
        _set(Xc, "px_close_norm_pct__ret5m", px_ret5m)
        _set(Xc, "px_close_norm_pct__absret5m", np.abs(px_ret5m))

        # volume log1p window
        vol_w = np.stack([volume[i0], volume[i1], volume[i2], volume[i3], volume[i4]], axis=1).astype(np.float64)
        vol_log = np.log1p(np.maximum(0.0, vol_w))
        vv = _agg5_rows(vol_log)
        for suf, arr in vv.items():
            _set(Xc, f"vol_log1p{suf}", arr)

        # missing close flags
        if "missing_close_n" in col_to_j or "missing_any" in col_to_j:
            miss = ~np.isfinite(close_w)
            miss_n = miss.sum(axis=1).astype(np.float64)
            _set(Xc, "missing_close_n", miss_n)
            _set(Xc, "missing_any", (miss_n > 0).astype(np.float64))

        # aggregated windows for each required series
        for name, arr in series_map.items():
            w = np.stack([arr[i0], arr[i1], arr[i2], arr[i3], arr[i4]], axis=1).astype(np.float64)
            a = _agg5_rows(w)
            for suf, vv2 in a.items():
                _set(Xc, f"{name}{suf}", vv2)

        pred[ii] = model.predict(Xc).astype(np.float32)

        if (s // chunk) % 2 == 0:
            print(f"  scored {e:,}/{idxs.size:,}")

    # Build selection tables
    yv = y_sell[valid]
    pv = pred[valid].astype(np.float64)

    base_win = float(np.mean(yv >= ret_thresh))

    def _summ(arr: np.ndarray) -> dict:
        a = np.asarray(arr, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"n": 0}
        return {
            "n": int(a.size),
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "p90": float(np.quantile(a, 0.9)),
            "p99": float(np.quantile(a, 0.99)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }

    rows = []

    def add_row(rule: str, mask_valid: np.ndarray) -> None:
        sel = mask_valid.astype(bool)
        sel_y = y_sell[valid][sel]
        if sel_y.size == 0:
            rows.append({"rule": rule, "selected": 0})
            return
        rows.append(
            {
                "rule": rule,
                "selected": int(sel_y.size),
                "win_rate_y>=ret": float(np.mean(sel_y >= ret_thresh)),
                "bad_rate_y<0": float(np.mean(sel_y < 0.0)),
                "mean_y": float(np.mean(sel_y)),
                "median_y": float(np.median(sel_y)),
                "p90_y": float(np.quantile(sel_y, 0.9)),
            }
        )

    # Absolute thresholds: use quantiles of prediction too (since scale is arbitrary)
    abs_thrs = [float(np.quantile(pv, q)) for q in [0.90, 0.95, 0.98, 0.99, 0.995]]
    abs_thrs = sorted(set(abs_thrs))
    for thr in abs_thrs:
        add_row(f"pred>=q(thr={thr:.6g})", pv >= thr)

    for q in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]:
        thr = float(np.quantile(pv, 1.0 - q))
        add_row(f"top_{q*100:.2f}% (thr={thr:.6g})", pv >= thr)

    rules_df = pd.DataFrame(rows)

    # Pick "best" by mean_y among rules with at least 1000 selections
    best = None
    if not rules_df.empty and "mean_y" in rules_df.columns:
        sub = rules_df[rules_df.get("selected", 0) >= 1000].copy()
        if not sub.empty:
            best_row = sub.sort_values("mean_y", ascending=False).iloc[0].to_dict()
            best = best_row

    summary = {
        "n_total": int(n),
        "n_valid": int(valid.sum()),
        "bars_start": str(bars["timestamp"].iloc[0]) if "timestamp" in bars.columns and len(bars) else None,
        "bars_end": str(bars["timestamp"].iloc[-1]) if "timestamp" in bars.columns and len(bars) else None,
        "horizon_min": int(horizon),
        "pre_min": int(pre_min),
        "fee_side": float(fee_side),
        "ret_thresh": float(ret_thresh),
        "base_rate_y>=ret": float(base_win),
        "y_sell_summary": _summ(yv),
        "pred_summary_on_valid": _summ(pv),
        "best_rule_min1000_by_mean_y": best,
    }

    # Sample rows for inspection
    sample_idx = np.where(valid)[0]
    rng = np.random.default_rng(42)
    take = min(20000, int(sample_idx.size))
    pick = rng.choice(sample_idx, size=take, replace=False)
    pick.sort()
    sample = pd.DataFrame(
        {
            "timestamp": bars.loc[pick, "timestamp"].to_numpy() if "timestamp" in bars.columns else pick,
            "pred": pred[pick].astype(np.float64),
            "y_sell": y_sell[pick].astype(np.float64),
        }
    )

    return summary, rules_df, sample


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest SELL entry model (ctx120 inliers) vs oracle 15m best exit")
    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="ETH 1m OHLCV+features parquet",
    )
    ap.add_argument(
        "--entry-model",
        default="data/entry_regressor_inliers_sell_ctx120_2026-01-04T18-55-10Z/models/entry_regressor_sell_inliers_ctx120_2026-01-04T18-55-10Z.joblib",
        help="SELL entry regressor joblib artifact (dict with {model, feature_cols, ...})",
    )
    ap.add_argument("--fee-total", type=float, default=0.001)
    ap.add_argument("--horizon-min", type=int, default=15)
    ap.add_argument("--pre-min", type=int, default=5)
    ap.add_argument("--ret-thresh", type=float, default=0.2)

    ap.add_argument("--ctx-windows", default="30,60,120")
    ap.add_argument("--chunk", type=int, default=120_000)

    ap.add_argument("--out-dir", default="data")

    args = ap.parse_args()

    art = joblib.load(args.entry_model)
    if not isinstance(art, dict) or "model" not in art or "feature_cols" not in art:
        raise SystemExit("entry model artifact must be a dict with keys: model, feature_cols")

    model = art["model"]
    feature_cols = list(art["feature_cols"])

    bars_path = Path(args.bars)
    if not bars_path.exists():
        raise SystemExit(f"bars not found: {bars_path}")

    # Need OHLCV + base features
    need_cols = ["timestamp", "open", "high", "low", "close", "volume"] + list(BASE_FEATS_DEFAULT)

    print(f"Loading bars: {bars_path}")
    bars = pd.read_parquet(bars_path, columns=need_cols)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    if not bars["timestamp"].is_monotonic_increasing:
        bars = bars.sort_values("timestamp").reset_index(drop=True)

    fee_side = float(args.fee_total) / 2.0
    horizon = int(args.horizon_min)
    pre_min = int(args.pre_min)
    ret_thresh = float(args.ret_thresh)

    ctx_windows = [int(x.strip()) for x in str(args.ctx_windows).split(",") if x.strip()]

    summary, rules_df, sample = backtest(
        bars=bars,
        model=model,
        feature_cols=feature_cols,
        fee_side=fee_side,
        horizon=horizon,
        pre_min=pre_min,
        ret_thresh=ret_thresh,
        ctx_windows=ctx_windows,
        chunk=int(args.chunk),
    )

    ts = now_ts()
    out_root = Path(args.out_dir) / f"backtest_sell_entry_ctx120_oracle15m_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    (out_root / f"summary_{ts}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    rules_df.to_csv(out_root / f"selection_rules_{ts}.csv", index=False)
    sample.to_csv(out_root / f"preds_sample_{ts}.csv", index=False)

    print(json.dumps(summary, indent=2))
    print("Wrote", out_root)


if __name__ == "__main__":
    main()
