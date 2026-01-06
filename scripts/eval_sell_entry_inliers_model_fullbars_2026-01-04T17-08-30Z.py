#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T17:08:30Z
"""Evaluate SELL inlier entry regressor on the full ETH 1m dataset.

User request
- "Run it on the entire dataset, backtest, and report the results"
- Specifically: for entries selected by the trained model, how many have >=0.2% oracle profitability
  with an oracle best exit in the next 15 minutes, and what is the distribution.

This uses the same label definition as scripts/train_eval_entry_regressor_oracle15m_2026-01-04T11-11-32Z.py:
- y_sell[t] = net_ret_pct_sell(close[t], min(close[t+1..t+15]), fee_side)

Selection
- Reports metrics for a set of absolute prediction thresholds AND top-q% selection rules.

Notes
- This evaluates per-minute entry candidates (no non-overlap constraint).
- Features are the 5-minute precontext descriptors (t-5..t-1), aligned to the model's feature_cols.
"""

from __future__ import annotations

import argparse
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
    """Build pre-5m feature matrix for all rows, then select requested columns.

    Each row t uses ONLY the previous pre_min minutes (t-pre_min .. t-1).
    """
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
    """For each i, return extreme over x[i+1:i+W+1]. mode in {'max','min'}.

    O(n) via deque on the reversed series.
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


def _summ(name: str, arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.9)),
        "p99": float(np.quantile(arr, 0.99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate SELL inlier entry model vs SELL 15m oracle best exit")
    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="ETH 1m OHLCV+features parquet",
    )
    ap.add_argument(
        "--entry-model",
        default="data/entry_regressor_inliers_sell_2026-01-04T16-49-43Z/models/entry_regressor_sell_inliers_2026-01-04T16-49-43Z.joblib",
        help="SELL entry regressor joblib artifact (dict with {model, feature_cols, ...})",
    )
    ap.add_argument("--fee-total", type=float, default=0.001, help="TOTAL round-trip fee (0.001=0.1% total)")
    ap.add_argument("--horizon-min", type=int, default=15)
    ap.add_argument("--pre-min", type=int, default=5)
    ap.add_argument("--ret-thresh", type=float, default=0.2)

    ap.add_argument("--chunk", type=int, default=200_000)
    args = ap.parse_args()

    bars_path = Path(args.bars)
    if not bars_path.exists():
        raise SystemExit(f"bars not found: {bars_path}")

    art = joblib.load(args.entry_model)
    if not isinstance(art, dict) or "model" not in art or "feature_cols" not in art:
        raise SystemExit("entry model artifact must be a dict with keys: model, feature_cols")

    model = art["model"]
    feature_cols = list(art["feature_cols"])

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

    print(f"Computing SELL oracle best-exit labels within {horizon}m...", flush=True)
    fut_min = _future_extrema_excl_current(close, horizon, mode="min")
    y_sell = net_ret_pct_sell(close, fut_min, fee_side)

    valid = (np.arange(n) >= pre_min) & np.isfinite(y_sell)
    valid &= np.isfinite(close)

    print(f"Valid minutes: {int(valid.sum()):,}/{n:,}", flush=True)

    print(f"Building pre-5m features for cols={len(feature_cols)} ...", flush=True)
    X_all = build_pre5m_feature_matrix(
        df,
        pre_min=pre_min,
        base_feat_cols=list(BASE_FEATS_DEFAULT),
        out_feature_cols=feature_cols,
    )

    # score model on valid region (avoid last horizon, but already excluded by y_sell finite)
    pred = np.full(n, np.nan, dtype=np.float32)
    idxs = np.where(valid)[0]

    print(f"Scoring model: n_valid={idxs.size:,} chunk={int(args.chunk):,} ...", flush=True)
    chunk = int(args.chunk)
    for s in range(0, idxs.size, chunk):
        e = min(idxs.size, s + chunk)
        ii = idxs[s:e]
        pred[ii] = model.predict(X_all[ii]).astype(np.float32)
        if (s // chunk) % 10 == 0:
            print(f"  scored {e:,}/{idxs.size:,}", flush=True)

    yv = y_sell[valid]
    pv = pred[valid].astype(np.float64)

    # overall base rates
    ret_thr = float(args.ret_thresh)
    base_win = float(np.mean(yv >= ret_thr))
    print("\n=== Overall (all valid minutes) ===")
    print(f"base_rate_y>= {ret_thr:.2f}% : {base_win:.6f}  (n={yv.size:,})")
    print("y_sell summary:", _summ("y_sell", yv))

    # selection rules
    abs_thrs = [0.2, 0.3, 0.4, 0.5, 0.6]
    top_qs = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]

    rows = []

    def add_row(name: str, mask: np.ndarray) -> None:
        mask = mask.astype(bool)
        sel_y = y_sell[mask]
        sel_y_win = sel_y[sel_y >= ret_thr]
        rows.append(
            {
                "rule": name,
                "selected": int(mask.sum()),
                "win_rate_y>=thr": float(np.mean(sel_y >= ret_thr)) if sel_y.size else float("nan"),
                "wins": int(sel_y_win.size),
                "sel_median": float(np.median(sel_y)) if sel_y.size else float("nan"),
                "sel_p90": float(np.quantile(sel_y, 0.9)) if sel_y.size else float("nan"),
                "sel_p99": float(np.quantile(sel_y, 0.99)) if sel_y.size else float("nan"),
                "win_median": float(np.median(sel_y_win)) if sel_y_win.size else float("nan"),
                "win_p90": float(np.quantile(sel_y_win, 0.9)) if sel_y_win.size else float("nan"),
                "win_p99": float(np.quantile(sel_y_win, 0.99)) if sel_y_win.size else float("nan"),
            }
        )

    base_mask = valid & np.isfinite(pred)

    print("\n=== Selection by absolute pred thresholds ===")
    for thr in abs_thrs:
        m = base_mask & (pred >= float(thr))
        add_row(f"pred>={thr:.2f}", m)

    print("\n=== Selection by top-q% over all valid preds ===")
    pv_f = pv[np.isfinite(pv)]
    if pv_f.size == 0:
        raise SystemExit("No finite predictions")

    for q in top_qs:
        thr = float(np.quantile(pv_f, 1.0 - q))
        m = base_mask & (pred >= thr)
        add_row(f"top{q*100:.3f}% (thr={thr:.4f})", m)

    out = pd.DataFrame(rows).sort_values("selected", ascending=False)

    # Print concise table
    show_cols = [
        "rule",
        "selected",
        "wins",
        "win_rate_y>=thr",
        "sel_median",
        "sel_p90",
        "sel_p99",
        "win_median",
        "win_p90",
        "win_p99",
    ]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print("\n=== Summary table (ret_thresh applies to y_sell) ===")
    print(out[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
