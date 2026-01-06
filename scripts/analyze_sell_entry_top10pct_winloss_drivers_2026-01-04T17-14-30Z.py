#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T17:14:30Z
"""Analyze drivers of big wins / big losses for top-10% SELL entries.

We:
- Load full ETH 1m dataset.
- Compute SELL oracle best-exit return within next 15m (y_sell).
- Score the trained SELL inlier entry regressor on all valid minutes.
- Select top 10% by predicted score.
- Within that selection:
  - define big winners as top 1% by y_sell
  - define big losers as bottom 1% by y_sell
  - also track "wins" as y_sell>=0.2 and "losses" as y_sell<0.0
- Report which pre-entry features differ most (robust median shift) for big winners/losers.

Outputs (timestamped)
- data/analysis_sell_entry_top10pct_winloss_<ts>/
  - summary.json
  - selection_sample.csv
  - big_winners.csv
  - big_losers.csv
  - feature_shifts_bigwin_vs_rest.csv
  - feature_shifts_bigloss_vs_rest.csv
  - feature_corr_with_y_in_top10.csv

Note
- This is diagnostic; it does not simulate non-overlap trading.
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

    vol_prev = _rolling_prev_matrix(vol, L)
    vol_log = np.log1p(np.maximum(0.0, vol_prev))
    vv = _agg_5(vol_log, "vol_log1p")
    for k, v in vv.items():
        _set(k, v)

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


def feature_shifts(df: pd.DataFrame, *, group_mask: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    g = group_mask.astype(bool)
    a = df.loc[~g, feature_cols]
    b = df.loc[g, feature_cols]

    rows = []
    for c in feature_cols:
        xa = pd.to_numeric(a[c], errors="coerce").to_numpy(np.float64)
        xb = pd.to_numeric(b[c], errors="coerce").to_numpy(np.float64)
        if np.isfinite(xa).sum() < 2000 or np.isfinite(xb).sum() < 50:
            continue
        med_a = float(np.nanmedian(xa))
        med_b = float(np.nanmedian(xb))
        scale = _robust_scale(pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64))
        z = (med_b - med_a) / scale if np.isfinite(scale) and scale > 0 else float("nan")
        rows.append(
            {
                "feature": c,
                "median_rest": med_a,
                "median_group": med_b,
                "robust_med_shift": float(z),
            }
        )

    out = pd.DataFrame(rows)
    out["abs_shift"] = out["robust_med_shift"].abs()
    out = out.sort_values("abs_shift", ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze big wins/losses for top-10% predicted SELL entries")
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
    ap.add_argument("--fee-total", type=float, default=0.001)
    ap.add_argument("--horizon-min", type=int, default=15)
    ap.add_argument("--pre-min", type=int, default=5)
    ap.add_argument("--top-frac", type=float, default=0.10, help="Top fraction by prediction to analyze")
    ap.add_argument("--big-quantile", type=float, default=0.01, help="Tail quantile for big winners/losers within selection")
    ap.add_argument("--ret-thresh", type=float, default=0.2)
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    bars_path = Path(args.bars)
    if not bars_path.exists():
        raise SystemExit(f"bars not found: {bars_path}")

    art = joblib.load(args.entry_model)
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

    print(f"Computing y_sell (oracle min close within {horizon}m)...", flush=True)
    fut_min = _future_extrema_excl_current(close, horizon, mode="min")
    y_sell = net_ret_pct_sell(close, fut_min, fee_side)

    valid = (np.arange(n) >= pre_min) & np.isfinite(y_sell) & np.isfinite(close)

    print(f"Building pre-5m features for cols={len(feature_cols)} ...", flush=True)
    X_all = build_pre5m_feature_matrix(
        df,
        pre_min=pre_min,
        base_feat_cols=list(BASE_FEATS_DEFAULT),
        out_feature_cols=feature_cols,
    )

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

    pv = pred[valid].astype(np.float64)
    pv = pv[np.isfinite(pv)]
    if pv.size == 0:
        raise SystemExit("No finite predictions")

    top_frac = float(args.top_frac)
    thr_pred = float(np.quantile(pv, 1.0 - top_frac))

    sel_mask = valid & np.isfinite(pred) & (pred >= thr_pred)
    sel_idx = np.where(sel_mask)[0]

    sel_y = y_sell[sel_mask]
    sel_p = pred[sel_mask].astype(np.float64)

    # Build analysis frame with features for selected entries
    sel_X = X_all[sel_idx]
    feat_df = pd.DataFrame(sel_X, columns=feature_cols)
    out_df = pd.DataFrame(
        {
            "idx": sel_idx.astype(np.int64),
            "timestamp": df.loc[sel_idx, "timestamp"].to_numpy(),
            "close": close[sel_idx].astype(np.float64),
            "pred": sel_p,
            "y_sell": sel_y.astype(np.float64),
            "is_win_0p2": (sel_y >= float(args.ret_thresh)).astype(np.int64),
            "is_loss_lt0": (sel_y < 0.0).astype(np.int64),
        }
    )
    out_df = pd.concat([out_df, feat_df], axis=1)

    q = float(args.big_quantile)
    lo = float(np.quantile(sel_y, q))
    hi = float(np.quantile(sel_y, 1.0 - q))

    big_loss_mask = out_df["y_sell"].to_numpy(np.float64) <= lo
    big_win_mask = out_df["y_sell"].to_numpy(np.float64) >= hi

    # correlations inside selection
    corr_rows = []
    yv = out_df["y_sell"].to_numpy(np.float64)
    for c in feature_cols:
        x = pd.to_numeric(out_df[c], errors="coerce").to_numpy(np.float64)
        if np.isfinite(x).sum() < 1000:
            continue
        corr = np.corrcoef(np.nan_to_num(x, nan=np.nanmedian(x)), yv)[0, 1]
        if not np.isfinite(corr):
            continue
        corr_rows.append({"feature": c, "pearson_corr": float(corr), "abs": float(abs(corr))})
    corr_df = pd.DataFrame(corr_rows).sort_values("abs", ascending=False).reset_index(drop=True)

    # feature shifts
    shifts_win = feature_shifts(out_df, group_mask=big_win_mask, feature_cols=feature_cols)
    shifts_loss = feature_shifts(out_df, group_mask=big_loss_mask, feature_cols=feature_cols)

    ts = now_ts()
    out_root = Path(args.out_dir) / f"analysis_sell_entry_top10pct_winloss_{ts}"
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
        "top_pred_threshold": float(thr_pred),
        "selected_n": int(out_df.shape[0]),
        "selected_win_rate_0p2": float(out_df["is_win_0p2"].mean()),
        "selected_loss_rate_lt0": float(out_df["is_loss_lt0"].mean()),
        "selected_y_sell_quantiles": {
            "p01": float(np.quantile(sel_y, 0.01)),
            "p10": float(np.quantile(sel_y, 0.10)),
            "p50": float(np.quantile(sel_y, 0.50)),
            "p90": float(np.quantile(sel_y, 0.90)),
            "p99": float(np.quantile(sel_y, 0.99)),
        },
        "big_quantile": float(q),
        "big_loss_cutoff": float(lo),
        "big_win_cutoff": float(hi),
        "big_loss_n": int(big_loss_mask.sum()),
        "big_win_n": int(big_win_mask.sum()),
    }

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Save examples + samples
    # Keep columns compact for the human-facing CSVs
    base_cols = ["idx", "timestamp", "close", "pred", "y_sell", "is_win_0p2", "is_loss_lt0"]

    # Sample of selection
    out_df[base_cols].sample(n=min(50000, len(out_df)), random_state=42).to_csv(out_root / "selection_sample.csv", index=False)

    # Big tails (include features)
    out_df.loc[big_win_mask].sort_values("y_sell", ascending=False).head(2000).to_csv(out_root / "big_winners.csv", index=False)
    out_df.loc[big_loss_mask].sort_values("y_sell", ascending=True).head(2000).to_csv(out_root / "big_losers.csv", index=False)

    shifts_win.to_csv(out_root / "feature_shifts_bigwin_vs_rest.csv", index=False)
    shifts_loss.to_csv(out_root / "feature_shifts_bigloss_vs_rest.csv", index=False)
    corr_df.to_csv(out_root / "feature_corr_with_y_in_top10.csv", index=False)

    print(json.dumps(summary, indent=2))
    print("Wrote", out_root)


if __name__ == "__main__":
    main()
