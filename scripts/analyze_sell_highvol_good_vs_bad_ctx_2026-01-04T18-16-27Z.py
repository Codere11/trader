#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T18:16:27Z
"""Analyze SELL high-vol: clean split good vs bad using context + oracle precontext features.

This is a *diagnostic* script (no shared pipeline changes).

Steps
1) Load 1m ETH bars (OHLCV + base 1m features).
2) Compute y_sell[t] = best-exit net return for a SELL within next horizon minutes.
3) Score the existing SELL entry model and select top-N% by prediction.
4) Inside the selection, define high-vol using proxy hits (>= hv_min_count of proxies in top hv_q).
5) In that HV subset, define:
   - good: y_sell >= ret_thresh
   - bad:  y_sell <  bad_thresh
   - neutral otherwise (excluded from training)
6) Build features:
   - within-5m precontext windows (same as prior) + within-5m shape features
   - NEW: 30/60/120m context state features at t
   - NEW: 30/60/120m exhaustion/bounce features on close (drawdown/bounce/pos-in-range)
7) Train a gate model; run 1D and 2D rule sweeps to find interpretable splits.

Outputs (timestamped)
- data/highvol_sell_goodbad_ctx_<ts>/
  - summary_<ts>.json
  - hv_proxy_thresholds_<ts>.json
  - gate_metrics_<ts>.json
  - gate_feature_importance_<ts>.csv
  - feature_shifts_good_vs_bad_<ts>.csv
  - guardrail_sweep_1d_<ts>.csv
  - guardrail_sweep_2d_<ts>.csv
  - examples_good_<ts>.csv, examples_bad_<ts>.csv (small samples)

Notes
- “Clean split” here means *interpretable* conditions that increase good_rate while reducing bad_rate
  at non-trivial coverage; perfect separability is not assumed.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
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
    """Match the feature construction used in learn_sell_highvol_gate_v2/v3 scripts."""
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
    idx = np.asarray(idx, dtype=np.int64)
    mats = []
    for k in range(L, 0, -1):
        mats.append(x[idx - k])
    return np.stack(mats, axis=1)


def _sign_changes(w: np.ndarray) -> np.ndarray:
    s = np.sign(w)
    s[s == 0] = np.nan
    changes = (np.isfinite(s[:, 1:]) & np.isfinite(s[:, :-1]) & (s[:, 1:] != s[:, :-1])).sum(axis=1)
    return changes.astype(np.float64)


def _add_within5m_features(*, base_df: pd.DataFrame, raw: pd.DataFrame, idx: np.ndarray, pre_min: int) -> pd.DataFrame:
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
    close_norm = (close_w / close_last[:, None] - 1.0) * 100.0

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


def _rolling_sum_nan_prefix(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    cs = np.cumsum(x)
    out[w - 1 :] = cs[w - 1 :] - np.concatenate(([0.0], cs[: n - w]))
    return out


def _rolling_std_pop_nan_prefix(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    cs = np.cumsum(x)
    cs2 = np.cumsum(x * x)
    sum_w = cs[w - 1 :] - np.concatenate(([0.0], cs[: n - w]))
    sum2_w = cs2[w - 1 :] - np.concatenate(([0.0], cs2[: n - w]))
    mean = sum_w / float(w)
    var = sum2_w / float(w) - mean * mean
    var = np.maximum(var, 0.0)
    out[w - 1 :] = np.sqrt(var)
    return out


def _add_context_state_features(df: pd.DataFrame, windows: list[int]) -> list[str]:
    """Add 30/60/120m context state + exhaustion metrics (in-place). Return new col names."""
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64, copy=False)
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(np.float64, copy=False)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(np.float64, copy=False)
    vol = pd.to_numeric(df["volume"], errors="coerce").to_numpy(np.float64, copy=False)

    s_close = pd.Series(close)
    s_high = pd.Series(high)
    s_low = pd.Series(low)

    added = []
    for w in windows:
        w = int(w)
        # momentum / returns
        mom = np.full_like(close, np.nan)
        mom[w:] = (close[w:] / close[:-w] - 1.0) * 100.0
        c = f"mom_{w}m_pct"
        df[c] = mom
        added.append(c)

        c = f"vol_std_{w}m"
        df[c] = _rolling_std_pop_nan_prefix(close, w)
        added.append(c)

        # range and normalization
        r = (s_high.rolling(w, min_periods=w).max() - s_low.rolling(w, min_periods=w).min()).to_numpy(np.float64)
        c = f"range_{w}m"
        df[c] = r
        added.append(c)

        c = f"range_norm_{w}m"
        df[c] = r / np.maximum(1e-9, close)
        added.append(c)

        # rolling min/max close for exhaustion proxies
        cmin = s_close.rolling(w, min_periods=w).min().to_numpy(np.float64)
        cmax = s_close.rolling(w, min_periods=w).max().to_numpy(np.float64)
        crng = cmax - cmin
        eps = 1e-9

        c = f"close_dd_from_{w}m_max_pct"
        df[c] = (cmax / np.maximum(1e-9, close) - 1.0) * 100.0
        added.append(c)

        c = f"close_bounce_from_{w}m_min_pct"
        df[c] = (close / np.maximum(1e-9, cmin) - 1.0) * 100.0
        added.append(c)

        c = f"close_pos_in_{w}m_range"
        df[c] = (close - cmin) / (crng + eps)
        added.append(c)

        # VWAP dev over window
        sum_v = _rolling_sum_nan_prefix(vol, w)
        sum_pv = _rolling_sum_nan_prefix(close * vol, w)
        vwap = sum_pv / np.maximum(1e-9, sum_v)
        vwap_dev = np.where(sum_v > 0.0, ((close - vwap) / np.maximum(1e-9, vwap)) * 100.0, 0.0)
        c = f"vwap_dev_{w}m"
        df[c] = vwap_dev
        added.append(c)

    return added


@dataclass(frozen=True)
class Cond:
    feature: str
    op: str  # ">=" or "<="
    thr: float

    def name(self) -> str:
        return f"{self.feature} {self.op} {self.thr:.6g}"


def _sweep_1d(hv: pd.DataFrame, *, feature_cols: list[str], ret_thresh: float, bad_thresh: float, quantiles: list[float]) -> pd.DataFrame:
    y = pd.to_numeric(hv["y_sell"], errors="coerce").to_numpy(np.float64)
    rows = []
    for c in feature_cols:
        x = pd.to_numeric(hv[c], errors="coerce").to_numpy(np.float64)
        xf = x[np.isfinite(x)]
        if xf.size < 500:
            continue
        qs = np.unique(np.nanquantile(xf, quantiles))
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
                        "selected_n": int(ys.size),
                        "good_rate": gr,
                        "bad_rate": br,
                        "neutral_rate": float(np.mean((ys >= bad_thresh) & (ys < ret_thresh))),
                        "median_y": float(np.median(ys)),
                        "p10_y": float(np.quantile(ys, 0.10)),
                        "p90_y": float(np.quantile(ys, 0.90)),
                        "score": float(score),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["score", "coverage"], ascending=[False, False]).reset_index(drop=True)


def _top_conditions_from_1d(sweep: pd.DataFrame, *, top_k: int, max_cond: int) -> list[Cond]:
    if sweep.empty:
        return []
    # Keep diverse: take top by score, but cap per-feature
    by_feat = {}
    conds: list[Cond] = []
    for _, r in sweep.sort_values("score", ascending=False).iterrows():
        f = str(r["feature"])
        by_feat.setdefault(f, 0)
        if by_feat[f] >= top_k:
            continue
        by_feat[f] += 1
        conds.append(Cond(feature=f, op=str(r["op"]), thr=float(r["threshold"])))
        if len(conds) >= max_cond:
            break
    return conds


def _mask_for_cond(hv: pd.DataFrame, cond: Cond) -> np.ndarray:
    x = pd.to_numeric(hv[cond.feature], errors="coerce").to_numpy(np.float64)
    if cond.op == ">=":
        return np.isfinite(x) & (x >= cond.thr)
    return np.isfinite(x) & (x <= cond.thr)


def _sweep_2d_from_conds(
    hv: pd.DataFrame,
    *,
    conds: list[Cond],
    ret_thresh: float,
    bad_thresh: float,
    min_cov: float,
    max_cov: float,
    max_pairs: int,
) -> pd.DataFrame:
    y = pd.to_numeric(hv["y_sell"], errors="coerce").to_numpy(np.float64)

    masks = [(_mask_for_cond(hv, c), c) for c in conds]

    rows = []
    pairs = 0
    for i in range(len(masks)):
        mi, ci = masks[i]
        for j in range(i + 1, len(masks)):
            mj, cj = masks[j]
            m = mi & mj
            cov = float(np.mean(m))
            if cov < min_cov or cov > max_cov:
                continue
            ys = y[m]
            if ys.size < 500:
                continue
            gr = float(np.mean(ys >= ret_thresh))
            br = float(np.mean(ys < bad_thresh))
            score = (gr - br) * (cov**0.5)
            rows.append(
                {
                    "cond1": ci.name(),
                    "cond2": cj.name(),
                    "coverage": cov,
                    "selected_n": int(ys.size),
                    "good_rate": gr,
                    "bad_rate": br,
                    "neutral_rate": float(np.mean((ys >= bad_thresh) & (ys < ret_thresh))),
                    "median_y": float(np.median(ys)),
                    "p10_y": float(np.quantile(ys, 0.10)),
                    "p90_y": float(np.quantile(ys, 0.90)),
                    "score": float(score),
                }
            )
            pairs += 1
            if pairs >= max_pairs:
                break
        if pairs >= max_pairs:
            break

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["score", "coverage"], ascending=[False, False]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze SELL high-vol good vs bad with context features")
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

    ap.add_argument("--gate-n-estimators", type=int, default=1200)
    ap.add_argument("--gate-test-frac", type=float, default=0.20)

    ap.add_argument("--ctx-windows", default="30,60,120")

    ap.add_argument("--out-dir", default="data")

    ap.add_argument("--sweep-quantiles", default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95")
    ap.add_argument("--sweep-topk-per-feature", type=int, default=2)
    ap.add_argument("--sweep-max-conds", type=int, default=120)
    ap.add_argument("--sweep-2d-min-cov", type=float, default=0.02)
    ap.add_argument("--sweep-2d-max-cov", type=float, default=0.90)
    ap.add_argument("--sweep-2d-max-pairs", type=int, default=15000)

    ap.add_argument("--examples-n", type=int, default=200)

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

    need_cols = ["timestamp", "open", "high", "low", "close", "volume"] + list(BASE_FEATS_DEFAULT)
    need_cols = sorted(set(need_cols))

    print(f"Loading bars: {bars_path}", flush=True)
    df = pd.read_parquet(bars_path, columns=need_cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if not df["timestamp"].is_monotonic_increasing:
        df = df.sort_values("timestamp").reset_index(drop=True)

    ctx_windows = [int(x.strip()) for x in str(args.ctx_windows).split(",") if x.strip()]
    print(f"Adding context state features for windows: {ctx_windows}", flush=True)
    ctx_added = _add_context_state_features(df, ctx_windows)

    n = int(len(df))
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64, copy=False)

    print(f"Computing y_sell within {horizon}m...", flush=True)
    fut_min = _future_extrema_excl_current(close, horizon, mode="min")
    y_sell = net_ret_pct_sell(close, fut_min, fee_side)

    valid = (np.arange(n) >= pre_min) & np.isfinite(y_sell) & np.isfinite(close)
    idxs = np.where(valid)[0]

    print(f"Building 5m descriptor matrix for entry model (cols={len(feat_cols)})...", flush=True)
    X_all = build_pre5m_feature_matrix(df, pre_min=pre_min, base_feat_cols=list(BASE_FEATS_DEFAULT), out_feature_cols=feat_cols)

    print("Scoring entry model...", flush=True)
    pred = np.full(n, np.nan, dtype=np.float32)
    # Chunk predict for memory safety
    chunk = 200_000
    for s in range(0, idxs.size, chunk):
        e = min(idxs.size, s + chunk)
        ii = idxs[s:e]
        pred[ii] = entry_model.predict(X_all[ii]).astype(np.float32)
        # Print every 2 chunks to avoid "frozen" appearance
        if (s // chunk) % 2 == 0:
            print(f"  scored {e:,}/{idxs.size:,}", flush=True)

    pv = pred[valid].astype(np.float64)
    pv = pv[np.isfinite(pv)]
    if pv.size == 0:
        raise SystemExit("No finite predictions")

    top_frac = float(args.top_frac)
    thr_top = float(np.quantile(pv, 1.0 - top_frac))
    sel_mask = valid & np.isfinite(pred) & (pred >= thr_top)
    sel_idx = np.where(sel_mask)[0]

    sel_base = pd.DataFrame(
        {
            "idx": sel_idx.astype(np.int64),
            "timestamp": df.loc[sel_idx, "timestamp"].to_numpy(),
            "pred": pred[sel_idx].astype(np.float64),
            "y_sell": y_sell[sel_idx].astype(np.float64),
        }
    )
    sel_feat = pd.DataFrame(X_all[sel_idx], columns=feat_cols)

    sel_ctx = df.loc[sel_idx, ctx_added].reset_index(drop=True)
    sel = pd.concat([sel_base, sel_feat.reset_index(drop=True), sel_ctx], axis=1)

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

    # Add within-5m shape features (computed only on HV indices)
    hv_idx = hv["idx"].to_numpy(np.int64)
    hv = _add_within5m_features(base_df=hv, raw=df, idx=hv_idx, pre_min=pre_min)

    ret_thresh = float(args.ret_thresh)
    bad_thresh = float(args.bad_thresh)

    hv["label_good"] = (hv["y_sell"] >= ret_thresh).astype(np.int32)
    hv["label_bad"] = (hv["y_sell"] < bad_thresh).astype(np.int32)
    hv["label_neutral"] = ((hv["y_sell"] >= bad_thresh) & (hv["y_sell"] < ret_thresh)).astype(np.int32)

    gate_df = hv[hv["label_neutral"] == 0].copy().reset_index(drop=True)
    gate_df["timestamp"] = pd.to_datetime(gate_df["timestamp"], utc=True, errors="coerce")
    gate_df = gate_df.sort_values("timestamp").reset_index(drop=True)

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

    # Early stopping to keep runtime bounded
    try:
        from lightgbm import early_stopping

        clf.fit(
            X_tr,
            y_tr,
            eval_set=[(X_te, y_te)],
            eval_metric="auc",
            callbacks=[early_stopping(stopping_rounds=100, verbose=False)],
        )
    except Exception:
        clf.fit(X_tr, y_tr)

    proba_te = clf.predict_proba(X_te)[:, 1].astype(np.float64)

    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y_te, proba_te)) if len(np.unique(y_te)) == 2 else float("nan")
    except Exception:
        auc = float("nan")

    # Retention / clean-split curve on test segment
    y_sell_te = gate_df["y_sell"].to_numpy(np.float64)[split:]
    qs = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
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
                "sel_median": float(np.median(sel_y)) if sel_y.size else float("nan"),
                "sel_p10": float(np.quantile(sel_y, 0.10)) if sel_y.size else float("nan"),
                "sel_p90": float(np.quantile(sel_y, 0.90)) if sel_y.size else float("nan"),
            }
        )

    # Shifts
    good_mask = gate_df["label_good"].to_numpy(np.int32) == 1
    bad_mask = gate_df["label_bad"].to_numpy(np.int32) == 1
    shifts = feature_shifts(gate_df, good_mask=good_mask, bad_mask=bad_mask, feature_cols=feat_all)

    # Sweeps
    quantiles = [float(x) for x in str(args.sweep_quantiles).split(",") if x.strip()]

    # Candidate features: top shift + all context state features + key snapback signals
    cand = []
    if not shifts.empty:
        cand.extend(shifts.head(60)["feature"].tolist())
    cand.extend(ctx_added)
    cand.extend(
        [
            "ret_1m_pct__min5",
            "ret_1m_pct__max5",
            "mom_3m_pct__min5",
            "mom_5m_pct__min5",
            "vwap_dev_5m__min5",
            "macd__min5",
            "px5__bounce_from_min",
            "px5__drawdown_from_max",
            "px5__pos_in_range",
            "vol_log1p__max5",
            "vol_log1p__min5",
        ]
    )
    cand = [c for c in dict.fromkeys(cand) if c in hv.columns and hv[c].dtype.kind in "fc"]

    sweep1 = _sweep_1d(hv, feature_cols=cand, ret_thresh=ret_thresh, bad_thresh=bad_thresh, quantiles=quantiles)

    conds = _top_conditions_from_1d(
        sweep1,
        top_k=int(args.sweep_topk_per_feature),
        max_cond=int(args.sweep_max_conds),
    )

    sweep2 = _sweep_2d_from_conds(
        hv,
        conds=conds,
        ret_thresh=ret_thresh,
        bad_thresh=bad_thresh,
        min_cov=float(args.sweep_2d_min_cov),
        max_cov=float(args.sweep_2d_max_cov),
        max_pairs=int(args.sweep_2d_max_pairs),
    )

    ts = now_ts()
    out_root = Path(args.out_dir) / f"highvol_sell_goodbad_ctx_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Feature importance
    imp = pd.DataFrame(
        {
            "feature": feat_all,
            "importance_gain": clf.booster_.feature_importance(importance_type="gain"),
            "importance_split": clf.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)

    # Example slices
    ex_n = int(args.examples_n)
    good_ex = hv[hv["label_good"] == 1].sample(n=min(ex_n, int((hv["label_good"] == 1).sum())), random_state=1)
    bad_ex = hv[hv["label_bad"] == 1].sample(n=min(ex_n, int((hv["label_bad"] == 1).sum())), random_state=2)

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
        "ctx_windows": ctx_windows,
        "ctx_added_cols_n": int(len(ctx_added)),
        "sweep1_rows": int(len(sweep1)) if not sweep1.empty else 0,
        "sweep2_rows": int(len(sweep2)) if not sweep2.empty else 0,
    }

    (out_root / f"summary_{ts}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_root / f"hv_proxy_thresholds_{ts}.json").write_text(json.dumps(hv_thr, indent=2) + "\n", encoding="utf-8")

    gate_metrics = {
        "auc_test": float(auc),
        "pos_rate_train": float(np.mean(y_tr)) if y_tr.size else float("nan"),
        "pos_rate_test": float(np.mean(y_te)) if y_te.size else float("nan"),
    }
    (out_root / f"gate_metrics_{ts}.json").write_text(json.dumps(gate_metrics, indent=2) + "\n", encoding="utf-8")

    pd.DataFrame(curve).to_csv(out_root / f"gate_retention_curve_{ts}.csv", index=False)

    imp.to_csv(out_root / f"gate_feature_importance_{ts}.csv", index=False)
    shifts.to_csv(out_root / f"feature_shifts_good_vs_bad_{ts}.csv", index=False)

    if not sweep1.empty:
        sweep1.to_csv(out_root / f"guardrail_sweep_1d_{ts}.csv", index=False)
    if not sweep2.empty:
        sweep2.to_csv(out_root / f"guardrail_sweep_2d_{ts}.csv", index=False)

    good_ex.to_csv(out_root / f"examples_good_{ts}.csv", index=False)
    bad_ex.to_csv(out_root / f"examples_bad_{ts}.csv", index=False)

    # Save the gate dataset (small ~10k) for offline inspection
    gate_df.to_parquet(out_root / f"gate_dataset_{ts}.parquet", index=False)

    print(json.dumps(summary, indent=2))
    print("Top features by gain:")
    print(imp.head(15).to_string(index=False))
    if not sweep2.empty:
        print("Top 2D guardrails:")
        print(
            sweep2.head(10)[
                [
                    "cond1",
                    "cond2",
                    "coverage",
                    "good_rate",
                    "bad_rate",
                    "median_y",
                    "p10_y",
                    "p90_y",
                    "score",
                ]
            ].to_string(index=False)
        )
    elif not sweep1.empty:
        print("Top 1D guardrails:")
        print(
            sweep1.head(10)[
                [
                    "feature",
                    "op",
                    "threshold",
                    "coverage",
                    "good_rate",
                    "bad_rate",
                    "median_y",
                    "p10_y",
                    "p90_y",
                    "score",
                ]
            ].to_string(index=False)
        )
    print("Wrote", out_root)


if __name__ == "__main__":
    main()
