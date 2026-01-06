#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-05T16:25:55Z
"""Analyze SELL strategy failures at high coverage (e.g., 10% of entries).

Goal
- Run a SELL entry+exit simulation using:
  - entry model + causal per-day thresholding targeting --target-frac (default 10%)
  - exit oracle-gap regressor with (tau, min_exit_k, hold_min)
- Collect trades and focus on BAD trades (realized_ret_1x_pct <= 0).
- Within BAD, focus on WORST30 (bottom 30% by realized_ret_1x_pct).
- Analyze 5-minute precontexts at:
  - entry decision minute (what entry model saw)
  - actual exit decision minute (what exit model saw)
  - oracle-best exit minute (what exit model could have seen if exiting optimally)

Attribution heuristic (entry vs exit)
- For each trade we compute oracle_best_ret_1x_pct = max over k=1..hold_min of ret_if_exit_now_1x_pct.
- If oracle_best_ret_1x_pct <= 0 => "entry-doomed" (no profitable outcome within horizon)
- If oracle_best_ret_1x_pct > 0 and realized_ret_1x_pct <= 0 => "exit-failed" (profit existed but was not captured)

Outputs
- Trades CSV
- Entry precontext separation CSVs
- Exit precontext separation CSVs

Notes
- Uses 1x net returns for attribution (liquidation/leverage is orthogonal). Optionally records wick-liq risk.
- Uses OOS windowing by default (start_iso set to the known OOS boundary), but can be overridden.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _rolling_prev_matrix(x: np.ndarray, L: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    mat = np.full((n, L), np.nan, dtype=np.float64)
    for k in range(1, L + 1):
        mat[k:, L - k] = x[: n - k]
    return mat


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


def _slope_5(y: np.ndarray) -> np.ndarray:
    xc = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    return np.nansum(y * xc[None, :], axis=1) / 10.0


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


def build_precontext_features(df: pd.DataFrame, *, L: int, base_feat_cols: list[str], ctx_series: dict[str, np.ndarray]) -> Tuple[pd.DataFrame, list[str]]:
    """Build the same precontext features used by SELL entry/exit."""
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

    for name, arr in ctx_series.items():
        arr_prev = _rolling_prev_matrix(arr, L)
        out.update(_agg_5(arr_prev, name))

    out["missing_close_n"] = np.sum(~np.isfinite(close_prev), axis=1).astype(np.float64)
    mats = [close_prev]
    for c in base_feat_cols:
        if c in df.columns:
            mats.append(_rolling_prev_matrix(pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64, copy=False), L))
    all_vals = np.column_stack(mats)
    out["missing_any"] = np.any(~np.isfinite(all_vals), axis=1).astype(np.float64)

    df_out = pd.DataFrame(out)
    return df_out, list(df_out.columns)


def score_in_chunks(model: Any, X: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    out = np.full((X.shape[0],), np.nan, dtype=np.float32)
    for s in range(0, X.shape[0], chunk):
        e = min(X.shape[0], s + chunk)
        out[s:e] = model.predict(X[s:e]).astype(np.float32)
    return out


def compute_causal_thresholds(
    *,
    dates: np.ndarray,
    scores_full: np.ndarray,
    target_frac: float,
    seed_days: int,
    min_prior_scores: int,
    max_prior_scores: int,
) -> Dict[object, float]:
    seed_days_n = max(0, int(seed_days))
    min_prior = max(0, int(min_prior_scores))
    max_prior = max(0, int(max_prior_scores))

    unique_days = pd.unique(dates)
    seed_set = set(unique_days[: min(seed_days_n, len(unique_days))]) if seed_days_n > 0 else set()

    def _orderstat_threshold(prior: np.ndarray, frac: float) -> float:
        prior = np.asarray(prior, dtype=np.float64)
        prior = prior[np.isfinite(prior)]
        if prior.size == 0:
            return float("inf")
        k = int(np.floor((1.0 - float(frac)) * float(prior.size - 1)))
        k = max(0, min(int(prior.size - 1), k))
        return float(np.partition(prior, k)[k])

    thr_by_day: Dict[object, float] = {}
    prior_scores: List[float] = []

    n = int(len(dates))
    i0 = 0
    while i0 < n:
        d = dates[i0]
        i1 = i0 + 1
        while i1 < n and dates[i1] == d:
            i1 += 1

        if d in seed_set:
            thr = float("inf")
        elif len(prior_scores) < min_prior or len(prior_scores) == 0:
            thr = float("inf")
        else:
            thr = _orderstat_threshold(np.asarray(prior_scores, dtype=np.float64), float(target_frac))
        thr_by_day[d] = float(thr)

        day_scores = scores_full[i0:i1]
        fin = day_scores[np.isfinite(day_scores)]
        if fin.size:
            prior_scores.extend([float(x) for x in fin])
            if max_prior > 0 and len(prior_scores) > max_prior:
                prior_scores[:] = prior_scores[-max_prior:]

        i0 = i1

    return thr_by_day


def liquidation_price_short(entry_px: float, fee_side: float, leverage: float) -> float:
    if leverage <= 1.0:
        return float("inf")
    target_net_mult = 1.0 - 1.0 / float(leverage)
    return float(entry_px) * (1.0 - float(fee_side)) / (max(1e-12, (1.0 + float(fee_side)) * target_net_mult))


def first_liq_breach_offset_short(high_window: np.ndarray, liq_px: float) -> int:
    if not np.isfinite(liq_px):
        return -1
    breach = np.where(high_window >= float(liq_px))[0]
    return int(breach[0]) if breach.size else -1


def robust_mad(x: np.ndarray) -> float:
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


def precontext_separation(
    *,
    X: np.ndarray,
    feature_names: List[str],
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    pair_name: str,
) -> pd.DataFrame:
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    Xa = X[mask_a]
    Xb = X[mask_b]

    if Xa.size == 0 or Xb.size == 0:
        return pd.DataFrame(columns=["pair", "feature", "median_a", "median_b", "robust_scale", "robust_z", "abs_robust_z", "n_a", "n_b"])  # type: ignore

    scales = np.asarray([robust_mad(X[:, j]) for j in range(X.shape[1])], dtype=np.float64)
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)

    med_a = np.nanmedian(Xa.astype(np.float64), axis=0)
    med_b = np.nanmedian(Xb.astype(np.float64), axis=0)

    z = (med_a - med_b) / scales

    out = pd.DataFrame(
        {
            "pair": pair_name,
            "feature": feature_names,
            "median_a": med_a,
            "median_b": med_b,
            "robust_scale": scales,
            "robust_z": z,
            "abs_robust_z": np.abs(z),
            "n_a": int(mask_a.sum()),
            "n_b": int(mask_b.sum()),
        }
    )
    out = out.sort_values("abs_robust_z", ascending=False).reset_index(drop=True)
    return out


def pick_exit_k(pred_gap: np.ndarray, *, hold_min: int, min_exit_k: int, tau: float) -> Tuple[int, str]:
    k0 = int(min_exit_k)
    for k in range(k0, int(hold_min) + 1):
        v = float(pred_gap[k - 1])
        if np.isfinite(v) and v <= float(tau):
            return int(k), "pred_gap<=tau"
    return int(hold_min), "hold_min"


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze bad trades at high entry coverage (SELL)")

    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
    )
    ap.add_argument(
        "--entry-model",
        default="data/entry_regressor_oracle15m_sell_ctx120_weighted_oracleentry5m_2026-01-04T22-45-20Z/models/entry_regressor_sell_oracle15m_ctx120_weighted_oracleentry5m_2026-01-04T22-45-20Z.joblib",
    )
    ap.add_argument(
        "--exit-model",
        default="data/exit_oracle_sell/exit_oracle_gap_regressor_sell_hold15_top10_2026-01-05T12-27-59Z.joblib",
    )

    ap.add_argument("--target-frac", type=float, default=0.10)
    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--exit-gap-min-exit-k", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.24)
    ap.add_argument("--fee-side", type=float, default=0.001)

    ap.add_argument("--threshold-seed-days", type=int, default=2)
    ap.add_argument("--threshold-min-prior-scores", type=int, default=2000)
    ap.add_argument("--threshold-max-prior-scores", type=int, default=200_000)

    ap.add_argument("--start-iso", default="2025-07-31T08:18:00Z")
    ap.add_argument("--end-iso", default=None)
    ap.add_argument(
        "--slice-prior-mins",
        type=int,
        default=100_000,
        help="How many minutes of history before start_iso to include (speeds up scoring).",
    )
    ap.add_argument("--warmup-mins", type=int, default=1440)

    ap.add_argument("--worst-frac", type=float, default=0.30, help="Fraction of BAD trades to consider WORST")
    ap.add_argument("--oracle-good-thr", type=float, default=0.20, help="Threshold for saying profit existed but exit missed")

    ap.add_argument("--liq-leverage", type=float, default=50.0, help="Only for reporting wick-liq risk; does not change 1x returns")

    ap.add_argument("--out-dir", default="data/backtests")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    t0 = time.time()
    ts = now_ts()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load bars
    bars = pd.read_parquet(Path(args.bars))
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    bars = bars.sort_values("timestamp").reset_index(drop=True)

    ts_open_full = pd.to_datetime(bars["timestamp"], utc=True)
    ts_arr_full = ts_open_full.to_numpy(dtype="datetime64[ns]")

    start_dt = pd.to_datetime(str(args.start_iso), utc=True, errors="coerce") if args.start_iso else None
    if args.start_iso and pd.isna(start_dt):
        raise SystemExit(f"Invalid --start-iso: {args.start_iso!r}")

    if start_dt is not None:
        start_i = int(np.searchsorted(ts_arr_full, start_dt.to_datetime64(), side="left"))
    else:
        start_i = 0

    if args.end_iso:
        end_dt = pd.to_datetime(str(args.end_iso), utc=True, errors="coerce")
        if pd.isna(end_dt):
            raise SystemExit(f"Invalid --end-iso: {args.end_iso!r}")
        end_i_excl = int(np.searchsorted(ts_arr_full, end_dt.to_datetime64(), side="left"))
    else:
        end_i_excl = int(len(ts_arr_full))

    # Slice to speed up scoring
    start_i_slice = max(0, int(start_i) - int(args.slice_prior_mins) - int(args.warmup_mins) - 10_000)
    sub = bars.iloc[start_i_slice:end_i_excl].copy()
    sub = sub.reset_index(drop=False).rename(columns={"index": "orig_i"})

    ts_open = pd.to_datetime(sub["timestamp"], utc=True)
    ts_arr = ts_open.to_numpy(dtype="datetime64[ns]")
    dates = ts_open.dt.date.to_numpy()

    if args.verbose:
        print(f"slice rows: {len(sub):,} (orig_i {start_i_slice}..{end_i_excl})")

    # Build features once
    ctx_series = _compute_ctx_series(sub, [30, 60, 120])
    base_cols = [c for c in BASE_FEATS_DEFAULT if c in sub.columns]
    pre_df, pre_names = build_precontext_features(sub, L=5, base_feat_cols=base_cols, ctx_series=ctx_series)
    pre_arr = pre_df.to_numpy(np.float32)
    pre_map = {n: j for j, n in enumerate(pre_names)}

    # Models
    entry_art = joblib.load(Path(args.entry_model))
    entry_model = entry_art["model"]
    entry_features = [str(c) for c in list(entry_art.get("feature_cols") or [])]

    exit_art = joblib.load(Path(args.exit_model))
    exit_model = exit_art["model"]
    exit_features_all = [str(c) for c in list(exit_art.get("feature_cols") or [])]

    # Exit feature layout
    dyn_names = {
        "mins_in_trade",
        "mins_remaining",
        "delta_mark_pct",
        "delta_mark_prev1_pct",
        "delta_mark_prev2_pct",
        "delta_mark_change_1m",
        "delta_mark_change_2m",
        "delta_mark_change_3m",
        "drawdown_from_peak_pct",
    }
    dyn_pos = {n: i for i, n in enumerate(exit_features_all) if n in dyn_names}
    pre_pos: List[int] = []
    exit_pre_names: List[str] = []
    for j, n in enumerate(exit_features_all):
        if n not in dyn_names:
            pre_pos.append(int(j))
            exit_pre_names.append(str(n))

    pre_idx_arr = np.asarray([int(pre_map.get(n, -1)) for n in exit_pre_names], dtype=np.int64)
    pre_missing_mask = pre_idx_arr < 0

    # Score entry model once
    X_df = pre_df[entry_features]
    good = np.isfinite(X_df.to_numpy(np.float64)).all(axis=1)
    good &= (np.arange(len(X_df), dtype=np.int64) >= int(args.warmup_mins))
    idxs = np.where(good)[0]
    scores_full = np.full((len(sub),), np.nan, dtype=np.float32)
    if idxs.size:
        scores_full[idxs] = score_in_chunks(entry_model, X_df.iloc[idxs].to_numpy(np.float32), chunk=200_000)

    thr_by_day = compute_causal_thresholds(
        dates=dates,
        scores_full=scores_full,
        target_frac=float(args.target_frac),
        seed_days=int(args.threshold_seed_days),
        min_prior_scores=int(args.threshold_min_prior_scores),
        max_prior_scores=int(args.threshold_max_prior_scores),
    )

    # Determine OOS window indices within sub
    if start_dt is not None:
        start_i_sub = int(np.searchsorted(ts_arr, start_dt.to_datetime64(), side="left"))
    else:
        start_i_sub = 0
    end_i_sub_excl = int(len(sub))
    if args.end_iso:
        end_i_sub_excl = int(np.searchsorted(ts_arr, pd.to_datetime(str(args.end_iso), utc=True).to_datetime64(), side="left"))

    open_arr = pd.to_numeric(sub["open"], errors="coerce").to_numpy(np.float64)
    close_arr = pd.to_numeric(sub["close"], errors="coerce").to_numpy(np.float64)
    high_arr = pd.to_numeric(sub["high"], errors="coerce").to_numpy(np.float64)

    hold_min = int(args.hold_min)

    trades: List[Dict[str, Any]] = []

    entry_feat_rows: List[np.ndarray] = []  # entry model features at entry decision
    exit_feat_rows: List[np.ndarray] = []  # exit model precontext features at actual exit decision
    exit_oracle_feat_rows: List[np.ndarray] = []  # exit model precontext features at oracle-best exit decision

    exit_reason_list: List[str] = []

    # Scan sequentially: one trade at a time
    i = int(max(start_i_sub, int(args.warmup_mins)))

    while i < int(end_i_sub_excl):
        s_i = float(scores_full[i])
        if not np.isfinite(s_i):
            i += 1
            continue

        thr_now = float(thr_by_day.get(dates[i], float("inf")))
        if not np.isfinite(thr_now) or s_i < thr_now:
            i += 1
            continue

        entry_decision_i = int(i)
        entry_idx = int(i + 1)
        if int(entry_idx + hold_min + 1) >= int(end_i_sub_excl):
            break

        entry_px = float(open_arr[entry_idx])
        if not (np.isfinite(entry_px) and entry_px > 0.0):
            i += 1
            continue

        # Exit decisions at closes entry_idx+1 .. entry_idx+hold_min
        decision_idx = np.arange(int(entry_idx + 1), int(entry_idx + hold_min + 1), dtype=np.int64)
        close_seq = close_arr[decision_idx]

        gross_mult = (float(entry_px) / np.maximum(1e-12, close_seq)).astype(np.float64)
        net_mult = gross_mult * (1.0 - float(args.fee_side)) / (1.0 + float(args.fee_side))
        cur_ret = (net_mult - 1.0) * 100.0  # shape (hold_min,)

        # Build exit feature matrix and score
        m = int(cur_ret.size)
        k = (np.arange(m, dtype=np.float32) + 1.0)
        mins_remaining = float(hold_min) - k

        prev1 = np.roll(cur_ret, 1)
        prev2 = np.roll(cur_ret, 2)
        prev3 = np.roll(cur_ret, 3)
        prev1[0] = np.nan
        prev2[:2] = np.nan
        prev3[:3] = np.nan

        peak = np.maximum.accumulate(cur_ret)
        drawdown = cur_ret - peak

        X_exit = np.empty((m, len(exit_features_all)), dtype=np.float32)

        # Fill precontext block
        pre_rows = pre_arr[decision_idx]
        if np.any(pre_missing_mask):
            pre_block = np.full((m, pre_idx_arr.size), np.nan, dtype=np.float32)
            ok = ~pre_missing_mask
            if np.any(ok):
                pre_block[:, ok] = pre_rows[:, pre_idx_arr[ok]]
        else:
            pre_block = pre_rows[:, pre_idx_arr]
        X_exit[:, np.asarray(pre_pos, dtype=np.int64)] = pre_block

        # Dynamic
        if "mins_in_trade" in dyn_pos:
            X_exit[:, dyn_pos["mins_in_trade"]] = k
        if "mins_remaining" in dyn_pos:
            X_exit[:, dyn_pos["mins_remaining"]] = mins_remaining
        if "delta_mark_pct" in dyn_pos:
            X_exit[:, dyn_pos["delta_mark_pct"]] = cur_ret.astype(np.float32)
        if "delta_mark_prev1_pct" in dyn_pos:
            X_exit[:, dyn_pos["delta_mark_prev1_pct"]] = prev1.astype(np.float32)
        if "delta_mark_prev2_pct" in dyn_pos:
            X_exit[:, dyn_pos["delta_mark_prev2_pct"]] = prev2.astype(np.float32)
        if "delta_mark_change_1m" in dyn_pos:
            X_exit[:, dyn_pos["delta_mark_change_1m"]] = (cur_ret - prev1).astype(np.float32)
        if "delta_mark_change_2m" in dyn_pos:
            X_exit[:, dyn_pos["delta_mark_change_2m"]] = (cur_ret - prev2).astype(np.float32)
        if "delta_mark_change_3m" in dyn_pos:
            X_exit[:, dyn_pos["delta_mark_change_3m"]] = (cur_ret - prev3).astype(np.float32)
        if "drawdown_from_peak_pct" in dyn_pos:
            X_exit[:, dyn_pos["drawdown_from_peak_pct"]] = drawdown.astype(np.float32)

        pred_gap = np.asarray(exit_model.predict(X_exit), dtype=np.float64)

        exit_k, exit_reason = pick_exit_k(pred_gap, hold_min=hold_min, min_exit_k=int(args.exit_gap_min_exit_k), tau=float(args.tau))
        exit_decision_i = int(entry_idx + exit_k)

        # Oracle best within horizon
        oracle_k = int(np.nanargmax(cur_ret) + 1)
        oracle_decision_i = int(entry_idx + oracle_k)

        realized_ret_1x = float(cur_ret[int(exit_k) - 1])
        oracle_best_ret_1x = float(cur_ret[int(oracle_k) - 1])

        # Wick liquidation risk (report-only)
        liq_px = liquidation_price_short(entry_px, float(args.fee_side), float(max(1.0, float(args.liq_leverage))))
        high_window = high_arr[entry_idx : (entry_idx + hold_min + 1)]
        liq_off = first_liq_breach_offset_short(high_window, liq_px)
        liq_within_exit = int(liq_off >= 0 and liq_off <= int(exit_k))

        # Store precontext vectors for analysis
        entry_feat_rows.append(X_df.iloc[entry_decision_i].to_numpy(np.float32))

        def _gather_pre(row: np.ndarray) -> np.ndarray:
            if not np.any(pre_missing_mask):
                return row[pre_idx_arr].astype(np.float32, copy=False)
            outv = np.full((pre_idx_arr.size,), np.nan, dtype=np.float32)
            ok = ~pre_missing_mask
            if np.any(ok):
                outv[ok] = row[pre_idx_arr[ok]].astype(np.float32, copy=False)
            return outv

        # Exit precontext (only) at actual exit decision minute
        exit_pre_act = pre_arr[exit_decision_i]
        exit_feat_rows.append(_gather_pre(exit_pre_act))

        # Exit precontext at oracle-best minute
        exit_pre_or = pre_arr[oracle_decision_i]
        exit_oracle_feat_rows.append(_gather_pre(exit_pre_or))

        exit_reason_list.append(str(exit_reason))

        trades.append(
            {
                "entry_time_open_utc": pd.to_datetime(ts_open.iloc[entry_idx], utc=True).isoformat().replace("+00:00", "Z"),
                "exit_time_close_utc": (pd.to_datetime(ts_open.iloc[exit_decision_i], utc=True) + pd.Timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
                "entry_decision_i": int(entry_decision_i),
                "entry_idx": int(entry_idx),
                "exit_decision_i": int(exit_decision_i),
                "exit_rel_min": int(exit_k),
                "exit_reason": str(exit_reason),
                "oracle_rel_min": int(oracle_k),
                "entry_price": float(entry_px),
                "entry_score": float(s_i),
                "entry_threshold": float(thr_now),
                "realized_ret_1x_pct": float(realized_ret_1x),
                "oracle_best_ret_1x_pct": float(oracle_best_ret_1x),
                "oracle_gap_to_realized_pct": float(oracle_best_ret_1x - realized_ret_1x),
                "liq_breach_off_look15": int(liq_off),
                "liq_within_exit": int(liq_within_exit),
            }
        )

        # move time forward to after exit
        i = int(exit_decision_i + 1)

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        raise SystemExit("No trades produced")

    # Build matrices
    entry_X = np.vstack(entry_feat_rows).astype(np.float32)
    exit_X = np.vstack(exit_feat_rows).astype(np.float32)
    exit_or_X = np.vstack(exit_oracle_feat_rows).astype(np.float32)

    # Masks
    realized = pd.to_numeric(trades_df["realized_ret_1x_pct"], errors="coerce").to_numpy(np.float64)
    oracle_best = pd.to_numeric(trades_df["oracle_best_ret_1x_pct"], errors="coerce").to_numpy(np.float64)

    good = realized > 0.0
    bad = realized <= 0.0

    # Worst-frac among bad (bottom by realized)
    bad_vals = realized[bad]
    if bad_vals.size:
        thr_worst = float(np.quantile(bad_vals, float(args.worst_frac)))
    else:
        thr_worst = float("nan")
    worst = bad & (realized <= thr_worst)

    entry_doomed = bad & (oracle_best <= 0.0)
    exit_failed = bad & (oracle_best > 0.0)
    exit_failed_strong = bad & (oracle_best >= float(args.oracle_good_thr))

    # Summary
    summ = {
        "created_utc": ts,
        "target_frac": float(args.target_frac),
        "tau": float(args.tau),
        "hold_min": int(args.hold_min),
        "min_exit_k": int(args.exit_gap_min_exit_k),
        "start_iso": (str(args.start_iso) if args.start_iso else None),
        "end_iso": (str(args.end_iso) if args.end_iso else None),
        "slice_prior_mins": int(args.slice_prior_mins),
        "n_trades": int(len(trades_df)),
        "n_bad": int(bad.sum()),
        "bad_rate": float(bad.mean()),
        "n_worst": int(worst.sum()),
        "worst_frac_of_bad": float(worst.sum()) / float(bad.sum()) if bad.sum() else 0.0,
        "worst_thr_realized_1x_pct": float(thr_worst),
        "n_entry_doomed": int(entry_doomed.sum()),
        "n_exit_failed": int(exit_failed.sum()),
        "n_exit_failed_strong": int(exit_failed_strong.sum()),
        "oracle_good_thr": float(args.oracle_good_thr),
        "mean_realized_1x_bad": float(np.mean(realized[bad])) if bad.sum() else float("nan"),
        "mean_oracle_best_1x_bad": float(np.mean(oracle_best[bad])) if bad.sum() else float("nan"),
        "mean_oracle_gap_bad": float(np.mean((oracle_best - realized)[bad])) if bad.sum() else float("nan"),
        "exit_reason_counts": trades_df["exit_reason"].value_counts(dropna=False).to_dict(),
        "liq_within_exit_rate": float(pd.to_numeric(trades_df["liq_within_exit"], errors="coerce").fillna(0).mean()),
    }

    out_root = out_dir / f"analysis_sell_badtrades_top{int(args.target_frac*100):02d}pct_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    trades_path = out_root / "trades.csv"
    trades_df.to_csv(trades_path, index=False)

    # Separations (entry)
    entry_sep_bad_vs_good = precontext_separation(
        X=entry_X,
        feature_names=entry_features,
        mask_a=bad,
        mask_b=good,
        pair_name="ENTRY_BAD_minus_GOOD",
    )
    entry_sep_worst_vs_restbad = precontext_separation(
        X=entry_X,
        feature_names=entry_features,
        mask_a=worst,
        mask_b=(bad & ~worst),
        pair_name="ENTRY_WORST_minus_REST_BAD",
    )

    # Separations (exit precontext at actual exit)
    exit_sep_bad_vs_good = precontext_separation(
        X=exit_X,
        feature_names=exit_pre_names,
        mask_a=bad,
        mask_b=good,
        pair_name="EXIT_BAD_minus_GOOD",
    )
    exit_sep_worst_vs_restbad = precontext_separation(
        X=exit_X,
        feature_names=exit_pre_names,
        mask_a=worst,
        mask_b=(bad & ~worst),
        pair_name="EXIT_WORST_minus_REST_BAD",
    )

    # Exit oracle-vs-actual (only for exit_failed_strong):
    # Compare (oracle_precontext - actual_exit_precontext) in exit-failed trades vs GOOD trades.
    D = (exit_or_X - exit_X).astype(np.float32)
    exit_fail_mask = exit_failed_strong
    if exit_fail_mask.sum() and good.sum():
        exit_or_minus_act = precontext_separation(
            X=D,
            feature_names=[f"delta_oracle_minus_actual__{n}" for n in exit_pre_names],
            mask_a=exit_fail_mask,
            mask_b=good,
            pair_name="EXIT_FAIL_STRONG_DELTA_ORACLE_MINUS_ACTUAL_vs_GOOD",
        )
    else:
        exit_or_minus_act = pd.DataFrame(columns=["pair", "feature", "median_a", "median_b", "robust_scale", "robust_z", "abs_robust_z", "n_a", "n_b"])  # type: ignore

    entry_sep_bad_vs_good.to_csv(out_root / "entry_precontext_bad_vs_good.csv", index=False)
    entry_sep_worst_vs_restbad.to_csv(out_root / "entry_precontext_worst_vs_restbad.csv", index=False)

    exit_sep_bad_vs_good.to_csv(out_root / "exit_precontext_bad_vs_good.csv", index=False)
    exit_sep_worst_vs_restbad.to_csv(out_root / "exit_precontext_worst_vs_restbad.csv", index=False)

    if not exit_or_minus_act.empty:
        exit_or_minus_act.to_csv(out_root / "exit_oracle_minus_actual_precontext_delta_exitfailedstrong.csv", index=False)

    (out_root / "summary.json").write_text(json.dumps(summ, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Wrote:")
    print(" ", out_root)
    print("Summary:")
    for k in [
        "n_trades",
        "n_bad",
        "bad_rate",
        "n_worst",
        "worst_thr_realized_1x_pct",
        "n_entry_doomed",
        "n_exit_failed",
        "n_exit_failed_strong",
        "liq_within_exit_rate",
    ]:
        if k in summ:
            print(f"  {k}: {summ[k]}")
    print(f"runtime_s: {time.time() - t0:.2f}")


if __name__ == "__main__":
    main()
