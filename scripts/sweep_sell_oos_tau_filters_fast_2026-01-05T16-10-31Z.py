#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-05T16:10:31Z
"""Lightning-fast sweep: SELL OOS tau × guardrails using a precomputed candidate set.

Key idea
- Do the expensive work ONCE:
  1) build precontext feature frame (same as current SELL backtest)
  2) score entry model once (causal thresholds)
  3) collect candidate entry decision indices in OOS
  4) for each candidate, precompute:
     - exit_model pred_gap curve for k=1..hold_min
     - realized_ret_1x curve for k=1..hold_min
     - first wick liquidation breach offset (assuming constant leverage cap)
     - guardrail features at entry decision (px_close_norm_pct__max5/min5/range5, vol_log1p__last/mean5, vol_std_5m__mean5)

- Then sweeping becomes trivial:
  - filter candidate list by guardrails
  - for each tau pick exit_k from pred_gap curve
  - run bankroll sim by iterating the (filtered) candidates list in time order

This is designed to make tau/filter sweeps extremely fast once the candidate pack is built.

Notes
- This matches the existing SELL OOS backtest conventions:
  - entry decision at bar close i; enter at open i+1
  - exit decision at closes entry_idx+1 .. entry_idx+hold_min
  - liquidation via wick breach on highs for SHORT
  - leverage schedule is capped by --max-leverage; with typical OOS bankroll this stays constant at the cap.

Outputs
- data/backtests/sweep_sell_oos_tau_filters_fast_*_<ts>.csv
- data/backtests/sweep_sell_oos_tau_filters_fast_meta_<ts>.json
- Optional candidate cache: data/backtests/sell_candidates_pack_*_<ts>.npz (+ meta json)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
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


def build_precontext_features(df: pd.DataFrame, *, L: int, base_feat_cols: list[str], ctx_series: dict[str, np.ndarray]) -> Tuple[pd.DataFrame, list[str]]:
    """Build the same precontext feature set used by SELL backtests.

    Convention: each row t uses ONLY the previous L minutes (t-L .. t-1).
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


@dataclass
class BankrollState:
    trading_equity: float
    bank_equity: float

    trade_floor_eur: float = 10.0
    profit_siphon_frac: float = 0.50
    bank_threshold_eur: float = 150.0
    liquidation_recap_bank_frac: float = 0.20

    n_external_topups: int = 0
    n_bank_recaps: int = 0
    total_external_topup: float = 0.0
    total_bank_recap: float = 0.0


def topup_to_floor(bankroll: BankrollState) -> float:
    if bankroll.trading_equity >= bankroll.trade_floor_eur:
        return 0.0
    needed = bankroll.trade_floor_eur - bankroll.trading_equity
    bankroll.trading_equity += needed
    bankroll.total_external_topup += needed
    bankroll.n_external_topups += 1
    return needed


def siphon_profit_to_bank(bankroll: BankrollState, profit: float, leverage: float) -> float:
    if profit <= 0.0:
        return 0.0
    lev = float(leverage)
    if lev not in (50.0, 100.0):
        return 0.0
    amt = profit * bankroll.profit_siphon_frac
    bankroll.trading_equity -= amt
    bankroll.bank_equity += amt
    return amt


def refinance_after_liquidation(bankroll: BankrollState) -> Tuple[float, str]:
    bankroll.trading_equity = 0.0

    if bankroll.bank_equity < bankroll.bank_threshold_eur:
        amt = bankroll.trade_floor_eur
        bankroll.trading_equity = amt
        bankroll.total_external_topup += amt
        bankroll.n_external_topups += 1
        return amt, "external"

    amt = bankroll.bank_equity * bankroll.liquidation_recap_bank_frac
    amt = min(amt, bankroll.bank_equity)
    bankroll.bank_equity -= amt
    bankroll.trading_equity += amt
    bankroll.total_bank_recap += amt
    bankroll.n_bank_recaps += 1
    return amt, "bank"


def _nanquantile(x: np.ndarray, q: float, default: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(default)
    try:
        return float(np.quantile(x, float(q)))
    except Exception:
        return float(default)


def liquidation_price_short(entry_px: float, fee_side: float, leverage: float) -> float:
    if leverage <= 1.0:
        return float("inf")
    target_net_mult = 1.0 - 1.0 / float(leverage)
    return float(entry_px) * (1.0 - float(fee_side)) / (max(1e-12, (1.0 + float(fee_side)) * target_net_mult))


def first_liq_breach_offset_short(high_window: np.ndarray, liq_px: float) -> int:
    """Return first offset in high_window where high >= liq_px, else -1."""
    if not np.isfinite(liq_px):
        return -1
    breach = np.where(high_window >= float(liq_px))[0]
    return int(breach[0]) if breach.size else -1


def pick_exit_k(pred_gap: np.ndarray, *, hold_min: int, min_exit_k: int, tau: float) -> Tuple[int, str]:
    k0 = int(min_exit_k)
    for k in range(k0, int(hold_min) + 1):
        v = float(pred_gap[k - 1])
        if np.isfinite(v) and v <= float(tau):
            return int(k), "pred_gap<=tau"
    return int(hold_min), "hold_min"


def simulate_from_candidates(
    *,
    cand_decision_i: np.ndarray,
    cand_entry_idx: np.ndarray,
    cand_pred_gap: np.ndarray,
    cand_ret_1x: np.ndarray,
    cand_liq_breach_off: np.ndarray,
    tau: float,
    hold_min: int,
    min_exit_k: int,
    fee_side: float,
    trade_floor_eur: float,
    profit_siphon_frac: float,
    bank_threshold_eur: float,
    liquidation_recap_bank_frac: float,
    use_margin_frac: float,
    leverage: float,
) -> Dict[str, Any]:
    bankroll = BankrollState(
        trading_equity=float(trade_floor_eur),
        bank_equity=0.0,
        trade_floor_eur=float(trade_floor_eur),
        profit_siphon_frac=float(profit_siphon_frac),
        bank_threshold_eur=float(bank_threshold_eur),
        liquidation_recap_bank_frac=float(liquidation_recap_bank_frac),
    )

    n = int(cand_decision_i.size)
    free_i = -10**18

    n_trades = 0
    n_liq = 0
    n_exit_pred = 0
    n_exit_hold = 0

    for t in range(n):
        di = int(cand_decision_i[t])
        if di < int(free_i):
            continue

        # bankroll + constant leverage (cap)
        topup_to_floor(bankroll)
        equity_before = float(bankroll.trading_equity)

        margin_eur = float(equity_before) * float(use_margin_frac)
        margin_eur = max(0.0, float(margin_eur))
        notional_eur = float(margin_eur) * float(leverage)

        pred_gap = cand_pred_gap[t]
        exit_k, reason = pick_exit_k(pred_gap, hold_min=int(hold_min), min_exit_k=int(min_exit_k), tau=float(tau))
        if reason == "pred_gap<=tau":
            n_exit_pred += 1
        else:
            n_exit_hold += 1

        entry_idx = int(cand_entry_idx[t])
        exit_idx = int(entry_idx + exit_k)

        # liquidation if breach offset happens before/at exit_k
        off = int(cand_liq_breach_off[t])
        liquidated = (off >= 0) and (off <= int(exit_k))

        if liquidated:
            n_liq += 1
            refinance_after_liquidation(bankroll)
        else:
            r1x = float(cand_ret_1x[t, int(exit_k) - 1])
            profit_eur = float(notional_eur) * (float(r1x) / 100.0)
            bankroll.trading_equity = float(equity_before) + float(profit_eur)
            siphon_profit_to_bank(bankroll, float(profit_eur), float(leverage))
            topup_to_floor(bankroll)

        n_trades += 1
        free_i = int(exit_idx + 1)

    final_total = float(bankroll.trading_equity + bankroll.bank_equity)
    return {
        "n_trades": int(n_trades),
        "n_liquidations": int(n_liq),
        "liq_rate": float(n_liq) / float(n_trades) if n_trades else 0.0,
        "n_external_topups": int(bankroll.n_external_topups),
        "total_external_topup_eur": float(bankroll.total_external_topup),
        "final_total_equity": float(final_total),
        "net_after_external_topups_eur": float(final_total - float(trade_floor_eur) - float(bankroll.total_external_topup)),
        "n_exit_pred_gap": int(n_exit_pred),
        "n_exit_hold_min": int(n_exit_hold),
    }


def build_candidate_pack(
    *,
    bars: pd.DataFrame,
    entry_art: Dict[str, Any],
    exit_art: Dict[str, Any],
    start_iso: str,
    end_iso: Optional[str],
    target_frac: float,
    hold_min: int,
    min_exit_k: int,
    fee_side: float,
    seed_days: int,
    min_prior_scores: int,
    max_prior_scores: int,
    max_leverage: int,
    leverage_safety_frac: float,
    use_margin_frac: float,
    warmup_mins: int,
    slice_prior_scores: int,
    verbose: bool,
) -> Dict[str, Any]:
    bars = bars.copy().sort_values("timestamp").reset_index(drop=True)
    ts_open_full = pd.to_datetime(bars["timestamp"], utc=True)
    ts_arr_full = ts_open_full.to_numpy(dtype="datetime64[ns]")

    start_dt = pd.to_datetime(str(start_iso), utc=True, errors="coerce")
    if pd.isna(start_dt):
        raise SystemExit(f"Invalid --start-iso: {start_iso!r}")
    start_i_oos = int(np.searchsorted(ts_arr_full, start_dt.to_datetime64(), side="left"))

    if end_iso:
        end_dt = pd.to_datetime(str(end_iso), utc=True, errors="coerce")
        if pd.isna(end_dt):
            raise SystemExit(f"Invalid --end-iso: {end_iso!r}")
        end_i_excl = int(np.searchsorted(ts_arr_full, end_dt.to_datetime64(), side="left"))
    else:
        end_i_excl = int(len(ts_arr_full))

    # Slice earlier history so causal thresholds have enough prior score distribution
    lookback = int(max(0, int(slice_prior_scores)))
    start_i_slice = max(0, int(start_i_oos) - int(lookback) - int(warmup_mins) - 10_000)

    sub = bars.iloc[start_i_slice:end_i_excl].copy()
    sub = sub.reset_index(drop=False).rename(columns={"index": "orig_i"})

    ts_open = pd.to_datetime(sub["timestamp"], utc=True)
    dates = ts_open.dt.date.to_numpy()

    if verbose:
        print(f"slice rows: {len(sub):,} (orig_i {start_i_slice}..{end_i_excl})")

    # Build features once
    ctx_series = _compute_ctx_series(sub, [30, 60, 120])
    base_cols = [c for c in BASE_FEATS_DEFAULT if c in sub.columns]
    pre_df, pre_names = build_precontext_features(sub, L=5, base_feat_cols=base_cols, ctx_series=ctx_series)

    entry_model = entry_art["model"]
    entry_features = [str(c) for c in list(entry_art.get("feature_cols") or [])]

    # Score entry model once
    X_df = pre_df[entry_features]
    good = np.isfinite(X_df.to_numpy(np.float64)).all(axis=1)
    good &= (np.arange(len(X_df), dtype=np.int64) >= int(warmup_mins))
    idxs = np.where(good)[0]
    scores_full = np.full((len(sub),), np.nan, dtype=np.float32)
    if idxs.size:
        scores_full[idxs] = score_in_chunks(entry_model, X_df.iloc[idxs].to_numpy(np.float32), chunk=200_000)

    thr_by_day = compute_causal_thresholds(
        dates=dates,
        scores_full=scores_full,
        target_frac=float(target_frac),
        seed_days=int(seed_days),
        min_prior_scores=int(min_prior_scores),
        max_prior_scores=int(max_prior_scores),
    )

    thr_arr = np.asarray([float(thr_by_day.get(d, float("inf"))) for d in dates], dtype=np.float64)

    # OOS mask within sub
    start_mask = (ts_open >= start_dt).to_numpy()
    if end_iso:
        end_dt = pd.to_datetime(str(end_iso), utc=True)
        oos_mask = start_mask & (ts_open < end_dt).to_numpy()
    else:
        oos_mask = start_mask

    # Candidate entry decisions are all indices i where score>=thr(day)
    eligible = oos_mask & np.isfinite(scores_full) & np.isfinite(thr_arr) & (scores_full.astype(np.float64) >= thr_arr)

    cand_i = np.where(eligible)[0].astype(np.int64)

    # Must have entry_idx and full horizon inside sub
    cand_i = cand_i[(cand_i + 1 + int(hold_min) + 1) < len(sub)]

    # IS eligible for fitting guardrail cutoffs
    eligible_is = (~oos_mask) & np.isfinite(scores_full) & np.isfinite(thr_arr) & (scores_full.astype(np.float64) >= thr_arr)

    pre_arr = pre_df.to_numpy(np.float32)
    pre_map = {n: j for j, n in enumerate(pre_names)}

    j_max5 = pre_map.get("px_close_norm_pct__max5", -1)
    j_min5 = pre_map.get("px_close_norm_pct__min5", -1)
    j_px_range5 = pre_map.get("px_close_norm_pct__range5", -1)

    j_vol_last = pre_map.get("vol_log1p__last", -1)
    j_vol_mean5 = pre_map.get("vol_log1p__mean5", -1)

    j_volstd5_mean5 = pre_map.get("vol_std_5m__mean5", -1)

    if j_max5 < 0 or j_min5 < 0 or j_vol_last < 0:
        raise SystemExit("Missing required guardrail features in precontext frame")

    # Optional (only needed for extended guardrails)
    if j_px_range5 < 0:
        j_px_range5 = -1
    if j_vol_mean5 < 0:
        j_vol_mean5 = -1
    if j_volstd5_mean5 < 0:
        j_volstd5_mean5 = -1

    # Extract IS distributions on eligible signal points
    is_idx = np.where(eligible_is)[0].astype(np.int64)
    is_px_max5 = pre_arr[is_idx, j_max5].astype(np.float64)
    is_px_min5 = pre_arr[is_idx, j_min5].astype(np.float64)
    is_vol_last = pre_arr[is_idx, j_vol_last].astype(np.float64)

    is_px_range5 = pre_arr[is_idx, j_px_range5].astype(np.float64) if j_px_range5 >= 0 else np.asarray([], dtype=np.float64)
    is_vol_mean5 = pre_arr[is_idx, j_vol_mean5].astype(np.float64) if j_vol_mean5 >= 0 else np.asarray([], dtype=np.float64)
    is_volstd5_mean5 = pre_arr[is_idx, j_volstd5_mean5].astype(np.float64) if j_volstd5_mean5 >= 0 else np.asarray([], dtype=np.float64)

    # Arrays for per-candidate trade evaluation
    open_arr = pd.to_numeric(sub["open"], errors="coerce").to_numpy(np.float64)
    close_arr = pd.to_numeric(sub["close"], errors="coerce").to_numpy(np.float64)
    high_arr = pd.to_numeric(sub["high"], errors="coerce").to_numpy(np.float64)

    exit_model = exit_art["model"]
    exit_features = [str(c) for c in list(exit_art.get("feature_cols") or [])]

    # Exit feature layout (dynamic vs precontext)
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
    dyn_pos = {n: i for i, n in enumerate(exit_features) if n in dyn_names}
    pre_pos: List[int] = []
    pre_names_exit: List[str] = []
    for j, n in enumerate(exit_features):
        if n not in dyn_names:
            pre_pos.append(int(j))
            pre_names_exit.append(str(n))
    pre_idx_arr = np.asarray([int(pre_map.get(n, -1)) for n in pre_names_exit], dtype=np.int64)
    pre_missing_mask = pre_idx_arr < 0

    # We assume leverage stays at the cap; use lev_eff for liquidation price.
    lev_cap = float(max(1, int(max_leverage)))
    lev_eff = float(lev_cap) * float(leverage_safety_frac)

    N = int(cand_i.size)
    cand_decision_i = cand_i.astype(np.int32)
    cand_entry_idx = (cand_i + 1).astype(np.int32)

    cand_px_max5 = pre_arr[cand_i, j_max5].astype(np.float32)
    cand_px_min5 = pre_arr[cand_i, j_min5].astype(np.float32)
    cand_vol_last = pre_arr[cand_i, j_vol_last].astype(np.float32)

    cand_px_range5 = pre_arr[cand_i, j_px_range5].astype(np.float32) if j_px_range5 >= 0 else np.full((N,), np.nan, dtype=np.float32)
    cand_vol_mean5 = pre_arr[cand_i, j_vol_mean5].astype(np.float32) if j_vol_mean5 >= 0 else np.full((N,), np.nan, dtype=np.float32)
    cand_volstd5_mean5 = pre_arr[cand_i, j_volstd5_mean5].astype(np.float32) if j_volstd5_mean5 >= 0 else np.full((N,), np.nan, dtype=np.float32)

    cand_pred_gap = np.full((N, int(hold_min)), np.nan, dtype=np.float32)
    cand_ret_1x = np.full((N, int(hold_min)), np.nan, dtype=np.float32)
    cand_liq_breach_off = np.full((N,), -1, dtype=np.int16)

    if verbose:
        print(f"OOS candidates (pre-filter): {N}")

    for t in range(N):
        di = int(cand_decision_i[t])
        entry_idx = int(cand_entry_idx[t])

        entry_px = float(open_arr[entry_idx])
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue

        # decision_idx are bar close indices
        decision_idx = np.arange(int(entry_idx + 1), int(entry_idx + hold_min + 1), dtype=np.int64)

        close_seq = close_arr[decision_idx]
        gross_mult = (float(entry_px) / np.maximum(1e-12, close_seq)).astype(np.float64)
        net_mult = gross_mult * (1.0 - float(fee_side)) / (1.0 + float(fee_side))
        cur_ret = (net_mult - 1.0) * 100.0
        cand_ret_1x[t, :] = cur_ret.astype(np.float32)

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

        X_exit = np.empty((m, len(exit_features)), dtype=np.float32)

        # precontext block for these decision minutes
        pre_rows = pre_arr[decision_idx]
        if np.any(pre_missing_mask):
            pre_block = np.full((m, pre_idx_arr.size), np.nan, dtype=np.float32)
            ok = ~pre_missing_mask
            if np.any(ok):
                pre_block[:, ok] = pre_rows[:, pre_idx_arr[ok]]
        else:
            pre_block = pre_rows[:, pre_idx_arr]
        X_exit[:, np.asarray(pre_pos, dtype=np.int64)] = pre_block

        # dynamic
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

        pg = np.asarray(exit_model.predict(X_exit), dtype=np.float64)
        cand_pred_gap[t, :] = pg.astype(np.float32)

        # liquidation breach offset on highs within full horizon
        high_window = high_arr[entry_idx : entry_idx + int(hold_min) + 1]
        liq_px = liquidation_price_short(entry_px, float(fee_side), float(max(1.0, lev_eff)))
        cand_liq_breach_off[t] = int(first_liq_breach_offset_short(high_window, liq_px))

    return {
        "sub": sub,
        "start_iso": str(start_iso),
        "end_iso": (str(end_iso) if end_iso else None),
        "target_frac": float(target_frac),
        "hold_min": int(hold_min),
        "min_exit_k": int(min_exit_k),
        "fee_side": float(fee_side),
        "max_leverage": int(max_leverage),
        "leverage_safety_frac": float(leverage_safety_frac),
        "use_margin_frac": float(use_margin_frac),
        "warmup_mins": int(warmup_mins),
        "slice_prior_scores": int(slice_prior_scores),
        "is_px_max5": is_px_max5,
        "is_px_min5": is_px_min5,
        "is_vol_last": is_vol_last,
        "is_px_range5": is_px_range5,
        "is_vol_mean5": is_vol_mean5,
        "is_volstd5_mean5": is_volstd5_mean5,
        "cand_decision_i": cand_decision_i,
        "cand_entry_idx": cand_entry_idx,
        "cand_px_max5": cand_px_max5,
        "cand_px_min5": cand_px_min5,
        "cand_vol_last": cand_vol_last,
        "cand_px_range5": cand_px_range5,
        "cand_vol_mean5": cand_vol_mean5,
        "cand_volstd5_mean5": cand_volstd5_mean5,
        "cand_pred_gap": cand_pred_gap,
        "cand_ret_1x": cand_ret_1x,
        "cand_liq_breach_off": cand_liq_breach_off,
        "orig_i_offset": int(start_i_slice),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fast sweep: SELL OOS tau × guardrails using precomputed candidates")

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

    ap.add_argument("--start-iso", default="2025-07-31T08:18:00Z")
    ap.add_argument("--end-iso", default=None)

    ap.add_argument("--target-frac", type=float, default=0.0005, help="Top fraction (0.0005 = 0.05%)")
    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--exit-gap-min-exit-k", type=int, default=2)
    ap.add_argument("--fee-side", type=float, default=0.001)

    ap.add_argument("--threshold-seed-days", type=int, default=2)
    ap.add_argument("--threshold-min-prior-scores", type=int, default=2000)
    ap.add_argument("--threshold-max-prior-scores", type=int, default=200_000)

    ap.add_argument("--max-leverage", type=int, default=50)
    ap.add_argument("--leverage-safety-frac", type=float, default=1.0)
    ap.add_argument("--use-margin-frac", type=float, default=1.0)

    ap.add_argument("--trade-floor-eur", type=float, default=10.0)
    ap.add_argument("--profit-siphon-frac", type=float, default=0.50)
    ap.add_argument("--bank-threshold-eur", type=float, default=150.0)
    ap.add_argument("--liquidation-recap-bank-frac", type=float, default=0.20)

    ap.add_argument(
        "--taus",
        default="0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30",
        help="Comma-separated tau values",
    )

    ap.add_argument("--q", default="0.60,0.65,0.70,0.75,0.80,0.85,0.90", help="Comma-separated quantiles for guardrails")
    ap.add_argument("--q-diag-only", action="store_true", help="Only use (q_px=q, q_vol=q) pairs")
    ap.add_argument("--modes", default="BASELINE,DUAL", help="Comma-separated: BASELINE, BUY_LIKE, SELL_REVERSE, DUAL")

    ap.add_argument("--warmup-mins", type=int, default=1440)
    ap.add_argument(
        "--slice-prior-scores",
        type=int,
        default=200_000,
        help="How many minutes of history before OOS to include for causal thresholds (controls runtime).",
    )

    ap.add_argument("--cache-in", default=None, help="Load candidates pack from .npz (skip model scoring)")
    ap.add_argument("--cache-out", default=None, help="Write candidates pack .npz to this path")

    ap.add_argument("--out-dir", default="data/backtests")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    t0 = time.time()
    ts = now_ts()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    taus = [float(x.strip()) for x in str(args.taus).split(",") if x.strip()]
    qs = [float(x.strip()) for x in str(args.q).split(",") if x.strip()]

    want_modes = {m.strip().upper() for m in str(args.modes).split(",") if m.strip()}
    known = {"BASELINE", "BUY_LIKE", "SELL_REVERSE", "DUAL"}
    bad = sorted(list(want_modes - known))
    if bad:
        raise SystemExit(f"Unknown mode(s) in --modes: {bad} (known: {sorted(list(known))})")

    pack: Dict[str, Any]

    if args.cache_in:
        npz = np.load(str(args.cache_in), allow_pickle=False)
        pack = {
            "start_iso": str(npz["start_iso"].item()),
            "end_iso": (str(npz["end_iso"].item()) if npz["end_iso"].item() else None),
            "target_frac": float(npz["target_frac"].item()),
            "hold_min": int(npz["hold_min"].item()),
            "min_exit_k": int(npz["min_exit_k"].item()),
            "fee_side": float(npz["fee_side"].item()),
            "max_leverage": int(npz["max_leverage"].item()),
            "leverage_safety_frac": float(npz["leverage_safety_frac"].item()),
            "use_margin_frac": float(npz["use_margin_frac"].item()),
            "is_px_max5": npz["is_px_max5"],
            "is_px_min5": npz["is_px_min5"],
            "is_vol_last": npz["is_vol_last"],
            "is_px_range5": (npz["is_px_range5"] if "is_px_range5" in npz.files else np.asarray([], dtype=np.float64)),
            "is_vol_mean5": (npz["is_vol_mean5"] if "is_vol_mean5" in npz.files else np.asarray([], dtype=np.float64)),
            "is_volstd5_mean5": (npz["is_volstd5_mean5"] if "is_volstd5_mean5" in npz.files else np.asarray([], dtype=np.float64)),
            "cand_decision_i": npz["cand_decision_i"],
            "cand_entry_idx": npz["cand_entry_idx"],
            "cand_px_max5": npz["cand_px_max5"],
            "cand_px_min5": npz["cand_px_min5"],
            "cand_vol_last": npz["cand_vol_last"],
            "cand_px_range5": (npz["cand_px_range5"] if "cand_px_range5" in npz.files else np.full((npz["cand_decision_i"].size,), np.nan, dtype=np.float32)),
            "cand_vol_mean5": (npz["cand_vol_mean5"] if "cand_vol_mean5" in npz.files else np.full((npz["cand_decision_i"].size,), np.nan, dtype=np.float32)),
            "cand_volstd5_mean5": (npz["cand_volstd5_mean5"] if "cand_volstd5_mean5" in npz.files else np.full((npz["cand_decision_i"].size,), np.nan, dtype=np.float32)),
            "cand_pred_gap": npz["cand_pred_gap"],
            "cand_ret_1x": npz["cand_ret_1x"],
            "cand_liq_breach_off": npz["cand_liq_breach_off"],
        }
        if args.verbose:
            print(f"Loaded candidates cache: {args.cache_in}")
    else:
        bars = pd.read_parquet(Path(args.bars))
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")

        entry_art = joblib.load(Path(args.entry_model))
        exit_art = joblib.load(Path(args.exit_model))

        pack = build_candidate_pack(
            bars=bars,
            entry_art=entry_art,
            exit_art=exit_art,
            start_iso=str(args.start_iso),
            end_iso=(str(args.end_iso) if args.end_iso else None),
            target_frac=float(args.target_frac),
            hold_min=int(args.hold_min),
            min_exit_k=int(args.exit_gap_min_exit_k),
            fee_side=float(args.fee_side),
            seed_days=int(args.threshold_seed_days),
            min_prior_scores=int(args.threshold_min_prior_scores),
            max_prior_scores=int(args.threshold_max_prior_scores),
            max_leverage=int(args.max_leverage),
            leverage_safety_frac=float(args.leverage_safety_frac),
            use_margin_frac=float(args.use_margin_frac),
            warmup_mins=int(args.warmup_mins),
            slice_prior_scores=int(args.slice_prior_scores),
            verbose=bool(args.verbose),
        )

        if args.cache_out:
            cache_path = Path(args.cache_out)
        else:
            cache_path = out_dir / f"sell_candidates_pack_top{int(float(args.target_frac)*100000):05d}ppm_{ts}.npz"

        np.savez_compressed(
            cache_path,
            start_iso=np.asarray(str(pack["start_iso"])),
            end_iso=np.asarray(str(pack["end_iso"] or "")),
            target_frac=np.asarray(float(pack["target_frac"])),
            hold_min=np.asarray(int(pack["hold_min"])),
            min_exit_k=np.asarray(int(pack["min_exit_k"])),
            fee_side=np.asarray(float(pack["fee_side"])),
            max_leverage=np.asarray(int(pack["max_leverage"])),
            leverage_safety_frac=np.asarray(float(pack["leverage_safety_frac"])),
            use_margin_frac=np.asarray(float(pack["use_margin_frac"])),
            is_px_max5=np.asarray(pack["is_px_max5"], dtype=np.float64),
            is_px_min5=np.asarray(pack["is_px_min5"], dtype=np.float64),
            is_vol_last=np.asarray(pack["is_vol_last"], dtype=np.float64),
            is_px_range5=np.asarray(pack.get("is_px_range5", np.asarray([], dtype=np.float64)), dtype=np.float64),
            is_vol_mean5=np.asarray(pack.get("is_vol_mean5", np.asarray([], dtype=np.float64)), dtype=np.float64),
            is_volstd5_mean5=np.asarray(pack.get("is_volstd5_mean5", np.asarray([], dtype=np.float64)), dtype=np.float64),
            cand_decision_i=np.asarray(pack["cand_decision_i"], dtype=np.int32),
            cand_entry_idx=np.asarray(pack["cand_entry_idx"], dtype=np.int32),
            cand_px_max5=np.asarray(pack["cand_px_max5"], dtype=np.float32),
            cand_px_min5=np.asarray(pack["cand_px_min5"], dtype=np.float32),
            cand_vol_last=np.asarray(pack["cand_vol_last"], dtype=np.float32),
            cand_px_range5=np.asarray(pack.get("cand_px_range5"), dtype=np.float32),
            cand_vol_mean5=np.asarray(pack.get("cand_vol_mean5"), dtype=np.float32),
            cand_volstd5_mean5=np.asarray(pack.get("cand_volstd5_mean5"), dtype=np.float32),
            cand_pred_gap=np.asarray(pack["cand_pred_gap"], dtype=np.float32),
            cand_ret_1x=np.asarray(pack["cand_ret_1x"], dtype=np.float32),
            cand_liq_breach_off=np.asarray(pack["cand_liq_breach_off"], dtype=np.int16),
        )

        meta_cache = {
            "created_utc": ts,
            "cache_path": str(cache_path),
            "n_candidates": int(np.asarray(pack["cand_decision_i"]).size),
            "target_frac": float(args.target_frac),
            "start_iso": str(args.start_iso),
            "end_iso": (str(args.end_iso) if args.end_iso else None),
            "slice_prior_scores": int(args.slice_prior_scores),
            "warmup_mins": int(args.warmup_mins),
        }
        cache_meta_path = cache_path.with_suffix(".json")
        cache_meta_path.write_text(json.dumps(meta_cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.verbose:
            print(f"Wrote candidates cache: {cache_path} and {cache_meta_path}")

    # Sweep
    cand_decision_i = np.asarray(pack["cand_decision_i"], dtype=np.int32)
    cand_entry_idx = np.asarray(pack["cand_entry_idx"], dtype=np.int32)
    cand_pred_gap = np.asarray(pack["cand_pred_gap"], dtype=np.float32)
    cand_ret_1x = np.asarray(pack["cand_ret_1x"], dtype=np.float32)
    cand_liq_breach_off = np.asarray(pack["cand_liq_breach_off"], dtype=np.int16)

    cand_px_max5 = np.asarray(pack["cand_px_max5"], dtype=np.float32)
    cand_px_min5 = np.asarray(pack["cand_px_min5"], dtype=np.float32)
    cand_vol_last = np.asarray(pack["cand_vol_last"], dtype=np.float32)

    cand_px_range5 = np.asarray(pack.get("cand_px_range5"), dtype=np.float32)
    cand_vol_mean5 = np.asarray(pack.get("cand_vol_mean5"), dtype=np.float32)
    cand_volstd5_mean5 = np.asarray(pack.get("cand_volstd5_mean5"), dtype=np.float32)

    is_px_max5 = np.asarray(pack["is_px_max5"], dtype=np.float64)
    is_px_min5 = np.asarray(pack["is_px_min5"], dtype=np.float64)
    is_vol_last = np.asarray(pack["is_vol_last"], dtype=np.float64)

    is_px_range5 = np.asarray(pack.get("is_px_range5", np.asarray([], dtype=np.float64)), dtype=np.float64)
    is_vol_mean5 = np.asarray(pack.get("is_vol_mean5", np.asarray([], dtype=np.float64)), dtype=np.float64)
    is_volstd5_mean5 = np.asarray(pack.get("is_volstd5_mean5", np.asarray([], dtype=np.float64)), dtype=np.float64)

    if bool(args.q_diag_only):
        q_pairs = [(q, q) for q in qs]
    else:
        q_pairs = [(q_px, q_vol) for q_px in qs for q_vol in qs]

    # Effective (constant) leverage used for PnL in this fast sim
    leverage = float(max(1, int(args.max_leverage))) * float(args.leverage_safety_frac)

    rows: List[Dict[str, Any]] = []

    # Precompute baseline mask
    all_mask = np.ones((cand_decision_i.size,), dtype=bool)

    for q_px, q_vol in q_pairs:
        max5_cap = _nanquantile(is_px_max5, float(q_px), default=float("inf"))
        min5_floor = _nanquantile(is_px_min5, 1.0 - float(q_px), default=float("-inf"))
        vol_cap = _nanquantile(is_vol_last, float(q_vol), default=float("inf"))

        mask_buy_like = (cand_px_max5.astype(np.float64) <= float(max5_cap)) & (cand_vol_last.astype(np.float64) <= float(vol_cap))
        mask_sell_reverse = (cand_px_min5.astype(np.float64) >= float(min5_floor)) & (cand_vol_last.astype(np.float64) <= float(vol_cap))
        mask_dual = mask_buy_like & mask_sell_reverse

        mode_to_mask = {
            "BASELINE": all_mask,
            "BUY_LIKE": mask_buy_like,
            "SELL_REVERSE": mask_sell_reverse,
            "DUAL": mask_dual,
        }

        for tau in taus:
            for mode in sorted(want_modes):
                m = mode_to_mask[str(mode)]
                idx = np.where(m)[0]

                r = simulate_from_candidates(
                    cand_decision_i=cand_decision_i[idx],
                    cand_entry_idx=cand_entry_idx[idx],
                    cand_pred_gap=cand_pred_gap[idx],
                    cand_ret_1x=cand_ret_1x[idx],
                    cand_liq_breach_off=cand_liq_breach_off[idx],
                    tau=float(tau),
                    hold_min=int(args.hold_min),
                    min_exit_k=int(args.exit_gap_min_exit_k),
                    fee_side=float(args.fee_side),
                    trade_floor_eur=float(args.trade_floor_eur),
                    profit_siphon_frac=float(args.profit_siphon_frac),
                    bank_threshold_eur=float(args.bank_threshold_eur),
                    liquidation_recap_bank_frac=float(args.liquidation_recap_bank_frac),
                    use_margin_frac=float(args.use_margin_frac),
                    leverage=float(leverage),
                )

                rows.append(
                    {
                        "mode": str(mode),
                        "q_px": float(q_px),
                        "q_vol": float(q_vol),
                        "tau": float(tau),
                        "guard_max5_cap": float(max5_cap) if np.isfinite(max5_cap) else None,
                        "guard_min5_floor": float(min5_floor) if np.isfinite(min5_floor) else None,
                        "guard_vol_last_cap": float(vol_cap) if np.isfinite(vol_cap) else None,
                        "n_candidates_after_filter": int(idx.size),
                        **r,
                    }
                )

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["net_after_external_topups_eur", "n_liquidations", "n_trades"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    out_csv = out_dir / f"sweep_sell_oos_tau_filters_fast_top{int(float(args.target_frac)*100000):05d}ppm_{ts}.csv"
    out.to_csv(out_csv, index=False)

    meta = {
        "created_utc": ts,
        "runtime_s": float(time.time() - t0),
        "start_iso": str(args.start_iso),
        "end_iso": (str(args.end_iso) if args.end_iso else None),
        "target_frac": float(args.target_frac),
        "max_leverage": int(args.max_leverage),
        "leverage_safety_frac": float(args.leverage_safety_frac),
        "use_margin_frac": float(args.use_margin_frac),
        "hold_min": int(args.hold_min),
        "exit_gap_min_exit_k": int(args.exit_gap_min_exit_k),
        "fee_side": float(args.fee_side),
        "taus": taus,
        "q": qs,
        "q_diag_only": bool(args.q_diag_only),
        "modes": sorted(list(want_modes)),
        "n_candidates_total": int(cand_decision_i.size),
        "out_csv": str(out_csv),
        "cache_in": (str(args.cache_in) if args.cache_in else None),
    }
    meta_path = out_dir / f"sweep_sell_oos_tau_filters_fast_meta_{ts}.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Wrote:", out_csv)
    print("Meta:", meta_path)
    with pd.option_context("display.max_rows", 20, "display.width", 220):
        cols = [
            "mode",
            "q_px",
            "q_vol",
            "tau",
            "n_candidates_after_filter",
            "n_trades",
            "n_liquidations",
            "liq_rate",
            "total_external_topup_eur",
            "final_total_equity",
            "net_after_external_topups_eur",
            "n_exit_pred_gap",
            "n_exit_hold_min",
        ]
        cols = [c for c in cols if c in out.columns]
        print(out[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
