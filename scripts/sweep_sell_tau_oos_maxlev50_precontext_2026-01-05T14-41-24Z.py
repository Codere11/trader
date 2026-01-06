#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-05T14:41:24Z
"""Sweep exit-gap tau for SELL entry+exit strategy on OOS window (no future leakage).

What this does
- Loads ETH-USD 1m dataset (with standard base features).
- Builds causal entry feature frame (5m precontext + ctx windows) once.
- Scores the entry model once.
- Computes causal per-day thresholds (prior-days-only) for target_frac=1%.
- Runs a fast tau sweep for the exit-gap policy with MAX LEVERAGE capped (e.g. 50x).
- Writes:
  - summary CSV of tau sweep
  - trades CSV for the best tau (by net after external topups)
  - precontext analysis CSV of feature separations across outcome buckets

Notes
- Exit model sees profitability delta (delta_mark_pct) and its short history.
- No future leakage in thresholding: thresholds computed day-by-day from prior scores only.
- OOS-only windowing: trades must fully fit inside [start_iso, end_iso).
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


def net_mult_sell(entry_px: float, exit_px: float, fee_side: float) -> float:
    entry_px = max(1e-12, float(entry_px))
    exit_px = max(1e-12, float(exit_px))
    gross_mult = entry_px / exit_px
    return float(gross_mult) * (1.0 - float(fee_side)) / (1.0 + float(fee_side))


def net_return_pct_sell(entry_px: float, exit_px: float, fee_side: float) -> float:
    return float((net_mult_sell(entry_px, exit_px, fee_side) - 1.0) * 100.0)


def liquidation_price_short(entry_px: float, fee_side: float, leverage: float) -> float:
    if leverage <= 1.0:
        return float("inf")
    target_net_mult = 1.0 - 1.0 / float(leverage)
    return float(entry_px) * (1.0 - float(fee_side)) / (max(1e-12, (1.0 + float(fee_side)) * target_net_mult))


def check_wick_liquidation_short(entry_px: float, high_window: np.ndarray, fee_side: float, leverage: float) -> Tuple[bool, int]:
    if leverage <= 1.0:
        return False, -1
    liq_px = liquidation_price_short(entry_px, fee_side, leverage)
    breach = np.where(high_window >= liq_px)[0]
    if breach.size == 0:
        return False, -1
    return True, int(breach[0])


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

    # Missing indicators expected by entry model.
    out["missing_close_n"] = np.sum(~np.isfinite(close_prev), axis=1).astype(np.float64)
    mats = [close_prev]
    for c in base_feat_cols:
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


def get_current_leverage(total_equity: float) -> float:
    if total_equity < 100_000:
        return 100.0
    if total_equity < 200_000:
        return 50.0
    if total_equity < 500_000:
        return 20.0
    if total_equity < 1_000_000:
        return 10.0
    return 1.0


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


def run_one_tau(
    *,
    tau: float,
    bars: pd.DataFrame,
    ts_open: pd.Series,
    dates: np.ndarray,
    scores_full: np.ndarray,
    thr_by_day: Dict[object, float],
    exit_model: Any,
    exit_features: List[str],
    exit_precontext_arr: np.ndarray,
    exit_precontext_map: Dict[str, int],
    start_iso: str,
    end_iso: Optional[str],
    hold_min: int,
    exit_gap_min_exit_k: int,
    fee_side: float,
    trade_floor_eur: float,
    profit_siphon_frac: float,
    bank_threshold_eur: float,
    liquidation_recap_bank_frac: float,
    use_margin_frac: float,
    leverage_safety_frac: float,
    max_leverage: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ts_arr = ts_open.to_numpy(dtype="datetime64[ns]")
    start_dt = pd.to_datetime(str(start_iso), utc=True, errors="coerce")
    if pd.isna(start_dt):
        raise SystemExit(f"Invalid start_iso: {start_iso!r}")
    start_i = int(np.searchsorted(ts_arr, start_dt.to_datetime64(), side="left"))

    if end_iso:
        end_dt = pd.to_datetime(str(end_iso), utc=True, errors="coerce")
        if pd.isna(end_dt):
            raise SystemExit(f"Invalid end_iso: {end_iso!r}")
        end_i_excl = int(np.searchsorted(ts_arr, end_dt.to_datetime64(), side="left"))
    else:
        end_i_excl = int(len(ts_arr))

    open_arr = pd.to_numeric(bars["open"], errors="coerce").to_numpy(np.float64)
    high_arr = pd.to_numeric(bars["high"], errors="coerce").to_numpy(np.float64)
    close_arr = pd.to_numeric(bars["close"], errors="coerce").to_numpy(np.float64)

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
    pre_names: List[str] = []
    for j, n in enumerate(exit_features):
        if n not in dyn_names:
            pre_pos.append(int(j))
            pre_names.append(str(n))
    pre_idx_arr = np.asarray([int(exit_precontext_map.get(n, -1)) for n in pre_names], dtype=np.int64)
    pre_missing_mask = pre_idx_arr < 0

    bankroll = BankrollState(
        trading_equity=float(trade_floor_eur),
        bank_equity=0.0,
        trade_floor_eur=float(trade_floor_eur),
        profit_siphon_frac=float(profit_siphon_frac),
        bank_threshold_eur=float(bank_threshold_eur),
        liquidation_recap_bank_frac=float(liquidation_recap_bank_frac),
    )

    trades: List[Dict[str, Any]] = []
    i = int(start_i)
    last_log = time.time()

    while i < int(end_i_excl):
        if (time.time() - last_log) > 20.0:
            last_log = time.time()

        s_i = float(scores_full[i])
        if not np.isfinite(s_i):
            i += 1
            continue

        thr_now = float(thr_by_day.get(dates[i], float("inf")))
        if not np.isfinite(thr_now) or s_i < thr_now:
            i += 1
            continue

        entry_decision_idx = int(i)
        entry_idx = int(i + 1)
        if entry_idx >= len(close_arr):
            break
        if int(entry_idx + hold_min + 1) > int(end_i_excl):
            break

        # bankroll and leverage
        topup_pre = topup_to_floor(bankroll)
        equity_before = float(bankroll.trading_equity)
        total_equity_before = float(bankroll.trading_equity + bankroll.bank_equity)

        lev_sched = float(get_current_leverage(total_equity_before))
        lev_cap = float(max(1, int(max_leverage)))
        leverage = float(min(lev_sched, lev_cap))
        lev_eff = float(leverage) * float(leverage_safety_frac)

        entry_px = float(open_arr[entry_idx])
        if not np.isfinite(entry_px) or entry_px <= 0:
            i += 1
            continue

        margin_eur = float(equity_before) * float(use_margin_frac)
        margin_eur = max(0.0, float(margin_eur))
        notional_eur = float(margin_eur) * float(lev_eff)

        decision_idx = np.arange(int(entry_idx + 1), int(entry_idx + hold_min + 1), dtype=np.int64)

        # profitability delta path (SELL)
        close_seq = close_arr[decision_idx]
        gross_mult = (float(entry_px) / np.maximum(1e-12, close_seq)).astype(np.float64)
        net_mult = gross_mult * (1.0 - float(fee_side)) / (1.0 + float(fee_side))
        cur_ret = (net_mult - 1.0) * 100.0

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

        pre_rows = exit_precontext_arr[decision_idx]
        if np.any(pre_missing_mask):
            pre_block = np.full((m, pre_idx_arr.size), np.nan, dtype=np.float32)
            ok = ~pre_missing_mask
            if np.any(ok):
                pre_block[:, ok] = pre_rows[:, pre_idx_arr[ok]]
        else:
            pre_block = pre_rows[:, pre_idx_arr]
        X_exit[:, np.asarray(pre_pos, dtype=np.int64)] = pre_block

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
        eligible = (k >= float(exit_gap_min_exit_k)) & np.isfinite(pred_gap) & (pred_gap <= float(tau))

        if np.any(eligible):
            j = int(np.where(eligible)[0][0])
            exit_reason = "pred_gap<=tau"
        else:
            j = int(m - 1)
            exit_reason = "hold_min"

        exit_idx = int(decision_idx[j])
        exit_px = float(close_arr[exit_idx])
        realized_ret_1x_pct = float(cur_ret[j])
        exit_pred_gap = float(pred_gap[j]) if np.isfinite(pred_gap[j]) else np.nan

        entry_time_open = pd.to_datetime(ts_open.iloc[entry_idx], utc=True).to_pydatetime()
        exit_time_close = pd.to_datetime(ts_open.iloc[exit_idx], utc=True).to_pydatetime() + pd.Timedelta(minutes=1)

        # liquidation check
        high_window = high_arr[entry_idx : exit_idx + 1]
        liq, liq_k = check_wick_liquidation_short(entry_px, high_window, float(fee_side), float(max(1.0, lev_eff)))
        liquidated = bool(liq)
        liq_source = ""

        if liquidated:
            realized_ret_1x_pct = -100.0
            profit_eur = -equity_before
            recap_amt, recap_source = refinance_after_liquidation(bankroll)
            topup_post = float(recap_amt)
            liq_source = str(recap_source)
            siphon_amt = 0.0
        else:
            profit_eur = float(notional_eur) * (float(realized_ret_1x_pct) / 100.0)
            bankroll.trading_equity = float(equity_before) + float(profit_eur)
            siphon_amt = siphon_profit_to_bank(bankroll, float(profit_eur), float(leverage))
            topup_post = topup_to_floor(bankroll)

        trades.append(
            {
                "tau": float(tau),
                "entry_time_open_utc": entry_time_open.isoformat().replace("+00:00", "Z"),
                "exit_time_close_utc": exit_time_close.isoformat().replace("+00:00", "Z"),
                "entry_decision_idx": int(entry_decision_idx),
                "entry_idx": int(entry_idx),
                "exit_idx": int(exit_idx),
                "exit_rel_min": int(j + 1),
                "entry_price": float(entry_px),
                "exit_price": float(exit_px),
                "realized_ret_1x_pct": float(realized_ret_1x_pct),
                "exit_reason": str(exit_reason),
                "exit_pred_gap_pct": float(exit_pred_gap),
                "exit_gap_min_exit_k": int(exit_gap_min_exit_k),
                "entry_score": float(s_i),
                "entry_threshold": float(thr_now),
                "leverage": float(leverage),
                "lev_eff": float(lev_eff),
                "profit_eur": float(profit_eur),
                "siphon_eur": float(siphon_amt),
                "topup_pre_eur": float(topup_pre),
                "topup_post_eur": float(topup_post),
                "trading_equity_end_eur": float(bankroll.trading_equity),
                "bank_equity_end_eur": float(bankroll.bank_equity),
                "total_equity_end_eur": float(bankroll.trading_equity + bankroll.bank_equity),
                "liquidated": int(liquidated),
                "liquidation_source": str(liq_source),
                "liquidation_rel_min": int(liq_k) if liquidated else -1,
            }
        )

        i = int(exit_idx + 1)

    df = pd.DataFrame(trades)
    if df.empty:
        summary = {
            "tau": float(tau),
            "n_trades": 0,
            "n_liquidations": 0,
            "n_external_topups": int(bankroll.n_external_topups),
            "total_external_topup_eur": float(bankroll.total_external_topup),
            "final_total_equity": float(bankroll.trading_equity + bankroll.bank_equity),
            "net_after_external_topups_eur": float((bankroll.trading_equity + bankroll.bank_equity) - float(trade_floor_eur) - float(bankroll.total_external_topup)),
        }
        return df, summary

    win = df["realized_ret_1x_pct"] >= 0.0
    summary = {
        "tau": float(tau),
        "n_trades": int(len(df)),
        "win_rate": float(win.mean()),
        "mean_ret_1x_pct": float(df["realized_ret_1x_pct"].mean()),
        "median_ret_1x_pct": float(df["realized_ret_1x_pct"].median()),
        "n_liquidations": int((df["liquidated"] == 1).sum()),
        "liq_rate": float((df["liquidated"] == 1).mean()),
        "n_external_topups": int(bankroll.n_external_topups),
        "n_bank_recaps": int(bankroll.n_bank_recaps),
        "total_external_topup_eur": float(bankroll.total_external_topup),
        "total_bank_recap_eur": float(bankroll.total_bank_recap),
        "final_trading_equity": float(bankroll.trading_equity),
        "final_bank_equity": float(bankroll.bank_equity),
        "final_total_equity": float(bankroll.trading_equity + bankroll.bank_equity),
        "net_after_external_topups_eur": float((bankroll.trading_equity + bankroll.bank_equity) - float(trade_floor_eur) - float(bankroll.total_external_topup)),
    }

    return df, summary


def analyze_precontext(
    *,
    trades: pd.DataFrame,
    precontext_df: pd.DataFrame,
    feature_cols: List[str],
    out_csv: Path,
) -> None:
    """Compare 5m precontext features across outcome buckets."""
    t = trades.copy()
    t["realized_ret_1x_pct"] = pd.to_numeric(t["realized_ret_1x_pct"], errors="coerce")
    t["liquidated"] = pd.to_numeric(t["liquidated"], errors="coerce").fillna(0).astype(int)

    # Buckets
    t["bucket"] = ""
    t.loc[t["liquidated"] == 1, "bucket"] = "LIQ"

    nonliq = t["liquidated"] == 0
    t.loc[nonliq & (t["realized_ret_1x_pct"] >= 0) & (t["realized_ret_1x_pct"] < 0.5), "bucket"] = "WIN_SMALL"
    t.loc[nonliq & (t["realized_ret_1x_pct"] >= 0.5), "bucket"] = "WIN_BIG"

    t.loc[nonliq & (t["realized_ret_1x_pct"] < 0) & (t["realized_ret_1x_pct"] > -0.5), "bucket"] = "LOSS_SMALL"
    t.loc[nonliq & (t["realized_ret_1x_pct"] <= -0.5), "bucket"] = "LOSS_BIG"

    # Only keep assigned
    t = t[t["bucket"] != ""].copy()
    if t.empty:
        return

    # Join precontext rows at entry_decision_idx (t-5..t-1 relative to entry decision)
    idx = pd.to_numeric(t["entry_decision_idx"], errors="coerce").astype(int)
    X = precontext_df.iloc[idx][feature_cols].copy()

    # robust scales computed from all involved rows
    scales = {c: robust_mad(pd.to_numeric(X[c], errors="coerce").to_numpy(np.float64)) for c in feature_cols}

    rows: List[Dict[str, Any]] = []
    buckets = ["WIN_BIG", "WIN_SMALL", "LOSS_SMALL", "LOSS_BIG", "LIQ"]

    med_by = {}
    for b in buckets:
        sel = (t["bucket"] == b).to_numpy()
        if sel.sum() == 0:
            continue
        med = X.loc[sel].median(numeric_only=True)
        med_by[b] = med

    def add_pair(a: str, b: str) -> None:
        if a not in med_by or b not in med_by:
            return
        da = med_by[a]
        db = med_by[b]
        for c in feature_cols:
            s = float(scales.get(c, 1.0) or 1.0)
            if not np.isfinite(s) or s <= 0:
                s = 1.0
            ra = float(da.get(c, np.nan))
            rb = float(db.get(c, np.nan))
            z = (ra - rb) / s if np.isfinite(ra) and np.isfinite(rb) else np.nan
            rows.append({
                "pair": f"{a}_minus_{b}",
                "feature": c,
                "median_a": ra,
                "median_b": rb,
                "robust_scale": s,
                "robust_z": z,
                "abs_robust_z": abs(z) if np.isfinite(z) else np.nan,
                "n_a": int((t["bucket"] == a).sum()),
                "n_b": int((t["bucket"] == b).sum()),
            })

    # comparisons of interest
    add_pair("WIN_BIG", "WIN_SMALL")
    add_pair("WIN_BIG", "LOSS_BIG")
    add_pair("WIN_BIG", "LIQ")
    add_pair("WIN_SMALL", "LOSS_BIG")
    add_pair("WIN_SMALL", "LIQ")

    out = pd.DataFrame(rows)
    if out.empty:
        return

    out = out.sort_values(["pair", "abs_robust_z"], ascending=[True, False]).reset_index(drop=True)
    out.to_csv(out_csv, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep SELL exit tau on OOS-only with max leverage cap; analyze 5m precontexts")

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

    ap.add_argument("--target-frac", type=float, default=0.01)
    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--exit-gap-min-exit-k", type=int, default=2)
    ap.add_argument("--fee-side", type=float, default=0.001)

    ap.add_argument("--threshold-seed-days", type=int, default=2)
    ap.add_argument("--threshold-min-prior-scores", type=int, default=2000)
    ap.add_argument("--threshold-max-prior-scores", type=int, default=200_000)

    ap.add_argument("--start-iso", default="2025-07-31T08:18:00Z")
    ap.add_argument("--end-iso", default=None)

    ap.add_argument("--max-leverage", type=int, default=50)
    ap.add_argument("--leverage-safety-frac", type=float, default=1.0)
    ap.add_argument("--use-margin-frac", type=float, default=1.0)

    # DO_NOT_TOUCH bankroll knobs
    ap.add_argument("--trade-floor-eur", type=float, default=10.0)
    ap.add_argument("--profit-siphon-frac", type=float, default=0.50)
    ap.add_argument("--bank-threshold-eur", type=float, default=150.0)
    ap.add_argument("--liquidation-recap-bank-frac", type=float, default=0.20)

    ap.add_argument(
        "--taus",
        default="0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30",
        help="Comma-separated tau values",
    )

    ap.add_argument("--out-dir", default="data/backtests")

    args = ap.parse_args()

    t0 = time.time()

    bars_path = Path(args.bars)
    bars = pd.read_parquet(bars_path)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    bars = bars.sort_values("timestamp").reset_index(drop=True)

    # Build features once
    ctx_series = _compute_ctx_series(bars, [30, 60, 120])
    base_cols = [c for c in BASE_FEATS_DEFAULT if c in bars.columns]

    pre_df, pre_names = build_precontext_features(bars, L=5, base_feat_cols=base_cols, ctx_series=ctx_series)

    # Entry scoring
    entry_art = joblib.load(Path(args.entry_model))
    entry_model = entry_art["model"]
    entry_features = [str(c) for c in list(entry_art.get("feature_cols") or [])]

    X_df = pre_df[entry_features]
    good = np.isfinite(X_df.to_numpy(np.float64)).all(axis=1)
    # warmup: at least 1 day so z-like stats stabilize (matches other scripts)
    warmup = 1440
    good &= (np.arange(len(X_df), dtype=np.int64) >= int(warmup))
    idxs = np.where(good)[0]

    scores_full = np.full((len(bars),), np.nan, dtype=np.float32)
    if idxs.size:
        scores_full[idxs] = score_in_chunks(entry_model, X_df.iloc[idxs].to_numpy(np.float32), chunk=200_000)

    ts_open = pd.to_datetime(bars["timestamp"], utc=True)
    dates = ts_open.dt.date.to_numpy()

    thr_by_day = compute_causal_thresholds(
        dates=dates,
        scores_full=scores_full,
        target_frac=float(args.target_frac),
        seed_days=int(args.threshold_seed_days),
        min_prior_scores=int(args.threshold_min_prior_scores),
        max_prior_scores=int(args.threshold_max_prior_scores),
    )

    # Exit precontext for exit model features (use same pre_df)
    exit_art = joblib.load(Path(args.exit_model))
    exit_model = exit_art["model"]
    exit_features = [str(c) for c in list(exit_art.get("feature_cols") or [])]

    # Need coverage of exit feature names: ensure we used default base cols.
    exit_pre_arr = pre_df.to_numpy(np.float32)
    exit_pre_map = {n: i for i, n in enumerate(pre_names)}

    taus = [float(x.strip()) for x in str(args.taus).split(",") if x.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = now_ts()

    sweep_rows = []
    best = None
    best_trades = None

    for tau in taus:
        df_tr, summ = run_one_tau(
            tau=float(tau),
            bars=bars,
            ts_open=ts_open,
            dates=dates,
            scores_full=scores_full,
            thr_by_day=thr_by_day,
            exit_model=exit_model,
            exit_features=exit_features,
            exit_precontext_arr=exit_pre_arr,
            exit_precontext_map=exit_pre_map,
            start_iso=str(args.start_iso),
            end_iso=(str(args.end_iso) if args.end_iso else None),
            hold_min=int(args.hold_min),
            exit_gap_min_exit_k=int(args.exit_gap_min_exit_k),
            fee_side=float(args.fee_side),
            trade_floor_eur=float(args.trade_floor_eur),
            profit_siphon_frac=float(args.profit_siphon_frac),
            bank_threshold_eur=float(args.bank_threshold_eur),
            liquidation_recap_bank_frac=float(args.liquidation_recap_bank_frac),
            use_margin_frac=float(args.use_margin_frac),
            leverage_safety_frac=float(args.leverage_safety_frac),
            max_leverage=int(args.max_leverage),
        )
        sweep_rows.append(summ)

        score = float(summ.get("net_after_external_topups_eur", float("-inf")))
        if best is None or score > float(best.get("net_after_external_topups_eur", float("-inf"))):
            best = summ
            best_trades = df_tr

    sweep_df = pd.DataFrame(sweep_rows).sort_values("tau").reset_index(drop=True)

    sweep_csv = out_dir / f"sweep_sell_oos_tau_maxlev{int(args.max_leverage)}_top{int(args.target_frac*100):02d}pct_{ts}.csv"
    sweep_df.to_csv(sweep_csv, index=False)

    best_tau = float(best["tau"]) if best else float("nan")
    best_trades_csv = out_dir / f"sweep_sell_oos_besttau_trades_maxlev{int(args.max_leverage)}_{ts}.csv"
    if best_trades is not None:
        best_trades.to_csv(best_trades_csv, index=False)

    # Precontext analysis on best tau
    analysis_csv = out_dir / f"sweep_sell_oos_besttau_precontext_analysis_maxlev{int(args.max_leverage)}_{ts}.csv"
    if best_trades is not None and not best_trades.empty:
        analyze_precontext(trades=best_trades, precontext_df=pre_df, feature_cols=pre_names, out_csv=analysis_csv)

    meta = {
        "created_utc": ts,
        "runtime_s": float(time.time() - t0),
        "start_iso": str(args.start_iso),
        "end_iso": (str(args.end_iso) if args.end_iso else None),
        "target_frac": float(args.target_frac),
        "max_leverage": int(args.max_leverage),
        "best": best,
        "sweep_csv": str(sweep_csv),
        "best_trades_csv": str(best_trades_csv),
        "analysis_csv": str(analysis_csv),
    }
    meta_path = out_dir / f"sweep_sell_oos_tau_maxlev{int(args.max_leverage)}_meta_{ts}.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== DONE ===")
    print("Sweep:", sweep_csv)
    print("Best tau:", best_tau)
    print("Best trades:", best_trades_csv)
    print("Precontext analysis:", analysis_csv)
    print("Meta:", meta_path)


if __name__ == "__main__":
    main()
