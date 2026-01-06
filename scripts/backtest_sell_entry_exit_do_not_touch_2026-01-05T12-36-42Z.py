#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-05T12:36:42Z
"""Backtest SELL entry + exit with DO NOT TOUCH strategy mechanics.

This script:
- Uses the best SELL entry regressor to select entries at multiple coverage levels (top 10%, 5%, 1%, 0.1%, 0.05%).
- Uses the oracle-gap exit regressor to decide exits minute-by-minute.
- Implements the DO NOT TOUCH strategy: starting with 10 EUR, leverage-based bankroll management,
  liquidation tracking, top-ups, and profit siphoning.
- Tracks liquidations via wick-breach approximation.
- Exit model sees real-time profitability delta at each decision point.

Outputs per coverage level:
- trades CSV
- daily CSV
- summary JSON
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def net_mult_sell(entry_px: float, exit_px: float, fee_side: float) -> float:
    """Net multiplier for SELL (short): profit when price drops."""
    entry_px = max(1e-12, float(entry_px))
    exit_px = max(1e-12, float(exit_px))
    # SHORT: sell high, buy low
    # gross_mult = entry / exit
    gross_mult = entry_px / exit_px
    return float(gross_mult) * (1.0 - float(fee_side)) / (1.0 + float(fee_side))


def net_return_pct_sell(entry_px: float, exit_px: float, fee_side: float) -> float:
    return float((net_mult_sell(entry_px, exit_px, fee_side) - 1.0) * 100.0)


# ---- Feature computation helpers ----

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


def build_entry_features(
    df: pd.DataFrame,
    *,
    L: int,
    base_feat_cols: list[str],
    ctx_series: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, list[str]]:
    """Build 5-minute precontext features for entry model."""
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

    # Add missing data indicators
    close_prev_mat = _rolling_prev_matrix(close, L)
    out["missing_close_n"] = np.sum(~np.isfinite(close_prev_mat), axis=1).astype(np.float64)
    
    # Check if any value in the precontext window is missing
    all_vals = np.column_stack([close_prev_mat] + [_rolling_prev_matrix(pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64, copy=False), L) for c in base_feat_cols if c in df.columns])
    out["missing_any"] = np.any(~np.isfinite(all_vals), axis=1).astype(np.float64)

    result_df = pd.DataFrame(out)
    feature_names = list(result_df.columns)
    return result_df, feature_names


def build_exit_features_row(
    *,
    feature_cols: List[str],
    hold_min: int,
    mins_in_trade: int,
    cur_ret: float,
    r_prev1: float,
    r_prev2: float,
    r_prev3: float,
    peak_ret: float,
    precontext_row: np.ndarray,
    precontext_feat_map: Dict[str, int],
) -> np.ndarray:
    """Build exit features for a single minute decision point.
    
    The exit model sees:
    - mins_in_trade, mins_remaining
    - delta_mark_pct (current profitability %)
    - delta_mark_prev1_pct, delta_mark_prev2_pct (profitability 1-2 minutes ago)
    - delta_mark_change_1m, delta_mark_change_2m, delta_mark_change_3m (rate of change)
    - drawdown_from_peak_pct
    - precontext features (5-minute lookback from decision point)
    """

    out: List[float] = []
    for c in feature_cols:
        if c == "mins_in_trade":
            out.append(float(mins_in_trade))
        elif c == "mins_remaining":
            out.append(float(max(0, int(hold_min) - int(mins_in_trade))))
        elif c == "delta_mark_pct":
            out.append(float(cur_ret))
        elif c == "delta_mark_prev1_pct":
            out.append(float(r_prev1))
        elif c == "delta_mark_prev2_pct":
            out.append(float(r_prev2))
        elif c == "delta_mark_change_1m":
            out.append(float(cur_ret - r_prev1) if np.isfinite(r_prev1) else float("nan"))
        elif c == "delta_mark_change_2m":
            out.append(float(cur_ret - r_prev2) if np.isfinite(r_prev2) else float("nan"))
        elif c == "delta_mark_change_3m":
            out.append(float(cur_ret - r_prev3) if np.isfinite(r_prev3) else float("nan"))
        elif c == "drawdown_from_peak_pct":
            out.append(float(cur_ret - peak_ret))
        else:
            # Precontext feature - use precomputed array
            idx = precontext_feat_map.get(c, -1)
            if idx >= 0 and idx < len(precontext_row):
                out.append(float(precontext_row[idx]))
            else:
                out.append(float("nan"))

    return np.asarray(out, dtype=np.float32)


# ---- Bankroll policy ----

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
    """Siphon profit only for 100x and 50x leverage."""
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


def liquidation_price_short(entry_px: float, fee_side: float, leverage: float) -> float:
    """Liquidation price for SHORT position."""
    if leverage <= 1.0:
        return float("inf")
    # For short, liq when price rises such that loss = 1/leverage
    # net_mult = entry/exit * (1-f)/(1+f) = 1 - 1/lev
    # exit = entry * (1-f) / ((1+f) * (1 - 1/lev))
    target_net_mult = 1.0 - 1.0 / float(leverage)
    liq_px = float(entry_px) * (1.0 - fee_side) / (max(1e-12, (1.0 + fee_side) * target_net_mult))
    return float(liq_px)


def check_wick_liquidation_short(entry_px: float, high_window: np.ndarray, fee_side: float, leverage: float) -> Tuple[bool, int]:
    """Check if SHORT position hit liquidation via wick (high breach)."""
    if leverage <= 1.0:
        return False, -1
    liq_px = liquidation_price_short(entry_px, fee_side, leverage)
    breach = np.where(high_window >= liq_px)[0]
    if breach.size == 0:
        return False, -1
    return True, int(breach[0])


def score_in_chunks(model: Any, X: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    out = np.full((X.shape[0],), np.nan, dtype=np.float32)
    for s in range(0, X.shape[0], chunk):
        e = min(X.shape[0], s + chunk)
        out[s:e] = model.predict(X[s:e]).astype(np.float32)
    return out


def get_current_leverage(total_equity: float) -> float:
    """Get leverage based on DO NOT TOUCH strategy rules."""
    if total_equity < 100_000:
        return 100.0
    elif total_equity < 200_000:
        return 50.0
    elif total_equity < 500_000:
        return 20.0
    elif total_equity < 1_000_000:
        return 10.0
    else:
        return 1.0


def simulate(
    *,
    bars: pd.DataFrame,
    entry_art: Dict[str, Any],
    exit_art: Dict[str, Any],
    target_frac: float,
    hold_min: int,
    exit_gap_tau: float,
    exit_gap_min_exit_k: int,
    warmup_mins: int,
    fee_side: float,
    trade_floor_eur: float,
    profit_siphon_frac: float,
    bank_threshold_eur: float,
    liquidation_recap_bank_frac: float,
    use_margin_frac: float,
    leverage_safety_frac: float,
    threshold_mode: str,
    seed_days: int,
    min_prior_scores: int,
    max_prior_scores: int,
    start_iso: str | None,
    end_iso: str | None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    
    t0 = time.time()
    
    bars = bars.copy().sort_values("timestamp").reset_index(drop=True)
    
    entry_model = entry_art["model"]
    entry_features = [str(c) for c in list(entry_art.get("feature_cols") or [])]
    entry_base_features = list(entry_art.get("base_features") or ["ret_1m_pct", "mom_3m_pct", "mom_5m_pct", "vol_std_5m", "range_5m", "range_norm_5m", "macd", "vwap_dev_5m"])

    exit_model = exit_art["model"]
    exit_features = [str(c) for c in list(exit_art.get("feature_cols") or [])]

    print(f"Building entry features (5m precontext + ctx120) ...", flush=True)
    ctx_series = _compute_ctx_series(bars, [30, 60, 120])
    entry_feat_df, entry_feat_names = build_entry_features(bars, L=5, base_feat_cols=entry_base_features, ctx_series=ctx_series)

    # Also build exit precontext features (5m lookback at each minute)
    print(f"Building exit precontext features (5m lookback) ...", flush=True)
    exit_precontext_df, exit_precontext_names = build_entry_features(bars, L=5, base_feat_cols=entry_base_features, ctx_series=ctx_series)
    
    # Convert to numpy array for fast indexing (CRITICAL for performance)
    exit_precontext_arr = exit_precontext_df.to_numpy(np.float32)
    exit_precontext_map = {name: i for i, name in enumerate(exit_precontext_names)}
    del exit_precontext_df  # Free memory

    # Score entry model
    print(f"Scoring entry model on all minutes ...", flush=True)
    scores_full = np.full((len(bars),), np.nan, dtype=np.float32)
    if not entry_features:
        raise SystemExit("Entry artifact feature_cols is empty")

    X_df = entry_feat_df[entry_features]
    good = np.isfinite(X_df.to_numpy(np.float64)).all(axis=1)
    good &= (np.arange(len(X_df), dtype=np.int64) >= int(warmup_mins))
    idxs = np.where(good)[0]

    if idxs.size > 0:
        scores = score_in_chunks(entry_model, X_df.iloc[idxs].to_numpy(np.float32), chunk=200_000)
        scores_full[idxs] = scores

    # Thresholding
    ts_open = pd.to_datetime(bars["timestamp"], utc=True)
    dates = ts_open.dt.date.to_numpy()

    mode = str(threshold_mode).strip().lower()
    if mode not in ("global", "causal"):
        raise SystemExit(f"Unknown threshold_mode={threshold_mode!r} (use global or causal)")

    def _orderstat_threshold(prior: np.ndarray, frac: float) -> float:
        prior = np.asarray(prior, dtype=np.float64)
        prior = prior[np.isfinite(prior)]
        if prior.size == 0:
            return float("inf")
        # We want the (1-frac) quantile => threshold above which we trade.
        k = int(np.floor((1.0 - float(frac)) * float(prior.size - 1)))
        k = max(0, min(int(prior.size - 1), k))
        return float(np.partition(prior, k)[k])

    if mode == "global":
        print(f"Computing global entry threshold for target_frac={target_frac} ...", flush=True)
        valid_scores = scores_full[np.isfinite(scores_full)]
        if valid_scores.size == 0:
            raise SystemExit("No valid entry scores")
        thr_global = _orderstat_threshold(valid_scores, float(target_frac))
        print(f"Entry score threshold (global): {thr_global:.6f}", flush=True)
        thr_by_day = None
    else:
        # Causal per-day thresholding using ONLY prior days' scores.
        # Seed days: collect scores but do not trade.
        seed_days_n = max(0, int(seed_days))
        min_prior = max(0, int(min_prior_scores))
        max_prior = max(0, int(max_prior_scores))

        unique_days = pd.unique(dates)
        seed_set = set(unique_days[: min(seed_days_n, len(unique_days))]) if seed_days_n > 0 else set()

        print(
            f"Computing causal per-day thresholds: target_frac={target_frac}, seed_days={seed_days_n}, min_prior_scores={min_prior}, max_prior_scores={max_prior}",
            flush=True,
        )

        thr_by_day = {}
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
            if day_scores.size:
                fin = day_scores[np.isfinite(day_scores)]
                if fin.size:
                    prior_scores.extend([float(x) for x in fin])
                    if max_prior > 0 and len(prior_scores) > max_prior:
                        prior_scores[:] = prior_scores[-max_prior:]

            i0 = i1

        thr_global = float("nan")

    # Basic arrays

    open_arr = pd.to_numeric(bars["open"], errors="coerce").to_numpy(np.float64)
    high_arr = pd.to_numeric(bars["high"], errors="coerce").to_numpy(np.float64)
    close_arr = pd.to_numeric(bars["close"], errors="coerce").to_numpy(np.float64)

    # Precompute exit feature layout once (dynamic vs precontext)
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

    n = int(len(bars))

    # Optional windowing (OOS-only evaluation, etc.)
    ts_arr = ts_open.to_numpy(dtype="datetime64[ns]")

    start_i = 0
    end_i_excl = n
    if start_iso:
        start_dt = pd.to_datetime(str(start_iso), utc=True, errors="coerce")
        if pd.isna(start_dt):
            raise SystemExit(f"Invalid --start-iso: {start_iso!r}")
        start_i = int(np.searchsorted(ts_arr, start_dt.to_datetime64(), side="left"))
    if end_iso:
        end_dt = pd.to_datetime(str(end_iso), utc=True, errors="coerce")
        if pd.isna(end_dt):
            raise SystemExit(f"Invalid --end-iso: {end_iso!r}")
        end_i_excl = int(np.searchsorted(ts_arr, end_dt.to_datetime64(), side="left"))

    # Start scanning at max(warmup, start_i)
    i = int(max(int(warmup_mins), int(start_i)))
    last_log_t = time.time()

    # Main pass: decide entry at bar close i -> enter at i+1 open; then decide exit on closes up to hold_min.
    while i < min(n, int(end_i_excl)):
        if (time.time() - last_log_t) > 15.0:
            print(f"  t={i:,}/{n:,} trades={len(trades):,}", flush=True)
            last_log_t = time.time()

        s_i = float(scores_full[i])
        if not np.isfinite(s_i):
            i += 1
            continue

        if mode == "global":
            thr_now = float(thr_global)
        else:
            thr_now = float(thr_by_day.get(dates[i], float("inf")))

        if not np.isfinite(thr_now) or s_i < thr_now:
            i += 1
            continue

        entry_decision_idx = int(i)
        entry_idx = int(i + 1)
        if entry_idx >= n:
            break
        if int(entry_idx + hold_min) >= n:
            break
        if int(entry_idx + hold_min + 1) > int(end_i_excl):
            # Not enough room in the evaluation window to complete the trade horizon.
            break

        # Bankroll + leverage
        topup_pre = topup_to_floor(bankroll)
        equity_before = float(bankroll.trading_equity)
        total_equity_before = float(bankroll.trading_equity + bankroll.bank_equity)

        leverage = get_current_leverage(total_equity_before)
        lev_eff = float(leverage) * float(leverage_safety_frac)

        entry_px = float(open_arr[entry_idx])
        if not np.isfinite(entry_px) or entry_px <= 0:
            i += 1
            continue

        margin_eur = float(equity_before) * float(use_margin_frac)
        margin_eur = max(0.0, float(margin_eur))
        notional_eur = float(margin_eur) * float(lev_eff)

        # Exit decision indices are bar closes from entry_idx+1 .. entry_idx+hold_min
        decision_idx = np.arange(int(entry_idx + 1), int(entry_idx + hold_min + 1), dtype=np.int64)

        # Profitability delta sequence (SELL) that the exit model sees as it moves.
        close_seq = close_arr[decision_idx]
        gross_mult = (float(entry_px) / np.maximum(1e-12, close_seq)).astype(np.float64)
        net_mult = gross_mult * (1.0 - float(fee_side)) / (1.0 + float(fee_side))
        cur_ret = (net_mult - 1.0) * 100.0  # shape (hold_min,)

        # Dynamic sequences
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

        # Build exit feature matrix in one go (m rows), then predict in one model call.
        X_exit = np.empty((m, len(exit_features)), dtype=np.float32)

        # Fill precontext block
        pre_rows = exit_precontext_arr[decision_idx]
        if np.any(pre_missing_mask):
            pre_block = np.full((m, pre_idx_arr.size), np.nan, dtype=np.float32)
            ok = ~pre_missing_mask
            if np.any(ok):
                pre_block[:, ok] = pre_rows[:, pre_idx_arr[ok]]
        else:
            pre_block = pre_rows[:, pre_idx_arr]
        X_exit[:, np.asarray(pre_pos, dtype=np.int64)] = pre_block

        # Fill dynamic columns (only if present)
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

        pred_gap = exit_model.predict(X_exit)
        pred_gap = np.asarray(pred_gap, dtype=np.float64)

        eligible = (k >= float(exit_gap_min_exit_k)) & np.isfinite(pred_gap) & (pred_gap <= float(exit_gap_tau))
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
        exit_time_close = pd.to_datetime(ts_open.iloc[exit_idx], utc=True).to_pydatetime() + timedelta(minutes=1)

        # Liquidation check (wick breach on high for SHORT)
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
                "exit_gap_tau": float(exit_gap_tau),
                "exit_gap_min_exit_k": int(exit_gap_min_exit_k),
                "entry_score": float(s_i),
                "entry_threshold": float(thr_now),
                "date": dates[entry_idx],
                "equity_before_eur": float(equity_before),
                "total_equity_before_eur": float(total_equity_before),
                "margin_eur": float(margin_eur),
                "notional_eur": float(notional_eur),
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

        # Jump to the minute after exit decision.
        i = int(exit_idx + 1)

    # end while loop

    df = pd.DataFrame(trades)
    if df.empty:
        return df, pd.DataFrame(), {
            "n_trades": 0,
            "final_trading_equity": bankroll.trading_equity,
            "final_bank_equity": bankroll.bank_equity,
            "final_total_equity": bankroll.trading_equity + bankroll.bank_equity,
        }

    daily = df.groupby("date", as_index=False).agg(
        n_trades=("realized_ret_1x_pct", "size"),
        mean_ret_1x_pct=("realized_ret_1x_pct", "mean"),
        sum_profit_eur=("profit_eur", "sum"),
        end_trading_equity=("trading_equity_end_eur", "last"),
        end_bank_equity=("bank_equity_end_eur", "last"),
        end_total_equity=("total_equity_end_eur", "last"),
        n_liquidations=("liquidated", "sum"),
        mean_exit_k=("exit_rel_min", "mean"),
    )

    win_mask = df["realized_ret_1x_pct"] >= 0.0
    n_wins = int(win_mask.sum())
    win_rate = float(n_wins) / float(len(df)) if len(df) > 0 else 0.0

    summary = {
        "n_trades": int(len(df)),
        "n_wins": n_wins,
        "win_rate": float(win_rate),
        "mean_ret_1x_pct": float(df["realized_ret_1x_pct"].mean()),
        "median_ret_1x_pct": float(df["realized_ret_1x_pct"].median()),
        "n_liquidations": int(df["liquidated"].sum()) if "liquidated" in df.columns else 0,
        "n_external_topups": int(bankroll.n_external_topups),
        "n_bank_recaps": int(bankroll.n_bank_recaps),
        "total_external_topup_eur": float(bankroll.total_external_topup),
        "total_bank_recap_eur": float(bankroll.total_bank_recap),
        "final_trading_equity": float(bankroll.trading_equity),
        "final_bank_equity": float(bankroll.bank_equity),
        "final_total_equity": float(bankroll.trading_equity + bankroll.bank_equity),
        "roi_pct": float((bankroll.trading_equity + bankroll.bank_equity - trade_floor_eur) / trade_floor_eur * 100.0),
        "threshold_mode": str(mode),
        "target_frac": float(target_frac),
        "start_iso": (str(start_iso) if start_iso else None),
        "end_iso": (str(end_iso) if end_iso else None),
        "created_utc": now_ts(),
        "runtime_s": float(time.time() - t0),
    }

    return df, daily, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest SELL entry + exit with DO NOT TOUCH strategy")

    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="ETH 1m bars parquet",
    )

    ap.add_argument(
        "--entry-model",
        default="data/entry_regressor_oracle15m_sell_ctx120_weighted_oracleentry5m_2026-01-04T22-45-20Z/models/entry_regressor_sell_oracle15m_ctx120_weighted_oracleentry5m_2026-01-04T22-45-20Z.joblib",
    )
    ap.add_argument(
        "--exit-model",
        default="data/exit_oracle_sell/exit_oracle_gap_regressor_sell_hold15_top10_2026-01-05T12-27-59Z.joblib",
    )

    ap.add_argument("--out-dir", default="data/backtests")

    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--coverage-levels", default="0.10,0.05,0.01,0.001,0.0005", help="Comma-separated target fractions")

    ap.add_argument("--exit-gap-tau", type=float, default=0.10)
    ap.add_argument("--exit-gap-min-exit-k", type=int, default=2)

    ap.add_argument("--warmup-mins", type=int, default=1440)

    # Entry selection thresholding
    ap.add_argument("--threshold-mode", default="global", choices=["global", "causal"], help="global=leaky quantile over all scores; causal=per-day quantile using prior days only")
    ap.add_argument("--seed-days", type=int, default=2)
    ap.add_argument("--min-prior-scores", type=int, default=2000)
    ap.add_argument("--max-prior-scores", type=int, default=200_000)

    # Optional evaluation window (e.g., OOS-only)
    ap.add_argument("--start-iso", default=None, help="If set, start evaluating at this timestamp (inclusive). Example: 2025-07-31T08:18:00Z")
    ap.add_argument("--end-iso", default=None, help="If set, stop evaluating at this timestamp (exclusive).")

    # DO NOT TOUCH strategy defaults
    ap.add_argument("--trade-floor-eur", type=float, default=10.0)
    ap.add_argument("--profit-siphon-frac", type=float, default=0.50)
    ap.add_argument("--bank-threshold-eur", type=float, default=150.0)
    ap.add_argument("--liquidation-recap-bank-frac", type=float, default=0.20)

    ap.add_argument("--fee-side", type=float, default=0.001)
    ap.add_argument("--use-margin-frac", type=float, default=1.0)
    ap.add_argument("--leverage-safety-frac", type=float, default=1.0)

    args = ap.parse_args()

    bars_path = Path(args.bars)
    if not bars_path.exists():
        raise SystemExit(f"bars not found: {bars_path}")

    print(f"Loading bars: {bars_path}", flush=True)
    bars = pd.read_parquet(bars_path)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    bars = bars.sort_values("timestamp").reset_index(drop=True)

    entry_path = Path(args.entry_model)
    if not entry_path.exists():
        raise SystemExit(f"entry model not found: {entry_path}")
    entry_art = joblib.load(entry_path)

    exit_path = Path(args.exit_model)
    if not exit_path.exists():
        raise SystemExit(f"exit model not found: {exit_path}")
    exit_art = joblib.load(exit_path)

    coverage_levels = [float(x.strip()) for x in args.coverage_levels.split(",")]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for target_frac in coverage_levels:
        print(f"\n{'='*80}")
        print(f"Running backtest for top {target_frac*100:.3f}% coverage")
        print(f"{'='*80}\n")

        trades, daily, summary = simulate(
            bars=bars,
            entry_art=entry_art,
            exit_art=exit_art,
            target_frac=float(target_frac),
            hold_min=int(args.hold_min),
            exit_gap_tau=float(args.exit_gap_tau),
            exit_gap_min_exit_k=int(args.exit_gap_min_exit_k),
            warmup_mins=int(args.warmup_mins),
            fee_side=float(args.fee_side),
            trade_floor_eur=float(args.trade_floor_eur),
            profit_siphon_frac=float(args.profit_siphon_frac),
            bank_threshold_eur=float(args.bank_threshold_eur),
            liquidation_recap_bank_frac=float(args.liquidation_recap_bank_frac),
            use_margin_frac=float(args.use_margin_frac),
            leverage_safety_frac=float(args.leverage_safety_frac),
            threshold_mode=str(args.threshold_mode),
            seed_days=int(args.seed_days),
            min_prior_scores=int(args.min_prior_scores),
            max_prior_scores=int(args.max_prior_scores),
            start_iso=(str(args.start_iso) if args.start_iso else None),
            end_iso=(str(args.end_iso) if args.end_iso else None),
        )

        ts = now_ts()
        cov_str = f"{int(target_frac*10000):05d}bps"
        trades_path = out_dir / f"backtest_sell_do_not_touch_trades_{cov_str}_{ts}.csv"
        daily_path = out_dir / f"backtest_sell_do_not_touch_daily_{cov_str}_{ts}.csv"
        summary_path = out_dir / f"backtest_sell_do_not_touch_summary_{cov_str}_{ts}.json"

        if not trades.empty:
            trades.to_csv(trades_path, index=False)
        if not daily.empty:
            daily.to_csv(daily_path, index=False)

        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"\n=== SUMMARY (top {target_frac*100:.3f}%) ===")
        for k in [
            "n_trades",
            "n_wins",
            "win_rate",
            "mean_ret_1x_pct",
            "median_ret_1x_pct",
            "n_liquidations",
            "n_external_topups",
            "n_bank_recaps",
            "total_external_topup_eur",
            "total_bank_recap_eur",
            "final_trading_equity",
            "final_bank_equity",
            "final_total_equity",
            "roi_pct",
            "runtime_s",
        ]:
            if k in summary:
                print(f"{k}: {summary[k]}")

        if not trades.empty:
            print(f"Trades: {trades_path}")
        if not daily.empty:
            print(f"Daily: {daily_path}")
        print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
