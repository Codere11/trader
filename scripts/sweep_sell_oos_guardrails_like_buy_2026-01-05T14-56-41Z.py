#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-05T14:56:41Z
"""Sweep OOS SELL entry guardrails inspired by the ETHUSD BUY guardrail constraints.

Motivation
- In ETHUSD BUY, we used simple entry-time guardrails to avoid "bad regime" trades:
  - cap px_close_norm_pct__max5 (avoid buying after sharp 5m dump / falling knife)
  - cap vol_log1p__last (avoid volume/volatility spikes)

Here we test analogous ideas for SELL on the OOS window:
- BUY_LIKE: cap px_close_norm_pct__max5 + cap vol_log1p__last
- SELL_REVERSE: floor px_close_norm_pct__min5 (avoid shorting after sharp 5m pump / squeeze risk) + cap vol_log1p__last
- DUAL: apply both px guards + cap vol_log1p__last

Cutoffs are fit on IS-only (timestamps < start_iso) among *eligible* signals (score >= causal per-day threshold).
Then applied to OOS backtest.

Outputs
- CSV sweep summary under data/backtests/
- meta JSON with chosen parameters and cutoffs
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
    """Build the same precontext feature set used by the SELL entry/exit scripts.

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

    # Missing indicators expected by the entry model.
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


def _nanquantile(x: np.ndarray, q: float, default: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(default)
    try:
        return float(np.quantile(x, float(q)))
    except Exception:
        return float(default)


def run_oos(
    *,
    tau: float,
    bars: pd.DataFrame,
    ts_open: pd.Series,
    dates: np.ndarray,
    scores_full: np.ndarray,
    thr_by_day: Dict[object, float],
    exit_model: Any,
    exit_features: List[str],
    pre_arr: np.ndarray,
    pre_map: Dict[str, int],
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
    guard_max5_cap: Optional[float],
    guard_min5_floor: Optional[float],
    guard_vol_last_cap: Optional[float],
) -> Dict[str, Any]:
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

    # Guard feature indices (entry decision row)
    j_max5 = pre_map.get("px_close_norm_pct__max5", -1)
    j_min5 = pre_map.get("px_close_norm_pct__min5", -1)
    j_vollast = pre_map.get("vol_log1p__last", -1)

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
    pre_idx_arr = np.asarray([int(pre_map.get(n, -1)) for n in pre_names], dtype=np.int64)
    pre_missing_mask = pre_idx_arr < 0

    bankroll = BankrollState(
        trading_equity=float(trade_floor_eur),
        bank_equity=0.0,
        trade_floor_eur=float(trade_floor_eur),
        profit_siphon_frac=float(profit_siphon_frac),
        bank_threshold_eur=float(bank_threshold_eur),
        liquidation_recap_bank_frac=float(liquidation_recap_bank_frac),
    )

    n_trades = 0
    n_liq = 0
    n_skip_guard = 0

    i = int(start_i)

    while i < int(end_i_excl):
        s_i = float(scores_full[i])
        if not np.isfinite(s_i):
            i += 1
            continue

        thr_now = float(thr_by_day.get(dates[i], float("inf")))
        if not np.isfinite(thr_now) or s_i < thr_now:
            i += 1
            continue

        # Guardrails on entry decision row (t uses precontext t-5..t-1)
        row = pre_arr[i]
        if guard_vol_last_cap is not None and j_vollast >= 0:
            v = float(row[j_vollast])
            if np.isfinite(v) and v > float(guard_vol_last_cap):
                n_skip_guard += 1
                i += 1
                continue

        if guard_max5_cap is not None and j_max5 >= 0:
            v = float(row[j_max5])
            if np.isfinite(v) and v > float(guard_max5_cap):
                n_skip_guard += 1
                i += 1
                continue

        if guard_min5_floor is not None and j_min5 >= 0:
            v = float(row[j_min5])
            if np.isfinite(v) and v < float(guard_min5_floor):
                n_skip_guard += 1
                i += 1
                continue

        entry_idx = int(i + 1)
        if int(entry_idx + hold_min + 1) > int(end_i_excl):
            break

        topup_to_floor(bankroll)
        equity_before = float(bankroll.trading_equity)
        total_equity_before = float(bankroll.trading_equity + bankroll.bank_equity)

        lev_sched = float(get_current_leverage(total_equity_before))
        leverage = float(min(lev_sched, float(max(1, int(max_leverage)))))
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

        # Fill dynamic columns
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
            j_exit = int(np.where(eligible)[0][0])
            exit_reason = "pred_gap<=tau"
        else:
            j_exit = int(m - 1)
            exit_reason = "hold_min"

        exit_idx = int(decision_idx[j_exit])

        # liquidation check
        high_window = high_arr[entry_idx : exit_idx + 1]
        liq, _ = check_wick_liquidation_short(entry_px, high_window, float(fee_side), float(max(1.0, lev_eff)))
        if liq:
            n_liq += 1
            refinance_after_liquidation(bankroll)
        else:
            realized_ret_1x_pct = float(cur_ret[j_exit])
            profit_eur = float(notional_eur) * (realized_ret_1x_pct / 100.0)
            bankroll.trading_equity = float(equity_before) + float(profit_eur)
            siphon_profit_to_bank(bankroll, float(profit_eur), float(leverage))
            topup_to_floor(bankroll)

        n_trades += 1
        i = int(exit_idx + 1)

    final_total = float(bankroll.trading_equity + bankroll.bank_equity)

    return {
        "tau": float(tau),
        "n_trades": int(n_trades),
        "n_liquidations": int(n_liq),
        "liq_rate": (float(n_liq) / float(n_trades) if n_trades else 0.0),
        "n_external_topups": int(bankroll.n_external_topups),
        "total_external_topup_eur": float(bankroll.total_external_topup),
        "final_total_equity": float(final_total),
        "net_after_external_topups_eur": float(final_total - float(trade_floor_eur) - float(bankroll.total_external_topup)),
        "n_skip_guard": int(n_skip_guard),
        "guard_max5_cap": (float(guard_max5_cap) if guard_max5_cap is not None else None),
        "guard_min5_floor": (float(guard_min5_floor) if guard_min5_floor is not None else None),
        "guard_vol_last_cap": (float(guard_vol_last_cap) if guard_vol_last_cap is not None else None),
        "_exit_gap_min_exit_k": int(exit_gap_min_exit_k),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep OOS SELL guardrails inspired by ETHUSD BUY constraints")

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

    ap.add_argument("--target-frac", type=float, default=0.0005)
    ap.add_argument("--start-iso", default="2025-07-31T08:18:00Z")
    ap.add_argument("--end-iso", default=None)

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

    ap.add_argument("--q", default="0.80,0.85,0.90,0.95", help="Quantiles to try for guardrail caps")
    ap.add_argument("--q-diag-only", action="store_true", help="If set, only evaluate (q_px=q, q_vol=q) pairs")
    ap.add_argument(
        "--modes",
        default="BASELINE,DUAL",
        help="Comma-separated modes to evaluate: BASELINE, BUY_LIKE, SELL_REVERSE, DUAL",
    )
    ap.add_argument("--out-dir", default="data/backtests")

    # Exit tuning is handled elsewhere; keep tau only as metadata for now.
    ap.add_argument("--tau", type=float, default=0.24)

    args = ap.parse_args()

    t0 = time.time()

    bars = pd.read_parquet(Path(args.bars))
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    bars = bars.sort_values("timestamp").reset_index(drop=True)

    ts_open = pd.to_datetime(bars["timestamp"], utc=True)
    dates = ts_open.dt.date.to_numpy()

    ctx_series = _compute_ctx_series(bars, [30, 60, 120])
    base_cols = [c for c in BASE_FEATS_DEFAULT if c in bars.columns]
    pre_df, pre_names = build_precontext_features(bars, L=5, base_feat_cols=base_cols, ctx_series=ctx_series)

    entry_art = joblib.load(Path(args.entry_model))
    entry_model = entry_art["model"]
    entry_features = [str(c) for c in list(entry_art.get("feature_cols") or [])]

    # Exit model
    exit_art = joblib.load(Path(args.exit_model))
    exit_model = exit_art["model"]
    exit_features = [str(c) for c in list(exit_art.get("feature_cols") or [])]

    X_df = pre_df[entry_features]
    warmup = 1440
    good = np.isfinite(X_df.to_numpy(np.float64)).all(axis=1)
    good &= (np.arange(len(X_df), dtype=np.int64) >= int(warmup))
    idxs = np.where(good)[0]

    scores_full = np.full((len(bars),), np.nan, dtype=np.float32)
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

    # Precompute per-row threshold for eligibility
    thr_arr = np.asarray([float(thr_by_day.get(d, float("inf"))) for d in dates], dtype=np.float64)

    start_dt = pd.to_datetime(str(args.start_iso), utc=True, errors="coerce")
    if pd.isna(start_dt):
        raise SystemExit(f"Invalid --start-iso: {args.start_iso!r}")

    is_mask = (ts_open < start_dt).to_numpy()
    eligible_is = is_mask & np.isfinite(scores_full) & np.isfinite(thr_arr) & (scores_full.astype(np.float64) >= thr_arr)

    pre_arr = pre_df.to_numpy(np.float32)
    pre_map = {n: i for i, n in enumerate(pre_names)}

    j_max5 = pre_map.get("px_close_norm_pct__max5")
    j_min5 = pre_map.get("px_close_norm_pct__min5")
    j_vollast = pre_map.get("vol_log1p__last")

    if j_max5 is None or j_min5 is None or j_vollast is None:
        raise SystemExit("Missing required guardrail features in precontext frame")

    v_max5_is = pre_arr[eligible_is, int(j_max5)].astype(np.float64)
    v_min5_is = pre_arr[eligible_is, int(j_min5)].astype(np.float64)
    v_vol_is = pre_arr[eligible_is, int(j_vollast)].astype(np.float64)

    qs = [float(x.strip()) for x in str(args.q).split(",") if x.strip()]

    want_modes = {m.strip().upper() for m in str(args.modes).split(",") if m.strip()}
    known = {"BASELINE", "BUY_LIKE", "SELL_REVERSE", "DUAL"}
    bad = sorted(list(want_modes - known))
    if bad:
        raise SystemExit(f"Unknown mode(s) in --modes: {bad} (known: {sorted(list(known))})")

    rows: List[Dict[str, Any]] = []

    def add_row(mode: str, q_px: Optional[float], q_vol: Optional[float], *, max5_cap, min5_floor, vol_cap) -> None:
        rows.append(
            {
                "mode": str(mode),
                "q_px": (float(q_px) if q_px is not None else None),
                "q_vol": (float(q_vol) if q_vol is not None else None),
                **run_oos(
                    tau=float(args.tau),
                    bars=bars,
                    ts_open=ts_open,
                    dates=dates,
                    scores_full=scores_full,
                    thr_by_day=thr_by_day,
                    exit_model=exit_model,
                    exit_features=exit_features,
                    pre_arr=pre_arr,
                    pre_map=pre_map,
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
                    guard_max5_cap=(float(max5_cap) if max5_cap is not None else None),
                    guard_min5_floor=(float(min5_floor) if min5_floor is not None else None),
                    guard_vol_last_cap=(float(vol_cap) if vol_cap is not None else None),
                ),
            }
        )

    # Baseline (no guardrails)
    if "BASELINE" in want_modes:
        add_row("BASELINE", None, None, max5_cap=None, min5_floor=None, vol_cap=None)

    # Evaluate guard modes
    q_pairs: List[Tuple[float, float]] = []
    if bool(args.q_diag_only):
        q_pairs = [(q, q) for q in qs]
    else:
        q_pairs = [(q_px, q_vol) for q_px in qs for q_vol in qs]

    for q_px, q_vol in q_pairs:
        max5_cap = _nanquantile(v_max5_is, q_px, default=float("inf"))
        min5_floor = _nanquantile(v_min5_is, 1.0 - float(q_px), default=float("-inf"))
        vol_cap = _nanquantile(v_vol_is, q_vol, default=float("inf"))

        if "BUY_LIKE" in want_modes:
            add_row("BUY_LIKE", q_px, q_vol, max5_cap=max5_cap, min5_floor=None, vol_cap=vol_cap)
        if "SELL_REVERSE" in want_modes:
            add_row("SELL_REVERSE", q_px, q_vol, max5_cap=None, min5_floor=min5_floor, vol_cap=vol_cap)
        if "DUAL" in want_modes:
            add_row("DUAL", q_px, q_vol, max5_cap=max5_cap, min5_floor=min5_floor, vol_cap=vol_cap)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = now_ts()
    out_csv = out_dir / f"sweep_sell_oos_guardrails_like_buy_maxlev{int(args.max_leverage)}_tau{str(args.tau).replace('.', 'p')}_{ts}.csv"
    out = pd.DataFrame(rows)
    # rank by net after external topups (higher is better)
    out = out.sort_values(["net_after_external_topups_eur", "n_liquidations", "n_trades"], ascending=[False, True, False]).reset_index(drop=True)
    out.to_csv(out_csv, index=False)

    meta = {
        "created_utc": ts,
        "runtime_s": float(time.time() - t0),
        "start_iso": str(args.start_iso),
        "end_iso": (str(args.end_iso) if args.end_iso else None),
        "target_frac": float(args.target_frac),
        "tau": float(args.tau),
        "max_leverage": int(args.max_leverage),
        "is_eligible_n": int(np.sum(eligible_is)),
        "q_grid": qs,
        "out_csv": str(out_csv),
    }
    meta_path = out_dir / f"sweep_sell_oos_guardrails_like_buy_meta_{ts}.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # print top few
    print("Wrote:", out_csv)
    print("Meta:", meta_path)
    with pd.option_context("display.max_rows", 20, "display.width", 200):
        cols = [
            "mode",
            "q_px",
            "q_vol",
            "n_trades",
            "n_liquidations",
            "liq_rate",
            "total_external_topup_eur",
            "final_total_equity",
            "net_after_external_topups_eur",
            "n_skip_guard",
            "guard_max5_cap",
            "guard_min5_floor",
            "guard_vol_last_cap",
        ]
        cols = [c for c in cols if c in out.columns]
        print(out[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
