#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-03T17:40:00Z
"""Offline backtest that mirrors the current live dYdX BTC-USD runner.

Targets parity with:
- Entry model: pattern_entry_regressor_v2 (pattern v2 feature frame)
- Entry selection: causal per-day threshold from prior scores (quantile targeting --target-frac)
  with min_prior_scores gating and seed_days bootstrapping.
- Exit: oracle-gap regressor (predict oracle_gap_pct = oracle_ret_pct - ret_if_exit_now_pct)
  Policy: exit at first k>=min_exit_k where pred_gap_pct <= tau, else force exit at hold_min.
- Bankroll policy: dYdX scheme approximation (trading + bank equity, floor topups, profit siphon,
  liquidation refinance). Liquidations are approximated via wick-breach of a leverage-based liq price.

Notes
- Bars are assumed to be 1m OHLCV with 'timestamp' as bar OPEN time.
- Decisions are made on minute CLOSE: decision_time = timestamp + 1 minute.
- Entry fills at next bar OPEN (i+1 open). Exit fills at current bar CLOSE (i close).
"""

from __future__ import annotations

import argparse
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

from binance_adapter.crossref_oracle_vs_agent_patterns import FEATURES, compute_feature_frame


# ---- Basic math helpers ----

def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def net_mult(entry_px: float, exit_px: float, fee_side: float) -> float:
    entry_px = max(1e-12, float(entry_px))
    exit_px = float(exit_px)
    gross_mult = exit_px / entry_px
    return float(gross_mult) * (1.0 - float(fee_side)) / (1.0 + float(fee_side))


def net_return_pct(entry_px: float, exit_px: float, fee_side: float) -> float:
    return float((net_mult(entry_px, exit_px, fee_side) - 1.0) * 100.0)


def sigmoid_arr(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def zscore_roll(s: pd.Series, win: int) -> pd.Series:
    mu = s.rolling(int(win), min_periods=int(win)).mean()
    sd = s.rolling(int(win), min_periods=int(win)).std()
    return (s - mu) / (sd + 1e-12)


# ---- Pattern entry v2 frame (must match training + live) ----

def slope5(series: pd.Series) -> pd.Series:
    # x = [-2,-1,0,1,2]
    return ((-2.0 * series.shift(4)) + (-1.0 * series.shift(3)) + (1.0 * series.shift(1)) + (2.0 * series)) / 10.0


def accel5(series: pd.Series) -> pd.Series:
    return (series - 2.0 * series.shift(2) + series.shift(4)) / 2.0


def build_pattern_frame_v2(bars: pd.DataFrame, base_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bars = bars.copy().sort_values("timestamp").reset_index(drop=True)
    bars["ts_min"] = pd.to_datetime(bars["timestamp"]).dt.floor("min")

    base = compute_feature_frame(bars)
    src = base[[c for c in FEATURES if c in base.columns]]

    missing = [f for f in base_features if f not in src.columns]
    if missing:
        raise ValueError(f"Missing required base features: {missing}")

    df = pd.DataFrame({"timestamp": pd.to_datetime(bars["timestamp"], utc=True)})

    # 5-minute pattern descriptors
    for f in base_features:
        s = pd.to_numeric(src[f], errors="coerce")
        df[f"{f}__last"] = s
        df[f"{f}__slope5"] = slope5(s)
        df[f"{f}__accel5"] = accel5(s)
        df[f"{f}__std5"] = s.rolling(5, min_periods=5).std()
        df[f"{f}__min5"] = s.rolling(5, min_periods=5).min()
        df[f"{f}__max5"] = s.rolling(5, min_periods=5).max()
        df[f"{f}__range5"] = df[f"{f}__max5"] - df[f"{f}__min5"]

    # Cross-feature correlations
    pairs = [
        ("macd", "ret_1m_pct"),
        ("vol_std_5m", "ret_1m_pct"),
        ("range_norm_5m", "ret_1m_pct"),
    ]
    for a, b in pairs:
        if a in base_features and b in base_features:
            df[f"corr5__{a}__{b}"] = pd.to_numeric(src[a], errors="coerce").rolling(5, min_periods=5).corr(
                pd.to_numeric(src[b], errors="coerce")
            )

    # Price context
    close = pd.to_numeric(bars["close"], errors="coerce")
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")

    df["px__ret1m_close"] = close.pct_change() * 100.0
    df["px__ret1m_abs"] = df["px__ret1m_close"].abs()
    df["px__range_norm1m"] = (high - low) / (close + 1e-12)
    df["px__range_norm1m_abs"] = df["px__range_norm1m"].abs()

    # 1-day z-scores (requires warmup)
    win = 1440
    df["z1d__px_ret1m"] = zscore_roll(df["px__ret1m_close"], win)
    df["z1d__px_ret1m_abs"] = df["z1d__px_ret1m"].abs()
    df["z1d__px_range1m"] = zscore_roll(df["px__range_norm1m"], win)
    df["z1d__px_range1m_abs"] = df["z1d__px_range1m"].abs()

    vol5 = df.get("vol_std_5m__last")
    rng5max = df.get("range_norm_5m__max5")
    if vol5 is not None:
        df["z1d__vol5"] = zscore_roll(pd.to_numeric(vol5, errors="coerce"), win)
        df["risk__ret1m_abs_over_vol5"] = df["px__ret1m_abs"] / (pd.to_numeric(vol5, errors="coerce").abs() + 1e-9)
        df["risk__range1m_over_vol5"] = df["px__range_norm1m"] / (pd.to_numeric(vol5, errors="coerce").abs() + 1e-9)

    if rng5max is not None:
        df["risk__range1m_over_range5max"] = df["px__range_norm1m"] / (pd.to_numeric(rng5max, errors="coerce").abs() + 1e-12)

    if "ret_1m_pct__last" in df.columns:
        df["ret_1m_pct__last_pos"] = df["ret_1m_pct__last"].clip(lower=0.0)
        df["ret_1m_pct__last_neg"] = (-df["ret_1m_pct__last"]).clip(lower=0.0)

    df["flag__ret1m_abs_z_gt3"] = (df["z1d__px_ret1m_abs"] > 3.0).astype(np.float32)
    df["flag__range1m_abs_z_gt3"] = (df["z1d__px_range1m_abs"] > 3.0).astype(np.float32)

    return df, src


def score_in_chunks(model: Any, X: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    out = np.full((X.shape[0],), np.nan, dtype=np.float32)
    for s in range(0, X.shape[0], chunk):
        e = min(X.shape[0], s + chunk)
        out[s:e] = model.predict(X[s:e]).astype(np.float32)
    return out


# ---- Exit indicators (ind_pos/ind_neg + ind_quick_exit_3m) ----

QE_MODELS: Dict[int, Dict[str, Any]] = {
    1: {
        "feature_cols": [
            "px_range_norm1m__t0",
            "px_range_norm1m__t1",
            "range_norm_5m__t1",
            "range_norm_5m__t2",
            "range_norm_5m__t0",
            "macd__t1",
            "macd__t2",
            "px_range_norm1m__t2",
            "macd__t0",
            "vol_std_5m__t2",
            "vol_std_5m__t1",
            "vol_std_5m__t0",
        ],
        "mean": [
            0.0017311684157958627,
            0.001799022583623574,
            0.005201049404186999,
            0.005219247844162671,
            0.00508836013540689,
            -14.8716822745053,
            -12.901003073577206,
            0.0022877535369816705,
            -15.30467478596776,
            97.63208412551606,
            98.8282850808809,
            94.23914311794302,
        ],
        "std": [
            0.002151103248995944,
            0.0019385443943925737,
            0.004336551503240184,
            0.004265534481971351,
            0.004291754603747401,
            136.32319643938536,
            133.23070322665765,
            0.0023559100334701955,
            138.63555088715978,
            81.06773524775568,
            82.59282483228851,
            87.7748030360668,
        ],
        "coef": [
            0.49821968896154567,
            0.3006536977305254,
            -0.03390883350241022,
            0.5491694390310403,
            -0.45858960448213576,
            -2.6440024697146813,
            1.2525218690586208,
            -0.131773273835023,
            1.2307795942830244,
            0.10173026078095815,
            0.18957380910723812,
            -0.24169722702957883,
        ],
        "intercept": -0.1262394305926545,
    },
    2: {
        "feature_cols": [
            "px_range_norm1m__t0",
            "px_range_norm1m__t1",
            "px_range_norm1m__t2",
            "range_norm_5m__t0",
            "range_norm_5m__t2",
            "range_norm_5m__t1",
            "vol_std_5m__t0",
            "macd__t2",
            "macd__t0",
            "macd__t1",
            "vol_std_5m__t2",
            "ind_neg",
        ],
        "mean": [
            0.0016825148751894745,
            0.0017311684157958627,
            0.001799022583623574,
            0.004911417996101815,
            0.005201049404186999,
            0.00508836013540689,
            85.41860345222413,
            -14.8716822745053,
            -14.54825509941764,
            -15.30467478596776,
            98.8282850808809,
            0.8437762536470834,
        ],
        "std": [
            0.002123881519424586,
            0.002151103248995944,
            0.0019385443943925737,
            0.004064453432659976,
            0.004336551503240184,
            0.004291754603747401,
            80.48006440317958,
            136.32319643938536,
            140.7627165829248,
            138.63555088715978,
            82.59282483228851,
            0.19991236876880172,
        ],
        "coef": [
            0.6610773181013107,
            0.5090862733850794,
            0.0669961618758457,
            -0.20776535018127928,
            0.5524467593663873,
            -0.5082489258371796,
            -0.04332494043509365,
            -1.6320144354352097,
            -1.850280060153793,
            3.2809397222096,
            0.08757782979542733,
            -0.07500164742692321,
        ],
        "intercept": -0.15862642859705967,
    },
    3: {
        "feature_cols": [
            "px_range_norm1m__t0",
            "px_range_norm1m__t1",
            "px_range_norm1m__t2",
            "drawdown_from_peak_pct",
            "range_norm_5m__t0",
            "vol_std_5m__t0",
            "range_norm_5m__t1",
            "range_norm_5m__t2",
            "delta_mark_change_2m",
            "vol_std_5m__t1",
            "ind_neg",
            "macd__t0",
        ],
        "mean": [
            0.0016334127545820165,
            0.0016825148751894745,
            0.0017311684157958627,
            -0.07294866189946678,
            0.004648579939007556,
            76.93326570808777,
            0.004911417996101815,
            0.00508836013540689,
            0.025025398578561613,
            85.41860345222413,
            0.8400210779410251,
            -13.219567950278769,
        ],
        "std": [
            0.001807875565210792,
            0.002123881519424586,
            0.002151103248995944,
            0.1287983162431893,
            0.0038768760588339408,
            73.64706830374843,
            0.004064453432659976,
            0.004291754603747401,
            0.21642660549499265,
            80.48006440317958,
            0.20381119533910882,
            142.51396637646718,
        ],
        "coef": [
            0.3531046695830277,
            0.28924432342328393,
            0.2985978304067285,
            -0.6468089689196372,
            -0.4753901879792831,
            0.298650559259968,
            0.13419032645676224,
            0.14406811962902843,
            0.1220210049887942,
            -0.2589707005521335,
            -0.08416181521140421,
            -0.1603943555151908,
        ],
        "intercept": -0.2709326807596019,
    },
}


def qe_feat_value(
    feat_name: str,
    *,
    decision_i: int,
    cur_ret: float,
    r_prev1: float,
    r_prev2: float,
    peak_ret: float,
    ind_neg: float,
    px_ret1m: np.ndarray,
    px_range1m: np.ndarray,
    exit_src: pd.DataFrame,
) -> float:
    if "__t" in feat_name:
        base, suf = feat_name.rsplit("__", 1)
        try:
            lag = int(suf[1:])
        except Exception:
            lag = 0
        j = int(decision_i) - int(lag)
        if j < 0:
            return float("nan")

        if base == "px_range_norm1m":
            return float(px_range1m[j])
        if base == "px_ret1m_close":
            return float(px_ret1m[j])

        if str(base) not in exit_src.columns:
            return float("nan")
        try:
            return float(pd.to_numeric(exit_src[str(base)], errors="coerce").to_numpy(np.float64)[j])
        except Exception:
            return float("nan")

    if feat_name == "drawdown_from_peak_pct":
        return float(cur_ret - peak_ret)
    if feat_name == "delta_mark_change_2m":
        return float(cur_ret - r_prev2) if np.isfinite(r_prev2) else float("nan")
    if feat_name == "delta_mark_change_1m":
        return float(cur_ret - r_prev1) if np.isfinite(r_prev1) else float("nan")
    if feat_name == "ind_neg":
        return float(ind_neg)

    return float("nan")


def ind_quick_exit_3m(
    *,
    mins_in_trade: int,
    decision_i: int,
    cur_ret: float,
    r_prev1: float,
    r_prev2: float,
    peak_ret: float,
    ind_neg: float,
    px_ret1m: np.ndarray,
    px_range1m: np.ndarray,
    exit_src: pd.DataFrame,
) -> float:
    k = int(mins_in_trade)
    if k not in (1, 2, 3):
        return float("nan")

    m = QE_MODELS.get(k)
    if m is None:
        return float("nan")

    feats = list(m["feature_cols"])
    mu = np.asarray(m["mean"], dtype=np.float64)
    sd = np.asarray(m["std"], dtype=np.float64)
    coef = np.asarray(m["coef"], dtype=np.float64)
    intercept = float(m["intercept"])

    x = np.asarray(
        [
            qe_feat_value(
                str(fn),
                decision_i=int(decision_i),
                cur_ret=float(cur_ret),
                r_prev1=float(r_prev1),
                r_prev2=float(r_prev2),
                peak_ret=float(peak_ret),
                ind_neg=float(ind_neg),
                px_ret1m=px_ret1m,
                px_range1m=px_range1m,
                exit_src=exit_src,
            )
            for fn in feats
        ],
        dtype=np.float64,
    )
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    xz = (x - mu) / (sd + 1e-12)
    logit = intercept + float(np.dot(coef, xz))
    return float(sigmoid_arr(np.asarray([logit], dtype=np.float64))[0])


def compute_ind_pos_neg_series(bars: pd.DataFrame, exit_src: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    close = pd.to_numeric(bars["close"], errors="coerce")
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")

    px_ret1m = close.pct_change() * 100.0
    px_range1m = (high - low) / (close + 1e-12)

    pre_ret_15m = close.pct_change(15) * 100.0
    pre_range_15m = (high.rolling(15, min_periods=15).max() / (low.rolling(15, min_periods=15).min() + 1e-12)) - 1.0
    pre_vol_15m = px_ret1m.rolling(15, min_periods=15).std()
    pre_absret_max_15m = px_ret1m.abs().rolling(15, min_periods=15).max()

    vol5 = px_ret1m.rolling(5, min_periods=5).std()

    z1d_vol5 = zscore_roll(vol5, 1440)
    z1d_ret1m_abs = zscore_roll(px_ret1m, 1440).abs()
    z1d_range1m_abs = zscore_roll(px_range1m, 1440).abs()

    dip1m = (-px_ret1m).clip(lower=0.0)
    z_dip1m = zscore_roll(dip1m, 1440)
    z_pre_vol_15m = zscore_roll(pre_vol_15m, 1440)
    z_pre_range_15m = zscore_roll(pre_range_15m, 1440)
    z_pre_absret_max_15m = zscore_roll(pre_absret_max_15m, 1440)
    down15 = (-pre_ret_15m).clip(lower=0.0)
    z_down15 = zscore_roll(down15, 1440)

    if "mom_5m_pct" in exit_src.columns:
        mom5 = pd.to_numeric(exit_src.get("mom_5m_pct"), errors="coerce")
    else:
        mom5 = pd.Series([np.nan] * len(exit_src))
    z_mom5 = zscore_roll(mom5, 1440)

    pos_raw = (
        0.80 * pd.to_numeric(z_mom5, errors="coerce")
        + 0.60 * pd.to_numeric(z_dip1m, errors="coerce")
        + 0.40 * pd.to_numeric(z_pre_vol_15m, errors="coerce")
        + 0.20 * pd.to_numeric(z_pre_range_15m, errors="coerce")
        - 0.80 * pd.to_numeric(z1d_vol5, errors="coerce")
        - 0.60 * pd.to_numeric(z1d_range1m_abs, errors="coerce")
    )

    neg_raw = (
        0.90 * pd.to_numeric(z1d_vol5, errors="coerce")
        + 0.80 * pd.to_numeric(z1d_range1m_abs, errors="coerce")
        + 0.60 * pd.to_numeric(z1d_ret1m_abs, errors="coerce")
        + 0.40 * pd.to_numeric(z_pre_range_15m, errors="coerce")
        + 0.40 * pd.to_numeric(z_pre_absret_max_15m, errors="coerce")
        + 0.40 * pd.to_numeric(z_down15, errors="coerce")
    )

    ind_pos = sigmoid_arr(pd.to_numeric(pos_raw, errors="coerce").to_numpy(np.float64))
    ind_neg = sigmoid_arr(pd.to_numeric(neg_raw, errors="coerce").to_numpy(np.float64))
    return ind_pos, ind_neg


def build_exit_gap_features(
    *,
    feature_cols: List[str],
    hold_min: int,
    decision_i: int,
    mins_in_trade: int,
    cur_ret: float,
    r_prev1: float,
    r_prev2: float,
    r_prev3: float,
    peak_ret: float,
    px_ret1m: np.ndarray,
    px_range1m: np.ndarray,
    exit_src: pd.DataFrame,
    ind_pos: np.ndarray,
    ind_neg: np.ndarray,
) -> np.ndarray:
    # ind_quick_exit_3m must use scalar ind_neg at decision_i (matches live)
    ind_neg_val = float(ind_neg[int(decision_i)]) if 0 <= int(decision_i) < len(ind_neg) else float("nan")
    ind_pos_val = float(ind_pos[int(decision_i)]) if 0 <= int(decision_i) < len(ind_pos) else float("nan")

    ind_qe = ind_quick_exit_3m(
        mins_in_trade=int(mins_in_trade),
        decision_i=int(decision_i),
        cur_ret=float(cur_ret),
        r_prev1=float(r_prev1),
        r_prev2=float(r_prev2),
        peak_ret=float(peak_ret),
        ind_neg=float(ind_neg_val),
        px_ret1m=px_ret1m,
        px_range1m=px_range1m,
        exit_src=exit_src,
    )

    def get_base_feat(base: str, j: int) -> float:
        if str(base) not in exit_src.columns:
            return float("nan")
        try:
            return float(pd.to_numeric(exit_src[str(base)], errors="coerce").to_numpy(np.float64)[j])
        except Exception:
            return float("nan")

    out: List[float] = []
    for c in feature_cols:
        if c == "mins_in_trade":
            out.append(float(mins_in_trade))
            continue
        if c == "mins_remaining":
            out.append(float(max(0, int(hold_min) - int(mins_in_trade))))
            continue
        if c == "delta_mark_pct":
            out.append(float(cur_ret))
            continue
        if c == "delta_mark_prev1_pct":
            out.append(float(r_prev1))
            continue
        if c == "delta_mark_prev2_pct":
            out.append(float(r_prev2))
            continue
        if c == "delta_mark_change_1m":
            out.append(float(cur_ret - r_prev1) if np.isfinite(r_prev1) else float("nan"))
            continue
        if c == "delta_mark_change_2m":
            out.append(float(cur_ret - r_prev2) if np.isfinite(r_prev2) else float("nan"))
            continue
        if c == "delta_mark_change_3m":
            out.append(float(cur_ret - r_prev3) if np.isfinite(r_prev3) else float("nan"))
            continue
        if c == "drawdown_from_peak_pct":
            out.append(float(cur_ret - peak_ret))
            continue

        if c.startswith("px_ret1m_close__t"):
            try:
                lag = int(c.split("__t", 1)[1])
            except Exception:
                lag = 0
            j = int(decision_i) - int(lag)
            out.append(float(px_ret1m[j]) if j >= 0 else float("nan"))
            continue

        if c.startswith("px_range_norm1m__t"):
            try:
                lag = int(c.split("__t", 1)[1])
            except Exception:
                lag = 0
            j = int(decision_i) - int(lag)
            out.append(float(px_range1m[j]) if j >= 0 else float("nan"))
            continue

        if c == "ind_pos":
            out.append(float(ind_pos_val))
            continue
        if c == "ind_neg":
            out.append(float(ind_neg_val))
            continue
        if c == "ind_quick_exit_3m":
            out.append(float(ind_qe))
            continue

        if "__t" in c:
            base, suf = c.rsplit("__", 1)
            try:
                lag = int(suf[1:])
            except Exception:
                lag = 0
            j = int(decision_i) - int(lag)
            out.append(get_base_feat(str(base), j) if j >= 0 else float("nan"))
            continue

        out.append(float("nan"))

    return np.asarray(out, dtype=np.float32)


# ---- Bankroll policy (approximate dYdX live scheme) ----

@dataclass
class BankrollState:
    trading_equity: float
    bank_equity: float

    trade_floor_usdc: float = 10.0
    profit_siphon_frac: float = 0.30
    bank_threshold_usdc: float = 280.0
    liquidation_recap_bank_frac: float = 0.10

    n_external_topups: int = 0
    n_bank_recaps: int = 0
    total_external_topup: float = 0.0
    total_bank_recap: float = 0.0


def topup_to_floor(bankroll: BankrollState) -> float:
    if bankroll.trading_equity >= bankroll.trade_floor_usdc:
        return 0.0
    needed = bankroll.trade_floor_usdc - bankroll.trading_equity
    bankroll.trading_equity += needed
    bankroll.total_external_topup += needed
    bankroll.n_external_topups += 1
    return needed


def siphon_profit_to_bank(bankroll: BankrollState, profit: float) -> float:
    if profit <= 0.0:
        return 0.0
    amt = profit * bankroll.profit_siphon_frac
    bankroll.trading_equity -= amt
    bankroll.bank_equity += amt
    return amt


def refinance_after_liquidation(bankroll: BankrollState) -> Tuple[float, str]:
    bankroll.trading_equity = 0.0

    if bankroll.bank_equity < bankroll.bank_threshold_usdc:
        amt = bankroll.trade_floor_usdc
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


def liquidation_price_long(entry_px: float, fee_side: float, leverage: float) -> float:
    if leverage <= 1.0:
        return 0.0
    target_net_mult = 1.0 - 1.0 / float(leverage)
    liq_px = float(entry_px) * (1.0 + fee_side) * target_net_mult / max(1e-12, (1.0 - fee_side))
    return float(liq_px)


def check_wick_liquidation(entry_px: float, low_window: np.ndarray, fee_side: float, leverage: float) -> Tuple[bool, int]:
    if leverage <= 1.0:
        return False, -1
    liq_px = liquidation_price_long(entry_px, fee_side, leverage)
    breach = np.where(low_window <= liq_px)[0]
    if breach.size == 0:
        return False, -1
    return True, int(breach[0])


def load_entry_model(path: Path) -> Dict[str, Any]:
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "model" not in obj or "feature_cols" not in obj:
        raise SystemExit(f"Unexpected entry model format (need keys: model, feature_cols): {path}")
    return obj


def load_exit_gap_model(path: Path) -> Dict[str, Any]:
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "model" not in obj or "feature_cols" not in obj:
        raise SystemExit(f"Unexpected exit-gap model format (need keys: model, feature_cols): {path}")
    return obj


def simulate(
    *,
    bars: pd.DataFrame,
    entry_art: Dict[str, Any],
    exit_gap_art: Dict[str, Any],
    target_frac: float,
    hold_min: int,
    exit_gap_tau: float,
    exit_gap_min_exit_k: int,
    warmup_mins: int,
    seed_days: int,
    min_prior_scores: int,
    max_prior_scores: int,
    fee_side: float,
    trade_floor_usdc: float,
    profit_siphon_frac: float,
    bank_threshold_usdc: float,
    liquidation_recap_bank_frac: float,
    use_margin_frac: float,
    leverage_safety_frac: float,
    max_leverage: int,
    market_max_leverage: int,
    max_bars: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    bars = bars.copy().sort_values("timestamp").reset_index(drop=True)
    if max_bars > 0:
        bars = bars.iloc[: int(max_bars)].copy().reset_index(drop=True)

    entry_model = entry_art["model"]
    entry_features = [str(c) for c in list(entry_art.get("feature_cols") or [])]
    entry_base_features = list(entry_art.get("base_features") or ["macd", "ret_1m_pct", "mom_5m_pct", "vol_std_5m", "range_norm_5m"])

    exit_gap_model = exit_gap_art["model"]
    exit_gap_features = [str(c) for c in list(exit_gap_art.get("feature_cols") or [])]

    # Build entry pattern frame + base src for exit
    t0 = time.time()
    pat, src = build_pattern_frame_v2(bars, base_features=entry_base_features)

    # Basic arrays
    ts_open = pd.to_datetime(bars["timestamp"], utc=True)
    dates = ts_open.dt.date.to_numpy()

    open_arr = pd.to_numeric(bars["open"], errors="coerce").to_numpy(np.float64)
    low_arr = pd.to_numeric(bars["low"], errors="coerce").to_numpy(np.float64)
    close_arr = pd.to_numeric(bars["close"], errors="coerce").to_numpy(np.float64)

    px_ret1m = (pd.Series(close_arr).pct_change() * 100.0).to_numpy(np.float64)
    px_range1m = ((pd.to_numeric(bars["high"], errors="coerce") - pd.to_numeric(bars["low"], errors="coerce")) / (pd.to_numeric(bars["close"], errors="coerce") + 1e-12)).to_numpy(np.float64)

    ind_pos, ind_neg = compute_ind_pos_neg_series(bars, src)

    # Score entry model on all minutes where features are finite.
    scores_full = np.full((len(bars),), np.nan, dtype=np.float32)
    if not entry_features:
        raise SystemExit("Entry artifact feature_cols is empty")

    X_df = pat[entry_features]
    good = np.isfinite(X_df.to_numpy(np.float64)).all(axis=1)
    good &= (np.arange(len(X_df), dtype=np.int64) >= int(warmup_mins))
    idxs = np.where(good)[0]

    if idxs.size > 0:
        scores = score_in_chunks(entry_model, X_df.iloc[idxs].to_numpy(np.float32), chunk=200_000)
        scores_full[idxs] = scores

    # Prepare day order
    unique_days: List[object] = []
    seen = set()
    for d in dates:
        if d not in seen:
            seen.add(d)
            unique_days.append(d)

    bankroll = BankrollState(
        trading_equity=float(trade_floor_usdc),
        bank_equity=0.0,
        trade_floor_usdc=float(trade_floor_usdc),
        profit_siphon_frac=float(profit_siphon_frac),
        bank_threshold_usdc=float(bank_threshold_usdc),
        liquidation_recap_bank_frac=float(liquidation_recap_bank_frac),
    )

    trades: List[Dict[str, Any]] = []
    open_trade: Optional[Dict[str, Any]] = None

    # leverage config
    mkt_max_lev = int(market_max_leverage)
    lev_cap = int(max_leverage) if int(max_leverage) > 0 else int(mkt_max_lev)
    lev_eff = float(lev_cap) * float(leverage_safety_frac)

    # Causal per-day thresholding.
    # We never trade during the first seed_days calendar days, but we DO collect their scores into
    # prior_scores so later thresholds match the offline/backtest logic.
    prior_scores: List[float] = []
    seed_n = max(0, int(seed_days))
    seed_set = set(unique_days[: min(seed_n, len(unique_days))]) if seed_n > 0 else set()

    cur_day = None
    scores_cur_day: List[float] = []
    thr_cur_day = float("inf")

    def _compute_thr_from_prior() -> float:
        if len(prior_scores) < int(min_prior_scores) or len(prior_scores) == 0:
            return float("inf")
        return float(np.quantile(np.asarray(prior_scores, dtype=np.float64), 1.0 - float(target_frac)))

    for i in range(len(bars)):
        day = dates[i]

        # Handle day roll
        if cur_day is None:
            cur_day = day
            thr_cur_day = float("inf") if (day in seed_set) else _compute_thr_from_prior()
        elif day != cur_day:
            # finish prior day
            prior_scores.extend([float(x) for x in scores_cur_day if np.isfinite(float(x))])
            if max_prior_scores > 0 and len(prior_scores) > max_prior_scores:
                prior_scores[:] = prior_scores[-max_prior_scores:]
            scores_cur_day = []

            cur_day = day
            thr_cur_day = float("inf") if (day in seed_set) else _compute_thr_from_prior()

        # Append score to current day pool if finite
        s_i = float(scores_full[i])
        if np.isfinite(s_i):
            scores_cur_day.append(float(s_i))

        # Entry decision at bar close i -> enter at i+1 open
        if open_trade is None:
            if day in seed_set:
                continue

            if not np.isfinite(s_i) or not np.isfinite(thr_cur_day):
                continue

            if float(s_i) < float(thr_cur_day):
                continue

            entry_idx = i + 1
            if entry_idx >= len(bars):
                break
            # ensure we have enough horizon to force-exit
            if int(entry_idx + hold_min) >= len(bars):
                break

            topup_pre = topup_to_floor(bankroll)
            equity_before = float(bankroll.trading_equity)

            entry_px = float(open_arr[entry_idx])
            if not np.isfinite(entry_px) or entry_px <= 0:
                continue

            margin_usdc = float(equity_before) * float(use_margin_frac)
            margin_usdc = max(0.0, float(margin_usdc))
            notional_usd = float(margin_usdc) * float(lev_eff)

            open_trade = {
                "entry_idx": int(entry_idx),
                "entry_time_open": pd.to_datetime(ts_open.iloc[entry_idx], utc=True).to_pydatetime(),
                "entry_px": float(entry_px),
                "entry_score": float(s_i),
                "entry_threshold": float(thr_cur_day),
                "equity_before_usdc": float(equity_before),
                "topup_pre_usdc": float(topup_pre),
                "margin_usdc": float(margin_usdc),
                "notional_usd": float(notional_usd),
                "lev_eff": float(lev_eff),
                "ret_hist": [],
                "peak_ret": -1e30,
            }
            continue

        # Progress open trade (decision on close of bar i)
        tr = open_trade
        entry_idx = int(tr["entry_idx"])
        k_rel = int(i - entry_idx)
        if k_rel <= 0:
            continue
        if k_rel > int(hold_min):
            k_rel = int(hold_min)

        entry_px = float(tr["entry_px"])
        exit_px = float(close_arr[i])
        cur_ret = float(net_return_pct(entry_px, exit_px, float(fee_side)))

        tr["peak_ret"] = float(max(float(tr.get("peak_ret") or -1e30), float(cur_ret)))
        peak_ret = float(tr["peak_ret"])

        tr["ret_hist"].append(float(cur_ret))
        r_prev1 = float(tr["ret_hist"][-2]) if len(tr["ret_hist"]) >= 2 else float("nan")
        r_prev2 = float(tr["ret_hist"][-3]) if len(tr["ret_hist"]) >= 3 else float("nan")
        r_prev3 = float(tr["ret_hist"][-4]) if len(tr["ret_hist"]) >= 4 else float("nan")

        x_row = build_exit_gap_features(
            feature_cols=exit_gap_features,
            hold_min=int(hold_min),
            decision_i=int(i),
            mins_in_trade=int(k_rel),
            cur_ret=float(cur_ret),
            r_prev1=float(r_prev1),
            r_prev2=float(r_prev2),
            r_prev3=float(r_prev3),
            peak_ret=float(peak_ret),
            px_ret1m=px_ret1m,
            px_range1m=px_range1m,
            exit_src=src,
            ind_pos=ind_pos,
            ind_neg=ind_neg,
        )

        pred_gap = None
        try:
            pred_gap = float(exit_gap_model.predict(np.asarray([x_row], dtype=np.float32))[0])
        except Exception:
            pred_gap = None

        exit_reason = ""
        if int(k_rel) >= int(hold_min):
            exit_reason = "hold_min"
        elif int(k_rel) >= int(exit_gap_min_exit_k) and pred_gap is not None and np.isfinite(float(pred_gap)) and float(pred_gap) <= float(exit_gap_tau):
            exit_reason = "pred_gap<=tau"

        if not exit_reason:
            continue

        # Exit now
        exit_idx = int(i)
        exit_time_close = pd.to_datetime(ts_open.iloc[exit_idx], utc=True).to_pydatetime() + timedelta(minutes=1)

        # Liquidation approximation (wick breach)
        low_window = low_arr[entry_idx : exit_idx + 1]
        liq, liq_k = check_wick_liquidation(entry_px, low_window, float(fee_side), float(max(1.0, lev_eff)))

        liquidated = bool(liq)
        liq_source = ""

        equity_before = float(tr["equity_before_usdc"])
        margin_usdc = float(tr["margin_usdc"])
        notional_usd = float(tr["notional_usd"])

        if liquidated:
            realized_ret_1x_pct = -100.0
            profit_usdc = -equity_before

            recap_amt, recap_source = refinance_after_liquidation(bankroll)
            topup_post = float(recap_amt)
            liq_source = str(recap_source)
            siphon_amt = 0.0
        else:
            realized_ret_1x_pct = float(cur_ret)
            profit_usdc = float(notional_usd) * (float(realized_ret_1x_pct) / 100.0)
            bankroll.trading_equity = float(equity_before) + float(profit_usdc)

            siphon_amt = siphon_profit_to_bank(bankroll, float(profit_usdc))
            topup_post = topup_to_floor(bankroll)

        trades.append(
            {
                "entry_time_open_utc": tr["entry_time_open"].isoformat().replace("+00:00", "Z"),
                "exit_time_close_utc": exit_time_close.isoformat().replace("+00:00", "Z"),
                "entry_idx": int(entry_idx),
                "exit_idx": int(exit_idx),
                "exit_rel_min": int(k_rel),
                "entry_price": float(entry_px),
                "exit_price": float(exit_px),
                "realized_ret_1x_pct": float(realized_ret_1x_pct),
                "exit_reason": str(exit_reason),
                "exit_pred_gap_pct": (float(pred_gap) if pred_gap is not None and np.isfinite(float(pred_gap)) else np.nan),
                "exit_gap_tau": float(exit_gap_tau),
                "exit_gap_min_exit_k": int(exit_gap_min_exit_k),
                "entry_score": float(tr["entry_score"]),
                "entry_threshold": float(tr["entry_threshold"]),
                "date": day,
                "equity_before_usdc": float(equity_before),
                "margin_usdc": float(margin_usdc),
                "notional_usd": float(notional_usd),
                "lev_eff": float(lev_eff),
                "profit_usdc": float(profit_usdc),
                "siphon_usdc": float(siphon_amt),
                "topup_pre_usdc": float(tr["topup_pre_usdc"]),
                "topup_post_usdc": float(topup_post),
                "trading_equity_end_usdc": float(bankroll.trading_equity),
                "bank_equity_end_usdc": float(bankroll.bank_equity),
                "liquidated": int(liquidated),
                "liquidation_source": str(liq_source),
                "liquidation_rel_min": int(liq_k) if liquidated else -1,
            }
        )

        open_trade = None

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
        sum_profit_usdc=("profit_usdc", "sum"),
        end_trading_equity=("trading_equity_end_usdc", "last"),
        end_bank_equity=("bank_equity_end_usdc", "last"),
        n_liquidations=("liquidated", "sum"),
        mean_exit_k=("exit_rel_min", "mean"),
    )

    summary = {
        "n_trades": int(len(df)),
        "n_liquidations": int(df["liquidated"].sum()) if "liquidated" in df.columns else 0,
        "n_external_topups": int(bankroll.n_external_topups),
        "n_bank_recaps": int(bankroll.n_bank_recaps),
        "total_external_topup_usdc": float(bankroll.total_external_topup),
        "total_bank_recap_usdc": float(bankroll.total_bank_recap),
        "final_trading_equity": float(bankroll.trading_equity),
        "final_bank_equity": float(bankroll.bank_equity),
        "final_total_equity": float(bankroll.trading_equity + bankroll.bank_equity),
        "created_utc": now_ts(),
        "runtime_s": float(time.time() - t0),
    }

    return df, daily, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline parity backtest: pattern entry v2 + exit-gap regressor + dYdX bankroll")

    ap.add_argument(
        "--market-csv",
        default=str(REPO_ROOT / "data" / "dydx_BTC-USD_1MIN_2026-01-02T18-50-42Z.csv"),
        help="CSV with timestamp, open, high, low, close, volume",
    )

    ap.add_argument(
        "--entry-model",
        default=str(REPO_ROOT / "data" / "pattern_entry_regressor" / "pattern_entry_regressor_v2_2026-01-02T23-51-17Z.joblib"),
    )
    ap.add_argument(
        "--exit-gap-model",
        default=str(REPO_ROOT / "data" / "exit_oracle" / "exit_oracle_gap_regressor_hold15_2026-01-03T16-13-20Z.joblib"),
    )

    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "backtests"))

    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--target-frac", type=float, default=0.0001)

    ap.add_argument("--exit-gap-tau", type=float, default=0.10)
    ap.add_argument("--exit-gap-min-exit-k", type=int, default=2)

    ap.add_argument("--warmup-mins", type=int, default=1440)

    ap.add_argument("--seed-days", type=int, default=2)
    ap.add_argument("--min-prior-scores", type=int, default=2000)
    ap.add_argument("--max-prior-scores", type=int, default=200_000)

    # Bankroll defaults match dydx_adapter/v4.py defaults.
    ap.add_argument("--trade-floor-usdc", type=float, default=10.0)
    ap.add_argument("--profit-siphon-frac", type=float, default=0.30)
    ap.add_argument("--bank-threshold-usdc", type=float, default=280.0)
    ap.add_argument("--liquidation-recap-bank-frac", type=float, default=0.10)

    ap.add_argument("--fee-side", type=float, default=0.001)
    ap.add_argument("--use-margin-frac", type=float, default=1.0)
    ap.add_argument("--leverage-safety-frac", type=float, default=1.0)
    ap.add_argument("--max-leverage", type=int, default=0, help="0 => use market-max leverage")
    ap.add_argument("--market-max-leverage", type=int, default=50)

    ap.add_argument("--max-bars", type=int, default=0, help="If >0, only use the first N bars (debug)")

    args = ap.parse_args()

    market_csv = Path(args.market_csv)
    if not market_csv.exists():
        raise SystemExit(f"market csv not found: {market_csv}")

    bars = pd.read_csv(market_csv, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    entry_art = load_entry_model(Path(args.entry_model))
    exit_gap_art = load_exit_gap_model(Path(args.exit_gap_model))

    trades, daily, summary = simulate(
        bars=bars,
        entry_art=entry_art,
        exit_gap_art=exit_gap_art,
        target_frac=float(args.target_frac),
        hold_min=int(args.hold_min),
        exit_gap_tau=float(args.exit_gap_tau),
        exit_gap_min_exit_k=int(args.exit_gap_min_exit_k),
        warmup_mins=int(args.warmup_mins),
        seed_days=int(args.seed_days),
        min_prior_scores=int(args.min_prior_scores),
        max_prior_scores=int(args.max_prior_scores),
        fee_side=float(args.fee_side),
        trade_floor_usdc=float(args.trade_floor_usdc),
        profit_siphon_frac=float(args.profit_siphon_frac),
        bank_threshold_usdc=float(args.bank_threshold_usdc),
        liquidation_recap_bank_frac=float(args.liquidation_recap_bank_frac),
        use_margin_frac=float(args.use_margin_frac),
        leverage_safety_frac=float(args.leverage_safety_frac),
        max_leverage=int(args.max_leverage),
        market_max_leverage=int(args.market_max_leverage),
        max_bars=int(args.max_bars),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = now_ts()
    trades_path = out_dir / f"backtest_live_dydx_exact_gap_trades_{ts}.csv"
    daily_path = out_dir / f"backtest_live_dydx_exact_gap_daily_{ts}.csv"
    summary_path = out_dir / f"backtest_live_dydx_exact_gap_summary_{ts}.json"

    trades.to_csv(trades_path, index=False)
    if not daily.empty:
        daily.to_csv(daily_path, index=False)

    import json

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== SUMMARY ===")
    for k in [
        "n_trades",
        "n_liquidations",
        "n_external_topups",
        "n_bank_recaps",
        "total_external_topup_usdc",
        "total_bank_recap_usdc",
        "final_trading_equity",
        "final_bank_equity",
        "final_total_equity",
        "runtime_s",
    ]:
        if k in summary:
            print(f"{k}: {summary[k]}")

    print("Trades:", trades_path)
    if not daily.empty:
        print("Daily:", daily_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
