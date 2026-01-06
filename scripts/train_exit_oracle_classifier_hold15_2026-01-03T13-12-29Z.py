#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-03T13:12:29Z
"""Train an exit model on oracle exits (dYdX BTC-USD).

Goal
- Build a well-labeled dataset to learn *oracle exit timing* within a fixed horizon.
- Features focus on the last 3 minutes of price/feature action plus delta movement
  (P/L vs entry, drawdown from peak), so the model can learn "sell into strength" patterns.
- Train a classifier to predict: "should we exit NOW?" (y=1 only at oracle_best_k).
- Evaluate via a causal policy on held-out trades: exit at the first minute where p>=tau,
  else hold to hold_min.

Causality / no leakage
- Entry decision at minute close i, enter at next open i+1.
- Labels (oracle_best_k) use future closes within [1..hold_min] after entry.
- Features at decision minute k use only candles <= that minute.

Notes
- This script uses the dYdX-trained pattern entry regressor v2 to define a realistic entry stream.
- For entry selection we assume a fixed hold_min holding period to enforce non-overlap.
  (i.e., entry selection is NOT biased by oracle exits).

Outputs
- Dataset parquet: per-minute rows in-trade with labels and engineered features.
- Model artifact (joblib) and evaluation CSV.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMClassifier
except Exception as e:  # pragma: no cover
    raise SystemExit(f"lightgbm import failed: {e}")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import FEATURES, compute_feature_frame


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def net_mult(entry_px: float, exit_px: float, fee_side: float) -> float:
    entry_px = max(1e-12, float(entry_px))
    exit_px = float(exit_px)
    gross_mult = exit_px / entry_px
    return float(gross_mult) * (1.0 - float(fee_side)) / (1.0 + float(fee_side))


def net_return_pct(entry_px: float, exit_px: float, fee_side: float) -> float:
    return float((net_mult(entry_px, exit_px, fee_side) - 1.0) * 100.0)


def slope5(series: pd.Series) -> pd.Series:
    # x = [-2,-1,0,1,2]
    return ((-2.0 * series.shift(4)) + (-1.0 * series.shift(3)) + (1.0 * series.shift(1)) + (2.0 * series)) / 10.0


def accel5(series: pd.Series) -> pd.Series:
    return (series - 2.0 * series.shift(2) + series.shift(4)) / 2.0


def zscore_roll(s: pd.Series, win: int) -> pd.Series:
    mu = s.rolling(int(win), min_periods=int(win)).mean()
    sd = s.rolling(int(win), min_periods=int(win)).std()
    return (s - mu) / (sd + 1e-12)


def build_pattern_frame_v2(bars: pd.DataFrame, base_features: List[str]) -> pd.DataFrame:
    """Must match scripts/pattern_entry_regressor_oracle_sweep_v2_2026-01-02T23-28-15Z.py."""
    t0 = time.time()

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

    # New indicators: spike/pump/vol regime
    win = 1440
    df["z1d__px_ret1m"] = zscore_roll(df["px__ret1m_close"], win)
    df["z1d__px_ret1m_abs"] = df["z1d__px_ret1m"].abs()
    df["z1d__px_range1m"] = zscore_roll(df["px__range_norm1m"], win)
    df["z1d__px_range1m_abs"] = df["z1d__px_range1m"].abs()

    # Context baselines
    vol5 = df.get("vol_std_5m__last")
    rng5max = df.get("range_norm_5m__max5")
    if vol5 is not None:
        df["z1d__vol5"] = zscore_roll(pd.to_numeric(vol5, errors="coerce"), win)
        df["risk__ret1m_abs_over_vol5"] = df["px__ret1m_abs"] / (pd.to_numeric(vol5, errors="coerce").abs() + 1e-9)
        df["risk__range1m_over_vol5"] = df["px__range_norm1m"] / (pd.to_numeric(vol5, errors="coerce").abs() + 1e-9)

    if rng5max is not None:
        df["risk__range1m_over_range5max"] = df["px__range_norm1m"] / (pd.to_numeric(rng5max, errors="coerce").abs() + 1e-12)

    # Directional decomposition
    if "ret_1m_pct__last" in df.columns:
        df["ret_1m_pct__last_pos"] = df["ret_1m_pct__last"].clip(lower=0.0)
        df["ret_1m_pct__last_neg"] = (-df["ret_1m_pct__last"]).clip(lower=0.0)

    # Simple extreme flags
    df["flag__ret1m_abs_z_gt3"] = (df["z1d__px_ret1m_abs"] > 3.0).astype(np.float32)
    df["flag__range1m_abs_z_gt3"] = (df["z1d__px_range1m_abs"] > 3.0).astype(np.float32)

    print(f"Pattern frame v2 built in {(time.time() - t0):.1f}s with {df.shape[1]-1} cols", flush=True)
    return df


def score_in_chunks(model: Any, X: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    out = np.full((X.shape[0],), np.nan, dtype=np.float32)
    for s in range(0, X.shape[0], chunk):
        e = min(X.shape[0], s + chunk)
        out[s:e] = model.predict(X[s:e]).astype(np.float32)
        if (s // chunk) % 5 == 0:
            print(f"  scored {e}/{X.shape[0]}")
    return out


@dataclass(frozen=True)
class Trade:
    trade_id: int
    signal_i: int
    entry_idx: int
    entry_time: pd.Timestamp
    entry_px: float
    oracle_k: int
    oracle_ret_pct: float


def build_entry_stream(
    *,
    ts_utc: pd.Series,
    dates: np.ndarray,
    scores_full: np.ndarray,
    hold_min: int,
    target_frac: float,
    seed_days: int,
    min_prior_scores: int,
    max_prior_scores: int,
) -> List[Trade]:
    """Causal per-day threshold + fixed-hold non-overlap entry stream."""

    unique_days: List[object] = []
    seen = set()
    for d in dates:
        if d not in seen:
            seen.add(d)
            unique_days.append(d)

    if len(unique_days) <= seed_days:
        raise ValueError(f"Not enough days to seed prior scores (have {len(unique_days)} days, need > {seed_days})")

    # seed prior score pool using first seed_days days' scores
    seed_set = set(unique_days[: int(seed_days)])
    prior_scores: List[float] = [float(s) for s, d in zip(scores_full, dates) if (d in seed_set and np.isfinite(float(s)))]
    if max_prior_scores > 0 and len(prior_scores) > max_prior_scores:
        prior_scores = list(prior_scores[-max_prior_scores:])

    trades: List[Trade] = []
    trade_id = 0
    blocked_until_signal_i = -10**18

    for d in unique_days[int(seed_days) :]:
        # compute threshold from prior score pool
        if len(prior_scores) >= int(min_prior_scores) and len(prior_scores) > 0:
            thr = float(np.quantile(np.asarray(prior_scores, dtype=np.float64), 1.0 - float(target_frac)))
        else:
            thr = float("inf")

        day_scores: List[float] = []
        day_is = np.where(dates == d)[0]

        for i in day_is:
            i = int(i)
            if i < int(blocked_until_signal_i):
                continue

            s = float(scores_full[i])
            if not np.isfinite(s):
                continue

            day_scores.append(float(s))

            if s < float(thr):
                continue

            entry_idx = i + 1
            exit_limit_idx = entry_idx + int(hold_min)
            if exit_limit_idx >= len(ts_utc):
                continue

            # oracle within horizon
            entry_time = ts_utc.iloc[entry_idx]
            entry_px = float("nan")
            try:
                entry_px = float(entry_open_arr[entry_idx])  # type: ignore[name-defined]
            except Exception:
                continue
            if not np.isfinite(entry_px) or entry_px <= 0:
                continue

            best_ret = -1e18
            best_k = 1
            for k in range(1, int(hold_min) + 1):
                exit_idx = entry_idx + int(k)
                px = float(close_arr[exit_idx])  # type: ignore[name-defined]
                r = net_return_pct(entry_px, px, float(fee_side))  # type: ignore[name-defined]
                if r > best_ret:
                    best_ret = float(r)
                    best_k = int(k)

            trades.append(
                Trade(
                    trade_id=int(trade_id),
                    signal_i=int(i),
                    entry_idx=int(entry_idx),
                    entry_time=pd.to_datetime(entry_time, utc=True),
                    entry_px=float(entry_px),
                    oracle_k=int(best_k),
                    oracle_ret_pct=float(best_ret),
                )
            )
            trade_id += 1

            # fixed-hold non-overlap: can only take a new entry starting the bar AFTER horizon close
            blocked_until_signal_i = int(entry_idx + int(hold_min)) + 1

        prior_scores.extend(day_scores)
        if max_prior_scores > 0 and len(prior_scores) > max_prior_scores:
            prior_scores = list(prior_scores[-max_prior_scores:])

    return trades


def build_oracle_exit_dataset(
    *,
    bars: pd.DataFrame,
    src_feats: pd.DataFrame,
    trades: List[Trade],
    hold_min: int,
    fee_side: float,
    lag_min: int = 3,
    exit_base_features: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Return (dataset_df, feature_cols).

    One row = one minute-in-trade (k=1..hold_min).
    Label y=1 iff this minute equals oracle_k for the trade.

    Features include:
    - k_rel/time
    - delta_mark_pct + recent deltas (1m/2m/3m)
    - drawdown_from_peak
    - last 3 minutes of price ret/range
    - last 3 minutes of selected computed features (exit_base_features)
    """

    ts_utc = pd.to_datetime(bars["timestamp"], utc=True)
    close = pd.to_numeric(bars["close"], errors="coerce")
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")

    px_ret1m = (close.pct_change() * 100.0).to_numpy(np.float64)
    px_range1m = ((high - low) / (close + 1e-12)).to_numpy(np.float64)

    # --- Causal regime indicators (NO FUTURE LEAKAGE) ---
    # These indicators are computed for every minute t using ONLY bars <= t.
    # They are intended to summarize "positive" vs "negative" setup signs observed in winner/loser analysis.

    def _zscore_roll_arr(x: np.ndarray, win: int = 1440) -> np.ndarray:
        s = pd.Series(pd.to_numeric(pd.Series(x), errors="coerce"), copy=False)
        mu = s.rolling(int(win), min_periods=int(win)).mean()
        sd = s.rolling(int(win), min_periods=int(win)).std()
        z = (s - mu) / (sd + 1e-12)
        return z.to_numpy(np.float64)

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-x))

    # 15-minute lookback metrics (causal)
    pre_ret_15m = (close.pct_change(15) * 100.0).to_numpy(np.float64)
    pre_range_15m = ((high.rolling(15, min_periods=15).max() / (low.rolling(15, min_periods=15).min() + 1e-12)) - 1.0).to_numpy(np.float64)
    pre_vol_15m = pd.Series(px_ret1m).rolling(15, min_periods=15).std().to_numpy(np.float64)
    pre_absret_max_15m = pd.Series(np.abs(px_ret1m)).rolling(15, min_periods=15).max().to_numpy(np.float64)

    # "1-day" z-scores (causal), matching the spirit of the pattern-v2 z1d_* features.
    vol5 = pd.Series(px_ret1m).rolling(5, min_periods=5).std().to_numpy(np.float64)
    z1d_vol5 = _zscore_roll_arr(vol5, 1440)
    z1d_ret1m_abs = np.abs(_zscore_roll_arr(px_ret1m, 1440))
    z1d_range1m_abs = np.abs(_zscore_roll_arr(px_range1m, 1440))

    # Components for indicators
    dip1m = np.clip(-px_ret1m, 0.0, None)
    z_dip1m = _zscore_roll_arr(dip1m, 1440)
    z_pre_vol_15m = _zscore_roll_arr(pre_vol_15m, 1440)
    z_pre_range_15m = _zscore_roll_arr(pre_range_15m, 1440)
    z_pre_absret_max_15m = _zscore_roll_arr(pre_absret_max_15m, 1440)
    down15 = np.clip(-pre_ret_15m, 0.0, None)
    z_down15 = _zscore_roll_arr(down15, 1440)

    # Note: mom_5m_pct is taken from src_feats (computed feature frame) so it remains causal.
    mom5 = pd.to_numeric(src_feats.get("mom_5m_pct"), errors="coerce").to_numpy(np.float64) if "mom_5m_pct" in src_feats.columns else np.full(len(src_feats), np.nan, dtype=np.float64)
    z_mom5 = _zscore_roll_arr(mom5, 1440)

    # Positive indicator: "dip in an active (but not shocky) regime".
    pos_raw = (
        0.80 * z_mom5
        + 0.60 * z_dip1m
        + 0.40 * z_pre_vol_15m
        + 0.20 * z_pre_range_15m
        - 0.80 * z1d_vol5
        - 0.60 * z1d_range1m_abs
    )

    # Negative indicator: "shock/whipsaw/crash regime" (where early exit often dominates).
    neg_raw = (
        0.90 * z1d_vol5
        + 0.80 * z1d_range1m_abs
        + 0.60 * z1d_ret1m_abs
        + 0.40 * z_pre_range_15m
        + 0.40 * z_pre_absret_max_15m
        + 0.40 * z_down15
    )

    ind_pos = _sigmoid(pos_raw)
    ind_neg = _sigmoid(neg_raw)

    # --- Learned "quick exit" risk indicator (STRICTLY first 3 minutes) ---
    # We learn k-specific linear models (k=1/2/3) predicting whether "exiting within 3 minutes"
    # would have saved a meaningful loss vs holding to 15.
    #
    # Label used for training (offline):
    #   y_leave3_saves_loss = (hold_ret < 0) and (max_{k<=3} ret_k - hold_ret >= 0.75pp)
    # Models trained on 1% entry dataset:
    #   data/exit_oracle/exit_oracle_dataset_hold15_frac0p01_2026-01-03T15-35-31Z.parquet
    # Report dir:
    #   data/analysis_quick_exit_first3m_2026-01-03T15-52-58Z/
    #
    # We embed the learned weights here to produce a causal, interpretable indicator:
    #   ind_quick_exit_3m = sigmoid( intercept_k + dot(coef_k, (x - mean_k)/std_k) )
    # and we only compute it for mins_in_trade in {1,2,3}. Otherwise NaN.

    QUICK_EXIT_MODELS: Dict[int, Dict[str, Any]] = {
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

    def _quick_exit_feat_value(
        feat_name: str,
        *,
        idx: int,
        cur_ret: float,
        r_prev1: float,
        r_prev2: float,
        peak_ret: float,
        feat_arr: Dict[str, np.ndarray],
        px_ret1m: np.ndarray,
        px_range1m: np.ndarray,
        ind_neg: np.ndarray,
    ) -> float:
        # lagged feature name form: "base__t{lag}"
        if "__t" in feat_name:
            base, suf = feat_name.rsplit("__", 1)
            try:
                lag = int(suf[1:])
            except Exception:
                lag = 0
            j = int(idx) - int(lag)
            if j < 0:
                return float("nan")
            if base == "px_range_norm1m":
                return float(px_range1m[j])
            if base == "px_ret1m_close":
                return float(px_ret1m[j])
            arr = feat_arr.get(str(base))
            return float(arr[j]) if arr is not None and j < len(arr) else float("nan")

        if feat_name == "drawdown_from_peak_pct":
            return float(cur_ret - peak_ret)
        if feat_name == "delta_mark_change_2m":
            return float(cur_ret - r_prev2) if np.isfinite(r_prev2) else float("nan")
        if feat_name == "delta_mark_change_1m":
            return float(cur_ret - r_prev1) if np.isfinite(r_prev1) else float("nan")
        if feat_name == "ind_neg":
            try:
                return float(ind_neg[int(idx)])
            except Exception:
                return float("nan")

        # fallback: direct feature array at idx
        arr = feat_arr.get(str(feat_name))
        return float(arr[int(idx)]) if arr is not None and int(idx) < len(arr) else float("nan")

    def _quick_exit_score(
        k: int,
        *,
        idx: int,
        cur_ret: float,
        r_prev1: float,
        r_prev2: float,
        peak_ret: float,
        feat_arr: Dict[str, np.ndarray],
        px_ret1m: np.ndarray,
        px_range1m: np.ndarray,
        ind_neg: np.ndarray,
    ) -> float:
        m = QUICK_EXIT_MODELS.get(int(k))
        if m is None:
            return float("nan")

        feats = list(m["feature_cols"])
        mu = np.asarray(m["mean"], dtype=np.float64)
        sd = np.asarray(m["std"], dtype=np.float64)
        coef = np.asarray(m["coef"], dtype=np.float64)
        intercept = float(m["intercept"])

        x = np.asarray(
            [
                _quick_exit_feat_value(
                    str(fn),
                    idx=int(idx),
                    cur_ret=float(cur_ret),
                    r_prev1=float(r_prev1),
                    r_prev2=float(r_prev2),
                    peak_ret=float(peak_ret),
                    feat_arr=feat_arr,
                    px_ret1m=px_ret1m,
                    px_range1m=px_range1m,
                    ind_neg=ind_neg,
                )
                for fn in feats
            ],
            dtype=np.float64,
        )

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        xz = (x - mu) / (sd + 1e-12)
        logit = intercept + float(np.dot(coef, xz))
        return float(_sigmoid(np.asarray([logit], dtype=np.float64))[0])

    # Validate + precompute exit_base_features arrays
    missing = [c for c in exit_base_features if c not in src_feats.columns]
    if missing:
        raise ValueError(f"Missing exit_base_features in computed feature frame: {missing}")

    feat_arr: Dict[str, np.ndarray] = {}
    for f in exit_base_features:
        feat_arr[str(f)] = pd.to_numeric(src_feats[f], errors="coerce").to_numpy(np.float64)

    ts_arr = ts_utc.to_numpy()
    close_arr_local = close.to_numpy(np.float64)

    rows: List[Dict[str, Any]] = []

    for tr in trades:
        entry_idx = int(tr.entry_idx)
        entry_px = float(tr.entry_px)

        # precompute per-minute returns for this trade
        rets: List[float] = []
        for k in range(1, int(hold_min) + 1):
            idx = entry_idx + int(k)
            px = float(close_arr_local[idx])
            rets.append(float(net_return_pct(entry_px, px, float(fee_side))))

        peak = -1e18
        for k in range(1, int(hold_min) + 1):
            idx = entry_idx + int(k)
            cur_ret = float(rets[k - 1])
            peak = max(peak, cur_ret)

            r_prev1 = float(rets[k - 2]) if k >= 2 else float("nan")
            r_prev2 = float(rets[k - 3]) if k >= 3 else float("nan")
            r_prev3 = float(rets[k - 4]) if k >= 4 else float("nan")

            row: Dict[str, Any] = {
                "trade_id": int(tr.trade_id),
                "signal_i": int(tr.signal_i),
                "entry_idx": int(entry_idx),
                "entry_time": tr.entry_time,
                "entry_px": float(entry_px),
                "oracle_k": int(tr.oracle_k),
                "oracle_ret_pct": float(tr.oracle_ret_pct),
                "k_rel": int(k),
                "decision_idx": int(idx),
                "decision_time": ts_arr[int(idx)],
                "y_oracle_exit": int(1 if int(k) == int(tr.oracle_k) else 0),
                "ret_if_exit_now_pct": float(cur_ret),
                "mins_in_trade": int(k),
                "mins_remaining": int(int(hold_min) - int(k)),
                "delta_mark_pct": float(cur_ret),
                "delta_mark_prev1_pct": r_prev1,
                "delta_mark_prev2_pct": r_prev2,
                "delta_mark_change_1m": (cur_ret - r_prev1) if np.isfinite(r_prev1) else float("nan"),
                "delta_mark_change_2m": (cur_ret - r_prev2) if np.isfinite(r_prev2) else float("nan"),
                "delta_mark_change_3m": (cur_ret - r_prev3) if np.isfinite(r_prev3) else float("nan"),
                "drawdown_from_peak_pct": float(cur_ret - peak),
            }

            # last 3 minutes of price info (decision candle and 2 lags)
            for lag in range(0, int(lag_min)):
                j = int(idx) - int(lag)
                suf = f"t{lag}"  # t0=current, t1=lag1, t2=lag2
                row[f"px_ret1m_close__{suf}"] = float(px_ret1m[j]) if j >= 0 else float("nan")
                row[f"px_range_norm1m__{suf}"] = float(px_range1m[j]) if j >= 0 else float("nan")

            # last 3 minutes of selected computed features
            for f in exit_base_features:
                arr = feat_arr[str(f)]
                for lag in range(0, int(lag_min)):
                    j = int(idx) - int(lag)
                    row[f"{f}__t{lag}"] = float(arr[j]) if j >= 0 else float("nan")

            # Two rolled regime indicators (causal; computed from <= decision_idx)
            try:
                row["ind_pos"] = float(ind_pos[int(idx)])
            except Exception:
                row["ind_pos"] = float("nan")
            try:
                row["ind_neg"] = float(ind_neg[int(idx)])
            except Exception:
                row["ind_neg"] = float("nan")

            # Learned quick-exit risk score (ONLY for first 3 minutes)
            if int(k) <= 3:
                try:
                    row["ind_quick_exit_3m"] = float(
                        _quick_exit_score(
                            int(k),
                            idx=int(idx),
                            cur_ret=float(cur_ret),
                            r_prev1=float(r_prev1),
                            r_prev2=float(r_prev2),
                            peak_ret=float(peak),
                            feat_arr=feat_arr,
                            px_ret1m=px_ret1m,
                            px_range1m=px_range1m,
                            ind_neg=ind_neg,
                        )
                    )
                except Exception:
                    row["ind_quick_exit_3m"] = float("nan")
            else:
                row["ind_quick_exit_3m"] = float("nan")

            rows.append(row)

    df = pd.DataFrame(rows)

    # Feature columns (exclude labels/ids/metadata)
    ignore = {
        "trade_id",
        "signal_i",
        "entry_idx",
        "entry_time",
        "entry_px",
        "oracle_k",
        "oracle_ret_pct",
        "k_rel",
        "decision_idx",
        "decision_time",
        "y_oracle_exit",
        "ret_if_exit_now_pct",
    }
    feat_cols = [c for c in df.columns if c not in ignore]

    return df, feat_cols


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score

        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def policy_eval(
    df: pd.DataFrame,
    proba: np.ndarray,
    *,
    hold_min: int,
    taus: List[float],
) -> pd.DataFrame:
    """Evaluate exit policy on a set of trades.

    Policy: for each trade, at minute k=1..hold_min, exit at the first k where p>=tau, else k=hold_min.
    Returns: mean/median return, win rate, mean exit_k.
    """

    work = df[["trade_id", "k_rel", "ret_if_exit_now_pct", "oracle_ret_pct", "oracle_k"]].copy()
    work["p_exit"] = proba.astype(np.float64)

    out_rows: List[Dict[str, float]] = []

    # Baselines
    by_trade = work.groupby("trade_id", sort=False)

    def _hold15_ret(g: pd.DataFrame) -> float:
        # ret at k=hold_min
        r = g.loc[g["k_rel"] == int(hold_min), "ret_if_exit_now_pct"]
        return float(r.iloc[0]) if len(r) else float("nan")

    hold_rets = by_trade.apply(_hold15_ret).to_numpy(np.float64)
    oracle_rets = by_trade["oracle_ret_pct"].first().to_numpy(np.float64)

    out_rows.append(
        {
            "tau": float("nan"),
            "policy": "hold",
            "n_trades": float(len(hold_rets)),
            "mean_ret_pct": float(np.nanmean(hold_rets)),
            "median_ret_pct": float(np.nanmedian(hold_rets)),
            "win_rate_gt0_pct": float(np.nanmean(hold_rets > 0.0) * 100.0),
            "mean_exit_k": float(hold_min),
        }
    )
    out_rows.append(
        {
            "tau": float("nan"),
            "policy": "oracle",
            "n_trades": float(len(oracle_rets)),
            "mean_ret_pct": float(np.nanmean(oracle_rets)),
            "median_ret_pct": float(np.nanmedian(oracle_rets)),
            "win_rate_gt0_pct": float(np.nanmean(oracle_rets > 0.0) * 100.0),
            "mean_exit_k": float(np.nanmean(by_trade["oracle_k"].first().to_numpy(np.float64))),
        }
    )

    # Learned policy sweep
    for tau in taus:
        chosen_rets: List[float] = []
        chosen_ks: List[int] = []

        for _, g in by_trade:
            g2 = g.sort_values("k_rel")
            pick = g2[g2["p_exit"] >= float(tau)]
            if len(pick) > 0:
                k = int(pick.iloc[0]["k_rel"])
                r = float(pick.iloc[0]["ret_if_exit_now_pct"])
            else:
                k = int(hold_min)
                r = float(g2.loc[g2["k_rel"] == int(hold_min), "ret_if_exit_now_pct"].iloc[0])

            chosen_rets.append(r)
            chosen_ks.append(k)

        arr = np.asarray(chosen_rets, dtype=np.float64)
        out_rows.append(
            {
                "tau": float(tau),
                "policy": "p>=tau",
                "n_trades": float(len(arr)),
                "mean_ret_pct": float(np.mean(arr)),
                "median_ret_pct": float(np.median(arr)),
                "win_rate_gt0_pct": float(np.mean(arr > 0.0) * 100.0),
                "mean_exit_k": float(np.mean(np.asarray(chosen_ks, dtype=np.float64))),
            }
        )

    return pd.DataFrame(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train an exit classifier on oracle exits (3-min window + delta features)")
    ap.add_argument(
        "--market-csv",
        default=str(REPO_ROOT / "data" / "dydx_BTC-USD_1MIN_2026-01-02T18-50-42Z.csv"),
        help="CSV with timestamp, open, high, low, close, volume",
    )
    ap.add_argument(
        "--entry-model",
        default=str(REPO_ROOT / "data" / "pattern_entry_regressor" / "pattern_entry_regressor_v2_2026-01-02T23-51-17Z.joblib"),
        help="pattern_entry_regressor_v2 joblib artifact",
    )
    ap.add_argument("--entry-target-frac", type=float, default=0.001, help="Entry take fraction (0.001=0.1%, 0.01=1%)")
    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--fee-side", type=float, default=0.001)

    ap.add_argument("--seed-days", type=int, default=2)
    ap.add_argument("--min-prior-scores", type=int, default=2000)
    ap.add_argument("--max-prior-scores", type=int, default=200_000)

    ap.add_argument(
        "--exit-base-features",
        default="macd,ret_1m_pct,mom_5m_pct,vol_std_5m,range_norm_5m",
        help="Comma-separated FEATURES columns to include (each with 3-minute lag window)",
    )

    ap.add_argument("--test-trade-frac", type=float, default=0.2)
    ap.add_argument("--trees", type=int, default=600)
    ap.add_argument("--learning-rate", type=float, default=0.05)

    ap.add_argument(
        "--taus",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,0.90",
        help="Policy thresholds to sweep: exit when p_exit >= tau",
    )

    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "exit_oracle"))
    ap.add_argument("--save-dataset", action="store_true")

    args = ap.parse_args()

    market_csv = Path(args.market_csv)
    if not market_csv.exists():
        raise SystemExit(f"market csv not found: {market_csv}")

    print("Loading market data...", flush=True)
    bars = pd.read_csv(market_csv, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    bars["ts_min"] = pd.to_datetime(bars["timestamp"]).dt.floor("min")

    hold_min = int(args.hold_min)
    fee = float(args.fee_side)

    # Global arrays used by build_entry_stream (kept module-level for speed)
    global close_arr, entry_open_arr, fee_side
    close_arr = bars["close"].to_numpy(np.float64)
    entry_open_arr = bars["open"].to_numpy(np.float64)
    fee_side = float(fee)

    print("Loading entry model...", flush=True)
    art = joblib.load(args.entry_model)
    if not isinstance(art, dict) or "model" not in art or "feature_cols" not in art:
        raise SystemExit("Entry artifact must be pattern_entry_regressor_v2 with keys: model, feature_cols")

    entry_model = art["model"]
    entry_feat_cols = list(art["feature_cols"])
    entry_base_feats = list(art.get("base_features") or ["macd", "ret_1m_pct", "mom_5m_pct", "vol_std_5m", "range_norm_5m"])

    print(f"Entry features: {len(entry_feat_cols)}  base_features={entry_base_feats}")

    print("Building pattern frame v2 (for entry scoring)...", flush=True)
    pat = build_pattern_frame_v2(bars, base_features=entry_base_feats)

    ts_utc = pd.to_datetime(pat["timestamp"], utc=True)
    dates = ts_utc.dt.date.to_numpy()

    # candidate decision indices
    min_i = 1440
    last_i = len(bars) - 2 - hold_min
    if last_i <= min_i:
        raise SystemExit("Not enough bars for hold horizon")

    cand_is = np.arange(int(min_i), int(last_i) + 1, dtype=np.int64)

    X_df = pat.loc[cand_is, entry_feat_cols]
    good = np.isfinite(X_df.to_numpy(np.float64)).all(axis=1)
    cand_is = cand_is[np.where(good)[0]]

    print(f"Scoring entry model on {len(cand_is)} candidate minutes...", flush=True)
    scores = score_in_chunks(entry_model, X_df.iloc[np.where(good)[0]].to_numpy(np.float32), chunk=200_000)

    # map into full-length array (NaN where invalid)
    scores_full = np.full((len(bars),), np.nan, dtype=np.float32)
    scores_full[cand_is] = scores

    print("Computing base feature frame (for exit features)...", flush=True)
    base = compute_feature_frame(bars)
    src = base[[c for c in FEATURES if c in base.columns]]

    # Build entry stream (fixed hold non-overlap)
    print("Selecting entries (causal per-day threshold + fixed-hold non-overlap)...", flush=True)
    trades = build_entry_stream(
        ts_utc=ts_utc,
        dates=dates,
        scores_full=scores_full,
        hold_min=hold_min,
        target_frac=float(args.entry_target_frac),
        seed_days=int(args.seed_days),
        min_prior_scores=int(args.min_prior_scores),
        max_prior_scores=int(args.max_prior_scores),
    )
    if not trades:
        raise SystemExit("No trades selected. Try lowering --min-prior-scores or increasing dataset span.")

    print(f"Trades selected: {len(trades)}")

    exit_base_feats = [x.strip() for x in str(args.exit_base_features).split(",") if x.strip()]

    print("Building oracle-exit dataset (per-minute rows)...", flush=True)
    ds, feat_cols = build_oracle_exit_dataset(
        bars=bars,
        src_feats=src,
        trades=trades,
        hold_min=hold_min,
        fee_side=fee,
        lag_min=3,
        exit_base_features=exit_base_feats,
    )

    pos_rate = float(ds["y_oracle_exit"].mean() * 100.0)
    print(f"Dataset rows: {len(ds)}  pos_rate={pos_rate:.3f}% (should be ~{100.0/hold_min:.3f}%)")

    # Split by trades (time-ordered)
    trade_order = (
        ds[["trade_id", "entry_time"]]
        .drop_duplicates("trade_id")
        .sort_values("entry_time")
        .reset_index(drop=True)
    )
    n_tr = len(trade_order)
    n_test = max(1, int(n_tr * float(args.test_trade_frac)))
    test_ids = set(trade_order.tail(n_test)["trade_id"].tolist())

    train_mask = ~ds["trade_id"].isin(test_ids)
    test_mask = ds["trade_id"].isin(test_ids)

    X_tr = ds.loc[train_mask, feat_cols].to_numpy(np.float32)
    y_tr = ds.loc[train_mask, "y_oracle_exit"].to_numpy(np.int32)
    X_te = ds.loc[test_mask, feat_cols].to_numpy(np.float32)
    y_te = ds.loc[test_mask, "y_oracle_exit"].to_numpy(np.int32)

    n_pos = int(np.sum(y_tr == 1))
    n_neg = int(np.sum(y_tr == 0))
    spw = float(n_neg / max(1, n_pos))

    print(f"Train rows: {len(y_tr)}  Test rows: {len(y_te)}  scale_pos_weight={spw:.1f}")

    clf = LGBMClassifier(
        n_estimators=int(args.trees),
        learning_rate=float(args.learning_rate),
        max_depth=8,
        num_leaves=128,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        scale_pos_weight=float(spw),
    )

    print("Training exit classifier...", flush=True)
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    print(f"Trained in {(time.time() - t0):.1f}s")

    p_te = clf.predict_proba(X_te)[:, 1]
    p_tr = clf.predict_proba(X_tr)[:, 1]

    print("=== Classification metrics ===")
    print(f"Train AUC: {safe_auc(y_tr, p_tr):.4f}  AP: {safe_ap(y_tr, p_tr):.4f}")
    print(f"Test  AUC: {safe_auc(y_te, p_te):.4f}  AP: {safe_ap(y_te, p_te):.4f}")

    # Policy evaluation on TEST trades only
    taus = [float(x.strip()) for x in str(args.taus).split(",") if x.strip()]
    eval_df = policy_eval(ds.loc[test_mask].reset_index(drop=True), p_te, hold_min=hold_min, taus=taus)

    print("\n=== Policy evaluation (TEST trades) ===")
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(eval_df.to_string(index=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = now_ts()

    if bool(args.save_dataset):
        ds_path = out_dir / f"exit_oracle_dataset_hold{hold_min}_frac{str(args.entry_target_frac).replace('.', 'p')}_{ts}.parquet"
        ds.to_parquet(ds_path, index=False)
        print(f"\nSaved dataset: {ds_path}")

    eval_path = out_dir / f"exit_oracle_eval_hold{hold_min}_frac{str(args.entry_target_frac).replace('.', 'p')}_{ts}.csv"
    eval_df.to_csv(eval_path, index=False)

    model_path = out_dir / f"exit_oracle_classifier_hold{hold_min}_frac{str(args.entry_target_frac).replace('.', 'p')}_{ts}.joblib"
    joblib.dump(
        {
            "model": clf,
            "feature_cols": feat_cols,
            "label": "y_oracle_exit",
            "context": {
                "market_csv": str(Path(args.market_csv).resolve()),
                "entry_model": str(Path(args.entry_model).resolve()),
                "entry_target_frac": float(args.entry_target_frac),
                "hold_min": int(hold_min),
                "fee_side": float(fee),
                "seed_days": int(args.seed_days),
                "min_prior_scores": int(args.min_prior_scores),
                "max_prior_scores": int(args.max_prior_scores),
                "exit_base_features": list(exit_base_feats),
                "lag_min": 3,
                "trees": int(args.trees),
                "learning_rate": float(args.learning_rate),
                "test_trade_frac": float(args.test_trade_frac),
                "taus": list(taus),
            },
            "metrics": {
                "train_auc": safe_auc(y_tr, p_tr),
                "train_ap": safe_ap(y_tr, p_tr),
                "test_auc": safe_auc(y_te, p_te),
                "test_ap": safe_ap(y_te, p_te),
            },
            "created_utc": ts,
        },
        model_path,
    )

    print(f"Saved eval:  {eval_path}")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
