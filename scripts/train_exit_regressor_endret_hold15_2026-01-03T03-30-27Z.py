#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-03T03:30:27Z
"""Train an exit regressor to predict end-of-trade return (15-minute horizon).

User specs implemented:
- Horizon: 15 minutes.
- Training target: after trade ends, what is the +/- % return? (fee-adjusted)
  -> label = net_return_pct(entry_open, exit_open_at_horizon, fees)
- Training uses ONLY entry_target_frac=0.0003 (fraction of candidate minutes; 0.0003 = 0.03%).
- NO tau sweep / NO exit policy simulation.
- On EACH minute (k_rel=1..15) features ALWAYS include:
  - mins_in_trade (elapsed minutes since entry)
  - delta_mark_pct (fee-adjusted P/L vs entry, marked at current close)
- Features include:
  - 5-minute pre-entry candle window stats (the 5 minutes BEFORE trade start)
  - rolling in-trade candle window stats (up to last 15 minutes of candles inside trade)

Causality:
- Entry decision at close i, entry at next open i+1.
- Per-minute features at k_rel use candles up to the decision close (no future).
- Label uses the fixed horizon exit open.

Outputs:
- Saves a joblib artifact containing model + feature list + metadata.
- Optionally saves dataset CSV.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import FEATURES, compute_feature_frame


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def fee_side_from_roundtrip_pct(roundtrip_fee_pct: float) -> float:
    fee_mult = 1.0 - float(roundtrip_fee_pct) / 100.0
    if fee_mult <= 0.0 or fee_mult >= 1.0:
        raise ValueError("roundtrip_fee_pct must be in (0, 100)")
    return float((1.0 - fee_mult) / (1.0 + fee_mult))


def net_return_pct(entry_px: float, exit_px: float, fee_side: float) -> float:
    gross_mult = float(exit_px) / max(1e-12, float(entry_px))
    net_mult = gross_mult * (1.0 - float(fee_side)) / (1.0 + float(fee_side))
    return float((net_mult - 1.0) * 100.0)


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
    """Build the v2 pattern frame used by the entry model artifact."""
    bars = bars.copy().sort_values("timestamp").reset_index(drop=True)
    bars["ts_min"] = pd.to_datetime(bars["timestamp"]).dt.floor("min")

    base = compute_feature_frame(bars)
    src = base[[c for c in FEATURES if c in base.columns]]

    missing = [f for f in base_features if f not in src.columns]
    if missing:
        raise ValueError(f"Missing required base features: {missing}")

    df = pd.DataFrame({"timestamp": pd.to_datetime(bars["timestamp"], utc=True)})

    for f in base_features:
        s = pd.to_numeric(src[f], errors="coerce")
        df[f"{f}__last"] = s
        df[f"{f}__slope5"] = slope5(s)
        df[f"{f}__accel5"] = accel5(s)
        df[f"{f}__std5"] = s.rolling(5, min_periods=5).std()
        df[f"{f}__min5"] = s.rolling(5, min_periods=5).min()
        df[f"{f}__max5"] = s.rolling(5, min_periods=5).max()
        df[f"{f}__range5"] = df[f"{f}__max5"] - df[f"{f}__min5"]

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

    close = pd.to_numeric(bars["close"], errors="coerce")
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")

    df["px__ret1m_close"] = close.pct_change() * 100.0
    df["px__ret1m_abs"] = df["px__ret1m_close"].abs()
    df["px__range_norm1m"] = (high - low) / (close + 1e-12)
    df["px__range_norm1m_abs"] = df["px__range_norm1m"].abs()

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

    return df


@dataclass
class EntryTrade:
    trade_id: int
    signal_i: int
    entry_idx: int
    entry_time: pd.Timestamp
    entry_px: float


def build_entries_from_entry_model(
    bars: pd.DataFrame,
    entry_model_artifact: dict,
    target_frac: float,
    lookback_days: int,
    min_prior_candidates_requested: int,
    hold_min: int,
) -> List[EntryTrade]:
    model = entry_model_artifact["model"]
    feat_cols = list(entry_model_artifact["feature_cols"])
    base_features = list(entry_model_artifact.get("base_features") or ["macd", "ret_1m_pct", "mom_5m_pct", "vol_std_5m", "range_norm_5m"])

    pat = build_pattern_frame_v2(bars, base_features=base_features)
    ts_utc = pat["timestamp"]
    dates = ts_utc.dt.date.to_numpy()

    # warmup for z1d indicators + enough future for horizon exit
    min_i = 1440
    last_i = len(bars) - 2 - int(hold_min)
    signal_is = np.arange(int(min_i), int(last_i) + 1, dtype=np.int64)

    X_df = pat.loc[signal_is, feat_cols]
    good = np.isfinite(X_df.to_numpy(np.float64)).all(axis=1)
    signal_is = signal_is[np.where(good)[0]]
    X = X_df.iloc[np.where(good)[0]].to_numpy(np.float32)

    print(f"Scoring entry model on {len(signal_is)} candidate minutes...", flush=True)
    scores = model.predict(X).astype(np.float64)

    unique_days: List[object] = []
    seen = set()
    for i in signal_is:
        d = dates[int(i)]
        if d not in seen:
            seen.add(d)
            unique_days.append(d)

    day_to_is: Dict[object, List[int]] = {}
    for d in unique_days:
        day_to_is[d] = [int(i) for i in signal_is[dates[signal_is] == d]]

    score_by_i: Dict[int, float] = {int(i): float(s) for i, s in zip(signal_is, scores)}

    window_cap = int(lookback_days) * 1440
    effective_min_prior = int(min(int(min_prior_candidates_requested), max(2000, int(0.5 * window_cap))))

    prior_days_scores: deque[list[float]] = deque(maxlen=int(lookback_days))

    entries: List[EntryTrade] = []
    trade_id = 0
    blocked_until_i = -10**18

    for d in unique_days:
        if len(prior_days_scores) > 0:
            prior_pool = np.asarray([x for day in prior_days_scores for x in day], dtype=np.float64)
            prior_pool = prior_pool[np.isfinite(prior_pool)]
        else:
            prior_pool = np.asarray([], dtype=np.float64)

        if len(prior_pool) >= effective_min_prior:
            thr = float(np.quantile(prior_pool, 1.0 - float(target_frac)))
            if not np.isfinite(thr):
                thr = float("inf")
        else:
            thr = float("inf")

        day_scores: list[float] = []

        for i in day_to_is[d]:
            if i < int(blocked_until_i):
                continue

            s = float(score_by_i.get(i, float("nan")))
            if not np.isfinite(s):
                continue

            day_scores.append(s)

            if s >= thr:
                entry_idx = int(i) + 1
                entry_time = pd.to_datetime(bars.iloc[entry_idx]["timestamp"], utc=True)
                entry_px = float(bars.iloc[entry_idx]["open"])

                entries.append(
                    EntryTrade(
                        trade_id=trade_id,
                        signal_i=int(i),
                        entry_idx=entry_idx,
                        entry_time=entry_time,
                        entry_px=entry_px,
                    )
                )
                trade_id += 1

                # block overlaps for full horizon
                blocked_until_i = entry_idx + int(hold_min)

        prior_days_scores.append(day_scores)

    return entries


def _slope(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 2:
        return 0.0
    if not np.isfinite(y).all():
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sum(x * x))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x * y) / denom)


def _accel(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 4:
        return 0.0
    m = n // 2
    return float(_slope(y[m:]) - _slope(y[:m]))


def candle_features(open_px: float, high_px: float, low_px: float, close_px: float) -> Dict[str, float]:
    c = max(1e-12, float(close_px))
    o = float(open_px)
    h = float(high_px)
    l = float(low_px)
    body = abs(c - o)
    upper = max(0.0, h - max(o, c))
    lower = max(0.0, min(o, c) - l)

    return {
        "oc_ret_pct": float((c / max(1e-12, o) - 1.0) * 100.0),
        "range_norm": float((h - l) / c),
        "body_norm": float(body / c),
        "upper_wick_norm": float(upper / c),
        "lower_wick_norm": float(lower / c),
    }


def window_stats(prefix: str, vals: List[float]) -> Dict[str, float]:
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {
            f"{prefix}__n": 0.0,
            f"{prefix}__last": 0.0,
            f"{prefix}__mean": 0.0,
            f"{prefix}__std": 0.0,
            f"{prefix}__min": 0.0,
            f"{prefix}__max": 0.0,
            f"{prefix}__range": 0.0,
            f"{prefix}__slope": 0.0,
            f"{prefix}__accel": 0.0,
        }

    return {
        f"{prefix}__n": float(len(v)),
        f"{prefix}__last": float(v[-1]),
        f"{prefix}__mean": float(np.mean(v)),
        f"{prefix}__std": float(np.std(v)),
        f"{prefix}__min": float(np.min(v)),
        f"{prefix}__max": float(np.max(v)),
        f"{prefix}__range": float(np.max(v) - np.min(v)),
        f"{prefix}__slope": float(_slope(v)),
        f"{prefix}__accel": float(_accel(v)),
    }


def build_exit_dataset(
    bars: pd.DataFrame,
    entries: List[EntryTrade],
    hold_min: int,
    fee_side: float,
    pre_len: int = 5,
    in_len: int = 15,
) -> pd.DataFrame:
    open_arr = bars["open"].to_numpy(np.float64)
    high_arr = bars["high"].to_numpy(np.float64)
    low_arr = bars["low"].to_numpy(np.float64)
    close_arr = bars["close"].to_numpy(np.float64)
    ts_arr = pd.to_datetime(bars["timestamp"], utc=True)

    rows: List[Dict[str, float | int | str]] = []

    for tr in entries:
        entry_idx = int(tr.entry_idx)
        entry_px = float(tr.entry_px)

        # Require pre-entry window to exist
        if entry_idx - int(pre_len) < 0:
            continue
        # Require full horizon to exist
        if entry_idx + int(hold_min) >= len(bars):
            continue

        # label: end-of-trade return at fixed horizon (exit fill at open)
        end_ret_pct = float(net_return_pct(entry_px, float(open_arr[entry_idx + int(hold_min)]), fee_side))

        # Pre-entry 5-minute window (minutes BEFORE trade start)
        pre_oc: List[float] = []
        pre_rng: List[float] = []
        pre_body: List[float] = []
        pre_up: List[float] = []
        pre_low: List[float] = []
        for c_idx in range(entry_idx - int(pre_len), entry_idx):
            feats = candle_features(
                open_px=float(open_arr[c_idx]),
                high_px=float(high_arr[c_idx]),
                low_px=float(low_arr[c_idx]),
                close_px=float(close_arr[c_idx]),
            )
            pre_oc.append(feats["oc_ret_pct"])
            pre_rng.append(feats["range_norm"])
            pre_body.append(feats["body_norm"])
            pre_up.append(feats["upper_wick_norm"])
            pre_low.append(feats["lower_wick_norm"])

        pre_stats: Dict[str, float] = {}
        pre_stats.update(window_stats("pre5_oc_ret", pre_oc))
        pre_stats.update(window_stats("pre5_rng", pre_rng))
        pre_stats.update(window_stats("pre5_body", pre_body))
        pre_stats.update(window_stats("pre5_upwick", pre_up))
        pre_stats.update(window_stats("pre5_lowwick", pre_low))

        mfe = -1e18

        for k_rel in range(1, int(hold_min) + 1):
            j_close = entry_idx + (k_rel - 1)
            if j_close >= len(bars):
                break

            # REQUIRED: time elapsed + delta at current close
            delta_mark = float(net_return_pct(entry_px, float(close_arr[j_close]), fee_side))
            mfe = max(mfe, delta_mark)
            drawdown = float(delta_mark - mfe)

            # In-trade rolling 15-minute lookback (inside trade), up to current close
            start = max(entry_idx, j_close - (int(in_len) - 1))
            end = j_close

            in_oc: List[float] = []
            in_rng: List[float] = []
            in_body: List[float] = []
            in_up: List[float] = []
            in_low: List[float] = []
            for c_idx in range(start, end + 1):
                feats = candle_features(
                    open_px=float(open_arr[c_idx]),
                    high_px=float(high_arr[c_idx]),
                    low_px=float(low_arr[c_idx]),
                    close_px=float(close_arr[c_idx]),
                )
                in_oc.append(feats["oc_ret_pct"])
                in_rng.append(feats["range_norm"])
                in_body.append(feats["body_norm"])
                in_up.append(feats["upper_wick_norm"])
                in_low.append(feats["lower_wick_norm"])

            f: Dict[str, float | int | str] = {
                "trade_id": int(tr.trade_id),
                "signal_i": int(tr.signal_i),
                "entry_idx": int(entry_idx),
                "entry_time": str(tr.entry_time),
                "decision_time": str(ts_arr.iloc[j_close]),
                "k_rel": int(k_rel),
                "mins_in_trade": float(k_rel),
                "delta_mark_pct": float(delta_mark),
                "drawdown_from_peak_mark_pct": float(drawdown),
                "label_end_ret_pct": float(end_ret_pct),
            }

            f.update(pre_stats)
            f.update(window_stats("in15_oc_ret", in_oc))
            f.update(window_stats("in15_rng", in_rng))
            f.update(window_stats("in15_body", in_body))
            f.update(window_stats("in15_upwick", in_up))
            f.update(window_stats("in15_lowwick", in_low))

            rows.append(f)

    df = pd.DataFrame(rows)

    # Numeric coercion + keep all rows (no NaNs by design)
    for c in df.columns:
        if c in ("entry_time", "decision_time"):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df


def split_by_trade(df: pd.DataFrame, test_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = df[["trade_id", "entry_time"]].drop_duplicates().copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades = trades.sort_values("entry_time").reset_index(drop=True)

    split = int(len(trades) * (1.0 - float(test_frac)))
    train_ids = set(trades.iloc[:split]["trade_id"].astype(int).tolist())
    test_ids = set(trades.iloc[split:]["trade_id"].astype(int).tolist())

    train = df[df["trade_id"].astype(int).isin(train_ids)].reset_index(drop=True)
    test = df[df["trade_id"].astype(int).isin(test_ids)].reset_index(drop=True)
    return train, test


def progress_eta_callback(period: int = 25):
    t0 = time.time()

    def _cb(env):
        it = int(env.iteration) + 1
        total = int(env.end_iteration)
        if it == 1:
            nonlocal t0
            t0 = time.time()
        if it % int(period) != 0 and it != total:
            return
        elapsed = time.time() - t0
        rate = elapsed / max(1, it)
        eta = rate * max(0, total - it)
        parts = []
        for tup in env.evaluation_result_list or []:
            data_name, metric_name, val, _ = tup
            parts.append(f"{data_name} {metric_name}={float(val):.6f}")
        evals = "  ".join(parts)
        print(f"[iter {it:>4}/{total}] {evals}  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m", flush=True)

    return _cb


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train exit regressor to predict end-of-trade return (hold=15)")
    ap.add_argument("--market-csv", required=True)
    ap.add_argument("--entry-model", required=True)

    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--entry-target-frac", type=float, default=0.0003, help="Fraction (0.0003 = 0.03%)")
    ap.add_argument("--entry-lookback-days", type=int, default=14)
    ap.add_argument("--entry-min-prior-candidates", type=int, default=50_000)

    ap.add_argument("--fee-roundtrip-pct", type=float, default=0.1)

    ap.add_argument("--trees", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--progress-period", type=int, default=25)
    ap.add_argument("--test-frac", type=float, default=0.2)

    ap.add_argument("--out-dir", default="data/exit_endret")
    ap.add_argument("--save-dataset", action="store_true")

    args = ap.parse_args()

    t0 = time.time()

    fee_side = fee_side_from_roundtrip_pct(float(args.fee_roundtrip_pct))
    print(f"Fee side used: {fee_side}")

    bars = pd.read_csv(args.market_csv, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    print(f"Bars: {len(bars)}  Range: {bars['timestamp'].min()} → {bars['timestamp'].max()}")

    entry_art = joblib.load(args.entry_model)

    print("\n=== STEP 1: Build entry trades (causal thresholding, non-overlapping) ===")
    entries = build_entries_from_entry_model(
        bars=bars,
        entry_model_artifact=entry_art,
        target_frac=float(args.entry_target_frac),
        lookback_days=int(args.entry_lookback_days),
        min_prior_candidates_requested=int(args.entry_min_prior_candidates),
        hold_min=int(args.hold_min),
    )
    print(f"Entry trades: {len(entries)}")
    if len(entries) < 30:
        print("Too few entries for training; aborting.")
        return

    print("\n=== STEP 2: Build per-minute exit dataset ===")
    ds = build_exit_dataset(
        bars=bars,
        entries=entries,
        hold_min=int(args.hold_min),
        fee_side=float(fee_side),
        pre_len=5,
        in_len=int(args.hold_min),
    )

    print(f"Dataset rows: {len(ds)} (expected ~ trades*hold_min)")
    print(f"Label end_ret_pct: mean={ds['label_end_ret_pct'].mean():.6f}  median={ds['label_end_ret_pct'].median():.6f}")

    train_df, test_df = split_by_trade(ds, test_frac=float(args.test_frac))
    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")

    # Features / label
    label_col = "label_end_ret_pct"
    meta_cols = {"trade_id", "signal_i", "entry_idx", "entry_time", "decision_time"}
    feat_cols = [c for c in ds.columns if c != label_col and c not in meta_cols]

    X_tr = train_df[feat_cols].to_numpy(np.float32)
    y_tr = train_df[label_col].to_numpy(np.float32)
    X_te = test_df[feat_cols].to_numpy(np.float32)
    y_te = test_df[label_col].to_numpy(np.float32)

    print(f"Features: {len(feat_cols)}")

    print("\n=== STEP 3: Train LightGBM regressor (500 trees) ===")
    model = LGBMRegressor(
        n_estimators=int(args.trees),
        learning_rate=float(args.learning_rate),
        max_depth=10,
        num_leaves=128,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_te, y_te)],
        eval_metric="rmse",
        callbacks=[progress_eta_callback(period=int(args.progress_period))],
    )

    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)

    metrics = {
        "train_rmse": rmse(pred_tr, y_tr),
        "test_rmse": rmse(pred_te, y_te),
        "train_mae": mae(pred_tr, y_tr),
        "test_mae": mae(pred_te, y_te),
        "test_pearson": float(np.corrcoef(pred_te.astype(np.float64), y_te.astype(np.float64))[0, 1]) if len(y_te) > 10 else float("nan"),
    }

    print("=== Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    out_dir = REPO_ROOT / str(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = now_ts()

    model_path = out_dir / f"exit_endret_regressor_hold{int(args.hold_min)}_{ts}.joblib"

    art = {
        "model": model,
        "features": feat_cols,
        "label": label_col,
        "context": {
            "market_csv": str(Path(args.market_csv).resolve()),
            "entry_model": str(Path(args.entry_model).resolve()),
            "entry_target_frac": float(args.entry_target_frac),
            "entry_lookback_days": int(args.entry_lookback_days),
            "entry_min_prior_candidates": int(args.entry_min_prior_candidates),
            "hold_min": int(args.hold_min),
            "fee_roundtrip_pct": float(args.fee_roundtrip_pct),
            "fee_side": float(fee_side),
            "notes": "Includes mins_in_trade + delta_mark_pct ALWAYS, plus pre-entry 5m window and rolling in-trade 15m window.",
            "entry_timing": "decide at close i, enter at open i+1",
            "label_timing": "exit at open entry_idx+hold_min",
        },
        "metrics": metrics,
        "created_utc": ts,
    }

    joblib.dump(art, model_path)
    print(f"\nSaved model: {model_path}")

    if bool(args.save_dataset):
        ds_path = out_dir / f"exit_endret_dataset_hold{int(args.hold_min)}_{ts}.csv"
        ds.to_csv(ds_path, index=False)
        print(f"Saved dataset: {ds_path}")

    print(f"Total elapsed: {(time.time() - t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
