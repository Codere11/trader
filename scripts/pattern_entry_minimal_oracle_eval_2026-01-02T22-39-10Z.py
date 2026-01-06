#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-02T22:39:10Z
"""Minimal pattern-based entry model + oracle-exit evaluation (causal).

Implements the "minimal implementation sketch":
1) Derive a small set of pattern descriptors from the last 5 minutes of a few base features
2) Train a small model (SGD logistic regression) to predict whether an entry has >= min_profit opportunity
3) Use the model as a live-style scorer (decide on minute close i, enter at next open i+1)
4) Enforce taking only the top target_frac of *candidate* minutes (flat minutes) using a causal threshold
5) Evaluate with oracle exits (best net return within hold_min minutes)

NO LOOKAHEAD:
- Pattern features for decision i depend only on candles <= i.
- Threshold uses only prior days' candidate scores.
- Oracle label/exit uses candles strictly after entry.

Outputs:
- Trades CSV
- Daily aggregates CSV
- Saved model artifact (joblib)
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

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


def net_return_pct_vec(entry_px: np.ndarray, exit_px: np.ndarray, fee_side: float) -> np.ndarray:
    entry_px = np.maximum(entry_px.astype(np.float64), 1e-12)
    exit_px = exit_px.astype(np.float64)
    gross_mult = exit_px / entry_px
    net_mult = gross_mult * (1.0 - float(fee_side)) / (1.0 + float(fee_side))
    return (net_mult - 1.0) * 100.0


def oracle_best_ret_nextopen(
    open_arr: np.ndarray,
    close_arr: np.ndarray,
    signal_is: np.ndarray,
    hold_min: int,
    fee_side: float,
) -> np.ndarray:
    """For each signal index i, entry at i+1 open; label is best net return within next hold_min mins (exit at close)."""
    entry_idx = signal_is + 1
    entry_px = open_arr[entry_idx]

    best = np.full(len(signal_is), -1e18, dtype=np.float64)
    for k in range(1, int(hold_min) + 1):
        exit_idx = entry_idx + k
        ret = net_return_pct_vec(entry_px, close_arr[exit_idx], fee_side)
        best = np.maximum(best, ret)
    return best.astype(np.float32)


def slope5(series: pd.Series) -> pd.Series:
    """Slope over a 5-point window ending at t using x=[-2,-1,0,1,2]."""
    # slope_t = (-2*y_{t-4} -1*y_{t-3} +0*y_{t-2} +1*y_{t-1} +2*y_t) / 10
    return (
        (-2.0 * series.shift(4))
        + (-1.0 * series.shift(3))
        + (1.0 * series.shift(1))
        + (2.0 * series)
    ) / 10.0


def accel5(series: pd.Series) -> pd.Series:
    """Acceleration proxy for 5-point window: slope(last3)-slope(first3)."""
    # accel_t = (y_t - 2*y_{t-2} + y_{t-4}) / 2
    return (series - 2.0 * series.shift(2) + series.shift(4)) / 2.0


def rolling_corr5(a: pd.Series, b: pd.Series) -> pd.Series:
    # small window, pandas does this in compiled code
    return a.rolling(5, min_periods=5).corr(b)


def build_pattern_frame(
    bars: pd.DataFrame,
    base_features: List[str],
) -> pd.DataFrame:
    """Return dataframe with derived pattern descriptors aligned to bars rows."""
    t0 = time.time()
    bars = bars.copy().sort_values("timestamp").reset_index(drop=True)
    bars["ts_min"] = pd.to_datetime(bars["timestamp"]).dt.floor("min")

    print("Computing base features...", flush=True)
    base = compute_feature_frame(bars)
    src = base[[c for c in FEATURES if c in base.columns]]

    missing = [f for f in base_features if f not in src.columns]
    if missing:
        raise ValueError(f"Missing required base features: {missing}. Available: {list(src.columns)[:20]}...")

    df = pd.DataFrame({"timestamp": pd.to_datetime(bars["timestamp"], utc=True)})

    # Patterns per feature
    for f in base_features:
        s = pd.to_numeric(src[f], errors="coerce")
        df[f"{f}__last"] = s
        df[f"{f}__slope5"] = slope5(s)
        df[f"{f}__accel5"] = accel5(s)
        df[f"{f}__std5"] = s.rolling(5, min_periods=5).std()
        df[f"{f}__min5"] = s.rolling(5, min_periods=5).min()
        df[f"{f}__max5"] = s.rolling(5, min_periods=5).max()
        df[f"{f}__range5"] = df[f"{f}__max5"] - df[f"{f}__min5"]

    # A few cross-feature correlations (keep minimal)
    pairs = [
        ("macd", "ret_1m_pct"),
        ("vol_std_5m", "ret_1m_pct"),
        ("range_norm_5m", "ret_1m_pct"),
    ]
    for a, b in pairs:
        if a in base_features and b in base_features:
            df[f"corr5__{a}__{b}"] = rolling_corr5(pd.to_numeric(src[a], errors="coerce"), pd.to_numeric(src[b], errors="coerce"))

    # Add simple price context (optional but helpful)
    df["px__ret1m_close"] = pd.to_numeric(bars["close"], errors="coerce").pct_change() * 100.0
    df["px__range_norm1m"] = (pd.to_numeric(bars["high"], errors="coerce") - pd.to_numeric(bars["low"], errors="coerce")) / pd.to_numeric(bars["close"], errors="coerce")

    print(f"Pattern frame built in {(time.time() - t0):.1f}s with {df.shape[1]-1} derived columns", flush=True)
    return df


def train_sgd_logistic(
    X: np.ndarray,
    y: np.ndarray,
    test_frac: float,
    random_state: int = 42,
) -> Tuple[object, Dict[str, float]]:
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    if not (0.0 < test_frac < 1.0):
        raise ValueError("test_frac must be in (0,1)")

    split = int(len(y) * (1.0 - test_frac))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # Small, fast model.
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-4,
                    penalty="l2",
                    max_iter=20,
                    tol=1e-4,
                    random_state=random_state,
                    class_weight="balanced",
                    verbose=1,  # prints epochs -> visual progress
                ),
            ),
        ]
    )

    t0 = time.time()
    clf.fit(X_tr, y_tr)
    train_s = time.time() - t0

    # Scores
    p_te = clf.predict_proba(X_te)[:, 1]
    p_tr = clf.predict_proba(X_tr)[:, 1]

    metrics = {
        "train_auc": float(roc_auc_score(y_tr, p_tr)) if len(np.unique(y_tr)) > 1 else float("nan"),
        "test_auc": float(roc_auc_score(y_te, p_te)) if len(np.unique(y_te)) > 1 else float("nan"),
        "train_ap": float(average_precision_score(y_tr, p_tr)) if len(np.unique(y_tr)) > 1 else float("nan"),
        "test_ap": float(average_precision_score(y_te, p_te)) if len(np.unique(y_te)) > 1 else float("nan"),
        "train_seconds": float(train_s),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "pos_rate_train": float(np.mean(y_tr) * 100.0),
        "pos_rate_test": float(np.mean(y_te) * 100.0),
    }

    return clf, metrics


def simulate_top_frac_oracle_exit(
    ts_utc: pd.Series,
    dates: np.ndarray,
    open_arr: np.ndarray,
    close_arr: np.ndarray,
    signal_is: np.ndarray,
    scores: np.ndarray,
    hold_min: int,
    fee_side: float,
    target_frac: float,
    lookback_days: int,
    min_prior_candidates: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Simulate: decide on close at i, enter at i+1 open if score >= threshold.

    Threshold:
      - computed from PRIOR days' candidate scores, rolling window of lookback_days.
      - candidates are minutes where flat and score is finite.

    Note: with a lookback window, you can't require more prior candidates than the window can hold.
    """
    if not (0.0 < target_frac < 1.0):
        raise ValueError("target_frac must be in (0,1)")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be >= 1")

    # Choose an attainable bootstrap size under a rolling window.
    # We cap the bootstrap requirement to a fraction of the window's theoretical capacity to avoid "never trade".
    window_cap = int(lookback_days) * 1440
    effective_min_prior = int(min(int(min_prior_candidates), max(2000, int(0.5 * window_cap))))
    print(
        f"Threshold bootstrap: effective_min_prior={effective_min_prior} (requested {int(min_prior_candidates)}; window_cap~{window_cap})",
        flush=True,
    )

    # Map i -> score
    score_by_i: Dict[int, float] = {int(i): float(s) for i, s in zip(signal_is, scores)}

    # Determine unique days present in signal indices, in order
    unique_days: List[object] = []
    seen = set()
    for i in signal_is:
        d = dates[int(i)]
        if d not in seen:
            seen.add(d)
            unique_days.append(d)

    prior_days_scores: deque[List[float]] = deque(maxlen=int(lookback_days))

    trades: List[Dict[str, object]] = []
    total_candidates = 0
    blocked_until_i = -10**18

    for d in unique_days:
        # Build prior pool
        if len(prior_days_scores) > 0:
            prior_pool = np.asarray([x for day in prior_days_scores for x in day], dtype=np.float64)
        else:
            prior_pool = np.asarray([], dtype=np.float64)

        # Robustness: ensure quantile never sees NaNs/Infs.
        if len(prior_pool) > 0:
            prior_pool = prior_pool[np.isfinite(prior_pool)]

        if len(prior_pool) >= int(effective_min_prior):
            thr = float(np.quantile(prior_pool, 1.0 - float(target_frac)))
            if not np.isfinite(thr):
                thr = float("inf")
        else:
            thr = float("inf")

        # iterate signal indices for this day
        day_is = [int(i) for i in signal_is[dates[signal_is] == d]]
        day_candidates: List[float] = []

        for i in day_is:
            if i < int(blocked_until_i):
                continue

            s = float(score_by_i.get(i, float("nan")))
            if not np.isfinite(s):
                continue

            # candidate (would-take) minute
            total_candidates += 1
            day_candidates.append(s)

            if s >= thr:
                entry_idx = i + 1
                entry_px = float(open_arr[entry_idx])

                # oracle exit
                best_ret = -1e18
                best_k = 1
                best_exit_idx = entry_idx + 1

                for k in range(1, int(hold_min) + 1):
                    exit_idx = entry_idx + k
                    ret = float(net_return_pct_vec(np.asarray([entry_px]), np.asarray([close_arr[exit_idx]]), fee_side)[0])
                    if ret > best_ret:
                        best_ret = ret
                        best_k = k
                        best_exit_idx = exit_idx

                trades.append(
                    {
                        "signal_time": ts_utc.iloc[i],
                        "signal_index": int(i),
                        "entry_time": ts_utc.iloc[entry_idx],
                        "entry_index": int(entry_idx),
                        "entry_open": float(entry_px),
                        "exit_time": ts_utc.iloc[best_exit_idx],
                        "exit_index": int(best_exit_idx),
                        "exit_close": float(close_arr[best_exit_idx]),
                        "oracle_best_k": int(best_k),
                        "oracle_best_ret_pct": float(best_ret),
                        "score": float(s),
                        "threshold": float(thr) if np.isfinite(thr) else thr,
                        "date": d,
                    }
                )

                blocked_until_i = int(best_exit_idx)

        prior_days_scores.append(day_candidates)

    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
        daily = pd.DataFrame()
        summary = {
            "n_trades": 0.0,
            "total_candidates": float(total_candidates),
            "take_rate_pct": 0.0,
        }
        return trades_df, daily, summary

    daily = (
        trades_df.groupby("date", as_index=False)
        .agg(
            n_trades=("oracle_best_ret_pct", "size"),
            mean_oracle_ret_pct=("oracle_best_ret_pct", "mean"),
            median_oracle_ret_pct=("oracle_best_ret_pct", "median"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    rets = trades_df["oracle_best_ret_pct"].to_numpy(np.float64)
    summary = {
        "n_trades": float(len(trades_df)),
        "mean_oracle_ret_pct": float(np.mean(rets)),
        "median_oracle_ret_pct": float(np.median(rets)),
        "win_rate_gt0_pct": float(np.mean(rets > 0.0) * 100.0),
        "win_rate_ge_minprofit_pct": float(np.mean(rets >= 0.17) * 100.0),
        "total_candidates": float(total_candidates),
        "take_rate_pct": float((len(trades_df) / max(1, total_candidates)) * 100.0),
    }

    return trades_df, daily, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal pattern entry model + oracle exit eval")
    ap.add_argument("--market-csv", required=True)
    ap.add_argument("--hold-min", type=int, default=10)
    ap.add_argument("--pre-min", type=int, default=5, help="Context minutes required (5-min window ends at signal candle)")
    ap.add_argument("--min-profit-pct", type=float, default=0.17)
    ap.add_argument("--fee-roundtrip-pct", type=float, default=0.1)
    ap.add_argument("--target-frac", type=float, default=0.01)
    ap.add_argument("--lookback-days", type=int, default=14)
    ap.add_argument("--min-prior-candidates", type=int, default=10_000)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--out-dir", default="data/pattern_entry")

    args = ap.parse_args()

    fee_side = fee_side_from_roundtrip_pct(float(args.fee_roundtrip_pct))
    fee_mult = (1.0 - fee_side) / (1.0 + fee_side)
    print(f"Fees: roundtrip={float(args.fee_roundtrip_pct):.6f}% -> fee_side={fee_side:.8f} (fee_mult={fee_mult:.6f})")

    print("Loading market CSV...", flush=True)
    bars = pd.read_csv(args.market_csv, parse_dates=["timestamp"])
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    print(f"Bars: {len(bars)}  Range: {bars['timestamp'].min()} → {bars['timestamp'].max()}")

    # Base features to patternize (minimal set)
    base_feats = ["macd", "ret_1m_pct", "mom_5m_pct", "vol_std_5m", "range_norm_5m"]

    pat = build_pattern_frame(bars, base_features=base_feats)

    ts_utc = pat["timestamp"]
    dates = ts_utc.dt.date.to_numpy()

    open_arr = bars["open"].to_numpy(np.float64)
    close_arr = bars["close"].to_numpy(np.float64)

    # Signal indices i: need 5-min pattern window (i>=4) and need entry at i+1 + hold_min future
    min_i = 4
    last_i = len(bars) - 2 - int(args.hold_min)
    signal_is = np.arange(max(min_i, int(args.pre_min) - 1), last_i + 1, dtype=np.int64)

    # Build label: oracle best ret for next-open entry
    print("Building oracle labels...", flush=True)
    y_best = oracle_best_ret_nextopen(
        open_arr=open_arr,
        close_arr=close_arr,
        signal_is=signal_is,
        hold_min=int(args.hold_min),
        fee_side=float(fee_side),
    )
    y = (y_best >= float(args.min_profit_pct)).astype(np.int8)

    # Build X from pattern columns (drop timestamp)
    feat_cols = [c for c in pat.columns if c != "timestamp"]
    X_df = pat.loc[signal_is, feat_cols]

    # Drop rows with NaNs
    good = np.isfinite(X_df.to_numpy(np.float64)).all(axis=1) & np.isfinite(y_best)
    X_df = X_df.iloc[np.where(good)[0]].reset_index(drop=True)
    y = y[np.where(good)[0]]
    y_best = y_best[np.where(good)[0]]
    signal_is = signal_is[np.where(good)[0]]

    print(f"Samples: {len(y)}  Pos rate (>= {args.min_profit_pct}%): {float(np.mean(y)*100.0):.2f}%")

    X = X_df.to_numpy(np.float32)

    print("Training SGD logistic (with epoch progress)...", flush=True)
    model, m = train_sgd_logistic(X=X, y=y, test_frac=float(args.test_frac))

    print("=== Classifier metrics ===")
    for k in ["pos_rate_train", "pos_rate_test", "train_auc", "test_auc", "train_ap", "test_ap", "train_seconds"]:
        print(f"  {k}: {m.get(k)}")

    # Score all signal minutes
    print("Scoring all minutes...", flush=True)
    scores = model.predict_proba(X)[:, 1].astype(np.float64)

    print(
        f"Simulating top-fraction candidate entries + oracle exits... (lookback_days={int(args.lookback_days)}, min_prior_candidates={int(args.min_prior_candidates)})",
        flush=True,
    )
    trades_df, daily_df, summary = simulate_top_frac_oracle_exit(
        ts_utc=ts_utc,
        dates=dates,
        open_arr=open_arr,
        close_arr=close_arr,
        signal_is=signal_is,
        scores=scores,
        hold_min=int(args.hold_min),
        fee_side=float(fee_side),
        target_frac=float(args.target_frac),
        lookback_days=int(args.lookback_days),
        min_prior_candidates=int(args.min_prior_candidates),
    )

    out_dir = REPO_ROOT / str(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = now_ts()

    trades_path = out_dir / f"pattern_entry_oracle_trades_{ts}.csv"
    daily_path = out_dir / f"pattern_entry_oracle_daily_{ts}.csv"
    model_path = out_dir / f"pattern_entry_model_{ts}.joblib"

    trades_df.to_csv(trades_path, index=False)
    if len(daily_df) > 0:
        daily_df.to_csv(daily_path, index=False)

    joblib.dump(
        {
            "model": model,
            "feature_cols": feat_cols,
            "base_features": base_feats,
            "label": f"oracle_best_ret_nextopen_ge_{float(args.min_profit_pct):.4f}_pct",
            "context": {
                "market_csv": str(Path(args.market_csv).resolve()),
                "hold_min": int(args.hold_min),
                "fee_roundtrip_pct": float(args.fee_roundtrip_pct),
                "fee_side": float(fee_side),
                "target_frac": float(args.target_frac),
                "lookback_days": int(args.lookback_days),
                "min_prior_candidates": int(args.min_prior_candidates),
                "n_samples": int(len(y)),
            },
            "metrics": m,
            "created_utc": ts,
        },
        model_path,
    )

    print("=== Oracle-exit eval summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print(f"\nTrades CSV: {trades_path}")
    print(f"Daily  CSV: {daily_path}")
    print(f"Model:      {model_path}")


if __name__ == "__main__":
    main()
