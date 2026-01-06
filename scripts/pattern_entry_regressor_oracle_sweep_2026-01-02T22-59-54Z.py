#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-02T22:59:54Z
"""Pattern-based entry regressor (500 trees) + oracle-exit sweep.

Keeps the same "pattern between top features" idea, but:
- trains a regressor to predict oracle_best_ret_nextopen_pct (continuous)
- uses LightGBM with ~500 trees (n_estimators=500)
- evaluates by taking the top X% of *candidate* minutes (flat minutes) using a causal rolling threshold
- oracle exits within hold_min minutes (entry next open, exit at best close)

Causality / no leakage:
- Features at decision i use only candles <= i (5-min window ending at i).
- Entry is at i+1 open.
- Label uses future closes (i+1 .. i+1+hold_min).
- Threshold for a day is computed only from prior days' candidate scores (rolling window).

Outputs:
- CSV with sweep results
- joblib model artifact
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
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


def net_return_pct_vec(entry_px: np.ndarray, exit_px: np.ndarray, fee_side: float) -> np.ndarray:
    entry_px = np.maximum(entry_px.astype(np.float64), 1e-12)
    exit_px = exit_px.astype(np.float64)
    gross_mult = exit_px / entry_px
    net_mult = gross_mult * (1.0 - float(fee_side)) / (1.0 + float(fee_side))
    return (net_mult - 1.0) * 100.0


def slope5(series: pd.Series) -> pd.Series:
    # x = [-2,-1,0,1,2]
    return ((-2.0 * series.shift(4)) + (-1.0 * series.shift(3)) + (1.0 * series.shift(1)) + (2.0 * series)) / 10.0


def accel5(series: pd.Series) -> pd.Series:
    return (series - 2.0 * series.shift(2) + series.shift(4)) / 2.0


def build_pattern_frame(bars: pd.DataFrame, base_features: List[str]) -> pd.DataFrame:
    t0 = time.time()

    bars = bars.copy().sort_values("timestamp").reset_index(drop=True)
    bars["ts_min"] = pd.to_datetime(bars["timestamp"]).dt.floor("min")

    print("Computing base features...", flush=True)
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

    # Minimal price context
    df["px__ret1m_close"] = pd.to_numeric(bars["close"], errors="coerce").pct_change() * 100.0
    df["px__range_norm1m"] = (pd.to_numeric(bars["high"], errors="coerce") - pd.to_numeric(bars["low"], errors="coerce")) / pd.to_numeric(
        bars["close"], errors="coerce"
    )

    print(f"Pattern frame built in {(time.time() - t0):.1f}s with {df.shape[1]-1} derived columns", flush=True)
    return df


def oracle_best_ret_and_k_nextopen(
    open_arr: np.ndarray,
    close_arr: np.ndarray,
    signal_is: np.ndarray,
    hold_min: int,
    fee_side: float,
) -> Tuple[np.ndarray, np.ndarray]:
    entry_idx = signal_is + 1
    entry_px = open_arr[entry_idx]

    best_ret = np.full(len(signal_is), -1e18, dtype=np.float64)
    best_k = np.ones(len(signal_is), dtype=np.int16)

    for k in range(1, int(hold_min) + 1):
        exit_idx = entry_idx + k
        ret = net_return_pct_vec(entry_px, close_arr[exit_idx], fee_side)
        better = ret > best_ret
        if np.any(better):
            best_ret[better] = ret[better]
            best_k[better] = int(k)

    return best_ret.astype(np.float32), best_k


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


def simulate_sweep(
    signal_is: np.ndarray,
    scores: np.ndarray,
    best_ret: np.ndarray,
    best_k: np.ndarray,
    ts_utc: pd.Series,
    dates: np.ndarray,
    lookback_days: int,
    min_prior_candidates_requested: int,
    target_fracs: List[float],
) -> pd.DataFrame:
    # Precompute day -> indices slice for speed
    unique_days: List[object] = []
    seen = set()
    for i in signal_is:
        d = dates[int(i)]
        if d not in seen:
            seen.add(d)
            unique_days.append(d)

    # Map i->(score,ret,k) for fast lookup
    score_by_i: Dict[int, float] = {int(i): float(s) for i, s in zip(signal_is, scores)}
    ret_by_i: Dict[int, float] = {int(i): float(r) for i, r in zip(signal_is, best_ret)}
    k_by_i: Dict[int, int] = {int(i): int(k) for i, k in zip(signal_is, best_k)}

    day_to_is: Dict[object, List[int]] = {}
    for d in unique_days:
        day_to_is[d] = [int(i) for i in signal_is[dates[signal_is] == d]]

    window_cap = int(lookback_days) * 1440
    effective_min_prior = int(min(int(min_prior_candidates_requested), max(2000, int(0.5 * window_cap))))

    rows: List[Dict[str, float]] = []

    for target_frac in target_fracs:
        if not (0.0 < float(target_frac) < 1.0):
            raise ValueError("target_frac must be in (0,1)")

        prior_days_scores: deque[list[float]] = deque(maxlen=int(lookback_days))
        total_candidates = 0
        taken = 0
        rets: List[float] = []
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

            day_candidates: list[float] = []
            for i in day_to_is[d]:
                if i < int(blocked_until_i):
                    continue

                s = float(score_by_i.get(i, float("nan")))
                if not np.isfinite(s):
                    continue

                total_candidates += 1
                day_candidates.append(s)

                if s >= thr:
                    taken += 1
                    rets.append(float(ret_by_i[i]))
                    exit_idx = int(i) + 1 + int(k_by_i[i])
                    blocked_until_i = int(exit_idx)

            prior_days_scores.append(day_candidates)

        out: Dict[str, float] = {
            "target_frac": float(target_frac),
            "effective_min_prior": float(effective_min_prior),
            "total_candidates": float(total_candidates),
            "n_trades": float(taken),
            "take_rate_pct": float((taken / max(1, total_candidates)) * 100.0),
        }
        if len(rets) > 0:
            arr = np.asarray(rets, dtype=np.float64)
            out.update(
                {
                    "mean_oracle_ret_pct": float(arr.mean()),
                    "median_oracle_ret_pct": float(np.median(arr)),
                    "win_rate_gt0_pct": float(np.mean(arr > 0.0) * 100.0),
                }
            )
        else:
            out.update(
                {
                    "mean_oracle_ret_pct": float("nan"),
                    "median_oracle_ret_pct": float("nan"),
                    "win_rate_gt0_pct": float("nan"),
                }
            )

        rows.append(out)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pattern entry regressor (500 trees) + oracle-exit sweep")
    ap.add_argument("--market-csv", required=True)
    ap.add_argument("--hold-min", type=int, default=10)
    ap.add_argument("--fee-roundtrip-pct", type=float, default=0.1)

    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--trees", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--progress-period", type=int, default=25)

    ap.add_argument("--lookback-days", type=int, default=14)
    ap.add_argument("--min-prior-candidates", type=int, default=50_000)
    ap.add_argument(
        "--target-fracs",
        default="0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001",
        help="Comma-separated fractions (0.01=1%, 0.0001=0.01%)",
    )

    ap.add_argument("--out-dir", default="data/pattern_entry_regressor")

    args = ap.parse_args()

    fee_side = fee_side_from_roundtrip_pct(float(args.fee_roundtrip_pct))
    fee_mult = (1.0 - fee_side) / (1.0 + fee_side)
    print(f"Fees: roundtrip={float(args.fee_roundtrip_pct):.6f}% -> fee_side={fee_side:.8f} (fee_mult={fee_mult:.6f})")

    t0 = time.time()
    bars = pd.read_csv(args.market_csv, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    print(f"Bars: {len(bars)}  Range: {bars['timestamp'].min()} → {bars['timestamp'].max()}")

    # Same base features as minimal classifier approach
    base_feats = ["macd", "ret_1m_pct", "mom_5m_pct", "vol_std_5m", "range_norm_5m"]

    pat = build_pattern_frame(bars, base_features=base_feats)
    ts_utc = pat["timestamp"]
    dates = ts_utc.dt.date.to_numpy()

    open_arr = bars["open"].to_numpy(np.float64)
    close_arr = bars["close"].to_numpy(np.float64)

    min_i = 4
    last_i = len(bars) - 2 - int(args.hold_min)
    signal_is = np.arange(min_i, last_i + 1, dtype=np.int64)

    print("Precomputing oracle best_ret + best_k...", flush=True)
    best_ret, best_k = oracle_best_ret_and_k_nextopen(
        open_arr=open_arr,
        close_arr=close_arr,
        signal_is=signal_is,
        hold_min=int(args.hold_min),
        fee_side=float(fee_side),
    )

    feat_cols = [c for c in pat.columns if c != "timestamp"]
    X_df = pat.loc[signal_is, feat_cols]

    # Drop NaNs
    good = np.isfinite(X_df.to_numpy(np.float64)).all(axis=1) & np.isfinite(best_ret)
    signal_is = signal_is[np.where(good)[0]]
    X = X_df.iloc[np.where(good)[0]].to_numpy(np.float32)
    y = best_ret[np.where(good)[0]].astype(np.float32)
    best_k = best_k[np.where(good)[0]]

    print(f"Samples: {len(y)}  y_mean={float(np.mean(y)):.4f}%  y_median={float(np.median(y)):.4f}%")
    print(f"Features: {len(feat_cols)}")

    test_frac = float(args.test_frac)
    if not (0.0 < test_frac < 1.0):
        raise ValueError("--test-frac must be in (0,1)")

    split = int(len(y) * (1.0 - test_frac))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    print(f"Train n={len(y_tr)}  Test n={len(y_te)}")

    print("Training LightGBM regressor (progress+ETA)...", flush=True)
    model = LGBMRegressor(
        n_estimators=int(args.trees),
        learning_rate=float(args.learning_rate),
        max_depth=10,
        num_leaves=128,
        min_child_samples=100,
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

    pred_te = model.predict(X_te)
    pred_tr = model.predict(X_tr)

    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def mae(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return float(np.mean(np.abs(a - b)))

    print("=== Regressor metrics ===")
    print(f"Train RMSE: {rmse(pred_tr, y_tr):.4f}%  MAE: {mae(pred_tr, y_tr):.4f}%")
    print(f"Test  RMSE: {rmse(pred_te, y_te):.4f}%  MAE: {mae(pred_te, y_te):.4f}%")
    if len(y_te) > 10:
        pear = float(np.corrcoef(pred_te.astype(np.float64), y_te.astype(np.float64))[0, 1])
        print(f"Test Pearson corr: {pear:.4f}")

    # Score all minutes for sweep
    print("Scoring all minutes for sweep...", flush=True)
    scores = model.predict(X).astype(np.float64)

    target_fracs = [float(x.strip()) for x in str(args.target_fracs).split(",") if x.strip()]

    print("\n=== Oracle-exit sweep (top X% of candidate minutes) ===")
    df = simulate_sweep(
        signal_is=signal_is,
        scores=scores,
        best_ret=y,
        best_k=best_k,
        ts_utc=ts_utc,
        dates=dates,
        lookback_days=int(args.lookback_days),
        min_prior_candidates_requested=int(args.min_prior_candidates),
        target_fracs=target_fracs,
    )

    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(df.to_string(index=False))

    out_dir = REPO_ROOT / str(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = now_ts()

    sweep_path = out_dir / f"pattern_entry_regressor_sweep_{ts}.csv"
    df.to_csv(sweep_path, index=False)

    model_path = out_dir / f"pattern_entry_regressor_{ts}.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_cols": feat_cols,
            "base_features": base_feats,
            "label": "oracle_best_ret_nextopen_pct",
            "context": {
                "market_csv": str(Path(args.market_csv).resolve()),
                "hold_min": int(args.hold_min),
                "fee_roundtrip_pct": float(args.fee_roundtrip_pct),
                "fee_side": float(fee_side),
                "trees": int(args.trees),
                "learning_rate": float(args.learning_rate),
                "lookback_days": int(args.lookback_days),
                "min_prior_candidates": int(args.min_prior_candidates),
                "target_fracs": [float(x) for x in target_fracs],
            },
            "metrics": {
                "train_rmse": float(rmse(pred_tr, y_tr)),
                "test_rmse": float(rmse(pred_te, y_te)),
                "train_mae": float(mae(pred_tr, y_tr)),
                "test_mae": float(mae(pred_te, y_te)),
                "test_pearson": float(np.corrcoef(pred_te.astype(np.float64), y_te.astype(np.float64))[0, 1]) if len(y_te) > 10 else float("nan"),
            },
            "created_utc": ts,
        },
        model_path,
    )

    print(f"\nSaved sweep CSV: {sweep_path}")
    print(f"Saved model:     {model_path}")
    print(f"Total elapsed:   {(time.time() - t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
