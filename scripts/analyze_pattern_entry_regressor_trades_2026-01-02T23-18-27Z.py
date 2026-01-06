#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-02T23:18:27Z
"""Analyze pattern-entry regressor selection: biggest wins/losses + crucial differences.

This script:
- Loads a trained pattern-entry regressor artifact (LightGBM) with base_features + feature_cols
- Rebuilds the exact pattern feature frame on a market CSV
- Computes oracle best return (entry next open, exit best close within hold_min)
- Simulates "take top target_frac of candidate minutes" with causal rolling threshold + blocking
- Saves per-trade CSV including features at signal time
- Prints:
  - Headline stats
  - Biggest wins/losses
  - Separation analysis: which pattern features differ most between top vs bottom quintile of realized oracle best return

NO LOOKAHEAD:
- Features use only candles <= signal index i (pattern window ends at i)
- Entry at i+1 open
- Oracle uses future closes i+1..i+1+hold_min
- Threshold for day D uses only prior days' candidate scores (rolling window)
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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import FEATURES, compute_feature_frame


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def zscore_roll(s: pd.Series, win: int) -> pd.Series:
    mu = s.rolling(int(win), min_periods=int(win)).mean()
    sd = s.rolling(int(win), min_periods=int(win)).std()
    return (s - mu) / (sd + 1e-12)


def net_return_pct_vec(entry_px: np.ndarray, exit_px: np.ndarray, fee_side: float) -> np.ndarray:
    entry_px = np.maximum(entry_px.astype(np.float64), 1e-12)
    exit_px = exit_px.astype(np.float64)
    gross_mult = exit_px / entry_px
    net_mult = gross_mult * (1.0 - float(fee_side)) / (1.0 + float(fee_side))
    return (net_mult - 1.0) * 100.0


def slope5(series: pd.Series) -> pd.Series:
    return ((-2.0 * series.shift(4)) + (-1.0 * series.shift(3)) + (1.0 * series.shift(1)) + (2.0 * series)) / 10.0


def accel5(series: pd.Series) -> pd.Series:
    return (series - 2.0 * series.shift(2) + series.shift(4)) / 2.0


def build_pattern_frame(bars: pd.DataFrame, base_features: List[str]) -> pd.DataFrame:
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

    df["px__ret1m_close"] = pd.to_numeric(bars["close"], errors="coerce").pct_change() * 100.0
    df["px__range_norm1m"] = (pd.to_numeric(bars["high"], errors="coerce") - pd.to_numeric(bars["low"], errors="coerce")) / pd.to_numeric(
        bars["close"], errors="coerce"
    )

    return df


def build_pattern_frame_v2(bars: pd.DataFrame, base_features: List[str]) -> pd.DataFrame:
    """Same as v1 + spike/pump/vol-regime indicators (must match v2 training script)."""
    df = build_pattern_frame(bars, base_features=base_features)

    # We need access to OHLC for extra indicators.
    close = pd.to_numeric(bars["close"], errors="coerce")
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")

    df["px__ret1m_abs"] = df["px__ret1m_close"].abs()
    df["px__range_norm1m_abs"] = df["px__range_norm1m"].abs()

    win = 1440
    df["z1d__px_ret1m"] = zscore_roll(df["px__ret1m_close"], win)
    df["z1d__px_ret1m_abs"] = df["z1d__px_ret1m"].abs()
    df["z1d__px_range1m"] = zscore_roll(df["px__range_norm1m"], win)
    df["z1d__px_range1m_abs"] = df["z1d__px_range1m"].abs()

    # Context baselines (computed from already-present pattern cols)
    if "vol_std_5m__last" in df.columns:
        df["z1d__vol5"] = zscore_roll(pd.to_numeric(df["vol_std_5m__last"], errors="coerce"), win)
        df["risk__ret1m_abs_over_vol5"] = df["px__ret1m_abs"] / (pd.to_numeric(df["vol_std_5m__last"], errors="coerce").abs() + 1e-9)
        df["risk__range1m_over_vol5"] = df["px__range_norm1m"] / (pd.to_numeric(df["vol_std_5m__last"], errors="coerce").abs() + 1e-9)

    if "range_norm_5m__max5" in df.columns:
        df["risk__range1m_over_range5max"] = df["px__range_norm1m"] / (pd.to_numeric(df["range_norm_5m__max5"], errors="coerce").abs() + 1e-12)

    if "ret_1m_pct__last" in df.columns:
        df["ret_1m_pct__last_pos"] = pd.to_numeric(df["ret_1m_pct__last"], errors="coerce").clip(lower=0.0)
        df["ret_1m_pct__last_neg"] = (-pd.to_numeric(df["ret_1m_pct__last"], errors="coerce")).clip(lower=0.0)

    df["flag__ret1m_abs_z_gt3"] = (df["z1d__px_ret1m_abs"] > 3.0).astype(np.float32)
    df["flag__range1m_abs_z_gt3"] = (df["z1d__px_range1m_abs"] > 3.0).astype(np.float32)

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


def simulate_trades(
    signal_is: np.ndarray,
    scores: np.ndarray,
    best_ret: np.ndarray,
    best_k: np.ndarray,
    ts_utc: pd.Series,
    dates: np.ndarray,
    target_frac: float,
    lookback_days: int,
    min_prior_candidates_requested: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if not (0.0 < float(target_frac) < 1.0):
        raise ValueError("target_frac must be in (0,1)")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be >= 1")

    # Attainable bootstrap
    window_cap = int(lookback_days) * 1440
    effective_min_prior = int(min(int(min_prior_candidates_requested), max(2000, int(0.5 * window_cap))))

    score_by_i: Dict[int, float] = {int(i): float(s) for i, s in zip(signal_is, scores)}
    ret_by_i: Dict[int, float] = {int(i): float(r) for i, r in zip(signal_is, best_ret)}
    k_by_i: Dict[int, int] = {int(i): int(k) for i, k in zip(signal_is, best_k)}

    unique_days: List[object] = []
    seen = set()
    for i in signal_is:
        d = dates[int(i)]
        if d not in seen:
            seen.add(d)
            unique_days.append(d)

    # Precompute day -> indices
    day_to_is: Dict[object, List[int]] = {}
    for d in unique_days:
        day_to_is[d] = [int(i) for i in signal_is[dates[signal_is] == d]]

    prior_days_scores: deque[list[float]] = deque(maxlen=int(lookback_days))

    trades: List[Dict[str, object]] = []
    total_candidates = 0
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
                k = int(k_by_i[i])
                entry_idx = int(i) + 1
                exit_idx = entry_idx + k

                trades.append(
                    {
                        "signal_time": ts_utc.iloc[int(i)],
                        "signal_index": int(i),
                        "entry_time": ts_utc.iloc[int(entry_idx)],
                        "entry_index": int(entry_idx),
                        "exit_time": ts_utc.iloc[int(exit_idx)],
                        "exit_index": int(exit_idx),
                        "oracle_best_k": int(k),
                        "oracle_best_ret_pct": float(ret_by_i[i]),
                        "score": float(s),
                        "threshold": float(thr) if np.isfinite(thr) else thr,
                        "date": d,
                    }
                )

                blocked_until_i = int(exit_idx)

        prior_days_scores.append(day_candidates)

    df = pd.DataFrame(trades)
    rets = df["oracle_best_ret_pct"].to_numpy(np.float64) if len(df) else np.asarray([], dtype=np.float64)

    summary = {
        "target_frac": float(target_frac),
        "lookback_days": float(lookback_days),
        "effective_min_prior": float(effective_min_prior),
        "total_candidates": float(total_candidates),
        "n_trades": float(len(df)),
        "take_rate_pct": float((len(df) / max(1, total_candidates)) * 100.0),
        "mean_oracle_ret_pct": float(rets.mean()) if len(rets) else float("nan"),
        "median_oracle_ret_pct": float(np.median(rets)) if len(rets) else float("nan"),
        "win_rate_gt0_pct": float(np.mean(rets > 0.0) * 100.0) if len(rets) else float("nan"),
        "share_ge_0p17_pct": float(np.mean(rets >= 0.17) * 100.0) if len(rets) else float("nan"),
    }

    return df, summary


def separation_report(trades: pd.DataFrame, feat_df: pd.DataFrame, top_q: float = 0.2, bottom_q: float = 0.2, top_k: int = 12) -> pd.DataFrame:
    """Compare feature distributions between best and worst trades by realized oracle_best_ret_pct."""
    if len(trades) < 50:
        return pd.DataFrame()

    t = trades.copy()
    t = t.sort_values("oracle_best_ret_pct").reset_index(drop=True)
    n = len(t)
    n_bot = max(1, int(n * bottom_q))
    n_top = max(1, int(n * top_q))

    bot = t.iloc[:n_bot]
    top = t.iloc[-n_top:]

    bot_X = feat_df.loc[bot["signal_index"].to_numpy(np.int64)]
    top_X = feat_df.loc[top["signal_index"].to_numpy(np.int64)]

    rows = []
    for c in feat_df.columns:
        a = bot_X[c].to_numpy(np.float64)
        b = top_X[c].to_numpy(np.float64)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 10 or len(b) < 10:
            continue

        ma, mb = float(a.mean()), float(b.mean())
        sa, sb = float(a.std()), float(b.std())
        pooled = (sa + sb) / 2.0
        effect = (mb - ma) / pooled if pooled > 1e-12 else float("nan")
        rows.append(
            {
                "feature": c,
                "mean_bottom": ma,
                "mean_top": mb,
                "delta_top_minus_bottom": (mb - ma),
                "effect_size": effect,
            }
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    out = out.sort_values("effect_size", ascending=False).head(int(top_k)).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze pattern-entry regressor trades")
    ap.add_argument("--market-csv", required=True)
    ap.add_argument("--model-artifact", required=True)
    ap.add_argument("--hold-min", type=int, default=10)
    ap.add_argument("--target-frac", type=float, default=0.001, help="Top X% of candidate minutes")
    ap.add_argument("--lookback-days", type=int, default=14)
    ap.add_argument("--min-prior-candidates", type=int, default=50_000)
    ap.add_argument("--out-dir", default="data/pattern_entry_regressor")
    ap.add_argument("--top-n", type=int, default=20)

    args = ap.parse_args()

    art = joblib.load(args.model_artifact)
    model = art["model"]
    base_feats = list(art["base_features"])
    feat_cols = list(art["feature_cols"])
    ctx = art.get("context", {})

    fee_side = float(ctx.get("fee_side"))

    print(f"Model: {args.model_artifact}")
    print(f"target_frac={float(args.target_frac)} ({float(args.target_frac)*100:.4f}%) lookback_days={int(args.lookback_days)}")
    print(f"fee_side={fee_side}")

    t0 = time.time()
    bars = pd.read_csv(args.market_csv, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    print(f"Bars: {len(bars)}")

    # Choose pattern builder based on what the model expects.
    needs_v2 = any(
        k in feat_cols
        for k in [
            "z1d__px_ret1m",
            "z1d__px_range1m",
            "risk__range1m_over_vol5",
            "flag__ret1m_abs_z_gt3",
        ]
    )

    if needs_v2:
        pat = build_pattern_frame_v2(bars, base_features=base_feats)
    else:
        pat = build_pattern_frame(bars, base_features=base_feats)

    ts_utc = pat["timestamp"]
    dates = ts_utc.dt.date.to_numpy()

    open_arr = bars["open"].to_numpy(np.float64)
    close_arr = bars["close"].to_numpy(np.float64)

    # Need 5-min patterns and, for v2, 1-day z-score window.
    min_i = 1440 if needs_v2 else 4
    last_i = len(bars) - 2 - int(args.hold_min)
    signal_is = np.arange(int(min_i), last_i + 1, dtype=np.int64)

    best_ret, best_k = oracle_best_ret_and_k_nextopen(
        open_arr=open_arr,
        close_arr=close_arr,
        signal_is=signal_is,
        hold_min=int(args.hold_min),
        fee_side=fee_side,
    )

    X_df = pat.loc[signal_is, feat_cols]
    good = np.isfinite(X_df.to_numpy(np.float64)).all(axis=1) & np.isfinite(best_ret)
    signal_is = signal_is[np.where(good)[0]]
    X_df = X_df.iloc[np.where(good)[0]].reset_index(drop=True)
    best_ret = best_ret[np.where(good)[0]]
    best_k = best_k[np.where(good)[0]]

    X = X_df.to_numpy(np.float32)

    print("Scoring...", flush=True)
    scores = model.predict(X).astype(np.float64)

    # Build index->row mapping back to original signal index for analysis.
    # We'll keep a feature frame indexed by original bar index.
    feat_by_signal_index = pd.DataFrame(
        X_df.to_numpy(np.float64),
        columns=feat_cols,
        index=signal_is,
    )

    trades, summ = simulate_trades(
        signal_is=signal_is,
        scores=scores,
        best_ret=best_ret,
        best_k=best_k,
        ts_utc=ts_utc,
        dates=dates,
        target_frac=float(args.target_frac),
        lookback_days=int(args.lookback_days),
        min_prior_candidates_requested=int(args.min_prior_candidates),
    )

    if len(trades) == 0:
        print("No trades produced.")
        return

    # Attach features for each trade (signal time)
    trade_feats = feat_by_signal_index.loc[trades["signal_index"].to_numpy(np.int64)].copy()
    trade_feats["oracle_best_ret_pct"] = trades["oracle_best_ret_pct"].to_numpy(np.float64)
    trade_feats["score"] = trades["score"].to_numpy(np.float64)

    out_dir = REPO_ROOT / str(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = now_ts()

    trades_path = out_dir / f"pattern_entry_regressor_trades_target{float(args.target_frac):.6f}_{ts}.csv"
    trades_out = pd.concat([trades.reset_index(drop=True), trade_feats.reset_index(drop=True)], axis=1)
    trades_out.to_csv(trades_path, index=False)

    print("=== Headline ===")
    for k, v in summ.items():
        print(f"  {k}: {v}")

    # Biggest wins/losses
    n_show = int(args.top_n)
    t_sorted = trades.sort_values("oracle_best_ret_pct", ascending=False).reset_index(drop=True)

    print(f"\n=== Biggest wins (top {n_show}) ===")
    print(t_sorted.head(n_show)[["signal_time", "oracle_best_ret_pct", "oracle_best_k", "score"]].to_string(index=False))

    print(f"\n=== Biggest losses (bottom {n_show}) ===")
    print(t_sorted.tail(n_show).sort_values("oracle_best_ret_pct")[["signal_time", "oracle_best_ret_pct", "oracle_best_k", "score"]].to_string(index=False))

    # Crucial differences: top vs bottom quintile by realized oracle_best_ret
    sep = separation_report(trades=trades, feat_df=feat_by_signal_index, top_q=0.2, bottom_q=0.2, top_k=15)
    if len(sep) > 0:
        print("\n=== Crucial differences (top 20% vs bottom 20% by realized oracle_best_ret_pct) ===")
        print(sep.to_string(index=False))

    print(f"\nSaved per-trade analysis CSV: {trades_path}")
    print(f"Elapsed: {(time.time() - t0):.1f}s")


if __name__ == "__main__":
    main()
