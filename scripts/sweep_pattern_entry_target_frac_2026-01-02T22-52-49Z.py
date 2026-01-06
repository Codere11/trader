#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-02T22:52:49Z
"""Sweep target_frac for the minimal pattern-based entry model (oracle-exit eval).

Loads the trained pattern-entry model artifact, rebuilds the same pattern feature frame,
computes scores for each signal minute (decide at close i, enter next open i+1), then
simulates trading with oracle exits while enforcing "take only the top X% of candidate minutes".

We sweep X down to 0.01%.

Causality:
- Threshold for day D is computed only from prior days' candidate scores (rolling window).
- Candidate minutes are those minutes where we're flat and have a finite score.
- Exit is oracle best within hold_min minutes; trade blocks new entries until exit candle closes.
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


def simulate(
    signal_is: np.ndarray,
    scores: np.ndarray,
    best_ret: np.ndarray,
    best_k: np.ndarray,
    ts_utc: pd.Series,
    dates: np.ndarray,
    hold_min: int,
    target_frac: float,
    lookback_days: int,
    min_prior_candidates_requested: int,
) -> Dict[str, float]:
    if not (0.0 < target_frac < 1.0):
        raise ValueError("target_frac must be in (0,1)")

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
        day_is = [int(i) for i in signal_is[dates[signal_is] == d]]

        for i in day_is:
            if i < int(blocked_until_i):
                continue

            s = float(score_by_i.get(i, float("nan")))
            if not np.isfinite(s):
                continue

            total_candidates += 1
            day_candidates.append(s)

            if s >= thr:
                taken += 1
                r = float(ret_by_i[i])
                rets.append(r)
                # entry at i+1; exit at i+1+best_k
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
                "win_rate_ge_0p17_pct": float(np.mean(arr >= 0.17) * 100.0),
            }
        )
    else:
        out.update(
            {
                "mean_oracle_ret_pct": float("nan"),
                "median_oracle_ret_pct": float("nan"),
                "win_rate_gt0_pct": float("nan"),
                "win_rate_ge_0p17_pct": float("nan"),
            }
        )

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep target_frac for pattern entry model")
    ap.add_argument("--market-csv", required=True)
    ap.add_argument("--model-artifact", required=True)
    ap.add_argument("--lookback-days", type=int, default=14)
    ap.add_argument("--min-prior-candidates", type=int, default=50_000)
    ap.add_argument(
        "--target-fracs",
        default="0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001",
        help="Comma-separated fractions (e.g. 0.01=1%, 0.0001=0.01%)",
    )
    ap.add_argument("--out-csv", default=None, help="Optional output CSV path")

    args = ap.parse_args()

    art = joblib.load(args.model_artifact)
    model = art["model"]
    feat_cols = list(art["feature_cols"])
    base_feats = list(art["base_features"])
    ctx = art.get("context", {})

    hold_min = int(ctx.get("hold_min", 10))
    fee_side = float(ctx.get("fee_side"))

    print(f"Model: {args.model_artifact}")
    print(f"hold_min={hold_min}  fee_side={fee_side}")

    t0 = time.time()
    bars = pd.read_csv(args.market_csv, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    print(f"Bars: {len(bars)}  Range: {bars['timestamp'].min()} → {bars['timestamp'].max()}")

    print("Building pattern frame...", flush=True)
    pat = build_pattern_frame(bars, base_features=base_feats)
    ts_utc = pat["timestamp"]
    dates = ts_utc.dt.date.to_numpy()

    open_arr = bars["open"].to_numpy(np.float64)
    close_arr = bars["close"].to_numpy(np.float64)

    min_i = 4
    last_i = len(bars) - 2 - hold_min
    signal_is = np.arange(min_i, last_i + 1, dtype=np.int64)

    X_df = pat.loc[signal_is, feat_cols]
    good = np.isfinite(X_df.to_numpy(np.float64)).all(axis=1)
    signal_is = signal_is[np.where(good)[0]]
    X = X_df.iloc[np.where(good)[0]].to_numpy(np.float32)

    print(f"Signal minutes: {len(signal_is)}")

    print("Scoring...", flush=True)
    scores = model.predict_proba(X)[:, 1].astype(np.float64)

    print("Precomputing oracle best_ret + best_k (for blocking + stats)...", flush=True)
    best_ret, best_k = oracle_best_ret_and_k_nextopen(
        open_arr=open_arr,
        close_arr=close_arr,
        signal_is=signal_is,
        hold_min=hold_min,
        fee_side=fee_side,
    )

    fracs = [float(x.strip()) for x in str(args.target_fracs).split(",") if x.strip()]
    rows: List[Dict[str, float]] = []

    print("\n=== Sweep (top X% of candidate minutes) ===")
    for f in fracs:
        print(f"\nSimulating target_frac={f} ({f*100:.4f}%)...", flush=True)
        row = simulate(
            signal_is=signal_is,
            scores=scores,
            best_ret=best_ret,
            best_k=best_k,
            ts_utc=ts_utc,
            dates=dates,
            hold_min=hold_min,
            target_frac=float(f),
            lookback_days=int(args.lookback_days),
            min_prior_candidates_requested=int(args.min_prior_candidates),
        )
        rows.append(row)
        print(
            "  n_trades={n_trades:.0f} take_rate={take_rate_pct:.3f}% mean={mean_oracle_ret_pct:.4f}% median={median_oracle_ret_pct:.4f}% win>0={win_rate_gt0_pct:.2f}%".format(
                **row
            ),
            flush=True,
        )

    df = pd.DataFrame(rows)
    # nicer ordering
    cols = [
        "target_frac",
        "n_trades",
        "take_rate_pct",
        "mean_oracle_ret_pct",
        "median_oracle_ret_pct",
        "win_rate_gt0_pct",
        "win_rate_ge_0p17_pct",
        "total_candidates",
        "effective_min_prior",
    ]
    df = df[[c for c in cols if c in df.columns]]

    print("\n=== Summary table ===")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 50, "display.width", 200):
        print(df.to_string(index=False))

    if args.out_csv:
        out_path = Path(args.out_csv)
    else:
        out_path = REPO_ROOT / "data" / "pattern_entry" / f"sweep_target_frac_{now_ts()}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nSaved sweep CSV: {out_path}")
    print(f"Total elapsed: {(time.time()-t0):.1f}s")


if __name__ == "__main__":
    main()
