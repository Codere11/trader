#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-02T22:00:41Z
"""Evaluate a causal live-style entry scorer with oracle exits.

Policy:
- Decision time: minute CLOSE at index i (uses features up through candle i)
- Execution: enter at next minute OPEN (index i+1)
- Candidate minutes ("would take"): when flat AND score >= min_score
- Quota: take only the top target_frac of candidate minutes (implemented as a per-day threshold computed from PRIOR days' candidate scores)
- Exit: oracle best net return within next hold_min minutes (entry at open, exit at close)

NO FUTURE LEAKAGE:
- Features for scoring at i use only <= i.
- Per-day threshold uses only prior days' candidate scores.

Outputs:
- Trades CSV with oracle exit details
- Daily aggregates
"""

from __future__ import annotations

import argparse
import sys
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


def net_return_pct(entry_px: float, exit_px: float, fee_side: float) -> float:
    """Net return % at 1x (accounts for entry and exit fees)."""
    gross_mult = float(exit_px) / max(1e-12, float(entry_px))
    net_mult = gross_mult * (1.0 - fee_side) / (1.0 + fee_side)
    return float((net_mult - 1.0) * 100.0)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    offset: int  # relative to decision index i (0 = candle i)
    kind: str  # 'open' | 'close' | 'feat'
    feat_name: str | None  # for kind='feat'


def parse_feature_specs(feature_names: List[str]) -> List[FeatureSpec]:
    specs: List[FeatureSpec] = []
    for n in feature_names:
        if n.startswith("pre"):
            rest = n[len("pre") :]
            # e.g. "5_macd" or "1_open"
            k_str, suffix = rest.split("_", 1)
            k = int(k_str)
            offset = -k
            if suffix in ("open", "close"):
                specs.append(FeatureSpec(name=n, offset=offset, kind=suffix, feat_name=None))
            else:
                specs.append(FeatureSpec(name=n, offset=offset, kind="feat", feat_name=suffix))
        elif n.startswith("entry_"):
            suffix = n[len("entry_") :]
            if suffix in ("open", "close"):
                # not expected, but allow
                specs.append(FeatureSpec(name=n, offset=0, kind=suffix, feat_name=None))
            else:
                specs.append(FeatureSpec(name=n, offset=0, kind="feat", feat_name=suffix))
        else:
            raise ValueError(f"Unrecognized feature name pattern: {n}")
    return specs


def oracle_best_exit(
    entry_idx: int,
    open_arr: np.ndarray,
    close_arr: np.ndarray,
    fee_side: float,
    hold_min: int,
) -> Tuple[float, int, int, float]:
    """Return (best_ret_pct, best_k, exit_idx, exit_px)."""
    entry_px = float(open_arr[entry_idx])
    best_ret = -1e18
    best_k = 1
    best_exit_idx = entry_idx + 1
    best_exit_px = float(close_arr[best_exit_idx])

    for k in range(1, int(hold_min) + 1):
        exit_idx = entry_idx + k
        exit_px = float(close_arr[exit_idx])
        ret = net_return_pct(entry_px, exit_px, fee_side)
        if ret > best_ret:
            best_ret = ret
            best_k = k
            best_exit_idx = exit_idx
            best_exit_px = exit_px

    return float(best_ret), int(best_k), int(best_exit_idx), float(best_exit_px)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate live-style entry scorer with oracle exits")
    ap.add_argument("--market-csv", required=True, help="CSV with timestamp, open, high, low, close, volume")
    ap.add_argument(
        "--entry-model",
        default=str(REPO_ROOT / "models" / "entry_regressor_dydx_oracle_2026-01-02T21-47-15Z.joblib"),
        help="Joblib artifact containing {'model','features','label',...}",
    )
    ap.add_argument("--target-frac", type=float, default=0.01, help="Take top X fraction of candidate minutes (e.g. 0.01 = top 1%)")
    ap.add_argument("--min-score", type=float, default=0.0, help="Candidate (would-take) condition: score >= min_score")
    ap.add_argument(
        "--min-prior-candidates",
        type=int,
        default=10_000,
        help="Require at least this many prior candidate scores before taking any trades (avoids day-1 accept-all).",
    )
    ap.add_argument("--hold-min", type=int, default=10, help="Oracle exit horizon in minutes")
    ap.add_argument("--fee-side", type=float, default=0.001, help="Per-side fee (0.001 = 0.1%)")
    ap.add_argument("--out-dir", default="data/entry_oracle_eval")

    args = ap.parse_args()

    bars = pd.read_csv(args.market_csv, parse_dates=["timestamp"])  # expected 1-min candles
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    bars["ts_min"] = pd.to_datetime(bars["timestamp"]).dt.floor("min")

    n = len(bars)
    if n < 100:
        raise ValueError("Not enough bars")

    art = joblib.load(args.entry_model)
    model = art["model"]
    feature_names = list(art["features"])

    specs = parse_feature_specs(feature_names)

    # Compute features (per-candle)
    base = compute_feature_frame(bars)
    src = base[[c for c in FEATURES if c in base.columns]]

    open_arr = bars["open"].to_numpy(np.float64)
    close_arr = bars["close"].to_numpy(np.float64)
    ts_arr = pd.to_datetime(bars["timestamp"], utc=True)

    feat_arrs: Dict[str, np.ndarray] = {}
    for feat in src.columns:
        feat_arrs[str(feat)] = src[feat].to_numpy(np.float64)

    # Validate specs
    for sp in specs:
        if sp.kind == "feat":
            assert sp.feat_name is not None
            if sp.feat_name not in feat_arrs:
                raise ValueError(f"Model expects feature '{sp.feat_name}' but it was not found in computed features")

    # Determine index bounds: need pre context, need entry at i+1, need hold_min future closes
    pre_min = 0
    for sp in specs:
        if sp.offset < 0:
            pre_min = max(pre_min, -sp.offset)

    hold_min = int(args.hold_min)
    last_decision_i = n - 2 - hold_min
    if last_decision_i <= pre_min:
        raise ValueError("Not enough bars for requested pre-context and hold horizon")

    decision_is = np.arange(pre_min, last_decision_i + 1, dtype=np.int64)

    # Precompute scores in batches (still no leakage; just efficiency)
    scores = np.full(len(decision_is), np.nan, dtype=np.float64)
    batch = 50_000

    for b0 in range(0, len(decision_is), batch):
        b1 = min(len(decision_is), b0 + batch)
        idxs = decision_is[b0:b1]
        X = np.empty((len(idxs), len(specs)), dtype=np.float32)

        for j, sp in enumerate(specs):
            src_idx = idxs + int(sp.offset)
            if sp.kind == "open":
                X[:, j] = open_arr[src_idx].astype(np.float32)
            elif sp.kind == "close":
                X[:, j] = close_arr[src_idx].astype(np.float32)
            else:
                X[:, j] = feat_arrs[sp.feat_name][src_idx].astype(np.float32)  # type: ignore[index]

        # mark rows with any NaN as NaN score
        good = np.isfinite(X).all(axis=1)
        if np.any(good):
            pred = model.predict(X[good])
            scores[b0:b1][good] = pred

    # Simulation with per-day causal threshold based on prior days' candidate scores
    target_frac = float(args.target_frac)
    if not (0.0 < target_frac < 1.0):
        raise ValueError("--target-frac must be in (0,1)")

    min_score = float(args.min_score)
    fee_side = float(args.fee_side)

    # Map decision index -> score
    score_by_i: Dict[int, float] = {int(i): float(s) for i, s in zip(decision_is, scores)}

    # Group decision indices by date (UTC)
    dates = ts_arr.dt.date.to_numpy()
    unique_dates = []
    seen = set()
    for i in decision_is:
        d = dates[int(i)]
        if d not in seen:
            seen.add(d)
            unique_dates.append(d)

    prior_candidate_scores: List[float] = []
    trades: List[Dict[str, object]] = []

    blocked_until_i = -10**18  # decision indices < this are in-trade (trade exits at close of exit_idx)

    min_prior = int(args.min_prior_candidates)

    for d in unique_dates:
        # Causal threshold computed from *prior* days' candidate pool.
        # If we don't have enough prior candidates yet, don't take trades.
        if len(prior_candidate_scores) >= min_prior:
            thr = float(np.quantile(np.asarray(prior_candidate_scores, dtype=np.float64), 1.0 - target_frac))
        else:
            thr = float("inf")

        day_candidate_scores: List[float] = []

        # iterate decisions in this day
        # (decision indices are contiguous in time, so scanning full range is fine)
        # find start/end for this day
        day_mask = (dates[decision_is] == d)
        day_is = decision_is[day_mask]

        for i in day_is:
            i_int = int(i)

            # if trade still open (exit at close of blocked_until_i), skip this decision minute
            if i_int < int(blocked_until_i):
                continue

            s = float(score_by_i.get(i_int, float("nan")))
            if not np.isfinite(s):
                continue

            # candidate pool = minutes we would take if unconstrained
            if s >= min_score:
                day_candidate_scores.append(s)

                # quota: only take top target_frac of candidates (implemented via thr)
                if s >= thr:
                    entry_idx = i_int + 1
                    entry_time = ts_arr.iloc[entry_idx]
                    entry_px = float(open_arr[entry_idx])

                    best_ret, best_k, exit_idx, exit_px = oracle_best_exit(
                        entry_idx=entry_idx,
                        open_arr=open_arr,
                        close_arr=close_arr,
                        fee_side=fee_side,
                        hold_min=hold_min,
                    )

                    exit_time = ts_arr.iloc[exit_idx]

                    trades.append(
                        {
                            "signal_time": ts_arr.iloc[i_int],
                            "signal_index": i_int,
                            "entry_time": entry_time,
                            "entry_index": entry_idx,
                            "entry_open": entry_px,
                            "exit_time": exit_time,
                            "exit_index": exit_idx,
                            "exit_close": exit_px,
                            "oracle_best_k": int(best_k),
                            "oracle_best_ret_pct": float(best_ret),
                            "score": float(s),
                            "threshold": float(thr),
                            "min_score": float(min_score),
                            "target_frac": float(target_frac),
                            "date": d,
                        }
                    )

                    # Block new entries until exit candle closes (i == exit_idx is allowed).
                    blocked_until_i = int(exit_idx)

        # Update prior score pool with this day's candidates (causal: used for future days only)
        prior_candidate_scores.extend(day_candidate_scores)

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = now_ts()

    trades_df = pd.DataFrame(trades)
    trades_path = out_dir / f"eval_entry_oracle_exit_trades_{ts}.csv"
    trades_df.to_csv(trades_path, index=False)

    if len(trades_df) == 0:
        print("No trades produced.")
        print(f"Wrote empty trades file: {trades_path}")
        return

    daily = (
        trades_df.groupby("date", as_index=False)
        .agg(
            n_trades=("oracle_best_ret_pct", "size"),
            mean_oracle_ret_pct=("oracle_best_ret_pct", "mean"),
            median_oracle_ret_pct=("oracle_best_ret_pct", "median"),
            p25_oracle_ret_pct=("oracle_best_ret_pct", lambda x: float(np.percentile(np.asarray(x), 25))),
            p75_oracle_ret_pct=("oracle_best_ret_pct", lambda x: float(np.percentile(np.asarray(x), 75))),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily_path = out_dir / f"eval_entry_oracle_exit_daily_{ts}.csv"
    daily.to_csv(daily_path, index=False)

    # headline stats
    rets = trades_df["oracle_best_ret_pct"].to_numpy(np.float64)
    print("=== Entry scorer + oracle exit evaluation ===")
    print(f"Model: {args.entry_model}")
    print(f"Market: {args.market_csv}")
    print(f"Trades: {len(trades_df)}")
    print(f"Mean oracle best ret: {float(np.mean(rets)):.4f}%")
    print(f"Median oracle best ret: {float(np.median(rets)):.4f}%")
    print(f"Win rate (oracle best ret > 0): {float(np.mean(rets > 0.0) * 100.0):.2f}%")
    print(f"Win rate (oracle best ret >= 0.17): {float(np.mean(rets >= 0.17) * 100.0):.2f}%")

    # quota realized vs candidate pool (in-sim)
    # candidates are those minutes we were flat, had finite score, and s >= min_score; but we only tracked per-day candidates.
    # reconstruct candidate count approximately: count of all scores >= min_score during flat minutes is expensive to recompute here;
    # however, prior_candidate_scores holds all candidate scores observed across days.
    total_candidates = len(prior_candidate_scores)
    if total_candidates > 0:
        print(f"Total candidate minutes (flat, score>=min_score): {total_candidates}")
        print(f"Take rate among candidates: {len(trades_df) / total_candidates * 100.0:.3f}% (target {target_frac*100:.3f}%)")

    print(f"\nTrades CSV: {trades_path}")
    print(f"Daily CSV:  {daily_path}")


if __name__ == "__main__":
    main()
