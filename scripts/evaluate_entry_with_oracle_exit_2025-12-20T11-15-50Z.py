#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import sys
import numpy as np
import pandas as pd

# Ensure repo root is on sys.path so "scripts" is importable as a namespace package
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Reuse helpers from the ranker evaluation
from scripts import eval_rule_ranker_strategy as base


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate entry ranker with true oracle exits (best achievable minute within hold window)")
    ap.add_argument("--market-csv", default="data/btc_profitability_analysis_filtered.csv")
    ap.add_argument("--rule-trades", default="data/pattern_runs/rule_best/rule_rule__rn0.0015__vw0.000__mn1__m30.050__vwmin0.020__sl5.0__rsi55.0.parquet")
    ap.add_argument("--entry-models-dir", default="models")
    ap.add_argument("--entry-stem", default="rule_best_ranker")
    ap.add_argument("--entry-threshold", type=float, default=0.52)
    ap.add_argument("--k-per-day", type=int, default=3)
    ap.add_argument("--hold-min", type=int, default=8)
    ap.add_argument("--fee-rate", type=float, default=0.001)
    ap.add_argument("--start-date", type=str, default="2022-01-01")
    ap.add_argument("--end-date", type=str, default="2025-12-18")
    # Entry prefilters
    ap.add_argument("--min-vol-std-5m", type=float, default=None)
    ap.add_argument("--min-range-norm-5m", type=float, default=None)
    ap.add_argument("--rsi-min", type=float, default=None)
    ap.add_argument("--rsi-max", type=float, default=None)
    ap.add_argument("--max-vwap-dev-abs", type=float, default=None)
    ap.add_argument("--min-mom-3m-pct", type=float, default=None)
    ap.add_argument("--allow-hours", type=str, default=None)
    # Outputs
    ap.add_argument("--out-trades", type=str, default="data/entry_ranked_outputs/oracle_exit_true_trades.csv")
    ap.add_argument("--out-daily", type=str, default="data/entry_ranked_outputs/oracle_exit_true_daily.csv")
    ap.add_argument("--metrics-out", type=str, default="data/entry_ranked_outputs/oracle_exit_true_metrics.csv")
    ap.add_argument("--save-selected", type=str, default=None, help="Optional path to save the exact selected entries (entry_ts list)")
    return ap.parse_args()


def simulate_oracle(entries: pd.DataFrame, day_lookup: dict, hold_min: int, fee_rate: float) -> tuple[list[dict], list[dict]]:
    trades: list[dict] = []
    daily_returns: dict[object, float] = {}
    if entries.empty:
        return trades, [dict(date=None, daily_return_pct=0.0)]

    for row in entries.sort_values("entry_ts").itertuples(index=False):
        day = day_lookup.get(row.entry_date)
        if day is None or day.empty:
            continue
        ts_arr = day["timestamp"].to_numpy(dtype="datetime64[ns]")
        entry_ts = np.datetime64(row.entry_ts)
        pos = int(np.searchsorted(ts_arr, entry_ts))
        if pos >= len(day):
            continue
        feats = base.build_exit_features(day, pos, hold_min, fee_rate)
        if feats.empty:
            continue
        # Oracle: choose the minute in [1..hold_min] with maximum since_entry_return_pct
        # Include minute 0 only if it is the only available (edge case)
        mask = (feats["elapsed_min"] >= 1) & (feats["elapsed_min"] <= hold_min)
        if not mask.any():
            exit_idx = int(min(len(feats) - 1, hold_min))
        else:
            subset = feats.loc[mask, ["elapsed_min", "since_entry_return_pct"]]
            rel_idx = int(subset["since_entry_return_pct"].to_numpy().argmax())
            exit_idx = int(subset.index[rel_idx])

        exit_row = feats.iloc[exit_idx]
        raw_pct = float(exit_row["since_entry_return_pct"])  # net of fees
        trades.append(
            dict(
                date=row.entry_date,
                entry_time=row.entry_ts,
                exit_time=exit_row["timestamp"],
                duration_min=int(exit_row["elapsed_min"]),
                raw_return_pct=raw_pct,
                leverage=1.0,
                trade_return_pct=raw_pct,  # no leverage for pure upper-bound assessment
                capital_before=np.nan,
                capital_after=np.nan,
                bank_before=np.nan,
                bank_after=np.nan,
                bank_delta=0.0,
                profit=np.nan,
                liquidated=False,
                liquidation_reason="",
                restart_source="",
                restart_amount=0.0,
            )
        )
        daily_returns[row.entry_date] = daily_returns.get(row.entry_date, 0.0) + raw_pct

    daily = [dict(date=d, daily_return_pct=v) for d, v in sorted(daily_returns.items())]
    return trades, daily


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start_date) if args.start_date else None
    end = pd.Timestamp(args.end_date) if args.end_date else None

    # Load base trades and market, compute entry features, score and select entries
    trades = pd.read_parquet(args.rule_trades)
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"])  # ensure schema
    trades["exit_ts"] = pd.to_datetime(trades["exit_ts"])   # not used here
    if start is not None:
        trades = trades[trades["entry_ts"] >= start]
    if end is not None:
        trades = trades[trades["entry_ts"] <= end]

    market = base.load_market(Path(args.market_csv), start, end)
    feat = base.compute_feature_frame(market)
    feat = feat.dropna(subset=base.ENTRY_FEATURES).reset_index(drop=True)

    trades = base.attach_entry_features(trades, feat)
    lgb_model, xgb_model = base.load_entry_models(Path(args.entry_models_dir), args.entry_stem)
    trades_scored = base.score_entries(trades, lgb_model, xgb_model)

    allow_hours: Optional[set[int]] = None
    if args.allow_hours:
        try:
            allow_hours = {int(h) for h in str(args.allow_hours).split(',') if h.strip()}
        except Exception:
            allow_hours = None

    filters = dict(
        min_vol_std_5m=args.min_vol_std_5m,
        min_range_norm_5m=args.min_range_norm_5m,
        rsi_min=args.rsi_min,
        rsi_max=args.rsi_max,
        max_vwap_dev_abs=args.max_vwap_dev_abs,
        min_mom_3m_pct=args.min_mom_3m_pct,
    )
    selected = base.select_entries(trades_scored, args.entry_threshold, args.k_per_day, args.hold_min, allow_hours, filters)

    # Optionally persist the exact selected entries for reproducibility checks
    if args.save_selected:
        sel_out = Path(args.save_selected)
        sel_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            'entry_ts': pd.to_datetime(selected['entry_ts']).astype('datetime64[ns]'),
            'score': selected.get('score', np.nan),
            'entry_date': selected.get('entry_date', pd.NaT)
        }).to_csv(sel_out, index=False)

    # Build day lookup and run oracle
    market["timestamp"] = market["ts_min"]
    market["date"] = market["timestamp"].dt.date
    day_lookup = {d: g.sort_values("timestamp").reset_index(drop=True) for d, g in market.groupby("date")}
    trades_out, daily_out = simulate_oracle(selected, day_lookup, args.hold_min, args.fee_rate)

    trades_df = pd.DataFrame(trades_out)
    daily_df = pd.DataFrame(daily_out)

    Path(args.out_trades).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_daily).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)

    trades_df.to_csv(args.out_trades, index=False)
    daily_df.to_csv(args.out_daily, index=False)

    # Minimal metrics for the oracle case
    final = dict(
        start_date=args.start_date,
        end_date=args.end_date,
        entry_threshold=args.entry_threshold,
        k_per_day=args.k_per_day,
        hold_min=args.hold_min,
        fee_rate=args.fee_rate,
        num_trades=len(trades_df),
        trade_days=len(daily_df),
        mean_trade_pct=float(trades_df["trade_return_pct"].mean()) if not trades_df.empty else 0.0,
        median_trade_pct=float(trades_df["trade_return_pct"].median()) if not trades_df.empty else 0.0,
        daily_mean_pct=float(daily_df["daily_return_pct"].mean()) if not daily_df.empty else 0.0,
    )
    pd.DataFrame([final]).to_csv(args.metrics_out, index=False)

    print("Executed trades:", len(trades_df))
    print("Average trade %:", final["mean_trade_pct"])
    print("Average daily %:", final["daily_mean_pct"]) 


if __name__ == "__main__":
    main()
