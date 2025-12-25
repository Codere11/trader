#!/usr/bin/env python3
"""Analyze "initial liquidations" when starting from each trade.

Definition used here:
- For each starting trade index s, start a fresh account with:
    balance = Starting trading balance (from DO_NOT_TOUCH)
    bank = 0
- Simulate trades s..end under the strategy, with optional hard liquidation:
    --liquidation-mode wick: liquidate if any 1m candle low between entry open
      and the chosen exit minute crosses the fee-adjusted liquidation threshold.
    --liquidation-mode close: liquidate if levered close return would make equity <= 0.
- Count "initial liquidations" = number of liquidated trades that occur while
  bank (BEFORE the trade) is < refinance_bank_threshold (150 EUR by default).
- Stop the simulation for that start index at the first time bank >= threshold,
  or when trades end.

Outputs:
- Per-start CSV with the count of initial liquidations and stop info.
- Prints summary stats + a small histogram of the distribution.

Example:
  python3 scripts/analyze_initial_liquidations_sweep_2025-12-23T09-43-51Z.py \
    --trades-csv data/exit_regression/eval_exit_regressor_trades_2025-12-20T14-42-40Z.csv \
    --market-csv data/btc_profitability_analysis_filtered.csv \
    --liquidation-mode wick \
    --leverages 75,80,100 \
    --out-dir data/initial_liquidations
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Strategy:
    tiers: list[tuple[float, float]]  # list of (equity_threshold_eur, leverage), ascending thresholds
    withdraw_half_levers: set[float]
    start_balance: float
    refinance_min_balance: float
    refinance_floor: float
    refinance_bank_threshold: float
    refinance_bank_fraction: float


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def parse_strategy(path: Path) -> Strategy:
    txt = path.read_text(encoding="utf-8")

    # Leverage tiers
    tiers: list[tuple[float, float]] = []
    lev_pat = re.compile(r"(\d+)x leverage until account reaches ([\d,]+)EUR")
    for m in lev_pat.finditer(txt):
        lev = float(m.group(1))
        thr = float(m.group(2).replace(",", ""))
        tiers.append((thr, lev))
    tiers.sort(key=lambda x: x[0])
    if tiers:
        tiers.append((float("inf"), 1.0))

    # Withdraw rule (as written in DO_NOT_TOUCH)
    withdraw_half_levers = {100.0, 50.0}

    # Starting balance
    m = re.search(r"Starting trading balance deposited: (\d+(?:\.\d+)?)EUR", txt)
    start_balance = float(m.group(1)) if m else 30.0

    # Refinancing
    refinance_min_balance = 5.0
    refinance_floor = start_balance
    refinance_bank_threshold = 150.0

    m2 = re.search(r"20% of the bank account is deposited", txt)
    refinance_bank_fraction = 0.20 if m2 else 0.20

    return Strategy(
        tiers=tiers,
        withdraw_half_levers=withdraw_half_levers,
        start_balance=start_balance,
        refinance_min_balance=refinance_min_balance,
        refinance_floor=refinance_floor,
        refinance_bank_threshold=refinance_bank_threshold,
        refinance_bank_fraction=refinance_bank_fraction,
    )


def override_first_tier_leverage(strat: Strategy, first_tier_lev: float) -> Strategy:
    tiers = list(strat.tiers)
    if not tiers:
        return strat

    tiers[0] = (tiers[0][0], float(first_tier_lev))

    withdraw_half_levers = set(strat.withdraw_half_levers)
    # If the base strategy withdraws at 100x, keep the same "withdraw at top tier" behavior.
    if 100.0 in withdraw_half_levers:
        withdraw_half_levers.remove(100.0)
        withdraw_half_levers.add(float(first_tier_lev))

    return Strategy(
        tiers=tiers,
        withdraw_half_levers=withdraw_half_levers,
        start_balance=strat.start_balance,
        refinance_min_balance=strat.refinance_min_balance,
        refinance_floor=strat.refinance_floor,
        refinance_bank_threshold=strat.refinance_bank_threshold,
        refinance_bank_fraction=strat.refinance_bank_fraction,
    )


def leverage_for_equity(equity: float, tiers: list[tuple[float, float]]) -> float:
    for thr, lev in tiers:
        if equity < thr:
            return lev
    return 1.0


def _map_ts_to_index(ts_sorted: np.ndarray, q: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(ts_sorted, q)
    ok = (pos < ts_sorted.size) & (ts_sorted[pos] == q)
    return np.where(ok, pos, -1).astype(np.int64)


def load_market_open_low(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = pd.read_csv(csv_path, parse_dates=["timestamp"], usecols=["timestamp", "open", "low"])
    raw["timestamp"] = raw["timestamp"].dt.floor("min")

    # If file is already 1m OHLCV there should be no duplicates; still guard.
    raw = raw.drop_duplicates("timestamp", keep="first")

    if not raw["timestamp"].is_monotonic_increasing:
        raw = raw.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    ts_sorted = raw["timestamp"].to_numpy(dtype="datetime64[ns]")
    openp = raw["open"].to_numpy(np.float64, copy=False)
    lowp = raw["low"].to_numpy(np.float64, copy=False)
    return ts_sorted, openp, lowp


def wick_liquidated_for_trade(
    *,
    lev: float,
    fee_side: float,
    idx0: int,
    exit_rel_min: int,
    openp: np.ndarray,
    lowp: np.ndarray,
) -> bool:
    if lev <= 1.0:
        return False
    if idx0 < 0:
        return False

    k = max(0, int(exit_rel_min))
    idx1 = min(len(lowp) - 1, idx0 + k)
    window = lowp[idx0 : idx1 + 1]
    if window.size == 0:
        return False

    entry_open = float(openp[idx0])
    liq_price = entry_open * (1.0 + fee_side) * (1.0 - 1.0 / float(lev)) / max(1e-12, (1.0 - fee_side))
    return bool(np.any(window <= liq_price))


def close_hard_liquidated_for_trade(*, lev: float, ret_pct_1x: float) -> bool:
    # Same as backtest: levered_mult <= 0 => hard liquidation.
    return bool((1.0 + (ret_pct_1x / 100.0) * float(lev)) <= 0.0)


def initial_liqs_per_start(
    *,
    trades: pd.DataFrame,
    strat: Strategy,
    liquidation_mode: str,
    market_ts: np.ndarray | None,
    market_open: np.ndarray | None,
    market_low: np.ndarray | None,
    fee_side: float,
) -> pd.DataFrame:
    trades = trades.copy()
    trades = trades.sort_values("entry_time").reset_index(drop=True)

    entry_times = pd.to_datetime(trades["entry_time"]).to_numpy(dtype="datetime64[ns]")
    ret_pct = trades["realized_ret_pct"].to_numpy(np.float64, copy=False)
    exit_rel_min = trades["exit_rel_min"].to_numpy(np.int64, copy=False)

    idx0 = None
    if liquidation_mode == "wick":
        assert market_ts is not None and market_open is not None and market_low is not None
        entry_mins = pd.to_datetime(trades["entry_time"]).dt.floor("min").to_numpy(dtype="datetime64[ns]")
        idx0 = _map_ts_to_index(market_ts, entry_mins)

    n = len(trades)
    out_rows: list[dict] = []

    for start in range(n):
        balance = float(strat.start_balance)
        bank = 0.0

        initial_liqs = 0
        n_sim_trades = 0
        stop_reason = "end_of_data"

        for j in range(start, n):
            bal_before = float(balance)
            bank_before = float(bank)

            lev = leverage_for_equity(balance, strat.tiers)

            liquidated = False
            if liquidation_mode == "wick":
                liquidated = wick_liquidated_for_trade(
                    lev=lev,
                    fee_side=float(fee_side),
                    idx0=int(idx0[j]) if idx0 is not None else -1,
                    exit_rel_min=int(exit_rel_min[j]),
                    openp=market_open,
                    lowp=market_low,
                )
                if (not liquidated) and close_hard_liquidated_for_trade(lev=lev, ret_pct_1x=float(ret_pct[j])):
                    liquidated = True
            elif liquidation_mode == "close":
                liquidated = close_hard_liquidated_for_trade(lev=lev, ret_pct_1x=float(ret_pct[j]))
            elif liquidation_mode == "none":
                liquidated = False
            else:
                raise ValueError(f"Unknown liquidation_mode={liquidation_mode!r}")

            if liquidated and bank_before < strat.refinance_bank_threshold:
                initial_liqs += 1

            if liquidated:
                pnl = -bal_before
                balance = 0.0
            else:
                pnl = bal_before * (float(ret_pct[j]) / 100.0) * float(lev)
                balance = bal_before + pnl

            # Withdraw
            if (not liquidated) and (float(lev) in strat.withdraw_half_levers) and pnl > 0.0:
                withdrawn = 0.5 * pnl
                balance -= withdrawn
                bank += withdrawn

            # Refinance
            if balance < strat.refinance_min_balance:
                if bank < strat.refinance_bank_threshold:
                    dep = max(0.0, strat.refinance_floor - balance)
                    balance += dep
                else:
                    dep = strat.refinance_bank_fraction * bank
                    bank -= dep
                    balance += dep

            n_sim_trades += 1

            if bank >= strat.refinance_bank_threshold:
                stop_reason = "bank_threshold_reached"
                break

        out_rows.append(
            {
                "start_index": int(start),
                "start_entry_time": pd.to_datetime(entry_times[start]),
                "initial_liquidations": int(initial_liqs),
                "reached_bank_threshold": bool(stop_reason == "bank_threshold_reached"),
                "n_trades_simulated": int(n_sim_trades),
                "end_balance": float(balance),
                "end_bank": float(bank),
                "end_equity": float(balance + bank),
                "stop_reason": stop_reason,
            }
        )

    return pd.DataFrame(out_rows)


def print_distribution(name: str, counts: pd.Series) -> None:
    counts = counts.astype(int)

    vc = counts.value_counts().sort_index()

    def pct(x: float) -> float:
        return 100.0 * float(x)

    print(f"\n{name}")
    print(f"  n_startpoints: {len(counts)}")
    print(f"  max_initial_liquidations: {int(counts.max()) if len(counts) else 0}")
    print(f"  mean: {counts.mean():.4f}")
    print(f"  median: {counts.median():.1f}")
    for q in [0.75, 0.90, 0.95, 0.99]:
        print(f"  p{int(q*100)}: {counts.quantile(q):.2f}")

    # Small histogram (all if small, else first 20 bins + tail)
    print("  histogram (count -> starts):")
    if len(vc) <= 25:
        for k, v in vc.items():
            print(f"    {int(k):>3}: {int(v):>5} ({pct(v/len(counts)):.2f}%)")
    else:
        head = vc.iloc[:20]
        tail = vc.iloc[20:]
        for k, v in head.items():
            print(f"    {int(k):>3}: {int(v):>5} ({pct(v/len(counts)):.2f}%)")
        print(f"    ... ({len(tail)} more bins)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze initial liquidations from each trade start")
    ap.add_argument("--do-not-touch", default=str(REPO_ROOT / "DO_NOT_TOUCH.txt"))
    ap.add_argument("--trades-csv", required=True)
    ap.add_argument("--liquidation-mode", choices=["none", "close", "wick"], default="wick")
    ap.add_argument("--market-csv", default=None, help="Required for --liquidation-mode wick")
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--leverages", default="75,80,100", help="Comma-separated list of first-tier leverages to sweep")
    ap.add_argument("--out-dir", default="data/initial_liquidations")
    args = ap.parse_args()

    strat_base = parse_strategy(Path(args.do_not_touch))
    trades = pd.read_csv(args.trades_csv, parse_dates=["entry_time"])  # expects realized_ret_pct + exit_rel_min

    if args.liquidation_mode == "wick" and not args.market_csv:
        raise SystemExit("--market-csv is required when --liquidation-mode wick")

    market_ts = market_open = market_low = None
    if args.liquidation_mode == "wick":
        market_ts, market_open, market_low = load_market_open_low(Path(args.market_csv))

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = now_ts()

    levs = [float(x.strip()) for x in str(args.leverages).split(",") if x.strip()]

    for lev in levs:
        strat = override_first_tier_leverage(strat_base, lev)
        df = initial_liqs_per_start(
            trades=trades,
            strat=strat,
            liquidation_mode=str(args.liquidation_mode),
            market_ts=market_ts,
            market_open=market_open,
            market_low=market_low,
            fee_side=float(args.fee),
        )

        out_path = out_dir / f"initial_liquidations_by_start_lev{int(lev)}_{ts}.csv"
        df.to_csv(out_path, index=False)

        print(f"\nWrote: {out_path}")
        print_distribution(f"Leverage {lev:.0f}x", df["initial_liquidations"])

        # Show worst startpoints
        maxv = int(df["initial_liquidations"].max()) if not df.empty else 0
        worst = df[df["initial_liquidations"] == maxv].head(10)
        if maxv > 0 and not worst.empty:
            print("  worst startpoints (first 10):")
            for _, r in worst.iterrows():
                print(f"    start_index={int(r['start_index'])} start_time={r['start_entry_time']} n_initial_liqs={int(r['initial_liquidations'])} simulated_trades={int(r['n_trades_simulated'])} stop={r['stop_reason']}")


if __name__ == "__main__":
    main()
