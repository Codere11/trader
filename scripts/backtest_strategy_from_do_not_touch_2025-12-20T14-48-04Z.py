#!/usr/bin/env python3
"""
Backtest the strategy specified in DO_NOT_TOUCH.txt against a trades CSV.
- Reads STRATEGY rules from DO_NOT_TOUCH.txt (leverage tiers, withdrawal policy, starting balance, refinance rules).
- Applies the policy per trade in chronological order using realized_ret_pct (net % at 1x) and leverage scaling.
- Outputs equity/bank over time + per-day summary.

Usage:
  python3 scripts/backtest_strategy_from_do_not_touch_2025-12-20T14-48-04Z.py \
    --do-not-touch DO_NOT_TOUCH.txt \
    --trades-csv data/exit_regression/eval_exit_regressor_trades_<ts>.csv \
    --out-dir data/backtests
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys

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


def parse_strategy(path: Path) -> Strategy:
    txt = path.read_text(encoding="utf-8")
    # Leverage tiers
    tiers = []
    # 1) 100x leverage until account reaches 100,000EUR
    lev_pat = re.compile(r"(\d+)x leverage until account reaches ([\d,]+)EUR")
    for m in lev_pat.finditer(txt):
        lev = float(m.group(1))
        thr = float(m.group(2).replace(",",""))
        tiers.append((thr, lev))
    # 5) 1x leverage after that in perpertuity
    tiers.sort(key=lambda x: x[0])
    # Ensure final tier of 1x after last threshold
    if tiers:
        tiers.append((float("inf"), 1.0))
    # Withdraw rule for 100x and 50x
    withdraw_half_levers = {100.0, 50.0}
    # Starting balance
    m = re.search(r"Starting trading balance deposited: (\d+(?:\.\d+)?)EUR", txt)
    start_balance = float(m.group(1)) if m else 30.0
    # Refinancing
    # If balance drops below 5EUR AND withdrawn/banked amount is less than 150EUR, refinance it up to 30EUR.
    refinance_min_balance = 5.0
    refinance_floor = start_balance
    refinance_bank_threshold = 150.0
    # If banked >= threshold, deposit 20% of bank as new starting balance
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


def leverage_for_equity(equity: float, tiers: list[tuple[float,float]]) -> float:
    for thr, lev in tiers:
        if equity < thr:
            return lev
    return 1.0


def minute_bars(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, parse_dates=["timestamp"], usecols=["timestamp", "open", "high", "low", "close", "volume"])
    if not raw["timestamp"].is_monotonic_increasing:
        raw = raw.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    raw["ts_min"] = raw["timestamp"].dt.floor("min")
    g = (
        raw.groupby("ts_min")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"))
        .reset_index()
        .rename(columns={"ts_min": "timestamp"})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return g


def _map_ts_to_index(ts_sorted: np.ndarray, q: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(ts_sorted, q)
    ok = (pos < ts_sorted.size) & (ts_sorted[pos] == q)
    return np.where(ok, pos, -1).astype(np.int64)


def _wick_liquidation_long(
    entry_open: float,
    low_window: np.ndarray,
    fee_side: float,
    leverage: float,
) -> tuple[bool, float, int]:
    """Return (liquidated, liq_price, rel_min_at_breach).

    Liquidation condition (simple): equity_mult = 1 + L * (net_mult - 1) <= 0.
    net_mult = (px*(1-fee)) / (entry_open*(1+fee)).

    We trigger on the *first* minute where candle low <= liq_price.
    """
    if leverage <= 1.0:
        return False, float("nan"), -1
    # Solve net_mult <= 1 - 1/L for px
    liq_price = entry_open * (1.0 + fee_side) * (1.0 - 1.0 / float(leverage)) / max(1e-12, (1.0 - fee_side))
    breach = np.where(low_window <= liq_price)[0]
    if breach.size == 0:
        return False, liq_price, -1
    return True, liq_price, int(breach[0])


def simulate(
    trades: pd.DataFrame,
    strat: Strategy,
    liquidation_mode: str = "none",
    market_csv: Path | None = None,
    fee_side: float = 0.001,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    trades = trades.copy()
    trades = trades.sort_values("entry_time").reset_index(drop=True)

    mkt = None
    ts_sorted = None
    openp = None
    lowp = None
    if liquidation_mode == "wick":
        if market_csv is None:
            raise SystemExit("--market-csv is required when --liquidation-mode wick")
        if "exit_rel_min" not in trades.columns:
            raise SystemExit("trades CSV must include exit_rel_min for wick liquidation")
        mkt = minute_bars(market_csv)
        ts_sorted = mkt["timestamp"].to_numpy(dtype="datetime64[ns]")
        openp = mkt["open"].to_numpy(np.float64, copy=False)
        lowp = mkt["low"].to_numpy(np.float64, copy=False)

    balance = strat.start_balance
    bank = 0.0
    total_ext_deposits = strat.start_balance

    n_ext_refinances_total = 0
    n_ext_refinances_before_first_bank150 = 0
    bank150_first_reached_at = None

    records = []

    for i, r in trades.iterrows():
        t = pd.to_datetime(r["entry_time"]) if not np.issubdtype(trades["entry_time"].dtype, np.datetime64) else r["entry_time"]
        ret_pct = float(r["realized_ret_pct"])  # net % at 1x

        bal_before = float(balance)
        bank_before = float(bank)

        lev = leverage_for_equity(balance, strat.tiers)

        # Liquidation checks
        liq_reason = ""
        wick_liq = False
        wick_liq_rel_min = -1
        wick_liq_price = float("nan")
        wick_min_low = float("nan")

        if liquidation_mode == "wick":
            entry_min = pd.to_datetime(t).floor("min").to_datetime64()
            idx0 = int(_map_ts_to_index(ts_sorted, np.array([entry_min], dtype="datetime64[ns]"))[0])
            if idx0 >= 0:
                k = int(r["exit_rel_min"])
                idx1 = min(len(lowp) - 1, idx0 + max(0, k))
                window = lowp[idx0 : idx1 + 1]
                wick_min_low = float(np.min(window)) if window.size else float("nan")
                wick_liq, wick_liq_price, rel = _wick_liquidation_long(float(openp[idx0]), window, float(fee_side), float(lev))
                if wick_liq:
                    wick_liq_rel_min = int(rel)
                    liq_reason = "wick"

        # Close-based hard liquidation (prevents impossible negative equity)
        if liquidation_mode in ("close", "wick"):
            levered_mult = 1.0 + (ret_pct / 100.0) * lev
            close_hard_liq = bool(levered_mult <= 0.0)
            if close_hard_liq and not liq_reason:
                liq_reason = "close"

        liquidated = bool(liq_reason)
        if liquidated:
            pnl = -bal_before
            balance = 0.0
        else:
            pnl = bal_before * (ret_pct / 100.0) * lev
            balance = bal_before + pnl

        withdrawn = 0.0
        if (not liquidated) and lev in strat.withdraw_half_levers and pnl > 0.0:
            withdrawn = 0.5 * pnl
            balance -= withdrawn
            bank += withdrawn

        # Refinance logic if balance drops below floor
        ext_deposit = 0.0
        if balance < strat.refinance_min_balance:
            if bank < strat.refinance_bank_threshold:
                # external top-up to floor
                ext_deposit = max(0.0, strat.refinance_floor - balance)
                balance += ext_deposit
                total_ext_deposits += ext_deposit
                if ext_deposit > 0.0:
                    n_ext_refinances_total += 1
                    if bank150_first_reached_at is None:
                        n_ext_refinances_before_first_bank150 += 1
            else:
                # take from bank: 20% of bank into account
                dep = strat.refinance_bank_fraction * bank
                bank -= dep
                balance += dep

        if bank150_first_reached_at is None and bank >= strat.refinance_bank_threshold:
            bank150_first_reached_at = t

        equity = balance + bank
        records.append({
            "entry_time": t,
            "lev": lev,
            "ret_pct_1x": ret_pct,
            "balance_before": bal_before,
            "bank_before": bank_before,
            "pnl": pnl,
            "liquidated": liquidated,
            "liquidation_reason": liq_reason,
            "wick_liquidated": bool(wick_liq),
            "wick_liq_rel_min": int(wick_liq_rel_min),
            "wick_liq_price": float(wick_liq_price) if wick_liq_price == wick_liq_price else None,
            "wick_min_low": float(wick_min_low) if wick_min_low == wick_min_low else None,
            "withdrawn": withdrawn,
            "ext_deposit": ext_deposit,
            "balance": balance,
            "bank": bank,
            "equity": equity,
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["entry_time"]).dt.date
    daily = df.groupby("date", as_index=False).agg(
        n_trades=("pnl", "size"),
        sum_pnl_eur=("pnl", "sum"),
        end_balance=("balance", "last"),
        end_bank=("bank", "last"),
        end_equity=("equity", "last"),
        mean_1x_ret_pct=("ret_pct_1x", "mean"),
        n_liquidations=("liquidated", "sum"),
        n_wick_liq=("wick_liquidated", "sum"),
        n_external_refinances=("ext_deposit", lambda s: int((s.to_numpy() > 0.0).sum())),
    )

    summary = {
        "total_ext_deposits": float(total_ext_deposits),
        "n_external_refinances_total": int(n_ext_refinances_total),
        "n_external_refinances_before_first_bank150": int(n_ext_refinances_before_first_bank150),
        "bank150_first_reached_at": bank150_first_reached_at,
        "final_balance": float(balance),
        "final_bank": float(bank),
        "final_equity": float(balance + bank),
        "n_trades": int(len(df)),
        "n_liquidations": int(df["liquidated"].sum()) if not df.empty else 0,
        "n_wick_liquidations": int(df["wick_liquidated"].sum()) if not df.empty else 0,
    }

    return df, daily, summary


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest DO_NOT_TOUCH strategy on trades CSV")
    ap.add_argument("--do-not-touch", default=str(REPO_ROOT / "DO_NOT_TOUCH.txt"))
    ap.add_argument("--trades-csv", required=True)
    ap.add_argument("--out-dir", default="data/backtests")
    ap.add_argument("--liquidation-mode", choices=["none", "close", "wick"], default="none", help="none=use realized_ret_pct only; close=cap losses at -100% (hard liquidation); wick=hard liquidation if any candle low crosses threshold")
    ap.add_argument("--market-csv", default=None, help="Required for --liquidation-mode wick; CSV with timestamp, open, high, low, close, volume")
    ap.add_argument("--fee", type=float, default=0.001, help="Per-side fee used for wick liquidation threshold")
    args = ap.parse_args()

    strat = parse_strategy(Path(args["--do-not-touch"]) if isinstance(args, dict) else Path(args.__dict__["do_not_touch"]))
    trades = pd.read_csv(args.trades_csv, parse_dates=["entry_time"])  # expects realized_ret_pct
    ts = now_ts()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = str(args.liquidation_mode)
    mkt_path = Path(args.market_csv) if args.market_csv else None

    per_trade, daily, summary = simulate(
        trades,
        strat,
        liquidation_mode=mode,
        market_csv=mkt_path,
        fee_side=float(args.fee),
    )

    pt_path = out_dir / f"backtest_per_trade_{ts}.csv"
    dy_path = out_dir / f"backtest_daily_{ts}.csv"
    per_trade.to_csv(pt_path, index=False)
    daily.to_csv(dy_path, index=False)

    print("Backtest per-trade:", pt_path)
    print("Backtest daily:", dy_path)
    print("Summary:")
    for k in [
        "n_trades",
        "n_liquidations",
        "n_wick_liquidations",
        "n_external_refinances_total",
        "n_external_refinances_before_first_bank150",
        "bank150_first_reached_at",
        "total_ext_deposits",
        "final_balance",
        "final_bank",
        "final_equity",
    ]:
        print(f"  {k}: {summary.get(k)}")


if __name__ == "__main__":
    main()
