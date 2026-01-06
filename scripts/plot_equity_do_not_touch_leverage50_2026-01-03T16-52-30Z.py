#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-03T16:52:30Z
"""Plot DO_NOT_TOUCH banking equity curve + trades/day for selected entry streams.

This fixes the earlier mistake where we simulated pure leveraged compounding (which will often
trend to ~0 over long horizons). Here we apply the DO_NOT_TOUCH banking rules:
- Starting trading balance: 30 EUR
- At 50x leverage: withdraw 50% of profit on winning trades into bank
- Refinance rules:
  * if balance < 5 and bank < 150: external top-up back to 30
  * if balance < 5 and bank >= 150: deposit 20% of bank into trading account

Exit logic (same as our current pipeline):
- Use oracle-gap regressor to predict oracle_gap_pct per minute
- Exit at first k>=min_exit_k where pred_gap_pct <= tau_gap, else exit at hold_min

Outputs
- data/equity_do_not_touch_leverage50_<ts>/
  - equity_trades_timeseries.png (2 panels: trades/day, end-of-day equity)
  - <name>_daily.csv
  - <name>_per_trade.csv

Equity definition
- equity = trading_balance + bank_balance
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def parse_named_dataset(s: str) -> Tuple[str, str]:
    if "=" not in s:
        raise ValueError("--dataset must be in name=path format")
    name, path = s.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError("invalid dataset spec")
    return name, path


def pick_exit_k(pred_gap: np.ndarray, *, hold_min: int, min_exit_k: int, tau_gap: float) -> int:
    for k in range(int(min_exit_k), int(hold_min) + 1):
        v = float(pred_gap[k - 1])
        if np.isfinite(v) and v <= float(tau_gap):
            return int(k)
    return int(hold_min)


def build_trade_returns_from_oracle_exit_dataset(
    *,
    dataset_parquet: Path,
    reg,
    feat_cols: List[str],
    hold_min: int,
    min_exit_k: int,
    tau_gap: float,
) -> pd.DataFrame:
    df = pd.read_parquet(dataset_parquet)

    need = {"trade_id", "k_rel", "entry_time", "ret_if_exit_now_pct"}
    if not need.issubset(df.columns):
        raise ValueError(f"dataset missing required columns: {sorted(list(need - set(df.columns)))}")

    df = df[(df["k_rel"] >= 1) & (df["k_rel"] <= int(hold_min))].copy()
    df = df.sort_values(["trade_id", "k_rel"]).reset_index(drop=True)

    sizes = df.groupby("trade_id")["k_rel"].count()
    full_ids = sizes[sizes == int(hold_min)].index.to_numpy()
    df = df[df["trade_id"].isin(full_ids)].copy().reset_index(drop=True)

    for c in feat_cols:
        if c not in df.columns:
            df[c] = np.nan

    X = df[feat_cols].to_numpy(np.float32)
    pred = reg.predict(X).astype(np.float64)

    work = df[["trade_id", "k_rel", "entry_time", "ret_if_exit_now_pct"]].copy()
    work["pred_gap_pct"] = pred
    work["entry_time"] = pd.to_datetime(work["entry_time"], utc=True)

    rows: List[Dict[str, object]] = []
    for tid, g in work.groupby("trade_id", sort=False):
        g2 = g.sort_values("k_rel")
        entry_time = g2["entry_time"].iloc[0]
        pred_gap = g2["pred_gap_pct"].to_numpy(np.float64)
        rets = g2["ret_if_exit_now_pct"].to_numpy(np.float64)
        k_exit = pick_exit_k(pred_gap, hold_min=hold_min, min_exit_k=min_exit_k, tau_gap=tau_gap)
        r = float(rets[int(k_exit) - 1])
        rows.append(
            {
                "trade_id": int(tid),
                "entry_time": entry_time,
                "exit_k": int(k_exit),
                "realized_ret_pct": float(r),
            }
        )

    tr = pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)
    if tr.empty:
        raise ValueError("no trades")
    return tr


@dataclass(frozen=True)
class DoNotTouchStrat:
    leverage: float
    withdraw_half_profit: bool
    start_balance: float
    refinance_min_balance: float
    refinance_floor: float
    refinance_bank_threshold: float
    refinance_bank_fraction: float


def simulate_do_not_touch(trades: pd.DataFrame, strat: DoNotTouchStrat) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    trades = trades.copy().sort_values("entry_time").reset_index(drop=True)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)

    balance = float(strat.start_balance)
    bank = 0.0
    total_ext_deposits = float(strat.start_balance)

    recs: List[Dict[str, object]] = []

    for _, r in trades.iterrows():
        t = pd.to_datetime(r["entry_time"], utc=True)
        ret_pct_1x = float(r["realized_ret_pct"])  # already net at 1x

        bal_before = float(balance)
        bank_before = float(bank)

        lev = float(strat.leverage)
        pnl = bal_before * (ret_pct_1x / 100.0) * lev
        balance = bal_before + pnl

        withdrawn = 0.0
        if strat.withdraw_half_profit and pnl > 0.0:
            withdrawn = 0.5 * pnl
            balance -= withdrawn
            bank += withdrawn

        ext_deposit = 0.0
        bank_deposit = 0.0

        if balance < float(strat.refinance_min_balance):
            if bank < float(strat.refinance_bank_threshold):
                ext_deposit = max(0.0, float(strat.refinance_floor) - balance)
                balance += ext_deposit
                total_ext_deposits += ext_deposit
            else:
                bank_deposit = float(strat.refinance_bank_fraction) * bank
                bank -= bank_deposit
                balance += bank_deposit

        equity = balance + bank

        recs.append(
            {
                "entry_time": t,
                "date": t.floor("D"),
                "lev": lev,
                "ret_pct_1x": ret_pct_1x,
                "balance_before": bal_before,
                "bank_before": bank_before,
                "pnl": pnl,
                "withdrawn": withdrawn,
                "ext_deposit": ext_deposit,
                "bank_deposit": bank_deposit,
                "balance": balance,
                "bank": bank,
                "equity": equity,
            }
        )

    per_trade = pd.DataFrame(recs)
    daily = (
        per_trade.groupby("date", as_index=False)
        .agg(
            trades=("pnl", "size"),
            end_balance=("balance", "last"),
            end_bank=("bank", "last"),
            end_equity=("equity", "last"),
            sum_pnl=("pnl", "sum"),
            sum_withdrawn=("withdrawn", "sum"),
            sum_ext_deposit=("ext_deposit", "sum"),
            sum_bank_deposit=("bank_deposit", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    summary: Dict[str, object] = {
        "n_trades": int(len(per_trade)),
        "total_ext_deposits": float(total_ext_deposits),
        "final_balance": float(balance),
        "final_bank": float(bank),
        "final_equity": float(balance + bank),
    }

    return per_trade, daily, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot DO_NOT_TOUCH equity (balance+bank) and trades/day")
    ap.add_argument("--model-joblib", required=True)
    ap.add_argument("--dataset", action="append", required=True, help="Repeated: name=/path/to/dataset.parquet")
    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--min-exit-k", type=int, default=2)
    ap.add_argument("--tau-gap", type=float, default=0.10)

    # DO_NOT_TOUCH params (defaults from DO_NOT_TOUCH.txt)
    ap.add_argument("--leverage", type=float, default=50.0)
    ap.add_argument("--start-balance", type=float, default=30.0)
    ap.add_argument("--refinance-min-balance", type=float, default=5.0)
    ap.add_argument("--refinance-floor", type=float, default=30.0)
    ap.add_argument("--refinance-bank-threshold", type=float, default=150.0)
    ap.add_argument("--refinance-bank-fraction", type=float, default=0.20)
    ap.add_argument("--withdraw-half-profit", action="store_true", default=True)

    ap.add_argument("--out-dir", default="data")

    args = ap.parse_args()

    model_path = Path(args.model_joblib)
    art = joblib.load(model_path)
    reg = art["model"]
    feat_cols = list(art["feature_cols"])

    strat = DoNotTouchStrat(
        leverage=float(args.leverage),
        withdraw_half_profit=True,  # for 50x per DO_NOT_TOUCH
        start_balance=float(args.start_balance),
        refinance_min_balance=float(args.refinance_min_balance),
        refinance_floor=float(args.refinance_floor),
        refinance_bank_threshold=float(args.refinance_bank_threshold),
        refinance_bank_fraction=float(args.refinance_bank_fraction),
    )

    hold_min = int(args.hold_min)
    min_exit_k = int(args.min_exit_k)
    tau_gap = float(args.tau_gap)

    named = [parse_named_dataset(s) for s in list(args.dataset)]

    out_root = Path(args.out_dir)
    out_dir = out_root / f"equity_do_not_touch_leverage{int(strat.leverage)}_{now_ts()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    series_daily: Dict[str, pd.DataFrame] = {}
    for name, p in named:
        ds = Path(p)
        trades = build_trade_returns_from_oracle_exit_dataset(
            dataset_parquet=ds,
            reg=reg,
            feat_cols=feat_cols,
            hold_min=hold_min,
            min_exit_k=min_exit_k,
            tau_gap=tau_gap,
        )

        per_trade, daily, summary = simulate_do_not_touch(trades, strat)
        per_trade.to_csv(out_dir / f"{name}_per_trade.csv", index=False)
        daily.to_csv(out_dir / f"{name}_daily.csv", index=False)

        series_daily[name] = daily
        print(f"{name}: {summary}")

    # Align dates (union)
    all_dates = pd.Index([])
    for d in series_daily.values():
        all_dates = all_dates.union(pd.to_datetime(d["date"], utc=True))
    all_dates = all_dates.sort_values()

    # Plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    ax0 = axes[0]
    for name, d in series_daily.items():
        dd = d.copy()
        dd["date"] = pd.to_datetime(dd["date"], utc=True)
        dd = dd.set_index(dd["date"])
        s = dd["trades"].reindex(all_dates).fillna(0.0)
        ax0.plot(all_dates, s.to_numpy(np.float64), label=f"{name} trades/day", linewidth=1.2)
    ax0.set_title("Trades per day")
    ax0.set_ylabel("trades/day")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper right")

    ax1 = axes[1]
    for name, d in series_daily.items():
        dd = d.copy()
        dd["date"] = pd.to_datetime(dd["date"], utc=True)
        dd = dd.set_index(dd["date"])
        s = dd["end_equity"].reindex(all_dates).ffill()
        ax1.plot(all_dates, s.to_numpy(np.float64), label=f"{name} equity (balance+bank)", linewidth=1.4)

    ax1.set_title(
        f"DO_NOT_TOUCH equity over time (L={strat.leverage:.0f}x, start={strat.start_balance}, "
        f"withdraw=50% profits, refinance<5EUR, exit: pred_gap<= {tau_gap:.2f}pp)"
    )
    ax1.set_ylabel("equity (EUR)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")

    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)

    fig.tight_layout()

    out_png = out_dir / "equity_trades_timeseries.png"
    fig.savefig(out_png, dpi=160)

    print(f"Wrote: {out_dir}")
    print(f"Image: {out_png}")


if __name__ == "__main__":
    main()
