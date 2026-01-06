#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-03T16:50:30Z
"""Plot account/equity growth over time at 50x leverage + daily trades.

Uses an oracle-gap regressor exit policy:
- predict oracle_gap_pct per minute
- exit at first k>=min_exit_k where pred_gap_pct <= tau_gap, else exit at hold_min

Equity update (simple leveraged compounding):
  equity *= max(0, 1 + leverage * ret_pct/100)
If the factor <= 0, we treat it as liquidation and equity becomes 0 thereafter.

Outputs a PNG with two panels:
- trades/day
- equity (50x) over time

Also writes per-series daily CSVs.
"""

from __future__ import annotations

import argparse
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


def build_trade_returns(
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

    rows = []
    for tid, g in work.groupby("trade_id", sort=False):
        g2 = g.sort_values("k_rel")
        entry_time = g2["entry_time"].iloc[0]
        pred_gap = g2["pred_gap_pct"].to_numpy(np.float64)
        rets = g2["ret_if_exit_now_pct"].to_numpy(np.float64)
        k_exit = pick_exit_k(pred_gap, hold_min=hold_min, min_exit_k=min_exit_k, tau_gap=tau_gap)
        r = float(rets[int(k_exit) - 1])
        rows.append({"trade_id": int(tid), "entry_time": entry_time, "exit_k": int(k_exit), "ret_pct": r})

    tr = pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)
    tr["date"] = tr["entry_time"].dt.floor("D")
    return tr


def daily_equity_series(trades: pd.DataFrame, *, start_equity: float, leverage: float) -> pd.DataFrame:
    if len(trades) == 0:
        raise ValueError("no trades")

    d0 = trades["date"].min()
    d1 = trades["date"].max()
    idx = pd.date_range(d0, d1, freq="D", tz="UTC")

    # trades/day
    trades_per_day = trades.groupby("date").size().reindex(idx, fill_value=0).astype(np.int64)

    # equity end-of-day
    eq = float(start_equity)
    eq_eod = []

    # pre-group trades by day for efficiency
    by_day = {d: g for d, g in trades.groupby("date")}

    for d in idx:
        g = by_day.get(d)
        if g is not None and eq > 0.0:
            # apply in chronological order
            for r in g.sort_values("entry_time")["ret_pct"].to_numpy(np.float64):
                factor = 1.0 + float(leverage) * (float(r) / 100.0)
                if factor <= 0.0:
                    eq = 0.0
                    break
                eq *= factor
        eq_eod.append(eq)

    out = pd.DataFrame({"date": idx, "trades": trades_per_day.to_numpy(), "equity": np.asarray(eq_eod, dtype=np.float64)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot equity curve (50x) and trades/day for entry streams")
    ap.add_argument("--model-joblib", required=True)
    ap.add_argument("--dataset", action="append", required=True, help="Repeated: name=/path/to/dataset.parquet")
    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--min-exit-k", type=int, default=2)
    ap.add_argument("--tau-gap", type=float, default=0.10)
    ap.add_argument("--leverage", type=float, default=50.0)
    ap.add_argument("--start-equity", type=float, default=10.0)
    ap.add_argument("--out-dir", default="data")

    args = ap.parse_args()

    model_path = Path(args.model_joblib)
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    art = joblib.load(model_path)
    reg = art["model"]
    feat_cols = list(art["feature_cols"])

    hold_min = int(args.hold_min)
    min_exit_k = int(args.min_exit_k)
    tau_gap = float(args.tau_gap)
    leverage = float(args.leverage)
    start_equity = float(args.start_equity)

    named = [parse_named_dataset(s) for s in list(args.dataset)]

    out_root = Path(args.out_dir)
    out_dir = out_root / f"equity_curve_leverage{int(leverage)}_{now_ts()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    series: Dict[str, pd.DataFrame] = {}
    for name, p in named:
        ds = Path(p)
        if not ds.exists():
            raise SystemExit(f"dataset not found: {ds}")

        tr = build_trade_returns(
            dataset_parquet=ds,
            reg=reg,
            feat_cols=feat_cols,
            hold_min=hold_min,
            min_exit_k=min_exit_k,
            tau_gap=tau_gap,
        )
        daily = daily_equity_series(tr, start_equity=start_equity, leverage=leverage)
        series[name] = daily

        daily.to_csv(out_dir / f"{name}_daily.csv", index=False)

        # quick summary
        final_eq = float(daily["equity"].iloc[-1])
        n_trades = int(tr.shape[0])
        print(f"{name}: trades={n_trades}  final_equity={final_eq:.6f}")

    # Align dates (union)
    all_dates = pd.Index([])
    for d in series.values():
        all_dates = all_dates.union(pd.to_datetime(d["date"]))
    all_dates = all_dates.sort_values()

    # Plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    ax0 = axes[0]
    for name, d in series.items():
        dd = d.copy()
        dd = dd.set_index(pd.to_datetime(dd["date"]))
        s = dd["trades"].reindex(all_dates).fillna(0.0)
        ax0.plot(all_dates, s.to_numpy(np.float64), label=f"{name} trades/day", linewidth=1.2)
    ax0.set_title("Trades per day")
    ax0.set_ylabel("trades/day")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper right")

    ax1 = axes[1]
    for name, d in series.items():
        dd = d.copy()
        dd = dd.set_index(pd.to_datetime(dd["date"]))
        s = dd["equity"].reindex(all_dates).ffill()
        ax1.plot(all_dates, s.to_numpy(np.float64), label=f"{name} equity (L={leverage:.0f}x)", linewidth=1.4)
    ax1.set_title(f"Account equity over time (start={start_equity}, exit: pred_gap<= {tau_gap:.2f}pp, min_exit_k={min_exit_k})")
    ax1.set_ylabel("equity")
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
