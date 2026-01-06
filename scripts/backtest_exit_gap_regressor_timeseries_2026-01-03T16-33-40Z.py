#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-03T16:33:40Z
"""Backtest oracle-gap regressor exit across entry fractions and plot daily stats.

We assume you already have per-minute oracle-exit datasets (parquet) with columns:
- trade_id, k_rel, entry_time
- ret_if_exit_now_pct
(and feature columns used by the regressor)

We load a trained oracle-gap regressor (LightGBM) and compute:
- pred_gap_pct per minute
- exit policy: exit at first k>=min_exit_k where pred_gap_pct <= tau_gap; else exit at hold_min

Outputs
- data/backtest_gap_timeseries_<ts>/
  - timeseries.png  (2 panels: trades/day and mean_ret_pct/day)
  - <name>_daily.csv  (date, trades, mean_ret_pct, sum_ret_pct)
  - summary.json

Note
- Daily profitability is reported as mean return per trade (pct) for trades whose entry_time falls on that day.
"""

from __future__ import annotations

import argparse
import json
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
    if not name:
        raise ValueError("dataset name empty")
    if not path:
        raise ValueError("dataset path empty")
    return name, path


def pick_exit_k(pred_gap: np.ndarray, *, hold_min: int, min_exit_k: int, tau_gap: float) -> int:
    # pred_gap is length hold_min for a single trade, indexed 0..hold_min-1
    for k in range(int(min_exit_k), int(hold_min) + 1):
        v = float(pred_gap[k - 1])
        if np.isfinite(v) and v <= float(tau_gap):
            return int(k)
    return int(hold_min)


def backtest_dataset(
    *,
    name: str,
    dataset_parquet: Path,
    reg,
    feat_cols: List[str],
    hold_min: int,
    min_exit_k: int,
    tau_gap: float,
) -> Dict[str, object]:
    df = pd.read_parquet(dataset_parquet)

    need = {"trade_id", "k_rel", "entry_time", "ret_if_exit_now_pct"}
    if not need.issubset(df.columns):
        raise ValueError(f"dataset missing required columns: {sorted(list(need - set(df.columns)))}")

    df = df[(df["k_rel"] >= 1) & (df["k_rel"] <= int(hold_min))].copy()
    df = df.sort_values(["trade_id", "k_rel"]).reset_index(drop=True)

    # keep only full-length trades
    sizes = df.groupby("trade_id")["k_rel"].count()
    full_ids = sizes[sizes == int(hold_min)].index.to_numpy()
    df = df[df["trade_id"].isin(full_ids)].copy().reset_index(drop=True)

    # ensure features exist
    for c in feat_cols:
        if c not in df.columns:
            df[c] = np.nan

    X = df[feat_cols].to_numpy(np.float32)
    pred = reg.predict(X).astype(np.float64)

    work = df[["trade_id", "k_rel", "entry_time", "ret_if_exit_now_pct"]].copy()
    work["pred_gap_pct"] = pred

    by = work.groupby("trade_id", sort=False)

    trade_rows: List[Dict[str, object]] = []
    for tid, g in by:
        g2 = g.sort_values("k_rel")
        entry_time = pd.to_datetime(g2["entry_time"].iloc[0], utc=True)

        pred_gap = g2["pred_gap_pct"].to_numpy(np.float64)
        ret_seq = g2["ret_if_exit_now_pct"].to_numpy(np.float64)

        k_exit = pick_exit_k(pred_gap, hold_min=hold_min, min_exit_k=min_exit_k, tau_gap=tau_gap)
        r = float(ret_seq[int(k_exit) - 1])

        trade_rows.append(
            {
                "trade_id": int(tid),
                "entry_time": entry_time,
                "entry_date": entry_time.date().isoformat(),
                "exit_k": int(k_exit),
                "ret_pct": float(r),
            }
        )

    tr = pd.DataFrame(trade_rows)
    if len(tr) == 0:
        raise ValueError("no trades in dataset")

    # daily agg by entry_date
    tr["entry_date"] = pd.to_datetime(tr["entry_date"])
    daily = (
        tr.groupby("entry_date")
        .agg(trades=("trade_id", "count"), mean_ret_pct=("ret_pct", "mean"), sum_ret_pct=("ret_pct", "sum"))
        .reset_index()
        .sort_values("entry_date")
    )

    overall = {
        "n_trades": int(len(tr)),
        "mean_ret_pct": float(tr["ret_pct"].mean()),
        "median_ret_pct": float(tr["ret_pct"].median()),
        "win_rate_gt0_pct": float((tr["ret_pct"] > 0.0).mean() * 100.0),
        "mean_exit_k": float(tr["exit_k"].mean()),
    }

    return {"name": name, "daily": daily, "overall": overall}


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest oracle-gap regressor exit; plot trades/day and mean profitability")
    ap.add_argument("--model-joblib", required=True)
    ap.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Repeated: name=/path/to/dataset.parquet",
    )
    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--min-exit-k", type=int, default=2)
    ap.add_argument("--tau-gap", type=float, default=0.10, help="Fixed tau gap in pct points")
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

    named = [parse_named_dataset(s) for s in list(args.dataset)]

    out_root = Path(args.out_dir)
    out_dir = out_root / f"backtest_gap_timeseries_{now_ts()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    for name, p in named:
        ds_path = Path(p)
        if not ds_path.exists():
            raise SystemExit(f"dataset not found: {ds_path}")

        print(f"Backtesting {name}: {ds_path}", flush=True)
        res = backtest_dataset(
            name=name,
            dataset_parquet=ds_path,
            reg=reg,
            feat_cols=feat_cols,
            hold_min=hold_min,
            min_exit_k=min_exit_k,
            tau_gap=tau_gap,
        )
        results.append(res)

        daily: pd.DataFrame = res["daily"]  # type: ignore[assignment]
        daily.to_csv(out_dir / f"{name}_daily.csv", index=False)

        print(f"  overall: {res['overall']}")

    # Build aligned daily index (union of dates)
    all_dates = pd.Index([])
    for res in results:
        d: pd.DataFrame = res["daily"]  # type: ignore[assignment]
        all_dates = all_dates.union(pd.to_datetime(d["entry_date"]))
    all_dates = all_dates.sort_values()

    # assemble per-series
    series = {}
    for res in results:
        name = str(res["name"])
        d: pd.DataFrame = res["daily"]  # type: ignore[assignment]
        idx = pd.to_datetime(d["entry_date"])
        s_trades = pd.Series(d["trades"].to_numpy(np.float64), index=idx).reindex(all_dates).fillna(0.0)
        s_mean = pd.Series(d["mean_ret_pct"].to_numpy(np.float64), index=idx).reindex(all_dates)
        series[name] = {"trades": s_trades, "mean_ret": s_mean}

    # Plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    ax0 = axes[0]
    for name, s in series.items():
        ax0.plot(all_dates, s["trades"].to_numpy(), label=f"{name} trades/day", linewidth=1.2)
    ax0.set_title(f"Trades per day (exit: pred_gap<= {tau_gap:.2f}pp, min_exit_k={min_exit_k})")
    ax0.set_ylabel("trades/day")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper right")

    ax1 = axes[1]
    for name, s in series.items():
        ax1.plot(all_dates, s["mean_ret"].to_numpy(), label=f"{name} mean_ret%/trade", linewidth=1.2)
    ax1.set_title("Mean profitability per trade by entry day (%)")
    ax1.set_ylabel("mean ret (%)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right")

    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)

    fig.tight_layout()

    out_png = out_dir / "timeseries.png"
    fig.savefig(out_png, dpi=150)

    summary = {
        "model": str(model_path),
        "hold_min": hold_min,
        "min_exit_k": min_exit_k,
        "tau_gap": tau_gap,
        "datasets": [{"name": str(n), "path": str(p)} for n, p in named],
        "overall": {str(res["name"]): res["overall"] for res in results},
        "out_dir": str(out_dir),
        "image": str(out_png),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nWrote: {out_dir}")
    print(f"Image: {out_png}")


if __name__ == "__main__":
    main()
