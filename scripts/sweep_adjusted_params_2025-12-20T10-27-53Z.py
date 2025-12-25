#!/usr/bin/env python3
"""
Sweep adjusted strategy parameters and record timestamped metrics without leakage.
Outputs CSV: data/sweeps/adjusted_param_sweep_<ts>.csv
"""
from __future__ import annotations

import itertools
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts import eval_rule_ranker_strategy as eval_mod  # type: ignore


@dataclass
class Config:
    entry_threshold: float = 0.52
    k_per_day: int = 3
    hold_min: int = 8
    min_profit_pct: float = 0.10
    thr_exit: float = 0.50
    end_date: str = "2025-12-18"
    min_vol_std_5m: Optional[float] = 20.0
    min_range_norm_5m: Optional[float] = 0.004
    rsi_min: Optional[float] = 40.0
    rsi_max: Optional[float] = 60.0
    max_vwap_dev_abs: Optional[float] = 0.20
    allow_hours: Optional[str] = "0,3,9,10,19"
    early_cut_at: int = 7
    enable_peak_stop: bool = True
    peak_stop_dd: float = -0.20
    peak_trigger_min_profit: float = 0.10


def run_one(cfg: Config) -> dict:
    # Build argv to call eval_mod.main()
    out_dir = REPO_ROOT / "data" / "entry_ranked_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    trades_path = out_dir / f"sweep_trades_{ts}.csv"
    daily_path = out_dir / f"sweep_daily_{ts}.csv"
    metrics_path = out_dir / f"sweep_metrics_{ts}.csv"

    argv = [
        "eval_rule_ranker_strategy.py",
        "--entry-threshold", str(cfg.entry_threshold),
        "--k-per-day", str(cfg.k_per_day),
        "--hold-min", str(cfg.hold_min),
        "--min-profit-pct", str(cfg.min_profit_pct),
        "--thr-exit", str(cfg.thr_exit),
        "--end-date", cfg.end_date,
        "--out-trades", str(trades_path),
        "--out-daily", str(daily_path),
        "--metrics-out", str(metrics_path),
    ]
    if cfg.min_vol_std_5m is not None:
        argv += ["--min-vol-std-5m", str(cfg.min_vol_std_5m)]
    if cfg.min_range_norm_5m is not None:
        argv += ["--min-range-norm-5m", str(cfg.min_range_norm_5m)]
    if cfg.rsi_min is not None:
        argv += ["--rsi-min", str(cfg.rsi_min)]
    if cfg.rsi_max is not None:
        argv += ["--rsi-max", str(cfg.rsi_max)]
    if cfg.max_vwap_dev_abs is not None:
        argv += ["--max-vwap-dev-abs", str(cfg.max_vwap_dev_abs)]
    if cfg.allow_hours:
        argv += ["--allow-hours", cfg.allow_hours]
    if cfg.early_cut_at:
        argv += ["--early-cut-at", str(cfg.early_cut_at)]
    if cfg.enable_peak_stop:
        argv += ["--enable-peak-stop", "--peak-stop-dd", str(cfg.peak_stop_dd), "--peak-trigger-min-profit", str(cfg.peak_trigger_min_profit)]

    old = sys.argv
    try:
        sys.argv = argv
        eval_mod.main()
    finally:
        sys.argv = old

    # Read metrics
    m = pd.read_csv(metrics_path).iloc[0].to_dict()
    m.update(asdict(cfg))
    m.update({
        "metrics_path": str(metrics_path),
        "daily_path": str(daily_path),
        "trades_path": str(trades_path),
    })
    return m


def main() -> None:
    ts_all = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_csv = REPO_ROOT / "data" / "sweeps" / f"adjusted_param_sweep_{ts_all}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    base = Config()
    grid = []
    for early_cut_at in [6, 7]:
        for min_profit_pct in [0.05, 0.10]:
            for thr_exit in [0.45, 0.50]:
                for (min_vol, min_rng) in [(10.0, 0.002), (20.0, 0.004)]:
                    for allow_hours in ["0,3,9,10,19", "0,3,9,10"]:
                        cfg = Config(
                            entry_threshold=base.entry_threshold,
                            k_per_day=base.k_per_day,
                            hold_min=base.hold_min,
                            min_profit_pct=min_profit_pct,
                            thr_exit=thr_exit,
                            end_date=base.end_date,
                            min_vol_std_5m=min_vol,
                            min_range_norm_5m=min_rng,
                            rsi_min=base.rsi_min,
                            rsi_max=base.rsi_max,
                            max_vwap_dev_abs=base.max_vwap_dev_abs,
                            allow_hours=allow_hours,
                            early_cut_at=early_cut_at,
                            enable_peak_stop=base.enable_peak_stop,
                            peak_stop_dd=base.peak_stop_dd,
                            peak_trigger_min_profit=base.peak_trigger_min_profit,
                        )
                        grid.append(cfg)

    rows = []
    for i, cfg in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}] Running: {cfg}")
        try:
            rows.append(run_one(cfg))
        except Exception as e:
            rows.append({"error": str(e), **asdict(cfg)})

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved sweep results to:", out_csv)


if __name__ == "__main__":
    main()
