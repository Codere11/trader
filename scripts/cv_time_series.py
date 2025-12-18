#!/usr/bin/env python3
import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# We reuse eval_buy_k3_filtered_exitmodel.py via subprocess for evaluation
import subprocess

SPLITS_PATH = Path('data/splits.json')
METRICS_DIR = Path('data/cv_metrics')
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def month_range(df_dates):
    months = sorted({(d.year, d.month) for d in df_dates})
    return [pd.Timestamp(year=y, month=m, day=1) for (y, m) in months]


def build_folds(dates: pd.Series, embargo_days: int = 1, n_val_months: int = 1):
    """Builds purged monthly folds.

    Each fold:
    - Train on all data from start up to (val_start - embargo_days).
    - Validate on a single month.
    """
    months = month_range(dates)
    folds = []
    # Loop from n_val_months to len(months) to ensure there's at least one month for validation
    for i in range(n_val_months, len(months)):
        val_start = months[i]
        val_end = (val_start + pd.offsets.MonthEnd(1)).normalize()  # Ensure end of day
        train_end = val_start - pd.Timedelta(days=embargo_days)

        folds.append({
            'train_start': str(dates.min().date()),
            'train_end': str(train_end.date()),
            'val_start': str(val_start.date()),
            'val_end': str(val_end.date()),
        })
    return folds

def run_eval(start_date: str, end_date: str, k_per_day: int, thr_exit: float, out_csv: Path, train_end_date_for_filter: str):
    \"\"\"Runs eval script; passes train_end_date for filtering ranked entries.\"\"\"
    cmd = [
        'python3', 'scripts/eval_buy_k3_filtered_exitmodel.py',
        '--start-date', start_date,
        '--end-date', end_date,
        '--k-per-day', str(k_per_day),
        '--thr-exit', str(thr_exit),
        '--metrics-out', str(out_csv),
        '--train-end-date-for-filter', train_end_date_for_filter, # Pass train end date
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k-grid', type=str, default='2,3,4')
    ap.add_argument('--thr-grid', type=str, default='0.45,0.50,0.55,0.60')
    ap.add_argument('--embargo-days', type=int, default=1)
    ap.add_argument('--holdout-start', type=str, default='2025-10-01')
    args = ap.parse_args()

    # Record splits
    splits = {
        'holdout_start': args.holdout_start,
        'embargo_days': args.embargo_days,
    }
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_PATH, 'w') as f:
        json.dump(splits, f, indent=2)

    # Load ranked entries to derive date universe
    E = pd.read_csv('data/entry_ranked_outputs/entries_with_best8_pnl.csv', parse_dates=['timestamp'])
    E['date'] = pd.to_datetime(E['date']).dt.date
    dates = pd.to_datetime(E['date'])

    # Build folds up to holdout
    holdout_start = pd.Timestamp(args.holdout_start)
    in_sample_mask = (dates < holdout_start)
    folds = build_folds(pd.to_datetime(dates[in_sample_mask]), embargo_days=args.embargo_days)

    k_vals = [int(x) for x in args.k_grid.split(',') if x]
    thr_vals = [float(x) for x in args.thr_grid.split(',') if x]

    results = []
    for f in folds:
        for k in k_vals:
            for thr in thr_vals:
                out_csv = METRICS_DIR / f"cv_{f['val_start']}_{f['val_end']}_k{k}_thr{thr:.2f}.csv"
                run_eval(f['val_start'], f['val_end'], k, thr, out_csv, f['train_end'])
                m = pd.read_csv(out_csv).iloc[0].to_dict()
                m.update({'k': k, 'thr_exit': thr, 'fold': f})
                results.append(m)

    if not results:
        print('No CV results produced.')
        return

    df = pd.DataFrame(results)
    agg = df.groupby(['k','thr_exit']).agg({'daily_mean_pct':'mean','max_dd_pct':'mean','sortino':'mean','final_capital':'mean'}).reset_index()
    agg.sort_values(['sortino','daily_mean_pct'], ascending=[False, False], inplace=True)
    best = agg.iloc[0].to_dict()
    with open(METRICS_DIR / 'best_cv.json', 'w') as f:
        json.dump(best, f, indent=2, default=str)
    print('Best CV params:', best)


if __name__ == '__main__':
    main()