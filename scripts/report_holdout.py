#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import subprocess
import pandas as pd

METRICS_DIR = Path('data/cv_metrics')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--holdout-start', type=str, default=None)
    args = ap.parse_args()

    best_path = METRICS_DIR / 'best_cv.json'
    if not best_path.exists():
        raise SystemExit('Missing data/cv_metrics/best_cv.json — run scripts/cv_time_series.py first')
    with open(best_path, 'r') as f:
        best = json.load(f)
    k = int(best['k'])
    thr = float(best['thr_exit'])

    # Determine holdout start from splits or arg
    splits_path = Path('data/splits.json')
    if args.holdout_start:
        holdout_start = args.holdout_start
    elif splits_path.exists():
        with open(splits_path, 'r') as f:
            holdout_start = json.load(f).get('holdout_start')
    else:
        holdout_start = '2025-10-01'

    out_csv = METRICS_DIR / 'holdout_metrics.csv'
    cmd = [
        'python3','scripts/eval_buy_k3_filtered_exitmodel.py',
        '--start-date', holdout_start,
        '--end-date', '2025-12-31',
        '--k-per-day', str(k),
        '--thr-exit', str(thr),
        '--metrics-out', str(out_csv),
        '--allow-holdout'
    ]
    subprocess.run(cmd, check=True)

    df = pd.read_csv(out_csv)
    print('Holdout summary:')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()