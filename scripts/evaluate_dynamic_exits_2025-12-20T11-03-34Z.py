#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts import eval_rule_ranker_strategy as eval_mod  # type: ignore


def main():
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')
    out_dir = REPO_ROOT / 'data' / 'entry_ranked_outputs'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_trades = out_dir / f'dyn_exit_trades_{ts}.csv'
    out_daily = out_dir / f'dyn_exit_daily_{ts}.csv'
    out_metrics = out_dir / f'dyn_exit_metrics_{ts}.csv'

    argv = [
        'eval_rule_ranker_strategy.py',
        '--entry-threshold','0.52',
        '--k-per-day','3',
        '--hold-min','8',
        '--min-profit-pct','0.05',
        '--thr-exit','0.45',
        '--end-date','2025-12-18',
        '--min-vol-std-5m','10',
'--min-range-norm-5m','0.003',
'--min-mom-3m-pct','0',
'--rsi-min','40','--rsi-max','60',
'--max-vwap-dev-abs','0.20',
'--allow-hours','0,3,9,10,19',
        '--early-cut-at','6',
        '--enable-peak-stop','--peak-stop-dd','-0.20','--peak-trigger-min-profit','0.10',
        '--out-trades', str(out_trades),
        '--out-daily', str(out_daily),
        '--metrics-out', str(out_metrics),
    ]

    old=sys.argv
    try:
        sys.argv=argv
        eval_mod.main()
    finally:
        sys.argv=old

    print('Trades:', out_trades)
    print('Daily:', out_daily)
    print('Metrics:', out_metrics)

if __name__ == '__main__':
    main()
