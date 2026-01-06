#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def read_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)


def compute_equity_from_daily(daily: pd.DataFrame, start_capital: float) -> pd.DataFrame:
    df = daily.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    ret = df['daily_return_pct'].fillna(0.0).to_numpy(dtype=float) / 100.0
    mult = np.cumprod(1.0 + ret)
    df['equity'] = start_capital * mult
    return df[['date','equity']]


def main():
    ap = argparse.ArgumentParser(description='Plot strategy trades and equity over time with regime overlays')
    ap.add_argument('--trades-csv', required=True)
    ap.add_argument('--daily-csv', required=True)
    ap.add_argument('--segmented-daily', required=True, help='segmented_tree_noise_*.{parquet,csv}')
    ap.add_argument('--start-capital', type=float, default=30.0)
    ap.add_argument('--out', default='data/plots/strategy_vs_regimes.png')
    args = ap.parse_args()

    trades = read_df(Path(args.trades_csv))
    daily = read_df(Path(args.daily_csv))
    seg = read_df(Path(args.segmented_daily))

    # Normalize dates
    trades['date'] = pd.to_datetime(trades['date'])
    daily['date'] = pd.to_datetime(daily['date'])
    seg['date'] = pd.to_datetime(seg['date'])

    # Daily trade counts
    tc = trades.groupby(trades['date']).size().rename('n_trades').reset_index()

    # Equity curve
    eq = compute_equity_from_daily(daily, float(args.start_capital))

    # Merge for plotting convenience
    frame = pd.date_range(seg['date'].min(), seg['date'].max(), freq='D')
    reg = seg.set_index('date')[['main_type','subtype']].reindex(frame).ffill().reset_index().rename(columns={'index':'date'})
    counts = reg.merge(tc, on='date', how='left').fillna({'n_trades':0})
    eq2 = reg.merge(eq, on='date', how='left').ffill()

    # Colors per main regime
    main_colors = {
        'TREND_UP': '#2ca02c',
        'TREND_DOWN': '#d62728',
        'CHOP': '#7f7f7f'
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios':[1,2]})

    # Background regime shading on ax2
    prev_main = None
    start_idx = 0
    dates = reg['date'].to_numpy()
    mains = reg['main_type'].to_numpy()
    subs = reg['subtype'].to_numpy()
    for i in range(len(reg)):
        if i == 0:
            prev_main = mains[i]
            start_idx = i
            continue
        if mains[i] != prev_main or i == len(reg)-1:
            end_i = i if mains[i] != prev_main else i
            ax2.axvspan(dates[start_idx], dates[end_i], color=main_colors.get(prev_main,'#cccccc'), alpha=0.08)
            # Label subtype in the middle
            mid = dates[start_idx + (end_i-start_idx)//2]
            ax2.text(mid, ax2.get_ylim()[1], f"{prev_main}", va='top', ha='center', fontsize=9, alpha=0.6)
            prev_main = mains[i]
            start_idx = i

    # Plot daily trade counts (bars)
    ax1.bar(counts['date'], counts['n_trades'], width=0.9, color='#1f77b4')
    ax1.set_ylabel('Trades/day')

    # Plot equity curve
    ax2.plot(eq2['date'], eq2['equity'], color='#1f77b4', linewidth=1.5, label='Equity')
    ax2.set_ylabel('Equity')

    # Improve x-axis
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))

    # Secondary annotation: subtype bands as small colored markers on bottom
    subtype_codes = {s:i for i,s in enumerate(sorted(seg['subtype'].unique()))}
    y0 = ax2.get_ylim()[0]
    y_band = y0 + 0.02*(ax2.get_ylim()[1]-y0)
    colors = plt.cm.tab20(np.linspace(0,1,len(subtype_codes)))
    cmap = {s: colors[i] for s,i in subtype_codes.items()}
    ax2.scatter(reg['date'], np.full(len(reg), y_band), c=[cmap[s] for s in subs], s=2, alpha=0.8, marker='s')

    ax2.set_title('Main strategy vs regimes (noise-based cleanliness)')

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print('Wrote plot:', out_path)

if __name__ == '__main__':
    main()
