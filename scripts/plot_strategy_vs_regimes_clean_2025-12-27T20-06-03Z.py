#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({
    'figure.figsize': (16, 6),
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.color': '#e6e6e6',
    'grid.linestyle': '-',
    'grid.linewidth': 0.6,
    'font.size': 10,
})


def read_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)


def compute_equity(daily: pd.DataFrame, start_capital: float) -> pd.DataFrame:
    df = daily.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    ret = df['daily_return_pct'].fillna(0.0).to_numpy(dtype=float) / 100.0
    mult = np.cumprod(1.0 + ret)
    df['equity'] = start_capital * mult
    return df[['date','equity']]


def main():
    ap = argparse.ArgumentParser(description='Clean plot: weekly trades and equity with regime overlays')
    ap.add_argument('--trades-csv', required=True)
    ap.add_argument('--daily-csv', required=True)
    ap.add_argument('--segmented-daily', required=True)
    ap.add_argument('--start-capital', type=float, default=30.0)
    ap.add_argument('--out', default='data/plots/strategy_vs_regimes_clean.png')
    args = ap.parse_args()

    trades = read_df(Path(args.trades_csv))
    daily = read_df(Path(args.daily_csv))
    seg = read_df(Path(args.segmented_daily))

    trades['date'] = pd.to_datetime(trades['date'])
    daily['date'] = pd.to_datetime(daily['date'])
    seg['date'] = pd.to_datetime(seg['date'])

    # Focus the x-axis to where strategy is active
    if len(trades):
        start = trades['date'].min() - pd.Timedelta(days=7)
        end = trades['date'].max() + pd.Timedelta(days=7)
        seg = seg[(seg['date'] >= start) & (seg['date'] <= end)].copy()
        daily = daily[(daily['date'] >= start) & (daily['date'] <= end)].copy()

    # Weekly trade counts (smoother and more visible)
    tc = trades.groupby('date').size().rename('n').reset_index()
    tc = tc.set_index('date').resample('W').sum().reset_index()

    # Equity curve (log scale can help show relative moves)
    eq = compute_equity(daily, float(args.start_capital))

    # Prepare regime frame (daily)
    frame = pd.date_range(seg['date'].min(), seg['date'].max(), freq='D') if len(seg) else pd.DatetimeIndex([])
    reg = seg.set_index('date')[['main_type','subtype']].reindex(frame).ffill().reset_index().rename(columns={'index':'date'})

    fig, ax = plt.subplots()

    # Regime shading blocks
    main_colors = {'TREND_UP':'#87d37c', 'TREND_DOWN':'#f5a5a5', 'CHOP':'#c7c7c7'}
    if len(reg):
        dates = reg['date'].to_numpy()
        mains = reg['main_type'].to_numpy()
        last = 0
        for i in range(1, len(reg)):
            if mains[i] != mains[i-1]:
                ax.axvspan(dates[last], dates[i], color=main_colors.get(mains[i-1],'#dddddd'), alpha=0.12, linewidth=0)
                last = i
        ax.axvspan(dates[last], dates[-1], color=main_colors.get(mains[-1],'#dddddd'), alpha=0.12, linewidth=0)

    # Twin axis: bars for weekly trades, line for equity
    ax2 = ax.twinx()
    if len(tc):
        ax2.bar(tc['date'], tc['n'], width=6, align='center', color='#1f77b4', alpha=0.5, label='Trades/week')
        ax2.set_ylabel('Trades/week')

    if len(eq):
        ax.plot(eq['date'], eq['equity'], color='#0d4b8f', linewidth=1.8, label='Equity')
        ax.set_ylabel('Equity (log)')
        ax.set_yscale('log')

    # Subtype ribbon at bottom
    if len(reg):
        subcats = sorted(reg['subtype'].dropna().unique())
        cmap = {s: plt.cm.tab20(i/20) for i,s in enumerate(subcats)}
        y0, y1 = ax.get_ylim()
        y = y0 * (0.999 if ax.get_yscale()=='log' else 1.0)  # small offset at bottom
        ax.scatter(reg['date'], np.full(len(reg), y), c=[cmap.get(s, (0,0,0,0.2)) for s in reg['subtype']], s=8, marker='s', alpha=0.85)

    # Legend and formatting
    ax.set_title('Strategy vs regimes (noise-based cleanliness): equity and weekly trades')
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(True, axis='y')

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print('Wrote plot:', out_path)

if __name__ == '__main__':
    main()
