#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    'ret_1m_pct',
    'mom_3m_pct',
    'mom_5m_pct',
    'vol_std_5m',
    'range_5m',
    'range_norm_5m',
    'macd',
    'vwap_dev_5m',
]


def _rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    cs = np.cumsum(x)
    # sum over [i-w+1, i]
    out[w - 1 :] = cs[w - 1 :] - np.concatenate(([0.0], cs[: n - w]))
    return out


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    return _rolling_sum(x, w) / float(w)


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    cs = np.cumsum(x)
    cs2 = np.cumsum(x * x)
    sum_w = cs[w - 1 :] - np.concatenate(([0.0], cs[: n - w]))
    sum2_w = cs2[w - 1 :] - np.concatenate(([0.0], cs2[: n - w]))
    mean = sum_w / float(w)
    var = sum2_w / float(w) - mean * mean
    var = np.maximum(var, 0.0)
    out[w - 1 :] = np.sqrt(var)
    return out


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    # Numpy EMA loop; ~2M points is fine.
    x = x.astype(np.float64, copy=False)
    n = x.size
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    alpha = 2.0 / (float(span) + 1.0)
    out[0] = x[0]
    for i in range(1, n):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def compute_feature_frame(mkt: pd.DataFrame) -> pd.DataFrame:
    # Requires columns: ts_min, close, high, low, volume
    close = mkt['close'].to_numpy(np.float64, copy=False)
    high = mkt['high'].to_numpy(np.float64, copy=False)
    low = mkt['low'].to_numpy(np.float64, copy=False)
    vol = mkt['volume'].to_numpy(np.float64, copy=False)
    n = close.size

    ret_1m = np.zeros(n, dtype=np.float64)
    ret_1m[1:] = (close[1:] / close[:-1] - 1.0) * 100.0

    mom_3m = np.full(n, np.nan, dtype=np.float64)
    mom_5m = np.full(n, np.nan, dtype=np.float64)
    mom_3m[3:] = (close[3:] / close[:-3] - 1.0) * 100.0
    mom_5m[5:] = (close[5:] / close[:-5] - 1.0) * 100.0

    vol_std_5m = _rolling_std(close, 5)

    # range_5m, rolling max/min: use pandas rolling (fast in C)
    s_high = pd.Series(high)
    s_low = pd.Series(low)
    range_5m = (s_high.rolling(5, min_periods=5).max() - s_low.rolling(5, min_periods=5).min()).to_numpy(np.float64)
    range_norm_5m = range_5m / np.maximum(1e-9, close)

    # slope_ols_5m removed (low-signal)
    slope_ols_5m = None

    # rsi_14 removed (low-signal)

    # MACD (12-26); macd_hist removed (low-signal)
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26

    # VWAP dev over last 5 minutes
    pxv = close * vol
    sum_v = _rolling_sum(vol, 5)
    sum_pv = _rolling_sum(pxv, 5)
    # Match monster precontext behavior: if sum_v == 0 => vwap_dev = 0 (not huge).
    vwap = sum_pv / np.maximum(1e-9, sum_v)
    vwap_dev_5m = np.where(sum_v > 0.0, ((close - vwap) / np.maximum(1e-9, vwap)) * 100.0, 0.0)

    # last3_same_sign removed (low-signal)

    feats = pd.DataFrame({
        'ts_min': mkt['ts_min'].to_numpy(dtype='datetime64[ns]'),
        'ret_1m_pct': ret_1m,
        'mom_3m_pct': mom_3m,
        'mom_5m_pct': mom_5m,
        'vol_std_5m': vol_std_5m,
        'range_5m': range_5m,
        'range_norm_5m': range_norm_5m,
        'macd': macd,
        'vwap_dev_5m': vwap_dev_5m,
    })

    return feats


def map_ts_to_index(ts_sorted: np.ndarray, q: np.ndarray) -> np.ndarray:
    # ts_sorted: datetime64[ns] increasing. q: datetime64[ns]
    pos = np.searchsorted(ts_sorted, q)
    ok = (pos < ts_sorted.size) & (ts_sorted[pos] == q)
    out = np.where(ok, pos, -1)
    return out.astype(np.int64)


def net_return_flat_round_trip(entry_px: np.ndarray, exit_px: np.ndarray, fee_round_trip: float) -> np.ndarray:
    fee_side = fee_round_trip / 2.0
    mult = (exit_px * (1.0 - fee_side)) / (entry_px * (1.0 + fee_side))
    return (mult - 1.0) * 100.0


def build_dyn5(feat_arrs: dict[str, np.ndarray], idx: np.ndarray, offsets: np.ndarray) -> pd.DataFrame:
    # idx shape (m,), offsets shape (k,)
    m = idx.size
    k = offsets.size
    mat = idx[:, None] + offsets[None, :]

    flat_idx = mat.reshape(-1)
    rel = np.tile(offsets, m)
    trade_row = np.repeat(np.arange(m, dtype=np.int64), k)

    valid = flat_idx >= 0
    flat_idx2 = flat_idx[valid]

    out = {
        'trade_row': trade_row[valid],
        'rel_min': rel[valid],
    }
    for name, arr in feat_arrs.items():
        out[name] = arr[flat_idx2]
    return pd.DataFrame(out)


def summarize_dyn(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    # Returns mean/median for each feature by group and rel_min
    agg = {}
    for f in FEATURES:
        agg[f'{f}__mean'] = (f, 'mean')
        agg[f'{f}__median'] = (f, 'median')
    # pandas named agg requires dict mapping col->list, so do it explicitly
    grouped = df.groupby(group_cols, as_index=False)
    pieces = []
    for key, g in grouped:
        row = {}
        if not isinstance(key, tuple):
            key = (key,)
        for c, v in zip(group_cols, key):
            row[c] = v
        for f in FEATURES:
            s = g[f]
            row[f'{f}__mean'] = float(s.mean())
            row[f'{f}__median'] = float(s.median())
        pieces.append(row)
    return pd.DataFrame(pieces)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--market-cache', required=True, help='CSV with timestamp, open, high, low, close, volume')
    ap.add_argument('--agent-trades', required=True, help='Agent trades CSV (monster) with entry_time/exit_time')
    ap.add_argument('--oracle-trades', required=True, help='Oracle trades CSV with entry_time/exit_time')
    ap.add_argument('--fee-round-trip', type=float, default=0.001)
    ap.add_argument('--out-dir', default='data/pattern_runs')
    ap.add_argument('--tag', default='crossref')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load market
    mkt = pd.read_csv(args.market_cache, parse_dates=['timestamp'], usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    mkt['ts_min'] = pd.to_datetime(mkt['timestamp']).dt.floor('min')
    mkt = mkt.sort_values('ts_min').reset_index(drop=True)

    ts_sorted = mkt['ts_min'].to_numpy(dtype='datetime64[ns]')

    # Precompute feature arrays once
    feats = compute_feature_frame(mkt)
    feat_arrs = {f: feats[f].to_numpy(np.float64, copy=False) if f != 'last3_same_sign' else feats[f].to_numpy(np.int8, copy=False).astype(np.float64) for f in FEATURES}

    # Load trades
    agent = pd.read_csv(args.agent_trades, parse_dates=['entry_time', 'exit_time'])
    oracle = pd.read_csv(args.oracle_trades, parse_dates=['entry_time', 'exit_time'])

    # Normalize timestamps to minute
    agent['entry_min'] = pd.to_datetime(agent['entry_time']).dt.floor('min')
    agent['exit_min'] = pd.to_datetime(agent['exit_time']).dt.floor('min')
    oracle['entry_min'] = pd.to_datetime(oracle['entry_time']).dt.floor('min')
    oracle['exit_min'] = pd.to_datetime(oracle['exit_time']).dt.floor('min')

    agent_entry_idx = map_ts_to_index(ts_sorted, agent['entry_min'].to_numpy(dtype='datetime64[ns]'))
    agent_exit_idx = map_ts_to_index(ts_sorted, agent['exit_min'].to_numpy(dtype='datetime64[ns]'))
    oracle_entry_idx = map_ts_to_index(ts_sorted, oracle['entry_min'].to_numpy(dtype='datetime64[ns]'))
    oracle_exit_idx = map_ts_to_index(ts_sorted, oracle['exit_min'].to_numpy(dtype='datetime64[ns]'))

    # Filter out unmatched
    agent_ok = (agent_entry_idx >= 0) & (agent_exit_idx >= 0)
    oracle_ok = (oracle_entry_idx >= 0) & (oracle_exit_idx >= 0)
    agent = agent.loc[agent_ok].reset_index(drop=True)
    oracle = oracle.loc[oracle_ok].reset_index(drop=True)
    agent_entry_idx = agent_entry_idx[agent_ok]
    agent_exit_idx = agent_exit_idx[agent_ok]
    oracle_entry_idx = oracle_entry_idx[oracle_ok]
    oracle_exit_idx = oracle_exit_idx[oracle_ok]

    # Compute comparable net returns under flat fee for both
    close = mkt['close'].to_numpy(np.float64, copy=False)

    agent_entry_px = close[agent_entry_idx]
    agent_exit_px = close[agent_exit_idx]
    agent['net_return_flat_pct'] = net_return_flat_round_trip(agent_entry_px, agent_exit_px, args.fee_round_trip)
    agent['win_flat'] = (agent['net_return_flat_pct'] > 0.0).astype(int)

    oracle_entry_px = close[oracle_entry_idx]
    oracle_exit_px = close[oracle_exit_idx]
    oracle['net_return_flat_pct'] = net_return_flat_round_trip(oracle_entry_px, oracle_exit_px, args.fee_round_trip)

    # Build dynamic 5m-before windows
    offsets = np.array([-5, -4, -3, -2, -1], dtype=np.int64)

    # Agent entry/exit
    a_pre_e = build_dyn5(feat_arrs, agent_entry_idx, offsets)
    a_pre_x = build_dyn5(feat_arrs, agent_exit_idx, offsets)
    a_pre_e['which'] = 'agent'; a_pre_e['phase'] = 'pre_entry'
    a_pre_x['which'] = 'agent'; a_pre_x['phase'] = 'pre_exit'

    # Oracle entry/exit
    o_pre_e = build_dyn5(feat_arrs, oracle_entry_idx, offsets)
    o_pre_x = build_dyn5(feat_arrs, oracle_exit_idx, offsets)
    o_pre_e['which'] = 'oracle'; o_pre_e['phase'] = 'pre_entry'
    o_pre_x['which'] = 'oracle'; o_pre_x['phase'] = 'pre_exit'

    # Attach trade-level fields
    # trade_row maps to row in respective df; add duration and net_return
    a_meta = agent[['entry_min', 'exit_min', 'duration_min', 'net_return_flat_pct', 'win_flat']].copy()
    o_meta = oracle[['entry_min', 'exit_min', 'duration_min', 'net_return_flat_pct']].copy()

    a_pre_e = a_pre_e.merge(a_meta.reset_index().rename(columns={'index': 'trade_row'}), on='trade_row', how='left')
    a_pre_x = a_pre_x.merge(a_meta.reset_index().rename(columns={'index': 'trade_row'}), on='trade_row', how='left')
    o_pre_e = o_pre_e.merge(o_meta.reset_index().rename(columns={'index': 'trade_row'}), on='trade_row', how='left')
    o_pre_x = o_pre_x.merge(o_meta.reset_index().rename(columns={'index': 'trade_row'}), on='trade_row', how='left')

    dyn = pd.concat([a_pre_e, a_pre_x, o_pre_e, o_pre_x], ignore_index=True)

    # Write parquet
    dyn_path = out_dir / f'{args.tag}_dyn5m.parquet'
    dyn.to_parquet(dyn_path, index=False)

    # Summaries
    sum_all = summarize_dyn(dyn, ['which', 'phase', 'rel_min'])
    sum_path = out_dir / f'{args.tag}_dyn5m_summary.csv'
    sum_all.to_csv(sum_path, index=False)

    # Agent winners/losers summary
    sum_agent_wl = summarize_dyn(dyn[dyn['which'] == 'agent'].assign(win_flat=dyn[dyn['which'] == 'agent']['win_flat']), ['phase', 'rel_min', 'win_flat'])
    sum_agent_wl_path = out_dir / f'{args.tag}_agent_winloss_dyn5m_summary.csv'
    sum_agent_wl.to_csv(sum_agent_wl_path, index=False)

    # Quick headline stats
    print('Loaded trades:')
    print(f'  agent trades : {len(agent):,}  (win_flat={agent["win_flat"].mean():.4f})')
    print(f'  oracle trades: {len(oracle):,}  (all net positive by construction? min net={oracle["net_return_flat_pct"].min():.6f}%)')
    print('Outputs:')
    print(f'  {dyn_path}')
    print(f'  {sum_path}')
    print(f'  {sum_agent_wl_path}')


if __name__ == '__main__':
    main()
