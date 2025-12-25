#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
import time, sys, math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Reuse feature computation
import sys
from pathlib import Path as _Path
_sys_root = _Path(__file__).resolve().parent
if str(_sys_root) not in sys.path:
    sys.path.append(str(_sys_root))
from precompute_features import precompute as compute_features


@dataclass
class EntryParams:
    range_norm_min: float
    vwap_dev_max: float  # at/below this (e.g., 0 or negative)
    require_any_neg_mom: bool  # mom_3m<=0 or mom_5m<=0

@dataclass
class ExitParams:
    mom3_min: float
    vwap_min: float
    slope_min: float
    rsi_min: float
    mom3_inval: float  # e.g., -0.03
    vwap_inval: float  # e.g., -0.04
    max_hold: int      # minutes


def load_market(csv_path: Path, start: str | None, end: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df['ts_min'] = pd.to_datetime(df['timestamp']).dt.floor('min')
    df = (df.groupby('ts_min')
            .agg(open=('open','first'), high=('high','max'), low=('low','min'), close=('close','last'), volume=('volume','sum'))
            .reset_index()
         )
    if start:
        df = df[df['ts_min'] >= pd.to_datetime(start)]
    if end:
        df = df[df['ts_min'] <= pd.to_datetime(end)]
    df = df.sort_values('ts_min').reset_index(drop=True)
    return df


def build_feature_frame(mkt: pd.DataFrame) -> pd.DataFrame:
    raw = mkt.rename(columns={'ts_min':'timestamp'})
    t0 = time.time()
    print(f"[features] Building features for {len(raw):,} minutes ...", flush=True)
    feat = compute_features(raw)
    dt = time.time() - t0
    print(f"[features] Done in {dt:.1f}s ({len(raw)/max(dt,1):.1f} rows/s)", flush=True)
    # Normalize names used in rules
    out = feat.rename(columns={'timestamp':'ts_min','vwap_dev':'vwap_dev'})
    return out


def rule_entry_mask(df: pd.DataFrame, ep: EntryParams) -> np.ndarray:
    cond = (df['range_norm_5m'] >= ep.range_norm_min) & (df['vwap_dev'] <= ep.vwap_dev_max)
    if ep.require_any_neg_mom:
        cond &= ((df['mom_3m'] <= 0.0) | (df['mom_5m'] <= 0.0))
    return cond.to_numpy()


def rule_exit_trigger(df: pd.DataFrame, i: int, start_i: int, xp: ExitParams) -> bool:
    dur = i - start_i
    if (df['mom_3m'].iat[i] >= xp.mom3_min) or (df['vwap_dev'].iat[i] >= xp.vwap_min) or (df['slope_ols_5m'].iat[i] >= xp.slope_min) or (df['rsi_14'].iat[i] >= xp.rsi_min):
        return True
    if (df['mom_3m'].iat[i] <= xp.mom3_inval) or (df['vwap_dev'].iat[i] <= xp.vwap_inval) or (dur >= xp.max_hold):
        return True
    return False


def simulate_rule_rule(df: pd.DataFrame, ep: EntryParams, xp: ExitParams) -> pd.DataFrame:
    in_pos = False
    entry_i = -1
    rows = []
    mask = rule_entry_mask(df, ep)
    n = len(df)
    for i in range(n):
        if not in_pos:
            if mask[i]:
                in_pos = True
                entry_i = i
                entry_price = float(df['close'].iat[i])
                entry_ts = df['ts_min'].iat[i]
        else:
            if rule_exit_trigger(df, i, entry_i, xp):
                exit_price = float(df['close'].iat[i])
                rows.append({
                    'entry_ts': entry_ts,
                    'exit_ts': df['ts_min'].iat[i],
                    'entry_idx': entry_i,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'duration_min': i - entry_i,
                    'net_return_flat_pct': (exit_price - entry_price) / entry_price * 100.0,
                    'model': 'rule->rule'
                })
                in_pos = False
                entry_i = -1
    return pd.DataFrame(rows)


def simulate_rule_oracle_exit(df: pd.DataFrame, ep: EntryParams, max_hold: int = 12) -> pd.DataFrame:
    in_pos = False
    entry_i = -1
    rows = []
    mask = rule_entry_mask(df, ep)
    n = len(df)
    for i in range(n):
        if not in_pos:
            if mask[i]:
                in_pos = True
                entry_i = i
                entry_price = float(df['close'].iat[i])
                entry_ts = df['ts_min'].iat[i]
        else:
            # oracle exit within [entry_i+1, entry_i+max_hold]
            j_end = min(entry_i + max_hold, n - 1)
            window = df['close'].iloc[entry_i+1:j_end+1].to_numpy()
            if window.size == 0:
                continue
            j_rel = int(np.argmax(window))
            j = entry_i + 1 + j_rel
            exit_price = float(df['close'].iat[j])
            rows.append({
                'entry_ts': entry_ts,
                'exit_ts': df['ts_min'].iat[j],
                'entry_idx': entry_i,
                'exit_idx': j,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'duration_min': j - entry_i,
                'net_return_flat_pct': (exit_price - entry_price) / entry_price * 100.0,
                'model': 'rule->oracle'
            })
            in_pos = False
            entry_i = -1
    return pd.DataFrame(rows)


def simulate_oracle_entry_rule_exit(df: pd.DataFrame, xp: ExitParams, max_hold: int = 12) -> pd.DataFrame:
    in_pos = False
    entry_i = -1
    rows = []
    n = len(df)
    for i in range(n):
        if not in_pos:
            # oracle entry if any future close within 12 min higher than current
            j_end = min(i + max_hold, n - 1)
            future = df['close'].iloc[i+1:j_end+1].to_numpy()
            if future.size == 0:
                continue
            if np.max(future) > df['close'].iat[i]:
                in_pos = True
                entry_i = i
                entry_price = float(df['close'].iat[i])
                entry_ts = df['ts_min'].iat[i]
        else:
            if rule_exit_trigger(df, i, entry_i, xp):
                exit_price = float(df['close'].iat[i])
                rows.append({
                    'entry_ts': entry_ts,
                    'exit_ts': df['ts_min'].iat[i],
                    'entry_idx': entry_i,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'duration_min': i - entry_i,
                    'net_return_flat_pct': (exit_price - entry_price) / entry_price * 100.0,
                    'model': 'oracle->rule'
                })
                in_pos = False
                entry_i = -1
    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {'n': 0, 'win_rate': np.nan, 'mean_ret': np.nan, 'med_ret': np.nan, 'cum_ret_pct': 0.0}
    n = len(trades)
    wr = float((trades['net_return_flat_pct'] > 0).mean())
    mean_ret = float(trades['net_return_flat_pct'].mean())
    med_ret = float(trades['net_return_flat_pct'].median())
    cum = float((trades['net_return_flat_pct'] / 100.0 + 1.0).prod() - 1.0)
    return {'n': n, 'win_rate': wr, 'mean_ret': mean_ret, 'med_ret': med_ret, 'cum_ret_pct': cum * 100.0}


def _run_with_progress(tag_prefix: str, combos: list[tuple], runner, out_dir: Path, log_rows: list[dict]):
    total = len(combos)
    t0 = time.time(); last = t0
    for idx, params in enumerate(combos, 1):
        start = time.time()
        trades, meta = runner(*params)
        summ = summarize(trades)
        tag = f"{tag_prefix}__{meta['tag']}"
        trades.to_parquet(out_dir / f"{tag}.parquet", index=False)
        log_rows.append({'task': tag_prefix, 'tag': tag, **summ, **meta})
        # incremental log write for safety
        pd.DataFrame(log_rows).to_csv(out_dir / 'sweep_log.csv', index=False)
        # progress line
        now = time.time()
        elapsed = now - t0
        rate = idx / max(elapsed, 1e-9)
        remaining = (total - idx) / max(rate, 1e-9)
        print(f"[{tag_prefix}] {idx}/{total} ({idx/total*100:.1f}%) in {elapsed:.1f}s | {rate:.2f} it/s | ETA {remaining:.1f}s | last {now-start:.2f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--market-csv', required=True)
    ap.add_argument('--start', default='2022-01-01')
    ap.add_argument('--end', default='2025-12-18')
    ap.add_argument('--out-dir', default='data/pattern_runs/rule_sweeps')
    ap.add_argument('--tasks', nargs='+', default=['rule_oracle', 'oracle_rule', 'rule_rule'])
    # modest grids by default
    ap.add_argument('--grid-range', nargs='+', type=float, default=[0.0012, 0.0015, 0.0020])
    ap.add_argument('--grid-vwapmax', nargs='+', type=float, default=[0.0, -0.005, -0.01])
    ap.add_argument('--grid-momneg', nargs='+', type=int, default=[1])
    ap.add_argument('--grid-mom3min', nargs='+', type=float, default=[0.03, 0.05])
    ap.add_argument('--grid-vwapmin', nargs='+', type=float, default=[0.02, 0.03])
    ap.add_argument('--grid-slopemin', nargs='+', type=float, default=[3.0, 5.0])
    ap.add_argument('--grid-rsimin', nargs='+', type=float, default=[55.0])
    ap.add_argument('--mom3-inval', type=float, default=-0.03)
    ap.add_argument('--vwap-inval', type=float, default=-0.04)
    ap.add_argument('--max-hold', type=int, default=12)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    mkt = load_market(Path(args.market_csv), args.start, args.end)
    feat = build_feature_frame(mkt)
    # Features frame already includes close; just drop rows without close
    df = feat.dropna(subset=['close']).reset_index(drop=True)

    logs = []

    # 1) rule entry -> oracle exit
    if 'rule_oracle' in args.tasks:
        combos = [(rn, vw, mn) for rn, vw, mn in product(args.grid_range, args.grid_vwapmax, args.grid_momneg)]
        def run_rule_oracle(rn, vw, mn):
            ep = EntryParams(rn, vw, bool(mn))
            trades = simulate_rule_oracle_exit(df, ep, max_hold=args.max_hold)
            return trades, {'tag': f"rn{rn:.4f}__vw{vw:.3f}__mn{int(mn)}", **ep.__dict__}
        _run_with_progress('rule_oracle', combos, run_rule_oracle, out_dir, logs)

    # 2) oracle entry -> rule exit
    if 'oracle_rule' in args.tasks:
        combos = [(m3, vwmn, slp, rsi) for m3, vwmn, slp, rsi in product(args.grid_mom3min, args.grid_vwapmin, args.grid_slopemin, args.grid_rsimin)]
        def run_oracle_rule(m3, vwmn, slp, rsi):
            xp = ExitParams(m3, vwmn, slp, rsi, args.mom3_inval, args.vwap_inval, args.max_hold)
            trades = simulate_oracle_entry_rule_exit(df, xp, max_hold=args.max_hold)
            return trades, {'tag': f"m3{m3:.3f}__vw{vwmn:.3f}__sl{slp:.1f}__rsi{rsi:.1f}", **xp.__dict__}
        _run_with_progress('oracle_rule', combos, run_oracle_rule, out_dir, logs)

    # 3) rule entry -> rule exit sweeps
    if 'rule_rule' in args.tasks:
        combos = [(rn, vw, mn, m3, vwmn, slp, rsi) for rn, vw, mn, m3, vwmn, slp, rsi in product(
            args.grid_range, args.grid_vwapmax, args.grid_momneg,
            args.grid_mom3min, args.grid_vwapmin, args.grid_slopemin, args.grid_rsimin
        )]
        def run_rule_rule(rn, vw, mn, m3, vwmn, slp, rsi):
            ep = EntryParams(rn, vw, bool(mn))
            xp = ExitParams(m3, vwmn, slp, rsi, args.mom3_inval, args.vwap_inval, args.max_hold)
            trades = simulate_rule_rule(df, ep, xp)
            return trades, {'tag': f"rn{rn:.4f}__vw{vw:.3f}__mn{int(mn)}__m3{m3:.3f}__vwmin{vwmn:.3f}__sl{slp:.1f}__rsi{rsi:.1f}", **ep.__dict__, **xp.__dict__}
        _run_with_progress('rule_rule', combos, run_rule_rule, out_dir, logs)

    log_df = pd.DataFrame(logs).sort_values(['task','cum_ret_pct'], ascending=[True, False])
    log_df.to_csv(out_dir / 'sweep_log.csv', index=False)
    print(f"Wrote {len(logs)} sweep runs to {out_dir}/sweep_log.csv")


if __name__ == '__main__':
    main()