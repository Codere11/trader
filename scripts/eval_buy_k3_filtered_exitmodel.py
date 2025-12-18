#!/usr/bin/env python3
import os
import argparse
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

RANKED_PATH = 'data/entry_ranked_outputs/entries_with_best8_pnl.csv'
BASE_PATH = 'data/btc_profitability_analysis_filtered.csv'
EXIT_MODEL_MAIN = 'models/exit_timing_model_weighted.joblib'
EXIT_MODEL_FALLBACK = 'models/exit_timing_model.joblib'
PRE_FILTER_PATH = 'models/preentry_mistake_filter.joblib'

K_PER_DAY = 3
HOLD_MIN = 8
FEE_RATE = 0.001
THR_EXIT = 0.50
MIN_PROFIT_PCT = 0.10

OUT_TRADES = Path('data/entry_ranked_outputs/buy_k3_filtered_exitmodel_trades.csv')
OUT_DAILY  = Path('data/entry_ranked_outputs/buy_k3_filtered_exitmodel_daily.csv')

EXIT_FEATURE_COLS = [
    'elapsed_min','since_entry_return_pct','prob_enter_at_entry','daily_pred_high_profit',
    'momentum_3m','momentum_5m','momentum_10m','volatility_3m','volatility_5m','volatility_10m',
    'drawdown_from_max_pct','since_entry_cummax','mins_since_peak'
]

@dataclass
class TradeResult:
    date: object
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    duration_min: int
    trade_return_pct: float


def build_precontext_window(day: pd.DataFrame, ts: pd.Timestamp, window: int = 5) -> dict:
    idx = day['timestamp'].searchsorted(ts) - 1
    start = max(0, idx - (window - 1))
    end = idx
    if end <= start or end < 0:
        return {}
    w = day.iloc[start:end+1].copy()
    feats = {}
    try:
        feats['mom_1m'] = w['close'].pct_change(1).iloc[-1] * 100
        feats['mom_3m'] = w['close'].pct_change(3).iloc[-1] * 100
        feats['mom_5m'] = (w['close'].iloc[-1] / w['close'].iloc[0] - 1.0) * 100
        feats['vol_std_5m'] = w['close'].std()
        feats['range_5m'] = w['high'].max() - w['low'].min()
        feats['range_norm_5m'] = feats['range_5m'] / max(1e-9, w['close'].iloc[-1])
        x = np.arange(len(w))
        m = np.polyfit(x, w['close'].values, 1)[0]
        feats['slope_ols_5m'] = m
        s = w['close']
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        feats['rsi_14'] = float(rsi.iloc[-1])
        ema12 = s.ewm(span=12, adjust=False).mean()
        ema26 = s.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        feats['macd'] = float(macd.iloc[-1])
        feats['macd_hist'] = float((macd - macd.ewm(span=9, adjust=False).mean()).iloc[-1])
        vwap = (w['close']*w['volume']).sum()/max(1e-9, w['volume'].sum())
        feats['vwap_dev'] = ((w['close'].iloc[-1]-vwap)/vwap)*100 if vwap>0 else 0.0
        last1 = s.pct_change(1).iloc[-1]
        last2 = s.pct_change(1).iloc[-2]
        last3 = s.pct_change(1).iloc[-3]
        feats['last3_same_sign'] = int(np.sign(last1)==np.sign(last2)==np.sign(last3))
    except Exception:
        return {}
    return feats


def passes_rule_gate(pre: dict) -> bool:
    if not pre:
        return False
    # Simple sanity rules (can be augmented by CLI thresholds)
    if pre.get('vol_std_5m', 0) <= 0:
        return False
    return True


def build_exit_features(day_slice: pd.DataFrame, entry_idx: int) -> pd.DataFrame:
    start_row = day_slice.iloc[entry_idx]
    entry_time = start_row['timestamp']
    entry_price = float(start_row['close']) * (1 + FEE_RATE)
    end_idx = min(len(day_slice) - 1, entry_idx + HOLD_MIN)
    seq = day_slice.iloc[entry_idx:end_idx + 1].copy().reset_index(drop=True)
    seq['elapsed_min'] = (seq['timestamp'] - entry_time).dt.total_seconds().div(60).astype(int)
    exit_value = seq['close'] * (1 - FEE_RATE)
    net_now = (exit_value / entry_price) - 1.0
    seq['since_entry_return_pct'] = net_now * 100.0
    seq['prob_enter_at_entry'] = 0.5
    seq['daily_pred_high_profit'] = 1
    for w in [3,5,10]:
        seq[f'momentum_{w}m'] = seq['close'].pct_change(w) * 100.0
        seq[f'volatility_{w}m'] = seq['close'].rolling(w).std()
    seq['since_entry_cummax'] = seq['since_entry_return_pct'].cummax()
    seq['drawdown_from_max_pct'] = seq['since_entry_return_pct'] - seq['since_entry_cummax']
    best = -1e9
    best_i = 0
    msp = []
    for i, v in enumerate(seq['since_entry_return_pct'].tolist()):
        if v > best:
            best = v
            best_i = i
        msp.append(i - best_i)
    seq['mins_since_peak'] = msp
    return seq[['timestamp','elapsed_min','since_entry_return_pct','prob_enter_at_entry','daily_pred_high_profit',
                'momentum_3m','momentum_5m','momentum_10m','volatility_3m','volatility_5m','volatility_10m',
                'drawdown_from_max_pct','since_entry_cummax','mins_since_peak']].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start-date', type=str, default=None)
    ap.add_argument('--end-date', type=str, default=None)
    ap.add_argument('--k-per-day', type=int, default=K_PER_DAY)
    ap.add_argument('--thr-exit', type=float, default=THR_EXIT)
    ap.add_argument('--min-profit-pct', type=float, default=MIN_PROFIT_PCT)
    ap.add_argument('--enable-peak-stop', action='store_true')
    ap.add_argument('--peak-stop-dd', type=float, default=-0.20)  # % from peak
    ap.add_argument('--peak-trigger-min-profit', type=float, default=0.10)
    ap.add_argument('--max-fail-prob', type=float, default=None)
    ap.add_argument('--min-score', type=float, default=None)
    ap.add_argument('--min-vol-std-5m', type=float, default=None)
    ap.add_argument('--max-vwap-dev', type=float, default=None)  # in %; e.g., 0.0 -> at/below VWAP
    ap.add_argument('--min-rsi', type=float, default=None)
    ap.add_argument('--max-rsi', type=float, default=None)
    ap.add_argument('--allow-hours', type=str, default=None)  # comma-separated hours like "10,15"
    ap.add_argument('--leverage', type=float, default=1.0)
    ap.add_argument('--metrics-out', type=str, default=None)
    ap.add_argument('--allow-holdout', action='store_true', help='Allow evaluation on holdout dates defined in splits file')
    ap.add_argument('--splits-file', type=str, default='data/splits.json', help='JSON with holdout split definitions {"holdout_start":"YYYY-MM-DD","embargo_days":int}')
    args = ap.parse_args()

    # Optional holdout guard
    holdout_start = None
    try:
        from pathlib import Path as _Path
        import json as _json
        sp = _Path(args.splits_file)
        if sp.exists():
            with open(sp, 'r') as _f:
                _spl = _json.load(_f)
            if _spl.get('holdout_start'):
                holdout_start = pd.Timestamp(_spl['holdout_start'])
    except Exception:
        holdout_start = None

    if holdout_start is not None and (args.start_date or args.end_date) and not args.allow_holdout:
        s = pd.Timestamp(args.start_date) if args.start_date else None
        e = pd.Timestamp(args.end_date) if args.end_date else None
        # Block if any overlap with holdout window [holdout_start, +inf)
        if (s and s >= holdout_start) or (e and e >= holdout_start) or (s and e and (s < holdout_start <= e)):
            raise SystemExit(f"Holdout guard: attempting to evaluate on or after {holdout_start.date()} without --allow-holdout")

    ranked = pd.read_csv(RANKED_PATH)
    ranked = ranked[ranked.get('pred_dir', 1) == 1].copy()
    ranked['timestamp'] = pd.to_datetime(ranked['timestamp'])
    ranked['date'] = pd.to_datetime(ranked['date']).dt.date
    if args.start_date:
        ranked = ranked[ranked['timestamp'] >= pd.Timestamp(args.start_date)]
    if args.end_date:
        ranked = ranked[ranked['timestamp'] <= pd.Timestamp(args.end_date)]

    base = pd.read_csv(BASE_PATH)
    base['timestamp'] = pd.to_datetime(base['timestamp'])
    base['date'] = pd.to_datetime(base['date']).dt.date
    if args.start_date:
        base = base[base['timestamp'] >= pd.Timestamp(args.start_date)]
    if args.end_date:
        base = base[base['timestamp'] <= pd.Timestamp(args.end_date)]
    base_by_date = {d: g.sort_values('timestamp').reset_index(drop=True) for d, g in base.groupby('date')}

    # Load models
    exit_model_path = EXIT_MODEL_MAIN if os.path.exists(EXIT_MODEL_MAIN) else EXIT_MODEL_FALLBACK
    exit_clf = joblib.load(exit_model_path)
    pre_payload = joblib.load(PRE_FILTER_PATH)
    pre_clf = pre_payload['model']
    pre_thr = float(pre_payload['threshold'])
    max_fail_prob = pre_thr if args.max_fail_prob is None else float(args.max_fail_prob)
    pre_feats = pre_payload['features']

    allow_hours = None
    if args.allow_hours:
        allow_hours = set(int(h) for h in args.allow_hours.split(',') if h.strip())

    capital = 100.0
    trades = []
    last_exit_ts = None

    for d, g in tqdm(ranked.sort_values('timestamp').groupby('date'), desc='Simulating by day'):
        day = base_by_date.get(d)
        if day is None or day.empty:
            continue
        g = g.sort_values('score', ascending=False)
        chosen_entries = []
        last_exit = None
        ts_arr = day['timestamp'].to_numpy(dtype='datetime64[ns]')

        # Selection
        for _, r in g.iterrows():
            ts = pd.to_datetime(r['timestamp'])
            if allow_hours is not None and ts.hour not in allow_hours:
                continue
            if last_exit is not None and ts < last_exit:
                continue
            pre = build_precontext_window(day, ts, 5)
            if not pre:
                continue
            pre['score'] = float(r.get('score', 0.0))
            pre['confidence'] = float(r.get('confidence', 0.5))
            pre['p_buy'] = float(r.get('p_buy', 0.5))
            if not passes_rule_gate(pre):
                continue
            if args.min_score is not None and pre['score'] < args.min_score:
                continue
            if args.min_vol_std_5m is not None and pre.get('vol_std_5m', 0.0) < args.min_vol_std_5m:
                continue
            if args.max_vwap_dev is not None and pre.get('vwap_dev', 0.0) > args.max_vwap_dev:
                continue
            if args.min_rsi is not None and pre.get('rsi_14', 50.0) < args.min_rsi:
                continue
            if args.max_rsi is not None and pre.get('rsi_14', 50.0) > args.max_rsi:
                continue

            Xpre = np.array([[pre.get(f, 0.0) for f in pre_feats]], dtype=float)
            p_fail = float(pre_clf.predict_proba(Xpre)[:,1])
            if p_fail >= max_fail_prob:
                continue
            chosen_entries.append(ts)
            last_exit = ts + timedelta(minutes=HOLD_MIN)
            if len(chosen_entries) >= args.k_per_day:
                break

        # Execute
        for ts in chosen_entries:
            if last_exit_ts is not None and ts < last_exit_ts:
                continue
            pos = int(np.searchsorted(ts_arr, np.datetime64(ts.to_datetime64())))
            if pos >= len(day):
                continue
            feats = build_exit_features(day, pos)
            X = feats[EXIT_FEATURE_COLS].fillna(0.0).values
            probs = exit_clf.predict_proba(X)[:, 1]
            window = (feats['elapsed_min'] >= 3) & (feats['elapsed_min'] <= HOLD_MIN)
            ok = window & (feats['since_entry_return_pct'] >= args.min_profit_pct)
            idxs = np.where(ok.values)[0]
            if len(idxs) == 0:
                target = int(min(HOLD_MIN, int(feats['elapsed_min'].max())))
                arr = feats['elapsed_min'].to_numpy()
                idx = int(np.searchsorted(arr, target, side='right') - 1)
                idx = max(0, min(idx, len(arr)-1))
            else:
                hit = np.where(probs[idxs] >= args.thr_exit)[0]
                idx = int(idxs[hit[0]]) if len(hit) else int(idxs[0])

            # Peak-protect stop (optional)
            if args.enable_peak_stop:
                # If hit a small profit, then draw down beyond threshold -> exit at first breach
                s = feats.copy()
                cond = (s['since_entry_return_pct'] >= args.peak_trigger_min_profit)
                if cond.any():
                    peak_idx = int(cond.argmax())
                    after = s.iloc[peak_idx:]
                    dd_hit = (after['drawdown_from_max_pct'] <= args.peak_stop_dd).to_numpy()
                    if dd_hit.any():
                        rel = int(np.argmax(dd_hit))
                        idx = min(idx, peak_idx + rel)

            entry_price = float(day.iloc[pos]['close'])
            exit_price = float(day.iloc[pos + idx]['close'])
            entry_cost = entry_price * (1 + FEE_RATE)
            exit_value = exit_price * (1 - FEE_RATE)
            trade_ret = (exit_value / entry_cost) - 1.0
            levered = 1.0 + args.leverage * trade_ret
            liq = False
            if levered <= 0:
                levered = 0.0
                liq = True
            # Update capital using levered return
            capital *= levered
            trades.append(TradeResult(
                date=d,
                entry_time=day.iloc[pos]['timestamp'],
                exit_time=day.iloc[pos + idx]['timestamp'],
                entry_price=entry_price,
                exit_price=exit_price,
                duration_min=int(feats.iloc[idx]['elapsed_min']),
                trade_return_pct=trade_ret * 100.0 * args.leverage,
            ))
            last_exit_ts = day.iloc[pos]['timestamp'] + timedelta(minutes=HOLD_MIN)

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    OUT_TRADES.parent.mkdir(parents=True, exist_ok=True)
    OUT_DAILY.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(OUT_TRADES, index=False)
    if not trades_df.empty:
        daily = trades_df.groupby('date')['trade_return_pct'].sum().reset_index(name='daily_return_pct')
    else:
        daily = pd.DataFrame(columns=['date','daily_return_pct'])
    daily.to_csv(OUT_DAILY, index=False)

    # Metrics
    def max_drawdown(series):
        # series is cumulative capital path in percent returns aggregated daily
        if series.empty:
            return 0.0
        peak = series.cummax()
        dd = (series / peak) - 1.0
        return float(dd.min()) * 100.0

    # Build a daily capital curve from daily %
    if len(daily):
        daily_cap = (1 + daily['daily_return_pct']/100.0).cumprod()
        dd_pct = max_drawdown(daily_cap)
        downs = daily['daily_return_pct'].clip(upper=0)
        sortino = float(daily['daily_return_pct'].mean()) / (float(downs.std()) + 1e-9)
        worst5 = float(daily['daily_return_pct'].quantile(0.05))
    else:
        dd_pct = 0.0
        sortino = 0.0
        worst5 = 0.0

    print(f'Executed trades: {len(trades_df):,}')
    print(f'Final capital (start=100): {capital:.2f}')
    if len(daily):
        print('Avg daily %:', float(daily['daily_return_pct'].mean()))
        print('Median daily %:', float(daily['daily_return_pct'].median()))
        print('Max DD %:', dd_pct)
        print('Sortino:', sortino)
        print('Worst 5% daily %:', worst5)

    if args.metrics_out:
        m = {
            'start_date': args.start_date,
            'end_date': args.end_date,
            'k_per_day': args.k_per_day,
            'thr_exit': args.thr_exit,
            'min_profit_pct': args.min_profit_pct,
            'enable_peak_stop': args.enable_peak_stop,
            'peak_stop_dd': args.peak_stop_dd,
            'peak_trigger_min_profit': args.peak_trigger_min_profit,
            'max_fail_prob': max_fail_prob,
            'min_score': args.min_score,
            'min_vol_std_5m': args.min_vol_std_5m,
            'max_vwap_dev': args.max_vwap_dev,
            'min_rsi': args.min_rsi,
            'max_rsi': args.max_rsi,
            'allow_hours': args.allow_hours,
            'leverage': args.leverage,
            'num_trades': int(len(trades_df)),
            'trade_days': int(len(daily)),
            'mean_trade_pct': float(trades_df['trade_return_pct'].mean()) if len(trades_df) else 0.0,
            'median_trade_pct': float(trades_df['trade_return_pct'].median()) if len(trades_df) else 0.0,
            'daily_mean_pct': float(daily['daily_return_pct'].mean()) if len(daily) else 0.0,
            'daily_median_pct': float(daily['daily_return_pct'].median()) if len(daily) else 0.0,
            'max_dd_pct': dd_pct,
            'sortino': sortino,
            'worst5_daily_pct': worst5,
            'final_capital': float(capital),
        }
        mo = Path(args.metrics_out)
        mo.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([m]).to_csv(mo, index=False)

if __name__ == '__main__':
    main()