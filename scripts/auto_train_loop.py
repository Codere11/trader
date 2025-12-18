import json
import os
import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm

TRAIN_END_DATE = '2099-12-31'  # Train on all data
MIN_PROFIT_PCT = 0.10   # guard and label
N_TREES = 400           # "400-neuron brain"
THR_GRID = [0.40, 0.45, 0.50, 0.55]
STATE_PATH = 'models/best_meta.json'
MODEL_PATH = 'models/exit_timing_model_autobest.joblib'
LOG_DIR = 'data/iter_logs'

FEATURES = [
    'elapsed_min','since_entry_return_pct','prob_enter_at_entry',
    'momentum_3m','momentum_5m','momentum_10m','volatility_3m','volatility_5m','volatility_10m',
    'drawdown_from_max_pct','since_entry_cummax','mins_since_peak'
]

os.makedirs('models', exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def load_best_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, 'r') as f:
            return json.load(f)
    return {"best_total": -1e9, "thr": None}


def save_best_state(state):
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)


def build_dataset():
    df = pd.read_csv('data/exit_timing_features.csv', parse_dates=['timestamp'])
    df['date'] = pd.to_datetime(df['date']).dt.date
    # outcome-weighted labels in 3-8m window
    W = df[(df['elapsed_min'] >= 3) & (df['elapsed_min'] <= 8)].copy()
    W['y'] = (W['since_entry_return_pct'] >= MIN_PROFIT_PCT).astype(int)
    # weight successful trades more (static outcomes)
    outcomes = pd.read_csv('data/trade_outcomes_all.csv') if os.path.exists('data/trade_outcomes_all.csv') else None
    if outcomes is not None:
        W['is_start'] = (W['elapsed_min'] == 0).astype(int)
        W['trade_id'] = W.groupby('date')['is_start'].cumsum()
        W = W.merge(outcomes[['date','trade_id','success']], on=['date','trade_id'], how='left')
        W['success'] = W['success'].fillna(0).astype(int)
        W['w'] = 1 + 2*W['success']
    else:
        W['w'] = 1.0

    # Apply dynamic feedback-based multipliers from previous rounds
    fb_df = load_feedback_multipliers()
    if fb_df is not None:
        # fb_df['date'] already as date, timestamp as datetime64[ns]
        W = W.merge(fb_df, on=['date', 'timestamp'], how='left')
        W['fb_mult'] = W['fb_mult'].fillna(1.0)
        W['w'] = W['w'] * W['fb_mult']

    train_mask = W['date'] <= pd.to_datetime(TRAIN_END_DATE).date()
    return W, train_mask


def simulate(test_df, booster, thr):
    T = test_df.copy()
    proba = booster.predict(T[FEATURES].values)
    T['prob_exit'] = proba
    T['is_start'] = (T['elapsed_min'] == 0).astype(int)
    T['trade_id'] = T.groupby('date')['is_start'].cumsum()
    chosen = []
    for (date, tid), g in T.groupby(['date','trade_id']):
        g = g.sort_values('elapsed_min')
        window = g[(g['elapsed_min'] >= 3) & (g['elapsed_min'] <= 8)]
        window = window[window['since_entry_return_pct'] >= MIN_PROFIT_PCT]
        pick = window[window['prob_exit'] >= thr].head(1)
        if pick.empty:
            fallback = g[g['elapsed_min'] == min(8, g['elapsed_min'].max())]
            pick = fallback.head(1)
        if not pick.empty:
            chosen.append(pick.iloc[0])
    chosen_df = pd.DataFrame(chosen)
    if chosen_df.empty:
        return 0.0, pd.DataFrame()
    total = float(chosen_df.groupby('date')['since_entry_return_pct'].sum().sum())
    return total, chosen_df


def log_reasons(round_id, chosen_df, booster):
    # SHAP-like per-feature contribs using pred_contrib
    X = chosen_df[FEATURES].values
    contrib = booster.predict(X, pred_contrib=True)  # last column is bias term
    cols = FEATURES + ['bias']
    contrib_df = pd.DataFrame(contrib, columns=cols)
    top3 = []
    for i in range(len(contrib_df)):
        s = contrib_df.iloc[i][FEATURES].abs().sort_values(ascending=False).head(3)
        top3.append(';'.join([f"{k}:{s[k]:.4f}" for k in s.index]))
    out = chosen_df[['date','timestamp','elapsed_min','since_entry_return_pct']].copy()
    out['top3_reasons'] = top3
    out.to_csv(os.path.join(LOG_DIR, f'round_{round_id}_chosen_trades.csv'), index=False)


def load_feedback_multipliers():
    """Aggregate feedback from all logged rounds into per-row weight multipliers.

    Rows (date, timestamp) with consistently good exits get their weight increased.
    Rows with bad exits get their weight reduced (but stay > 0).
    Also applies trade-level penalties for trades with multiple bad exits.
    """
    pattern = os.path.join(LOG_DIR, 'round_*_chosen_trades.csv')
    files = glob.glob(pattern)
    if not files:
        return None

    frames = []
    for path in files:
        try:
            df = pd.read_csv(path, parse_dates=['timestamp'])
        except Exception:
            continue
        if df.empty:
            continue
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['is_good'] = (df['since_entry_return_pct'] >= MIN_PROFIT_PCT).astype(int)
        frames.append(df[['date', 'timestamp', 'elapsed_min', 'is_good']])

    if not frames:
        return None

    all_logs = pd.concat(frames, ignore_index=True)

    # Aggressive multiplicative scheme
    GOOD_FACTOR = 0.5   # +50% per good exit instance
    BAD_FACTOR = 0.7    # -70% per bad exit instance
    MIN_MULT = 0.05

    # Per-timestamp multipliers
    mult = {}
    for _, row in all_logs.iterrows():
        key = (row['date'], row['timestamp'])
        m = mult.get(key, 1.0)
        if row['is_good']:
            m *= (1.0 + GOOD_FACTOR)
        else:
            m *= (1.0 - BAD_FACTOR)
        if m < MIN_MULT:
            m = MIN_MULT
        mult[key] = m

    # Trade-level penalties: identify trades with repeatedly bad exits
    all_logs['is_start'] = (all_logs['elapsed_min'] == 0).astype(int)
    all_logs = all_logs.sort_values(['date', 'timestamp']).reset_index(drop=True)
    all_logs['trade_id'] = all_logs.groupby('date')['is_start'].cumsum()
    
    trade_stats = all_logs.groupby(['date', 'trade_id']).agg(
        n_exits=('is_good', 'size'),
        n_bad=('is_good', lambda x: (x == 0).sum())
    ).reset_index()
    trade_stats['bad_ratio'] = trade_stats['n_bad'] / trade_stats['n_exits']
    
    # If trade has >50% bad exits across rounds, apply trade-level penalty
    TRADE_PENALTY = 0.5  # additional ×0.5 multiplier
    bad_trades = trade_stats[trade_stats['bad_ratio'] > 0.5][['date', 'trade_id']]
    
    trade_mult = {}
    for _, row in bad_trades.iterrows():
        trade_mult[(row['date'], row['trade_id'])] = TRADE_PENALTY
    
    # Merge trade penalties back to timestamps
    all_logs['trade_mult'] = all_logs.apply(
        lambda r: trade_mult.get((r['date'], r['trade_id']), 1.0), axis=1
    )
    
    for _, row in all_logs.iterrows():
        key = (row['date'], row['timestamp'])
        if key in mult:
            mult[key] *= row['trade_mult']
            if mult[key] < MIN_MULT:
                mult[key] = MIN_MULT

    keys = list(mult.keys())
    fb_df = pd.DataFrame({
        'date': [k[0] for k in keys],
        'timestamp': [k[1] for k in keys],
        'fb_mult': [mult[k] for k in keys],
    })
    return fb_df


def main(rounds=-1):  # rounds < 0 => loop until Ctrl+C
    state = load_best_state()

    r = 0
    try:
        while True:
            r += 1
            if rounds > 0 and r > rounds:
                break

            # Rebuild dataset each round so feedback from previous rounds is used
            W, train_mask = build_dataset()
            X_train = W.loc[train_mask, FEATURES].values
            y_train = W.loc[train_mask, 'y'].values
            w_train = W.loc[train_mask, 'w'].values
            # Evaluate on same data (in-sample) for feedback loop
            test_df = W.loc[train_mask].copy()

            print(f"\n[Round {r}] Training 400-tree LightGBM...")
            train_set = lgb.Dataset(X_train, label=y_train, weight=w_train, feature_name=FEATURES)
            params = dict(objective='binary', learning_rate=0.05, num_leaves=63, n_estimators=N_TREES,
                          feature_pre_filter=False, verbosity=-1)
            
            # Train with progress bar
            with tqdm(total=N_TREES, desc=f"Round {r} Training", unit="tree") as pbar:
                def callback(env):
                    pbar.update(1)
                booster = lgb.train(params, train_set, num_boost_round=N_TREES, callbacks=[callback])

            print("Sweeping exit thresholds...")
            best_thr, best_total, best_df = None, -1e9, None
            for thr in tqdm(THR_GRID, desc=f"Round {r} Threshold Sweep", unit="thr"):
                total, chosen = simulate(test_df, booster, thr)
                tqdm.write(f"thr={thr:.2f} total={total:.2f}% trades={len(chosen)}")
                if total > best_total:
                    best_thr, best_total, best_df = thr, total, chosen

            improved = best_total > state.get('best_total', -1e9) + 1e-6
            
            # Calculate KPIs for this round
            n_days = len(best_df['date'].unique())
            n_trades = len(best_df)
            avg_return_per_trade = best_df['since_entry_return_pct'].mean()
            median_return_per_trade = best_df['since_entry_return_pct'].median()
            avg_daily_pnl = best_total / n_days if n_days > 0 else 0
            
            print(f"\n=== Round {r} Results ===")
            print(f"This round: thr={best_thr:.2f} | total_return={best_total:.2f}% | trades={n_trades} | days={n_days}")
            print(f"Avg per trade: {avg_return_per_trade:.3f}% | Median: {median_return_per_trade:.3f}%")
            print(f"Avg daily PnL: {avg_daily_pnl:.3f}%")
            print(f"Improved: {improved}")
            
            if state.get('best_total') and state['best_total'] > -1e9:
                print(f"\n=== CURRENT BEST MODEL ===")
                print(f"Best total return: {state['best_total']:.2f}%")
                print(f"Best threshold: {state.get('thr', 'N/A')}")
            
            # Log reasons for chosen trades
            log_reasons(r, best_df, booster)

            if improved:
                booster.save_model(MODEL_PATH.replace('.joblib', '.txt'))
                state['best_total'] = best_total
                state['thr'] = best_thr
                save_best_state(state)
                print(f"\n✓ NEW BEST MODEL SAVED: {MODEL_PATH} (thr={best_thr:.2f})\n")
            else:
                print(f"\n- No improvement; best model unchanged.\n")
    except KeyboardInterrupt:
        print("\nStopped by user. Current best:", state)

if __name__ == '__main__':
    # Allow: python3 auto_train_loop.py --rounds 5 or --rounds -1 (infinite)
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=-1)
    args = p.parse_args()
    main(rounds=args.rounds)
