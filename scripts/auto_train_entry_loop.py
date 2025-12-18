import json
import os
import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm

TRAIN_END_DATE = '2099-12-31'  # Train on all data
MIN_SUCCESS_PCT = 0.10  # success criteria for trades
N_TREES = 400
THR_GRID = [0.50, 0.55, 0.60, 0.65]
STATE_PATH = 'models/entry_best_meta.json'
MODEL_PATH = 'models/entry_timing_model_autobest.txt'
LOG_DIR = 'data/entry_iter_logs'
WINDOW_MIN = 3
MAX_DRIFT_PCT = 0.30

FEATURES = [
    'momentum_5m','volatility_5m','price_range_5m','volume_avg_5m',
    'momentum_15m','volatility_15m','price_range_15m',
    'price_change_lag_1','price_change_lag_3','price_change_lag_5'
]

os.makedirs('models', exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def load_best_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, 'r') as f:
            return json.load(f)
    return {"best_prec": -1e9, "thr": None}


def save_best_state(state):
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)


def load_feedback_multipliers():
    """Aggregate feedback from all logged rounds into per-row weight multipliers.
    
    Rows (date, timestamp) with consistently successful entries get weight increased.
    Rows with failed entries get weight reduced.
    Also applies trade-level penalties for entries leading to consistently bad trades.
    """
    pattern = os.path.join(LOG_DIR, 'round_*_chosen_entries.csv')
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
        df['is_good'] = (df['trade_success'] >= 1).astype(int)
        frames.append(df[['date', 'timestamp', 'is_good']])

    if not frames:
        return None

    all_logs = pd.concat(frames, ignore_index=True)

    # Aggressive multiplicative scheme
    GOOD_FACTOR = 0.5   # +50% per good entry instance
    BAD_FACTOR = 0.7    # -70% per bad entry instance
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

    # Trade-level penalties (each entry → one trade)
    trade_stats = all_logs.groupby(['date', 'timestamp']).agg(
        n_rounds=('is_good', 'size'),
        n_bad=('is_good', lambda x: (x == 0).sum())
    ).reset_index()
    trade_stats['bad_ratio'] = trade_stats['n_bad'] / trade_stats['n_rounds']
    
    # If entry has >50% bad outcomes, apply trade-level penalty
    TRADE_PENALTY = 0.5
    bad_entries = trade_stats[trade_stats['bad_ratio'] > 0.5][['date', 'timestamp']]
    
    for _, row in bad_entries.iterrows():
        key = (row['date'], row['timestamp'])
        if key in mult:
            mult[key] *= TRADE_PENALTY
            if mult[key] < MIN_MULT:
                mult[key] = MIN_MULT

    keys = list(mult.keys())
    fb_df = pd.DataFrame({
        'date': [k[0] for k in keys],
        'timestamp': [k[1] for k in keys],
        'fb_mult': [mult[k] for k in keys],
    })
    return fb_df


def build_dataset():
    df = pd.read_csv('data/entry_timing_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Load outcomes
    outcomes = pd.read_csv('data/trade_outcomes_all.csv') if os.path.exists('data/trade_outcomes_all.csv') else None
    if outcomes is not None:
        outcomes['entry_time'] = pd.to_datetime(outcomes['entry_time'])
        df = df.merge(
            outcomes[['date','entry_time','success']],
            left_on=['date','timestamp'], right_on=['date','entry_time'], how='left'
        )
        df.drop(columns=['entry_time'], inplace=True)
        df['success'] = df['success'].fillna(0).astype(int)
        # Base weights
        df['w'] = 1.0
        df.loc[df['target_enter'] == 1, 'w'] = 1.5
        df.loc[(df['target_enter'] == 1) & (df['success'] == 1), 'w'] = 5.0
    else:
        df['w'] = 1.0

    # Apply dynamic feedback multipliers
    fb_df = load_feedback_multipliers()
    if fb_df is not None:
        df = df.merge(fb_df, on=['date', 'timestamp'], how='left')
        df['fb_mult'] = df['fb_mult'].fillna(1.0)
        df['w'] = df['w'] * df['fb_mult']

    train_mask = df['date'] <= pd.to_datetime(TRAIN_END_DATE).date()
    return df, train_mask


def fuzzy_metrics(df_test, thr: float, gate_daily: bool = True):
    """Fuzzy precision/recall with ±3min window and ≤0.3% price drift."""
    T = df_test.copy()
    pred_mask = (T['prob'] >= thr)
    if gate_daily:
        pred_mask &= (T['daily_pred_high_profit'] == 1)
    T['pred'] = pred_mask.astype(int)

    fuzzy_tp = fuzzy_fp = fuzzy_fn = 0
    for date, g in T.groupby('date'):
        g = g.sort_values('timestamp')
        true_mask = (g['target_enter'] == 1)
        pred_mask = (g['pred'] == 1)
        true_times = g.loc[true_mask, 'timestamp'].to_numpy()
        true_prices = g.loc[true_mask, 'close'].to_numpy()
        pred_times = g.loc[pred_mask, 'timestamp'].to_numpy()
        pred_prices = g.loc[pred_mask, 'close'].to_numpy()
        if len(pred_times) == 0 and len(true_times) == 0:
            continue
        if len(pred_times) == 0 and len(true_times) > 0:
            fuzzy_fn += len(true_times)
            continue
        used_true = np.zeros(len(true_times), dtype=bool)
        true_secs = true_times.astype('datetime64[s]').astype('int64')
        pred_secs = pred_times.astype('datetime64[s]').astype('int64')
        window = WINDOW_MIN * 60
        for i in range(len(pred_secs)):
            ps = pred_secs[i]
            j = np.searchsorted(true_secs, ps)
            candidates = []
            if j < len(true_secs):
                candidates.append(j)
            if j-1 >= 0:
                candidates.append(j-1)
            matched = False
            for c in candidates:
                if used_true[c]:
                    continue
                dt = abs(int(ps - true_secs[c]))
                if dt <= window:
                    drift = abs(pred_prices[i] - true_prices[c]) / true_prices[c] * 100.0
                    if drift <= MAX_DRIFT_PCT:
                        used_true[c] = True
                        fuzzy_tp += 1
                        matched = True
                        break
            if not matched:
                fuzzy_fp += 1
        fuzzy_fn += (~used_true).sum()
    prec = fuzzy_tp / (fuzzy_tp + fuzzy_fp) if (fuzzy_tp + fuzzy_fp) > 0 else 0.0
    rec  = fuzzy_tp / (fuzzy_tp + fuzzy_fn) if (fuzzy_tp + fuzzy_fn) > 0 else 0.0
    return prec, rec, fuzzy_tp, fuzzy_fp, fuzzy_fn


def simulate_and_log(df_test, booster, thr, round_id):
    """Generate entry signals, evaluate, and log chosen entries with outcomes."""
    proba = booster.predict(df_test[FEATURES].values)
    df_test = df_test.copy()
    df_test['prob'] = proba
    
    # Apply threshold with daily gate
    pred_mask = (df_test['prob'] >= thr) & (df_test['daily_pred_high_profit'] == 1)
    df_test['pred'] = pred_mask.astype(int)
    
    chosen = df_test[df_test['pred'] == 1].copy()
    if chosen.empty:
        return 0.0, 0.0, chosen
    
    # Get trade outcomes for chosen entries
    outcomes = pd.read_csv('data/trade_outcomes_all.csv') if os.path.exists('data/trade_outcomes_all.csv') else None
    if outcomes is not None:
        outcomes['entry_time'] = pd.to_datetime(outcomes['entry_time'])
        cols = [c for c in ['date','entry_time','success','return_pct'] if c in outcomes.columns]
        chosen = chosen.merge(
            outcomes[cols],
            left_on=['date','timestamp'], right_on=['date','entry_time'], how='left'
        )
        if 'success' in chosen.columns:
            chosen['trade_success'] = chosen['success'].fillna(0)
        else:
            chosen['trade_success'] = 0
        if 'return_pct' in chosen.columns:
            chosen['net_return'] = chosen['return_pct'].fillna(0)
        else:
            chosen['net_return'] = 0
    else:
        chosen['trade_success'] = 0
        chosen['net_return'] = 0
    
    # Fuzzy metrics
    prec, rec, tp, fp, fn = fuzzy_metrics(df_test, thr, gate_daily=True)
    
    # Log
    out = chosen[['date','timestamp','close','trade_success','net_return']].copy()
    out.to_csv(os.path.join(LOG_DIR, f'round_{round_id}_chosen_entries.csv'), index=False)
    
    return prec, rec, chosen


def main(rounds=-1):
    state = load_best_state()

    r = 0
    try:
        while True:
            r += 1
            if rounds > 0 and r > rounds:
                break

            # Rebuild dataset each round
            df, train_mask = build_dataset()
            X_train = df.loc[train_mask, FEATURES].values
            y_train = df.loc[train_mask, 'target_enter'].values
            w_train = df.loc[train_mask, 'w'].values
            # Evaluate on same data (in-sample) for feedback loop
            df_test = df.loc[train_mask].copy()

            print(f"\n[Round {r}] Training {N_TREES}-tree entry LightGBM...")
            train_set = lgb.Dataset(X_train, label=y_train, weight=w_train, feature_name=FEATURES)
            params = dict(objective='binary', learning_rate=0.05, num_leaves=63, n_estimators=N_TREES,
                          feature_pre_filter=False, verbosity=-1)
            
            with tqdm(total=N_TREES, desc=f"Round {r} Entry Training", unit="tree") as pbar:
                def callback(env):
                    pbar.update(1)
                booster = lgb.train(params, train_set, num_boost_round=N_TREES, callbacks=[callback])

            print("Sweeping entry thresholds...")
            best_thr, best_prec, best_rec, best_df = None, -1e9, 0, None
            for thr in tqdm(THR_GRID, desc=f"Round {r} Entry Threshold Sweep", unit="thr"):
                prec, rec, chosen = simulate_and_log(df_test, booster, thr, r)
                tqdm.write(f"thr={thr:.2f} prec={prec:.3f} rec={rec:.3f} entries={len(chosen)}")
                if prec > best_prec:
                    best_thr, best_prec, best_rec, best_df = thr, prec, rec, chosen

            improved = best_prec > state.get('best_prec', -1e9) + 1e-6
            
            # Calculate KPIs
            n_days = len(best_df['date'].unique()) if not best_df.empty else 0
            n_entries = len(best_df)
            success_rate = best_df['trade_success'].mean() if not best_df.empty else 0
            avg_return = best_df['net_return'].mean() if not best_df.empty else 0
            
            print(f"\n=== Round {r} Entry Results ===")
            print(f"This round: thr={best_thr:.2f} | fuzzy_prec={best_prec:.3f} | fuzzy_rec={best_rec:.3f}")
            print(f"Entries: {n_entries} | Days: {n_days} | Success rate: {success_rate:.3f}")
            print(f"Avg return per entry: {avg_return:.3f}%")
            print(f"Improved: {improved}")
            
            if state.get('best_prec') and state['best_prec'] > -1e9:
                print(f"\n=== CURRENT BEST ENTRY MODEL ===")
                print(f"Best precision: {state['best_prec']:.3f}")
                print(f"Best threshold: {state.get('thr', 'N/A')}")

            if improved:
                booster.save_model(MODEL_PATH)
                state['best_prec'] = best_prec
                state['thr'] = best_thr
                save_best_state(state)
                print(f"\n✓ NEW BEST ENTRY MODEL SAVED: {MODEL_PATH} (thr={best_thr:.2f})\n")
            else:
                print(f"\n- No improvement; best entry model unchanged.\n")
    except KeyboardInterrupt:
        print("\nStopped by user. Current best:", state)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=-1)
    args = p.parse_args()
    main(rounds=args.rounds)
