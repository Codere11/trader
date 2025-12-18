#!/usr/bin/env python3
import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm

SAMPLES_PATH = 'data/exit_training/exit_samples.csv'

print('[1/4] Loading exit samples...')
df = pd.read_csv(SAMPLES_PATH)
print(f'  Samples: {len(df):,}, entries: {df["entry_id"].nunique():,}')

print('[2/4] Preparing train/validation split (by entry_date)...')
df['entry_date'] = pd.to_datetime(df['entry_date']).dt.date

dates = sorted(df['entry_date'].unique())
split_idx = int(len(dates) * 0.9)
train_dates = set(dates[:split_idx])

train_df = df[df['entry_date'].isin(train_dates)].copy()
valid_df = df[~df['entry_date'].isin(train_dates)].copy()

print(f'  Train entries: {train_df["entry_id"].nunique():,}, samples: {len(train_df):,}')
print(f'  Valid entries: {valid_df["entry_id"].nunique():,}, samples: {len(valid_df):,}')

feat_cols = [c for c in df.columns if c.startswith('feat_')]
print(f'  Using features: {feat_cols}')

X_tr = train_df[feat_cols].values
y_tr = train_df['exit_now'].astype(int).values
X_va = valid_df[feat_cols].values
y_va = valid_df['exit_now'].astype(int).values

print('[3/4] Training LightGBM exit classifier...')

train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols)
valid_set = lgb.Dataset(X_va, label=y_va, feature_name=feat_cols)

params = dict(
    objective='binary',
    metric=['binary_logloss','auc'],
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=100,
    feature_pre_filter=False,
    random_state=42,
    verbosity=-1,
)

clf = lgb.train(
    params,
    train_set,
    num_boost_round=800,
    valid_sets=[valid_set],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=100)],
)

model_path = 'models/exit_classifier_lgb.txt'
clf.save_model(model_path)
print(f'  Saved exit classifier to {model_path}')

print('[4/4] Evaluating exit policy on validation set (per-entry simulation)...')

# Attach predictions to validation rows
probs = clf.predict(X_va)
valid_df = valid_df.copy()
valid_df['p_exit_now'] = probs

# Simple policy: for each entry, step t=0.., exit at first t where p>=thr; if none, exit at last t
THR = 0.6

results = []
for entry_id, g in tqdm(valid_df.groupby('entry_id'), desc='Simulating exits'):
    g = g.sort_values('t_offset')
    p = g['p_exit_now'].values
    idx = np.where(p >= THR)[0]
    if len(idx) == 0:
        chosen = g.iloc[-1]
    else:
        chosen = g.iloc[idx[0]]
    best = g['best8_net_pct'].iloc[0]
    pnl = chosen['pnl_if_exit_pct']
    results.append({
        'entry_id': entry_id,
        'chosen_t': chosen['t_offset'],
        'chosen_pnl': pnl,
        'best_pnl': best,
    })

res_df = pd.DataFrame(results)

# Metrics
mask_pos = res_df['best_pnl'] > 0
capture_ratio = (res_df.loc[mask_pos, 'chosen_pnl'] / res_df.loc[mask_pos, 'best_pnl']).clip(upper=1.0)

print('\n=== EXIT POLICY EVALUATION (validation) ===')
print(f'Entries evaluated: {len(res_df):,}')
print(f'Positive-potential entries: {mask_pos.sum():,}')
print(f'Average chosen PnL: {res_df["chosen_pnl"].mean():.3f}%')
print(f'Median chosen PnL: {res_df["chosen_pnl"].median():.3f}%')
print(f'Average best-8min PnL: {res_df["best_pnl"].mean():.3f}%')
print(f'Capture ratio (mean over positive-potential): {capture_ratio.mean():.3f}')
print(f'Capture ratio (median over positive-potential): {capture_ratio.median():.3f}')

res_df.to_csv('data/exit_training/exit_policy_eval_val.csv', index=False)
print('Saved per-entry evaluation to data/exit_training/exit_policy_eval_val.csv')
