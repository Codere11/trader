#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import joblib

LOG_DIR = 'data/entry_directional_logs'
MODEL_PATH = 'models/entry_mistake_filter.joblib'
RANDOM_STATE = 42

# Given contexts_agg.csv with one row per entry, train a classifier to predict entry_correct (1) vs mistake (0).

FEATURE_COLS = [
    'ctx_close_mean','ctx_close_std','ctx_close_min','ctx_close_max','ctx_close_range','ctx_close_slope',
    'ctx_vol_mean','ctx_vol_std',
    'entry_momentum_5m','entry_volatility_5m','entry_price_range_5m','entry_volume_avg_5m',
    'entry_momentum_15m','entry_volatility_15m','entry_price_range_15m',
    'entry_price_change_lag_1','entry_price_change_lag_3','entry_price_change_lag_5',
    'pred_dir','confidence'
]


def load_contexts_raw(path=os.path.join(LOG_DIR, 'contexts.csv')) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['entry_timestamp'] = pd.to_datetime(df['entry_timestamp'])
    return df


def aggregate_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized per-entry aggregation as a fallback when only raw contexts.csv exists."""
    # Aggregate close/volume with groupby
    agg_close = df.groupby('entry_timestamp')['close'].agg(
        ctx_close_mean='mean',
        ctx_close_std='std',
        ctx_close_min='min',
        ctx_close_max='max'
    ).reset_index()
    agg_close['ctx_close_range'] = agg_close['ctx_close_max'] - agg_close['ctx_close_min']

    if 'volume' in df.columns:
        agg_vol = df.groupby('entry_timestamp')['volume'].agg(
            ctx_vol_mean='mean',
            ctx_vol_std='std'
        ).reset_index()
        agg = pd.merge(agg_close, agg_vol, on='entry_timestamp', how='left')
    else:
        agg = agg_close

    # Slope proxy: (last close - first close) / window_len using rel_min ordering
    df_sorted = df.sort_values(['entry_timestamp','rel_min'])
    first = df_sorted.groupby('entry_timestamp')['close'].first()
    last = df_sorted.groupby('entry_timestamp')['close'].last()
    counts = df_sorted.groupby('entry_timestamp')['close'].size().clip(lower=1)
    slope = (last - first) / counts
    slope = slope.reset_index().rename(columns={'close':'ctx_close_slope'})
    agg = agg.merge(slope, on='entry_timestamp', how='left')

    # Metadata at entry (rel_min == 0)
    at_entry = df[df['rel_min'] == 0].copy()
    meta_cols = ['entry_timestamp','entry_date','entry_pred_dir','entry_confidence','entry_correct']
    if 'entry_date' not in at_entry.columns:
        at_entry['entry_date'] = pd.to_datetime(at_entry['timestamp']).dt.date
    meta = at_entry[meta_cols].rename(columns={
        'entry_pred_dir':'pred_dir',
        'entry_confidence':'confidence',
        'entry_correct':'correct'
    })
    # Attach engineered features at entry
    for col in ['momentum_5m','volatility_5m','price_range_5m','volume_avg_5m',
                'momentum_15m','volatility_15m','price_range_15m',
                'price_change_lag_1','price_change_lag_3','price_change_lag_5']:
        if col in at_entry.columns:
            meta[f'entry_{col}'] = at_entry[col].astype(float)

    agg = agg.merge(meta, on='entry_timestamp', how='left')
    return agg


def train_filter(df_agg: pd.DataFrame):
    # Normalize/derive dates and chronological split
    if 'entry_date' in df_agg.columns:
        df_agg['entry_date'] = pd.to_datetime(df_agg['entry_date'])
    elif 'date' in df_agg.columns:
        df_agg['entry_date'] = pd.to_datetime(df_agg['date'])
    elif 'timestamp' in df_agg.columns:
        df_agg['entry_date'] = pd.to_datetime(df_agg['timestamp']).dt.date
        df_agg['entry_date'] = pd.to_datetime(df_agg['entry_date'])
    else:
        raise KeyError('No entry_date/date/timestamp column found in aggregated data')
    df_agg['date'] = df_agg['entry_date'].dt.date

    # Feature selection: keep only known numeric feature cols if present
    feat_cols = [c for c in FEATURE_COLS if c in df_agg.columns]
    # Fill/clean
    df_agg[feat_cols] = df_agg[feat_cols].astype(float)
    df_agg[feat_cols] = df_agg[feat_cols].fillna(0.0)
    # Target as integer 0/1
    df_agg['correct'] = df_agg['correct'].astype(int)

    # Train/valid/test by date: train<=2024-12-31, valid=last 10% of train dates for early stopping, test>cutoff
    cutoff = pd.to_datetime('2024-12-31').date()
    train_mask = df_agg['date'] <= cutoff
    # If nothing falls before/equal to cutoff, train on all data
    if not train_mask.any():
        train_mask = pd.Series(True, index=df_agg.index)

    train_df = df_agg.loc[train_mask]
    test_df  = df_agg.loc[~train_mask]

    # Chrono valid split inside train
    unique_dates = sorted(train_df['date'].unique())
    if len(unique_dates) > 1:
        k = max(1, int(0.1 * len(unique_dates)))
        valid_dates = set(unique_dates[-k:])
        valid_mask = train_df['date'].isin(valid_dates)
    else:
        valid_mask = pd.Series(False, index=train_df.index)

    X_tr = train_df.loc[~valid_mask, feat_cols]
    y_tr = train_df.loc[~valid_mask, 'correct'].astype(int)
    X_va = train_df.loc[valid_mask, feat_cols]
    y_va = train_df.loc[valid_mask, 'correct'].astype(int)

    # Compute scale_pos_weight to handle imbalance safely
    pos = int((y_tr == 1).sum())
    neg = int((y_tr == 0).sum())
    spw = float(neg / pos) if pos > 0 else 1.0

    clf = lgb.LGBMClassifier(
        objective='binary',
        random_state=RANDOM_STATE,
        n_estimators=2000,
        learning_rate=0.05,
        scale_pos_weight=spw,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
    )

    if len(X_va) > 0:
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc', callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        clf.fit(X_tr, y_tr)

    joblib.dump(clf, MODEL_PATH)

    if len(test_df) > 0:
        X_te = test_df[feat_cols]
        y_te = test_df['correct']
        p = clf.predict_proba(X_te)[:,1]
        auc = roc_auc_score(y_te, p)
        yhat = (p >= 0.5).astype(int)
        acc = accuracy_score(y_te, yhat)
        print(f'Test AUC={auc:.3f} ACC={acc:.3f} (cutoff 0.5) on {len(X_te)} entries')
    else:
        print('No test split available; model trained on all train data.')


def main():
    agg_path = os.path.join(LOG_DIR, 'contexts_agg.csv')
    raw_path = os.path.join(LOG_DIR, 'contexts.csv')

    if os.path.exists(agg_path):
        print(f'Loading aggregated contexts from {agg_path}')
        agg = pd.read_csv(agg_path)
        # Ensure expected columns exist
        if 'entry_timestamp' in agg.columns:
            try:
                agg['entry_timestamp'] = pd.to_datetime(agg['entry_timestamp'])
            except Exception:
                pass
    elif os.path.exists(raw_path):
        print(f'Aggregated file not found; loading raw contexts from {raw_path} and aggregating (vectorized)...')
        ctx = load_contexts_raw(raw_path)
        agg = aggregate_features_vectorized(ctx)
        agg.to_csv(agg_path, index=False)
        print(f'Saved aggregated context features to {agg_path} ({len(agg)} entries)')
    else:
        print(f'Missing contexts at {agg_path} and {raw_path}. Run train_entry_directional_min50.py first.')
        return

    train_filter(agg)

if __name__ == '__main__':
    main()
