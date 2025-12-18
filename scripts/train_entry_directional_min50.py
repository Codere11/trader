#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
import joblib

# Config
FEE_RATE = 0.001           # 0.1% taker per side (approx Binance spot)
ROUND_TRIP = 2 * FEE_RATE  # buy+sell or sell+buy
LOOKAHEAD_MIN = 8          # minutes to judge correctness
CONTEXT_MIN = 10           # minutes of pre-entry context to log
START_HOUR = 10            # train on/after 10:00 for consistency with other scripts
TRAIN_END_DATE = '2024-12-31'
MODEL_PATH = 'models/entry_directional_min50.joblib'
LOG_DIR = 'data/entry_directional_logs'
N_ESTIMATORS = 400
RANDOM_STATE = 42

FEATURES = [
    # basic price/volume context — echo of existing minute feature set
    'momentum_5m','volatility_5m','price_range_5m','volume_avg_5m',
    'momentum_15m','volatility_15m','price_range_15m',
    'price_change_lag_1','price_change_lag_3','price_change_lag_5'
]

os.makedirs('models', exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def create_minute_features(df: pd.DataFrame, window_sizes=[5, 15, 30]) -> pd.DataFrame:
    df = df.copy()
    # Per-day rolling features (to avoid lookahead leakage across days)
    for window in window_sizes:
        df[f'volatility_{window}m'] = df.groupby('date')['close'].transform(lambda s: s.rolling(window).std())
        df[f'momentum_{window}m'] = df.groupby('date')['close'].transform(lambda s: s.pct_change(window)) * 100
        df[f'volume_avg_{window}m'] = df.groupby('date')['volume'].transform(lambda s: s.rolling(window).mean())
        high_max = df.groupby('date')['high'].transform(lambda s: s.rolling(window).max())
        low_min  = df.groupby('date')['low'].transform(lambda s: s.rolling(window).min())
        df[f'price_range_{window}m'] = high_max - low_min
    for lag in [1, 3, 5]:
        df[f'price_change_lag_{lag}'] = df.groupby('date')['price_change_pct'].shift(lag)
    return df


def add_future_window_extrema(df: pd.DataFrame, lookahead: int = LOOKAHEAD_MIN) -> pd.DataFrame:
    """Compute future max(high) and min(low) within next `lookahead` minutes per day."""
    df = df.copy()
    df = df.sort_values(['date', 'timestamp']).reset_index(drop=True)
    df['fut_max_high'] = np.nan
    df['fut_min_low'] = np.nan

    for d, g in tqdm(df.groupby('date'), desc='Future window extrema by day'):
        highs = g['high'].to_numpy()
        lows  = g['low'].to_numpy()
        n = len(g)
        max_high = np.full(n, np.nan)
        min_low  = np.full(n, np.nan)
        # For each index i, compute extrema over (i+1 ... i+lookahead)
        for i in range(n):
            j2 = min(n, i + lookahead + 1)
            if i + 1 >= j2:
                continue
            max_high[i] = np.max(highs[i+1:j2])
            min_low[i]  = np.min(lows[i+1:j2])
        df.loc[g.index, 'fut_max_high'] = max_high
        df.loc[g.index, 'fut_min_low'] = min_low
    return df


def make_direction_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create BUY/SELL labels and correctness targets under 8-min rule with round-trip fee.
    - BUY is correct if fut_max_high >= close * (1 + ROUND_TRIP)
    - SELL is correct if fut_min_low  <= close * (1 - ROUND_TRIP)
    Labels for training (y): BUY=1, SELL=0. Ties resolved by larger potential move.
    """
    df = df.copy()
    close = df['close'].astype(float)
    fut_max = df['fut_max_high'].astype(float)
    fut_min = df['fut_min_low'].astype(float)

    buy_trigger  = close * (1.0 + ROUND_TRIP)
    sell_trigger = close * (1.0 - ROUND_TRIP)

    buy_correct  = (fut_max >= buy_trigger)
    sell_correct = (fut_min <= sell_trigger)

    # Potential moves as percentages (best-case in each direction over the window)
    up_move   = (fut_max / close) - 1.0
    down_move = 1.0 - (fut_min / close)

    # Training label: choose direction with larger potential move
    # Fallbacks: if only one is finite, choose that; if neither finite, default to BUY.
    y = np.where((up_move >= down_move), 1, 0).astype(int)

    df['y'] = y                  # model target: BUY=1, SELL=0
    df['buy_correct'] = buy_correct.astype(int)
    df['sell_correct'] = sell_correct.astype(int)
    df['up_move_pct'] = up_move * 100.0
    df['down_move_pct'] = down_move * 100.0
    return df


def load_data() -> pd.DataFrame:
    df = pd.read_csv('data/btc_profitability_analysis_filtered.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['hour'] = df['timestamp'].dt.hour
    # Keep 10:00+ minutes to align with other components
    df = df[df['hour'] >= START_HOUR].copy()
    return df


def add_context_rollups(df: pd.DataFrame, window: int = CONTEXT_MIN) -> pd.DataFrame:
    """Precompute per-entry 10-minute aggregate context using rolling stats per day."""
    df = df.copy()
    # Rolling over past `window` minutes inclusive of current row
    g = df.groupby('date')
    close = g['close']
    vol = g['volume'] if 'volume' in df.columns else None

    df['ctx_close_mean'] = close.transform(lambda s: s.rolling(window, min_periods=1).mean())
    df['ctx_close_std'] = close.transform(lambda s: s.rolling(window, min_periods=1).std())
    df['ctx_close_min'] = close.transform(lambda s: s.rolling(window, min_periods=1).min())
    df['ctx_close_max'] = close.transform(lambda s: s.rolling(window, min_periods=1).max())
    df['ctx_close_range'] = df['ctx_close_max'] - df['ctx_close_min']
    # Simple slope proxy: last-first over window length (percentage per minute of price)
    first_close = close.transform(lambda s: s.rolling(window, min_periods=1).apply(lambda x: x[0], raw=True))
    last_close = close.transform(lambda s: s.rolling(window, min_periods=1).apply(lambda x: x[-1], raw=True))
    df['ctx_close_slope'] = (last_close - first_close) / np.maximum(1, window)

    if vol is not None:
        df['ctx_vol_mean'] = vol.transform(lambda s: s.rolling(window, min_periods=1).mean())
        df['ctx_vol_std'] = vol.transform(lambda s: s.rolling(window, min_periods=1).std())

    return df


def build_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    df = load_data()
    df = create_minute_features(df)
    df = add_future_window_extrema(df, LOOKAHEAD_MIN)
    df = make_direction_labels(df)
    df = add_context_rollups(df, CONTEXT_MIN)

    # Regime/normalized features
    # Hour-of-day sin/cos and day-of-week
    ts = pd.to_datetime(df['timestamp'])
    df['hour'] = ts.dt.hour
    df['dow'] = ts.dt.dayofweek
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24.0)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24.0)
    # Normalized range/std
    df['ctx_range_norm'] = df['ctx_close_range'] / df['close']
    df['ctx_std_norm'] = df['ctx_close_std'] / df['ctx_close_mean'].replace(0, np.nan)

    # Drop rows missing engineered features or future window
    need = FEATURES + ['fut_max_high','fut_min_low','y']
    df = df.dropna(subset=need).copy()

    X = df[FEATURES]
    y = df['y']
    dates = pd.to_datetime(df['date'])
    return X, y, dates, df


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMClassifier:
    clf = lgb.LGBMClassifier(
        objective='binary',
        random_state=RANDOM_STATE,
        n_estimators=N_ESTIMATORS,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def select_entries_min50_per_day(
    df_day: pd.DataFrame,
    proba_buy: np.ndarray,
    k_min: int = 50,
    k_max: int = 170,
    exact_k: int | None = None,
) -> pd.DataFrame:
    """Select per-day entries.
    - If exact_k is provided, pick exactly that many by highest confidence (or all if fewer rows).
    - Otherwise enforce min/max constraints (legacy behavior).
    Direction = argmax{P(BUY), 1-P(BUY)}; confidence = max of the two.
    """
    d = df_day.copy()
    p_buy = proba_buy
    p_sell = 1.0 - p_buy
    conf = np.maximum(p_buy, p_sell)
    pred_dir = np.where(p_buy >= p_sell, 1, 0)  # 1=BUY, 0=SELL

    d['p_buy'] = p_buy
    d['p_sell'] = p_sell
    d['confidence'] = conf
    d['pred_dir'] = pred_dir

    d = d.sort_values('confidence', ascending=False)
    n_total = len(d)
    if exact_k is not None:
        n_pick = min(exact_k, n_total)
    else:
        n_pick = min(max(k_min, n_total), k_max)
    chosen = d.head(n_pick)
    return chosen


def evaluate_min50(dfall: pd.DataFrame, dates: pd.Series, clf, log_agg_only: bool = True, exact_k: int | None = 50) -> pd.DataFrame:
    """Evaluate with per-day selection.
    - If exact_k is provided (default 50), pick exactly that many entries per day.
    - Otherwise, legacy min-50 behavior applies.
    Returns per-entry log. If log_agg_only, writes one aggregated row per chosen entry
    to LOG_DIR/contexts_agg.csv and skips raw per-minute contexts.
    """
    X_all = dfall[FEATURES]
    p_buy_all = pd.Series(clf.predict_proba(X_all)[:, 1], index=dfall.index)

    logs = []
    agg_rows = []
    raw_ctx_rows = []
    for date, idx in tqdm(list(dfall.groupby(dfall['date']).groups.items()), desc='Evaluating days'):
        g = dfall.loc[idx].copy().sort_values('timestamp')
        proba = p_buy_all.loc[g.index].values
        chosen = select_entries_min50_per_day(g, proba, k_min=50, exact_k=exact_k)

        # Correctness under rule within next 8 minutes including fees
        is_buy = (chosen['pred_dir'] == 1)
        is_sell = ~is_buy
        buy_ok = (chosen['fut_max_high'] >= chosen['close'] * (1.0 + ROUND_TRIP))
        sell_ok = (chosen['fut_min_low'] <= chosen['close'] * (1.0 - ROUND_TRIP))
        correct = (is_buy & buy_ok) | (is_sell & sell_ok)

        out = chosen[['timestamp','date','close','p_buy','p_sell','confidence','pred_dir','fut_max_high','fut_min_low']].copy()
        out['correct'] = correct.astype(int)
        logs.append(out)

        # Aggregated context row pulled directly from precomputed rollups at the entry row
        present_cols = [
            'ctx_close_mean','ctx_close_std','ctx_close_min','ctx_close_max','ctx_close_range','ctx_close_slope',
            'ctx_vol_mean','ctx_vol_std'
        ]
        present_cols = [c for c in present_cols if c in g.columns]
        entry_cols = ['timestamp','date','pred_dir','confidence','correct']
        for _, r in out.iterrows():
            row = g.loc[g['timestamp'] == r['timestamp']]
            if row.empty:
                continue
            base = {**{k: r[k] for k in ['timestamp','date','pred_dir','confidence','correct']}}
            for c in present_cols:
                base[c] = float(row.iloc[0][c]) if c in row.columns and pd.notna(row.iloc[0][c]) else np.nan
            # also attach engineered features at entry
            for col in FEATURES:
                if col in row.columns:
                    base[f'entry_{col}'] = float(row.iloc[0][col]) if pd.notna(row.iloc[0][col]) else np.nan
            agg_rows.append(base)

        if not log_agg_only:
            # Optional: maintain legacy raw 10-min contexts (disabled by default)
            g_reset = g.reset_index()
            t_to_pos = {ts: i for i, ts in enumerate(g_reset['timestamp'])}
            for _, r in out.iterrows():
                ts = r['timestamp']
                if ts not in t_to_pos:
                    continue
                pos = t_to_pos[ts]
                start = max(0, pos - CONTEXT_MIN)
                ctx = g_reset.iloc[start:pos+1].copy()
                ctx['rel_min'] = np.arange(-(pos-start), 1)
                ctx['entry_timestamp'] = ts
                ctx['entry_date'] = r['date']
                ctx['entry_pred_dir'] = r['pred_dir']
                ctx['entry_confidence'] = r['confidence']
                ctx['entry_correct'] = r['correct']
                keep_cols = ['entry_timestamp','entry_date','entry_pred_dir','entry_confidence','entry_correct',
                             'timestamp','rel_min','close','high','low','volume'] + FEATURES
                present = [c for c in keep_cols if c in ctx.columns]
                raw_ctx_rows.append(ctx[present])

    # Write aggregated contexts to disk
    if agg_rows:
        agg_df = pd.DataFrame(agg_rows)
        agg_path = os.path.join(LOG_DIR, 'contexts_agg.csv')
        agg_df.to_csv(agg_path, index=False)
        print(f'Saved aggregated context features to {agg_path} ({len(agg_df)} entries)')

    # Optionally write raw contexts
    if (not log_agg_only) and raw_ctx_rows:
        ctx_df = pd.concat(raw_ctx_rows, ignore_index=True)
        ctx_path = os.path.join(LOG_DIR, 'contexts.csv')
        ctx_df.to_csv(ctx_path, index=False)
        print(f'Saved raw contexts to {ctx_path} ({len(ctx_df)} rows)')

    if not logs:
        return pd.DataFrame()
    return pd.concat(logs, ignore_index=True)


def main():
    print('Building dataset...')
    X, y, dates, df_full = build_dataset()

    # Chronological split
    train_mask = pd.to_datetime(dates).dt.date <= pd.to_datetime(TRAIN_END_DATE).date()
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, df_test = X[~train_mask], df_full[~train_mask].copy()

    print(f'Train samples: {len(X_train):,}; Test samples: {len(X_test):,}')

    print('Training LightGBM (BUY/SELL)...')
    clf = train_model(X_train, y_train)

    # Save model
    joblib.dump(clf, MODEL_PATH)
    print(f'Saved model to {MODEL_PATH}')

    # Evaluate with MIN 50 entries/day on test period
    print('Evaluating with exactly 50 entries per day (8-min correctness incl. fees)...')
    logs = evaluate_min50(df_test, pd.to_datetime(df_test['date']), clf, log_agg_only=True, exact_k=50)

    if logs.empty:
        print('No evaluation logs produced.')
        return

    # Per-day accuracy
    per_day = logs.groupby('date')['correct'].agg(['mean','size']).reset_index().rename(columns={'mean':'accuracy','size':'entries'})
    out_path = os.path.join(LOG_DIR, 'eval_min50_logs.csv')
    logs.to_csv(out_path, index=False)
    print(f'Saved per-entry eval logs to {out_path} ({len(logs)} rows)')

    print('\nPer-day summary:')
    print(per_day.head(15))
    print(f"Days evaluated: {len(per_day)} | Avg accuracy: {per_day['accuracy'].mean():.3f} | Avg entries/day: {per_day['entries'].mean():.1f}")


if __name__ == '__main__':
    main()
