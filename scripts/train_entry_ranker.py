#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm
import joblib
from numba import jit

# Reuse dataset construction from directional script via CSVs
# We assume data/btc_profitability_analysis_filtered.csv is the base and we rebuild features here.

FEE_RATE = 0.001
ROUND_TRIP = 2 * FEE_RATE
LOOKAHEAD_MIN = 8
START_HOUR = 10
MODEL_PATH = 'models/entry_ranker_lambdamart.txt'
RANDOM_STATE = 42

FEATURES_BASE = [
    'momentum_5m','volatility_5m','price_range_5m','volume_avg_5m',
    'momentum_15m','volatility_15m','price_range_15m',
    'price_change_lag_1','price_change_lag_3','price_change_lag_5'
]
ROLLUP_FEATURES = [
    'ctx_close_mean','ctx_close_std','ctx_close_min','ctx_close_max','ctx_close_range','ctx_close_slope',
    'ctx_vol_mean','ctx_vol_std','ctx_range_norm','ctx_std_norm','ctx_skew_10m'
]
REGIME_FEATURES = ['hour_sin','hour_cos','dow']
MICRO_FEATURES = [
    'ret_1m','ret_2m','ret_3m','last3_same_sign',
    'z_momentum_5m','z_price_range_5m',  # removed z_volatility_5m (redundant with regime)
    'vwap_dev','order_flow_proxy','acceleration','rejection_count'
]
SUCCESS_FILTERS = [
    'vol_regime_high','volume_regime_high','price_near_extreme'
]
TECH_INDICATORS = [
    'ema_cross','rsi_14','macd','macd_signal','macd_hist'
]
ALL_FEATURES = FEATURES_BASE + ROLLUP_FEATURES + REGIME_FEATURES + MICRO_FEATURES + SUCCESS_FILTERS + TECH_INDICATORS + ['confidence']

os.makedirs('models', exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv('data/btc_profitability_analysis_filtered.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['hour'] = df['timestamp'].dt.hour
    df = df[df['hour'] >= START_HOUR].copy()
    return df


def create_minute_features(df: pd.DataFrame, window_sizes=[5, 15, 30]) -> pd.DataFrame:
    df = df.copy()
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
    df = df.copy().sort_values(['date','timestamp']).reset_index(drop=True)
    df['fut_max_high'] = np.nan
    df['fut_min_low'] = np.nan
    for d, g in tqdm(df.groupby('date'), desc='Future window extrema by day'):
        highs = g['high'].to_numpy()
        lows  = g['low'].to_numpy()
        n = len(g)
        max_high = np.full(n, np.nan)
        min_low  = np.full(n, np.nan)
        for i in range(n):
            j2 = min(n, i + lookahead + 1)
            if i + 1 >= j2:
                continue
            max_high[i] = np.max(highs[i+1:j2])
            min_low[i]  = np.min(lows[i+1:j2])
        df.loc[g.index, 'fut_max_high'] = max_high
        df.loc[g.index, 'fut_min_low'] = min_low
    return df


def add_rollups_and_regime(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby('date')
    close = g['close']
    vol = g['volume'] if 'volume' in df.columns else None
    df['ctx_close_mean'] = close.transform(lambda s: s.rolling(window, min_periods=1).mean())
    df['ctx_close_std'] = close.transform(lambda s: s.rolling(window, min_periods=1).std())
    df['ctx_close_min'] = close.transform(lambda s: s.rolling(window, min_periods=1).min())
    df['ctx_close_max'] = close.transform(lambda s: s.rolling(window, min_periods=1).max())
    df['ctx_close_range'] = df['ctx_close_max'] - df['ctx_close_min']
    first_close = close.transform(lambda s: s.rolling(window, min_periods=1).apply(lambda x: x[0], raw=True))
    last_close = close.transform(lambda s: s.rolling(window, min_periods=1).apply(lambda x: x[-1], raw=True))
    df['ctx_close_slope'] = (last_close - first_close) / max(1, window)
    if vol is not None:
        df['ctx_vol_mean'] = vol.transform(lambda s: s.rolling(window, min_periods=1).mean())
        df['ctx_vol_std'] = vol.transform(lambda s: s.rolling(window, min_periods=1).std())
    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.dayofweek
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24.0)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24.0)
    df['ctx_range_norm'] = df['ctx_close_range'] / df['close']
    df['ctx_std_norm'] = df['ctx_close_std'] / df['ctx_close_mean'].replace(0, np.nan)
    # Rolling skew of close over window
    df['ctx_skew_10m'] = g['close'].transform(lambda s: s.rolling(window, min_periods=3).skew())
    return df


def add_micro_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Minute returns
    df['ret_1m'] = df.groupby('date')['close'].pct_change(1) * 100
    df['ret_2m'] = df.groupby('date')['close'].pct_change(2) * 100
    df['ret_3m'] = df.groupby('date')['close'].pct_change(3) * 100
    # Same-sign consistency in last 3 one-minute returns
    sign1 = np.sign(df['ret_1m'])
    sign2 = np.sign(df['ret_2m'])
    sign3 = np.sign(df['ret_3m'])
    df['last3_same_sign'] = ((sign1 == sign2) & (sign2 == sign3)).astype(int)

    # Causal (real-time) z-scores: use expanding mean/std up to the current minute within the day.
    def zscore_expanding(s: pd.Series, min_periods: int = 30) -> pd.Series:
        mean = s.expanding(min_periods=min_periods).mean()
        std = s.expanding(min_periods=min_periods).std()
        z = (s - mean) / std.replace(0, np.nan)
        return z.fillna(0.0)

    for col in ['momentum_5m', 'price_range_5m']:
        if col in df.columns:
            df[f'z_{col}'] = df.groupby('date')[col].transform(lambda s: zscore_expanding(s, 30))

    return df


def add_advanced_microstructure(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add advanced microstructure features from rolling windows."""
    df = df.copy()
    
    # Need open/high/low for some features; if missing, approximate
    if 'open' not in df.columns:
        df['open'] = df['close']  # fallback
    if 'volume' not in df.columns:
        df['volume'] = 0.0
    
    g = df.groupby('date')
    close = df['close']
    high = df['high'] if 'high' in df.columns else close
    low = df['low'] if 'low' in df.columns else close
    vol = df['volume']
    
    # 1. VWAP deviation: (close - session_vwap) / session_vwap
    # Approximate session VWAP from start of day to current row
    cum_vol = g['volume'].cumsum()
    cum_pv = (df['close'] * df['volume']).groupby(df['date']).cumsum()
    session_vwap = cum_pv / cum_vol.replace(0, np.nan)
    df['vwap_dev'] = ((close - session_vwap) / session_vwap * 100).fillna(0.0)
    
    # 2. Order flow proxy: cumulative (close - open) over rolling window, volume-weighted
    df['co_diff'] = df['close'] - df['open']
    df['vw_co'] = df['co_diff'] * df['volume']
    rolling_vw_co = g['vw_co'].transform(lambda s: s.rolling(window, min_periods=1).sum())
    rolling_vol = g['volume'].transform(lambda s: s.rolling(window, min_periods=1).sum())
    df['order_flow_proxy'] = (rolling_vw_co / rolling_vol.replace(0, np.nan)).fillna(0.0)
    
    # 3. Acceleration: slope of last 3 minutes vs full 10-minute slope
    # Full 10-min slope already in ctx_close_slope; compute last-3-min slope
    def rolling_slope(s, w):
        def slope(arr):
            if len(arr) < 2:
                return 0.0
            x = np.arange(len(arr))
            A = np.vstack([x, np.ones(len(x))]).T
            m, _ = np.linalg.lstsq(A, arr, rcond=None)[0]
            return m
        return s.rolling(w, min_periods=2).apply(slope, raw=True)
    
    slope_3m = g['close'].transform(lambda s: rolling_slope(s, 3))
    # Acceleration = recent slope / longer slope (if longer slope exists)
    df['acceleration'] = (slope_3m / (df['ctx_close_slope'].replace(0, np.nan))).fillna(0.0)
    
    # 4. Rejection count: how many times price touched high/low extremes and reversed (numba JIT)
    @jit(nopython=True)
    def count_rejections_fast(highs, lows, closes):
        if len(closes) < 2:
            return 0
        rej = 0
        for i in range(len(closes)-1):
            h, l, c, c_next = highs[i], lows[i], closes[i], closes[i+1]
            rng = h - l
            if rng < 1e-6:
                continue
            if abs(c - h) / rng < 0.01 and c_next < c:
                rej += 1
            elif abs(c - l) / rng < 0.01 and c_next > c:
                rej += 1
        return rej
    
    @jit(nopython=True)
    def rolling_rejection_fast(highs, lows, closes, w):
        n = len(closes)
        result = np.zeros(n)
        for i in range(n):
            start = max(0, i - w + 1)
            result[i] = count_rejections_fast(highs[start:i+1], lows[start:i+1], closes[start:i+1])
        return result
    
    rej_counts = []
    for d, day_df in tqdm(list(df.groupby('date')), desc='Computing rejection counts', leave=False):
        day_df = day_df.reset_index(drop=True)
        highs = day_df['high'].values if 'high' in day_df.columns else day_df['close'].values
        lows = day_df['low'].values if 'low' in day_df.columns else day_df['close'].values
        closes = day_df['close'].values
        rej = rolling_rejection_fast(highs, lows, closes, window)
        rej_counts.extend(rej)
    df['rejection_count'] = rej_counts
    
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add classic technical indicators: EMA crossover, RSI, MACD."""
    df = df.copy()
    g = df.groupby('date')
    close = df['close']
    
    # 1. EMA Crossover: EMA(9) - EMA(21)
    ema9 = g['close'].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    ema21 = g['close'].transform(lambda s: s.ewm(span=21, adjust=False).mean())
    df['ema_cross'] = ema9 - ema21
    
    # 2. RSI (14-period)
    def compute_rsi(s, period=14):
        delta = s.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)
    
    df['rsi_14'] = g['close'].transform(lambda s: compute_rsi(s, 14))
    
    # 3. MACD (12, 26, 9)
    ema12 = g['close'].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = g['close'].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df['macd'] = ema12 - ema26
    # MACD signal: 9-period EMA of MACD per group
    df['macd_signal'] = g['macd'].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    return df


def add_success_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Add explicit filters based on success pattern analysis.

    IMPORTANT: These must be causal (real-time). Do NOT use whole-day quantiles/means.
    """
    df = df.copy()

    def zscore_expanding(s: pd.Series, min_periods: int = 30) -> pd.Series:
        mean = s.expanding(min_periods=min_periods).mean()
        std = s.expanding(min_periods=min_periods).std()
        z = (s - mean) / std.replace(0, np.nan)
        return z.fillna(0.0)

    # 1. High volatility/range regime: high relative to what we've seen so far today
    z_std = df.groupby('date')['ctx_close_std'].transform(lambda s: zscore_expanding(s, 30))
    z_rng = df.groupby('date')['ctx_close_range'].transform(lambda s: zscore_expanding(s, 30))
    df['vol_regime_high'] = ((z_std > 0.5) | (z_rng > 0.5)).astype(int)

    # 2. High volume regime: high relative to what we've seen so far today
    if 'ctx_vol_mean' in df.columns:
        z_vol = df.groupby('date')['ctx_vol_mean'].transform(lambda s: zscore_expanding(s, 30))
        df['volume_regime_high'] = (z_vol > 0.5).astype(int)
    else:
        df['volume_regime_high'] = 0

    # 3. Price near extreme: within bottom 20% or top 20% of recent 10-min range (already causal)
    near_low = (df['close'] - df['ctx_close_min']) / (df['ctx_close_range'] + 1e-6) < 0.2
    near_high = (df['ctx_close_max'] - df['close']) / (df['ctx_close_range'] + 1e-6) < 0.2
    df['price_near_extreme'] = (near_low | near_high).astype(int)

    return df


def build_rank_dataset():
    print('[1/8] Loading and processing data...')
    df = load_data()
    print(f'  Loaded {len(df):,} rows')
    print('[2/8] Creating minute features...')
    df = create_minute_features(df)
    print('[3/8] Computing future window extrema...')
    df = add_future_window_extrema(df, LOOKAHEAD_MIN)
    print('[4/8] Adding rollups and regime...')
    df = add_rollups_and_regime(df, 10)
    print('[5/8] Adding micro features...')
    df = add_micro_features(df)
    print('[6/8] Adding advanced microstructure (this may take 1-2 minutes)...')
    df = add_advanced_microstructure(df, 10)
    print('[7/8] Adding technical indicators...')
    df = add_technical_indicators(df)
    print('[8/8] Adding success filters...')
    df = add_success_filters(df)
    print(f'  Final dataset: {len(df):,} rows, {len(df.columns)} columns')

    print('Computing directional labels...')
    # Compute BUY/SELL confidence and direction from a simple heuristic: p_buy via up_move potential proxy
    # Prefer to reuse a trained directional classifier if available; otherwise derive confidence from existing features.
    # Here, we approximate confidence with max of momentum-based signals to keep it lightweight if model is absent.
    if os.path.exists('models/entry_directional_min50.joblib'):
        try:
            mdl = joblib.load('models/entry_directional_min50.joblib')
            X = df[FEATURES_BASE]
            df['p_buy'] = mdl.predict_proba(X)[:,1]
        except Exception:
            df['p_buy'] = 0.5
    else:
        df['p_buy'] = 0.5
    df['p_sell'] = 1.0 - df['p_buy']
    df['confidence'] = np.maximum(df['p_buy'], df['p_sell'])
    df['pred_dir'] = (df['p_buy'] >= df['p_sell']).astype(int)  # 1=BUY, 0=SELL

    # Label per minute based on predicted direction and 8-min fees rule
    buy_ok = (df['fut_max_high'] >= df['close'] * (1.0 + ROUND_TRIP))
    sell_ok = (df['fut_min_low'] <= df['close'] * (1.0 - ROUND_TRIP))
    df['correct'] = ((df['pred_dir']==1) & buy_ok) | ((df['pred_dir']==0) & sell_ok)
    df['correct'] = df['correct'].fillna(0).astype(int)

    # Drop rows missing necessary features
    need = FEATURES_BASE + ['fut_max_high','fut_min_low']
    df = df.dropna(subset=need).copy()

    print('Splitting train/validation...')
    # Train/valid split by date (chronological)
    df['date'] = pd.to_datetime(df['date']).dt.date
    dates = sorted(df['date'].unique())
    if len(dates) < 10:
        split = int(0.8*len(dates))
    else:
        split = int(0.9*len(dates))
    train_dates = set(dates[:split])

    train_df = df[df['date'].isin(train_dates)].copy()
    valid_df = df[~df['date'].isin(train_dates)].copy()
    print(f'  Train: {len(train_df):,} rows ({len(train_dates)} days)')
    print(f'  Valid: {len(valid_df):,} rows ({len(dates)-len(train_dates)} days)')

    feat_cols = [c for c in ALL_FEATURES if c in df.columns]
    X_tr = train_df[feat_cols].fillna(0.0).values
    y_tr = train_df['correct'].values
    grp_tr = train_df.groupby('date').size().to_list()

    X_va = valid_df[feat_cols].fillna(0.0).values
    y_va = valid_df['correct'].values
    grp_va = valid_df.groupby('date').size().to_list()

    return feat_cols, (X_tr, y_tr, grp_tr), (X_va, y_va, grp_va), valid_df


def train_ranker():
    feat_cols, train_pack, valid_pack, valid_df = build_rank_dataset()
    X_tr, y_tr, g_tr = train_pack
    X_va, y_va, g_va = valid_pack

    # Train LightGBM ranker
    print('Training LightGBM ranker...')
    train_set = lgb.Dataset(X_tr, label=y_tr, group=g_tr, feature_name=list(feat_cols))
    valid_set = lgb.Dataset(X_va, label=y_va, group=g_va, feature_name=list(feat_cols))

    params_lgb = dict(
        objective='lambdarank',
        metric='map',
        eval_at=[50,100,200],
        num_leaves=63,
        learning_rate=0.05,
        min_data_in_leaf=50,
        feature_pre_filter=False,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )

    print('  - Training 1000 boosting rounds with early stopping...')
    lgb_booster = lgb.train(params_lgb, train_set, num_boost_round=1000, valid_sets=[valid_set],
                            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(period=100)])

    lgb_booster.save_model(MODEL_PATH)
    print(f'Saved LightGBM ranker to {MODEL_PATH}')

    # Train XGBoost ranker
    print('Training XGBoost ranker...')
    xgb_model_path = MODEL_PATH.replace('.txt', '_xgb.json')
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtrain.set_group(g_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va)
    dvalid.set_group(g_va)
    
    params_xgb = {
        'objective': 'rank:pairwise',
        'eval_metric': 'map@50',
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': RANDOM_STATE,
    }
    
    print('  - Training 1000 boosting rounds with early stopping...')
    xgb_booster = xgb.train(params_xgb, dtrain, num_boost_round=1000, evals=[(dvalid, 'valid')],
                            early_stopping_rounds=150, verbose_eval=100)
    
    xgb_booster.save_model(xgb_model_path)
    print(f'Saved XGBoost ranker to {xgb_model_path}')

    # Ensemble: average LightGBM and XGBoost scores
    print('Computing ensemble scores...')
    lgb_scores = lgb_booster.predict(X_va)
    xgb_scores = xgb_booster.predict(dvalid)
    ensemble_scores = (lgb_scores + xgb_scores) / 2.0
    
    valid_df = valid_df.copy()
    valid_df['score'] = ensemble_scores
    
    prec_stats = []
    for k in [50,100,200]:
        num = den = 0
        for d, g in valid_df.groupby('date'):
            g = g.sort_values('score', ascending=False).head(k)
            if len(g)==0:
                continue
            num += int(g['correct'].sum())
            den += len(g)
        prec = (num/den) if den>0 else 0.0
        prec_stats.append((k, prec))
    print('Ensemble validation precision:', {k: round(p,3) for k,p in prec_stats})


if __name__ == '__main__':
    train_ranker()