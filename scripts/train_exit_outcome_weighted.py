import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from tqdm import tqdm

TRAIN_END_DATE = '2024-12-31'
THR_LIST = [0.35, 0.40, 0.45, 0.50, 0.55]
MIN_PROFIT_PCT = 0.10

FEATURES = [
    'elapsed_min','since_entry_return_pct','prob_enter_at_entry','daily_pred_high_profit',
    'momentum_3m','momentum_5m','momentum_10m','volatility_3m','volatility_5m','volatility_10m',
    'drawdown_from_max_pct','since_entry_cummax','mins_since_peak'
]


def main():
    print('Loading exit features and trade outcomes...')
    Xdf = pd.read_csv('data/exit_timing_features.csv')
    Xdf['timestamp'] = pd.to_datetime(Xdf['timestamp'])
    Xdf['date'] = Xdf['timestamp'].dt.date

    outcomes = pd.read_csv('data/trade_outcomes_all.csv')

    print('Constructing outcome-weighted labels (3–8m window, >=0.10% positive)...')
    # Identify start rows to get trade_id mapping consistent
    Xdf['is_start'] = (Xdf['elapsed_min'] == 0).astype(int)
    Xdf['trade_id'] = Xdf.groupby('date')['is_start'].cumsum()

    # Filter to window and build labels
    W = Xdf[(Xdf['elapsed_min'] >= 3) & (Xdf['elapsed_min'] <= 8)].copy()
    W['y'] = (W['since_entry_return_pct'] >= MIN_PROFIT_PCT).astype(int)

    # Join outcomes for weights
    W = W.merge(outcomes[['date','trade_id','success']], on=['date','trade_id'], how='left')
    W['success'] = W['success'].fillna(0).astype(int)
    # Weights: 3x for minutes from successful trades, 1x otherwise
    W['w'] = 1 + 2*W['success']

    # Train/test split
    train_mask = W['date'] <= pd.to_datetime(TRAIN_END_DATE).date()
    X_train = W.loc[train_mask, FEATURES]
    y_train = W.loc[train_mask, 'y']
    w_train = W.loc[train_mask, 'w']
    X_test  = W.loc[~train_mask, FEATURES]
    y_test  = W.loc[~train_mask, 'y']

    print(f'Train rows: {len(X_train):,}; Test rows: {len(X_test):,}')

    clf = lgb.LGBMClassifier(objective='binary', random_state=42, n_estimators=500, learning_rate=0.05)
    print('Training with outcome weights...')
    clf.fit(X_train, y_train, sample_weight=w_train)

    # Save the model
    joblib.dump(clf, 'models/exit_timing_model_weighted.joblib')

    # Evaluate profit via simulation on 2025
    print('\nSimulating on 2025 (per-trade choose first prob>=thr in 3–8m; fallback 8m)...')
    T = Xdf[Xdf['date'] > pd.to_datetime(TRAIN_END_DATE).date()].copy()
    T['prob_exit'] = clf.predict_proba(T[FEATURES])[:,1]
    T['is_start'] = (T['elapsed_min'] == 0).astype(int)
    T['trade_id'] = T.groupby('date')['is_start'].cumsum()

    def simulate(thr: float):
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
            return 0, 0.0, 0.0
        total = float(chosen_df.groupby('date')['since_entry_return_pct'].sum().sum())
        avg = float(chosen_df['since_entry_return_pct'].mean())
        n = int(chosen_df.shape[0])
        return n, avg, total

    best = None
    for thr in THR_LIST:
        n, avg, total = simulate(thr)
        print(f"thr={thr:.2f} -> trades={n}, avg={avg:.3f}%, total={total:.2f}%")
        if best is None or total > best[2]:
            best = (thr, n, total, avg)

    if best:
        print(f"\nBest threshold: {best[0]:.2f}  trades={best[1]}  total={best[2]:.2f}%  avg={best[3]:.3f}%")
    else:
        print('No trades produced.')

if __name__ == '__main__':
    main()
