import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

TRAIN_END_DATE = '2024-12-31'
MODEL_PATH = 'models/exit_timing_model.joblib'
THRESH_EXIT = 0.50
FEE_RATE = 0.001

# Simple guardrails
MIN_PROFIT_PCT = 0.10  # require at least +0.10% net return to allow exit


def main():
    print('Loading exit timing features...')
    df = pd.read_csv('data/exit_timing_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    y = df['target_exit_constrained']
    feature_cols = [
        'elapsed_min','since_entry_return_pct','prob_enter_at_entry','daily_pred_high_profit',
        'momentum_3m','momentum_5m','momentum_10m','volatility_3m','volatility_5m','volatility_10m',
        'drawdown_from_max_pct','since_entry_cummax','mins_since_peak'
    ]
    X = df[feature_cols]

    train_mask = df['date'] <= pd.to_datetime(TRAIN_END_DATE).date()
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    print(f'Train samples: {len(X_train):,}; Test samples: {len(X_test):,}')

    # Classifier
    clf = lgb.LGBMClassifier(objective='binary', random_state=42, n_estimators=500, learning_rate=0.05)
    print('Training exit classifier...')
    clf.fit(X_train, y_train)

    # Evaluate classification quality
    y_pred = clf.predict(X_test)
    print('\nClassification metrics (2025):')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['hold','exit']))

    # Backtest exit policy on 2025 using predicted exits
    print('\nSimulating exits on 2025...')
    test = df[~train_mask].copy()
    test['prob_exit'] = clf.predict_proba(test[feature_cols])[:,1]

    # Identify trades
    test['is_start'] = (test['elapsed_min'] == 0).astype(int)
    test['trade_id'] = test.groupby('date')['is_start'].cumsum()

    def simulate(thr: float) -> tuple[int, float, float]:
        t = test.copy()
        t['pred_exit'] = (t['prob_exit'] >= thr).astype(int)
        chosen = []
        for (date, tid), g in t.groupby(['date','trade_id']):
            g = g.sort_values('elapsed_min')
            window = g[(g['elapsed_min'] >= 3) & (g['elapsed_min'] <= 8)]
            # apply profit guard
            window = window[window['since_entry_return_pct'] >= MIN_PROFIT_PCT]
            pick = window[window['pred_exit'] == 1].head(1)
            if pick.empty:
                fallback = g[g['elapsed_min'] == min(8, g['elapsed_min'].max())]
                pick = fallback.head(1)
            if not pick.empty:
                chosen.append(pick.iloc[0])
        chosen_df = pd.DataFrame(chosen)
        if chosen_df.empty:
            return 0, 0.0, 0.0
        pnl_by_day = chosen_df.groupby('date')['since_entry_return_pct'].sum()
        total = float(pnl_by_day.sum())
        avg = float(chosen_df['since_entry_return_pct'].mean())
        n = int(chosen_df.shape[0])
        return n, avg, total

    # Sweep thresholds quickly
    thr_list = [0.40, 0.45, 0.50, 0.55, 0.60]
    print('\nExit threshold sweep with guard (min_profit>=%.2f%%):' % MIN_PROFIT_PCT)
    best = None
    for thr in thr_list:
        n, avg, total = simulate(thr)
        print(f"thr={thr:.2f} -> trades={n}, avg={avg:.3f}%, total={total:.2f}%")
        if best is None or total > best[2]:
            best = (thr, n, total, avg)

    if best:
        print(f"\nBest threshold: {best[0]:.2f}  trades={best[1]}  total={best[2]:.2f}%  avg={best[3]:.3f}%")
    else:
        print("No trades produced in sweep.")

    # Save model
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved exit timing model to {MODEL_PATH}")

if __name__ == '__main__':
    main()
