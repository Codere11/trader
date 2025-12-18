#!/usr/bin/env python3
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

IN_DIR = Path('data/entry_ranked_outputs/failure_precontext')
MODEL_PATH = 'models/preentry_mistake_filter.joblib'

FEATURES = [
    'mom_1m','mom_3m','mom_5m','vol_std_5m','range_5m','range_norm_5m',
    'slope_ols_5m','rsi_14','macd','macd_hist','vwap_dev','last3_same_sign',
    'score','confidence','p_buy'
]


def main():
    fail = pd.read_csv(IN_DIR / 'failures_precontext.csv')
    win  = pd.read_csv(IN_DIR / 'winners_precontext_sample.csv')

    F = pd.concat([fail.assign(y=1), win.assign(y=0)], ignore_index=True)
    F = F.dropna(subset=FEATURES)

    X = F[FEATURES].copy()
    # cast booleans/ints
    if 'last3_same_sign' in X.columns:
        X['last3_same_sign'] = X['last3_same_sign'].astype(int)
    y = F['y'].astype(int)

    # simple scaler-free model
    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(objective='binary', n_estimators=400, learning_rate=0.05, num_leaves=63, random_state=42)
    clf.fit(X, y)

    # choose threshold targeting precision on failures
    from sklearn.metrics import precision_recall_curve
    prob = clf.predict_proba(X)[:,1]
    prec, rec, thr = precision_recall_curve(y, prob)
    # target precision >= 0.65 for failure class if possible
    target_thr = 0.5
    cand = [(p, r, t) for p, r, t in zip(prec, rec, thr) if p >= 0.65]
    if cand:
        # pick threshold with highest recall under the precision constraint
        target_thr = max(cand, key=lambda x: x[1])[2]

    payload = {
        'model': clf,
        'features': FEATURES,
        'threshold': float(target_thr)
    }
    joblib.dump(payload, MODEL_PATH)
    print(f'Saved pre-entry mistake filter to {MODEL_PATH} (threshold={target_thr:.3f})')

if __name__ == '__main__':
    main()
