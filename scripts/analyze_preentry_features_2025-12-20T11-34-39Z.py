#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif

from binance_adapter.crossref_oracle_vs_agent_patterns import compute_feature_frame


def build_minute_bars(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, parse_dates=['timestamp'])
    # normalize to minute bars
    g = (
        raw.assign(ts_min=pd.to_datetime(raw['timestamp']).dt.floor('min'))
           .groupby('ts_min')
           .agg(open=('open','first'), high=('high','max'), low=('low','min'), close=('close','last'), volume=('volume','sum'))
           .reset_index()
    )
    return g


def label_future_profit(df: pd.DataFrame, hold_min: int = 8, thr_profit_pct: float = 0.10) -> pd.Series:
    close = df['close'].to_numpy(np.float64, copy=False)
    n = close.size
    # forward maximum return within next hold_min minutes relative to current close
    fwd_max = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        j2 = min(n, i + hold_min + 1)
        if i + 1 >= j2:
            fwd_max[i] = np.nan
            continue
        mx = np.max(close[i+1:j2])
        fwd_max[i] = (mx / close[i] - 1.0) * 100.0
    y = (fwd_max >= thr_profit_pct).astype(np.float32)
    # last few rows cannot be labeled
    y[np.isnan(fwd_max)] = np.nan
    return pd.Series(y, index=df.index, name='y')


def main():
    ap = argparse.ArgumentParser(description='Analyze 5-minute pre-entry context features on BTCUSDT data (no entry/exit logic).')
    ap.add_argument('--market-csv', default='data/btc_profitability_analysis_filtered.csv')
    ap.add_argument('--limit', type=int, default=300000, help='Use last N rows for speed (None = all)')
    ap.add_argument('--hold-min', type=int, default=8)
    ap.add_argument('--thr-profit-pct', type=float, default=0.10)
    ap.add_argument('--random-state', type=int, default=42)
    args = ap.parse_args()

    src = Path(args.market_csv)
    if not src.exists():
        raise SystemExit(f'Missing market CSV: {src}')

    mkt = build_minute_bars(src)
    if args.limit is not None and args.limit > 0 and len(mkt) > args.limit:
        mkt = mkt.sort_values('ts_min').iloc[-args.limit:].reset_index(drop=True)

    feats = compute_feature_frame(mkt)

    # binary label: can we hit +thr within next hold_min minutes?
    lab = label_future_profit(mkt.rename(columns={'ts_min':'timestamp'}), hold_min=args.hold_min, thr_profit_pct=args.thr_profit_pct)
    df = feats.join(lab)
    df = df.dropna(subset=['y'])

    X = df.drop(columns=['ts_min','y'])
    y = df['y'].astype(int)

    X = X.fillna(0.0).astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, shuffle=False)

    # Quick model: RandomForest for robust importances
    clf = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=50, n_jobs=-1, random_state=args.random_state)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, prob)

    # Importances
    rf_imp = pd.Series(clf.feature_importances_, index=X.columns, name='rf_importance')
    pi = permutation_importance(clf, Xte, yte, n_repeats=8, random_state=args.random_state, n_jobs=-1)
    perm_imp = pd.Series(pi.importances_mean, index=X.columns, name='perm_importance')
    mi = pd.Series(mutual_info_classif(Xtr, ytr, random_state=args.random_state, discrete_features=False), index=X.columns, name='mutual_info')

    out = pd.concat([rf_imp, perm_imp, mi], axis=1)
    out['rank_mean'] = out.rank(ascending=False).mean(axis=1)
    out = out.sort_values('rank_mean')

    print(f'AUC on holdout: {auc:.4f}')
    print('\nTop 10 helpful features:')
    print(out.head(10).round(6).to_string())

    print('\nLikely noise (bottom 10):')
    print(out.tail(10).round(6).to_string())


if __name__ == '__main__':
    main()
