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

from binance_adapter.crossref_oracle_vs_agent_patterns import compute_feature_frame, map_ts_to_index


FEATURE_AGG = "last"  # choose from: last, mean


def build_minute_bars(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, parse_dates=['timestamp'])
    g = (
        raw.assign(ts_min=pd.to_datetime(raw['timestamp']).dt.floor('min'))
           .groupby('ts_min')
           .agg(open=('open','first'), high=('high','max'), low=('low','min'), close=('close','last'), volume=('volume','sum'))
           .reset_index()
    )
    return g


def label_future_profit_from_entry(mkt: pd.DataFrame, entry_idx: np.ndarray, hold_min: int, thr_profit_pct: float) -> np.ndarray:
    close = mkt['close'].to_numpy(np.float64, copy=False)
    n = close.size
    y = np.full(entry_idx.size, np.nan, dtype=np.float64)
    for i, idx in enumerate(entry_idx):
        if idx < 0 or idx >= n - 1:
            y[i] = np.nan
            continue
        j2 = min(n, idx + hold_min + 1)
        if idx + 1 >= j2:
            y[i] = np.nan
            continue
        mx = np.max(close[idx+1:j2])
        y[i] = 1.0 if ((mx / close[idx] - 1.0) * 100.0) >= thr_profit_pct else 0.0
    return y


def main():
    ap = argparse.ArgumentParser(description='Analyze 5-minute pre-entry contexts: which features help most vs noise.')
    ap.add_argument('--market-csv', default='data/btc_profitability_analysis_filtered.csv')
    ap.add_argument('--entries-csv', required=True, help='CSV with entry_time column (datetime)')
    ap.add_argument('--limit', type=int, default=None, help='Use last N minutes of market for speed (optional)')
    ap.add_argument('--hold-min', type=int, default=8)
    ap.add_argument('--thr-profit-pct', type=float, default=0.10)
    ap.add_argument('--n-estimators', type=int, default=400)
    ap.add_argument('--min-samples-leaf', type=int, default=50)
    ap.add_argument('--random-state', type=int, default=42)
    args = ap.parse_args()

    mkt = build_minute_bars(Path(args.market_csv))
    if args.limit is not None and args.limit > 0 and len(mkt) > args.limit:
        mkt = mkt.sort_values('ts_min').iloc[-args.limit:].reset_index(drop=True)

    feats = compute_feature_frame(mkt)
    ts_sorted = feats['ts_min'].to_numpy(dtype='datetime64[ns]')

    entries_raw = pd.read_csv(args.entries_csv)
    # Support either 'entry_time' or 'entry_ts'
    if 'entry_time' in entries_raw.columns:
        entries_raw['entry_min'] = pd.to_datetime(entries_raw['entry_time']).dt.floor('min')
    elif 'entry_ts' in entries_raw.columns:
        entries_raw['entry_min'] = pd.to_datetime(entries_raw['entry_ts']).dt.floor('min')
    else:
        raise SystemExit("entries CSV must contain 'entry_time' or 'entry_ts'")
    entries = entries_raw
    eidx = map_ts_to_index(ts_sorted, entries['entry_min'].to_numpy(dtype='datetime64[ns]'))
    ok = eidx >= 0
    entries = entries.loc[ok].reset_index(drop=True)
    eidx = eidx[ok]

    # Build 5-minute precontext rows for each entry: rel in {-5,-4,-3,-2,-1}
    offsets = np.array([-5, -4, -3, -2, -1], dtype=np.int64)
    mat = eidx[:, None] + offsets[None, :]
    valid = mat >= 0

    Fcols = [c for c in feats.columns if c not in ('ts_min',)]
    # gather feature arrays
    parr = {c: feats[c].to_numpy(np.float64, copy=False) for c in Fcols}

    rows = []
    for i in range(mat.shape[0]):
        idxs = mat[i][valid[i]]
        d = {'entry_idx': int(eidx[i]), 'entry_time': entries.loc[i, 'entry_min']}
        if FEATURE_AGG == 'last':
            for c, arr in parr.items():
                d[c] = float(arr[idxs[-1]]) if idxs.size > 0 else np.nan
        else:  # mean
            for c, arr in parr.items():
                v = arr[idxs]
                d[c] = float(np.nanmean(v)) if v.size else np.nan
        rows.append(d)
    Xdf = pd.DataFrame(rows)

    # Labels: based on future outcome after entry
    y = label_future_profit_from_entry(mkt, eidx, args.hold_min, args.thr_profit_pct)
    Xdf['y'] = y
    Xdf = Xdf.dropna(subset=['y']).reset_index(drop=True)

    X = Xdf.drop(columns=['entry_idx','entry_time','y'])
    y = Xdf['y'].astype(int)
    X = X.fillna(0.0).astype(np.float32)

    # Chronological split (respecting time)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, shuffle=False)

    clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=None, min_samples_leaf=args.min_samples_leaf, n_jobs=-1, random_state=args.random_state)
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

    print(f'Entries used: {len(X)}  AUC on holdout: {auc:.4f}')
    print('\nTop 15 helpful precontext features (5m window):')
    print(out.head(15).round(6).to_string())

    print('\nLikely noise (bottom 15):')
    print(out.tail(15).round(6).to_string())


if __name__ == '__main__':
    main()
