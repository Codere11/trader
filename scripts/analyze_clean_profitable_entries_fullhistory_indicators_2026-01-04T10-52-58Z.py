#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T10:52:58Z
"""Analyze what's left: cleaned profitable entries using full-history indicators.

Inputs
- Full 1m ETH-USD candles with base features:
    data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet
- Clean + profitable (>=0.2%) entry datasets:
    data/entry_precontext_clean_top10pct_retge0p2_*/BUY_clean_top10pct_retge0p2_*.parquet
    data/entry_precontext_clean_top10pct_retge0p2_*/SELL_clean_top10pct_retge0p2_*.parquet

What it does
- Builds *full-history* indicators on the entire 1m series (RSI/ATR/BB/EMA, multi-horizon returns/vol/range, slopes).
- Joins indicators at each trade entry timestamp.
- Scores:
  1) Univariate ranking by:
     - Spearman corr with trade_net_return_pct
     - AUC / balanced-accuracy for separating high vs low returns
       (labels = top-q vs bottom-q returns; middle dropped)
  2) Pair ranking using logistic regression on the same high/low task.

Notes
- This intentionally does NOT use the 5-minute precontext descriptor columns.
- Since the dataset is already filtered to trade_net_return_pct>=0.2, "accuracy" is measured
  for distinguishing *better vs worse* winners (high vs low quantiles), not winner vs loser.

Outputs (timestamped)
- data/analysis_fullhistory_indicators_<ts>/BUY/...
- data/analysis_fullhistory_indicators_<ts>/SELL/...
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _pick_latest(glob_pat: str) -> Path:
    c = sorted(Path("data").glob(glob_pat))
    if not c:
        raise SystemExit(f"No matches for: data/{glob_pat}")
    return c[-1]


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)

    # Wilder-style smoothing (EMA with alpha=1/period)
    roll_up = up.ewm(alpha=1.0 / float(period), adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / float(period), adjust=False).mean()

    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1.0 / float(period), adjust=False).mean()
    return 100.0 * atr / close.replace(0.0, np.nan)


def _build_bar_features(bars: pd.DataFrame) -> pd.DataFrame:
    """bars indexed by timestamp."""
    close = bars["close"].astype(np.float64)
    high = bars["high"].astype(np.float64)
    low = bars["low"].astype(np.float64)
    vol = bars["volume"].astype(np.float64)

    out = pd.DataFrame(index=bars.index)

    # Time-of-day features (crypto still has intraday microstructure)
    out["bar_hour"] = bars.index.hour.astype(np.int16)
    out["bar_dow"] = bars.index.dayofweek.astype(np.int16)

    # Basic transforms
    out["bar_log_close"] = np.log(close.replace(0.0, np.nan))
    out["bar_vol_log1p"] = np.log1p(vol.clip(lower=0.0))

    # Returns / momentum horizons (percent)
    for w in [5, 15, 60, 240, 1440]:
        out[f"bar_ret_{w}m_pct"] = 100.0 * (close / close.shift(w) - 1.0)
        out[f"bar_vol_std_{w}m"] = (
            pd.to_numeric(bars.get("ret_1m_pct", pd.Series(index=bars.index, dtype=np.float64)), errors="coerce")
            .astype(np.float64)
            .rolling(w, min_periods=w)
            .std()
        )
        hh = high.rolling(w, min_periods=w).max()
        ll = low.rolling(w, min_periods=w).min()
        out[f"bar_range_{w}m_pct"] = 100.0 * (hh - ll) / close.replace(0.0, np.nan)

        # Simple slope proxies (per-minute)
        out[f"bar_slope_logclose_{w}m"] = (out["bar_log_close"] - out["bar_log_close"].shift(w)) / float(w)
        out[f"bar_slope_macd_{w}m"] = (pd.to_numeric(bars.get("macd"), errors="coerce") - pd.to_numeric(bars.get("macd"), errors="coerce").shift(w)) / float(w)

    # RSI
    out["bar_rsi_14"] = _rsi(close, 14)
    out["bar_rsi_28"] = _rsi(close, 28)

    # RSI slopes
    for w in [15, 60, 240]:
        out[f"bar_slope_rsi14_{w}m"] = (out["bar_rsi_14"] - out["bar_rsi_14"].shift(w)) / float(w)

    # ATR
    out["bar_atr14_pct"] = _atr_pct(high, low, close, 14)

    # Bollinger (20)
    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    out["bar_bb20_z"] = (close - sma20) / std20.replace(0.0, np.nan)
    out["bar_bb20_width_pct"] = 100.0 * (4.0 * std20) / sma20.replace(0.0, np.nan)

    # EMAs and trend spreads
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    out["bar_ema20_dev_pct"] = 100.0 * (close / ema20 - 1.0)
    out["bar_ema60_dev_pct"] = 100.0 * (close / ema60 - 1.0)
    out["bar_ema200_dev_pct"] = 100.0 * (close / ema200 - 1.0)
    out["bar_ema20_ema60_spread_pct"] = 100.0 * (ema20 / ema60 - 1.0)

    # MACD signal / hist
    macd = pd.to_numeric(bars.get("macd"), errors="coerce").astype(np.float64)
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    out["bar_macd"] = macd
    out["bar_macd_signal"] = macd_sig
    out["bar_macd_hist"] = macd - macd_sig

    # Existing short-window deviations
    if "vwap_dev_5m" in bars.columns:
        out["bar_vwap_dev_5m"] = pd.to_numeric(bars["vwap_dev_5m"], errors="coerce").astype(np.float64)

    # Reduce memory
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].astype(np.float32)

    return out


def _time_split_idx(n: int, test_frac: float) -> int:
    return int(n * (1.0 - float(test_frac)))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    # Spearman via rank correlation; ignore NaNs.
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx = pd.Series(x[m]).rank().to_numpy(np.float64)
    ry = pd.Series(y[m]).rank().to_numpy(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def _best_threshold_bal_acc(x_tr: np.ndarray, y_tr: np.ndarray, direction: int) -> float:
    # scan thresholds on train; direction=+1 means predict 1 when x>=thr, -1 means predict 1 when x<=thr
    x = x_tr[np.isfinite(x_tr)]
    if x.size == 0:
        return float("nan")
    qs = np.linspace(0.02, 0.98, 97)
    thrs = np.quantile(x, qs)

    best_thr = float(thrs[0])
    best = -1.0
    for thr in thrs:
        if direction > 0:
            pred = (x_tr >= thr).astype(int)
        else:
            pred = (x_tr <= thr).astype(int)
        m = np.isfinite(x_tr)
        if m.sum() < 3:
            continue
        s = balanced_accuracy_score(y_tr[m], pred[m])
        if s > best:
            best = float(s)
            best_thr = float(thr)
    return best_thr


def _auc_signed(y: np.ndarray, x: np.ndarray) -> tuple[float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return float("nan"), 1
    try:
        auc = float(roc_auc_score(y[m], x[m]))
    except Exception:
        return float("nan"), 1
    if auc >= 0.5:
        return auc, 1
    return 1.0 - auc, -1


def _make_hi_lo_labels(y: np.ndarray, q_lo: float, q_hi: float) -> tuple[np.ndarray, float, float]:
    lo = float(np.nanquantile(y, q_lo))
    hi = float(np.nanquantile(y, q_hi))
    lab = np.full_like(y, fill_value=-1, dtype=np.int8)
    lab[y <= lo] = 0
    lab[y >= hi] = 1
    return lab, lo, hi


def _analyze_side(
    *,
    side: str,
    trades: pd.DataFrame,
    bar_feat: pd.DataFrame,
    out_dir: Path,
    test_frac: float,
    q_lo: float,
    q_hi: float,
    top_n_for_pairs: int,
) -> None:
    side = str(side).upper()
    d = out_dir / side
    d.mkdir(parents=True, exist_ok=True)

    trades = trades.copy()
    trades["trade_entry_time"] = pd.to_datetime(trades["trade_entry_time"], utc=True, errors="coerce")
    trades = trades.sort_values("trade_entry_time").set_index("trade_entry_time")

    # Join full-history bar features at entry time
    joined = trades[["trade_net_return_pct"]].join(bar_feat, how="left")

    # Basic join diagnostics
    join_missing = float(joined.filter(like="bar_").isna().mean().mean())

    y = pd.to_numeric(joined["trade_net_return_pct"], errors="coerce").to_numpy(np.float64)
    feat_cols = [c for c in joined.columns if c.startswith("bar_")]

    # Drop columns with too much missingness (mostly early warmup windows)
    miss_by_col = joined[feat_cols].isna().mean()
    feat_cols = [c for c in feat_cols if float(miss_by_col[c]) <= 0.20]

    # time split
    n = len(joined)
    split = _time_split_idx(n, test_frac)

    y_tr, y_te = y[:split], y[split:]

    # hi/lo labels based on train quantiles
    lab_tr, lo_thr, hi_thr = _make_hi_lo_labels(y_tr, q_lo=q_lo, q_hi=q_hi)

    # apply same thresholds to test
    lab_te = np.full_like(y_te, fill_value=-1, dtype=np.int8)
    lab_te[y_te <= lo_thr] = 0
    lab_te[y_te >= hi_thr] = 1

    # Only extremes
    m_tr = lab_tr >= 0
    m_te = lab_te >= 0

    meta = {
        "side": side,
        "n": int(n),
        "split": int(split),
        "test_frac": float(test_frac),
        "q_lo": float(q_lo),
        "q_hi": float(q_hi),
        "y_train_lo": float(lo_thr),
        "y_train_hi": float(hi_thr),
        "n_train_extremes": int(m_tr.sum()),
        "n_test_extremes": int(m_te.sum()),
        "join_missing_mean": float(join_missing),
        "n_features_after_missing_filter": int(len(feat_cols)),
    }

    (d / f"meta_{out_dir.name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # --- Univariate ---
    uni_rows = []
    for c in feat_cols:
        x = pd.to_numeric(joined[c], errors="coerce").to_numpy(np.float64)
        x_tr, x_te = x[:split], x[split:]

        sp = _spearman(x_tr, y_tr)

        # AUC / direction on test extremes
        auc, direction = _auc_signed(lab_te[m_te].astype(int), x_te[m_te])

        # Balanced accuracy with threshold chosen on train extremes
        thr = _best_threshold_bal_acc(x_tr[m_tr], lab_tr[m_tr].astype(int), direction)
        if np.isfinite(thr):
            if direction > 0:
                pred = (x_te[m_te] >= thr).astype(int)
            else:
                pred = (x_te[m_te] <= thr).astype(int)
            bal_acc = float(balanced_accuracy_score(lab_te[m_te].astype(int), pred)) if m_te.sum() else float("nan")
        else:
            bal_acc = float("nan")

        # Stability: early vs late half of test extremes
        if m_te.sum() >= 40:
            idx = np.where(m_te)[0]
            half = len(idx) // 2
            i0 = idx[:half]
            i1 = idx[half:]
            a0, _ = _auc_signed(lab_te[i0].astype(int), x_te[i0])
            a1, _ = _auc_signed(lab_te[i1].astype(int), x_te[i1])
            stab = float(abs(a0 - a1))
        else:
            stab = float("nan")

        uni_rows.append(
            {
                "feature": c,
                "missing_frac": float(miss_by_col.get(c, np.nan)),
                "spearman_train": float(sp),
                "auc_test_extremes": float(auc),
                "bal_acc_test_extremes": float(bal_acc),
                "direction": int(direction),
                "stability_abs_auc_diff": float(stab),
            }
        )

    uni = pd.DataFrame(uni_rows).sort_values(["auc_test_extremes", "bal_acc_test_extremes"], ascending=False)
    uni_path = d / f"univariate_bar_features_{out_dir.name}.csv"
    uni.to_csv(uni_path, index=False)

    # --- Pairs (logistic regression) ---
    top = uni.dropna(subset=["auc_test_extremes"]).head(int(top_n_for_pairs))
    top_feats = top["feature"].tolist()

    X = joined[top_feats].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    # NaN -> column median
    med = np.nanmedian(X[:split], axis=0)
    ii = np.where(~np.isfinite(X))
    if ii[0].size:
        X[ii] = med[ii[1]]

    # Standardize using train
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xs_tr, Xs_te = Xs[:split], Xs[split:]

    yb_tr = lab_tr.astype(int)
    yb_te = lab_te.astype(int)

    pair_rows = []
    for i in range(len(top_feats)):
        for j in range(i + 1, len(top_feats)):
            f1, f2 = top_feats[i], top_feats[j]
            X2_tr = Xs_tr[m_tr][:, [i, j]]
            X2_te = Xs_te[m_te][:, [i, j]]

            if X2_tr.shape[0] < 30 or X2_te.shape[0] < 30:
                continue

            try:
                clf = LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="liblinear",
                    random_state=42,
                    max_iter=1000,
                )
                clf.fit(X2_tr, yb_tr[m_tr])
                proba = clf.predict_proba(X2_te)[:, 1]
                auc = float(roc_auc_score(yb_te[m_te], proba))
                pred = (proba >= 0.5).astype(int)
                bal_acc = float(balanced_accuracy_score(yb_te[m_te], pred))
            except Exception:
                continue

            pair_rows.append(
                {
                    "f1": f1,
                    "f2": f2,
                    "auc_test_extremes": auc,
                    "bal_acc_test_extremes": bal_acc,
                }
            )

    pairs = pd.DataFrame(pair_rows).sort_values(["auc_test_extremes", "bal_acc_test_extremes"], ascending=False)
    pairs_path = d / f"pairs_logreg_top{len(top_feats)}_{out_dir.name}.csv"
    pairs.to_csv(pairs_path, index=False)

    print(
        f"[{side}] joined={len(joined):,} features={len(feat_cols):,} (top_for_pairs={len(top_feats)}) test_extremes={int(m_te.sum()):,} ",
        flush=True,
    )
    print(f"[{side}] wrote: {uni_path}", flush=True)
    print(f"[{side}] wrote: {pairs_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze cleaned profitable trades using full-history indicators")
    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="Full 1m candles+features parquet",
    )
    ap.add_argument(
        "--buy",
        default=str(_pick_latest("entry_precontext_clean_top10pct_retge0p2_*/BUY_clean_top10pct_retge0p2_*.parquet")),
        help="BUY cleaned profitable parquet",
    )
    ap.add_argument(
        "--sell",
        default=str(_pick_latest("entry_precontext_clean_top10pct_retge0p2_*/SELL_clean_top10pct_retge0p2_*.parquet")),
        help="SELL cleaned profitable parquet",
    )
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--q-lo", type=float, default=0.25)
    ap.add_argument("--q-hi", type=float, default=0.75)
    ap.add_argument("--top-n-for-pairs", type=int, default=30)
    args = ap.parse_args()

    ts = _now_ts()
    out_dir = Path(args.out_dir) / f"analysis_fullhistory_indicators_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading bars: {args.bars}", flush=True)
    bars = pd.read_parquet(args.bars)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    bars = bars.sort_values("timestamp").set_index("timestamp")

    print("Building full-history indicators...", flush=True)
    bar_feat = _build_bar_features(bars)

    # Load trades
    buy = pd.read_parquet(args.buy)
    sell = pd.read_parquet(args.sell)

    run_meta = {
        "created_utc": ts,
        "bars": str(args.bars),
        "buy": str(args.buy),
        "sell": str(args.sell),
        "test_frac": float(args.test_frac),
        "q_lo": float(args.q_lo),
        "q_hi": float(args.q_hi),
        "top_n_for_pairs": int(args.top_n_for_pairs),
    }
    (out_dir / f"run_meta_{out_dir.name}.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    _analyze_side(
        side="BUY",
        trades=buy,
        bar_feat=bar_feat,
        out_dir=out_dir,
        test_frac=float(args.test_frac),
        q_lo=float(args.q_lo),
        q_hi=float(args.q_hi),
        top_n_for_pairs=int(args.top_n_for_pairs),
    )
    _analyze_side(
        side="SELL",
        trades=sell,
        bar_feat=bar_feat,
        out_dir=out_dir,
        test_frac=float(args.test_frac),
        q_lo=float(args.q_lo),
        q_hi=float(args.q_hi),
        top_n_for_pairs=int(args.top_n_for_pairs),
    )

    print("Done.")
    print("Outputs:", out_dir)


if __name__ == "__main__":
    main()
