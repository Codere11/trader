#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T14:38:54Z
"""Build an ETH-USD (dYdX) oracle-exit dataset with 5-minute precontext features.

Intent
- Use the current best ETH-USD BUY entry regressor to select entries around ~3 trades/day.
- For each entry, compute the oracle best exit within a fixed horizon (hold_min).
- Generate per-minute in-trade rows (k_rel=1..hold_min) with:
  - realized ret_if_exit_now_pct at that minute's close
  - oracle_k / oracle_ret_pct
  - 5-minute precontext features at the decision minute (t-5..t-1)
  - simple in-trade state features (mins_in_trade, drawdown_from_peak, etc.)

Output
- Parquet dataset under data/exit_oracle/
  exit_oracle_dataset_ETH-USD_pre5m_hold<hold>_thr<thr>_<ts>.parquet
- Small CSV sample + run_meta.json

Notes
- Entry decision at close[i]; entry at open[i+1].
- Exit decision at close[entry_idx + k].
- Fees: fee_total is round-trip; per-side fee = fee_total/2.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent

BASE_FEATS_DEFAULT = [
    "ret_1m_pct",
    "mom_3m_pct",
    "mom_5m_pct",
    "vol_std_5m",
    "range_5m",
    "range_norm_5m",
    "macd",
    "vwap_dev_5m",
]

CORE_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def net_ret_pct_long(entry_px: float, exit_px: float, fee_side: float) -> float:
    entry_px = float(entry_px)
    exit_px = float(exit_px)
    f = float(fee_side)
    if not (math.isfinite(entry_px) and math.isfinite(exit_px)):
        return float("nan")
    if entry_px <= 0.0 or exit_px <= 0.0:
        return float("nan")
    mult = (exit_px * (1.0 - f)) / (entry_px * (1.0 + f))
    return float((mult - 1.0) * 100.0)


def _rolling_prev_matrix(x: np.ndarray, L: int) -> np.ndarray:
    """Return mat shape (n, L) with mat[t, i] = x[t-L+i] (previous L values), NaN where unavailable."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    mat = np.full((n, L), np.nan, dtype=np.float64)
    for k in range(1, L + 1):
        mat[k:, L - k] = x[: n - k]
    return mat


def _slope_5(y: np.ndarray) -> np.ndarray:
    """Slope over 5 points with x=[-5,-4,-3,-2,-1] (centered [-2,-1,0,1,2])."""
    xc = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    denom = 10.0
    return np.nansum(y * xc[None, :], axis=1) / denom


def _safe_nanstd(x: np.ndarray) -> np.ndarray:
    import warnings

    out = np.full(x.shape[0], np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m = np.nanmean(x, axis=1)
        ok = np.isfinite(m)
        if np.any(ok):
            out[ok] = np.nanmean((x[ok] - m[ok, None]) ** 2, axis=1) ** 0.5
    return out


def _agg_5(x: np.ndarray, prefix: str) -> Dict[str, np.ndarray]:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean5 = np.nanmean(x, axis=1)
        min5 = np.nanmin(x, axis=1)
        max5 = np.nanmax(x, axis=1)

    return {
        f"{prefix}__last": x[:, -1],
        f"{prefix}__mean5": mean5,
        f"{prefix}__std5": _safe_nanstd(x),
        f"{prefix}__min5": min5,
        f"{prefix}__max5": max5,
        f"{prefix}__range5": max5 - min5,
        f"{prefix}__slope5": _slope_5(x),
    }


def build_pre5m_feature_matrix(
    df: pd.DataFrame,
    *,
    pre_min: int,
    base_feat_cols: List[str],
    out_feature_cols: List[str],
) -> np.ndarray:
    """Build pre-5m feature matrix for *all* rows, then select requested columns.

    Features follow the same convention as scripts/train_eval_entry_regressor_oracle15m_2026-01-04T11-11-32Z.py:
    each row t uses ONLY the previous pre_min minutes (t-pre_min .. t-1).
    """
    L = int(pre_min)
    if L != 5:
        raise SystemExit("This script currently supports --pre-min=5 only")

    n = len(df)
    col_to_j = {c: j for j, c in enumerate(out_feature_cols)}

    X = np.full((n, len(out_feature_cols)), np.nan, dtype=np.float32)

    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64, copy=False)
    vol = pd.to_numeric(df["volume"], errors="coerce").to_numpy(np.float64, copy=False)

    # ---- close-normalized window ----
    close_prev = _rolling_prev_matrix(close, L)
    close_last = close_prev[:, -1]
    close_last = np.where((np.isfinite(close_last)) & (close_last > 0.0), close_last, np.nan)
    close_norm = (close_prev / close_last[:, None] - 1.0) * 100.0

    px = _agg_5(close_norm, "px_close_norm_pct")
    # extras
    px_ret5m = close_norm[:, -1] - close_norm[:, 0]
    px_absret5m = np.abs(px_ret5m)

    def _set(name: str, arr: np.ndarray) -> None:
        j = col_to_j.get(name)
        if j is None:
            return
        X[:, j] = np.asarray(arr, dtype=np.float32)

    for k, v in px.items():
        _set(k, v)
    _set("px_close_norm_pct__ret5m", px_ret5m)
    _set("px_close_norm_pct__absret5m", px_absret5m)
    _set("px_close_norm_pct__m5", close_norm[:, 0])
    _set("px_close_norm_pct__m4", close_norm[:, 1])
    _set("px_close_norm_pct__m3", close_norm[:, 2])
    _set("px_close_norm_pct__m2", close_norm[:, 3])
    _set("px_close_norm_pct__m1", close_norm[:, 4])

    miss = np.isnan(close_prev)
    miss_n = miss.sum(axis=1).astype(np.float64)
    _set("missing_close_n", miss_n)
    _set("missing_any", (miss_n > 0).astype(np.float64))

    # ---- volume log1p window ----
    vol_prev = _rolling_prev_matrix(vol, L)
    vol_log = np.log1p(np.maximum(0.0, vol_prev))
    vv = _agg_5(vol_log, "vol_log1p")
    for k, v in vv.items():
        _set(k, v)

    # ---- base feature windows ----
    for c in base_feat_cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64, copy=False)
        x_prev = _rolling_prev_matrix(x, L)
        a = _agg_5(x_prev, c)
        for k, v in a.items():
            _set(k, v)

    return X


@dataclass(frozen=True)
class Trade:
    trade_id: int
    signal_i: int
    entry_idx: int
    entry_time: pd.Timestamp
    entry_px: float
    entry_pred: float


def select_trades_fixed_hold(
    *,
    ts_utc: pd.Series,
    open_arr: np.ndarray,
    pred_by_i: np.ndarray,
    entry_threshold: float,
    hold_min: int,
    pre_min: int,
    max_trades: int,
) -> List[Trade]:
    n = len(ts_utc)
    last_signal_i = n - 2 - int(hold_min)

    trades: List[Trade] = []
    blocked_until_i = -10**18
    trade_id = 0

    start_i = max(0, int(pre_min))

    for i in range(int(start_i), int(last_signal_i) + 1):
        if i < int(blocked_until_i):
            continue

        p = float(pred_by_i[i])
        if not np.isfinite(p):
            continue
        if p < float(entry_threshold):
            continue

        entry_idx = int(i) + 1
        entry_px = float(open_arr[entry_idx])
        if not (np.isfinite(entry_px) and entry_px > 0.0):
            continue

        trades.append(
            Trade(
                trade_id=int(trade_id),
                signal_i=int(i),
                entry_idx=int(entry_idx),
                entry_time=pd.to_datetime(ts_utc.iloc[entry_idx], utc=True),
                entry_px=float(entry_px),
                entry_pred=float(p),
            )
        )
        trade_id += 1

        blocked_until_i = entry_idx + int(hold_min)
        if max_trades > 0 and len(trades) >= int(max_trades):
            break

    return trades


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ETH-USD exit oracle dataset with 5m precontext features")
    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="ETH 1m OHLCV+features parquet",
    )
    ap.add_argument(
        "--entry-model",
        default="data/entry_regressor_oracle15m_2026-01-04T11-28-38Z/models/entry_regressor_buy_oracle15m_2026-01-04T11-28-38Z.joblib",
        help="BUY entry regressor joblib artifact (dict with {model, feature_cols, ...})",
    )
    ap.add_argument("--entry-threshold", type=float, default=0.40, help="Fixed threshold on entry model prediction")
    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--pre-min", type=int, default=5)
    ap.add_argument("--fee-total", type=float, default=0.001, help="TOTAL round-trip fee (0.001=0.1% total)")
    ap.add_argument("--symbol", default="ETH-USD")
    ap.add_argument("--out-dir", default="data/exit_oracle")
    ap.add_argument("--max-trades", type=int, default=0, help="If >0, cap number of trades (debug)")
    ap.add_argument("--sample-csv-rows", type=int, default=50_000)

    args = ap.parse_args()

    bars_path = Path(args.bars)
    if not bars_path.exists():
        raise SystemExit(f"bars not found: {bars_path}")

    entry_art = joblib.load(args.entry_model)
    if not isinstance(entry_art, dict) or "model" not in entry_art or "feature_cols" not in entry_art:
        raise SystemExit("entry model artifact must be a dict with keys: model, feature_cols")

    model = entry_art["model"]
    feature_cols = list(entry_art["feature_cols"])
    if int(entry_art.get("pre_min", args.pre_min)) != int(args.pre_min):
        raise SystemExit(f"entry model pre_min={entry_art.get('pre_min')} but --pre-min={args.pre_min}")

    hold_min = int(args.hold_min)
    pre_min = int(args.pre_min)
    fee_total = float(args.fee_total)
    fee_side = float(fee_total) / 2.0

    need_cols = CORE_COLS + list(BASE_FEATS_DEFAULT)

    print(f"Loading bars: {bars_path}", flush=True)
    t0 = time.time()
    df = pd.read_parquet(bars_path, columns=need_cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if not df["timestamp"].is_monotonic_increasing:
        df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    n = len(df)
    print(f"Bars: {n:,}  range=[{df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}]  loaded in {(time.time()-t0):.1f}s")

    ts_utc = df["timestamp"]
    open_arr = pd.to_numeric(df["open"], errors="coerce").to_numpy(np.float64, copy=False)
    close_arr = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64, copy=False)

    print(f"Building precontext features for ALL minutes (n={n:,}, cols={len(feature_cols)})...", flush=True)
    t1 = time.time()
    X_all = build_pre5m_feature_matrix(
        df,
        pre_min=pre_min,
        base_feat_cols=list(BASE_FEATS_DEFAULT),
        out_feature_cols=feature_cols,
    )
    print(f"Built feature matrix in {(time.time()-t1):.1f}s", flush=True)

    # Score entry model across all minutes (NaN features are OK; we ignore non-finite preds).
    print("Scoring entry model across all minutes...", flush=True)
    pred = np.full((n,), np.nan, dtype=np.float32)

    # Only score indices where we can still complete the horizon.
    last_signal_i = n - 2 - hold_min
    idxs = np.arange(int(pre_min), int(last_signal_i) + 1, dtype=np.int64)

    # Predict in chunks to control peak memory.
    chunk = 200_000
    for s in range(0, idxs.size, chunk):
        e = min(idxs.size, s + chunk)
        ii = idxs[s:e]
        pred[ii] = model.predict(X_all[ii]).astype(np.float32)
        if (s // chunk) % 5 == 0:
            print(f"  scored {e:,}/{idxs.size:,}", flush=True)

    print("Selecting trades (fixed hold non-overlap)...", flush=True)
    trades = select_trades_fixed_hold(
        ts_utc=ts_utc,
        open_arr=open_arr,
        pred_by_i=pred,
        entry_threshold=float(args.entry_threshold),
        hold_min=hold_min,
        pre_min=pre_min,
        max_trades=int(args.max_trades),
    )

    if not trades:
        raise SystemExit("No trades selected. Try lowering --entry-threshold.")

    print(f"Trades selected: {len(trades):,}", flush=True)

    # Build per-minute rows.
    rows: List[Dict[str, object]] = []

    for tr in trades:
        entry_idx = int(tr.entry_idx)
        entry_px = float(tr.entry_px)

        # returns at each minute in trade
        rets = []
        for k in range(1, hold_min + 1):
            idx = entry_idx + k
            rets.append(net_ret_pct_long(entry_px, float(close_arr[idx]), fee_side))
        rets_arr = np.asarray(rets, dtype=np.float64)

        if not np.isfinite(rets_arr).any():
            continue

        oracle_k = int(np.nanargmax(rets_arr) + 1)
        oracle_ret = float(np.nanmax(rets_arr))

        peak = -1e18
        for k in range(1, hold_min + 1):
            idx = entry_idx + k
            cur_ret = float(rets_arr[k - 1])
            peak = max(peak, cur_ret)

            r_prev1 = float(rets_arr[k - 2]) if k >= 2 else float("nan")
            r_prev2 = float(rets_arr[k - 3]) if k >= 3 else float("nan")
            r_prev3 = float(rets_arr[k - 4]) if k >= 4 else float("nan")

            row: Dict[str, object] = {
                "trade_id": int(tr.trade_id),
                "signal_i": int(tr.signal_i),
                "entry_idx": int(entry_idx),
                "entry_time": tr.entry_time,
                "entry_px": float(entry_px),
                "entry_pred": float(tr.entry_pred),
                "entry_threshold": float(args.entry_threshold),
                "oracle_k": int(oracle_k),
                "oracle_ret_pct": float(oracle_ret),
                "k_rel": int(k),
                "decision_idx": int(idx),
                "decision_time": pd.to_datetime(ts_utc.iloc[idx], utc=True),
                "y_oracle_exit": int(1 if int(k) == int(oracle_k) else 0),
                "ret_if_exit_now_pct": float(cur_ret),
                "mins_in_trade": int(k),
                "mins_remaining": int(hold_min - k),
                "delta_mark_pct": float(cur_ret),
                "delta_mark_prev1_pct": r_prev1,
                "delta_mark_prev2_pct": r_prev2,
                "delta_mark_change_1m": (cur_ret - r_prev1) if np.isfinite(r_prev1) else float("nan"),
                "delta_mark_change_2m": (cur_ret - r_prev2) if np.isfinite(r_prev2) else float("nan"),
                "delta_mark_change_3m": (cur_ret - r_prev3) if np.isfinite(r_prev3) else float("nan"),
                # <=0 by construction
                "drawdown_from_peak_pct": float(cur_ret - peak),
            }

            # Attach 5m precontext features at the DECISION minute.
            # This row uses t-5..t-1 only, consistent with entry model.
            feat_row = X_all[idx]
            for j, name in enumerate(feature_cols):
                row[name] = float(feat_row[j]) if np.isfinite(float(feat_row[j])) else float("nan")

            rows.append(row)

    ds = pd.DataFrame(rows)
    if ds.empty:
        raise SystemExit("Built empty dataset; check inputs.")

    # Basic sanity
    pos_rate = float(ds["y_oracle_exit"].mean() * 100.0)
    print(f"Dataset rows: {len(ds):,}  trades: {ds['trade_id'].nunique():,}  pos_rate={pos_rate:.3f}%", flush=True)

    out_dir = REPO_ROOT / str(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = now_ts()

    thr_tag = str(float(args.entry_threshold)).replace(".", "p")
    sym = str(args.symbol)

    out_path = out_dir / f"exit_oracle_dataset_{sym}_pre5m_hold{hold_min}_thr{thr_tag}_{ts}.parquet"
    ds.to_parquet(out_path, index=False)

    sample_n = max(0, int(args.sample_csv_rows))
    if sample_n > 0:
        sample_path = out_dir / f"exit_oracle_dataset_{sym}_pre5m_hold{hold_min}_thr{thr_tag}_{ts}.sample.csv"
        ds.sample(n=min(sample_n, len(ds)), random_state=42).to_csv(sample_path, index=False)
    else:
        sample_path = None

    # Distribution of oracle_k
    ok_dist = ds.loc[ds["y_oracle_exit"] == 1, "oracle_k"].value_counts().sort_index().to_dict()

    run_meta = {
        "created_utc": ts,
        "symbol": sym,
        "bars": str(bars_path),
        "entry_model": str(Path(args.entry_model)),
        "entry_threshold": float(args.entry_threshold),
        "hold_min": int(hold_min),
        "pre_min": int(pre_min),
        "fee_total": float(fee_total),
        "fee_side": float(fee_side),
        "n_bars": int(n),
        "n_trades": int(ds["trade_id"].nunique()),
        "n_rows": int(len(ds)),
        "pos_rate_pct": float(pos_rate),
        "oracle_k_dist": ok_dist,
        "feature_cols": feature_cols,
        "columns": list(ds.columns),
    }

    meta_path = out_dir / f"exit_oracle_dataset_{sym}_pre5m_hold{hold_min}_thr{thr_tag}_{ts}.run_meta.json"
    meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(f"Saved dataset: {out_path}")
    if sample_path is not None:
        print(f"Saved sample:  {sample_path}")
    print(f"Saved meta:    {meta_path}")


if __name__ == "__main__":
    main()
