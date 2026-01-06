#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T11:11:32Z
"""Train an entry regressor from 5-minute precontexts, evaluate on entire ETH database.

Task (as requested)
- Train an entry regressor using 5-minute precontext features.
- Evaluate it on finding entries across the entire ETH 1m dataset.
- An entry is a "win" if within 15 minutes an oracle exit of at least 0.2% net return is possible.
  Higher profitability should score higher.

Definitions
- Entry at timestamp t uses ONLY the prior 5 minutes (t-5..t-1) as features (same convention as earlier pre_entry contexts).
- Entry price uses close[t].
- Oracle 15m label uses closes in t+1..t+15.
- Net return uses the same fee model as the oracle trader:
    BUY: (exit*(1-f))/(entry*(1+f)) - 1
    SELL: (entry*(1-f))/(exit*(1+f)) - 1

Models
- Two separate regressors: BUY and SELL.
- Base model: LightGBM regressor.

Outputs (timestamped)
- data/entry_regressor_oracle15m_<ts>/
  - metrics.json
  - thresholds.csv
  - per_side_metrics.json
  - preds_sample.csv
  - preds_full.parquet (optional; can be big)
  - models/entry_regressor_buy_oracle15m_<ts>.joblib
  - models/entry_regressor_sell_oracle15m_<ts>.joblib

Note
- This trains on ALL minutes (not only oracle-trade entries) so evaluation on the full database is meaningful.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from lightgbm.callback import log_evaluation
from sklearn.metrics import roc_auc_score


FEATURE_COLS_DEFAULT = [
    "ret_1m_pct",
    "mom_3m_pct",
    "mom_5m_pct",
    "vol_std_5m",
    "range_5m",
    "range_norm_5m",
    "macd",
    "vwap_dev_5m",
]


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def net_ret_pct_buy(entry_px: np.ndarray, exit_px: np.ndarray, f: float) -> np.ndarray:
    entry_px = np.asarray(entry_px, dtype=np.float64)
    exit_px = np.asarray(exit_px, dtype=np.float64)
    mult = (exit_px * (1.0 - f)) / (entry_px * (1.0 + f))
    return (mult - 1.0) * 100.0


def net_ret_pct_sell(entry_px: np.ndarray, exit_px: np.ndarray, f: float) -> np.ndarray:
    entry_px = np.asarray(entry_px, dtype=np.float64)
    exit_px = np.asarray(exit_px, dtype=np.float64)
    mult = (entry_px * (1.0 - f)) / (exit_px * (1.0 + f))
    return (mult - 1.0) * 100.0


def _rolling_prev_matrix(x: np.ndarray, L: int) -> np.ndarray:
    """Return mat shape (n, L) with mat[t, i] = x[t-L+i] (previous L values), NaN where unavailable."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    mat = np.full((n, L), np.nan, dtype=np.float64)
    for k in range(1, L + 1):
        # k=1 -> previous 1; stored at last column
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


def _agg_5(x: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
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


def _past_5m_features(df: pd.DataFrame, *, L: int, feat_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Build 5-minute precontext descriptor features for each timestamp (using t-5..t-1).

    Returns (features_df aligned to df rows, feature_names).
    """
    close = df["close"].to_numpy(np.float64, copy=False)
    vol = df["volume"].to_numpy(np.float64, copy=False)

    close_prev = _rolling_prev_matrix(close, L)
    close_last = close_prev[:, -1]

    # close norm % relative to last precontext close (t-1)
    close_norm = (close_prev / np.maximum(1e-12, close_last[:, None]) - 1.0) * 100.0

    out: dict[str, np.ndarray] = {}

    # price descriptors
    out.update(_agg_5(close_norm, "px_close_norm_pct"))
    out["px_close_norm_pct__ret5m"] = close_norm[:, -1] - close_norm[:, 0]
    out["px_close_norm_pct__absret5m"] = np.abs(out["px_close_norm_pct__ret5m"])

    # keep raw 5-d path too
    out["px_close_norm_pct__m5"] = close_norm[:, 0]
    out["px_close_norm_pct__m4"] = close_norm[:, 1]
    out["px_close_norm_pct__m3"] = close_norm[:, 2]
    out["px_close_norm_pct__m2"] = close_norm[:, 3]
    out["px_close_norm_pct__m1"] = close_norm[:, 4]

    # volume
    vol_prev = _rolling_prev_matrix(vol, L)
    vol_log = np.log1p(np.maximum(0.0, vol_prev))
    out.update(_agg_5(vol_log, "vol_log1p"))

    # base feature columns (ret/mom/vol/macd/vwapdev) over last 5 mins
    for c in feat_cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64, copy=False)
        x_prev = _rolling_prev_matrix(x, L)
        out.update(_agg_5(x_prev, c))

    # missingness of close in window
    miss = np.isnan(close_prev)
    miss_n = miss.sum(axis=1).astype(np.float64)
    out["missing_close_n"] = miss_n
    out["missing_any"] = (miss_n > 0).astype(np.float64)

    feat_df = pd.DataFrame(out)

    # downcast floats
    for c in feat_df.columns:
        feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce").astype(np.float32)

    return feat_df, list(feat_df.columns)


def _future_extrema_excl_current(x: np.ndarray, W: int, *, mode: str) -> np.ndarray:
    """For each i, return extreme over x[i+1:i+W+1]. mode in {'max','min'}.

    O(n) via deque on the reversed series.
    """
    from collections import deque

    x = np.asarray(x, dtype=np.float64)
    n = x.size
    b = x[::-1]

    ans_rev = np.full(n, np.nan, dtype=np.float64)
    dq: deque[int] = deque()

    def better(a: float, b: float) -> bool:
        return (a >= b) if mode == "max" else (a <= b)

    for j in range(n):
        # drop indices outside [j-W, j-1]
        while dq and dq[0] < j - W:
            dq.popleft()

        # answer excludes current j
        ans_rev[j] = b[dq[0]] if dq else np.nan

        v = b[j]
        if np.isfinite(v):
            while dq and np.isfinite(b[dq[-1]]) and better(v, b[dq[-1]]):
                dq.pop()
            dq.append(j)
        else:
            # if NaN, just append (it won't dominate)
            dq.append(j)

    return ans_rev[::-1]


@dataclass
class EvalRow:
    name: str
    selected: int
    win_rate: float
    mean_true_ret: float
    median_true_ret: float
    p90_true_ret: float


def _summarize_selection(true_ret: np.ndarray, mask: np.ndarray) -> EvalRow:
    sel = mask.astype(bool)
    tr = true_ret[sel]
    if tr.size == 0:
        return EvalRow(name="", selected=0, win_rate=float("nan"), mean_true_ret=float("nan"), median_true_ret=float("nan"), p90_true_ret=float("nan"))
    win = (tr >= 0.2).astype(np.float64)
    return EvalRow(
        name="",
        selected=int(tr.size),
        win_rate=float(win.mean()),
        mean_true_ret=float(np.mean(tr)),
        median_true_ret=float(np.median(tr)),
        p90_true_ret=float(np.quantile(tr, 0.90)),
    )


def _trades_per_day(ts: pd.Series, mask: np.ndarray) -> dict[str, float]:
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    sel = mask.astype(bool)
    s = pd.Series(sel.astype(int), index=t.dt.date)
    by = s.groupby(level=0).sum()
    return {
        "median": float(by.median()) if len(by) else 0.0,
        "p90": float(by.quantile(0.9)) if len(by) else 0.0,
        "max": float(by.max()) if len(by) else 0.0,
        "mean": float(by.mean()) if len(by) else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/eval entry regressors using 5m precontexts vs oracle 15m exits")
    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="ETH 1m dataset parquet",
    )
    ap.add_argument("--fee-total", type=float, default=0.001, help="TOTAL round-trip fee (0.001=0.1% total)")
    ap.add_argument("--horizon-min", type=int, default=15)
    ap.add_argument("--pre-min", type=int, default=5)
    ap.add_argument("--ret-thresh", type=float, default=0.2)

    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--val-frac", type=float, default=0.10, help="Validation taken from the tail of train")
    ap.add_argument("--max-train-rows", type=int, default=0, help="If >0, subsample training rows (chronological)")
    ap.add_argument("--side", default="BUY", choices=["BUY", "SELL", "BOTH", "buy", "sell", "both"], help="Train/eval side(s); default BUY")
    ap.add_argument("--lgb-log-period", type=int, default=50, help="LightGBM log period (0 disables)")

    ap.add_argument(
        "--feature-keep-root",
        default="data/eth_pruned_coinflip_2026-01-04T11-02-48Z",
        help="Folder with entry_keep_features_BUY_*.json and entry_keep_features_SELL_*.json (optional)",
    )
    ap.add_argument("--no-keep-lists", action="store_true", help="Ignore keep lists; use all computed features")

    ap.add_argument("--save-full-preds", action="store_true", help="Write preds_full.parquet (can be big)")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    ts = now_ts()
    out_root = Path(args.out_dir) / f"entry_regressor_oracle15m_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    models_dir = out_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    side_arg = str(args.side).upper()
    sides = ["BUY", "SELL"] if side_arg == "BOTH" else [side_arg]

    print(f"[1/6] Loading bars: {args.bars}", flush=True)
    df = pd.read_parquet(args.bars)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"[1/6] Bars loaded: rows={len(df):,}  start={df['timestamp'].iloc[0]}  end={df['timestamp'].iloc[-1]}", flush=True)

    feat_cols = [c for c in FEATURE_COLS_DEFAULT if c in df.columns]

    print(f"[2/6] Building {int(args.pre_min)}-minute precontext features (t-5..t-1) ...", flush=True)
    pre_L = int(args.pre_min)
    X_df, X_names = _past_5m_features(df, L=pre_L, feat_cols=feat_cols)
    print(f"[2/6] Precontext features built: n_features={len(X_names):,}", flush=True)

    print(f"[3/6] Building oracle {int(args.horizon_min)}m best-exit labels ...", flush=True)
    close = df["close"].to_numpy(np.float64, copy=False)
    W = int(args.horizon_min)
    fut_max = _future_extrema_excl_current(close, W, mode="max")
    fut_min = _future_extrema_excl_current(close, W, mode="min")

    per_side_fee = float(args.fee_total) / 2.0

    y_buy = net_ret_pct_buy(close, fut_max, per_side_fee)
    y_sell = net_ret_pct_sell(close, fut_min, per_side_fee)

    # valid labels: need future window available and precontext available
    valid = (np.arange(len(df)) >= pre_L) & np.isfinite(y_buy)
    if "SELL" in sides:
        valid &= np.isfinite(y_sell)

    print(f"[3/6] Labels ready. valid_rows={int(valid.sum()):,}/{len(df):,}", flush=True)

    # drop last horizon window (where fut_max/min is NaN)
    # already covered by finite check.

    # feature keep lists
    keep = {"BUY": None, "SELL": None}
    if not bool(args.no_keep_lists):
        keep_root = Path(args.feature_keep_root)
        buy_files = sorted(keep_root.glob("entry_keep_features_BUY_*.json"))
        sell_files = sorted(keep_root.glob("entry_keep_features_SELL_*.json"))
        if buy_files and sell_files:
            keep["BUY"] = json.loads(buy_files[-1].read_text(encoding="utf-8"))
            keep["SELL"] = json.loads(sell_files[-1].read_text(encoding="utf-8"))

    def select_cols(side: str) -> list[str]:
        if keep[side] is None:
            return list(X_names)
        # intersection
        s = set(keep[side])
        return [c for c in X_names if c in s]

    # split indices
    n = int(df.shape[0])
    test_frac = float(args.test_frac)
    split_test = int(n * (1.0 - test_frac))

    idx_all = np.arange(n)
    idx_train_all = idx_all[:split_test]
    idx_test_all = idx_all[split_test:]

    # restrict to valid
    train_mask = valid & (idx_all < split_test)
    test_mask = valid & (idx_all >= split_test)

    # optional subsample training
    train_idx = np.where(train_mask)[0]
    if int(args.max_train_rows) > 0 and train_idx.size > int(args.max_train_rows):
        train_idx = train_idx[-int(args.max_train_rows) :]  # keep most recent training slice

    test_idx = np.where(test_mask)[0]

    # validation from tail of training
    val_frac = float(args.val_frac)
    n_tr = int(train_idx.size)
    n_val = int(n_tr * val_frac)
    if n_val < 1000:
        n_val = min(5000, n_tr // 5) if n_tr >= 5000 else max(0, n_tr // 10)

    if n_val <= 0:
        raise SystemExit("Not enough training data for validation split")

    tr_idx = train_idx[: n_tr - n_val]
    va_idx = train_idx[n_tr - n_val :]

    run_meta = {
        "created_utc": ts,
        "bars": str(args.bars),
        "fee_total": float(args.fee_total),
        "per_side_fee": float(per_side_fee),
        "horizon_min": int(args.horizon_min),
        "pre_min": int(args.pre_min),
        "ret_thresh": float(args.ret_thresh),
        "test_frac": float(args.test_frac),
        "val_frac": float(args.val_frac),
        "max_train_rows": int(args.max_train_rows),
        "use_keep_lists": not bool(args.no_keep_lists),
        "feature_keep_root": str(args.feature_keep_root),
        "n_total": int(n),
        "n_valid": int(valid.sum()),
        "n_train": int(tr_idx.size),
        "n_val": int(va_idx.size),
        "n_test": int(test_idx.size),
    }
    (out_root / f"run_meta_{ts}.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    results = {}

    def train_side(side: str, y: np.ndarray) -> tuple[LGBMRegressor, np.ndarray, dict]:
        cols = select_cols(side)
        X = X_df[cols].to_numpy(np.float32, copy=False)

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        print(
            f"[4/6] Training {side}: rows(train)={len(y_tr):,} rows(val)={len(y_va):,} n_features={len(cols):,} n_estimators=500",
            flush=True,
        )

        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=256,
            max_depth=-1,
            min_child_samples=300,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        callbacks = []
        if int(args.lgb_log_period) > 0:
            callbacks.append(log_evaluation(period=int(args.lgb_log_period)))

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="l2",
            callbacks=callbacks,
        )

        # preds for all valid rows (for evaluation)
        pred_all = np.full(n, np.nan, dtype=np.float64)
        pred_all[valid] = model.predict(X[valid])

        # metrics on test
        y_test = y[test_idx]
        p_test = pred_all[test_idx]
        win_test = (y_test >= float(args.ret_thresh)).astype(int)
        auc = float("nan")
        if len(np.unique(win_test)) == 2:
            auc = float(roc_auc_score(win_test, p_test))

        side_meta = {
            "side": side,
            "n_features": int(len(cols)),
            "auc_test_win": float(auc),
            "pred_test_mean": float(np.mean(p_test)),
            "pred_test_p90": float(np.quantile(p_test, 0.9)),
            "y_test_mean": float(np.mean(y_test)),
            "y_test_p90": float(np.quantile(y_test, 0.9)),
        }

        return model, pred_all, side_meta

    print(f"[4/6] Training sides: {sides}", flush=True)

    models: dict[str, LGBMRegressor] = {}
    preds: dict[str, np.ndarray] = {}
    metas: dict[str, dict] = {}

    for side in sides:
        y = y_buy if side == "BUY" else y_sell
        model, pred, meta = train_side(side, y)
        models[side] = model
        preds[side] = pred
        metas[side] = meta

        joblib.dump(
            {
                "created_utc": ts,
                "side": side,
                "feature_cols": select_cols(side),
                "horizon_min": int(args.horizon_min),
                "pre_min": int(args.pre_min),
                "fee_total": float(args.fee_total),
                "ret_thresh": float(args.ret_thresh),
                "model": model,
                "metrics": meta,
            },
            models_dir / f"entry_regressor_{side.lower()}_oracle15m_{ts}.joblib",
        )

    print("[5/6] Scoring selection on test split...", flush=True)

    if len(sides) == 1:
        side = sides[0]
        chosen_side = np.full(n, side, dtype=object)
        chosen_pred = preds[side]
        chosen_true = y_buy if side == "BUY" else y_sell
        best_true = chosen_true.copy()
    else:
        pred_buy = preds["BUY"]
        pred_sell = preds["SELL"]
        chosen_side = np.where(pred_buy >= pred_sell, "BUY", "SELL")
        chosen_pred = np.where(pred_buy >= pred_sell, pred_buy, pred_sell)
        chosen_true = np.where(pred_buy >= pred_sell, y_buy, y_sell)
        best_true = np.maximum(y_buy, y_sell)

    # Evaluate thresholds on test window
    test_sel = np.where(test_mask)[0]
    ct = chosen_true[test_sel]
    cp = chosen_pred[test_sel]

    thresholds = []

    # absolute thresholds
    for thr in [0.2, 0.25, 0.3, 0.4, 0.5]:
        m = test_mask & (chosen_pred >= thr)
        row = _summarize_selection(chosen_true, m)
        thresholds.append(
            {
                "rule": f"pred>= {thr:.2f}",
                "selected": row.selected,
                "win_rate": row.win_rate,
                "mean_true_ret": row.mean_true_ret,
                "median_true_ret": row.median_true_ret,
                "p90_true_ret": row.p90_true_ret,
                "trades_per_day_mean": _trades_per_day(df["timestamp"], m)["mean"],
            }
        )

    # top-k% thresholds (on test)
    for q in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
        thr = float(np.nanquantile(cp, 1.0 - q))
        m = test_mask & (chosen_pred >= thr)
        row = _summarize_selection(chosen_true, m)
        thresholds.append(
            {
                "rule": f"top_{q*100:.1f}% (thr={thr:.4f})",
                "selected": row.selected,
                "win_rate": row.win_rate,
                "mean_true_ret": row.mean_true_ret,
                "median_true_ret": row.median_true_ret,
                "p90_true_ret": row.p90_true_ret,
                "trades_per_day_mean": _trades_per_day(df["timestamp"], m)["mean"],
            }
        )

    thresholds_df = pd.DataFrame(thresholds)
    thresholds_df.to_csv(out_root / f"thresholds_{ts}.csv", index=False)

    # headline metrics
    win_test = (ct >= float(args.ret_thresh)).astype(int)
    auc_comb = float("nan")
    if len(np.unique(win_test)) == 2:
        auc_comb = float(roc_auc_score(win_test, cp))

    metrics = {**metas}
    metrics["SELECTION"] = {
        "sides_trained": sides,
        "auc_test_win": float(auc_comb),
        "test_mean_true_ret_chosen": float(np.mean(ct)),
        "test_mean_true_ret_best_possible": float(np.mean(best_true[test_sel])),
        "test_win_rate_chosen": float(np.mean(ct >= float(args.ret_thresh))),
        "test_win_rate_best_possible": float(np.mean(best_true[test_sel] >= float(args.ret_thresh))),
    }

    (out_root / f"metrics_{ts}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # write predictions sample
    sample = pd.DataFrame({"timestamp": df["timestamp"], "chosen_side": chosen_side, "chosen_pred": chosen_pred, "chosen_true": chosen_true})
    if "BUY" in sides:
        sample["y_buy_best15m"] = y_buy
        sample["pred_buy"] = preds.get("BUY")
    if "SELL" in sides:
        sample["y_sell_best15m"] = y_sell
        sample["pred_sell"] = preds.get("SELL")

    sample.loc[test_mask].sample(n=min(20000, int(test_mask.sum())), random_state=42).to_csv(out_root / f"preds_sample_{ts}.csv", index=False)

    if bool(args.save_full_preds):
        sample.to_parquet(out_root / f"preds_full_{ts}.parquet", index=False)

    print("[6/6] Done.", flush=True)
    print("Outputs:", out_root, flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
