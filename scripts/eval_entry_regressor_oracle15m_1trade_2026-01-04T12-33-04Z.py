#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T12:33:04Z
"""Evaluate a trained entry regressor with a 1-trade-at-a-time simulator (no retraining).

What this does
- Loads a saved BUY entry regressor artifact (.joblib) produced by train_eval_entry_regressor_oracle15m_*.py
- Recomputes the same 5-minute precontext features over the full ETH 1m dataset
- Runs inference to get pred[t]
- Computes the oracle best-exit within horizon (default 15m) and the corresponding net return
- Evaluates 1-trade-at-a-time execution on the TEST split by default:
  - Take an entry when pred[t] >= threshold
  - Exit at the best oracle exit within the next horizon minutes (max close for BUY)
  - Next trade can only start after exit (no overlap)

Outputs (timestamped)
- data/entry_regressor_oracle15m_1trade_<ts>/
  - one_trade_thresholds_<ts>.csv
  - one_trade_trades_sample_<ts>.csv
  - meta_<ts>.json

Notes
- "Win" means realized oracle-exit net_return_pct >= ret_thresh (default 0.2).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def net_ret_pct_buy(entry_px: float, exit_px: float, per_side_fee: float) -> float:
    f = float(per_side_fee)
    mult = (float(exit_px) * (1.0 - f)) / (float(entry_px) * (1.0 + f))
    return (mult - 1.0) * 100.0


def _rolling_prev_matrix(x: np.ndarray, L: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    mat = np.full((n, L), np.nan, dtype=np.float64)
    for k in range(1, L + 1):
        mat[k:, L - k] = x[: n - k]
    return mat


def _slope_5(y: np.ndarray) -> np.ndarray:
    xc = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    return np.nansum(y * xc[None, :], axis=1) / 10.0


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


def build_5m_precontext_feature_arrays(
    *,
    df: pd.DataFrame,
    pre_min: int,
    base_feat_cols: list[str],
) -> dict[str, np.ndarray]:
    """Rebuild the same 5m descriptor features as the training script, but return numpy arrays."""
    L = int(pre_min)

    close = df["close"].to_numpy(np.float64, copy=False)
    vol = df["volume"].to_numpy(np.float64, copy=False)

    close_prev = _rolling_prev_matrix(close, L)
    close_last = close_prev[:, -1]
    close_norm = (close_prev / np.maximum(1e-12, close_last[:, None]) - 1.0) * 100.0

    feat: dict[str, np.ndarray] = {}

    feat.update(_agg_5(close_norm, "px_close_norm_pct"))
    feat["px_close_norm_pct__ret5m"] = close_norm[:, -1] - close_norm[:, 0]
    feat["px_close_norm_pct__absret5m"] = np.abs(feat["px_close_norm_pct__ret5m"])

    feat["px_close_norm_pct__m5"] = close_norm[:, 0]
    feat["px_close_norm_pct__m4"] = close_norm[:, 1]
    feat["px_close_norm_pct__m3"] = close_norm[:, 2]
    feat["px_close_norm_pct__m2"] = close_norm[:, 3]
    feat["px_close_norm_pct__m1"] = close_norm[:, 4]

    vol_prev = _rolling_prev_matrix(vol, L)
    vol_log = np.log1p(np.maximum(0.0, vol_prev))
    feat.update(_agg_5(vol_log, "vol_log1p"))

    for c in base_feat_cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64, copy=False)
        x_prev = _rolling_prev_matrix(x, L)
        feat.update(_agg_5(x_prev, c))

    miss = np.isnan(close_prev)
    miss_n = miss.sum(axis=1).astype(np.float64)
    feat["missing_close_n"] = miss_n
    feat["missing_any"] = (miss_n > 0).astype(np.float64)

    # downcast
    for k, v in list(feat.items()):
        feat[k] = np.asarray(v, dtype=np.float32)

    return feat


def best_oracle_exit_within_horizon_buy(
    *,
    close: np.ndarray,
    entry_i: int,
    horizon_min: int,
) -> int:
    """Return the exit index in (entry_i+1 .. entry_i+horizon_min] maximizing close."""
    n = int(close.size)
    lo = int(entry_i + 1)
    hi = int(min(n, entry_i + int(horizon_min) + 1))
    if lo >= hi:
        return int(entry_i)
    w = close[lo:hi]
    j = int(np.nanargmax(w))  # assumes price series is clean
    return int(lo + j)


def simulate_one_trade_at_a_time_buy(
    *,
    timestamps: np.ndarray,
    close: np.ndarray,
    pred: np.ndarray,
    valid: np.ndarray,
    start_i: int,
    end_i: int,
    per_side_fee: float,
    horizon_min: int,
    ret_thresh: float,
    pred_threshold: float,
) -> dict:
    """Simulate 1-at-a-time trades using BUY oracle exit."""
    n = int(close.size)
    start_i = int(max(0, start_i))
    end_i = int(min(n, end_i))

    mask = (np.arange(n) >= start_i) & (np.arange(n) < end_i) & valid & np.isfinite(pred) & (pred >= float(pred_threshold))
    entries = np.where(mask)[0]

    trades = []
    pos = 0
    cur = start_i

    while pos < entries.size:
        i = int(entries[pos])
        if i < cur:
            pos += 1
            continue

        exit_i = best_oracle_exit_within_horizon_buy(close=close, entry_i=i, horizon_min=int(horizon_min))
        if exit_i <= i:
            pos += 1
            continue

        r = net_ret_pct_buy(close[i], close[exit_i], per_side_fee)
        trades.append(
            {
                "entry_i": i,
                "exit_i": int(exit_i),
                "entry_time": pd.Timestamp(timestamps[i]).tz_localize("UTC") if timestamps.dtype.kind == "M" else timestamps[i],
                "exit_time": pd.Timestamp(timestamps[exit_i]).tz_localize("UTC") if timestamps.dtype.kind == "M" else timestamps[exit_i],
                "duration_min": int(exit_i - i),
                "pred": float(pred[i]),
                "true_ret_pct": float(r),
                "win": int(r >= float(ret_thresh)),
            }
        )

        cur = int(exit_i + 1)
        pos = int(np.searchsorted(entries, cur, side="left"))

    if trades:
        tr = np.array([t["true_ret_pct"] for t in trades], dtype=np.float64)
        win = np.array([t["win"] for t in trades], dtype=np.float64)
        out = {
            "selected": int(len(trades)),
            "win_rate": float(win.mean()),
            "mean_true_ret": float(tr.mean()),
            "median_true_ret": float(np.median(tr)),
            "p90_true_ret": float(np.quantile(tr, 0.90)),
            "mean_duration_min": float(np.mean([t["duration_min"] for t in trades])),
        }
    else:
        out = {
            "selected": 0,
            "win_rate": float("nan"),
            "mean_true_ret": float("nan"),
            "median_true_ret": float("nan"),
            "p90_true_ret": float("nan"),
            "mean_duration_min": float("nan"),
        }

    out["trades"] = trades
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate saved entry regressor via 1-trade-at-a-time oracle-15m sim")
    ap.add_argument(
        "--model",
        default="",
        help="Path to entry_regressor_buy_oracle15m_*.joblib (if empty, auto-pick latest)",
    )
    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="ETH 1m dataset parquet",
    )
    ap.add_argument("--side", default="BUY", choices=["BUY"], help="Only BUY supported in this quick evaluator")
    ap.add_argument("--on", default="test", choices=["test", "full"], help="Evaluate on test split or full dataset")
    ap.add_argument("--test-frac", type=float, default=0.20, help="Test split fraction (must match training for apples-to-apples)")
    ap.add_argument(
        "--export-top-frac",
        type=float,
        default=0.0,
        help="If >0, export full trade list for the top-X%% rule (e.g. 0.05 for top 5%%) as parquet",
    )

    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    ts = now_ts()
    out_root = Path(args.out_dir) / f"entry_regressor_oracle15m_1trade_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    # pick model
    if args.model:
        model_path = Path(args.model)
    else:
        c = sorted(Path("data").glob("entry_regressor_oracle15m_*/models/entry_regressor_buy_oracle15m_*.joblib"))
        if not c:
            c = sorted(Path("data").glob("entry_regressor_oracle15m_*/entry_regressor_buy_oracle15m_*.joblib"))
        if not c:
            raise SystemExit("Could not auto-find a BUY model joblib under data/entry_regressor_oracle15m_*/")
        model_path = c[-1]

    # Allow passing a run directory; look for models/*.joblib.
    if model_path.is_dir():
        c = sorted(model_path.glob("models/entry_regressor_buy_oracle15m_*.joblib"))
        if not c:
            c = sorted(model_path.glob("entry_regressor_buy_oracle15m_*.joblib"))
        if not c:
            raise SystemExit(f"No BUY model joblib found under: {model_path}")
        model_path = c[-1]

    # Allow passing a path missing the models/ segment.
    if not model_path.exists():
        alt = model_path.parent / "models" / model_path.name
        if alt.exists():
            model_path = alt

    print(f"Loading model: {model_path}", flush=True)
    art = joblib.load(model_path)
    model = art["model"]

    pre_min = int(art.get("pre_min", 5))
    horizon_min = int(art.get("horizon_min", 15))
    fee_total = float(art.get("fee_total", 0.001))
    per_side_fee = fee_total / 2.0
    ret_thresh = float(art.get("ret_thresh", 0.2))
    keep_cols = list(art.get("feature_cols", []))

    if not keep_cols:
        raise SystemExit("Model artifact missing feature_cols")

    # load bars (only columns needed for feature build)
    base_feat_cols = [
        c
        for c in [
            "ret_1m_pct",
            "mom_3m_pct",
            "mom_5m_pct",
            "vol_std_5m",
            "range_5m",
            "range_norm_5m",
            "macd",
            "vwap_dev_5m",
        ]
    ]

    cols = ["timestamp", "close", "volume"] + base_feat_cols

    print(f"Loading bars: {args.bars}", flush=True)
    df = pd.read_parquet(args.bars, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = int(len(df))

    print(f"Rebuilding 5m precontext features for inference (n={n:,}) ...", flush=True)
    feat = build_5m_precontext_feature_arrays(df=df, pre_min=pre_min, base_feat_cols=[c for c in base_feat_cols if c in df.columns])

    missing = [c for c in keep_cols if c not in feat]
    if missing:
        raise SystemExit(f"Missing required feature cols for model: {missing[:10]} (total missing={len(missing)})")

    X = np.column_stack([feat[c] for c in keep_cols]).astype(np.float32, copy=False)

    # validity: need full precontext (t>=pre_min) and future window available
    idx = np.arange(n)
    valid = idx >= int(pre_min)
    valid &= idx < (n - int(horizon_min) - 1)

    print(f"Running inference for BUY on valid rows: {int(valid.sum()):,}/{n:,} ...", flush=True)
    pred = np.full(n, np.nan, dtype=np.float64)
    pred[valid] = model.predict(X[valid])

    # evaluation window
    split_test = int(n * (1.0 - float(args.test_frac)))
    if args.on == "test":
        start_i, end_i = split_test, n
    else:
        start_i, end_i = 0, n

    print(f"Simulating 1-trade-at-a-time on {args.on} window: idx[{start_i}:{end_i}] ...", flush=True)

    close = df["close"].to_numpy(np.float64, copy=False)
    timestamps = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")

    rules = []
    # same core threshold grid as before
    abs_thrs = [0.20, 0.25, 0.30, 0.40, 0.50]
    top_fracs = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]

    # Compute top-k thresholds from predictions in the evaluation window
    p_win = pred[(idx >= start_i) & (idx < end_i) & valid]

    for thr in abs_thrs:
        res = simulate_one_trade_at_a_time_buy(
            timestamps=timestamps,
            close=close,
            pred=pred,
            valid=valid,
            start_i=start_i,
            end_i=end_i,
            per_side_fee=per_side_fee,
            horizon_min=horizon_min,
            ret_thresh=ret_thresh,
            pred_threshold=float(thr),
        )
        rules.append(
            {
                "rule": f"pred>= {thr:.2f}",
                "selected": res["selected"],
                "win_rate": res["win_rate"],
                "mean_true_ret": res["mean_true_ret"],
                "median_true_ret": res["median_true_ret"],
                "p90_true_ret": res["p90_true_ret"],
                "mean_duration_min": res["mean_duration_min"],
            }
        )

    for q in top_fracs:
        if p_win.size == 0:
            continue
        thr = float(np.nanquantile(p_win, 1.0 - float(q)))
        res = simulate_one_trade_at_a_time_buy(
            timestamps=timestamps,
            close=close,
            pred=pred,
            valid=valid,
            start_i=start_i,
            end_i=end_i,
            per_side_fee=per_side_fee,
            horizon_min=horizon_min,
            ret_thresh=ret_thresh,
            pred_threshold=float(thr),
        )
        rules.append(
            {
                "rule": f"top_{q*100:.1f}% (thr={thr:.4f})",
                "selected": res["selected"],
                "win_rate": res["win_rate"],
                "mean_true_ret": res["mean_true_ret"],
                "median_true_ret": res["median_true_ret"],
                "p90_true_ret": res["p90_true_ret"],
                "mean_duration_min": res["mean_duration_min"],
            }
        )

    out_df = pd.DataFrame(rules)
    out_df.to_csv(out_root / f"one_trade_thresholds_{ts}.csv", index=False)

    # Save a trade sample for a representative rule (pred>=0.30)
    sample_rule = 0.30
    res = simulate_one_trade_at_a_time_buy(
        timestamps=timestamps,
        close=close,
        pred=pred,
        valid=valid,
        start_i=start_i,
        end_i=end_i,
        per_side_fee=per_side_fee,
        horizon_min=horizon_min,
        ret_thresh=ret_thresh,
        pred_threshold=float(sample_rule),
    )
    trades = pd.DataFrame(res["trades"])
    if not trades.empty:
        trades.sample(n=min(20000, len(trades)), random_state=42).to_csv(out_root / f"one_trade_trades_sample_{ts}.csv", index=False)

    # Optional: export full trade list for a top-X% rule
    if float(args.export_top_frac) > 0.0:
        q = float(args.export_top_frac)
        if not (0.0 < q < 1.0):
            raise SystemExit("--export-top-frac must be in (0,1)")
        thr = float(np.nanquantile(p_win, 1.0 - q))
        res2 = simulate_one_trade_at_a_time_buy(
            timestamps=timestamps,
            close=close,
            pred=pred,
            valid=valid,
            start_i=start_i,
            end_i=end_i,
            per_side_fee=per_side_fee,
            horizon_min=horizon_min,
            ret_thresh=ret_thresh,
            pred_threshold=float(thr),
        )
        trades2 = pd.DataFrame(res2["trades"])
        pct = int(round(q * 100.0))
        out_p = out_root / f"one_trade_trades_top{pct}pct_{ts}.parquet"
        trades2.to_parquet(out_p, index=False)
        print(f"Exported full trades for top {q*100:.1f}% (thr={thr:.6f}) -> {out_p} rows={len(trades2):,}", flush=True)

    meta = {
        "created_utc": ts,
        "model": str(model_path),
        "bars": str(args.bars),
        "side": "BUY",
        "on": str(args.on),
        "test_frac": float(args.test_frac),
        "pre_min": int(pre_min),
        "horizon_min": int(horizon_min),
        "fee_total": float(fee_total),
        "per_side_fee": float(per_side_fee),
        "ret_thresh": float(ret_thresh),
        "n_bars": int(n),
        "n_valid": int(valid.sum()),
        "n_features": int(len(keep_cols)),
    }
    (out_root / f"meta_{ts}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Done.")
    print("Outputs:", out_root)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
