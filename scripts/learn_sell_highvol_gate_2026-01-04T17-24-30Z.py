#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T17:24:30Z
"""Isolate high-vol regimes within top-10% model-selected SELL entries and separate good vs bad.

User goal (SELL)
- Start from model-selected entries (top 10% by entry model score).
- Within those, isolate *high-vol regimes* where volatility/range/volume proxy signals are high.
- Cleanly separate good high-vol entries from bad high-vol entries using 5-minute precontext features.

Definitions
- y_sell[t] is SELL oracle best-exit net return within the next horizon minutes:
    y_sell[t] = net_ret_pct_sell(close[t], min(close[t+1..t+horizon]), fee_side)
- top10 selection uses the trained entry model score threshold at the 90th percentile.
- high-vol regime is defined on the selected set using proxy quantiles:
    proxies = {vol_std_5m__last, range_norm_5m__max5, vol_log1p__last, range_5m__max5}
  high-vol if at least `--hv-min-count` of these proxies are >= their `--hv-q` quantile (computed within top10).

Good vs bad labels inside the high-vol subset
- good: y_sell >= ret_thresh (default 0.2)
- bad:  y_sell < bad_thresh (default 0.0)
- rows in [bad_thresh, ret_thresh) are treated as "neutral" and excluded from gate training.

Gate model
- LightGBM binary classifier (500 trees) trained on high-vol subset only.
- Features are the same 5m precontext descriptors used by the entry model (feature_cols from the artifact).
- Reports AUC and a retention curve: keep_frac vs achieved win_rate (y_sell>=ret_thresh).

Outputs (timestamped)
- data/highvol_sell_gate_<ts>/
  - summary_<ts>.json
  - hv_proxy_thresholds_<ts>.json
  - gate_metrics_<ts>.json
  - gate_feature_importance_<ts>.csv
  - gate_threshold_curve_<ts>.csv
  - examples_bigwins_<ts>.csv
  - examples_biglosses_<ts>.csv

Notes
- This is a diagnostic/selection gate; it does NOT enforce non-overlap execution.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier


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


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def net_ret_pct_sell(entry_px: np.ndarray, exit_px: np.ndarray, f: float) -> np.ndarray:
    entry_px = np.asarray(entry_px, dtype=np.float64)
    exit_px = np.asarray(exit_px, dtype=np.float64)
    mult = (entry_px * (1.0 - f)) / (exit_px * (1.0 + f))
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


def build_pre5m_feature_matrix(
    df: pd.DataFrame,
    *,
    pre_min: int,
    base_feat_cols: list[str],
    out_feature_cols: list[str],
) -> np.ndarray:
    L = int(pre_min)
    if L != 5:
        raise SystemExit("This script currently supports --pre-min=5 only")

    n = len(df)
    col_to_j = {c: j for j, c in enumerate(out_feature_cols)}
    X = np.full((n, len(out_feature_cols)), np.nan, dtype=np.float32)

    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64, copy=False)
    vol = pd.to_numeric(df["volume"], errors="coerce").to_numpy(np.float64, copy=False)

    def _set(name: str, arr: np.ndarray) -> None:
        j = col_to_j.get(name)
        if j is None:
            return
        X[:, j] = np.asarray(arr, dtype=np.float32)

    close_prev = _rolling_prev_matrix(close, L)
    close_last = close_prev[:, -1]
    close_last = np.where((np.isfinite(close_last)) & (close_last > 0.0), close_last, np.nan)
    close_norm = (close_prev / close_last[:, None] - 1.0) * 100.0

    px = _agg_5(close_norm, "px_close_norm_pct")
    px_ret5m = close_norm[:, -1] - close_norm[:, 0]
    px_absret5m = np.abs(px_ret5m)

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

    vol_prev = _rolling_prev_matrix(vol, L)
    vol_log = np.log1p(np.maximum(0.0, vol_prev))
    vv = _agg_5(vol_log, "vol_log1p")
    for k, v in vv.items():
        _set(k, v)

    for c in base_feat_cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64, copy=False)
        x_prev = _rolling_prev_matrix(x, L)
        a = _agg_5(x_prev, c)
        for k, v in a.items():
            _set(k, v)

    return X


def _future_extrema_excl_current(x: np.ndarray, W: int, *, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    b = x[::-1]

    ans_rev = np.full(n, np.nan, dtype=np.float64)
    dq: deque[int] = deque()

    def better(a: float, b: float) -> bool:
        return (a >= b) if mode == "max" else (a <= b)

    for j in range(n):
        while dq and dq[0] < j - W:
            dq.popleft()

        ans_rev[j] = b[dq[0]] if dq else np.nan

        v = b[j]
        if np.isfinite(v):
            while dq and np.isfinite(b[dq[-1]]) and better(v, b[dq[-1]]):
                dq.pop()
            dq.append(j)
        else:
            dq.append(j)

    return ans_rev[::-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Learn a high-vol gate for top-10% SELL entries")
    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="ETH 1m OHLCV+features parquet",
    )
    ap.add_argument(
        "--entry-model",
        default="data/entry_regressor_inliers_sell_2026-01-04T16-49-43Z/models/entry_regressor_sell_inliers_2026-01-04T16-49-43Z.joblib",
        help="SELL entry regressor joblib artifact (dict with {model, feature_cols, ...})",
    )

    ap.add_argument("--fee-total", type=float, default=0.001)
    ap.add_argument("--horizon-min", type=int, default=15)
    ap.add_argument("--pre-min", type=int, default=5)

    ap.add_argument("--top-frac", type=float, default=0.10)

    ap.add_argument("--hv-q", type=float, default=0.90, help="Quantile for defining high proxy values within top10")
    ap.add_argument("--hv-min-count", type=int, default=2, help="Require at least this many proxies above hv-q")

    ap.add_argument("--ret-thresh", type=float, default=0.2, help="good if y_sell >= this")
    ap.add_argument("--bad-thresh", type=float, default=0.0, help="bad if y_sell < this")

    ap.add_argument("--gate-n-estimators", type=int, default=500)
    ap.add_argument("--gate-test-frac", type=float, default=0.20)

    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--out-dir", default="data")

    args = ap.parse_args()

    bars_path = Path(args.bars)
    if not bars_path.exists():
        raise SystemExit(f"bars not found: {bars_path}")

    art = joblib.load(args.entry_model)
    if not isinstance(art, dict) or "model" not in art or "feature_cols" not in art:
        raise SystemExit("entry model artifact must be a dict with keys: model, feature_cols")

    entry_model = art["model"]
    feat_cols = list(art["feature_cols"])

    pre_min = int(args.pre_min)
    horizon = int(args.horizon_min)
    fee_side = float(args.fee_total) / 2.0

    need_cols = ["timestamp", "close", "volume"] + list(BASE_FEATS_DEFAULT)
    print(f"Loading bars: {bars_path}", flush=True)
    df = pd.read_parquet(bars_path, columns=need_cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if not df["timestamp"].is_monotonic_increasing:
        df = df.sort_values("timestamp").reset_index(drop=True)

    n = int(len(df))
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64, copy=False)

    print(f"Computing y_sell (oracle min close within {horizon}m)...", flush=True)
    fut_min = _future_extrema_excl_current(close, horizon, mode="min")
    y_sell = net_ret_pct_sell(close, fut_min, fee_side)

    valid = (np.arange(n) >= pre_min) & np.isfinite(y_sell) & np.isfinite(close)
    idxs = np.where(valid)[0]
    print(f"Valid minutes: {idxs.size:,}/{n:,}", flush=True)

    print(f"Building 5m precontext features (cols={len(feat_cols)})...", flush=True)
    X_all = build_pre5m_feature_matrix(df, pre_min=pre_min, base_feat_cols=list(BASE_FEATS_DEFAULT), out_feature_cols=feat_cols)

    print("Scoring entry model on valid minutes...", flush=True)
    pred = np.full(n, np.nan, dtype=np.float32)
    chunk = int(args.chunk)
    for s in range(0, idxs.size, chunk):
        e = min(idxs.size, s + chunk)
        ii = idxs[s:e]
        pred[ii] = entry_model.predict(X_all[ii]).astype(np.float32)
        if (s // chunk) % 10 == 0:
            print(f"  scored {e:,}/{idxs.size:,}", flush=True)

    pv = pred[valid].astype(np.float64)
    pv = pv[np.isfinite(pv)]
    if pv.size == 0:
        raise SystemExit("No finite predictions")

    top_frac = float(args.top_frac)
    thr_top = float(np.quantile(pv, 1.0 - top_frac))
    sel_mask = valid & np.isfinite(pred) & (pred >= thr_top)
    sel_idx = np.where(sel_mask)[0]

    # Build selected frame with features
    sel_X = X_all[sel_idx]
    sel_feat = pd.DataFrame(sel_X, columns=feat_cols)
    sel = pd.DataFrame(
        {
            "idx": sel_idx.astype(np.int64),
            "timestamp": df.loc[sel_idx, "timestamp"].to_numpy(),
            "pred": pred[sel_idx].astype(np.float64),
            "y_sell": y_sell[sel_idx].astype(np.float64),
        }
    )
    sel = pd.concat([sel, sel_feat], axis=1)

    # High-vol proxy definitions (must exist in feature columns)
    proxies = [
        "vol_std_5m__last",
        "range_norm_5m__max5",
        "vol_log1p__last",
        "range_5m__max5",
    ]
    proxies = [p for p in proxies if p in sel.columns]
    if len(proxies) < 2:
        raise SystemExit(f"Not enough proxy columns present for high-vol definition. Found: {proxies}")

    hv_q = float(args.hv_q)
    hv_thr = {p: float(np.nanquantile(pd.to_numeric(sel[p], errors="coerce"), hv_q)) for p in proxies}

    hv_hits = np.zeros(len(sel), dtype=np.int32)
    for p in proxies:
        hv_hits += (pd.to_numeric(sel[p], errors="coerce").to_numpy(np.float64) >= hv_thr[p]).astype(np.int32)

    hv_min_count = int(args.hv_min_count)
    sel["hv_hits"] = hv_hits
    sel["is_high_vol"] = (hv_hits >= hv_min_count).astype(np.int32)

    hv = sel[sel["is_high_vol"] == 1].copy().reset_index(drop=True)

    ret_thresh = float(args.ret_thresh)
    bad_thresh = float(args.bad_thresh)

    hv["label_good"] = (hv["y_sell"] >= ret_thresh).astype(np.int32)
    hv["label_bad"] = (hv["y_sell"] < bad_thresh).astype(np.int32)
    hv["label_neutral"] = ((hv["y_sell"] >= bad_thresh) & (hv["y_sell"] < ret_thresh)).astype(np.int32)

    # Gate training dataset: exclude neutrals
    gate_df = hv[hv["label_neutral"] == 0].copy().reset_index(drop=True)
    if gate_df.empty:
        raise SystemExit("High-vol gate dataset empty after removing neutrals; adjust thresholds")

    # Chronological split by timestamp
    gate_df["timestamp"] = pd.to_datetime(gate_df["timestamp"], utc=True, errors="coerce")
    gate_df = gate_df.sort_values("timestamp").reset_index(drop=True)

    y_cls = gate_df["label_good"].to_numpy(np.int32)
    X_cols = list(feat_cols)
    X = gate_df[X_cols].to_numpy(np.float32, copy=False)

    n_gate = int(len(gate_df))
    split = int(n_gate * (1.0 - float(args.gate_test_frac)))
    X_tr, y_tr = X[:split], y_cls[:split]
    X_te, y_te = X[split:], y_cls[split:]

    # Train classifier
    clf = LGBMClassifier(
        n_estimators=int(args.gate_n_estimators),
        learning_rate=0.05,
        num_leaves=128,
        max_depth=-1,
        min_child_samples=200,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        class_weight="balanced",
    )
    clf.fit(X_tr, y_tr)

    proba_te = clf.predict_proba(X_te)[:, 1].astype(np.float64)

    # AUC
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y_te, proba_te)) if len(np.unique(y_te)) == 2 else float("nan")
    except Exception:
        auc = float("nan")

    # Threshold curve on test
    # evaluate keeping top-k% by proba and compute achieved win rate vs y_sell>=ret_thresh
    y_sell_te = gate_df["y_sell"].to_numpy(np.float64)[split:]

    qs = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80]
    curve = []
    for q in qs:
        thr = float(np.quantile(proba_te, 1.0 - q))
        m = proba_te >= thr
        sel_y = y_sell_te[m]
        curve.append(
            {
                "keep_top_frac": float(q),
                "threshold": float(thr),
                "selected": int(m.sum()),
                "win_rate_y>=ret": float(np.mean(sel_y >= ret_thresh)) if sel_y.size else float("nan"),
                "sel_median": float(np.median(sel_y)) if sel_y.size else float("nan"),
                "sel_p90": float(np.quantile(sel_y, 0.9)) if sel_y.size else float("nan"),
            }
        )

    ts = now_ts()
    out_root = Path(args.out_dir) / f"highvol_sell_gate_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Save summaries
    summary = {
        "created_utc": ts,
        "bars": str(bars_path),
        "entry_model": str(Path(args.entry_model)),
        "fee_total": float(args.fee_total),
        "fee_side": float(fee_side),
        "horizon_min": int(horizon),
        "pre_min": int(pre_min),
        "top_frac": float(top_frac),
        "top_pred_threshold": float(thr_top),
        "selected_n": int(len(sel)),
        "hv_q": float(hv_q),
        "hv_min_count": int(hv_min_count),
        "high_vol_n": int(len(hv)),
        "high_vol_frac_of_top": float(len(hv) / max(1, len(sel))),
        "hv_good_rate_y>=ret": float(np.mean(hv["y_sell"] >= ret_thresh)) if len(hv) else float("nan"),
        "hv_bad_rate_y<bad": float(np.mean(hv["y_sell"] < bad_thresh)) if len(hv) else float("nan"),
        "gate_dataset_n": int(n_gate),
        "gate_train_n": int(split),
        "gate_test_n": int(n_gate - split),
        "gate_auc_test": float(auc),
        "ret_thresh": float(ret_thresh),
        "bad_thresh": float(bad_thresh),
    }
    (out_root / f"summary_{ts}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    (out_root / f"hv_proxy_thresholds_{ts}.json").write_text(json.dumps(hv_thr, indent=2) + "\n", encoding="utf-8")

    gate_metrics = {
        "auc_test": float(auc),
        "pos_rate_test": float(np.mean(y_te)) if y_te.size else float("nan"),
        "pos_rate_train": float(np.mean(y_tr)) if y_tr.size else float("nan"),
    }
    (out_root / f"gate_metrics_{ts}.json").write_text(json.dumps(gate_metrics, indent=2) + "\n", encoding="utf-8")

    # Feature importance
    imp = pd.DataFrame(
        {
            "feature": X_cols,
            "importance_gain": clf.booster_.feature_importance(importance_type="gain"),
            "importance_split": clf.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    imp.to_csv(out_root / f"gate_feature_importance_{ts}.csv", index=False)

    pd.DataFrame(curve).to_csv(out_root / f"gate_threshold_curve_{ts}.csv", index=False)

    # Save some examples for physical inspection
    # Big wins / losses inside high-vol set
    hv_sorted = hv.sort_values("y_sell")
    hv_sorted.tail(2000).sort_values("y_sell", ascending=False).to_csv(out_root / f"examples_bigwins_{ts}.csv", index=False)
    hv_sorted.head(2000).to_csv(out_root / f"examples_biglosses_{ts}.csv", index=False)

    print(json.dumps(summary, indent=2))
    print("High-vol proxy thresholds:")
    print(json.dumps(hv_thr, indent=2))
    print("Wrote", out_root)


if __name__ == "__main__":
    main()
