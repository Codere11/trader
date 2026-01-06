#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T13:28:44Z
"""Build a 'violent candidates' dataset over the FULL ETH history (BUY), then label it.

You requested
- Consider the trades the system would take over the entire dataset via a SCORE THRESHOLD.
- Isolate 'violent' 5-minute precontexts.
- Label them as:
    FAST_HIT: can hit >=0.2% net return within 5 minutes after entry
    SLOW_HIT: can hit >=0.2% net return within 15 minutes, but not within 5
    NO_HIT: cannot hit >=0.2% within 15 minutes
- This dataset is for hypertuning a model to separate good-violent vs bad-violent.

This script:
1) Loads a trained BUY model artifact to reproduce its features and compute pred[t] over the full history.
2) Selects candidate minutes where pred[t] >= pred_threshold.
3) Computes violent score (based only on 5m precontext) and keeps the top violent_frac within candidates.
4) Labels each kept candidate with oracle outcomes in the next horizon minutes.

Outputs (timestamped)
- data/violent_candidates_oracle15m_<ts>/
  - violent_candidates.parquet
  - summary.json
  - univariate_discriminators.csv

No retraining in this script.
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


def net_ret_pct_buy(entry_px: np.ndarray, exit_px: np.ndarray, per_side_fee: float) -> np.ndarray:
    f = float(per_side_fee)
    mult = (exit_px * (1.0 - f)) / (entry_px * (1.0 + f))
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


def build_features_for_model(df: pd.DataFrame, *, pre_min: int, base_feat_cols: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Build 5m precontext descriptors for every row (t uses t-5..t-1)."""
    L = int(pre_min)

    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64, copy=False)
    vol = pd.to_numeric(df["volume"], errors="coerce").to_numpy(np.float64, copy=False)

    close_prev = _rolling_prev_matrix(close, L)
    close_last = close_prev[:, -1]
    close_norm = (close_prev / np.maximum(1e-12, close_last[:, None]) - 1.0) * 100.0

    feat: dict[str, np.ndarray] = {}

    # price
    feat.update(_agg_5(close_norm, "px_close_norm_pct"))
    feat["px_close_norm_pct__ret5m"] = close_norm[:, -1] - close_norm[:, 0]
    feat["px_close_norm_pct__absret5m"] = np.abs(feat["px_close_norm_pct__ret5m"])
    feat["px_close_norm_pct__m5"] = close_norm[:, 0]
    feat["px_close_norm_pct__m4"] = close_norm[:, 1]
    feat["px_close_norm_pct__m3"] = close_norm[:, 2]
    feat["px_close_norm_pct__m2"] = close_norm[:, 3]
    feat["px_close_norm_pct__m1"] = close_norm[:, 4]

    # volume
    vol_prev = _rolling_prev_matrix(vol, L)
    vol_log = np.log1p(np.maximum(0.0, vol_prev))
    feat.update(_agg_5(vol_log, "vol_log1p"))

    # base feature columns
    for c in base_feat_cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64, copy=False)
        x_prev = _rolling_prev_matrix(x, L)
        feat.update(_agg_5(x_prev, c))

    # missingness
    miss_n = np.isnan(close_prev).sum(axis=1).astype(np.float64)
    feat["missing_close_n"] = miss_n
    feat["missing_any"] = (miss_n > 0).astype(np.float64)

    # extra structural features (important for violent good vs bad)
    prev_close = close_prev
    lo = np.nanmin(prev_close, axis=1)
    hi = np.nanmax(prev_close, axis=1)
    feat["pos_in_range_5m"] = (close_last - lo) / np.maximum(1e-12, (hi - lo))

    # 1m return microstructure in window
    r1 = (prev_close[:, 1:] / np.maximum(1e-12, prev_close[:, :-1]) - 1.0) * 100.0  # (n,4)
    feat["last1m_ret_in_win"] = r1[:, -1]
    feat["max_abs_1m_ret_in_win"] = np.nanmax(np.abs(r1), axis=1)
    feat["sum_abs_1m_ret_in_win"] = np.nansum(np.abs(r1), axis=1)
    feat["sign_changes_1m"] = np.sum(np.sign(r1)[:, 1:] != np.sign(r1)[:, :-1], axis=1).astype(np.float64)

    # volume baseline vs burst
    v_mean = np.nanmean(vol_prev, axis=1)
    v_last = vol_prev[:, -1]
    v_max = np.nanmax(vol_prev, axis=1)
    v_std = _safe_nanstd(vol_prev)
    feat["vol_mean_5m"] = v_mean
    feat["vol_last_over_mean_5m"] = v_last / np.maximum(1e-12, v_mean)
    feat["vol_max_over_mean_5m"] = v_max / np.maximum(1e-12, v_mean)
    feat["vol_std_over_mean_5m"] = v_std / np.maximum(1e-12, v_mean)

    names = list(feat.keys())
    return feat, names


def oracle_hit_time_buy(
    *,
    close: np.ndarray,
    entry_i: np.ndarray,
    horizon_min: int,
    per_side_fee: float,
    ret_thresh: float,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """For each entry index, compute (best_ret_15m, time_to_hit), where time_to_hit in [1..horizon] or 0 if no hit."""
    close = np.asarray(close, dtype=np.float64)
    idx = np.asarray(entry_i, dtype=np.int64)
    n = int(close.size)
    W = int(horizon_min)

    best_ret = np.full(idx.size, np.nan, dtype=np.float64)
    t_hit = np.zeros(idx.size, dtype=np.int16)

    offs = np.arange(1, W + 1, dtype=np.int64)

    for s in range(0, idx.size, int(chunk)):
        e = min(idx.size, s + int(chunk))
        ii = idx[s:e]

        # build future index matrix (e-s, W)
        mat_idx = ii[:, None] + offs[None, :]
        mat_idx = np.clip(mat_idx, 0, n - 1)
        fut = close[mat_idx]

        entry_px = close[ii]

        # best exit ret
        fut_max = np.nanmax(fut, axis=1)
        br = net_ret_pct_buy(entry_px, fut_max, per_side_fee)
        best_ret[s:e] = br

        # time-to-hit for threshold
        # compute ret per minute
        rmat = net_ret_pct_buy(entry_px[:, None], fut, per_side_fee)
        hit = rmat >= float(ret_thresh)
        # first hit index (+1 because offsets start at 1). if none -> 0
        any_hit = hit.any(axis=1)
        first = np.argmax(hit, axis=1).astype(np.int16) + 1
        first = np.where(any_hit, first, 0).astype(np.int16)
        t_hit[s:e] = first

    return best_ret, t_hit


def main() -> None:
    ap = argparse.ArgumentParser(description="Build violent candidate dataset from model-threshold entries + oracle labels")
    ap.add_argument(
        "--model",
        default="data/entry_regressor_oracle15m_2026-01-04T11-28-38Z/models/entry_regressor_buy_oracle15m_2026-01-04T11-28-38Z.joblib",
        help="BUY model artifact joblib",
    )
    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="ETH 1m dataset parquet",
    )
    ap.add_argument("--pred-threshold", type=float, default=0.20, help="Candidate if pred>=this")
    ap.add_argument("--violent-frac", type=float, default=0.20, help="Keep top X fraction most-violent among candidates")
    ap.add_argument("--horizon-min", type=int, default=15)
    ap.add_argument("--fast-min", type=int, default=5)
    ap.add_argument("--ret-thresh", type=float, default=0.2)
    ap.add_argument("--fee-total", type=float, default=0.001)
    ap.add_argument("--chunk", type=int, default=250000, help="Chunk size for oracle labeling")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    ts = now_ts()
    out_root = Path(args.out_dir) / f"violent_candidates_oracle15m_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}", flush=True)
    art = joblib.load(Path(args.model))
    model = art["model"]
    keep_cols = list(art.get("feature_cols", []))
    pre_min = int(art.get("pre_min", 5))

    per_side_fee = float(args.fee_total) / 2.0

    print(f"Loading bars: {args.bars}", flush=True)
    cols = ["timestamp", "close", "volume"] + [c for c in FEATURE_COLS_DEFAULT]
    df = pd.read_parquet(args.bars, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    base_cols = [c for c in FEATURE_COLS_DEFAULT if c in df.columns]
    n = int(len(df))

    print(f"Building 5m features for inference (n={n:,}) ...", flush=True)
    feat, feat_names = build_features_for_model(df, pre_min=pre_min, base_feat_cols=base_cols)

    missing = [c for c in keep_cols if c not in feat]
    if missing:
        raise SystemExit(f"Model expects missing feature cols: {missing[:10]} (total={len(missing)})")

    # inference matrix
    X = np.column_stack([feat[c] for c in keep_cols]).astype(np.float32, copy=False)

    valid = (np.arange(n) >= pre_min) & (np.arange(n) < (n - int(args.horizon_min) - 1))

    pred = np.full(n, np.nan, dtype=np.float64)
    print(f"Running inference on valid rows: {int(valid.sum()):,}/{n:,} ...", flush=True)
    pred[valid] = model.predict(X[valid])

    # candidates by score threshold
    cand = valid & np.isfinite(pred) & (pred >= float(args.pred_threshold))
    cand_idx = np.where(cand)[0]
    print(f"Candidates (pred>={args.pred_threshold}): {cand_idx.size:,}", flush=True)
    if cand_idx.size == 0:
        raise SystemExit("No candidates at this threshold")

    # violent score among candidates: composite percentile of (absret5m, range5m, vol_std_5m_last)
    absret = np.asarray(feat["px_close_norm_pct__absret5m"], dtype=np.float64)
    rng = np.asarray(feat["px_close_norm_pct__range5"], dtype=np.float64)
    # fallback if feature missing
    vol_last = np.asarray(feat.get("vol_std_5m__last", np.zeros(n, dtype=np.float64)), dtype=np.float64)

    a = absret[cand_idx]
    b = rng[cand_idx]
    c = vol_last[cand_idx]

    # percentile ranks
    ra = pd.Series(a).rank(pct=True).to_numpy(np.float64)
    rb = pd.Series(b).rank(pct=True).to_numpy(np.float64)
    rc = pd.Series(c).rank(pct=True).to_numpy(np.float64)
    comp = np.maximum.reduce([ra, rb, rc])

    keep_n = int(np.ceil(float(args.violent_frac) * cand_idx.size))
    order = np.argsort(comp)
    keep_idx = cand_idx[order[-keep_n:]]

    print(f"Violent kept: {keep_idx.size:,}/{cand_idx.size:,} (violent_frac={args.violent_frac})", flush=True)

    # oracle labels for kept
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64, copy=False)
    best_ret, t_hit = oracle_hit_time_buy(
        close=close,
        entry_i=keep_idx,
        horizon_min=int(args.horizon_min),
        per_side_fee=per_side_fee,
        ret_thresh=float(args.ret_thresh),
        chunk=int(args.chunk),
    )

    # class labels
    # 0=NO_HIT, 1=SLOW_HIT (hit within 15 but not within fast_min), 2=FAST_HIT (hit within fast_min)
    fast_min = int(args.fast_min)
    y_cls = np.zeros(keep_idx.size, dtype=np.int8)
    y_cls[(t_hit > 0) & (t_hit <= fast_min)] = 2
    y_cls[(t_hit > fast_min) & (t_hit <= int(args.horizon_min))] = 1

    # build output dataframe: store minimal set + all feature columns for those indices
    out = pd.DataFrame(
        {
            "timestamp": df.loc[keep_idx, "timestamp"].to_numpy(),
            "entry_i": keep_idx.astype(np.int64),
            "pred": pred[keep_idx].astype(np.float32),
            "best_ret_15m": best_ret.astype(np.float32),
            "t_hit": t_hit.astype(np.int16),
            "y_cls": y_cls.astype(np.int8),
        }
    )

    # attach features (precontext only)
    use_cols = [
        # keep model cols + extra discriminators
        *list(dict.fromkeys(keep_cols + [
            "pos_in_range_5m",
            "vol_mean_5m",
            "vol_last_over_mean_5m",
            "vol_max_over_mean_5m",
            "vol_std_over_mean_5m",
            "last1m_ret_in_win",
            "max_abs_1m_ret_in_win",
            "sum_abs_1m_ret_in_win",
            "sign_changes_1m",
            "px_close_norm_pct__absret5m",
            "px_close_norm_pct__range5",
            "px_close_norm_pct__slope5",
        ]))
    ]
    use_cols = [c for c in use_cols if c in feat]

    for c in use_cols:
        out[c] = np.asarray(feat[c][keep_idx], dtype=np.float32)

    out_path = out_root / "violent_candidates.parquet"
    out.to_parquet(out_path, index=False)

    # summary
    cls_counts = {"NO_HIT": int((y_cls == 0).sum()), "SLOW_HIT": int((y_cls == 1).sum()), "FAST_HIT": int((y_cls == 2).sum())}
    summary = {
        "created_utc": ts,
        "model": str(args.model),
        "bars": str(args.bars),
        "pred_threshold": float(args.pred_threshold),
        "violent_frac": float(args.violent_frac),
        "pre_min": int(pre_min),
        "horizon_min": int(args.horizon_min),
        "fast_min": int(args.fast_min),
        "ret_thresh": float(args.ret_thresh),
        "fee_total": float(args.fee_total),
        "n_bars": int(n),
        "n_candidates": int(cand_idx.size),
        "n_violent": int(keep_idx.size),
        "class_counts": cls_counts,
        "class_fracs": {k: (v / float(keep_idx.size)) for k, v in cls_counts.items()},
    }

    # univariate discriminators: FAST vs STALL (NO_HIT) and HIT15 vs NO_HIT
    # Here STALL=NO_HIT only; SLOW_HIT counts as hit.
    rows = []
    xdf = out

    # binary labels
    y_hit15 = (xdf["y_cls"].to_numpy(np.int8) > 0).astype(int)
    y_fast = (xdf["y_cls"].to_numpy(np.int8) == 2).astype(int)

    for c in use_cols:
        x = pd.to_numeric(xdf[c], errors="coerce").to_numpy(np.float64)
        m = np.isfinite(x)
        if m.sum() < 200:
            continue
        # hit15 vs no
        auc1 = float("nan")
        if len(np.unique(y_hit15[m])) == 2:
            auc1 = float(roc_auc_score(y_hit15[m], x[m]))
            if auc1 < 0.5:
                auc1 = 1.0 - auc1
        # fast vs rest
        auc2 = float("nan")
        if len(np.unique(y_fast[m])) == 2:
            auc2 = float(roc_auc_score(y_fast[m], x[m]))
            if auc2 < 0.5:
                auc2 = 1.0 - auc2

        rows.append({"feature": c, "auc_hit15_vs_no": auc1, "auc_fast_vs_rest": auc2})

    disc = pd.DataFrame(rows).sort_values(["auc_hit15_vs_no", "auc_fast_vs_rest"], ascending=False)
    disc.to_csv(out_root / "univariate_discriminators.csv", index=False)

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done.")
    print("Output:", out_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
