#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-03T16:15:30Z
"""Evaluate an oracle-gap exit regressor without manual thresholds.

Given a regressor that predicts oracle_gap_pct = oracle_ret_pct - ret_if_exit_now_pct,
we can derive *threshold-free* policies:

1) local_min_pred_gap (causal / sequential):
   - scan minutes k=min_exit_k..hold_min
   - keep the best (lowest) predicted gap seen so far
   - exit at the first time the predicted gap starts increasing (i.e. exit at k-1)
   This uses only information available up to the current minute.

2) argmin_pred_gap (non-causal upper bound):
   - choose k that minimizes predicted gap within the horizon.

This script also prints hold/oracle baselines on a held-out, time-ordered test split.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def time_ordered_trade_split(df: pd.DataFrame, test_trade_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    if "entry_time" in df.columns:
        order = df[["trade_id", "entry_time"]].drop_duplicates("trade_id").sort_values("entry_time")
    else:
        order = df[["trade_id"]].drop_duplicates("trade_id").sort_values("trade_id")

    tids = order["trade_id"].to_numpy()
    n = len(tids)
    n_test = max(1, int(n * float(test_trade_frac)))
    return tids[: n - n_test], tids[n - n_test :]


def policy_hold(R: np.ndarray, hold_min: int) -> np.ndarray:
    return R[:, hold_min - 1]


def policy_oracle(df: pd.DataFrame, trade_ids: np.ndarray) -> np.ndarray:
    o = df[df["trade_id"].isin(trade_ids)][["trade_id", "oracle_ret_pct"]].drop_duplicates("trade_id")
    o = o.sort_values("trade_id")
    return o["oracle_ret_pct"].to_numpy(np.float64)


def local_min_policy(pred_gap: np.ndarray, hold_min: int, min_exit_k: int) -> np.ndarray:
    """Return chosen exit_k (1..hold_min) per trade using local-minimum rule."""
    N, K = pred_gap.shape
    assert K == hold_min

    exit_k = np.full((N,), hold_min, dtype=np.int64)

    for i in range(N):
        # initialize at min_exit_k
        k0 = max(1, int(min_exit_k))
        if k0 >= hold_min:
            exit_k[i] = hold_min
            continue

        prev = float(pred_gap[i, k0 - 1])
        best_k = int(k0)

        for k in range(k0 + 1, hold_min + 1):
            cur = float(pred_gap[i, k - 1])
            # if the predicted gap starts increasing, exit at the previous best
            if np.isfinite(cur) and np.isfinite(prev) and cur > prev:
                exit_k[i] = int(best_k)
                break

            prev = cur
            best_k = int(k)

    return exit_k


def argmin_policy(pred_gap: np.ndarray, hold_min: int, min_exit_k: int) -> np.ndarray:
    N, K = pred_gap.shape
    assert K == hold_min

    start = max(1, int(min_exit_k)) - 1
    sub = pred_gap[:, start:]
    # nan-safe: treat nan as +inf (never chosen)
    sub2 = np.where(np.isfinite(sub), sub, np.inf)
    idx = np.argmin(sub2, axis=1)  # 0-based within sub
    return (idx + 1 + start).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate oracle-gap regressor policies (no thresholds)")
    ap.add_argument("--model-joblib", required=True)
    ap.add_argument("--dataset-parquet", required=True)
    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--test-trade-frac", type=float, default=0.2)
    ap.add_argument("--min-exit-k", type=int, default=2)

    args = ap.parse_args()

    model_path = Path(args.model_joblib)
    ds_path = Path(args.dataset_parquet)
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")
    if not ds_path.exists():
        raise SystemExit(f"dataset not found: {ds_path}")

    hold_min = int(args.hold_min)
    min_exit_k = int(args.min_exit_k)

    art = joblib.load(model_path)
    reg = art["model"]
    feat_cols = list(art["feature_cols"])

    df = pd.read_parquet(ds_path)
    df = df[(df["k_rel"] >= 1) & (df["k_rel"] <= hold_min)].copy()
    df = df.sort_values(["trade_id", "k_rel"]).reset_index(drop=True)

    # keep only full-length trades
    sizes = df.groupby("trade_id")["k_rel"].count()
    full_ids = sizes[sizes == hold_min].index.to_numpy()
    df = df[df["trade_id"].isin(full_ids)].copy().reset_index(drop=True)

    # compute label (not used for policy, but keeps parity)
    df["oracle_gap_pct"] = pd.to_numeric(df["oracle_ret_pct"], errors="coerce") - pd.to_numeric(df["ret_if_exit_now_pct"], errors="coerce")

    # ensure feature columns exist
    for c in feat_cols:
        if c not in df.columns:
            df[c] = np.nan

    X = df[feat_cols].to_numpy(np.float32)
    pred = reg.predict(X).astype(np.float64)

    # test split
    _, test_ids = time_ordered_trade_split(df, test_trade_frac=float(args.test_trade_frac))
    dte = df[df["trade_id"].isin(test_ids)].copy().reset_index(drop=True)
    pte = pred[np.where(df["trade_id"].isin(test_ids).to_numpy())[0]]

    # pivot returns and preds
    R = dte.pivot(index="trade_id", columns="k_rel", values="ret_if_exit_now_pct").reindex(columns=range(1, hold_min + 1))
    P = pd.DataFrame({"trade_id": dte["trade_id"].to_numpy(), "k_rel": dte["k_rel"].to_numpy(), "pred_gap": pte})
    P = P.pivot(index="trade_id", columns="k_rel", values="pred_gap").reindex(columns=range(1, hold_min + 1))

    Rv = R.to_numpy(np.float64)
    Pv = P.to_numpy(np.float64)

    hold = policy_hold(Rv, hold_min=hold_min)
    oracle = policy_oracle(dte, trade_ids=R.index.to_numpy())

    print(f"n_test_trades={Rv.shape[0]}")
    print(f"hold_mean={float(np.mean(hold)):.6f}%  hold_median={float(np.median(hold)):.6f}%")
    print(f"oracle_mean={float(np.mean(oracle)):.6f}%  oracle_median={float(np.median(oracle)):.6f}%")

    # local-min policy
    ek_local = local_min_policy(Pv, hold_min=hold_min, min_exit_k=min_exit_k)
    chosen_local = Rv[np.arange(Rv.shape[0]), ek_local - 1]
    print(
        f"local_min_pred_gap: mean={float(np.mean(chosen_local)):.6f}%  median={float(np.median(chosen_local)):.6f}%"
        f"  win%={float(np.mean(chosen_local>0)*100.0):.3f}  mean_exit_k={float(np.mean(ek_local)):.3f}"
    )

    # argmin policy (upper bound)
    ek_argmin = argmin_policy(Pv, hold_min=hold_min, min_exit_k=min_exit_k)
    chosen_argmin = Rv[np.arange(Rv.shape[0]), ek_argmin - 1]
    print(
        f"argmin_pred_gap:     mean={float(np.mean(chosen_argmin)):.6f}%  median={float(np.median(chosen_argmin)):.6f}%"
        f"  win%={float(np.mean(chosen_argmin>0)*100.0):.3f}  mean_exit_k={float(np.mean(ek_argmin)):.3f}"
    )


if __name__ == "__main__":
    main()
