#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-03T15:25:00Z
"""Evaluate a trained profit-optimized hazard exit policy on an oracle-exit dataset.

This script is used to "backtest" a policy trained on one entry stream (e.g. 1%)
against another entry stream (e.g. 0.1%), with NO re-training.

Inputs
- model .pt (state_dict)
- meta .json (feature_cols, standardization mean/std, model hyperparams)
- dataset parquet produced by train_exit_oracle_classifier_hold15_*.py (must contain ret_if_exit_now_pct)

Outputs
- Prints policy evaluation table (hold baseline, expected_hazard, h>=tau sweep)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


class PerStepMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, k, f = x.shape
        y = self.net(x.reshape(b * k, f))
        return y.reshape(b, k)


def simulate_threshold_policy(logits: np.ndarray, rets: np.ndarray, taus: List[float]) -> pd.DataFrame:
    h = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    N, K = h.shape

    rows: List[Dict[str, float]] = []

    hold_rets = rets[:, K - 1]
    rows.append(
        {
            "tau": float("nan"),
            "policy": "hold",
            "n_trades": float(N),
            "mean_ret_pct": float(np.mean(hold_rets)),
            "median_ret_pct": float(np.median(hold_rets)),
            "win_rate_gt0_pct": float(np.mean(hold_rets > 0.0) * 100.0),
            "mean_exit_k": float(K),
        }
    )

    # expected-return under stochastic hazards
    surv = np.ones((N,), dtype=np.float64)
    exp = np.zeros((N,), dtype=np.float64)
    for k in range(K):
        if k < K - 1:
            p = surv * h[:, k]
            exp += p * rets[:, k]
            surv *= (1.0 - h[:, k])
        else:
            exp += surv * rets[:, k]

    rows.append(
        {
            "tau": float("nan"),
            "policy": "expected_hazard",
            "n_trades": float(N),
            "mean_ret_pct": float(np.mean(exp)),
            "median_ret_pct": float(np.median(exp)),
            "win_rate_gt0_pct": float(np.mean(exp > 0.0) * 100.0),
            "mean_exit_k": float("nan"),
        }
    )

    for tau in taus:
        exit_k = np.full((N,), K, dtype=np.int64)
        for k in range(K - 1):
            hit = (exit_k == K) & (h[:, k] >= float(tau))
            exit_k[hit] = k + 1
        chosen = rets[np.arange(N), exit_k - 1]
        rows.append(
            {
                "tau": float(tau),
                "policy": "h>=tau",
                "n_trades": float(N),
                "mean_ret_pct": float(np.mean(chosen)),
                "median_ret_pct": float(np.median(chosen)),
                "win_rate_gt0_pct": float(np.mean(chosen > 0.0) * 100.0),
                "mean_exit_k": float(np.mean(exit_k.astype(np.float64))),
            }
        )

    return pd.DataFrame(rows)


def build_xy(df: pd.DataFrame, ids: np.ndarray, hold_min: int, feat_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    d = df[df["trade_id"].isin(ids)].copy()
    d = d.sort_values(["trade_id", "k_rel"]).reset_index(drop=True)

    # Ensure k_norm exists if requested
    if "k_norm" in feat_cols and "k_norm" not in d.columns:
        d["k_norm"] = d["k_rel"].astype(np.float32) / float(hold_min)

    X = d.pivot(index="trade_id", columns="k_rel", values=feat_cols)
    R = d.pivot(index="trade_id", columns="k_rel", values="ret_if_exit_now_pct")

    X = X.reindex(columns=pd.MultiIndex.from_product([feat_cols, range(1, hold_min + 1)]))
    R = R.reindex(columns=range(1, hold_min + 1))

    Xv = X.to_numpy(dtype=np.float32).reshape(len(ids), len(feat_cols), hold_min).transpose(0, 2, 1)
    Rv = R.to_numpy(dtype=np.float32)

    Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)
    Rv = np.nan_to_num(Rv, nan=0.0, posinf=0.0, neginf=0.0)

    return Xv, Rv


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate profit hazard exit policy")
    ap.add_argument("--model-pt", required=True)
    ap.add_argument("--meta-json", required=True)
    ap.add_argument("--dataset-parquet", required=True)
    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument(
        "--taus",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,0.90",
    )
    args = ap.parse_args()

    meta = json.loads(Path(args.meta_json).read_text(encoding="utf-8"))

    feat_cols = list(meta["feature_cols"])
    mu = np.asarray(meta["standardize"]["mean"], dtype=np.float32)
    sd = np.asarray(meta["standardize"]["std"], dtype=np.float32)

    hidden = int(meta["model"]["hidden"])
    dropout = float(meta["model"]["dropout"])

    hold_min = int(args.hold_min)

    df = pd.read_parquet(Path(args.dataset_parquet))
    df = df[(df["k_rel"] >= 1) & (df["k_rel"] <= hold_min)].copy()

    sizes = df.groupby("trade_id")["k_rel"].count()
    full_ids = sizes[sizes == hold_min].index.to_numpy()
    df = df[df["trade_id"].isin(full_ids)].copy().reset_index(drop=True)

    # Time-ordered list
    if "entry_time" in df.columns:
        order = df[["trade_id", "entry_time"]].drop_duplicates("trade_id").sort_values("entry_time")
    else:
        order = df[["trade_id"]].drop_duplicates("trade_id").sort_values("trade_id")

    trade_ids = order["trade_id"].to_numpy()

    X, R = build_xy(df, trade_ids, hold_min=hold_min, feat_cols=feat_cols)

    # standardize
    X = (X - mu) / sd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PerStepMLP(d_in=int(X.shape[-1]), d_hidden=hidden, dropout=dropout).to(device)
    state = torch.load(Path(args.model_pt), map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()

    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(xb).detach().cpu().numpy().astype(np.float64)

    taus = [float(x.strip()) for x in str(args.taus).split(",") if x.strip()]
    eval_df = simulate_threshold_policy(logits, R.astype(np.float64), taus=taus)

    # Print baselines from dataset (oracle if present)
    hold_rets = R[:, hold_min - 1].astype(np.float64)
    print(f"n_trades={len(trade_ids)}")
    print(f"hold_mean={float(np.mean(hold_rets)):.6f}%  hold_median={float(np.median(hold_rets)):.6f}%")
    if "oracle_ret_pct" in df.columns:
        o = df[["trade_id", "oracle_ret_pct"]].drop_duplicates("trade_id").sort_values("trade_id")
        oracle = o["oracle_ret_pct"].to_numpy(np.float64)
        if len(oracle) == len(trade_ids):
            print(f"oracle_mean={float(np.mean(oracle)):.6f}%  oracle_median={float(np.median(oracle)):.6f}%")

    print("\n=== Policy evaluation ===")
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(eval_df.to_string(index=False))


if __name__ == "__main__":
    main()
