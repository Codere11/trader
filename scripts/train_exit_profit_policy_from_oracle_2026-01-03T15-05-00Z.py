#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-03T15:05:00Z
"""Train a profitability-optimized exit policy using oracle-exit episodes.

Intent (per user request)
- DO NOT train to predict oracle_k or to imitate oracle exit timing.
- Use the oracle-exit *dataset* as the learning source (i.e., trade episodes with a fixed horizon),
  but optimize the policy directly for profitability.

How
- We model an exit policy as per-minute hazards h_k = sigmoid(f(x_k)).
- The resulting exit-time distribution within a trade is:
    p(exit=k) = h_k * Π_{j<k} (1 - h_j)   for k=1..hold_min-1
    p(exit=hold_min) = Π_{j<hold_min} (1 - h_j)  (forced exit at horizon)
- Objective: maximize expected net return within the horizon:
    E[ret] = Σ_k p(exit=k) * ret_if_exit_now_pct(k)
  where ret_if_exit_now_pct(k) is the realized net return if we exited at minute k.

Outputs
- Writes a model artifact under data/exit_oracle/
- Writes an evaluation CSV under data/exit_oracle/

Notes
- This is offline policy learning. It uses realized returns for exits at each k (supervision),
  not the oracle exit labels.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


@dataclass(frozen=True)
class Split:
    train_ids: np.ndarray
    test_ids: np.ndarray


class TradeDataset(Dataset):
    def __init__(self, X: np.ndarray, R: np.ndarray):
        # X: [N, K, F], R: [N, K]
        self.X = torch.tensor(X, dtype=torch.float32)
        self.R = torch.tensor(R, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.R[idx]


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
        # x: [B, K, F] -> logits: [B, K]
        b, k, f = x.shape
        y = self.net(x.reshape(b * k, f))
        return y.reshape(b, k)


def expected_return_from_logits(logits: torch.Tensor, rets: torch.Tensor) -> torch.Tensor:
    """Compute expected return per trade given hazard logits.

    logits: [B, K]
    rets:   [B, K]

    h_k = sigmoid(logit_k)
    p_k = h_k * Π_{j<k} (1 - h_j) for k<K
    p_K = Π_{j<K} (1 - h_j)

    Expected return = Σ_k p_k * ret_k
    """

    h = torch.sigmoid(logits)
    b, K = h.shape

    # survival_{k} = Π_{j<k} (1 - h_j)  (survival before deciding at k)
    surv = torch.ones((b,), dtype=h.dtype, device=h.device)
    exp_ret = torch.zeros((b,), dtype=rets.dtype, device=rets.device)

    for k in range(K):
        if k < K - 1:
            p = surv * h[:, k]
            exp_ret = exp_ret + p * rets[:, k]
            surv = surv * (1.0 - h[:, k])
        else:
            # forced exit at horizon
            p = surv
            exp_ret = exp_ret + p * rets[:, k]

    return exp_ret


def simulate_threshold_policy(logits: np.ndarray, rets: np.ndarray, taus: List[float]) -> pd.DataFrame:
    """Deterministic policy: exit at first k where sigmoid(logit)>=tau else hold to K.

    logits: [N, K]
    rets: [N, K]
    """

    h = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    N, K = h.shape

    rows: List[Dict[str, float]] = []

    # baseline hold
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

    # expected-return (stochastic) under hazards
    # compute exact expectation:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Train profitability-optimized hazard exit policy")
    ap.add_argument(
        "--dataset-parquet",
        default="data/exit_oracle/exit_oracle_dataset_hold15_frac0p01_2026-01-03T15-01-09Z.parquet",
        help="Oracle-exit dataset parquet (per-minute in-trade rows)",
    )
    ap.add_argument("--hold-min", type=int, default=15)

    ap.add_argument("--test-trade-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-trades", type=int, default=512)

    ap.add_argument(
        "--taus",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,0.90",
    )

    ap.add_argument("--out-dir", default="data/exit_oracle")

    args = ap.parse_args()
    set_seed(int(args.seed))

    ds_path = Path(args.dataset_parquet)
    if not ds_path.exists():
        raise SystemExit(f"dataset not found: {ds_path}")

    hold_min = int(args.hold_min)
    if hold_min <= 1:
        raise SystemExit("--hold-min must be >= 2")

    print(f"Loading dataset: {ds_path}", flush=True)
    df = pd.read_parquet(ds_path)

    need_cols = {"trade_id", "k_rel", "ret_if_exit_now_pct"}
    if not need_cols.issubset(df.columns):
        raise SystemExit(f"dataset missing required columns: {sorted(list(need_cols - set(df.columns)))}")

    # feature columns = numeric columns excluding known meta/labels
    drop_cols = {
        "trade_id",
        "k_rel",
        "entry_time",
        "oracle_k",
        "oracle_ret_pct",
        "ret_if_exit_now_pct",
        "y_oracle_exit",
        "created_utc",
    }
    feat_cols = [c for c in df.columns if c not in drop_cols]

    # Keep only numeric feature cols
    good_feat_cols: List[str] = []
    for c in feat_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            good_feat_cols.append(str(c))
    feat_cols = good_feat_cols

    if not feat_cols:
        raise SystemExit("No numeric feature columns found")

    # ensure we have complete sequences per trade
    df = df.sort_values(["trade_id", "k_rel"]).reset_index(drop=True)

    # Keep only k_rel in [1..hold_min]
    df = df[(df["k_rel"] >= 1) & (df["k_rel"] <= hold_min)].copy()

    # group sizes
    sizes = df.groupby("trade_id")["k_rel"].count()
    full_ids = sizes[sizes == hold_min].index.to_numpy()
    if len(full_ids) == 0:
        raise SystemExit("No full-length trades (unexpected)")

    df = df[df["trade_id"].isin(full_ids)].copy().reset_index(drop=True)

    # time-ordered split by entry_time if available, else by trade_id numeric
    if "entry_time" in df.columns:
        order = df[["trade_id", "entry_time"]].drop_duplicates("trade_id").sort_values("entry_time")
    else:
        order = df[["trade_id"]].drop_duplicates("trade_id").sort_values("trade_id")

    trade_ids = order["trade_id"].to_numpy()
    n_tr = len(trade_ids)
    n_test = max(1, int(n_tr * float(args.test_trade_frac)))

    split = Split(train_ids=trade_ids[: n_tr - n_test], test_ids=trade_ids[n_tr - n_test :])

    # Build tensors [N, K, F] and [N, K]
    def build_xy(ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d = df[df["trade_id"].isin(ids)].copy()
        # add time feature
        d["k_norm"] = d["k_rel"].astype(np.float32) / float(hold_min)
        cols = feat_cols + ["k_norm"]

        # pivot into [trade, k]
        X = d.pivot(index="trade_id", columns="k_rel", values=cols)
        R = d.pivot(index="trade_id", columns="k_rel", values="ret_if_exit_now_pct")

        # ensure columns 1..K
        X = X.reindex(columns=pd.MultiIndex.from_product([cols, range(1, hold_min + 1)]))
        R = R.reindex(columns=range(1, hold_min + 1))

        Xv = X.to_numpy(dtype=np.float32).reshape(len(ids), len(cols), hold_min).transpose(0, 2, 1)
        Rv = R.to_numpy(dtype=np.float32)

        # sanitize
        Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)
        Rv = np.nan_to_num(Rv, nan=0.0, posinf=0.0, neginf=0.0)
        return Xv, Rv

    X_tr, R_tr = build_xy(split.train_ids)
    X_te, R_te = build_xy(split.test_ids)

    # standardize features based on train (per feature dimension)
    mu = X_tr.reshape(-1, X_tr.shape[-1]).mean(axis=0)
    sd = X_tr.reshape(-1, X_tr.shape[-1]).std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd

    print(f"Trades train={len(split.train_ids)} test={len(split.test_ids)}  hold_min={hold_min}  n_feat={X_tr.shape[-1]}")

    # baselines from data
    hold_ret = R_te[:, hold_min - 1]
    oracle_ret = None
    if "oracle_ret_pct" in df.columns and "trade_id" in df.columns:
        o = df[df["trade_id"].isin(split.test_ids)][["trade_id", "oracle_ret_pct"]].drop_duplicates("trade_id")
        oracle_ret = o["oracle_ret_pct"].to_numpy(np.float64)

    print(f"Test baseline hold mean={float(np.mean(hold_ret)):.4f}%  median={float(np.median(hold_ret)):.4f}%")
    if oracle_ret is not None and len(oracle_ret) == len(split.test_ids):
        print(f"Test oracle   mean={float(np.mean(oracle_ret)):.4f}%  median={float(np.median(oracle_ret)):.4f}%")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PerStepMLP(d_in=int(X_tr.shape[-1]), d_hidden=int(args.hidden), dropout=float(args.dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_loader = DataLoader(TradeDataset(X_tr, R_tr), batch_size=int(args.batch_trades), shuffle=True, drop_last=False)

    model.train()
    for ep in range(1, int(args.epochs) + 1):
        losses = []
        exp_rets = []
        for xb, rb in train_loader:
            xb = xb.to(device)
            rb = rb.to(device)
            logits = model(xb)
            exp = expected_return_from_logits(logits, rb)
            loss = -torch.mean(exp)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu().item()))
            exp_rets.append(float(exp.detach().mean().cpu().item()))

        print(f"epoch {ep:02d}  loss={np.mean(losses):.6f}  exp_ret_train={np.mean(exp_rets):.6f}%", flush=True)

    # Evaluate on test
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_te, dtype=torch.float32, device=device)
        rb = torch.tensor(R_te, dtype=torch.float32, device=device)
        logits = model(xb).detach().cpu().numpy().astype(np.float64)

    taus = [float(x.strip()) for x in str(args.taus).split(",") if x.strip()]
    eval_df = simulate_threshold_policy(logits, R_te.astype(np.float64), taus=taus)

    print("\n=== Policy evaluation (TEST trades) ===")
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(eval_df.to_string(index=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = now_ts()

    model_path = out_dir / f"exit_profit_policy_hold{hold_min}_frac0p01_{ts}.pt"
    meta_path = out_dir / f"exit_profit_policy_hold{hold_min}_frac0p01_{ts}.json"
    eval_path = out_dir / f"exit_profit_policy_eval_hold{hold_min}_frac0p01_{ts}.csv"

    torch.save({"state_dict": model.state_dict()}, model_path)
    eval_df.to_csv(eval_path, index=False)

    meta = {
        "created_utc": ts,
        "dataset_parquet": str(ds_path.resolve()),
        "hold_min": int(hold_min),
        "feature_cols": feat_cols + ["k_norm"],
        "standardize": {"mean": mu.tolist(), "std": sd.tolist()},
        "model": {
            "type": "PerStepMLP",
            "hidden": int(args.hidden),
            "dropout": float(args.dropout),
        },
        "train": {
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "batch_trades": int(args.batch_trades),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "device": str(device),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSaved model: {model_path}")
    print(f"Saved meta:  {meta_path}")
    print(f"Saved eval:  {eval_path}")


if __name__ == "__main__":
    main()
