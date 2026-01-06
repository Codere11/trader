#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T15:29:29Z
"""Analyze liquidation (wick) trades for ETH-USD under our entry+exit pipeline.

Goal
- Identify which trades liquidate under DO_NOT_TOUCH-like bankroll rules.
- Compare liquidated vs non-liquidated trades and rank "warning" signals.
- Produce artifacts that help decide: entry filter, exit aggressiveness, or leverage gating.

Inputs
- ETH oracle-exit dataset (per-minute rows within trade) with:
  trade_id, k_rel, entry_time, entry_pred, ret_if_exit_now_pct, pred_gap_pct features, etc.
- ETH 1m market CSV (timestamp, open, low) to detect wick liquidation.

Bankroll model
- Same tiered leverage schedule as DO_NOT_TOUCH.txt (100x/50x/20x/10x/1x).
- Withdraw 50% of profit into bank on winning trades at 100x and 50x.
- Refill rule (override-able): if balance < refill_min_balance and bank < 150 => external top-up to refill_floor.
- If bank >= 150 and balance < refill_min_balance => deposit 20% of bank into balance.

Liquidation model
- Long-only liquidation threshold (simple): equity_mult <= 0.
  Equivalent liq_price for candle lows:
    liq_price = entry_open*(1+fee_side)*(1-1/L)/(1-fee_side)
  Liquidates on first minute where low <= liq_price within [entry_min .. entry_min+exit_k].

Outputs
- data/liquidation_analysis_ethusd_<ts>/
  - per_trade.csv (per-trade simulation incl. liq details)
  - entry_univariate_auc.csv (warning ranks using entry features)
  - early_univariate_auc.csv (warning ranks using early-trade features)
  - summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _safe_float(x) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return float("nan")


def auc_mann_whitney(y: np.ndarray, s: np.ndarray) -> float:
    """AUC via Mann–Whitney U; handles NaNs by dropping them."""
    y = np.asarray(y)
    s = np.asarray(s)
    m = np.isfinite(s) & np.isfinite(y)
    y = y[m].astype(int)
    s = s[m].astype(float)
    if y.size == 0:
        return float("nan")
    if len(np.unique(y)) < 2:
        return float("nan")

    # ranks with average for ties
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)

    # tie correction: average ranks within ties
    # (stable grouping by sorted values)
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i + 1
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sum_ranks_pos = float(ranks[y == 1].sum())
    # U statistic for positives
    u = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    auc = u / (n_pos * n_neg)
    return float(auc)


@dataclass(frozen=True)
class Strat:
    tiers: List[Tuple[float, float]]  # (equity_threshold, leverage)
    withdraw_half_levers: set[float]

    start_balance: float
    refill_min_balance: float
    refill_floor: float

    bank_threshold: float
    bank_fraction: float


def parse_do_not_touch(path: Path) -> Strat:
    txt = path.read_text(encoding="utf-8")

    lev_pat = re.compile(r"(\d+)x leverage until account reaches ([\d,]+)EUR")
    tiers: List[Tuple[float, float]] = []
    for m in lev_pat.finditer(txt):
        lev = float(m.group(1))
        thr = float(m.group(2).replace(",", ""))
        tiers.append((thr, lev))
    tiers.sort(key=lambda x: x[0])
    if tiers:
        tiers.append((float("inf"), 1.0))

    withdraw_half_levers = {100.0, 50.0}

    m = re.search(r"Starting trading balance deposited: (\d+(?:\.\d+)?)EUR", txt)
    start_balance = float(m.group(1)) if m else 30.0

    return Strat(
        tiers=tiers,
        withdraw_half_levers=withdraw_half_levers,
        start_balance=start_balance,
        refill_min_balance=5.0,
        refill_floor=start_balance,
        bank_threshold=150.0,
        bank_fraction=0.20,
    )


def leverage_for_equity(equity: float, tiers: List[Tuple[float, float]]) -> float:
    for thr, lev in tiers:
        if float(equity) < float(thr):
            return float(lev)
    return 1.0


def liq_price_long(entry_open: float, *, fee_side: float, lev: float) -> float:
    if not (math.isfinite(entry_open) and entry_open > 0.0):
        return float("nan")
    if lev <= 1.0:
        return float("nan")
    f = float(fee_side)
    return float(entry_open * (1.0 + f) * (1.0 - 1.0 / float(lev)) / max(1e-12, (1.0 - f)))


def map_ts_to_index(ts_sorted: np.ndarray, query: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(ts_sorted, query)
    ok = (pos < ts_sorted.size) & (ts_sorted[pos] == query)
    return np.where(ok, pos, -1).astype(np.int64)


def build_trade_table(
    *,
    ds: pd.DataFrame,
    feat_cols: List[str],
    reg,
    hold_min: int,
    min_exit_k: int,
    tau_gap: float,
) -> pd.DataFrame:
    """From per-minute oracle dataset produce per-trade rows with chosen exit_k and return."""
    need = {"trade_id", "k_rel", "entry_time", "entry_pred", "ret_if_exit_now_pct"}
    if not need.issubset(ds.columns):
        raise SystemExit(f"dataset missing required columns: {sorted(list(need - set(ds.columns)))}")

    ds = ds[(ds["k_rel"] >= 1) & (ds["k_rel"] <= int(hold_min))].copy()
    ds = ds.sort_values(["trade_id", "k_rel"]).reset_index(drop=True)

    sizes = ds.groupby("trade_id")["k_rel"].count()
    full_ids = sizes[sizes == int(hold_min)].index.to_numpy()
    ds = ds[ds["trade_id"].isin(full_ids)].copy().reset_index(drop=True)

    # ensure features present
    for c in feat_cols:
        if c not in ds.columns:
            ds[c] = np.nan

    X = ds[feat_cols].to_numpy(np.float32)
    pred_gap = reg.predict(X).astype(np.float64)
    ds["pred_gap_pct"] = pred_gap

    rows: List[Dict[str, object]] = []
    for tid, g in ds.groupby("trade_id", sort=False):
        g2 = g.sort_values("k_rel")
        entry_time = pd.to_datetime(g2["entry_time"].iloc[0], utc=True)
        entry_pred = float(g2["entry_pred"].iloc[0])
        gap_arr = g2["pred_gap_pct"].to_numpy(np.float64)
        ret_arr = g2["ret_if_exit_now_pct"].to_numpy(np.float64)

        k_exit = int(hold_min)
        for k in range(int(min_exit_k), int(hold_min) + 1):
            v = float(gap_arr[k - 1])
            if np.isfinite(v) and v <= float(tau_gap):
                k_exit = int(k)
                break

        r1x = float(ret_arr[k_exit - 1])

        # entry features: take k_rel=1 row as "at/just after entry" proxy
        entry_feat = g2.iloc[0]
        rec: Dict[str, object] = {
            "trade_id": int(tid),
            "entry_time": entry_time,
            "entry_pred": float(entry_pred),
            "exit_k": int(k_exit),
            "realized_ret_pct_1x": float(r1x),
        }

        for c in feat_cols:
            rec[f"entryfeat__{c}"] = _safe_float(entry_feat.get(c))

        # early features: k_rel=2 and k_rel=3 if available
        for k in (2, 3):
            if int(hold_min) >= k:
                rowk = g2.iloc[k - 1]
                # common early state features if present
                for c in [
                    "delta_mark_pct",
                    "delta_mark_prev1_pct",
                    "delta_mark_prev2_pct",
                    "delta_mark_change_1m",
                    "delta_mark_change_2m",
                    "drawdown_from_peak_pct",
                ]:
                    if c in rowk.index:
                        rec[f"k{k}__{c}"] = _safe_float(rowk[c])

        rows.append(rec)

    return pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)


def simulate_bankroll_and_liqs(
    *,
    trades: pd.DataFrame,
    market_ts: np.ndarray,
    market_open: np.ndarray,
    market_low: np.ndarray,
    strat: Strat,
    fee_side: float,
) -> pd.DataFrame:
    """Simulate bankroll and label wick liquidation trades."""

    out = trades.copy().sort_values("entry_time").reset_index(drop=True)
    out["entry_min"] = pd.to_datetime(out["entry_time"], utc=True).dt.floor("min").to_numpy(dtype="datetime64[ns]")
    idx0 = map_ts_to_index(market_ts, out["entry_min"].to_numpy(dtype="datetime64[ns]"))
    out["market_idx0"] = idx0

    balance = float(strat.start_balance)
    bank = 0.0
    ext_topups_total = float(strat.start_balance)  # include initial deposit

    records = []

    for r in out.itertuples(index=False):
        t = pd.to_datetime(r.entry_time, utc=True)
        k = int(r.exit_k)
        ret_pct_1x = float(r.realized_ret_pct_1x)

        bal_before = float(balance)
        bank_before = float(bank)

        lev = leverage_for_equity(balance, strat.tiers)

        # wick liquidation detection
        liq = False
        liq_reason = ""
        liq_px = float("nan")
        min_low = float("nan")
        breach_rel = -1

        j0 = int(r.market_idx0)
        if j0 >= 0:
            entry_open = float(market_open[j0])
            liq_px = liq_price_long(entry_open, fee_side=float(fee_side), lev=float(lev))

            j1 = min(len(market_low) - 1, j0 + max(0, int(k)))
            window = market_low[j0 : j1 + 1]
            if window.size:
                min_low = float(np.min(window))
                if np.isfinite(liq_px):
                    breach = np.where(window <= liq_px)[0]
                    if breach.size > 0:
                        liq = True
                        liq_reason = "wick"
                        breach_rel = int(breach[0])

        # close-hard backstop
        if (not liq) and (1.0 + (ret_pct_1x / 100.0) * float(lev) <= 0.0):
            liq = True
            liq_reason = "close"

        if liq:
            pnl = -bal_before
            balance = 0.0
        else:
            pnl = bal_before * (ret_pct_1x / 100.0) * float(lev)
            balance = bal_before + pnl

        withdrawn = 0.0
        if (not liq) and (float(lev) in strat.withdraw_half_levers) and pnl > 0.0:
            withdrawn = 0.5 * pnl
            balance -= withdrawn
            bank += withdrawn

        ext_deposit = 0.0
        bank_deposit = 0.0
        if balance < float(strat.refill_min_balance):
            if bank < float(strat.bank_threshold):
                ext_deposit = max(0.0, float(strat.refill_floor) - balance)
                balance += ext_deposit
                ext_topups_total += ext_deposit
            else:
                bank_deposit = float(strat.bank_fraction) * bank
                bank -= bank_deposit
                balance += bank_deposit

        equity = balance + bank

        rec = {
            "trade_id": int(r.trade_id),
            "entry_time": t,
            "entry_pred": float(r.entry_pred),
            "exit_k": int(k),
            "realized_ret_pct_1x": float(ret_pct_1x),
            "lev": float(lev),
            "balance_before": bal_before,
            "bank_before": bank_before,
            "pnl": float(pnl),
            "liquidated": bool(liq),
            "liquidation_reason": liq_reason,
            "liq_price": float(liq_px) if np.isfinite(liq_px) else float("nan"),
            "min_low": float(min_low) if np.isfinite(min_low) else float("nan"),
            "breach_rel_min": int(breach_rel),
            "withdrawn": float(withdrawn),
            "ext_deposit": float(ext_deposit),
            "bank_deposit": float(bank_deposit),
            "balance": float(balance),
            "bank": float(bank),
            "equity": float(equity),
            "ext_total_incl_start": float(ext_topups_total),
        }

        # attach original entry + early features columns (prefixed already)
        for c in out.columns:
            if c.startswith("entryfeat__") or c.startswith("k2__") or c.startswith("k3__"):
                rec[c] = _safe_float(getattr(r, c))

        records.append(rec)

    return pd.DataFrame(records).sort_values("entry_time").reset_index(drop=True)


def rank_univariate(df: pd.DataFrame, *, y_col: str, feat_cols: List[str]) -> pd.DataFrame:
    y = df[y_col].to_numpy(np.int32)
    rows = []
    for c in feat_cols:
        s = df[c].to_numpy(np.float64)
        auc = auc_mann_whitney(y, s)
        if np.isfinite(auc):
            auc2 = max(auc, 1.0 - auc)  # directionless
        else:
            auc2 = float("nan")
        rows.append(
            {
                "feature": c,
                "auc": float(auc),
                "auc_abs": float(auc2),
                "mean_liq": float(np.nanmean(s[y == 1])) if (y == 1).any() else float("nan"),
                "mean_ok": float(np.nanmean(s[y == 0])) if (y == 0).any() else float("nan"),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("auc_abs", ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze wick liquidations for ETH-USD and rank warning signals")
    ap.add_argument(
        "--dataset-parquet",
        default="data/exit_oracle/exit_oracle_dataset_ETH-USD_pre5m_hold15_thr0p7_2026-01-04T14-55-44Z.parquet",
    )
    ap.add_argument(
        "--exit-model",
        default="data/exit_oracle/exit_oracle_gap_regressor_hold15_2026-01-04T14-57-25Z.joblib",
    )
    ap.add_argument("--market-csv", default="data/dydx_ETH-USD_1MIN_full_2026-01-03T23-48-53Z.csv")

    ap.add_argument("--entry-threshold", type=float, default=0.998)
    ap.add_argument("--tau-gap", type=float, default=0.75)
    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--min-exit-k", type=int, default=2)

    ap.add_argument("--fee-side", type=float, default=0.0005)

    # bankroll overrides
    ap.add_argument("--start-balance", type=float, default=10.0)
    ap.add_argument("--refill-floor", type=float, default=10.0)
    ap.add_argument("--refill-min-balance", type=float, default=5.0)

    # leverage override: replace the first tier leverage (e.g. make "100x" become "50x")
    ap.add_argument(
        "--first-tier-leverage",
        type=float,
        default=None,
        help="If set, overrides the leverage used in the first DO_NOT_TOUCH tier (e.g. 50 to replace 100x).",
    )

    ap.add_argument("--do-not-touch", default="DO_NOT_TOUCH.txt")

    ap.add_argument("--out-dir", default="data")

    args = ap.parse_args()

    # Load market
    mkt = pd.read_csv(args.market_csv, parse_dates=["timestamp"], usecols=["timestamp", "open", "low"])
    mkt["timestamp"] = mkt["timestamp"].dt.floor("min")
    mkt = mkt.drop_duplicates("timestamp", keep="first")
    if not mkt["timestamp"].is_monotonic_increasing:
        mkt = mkt.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    ts_sorted = mkt["timestamp"].to_numpy(dtype="datetime64[ns]")
    openp = mkt["open"].to_numpy(np.float64, copy=False)
    lowp = mkt["low"].to_numpy(np.float64, copy=False)

    # Load exit model
    art = joblib.load(args.exit_model)
    reg = art["model"] if isinstance(art, dict) else art
    feat_cols = list(art.get("feature_cols") if isinstance(art, dict) else [])

    # Load oracle dataset
    ds = pd.read_parquet(args.dataset_parquet)

    # Filter by entry threshold
    if "entry_pred" not in ds.columns:
        raise SystemExit("dataset missing entry_pred")

    ds = ds[pd.to_numeric(ds["entry_pred"], errors="coerce") >= float(args.entry_threshold)].copy()
    if ds.empty:
        raise SystemExit("no rows after entry threshold filter")

    # Build per-trade list with chosen exit
    trades = build_trade_table(
        ds=ds,
        feat_cols=feat_cols,
        reg=reg,
        hold_min=int(args.hold_min),
        min_exit_k=int(args.min_exit_k),
        tau_gap=float(args.tau_gap),
    )

    # Bankroll rules
    base_strat = parse_do_not_touch(Path(args.do_not_touch))

    tiers = list(base_strat.tiers)
    if args.first_tier_leverage is not None and len(tiers) > 0:
        thr0, _lev0 = tiers[0]
        tiers[0] = (float(thr0), float(args.first_tier_leverage))

    strat = Strat(
        tiers=tiers,
        withdraw_half_levers=base_strat.withdraw_half_levers,
        start_balance=float(args.start_balance),
        refill_min_balance=float(args.refill_min_balance),
        refill_floor=float(args.refill_floor),
        bank_threshold=base_strat.bank_threshold,
        bank_fraction=base_strat.bank_fraction,
    )

    sim = simulate_bankroll_and_liqs(
        trades=trades,
        market_ts=ts_sorted,
        market_open=openp,
        market_low=lowp,
        strat=strat,
        fee_side=float(args.fee_side),
    )

    out_root = Path(args.out_dir) / f"liquidation_analysis_ethusd_{now_ts()}"
    out_root.mkdir(parents=True, exist_ok=True)

    per_trade_path = out_root / "per_trade.csv"
    sim.to_csv(per_trade_path, index=False)

    # Feature ranking
    y = "liquidated"
    entry_feats = [c for c in sim.columns if c.startswith("entryfeat__")]
    early_feats = [c for c in sim.columns if c.startswith("k2__") or c.startswith("k3__")]

    entry_rank = rank_univariate(sim, y_col=y, feat_cols=entry_feats)
    early_rank = rank_univariate(sim, y_col=y, feat_cols=early_feats)

    entry_rank_path = out_root / "entry_univariate_auc.csv"
    early_rank_path = out_root / "early_univariate_auc.csv"

    entry_rank.to_csv(entry_rank_path, index=False)
    early_rank.to_csv(early_rank_path, index=False)

    # Summary
    liq = sim[sim["liquidated"] == True]
    summary = {
        "created_utc": now_ts(),
        "entry_threshold": float(args.entry_threshold),
        "tau_gap": float(args.tau_gap),
        "start_balance": float(strat.start_balance),
        "refill_floor": float(strat.refill_floor),
        "refill_min_balance": float(strat.refill_min_balance),
        "fee_side": float(args.fee_side),
        "n_trades": int(len(sim)),
        "n_liquidations": int(len(liq)),
        "liq_by_lev": liq["lev"].value_counts().sort_index().to_dict(),
        "external_total_incl_start_end": float(sim["ext_total_incl_start"].iloc[-1]) if len(sim) else float("nan"),
        "final_equity": float(sim["equity"].iloc[-1]) if len(sim) else float("nan"),
        "top_entry_features": entry_rank.head(15).to_dict(orient="records"),
        "top_early_features": early_rank.head(15).to_dict(orient="records"),
    }

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Wrote:", out_root)
    print(" per_trade:", per_trade_path)
    print(" entry_rank:", entry_rank_path)
    print(" early_rank:", early_rank_path)
    print(" summary:", out_root / "summary.json")
    print("Liquidations:", int(len(liq)), "of", int(len(sim)))


if __name__ == "__main__":
    main()
