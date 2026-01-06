#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T15:57:29Z
"""Joint sweep: ETH-USD entry+exit under DO_NOT_TOUCH with 50x first tier.

What this does
- Loads the ETH oracle-exit dataset (per-minute rows within each trade).
- Loads the trained exit oracle-gap regressor and computes pred_gap_pct for every row.
- Builds per-trade arrays (entry_time, entry_pred, per-minute ret_if_exit_now_pct, per-minute pred_gap_pct).
- Sweeps:
  - entry_threshold: keep trades with entry_pred >= threshold
  - tau_gap: exit at first k>=min_exit_k where pred_gap_pct <= tau_gap else hold_min
  - min_exit_k
- Simulates DO_NOT_TOUCH bankroll with wick liquidation, but overrides first tier leverage to 50x.

Output
- Writes data/sweeps_ethusd_50x_joint_<ts>/results.csv
- Writes data/sweeps_ethusd_50x_joint_<ts>/top_configs.json

Notes
- This is not retraining models yet; it is hyperparameter tuning of the *policy* that couples entry selection + exit policy.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


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

    # Defaults in DO_NOT_TOUCH; caller typically overrides start/refill for ETH.
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


def _map_ts_to_index(ts_sorted: np.ndarray, query: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(ts_sorted, query)
    ok = (pos < ts_sorted.size) & (ts_sorted[pos] == query)
    return np.where(ok, pos, -1).astype(np.int64)


def pick_exit_k_for_trade(pred_gap: np.ndarray, *, hold_min: int, min_exit_k: int, tau_gap: float) -> int:
    # pred_gap indexed 0..hold_min-1 representing k=1..hold_min
    for k in range(int(min_exit_k), int(hold_min) + 1):
        v = float(pred_gap[k - 1])
        if np.isfinite(v) and v <= float(tau_gap):
            return int(k)
    return int(hold_min)


def build_trade_mats(
    *,
    ds: pd.DataFrame,
    feat_cols: List[str],
    reg,
    hold_min: int,
) -> Dict[str, object]:
    """Return per-trade matrices and metadata, ordered by entry_time.

    Returns
    - trade_ids: (N,)
    - entry_time: (N,) datetime64[ns]
    - entry_pred: (N,)
    - R: (N, hold_min) ret_if_exit_now_pct
    - G: (N, hold_min) pred_gap_pct
    """

    need = {"trade_id", "k_rel", "entry_time", "entry_pred", "ret_if_exit_now_pct"}
    missing = sorted(list(need - set(ds.columns)))
    if missing:
        raise SystemExit(f"dataset missing required columns: {missing}")

    d = ds[(ds["k_rel"] >= 1) & (ds["k_rel"] <= int(hold_min))].copy()
    d["entry_time"] = pd.to_datetime(d["entry_time"], utc=True)
    d = d.sort_values(["trade_id", "k_rel"]).reset_index(drop=True)

    # keep only full-length trades
    sizes = d.groupby("trade_id")["k_rel"].count()
    full_ids = sizes[sizes == int(hold_min)].index.to_numpy()
    d = d[d["trade_id"].isin(full_ids)].copy().reset_index(drop=True)

    # ensure features exist
    for c in feat_cols:
        if c not in d.columns:
            d[c] = np.nan

    X = d[feat_cols].to_numpy(np.float32)
    pred_gap = reg.predict(X).astype(np.float64)
    d["pred_gap_pct"] = pred_gap

    # order trades by entry_time
    order = d[["trade_id", "entry_time", "entry_pred"]].drop_duplicates("trade_id").sort_values("entry_time")
    trade_ids = order["trade_id"].to_numpy(np.int64)
    entry_time = order["entry_time"].to_numpy(dtype="datetime64[ns]")
    entry_pred = pd.to_numeric(order["entry_pred"], errors="coerce").to_numpy(np.float64)

    # pivot R and G
    R = (
        d.pivot(index="trade_id", columns="k_rel", values="ret_if_exit_now_pct")
        .reindex(index=trade_ids, columns=range(1, hold_min + 1))
        .to_numpy(np.float64)
    )
    G = (
        d.pivot(index="trade_id", columns="k_rel", values="pred_gap_pct")
        .reindex(index=trade_ids, columns=range(1, hold_min + 1))
        .to_numpy(np.float64)
    )

    # NaN safety: treat missing preds as +inf (never triggers), missing returns as 0.
    G = np.where(np.isfinite(G), G, np.inf)
    R = np.where(np.isfinite(R), R, 0.0)

    return {
        "trade_ids": trade_ids,
        "entry_time": entry_time,
        "entry_pred": entry_pred,
        "R": R,
        "G": G,
    }


def precompute_market_windows(
    *,
    market_csv: Path,
    entry_time: np.ndarray,
    hold_min: int,
) -> Dict[str, object]:
    """Precompute entry_index, entry_open, and cumulative min lows per trade for k=0..hold_min."""

    mkt = pd.read_csv(market_csv, parse_dates=["timestamp"], usecols=["timestamp", "open", "low"])
    mkt["timestamp"] = mkt["timestamp"].dt.floor("min")
    mkt = mkt.drop_duplicates("timestamp", keep="first")
    if not mkt["timestamp"].is_monotonic_increasing:
        mkt = mkt.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    ts_sorted = mkt["timestamp"].to_numpy(dtype="datetime64[ns]")
    openp = mkt["open"].to_numpy(np.float64, copy=False)
    lowp = mkt["low"].to_numpy(np.float64, copy=False)

    entry_min = pd.to_datetime(entry_time, utc=True).floor("min").to_numpy(dtype="datetime64[ns]")
    idx0 = _map_ts_to_index(ts_sorted, entry_min)

    N = int(len(idx0))
    K = int(hold_min)

    entry_open = np.full((N,), np.nan, dtype=np.float64)
    min_low_cum = np.full((N, K + 1), np.nan, dtype=np.float64)  # k=0..K

    for i in range(N):
        j0 = int(idx0[i])
        if j0 < 0:
            continue
        entry_open[i] = float(openp[j0])

        j1 = min(len(lowp) - 1, j0 + K)
        window = lowp[j0 : j1 + 1]
        if window.size == 0:
            continue
        # cumulative min over window length (<=K+1)
        cum = np.minimum.accumulate(window.astype(np.float64, copy=False))
        # pad to K+1 with last value if needed
        if cum.size < K + 1:
            pad = np.full((K + 1 - cum.size,), float(cum[-1]), dtype=np.float64)
            cum = np.concatenate([cum, pad], axis=0)
        min_low_cum[i, :] = cum[: K + 1]

    return {
        "idx0": idx0,
        "entry_open": entry_open,
        "min_low_cum": min_low_cum,
        "ts_sorted": ts_sorted,
    }


def simulate_do_not_touch(
    *,
    entry_time: np.ndarray,
    entry_pred: np.ndarray,
    R: np.ndarray,
    G: np.ndarray,
    market_idx0: np.ndarray,
    entry_open: np.ndarray,
    min_low_cum: np.ndarray,
    hold_min: int,
    min_exit_k: int,
    tau_gap: float,
    entry_threshold: float,
    strat: Strat,
    fee_side: float,
) -> Dict[str, object]:
    """Simulate bankroll for a given (entry_threshold, tau_gap, min_exit_k)."""

    # Select trades by entry_threshold
    mask = np.isfinite(entry_pred) & (entry_pred >= float(entry_threshold))
    if not np.any(mask):
        return {
            "entry_threshold": float(entry_threshold),
            "tau_gap": float(tau_gap),
            "min_exit_k": int(min_exit_k),
            "n_trades": 0,
        }

    idx = np.where(mask)[0]

    t = entry_time[idx]
    Rm = R[idx]
    Gm = G[idx]
    idx0 = market_idx0[idx]
    eopen = entry_open[idx]
    minlow = min_low_cum[idx]

    n = int(len(idx))

    # Precompute exit_k and ret at exit for this policy
    K = int(hold_min)
    exit_k = np.full((n,), K, dtype=np.int64)
    # brute per trade; K small and n ~ O(1k)
    for i in range(n):
        exit_k[i] = pick_exit_k_for_trade(Gm[i], hold_min=K, min_exit_k=int(min_exit_k), tau_gap=float(tau_gap))

    # ret at chosen exit
    ret_1x = Rm[np.arange(n), exit_k - 1].astype(np.float64)

    # Trade frequency stats
    # Note: pd.to_datetime on a numpy array returns a DatetimeIndex.
    t_idx = pd.to_datetime(t, utc=True)
    t0 = t_idx.min()
    t1 = t_idx.max()
    cal_days = int((t1.floor("D") - t0.floor("D")).days + 1) if n else 0
    trade_days = int(pd.Index(t_idx.date).nunique()) if n else 0

    # Bankroll simulation
    balance = float(strat.start_balance)
    bank = 0.0
    ext_total = float(strat.start_balance)  # include initial deposit

    n_liq = 0
    n_init_liq = 0
    n_ext_topups = 0
    ext_topups_only = 0.0
    bank150_first = None

    for i in range(n):
        bal_before = float(balance)
        bank_before = float(bank)

        lev = leverage_for_equity(balance, strat.tiers)

        # liquidation checks
        liq = False

        # wick: compare min low over window [0..exit_k] to liq price
        j0 = int(idx0[i])
        if j0 >= 0 and np.isfinite(eopen[i]):
            liq_px = liq_price_long(float(eopen[i]), fee_side=float(fee_side), lev=float(lev))
            if np.isfinite(liq_px):
                # min low within first exit_k minutes (inclusive of entry minute) => cum index exit_k
                ml = float(minlow[i, int(exit_k[i])]) if np.isfinite(minlow[i, int(exit_k[i])]) else float("nan")
                if np.isfinite(ml) and ml <= liq_px:
                    liq = True

        # close-hard backstop
        if not liq:
            if (1.0 + (float(ret_1x[i]) / 100.0) * float(lev)) <= 0.0:
                liq = True

        if liq:
            n_liq += 1
            if bank_before < float(strat.bank_threshold):
                n_init_liq += 1
            pnl = -bal_before
            balance = 0.0
        else:
            pnl = bal_before * (float(ret_1x[i]) / 100.0) * float(lev)
            balance = bal_before + pnl

        # withdraw half of profit at 50x/100x per DO_NOT_TOUCH
        if (not liq) and (float(lev) in strat.withdraw_half_levers) and pnl > 0.0:
            withdrawn = 0.5 * pnl
            balance -= withdrawn
            bank += withdrawn

        # refinance
        if balance < float(strat.refill_min_balance):
            if bank < float(strat.bank_threshold):
                dep = max(0.0, float(strat.refill_floor) - balance)
                if dep > 0.0:
                    balance += dep
                    ext_total += dep
                    ext_topups_only += dep
                    n_ext_topups += 1
            else:
                dep = float(strat.bank_fraction) * bank
                bank -= dep
                balance += dep

        if bank150_first is None and bank >= float(strat.bank_threshold):
            bank150_first = pd.to_datetime(t_idx[i], utc=True)

    final_equity = float(balance + bank)

    return {
        "entry_threshold": float(entry_threshold),
        "tau_gap": float(tau_gap),
        "min_exit_k": int(min_exit_k),
        "hold_min": int(hold_min),
        "n_trades": int(n),
        "trade_days": int(trade_days),
        "calendar_days_inclusive": int(cal_days),
        "trades_per_trade_day": float(n / trade_days) if trade_days else 0.0,
        "trades_per_calendar_day": float(n / cal_days) if cal_days else 0.0,
        "n_liquidations": int(n_liq),
        "initial_liquidations": int(n_init_liq),
        "n_ext_topups_total": int(n_ext_topups),
        "ext_topups_only": float(ext_topups_only),
        "ext_total_incl_start_end": float(ext_total),
        "bank150_first_reached_at": bank150_first.isoformat() if bank150_first is not None else None,
        "final_equity": float(final_equity),
    }


def parse_float_list(s: str) -> List[float]:
    out: List[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep ETH-USD entry+exit params under DO_NOT_TOUCH (50x first tier)")
    ap.add_argument(
        "--dataset-parquet",
        default="data/exit_oracle/exit_oracle_dataset_ETH-USD_pre5m_hold15_thr0p7_2026-01-04T14-55-44Z.parquet",
    )
    ap.add_argument(
        "--exit-model",
        default="data/exit_oracle/exit_oracle_gap_regressor_hold15_2026-01-04T14-57-25Z.joblib",
    )
    ap.add_argument("--market-csv", default="data/dydx_ETH-USD_1MIN_full_2026-01-03T23-48-53Z.csv")

    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--fee-side", type=float, default=0.0005)

    ap.add_argument(
        "--entry-thresholds",
        default="0.70,0.80,0.90,0.95,0.96,0.97,0.98,0.99,0.995,0.996,0.997,0.998,0.999",
        help="Comma-separated entry_pred thresholds.",
    )
    ap.add_argument(
        "--tau-gaps",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.75,0.80,0.90",
        help="Comma-separated tau_gap values (gap pct points).",
    )
    ap.add_argument(
        "--min-exit-ks",
        default="1,2,3",
        help="Comma-separated min_exit_k values.",
    )

    # bankroll overrides
    ap.add_argument("--do-not-touch", default="DO_NOT_TOUCH.txt")
    ap.add_argument("--first-tier-leverage", type=float, default=50.0)
    ap.add_argument("--start-balance", type=float, default=10.0)
    ap.add_argument("--refill-floor", type=float, default=10.0)
    ap.add_argument("--refill-min-balance", type=float, default=5.0)

    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--top-n", type=int, default=25)

    args = ap.parse_args()

    hold_min = int(args.hold_min)

    entry_thresholds = parse_float_list(args.entry_thresholds)
    tau_gaps = parse_float_list(args.tau_gaps)
    min_exit_ks = parse_int_list(args.min_exit_ks)

    # Load model
    art = joblib.load(Path(args.exit_model))
    reg = art["model"] if isinstance(art, dict) else art
    feat_cols = list(art.get("feature_cols") if isinstance(art, dict) else [])

    # Load dataset
    ds = pd.read_parquet(Path(args.dataset_parquet))

    mats = build_trade_mats(ds=ds, feat_cols=feat_cols, reg=reg, hold_min=hold_min)

    entry_time = mats["entry_time"]  # np.ndarray
    entry_pred = mats["entry_pred"]  # np.ndarray
    R = mats["R"]
    G = mats["G"]

    # Market precompute
    mkt = precompute_market_windows(market_csv=Path(args.market_csv), entry_time=entry_time, hold_min=hold_min)

    # Strategy
    base = parse_do_not_touch(Path(args.do_not_touch))
    tiers = list(base.tiers)
    if tiers and args.first_tier_leverage is not None:
        thr0, _ = tiers[0]
        tiers[0] = (float(thr0), float(args.first_tier_leverage))

    strat = Strat(
        tiers=tiers,
        withdraw_half_levers=base.withdraw_half_levers,
        start_balance=float(args.start_balance),
        refill_min_balance=float(args.refill_min_balance),
        refill_floor=float(args.refill_floor),
        bank_threshold=base.bank_threshold,
        bank_fraction=base.bank_fraction,
    )

    rows: List[Dict[str, object]] = []

    total = len(entry_thresholds) * len(tau_gaps) * len(min_exit_ks)
    done = 0

    for thr in sorted(entry_thresholds):
        for tau in sorted(tau_gaps):
            for mk in sorted(min_exit_ks):
                res = simulate_do_not_touch(
                    entry_time=entry_time,
                    entry_pred=entry_pred,
                    R=R,
                    G=G,
                    market_idx0=mkt["idx0"],
                    entry_open=mkt["entry_open"],
                    min_low_cum=mkt["min_low_cum"],
                    hold_min=hold_min,
                    min_exit_k=int(mk),
                    tau_gap=float(tau),
                    entry_threshold=float(thr),
                    strat=strat,
                    fee_side=float(args.fee_side),
                )
                rows.append(res)

                done += 1
                if done % 50 == 0 or done == total:
                    print(f"progress: {done}/{total}", flush=True)

    df = pd.DataFrame(rows)

    # Rank: minimize initial_liquidations, then total liqs, then ext_total, then maximize final_equity.
    # (If a config has n_trades=0, push it to the bottom.)
    def _rank_row(r: pd.Series) -> tuple:
        n_trades = int(r.get("n_trades", 0) or 0)
        if n_trades <= 0:
            return (10**9, 10**9, float("inf"), float("inf"), 10**9)
        return (
            int(r.get("initial_liquidations", 10**9) or 10**9),
            int(r.get("n_liquidations", 10**9) or 10**9),
            float(r.get("ext_total_incl_start_end", float("inf")) or float("inf")),
            -float(r.get("final_equity", 0.0) or 0.0),
            -float(r.get("trades_per_calendar_day", 0.0) or 0.0),
        )

    df["rank_key"] = df.apply(_rank_row, axis=1)
    df = df.sort_values("rank_key", ascending=True).drop(columns=["rank_key"]).reset_index(drop=True)

    out_root = Path(args.out_dir) / f"sweeps_ethusd_50x_joint_{now_ts()}"
    out_root.mkdir(parents=True, exist_ok=True)

    results_csv = out_root / "results.csv"
    df.to_csv(results_csv, index=False)

    top_n = int(args.top_n)
    top_df = df.head(top_n).copy()
    (out_root / "top_configs.json").write_text(
        json.dumps({"created_utc": now_ts(), "top": top_df.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote: {results_csv}")
    print("\nTop configs:")
    with pd.option_context("display.max_rows", min(top_n, 50), "display.width", 200):
        cols = [
            "entry_threshold",
            "tau_gap",
            "min_exit_k",
            "n_trades",
            "n_liquidations",
            "initial_liquidations",
            "ext_total_incl_start_end",
            "final_equity",
            "trades_per_calendar_day",
            "bank150_first_reached_at",
        ]
        cols = [c for c in cols if c in top_df.columns]
        print(top_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
