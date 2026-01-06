#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T16:06:22Z
"""Guardrail sweep: ETH-USD entry+exit under DO_NOT_TOUCH (50x first tier) with usability constraints.

Goal
- Jointly tune:
  - entry_threshold (entry_pred >= threshold)
  - entry guardrails (caps on volatility/overextension features)
  - exit policy (tau_gap, min_exit_k)
- Evaluate under DO_NOT_TOUCH bankroll rules with wick liquidation and first-tier leverage overridden to 50x.
- Find configs that satisfy:
  - external topups (topups-only) <= max_initial_topups_eur
  - trades_per_calendar_day >= min_trades_per_day
  while minimizing liquidations and maximizing equity.

Implementation notes
- Uses the oracle exit dataset produced from entry selections (thr0.7 builder) and thus can only trade when
  that dataset has trades; guardrails further filter the available trades.
- External topups happen only while bank < 150 (after that, refinancing is from bank), so "topups-only"
  corresponds to "initial external topups".

Outputs
- data/sweeps_ethusd_50x_guardrails_<ts>/results.csv
- data/sweeps_ethusd_50x_guardrails_<ts>/feasible.csv  (configs meeting the constraints)
- data/sweeps_ethusd_50x_guardrails_<ts>/top_configs.json
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


def pick_exit_k(pred_gap: np.ndarray, *, hold_min: int, min_exit_k: int, tau_gap: float) -> int:
    for k in range(int(min_exit_k), int(hold_min) + 1):
        if float(pred_gap[k - 1]) <= float(tau_gap):
            return int(k)
    return int(hold_min)


def load_trade_mats(
    *,
    dataset_parquet: Path,
    exit_model: Path,
    hold_min: int,
    entry_feat_cols: List[str],
) -> Dict[str, object]:
    art = joblib.load(exit_model)
    reg = art["model"] if isinstance(art, dict) else art
    feat_cols = list(art.get("feature_cols") if isinstance(art, dict) else [])

    df = pd.read_parquet(dataset_parquet)
    need = {"trade_id", "k_rel", "entry_time", "entry_pred", "ret_if_exit_now_pct"}
    missing = sorted(list(need - set(df.columns)))
    if missing:
        raise SystemExit(f"dataset missing required columns: {missing}")

    df = df[(df["k_rel"] >= 1) & (df["k_rel"] <= int(hold_min))].copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df = df.sort_values(["trade_id", "k_rel"]).reset_index(drop=True)

    # keep only full-length trades
    sizes = df.groupby("trade_id")["k_rel"].count()
    full_ids = sizes[sizes == int(hold_min)].index.to_numpy()
    df = df[df["trade_id"].isin(full_ids)].copy().reset_index(drop=True)

    # ensure exit features
    for c in feat_cols:
        if c not in df.columns:
            df[c] = np.nan

    X = df[feat_cols].to_numpy(np.float32)
    df["pred_gap_pct"] = reg.predict(X).astype(np.float64)

    # order trades by entry_time
    order = df[["trade_id", "entry_time", "entry_pred"]].drop_duplicates("trade_id").sort_values("entry_time")
    trade_ids = order["trade_id"].to_numpy(np.int64)
    entry_time = order["entry_time"].to_numpy(dtype="datetime64[ns]")
    entry_pred = pd.to_numeric(order["entry_pred"], errors="coerce").to_numpy(np.float64)

    # pivot per-trade sequences
    R = (
        df.pivot(index="trade_id", columns="k_rel", values="ret_if_exit_now_pct")
        .reindex(index=trade_ids, columns=range(1, hold_min + 1))
        .to_numpy(np.float64)
    )
    G = (
        df.pivot(index="trade_id", columns="k_rel", values="pred_gap_pct")
        .reindex(index=trade_ids, columns=range(1, hold_min + 1))
        .to_numpy(np.float64)
    )

    # entry features at k_rel=1
    k1 = df[df["k_rel"] == 1].copy()
    k1 = k1.drop_duplicates("trade_id").set_index("trade_id")

    # always include entry_pred as a feature for completeness
    entry_feat_cols = list(dict.fromkeys(["entry_pred", *entry_feat_cols]))

    entry_feat = np.full((len(trade_ids), len(entry_feat_cols)), np.nan, dtype=np.float64)
    for j, c in enumerate(entry_feat_cols):
        if c == "entry_pred":
            entry_feat[:, j] = entry_pred
            continue
        if c in k1.columns:
            s = pd.to_numeric(k1[c], errors="coerce")
            entry_feat[:, j] = s.reindex(trade_ids).to_numpy(np.float64)

    # NaN handling:
    # - pred_gap: NaN => +inf (never triggers)
    # - returns: NaN => 0
    G = np.where(np.isfinite(G), G, np.inf)
    R = np.where(np.isfinite(R), R, 0.0)

    return {
        "trade_ids": trade_ids,
        "entry_time": entry_time,
        "entry_pred": entry_pred,
        "R": R,
        "G": G,
        "entry_feat_cols": entry_feat_cols,
        "entry_feat": entry_feat,
    }


def precompute_market_windows(*, market_csv: Path, entry_time: np.ndarray, hold_min: int) -> Dict[str, object]:
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
    min_low_cum = np.full((N, K + 1), np.nan, dtype=np.float64)

    for i in range(N):
        j0 = int(idx0[i])
        if j0 < 0:
            continue
        entry_open[i] = float(openp[j0])
        j1 = min(len(lowp) - 1, j0 + K)
        window = lowp[j0 : j1 + 1]
        if window.size == 0:
            continue
        cum = np.minimum.accumulate(window.astype(np.float64, copy=False))
        if cum.size < K + 1:
            pad = np.full((K + 1 - cum.size,), float(cum[-1]), dtype=np.float64)
            cum = np.concatenate([cum, pad], axis=0)
        min_low_cum[i, :] = cum[: K + 1]

    return {"idx0": idx0, "entry_open": entry_open, "min_low_cum": min_low_cum}


def main() -> None:
    ap = argparse.ArgumentParser(description="Guardrail sweep for ETH-USD under DO_NOT_TOUCH 50x-first-tier")

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
        default="0.70,0.72,0.74,0.76",
        help="Comma-separated entry_pred thresholds to try.",
    )
    ap.add_argument(
        "--tau-gaps",
        default="0.70,0.75,0.80,0.85,0.90",
        help="Comma-separated tau_gap values.",
    )
    ap.add_argument(
        "--min-exit-ks",
        default="1,2",
        help="Comma-separated min_exit_k values.",
    )

    # Guardrail features (entry-time columns in the oracle dataset)
    ap.add_argument(
        "--guard-max5-col",
        default="px_close_norm_pct__max5",
        help="First guardrail feature (cap).",
    )
    ap.add_argument(
        "--guard-vol-col",
        default="vol_log1p__last",
        help="Second guardrail feature (cap).",
    )
    ap.add_argument(
        "--guard-qs",
        default="0.80,0.85,0.90,0.95",
        help="Quantiles to use for guardrail caps (per feature, within each entry_threshold slice).",
    )

    # Constraints
    ap.add_argument("--min-trades-per-day", type=float, default=0.8)
    ap.add_argument("--max-initial-topups-eur", type=float, default=120.0, help="External topups only (excludes initial start deposit).")

    # bankroll overrides
    ap.add_argument("--do-not-touch", default="DO_NOT_TOUCH.txt")
    ap.add_argument("--first-tier-leverage", type=float, default=50.0)
    ap.add_argument("--start-balance", type=float, default=10.0)
    ap.add_argument("--refill-floor", type=float, default=10.0)
    ap.add_argument("--refill-min-balance", type=float, default=5.0)

    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--top-n", type=int, default=30)

    args = ap.parse_args()

    hold_min = int(args.hold_min)

    entry_thresholds = parse_float_list(args.entry_thresholds)
    tau_gaps = parse_float_list(args.tau_gaps)
    min_exit_ks = parse_int_list(args.min_exit_ks)
    guard_qs = parse_float_list(args.guard_qs)

    # Load trades
    mats = load_trade_mats(
        dataset_parquet=Path(args.dataset_parquet),
        exit_model=Path(args.exit_model),
        hold_min=hold_min,
        entry_feat_cols=[str(args.guard_max5_col), str(args.guard_vol_col)],
    )

    entry_time = mats["entry_time"]
    entry_pred = mats["entry_pred"]
    R = mats["R"]
    G = mats["G"]
    entry_feat_cols = mats["entry_feat_cols"]
    entry_feat = mats["entry_feat"]  # (N, d)

    col_to_j = {c: j for j, c in enumerate(entry_feat_cols)}
    j_max5 = col_to_j.get(str(args.guard_max5_col))
    j_vol = col_to_j.get(str(args.guard_vol_col))
    if j_max5 is None or j_vol is None:
        raise SystemExit("guardrail columns not found in entry features")

    # Market precompute
    mkt = precompute_market_windows(market_csv=Path(args.market_csv), entry_time=entry_time, hold_min=hold_min)
    market_idx0 = mkt["idx0"]
    entry_open = mkt["entry_open"]
    min_low_cum = mkt["min_low_cum"]

    # Strategy
    base = parse_do_not_touch(Path(args.do_not_touch))
    tiers = list(base.tiers)
    if tiers:
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

    # Precompute exit outcomes per (tau_gap, min_exit_k) over all trades
    exit_cache: Dict[Tuple[float, int], Tuple[np.ndarray, np.ndarray]] = {}
    for mk in sorted(min_exit_ks):
        for tau in sorted(tau_gaps):
            exit_k = np.full((len(entry_pred),), int(hold_min), dtype=np.int64)
            for i in range(len(exit_k)):
                exit_k[i] = pick_exit_k(G[i], hold_min=hold_min, min_exit_k=int(mk), tau_gap=float(tau))
            ret_1x = R[np.arange(len(exit_k)), exit_k - 1].astype(np.float64)
            exit_cache[(float(tau), int(mk))] = (exit_k, ret_1x)

    rows: List[Dict[str, object]] = []

    # Sweep
    total = len(entry_thresholds) * len(guard_qs) * len(guard_qs) * len(tau_gaps) * len(min_exit_ks)
    done = 0

    for thr in sorted(entry_thresholds):
        base_mask = np.isfinite(entry_pred) & (entry_pred >= float(thr))
        if not base_mask.any():
            continue

        # guardrail cutoffs computed within the base slice
        v_max5 = entry_feat[base_mask, j_max5]
        v_vol = entry_feat[base_mask, j_vol]

        # if a column is all NaN, quantiles will be NaN; handle by making cutoff +inf (no filter)
        max5_cut_by_q = {}
        vol_cut_by_q = {}
        for q in guard_qs:
            try:
                max5_cut_by_q[q] = float(np.nanquantile(v_max5, q)) if np.isfinite(np.nanmean(v_max5)) else float("inf")
            except Exception:
                max5_cut_by_q[q] = float("inf")
            try:
                vol_cut_by_q[q] = float(np.nanquantile(v_vol, q)) if np.isfinite(np.nanmean(v_vol)) else float("inf")
            except Exception:
                vol_cut_by_q[q] = float("inf")

        for q1 in sorted(guard_qs):
            max5_cut = float(max5_cut_by_q[q1])
            m1 = base_mask & (np.isnan(entry_feat[:, j_max5]) | (entry_feat[:, j_max5] <= max5_cut))
            for q2 in sorted(guard_qs):
                vol_cut = float(vol_cut_by_q[q2])
                mask = m1 & (np.isnan(entry_feat[:, j_vol]) | (entry_feat[:, j_vol] <= vol_cut))

                idx = np.where(mask)[0]
                if idx.size == 0:
                    continue

                # trade frequency (calendar days inclusive)
                t_idx = pd.to_datetime(entry_time[idx], utc=True)
                t0 = t_idx.min()
                t1 = t_idx.max()
                cal_days = int((t1.floor("D") - t0.floor("D")).days + 1)
                trades_per_calendar_day = float(idx.size / cal_days) if cal_days > 0 else 0.0

                # Quick prune: if can't meet min trades/day, skip all exit policies
                if trades_per_calendar_day < float(args.min_trades_per_day):
                    done += len(tau_gaps) * len(min_exit_ks)
                    continue

                for mk in sorted(min_exit_ks):
                    for tau in sorted(tau_gaps):
                        exit_k_all, ret_1x_all = exit_cache[(float(tau), int(mk))]
                        exit_k = exit_k_all[idx]
                        ret_1x = ret_1x_all[idx]

                        # Bankroll simulation (loop per trade; n <= ~1100)
                        balance = float(strat.start_balance)
                        bank = 0.0

                        # External topups can occur whenever bank < bank_threshold.
                        # For your constraint, we track INITIAL topups only (before bank first reaches >= threshold).
                        topups_only_total = 0.0
                        n_topups_total = 0
                        topups_only_initial = 0.0
                        n_topups_initial = 0

                        n_liq = 0
                        n_init_liq = 0  # liquidations while bank_before < threshold
                        bank150_first = None

                        for j in range(len(idx)):
                            ii = int(idx[j])

                            bal_before = float(balance)
                            bank_before = float(bank)

                            lev = leverage_for_equity(balance, strat.tiers)

                            # liquidation
                            liq = False
                            j0 = int(market_idx0[ii])
                            if j0 >= 0 and np.isfinite(entry_open[ii]):
                                liq_px = liq_price_long(float(entry_open[ii]), fee_side=float(args.fee_side), lev=float(lev))
                                if np.isfinite(liq_px):
                                    ml = float(min_low_cum[ii, int(exit_k[j])])
                                    if np.isfinite(ml) and ml <= liq_px:
                                        liq = True

                            if (not liq) and (1.0 + (float(ret_1x[j]) / 100.0) * float(lev) <= 0.0):
                                liq = True

                            if liq:
                                n_liq += 1
                                if bank_before < float(strat.bank_threshold):
                                    n_init_liq += 1
                                pnl = -bal_before
                                balance = 0.0
                            else:
                                pnl = bal_before * (float(ret_1x[j]) / 100.0) * float(lev)
                                balance = bal_before + pnl

                            # withdraw
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
                                        topups_only_total += dep
                                        n_topups_total += 1
                                        # only count as "initial" if bank150 has not been reached yet
                                        if bank150_first is None:
                                            topups_only_initial += dep
                                            n_topups_initial += 1
                                else:
                                    dep = float(strat.bank_fraction) * bank
                                    bank -= dep
                                    balance += dep

                            if bank150_first is None and bank >= float(strat.bank_threshold):
                                bank150_first = pd.to_datetime(t_idx[j], utc=True)

                        final_equity = float(balance + bank)

                        rows.append(
                            {
                                "entry_threshold": float(thr),
                                "tau_gap": float(tau),
                                "min_exit_k": int(mk),
                                "hold_min": int(hold_min),
                                "guard_max5_col": str(args.guard_max5_col),
                                "guard_vol_col": str(args.guard_vol_col),
                                "guard_max5_q": float(q1),
                                "guard_vol_q": float(q2),
                                "guard_max5_cut": float(max5_cut),
                                "guard_vol_cut": float(vol_cut),
                                "n_trades": int(len(idx)),
                                "calendar_days_inclusive": int(cal_days),
                                "trades_per_calendar_day": float(trades_per_calendar_day),
                                "n_liquidations": int(n_liq),
                                "initial_liquidations": int(n_init_liq),
                                "n_topups_total": int(n_topups_total),
                                "topups_only_total": float(topups_only_total),
                                "n_topups_initial": int(n_topups_initial),
                                "topups_only_initial": float(topups_only_initial),
                                "ext_total_incl_start_end": float(topups_only_total + float(strat.start_balance)),
                                "bank150_first_reached_at": bank150_first.isoformat() if bank150_first is not None else None,
                                "final_equity": float(final_equity),
                            }
                        )

                        done += 1
                        if done % 50 == 0 or done == total:
                            print(f"progress: {done}/{total}", flush=True)

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("No rows produced")

    # Feasibility filter
    feasible = out[
        (out["topups_only_initial"] <= float(args.max_initial_topups_eur))
        & (out["trades_per_calendar_day"] >= float(args.min_trades_per_day))
    ].copy()

    # Ranking: primarily minimize liquidations, then maximize equity.
    def _rank(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["rank_key"] = list(zip(df.initial_liquidations, df.n_liquidations, df.topups_only_initial, -df.final_equity))
        return df.sort_values("rank_key").drop(columns=["rank_key"]).reset_index(drop=True)

    out = _rank(out)
    feasible = _rank(feasible) if len(feasible) else feasible

    out_root = Path(args.out_dir) / f"sweeps_ethusd_50x_guardrails_{now_ts()}"
    out_root.mkdir(parents=True, exist_ok=True)

    out.to_csv(out_root / "results.csv", index=False)
    if len(feasible):
        feasible.to_csv(out_root / "feasible.csv", index=False)

    top_n = int(args.top_n)
    top = feasible.head(top_n) if len(feasible) else out.head(top_n)
    (out_root / "top_configs.json").write_text(
        json.dumps(
            {
                "created_utc": now_ts(),
                "constraints": {
                    "min_trades_per_calendar_day": float(args.min_trades_per_day),
                    "max_initial_topups_eur": float(args.max_initial_topups_eur),
                },
                "has_feasible": bool(len(feasible)),
                "top": top.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote: {out_root}")
    print(f"  results: {out_root / 'results.csv'}")
    if len(feasible):
        print(f"  feasible: {out_root / 'feasible.csv'}  (n={len(feasible)})")
    else:
        print("  feasible: none")

    print("\nTop configs (feasible if any, else best overall):")
    cols = [
        "entry_threshold",
        "tau_gap",
        "min_exit_k",
        "guard_max5_q",
        "guard_vol_q",
        "n_trades",
        "trades_per_calendar_day",
        "topups_only_initial",
        "initial_liquidations",
        "n_liquidations",
        "final_equity",
        "bank150_first_reached_at",
    ]
    cols = [c for c in cols if c in top.columns]
    with pd.option_context("display.max_rows", min(top_n, 50), "display.width", 200):
        print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
