#!/usr/bin/env python3
"""Timestamp (UTC): 2026-01-02T17:20:36Z

Sweep entry-score percentile thresholds for the current best pipeline:
- Entry regressor (BTCUSDT)
- Exit regressor (BTCUSDT)

For each percentile threshold, we:
- Compute threshold value from entry scores within [2024-01-01, now]
- Generate trades using:
  - entry decision at minute i close; enter at minute i+1 open
  - exit selection over 1..H minutes after entry (default H=10) using "last max" over exit predictions
  - max entries per day (default 2)
  - no overlap: next entry cannot occur until previous trade exits
- Report:
  - total profitability (1x): sum(realized_ret_pct)
  - daily profitability % (1x): mean over days of (sum realized_ret_pct per day)
  - total number of trades
  - number of liquidations (strategy simulation, wick+close hard liq) using DO_NOT_TOUCH leverage tiers

Outputs a timestamped CSV under data/sweeps/.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import FEATURES, compute_feature_frame

_MKT_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


# ------------------- feature parity helpers -------------------

_MEAN_SUFFIX_RE = re.compile(r"_mean_(\d+)m$")


def _infer_pre_mins_from_features(feat_names: list[str]) -> list[int]:
    mins: set[int] = set()
    for c in feat_names:
        m = _MEAN_SUFFIX_RE.search(str(c))
        if not m:
            continue
        try:
            k = int(m.group(1))
        except Exception:
            continue
        if k > 0:
            mins.add(k)
    return sorted(mins)


def _build_feature_frame(bars: pd.DataFrame, feat_names: list[str], pre_mins: list[int]) -> pd.DataFrame:
    """Compute base FEATURES then add rolling means for each pre_min required by artifacts."""
    base = compute_feature_frame(bars.rename(columns={"timestamp": "ts_min"}))
    src = base[[c for c in FEATURES if c in base.columns]]

    full = src
    for m in sorted({int(x) for x in pre_mins if int(x) > 0}):
        ctx_mean = src.rolling(int(m), min_periods=int(m)).mean().add_suffix(f"_mean_{int(m)}m")
        full = pd.concat([full, ctx_mean], axis=1)

    # Ensure all requested feature columns exist (fill missing with NaN).
    for c in feat_names:
        if c not in full.columns:
            full[c] = np.nan

    out = pd.concat([base[["ts_min"]], full[feat_names]], axis=1).rename(columns={"ts_min": "timestamp"})
    return out


def _load_model_payload(path: Path) -> tuple[object, list[str]]:
    payload = joblib.load(path)
    model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    feat_names = list(payload["features"]) if isinstance(payload, dict) and "features" in payload else list(FEATURES)
    return model, feat_names


def _last_max_k(preds: np.ndarray) -> tuple[int, float]:
    """Return (best_k, best_pred) using tie-break-to-last semantics (>=). preds is length H."""
    best = -1e30
    best_k = 1
    for k, y in enumerate(preds, start=1):
        if y is None or not np.isfinite(y):
            continue
        if float(y) >= float(best):
            best = float(y)
            best_k = int(k)
    return int(best_k), float(best)


def net_return_pct(entry_px_open: float, exit_px_close: float, fee_side: float) -> float:
    mult = (float(exit_px_close) * (1.0 - float(fee_side))) / (float(entry_px_open) * (1.0 + float(fee_side)))
    return (mult - 1.0) * 100.0


# ------------------- strategy / liquidation simulation -------------------

@dataclass
class Strategy:
    tiers: list[tuple[float, float]]  # list of (equity_threshold_eur, leverage), ascending thresholds
    withdraw_half_levers: set[float]
    start_balance: float
    refinance_min_balance: float
    refinance_floor: float
    refinance_bank_threshold: float
    refinance_bank_fraction: float


def parse_strategy(path: Path) -> Strategy:
    txt = path.read_text(encoding="utf-8")
    tiers: list[tuple[float, float]] = []
    lev_pat = re.compile(r"(\d+)x leverage until account reaches ([\d,]+)EUR")
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

    refinance_min_balance = 5.0
    refinance_floor = start_balance
    refinance_bank_threshold = 150.0

    m2 = re.search(r"20% of the bank account is deposited", txt)
    refinance_bank_fraction = 0.20 if m2 else 0.20

    return Strategy(
        tiers=tiers,
        withdraw_half_levers=withdraw_half_levers,
        start_balance=start_balance,
        refinance_min_balance=refinance_min_balance,
        refinance_floor=refinance_floor,
        refinance_bank_threshold=refinance_bank_threshold,
        refinance_bank_fraction=refinance_bank_fraction,
    )


def leverage_for_equity(equity: float, tiers: list[tuple[float, float]]) -> float:
    for thr, lev in tiers:
        if float(equity) < float(thr):
            return float(lev)
    return 1.0


def wick_liquidated_for_trade(*, lev: float, fee_side: float, entry_open: float, low_window: np.ndarray) -> tuple[bool, float]:
    if float(lev) <= 1.0:
        return False, float("nan")
    # Solve: 1 + L*(net_mult - 1) <= 0, net_mult = (px*(1-fee))/(entry*(1+fee))
    liq_price = float(entry_open) * (1.0 + float(fee_side)) * (1.0 - 1.0 / float(lev)) / max(1e-12, (1.0 - float(fee_side)))
    breached = bool(np.any(low_window <= liq_price)) if low_window.size else False
    return breached, float(liq_price)


def simulate_strategy(
    trades: pd.DataFrame,
    *,
    strat: Strategy,
    lowp: np.ndarray,
    openp: np.ndarray,
    fee_side: float,
) -> dict:
    # trades must include: entry_index (int), exit_rel_min (int), realized_ret_pct (float)
    balance = float(strat.start_balance)
    bank = 0.0
    total_ext_deposits = float(strat.start_balance)

    n_liq = 0
    n_wick_liq = 0

    for _, r in trades.iterrows():
        idx0 = int(r["entry_index"])
        k = int(r["exit_rel_min"])
        ret_pct = float(r["realized_ret_pct"])  # net % at 1x

        bal_before = float(balance)
        bank_before = float(bank)

        lev = leverage_for_equity(balance, strat.tiers)

        # wick liquidation check over [entry_idx .. entry_idx+k]
        idx1 = min(len(lowp) - 1, idx0 + max(0, k))
        window = lowp[idx0 : idx1 + 1]
        wick_liq, _liq_px = wick_liquidated_for_trade(
            lev=float(lev), fee_side=float(fee_side), entry_open=float(openp[idx0]), low_window=window
        )

        # close hard liquidation (prevents negative equity)
        close_hard_liq = bool((1.0 + (ret_pct / 100.0) * float(lev)) <= 0.0)

        liquidated = bool(wick_liq or close_hard_liq)
        if liquidated:
            n_liq += 1
            if wick_liq:
                n_wick_liq += 1
            pnl = -bal_before
            balance = 0.0
        else:
            pnl = bal_before * (ret_pct / 100.0) * float(lev)
            balance = bal_before + pnl

        # Withdraw half of profitable trades at top leverage tiers
        if (not liquidated) and (float(lev) in strat.withdraw_half_levers) and pnl > 0.0:
            withdrawn = 0.5 * pnl
            balance -= withdrawn
            bank += withdrawn

        # Refinance
        if balance < float(strat.refinance_min_balance):
            if bank < float(strat.refinance_bank_threshold):
                dep = max(0.0, float(strat.refinance_floor) - balance)
                balance += dep
                total_ext_deposits += dep
            else:
                dep = float(strat.refinance_bank_fraction) * bank
                bank -= dep
                balance += dep

        # (bank_before is currently unused, but keeping structure similar to other scripts)
        _ = bank_before

    final_equity = float(balance + bank)
    profit_eur = float(final_equity - total_ext_deposits)
    profit_pct_of_deposits = float((final_equity / total_ext_deposits - 1.0) * 100.0) if total_ext_deposits > 0 else float("nan")

    return {
        "n_liquidations": int(n_liq),
        "n_wick_liquidations": int(n_wick_liq),
        "total_ext_deposits_eur": float(total_ext_deposits),
        "final_balance_eur": float(balance),
        "final_bank_eur": float(bank),
        "final_equity_eur": float(final_equity),
        "profit_eur": float(profit_eur),
        "profit_pct_of_deposits": float(profit_pct_of_deposits),
    }


# ------------------- main sweep -------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep entry-score percentile thresholds (2024->now)")
    ap.add_argument("--market-csv", default=str(REPO_ROOT / "data" / "btc_profitability_analysis_filtered.csv"))
    ap.add_argument("--do-not-touch", default=str(REPO_ROOT / "DO_NOT_TOUCH.txt"))
    ap.add_argument("--entry-model", default=str(REPO_ROOT / "models" / "entry_regressor_btcusdt_2025-12-20T13-47-55Z.joblib"))
    ap.add_argument("--exit-model", default=str(REPO_ROOT / "models" / "exit_regressor_btcusdt_2025-12-20T14-41-31Z.joblib"))
    ap.add_argument("--start", default="2024-01-01", help="YYYY-MM-DD")
    ap.add_argument("--hold-min", type=int, default=10)
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--max-entries-per-day", type=int, default=2)
    ap.add_argument("--allow-overlap", action="store_true", help="If set, allow overlapping trades (default: no overlap)")
    ap.add_argument(
        "--percentiles",
        default="",
        help="Comma-separated percentiles (e.g. '95,99,99.5'). If empty, uses --percentile-start..end..step.",
    )
    ap.add_argument("--percentile-start", type=float, default=99.90)
    ap.add_argument("--percentile-end", type=float, default=99.99)
    ap.add_argument("--percentile-step", type=float, default=0.01)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "sweeps"))
    args = ap.parse_args()

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)

    # Load market data (UTC)
    mkt = pd.read_csv(
        args.market_csv,
        usecols=_MKT_COLS,
        parse_dates=["timestamp"],
        dtype={"open": "float32", "high": "float32", "low": "float32", "close": "float32", "volume": "float32"},
    )
    mkt["timestamp"] = pd.to_datetime(mkt["timestamp"], utc=True, errors="coerce")
    mkt = mkt.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # Keep a small buffer before start for rolling windows
    buffer_start = pd.Timestamp(start_dt) - pd.Timedelta(days=1)
    mkt = mkt[mkt["timestamp"] >= buffer_start].reset_index(drop=True)

    if mkt.empty:
        raise SystemExit("No market data after buffer_start; check --market-csv/--start")

    # Models + feature names
    entry_model, entry_feat = _load_model_payload(Path(args.entry_model))
    exit_model, exit_feat = _load_model_payload(Path(args.exit_model))

    feat_union = list(dict.fromkeys([*entry_feat, *exit_feat]))
    pre_mins = _infer_pre_mins_from_features(feat_union)

    feats = _build_feature_frame(mkt, feat_union, pre_mins)

    X_entry = feats[entry_feat].to_numpy(dtype=np.float32)
    X_exit = feats[exit_feat].to_numpy(dtype=np.float32)

    entry_scores = np.asarray(entry_model.predict(X_entry), dtype=np.float64)
    exit_preds = np.asarray(exit_model.predict(X_exit), dtype=np.float64)

    ts = feats["timestamp"].to_numpy()
    ts_pd = pd.Series(pd.to_datetime(ts, utc=True))

    # Entry is assumed at the same-minute open (matches existing offline scripts in this repo).
    # Exit predictions are evaluated on minutes 1..H after entry.
    openp = mkt["open"].to_numpy(np.float64, copy=False)
    lowp = mkt["low"].to_numpy(np.float64, copy=False)
    closep = mkt["close"].to_numpy(np.float64, copy=False)

    # Percentiles
    pct_list: list[float] = []
    raw = str(getattr(args, "percentiles", "") or "").strip()
    if raw:
        for x in raw.split(","):
            x = x.strip().replace("%", "")
            if not x:
                continue
            pct_list.append(float(x))
    else:
        p0 = float(getattr(args, "percentile_start", 99.90))
        p1 = float(getattr(args, "percentile_end", 99.99))
        step = float(getattr(args, "percentile_step", 0.01))
        if step <= 0:
            raise SystemExit("--percentile-step must be > 0")
        if p1 < p0:
            raise SystemExit("--percentile-end must be >= --percentile-start")
        # inclusive range with float step (rounded to 2 decimals for stability)
        n = int(round((p1 - p0) / step))
        pct_list = [round(p0 + i * step, 2) for i in range(n + 1)]

    # Compute thresholds from decision-score distribution within [start_dt, end]
    mask_scores = (ts_pd >= pd.Timestamp(start_dt)) & np.isfinite(entry_scores)
    scores_window = entry_scores[mask_scores]
    if scores_window.size == 0:
        raise SystemExit("No finite entry scores in the requested window.")

    thresholds = {p: float(np.percentile(scores_window, p)) for p in pct_list}

    strat = parse_strategy(Path(args.do_not_touch))

    out_rows: list[dict] = []
    out_trades_dir = Path(args.out_dir)
    out_trades_dir.mkdir(parents=True, exist_ok=True)

    for p in pct_list:
        thr = float(thresholds[p])

        trades: list[dict] = []
        next_free_idx = 0
        entries_today = 0
        cur_day = None

        H = int(args.hold_min)
        fee = float(args.fee)

        # iterate candidate entries at minute i open
        for i in range(0, len(ts_pd) - (H + 2)):
            j = i  # entry index

            entry_time = ts_pd.iloc[j].to_pydatetime()
            if entry_time < start_dt:
                continue

            d = entry_time.date()
            if cur_day != d:
                cur_day = d
                entries_today = 0

            if entries_today >= int(args.max_entries_per_day):
                continue

            if (not bool(args.allow_overlap)) and j < int(next_free_idx):
                continue

            s = float(entry_scores[j])
            if not np.isfinite(s) or s < thr:
                continue

            # Exit selection: evaluate at minutes 1..H after entry (indices j+1..j+H)
            preds = exit_preds[(j + 1) : (j + 1 + H)]
            if preds.size == 0:
                continue

            best_k, best_pred = _last_max_k(preds)
            exit_idx = j + int(best_k)
            if exit_idx >= len(closep):
                break

            realized = net_return_pct(float(openp[j]), float(closep[exit_idx]), fee)

            trades.append(
                {
                    "percentile": float(p),
                    "threshold": float(thr),
                    "entry_time": entry_time,
                    "entry_index": int(j),
                    "exit_rel_min": int(best_k),
                    "predicted_ret_pct": float(best_pred) if np.isfinite(best_pred) else np.nan,
                    "realized_ret_pct": float(realized),
                    "entry_score": float(s),
                    "entry_open": float(openp[j]),
                    "exit_close": float(closep[exit_idx]),
                }
            )

            entries_today += 1
            if not bool(args.allow_overlap):
                next_free_idx = int(exit_idx) + 1

        tdf = pd.DataFrame(trades)
        if tdf.empty:
            out_rows.append(
                {
                    "percentile": float(p),
                    "threshold": float(thr),
                    "n_trades": 0,
                    "total_profitability_1x_sum_pct": 0.0,
                    "daily_profitability_1x_mean_sum_pct": np.nan,
                    "n_liquidations": 0,
                    "n_wick_liquidations": 0,
                    "final_equity_eur": float(strat.start_balance),
                    "profit_pct_of_deposits": 0.0,
                }
            )
            continue

        tdf["date"] = pd.to_datetime(tdf["entry_time"], utc=True).dt.date
        daily_sum = tdf.groupby("date")["realized_ret_pct"].sum()

        strat_sum = simulate_strategy(tdf, strat=strat, lowp=lowp, openp=openp, fee_side=float(fee))

        out_rows.append(
            {
                "percentile": float(p),
                "threshold": float(thr),
                "n_trades": int(len(tdf)),
                "total_profitability_1x_sum_pct": float(tdf["realized_ret_pct"].sum()),
                "daily_profitability_1x_mean_sum_pct": float(daily_sum.mean()) if len(daily_sum) else np.nan,
                "n_liquidations": int(strat_sum["n_liquidations"]),
                "n_wick_liquidations": int(strat_sum["n_wick_liquidations"]),
                "final_equity_eur": float(strat_sum["final_equity_eur"]),
                "profit_pct_of_deposits": float(strat_sum["profit_pct_of_deposits"]),
            }
        )

    out = pd.DataFrame(out_rows).sort_values(["percentile"], ascending=True).reset_index(drop=True)

    ts_out = _now_ts()
    out_path = Path(args.out_dir) / f"entry_exit_threshold_sweep_{ts_out}.csv"
    out.to_csv(out_path, index=False)

    print(out.to_string(index=False))
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
