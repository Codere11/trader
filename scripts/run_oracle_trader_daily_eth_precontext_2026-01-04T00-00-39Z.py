#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T00:00:39Z
"""Run a per-day oracle trader on a 1-minute OHLCV+features dataset and save trades + precontexts.

Oracle trader definition (implemented):
- One position at a time (flat / long / short).
- Uses perfect future knowledge within each day to maximize end-of-day capital (percentage return).
- Trades are allowed to flip long<->short at a timestamp (pays both fees).
- Closes all positions by end of each UTC day.

Precontext artifacts:
- Separately saves 5-minute pre-entry and 5-minute pre-exit contexts.
- Context rows are aligned to exact minute offsets (e.g., -5..-1); missing minutes are recorded as NaN rows.
- Includes OHLCV + standard feature columns.

Input expectations:
- A dataset file with columns:
    timestamp, open, high, low, close, volume, ...features...
  Timestamp must be parseable as UTC datetimes.

Outputs (timestamped filenames):
- data/oracle_trader/oracle_daily_<SYMBOL>_trades_<ts>.csv
- data/oracle_trader/oracle_daily_<SYMBOL>_summary_<ts>.csv
- data/oracle_precontext/oracle_daily_<SYMBOL>_pre_entry_dyn5m_<ts>.parquet (+ .csv)
- data/oracle_precontext/oracle_daily_<SYMBOL>_pre_exit_dyn5m_<ts>.parquet (+ .csv)
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


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

CORE_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass
class DpState:
    # For each minute, store backpointer into previous minute state.
    prev: np.ndarray  # int8 codes
    entries: np.ndarray  # int32 number of entries taken to reach state


StateCode = Literal[0, 1, 2]  # 0=cash, 1=long, 2=short


def _now_ts() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _parse_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise SystemExit("dataset missing 'timestamp' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Keep only expected columns if present (plus any extras requested).
    if not all(c in df.columns for c in CORE_COLS):
        missing = [c for c in CORE_COLS if c not in df.columns]
        raise SystemExit(f"dataset missing columns: {missing}")

    # Sort/dedupe to make searchsorted valid.
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    return df


def _better(a_val: float, a_n: int, b_val: float, b_n: int, eps: float) -> bool:
    """Return True if (a_val, a_n) is better than (b_val, b_n)."""
    if a_val > b_val + eps:
        return True
    if abs(a_val - b_val) <= eps and a_n > b_n:
        return True
    return False


def _run_daily_dp(prices: np.ndarray, per_side_fee: float, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (state_path, entry_count_path) for a single day.

    prices: close prices for each minute in the day (float64)

    State model:
    - cash: USD wealth
    - long: coin amount
    - short: k = USD*price-like invariant (so USD wealth at price p is k/(p*(1+fee)))

    Objective: maximize end-of-day cash.
    Secondary tie-break: maximize number of entries (trade frequency).
    """

    n = int(prices.size)
    if n <= 1:
        # Nothing to do.
        return np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.int32)

    f = float(per_side_fee)
    if f < 0:
        raise ValueError("per_side_fee must be >= 0")

    # DP arrays (values in each state's native unit)
    cash = np.empty(n, dtype=np.float64)
    longc = np.empty(n, dtype=np.float64)
    shortk = np.empty(n, dtype=np.float64)

    cash_bp = DpState(prev=np.full(n, -1, dtype=np.int8), entries=np.zeros(n, dtype=np.int32))
    long_bp = DpState(prev=np.full(n, -1, dtype=np.int8), entries=np.zeros(n, dtype=np.int32))
    short_bp = DpState(prev=np.full(n, -1, dtype=np.int8), entries=np.zeros(n, dtype=np.int32))

    p0 = float(prices[0])
    if not np.isfinite(p0) or p0 <= 0:
        raise ValueError("invalid first price")

    # Start day with 1.0 cash (normalized)
    cash[0] = 1.0
    cash_bp.prev[0] = 0
    cash_bp.entries[0] = 0

    # Allow opening long/short at t=0
    longc[0] = cash[0] / (p0 * (1.0 + f))
    long_bp.prev[0] = 0
    long_bp.entries[0] = 1

    shortk[0] = cash[0] * p0 * (1.0 - f)
    short_bp.prev[0] = 0
    short_bp.entries[0] = 1

    for t in range(1, n):
        p = float(prices[t])
        if not np.isfinite(p) or p <= 0:
            # Hard fail: price series should be clean.
            raise ValueError(f"invalid price at t={t}: {p}")

        # ----- cash[t] -----
        best_val = cash[t - 1]
        best_prev: StateCode = 0
        best_n = int(cash_bp.entries[t - 1])

        # from long -> cash (sell)
        cand_val = longc[t - 1] * p * (1.0 - f)
        cand_n = int(long_bp.entries[t - 1])
        if _better(cand_val, cand_n, best_val, best_n, eps):
            best_val, best_prev, best_n = cand_val, 1, cand_n

        # from short -> cash (buy back)
        cand_val = shortk[t - 1] / (p * (1.0 + f))
        cand_n = int(short_bp.entries[t - 1])
        if _better(cand_val, cand_n, best_val, best_n, eps):
            best_val, best_prev, best_n = cand_val, 2, cand_n

        cash[t] = best_val
        cash_bp.prev[t] = int(best_prev)
        cash_bp.entries[t] = best_n

        # ----- long[t] (coin amount) -----
        best_val = longc[t - 1]
        best_prev = 1
        best_n = int(long_bp.entries[t - 1])

        # from cash -> long (buy)
        cand_val = cash[t - 1] / (p * (1.0 + f))
        cand_n = int(cash_bp.entries[t - 1]) + 1
        if _better(cand_val, cand_n, best_val, best_n, eps):
            best_val, best_prev, best_n = cand_val, 0, cand_n

        # from short -> long (close short then buy)
        cand_val = shortk[t - 1] / ((p * (1.0 + f)) * (p * (1.0 + f)))
        cand_n = int(short_bp.entries[t - 1]) + 1
        if _better(cand_val, cand_n, best_val, best_n, eps):
            best_val, best_prev, best_n = cand_val, 2, cand_n

        longc[t] = best_val
        long_bp.prev[t] = int(best_prev)
        long_bp.entries[t] = best_n

        # ----- short[t] (k invariant) -----
        best_val = shortk[t - 1]
        best_prev = 2
        best_n = int(short_bp.entries[t - 1])

        # from cash -> short (sell)
        cand_val = cash[t - 1] * p * (1.0 - f)
        cand_n = int(cash_bp.entries[t - 1]) + 1
        if _better(cand_val, cand_n, best_val, best_n, eps):
            best_val, best_prev, best_n = cand_val, 0, cand_n

        # from long -> short (sell long then sell short)
        cand_val = longc[t - 1] * (p * (1.0 - f)) * (p * (1.0 - f))
        cand_n = int(long_bp.entries[t - 1]) + 1
        if _better(cand_val, cand_n, best_val, best_n, eps):
            best_val, best_prev, best_n = cand_val, 1, cand_n

        shortk[t] = best_val
        short_bp.prev[t] = int(best_prev)
        short_bp.entries[t] = best_n

    # Force end-of-day cash (close by end): we end in cash at t=n-1.
    state = np.empty(n, dtype=np.int8)
    entry_count = np.empty(n, dtype=np.int32)

    cur_state: StateCode = 0
    state[n - 1] = int(cur_state)
    entry_count[n - 1] = int(cash_bp.entries[n - 1])

    for t in range(n - 1, 0, -1):
        if cur_state == 0:
            prev_state = int(cash_bp.prev[t])
            entry_count[t - 1] = int(
                cash_bp.entries[t - 1] if prev_state == 0 else (long_bp.entries[t - 1] if prev_state == 1 else short_bp.entries[t - 1])
            )
        elif cur_state == 1:
            prev_state = int(long_bp.prev[t])
            entry_count[t - 1] = int(
                cash_bp.entries[t - 1] if prev_state == 0 else (long_bp.entries[t - 1] if prev_state == 1 else short_bp.entries[t - 1])
            )
        else:
            prev_state = int(short_bp.prev[t])
            entry_count[t - 1] = int(
                cash_bp.entries[t - 1] if prev_state == 0 else (long_bp.entries[t - 1] if prev_state == 1 else short_bp.entries[t - 1])
            )

        cur_state = 0 if prev_state == 0 else (1 if prev_state == 1 else 2)
        state[t - 1] = int(cur_state)

    return state, entry_count


def _net_ret_pct(side: str, entry_px: float, exit_px: float, per_side_fee: float) -> float:
    f = float(per_side_fee)
    entry_px = float(entry_px)
    exit_px = float(exit_px)
    if entry_px <= 0 or exit_px <= 0:
        return float("nan")

    if side == "BUY":
        mult = (exit_px * (1.0 - f)) / (entry_px * (1.0 + f))
    elif side == "SELL":
        mult = (entry_px * (1.0 - f)) / (exit_px * (1.0 + f))
    else:
        raise ValueError(f"bad side {side}")

    return (mult - 1.0) * 100.0


def _extract_trades_for_day(
    ts: np.ndarray,
    px: np.ndarray,
    state: np.ndarray,
    per_side_fee: float,
    global_offset: int,
    date_utc: dt.date,
) -> list[dict]:
    trades: list[dict] = []

    prev = int(state[0])
    entry_i: int | None = None
    entry_side: str | None = None

    def _side_from_code(code: int) -> str:
        return "BUY" if code == 1 else "SELL"

    if prev != 0:
        entry_i = 0
        entry_side = _side_from_code(prev)

    for i in range(1, int(len(state))):
        cur = int(state[i])
        if cur == prev:
            continue

        # Close prior position (if any)
        if prev != 0 and entry_i is not None and entry_side is not None:
            exit_i = i
            epx = float(px[entry_i])
            xpx = float(px[exit_i])
            r = _net_ret_pct(entry_side, epx, xpx, per_side_fee)
            trades.append(
                dict(
                    date=str(date_utc),
                    entry_time=pd.Timestamp(ts[entry_i]).tz_localize("UTC"),
                    exit_time=pd.Timestamp(ts[exit_i]).tz_localize("UTC"),
                    duration_min=int((ts[exit_i] - ts[entry_i]) / np.timedelta64(1, "m")),
                    side=entry_side,
                    entry_px=epx,
                    exit_px=xpx,
                    net_return_pct=float(r),
                    entry_idx=int(global_offset + entry_i),
                    exit_idx=int(global_offset + exit_i),
                )
            )

        # Open new position (if any)
        if cur != 0:
            entry_i = i
            entry_side = _side_from_code(cur)
        else:
            entry_i = None
            entry_side = None

        prev = cur

    # Ensure we're flat by end; if not, close at last index.
    if prev != 0 and entry_i is not None and entry_side is not None:
        exit_i = int(len(state) - 1)
        epx = float(px[entry_i])
        xpx = float(px[exit_i])
        r = _net_ret_pct(entry_side, epx, xpx, per_side_fee)
        trades.append(
            dict(
                date=str(date_utc),
                entry_time=pd.Timestamp(ts[entry_i]).tz_localize("UTC"),
                exit_time=pd.Timestamp(ts[exit_i]).tz_localize("UTC"),
                duration_min=int((ts[exit_i] - ts[entry_i]) / np.timedelta64(1, "m")),
                side=entry_side,
                entry_px=epx,
                exit_px=xpx,
                net_return_pct=float(r),
                entry_idx=int(global_offset + entry_i),
                exit_idx=int(global_offset + exit_i),
            )
        )

    return trades


def _precontext_rows(
    df: pd.DataFrame,
    ts_sorted: np.ndarray,
    trade_row: int,
    when: pd.Timestamp,
    *,
    phase: str,
    rel_offsets: np.ndarray,
    feature_cols: list[str],
    trade_meta: dict,
) -> list[dict]:
    out: list[dict] = []

    # Convert to numpy datetime64[ns] (tz-naive UTC) for searchsorted.
    when64 = np.datetime64(when.tz_convert("UTC").to_datetime64())

    for off in rel_offsets:
        t_ctx = when64 + np.timedelta64(int(off), "m")
        pos = int(np.searchsorted(ts_sorted, t_ctx))
        if 0 <= pos < ts_sorted.size and ts_sorted[pos] == t_ctx:
            row = df.iloc[pos]
            d = {c: row.get(c, np.nan) for c in (CORE_COLS + feature_cols) if c in df.columns}
            d["ctx_timestamp"] = pd.Timestamp(t_ctx).tz_localize("UTC")
        else:
            # Missing minute: emit NaNs for all values.
            d = {c: np.nan for c in (CORE_COLS + feature_cols) if c in df.columns}
            d["ctx_timestamp"] = pd.Timestamp(t_ctx).tz_localize("UTC")

        d["trade_row"] = int(trade_row)
        d["phase"] = str(phase)
        d["rel_min"] = int(off)
        d["anchor_time"] = when
        for k, v in (trade_meta or {}).items():
            d[k] = v
        out.append(d)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-day optimal oracle trader + entry/exit 5m precontexts")
    ap.add_argument(
        "--dataset",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="Path to ETH-USD 1MIN dataset with features (CSV or Parquet)",
    )
    ap.add_argument("--symbol", default="ETH-USD", help="Symbol label used in filenames")
    ap.add_argument("--fee-total", type=float, default=0.001, help="TOTAL round-trip fee (e.g. 0.001 = 0.1% total)")
    ap.add_argument("--pre-min", type=int, default=5, help="Minutes of precontext to save")
    ap.add_argument("--eps", type=float, default=1e-12, help="Epsilon for tie-breaking equal value; prefers more trades")
    ap.add_argument("--max-days", type=int, default=0, help="If >0, process only the first N days (debug)")

    ap.add_argument("--out-trades-dir", default="data/oracle_trader")
    ap.add_argument("--out-precontext-dir", default="data/oracle_precontext")

    args = ap.parse_args()

    df = _parse_dataset(Path(args.dataset))

    feature_cols = [c for c in FEATURE_COLS_DEFAULT if c in df.columns]

    df["date"] = df["timestamp"].dt.date

    ts_sorted = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    px_all = df["close"].to_numpy(np.float64, copy=False)

    per_side_fee = float(args.fee_total) / 2.0
    pre_min = int(args.pre_min)
    rel_offsets = np.arange(-pre_min, 0, dtype=np.int64)

    all_trades: list[dict] = []
    entry_ctx_rows: list[dict] = []
    exit_ctx_rows: list[dict] = []

    groups = df.groupby("date", sort=True).groups
    days = list(groups.keys())
    if int(args.max_days) > 0:
        days = days[: int(args.max_days)]

    next_trade_row = 0

    # Iterate days using index ranges (fast; avoids copying dataframes per group).
    for di, day in enumerate(days):
        idxs = groups[day]
        if len(idxs) == 0:
            continue
        # group indices are sorted because df is sorted by timestamp.
        start = int(idxs[0])
        end = int(idxs[-1])

        ts_day = ts_sorted[start : end + 1]
        px_day = px_all[start : end + 1]

        # Per-day DP (force flat by end-of-day).
        state_path, _ = _run_daily_dp(px_day.astype(np.float64, copy=False), per_side_fee=per_side_fee, eps=float(args.eps))

        trades = _extract_trades_for_day(
            ts=ts_day,
            px=px_day,
            state=state_path,
            per_side_fee=per_side_fee,
            global_offset=start,
            date_utc=day,
        )

        if trades:
            # Assign a stable trade_row id for each trade (used to link to precontext rows).
            for tr in trades:
                tr["trade_row"] = int(next_trade_row)
                next_trade_row += 1
            all_trades.extend(trades)

            # Precontext extraction (entry and exit).
            for tr in trades:
                trade_row = int(tr["trade_row"])
                entry_time = pd.to_datetime(tr["entry_time"], utc=True)
                exit_time = pd.to_datetime(tr["exit_time"], utc=True)

                trade_meta = {
                    "trade_date": tr.get("date"),
                    "trade_entry_time": entry_time,
                    "trade_exit_time": exit_time,
                    "trade_duration_min": tr.get("duration_min"),
                    "trade_side": tr.get("side"),
                    "trade_net_return_pct": tr.get("net_return_pct"),
                    "trade_entry_idx": tr.get("entry_idx"),
                    "trade_exit_idx": tr.get("exit_idx"),
                }

                entry_ctx_rows.extend(
                    _precontext_rows(
                        df,
                        ts_sorted,
                        trade_row,
                        entry_time,
                        phase="pre_entry",
                        rel_offsets=rel_offsets,
                        feature_cols=feature_cols,
                        trade_meta=trade_meta,
                    )
                )
                exit_ctx_rows.extend(
                    _precontext_rows(
                        df,
                        ts_sorted,
                        trade_row,
                        exit_time,
                        phase="pre_exit",
                        rel_offsets=rel_offsets,
                        feature_cols=feature_cols,
                        trade_meta=trade_meta,
                    )
                )

        if (di + 1) % 50 == 0:
            print(f"Processed days: {di+1}/{len(days)}  trades_so_far={len(all_trades):,}")

    if not all_trades:
        raise SystemExit("No trades produced.")

    trades_df = pd.DataFrame(all_trades)
    # Sort by time for downstream tools, but keep trade_row stable.
    trades_df = trades_df.sort_values(["entry_time", "exit_time", "trade_row"]).reset_index(drop=True)

    # Summary stats
    mult = float(np.prod(1.0 + trades_df["net_return_pct"].to_numpy(np.float64) / 100.0))
    summary = pd.DataFrame(
        [
            dict(
                symbol=str(args.symbol),
                dataset=str(args.dataset),
                fee_total=float(args.fee_total),
                fee_per_side=float(per_side_fee),
                n_trades=int(len(trades_df)),
                final_multiplier=mult,
                final_return_pct=(mult - 1.0) * 100.0,
                start_ts=str(df["timestamp"].iloc[0]),
                end_ts=str(df["timestamp"].iloc[-1]),
                created_utc=_now_ts(),
            )
        ]
    )

    out_trades_dir = Path(args.out_trades_dir)
    out_trades_dir.mkdir(parents=True, exist_ok=True)

    out_pre_dir = Path(args.out_precontext_dir)
    out_pre_dir.mkdir(parents=True, exist_ok=True)

    ts_out = _now_ts()
    safe_sym = str(args.symbol).replace("/", "-")

    out_trades = out_trades_dir / f"oracle_daily_{safe_sym}_trades_{ts_out}.csv"
    out_summary = out_trades_dir / f"oracle_daily_{safe_sym}_summary_{ts_out}.csv"

    trades_df.to_csv(out_trades, index=False)
    summary.to_csv(out_summary, index=False)

    entry_ctx = pd.DataFrame(entry_ctx_rows)
    exit_ctx = pd.DataFrame(exit_ctx_rows)

    out_entry_parq = out_pre_dir / f"oracle_daily_{safe_sym}_pre_entry_dyn{pre_min}m_{ts_out}.parquet"
    out_entry_csv_sample = out_pre_dir / f"oracle_daily_{safe_sym}_pre_entry_dyn{pre_min}m_{ts_out}.sample.csv"

    out_exit_parq = out_pre_dir / f"oracle_daily_{safe_sym}_pre_exit_dyn{pre_min}m_{ts_out}.parquet"
    out_exit_csv_sample = out_pre_dir / f"oracle_daily_{safe_sym}_pre_exit_dyn{pre_min}m_{ts_out}.sample.csv"

    entry_ctx.to_parquet(out_entry_parq, index=False)
    exit_ctx.to_parquet(out_exit_parq, index=False)

    # CSV samples for quick inspection (full CSVs can be huge).
    entry_ctx.sample(n=min(20000, len(entry_ctx)), random_state=42).to_csv(out_entry_csv_sample, index=False)
    exit_ctx.sample(n=min(20000, len(exit_ctx)), random_state=42).to_csv(out_exit_csv_sample, index=False)

    print("Wrote trades:", out_trades)
    print("Wrote summary:", out_summary)
    print("Wrote entry precontext:", out_entry_parq)
    print("Wrote exit precontext:", out_exit_parq)
    print("Wrote entry precontext sample:", out_entry_csv_sample)
    print("Wrote exit precontext sample:", out_exit_csv_sample)


if __name__ == "__main__":
    main()
