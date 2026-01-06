#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-05T10:28:20Z
"""Build an ETH-USD (dYdX) SELL oracle-exit dataset from top-10% entry selections.

User intent
- We only care about exits *conditional on entry selection*.
- Use the current best SELL entry model to pick entry moments.
- Build an exit dataset covering the next 15 minutes after each chosen entry.
- Include (a) all decision-time features and (b) rolling PnL (%) at each minute.
- Also include 5-minute precontext at the oracle-exit moment as a helper view.

Definitions
- Entry decision at minute i (close[i]); entry fill at open[i+1].
- Exit decision at minute j (close[j]) for j=entry_idx+1 .. entry_idx+hold_min.
- SELL PnL at decision minute j:
    ret_if_exit_now_pct = net_ret_pct_sell(entry_px=open[entry_idx], exit_px=close[j], fee_side)
- Oracle exit within horizon: choose j that maximizes ret_if_exit_now_pct (equivalently min price).

Outputs (timestamped)
- data/exit_oracle_sell/
  - exit_oracle_rows_sell_hold15_top10_<ts>.parquet
    Per-minute rows: one row per (trade_id, k_rel=1..15) with:
      - decision-time model features (feature_cols from entry model)
      - rolling PnL (ret_if_exit_now_pct) and short history deltas
      - oracle_k / oracle_ret_pct / y_oracle_exit
  - exit_oracle_trades_sell_hold15_top10_<ts>.parquet
    Per-trade wide table: delta_k1..delta_k15 + oracle info.
  - exit_oracle_oracleexit_precontext_sell_hold15_top10_<ts>.parquet
    One row per trade at oracle exit (k_rel==oracle_k) with features and oracle_ret_pct.
  - run_meta_<ts>.json

Notes
- By default we enforce non-overlap between trades (blocked until entry_idx+hold_min).
  You can allow overlaps with --allow-overlap.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd


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

CORE_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def net_ret_pct_sell(entry_px: float, exit_px: float, fee_side: float) -> float:
    entry_px = float(entry_px)
    exit_px = float(exit_px)
    f = float(fee_side)
    if not (np.isfinite(entry_px) and np.isfinite(exit_px)):
        return float("nan")
    if entry_px <= 0.0 or exit_px <= 0.0:
        return float("nan")
    mult = (entry_px * (1.0 - f)) / (exit_px * (1.0 + f))
    return float((mult - 1.0) * 100.0)


def _future_extrema_excl_current(x: np.ndarray, W: int, *, mode: str) -> np.ndarray:
    """For each i, return extreme over x[i+1:i+W+1] (min/max). O(n) via deque on reversed series."""
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


def _rolling_sum_nan_prefix(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    cs = np.cumsum(x)
    out[w - 1 :] = cs[w - 1 :] - np.concatenate(([0.0], cs[: n - w]))
    return out


def compute_ctx_series(bars: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Compute causal context series at each minute using only t-1 and earlier."""
    close_prev = pd.to_numeric(bars["close"], errors="coerce").shift(1)
    high_prev = pd.to_numeric(bars["high"], errors="coerce").shift(1)
    low_prev = pd.to_numeric(bars["low"], errors="coerce").shift(1)
    vol_prev = pd.to_numeric(bars["volume"], errors="coerce").shift(1)

    out = pd.DataFrame(index=bars.index)

    for w in windows:
        w = int(w)
        out[f"mom_{w}m_pct"] = close_prev.pct_change(w) * 100.0
        out[f"vol_std_{w}m"] = close_prev.rolling(w, min_periods=w).std(ddof=0)

        rng = high_prev.rolling(w, min_periods=w).max() - low_prev.rolling(w, min_periods=w).min()
        out[f"range_{w}m"] = rng
        out[f"range_norm_{w}m"] = rng / close_prev.replace(0, np.nan)

        cmax = close_prev.rolling(w, min_periods=w).max()
        cmin = close_prev.rolling(w, min_periods=w).min()
        crng = cmax - cmin
        eps = 1e-9

        out[f"close_dd_from_{w}m_max_pct"] = (cmax / close_prev.replace(0, np.nan) - 1.0) * 100.0
        out[f"close_bounce_from_{w}m_min_pct"] = (close_prev / cmin.replace(0, np.nan) - 1.0) * 100.0
        out[f"close_pos_in_{w}m_range"] = (close_prev - cmin) / (crng + eps)

        v = vol_prev.to_numpy(np.float64, copy=False)
        c = close_prev.to_numpy(np.float64, copy=False)
        sum_v = _rolling_sum_nan_prefix(np.nan_to_num(v, nan=0.0), w)
        sum_pv = _rolling_sum_nan_prefix(np.nan_to_num(v, nan=0.0) * np.nan_to_num(c, nan=0.0), w)
        vwap = sum_pv / np.maximum(1e-9, sum_v)
        vwap_dev = np.where(sum_v > 0.0, ((c - vwap) / np.maximum(1e-9, vwap)) * 100.0, 0.0)
        out[f"vwap_dev_{w}m"] = vwap_dev

    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float32)

    return out


def _safe_nanstd_rows(x: np.ndarray) -> np.ndarray:
    import warnings

    out = np.full(x.shape[0], np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m = np.nanmean(x, axis=1)
        ok = np.isfinite(m)
        if np.any(ok):
            out[ok] = np.nanmean((x[ok] - m[ok, None]) ** 2, axis=1) ** 0.5
    return out


def _slope_5_rows(y: np.ndarray) -> np.ndarray:
    xc = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    return np.nansum(y * xc[None, :], axis=1) / 10.0


def _agg5_rows(x: np.ndarray) -> dict[str, np.ndarray]:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean5 = np.nanmean(x, axis=1)
        min5 = np.nanmin(x, axis=1)
        max5 = np.nanmax(x, axis=1)

    return {
        "__last": x[:, -1],
        "__mean5": mean5,
        "__std5": _safe_nanstd_rows(x),
        "__min5": min5,
        "__max5": max5,
        "__range5": max5 - min5,
        "__slope5": _slope_5_rows(x),
    }


def build_X_for_indices(
    *,
    bars: pd.DataFrame,
    feature_cols: list[str],
    ctx_windows: list[int],
    idx: np.ndarray,
) -> np.ndarray:
    """Build model feature matrix (columns=feature_cols) for the given indices.

    Features follow entry-model convention: for decision minute t=idx, use only t-5..t-1.
    """
    ii = np.asarray(idx, dtype=np.int64)

    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(np.float64, copy=False)
    volume = pd.to_numeric(bars["volume"], errors="coerce").to_numpy(np.float64, copy=False)

    base_arrs = {c: pd.to_numeric(bars[c], errors="coerce").to_numpy(np.float64, copy=False) for c in BASE_FEATS_DEFAULT}
    ctx_df = compute_ctx_series(bars, ctx_windows)
    ctx_arrs = {c: pd.to_numeric(ctx_df[c], errors="coerce").to_numpy(np.float64, copy=False) for c in ctx_df.columns}

    col_to_j = {c: j for j, c in enumerate(feature_cols)}

    prefixes = set()
    for c in feature_cols:
        if "__" in c:
            prefixes.add(c.split("__", 1)[0])
    prefixes.discard("px_close_norm_pct")
    prefixes.discard("vol_log1p")

    series_map: dict[str, np.ndarray] = {}
    for p in prefixes:
        if p in base_arrs:
            series_map[p] = base_arrs[p]
        elif p in ctx_arrs:
            series_map[p] = ctx_arrs[p]

    m = int(ii.size)
    X = np.full((m, len(feature_cols)), np.nan, dtype=np.float32)

    def _set(name: str, arr: np.ndarray) -> None:
        j = col_to_j.get(name)
        if j is None:
            return
        X[:, j] = np.asarray(arr, dtype=np.float32)

    # indices for t-5..t-1
    i0 = ii - 5
    i1 = ii - 4
    i2 = ii - 3
    i3 = ii - 2
    i4 = ii - 1

    close_w = np.stack([close[i0], close[i1], close[i2], close[i3], close[i4]], axis=1)
    close_last = close_w[:, -1]
    close_last = np.where((np.isfinite(close_last)) & (close_last > 0.0), close_last, np.nan)
    close_norm = (close_w / close_last[:, None] - 1.0) * 100.0

    px = _agg5_rows(close_norm)
    for suf, arr in px.items():
        _set(f"px_close_norm_pct{suf}", arr)
    px_ret5m = close_norm[:, -1] - close_norm[:, 0]
    _set("px_close_norm_pct__ret5m", px_ret5m)
    _set("px_close_norm_pct__absret5m", np.abs(px_ret5m))
    _set("px_close_norm_pct__m5", close_norm[:, 0])
    _set("px_close_norm_pct__m4", close_norm[:, 1])
    _set("px_close_norm_pct__m3", close_norm[:, 2])
    _set("px_close_norm_pct__m2", close_norm[:, 3])
    _set("px_close_norm_pct__m1", close_norm[:, 4])

    # vol_log1p
    vol_w = np.stack([volume[i0], volume[i1], volume[i2], volume[i3], volume[i4]], axis=1)
    vol_log = np.log1p(np.maximum(0.0, vol_w))
    vv = _agg5_rows(vol_log)
    for suf, arr in vv.items():
        _set(f"vol_log1p{suf}", arr)

    if "missing_close_n" in col_to_j or "missing_any" in col_to_j:
        miss = ~np.isfinite(close_w)
        miss_n = miss.sum(axis=1).astype(np.float64)
        _set("missing_close_n", miss_n)
        _set("missing_any", (miss_n > 0).astype(np.float64))

    for name, arr0 in series_map.items():
        w = np.stack([arr0[i0], arr0[i1], arr0[i2], arr0[i3], arr0[i4]], axis=1)
        a = _agg5_rows(w)
        for suf, v in a.items():
            _set(f"{name}{suf}", v)

    return X


@dataclass(frozen=True)
class Trade:
    trade_id: int
    signal_i: int
    entry_idx: int
    entry_time: pd.Timestamp
    entry_px: float
    entry_pred: float


def select_trades(
    *,
    ts_utc: pd.Series,
    open_arr: np.ndarray,
    pred_by_i: np.ndarray,
    thr: float,
    hold_min: int,
    pre_min: int,
    allow_overlap: bool,
) -> list[Trade]:
    n = len(ts_utc)
    last_signal_i = n - 2 - int(hold_min)

    trades: list[Trade] = []
    blocked_until_i = -10**18
    trade_id = 0

    start_i = max(0, int(pre_min))

    for i in range(int(start_i), int(last_signal_i) + 1):
        if (not allow_overlap) and i < int(blocked_until_i):
            continue

        p = float(pred_by_i[i])
        if not np.isfinite(p):
            continue
        if p < float(thr):
            continue

        entry_idx = int(i) + 1
        entry_px = float(open_arr[entry_idx])
        if not (np.isfinite(entry_px) and entry_px > 0.0):
            continue

        trades.append(
            Trade(
                trade_id=int(trade_id),
                signal_i=int(i),
                entry_idx=int(entry_idx),
                entry_time=pd.to_datetime(ts_utc.iloc[entry_idx], utc=True),
                entry_px=float(entry_px),
                entry_pred=float(p),
            )
        )
        trade_id += 1

        if not allow_overlap:
            blocked_until_i = entry_idx + int(hold_min)

    return trades


def main() -> None:
    ap = argparse.ArgumentParser(description="Build SELL exit-oracle dataset from top-10% entry selections")
    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
        help="ETH 1m OHLCV+features parquet",
    )
    ap.add_argument(
        "--entry-model",
        default="data/entry_regressor_oracle15m_sell_ctx120_weighted_2026-01-04T19-54-19Z/models/entry_regressor_sell_oracle15m_ctx120_weighted_2026-01-04T19-54-19Z.joblib",
        help="SELL entry regressor artifact (dict with {model, feature_cols, ...})",
    )
    ap.add_argument("--top-frac", type=float, default=0.10, help="Entry selection fraction by prediction quantile")
    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--pre-min", type=int, default=5)
    ap.add_argument("--fee-total", type=float, default=0.001, help="TOTAL round-trip fee")
    ap.add_argument("--allow-overlap", action="store_true")

    ap.add_argument("--ctx-windows", default="30,60,120")
    ap.add_argument("--chunk", type=int, default=120_000)

    ap.add_argument("--out-dir", default="data/exit_oracle_sell")

    args = ap.parse_args()

    bars_path = Path(args.bars)
    if not bars_path.exists():
        raise SystemExit(f"bars not found: {bars_path}")

    entry_art = joblib.load(args.entry_model)
    if not isinstance(entry_art, dict) or "model" not in entry_art or "feature_cols" not in entry_art:
        raise SystemExit("entry model artifact must be a dict with keys: model, feature_cols")

    model = entry_art["model"]
    feature_cols = list(entry_art["feature_cols"])

    hold_min = int(args.hold_min)
    pre_min = int(args.pre_min)
    fee_side = float(args.fee_total) / 2.0

    ctx_windows = [int(x.strip()) for x in str(args.ctx_windows).split(",") if x.strip()]

    need_cols = CORE_COLS + list(BASE_FEATS_DEFAULT)

    print(f"Loading bars: {bars_path}", flush=True)
    bars = pd.read_parquet(bars_path, columns=need_cols)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    if not bars["timestamp"].is_monotonic_increasing:
        bars = bars.sort_values("timestamp").reset_index(drop=True)

    n = int(len(bars))
    ts_utc = bars["timestamp"]

    open_arr = pd.to_numeric(bars["open"], errors="coerce").to_numpy(np.float64, copy=False)
    close_arr = pd.to_numeric(bars["close"], errors="coerce").to_numpy(np.float64, copy=False)

    # oracle label (for reference): best possible exit within hold_min
    fut_min = _future_extrema_excl_current(close_arr, hold_min, mode="min")
    y_sell_oracle15m = np.asarray([net_ret_pct_sell(c, f, fee_side) for c, f in zip(close_arr, fut_min)], dtype=np.float64)

    valid = (np.arange(n) >= pre_min) & np.isfinite(close_arr) & np.isfinite(y_sell_oracle15m)
    last_signal_i = n - 2 - hold_min
    valid_signal = valid & (np.arange(n) <= last_signal_i)
    signal_idx = np.where(valid_signal)[0]

    print(f"Scoring entry model on {signal_idx.size:,} candidate minutes...", flush=True)
    pred = np.full(n, np.nan, dtype=np.float32)

    for s in range(0, signal_idx.size, int(args.chunk)):
        e = min(signal_idx.size, s + int(args.chunk))
        ii = signal_idx[s:e]
        Xc = build_X_for_indices(bars=bars, feature_cols=feature_cols, ctx_windows=ctx_windows, idx=ii)
        pred[ii] = model.predict(Xc).astype(np.float32)
        if (s // int(args.chunk)) % 2 == 0:
            print(f"  scored {e:,}/{signal_idx.size:,}", flush=True)

    pv = pred[signal_idx].astype(np.float64)
    pv = pv[np.isfinite(pv)]
    if pv.size == 0:
        raise SystemExit("No finite predictions")

    top_frac = float(args.top_frac)
    thr = float(np.quantile(pv, 1.0 - top_frac))
    print(f"Top-{top_frac*100:.2f}% entry threshold (pred): {thr:.6g}", flush=True)

    print("Selecting trades...", flush=True)
    trades = select_trades(
        ts_utc=ts_utc,
        open_arr=open_arr,
        pred_by_i=pred,
        thr=thr,
        hold_min=hold_min,
        pre_min=pre_min,
        allow_overlap=bool(args.allow_overlap),
    )

    if not trades:
        raise SystemExit("No trades selected. Something is wrong with thresholding.")

    print(f"Trades selected: {len(trades):,}  (allow_overlap={bool(args.allow_overlap)})", flush=True)

    # Build per-trade delta paths and per-minute rows
    trade_rows: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []

    for tr in trades:
        entry_idx = int(tr.entry_idx)
        entry_px = float(tr.entry_px)

        # ensure horizon exists
        if entry_idx + hold_min >= n:
            continue

        # delta path for each minute k=1..hold
        rets = []
        for k in range(1, hold_min + 1):
            j = entry_idx + k
            rets.append(net_ret_pct_sell(entry_px, float(close_arr[j]), fee_side))
        rets_arr = np.asarray(rets, dtype=np.float64)
        if not np.isfinite(rets_arr).any():
            continue

        oracle_k = int(np.nanargmax(rets_arr) + 1)
        oracle_ret = float(np.nanmax(rets_arr))

        trow: dict[str, object] = {
            "trade_id": int(tr.trade_id),
            "signal_i": int(tr.signal_i),
            "entry_idx": int(entry_idx),
            "entry_time": tr.entry_time,
            "entry_px": float(entry_px),
            "entry_pred": float(tr.entry_pred),
            "oracle_k": int(oracle_k),
            "oracle_ret_pct": float(oracle_ret),
        }
        for k in range(1, hold_min + 1):
            trow[f"delta_k{k:02d}_pct"] = float(rets_arr[k - 1])
        trade_rows.append(trow)

        peak = -1e18
        for k in range(1, hold_min + 1):
            j = entry_idx + k
            cur_ret = float(rets_arr[k - 1])
            peak = max(peak, cur_ret)

            r_prev1 = float(rets_arr[k - 2]) if k >= 2 else float("nan")
            r_prev2 = float(rets_arr[k - 3]) if k >= 3 else float("nan")
            r_prev3 = float(rets_arr[k - 4]) if k >= 4 else float("nan")

            row: dict[str, object] = {
                "trade_id": int(tr.trade_id),
                "signal_i": int(tr.signal_i),
                "entry_idx": int(entry_idx),
                "entry_time": tr.entry_time,
                "entry_px": float(entry_px),
                "entry_pred": float(tr.entry_pred),
                "top_frac": float(top_frac),
                "entry_pred_threshold": float(thr),
                "hold_min": int(hold_min),
                "oracle_k": int(oracle_k),
                "oracle_ret_pct": float(oracle_ret),
                "k_rel": int(k),
                "decision_idx": int(j),
                "decision_time": pd.to_datetime(ts_utc.iloc[j], utc=True),
                "y_oracle_exit": int(1 if int(k) == int(oracle_k) else 0),
                "ret_if_exit_now_pct": float(cur_ret),
                "oracle_gap_pct": float(oracle_ret - cur_ret),
                "mins_in_trade": int(k),
                "mins_remaining": int(hold_min - k),
                "delta_mark_pct": float(cur_ret),
                "delta_mark_prev1_pct": r_prev1,
                "delta_mark_prev2_pct": r_prev2,
                "delta_mark_change_1m": (cur_ret - r_prev1) if np.isfinite(r_prev1) else float("nan"),
                "delta_mark_change_2m": (cur_ret - r_prev2) if np.isfinite(r_prev2) else float("nan"),
                "delta_mark_change_3m": (cur_ret - r_prev3) if np.isfinite(r_prev3) else float("nan"),
                "drawdown_from_peak_pct": float(cur_ret - peak),
            }

            # attach decision-time features (entry-model feature space)
            Xj = build_X_for_indices(bars=bars, feature_cols=feature_cols, ctx_windows=ctx_windows, idx=np.asarray([j], dtype=np.int64))
            for col_i, name in enumerate(feature_cols):
                v = float(Xj[0, col_i])
                row[name] = v if np.isfinite(v) else float("nan")

            rows.append(row)

    ds_rows = pd.DataFrame(rows)
    ds_trades = pd.DataFrame(trade_rows)

    if ds_rows.empty:
        raise SystemExit("Built empty dataset")

    # convenience: oracle-exit precontext rows
    ds_oracle = ds_rows[ds_rows["y_oracle_exit"] == 1].copy().reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = now_ts()

    out_rows = out_dir / f"exit_oracle_rows_sell_hold{hold_min}_top10_{ts}.parquet"
    out_trades = out_dir / f"exit_oracle_trades_sell_hold{hold_min}_top10_{ts}.parquet"
    out_oracle = out_dir / f"exit_oracle_oracleexit_precontext_sell_hold{hold_min}_top10_{ts}.parquet"
    meta_path = out_dir / f"run_meta_{ts}.json"

    ds_rows.to_parquet(out_rows, index=False)
    ds_trades.to_parquet(out_trades, index=False)
    ds_oracle.to_parquet(out_oracle, index=False)

    run_meta = {
        "created_utc": ts,
        "bars": str(bars_path),
        "entry_model": str(Path(args.entry_model)),
        "top_frac": float(top_frac),
        "hold_min": int(hold_min),
        "pre_min": int(pre_min),
        "fee_total": float(args.fee_total),
        "fee_side": float(fee_side),
        "allow_overlap": bool(args.allow_overlap),
        "ctx_windows": ctx_windows,
        "entry_pred_threshold": float(thr),
        "n_rows": int(len(ds_rows)),
        "n_trades": int(ds_rows["trade_id"].nunique()),
        "columns_rows": list(ds_rows.columns),
        "columns_trades": list(ds_trades.columns),
        "feature_cols": feature_cols,
    }
    meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(f"Saved rows:   {out_rows}")
    print(f"Saved trades: {out_trades}")
    print(f"Saved oracle: {out_oracle}")
    print(f"Saved meta:   {meta_path}")


if __name__ == "__main__":
    main()
