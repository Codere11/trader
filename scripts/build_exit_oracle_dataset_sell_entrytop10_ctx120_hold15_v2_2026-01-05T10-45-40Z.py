#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-05T10:45:40Z
"""Build an ETH-USD (dYdX) SELL exit-oracle dataset from top-10% entry selections (fast, streaming).

User specs
- Use the (best) SELL entry scorer to select entries (top 10% by pred).
- For each chosen entry, include the next 15 minutes (k=1..15):
  - ALL model features at each decision minute
  - rolling trade % delta (PnL in %) at each minute (and short history)
- Also produce a helper view:
  - rows at the oracle exit time (k==oracle_k) with 5-minute precontext features + oracle_ret_pct.
- Intended for training a 500-tree exit regressor.

Key points
- Entry decision at close[i] => entry at open[i+1].
- Decision minutes are close[entry_idx + k] for k=1..hold_min.
- SELL PnL at decision minute: ret_if_exit_now_pct = net_ret_pct_sell(entry_open, close_at_k, fee_side).
- Oracle exit = argmax over k of ret_if_exit_now_pct.

Outputs (timestamped)
- data/exit_oracle_sell/
  - exit_oracle_rows_sell_hold15_top10_<ts>.parquet (streamed)
  - exit_oracle_trades_sell_hold15_top10_<ts>.parquet (wide delta paths)
  - exit_oracle_oracleexit_precontext_sell_hold15_top10_<ts>.parquet (one row per trade at oracle exit)
  - run_meta_<ts>.json

Perf
- Avoids recomputing ctx series per row; builds features in chunks.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import joblib
import pyarrow as pa
import pyarrow.parquet as pq


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


def net_ret_pct_sell(entry_px: np.ndarray, exit_px: np.ndarray, fee_side: float) -> np.ndarray:
    entry_px = np.asarray(entry_px, dtype=np.float64)
    exit_px = np.asarray(exit_px, dtype=np.float64)
    f = float(fee_side)
    mult = (entry_px * (1.0 - f)) / (exit_px * (1.0 + f))
    return (mult - 1.0) * 100.0


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


@dataclass(frozen=True)
class Trade:
    trade_id: int
    signal_i: int
    entry_idx: int
    entry_time: pd.Timestamp
    entry_px: float
    entry_pred: float


class FeatureBuilder:
    def __init__(
        self,
        *,
        close: np.ndarray,
        volume: np.ndarray,
        base_arrs: dict[str, np.ndarray],
        ctx_arrs: dict[str, np.ndarray],
        feature_cols: list[str],
    ) -> None:
        self.close = np.asarray(close, dtype=np.float64)
        self.volume = np.asarray(volume, dtype=np.float64)
        self.feature_cols = list(feature_cols)
        self.col_to_j = {c: j for j, c in enumerate(self.feature_cols)}

        prefixes = set()
        for c in self.feature_cols:
            if "__" in c:
                prefixes.add(c.split("__", 1)[0])
        prefixes.discard("px_close_norm_pct")
        prefixes.discard("vol_log1p")

        series_map: dict[str, np.ndarray] = {}
        for p in prefixes:
            if p in base_arrs:
                series_map[p] = np.asarray(base_arrs[p], dtype=np.float64)
            elif p in ctx_arrs:
                series_map[p] = np.asarray(ctx_arrs[p], dtype=np.float64)
        self.series_map = series_map

    def build_X(self, idx: np.ndarray) -> np.ndarray:
        ii = np.asarray(idx, dtype=np.int64)
        m = int(ii.size)
        X = np.full((m, len(self.feature_cols)), np.nan, dtype=np.float32)

        def _set(name: str, arr: np.ndarray) -> None:
            j = self.col_to_j.get(name)
            if j is None:
                return
            X[:, j] = np.asarray(arr, dtype=np.float32)

        i0 = ii - 5
        i1 = ii - 4
        i2 = ii - 3
        i3 = ii - 2
        i4 = ii - 1

        close_w = np.stack([self.close[i0], self.close[i1], self.close[i2], self.close[i3], self.close[i4]], axis=1)
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

        vol_w = np.stack([self.volume[i0], self.volume[i1], self.volume[i2], self.volume[i3], self.volume[i4]], axis=1)
        vol_log = np.log1p(np.maximum(0.0, vol_w))
        vv = _agg5_rows(vol_log)
        for suf, arr in vv.items():
            _set(f"vol_log1p{suf}", arr)

        if "missing_close_n" in self.col_to_j or "missing_any" in self.col_to_j:
            miss = ~np.isfinite(close_w)
            miss_n = miss.sum(axis=1).astype(np.float64)
            _set("missing_close_n", miss_n)
            _set("missing_any", (miss_n > 0).astype(np.float64))

        for name, arr0 in self.series_map.items():
            w = np.stack([arr0[i0], arr0[i1], arr0[i2], arr0[i3], arr0[i4]], axis=1)
            a = _agg5_rows(w)
            for suf, v in a.items():
                _set(f"{name}{suf}", v)

        return X


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
    ap = argparse.ArgumentParser(description="Build SELL exit-oracle dataset from top-10% entry selections (streaming)")
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
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--pre-min", type=int, default=5)
    ap.add_argument("--fee-total", type=float, default=0.001)
    ap.add_argument("--allow-overlap", action="store_true")

    ap.add_argument("--ctx-windows", default="30,60,120")
    ap.add_argument("--score-chunk", type=int, default=120_000)
    ap.add_argument("--write-chunk", type=int, default=60_000, help="Rows per parquet write chunk")

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
    volume_arr = pd.to_numeric(bars["volume"], errors="coerce").to_numpy(np.float64, copy=False)

    # ctx series (once)
    print(f"Computing ctx series (windows={ctx_windows}) ...", flush=True)
    ctx_df = compute_ctx_series(bars, ctx_windows)
    ctx_arrs = {c: pd.to_numeric(ctx_df[c], errors="coerce").to_numpy(np.float64, copy=False) for c in ctx_df.columns}

    base_arrs = {c: pd.to_numeric(bars[c], errors="coerce").to_numpy(np.float64, copy=False) for c in BASE_FEATS_DEFAULT}

    fb = FeatureBuilder(close=close_arr, volume=volume_arr, base_arrs=base_arrs, ctx_arrs=ctx_arrs, feature_cols=feature_cols)

    # Candidate indices for entry decisions
    last_signal_i = n - 2 - hold_min
    valid_signal = (np.arange(n) >= pre_min) & (np.arange(n) <= last_signal_i) & np.isfinite(close_arr)
    signal_idx = np.where(valid_signal)[0]

    print(f"Scoring entry model on {signal_idx.size:,} candidate minutes...", flush=True)
    pred = np.full(n, np.nan, dtype=np.float32)

    sc = int(args.score_chunk)
    for s in range(0, signal_idx.size, sc):
        e = min(signal_idx.size, s + sc)
        ii = signal_idx[s:e]
        Xc = fb.build_X(ii)
        pred[ii] = model.predict(Xc).astype(np.float32)
        if (s // sc) % 2 == 0:
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
        raise SystemExit("No trades selected")

    print(f"Trades selected: {len(trades):,} (allow_overlap={bool(args.allow_overlap)})", flush=True)

    # Convert trades to arrays
    T = int(len(trades))
    trade_id = np.asarray([t.trade_id for t in trades], dtype=np.int64)
    signal_i = np.asarray([t.signal_i for t in trades], dtype=np.int64)
    entry_idx = np.asarray([t.entry_idx for t in trades], dtype=np.int64)
    entry_time = np.asarray([np.datetime64(t.entry_time.to_datetime64()) for t in trades])
    entry_px = np.asarray([t.entry_px for t in trades], dtype=np.float64)
    entry_pred = np.asarray([t.entry_pred for t in trades], dtype=np.float32)

    k = np.arange(1, hold_min + 1, dtype=np.int64)
    decision_idx_mat = entry_idx[:, None] + k[None, :]

    exit_close = close_arr[decision_idx_mat]
    ret_mat = net_ret_pct_sell(entry_px[:, None], exit_close, fee_side).astype(np.float32)

    oracle_k = (np.nanargmax(ret_mat.astype(np.float64), axis=1) + 1).astype(np.int16)
    oracle_ret = np.nanmax(ret_mat.astype(np.float64), axis=1).astype(np.float32)

    # Per-trade wide table (delta path)
    trades_df = pd.DataFrame(
        {
            "trade_id": trade_id,
            "signal_i": signal_i,
            "entry_idx": entry_idx,
            "entry_time": pd.to_datetime(entry_time, utc=True),
            "entry_px": entry_px.astype(np.float64),
            "entry_pred": entry_pred.astype(np.float32),
            "oracle_k": oracle_k.astype(np.int16),
            "oracle_ret_pct": oracle_ret.astype(np.float32),
        }
    )
    for kk in range(1, hold_min + 1):
        trades_df[f"delta_k{kk:02d}_pct"] = ret_mat[:, kk - 1].astype(np.float32)

    # Oracle-exit precontext table (features at decision_idx = entry_idx + oracle_k)
    oracle_dec_idx = (entry_idx + oracle_k.astype(np.int64)).astype(np.int64)

    # Stream per-minute rows to parquet
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = now_ts()

    out_rows = out_dir / f"exit_oracle_rows_sell_hold{hold_min}_top10_{ts}.parquet"
    out_trades = out_dir / f"exit_oracle_trades_sell_hold{hold_min}_top10_{ts}.parquet"
    out_oracle = out_dir / f"exit_oracle_oracleexit_precontext_sell_hold{hold_min}_top10_{ts}.parquet"
    meta_path = out_dir / f"run_meta_{ts}.json"

    print(f"Writing trades table: {out_trades}", flush=True)
    trades_df.to_parquet(out_trades, index=False)

    # Flat arrays for per-minute rows
    N = int(T * hold_min)
    trade_id_f = np.repeat(trade_id, hold_min)
    signal_i_f = np.repeat(signal_i, hold_min)
    entry_idx_f = np.repeat(entry_idx, hold_min)
    entry_time_f = np.repeat(entry_time, hold_min)
    entry_px_f = np.repeat(entry_px.astype(np.float64), hold_min)
    entry_pred_f = np.repeat(entry_pred.astype(np.float32), hold_min)

    k_rel_f = np.tile(k.astype(np.int16), T)
    decision_idx_f = decision_idx_mat.reshape(-1)

    ret_f = ret_mat.reshape(-1)

    # oracle per row
    oracle_k_f = np.repeat(oracle_k.astype(np.int16), hold_min)
    oracle_ret_f = np.repeat(oracle_ret.astype(np.float32), hold_min)
    y_oracle_exit_f = (k_rel_f.astype(np.int16) == oracle_k_f).astype(np.int8)
    oracle_gap_f = (oracle_ret_f - ret_f).astype(np.float32)

    # prev deltas and drawdown-from-peak
    prev1 = np.concatenate([np.full((T, 1), np.nan, dtype=np.float32), ret_mat[:, :-1]], axis=1)
    prev2 = np.concatenate([np.full((T, 2), np.nan, dtype=np.float32), ret_mat[:, :-2]], axis=1)
    prev3 = np.concatenate([np.full((T, 3), np.nan, dtype=np.float32), ret_mat[:, :-3]], axis=1)

    prev1_f = prev1.reshape(-1)
    prev2_f = prev2.reshape(-1)
    prev3_f = prev3.reshape(-1)

    ch1_f = (ret_f - prev1_f).astype(np.float32)
    ch2_f = (ret_f - prev2_f).astype(np.float32)
    ch3_f = (ret_f - prev3_f).astype(np.float32)

    peak = np.maximum.accumulate(ret_mat.astype(np.float64), axis=1).astype(np.float32)
    drawdown = (ret_mat - peak).astype(np.float32)
    drawdown_f = drawdown.reshape(-1)

    mins_in_trade_f = k_rel_f.astype(np.int16)
    mins_remaining_f = (hold_min - k_rel_f.astype(np.int64)).astype(np.int16)

    # decision times
    decision_time_f = ts_utc.iloc[decision_idx_f].to_numpy()

    write_chunk = int(args.write_chunk)

    writer: pq.ParquetWriter | None = None

    def write_table(df_chunk: pd.DataFrame) -> None:
        nonlocal writer
        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_rows.as_posix(), table.schema, compression="zstd")
        writer.write_table(table)

    print(f"Writing per-minute rows (N={N:,}) to {out_rows} ...", flush=True)

    for s in range(0, N, write_chunk):
        e = min(N, s + write_chunk)
        ii = np.arange(s, e, dtype=np.int64)

        idx_dec = decision_idx_f[s:e]
        Xc = fb.build_X(idx_dec)

        df_chunk = pd.DataFrame(
            {
                "trade_id": trade_id_f[s:e].astype(np.int64),
                "signal_i": signal_i_f[s:e].astype(np.int64),
                "entry_idx": entry_idx_f[s:e].astype(np.int64),
                "entry_time": pd.to_datetime(entry_time_f[s:e], utc=True),
                "entry_px": entry_px_f[s:e].astype(np.float64),
                "entry_pred": entry_pred_f[s:e].astype(np.float32),
                "top_frac": float(top_frac),
                "entry_pred_threshold": float(thr),
                "hold_min": int(hold_min),
                "oracle_k": oracle_k_f[s:e].astype(np.int16),
                "oracle_ret_pct": oracle_ret_f[s:e].astype(np.float32),
                "k_rel": k_rel_f[s:e].astype(np.int16),
                "decision_idx": idx_dec.astype(np.int64),
                "decision_time": pd.to_datetime(decision_time_f[s:e], utc=True),
                "y_oracle_exit": y_oracle_exit_f[s:e].astype(np.int8),
                "ret_if_exit_now_pct": ret_f[s:e].astype(np.float32),
                "oracle_gap_pct": oracle_gap_f[s:e].astype(np.float32),
                "mins_in_trade": mins_in_trade_f[s:e].astype(np.int16),
                "mins_remaining": mins_remaining_f[s:e].astype(np.int16),
                "delta_mark_pct": ret_f[s:e].astype(np.float32),
                "delta_mark_prev1_pct": prev1_f[s:e].astype(np.float32),
                "delta_mark_prev2_pct": prev2_f[s:e].astype(np.float32),
                "delta_mark_change_1m": ch1_f[s:e].astype(np.float32),
                "delta_mark_change_2m": ch2_f[s:e].astype(np.float32),
                "delta_mark_change_3m": ch3_f[s:e].astype(np.float32),
                "drawdown_from_peak_pct": drawdown_f[s:e].astype(np.float32),
            }
        )

        # attach feature columns
        feat_df = pd.DataFrame(Xc, columns=feature_cols)
        df_chunk = pd.concat([df_chunk, feat_df], axis=1)

        write_table(df_chunk)

        if (s // write_chunk) % 10 == 0:
            print(f"  wrote {e:,}/{N:,}", flush=True)

    if writer is not None:
        writer.close()

    # Oracle exit precontext (features at oracle decision idx)
    print(f"Writing oracle-exit precontext rows: {out_oracle}", flush=True)
    X_oracle = []
    chunk = 120_000
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        X_oracle.append(fb.build_X(oracle_dec_idx[s:e]))
    Xo = np.vstack(X_oracle).astype(np.float32)

    oracle_df = pd.DataFrame(
        {
            "trade_id": trade_id,
            "signal_i": signal_i,
            "entry_idx": entry_idx,
            "entry_time": pd.to_datetime(entry_time, utc=True),
            "entry_px": entry_px.astype(np.float64),
            "entry_pred": entry_pred.astype(np.float32),
            "oracle_k": oracle_k.astype(np.int16),
            "oracle_ret_pct": oracle_ret.astype(np.float32),
            "oracle_decision_idx": oracle_dec_idx.astype(np.int64),
            "oracle_decision_time": pd.to_datetime(ts_utc.iloc[oracle_dec_idx].to_numpy(), utc=True),
        }
    )
    oracle_df = pd.concat([oracle_df, pd.DataFrame(Xo, columns=feature_cols)], axis=1)
    oracle_df.to_parquet(out_oracle, index=False)

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
        "n_trades": int(T),
        "n_rows": int(N),
        "feature_cols": feature_cols,
        "paths": {
            "rows": out_rows.as_posix(),
            "trades": out_trades.as_posix(),
            "oracle_exit_precontext": out_oracle.as_posix(),
        },
    }
    meta_path.write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")

    print("Done.", flush=True)
    print("Rows:", out_rows, flush=True)
    print("Trades:", out_trades, flush=True)
    print("Oracle:", out_oracle, flush=True)
    print("Meta:", meta_path, flush=True)


if __name__ == "__main__":
    main()
