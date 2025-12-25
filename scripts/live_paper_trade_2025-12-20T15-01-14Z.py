#!/usr/bin/env python3
"""
Minute-level paper trading runner (replay or follow) with strict feature parity.
- Ensures the model sees the exact same feature columns/order as in training by reading payload["features"] and payload["context"].
- Entry: regressor score vs causal per-day threshold; execute at next-minute open.
- Exit: greedy_causal on minute closes up to 10 minutes; auto-close at 10.
- Replay mode (default): fetch 1m futures klines via REST for a given period and simulate decisions minute-by-minute.

Outputs under --out-dir (default data/live):
- minute_bars_{date}.csv (raw bars as used)
- trades_{ts}.csv, daily_{ts}.csv (same format as evaluator/backtests)

NOTE: For exact feature parity, this runner uses compute_feature_frame() and reconstructs any context columns found in the entry model payload.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
import subprocess
import sys
from typing import Optional, List, Dict, Any, Tuple, Union, Mapping

import joblib
import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from contracts import AuditLogger
from contracts.v1 import (
    KIND_ENTRY_DECISION,
    KIND_MARKET_BAR_1M_CLOSED,
    KIND_RUN_META,
    KIND_TRADE_CLOSED,
    KIND_TRADE_MODE_SWITCH,
)

from binance_adapter.crossref_oracle_vs_agent_patterns import FEATURES, compute_feature_frame
from binance_adapter.futures_exec import (
    BinanceAPIError,
    BinanceUSDMFuturesClient,
    FuturesLongOnlyExecutor,
    fetch_futures_exchange_info,
    fetch_spot_price,
)

from live_state import (
    GateState,
    StrategyLedger,
    load_gate_state,
    load_strategy_ledger,
    save_gate_state,
    save_strategy_ledger,
)

BINANCE_FAPI = "https://fapi.binance.com"

@dataclass
class Models:
    entry_model: Any
    exit_model: Any
    entry_features: List[str]
    exit_features: List[str]

    # Context windows used during training (if present in the artifacts).
    entry_pre_min: int
    exit_pre_min: int

    # Back-compat / convenience: max of entry+exit pre_min.
    pre_min: int

    entry_artifact: str
    exit_artifact: str
    entry_created_utc: Optional[str]
    exit_created_utc: Optional[str]


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _iso_utc(ts_like: Any) -> str:
    # Convert anything pandas can parse into an ISO8601 UTC string.
    return pd.to_datetime(ts_like, utc=True).to_pydatetime().isoformat()


def _finite_or_none(x: Any) -> Optional[float]:
    try:
        fx = float(x)
    except Exception:
        return None
    return fx if np.isfinite(fx) else None


def _git_commit_hash(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        s = out.decode("utf-8", errors="replace").strip()
        return s if s else None
    except Exception:
        return None


def _audit_market_bar_1m_closed(audit: AuditLogger, *, symbol: str, source: str, bar_row: Mapping[str, Any]) -> None:
    t0 = pd.to_datetime(bar_row["timestamp"], utc=True)
    t1 = t0 + timedelta(minutes=1)
    audit.write(
        {
            "kind": KIND_MARKET_BAR_1M_CLOSED,
            "symbol": symbol,
            "source": source,
            "interval": "1m",
            "bar_open_time_utc": t0.to_pydatetime().isoformat(),
            "bar_close_time_utc": t1.to_pydatetime().isoformat(),
            "open": float(bar_row["open"]),
            "high": float(bar_row["high"]),
            "low": float(bar_row["low"]),
            "close": float(bar_row["close"]),
            "volume": float(bar_row["volume"]),
        }
    )


def _audit_entry_decision(
    audit: AuditLogger,
    *,
    symbol: str,
    bar_open_ts: Any,
    score: float,
    threshold: float,
    planned_entry_ts: Any,
    models: Models,
    feats_row: pd.Series,
    policy: str = "threshold_online_causal_next_open",
    forced: bool = False,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    t0 = pd.to_datetime(bar_open_ts, utc=True)
    t1 = t0 + timedelta(minutes=1)

    feature_names = list(models.entry_features)
    feature_values = [_finite_or_none(feats_row.get(c)) for c in feature_names]

    payload: Dict[str, Any] = {
        "kind": KIND_ENTRY_DECISION,
        "symbol": symbol,
        "decision_time_utc": t1.to_pydatetime().isoformat(),
        "bar_open_time_utc": t0.to_pydatetime().isoformat(),
        "bar_close_time_utc": t1.to_pydatetime().isoformat(),
        "score": float(score),
        "threshold": float(threshold),
        "action": "enter",
        "planned_entry_time_utc": _iso_utc(planned_entry_ts),
        "policy": str(policy),
        "forced": bool(forced),
        "feature_names": feature_names,
        "feature_values": feature_values,
        "model": {
            "role": "entry",
            "artifact": str(models.entry_artifact),
            "created_utc": str(models.entry_created_utc or "unknown"),
            "features": feature_names,
        },
    }
    if extra:
        payload.update(dict(extra))
    audit.write(payload)


def _audit_trade_closed(
    audit: AuditLogger,
    *,
    symbol: str,
    paper: bool,
    entry_time_utc: str,
    exit_time_utc: str,
    entry_price: float,
    exit_price: float,
    fee_side: float,
    exit_rel_min: int,
    realized_ret_pct: float,
    predicted_ret_pct: Optional[float],
    entry_score: Optional[float],
    entry_threshold: Optional[float],
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "kind": KIND_TRADE_CLOSED,
        "symbol": symbol,
        "paper": bool(paper),
        "entry_time_utc": entry_time_utc,
        "exit_time_utc": exit_time_utc,
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "fee_side": float(fee_side),
        "exit_rel_min": int(exit_rel_min),
        "realized_ret_pct": float(realized_ret_pct),
        "predicted_ret_pct": predicted_ret_pct,
        "entry_score": entry_score,
        "entry_threshold": entry_threshold,
    }
    if extra:
        payload.update(dict(extra))
    audit.write(payload)


_MEAN_SUFFIX_RE = re.compile(r"_mean_(\d+)m$")


def _parse_pre_min(payload: Any) -> int:
    if not isinstance(payload, dict):
        return 0
    ctx = payload.get("context")
    if not isinstance(ctx, dict):
        return 0
    pm = ctx.get("pre_min")
    if pm is None:
        return 0
    try:
        return max(0, int(pm))
    except Exception:
        return 0


def _union_feature_names(a: List[str], b: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for c in list(a) + list(b):
        s = str(c)
        if s in seen:
            continue
        out.append(s)
        seen.add(s)
    return out


def _infer_pre_mins_from_features(feat_names: List[str]) -> List[int]:
    mins = set()
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


def load_models(entry_path: Path, exit_path: Path) -> Models:
    ep = joblib.load(entry_path)
    xp = joblib.load(exit_path)

    entry_model = ep["model"] if isinstance(ep, dict) and "model" in ep else ep
    exit_model = xp["model"] if isinstance(xp, dict) and "model" in xp else xp

    entry_features = list(ep["features"]) if isinstance(ep, dict) and "features" in ep else list(FEATURES)
    exit_features = list(xp["features"]) if isinstance(xp, dict) and "features" in xp else list(FEATURES)

    entry_pre_min = _parse_pre_min(ep) if isinstance(ep, dict) else 5
    exit_pre_min = _parse_pre_min(xp) if isinstance(xp, dict) else int(entry_pre_min)
    pre_min = int(max(int(entry_pre_min), int(exit_pre_min), 0))

    entry_created_utc = ep.get("created_utc") if isinstance(ep, dict) else None
    exit_created_utc = xp.get("created_utc") if isinstance(xp, dict) else None

    return Models(
        entry_model=entry_model,
        exit_model=exit_model,
        entry_features=entry_features,
        exit_features=exit_features,
        entry_pre_min=int(entry_pre_min),
        exit_pre_min=int(exit_pre_min),
        pre_min=int(pre_min),
        entry_artifact=str(entry_path),
        exit_artifact=str(exit_path),
        entry_created_utc=str(entry_created_utc) if entry_created_utc else None,
        exit_created_utc=str(exit_created_utc) if exit_created_utc else None,
    )


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    # USDT-M futures 1m klines
    params = {"symbol": symbol, "interval": "1m", "startTime": start_ms, "endTime": end_ms, "limit": 1500}
    r = requests.get(f"{BINANCE_FAPI}/fapi/v1/klines", params=params, timeout=10)
    r.raise_for_status()
    arr = r.json()
    if not arr:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])  
    df = pd.DataFrame(arr, columns=[
        "open_time","open","high","low","close","volume","close_time","qav","trades","tbbav","tbqav","ignore"
    ])
    df = df[["open_time","open","high","low","close","volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df = df.rename(columns={"open_time":"_open_time"})
    df = df[["timestamp","open","high","low","close","volume"]]
    return df


def minute_bars_from_klines(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    # Pull in chunks (<= 1500 mins)
    out = []
    s = start
    while s < end:
        e = min(s + timedelta(minutes=1490), end)
        df = fetch_klines(symbol, int(s.timestamp() * 1000), int(e.timestamp() * 1000))
        if not df.empty:
            out.append(df)
        s = e
        time.sleep(0.05)  # polite
    if not out:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    bars = pd.concat(out, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return bars


def minute_bars_from_dir(bars_dir: Path, start: datetime, end: datetime) -> pd.DataFrame:
    """Load persisted minute_bars_YYYY-MM-DD.csv files from disk.

    This is used for reproducible/offline replay.
    """
    bars_dir = Path(bars_dir)
    if not bars_dir.exists():
        raise SystemExit(f"bars-dir does not exist: {bars_dir}")

    # end is exclusive
    end_excl = end
    last_day = (end_excl - timedelta(seconds=1)).date() if end_excl > start else start.date()

    frames: List[pd.DataFrame] = []
    d = start.date()
    while d <= last_day:
        p = bars_dir / f"minute_bars_{d}.csv"
        if not p.exists():
            raise SystemExit(f"Missing bars file for replay: {p}")
        df = pd.read_csv(p, parse_dates=["timestamp"])
        frames.append(df)
        d = d + timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    bars = pd.concat(frames, ignore_index=True)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    mask = (bars["timestamp"] >= pd.Timestamp(start)) & (bars["timestamp"] < pd.Timestamp(end_excl))
    bars = bars.loc[mask].reset_index(drop=True)

    return bars


def build_feature_frame(
    bars: pd.DataFrame,
    pre_mins: Any,
    feat_names: List[str],
    *,
    return_missing: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[str]]]:
    # Compute base features, then reconstruct context mean(s) to match training.

    if isinstance(pre_mins, (list, tuple, set)):
        mins_in = [int(x) for x in list(pre_mins)]
    else:
        mins_in = [int(pre_mins)]
    mins = sorted({int(x) for x in mins_in if int(x) > 0})

    base = compute_feature_frame(bars.rename(columns={"timestamp": "ts_min"}))
    src = base[[c for c in FEATURES if c in base.columns]]

    full = src
    for m in mins:
        ctx_mean = src.rolling(int(m), min_periods=int(m)).mean().add_suffix(f"_mean_{int(m)}m")
        full = pd.concat([full, ctx_mean], axis=1)

    missing = [c for c in feat_names if c not in full.columns]
    for c in missing:
        full[c] = np.nan

    out = pd.concat([base[["ts_min"]], full[feat_names]], axis=1).rename(columns={"ts_min": "timestamp"})
    if return_missing:
        return out, missing
    return out


def greedy_causal_exit(preds: List[float]) -> int:
    # Return k index (1-based) to exit
    best = -1e30
    best_k = None
    for k, yhat in enumerate(preds, start=1):
        if yhat >= best:
            best = yhat
            best_k = k
    return best_k if best_k is not None else 10


def simulate_replay(
    symbol: str,
    models: Models,
    start: datetime,
    end: datetime,
    thresholds_csv: Optional[Path],
    target_frac: float,
    max_entries_per_day: int,
    fee_side: float,
    out_dir: Path,
    *,
    bars_dir: Optional[Path] = None,
    audit: Optional[AuditLogger] = None,
) -> Dict[str, Path]:
    symbol_u = symbol.upper()

    if bars_dir is not None:
        bars = minute_bars_from_dir(Path(bars_dir), start, end)
    else:
        bars = minute_bars_from_klines(symbol_u, start, end)
    if bars.empty:
        raise SystemExit("No minute bars fetched for the requested period.")
    # Persist bars by date for audit
    bars["date"] = pd.to_datetime(bars["timestamp"]).dt.date
    for d, g in bars.groupby("date"):
        p = out_dir / f"minute_bars_{d}.csv"; p.parent.mkdir(parents=True, exist_ok=True); g.drop(columns=["date"]).to_csv(p, index=False)

    # Build features with parity (union of entry+exit model features)
    feat_union = _union_feature_names(models.entry_features, models.exit_features)
    req_pre_mins = set(_infer_pre_mins_from_features(feat_union))
    for pm in [int(models.entry_pre_min), int(models.exit_pre_min)]:
        if pm > 0:
            req_pre_mins.add(int(pm))
    feats = build_feature_frame(bars, sorted(req_pre_mins), feat_union)

    # Entry scoring
    entry_scores = pd.DataFrame({
        "timestamp": feats["timestamp"].to_numpy(),
        "score": models.entry_model.predict(feats[models.entry_features].to_numpy(dtype=np.float32))
    })
    entry_scores["date"] = pd.to_datetime(entry_scores["timestamp"]).dt.date

    # Thresholds
    thr_map: Dict[Any, float] = {}
    if thresholds_csv and thresholds_csv.exists():
        thr_df = pd.read_csv(thresholds_csv, parse_dates=["date"]) if "date" in pd.read_csv(thresholds_csv, nrows=0).columns else pd.read_csv(thresholds_csv)
        if "date" in thr_df.columns:
            thr_df["date"] = pd.to_datetime(thr_df["date"]).dt.date
            for _, r in thr_df.iterrows():
                if pd.notna(r.get("threshold")):
                    thr_map[r["date"]] = float(r["threshold"])        
    # Online thresholds from prior days if not provided
    days = sorted(entry_scores["date"].unique())
    prior_scores: List[float] = []
    for d in days:
        if d not in thr_map:
            if prior_scores:
                q = float(np.quantile(np.array(prior_scores, dtype=np.float64), 1.0 - float(target_frac)))
                thr_map[d] = q
            else:
                thr_map[d] = np.inf  # no trading on first day without prior
        # accumulate current day scores after the day ends
        prior_scores.extend(entry_scores.loc[entry_scores["date"]==d, "score"].tolist())

    # Iterate minutes, decide entries/exits
    entries_today = 0
    cur_day: Any = None
    open_trades: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []

    # For price lookup
    ts_to_row = {t: i for i, t in enumerate(bars["timestamp"]) }

    for i in range(len(bars)-11):  # ensure we can read next 10 minutes for exit cap
        t = pd.Timestamp(bars.iloc[i]["timestamp"]).to_pydatetime()
        d = t.date()
        if cur_day != d:
            cur_day = d
            entries_today = 0
        # Close-of-minute features at time t (computed above); entry fills at t+1 open
        # Entry decision
        thr = thr_map.get(d, np.inf)
        score_i = float(entry_scores.iloc[i]["score"]) if i < len(entry_scores) else -np.inf
        if score_i >= thr and entries_today < max_entries_per_day:
            # Place entry at next minute open
            j = i + 1
            if j < len(bars):
                if audit is not None:
                    # Decision computed on bar i (closed), entry assumed at bar j open.
                    _audit_market_bar_1m_closed(audit, symbol=symbol_u, source="binance_fapi_rest", bar_row=bars.iloc[i])
                    _audit_entry_decision(
                        audit,
                        symbol=symbol_u,
                        bar_open_ts=bars.iloc[i]["timestamp"],
                        score=float(score_i),
                        threshold=float(thr),
                        planned_entry_ts=bars.iloc[j]["timestamp"],
                        models=models,
                        feats_row=feats.iloc[i],
                    )

                e_open = float(bars.iloc[j]["open"])
                open_trades.append({
                    "entry_index": j,
                    "entry_time": bars.iloc[j]["timestamp"],
                    "entry_open": e_open,
                    "entry_score": float(score_i),
                    "entry_threshold": float(thr),
                    "running_best": -1e30,
                    "pred_path": [],
                })
                entries_today += 1
        # Progress open trades
        still_open = []
        for tr in open_trades:
            j0 = int(tr["entry_index"])  # index of entry open minute
            k = i - j0
            if k <= 0:
                still_open.append(tr)
                continue
            if k > 10:
                k = 10

            yhat = None
            if i < len(feats):
                try:
                    x = feats.iloc[i][models.exit_features].to_numpy(dtype=np.float32)[None, :]
                    yhat = float(models.exit_model.predict(x)[0])
                except Exception:
                    yhat = None

            updated = False
            if yhat is not None and np.isfinite(yhat):
                if yhat >= tr["running_best"]:
                    tr["running_best"] = yhat
                    tr["best_k"] = k
                    updated = True
                tr["pred_path"].append(yhat)
            else:
                tr["pred_path"].append(float("nan"))

            # Exit when a new best is observed (greedy) OR at cap 10
            should_exit = updated or (k >= 10)
            if should_exit:
                exit_k = int(tr.get("best_k") or 10)
                exit_idx = min(j0 + exit_k, len(bars) - 1)
                e_close = float(bars.iloc[exit_idx]["close"])
                fee = fee_side
                mult = (e_close * (1.0 - fee)) / (tr["entry_open"] * (1.0 + fee))
                realized = (mult - 1.0) * 100.0

                pred_raw = float(tr.get("running_best", float("nan")))
                pred = None
                if np.isfinite(pred_raw) and pred_raw > -1e29:
                    pred = float(pred_raw)

                exit_time = pd.to_datetime(bars.iloc[exit_idx]["timestamp"], utc=True) + timedelta(minutes=1)

                if audit is not None:
                    _audit_trade_closed(
                        audit,
                        symbol=symbol_u,
                        paper=True,
                        entry_time_utc=_iso_utc(tr["entry_time"]),
                        exit_time_utc=_iso_utc(exit_time),
                        entry_price=float(tr["entry_open"]),
                        exit_price=float(e_close),
                        fee_side=float(fee_side),
                        exit_rel_min=int(exit_k),
                        realized_ret_pct=float(realized),
                        predicted_ret_pct=pred,
                        entry_score=_finite_or_none(tr.get("entry_score")),
                        entry_threshold=_finite_or_none(tr.get("entry_threshold")),
                    )

                records.append({
                    "entry_time": pd.to_datetime(tr["entry_time"]).to_pydatetime(),
                    "exit_time": exit_time.to_pydatetime(),
                    "exit_rel_min": exit_k,
                    "predicted_ret_pct": pred,
                    "realized_ret_pct": realized,
                })
            else:
                still_open.append(tr)
        open_trades = still_open

    # Flush any remaining open trades at cap 10
    for tr in open_trades:
        j0 = int(tr["entry_index"])  
        exit_idx = min(j0 + 10, len(bars)-1)
        e_close = float(bars.iloc[exit_idx]["close"])  
        mult = (e_close * (1.0 - fee_side)) / (tr["entry_open"] * (1.0 + fee_side))
        realized = (mult - 1.0) * 100.0

        pred_raw = float(tr.get("running_best", float("nan")))
        pred = None
        if np.isfinite(pred_raw) and pred_raw > -1e29:
            pred = float(pred_raw)

        exit_time = pd.to_datetime(bars.iloc[exit_idx]["timestamp"], utc=True) + timedelta(minutes=1)

        if audit is not None:
            _audit_trade_closed(
                audit,
                symbol=symbol_u,
                paper=True,
                entry_time_utc=_iso_utc(tr["entry_time"]),
                exit_time_utc=_iso_utc(exit_time),
                entry_price=float(tr["entry_open"]),
                exit_price=float(e_close),
                fee_side=float(fee_side),
                exit_rel_min=10,
                realized_ret_pct=float(realized),
                predicted_ret_pct=pred,
                entry_score=_finite_or_none(tr.get("entry_score")),
                entry_threshold=_finite_or_none(tr.get("entry_threshold")),
            )

        records.append({
            "entry_time": pd.to_datetime(tr["entry_time"]).to_pydatetime(),
            "exit_time": exit_time.to_pydatetime(),
            "exit_rel_min": 10,
            "predicted_ret_pct": pred,
            "realized_ret_pct": realized,
        })

    if not records:
        trades = pd.DataFrame(
            columns=["entry_time", "exit_time", "exit_rel_min", "predicted_ret_pct", "realized_ret_pct", "date"]
        )
        daily = pd.DataFrame(
            columns=[
                "date",
                "n_trades",
                "mean_daily_pct",
                "sum_daily_pct",
                "median_daily_pct",
                "top_day_pct",
                "worst_day_pct",
            ]
        )
    else:
        trades = pd.DataFrame(records)
        trades.sort_values("entry_time", inplace=True)
        trades["date"] = pd.to_datetime(trades["entry_time"]).dt.date
        daily = trades.groupby("date", as_index=False).agg(
            n_trades=("realized_ret_pct", "size"),
            mean_daily_pct=("realized_ret_pct", "mean"),
            sum_daily_pct=("realized_ret_pct", "sum"),
            median_daily_pct=("realized_ret_pct", "median"),
            top_day_pct=("realized_ret_pct", "max"),
            worst_day_pct=("realized_ret_pct", "min"),
        )

    ts = _now_ts()
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / f"trades_{ts}.csv"
    daily_path = out_dir / f"daily_{ts}.csv"
    trades.to_csv(trades_path, index=False)
    daily.to_csv(daily_path, index=False)

    return {"trades": trades_path, "daily": daily_path}


# ---- LIVE MODE IMPLEMENTATION ----
class LiveRunner:
    def __init__(
        self,
        symbol: str,
        models: Models,
        thresholds_csv: Optional[Path],
        thresholds_live_out: Optional[Path],
        target_frac: float,
        max_entries_per_day: int,
        fee_side: float,
        out_dir: Path,
        *,
        trade_mode: str = "paper",  # paper | auto | real
        gate_window: int = 5,
        trade_margin_eur: float = 15.0,
        eurusdt: float = 0.0,
        binance_api_key: str = "",
        binance_api_secret: str = "",
        binance_base_url: str = BINANCE_FAPI,
        margin_type: str = "ISOLATED",
        allow_leverage_above_tier: bool = True,
        max_open_real: int = 1,
        gate_state_path: Optional[Path] = None,
        ledger_path: Optional[Path] = None,
        force_entry: bool = False,
        stop_after_minutes: int = 0,
        grace_sec: float = 1.5,
        backfill_hours: int = 48,
        stall_seconds: int = 120,
        alert_cooldown_seconds: int = 300,
        alerts_file: Optional[Path] = None,
        audit: Optional[AuditLogger] = None,
        metrics_port: int = 0,
        health_max_staleness_sec: float = 120.0,
        eod_send_email: bool = False,
        ws_base: str = "wss://fstream.binance.com/stream",
    ):
        self.symbol = symbol.upper()
        self.stream_name = f"{self.symbol.lower()}@kline_1m"
        self.models = models

        self.target_frac = float(target_frac)
        self.max_entries_per_day = int(max_entries_per_day)
        self.fee_side = float(fee_side)

        self.trade_mode_cfg = str(trade_mode).lower().strip()
        if self.trade_mode_cfg not in {"paper", "auto", "real"}:
            raise SystemExit(f"Invalid --trade-mode: {trade_mode!r} (expected paper|auto|real)")

        out_dir = Path(out_dir)
        self.gate_state_path = Path(gate_state_path) if gate_state_path else (out_dir / "gate_state.json")
        self.ledger_path = Path(ledger_path) if ledger_path else (out_dir / "strategy_ledger.json")

        self.gate: GateState = load_gate_state(self.gate_state_path, window=int(gate_window))
        self.ledger: StrategyLedger = load_strategy_ledger(self.ledger_path)

        self.trade_margin_eur = float(trade_margin_eur)
        self._eurusdt_override = float(eurusdt) if float(eurusdt) > 0.0 else None
        self._eurusdt_cached: Optional[Tuple[float, float]] = None  # (rate, wall_time)

        self.margin_type = str(margin_type).upper().strip() or "ISOLATED"
        self.allow_leverage_above_tier = bool(allow_leverage_above_tier)
        self.max_open_real = int(max_open_real)

        self.binance_base_url = str(binance_base_url).rstrip("/")
        self._binance_api_key = str(binance_api_key)
        self._binance_api_secret = str(binance_api_secret)

        # Sticky real enablement: either forced real, or auto + gate enabled (persisted).
        self.real_enabled = bool(self.trade_mode_cfg == "real" or (self.trade_mode_cfg == "auto" and self.gate.enabled))

        self._futures_client: Optional[BinanceUSDMFuturesClient] = None
        self._futures_exec: Optional[FuturesLongOnlyExecutor] = None
        self._position_side: Optional[str] = None
        self._real_position: Optional[Dict[str, Any]] = None

        # If stablecoin siphon fails, block further real entries (so we don't keep trading while deviating from strategy).
        self._block_real_entries: bool = False

        self.force_entry = bool(force_entry)
        self.stop_after_minutes = int(stop_after_minutes)
        self._force_entry_done = False
        self._start_wall: Optional[float] = None

        self.out_dir = out_dir
        self.grace_sec = float(grace_sec)
        self.backfill_hours = int(backfill_hours)
        self.ws_base = ws_base

        # Preflight credentials early in auto/real so we fail fast.
        if self.trade_mode_cfg in {"auto", "real"}:
            if not (self._binance_api_key and self._binance_api_secret):
                raise SystemExit(
                    "Binance API credentials required for --trade-mode auto/real. "
                    "Set BINANCE_API_KEY and BINANCE_API_SECRET env vars (recommended) or pass --binance-api-key/--binance-api-secret."
                )
            self._ensure_executor(preflight_only=True)

        self.thresholds_csv = thresholds_csv
        self.thresholds_live_out = thresholds_live_out

        self.stall_seconds = int(stall_seconds)
        self.alert_cooldown_seconds = int(alert_cooldown_seconds)
        self.alerts_file = alerts_file
        self.audit = audit

        self.metrics_port = int(metrics_port)
        self.health_max_staleness_sec = float(health_max_staleness_sec)
        self.eod_send_email = bool(eod_send_email)

        self.last_closed_wall = time.time()
        self._last_alert_by_kind: Dict[str, float] = {}

        self.thr_map: Dict[Any, float] = {}  # date -> threshold
        self.persisted_thr_dates: set = set()

        self.prior_scores: List[float] = []
        self._scores_cur_day: List[float] = []

        self.entries_today = 0
        self.cur_day = None

        self.open_trades: List[Dict[str, Any]] = []
        self.records: List[Dict[str, Any]] = []

        # rolling bars DataFrame
        self.bars = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        self.feats: Optional[pd.DataFrame] = None
        self._missing_cols: List[str] = []

        # Load thresholds from seed CSV and prior live persistence file (if any)
        self._load_thresholds_csv(thresholds_csv)
        self._load_thresholds_csv(thresholds_live_out, record_persisted=True)

    def _alert(self, kind: str, message: str, **fields: Any) -> None:
        now = time.time()
        last = float(self._last_alert_by_kind.get(kind, 0.0))
        if now - last < float(self.alert_cooldown_seconds):
            return
        self._last_alert_by_kind[kind] = now

        payload = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "message": message,
            **fields,
        }
        extra = f" {fields}" if fields else ""
        print(f"[ALERT] {payload['ts_utc']} {kind}: {message}{extra}")

        if self.alerts_file:
            self.alerts_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.alerts_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")

    def _stall_monitor(self) -> None:
        while True:
            time.sleep(1.0)
            dt = time.time() - float(self.last_closed_wall)
            if dt > float(self.stall_seconds):
                self._alert("feed_stall", "No closed 1m kline received recently", seconds_since_last=round(dt, 1))

    def _load_thresholds_csv(self, path: Optional[Path], *, record_persisted: bool = False) -> None:
        if not path or not path.exists():
            return
        try:
            df = pd.read_csv(path, parse_dates=["date"])
        except Exception:
            df = pd.read_csv(path)
        if "date" not in df.columns or "threshold" not in df.columns:
            return
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        for _, r in df.iterrows():
            d = r.get("date")
            thr = r.get("threshold")
            if pd.isna(d) or pd.isna(thr):
                continue
            try:
                self.thr_map[d] = float(thr)
                if record_persisted:
                    self.persisted_thr_dates.add(d)
            except Exception:
                continue

    def _append_threshold(self, d, threshold: float) -> None:
        if self.thresholds_live_out is None:
            return
        if d in self.persisted_thr_dates:
            return
        if not np.isfinite(threshold):
            self._alert("threshold_not_finite", "Threshold is not finite; not persisting", date=str(d), threshold=str(threshold))
            return
        path = self.thresholds_live_out
        path.parent.mkdir(parents=True, exist_ok=True)
        header = not path.exists()
        row = {
            "date": str(d),
            "threshold": float(threshold),
            "n_prior_scores": int(len(self.prior_scores)),
            "target_frac": float(self.target_frac),
            "created_utc": _now_ts(),
        }
        pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)
        self.persisted_thr_dates.add(d)

    def _persist_bar(self, row: pd.Series) -> None:
        d = pd.to_datetime(row["timestamp"]).date()
        p = self.out_dir / f"minute_bars_{d}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        header = not p.exists()
        pd.DataFrame([row]).to_csv(p, index=False, header=header, mode='a')

    def _ensure_feats(self) -> None:
        # Build/extend feature frame with parity using the full bars frame
        out, missing = build_feature_frame(
            self.bars,
            sorted({
                *set(_infer_pre_mins_from_features(_union_feature_names(self.models.entry_features, self.models.exit_features))),
                *{int(self.models.entry_pre_min), int(self.models.exit_pre_min)},
            }),
            _union_feature_names(self.models.entry_features, self.models.exit_features),
            return_missing=True,
        )
        self.feats = out
        self._missing_cols = missing

    def _entry_threshold_for_day(self, d):
        if d in self.thr_map:
            return self.thr_map[d]
        if self.prior_scores:
            q = float(np.quantile(np.array(self.prior_scores, dtype=np.float64), 1.0 - self.target_frac))
            self.thr_map[d] = q
            return q
        return np.inf

    def _on_day_roll(self, finished_day) -> None:
        # Move finished day scores into prior_scores (to keep thresholds causal)
        if self._scores_cur_day:
            self.prior_scores.extend(self._scores_cur_day)
        self._scores_cur_day = []
        self.entries_today = 0

        # Persist a snapshot and kick off an EOD report.
        try:
            self._flush_outputs()
        except Exception as e:
            self._alert("flush_outputs_error", "Failed to flush live outputs on day roll", error=str(e))

        # Run EOD report generation (non-blocking).
        self._spawn_eod_report(finished_day)

        # Evaluate auto go-live gate at EOD/day-roll.
        self._maybe_enable_real_after_eod(finished_day)

    def _score_entry_index(self, i: int) -> float:
        # Score at feats row i (exact feature order from payload)
        x = self.feats.loc[i, self.models.entry_features].to_numpy(dtype=np.float32)[None, :]
        return float(self.models.entry_model.predict(x)[0])

    def _predict_exit_index(self, i: int) -> float:
        x = self.feats.loc[i, self.models.exit_features].to_numpy(dtype=np.float32)[None, :]
        return float(self.models.exit_model.predict(x)[0])

    def _effective_trade_mode(self) -> str:
        # Effective mode used for *new* trades.
        if self.trade_mode_cfg == "paper":
            return "paper"
        if self.trade_mode_cfg == "real":
            return "real"
        # auto
        return "real" if self.real_enabled else "paper"

    def _eurusdt_rate(self) -> float:
        if self._eurusdt_override is not None:
            return float(self._eurusdt_override)

        # Cache for 10 minutes.
        if self._eurusdt_cached is not None:
            rate, wall = self._eurusdt_cached
            if (time.time() - float(wall)) < 600.0:
                return float(rate)

        rate = float(fetch_spot_price("EURUSDT"))
        self._eurusdt_cached = (float(rate), time.time())
        return float(rate)

    def _margin_eur_for_next_trade(self) -> float:
        # Clamp to available trading capital (banked portion is not deployable by design).
        cap = float(self.ledger.trading_capital_eur)
        return max(0.0, min(float(self.trade_margin_eur), cap))

    def _target_leverage_for_next_trade(self) -> int:
        eq = float(self.ledger.equity_eur())
        return int(StrategyLedger.target_leverage_from_equity(eq))

    def _ensure_executor(self, *, preflight_only: bool) -> None:
        if self._futures_exec is not None and self._futures_client is not None:
            return

        self._futures_client = BinanceUSDMFuturesClient(
            api_key=self._binance_api_key,
            api_secret=self._binance_api_secret,
            base_url=self.binance_base_url,
        )

        # Public exchange filters.
        filters = fetch_futures_exchange_info(self.symbol, base_url=self.binance_base_url)
        self._futures_exec = FuturesLongOnlyExecutor(
            client=self._futures_client,
            filters=filters,
            margin_type=self.margin_type,
        )

        # Detect position mode (one-way vs hedge).
        try:
            ps = self._futures_client.get_position_side_dual()
            dual = ps.get("dualSidePosition")
            dual_b = str(dual).lower() == "true" if isinstance(dual, str) else bool(dual)
            self._position_side = "LONG" if dual_b else None
        except Exception as e:
            raise SystemExit(f"Failed to query Binance position mode: {e}") from e

        # Read-only sanity calls to verify creds work.
        if not preflight_only:
            return
        _ = self._futures_client.get_account()
        _ = self._futures_client.get_leverage_brackets(self.symbol)

    def _audit_trade_mode_switch(self, *, from_mode: str, to_mode: str, reason: str, extra: Optional[Mapping[str, Any]] = None) -> None:
        if self.audit is None:
            return
        payload: Dict[str, Any] = {
            "kind": KIND_TRADE_MODE_SWITCH,
            "symbol": self.symbol,
            "from_trade_mode": str(from_mode),
            "to_trade_mode": str(to_mode),
            "reason": str(reason),
        }
        if extra:
            payload.update(dict(extra))
        self.audit.write(payload)

    def _spawn_eod_report(self, finished_day) -> None:
        # Keep this non-blocking: day-roll happens inside the WS callback.
        try:
            import threading

            t = threading.Thread(target=self._run_eod_report, args=(finished_day,), daemon=True)
            t.start()
        except Exception as e:
            self._alert("eod_report_thread_error", "Failed to spawn EOD report thread", error=str(e))

    def _run_eod_report(self, finished_day) -> None:
        # Generates HTML+CSV report under <out_dir>/reports.
        try:
            ts = _now_ts()
            reports_dir = self.out_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            day_s = str(finished_day)
            html_out = reports_dir / f"eod_report_{day_s}_{ts}.html"
            csv_out = reports_dir / f"eod_summary_{day_s}_{ts}.csv"

            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "eod_report_2025-12-22T15-48-42Z.py"),
                "--out-dir",
                str(self.out_dir),
                "--html-out",
                str(html_out),
                "--csv-out",
                str(csv_out),
            ]
            if bool(self.eod_send_email):
                cmd.append("--send-email")

            res = subprocess.run(cmd, check=False)
            if int(res.returncode) != 0:
                self._alert(
                    "eod_report_nonzero_exit",
                    "EOD report process exited non-zero",
                    finished_day=str(finished_day),
                    returncode=int(res.returncode),
                )
        except Exception as e:
            self._alert("eod_report_error", "EOD report generation failed", error=str(e))

    def _serve_http(self) -> None:
        # Minimal HTTP server for /health and /metrics.
        try:
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
            from urllib.parse import urlparse

            runner = self
            host = "0.0.0.0"
            port = int(getattr(self, "metrics_port", 0) or 0)
            if port <= 0:
                return

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    p = urlparse(self.path).path
                    now = time.time()
                    age = float(now - float(getattr(runner, "last_closed_wall", 0.0)))

                    if p == "/health":
                        max_age = float(getattr(runner, "health_max_staleness_sec", 120.0))
                        ok = bool(age <= max_age)
                        status = 200 if ok else 503
                        payload = {
                            "ok": ok,
                            "last_closed_bar_age_seconds": round(age, 3),
                            "max_staleness_seconds": max_age,
                            "trade_mode_cfg": str(getattr(runner, "trade_mode_cfg", "")),
                            "trade_mode_effective": str(runner._effective_trade_mode()),
                            "gate_enabled": bool(getattr(getattr(runner, "gate", None), "enabled", False)),
                            "paper_trades_total": int(getattr(getattr(runner, "gate", None), "paper_trades_total", 0)),
                        }
                        body = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
                        self.send_response(status)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return

                    if p == "/metrics":
                        # Prometheus text format.
                        gate = getattr(runner, "gate", None)
                        gate_enabled = 1 if bool(getattr(gate, "enabled", False)) else 0
                        paper_trades_total = int(getattr(gate, "paper_trades_total", 0))

                        open_trades_total = int(len(getattr(runner, "open_trades", []) or []))
                        records_total = int(len(getattr(runner, "records", []) or []))

                        real_trades_total = 0
                        try:
                            real_trades_total = int(sum(1 for r in (runner.records or []) if not bool(r.get("paper", True))))
                        except Exception:
                            real_trades_total = 0

                        lines = [
                            "# HELP autotrader_last_closed_bar_age_seconds Seconds since last closed 1m bar was received",
                            "# TYPE autotrader_last_closed_bar_age_seconds gauge",
                            f"autotrader_last_closed_bar_age_seconds {age}",
                            "# HELP autotrader_open_trades_total Number of open trades tracked by the runner",
                            "# TYPE autotrader_open_trades_total gauge",
                            f"autotrader_open_trades_total {open_trades_total}",
                            "# HELP autotrader_records_total Number of closed trade records buffered in memory",
                            "# TYPE autotrader_records_total gauge",
                            f"autotrader_records_total {records_total}",
                            "# HELP autotrader_real_trades_total Number of closed real trades recorded",
                            "# TYPE autotrader_real_trades_total gauge",
                            f"autotrader_real_trades_total {real_trades_total}",
                            "# HELP autotrader_gate_enabled Whether the auto go-live gate has enabled real trading",
                            "# TYPE autotrader_gate_enabled gauge",
                            f"autotrader_gate_enabled {gate_enabled}",
                            "# HELP autotrader_paper_trades_total Total completed paper trades (persisted)",
                            "# TYPE autotrader_paper_trades_total counter",
                            f"autotrader_paper_trades_total {paper_trades_total}",
                        ]
                        body = ("\n".join(lines) + "\n").encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "text/plain; version=0.0.4")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return

                    self.send_response(404)
                    self.end_headers()

                def log_message(self, format, *args):
                    # quiet
                    return

            httpd = ThreadingHTTPServer((host, port), Handler)
            httpd.serve_forever()
        except Exception as e:
            self._alert("http_server_error", "HTTP server crashed", error=str(e))

    def _maybe_enable_real_after_eod(self, finished_day) -> None:
        # Auto go-live after EOD once we have >= N completed paper trades.
        if self.trade_mode_cfg != "auto":
            return
        if self.real_enabled:
            return
        if not self.gate.should_enable_real():
            return

        self.gate.enable_real()
        save_gate_state(self.gate, self.gate_state_path)
        self.real_enabled = True

        self._alert(
            "trade_mode_switch",
            "EOD gate met; switching to real trading (sticky)",
            finished_day=str(finished_day),
            window=int(self.gate.window),
            paper_trades_total=int(getattr(self.gate, "paper_trades_total", 0)),
        )

        self._audit_trade_mode_switch(
            from_mode="auto",
            to_mode="real",
            reason=f"EOD gate met at day-roll {finished_day}: paper_trades_total >= window",
            extra={
                "gate": {
                    "window": int(self.gate.window),
                    "paper_trades_total": int(getattr(self.gate, "paper_trades_total", 0)),
                    "outcomes": list(self.gate.outcomes),
                    "trade_entry_times_utc": list(self.gate.trade_entry_times_utc),
                    "enabled_utc": self.gate.enabled_utc,
                }
            },
        )

    def _open_real_long(self, *, notional_usdt: float, client_order_id: str) -> Dict[str, Any]:
        self._ensure_executor(preflight_only=False)
        assert self._futures_client is not None
        assert self._futures_exec is not None

        px = float(self._futures_client.mark_price(self.symbol))
        qty = float(self._futures_exec.qty_for_notional(notional_usdt=float(notional_usdt), price_usdt=px))

        res = self._futures_exec.open_long_market(
            symbol=self.symbol,
            qty=float(qty),
            client_order_id=str(client_order_id),
            position_side=self._position_side,
        )

        entry_price = float(res.avg_price) if float(res.avg_price) > 0 else float(self._futures_client.ticker_price(self.symbol))
        entry_qty = float(res.executed_qty) if float(res.executed_qty) > 0 else float(qty)

        return {
            "entry_order_id": res.order_id,
            "entry_client_order_id": res.client_order_id,
            "qty": float(entry_qty),
            "entry_price": float(entry_price),
        }

    def _close_real_long(self, *, qty: float, client_order_id: str) -> Dict[str, Any]:
        self._ensure_executor(preflight_only=False)
        assert self._futures_client is not None
        assert self._futures_exec is not None

        res = self._futures_exec.close_long_market(
            symbol=self.symbol,
            qty=float(qty),
            client_order_id=str(client_order_id),
            position_side=self._position_side,
        )

        exit_price = float(res.avg_price) if float(res.avg_price) > 0 else float(self._futures_client.ticker_price(self.symbol))
        exit_qty = float(res.executed_qty) if float(res.executed_qty) > 0 else float(qty)

        return {
            "exit_order_id": res.order_id,
            "exit_client_order_id": res.client_order_id,
            "qty": float(exit_qty),
            "exit_price": float(exit_price),
        }

    def run(self) -> None:
        try:
            import websocket  # websocket-client
        except Exception as e:
            raise SystemExit("websocket-client is required: pip install websocket-client") from e
        import threading

        self.last_closed_wall = time.time()
        self._start_wall = time.time()
        threading.Thread(target=self._stall_monitor, daemon=True).start()

        if int(getattr(self, "metrics_port", 0)) > 0:
            threading.Thread(target=self._serve_http, daemon=True).start()
            self._alert("http_server", "HTTP server started", port=int(self.metrics_port))

        # Warm backfill for feature windows
        end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        start = end - timedelta(hours=self.backfill_hours)
        self.bars = minute_bars_from_klines(self.symbol, start, end)
        if self.bars.empty:
            raise SystemExit("Backfill failed: no klines fetched")
        # persist
        for d, g in self.bars.assign(date=lambda x: pd.to_datetime(x['timestamp']).dt.date).groupby('date'):
            p = self.out_dir / f"minute_bars_{d}.csv"; p.parent.mkdir(parents=True, exist_ok=True); g.drop(columns=['date']).to_csv(p, index=False)
        self._ensure_feats()
        # Seed prior scores with all fully-completed past days from backfill
        df_scores = pd.DataFrame({
            'timestamp': self.feats['timestamp'],
            'score': self.models.entry_model.predict(self.feats[self.models.entry_features].to_numpy(dtype=np.float32))
        })
        df_scores['date'] = pd.to_datetime(df_scores['timestamp']).dt.date
        today = datetime.now(timezone.utc).date()
        if len(df_scores) > 0:
            prior = df_scores[df_scores["date"] < today]["score"].tolist()
            self.prior_scores.extend(prior)
            self._scores_cur_day = df_scores[df_scores["date"] == today]["score"].tolist()
        self.cur_day = today

        thr_today = float(self._entry_threshold_for_day(today))
        self._append_threshold(today, thr_today)

        # WS setup
        url = f"{self.ws_base}?streams={self.stream_name}"

        def on_message(ws, message):
            try:
                obj = json.loads(message)
                k = obj.get("data", {}).get("k", {})
                if not k or not bool(k.get("x")):
                    return

                # Feed heartbeat for stall detection
                self.last_closed_wall = time.time()

                # Build bar row from closed kline
                ts = pd.to_datetime(k["t"], unit="ms", utc=True)
                row = {
                    "timestamp": ts,
                    "open": float(k["o"]),
                    "high": float(k["h"]),
                    "low": float(k["l"]),
                    "close": float(k["c"]),
                    "volume": float(k["v"]),
                }

                # Grace delay to avoid edge-of-minute partials
                time.sleep(self.grace_sec)

                # Enforce monotonic minute sequence
                persist_bar = True
                if len(self.bars) > 0:
                    prev_ts = pd.to_datetime(self.bars.iloc[-1]["timestamp"])
                    if ts < prev_ts:
                        self._alert(
                            "kline_out_of_order",
                            "Received out-of-order closed kline; ignoring",
                            ts=str(ts),
                            prev=str(prev_ts),
                        )
                        return
                    if ts == prev_ts:
                        self._alert("kline_duplicate", "Duplicate closed kline; replacing last", ts=str(ts))
                        self.bars.iloc[-1] = pd.Series(row)
                        persist_bar = False
                    else:
                        gap_sec = (ts - prev_ts).total_seconds()
                        if gap_sec > 90:
                            self._alert(
                                "kline_gap",
                                "Gap between closed klines",
                                gap_seconds=round(gap_sec, 1),
                                prev=str(prev_ts),
                                ts=str(ts),
                            )
                        self.bars = pd.concat([self.bars, pd.DataFrame([row])], ignore_index=True)
                else:
                    self.bars = pd.concat([self.bars, pd.DataFrame([row])], ignore_index=True)

                if persist_bar:
                    self._persist_bar(pd.Series(row))

                self._ensure_feats()
                if self._missing_cols:
                    self._alert(
                        "feature_missing_cols",
                        "Missing required feature columns; skipping model decisions",
                        missing=self._missing_cols,
                    )

                i = len(self.bars) - 1  # index of just-closed bar
                d = ts.date()

                # Day roll handling: persist next-day threshold at roll
                if self.cur_day is None:
                    self.cur_day = d
                    thr0 = float(self._entry_threshold_for_day(d))
                    self._append_threshold(d, thr0)

                if d != self.cur_day:
                    self._on_day_roll(self.cur_day)
                    self.cur_day = d
                    self.entries_today = 0
                    thr_new = float(self._entry_threshold_for_day(d))
                    self._append_threshold(d, thr_new)

                # Entry decision for bar i (fill at i+1 open when it arrives)
                thr = float(self._entry_threshold_for_day(d))
                score_i = None
                if not self._missing_cols:
                    try:
                        score_i = float(self._score_entry_index(i))
                    except Exception as e:
                        self._alert("entry_predict_error", "Entry model predict failed", error=str(e))
                if score_i is None or not np.isfinite(score_i):
                    if score_i is not None:
                        self._alert("entry_pred_nan", "Entry prediction not finite", score=str(score_i), ts=str(ts))
                    score_i = -np.inf
                else:
                    # record for tomorrow's threshold (end-of-day roll)
                    self._scores_cur_day.append(score_i)

                thr_finite = bool(np.isfinite(thr))
                thr_audit = float(thr) if thr_finite else float(1e30)
                if (not thr_finite) and self.force_entry and (not self._force_entry_done):
                    self._alert(
                        "threshold_not_finite",
                        "Entry threshold is not finite; using sentinel for forced entry audit",
                        threshold=str(thr),
                    )

                trade_mode_eff = self._effective_trade_mode()

                score_finite = bool(np.isfinite(score_i))
                should_force = (
                    bool(self.force_entry)
                    and trade_mode_eff == "paper"
                    and (not bool(self._force_entry_done))
                    and (not self._missing_cols)
                    and score_finite
                    and (self.entries_today < self.max_entries_per_day)
                )
                should_enter = (not self._missing_cols) and score_finite and (self.entries_today < self.max_entries_per_day) and (
                    (score_i >= thr) or should_force
                )
                forced = bool(should_force and (score_i < thr))

                if should_enter:
                    planned_entry_ts = pd.to_datetime(self.bars.iloc[i]["timestamp"], utc=True) + timedelta(minutes=1)

                    if self.audit is not None and self.feats is not None:
                        _audit_market_bar_1m_closed(
                            self.audit,
                            symbol=self.symbol,
                            source="binance_fapi_ws",
                            bar_row=self.bars.iloc[i],
                        )
                        _audit_entry_decision(
                            self.audit,
                            symbol=self.symbol,
                            bar_open_ts=self.bars.iloc[i]["timestamp"],
                            score=float(score_i),
                            threshold=float(thr_audit),
                            planned_entry_ts=planned_entry_ts,
                            models=self.models,
                            feats_row=self.feats.iloc[i],
                            policy=("forced_next_open" if forced else "threshold_online_causal_next_open"),
                            forced=forced,
                            extra={
                                "trade_mode_cfg": str(self.trade_mode_cfg),
                                "trade_mode_effective": str(trade_mode_eff),
                                "real_enabled": bool(self.real_enabled),
                            },
                        )

                    if trade_mode_eff == "real":
                        if self._block_real_entries:
                            self._alert(
                                "real_entry_blocked",
                                "Real entries blocked due to stablecoin siphon failure",
                            )
                        else:
                            open_real = int(
                                sum(
                                    1
                                    for x in self.open_trades
                                    if (not x.get("pending_fill")) and x.get("trade_mode") == "real"
                                )
                            )
                            if open_real >= int(self.max_open_real) or self._real_position is not None:
                                self._alert(
                                    "real_entry_skipped_open_position",
                                    "Skipping real entry: max open real positions reached",
                                    open_real=int(open_real),
                                    max_open_real=int(self.max_open_real),
                                )
                            else:
                                margin_eur = float(self._margin_eur_for_next_trade())
                                if margin_eur <= 0.0:
                                    self._alert(
                                        "real_entry_skipped_no_capital",
                                        "Skipping real entry: no trading capital available",
                                        trading_capital_eur=float(self.ledger.trading_capital_eur),
                                        trade_margin_eur=float(self.trade_margin_eur),
                                    )
                                else:
                                    eurusdt = float(self._eurusdt_rate())
                                    target_lev = int(self._target_leverage_for_next_trade())
                                    notional_est = float(margin_eur) * float(target_lev) * float(eurusdt)

                                    try:
                                        self._ensure_executor(preflight_only=False)
                                        assert self._futures_exec is not None
                                        lev_set = int(
                                            self._futures_exec.ensure_margin_and_leverage(
                                                symbol=self.symbol,
                                                target_leverage=int(target_lev),
                                                notional_estimate=float(notional_est),
                                                allow_above_target=bool(self.allow_leverage_above_tier),
                                                margin_usdt=float(margin_eur) * float(eurusdt),
                                            )
                                        )
                                        notional_usdt = float(margin_eur) * float(lev_set) * float(eurusdt)

                                        coid = f"pt_entry_{int(time.time())}_{int(i)}"
                                        entry_meta = self._open_real_long(notional_usdt=float(notional_usdt), client_order_id=coid)

                                        self._real_position = {
                                            "qty": float(entry_meta["qty"]),
                                            "entry_price": float(entry_meta["entry_price"]),
                                            "entry_order_id": entry_meta.get("entry_order_id"),
                                            "entry_client_order_id": entry_meta.get("entry_client_order_id"),
                                            "leverage": int(lev_set),
                                            "margin_eur": float(margin_eur),
                                            "eurusdt": float(eurusdt),
                                            "notional_usdt": float(notional_usdt),
                                            "planned_entry_time_utc": _iso_utc(planned_entry_ts),
                                        }

                                        self.open_trades.append(
                                            {
                                                "pending_fill": False,
                                                "trade_mode": "real",
                                                "entry_index": i + 1,
                                                "entry_time": planned_entry_ts,
                                                "entry_open": float(entry_meta["entry_price"]),
                                                "qty": float(entry_meta["qty"]),
                                                "entry_order_id": entry_meta.get("entry_order_id"),
                                                "entry_client_order_id": entry_meta.get("entry_client_order_id"),
                                                "leverage": int(lev_set),
                                                "margin_eur": float(margin_eur),
                                                "eurusdt": float(eurusdt),
                                                "notional_usdt": float(notional_usdt),
                                                "entry_score": float(score_i),
                                                "entry_threshold": float(thr_audit),
                                                "running_best": -1e30,
                                                "best_k": None,
                                            }
                                        )
                                        self.entries_today += 1
                                    except Exception as e:
                                        self._alert("real_entry_error", "Real entry failed", error=str(e))
                    else:
                        self.open_trades.append(
                            {
                                "pending_fill": True,
                                "trade_mode": "paper",
                                "entry_index": i + 1,
                                "entry_time": None,
                                "entry_open": None,
                                "entry_score": float(score_i),
                                "entry_threshold": float(thr_audit),
                                "running_best": -1e30,
                                "best_k": None,
                            }
                        )
                        self.entries_today += 1
                        if self.force_entry and (not self._force_entry_done):
                            self._force_entry_done = True

                # Fill pending entries whose entry_index == i (we now know open[i])
                still = []
                for tr in self.open_trades:
                    if tr.get("pending_fill") and tr.get("entry_index") == i:
                        tr["pending_fill"] = False
                        tr["entry_time"] = self.bars.iloc[i]["timestamp"]
                        tr["entry_open"] = float(self.bars.iloc[i]["open"])
                    still.append(tr)
                self.open_trades = still

                # Progress open trades (greedy_causal exit; evaluate at minutes 1..10 after entry)
                still2 = []
                for tr in self.open_trades:
                    if tr.get("pending_fill"):
                        still2.append(tr)
                        continue

                    j0 = int(tr["entry_index"])
                    k_rel = i - j0
                    if k_rel <= 0:
                        still2.append(tr)
                        continue
                    if k_rel > 10:
                        k_rel = 10

                    updated = False
                    yhat = None
                    if not self._missing_cols:
                        try:
                            yhat = float(self._predict_exit_index(i))
                        except Exception as e:
                            self._alert("exit_predict_error", "Exit model predict failed", error=str(e))

                    if yhat is not None and np.isfinite(yhat):
                        if yhat >= float(tr.get("running_best", -1e30)):
                            tr["running_best"] = yhat
                            tr["best_k"] = k_rel
                            updated = True
                    elif yhat is not None:
                        self._alert("exit_pred_nan", "Exit prediction not finite", yhat=str(yhat), ts=str(ts))

                    if updated or k_rel >= 10:
                        exit_k = int(tr.get("best_k") or 10)
                        exit_idx = min(j0 + exit_k, len(self.bars) - 1)

                        pred_raw = float(tr.get("running_best", float("nan")))
                        pred = None
                        if np.isfinite(pred_raw) and pred_raw > -1e29:
                            pred = float(pred_raw)

                        exit_time = pd.to_datetime(self.bars.iloc[exit_idx]["timestamp"], utc=True) + timedelta(minutes=1)

                        trade_mode = str(tr.get("trade_mode") or "paper")
                        if trade_mode == "real":
                            try:
                                qty = float(tr.get("qty") or 0.0)
                                if qty <= 0.0 and self._real_position:
                                    qty = float(self._real_position.get("qty") or 0.0)
                                if qty <= 0.0:
                                    raise RuntimeError("Real position qty is missing")

                                coid = f"pt_exit_{int(time.time())}_{int(i)}"
                                close_meta = self._close_real_long(qty=float(qty), client_order_id=coid)
                                exit_px = float(close_meta["exit_price"])

                                entry_px = float(tr.get("entry_open") or 0.0)
                                if entry_px <= 0.0 and self._real_position:
                                    entry_px = float(self._real_position.get("entry_price") or 0.0)
                                if entry_px <= 0.0:
                                    raise RuntimeError("Real position entry price is missing")

                                mult = (exit_px * (1.0 - self.fee_side)) / (entry_px * (1.0 + self.fee_side))
                                realized = (mult - 1.0) * 100.0

                                lev = int(tr.get("leverage") or (self._real_position or {}).get("leverage") or 1)
                                margin_eur = float(tr.get("margin_eur") or (self._real_position or {}).get("margin_eur") or self.trade_margin_eur)
                                eurusdt = float(tr.get("eurusdt") or (self._real_position or {}).get("eurusdt") or self._eurusdt_rate())
                                notional_usdt = float(tr.get("notional_usdt") or (self._real_position or {}).get("notional_usdt") or (margin_eur * lev * eurusdt))

                                # Siphon 50% of profitable trades (when leverage > 10x) to stablecoin (USDT) in spot wallet.
                                pnl_eur = float(
                                    StrategyLedger.compute_pnl_eur(
                                        realized_ret_pct=float(realized),
                                        leverage=int(lev),
                                        margin_eur=float(margin_eur),
                                    )
                                )
                                bank_target_eur = float(
                                    StrategyLedger.default_bank_delta_eur(pnl_eur=float(pnl_eur), leverage=int(lev))
                                )

                                siphon: Dict[str, Any] = {"attempted": False}
                                bank_override_eur: Optional[float] = None

                                if bank_target_eur > 0.0 and float(eurusdt) > 0.0:
                                    req_usdt = float(bank_target_eur) * float(eurusdt)
                                    amt_usdt = float(req_usdt)
                                    try:
                                        assert self._futures_client is not None
                                        avail = self._futures_client.futures_available_balance("USDT")
                                        if avail is not None:
                                            amt_usdt = min(float(amt_usdt), float(avail))

                                        if amt_usdt > 0.0:
                                            tx = self._futures_client.futures_transfer(
                                                asset="USDT",
                                                amount=float(amt_usdt),
                                                transfer_type=2,  # futures -> spot
                                            )
                                            tran_id = tx.get("tranId") if isinstance(tx, dict) else None
                                            siphon = {
                                                "attempted": True,
                                                "asset": "USDT",
                                                "requested_usdt": float(req_usdt),
                                                "transferred_usdt": float(amt_usdt),
                                                "tran_id": tran_id,
                                            }
                                            bank_override_eur = float(amt_usdt) / float(eurusdt)
                                        else:
                                            siphon = {
                                                "attempted": True,
                                                "asset": "USDT",
                                                "requested_usdt": float(req_usdt),
                                                "transferred_usdt": 0.0,
                                                "reason": "insufficient_available_balance",
                                            }
                                            bank_override_eur = 0.0
                                    except Exception as e:
                                        self._alert(
                                            "stablecoin_siphon_failed",
                                            "Stablecoin siphon failed; blocking further real entries",
                                            error=str(e),
                                        )
                                        self._block_real_entries = True
                                        siphon = {
                                            "attempted": True,
                                            "asset": "USDT",
                                            "requested_usdt": float(req_usdt),
                                            "transferred_usdt": 0.0,
                                            "error": str(e),
                                        }
                                        bank_override_eur = 0.0

                                deltas = self.ledger.apply_trade_closed(
                                    realized_ret_pct=float(realized),
                                    leverage=int(lev),
                                    margin_eur=float(margin_eur),
                                    bank_override_eur=bank_override_eur,
                                )
                                save_strategy_ledger(self.ledger, self.ledger_path)

                                extra = {
                                    "trade_mode": "real",
                                    "qty": float(qty),
                                    "leverage": int(lev),
                                    "margin_eur": float(margin_eur),
                                    "eurusdt": float(eurusdt),
                                    "notional_usdt": float(notional_usdt),
                                    "stablecoin_siphon": siphon,
                                    "orders": {
                                        "entry_order_id": tr.get("entry_order_id"),
                                        "entry_client_order_id": tr.get("entry_client_order_id"),
                                        "exit_order_id": close_meta.get("exit_order_id"),
                                        "exit_client_order_id": close_meta.get("exit_client_order_id"),
                                    },
                                    "ledger": {
                                        "pnl_eur": deltas.get("pnl_eur"),
                                        "bank_delta_eur": deltas.get("bank_delta_eur"),
                                        "refinance_delta_eur": deltas.get("refinance_delta_eur"),
                                        "capital_post_eur": deltas.get("capital_post_eur"),
                                        "bank_post_eur": deltas.get("bank_post_eur"),
                                    },
                                }

                                if self.audit is not None:
                                    _audit_trade_closed(
                                        self.audit,
                                        symbol=self.symbol,
                                        paper=False,
                                        entry_time_utc=_iso_utc(tr["entry_time"]),
                                        exit_time_utc=_iso_utc(exit_time),
                                        entry_price=float(entry_px),
                                        exit_price=float(exit_px),
                                        fee_side=float(self.fee_side),
                                        exit_rel_min=int(exit_k),
                                        realized_ret_pct=float(realized),
                                        predicted_ret_pct=pred,
                                        entry_score=_finite_or_none(tr.get("entry_score")),
                                        entry_threshold=_finite_or_none(tr.get("entry_threshold")),
                                        extra=extra,
                                    )

                                self.records.append(
                                    {
                                        "paper": False,
                                        "trade_mode": "real",
                                        "entry_time": pd.to_datetime(tr["entry_time"]).to_pydatetime(),
                                        "exit_time": exit_time.to_pydatetime(),
                                        "exit_rel_min": exit_k,
                                        "predicted_ret_pct": pred,
                                        "realized_ret_pct": realized,
                                        "leverage": int(lev),
                                        "margin_eur": float(margin_eur),
                                        "siphon_requested_usdt": float((siphon or {}).get("requested_usdt") or 0.0),
                                        "siphon_transferred_usdt": float((siphon or {}).get("transferred_usdt") or 0.0),
                                        "siphon_tran_id": (siphon or {}).get("tran_id"),
                                        "siphon_error": (siphon or {}).get("error"),
                                        **{f"ledger_{k}": v for k, v in deltas.items()},
                                    }
                                )

                                # Clear tracked position.
                                self._real_position = None
                            except Exception as e:
                                self._alert("real_exit_error", "Real exit failed; keeping position open", error=str(e))
                                still2.append(tr)
                                continue
                        else:
                            e_close = float(self.bars.iloc[exit_idx]["close"])
                            mult = (e_close * (1.0 - self.fee_side)) / (float(tr["entry_open"]) * (1.0 + self.fee_side))
                            realized = (mult - 1.0) * 100.0

                            if self.audit is not None:
                                _audit_trade_closed(
                                    self.audit,
                                    symbol=self.symbol,
                                    paper=True,
                                    entry_time_utc=_iso_utc(tr["entry_time"]),
                                    exit_time_utc=_iso_utc(exit_time),
                                    entry_price=float(tr["entry_open"]),
                                    exit_price=float(e_close),
                                    fee_side=float(self.fee_side),
                                    exit_rel_min=int(exit_k),
                                    realized_ret_pct=float(realized),
                                    predicted_ret_pct=pred,
                                    entry_score=_finite_or_none(tr.get("entry_score")),
                                    entry_threshold=_finite_or_none(tr.get("entry_threshold")),
                                    extra={"trade_mode": "paper"},
                                )

                            # Update gate on paper trades.
                            try:
                                self.gate.record_paper_trade(entry_time_utc=_iso_utc(tr["entry_time"]), realized_ret_pct=float(realized))
                                save_gate_state(self.gate, self.gate_state_path)
                            except Exception as e:
                                self._alert("gate_update_error", "Failed to update gate state", error=str(e))

                            self.records.append(
                                {
                                    "paper": True,
                                    "trade_mode": "paper",
                                    "entry_time": pd.to_datetime(tr["entry_time"]).to_pydatetime(),
                                    "exit_time": exit_time.to_pydatetime(),
                                    "exit_rel_min": exit_k,
                                    "predicted_ret_pct": pred,
                                    "realized_ret_pct": realized,
                                }
                            )
                    else:
                        still2.append(tr)
                self.open_trades = still2

                # Periodically flush outputs
                if len(self.records) and (len(self.records) % 20 == 0):
                    self._flush_outputs()

                if self.stop_after_minutes and self._start_wall is not None:
                    if (time.time() - float(self._start_wall)) >= float(self.stop_after_minutes) * 60.0:
                        self._alert(
                            "stop_after_minutes",
                            "Stop-after-minutes reached; closing websocket",
                            stop_after_minutes=int(self.stop_after_minutes),
                        )
                        ws.close()
                        return
            except Exception as e:
                self._alert("live_error", "Unhandled exception in on_message", error=str(e))

        def on_error(ws, error):
            print("ws_error:", error)
        def on_close(ws, status, msg):
            print("ws_closed", status, msg)
            self._flush_outputs()
        def on_open(ws):
            print("ws_opened")

        ws = websocket.WebSocketApp(url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
        # run forever (blocking)
        ws.run_forever(ping_interval=15, ping_timeout=10)

    def _flush_outputs(self):
        if not self.records:
            return
        trades = pd.DataFrame(self.records)
        trades.sort_values('entry_time', inplace=True)
        trades['date'] = pd.to_datetime(trades['entry_time']).dt.date
        daily = trades.groupby('date', as_index=False).agg(
            n_trades=('realized_ret_pct','size'),
            mean_daily_pct=('realized_ret_pct','mean'),
            sum_daily_pct=('realized_ret_pct','sum'),
            median_daily_pct=('realized_ret_pct','median'),
            top_day_pct=('realized_ret_pct','max'),
            worst_day_pct=('realized_ret_pct','min'),
        )
        ts = _now_ts()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        trades.to_csv(self.out_dir / f"trades_{ts}.csv", index=False)
        daily.to_csv(self.out_dir / f"daily_{ts}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Live/replay minute-level paper trader with feature parity")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument(
        "--entry-model",
        default=str(REPO_ROOT / "models" / "entry_regressor_btcusdt_2025-12-20T13-47-55Z.joblib"),
    )
    ap.add_argument(
        "--exit-model",
        default=str(REPO_ROOT / "models" / "exit_regressor_btcusdt_2025-12-20T14-41-31Z.joblib"),
    )

    # Seed thresholds (optional)
    ap.add_argument(
        "--thresholds-csv",
        default=str(REPO_ROOT / "data" / "exit_regression" / "entry_score_thresholds_online_2025-12-20T14-27-34Z.csv"),
        help="Optional seed thresholds CSV (date, threshold). Use '' to disable.",
    )

    # Live persistence + alerts
    ap.add_argument(
        "--thresholds-live-out",
        default=str(REPO_ROOT / "data" / "live" / "entry_score_thresholds_online_live.csv"),
        help="Append-only CSV written at day roll (date, threshold, n_prior_scores, ...). Use '' to disable.",
    )
    ap.add_argument(
        "--alerts-file",
        default="",
        help="If set, append JSONL alerts here (still prints to stdout).",
    )
    ap.add_argument(
        "--audit-log",
        default=str(REPO_ROOT / "data" / "live" / f"audit_{_now_ts()}.jsonl"),
        help="Append-only JSONL audit log (validated against contracts/v1). Use '' to disable.",
    )
    ap.add_argument("--stall-seconds", type=int, default=120)
    ap.add_argument("--alert-cooldown-seconds", type=int, default=300)

    ap.add_argument(
        "--metrics-port",
        type=int,
        default=0,
        help="If >0, start an HTTP server with /health and /metrics on this port",
    )
    ap.add_argument(
        "--health-max-staleness-sec",
        type=float,
        default=120.0,
        help="Max seconds since last closed 1m bar for /health to be OK",
    )
    ap.add_argument(
        "--eod-send-email",
        action="store_true",
        help="If set, the day-roll EOD report will also be emailed (requires EMAIL_TO/EMAIL_FROM/SMTP_* env vars)",
    )

    ap.add_argument("--target-frac", type=float, default=0.001)
    ap.add_argument("--max-entries-per-day", type=int, default=2)
    ap.add_argument("--fee", type=float, default=0.001)

    ap.add_argument(
        "--force-entry",
        action="store_true",
        help="Force exactly one entry (paper) as soon as possible (still scores model) to validate live plumbing.",
    )
    ap.add_argument(
        "--stop-after-minutes",
        type=int,
        default=0,
        help="If >0, stop live mode after this many minutes (closes websocket to trigger output flush).",
    )

    # Trade mode (live-only)
    ap.add_argument(
        "--trade-mode",
        choices=["paper", "auto", "real"],
        default="paper",
        help="paper: simulated only; auto: start paper then switch to real after gate; real: always real",
    )
    ap.add_argument(
        "--gate-window",
        type=int,
        default=5,
        help="Minimum number of completed paper trades required before switching to real at the first UTC day-roll (EOD) in --trade-mode auto",
    )
    ap.add_argument("--trade-margin-eur", type=float, default=15.0, help="Margin per trade (EUR) used for sizing")
    ap.add_argument("--eurusdt", type=float, default=0.0, help="Override EURUSDT conversion rate (0 = fetch from Binance)")

    ap.add_argument("--binance-api-key", type=str, default="", help="Binance API key (or set BINANCE_API_KEY env var)")
    ap.add_argument("--binance-api-secret", type=str, default="", help="Binance API secret (or set BINANCE_API_SECRET env var)")
    ap.add_argument("--binance-base-url", type=str, default=BINANCE_FAPI, help="Binance USDT-M futures base URL")
    ap.add_argument("--margin-type", type=str, default="ISOLATED", help="ISOLATED or CROSSED")
    ap.add_argument(
        "--allow-leverage-above-tier",
        dest="allow_leverage_above_tier",
        action="store_true",
        default=True,
        help="Always use the highest leverage Binance allows for the trade size (default: enabled)",
    )
    ap.add_argument(
        "--respect-tier-leverage",
        dest="allow_leverage_above_tier",
        action="store_false",
        help="Disable max leverage; cap leverage at the strategy tier target",
    )
    ap.add_argument("--max-open-real", type=int, default=1, help="Max concurrent open real positions")
    ap.add_argument(
        "--gate-state-path",
        type=str,
        default="",
        help="Path to persisted gate_state.json (default: <out-dir>/gate_state.json)",
    )
    ap.add_argument(
        "--ledger-path",
        type=str,
        default="",
        help="Path to persisted strategy_ledger.json (default: <out-dir>/strategy_ledger.json)",
    )

    ap.add_argument("--mode", choices=["replay", "live"], default="replay")

    # Replay-only
    ap.add_argument("--start", type=str, default=None, help="UTC start (YYYY-MM-DD) for replay")
    ap.add_argument("--end", type=str, default=None, help="UTC end (YYYY-MM-DD, exclusive) for replay")
    ap.add_argument(
        "--bars-dir",
        type=str,
        default="",
        help="If set, load minute_bars_YYYY-MM-DD.csv files from this directory instead of fetching from Binance (reproducible/offline replay)",
    )

    # Live-only
    ap.add_argument("--grace-sec", type=float, default=1.5)
    ap.add_argument("--backfill-hours", type=int, default=48)

    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "live"))
    args = ap.parse_args()

    models = load_models(Path(args.entry_model), Path(args.exit_model))

    thresholds_csv = Path(args.thresholds_csv) if args.thresholds_csv else None
    thresholds_live_out = Path(args.thresholds_live_out) if args.thresholds_live_out else None
    alerts_file = Path(args.alerts_file) if args.alerts_file else None
    audit_log = Path(args.audit_log) if args.audit_log else None

    audit_cm = AuditLogger.create(audit_log) if audit_log else nullcontext()
    with audit_cm as audit:
        if audit is not None:
            audit.write(
                {
                    "kind": KIND_RUN_META,
                    "mode": args.mode,
                    "symbol": args.symbol.upper(),
                    "models": {
                        "entry": {
                            "artifact": models.entry_artifact,
                            "created_utc": models.entry_created_utc,
                            "n_features": len(models.entry_features),
                        },
                        "exit": {
                            "artifact": models.exit_artifact,
                            "created_utc": models.exit_created_utc,
                            "n_features": len(models.exit_features),
                        },
                    "pre_min": int(models.pre_min),
                    "entry_pre_min": int(models.entry_pre_min),
                    "exit_pre_min": int(models.exit_pre_min),
                    },
                    "code_version": {
                        "script": Path(__file__).name,
                        "git_commit": _git_commit_hash(REPO_ROOT),
                    },
                }
            )

        if args.mode == "replay":
            if not args.start or not args.end:
                raise SystemExit("--start and --end are required in replay mode")
            start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
            end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

            bars_dir = Path(args.bars_dir) if getattr(args, "bars_dir", "") else None

            out_paths = simulate_replay(
                symbol=args.symbol,
                models=models,
                start=start,
                end=end,
                thresholds_csv=thresholds_csv,
                target_frac=float(args.target_frac),
                max_entries_per_day=int(args.max_entries_per_day),
                fee_side=float(args.fee),
                out_dir=Path(args.out_dir),
                bars_dir=bars_dir,
                audit=audit,
            )
            print("Trades:", out_paths["trades"])
            print("Daily:", out_paths["daily"])
            if audit_log:
                print("Audit:", audit_log)
            return

        api_key = str(args.binance_api_key or os.getenv("BINANCE_API_KEY") or "")
        api_secret = str(args.binance_api_secret or os.getenv("BINANCE_API_SECRET") or "")

        runner = LiveRunner(
            symbol=args.symbol,
            models=models,
            thresholds_csv=thresholds_csv,
            thresholds_live_out=thresholds_live_out,
            target_frac=float(args.target_frac),
            max_entries_per_day=int(args.max_entries_per_day),
            fee_side=float(args.fee),
            out_dir=Path(args.out_dir),
            trade_mode=str(args.trade_mode),
            gate_window=int(args.gate_window),
            trade_margin_eur=float(args.trade_margin_eur),
            eurusdt=float(args.eurusdt),
            binance_api_key=api_key,
            binance_api_secret=api_secret,
            binance_base_url=str(args.binance_base_url),
            margin_type=str(args.margin_type),
            allow_leverage_above_tier=bool(args.allow_leverage_above_tier),
            max_open_real=int(args.max_open_real),
            gate_state_path=(Path(args.gate_state_path) if args.gate_state_path else None),
            ledger_path=(Path(args.ledger_path) if args.ledger_path else None),
            force_entry=bool(args.force_entry),
            stop_after_minutes=int(args.stop_after_minutes),
            grace_sec=float(args.grace_sec),
            backfill_hours=int(args.backfill_hours),
            stall_seconds=int(args.stall_seconds),
            alert_cooldown_seconds=int(args.alert_cooldown_seconds),
            alerts_file=alerts_file,
            audit=audit,
            metrics_port=int(args.metrics_port),
            health_max_staleness_sec=float(args.health_max_staleness_sec),
            eod_send_email=bool(args.eod_send_email),
        )
        try:
            runner.run()
        except KeyboardInterrupt:
            print("\nInterrupted; flushing outputs...")
            runner._flush_outputs()


if __name__ == "__main__":
    main()
