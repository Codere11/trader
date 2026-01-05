#!/usr/bin/env python3
# Timestamp (UTC): 2025-12-24T13:14:02Z
"""Live/paper dYdX v4 runner for BTC-USD perpetual.

Implements the two-subaccount scheme:
- trading subaccount: used for trading; uses (almost) all available collateral each attempt
- bank subaccount: receives a configurable fraction of realized profits after each closed trade (default 30%)
- if trading equity drops below a floor, top up from bank back to the floor (between trades)

Strategy logic (UPDATED 2026-01-03):
- Entry model: pattern_entry_regressor_v2 (dYdX-trained) using the same feature engineering as offline training.
- Entry selection: causal per-day online threshold (quantile of prior scores) targeting --target-frac.
- Exit (NEW): oracle-gap regressor policy (predict oracle_gap_pct = oracle_ret_pct - ret_if_exit_now_pct).
  Policy: exit at the first minute k>=min_exit_k where pred_gap_pct <= tau_gap; else force-exit at hold_min.

Notes:
- Backfill is required so rolling features (incl. 1d z-scores) are defined.

WARNING: In --trade-mode real, this places real orders and transfers.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import queue
import re
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


@dataclass
class ExitGapRegressor:
    model: Any
    feature_cols: List[str]
    artifact: str
    created_utc: Optional[str]


@dataclass
class TopupBudget:
    """Persisted per-agent budget for initial (pre-threshold) topups.

    We treat bank->trading *floor topups while bank_equity < bank_threshold* as consuming
    the agent's initial external topup budget. Once exhausted, the agent will refuse
    new entries (but will always allow exits).
    """

    budget_usdc: float
    spent_usdc: float
    state_path: Path

    def remaining_usdc(self) -> float:
        return max(0.0, float(self.budget_usdc) - float(self.spent_usdc))

    def exhausted(self) -> bool:
        return float(self.spent_usdc) >= float(self.budget_usdc)

    def load(self) -> None:
        try:
            if not self.state_path.exists():
                return
            obj = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.spent_usdc = float(obj.get("spent_usdc") or obj.get("initial_topup_spent_usdc") or 0.0)
        except Exception:
            return

    def save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_utc": _now_ts(),
                "budget_usdc": float(self.budget_usdc),
                "spent_usdc": float(self.spent_usdc),
            }
            self.state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except Exception:
            return


def load_exit_gap_regressor(path: Optional[Path]) -> Optional[ExitGapRegressor]:
    if path is None:
        return None
    p = Path(path)
    if not str(p).strip():
        return None
    obj = joblib.load(p)
    if not isinstance(obj, dict) or "model" not in obj or "feature_cols" not in obj:
        raise SystemExit(f"Unexpected exit gap regressor artifact format: {p}")
    return ExitGapRegressor(
        model=obj["model"],
        feature_cols=[str(c) for c in list(obj.get("feature_cols") or [])],
        artifact=p.name,
        created_utc=obj.get("created_utc"),
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from contracts import AuditLogger
from contracts.v1 import (
    KIND_ENTRY_DECISION,
    KIND_MARKET_BAR_1M_CLOSED,
    KIND_RUN_META,
    KIND_TRADE_CLOSED,
)

from binance_adapter.crossref_oracle_vs_agent_patterns import FEATURES, compute_feature_frame


# --- Pattern entry v2 feature engineering (must match training) ---

def _slope5(series: pd.Series) -> pd.Series:
    # x = [-2,-1,0,1,2]
    return ((-2.0 * series.shift(4)) + (-1.0 * series.shift(3)) + (1.0 * series.shift(1)) + (2.0 * series)) / 10.0


def _accel5(series: pd.Series) -> pd.Series:
    return (series - 2.0 * series.shift(2) + series.shift(4)) / 2.0


def _zscore_roll(s: pd.Series, win: int) -> pd.Series:
    mu = s.rolling(int(win), min_periods=int(win)).mean()
    sd = s.rolling(int(win), min_periods=int(win)).std()
    return (s - mu) / (sd + 1e-12)


def build_entry_pattern_frame_v2(bars: pd.DataFrame, base_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the v2 pattern frame used by pattern_entry_regressor_v2.

    Must match scripts/pattern_entry_regressor_oracle_sweep_v2_2026-01-02T23-28-15Z.py.

    Returns (pattern_frame, base_feature_frame_src).
    """
    bars = bars.copy().sort_values("timestamp").reset_index(drop=True)
    bars["ts_min"] = pd.to_datetime(bars["timestamp"]).dt.floor("min")

    base = compute_feature_frame(bars)
    src = base[[c for c in FEATURES if c in base.columns]]

    missing = [f for f in base_features if f not in src.columns]
    if missing:
        raise ValueError(f"Missing required base features: {missing}")

    df = pd.DataFrame({"timestamp": pd.to_datetime(bars["timestamp"], utc=True)})

    # 5-minute pattern descriptors
    for f in base_features:
        s = pd.to_numeric(src[f], errors="coerce")
        df[f"{f}__last"] = s
        df[f"{f}__slope5"] = _slope5(s)
        df[f"{f}__accel5"] = _accel5(s)
        df[f"{f}__std5"] = s.rolling(5, min_periods=5).std()
        df[f"{f}__min5"] = s.rolling(5, min_periods=5).min()
        df[f"{f}__max5"] = s.rolling(5, min_periods=5).max()
        df[f"{f}__range5"] = df[f"{f}__max5"] - df[f"{f}__min5"]

    # Cross-feature correlations
    pairs = [
        ("macd", "ret_1m_pct"),
        ("vol_std_5m", "ret_1m_pct"),
        ("range_norm_5m", "ret_1m_pct"),
    ]
    for a, b in pairs:
        if a in base_features and b in base_features:
            df[f"corr5__{a}__{b}"] = pd.to_numeric(src[a], errors="coerce").rolling(5, min_periods=5).corr(
                pd.to_numeric(src[b], errors="coerce")
            )

    # Price context
    close = pd.to_numeric(bars["close"], errors="coerce")
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")

    df["px__ret1m_close"] = close.pct_change() * 100.0
    df["px__ret1m_abs"] = df["px__ret1m_close"].abs()
    df["px__range_norm1m"] = (high - low) / (close + 1e-12)
    df["px__range_norm1m_abs"] = df["px__range_norm1m"].abs()

    # New indicators: spike/pump/vol regime
    win = 1440
    df["z1d__px_ret1m"] = _zscore_roll(df["px__ret1m_close"], win)
    df["z1d__px_ret1m_abs"] = df["z1d__px_ret1m"].abs()
    df["z1d__px_range1m"] = _zscore_roll(df["px__range_norm1m"], win)
    df["z1d__px_range1m_abs"] = df["z1d__px_range1m"].abs()

    # Context baselines
    vol5 = df.get("vol_std_5m__last")
    rng5max = df.get("range_norm_5m__max5")
    if vol5 is not None:
        df["z1d__vol5"] = _zscore_roll(pd.to_numeric(vol5, errors="coerce"), win)
        df["risk__ret1m_abs_over_vol5"] = df["px__ret1m_abs"] / (pd.to_numeric(vol5, errors="coerce").abs() + 1e-9)
        df["risk__range1m_over_vol5"] = df["px__range_norm1m"] / (pd.to_numeric(vol5, errors="coerce").abs() + 1e-9)

    if rng5max is not None:
        df["risk__range1m_over_range5max"] = df["px__range_norm1m"] / (pd.to_numeric(rng5max, errors="coerce").abs() + 1e-12)

    # Directional decomposition
    if "ret_1m_pct__last" in df.columns:
        df["ret_1m_pct__last_pos"] = df["ret_1m_pct__last"].clip(lower=0.0)
        df["ret_1m_pct__last_neg"] = (-df["ret_1m_pct__last"]).clip(lower=0.0)

    # Simple extreme flags
    df["flag__ret1m_abs_z_gt3"] = (df["z1d__px_ret1m_abs"] > 3.0).astype(np.float32)
    df["flag__range1m_abs_z_gt3"] = (df["z1d__px_range1m_abs"] > 3.0).astype(np.float32)

    return df, src

from dydx_adapter import load_dotenv
from dydx_adapter.v4 import DydxV4Config, connect_v4, new_client_id, usdc_to_quantums

from dydx_v4_client import OrderFlags
from dydx_v4_client.indexer.candles_resolution import CandlesResolution
from dydx_v4_client.indexer.rest.constants import OrderType
from dydx_v4_client.indexer.rest.indexer_client import IndexerClient
from dydx_v4_client.indexer.socket.websocket import IndexerSocket
from dydx_v4_client.node.client import QueryNodeClient
from dydx_v4_client.node.market import Market, since_now
from dydx_v4_client.node.message import subaccount

from v4_proto.dydxprotocol.clob.order_pb2 import Order


def _pb_to_jsonable(x: Any) -> Any:
    """Best-effort conversion for protobuf messages returned by dydx-v4-client.

    Audit logs must be strict JSON; protobuf objects are not JSON-serializable.
    """

    try:
        return QueryNodeClient.transcode_response(x)  # Message -> dict/list
    except Exception:
        return str(x)


@dataclass
class Models:
    entry_model: Any
    entry_features: List[str]

    # Pattern entry v2 uses these base features to build its derived frame.
    entry_base_features: List[str]

    # Optional legacy exit model (disabled for the fixed-hold strategy).
    exit_model: Optional[Any]
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
    return pd.to_datetime(ts_like, utc=True).to_pydatetime().isoformat()


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


BANK_SCHEMA_VERSION = "bank_v1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(obj: Mapping[str, Any]) -> str:
    """Deterministic JSON encoding for hashing (sorted keys, strict JSON)."""

    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _txhash_hex(txhash: Any) -> Optional[str]:
    """Best-effort conversion of txhash field(s) to hex string.

    dydx-v4-client transcodes txhash into either:
    - list[int] (bytes)
    - hex string
    """

    if txhash is None:
        return None

    if isinstance(txhash, str):
        s = txhash.strip()
        return s or None

    if isinstance(txhash, (bytes, bytearray)):
        return bytes(txhash).hex()

    if isinstance(txhash, list):
        try:
            b = bytes(int(x) & 0xFF for x in txhash)
            return b.hex()
        except Exception:
            return None

    return None


@dataclass
class BankTradeLogger:
    """Append-only, hash-chained JSONL log intended for "bank-grade" trade reporting."""

    path: Path
    run_id: str
    autoflush: bool = True

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # If appending to an existing log, resume the hash chain from the last record.
        self._seq = 0
        self._prev_hash: Optional[str] = None
        self._trade_seq = 0
        try:
            if self.path.exists() and self.path.stat().st_size > 0:
                # Read the last ~8KB and parse the last non-empty line.
                tail_n = 8192
                with open(self.path, "rb") as rf:
                    sz = int(rf.seek(0, 2) or 0)
                    rf.seek(max(0, sz - tail_n))
                    data = rf.read().decode("utf-8", errors="replace")
                lines = [ln for ln in data.splitlines() if ln.strip()]
                if lines:
                    last = json.loads(lines[-1])
                    if isinstance(last, dict):
                        self._seq = int(last.get("seq") or 0)
                        self._prev_hash = str(last.get("hash")) if last.get("hash") else None

                    # Resume trade_seq from the most recent bank_trade_closed record (the last line may be a snapshot/event).
                    for ln in reversed(lines):
                        try:
                            obj = json.loads(ln)
                        except Exception:
                            continue
                        if not isinstance(obj, dict):
                            continue
                        if str(obj.get("kind")) != "bank_trade_closed":
                            continue
                        if obj.get("trade_seq") is None:
                            continue
                        try:
                            self._trade_seq = int(obj.get("trade_seq") or 0)
                        except Exception:
                            self._trade_seq = 0
                        break
        except Exception:
            # If anything goes wrong, just start a new chain.
            self._seq = 0
            self._prev_hash = None
            self._trade_seq = 0

        self._fh = open(self.path, "a", encoding="utf-8")

    @classmethod
    def create(cls, path: Path, *, run_id: str, autoflush: bool = True) -> "BankTradeLogger":
        return cls(path=Path(path), run_id=str(run_id), autoflush=autoflush)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def _write(self, record: Mapping[str, Any]) -> None:
        obj: Dict[str, Any] = dict(record)

        obj.setdefault("bank_schema_version", BANK_SCHEMA_VERSION)
        obj.setdefault("run_id", str(self.run_id))
        obj.setdefault("record_time_utc", _utc_now_iso())

        self._seq += 1
        obj.setdefault("seq", int(self._seq))
        obj.setdefault("prev_hash", self._prev_hash)

        # Hash covers the full record (including prev_hash and seq), excluding the hash field itself.
        to_hash = dict(obj)
        to_hash.pop("hash", None)
        to_hash.pop("hash_algo", None)
        h = _sha256_hex(_canonical_json(to_hash))

        obj["hash_algo"] = "sha256"
        obj["hash"] = h

        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        self._fh.write(line + "\n")
        if self.autoflush:
            self._fh.flush()

        self._prev_hash = h

    def write_meta(self, meta: Mapping[str, Any]) -> None:
        self._write({"kind": "bank_log_meta", **dict(meta)})

    def write_event(self, kind: str, payload: Mapping[str, Any]) -> None:
        # Generic bank-proof event. Keep it JSON-only and hash-chained.
        k = str(kind).strip() or "bank_event"
        self._write({"kind": k, **dict(payload)})

    def write_trade_closed(self, trade: Mapping[str, Any]) -> None:
        self._trade_seq += 1
        self._write({"kind": "bank_trade_closed", "trade_seq": int(self._trade_seq), **dict(trade)})

    def __enter__(self) -> "BankTradeLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _finite_or_none(x: Any) -> Optional[float]:
    try:
        fx = float(x)
    except Exception:
        return None
    return fx if np.isfinite(fx) else None


def net_return_pct(entry_px_open: float, exit_px: float, fee_side: float) -> float:
    """Net return % from entry open -> exit price, net of symmetric per-side fees.

    Matches the training/backtest label definition used by build_exit_regression_dataset_2025-12-20T14-18-35Z.py.
    """

    e = float(entry_px_open)
    x = float(exit_px)
    fee = max(0.0, float(fee_side))
    mult = (x * (1.0 - fee)) / max(1e-12, (e * (1.0 + fee)))
    return (mult - 1.0) * 100.0


def realized_ret_pct_from_fills(
    *,
    entry_fill_px: float,
    exit_fill_px: float,
    size_btc: float,
    entry_fee_usdc: float,
    exit_fee_usdc: float,
) -> float:
    """Realized return % from actual fills (handles absolute fees in USDC).

    Computes return on entry notional (including entry fee):
        (exit_notional - exit_fee) / (entry_notional + entry_fee) - 1

    This is what we should report for real trades; it will match equity deltas.
    """

    e_px = float(entry_fill_px)
    x_px = float(exit_fill_px)
    q = max(0.0, float(size_btc))
    e_fee = max(0.0, float(entry_fee_usdc))
    x_fee = max(0.0, float(exit_fee_usdc))

    entry_notional = (e_px * q) + e_fee
    exit_notional = (x_px * q) - x_fee

    mult = float(exit_notional) / max(1e-12, float(entry_notional))
    return (mult - 1.0) * 100.0


_MEAN_SUFFIX_RE = re.compile(r"_mean_(\d+)m$")


def _parse_pre_min(payload: Mapping[str, Any]) -> int:
    ctx = payload.get("context") if isinstance(payload, Mapping) else None
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


def load_models(entry_path: Path, exit_path: Optional[Path]) -> Models:
    eobj = joblib.load(entry_path)

    if not isinstance(eobj, dict) or "model" not in eobj:
        raise SystemExit(f"Unexpected entry model artifact format: {entry_path}")

    # Support both legacy entry artifacts (key: 'features') and pattern_entry_regressor_v2 artifacts (key: 'feature_cols').
    if "features" in eobj:
        entry_feats = list(eobj.get("features") or [])
    elif "feature_cols" in eobj:
        entry_feats = list(eobj.get("feature_cols") or [])
    else:
        raise SystemExit(f"Entry model artifact missing 'features' or 'feature_cols': {entry_path}")

    entry_base_features = list(eobj.get("base_features") or ["macd", "ret_1m_pct", "mom_5m_pct", "vol_std_5m", "range_norm_5m"])

    entry_pre_min = int(_parse_pre_min(eobj))

    exit_model = None
    exit_feats: List[str] = []
    exit_pre_min = 0
    exit_artifact = ""
    exit_created_utc: Optional[str] = None

    if exit_path is not None and str(exit_path).strip():
        xobj = joblib.load(exit_path)
        if not isinstance(xobj, dict) or "model" not in xobj or "features" not in xobj:
            raise SystemExit(f"Unexpected exit model artifact format: {exit_path}")
        exit_model = xobj["model"]
        exit_feats = list(xobj.get("features") or [])
        exit_pre_min = int(_parse_pre_min(xobj))
        exit_artifact = Path(exit_path).name
        exit_created_utc = xobj.get("created_utc")

    pre_min = int(max(entry_pre_min, exit_pre_min, 0))

    return Models(
        entry_model=eobj["model"],
        entry_features=entry_feats,
        entry_base_features=entry_base_features,
        exit_model=exit_model,
        exit_features=exit_feats,
        entry_pre_min=entry_pre_min,
        exit_pre_min=exit_pre_min,
        pre_min=pre_min,
        entry_artifact=Path(entry_path).name,
        exit_artifact=exit_artifact,
        entry_created_utc=eobj.get("created_utc"),
        exit_created_utc=exit_created_utc,
    )


def build_feature_frame(
    bars: pd.DataFrame,
    pre_mins: Any,
    feat_names: List[str],
    *,
    return_missing: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build a feature frame that supports multiple context windows.

    This is used to keep entry+exit models independent: we build the union of feature
    columns and include rolling context means for every required pre_min.
    """

    # Normalize pre_mins input.
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
    return out, missing


class CandleAssembler:
    """Convert dYdX candle stream (updates within the minute) into CLOSED 1m bars."""

    def __init__(self) -> None:
        self._cur_started_at: Optional[pd.Timestamp] = None
        self._cur: Optional[Dict[str, Any]] = None

    @staticmethod
    def _parse(c: Mapping[str, Any]) -> Dict[str, Any]:
        ts = pd.to_datetime(c.get("startedAt"), utc=True)
        return {
            "timestamp": ts,
            "open": float(c.get("open")),
            "high": float(c.get("high")),
            "low": float(c.get("low")),
            "close": float(c.get("close")),
            "volume": float(c.get("baseTokenVolume") or 0.0),
        }

    def ingest(self, candle: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        row = self._parse(candle)
        ts = pd.to_datetime(row["timestamp"], utc=True)

        if self._cur_started_at is None:
            self._cur_started_at = ts
            self._cur = dict(row)
            return None

        prev_ts = pd.to_datetime(self._cur_started_at, utc=True)

        if ts < prev_ts:
            return None

        if ts == prev_ts:
            # in-minute update
            self._cur = dict(row)
            return None

        # ts > prev_ts: previous minute is now closed
        closed = dict(self._cur or {})
        self._cur_started_at = ts
        self._cur = dict(row)
        return closed


async def _fetch_market_meta(clients, ticker: str) -> Dict[str, Any]:
    resp = await clients.indexer.markets.get_perpetual_markets(market=str(ticker))
    mkts = resp.get("markets") if isinstance(resp, dict) else None
    if isinstance(mkts, dict) and ticker in mkts:
        return dict(mkts[ticker])
    if isinstance(mkts, dict) and len(mkts) == 1:
        return dict(list(mkts.values())[0])
    raise RuntimeError(f"Failed to fetch market meta for {ticker}: keys={list((resp or {}).keys())}")


async def _fetch_oracle_price(clients, ticker: str) -> float:
    mk = await _fetch_market_meta(clients, ticker)
    px = mk.get("oraclePrice")
    if px is None:
        raise RuntimeError(f"Market {ticker} missing oraclePrice")
    return float(px)


def _imf_to_max_leverage(imf: float) -> int:
    if not np.isfinite(imf) or imf <= 0:
        return 1
    return max(1, int(1.0 / float(imf)))


async def _get_subaccount_equity_usdc(clients, *, address: str, subaccount_number: int) -> float:
    res = await clients.indexer.account.get_subaccount(address, int(subaccount_number))
    if not isinstance(res, dict):
        raise RuntimeError(f"Unexpected subaccount response type: {type(res)}")

    # Try top-level first.
    for k in ["equity", "freeCollateral", "quoteBalance", "balance"]:
        if res.get(k) is not None:
            try:
                return float(res[k])
            except Exception:
                pass

    sub = res.get("subaccount") if isinstance(res.get("subaccount"), dict) else None
    if sub is not None:
        for k in ["equity", "freeCollateral", "quoteBalance", "balance"]:
            if sub.get(k) is not None:
                try:
                    return float(sub[k])
                except Exception:
                    pass

    raise RuntimeError(f"Could not extract equity/freeCollateral from subaccount response keys={list(res.keys())}")


async def _get_open_position_size_btc(clients, *, address: str, subaccount_number: int, market: str) -> float:
    """Return current perp position size (BTC) for a given market; 0.0 if none."""

    res = await clients.indexer.account.get_subaccount(address, int(subaccount_number))
    sub = res.get("subaccount") if isinstance(res, dict) and isinstance(res.get("subaccount"), dict) else None
    opp = sub.get("openPerpetualPositions") if isinstance(sub, dict) else None
    if not isinstance(opp, dict) or not opp:
        return 0.0

    p = opp.get(str(market).upper())
    if not isinstance(p, dict):
        return 0.0

    # dYdX indexer returns size as a string.
    try:
        return float(p.get("size") or 0.0)
    except Exception:
        return 0.0


def _parse_dt_utc(ts: Any) -> Optional[datetime]:
    try:
        return pd.to_datetime(ts, utc=True).to_pydatetime()
    except Exception:
        return None


async def _wait_for_fills(
    clients,
    *,
    address: str,
    subaccount_number: int,
    market: str,
    side: str,
    since_dt: datetime,
    min_size_btc: float,
    timeout_s: float = 10.0,
    poll_s: float = 0.5,
) -> Dict[str, Any]:
    """Poll indexer fills until we observe >= min_size_btc filled since since_dt.

    Returns a summary dict with vwap/fees and the matching fills (newest-first from API).
    Also includes first/last fill timestamps.
    """

    side_u = str(side).upper()
    want = max(0.0, float(min_size_btc))
    deadline = time.time() + float(timeout_s)

    best: Dict[str, Any] = {
        "found": False,
        "side": side_u,
        "market": str(market).upper(),
        "since_time_utc": since_dt.isoformat(),
        "size_btc": 0.0,
        "vwap_price": None,
        "fee_usdc": 0.0,
        "n_fills": 0,
        "fills": [],
    }

    while time.time() < deadline:
        try:
            resp = await clients.indexer.account.get_subaccount_fills(
                address=str(address),
                subaccount_number=int(subaccount_number),
                ticker=str(market).upper(),
                limit=50,
            )
        except Exception:
            await asyncio.sleep(float(poll_s))
            continue

        fills = resp.get("fills") if isinstance(resp, dict) else None
        if not isinstance(fills, list):
            await asyncio.sleep(float(poll_s))
            continue

        picked: List[Dict[str, Any]] = []
        for f in fills:
            if not isinstance(f, dict):
                continue
            if str(f.get("side") or "").upper() != side_u:
                continue
            dt = _parse_dt_utc(f.get("createdAt"))
            if dt is None or dt < since_dt:
                continue
            picked.append(f)

        # Compute VWAP over picked fills (which may include partials).
        qty = 0.0
        notional = 0.0
        fee_usdc = 0.0
        for f in picked:
            try:
                q = float(f.get("size") or 0.0)
                px = float(f.get("price") or 0.0)
                fee_usdc += float(f.get("fee") or 0.0)
            except Exception:
                continue
            if q <= 0 or px <= 0:
                continue
            qty += q
            notional += q * px

        if qty > 0:
            dts: List[datetime] = []
            for f in picked:
                dt = _parse_dt_utc(f.get("createdAt"))
                if dt is not None:
                    dts.append(dt)

            first_dt = min(dts) if dts else None
            last_dt = max(dts) if dts else None

            best = {
                "found": True,
                "side": side_u,
                "market": str(market).upper(),
                "since_time_utc": since_dt.isoformat(),
                "size_btc": float(qty),
                "vwap_price": float(notional / max(1e-12, qty)),
                "fee_usdc": float(fee_usdc),
                "n_fills": int(len(picked)),
                "first_fill_time_utc": (first_dt.isoformat().replace("+00:00", "Z") if first_dt else None),
                "last_fill_time_utc": (last_dt.isoformat().replace("+00:00", "Z") if last_dt else None),
                "fills": [
                    {
                        "createdAt": f.get("createdAt"),
                        "side": f.get("side"),
                        "size": f.get("size"),
                        "price": f.get("price"),
                        "fee": f.get("fee"),
                        "liquidity": f.get("liquidity"),
                        "orderId": f.get("orderId"),
                    }
                    for f in picked
                ],
            }

        if qty >= want and want > 0:
            return best

        await asyncio.sleep(float(poll_s))

    return best


async def _ensure_trading_floor(clients, cfg: DydxV4Config, *, budget: Optional[TopupBudget] = None) -> Dict[str, Any]:
    """Top up trading subaccount from bank to reach the configured floor (between trades).

    Policy:
    - Only allow bank->trading floor topups when bank_equity < cfg.bank_threshold_usdc.
    - If a TopupBudget is provided, floor topups in this regime consume budget and are capped.
    """

    trading_eq = float(await _get_subaccount_equity_usdc(clients, address=cfg.address, subaccount_number=cfg.subaccount_trading))
    floor = float(cfg.trade_floor_usdc)

    if trading_eq >= floor:
        return {"topped_up": False, "needed_usdc": 0.0, "transferred_usdc": 0.0}

    needed = float(floor - trading_eq)

    bank_eq = float(await _get_subaccount_equity_usdc(clients, address=cfg.address, subaccount_number=cfg.subaccount_bank))
    bank_threshold = float(getattr(cfg, "bank_threshold_usdc", float("inf")))

    # IMPORTANT: user policy - only top up to floor when bank is below the threshold.
    if float(bank_eq) >= float(bank_threshold):
        return {
            "topped_up": False,
            "needed_usdc": float(needed),
            "transferred_usdc": 0.0,
            "skipped": True,
            "skip_reason": "bank_equity_at_or_above_threshold",
            "bank_equity_usdc": float(bank_eq),
            "bank_threshold_usdc": float(bank_threshold),
            "trading_equity_usdc": float(trading_eq),
            "floor_usdc": float(floor),
        }

    if budget is not None and budget.exhausted():
        return {
            "topped_up": False,
            "needed_usdc": float(needed),
            "transferred_usdc": 0.0,
            "skipped": True,
            "skip_reason": "initial_topup_budget_exhausted",
            "budget_usdc": float(budget.budget_usdc),
            "spent_usdc": float(budget.spent_usdc),
            "bank_equity_usdc": float(bank_eq),
            "bank_threshold_usdc": float(bank_threshold),
            "trading_equity_usdc": float(trading_eq),
            "floor_usdc": float(floor),
        }

    remaining = float(budget.remaining_usdc()) if budget is not None else float("inf")
    transfer = float(min(float(needed), float(bank_eq), float(remaining)))

    if transfer <= 0.0:
        return {
            "topped_up": False,
            "needed_usdc": float(needed),
            "transferred_usdc": 0.0,
            "skipped": True,
            "skip_reason": "no_transfer_possible",
            "budget_usdc": (float(budget.budget_usdc) if budget is not None else None),
            "spent_usdc": (float(budget.spent_usdc) if budget is not None else None),
            "bank_equity_usdc": float(bank_eq),
            "bank_threshold_usdc": float(bank_threshold),
            "trading_equity_usdc": float(trading_eq),
            "floor_usdc": float(floor),
        }

    amt_q = usdc_to_quantums(float(transfer))
    if int(amt_q) <= 0:
        return {
            "topped_up": False,
            "needed_usdc": float(needed),
            "transferred_usdc": 0.0,
            "skipped": True,
            "skip_reason": "transfer_amount_rounded_to_zero",
            "budget_usdc": (float(budget.budget_usdc) if budget is not None else None),
            "spent_usdc": (float(budget.spent_usdc) if budget is not None else None),
            "bank_equity_usdc": float(bank_eq),
            "bank_threshold_usdc": float(bank_threshold),
            "trading_equity_usdc": float(trading_eq),
            "floor_usdc": float(floor),
        }

    tx = await clients.node.transfer(
        clients.wallet,
        subaccount(cfg.address, int(cfg.subaccount_bank)),
        subaccount(cfg.address, int(cfg.subaccount_trading)),
        int(clients.usdc_asset_id),
        int(amt_q),
    )

    if budget is not None:
        budget.spent_usdc = float(budget.spent_usdc) + float(transfer)
        budget.save()

    return {
        "topped_up": True,
        "needed_usdc": float(needed),
        "transferred_usdc": float(transfer),
        "budget_usdc": (float(budget.budget_usdc) if budget is not None else None),
        "spent_usdc": (float(budget.spent_usdc) if budget is not None else None),
        "bank_equity_usdc": float(bank_eq),
        "bank_threshold_usdc": float(bank_threshold),
        "tx": _pb_to_jsonable(tx),
    }


async def _refinance_after_liquidation(clients, cfg: DydxV4Config, *, budget: Optional[TopupBudget] = None) -> Dict[str, Any]:
    """Refinance trading subaccount after a detected liquidation.

    Policy:
    - if bank_equity < cfg.bank_threshold_usdc: top up trading to cfg.trade_floor_usdc
    - else: transfer cfg.liquidation_recap_bank_frac of bank equity to trading
    """

    floor = float(cfg.trade_floor_usdc)
    bank_threshold = float(getattr(cfg, "bank_threshold_usdc", 280.0))
    recap_frac = float(getattr(cfg, "liquidation_recap_bank_frac", 0.10))

    trading_eq = float(await _get_subaccount_equity_usdc(clients, address=cfg.address, subaccount_number=cfg.subaccount_trading))
    bank_eq = float(await _get_subaccount_equity_usdc(clients, address=cfg.address, subaccount_number=cfg.subaccount_bank))

    action = ""
    requested = 0.0

    if float(bank_eq) < float(bank_threshold):
        action = "refill_to_floor"
        requested = max(0.0, float(floor) - float(trading_eq))
        # If the bank can't cover it, transfer what we can.
        requested = min(float(requested), float(bank_eq))

        # Budget applies only while bank < threshold.
        if budget is not None and budget.exhausted():
            return {
                "attempted": False,
                "action": str(action),
                "requested_usdc": float(requested),
                "transferred_usdc": 0.0,
                "budget_usdc": float(budget.budget_usdc),
                "spent_usdc": float(budget.spent_usdc),
                "skip_reason": "initial_topup_budget_exhausted",
                "bank_equity_usdc": float(bank_eq),
                "trading_equity_usdc": float(trading_eq),
                "bank_threshold_usdc": float(bank_threshold),
                "trade_floor_usdc": float(floor),
                "liquidation_recap_bank_frac": float(recap_frac),
            }

        if budget is not None:
            requested = min(float(requested), float(budget.remaining_usdc()))
    else:
        action = "bank_fraction"
        requested = max(0.0, float(bank_eq) * float(recap_frac))
        requested = min(float(requested), float(bank_eq))

    if requested <= 0.0:
        return {
            "attempted": False,
            "action": str(action),
            "requested_usdc": float(requested),
            "transferred_usdc": 0.0,
            "bank_equity_usdc": float(bank_eq),
            "trading_equity_usdc": float(trading_eq),
            "bank_threshold_usdc": float(bank_threshold),
            "trade_floor_usdc": float(floor),
            "liquidation_recap_bank_frac": float(recap_frac),
        }

    amt_q = usdc_to_quantums(float(requested))
    if int(amt_q) <= 0:
        return {
            "attempted": False,
            "action": str(action),
            "requested_usdc": float(requested),
            "transferred_usdc": 0.0,
            "bank_equity_usdc": float(bank_eq),
            "trading_equity_usdc": float(trading_eq),
            "bank_threshold_usdc": float(bank_threshold),
            "trade_floor_usdc": float(floor),
            "liquidation_recap_bank_frac": float(recap_frac),
            "skip_reason": "requested_amount_rounded_to_zero",
        }

    tx = await clients.node.transfer(
        clients.wallet,
        subaccount(cfg.address, int(cfg.subaccount_bank)),
        subaccount(cfg.address, int(cfg.subaccount_trading)),
        int(clients.usdc_asset_id),
        int(amt_q),
    )

    transferred = float(requested)

    if budget is not None and str(action) == "refill_to_floor" and float(bank_eq) < float(bank_threshold):
        budget.spent_usdc = float(budget.spent_usdc) + float(transferred)
        budget.save()

    return {
        "attempted": True,
        "action": str(action),
        "requested_usdc": float(requested),
        "transferred_usdc": float(transferred),
        "budget_usdc": (float(budget.budget_usdc) if budget is not None else None),
        "spent_usdc": (float(budget.spent_usdc) if budget is not None else None),
        "bank_equity_usdc": float(bank_eq),
        "trading_equity_usdc": float(trading_eq),
        "bank_threshold_usdc": float(bank_threshold),
        "trade_floor_usdc": float(floor),
        "liquidation_recap_bank_frac": float(recap_frac),
        "tx": _pb_to_jsonable(tx),
    }


async def _siphon_profit_and_topup(
    clients,
    cfg: DydxV4Config,
    *,
    equity_before: float,
    budget: Optional[TopupBudget] = None,
) -> Dict[str, Any]:
    """After a position is closed, siphon a configured fraction of realized profit to bank and top up to floor."""

    equity_after = float(await _get_subaccount_equity_usdc(clients, address=cfg.address, subaccount_number=cfg.subaccount_trading))
    profit = float(equity_after - float(equity_before))

    siphon = {"attempted": False, "profit_usdc": float(profit), "requested_usdc": 0.0, "transferred_usdc": 0.0}

    if profit > 0.0 and float(cfg.profit_siphon_frac) > 0.0:
        amt = float(profit) * float(cfg.profit_siphon_frac)
        if amt > 0.0:
            amt_q = usdc_to_quantums(amt)
            tx = await clients.node.transfer(
                clients.wallet,
                subaccount(cfg.address, int(cfg.subaccount_trading)),
                subaccount(cfg.address, int(cfg.subaccount_bank)),
                int(clients.usdc_asset_id),
                int(amt_q),
            )
            siphon = {
                "attempted": True,
                "profit_usdc": float(profit),
                "requested_usdc": float(amt),
                "transferred_usdc": float(amt),
                "tx": _pb_to_jsonable(tx),
            }

    # Top up (after siphon)
    topup = await _ensure_trading_floor(clients, cfg, budget=budget)

    return {
        "equity_before_usdc": float(equity_before),
        "equity_after_usdc": float(equity_after),
        "profit_usdc": float(profit),
        "siphon": siphon,
        "topup": topup,
    }


class DydxRunner:
    def __init__(
        self,
        *,
        market: str,
        models: Models,
        target_frac: float,
        hold_min: int,
        out_dir: Path,
        trade_mode: str,
        backfill_hours: int,
        thresholds_csv: Optional[Path],
        audit: Optional[AuditLogger],
        bank_log: Optional[BankTradeLogger],
        cfg: DydxV4Config,
        clients,
        loop: asyncio.AbstractEventLoop,
        exit_gap: Optional[ExitGapRegressor] = None,
        exit_gap_tau: float = 0.10,
        exit_gap_min_exit_k: int = 2,
        thresholds_live_out: Optional[Path] = None,
        prior_scores_path: Optional[Path] = None,
        max_prior_scores: int = 200_000,
        min_prior_scores: int = 2_000,
        seed_days: int = 2,
        retention_days: int = 0,
        metrics_port: int = 0,
        health_max_staleness_sec: float = 120.0,
        alert_cooldown_seconds: int = 300,
        initial_topup_budget_usdc: float = 50.0,
    ):
        self.market = str(market).upper()
        self.models = models
        self.target_frac = float(target_frac)
        self.hold_min = int(hold_min)
        self.out_dir = Path(out_dir)
        self.trade_mode = str(trade_mode).lower().strip()
        self.backfill_hours = int(backfill_hours)
        self.thresholds_csv = thresholds_csv
        self.audit = audit
        self.bank_log = bank_log
        self.cfg = cfg
        self.clients = clients
        self.loop = loop

        self.thresholds_live_out = Path(thresholds_live_out) if thresholds_live_out else None
        self.prior_scores_path = Path(prior_scores_path) if prior_scores_path else None
        self.max_prior_scores = int(max_prior_scores)
        self.min_prior_scores = int(min_prior_scores)
        self.seed_days = int(seed_days)
        self.retention_days = int(retention_days)

        self.metrics_port = int(metrics_port)
        self.health_max_staleness_sec = float(health_max_staleness_sec)
        self.alert_cooldown_seconds = int(alert_cooldown_seconds)
        self.last_closed_wall = time.time()
        self._last_alert_by_kind: Dict[str, float] = {}

        # Initial topup budget (per agent / per trading subaccount).
        st = re.sub(r"[^A-Za-z0-9]+", "_", str(self.market).lower()).strip("_")
        budget_path = self.out_dir / f"topup_budget_state_{st}_tr{int(self.cfg.subaccount_trading)}_bank{int(self.cfg.subaccount_bank)}.json"
        self.topup_budget = TopupBudget(
            budget_usdc=float(initial_topup_budget_usdc),
            spent_usdc=0.0,
            state_path=budget_path,
        )
        self.topup_budget.load()
        self.topup_budget.save()

        self.bars = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])  # closed bars only
        self.feats: Optional[pd.DataFrame] = None
        self.exit_src: Optional[pd.DataFrame] = None

        self.exit_gap = exit_gap
        self.exit_gap_tau = float(exit_gap_tau)
        self.exit_gap_min_exit_k = int(exit_gap_min_exit_k)

        # Entry features are built via the pattern v2 frame; legacy exit features are unused in this strategy.
        self._feat_union: List[str] = _union_feature_names(self.models.entry_features, self.models.exit_features)
        req_pre_mins = set(_infer_pre_mins_from_features(self._feat_union))
        for pm in [int(self.models.entry_pre_min), int(self.models.exit_pre_min)]:
            if pm > 0:
                req_pre_mins.add(int(pm))
        self._req_pre_mins: List[int] = sorted(req_pre_mins)

        self._missing_cols: List[str] = []
        self._missing_entry_cols: List[str] = []
        self._missing_exit_cols: List[str] = []

        self.thr_map: Dict[Any, float] = {}
        self.persisted_thr_dates: set = set()

        self.prior_scores: List[float] = []
        self._scores_cur_day: List[float] = []
        self.cur_day = None

        self.open_trade: Optional[Dict[str, Any]] = None
        self.records: List[Dict[str, Any]] = []

        # Real-mode safety: require multiple consecutive confirmations before clearing open_trade.
        self._open_pos_zero_streak: int = 0

        # Load seed thresholds.
        if self.thresholds_csv and self.thresholds_csv.exists():
            self._load_thresholds_csv(self.thresholds_csv)

        # Load persisted thresholds (so restarts keep the same daily threshold values).
        if self.thresholds_live_out and self.thresholds_live_out.exists():
            self._load_thresholds_csv(self.thresholds_live_out, record_persisted=True)

        # Load persisted prior score sample/window (so restarts keep the threshold distribution).
        if self.prior_scores_path and self.prior_scores_path.exists():
            self._load_prior_scores(self.prior_scores_path)

    def _alert(self, kind: str, message: str, **fields: Any) -> None:
        now = time.time()
        last = float(self._last_alert_by_kind.get(kind, 0.0))
        if now - last < float(self.alert_cooldown_seconds):
            return
        self._last_alert_by_kind[kind] = now

        payload = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "kind": str(kind),
            "message": str(message),
            **fields,
        }
        extra = f" {fields}" if fields else ""
        print(f"[ALERT] {payload['ts_utc']} {kind}: {message}{extra}")

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
                            "trade_mode": str(getattr(runner, "trade_mode", "")),
                            "has_open_trade": bool(getattr(runner, "open_trade", None) is not None),
                            "records_total": int(len(getattr(runner, "records", []) or [])),
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
                        has_open_trade = 1 if getattr(runner, "open_trade", None) is not None else 0
                        records_total = int(len(getattr(runner, "records", []) or []))

                        lines = [
                            "# HELP autotrader_last_closed_bar_age_seconds Seconds since last closed 1m bar was received",
                            "# TYPE autotrader_last_closed_bar_age_seconds gauge",
                            f"autotrader_last_closed_bar_age_seconds {age}",
                            "# HELP autotrader_has_open_trade Whether the runner currently tracks an open trade",
                            "# TYPE autotrader_has_open_trade gauge",
                            f"autotrader_has_open_trade {has_open_trade}",
                            "# HELP autotrader_records_total Number of closed trade records buffered in memory",
                            "# TYPE autotrader_records_total gauge",
                            f"autotrader_records_total {records_total}",
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

    def _load_thresholds_csv(self, path: Path, *, record_persisted: bool = False) -> None:
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
        if not np.isfinite(float(threshold)):
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

    def _cap_prior_scores(self) -> None:
        m = int(self.max_prior_scores)
        if m <= 0:
            return
        if len(self.prior_scores) > m:
            self.prior_scores = list(self.prior_scores[-m:])

    def _load_prior_scores(self, path: Path) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            arr = obj.get("scores") if isinstance(obj, dict) else obj
            if not isinstance(arr, list):
                return
            out: List[float] = []
            for x in arr:
                try:
                    fx = float(x)
                except Exception:
                    continue
                if not np.isfinite(fx):
                    continue
                out.append(fx)
            if out:
                self.prior_scores = out
                self._cap_prior_scores()
        except Exception as e:
            self._alert("prior_scores_load_error", "Failed to load prior_scores", path=str(path), error=str(e))

    def _save_prior_scores(self) -> None:
        if self.prior_scores_path is None:
            return
        try:
            self._cap_prior_scores()
            self.prior_scores_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "n_scores": int(len(self.prior_scores)),
                "scores": list(self.prior_scores),
            }
            tmp = self.prior_scores_path.with_suffix(self.prior_scores_path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            tmp.replace(self.prior_scores_path)
        except Exception as e:
            self._alert("prior_scores_save_error", "Failed to save prior_scores", path=str(self.prior_scores_path), error=str(e))

    def _cleanup_out_dir(self) -> None:
        days = int(self.retention_days)
        if days <= 0:
            return
        cutoff = time.time() - float(days) * 86400.0
        pats = [
            "minute_bars_*.csv",
            "trades_*.csv",
            "daily_*.csv",
            "audit_*.jsonl",
            "bank_trades_*.jsonl",
            "bank_proof_*.jsonl",
        ]
        for pat in pats:
            for p in self.out_dir.glob(pat):
                try:
                    if not p.is_file():
                        continue
                    if float(p.stat().st_mtime) < cutoff:
                        p.unlink(missing_ok=True)
                except Exception:
                    continue

    def _bank_write(self, kind: str, payload: Mapping[str, Any]) -> None:
        if self.bank_log is None:
            return
        try:
            self.bank_log.write_event(str(kind), dict(payload))
        except Exception as e:
            self._alert("bank_log_write_error", "Bank-proof log write failed", error=str(e), kind=str(kind))

    def _bank_snapshot(self, stage: str, extra: Optional[Mapping[str, Any]] = None) -> None:
        if self.bank_log is None:
            return

        payload: Dict[str, Any] = {
            "stage": str(stage),
            "symbol": str(self.market),
            "trade_mode": str(self.trade_mode),
            "address": str(self.cfg.address),
            "subaccounts": {"trading": int(self.cfg.subaccount_trading), "bank": int(self.cfg.subaccount_bank)},
        }

        if str(self.trade_mode).lower() == "real":
            try:
                trading_eq = float(
                    self.loop.run_until_complete(
                        _get_subaccount_equity_usdc(
                            self.clients,
                            address=self.cfg.address,
                            subaccount_number=int(self.cfg.subaccount_trading),
                        )
                    )
                )
            except Exception:
                trading_eq = float("nan")

            try:
                bank_eq = float(
                    self.loop.run_until_complete(
                        _get_subaccount_equity_usdc(
                            self.clients,
                            address=self.cfg.address,
                            subaccount_number=int(self.cfg.subaccount_bank),
                        )
                    )
                )
            except Exception:
                bank_eq = float("nan")

            pos_sz = self._safe_get_live_position_size_btc()

            payload.update(
                {
                    "trading_equity_usdc": _finite_or_none(trading_eq),
                    "bank_equity_usdc": _finite_or_none(bank_eq),
                    "open_position_size_btc": _finite_or_none(pos_sz),
                }
            )

        if extra:
            payload.update(dict(extra))

        self._bank_write("bank_balance_snapshot", payload)

    def _safe_get_live_position_size_btc(self) -> Optional[float]:
        if str(self.trade_mode).lower() != "real":
            return 0.0
        try:
            return float(
                self.loop.run_until_complete(
                    _get_open_position_size_btc(
                        self.clients,
                        address=self.cfg.address,
                        subaccount_number=int(self.cfg.subaccount_trading),
                        market=self.market,
                    )
                )
            )
        except Exception as e:
            self._alert("pos_size_query_error", "Failed to query open position size", error=str(e))
            return None

    def reconcile_startup(self) -> None:
        """On startup in real mode, flatten any leftover position so we never 'adopt' unknown state."""
        if str(self.trade_mode).lower() != "real":
            return

        self._bank_snapshot("startup_begin")

        sz0 = self._safe_get_live_position_size_btc()
        if sz0 is None:
            return
        if abs(float(sz0)) <= 0.0:
            self._bank_snapshot("startup_no_position")
            return

        self._alert("startup_orphan_position", "Open position detected on startup; flattening", size_btc=float(sz0))
        self._bank_write(
            "bank_startup_orphan_position",
            {
                "symbol": str(self.market),
                "trade_mode": str(self.trade_mode),
                "size_btc": _finite_or_none(sz0),
                "address": str(self.cfg.address),
                "subaccounts": {"trading": int(self.cfg.subaccount_trading), "bank": int(self.cfg.subaccount_bank)},
            },
        )

        try:
            close_meta = self.loop.run_until_complete(self._close_real_long(size_btc=float(abs(float(sz0)))))
            txhash = _txhash_hex((((close_meta.get("tx") or {}).get("txResponse") or {}).get("txhash"))) if isinstance(close_meta, dict) else None
            self._bank_write(
                "bank_startup_flatten_attempt",
                {
                    "symbol": str(self.market),
                    "trade_mode": str(self.trade_mode),
                    "size_btc": _finite_or_none(sz0),
                    "exit": close_meta if isinstance(close_meta, dict) else None,
                    "txhash": txhash,
                },
            )
        except Exception as e:
            self._bank_write("bank_startup_flatten_failed", {"error": str(e)})
            raise RuntimeError(f"Failed to flatten startup position: {e}") from e

        # Wait briefly for indexer to reflect flat position.
        deadline = time.time() + 30.0
        while time.time() < deadline:
            time.sleep(0.75)
            sz = self._safe_get_live_position_size_btc()
            if sz is None:
                continue
            if abs(float(sz)) <= 0.0:
                self._alert("startup_flatten_ok", "Startup flatten succeeded")
                self._bank_snapshot("startup_flatten_ok")
                return

        self._bank_write("bank_startup_flatten_timeout", {"error": "timeout"})
        raise RuntimeError("Startup flatten timed out; position still appears open")

    def _entry_threshold_for_day(self, d):
        if d in self.thr_map:
            return self.thr_map[d]

        # Match offline/backtest gating: require a minimum sample size before thresholds become active.
        if not self.prior_scores or int(len(self.prior_scores)) < int(getattr(self, "min_prior_scores", 0) or 0):
            self.thr_map[d] = float("inf")
            return self.thr_map[d]

        q = float(np.quantile(np.array(self.prior_scores, dtype=np.float64), 1.0 - self.target_frac))
        self.thr_map[d] = q
        return q

    def _ensure_feats(self) -> None:
        # For the v2 pattern entry model we build the same derived frame used in training.
        # Also keep the underlying base feature frame for exit-gap model features.
        try:
            pat, src = build_entry_pattern_frame_v2(self.bars, base_features=self.models.entry_base_features)
        except Exception as e:
            self.feats = None
            self.exit_src = None
            self._missing_cols = list(self.models.entry_features)
            self._missing_entry_cols = list(self.models.entry_features)
            self._missing_exit_cols = []
            self._alert("feature_build_error", "Failed to build pattern entry features", error=str(e))
            return

        self.feats = pat
        self.exit_src = src

        i = len(self.bars) - 1
        # Training required >= 1 day warmup for z1d_* indicators.
        if int(i) < 1440:
            self._missing_cols = list(self.models.entry_features)
            self._missing_entry_cols = list(self.models.entry_features)
            self._missing_exit_cols = []
            return

        row = self.feats.loc[i, self.models.entry_features]
        vals = pd.to_numeric(row, errors="coerce").to_numpy(dtype=np.float64)
        bad = ~np.isfinite(vals)
        missing = [c for c, b in zip(self.models.entry_features, bad) if bool(b)]

        self._missing_cols = list(missing)
        self._missing_entry_cols = list(missing)
        self._missing_exit_cols = []

    def _persist_bar(self, row: Mapping[str, Any]) -> None:
        d = pd.to_datetime(row["timestamp"], utc=True).date()
        p = self.out_dir / f"minute_bars_{d}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        header = not p.exists()
        pd.DataFrame([row]).to_csv(p, index=False, header=header, mode="a")

    def _audit_market_bar_1m_closed(self, bar_row: Mapping[str, Any]) -> None:
        if self.audit is None:
            return
        t0 = pd.to_datetime(bar_row["timestamp"], utc=True)
        t1 = t0 + timedelta(minutes=1)
        self.audit.write(
            {
                "kind": KIND_MARKET_BAR_1M_CLOSED,
                "symbol": self.market,
                "source": "dydx_indexer_ws",
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

    def _audit_entry_decision(self, *, bar_open_ts: Any, score: float, threshold: float, planned_entry_ts: Any, feats_row: pd.Series) -> None:
        if self.audit is None:
            return
        t0 = pd.to_datetime(bar_open_ts, utc=True)
        t1 = t0 + timedelta(minutes=1)

        feature_names = list(self.models.entry_features)
        feature_values = [_finite_or_none(feats_row.get(c)) for c in feature_names]

        self.audit.write(
            {
                "kind": KIND_ENTRY_DECISION,
                "symbol": self.market,
                "decision_time_utc": t1.to_pydatetime().isoformat(),
                "bar_open_time_utc": t0.to_pydatetime().isoformat(),
                "bar_close_time_utc": t1.to_pydatetime().isoformat(),
                "score": float(score),
                "threshold": float(threshold),
                "action": "enter",
                "planned_entry_time_utc": _iso_utc(planned_entry_ts),
                "policy": "threshold_online_causal_next_open",
                "forced": False,
                "feature_names": feature_names,
                "feature_values": feature_values,
                "model": {
                    "role": "entry",
                    "artifact": str(self.models.entry_artifact),
                    "created_utc": str(self.models.entry_created_utc or "unknown"),
                    "features": feature_names,
                },
                "exchange": "dydx_v4",
                "subaccounts": {
                    "trading": int(self.cfg.subaccount_trading),
                    "bank": int(self.cfg.subaccount_bank),
                },
            }
        )

    def _audit_trade_closed(
        self,
        *,
        paper: bool,
        entry_time_utc: str,
        exit_time_utc: str,
        entry_price: float,
        exit_price: float,
        exit_rel_min: int,
        realized_ret_pct: float,
        predicted_ret_pct: Optional[float],
        entry_score: Optional[float],
        entry_threshold: Optional[float],
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self.audit is None:
            return
        payload: Dict[str, Any] = {
            "kind": KIND_TRADE_CLOSED,
            "symbol": self.market,
            "paper": bool(paper),
            "entry_time_utc": str(entry_time_utc),
            "exit_time_utc": str(exit_time_utc),
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "fee_side": float(self.cfg.fee_side),
            "exit_rel_min": int(exit_rel_min),
            "realized_ret_pct": float(realized_ret_pct),
            "predicted_ret_pct": predicted_ret_pct,
            "entry_score": entry_score,
            "entry_threshold": entry_threshold,
            "exchange": "dydx_v4",
        }
        if extra:
            payload.update(dict(extra))
        self.audit.write(payload)

    def _score_entry_index(self, i: int) -> float:
        assert self.feats is not None
        x = self.feats.loc[i, self.models.entry_features].to_numpy(dtype=np.float32)[None, :]
        return float(self.models.entry_model.predict(x)[0])

    # ---- Exit-gap regressor feature engineering (must match oracle-exit dataset builder) ----

    _QE_MODELS: Dict[int, Dict[str, Any]] = {
        1: {
            "feature_cols": [
                "px_range_norm1m__t0",
                "px_range_norm1m__t1",
                "range_norm_5m__t1",
                "range_norm_5m__t2",
                "range_norm_5m__t0",
                "macd__t1",
                "macd__t2",
                "px_range_norm1m__t2",
                "macd__t0",
                "vol_std_5m__t2",
                "vol_std_5m__t1",
                "vol_std_5m__t0",
            ],
            "mean": [
                0.0017311684157958627,
                0.001799022583623574,
                0.005201049404186999,
                0.005219247844162671,
                0.00508836013540689,
                -14.8716822745053,
                -12.901003073577206,
                0.0022877535369816705,
                -15.30467478596776,
                97.63208412551606,
                98.8282850808809,
                94.23914311794302,
            ],
            "std": [
                0.002151103248995944,
                0.0019385443943925737,
                0.004336551503240184,
                0.004265534481971351,
                0.004291754603747401,
                136.32319643938536,
                133.23070322665765,
                0.0023559100334701955,
                138.63555088715978,
                81.06773524775568,
                82.59282483228851,
                87.7748030360668,
            ],
            "coef": [
                0.49821968896154567,
                0.3006536977305254,
                -0.03390883350241022,
                0.5491694390310403,
                -0.45858960448213576,
                -2.6440024697146813,
                1.2525218690586208,
                -0.131773273835023,
                1.2307795942830244,
                0.10173026078095815,
                0.18957380910723812,
                -0.24169722702957883,
            ],
            "intercept": -0.1262394305926545,
        },
        2: {
            "feature_cols": [
                "px_range_norm1m__t0",
                "px_range_norm1m__t1",
                "px_range_norm1m__t2",
                "range_norm_5m__t0",
                "range_norm_5m__t2",
                "range_norm_5m__t1",
                "vol_std_5m__t0",
                "macd__t2",
                "macd__t0",
                "macd__t1",
                "vol_std_5m__t2",
                "ind_neg",
            ],
            "mean": [
                0.0016825148751894745,
                0.0017311684157958627,
                0.001799022583623574,
                0.004911417996101815,
                0.005201049404186999,
                0.00508836013540689,
                85.41860345222413,
                -14.8716822745053,
                -14.54825509941764,
                -15.30467478596776,
                98.8282850808809,
                0.8437762536470834,
            ],
            "std": [
                0.002123881519424586,
                0.002151103248995944,
                0.0019385443943925737,
                0.004064453432659976,
                0.004336551503240184,
                0.004291754603747401,
                80.48006440317958,
                136.32319643938536,
                140.7627165829248,
                138.63555088715978,
                82.59282483228851,
                0.19991236876880172,
            ],
            "coef": [
                0.6610773181013107,
                0.5090862733850794,
                0.0669961618758457,
                -0.20776535018127928,
                0.5524467593663873,
                -0.5082489258371796,
                -0.04332494043509365,
                -1.6320144354352097,
                -1.850280060153793,
                3.2809397222096,
                0.08757782979542733,
                -0.07500164742692321,
            ],
            "intercept": -0.15862642859705967,
        },
        3: {
            "feature_cols": [
                "px_range_norm1m__t0",
                "px_range_norm1m__t1",
                "px_range_norm1m__t2",
                "drawdown_from_peak_pct",
                "range_norm_5m__t0",
                "vol_std_5m__t0",
                "range_norm_5m__t1",
                "range_norm_5m__t2",
                "delta_mark_change_2m",
                "vol_std_5m__t1",
                "ind_neg",
                "macd__t0",
            ],
            "mean": [
                0.0016334127545820165,
                0.0016825148751894745,
                0.0017311684157958627,
                -0.07294866189946678,
                0.004648579939007556,
                76.93326570808777,
                0.004911417996101815,
                0.00508836013540689,
                0.025025398578561613,
                85.41860345222413,
                0.8400210779410251,
                -13.219567950278769,
            ],
            "std": [
                0.001807875565210792,
                0.002123881519424586,
                0.002151103248995944,
                0.1287983162431893,
                0.0038768760588339408,
                73.64706830374843,
                0.004064453432659976,
                0.004291754603747401,
                0.21642660549499265,
                80.48006440317958,
                0.20381119533910882,
                142.51396637646718,
            ],
            "coef": [
                0.3531046695830277,
                0.28924432342328393,
                0.2985978304067285,
                -0.6468089689196372,
                -0.4753901879792831,
                0.298650559259968,
                0.13419032645676224,
                0.14406811962902843,
                0.1220210049887942,
                -0.2589707005521335,
                -0.08416181521140421,
                -0.1603943555151908,
            ],
            "intercept": -0.2709326807596019,
        },
    }

    @staticmethod
    def _sigmoid_scalar(x: float) -> float:
        x = float(np.clip(float(x), -20.0, 20.0))
        return float(1.0 / (1.0 + np.exp(-x)))

    @staticmethod
    def _zscore_roll_series(s: pd.Series, win: int) -> pd.Series:
        mu = s.rolling(int(win), min_periods=int(win)).mean()
        sd = s.rolling(int(win), min_periods=int(win)).std()
        return (s - mu) / (sd + 1e-12)

    def _compute_ind_pos_neg_at(self, idx: int) -> Tuple[float, float]:
        if self.exit_src is None or self.bars is None or len(self.bars) == 0:
            return float("nan"), float("nan")

        close = pd.to_numeric(self.bars["close"], errors="coerce")
        high = pd.to_numeric(self.bars["high"], errors="coerce")
        low = pd.to_numeric(self.bars["low"], errors="coerce")

        px_ret1m = close.pct_change() * 100.0
        px_range1m = (high - low) / (close + 1e-12)

        pre_ret_15m = close.pct_change(15) * 100.0
        pre_range_15m = (high.rolling(15, min_periods=15).max() / (low.rolling(15, min_periods=15).min() + 1e-12)) - 1.0
        pre_vol_15m = px_ret1m.rolling(15, min_periods=15).std()
        pre_absret_max_15m = px_ret1m.abs().rolling(15, min_periods=15).max()

        vol5 = px_ret1m.rolling(5, min_periods=5).std()

        z1d_vol5 = self._zscore_roll_series(vol5, 1440)
        z1d_ret1m_abs = self._zscore_roll_series(px_ret1m, 1440).abs()
        z1d_range1m_abs = self._zscore_roll_series(px_range1m, 1440).abs()

        dip1m = (-px_ret1m).clip(lower=0.0)
        z_dip1m = self._zscore_roll_series(dip1m, 1440)
        z_pre_vol_15m = self._zscore_roll_series(pre_vol_15m, 1440)
        z_pre_range_15m = self._zscore_roll_series(pre_range_15m, 1440)
        z_pre_absret_max_15m = self._zscore_roll_series(pre_absret_max_15m, 1440)
        down15 = (-pre_ret_15m).clip(lower=0.0)
        z_down15 = self._zscore_roll_series(down15, 1440)

        mom5 = pd.to_numeric(self.exit_src.get("mom_5m_pct"), errors="coerce") if "mom_5m_pct" in self.exit_src.columns else pd.Series([np.nan] * len(self.exit_src))
        z_mom5 = self._zscore_roll_series(mom5, 1440)

        try:
            pos_raw = (
                0.80 * float(z_mom5.iloc[idx])
                + 0.60 * float(z_dip1m.iloc[idx])
                + 0.40 * float(z_pre_vol_15m.iloc[idx])
                + 0.20 * float(z_pre_range_15m.iloc[idx])
                - 0.80 * float(z1d_vol5.iloc[idx])
                - 0.60 * float(z1d_range1m_abs.iloc[idx])
            )
            neg_raw = (
                0.90 * float(z1d_vol5.iloc[idx])
                + 0.80 * float(z1d_range1m_abs.iloc[idx])
                + 0.60 * float(z1d_ret1m_abs.iloc[idx])
                + 0.40 * float(z_pre_range_15m.iloc[idx])
                + 0.40 * float(z_pre_absret_max_15m.iloc[idx])
                + 0.40 * float(z_down15.iloc[idx])
            )
        except Exception:
            return float("nan"), float("nan")

        ind_pos = self._sigmoid_scalar(pos_raw)
        ind_neg = self._sigmoid_scalar(neg_raw)
        return float(ind_pos), float(ind_neg)

    def _qe_feat_value(
        self,
        feat_name: str,
        *,
        decision_i: int,
        cur_ret: float,
        r_prev1: float,
        r_prev2: float,
        peak_ret: float,
        ind_neg: float,
    ) -> float:
        # lagged feature name form: "base__t{lag}"
        if "__t" in feat_name:
            base, suf = feat_name.rsplit("__", 1)
            try:
                lag = int(suf[1:])
            except Exception:
                lag = 0
            j = int(decision_i) - int(lag)
            if j < 0 or j >= len(self.bars):
                return float("nan")

            if base == "px_range_norm1m":
                c = float(self.bars.iloc[j]["close"])
                h = float(self.bars.iloc[j]["high"])
                l = float(self.bars.iloc[j]["low"])
                return float((h - l) / (c + 1e-12))

            if self.exit_src is None:
                return float("nan")

            arr = self.exit_src.get(str(base))
            if arr is None:
                return float("nan")
            try:
                return float(pd.to_numeric(arr, errors="coerce").to_numpy(np.float64)[j])
            except Exception:
                return float("nan")

        if feat_name == "drawdown_from_peak_pct":
            return float(cur_ret - peak_ret)
        if feat_name == "delta_mark_change_2m":
            return float(cur_ret - r_prev2) if np.isfinite(r_prev2) else float("nan")
        if feat_name == "delta_mark_change_1m":
            return float(cur_ret - r_prev1) if np.isfinite(r_prev1) else float("nan")
        if feat_name == "ind_neg":
            return float(ind_neg)

        return float("nan")

    def _compute_ind_quick_exit_3m(self, *, mins_in_trade: int, decision_i: int, cur_ret: float, r_prev1: float, r_prev2: float, peak_ret: float, ind_neg: float) -> float:
        k = int(mins_in_trade)
        if k not in (1, 2, 3):
            return float("nan")
        m = self._QE_MODELS.get(k)
        if m is None:
            return float("nan")

        feats = list(m["feature_cols"])
        mu = np.asarray(m["mean"], dtype=np.float64)
        sd = np.asarray(m["std"], dtype=np.float64)
        coef = np.asarray(m["coef"], dtype=np.float64)
        intercept = float(m["intercept"])

        x = np.asarray(
            [
                self._qe_feat_value(
                    str(fn),
                    decision_i=int(decision_i),
                    cur_ret=float(cur_ret),
                    r_prev1=float(r_prev1),
                    r_prev2=float(r_prev2),
                    peak_ret=float(peak_ret),
                    ind_neg=float(ind_neg),
                )
                for fn in feats
            ],
            dtype=np.float64,
        )
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        xz = (x - mu) / (sd + 1e-12)
        logit = intercept + float(np.dot(coef, xz))
        return float(self._sigmoid_scalar(logit))

    def _exit_gap_features_for_decision(
        self,
        *,
        decision_i: int,
        mins_in_trade: int,
        entry_px: float,
        cur_ret: float,
        r_prev1: float,
        r_prev2: float,
        r_prev3: float,
        peak_ret: float,
    ) -> np.ndarray:
        # Build values in the exact order required by the model artifact.
        if self.exit_gap is None:
            return np.zeros((0,), dtype=np.float32)

        hold_min = int(self.hold_min)

        # price-derived arrays for lags
        close = pd.to_numeric(self.bars["close"], errors="coerce")
        high = pd.to_numeric(self.bars["high"], errors="coerce")
        low = pd.to_numeric(self.bars["low"], errors="coerce")
        px_ret1m = (close.pct_change() * 100.0).to_numpy(np.float64)
        px_range1m = ((high - low) / (close + 1e-12)).to_numpy(np.float64)

        ind_pos, ind_neg = self._compute_ind_pos_neg_at(int(decision_i))
        ind_qe = self._compute_ind_quick_exit_3m(
            mins_in_trade=int(mins_in_trade),
            decision_i=int(decision_i),
            cur_ret=float(cur_ret),
            r_prev1=float(r_prev1),
            r_prev2=float(r_prev2),
            peak_ret=float(peak_ret),
            ind_neg=float(ind_neg),
        )

        def _get_base_feat(base: str, j: int) -> float:
            if self.exit_src is None:
                return float("nan")
            if str(base) not in self.exit_src.columns:
                return float("nan")
            try:
                arr = pd.to_numeric(self.exit_src[str(base)], errors="coerce").to_numpy(np.float64)
                return float(arr[j])
            except Exception:
                return float("nan")

        out: List[float] = []
        for c in self.exit_gap.feature_cols:
            if c == "mins_in_trade":
                out.append(float(mins_in_trade))
                continue
            if c == "mins_remaining":
                out.append(float(max(0, hold_min - int(mins_in_trade))))
                continue
            if c == "delta_mark_pct":
                out.append(float(cur_ret))
                continue
            if c == "delta_mark_prev1_pct":
                out.append(float(r_prev1))
                continue
            if c == "delta_mark_prev2_pct":
                out.append(float(r_prev2))
                continue
            if c == "delta_mark_change_1m":
                out.append(float(cur_ret - r_prev1) if np.isfinite(r_prev1) else float("nan"))
                continue
            if c == "delta_mark_change_2m":
                out.append(float(cur_ret - r_prev2) if np.isfinite(r_prev2) else float("nan"))
                continue
            if c == "delta_mark_change_3m":
                out.append(float(cur_ret - r_prev3) if np.isfinite(r_prev3) else float("nan"))
                continue
            if c == "drawdown_from_peak_pct":
                out.append(float(cur_ret - peak_ret))
                continue

            if c.startswith("px_ret1m_close__t"):
                try:
                    lag = int(c.split("__t", 1)[1])
                except Exception:
                    lag = 0
                j = int(decision_i) - int(lag)
                out.append(float(px_ret1m[j]) if j >= 0 else float("nan"))
                continue
            if c.startswith("px_range_norm1m__t"):
                try:
                    lag = int(c.split("__t", 1)[1])
                except Exception:
                    lag = 0
                j = int(decision_i) - int(lag)
                out.append(float(px_range1m[j]) if j >= 0 else float("nan"))
                continue

            if c == "ind_pos":
                out.append(float(ind_pos))
                continue
            if c == "ind_neg":
                out.append(float(ind_neg))
                continue
            if c == "ind_quick_exit_3m":
                out.append(float(ind_qe))
                continue

            # base feature lags: e.g. macd__t0
            if "__t" in c:
                base, suf = c.rsplit("__", 1)
                try:
                    lag = int(suf[1:])
                except Exception:
                    lag = 0
                j = int(decision_i) - int(lag)
                if j < 0:
                    out.append(float("nan"))
                else:
                    out.append(float(_get_base_feat(str(base), int(j))))
                continue

            # fallback: unknown feature
            out.append(float("nan"))

        return np.asarray(out, dtype=np.float32)

    def _predict_exit_index(self, i: int) -> float:
        # Legacy exit model not used in this strategy.
        raise RuntimeError("Legacy exit model disabled")

    async def _open_real_long(self, *, equity_before: float) -> Dict[str, Any]:
        mk = await _fetch_market_meta(self.clients, self.market)
        imf = float(mk.get("initialMarginFraction") or 1.0)
        max_lev_imf = int(_imf_to_max_leverage(imf))
        user_cap = int(getattr(self.cfg, "max_leverage", 0) or 0)
        max_lev = int(max_lev_imf) if user_cap <= 0 else min(int(user_cap), int(max_lev_imf))

        px = float(mk.get("oraclePrice") or await _fetch_oracle_price(self.clients, self.market))
        if px <= 0:
            raise RuntimeError("oraclePrice not positive")

        # For IOC "market" orders, the provided price acts like a limit. Use a small
        # slippage buffer so buys cross the ask and sells cross the bid.
        try:
            slip = float(os.getenv("DYDX_ORDER_SLIPPAGE_FRAC", "0.002"))
        except Exception:
            slip = 0.002
        slip = max(0.0, min(0.05, slip))
        limit_px = float(px) * (1.0 + slip)

        # Use (almost) all trading equity as margin.
        margin_usdc = float(equity_before) * float(self.cfg.use_margin_frac)
        margin_usdc = max(0.0, float(margin_usdc))

        # Max notional from initial margin constraint.
        max_notional_imf = float(margin_usdc) / max(1e-12, float(imf))

        # Optional user leverage cap (only applies when explicitly configured).
        max_notional_user = float("inf")
        if int(user_cap) > 0:
            max_notional_user = float(margin_usdc) * float(max_lev)

        notional_usd = float(min(max_notional_imf, max_notional_user)) * float(self.cfg.leverage_safety_frac)

        if notional_usd <= 0:
            raise RuntimeError("Computed notional_usd <= 0")

        size_btc = float(notional_usd) / float(px)

        mkt = Market(mk)
        coid = int(new_client_id())
        oid = mkt.order_id(
            address=self.cfg.address,
            subaccount_number=int(self.cfg.subaccount_trading),
            client_id=int(coid),
            order_flags=int(OrderFlags.SHORT_TERM),
        )

        # Compute quantized size the same way the order builder does (so our stored size matches step sizes).
        quantums = int(mkt.calculate_quantums(float(size_btc)))
        size_btc_q = float(quantums) * (10 ** float(mk["atomicResolution"]))

        # SHORT_TERM orders require a non-zero good_til_block.
        good_til_block = int(await self.clients.node.latest_block_height()) + 20

        order = mkt.order(
            order_id=oid,
            order_type=OrderType.MARKET,
            side=Order.SIDE_BUY,
            size=float(size_btc_q),
            price=float(limit_px),
            time_in_force=Order.TIME_IN_FORCE_UNSPECIFIED,
            reduce_only=False,
            post_only=False,
            good_til_block=int(good_til_block),
        )

        sent_at = datetime.now(timezone.utc)
        tx = await self.clients.node.place_order(self.clients.wallet, order)

        fill = await _wait_for_fills(
            self.clients,
            address=self.cfg.address,
            subaccount_number=int(self.cfg.subaccount_trading),
            market=self.market,
            side="BUY",
            since_dt=sent_at - timedelta(seconds=5),
            min_size_btc=float(size_btc_q),
            timeout_s=10.0,
            poll_s=0.5,
        )

        fill_assumed = False
        filled_sz = float(fill.get("size_btc") or 0.0)
        filled_px = float(fill.get("vwap_price") or 0.0)
        fee_usdc = float(fill.get("fee_usdc") or 0.0)

        # Fallbacks for indexer lag: wait longer, then fall back to live position size.
        if filled_sz <= 0.0 or filled_px <= 0.0:
            fill2 = await _wait_for_fills(
                self.clients,
                address=self.cfg.address,
                subaccount_number=int(self.cfg.subaccount_trading),
                market=self.market,
                side="BUY",
                since_dt=sent_at - timedelta(seconds=5),
                min_size_btc=float(1e-12),
                timeout_s=30.0,
                poll_s=1.0,
            )
            if float(fill2.get("size_btc") or 0.0) > 0.0 and float(fill2.get("vwap_price") or 0.0) > 0.0:
                fill = fill2
                filled_sz = float(fill.get("size_btc") or 0.0)
                filled_px = float(fill.get("vwap_price") or 0.0)
                fee_usdc = float(fill.get("fee_usdc") or 0.0)

        if filled_sz <= 0.0 or filled_px <= 0.0:
            try:
                pos_sz = float(
                    await _get_open_position_size_btc(
                        self.clients,
                        address=self.cfg.address,
                        subaccount_number=int(self.cfg.subaccount_trading),
                        market=self.market,
                    )
                )
            except Exception:
                pos_sz = 0.0

            if abs(float(pos_sz)) > 0.0:
                fill_assumed = True
                filled_sz = abs(float(pos_sz))
                filled_px = float(px)
                fee_usdc = float(fee_usdc)  # unknown if indexer hasn't reported yet

        return {
            "client_id": int(coid),
            "tx": _pb_to_jsonable(tx),
            "order": _pb_to_jsonable(order),
            "oracle_price": float(px),
            "limit_price": float(limit_px),
            "slippage_frac": float(slip),
            "order_sent_time_utc": sent_at.isoformat().replace("+00:00", "Z"),
            "fill": fill,
            "fill_time_utc": fill.get("first_fill_time_utc"),
            "fill_price": float(filled_px),
            "fill_size_btc": float(filled_sz),
            "fee_usdc": float(fee_usdc),
            "fill_assumed": bool(fill_assumed),
            "imf": float(imf),
            "max_lev": int(max_lev),
            "margin_usdc": float(margin_usdc),
            "notional_usd": float(notional_usd),
            "size_btc": float(size_btc_q),
            "size_btc_quantums": int(quantums),
        }

    async def _close_real_long(self, *, size_btc: float) -> Dict[str, Any]:
        mk = await _fetch_market_meta(self.clients, self.market)
        px = float(mk.get("oraclePrice") or await _fetch_oracle_price(self.clients, self.market))

        try:
            slip = float(os.getenv("DYDX_ORDER_SLIPPAGE_FRAC", "0.002"))
        except Exception:
            slip = 0.002
        slip = max(0.0, min(0.05, slip))
        limit_px = float(px) * (1.0 - slip)

        mkt = Market(mk)

        coid = int(new_client_id())
        oid = mkt.order_id(
            address=self.cfg.address,
            subaccount_number=int(self.cfg.subaccount_trading),
            client_id=int(coid),
            order_flags=int(OrderFlags.SHORT_TERM),
        )

        # SHORT_TERM orders require a non-zero good_til_block.
        good_til_block = int(await self.clients.node.latest_block_height()) + 20

        order = mkt.order(
            order_id=oid,
            order_type=OrderType.MARKET,
            side=Order.SIDE_SELL,
            size=float(size_btc),
            price=float(limit_px),
            time_in_force=Order.TIME_IN_FORCE_UNSPECIFIED,
            reduce_only=True,
            post_only=False,
            good_til_block=int(good_til_block),
        )

        sent_at = datetime.now(timezone.utc)
        tx = await self.clients.node.place_order(self.clients.wallet, order)

        fill = await _wait_for_fills(
            self.clients,
            address=self.cfg.address,
            subaccount_number=int(self.cfg.subaccount_trading),
            market=self.market,
            side="SELL",
            since_dt=sent_at - timedelta(seconds=5),
            min_size_btc=float(size_btc),
            timeout_s=10.0,
            poll_s=0.5,
        )

        fill_assumed = False
        filled_sz = float(fill.get("size_btc") or 0.0)
        filled_px = float(fill.get("vwap_price") or 0.0)
        fee_usdc = float(fill.get("fee_usdc") or 0.0)

        if filled_sz <= 0.0 or filled_px <= 0.0:
            fill2 = await _wait_for_fills(
                self.clients,
                address=self.cfg.address,
                subaccount_number=int(self.cfg.subaccount_trading),
                market=self.market,
                side="SELL",
                since_dt=sent_at - timedelta(seconds=5),
                min_size_btc=float(1e-12),
                timeout_s=30.0,
                poll_s=1.0,
            )
            if float(fill2.get("size_btc") or 0.0) > 0.0 and float(fill2.get("vwap_price") or 0.0) > 0.0:
                fill = fill2
                filled_sz = float(fill.get("size_btc") or 0.0)
                filled_px = float(fill.get("vwap_price") or 0.0)
                fee_usdc = float(fill.get("fee_usdc") or 0.0)

        if filled_sz <= 0.0 or filled_px <= 0.0:
            # If the position is already flat, assume the close succeeded even if fills lag.
            try:
                pos_sz = float(
                    await _get_open_position_size_btc(
                        self.clients,
                        address=self.cfg.address,
                        subaccount_number=int(self.cfg.subaccount_trading),
                        market=self.market,
                    )
                )
            except Exception:
                pos_sz = float("nan")

            if np.isfinite(pos_sz) and abs(float(pos_sz)) <= 0.0:
                fill_assumed = True
                filled_sz = float(size_btc)
                filled_px = float(px)
                # fee_usdc unknown until indexer reports

        return {
            "client_id": int(coid),
            "order": _pb_to_jsonable(order),
            "tx": _pb_to_jsonable(tx),
            "oracle_price": float(px),
            "limit_price": float(limit_px),
            "slippage_frac": float(slip),
            "order_sent_time_utc": sent_at.isoformat().replace("+00:00", "Z"),
            "fill": fill,
            "fill_time_utc": fill.get("last_fill_time_utc"),
            "fill_price": float(filled_px),
            "fill_size_btc": float(filled_sz),
            "fee_usdc": float(fee_usdc),
            "fill_assumed": bool(fill_assumed),
        }

    def _rest_fetch_1m_bars(self, *, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> List[Dict[str, Any]]:
        """Fetch CLOSED 1m bars from indexer REST (best-effort), returns list sorted by timestamp ascending."""
        s = pd.to_datetime(start_utc, utc=True)
        e = pd.to_datetime(end_utc, utc=True)
        if e < s:
            return []

        try:
            resp = self.loop.run_until_complete(
                self.clients.indexer.markets.get_perpetual_market_candles(
                    market=str(self.market),
                    resolution="1MIN",
                    from_iso=s.to_pydatetime().isoformat().replace("+00:00", "Z"),
                    to_iso=e.to_pydatetime().isoformat().replace("+00:00", "Z"),
                    limit=1000,
                )
            )
        except Exception as ex:
            self._alert("rest_backfill_failed", "REST backfill failed", error=str(ex))
            return []

        candles = resp.get("candles") if isinstance(resp, dict) else None
        if not isinstance(candles, list) or not candles:
            return []

        df = pd.DataFrame(candles)
        if df.empty:
            return []

        df["timestamp"] = pd.to_datetime(df["startedAt"], utc=True)
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["baseTokenVolume"].astype(float)

        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df = df[(df["timestamp"] >= s) & (df["timestamp"] <= e)].copy()

        return df.to_dict("records")

    def on_closed_bar(self, bar: Mapping[str, Any], *, allow_trade_logic: bool = True) -> None:
        # Feed heartbeat (used by /health and stall watchdog).
        self.last_closed_wall = time.time()

        # Append bar
        row = {
            "timestamp": pd.to_datetime(bar["timestamp"], utc=True),
            "open": float(bar["open"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "close": float(bar["close"]),
            "volume": float(bar["volume"]),
        }

        if len(self.bars) > 0:
            prev_ts = pd.to_datetime(self.bars.iloc[-1]["timestamp"], utc=True)
            ts = pd.to_datetime(row["timestamp"], utc=True)
            if ts < prev_ts:
                return
            if ts == prev_ts:
                # replace
                self.bars.iloc[-1] = pd.Series(row)
                return

            # If we detect gaps, backfill missing bars via REST before proceeding.
            # This keeps feature computation consistent and avoids "missing minute" drift.
            dt = ts - prev_ts
            if dt > timedelta(minutes=1):
                gap_mins = int(dt.total_seconds() // 60) - 1
                self._alert(
                    "bar_gap",
                    "Detected missing 1m bars; attempting REST backfill",
                    prev_ts=str(prev_ts),
                    ts=str(ts),
                    gap_mins=int(gap_mins),
                )

                if self.bank_log is not None:
                    try:
                        self._bank_snapshot(
                            "bar_gap_detected",
                            {"prev_ts": str(prev_ts), "ts": str(ts), "gap_mins": int(gap_mins)},
                        )
                    except Exception:
                        pass

                # Backfill the strictly missing interval.
                bf_start = prev_ts + timedelta(minutes=1)
                bf_end = ts - timedelta(minutes=1)
                missing_rows = self._rest_fetch_1m_bars(start_utc=bf_start, end_utc=bf_end)
                for r in missing_rows:
                    try:
                        self.on_closed_bar(r, allow_trade_logic=False)
                    except Exception:
                        # best effort
                        pass

        self.bars = pd.concat([self.bars, pd.DataFrame([row])], ignore_index=True)
        self._persist_bar(row)
        self._audit_market_bar_1m_closed(row)

        # Day roll + scoring
        self._ensure_feats()

        i = len(self.bars) - 1
        ts = pd.to_datetime(self.bars.iloc[i]["timestamp"], utc=True)
        d = ts.date()

        if self.cur_day is None:
            self.cur_day = d

            # If we don't have persisted prior scores yet, seed them from the initial backfill so
            # the live threshold distribution matches the offline/backtest logic as closely as possible.
            if (not self.prior_scores) and self.feats is not None and len(self.feats) > 0 and not self._missing_entry_cols:
                try:
                    X_df = self.feats[self.models.entry_features]
                    good = np.isfinite(X_df.to_numpy(np.float64)).all(axis=1)
                    good &= (np.arange(len(X_df), dtype=np.int64) >= 1440)

                    if np.any(good):
                        idxs = np.where(good)[0]
                        scores = self.models.entry_model.predict(X_df.iloc[idxs].to_numpy(dtype=np.float32))
                        df_scores = pd.DataFrame({"timestamp": self.feats.loc[idxs, "timestamp"], "score": scores})
                        df_scores["timestamp"] = pd.to_datetime(df_scores["timestamp"], utc=True)
                        df_scores["date"] = pd.to_datetime(df_scores["timestamp"]).dt.date
                        df_scores = df_scores.sort_values("timestamp").reset_index(drop=True)

                        # Seed-day logic: fill prior_scores with the first N days in the backfill.
                        days = []
                        seen = set()
                        for x in df_scores["date"].tolist():
                            if x not in seen:
                                seen.add(x)
                                days.append(x)

                        seed_n = max(0, int(getattr(self, "seed_days", 0) or 0))

                        # Preferred: strict seed-days logic (matches offline scripts).
                        if seed_n > 0 and len(days) > seed_n:
                            seed_set = set(days[:seed_n])
                            prior0 = df_scores[df_scores["date"].isin(seed_set)]["score"].to_numpy(np.float64)
                            self.prior_scores.extend([float(x) for x in prior0 if np.isfinite(float(x))])

                            # Walk remaining days up to today to keep day boundaries causal.
                            for day in days[seed_n:]:
                                if day >= d:
                                    break
                                day_scores = df_scores[df_scores["date"] == day]["score"].to_numpy(np.float64)
                                if len(self.prior_scores) >= int(self.min_prior_scores):
                                    self.thr_map[day] = float(np.quantile(np.asarray(self.prior_scores, dtype=np.float64), 1.0 - self.target_frac))
                                self.prior_scores.extend([float(x) for x in day_scores if np.isfinite(float(x))])
                                self._cap_prior_scores()

                            # Seed today's already-seen scores into _scores_cur_day so tomorrow's threshold matches
                            # what it would have been if the bot ran from midnight.
                            todays = df_scores[df_scores["date"] == d]["score"].to_numpy(np.float64)
                            self._scores_cur_day.extend([float(x) for x in todays if np.isfinite(float(x))])

                        # Fallback: if the backfill window doesn't span enough days to satisfy seed_n,
                        # still seed from *prior-day* scores (no leakage) so we can trade today.
                        elif len(days) >= 1:
                            prior_any = df_scores[df_scores["date"] < d]["score"].to_numpy(np.float64)
                            self.prior_scores.extend([float(x) for x in prior_any if np.isfinite(float(x))])
                            todays = df_scores[df_scores["date"] == d]["score"].to_numpy(np.float64)
                            self._scores_cur_day.extend([float(x) for x in todays if np.isfinite(float(x))])

                except Exception:
                    pass

            # Persist today's threshold (if finite) and cap score history.
            self._cap_prior_scores()
            try:
                thr0 = float(self._entry_threshold_for_day(d))
                self._append_threshold(d, thr0)
            except Exception:
                pass

            # Persist seeded state so restarts preserve parity.
            try:
                self._save_prior_scores()
            except Exception:
                pass

        if d != self.cur_day:
            finished_day = self.cur_day

            # Move finished day scores to prior (to keep thresholds causal).
            if self._scores_cur_day:
                try:
                    self.prior_scores.extend([float(x) for x in self._scores_cur_day if np.isfinite(float(x))])
                except Exception:
                    self.prior_scores.extend([float(x) for x in self._scores_cur_day])
            self._scores_cur_day = []

            # Persist score history + optional retention cleanup.
            self._cap_prior_scores()
            self._save_prior_scores()
            try:
                self._cleanup_out_dir()
            except Exception:
                pass

            self.cur_day = d

            # Persist new day's threshold (if finite).
            try:
                thr_new = float(self._entry_threshold_for_day(d))
                self._append_threshold(d, thr_new)
            except Exception:
                pass

            # Bank-proof daily snapshot (start of new day).
            if self.bank_log is not None:
                try:
                    self._bank_snapshot("day_roll", {"finished_day": str(finished_day), "new_day": str(d)})
                except Exception:
                    pass

        # Real-mode reconciliation: ensure we never trade with an orphaned open position.
        if bool(allow_trade_logic) and str(self.trade_mode).lower() == "real":
            live_sz = self._safe_get_live_position_size_btc()
            if live_sz is not None:
                live_abs = abs(float(live_sz))

                # If we don't think we have an open trade but there is a live position, flatten it.
                if self.open_trade is None and live_abs > 0.0:
                    # Confirm once (avoid reacting to transient indexer weirdness).
                    time.sleep(0.25)
                    live_sz2 = self._safe_get_live_position_size_btc()
                    if live_sz2 is not None and abs(float(live_sz2)) > 0.0:
                        self._alert("orphan_position", "Live position open but no open_trade tracked; flattening", size_btc=float(live_sz2))

                        if self.bank_log is not None:
                            try:
                                self._bank_snapshot(
                                    "orphan_position_detected",
                                    {
                                        "size_btc": float(abs(float(live_sz2))),
                                        "note": "Live position open but no open_trade tracked; flattening.",
                                    },
                                )
                            except Exception:
                                pass

                        try:
                            close_meta = self.loop.run_until_complete(self._close_real_long(size_btc=float(abs(float(live_sz2)))))
                            if self.bank_log is not None:
                                try:
                                    txhash = _txhash_hex((((close_meta.get("tx") or {}).get("txResponse") or {}).get("txhash")))
                                    self.bank_log.write_event(
                                        "bank_orphan_position_flatten",
                                        {
                                            "symbol": self.market,
                                            "trade_mode": "real",
                                            "direction": "LONG",
                                            "size_btc": float(abs(float(live_sz2))),
                                            "txhash": txhash,
                                            "meta": close_meta,
                                        },
                                    )
                                    self._bank_snapshot("orphan_position_flattened", {"txhash": txhash})
                                except Exception:
                                    pass
                        except Exception as e:
                            self._alert("orphan_position_close_failed", "Failed to flatten orphan position", error=str(e))
                            if self.bank_log is not None:
                                try:
                                    self.bank_log.write_event(
                                        "bank_orphan_position_flatten_failed",
                                        {
                                            "symbol": self.market,
                                            "trade_mode": "real",
                                            "direction": "LONG",
                                            "size_btc": float(abs(float(live_sz2))),
                                            "error": str(e),
                                        },
                                    )
                                except Exception:
                                    pass
                        return

                # If we track an open real trade but the position appears flat, clear the state.
                if isinstance(self.open_trade, dict) and str(self.open_trade.get("trade_mode") or "").lower() == "real":
                    if live_abs <= 0.0:
                        self._open_pos_zero_streak += 1
                        if int(self._open_pos_zero_streak) >= 2:
                            self._alert(
                                "position_missing",
                                "Open trade tracked but live position is flat; clearing trade state",
                                streak=int(self._open_pos_zero_streak),
                            )

                            trade_id = str((self.open_trade or {}).get("trade_id") or "")

                            if self.bank_log is not None:
                                try:
                                    self.bank_log.write_event(
                                        "bank_position_missing_clear",
                                        {
                                            "symbol": self.market,
                                            "trade_mode": "real",
                                            "direction": "LONG",
                                            "trade_id": trade_id,
                                            "streak": int(self._open_pos_zero_streak),
                                            "note": "Open trade tracked but live position is flat; clearing trade state.",
                                        },
                                    )
                                    self._bank_snapshot("position_missing_clear", {"trade_id": trade_id})
                                except Exception:
                                    pass

                            # Treat this as a liquidation / external close and apply bankroll policy.
                            try:
                                refi = self.loop.run_until_complete(
                                    _refinance_after_liquidation(self.clients, self.cfg, budget=self.topup_budget)
                                )
                                if self.bank_log is not None:
                                    try:
                                        txhash = _txhash_hex((((refi.get("tx") or {}).get("txResponse") or {}).get("txhash")))
                                        self.bank_log.write_event(
                                            "bank_liquidation_refinance",
                                            {
                                                "symbol": self.market,
                                                "trade_mode": "real",
                                                "trade_id": trade_id,
                                                "refinance": refi,
                                                "txhash": txhash,
                                            },
                                        )
                                        self._bank_snapshot(
                                            "after_liquidation_refinance",
                                            {
                                                "trade_id": trade_id,
                                                "action": str(refi.get("action") or ""),
                                                "transferred_usdc": _finite_or_none(refi.get("transferred_usdc")),
                                                "txhash": txhash,
                                            },
                                        )
                                    except Exception:
                                        pass
                            except Exception as e:
                                self._alert("liquidation_refinance_failed", "Failed to refinance after liquidation", error=str(e))
                                if self.bank_log is not None:
                                    try:
                                        self.bank_log.write_event(
                                            "bank_liquidation_refinance_failed",
                                            {
                                                "symbol": self.market,
                                                "trade_mode": "real",
                                                "trade_id": trade_id,
                                                "error": str(e),
                                            },
                                        )
                                    except Exception:
                                        pass

                            self.open_trade = None
                            self._open_pos_zero_streak = 0
                            return
                    else:
                        self._open_pos_zero_streak = 0
                        # Keep stored size in sync (helps reduce-only sizing).
                        try:
                            self.open_trade["size_btc"] = float(live_abs)
                        except Exception:
                            pass

        thr = float(self._entry_threshold_for_day(d))
        score_i: Optional[float] = None
        if not self._missing_entry_cols:
            try:
                score_i = float(self._score_entry_index(i))
                if np.isfinite(score_i):
                    self._scores_cur_day.append(float(score_i))
            except Exception:
                score_i = None

        planned_entry_ts = ts + timedelta(minutes=1)

        # Block NEW entries unless the entry model has all required columns.
        # Exit model is disabled for the fixed-hold strategy.
        ready_for_new_entries = (not self._missing_entry_cols)

        if (
            bool(allow_trade_logic)
            and ready_for_new_entries
            and score_i is not None
            and np.isfinite(score_i)
            and score_i >= thr
            and self.open_trade is None
        ):
            self._audit_entry_decision(bar_open_ts=ts, score=float(score_i), threshold=float(thr), planned_entry_ts=planned_entry_ts, feats_row=self.feats.iloc[i])

            trade_id = uuid.uuid4().hex

            if self.trade_mode == "paper":
                # Entry fills at next minute open (we learn it once that bar is available).
                self.open_trade = {
                    "trade_id": trade_id,
                    "trade_mode": "paper",
                    "pending_fill": True,
                    "entry_index": i + 1,
                    "entry_time": None,
                    "entry_open": None,
                    "planned_entry_time": planned_entry_ts,
                    "entry_score": float(score_i),
                    "entry_threshold": float(thr),
                    "running_best": -1e30,
                    "best_k": None,
                }
            else:
                # Bank-proof snapshot before attempting a real entry.
                if self.bank_log is not None:
                    try:
                        self._bank_snapshot(
                            "before_entry",
                            {"trade_id": trade_id, "entry_score": _finite_or_none(score_i), "entry_threshold": _finite_or_none(thr)},
                        )
                    except Exception:
                        pass

                # Hard stop: do not open new trades if the initial topup budget is exhausted.
                if self.topup_budget.exhausted():
                    self._alert(
                        "entry_skipped_budget_exhausted",
                        "Initial topup budget exhausted; skipping entry",
                        trade_id=str(trade_id),
                        budget_usdc=float(self.topup_budget.budget_usdc),
                        spent_usdc=float(self.topup_budget.spent_usdc),
                    )
                    if self.bank_log is not None:
                        try:
                            self.bank_log.write_event(
                                "bank_entry_skipped_budget_exhausted",
                                {
                                    "symbol": self.market,
                                    "trade_id": trade_id,
                                    "trade_mode": "real",
                                    "direction": "LONG",
                                    "budget_usdc": float(self.topup_budget.budget_usdc),
                                    "spent_usdc": float(self.topup_budget.spent_usdc),
                                },
                            )
                        except Exception:
                            pass
                    return

                # real: ensure floor BEFORE measuring equity baseline (so profit calc doesn't treat top-ups as profit).
                topup_pre = self.loop.run_until_complete(_ensure_trading_floor(self.clients, self.cfg, budget=self.topup_budget))
                equity_before = float(
                    self.loop.run_until_complete(
                        _get_subaccount_equity_usdc(
                            self.clients,
                            address=self.cfg.address,
                            subaccount_number=self.cfg.subaccount_trading,
                        )
                    )
                )

                # If we could not reach the floor (budget exhausted / bank too small), refuse the entry.
                if float(equity_before) < float(self.cfg.trade_floor_usdc):
                    self._alert(
                        "entry_skipped_no_floor",
                        "Trading equity below floor; skipping entry",
                        trade_id=str(trade_id),
                        trading_equity_usdc=float(equity_before),
                        trade_floor_usdc=float(self.cfg.trade_floor_usdc),
                        topup_pre=topup_pre,
                    )
                    if self.bank_log is not None:
                        try:
                            self.bank_log.write_event(
                                "bank_entry_skipped_no_floor",
                                {
                                    "symbol": self.market,
                                    "trade_id": trade_id,
                                    "trade_mode": "real",
                                    "direction": "LONG",
                                    "trading_equity_usdc": float(equity_before),
                                    "trade_floor_usdc": float(self.cfg.trade_floor_usdc),
                                    "topup_pre": topup_pre,
                                },
                            )
                        except Exception:
                            pass
                    return

                if self.bank_log is not None:
                    try:
                        topup_pre_txhash = _txhash_hex((((((topup_pre or {}).get("tx") or {}).get("txResponse") or {}).get("txhash"))))
                        self._bank_snapshot(
                            "after_topup_pre",
                            {
                                "trade_id": trade_id,
                                "topped_up": bool((topup_pre or {}).get("topped_up")),
                                "transferred_usdc": _finite_or_none((topup_pre or {}).get("transferred_usdc")),
                                "txhash": topup_pre_txhash,
                                "trading_equity_before_usdc": float(equity_before),
                            },
                        )
                    except Exception:
                        pass

                try:
                    entry = self.loop.run_until_complete(self._open_real_long(equity_before=equity_before))
                except Exception as e:
                    print("Failed to open real long:", e)
                    return

                if self.bank_log is not None:
                    try:
                        entry_txhash = _txhash_hex((((entry.get("tx") or {}).get("txResponse") or {}).get("txhash")))
                        self.bank_log.write_event(
                            "bank_entry_order",
                            {
                                "symbol": self.market,
                                "trade_id": trade_id,
                                "trade_mode": "real",
                                "direction": "LONG",
                                "entry_time_utc": _iso_utc(planned_entry_ts),
                                "txhash": entry_txhash,
                                "meta": entry,
                            },
                        )
                        self._bank_snapshot("after_entry_order", {"trade_id": trade_id, "txhash": entry_txhash})
                    except Exception:
                        pass

                filled_sz = float(entry.get("fill_size_btc") or 0.0)
                filled_px = float(entry.get("fill_price") or 0.0)

                if filled_sz <= 0.0 or filled_px <= 0.0:
                    # Last-resort: if a live position exists, treat the entry as filled.
                    pos_sz = self._safe_get_live_position_size_btc()
                    if pos_sz is not None and abs(float(pos_sz)) > 0.0:
                        filled_sz = abs(float(pos_sz))
                        filled_px = float(entry.get("oracle_price") or filled_px or 0.0)
                        entry["fill_assumed"] = True
                    else:
                        self._alert(
                            "entry_fill_missing",
                            "Entry order did not fill (or fill not visible yet); skipping trade",
                            filled_sz=float(filled_sz),
                            filled_px=float(filled_px),
                        )
                        if self.bank_log is not None:
                            try:
                                entry_txhash = _txhash_hex((((entry.get("tx") or {}).get("txResponse") or {}).get("txhash")))
                                self.bank_log.write_event(
                                    "bank_entry_fill_missing",
                                    {
                                        "symbol": self.market,
                                        "trade_id": trade_id,
                                        "trade_mode": "real",
                                        "direction": "LONG",
                                        "entry_time_utc": _iso_utc(planned_entry_ts),
                                        "txhash": entry_txhash,
                                        "note": "Entry order did not fill (or fill not visible yet); skipping trade.",
                                        "filled_sz": float(filled_sz),
                                        "filled_px": float(filled_px),
                                        "meta": entry,
                                    },
                                )
                                self._bank_snapshot("entry_fill_missing", {"trade_id": trade_id, "txhash": entry_txhash})
                            except Exception:
                                pass
                        return

                self.open_trade = {
                    "trade_id": trade_id,
                    "trade_mode": "real",
                    "pending_fill": False,
                    "entry_index": i + 1,
                    "entry_time": planned_entry_ts,
                    "entry_order_sent_time_utc": entry.get("order_sent_time_utc"),
                    "entry_fill_time_utc": entry.get("fill_time_utc"),
                    "entry_open": float(filled_px),
                    "entry_fee_usdc": float(entry.get("fee_usdc") or 0.0),
                    "size_btc": float(filled_sz),
                    "equity_before_usdc": float(equity_before),
                    "topup_pre": topup_pre,
                    "entry_score": float(score_i),
                    "entry_threshold": float(thr),
                    "running_best": -1e30,
                    "best_k": None,
                    "entry": entry,
                }

                # Bank-proof log: trade opened.
                if self.bank_log is not None:
                    try:
                        entry_txhash = _txhash_hex((((entry.get("tx") or {}).get("txResponse") or {}).get("txhash")))
                        topup_pre_txhash = _txhash_hex((((((topup_pre or {}).get("tx") or {}).get("txResponse") or {}).get("txhash"))))
                        self.bank_log.write_event(
                            "bank_trade_opened",
                            {
                                "symbol": self.market,
                                "trade_id": trade_id,
                                "trade_mode": "real",
                                "direction": "LONG",
                                "entry_time_utc": _iso_utc(planned_entry_ts),
                                "entry_fill_time_utc": entry.get("fill_time_utc"),
                                "size_btc": float(filled_sz),
                                "entry_price": float(filled_px),
                                "fees": {"entry_fee_usdc": float(entry.get("fee_usdc") or 0.0)},
                                "signals": {
                                    "entry_score": _finite_or_none(score_i),
                                    "entry_threshold": _finite_or_none(thr),
                                },
                                "models": {
                                    "entry_artifact": str(self.models.entry_artifact),
                                    "exit_artifact": str(self.models.exit_artifact),
                                    "exit_gap_artifact": (str(self.exit_gap.artifact) if self.exit_gap is not None else None),
                                    "exit_gap_created_utc": (str(self.exit_gap.created_utc) if self.exit_gap is not None else None),
                                    "exit_gap_tau": (float(self.exit_gap_tau) if self.exit_gap is not None else None),
                                    "exit_gap_min_exit_k": (int(self.exit_gap_min_exit_k) if self.exit_gap is not None else None),
                                },
                                "balances": {
                                    "trading_equity_before_usdc": float(equity_before),
                                },
                                "transfers": {
                                    "topup_pre": {
                                        "topped_up": bool((topup_pre or {}).get("topped_up")),
                                        "transferred_usdc": _finite_or_none((topup_pre or {}).get("transferred_usdc")),
                                        "txhash": topup_pre_txhash,
                                    }
                                },
                                "orders": {"entry": {"txhash": entry_txhash, "meta": entry}},
                            },
                        )
                        self._bank_snapshot("after_entry")
                    except Exception as e:
                        self._alert("bank_trade_open_log_error", "Failed to write bank_trade_opened", error=str(e))


        # Progress open trade
        if not bool(allow_trade_logic):
            return
        if self.open_trade is None:
            return

        tr = self.open_trade

        # Fill paper entry at the bar open once we have that bar.
        if bool(tr.get("pending_fill")) and tr.get("trade_mode") == "paper":
            if int(tr.get("entry_index") or -1) == int(i):
                tr["pending_fill"] = False
                tr["entry_time"] = self.bars.iloc[i]["timestamp"]
                tr["entry_open"] = float(self.bars.iloc[i]["open"])
            else:
                return

        j0 = int(tr.get("entry_index") or 0)
        k_rel = int(i - j0)
        if k_rel <= 0:
            return
        # Exit policy: oracle-gap regressor (if configured), else fixed-hold.
        hold_min = int(getattr(self, "hold_min", 15))

        # Cap k_rel for reporting.
        if k_rel > int(hold_min):
            k_rel = int(hold_min)

        pred_gap: Optional[float] = None
        exit_reason = ""

        if self.exit_gap is not None and self.exit_src is not None:
            # Initialize per-trade state
            if "ret_hist" not in tr:
                tr["ret_hist"] = []
                tr["peak_ret"] = -1e30

            entry_px = float(tr.get("entry_open") or 0.0)
            cur_px = float(self.bars.iloc[i]["close"])
            cur_ret = float(net_return_pct(entry_px, cur_px, float(self.cfg.fee_side)))

            # update peak (as in dataset builder: peak includes current, then drawdown <= 0)
            tr["peak_ret"] = float(max(float(tr.get("peak_ret") or -1e30), cur_ret))
            peak_ret = float(tr.get("peak_ret") or cur_ret)

            # update return history
            tr["ret_hist"].append(float(cur_ret))
            r_prev1 = float(tr["ret_hist"][-2]) if len(tr["ret_hist"]) >= 2 else float("nan")
            r_prev2 = float(tr["ret_hist"][-3]) if len(tr["ret_hist"]) >= 3 else float("nan")
            r_prev3 = float(tr["ret_hist"][-4]) if len(tr["ret_hist"]) >= 4 else float("nan")

            # compute per-feature vector for this decision
            x_row = self._exit_gap_features_for_decision(
                decision_i=int(i),
                mins_in_trade=int(k_rel),
                entry_px=float(entry_px),
                cur_ret=float(cur_ret),
                r_prev1=r_prev1,
                r_prev2=r_prev2,
                r_prev3=r_prev3,
                peak_ret=float(peak_ret),
            )

            try:
                pred_gap = float(self.exit_gap.model.predict(np.asarray([x_row], dtype=np.float32))[0])
            except Exception:
                pred_gap = None

            # decide exit
            if int(k_rel) >= int(hold_min):
                exit_reason = "hold_min"
            elif int(k_rel) >= int(self.exit_gap_min_exit_k) and pred_gap is not None and np.isfinite(pred_gap) and float(pred_gap) <= float(self.exit_gap_tau):
                exit_reason = "pred_gap<=tau"

            tr["exit_pred_gap_pct_last"] = pred_gap
            tr["exit_reason_last"] = exit_reason
        else:
            if int(k_rel) >= int(hold_min):
                exit_reason = "hold_min"

        if exit_reason:
            exit_k = int(k_rel)
            exit_time = ts + timedelta(minutes=1)
            pred = None

            # Bank-proof snapshot right before we attempt to exit a real trade.
            if str(tr.get("trade_mode") or "").lower() == "real":
                try:
                    self._bank_snapshot(
                        "before_exit",
                        {
                            "trade_id": str(tr.get("trade_id") or ""),
                            "exit_rel_min": int(exit_k),
                            "exit_reason": str(exit_reason),
                            "exit_pred_gap_pct": _finite_or_none(pred_gap),
                            "exit_gap_tau": float(self.exit_gap_tau) if self.exit_gap is not None else None,
                            "exit_gap_min_exit_k": int(self.exit_gap_min_exit_k) if self.exit_gap is not None else None,
                        },
                    )
                except Exception:
                    pass

            if tr.get("trade_mode") == "paper":
                entry_px = float(tr.get("entry_open") or 0.0)
                exit_px = float(self.bars.iloc[i]["close"])
                realized = float(net_return_pct(entry_px, exit_px, float(self.cfg.fee_side)))

                self._audit_trade_closed(
                    paper=True,
                    entry_time_utc=_iso_utc(tr.get("entry_time")),
                    exit_time_utc=_iso_utc(exit_time),
                    entry_price=float(entry_px),
                    exit_price=float(exit_px),
                    exit_rel_min=int(exit_k),
                    realized_ret_pct=float(realized),
                    predicted_ret_pct=pred,
                    entry_score=_finite_or_none(tr.get("entry_score")),
                    entry_threshold=_finite_or_none(tr.get("entry_threshold")),
                    extra={
                        "trade_mode": "paper",
                        "entry_fill_time_utc": _iso_utc(tr.get("entry_time")),
                        "exit_fill_time_utc": _iso_utc(exit_time),
                        "entry_fill_latency_s": None,
                        "exit_fill_latency_s": None,
                        "exit_reason": str(exit_reason),
                        "exit_pred_gap_pct": _finite_or_none(pred_gap),
                        "exit_gap_tau": float(self.exit_gap_tau) if self.exit_gap is not None else None,
                        "exit_gap_min_exit_k": int(self.exit_gap_min_exit_k) if self.exit_gap is not None else None,
                    },
                )

                self.records.append(
                    {
                        "paper": True,
                        "trade_mode": "paper",
                        "entry_time": pd.to_datetime(tr.get("entry_time"), utc=True).to_pydatetime(),
                        "entry_fill_time": pd.to_datetime(tr.get("entry_time"), utc=True).to_pydatetime(),
                        "exit_time": pd.to_datetime(exit_time, utc=True).to_pydatetime(),
                        "exit_fill_time": pd.to_datetime(exit_time, utc=True).to_pydatetime(),
                        "entry_fill_latency_s": None,
                        "exit_fill_latency_s": None,
                        "exit_rel_min": int(exit_k),
                        "exit_reason": str(exit_reason),
                        "exit_pred_gap_pct": _finite_or_none(pred_gap),
                        "exit_gap_tau": float(self.exit_gap_tau) if self.exit_gap is not None else None,
                        "exit_gap_min_exit_k": int(self.exit_gap_min_exit_k) if self.exit_gap is not None else None,
                        "predicted_ret_pct": pred,
                        "realized_ret_pct": float(realized),
                    }
                )
                self.open_trade = None
                return

            # real
            equity_before = float(tr.get("equity_before_usdc") or 0.0)

            # Close whatever size is actually open (prevents reduce-only errors if entry partially filled).
            live_sz = float(
                self.loop.run_until_complete(
                    _get_open_position_size_btc(
                        self.clients,
                        address=self.cfg.address,
                        subaccount_number=int(self.cfg.subaccount_trading),
                        market=self.market,
                    )
                )
            )
            size_btc = abs(float(live_sz)) if abs(float(live_sz)) > 0.0 else float(tr.get("size_btc") or 0.0)

            try:
                close_meta = self.loop.run_until_complete(self._close_real_long(size_btc=float(size_btc)))
            except Exception as e:
                print("Failed to close real long:", e)
                return

            # Bank-proof log: exit order attempt.
            if self.bank_log is not None:
                try:
                    exit_txhash = _txhash_hex((((close_meta.get("tx") or {}).get("txResponse") or {}).get("txhash")))
                    self.bank_log.write_event(
                        "bank_exit_order",
                        {
                            "symbol": self.market,
                            "trade_id": str(tr.get("trade_id") or ""),
                            "trade_mode": "real",
                            "direction": "LONG",
                            "exit_time_utc": _iso_utc(exit_time),
                            "exit_rel_min": int(exit_k),
                            "txhash": exit_txhash,
                            "meta": close_meta,
                        },
                    )
                    self._bank_snapshot("after_exit_order", {"trade_id": str(tr.get("trade_id") or ""), "txhash": exit_txhash})
                except Exception as e:
                    self._alert("bank_exit_order_log_error", "Failed to write bank_exit_order", error=str(e))

            exit_fill_px = float(close_meta.get("fill_price") or 0.0)
            exit_filled_sz = float(close_meta.get("fill_size_btc") or 0.0)
            if exit_fill_px <= 0.0 or exit_filled_sz <= 0.0:
                # If the position is flat, assume we closed even if fills lag.
                pos_sz2 = self._safe_get_live_position_size_btc()
                if pos_sz2 is not None and abs(float(pos_sz2)) <= 0.0:
                    close_meta["fill_assumed"] = True
                    exit_fill_px = float(close_meta.get("oracle_price") or exit_fill_px or 0.0)
                    exit_filled_sz = float(size_btc)
                else:
                    self._alert("exit_fill_missing", "Exit not filled (or not visible yet); keeping position open")
                    if self.bank_log is not None:
                        try:
                            self.bank_log.write_event(
                                "bank_exit_fill_missing",
                                {
                                    "symbol": self.market,
                                    "trade_id": str(tr.get("trade_id") or ""),
                                    "trade_mode": "real",
                                    "direction": "LONG",
                                    "exit_time_utc": _iso_utc(exit_time),
                                    "exit_rel_min": int(exit_k),
                                    "note": "Exit not filled (or not visible yet); keeping position open.",
                                    "meta": close_meta,
                                },
                            )
                            self._bank_snapshot("exit_fill_missing", {"trade_id": str(tr.get("trade_id") or "")})
                        except Exception:
                            pass
                    return

            # Give indexer a moment to reflect equity.
            time.sleep(0.75)
            bank_ops = self.loop.run_until_complete(
                _siphon_profit_and_topup(self.clients, self.cfg, equity_before=equity_before, budget=self.topup_budget)
            )

            # Bank-proof snapshot after close + post-trade internal transfers.
            if self.bank_log is not None:
                try:
                    self.bank_log.write_event(
                        "bank_post_trade_ops",
                        {
                            "symbol": self.market,
                            "trade_id": str(tr.get("trade_id") or ""),
                            "trade_mode": "real",
                            "direction": "LONG",
                            "exit_time_utc": _iso_utc(exit_time),
                            "bank_ops": bank_ops,
                        },
                    )
                    self._bank_snapshot(
                        "after_post_trade_ops",
                        {
                            "trade_id": str(tr.get("trade_id") or ""),
                            "profit_usdc": _finite_or_none(bank_ops.get("profit_usdc")),
                            "net_bank_flow_usdc": _finite_or_none(
                                float(((bank_ops.get("siphon") or {}).get("transferred_usdc") or 0.0))
                                - float(((tr.get("topup_pre") or {}).get("transferred_usdc") or 0.0))
                                - float(((bank_ops.get("topup") or {}).get("transferred_usdc") or 0.0))
                            ),
                        },
                    )
                except Exception as e:
                    self._alert("bank_post_trade_ops_log_error", "Failed to write bank_post_trade_ops", error=str(e))

            entry_fill_px = float(tr.get("entry_open") or 0.0)
            entry_fee = float(tr.get("entry_fee_usdc") or 0.0)
            exit_fee = float(close_meta.get("fee_usdc") or 0.0)
            entry_filled_sz = float(tr.get("size_btc") or 0.0)
            closed_sz = float(min(entry_filled_sz, exit_filled_sz)) if entry_filled_sz > 0.0 and exit_filled_sz > 0.0 else float(entry_filled_sz)

            realized_fill = float(
                realized_ret_pct_from_fills(
                    entry_fill_px=entry_fill_px,
                    exit_fill_px=exit_fill_px,
                    size_btc=closed_sz,
                    entry_fee_usdc=entry_fee,
                    exit_fee_usdc=exit_fee,
                )
            )
            realized_assumed_fee = float(net_return_pct(entry_fill_px, exit_fill_px, float(self.cfg.fee_side)))
            exit_pred_gap_pct = _finite_or_none(tr.get("exit_pred_gap_pct_last"))
            exit_reason = str(tr.get("exit_reason_last") or "")

            entry_fill_latency_s: Optional[float] = None
            exit_fill_latency_s: Optional[float] = None
            try:
                sent = tr.get("entry_order_sent_time_utc")
                filled = tr.get("entry_fill_time_utc")
                if sent and filled:
                    dt_sent = pd.to_datetime(sent, utc=True)
                    dt_fill = pd.to_datetime(filled, utc=True)
                    entry_fill_latency_s = float((dt_fill - dt_sent).total_seconds())
            except Exception:
                entry_fill_latency_s = None

            try:
                sent = close_meta.get("order_sent_time_utc")
                filled = close_meta.get("fill_time_utc")
                if sent and filled:
                    dt_sent = pd.to_datetime(sent, utc=True)
                    dt_fill = pd.to_datetime(filled, utc=True)
                    exit_fill_latency_s = float((dt_fill - dt_sent).total_seconds())
            except Exception:
                exit_fill_latency_s = None

            extra = {
                "trade_mode": "real",
                "entry": tr.get("entry"),
                "exit": close_meta,
                "bank_ops": bank_ops,
                "topup_pre": tr.get("topup_pre"),
                "filled_size_btc": float(closed_sz),
                "entry_fee_usdc": float(entry_fee),
                "exit_fee_usdc": float(exit_fee),
                "realized_ret_pct_fill": float(realized_fill),
                "entry_order_sent_time_utc": tr.get("entry_order_sent_time_utc"),
                "exit_order_sent_time_utc": close_meta.get("order_sent_time_utc"),
                "entry_fill_time_utc": tr.get("entry_fill_time_utc"),
                "exit_fill_time_utc": close_meta.get("fill_time_utc"),
                "entry_fill_latency_s": _finite_or_none(entry_fill_latency_s),
                "exit_fill_latency_s": _finite_or_none(exit_fill_latency_s),
            }

            # Primary metric: money made (USDC).
            try:
                pnl_usdc = float(bank_ops.get("profit_usdc") or 0.0)
                siphon_usdc = float(((bank_ops.get("siphon") or {}).get("transferred_usdc") or 0.0))
                topup_pre_usdc = float(((tr.get("topup_pre") or {}).get("transferred_usdc") or 0.0))
                topup_post_usdc = float(((bank_ops.get("topup") or {}).get("transferred_usdc") or 0.0))
                print(
                    "Trade closed (real): profit_usdc="
                    f"{pnl_usdc:+.6f} siphon_to_bank={siphon_usdc:+.6f} topup_pre={topup_pre_usdc:+.6f} topup_post={topup_post_usdc:+.6f}"
                )
            except Exception:
                pass

            # Bank-grade per-trade log (hash-chained JSONL).
            if self.bank_log is not None:
                try:
                    entry_meta = tr.get("entry") if isinstance(tr.get("entry"), dict) else {}
                    exit_meta = close_meta if isinstance(close_meta, dict) else {}

                    topup_pre = tr.get("topup_pre") if isinstance(tr.get("topup_pre"), dict) else {}
                    siphon = bank_ops.get("siphon") if isinstance(bank_ops.get("siphon"), dict) else {}
                    topup_post = bank_ops.get("topup") if isinstance(bank_ops.get("topup"), dict) else {}

                    entry_txhash = _txhash_hex((((entry_meta.get("tx") or {}).get("txResponse") or {}).get("txhash")))
                    exit_txhash = _txhash_hex((((exit_meta.get("tx") or {}).get("txResponse") or {}).get("txhash")))

                    topup_pre_txhash = _txhash_hex((((topup_pre.get("tx") or {}).get("txResponse") or {}).get("txhash")))
                    siphon_txhash = _txhash_hex((((siphon.get("tx") or {}).get("txResponse") or {}).get("txhash")))
                    topup_post_txhash = _txhash_hex((((topup_post.get("tx") or {}).get("txResponse") or {}).get("txhash")))

                    siphon_amt = float((siphon.get("transferred_usdc") or 0.0))
                    topup_pre_amt = float((topup_pre.get("transferred_usdc") or 0.0))
                    topup_post_amt = float((topup_post.get("transferred_usdc") or 0.0))

                    trade_rec = {
                        "symbol": self.market,
                        "trade_id": str(tr.get("trade_id") or ""),
                        "trade_mode": "real",
                        "direction": "LONG",
                        "entry_time_utc": _iso_utc(tr.get("entry_time")),
                        "exit_time_utc": _iso_utc(exit_time),
                        "entry_fill_time_utc": tr.get("entry_fill_time_utc"),
                        "exit_fill_time_utc": exit_meta.get("fill_time_utc"),
                        "size_btc": float(closed_sz),
                        "entry_price": float(entry_fill_px),
                        "exit_price": float(exit_fill_px),
                        "money_made_usdc": float(bank_ops.get("profit_usdc") or 0.0),
                        "profit_usdc": float(bank_ops.get("profit_usdc") or 0.0),
                        "signals": {
                            "entry_score": _finite_or_none(tr.get("entry_score")),
                            "entry_threshold": _finite_or_none(tr.get("entry_threshold")),
                            "exit_pred_gap_pct": exit_pred_gap_pct,
                            "exit_reason": str(exit_reason),
                            "exit_gap_tau": float(self.exit_gap_tau) if self.exit_gap is not None else None,
                            "exit_gap_min_exit_k": int(self.exit_gap_min_exit_k) if self.exit_gap is not None else None,
                            "exit_rel_min": int(exit_k),
                        },
                        "fees": {"entry_fee_usdc": float(entry_fee), "exit_fee_usdc": float(exit_fee)},
                        "returns": {
                            "realized_ret_pct_assumed_fee": float(realized_assumed_fee),
                            "realized_ret_pct_fill": float(realized_fill),
                        },
                        "balances": {
                            "trading_equity_before_usdc": float(bank_ops.get("equity_before_usdc") or equity_before),
                            "trading_equity_after_close_usdc": float(bank_ops.get("equity_after_usdc") or 0.0),
                            "trade_floor_usdc": float(self.cfg.trade_floor_usdc),
                        },
                        "transfers": {
                            "topup_pre": {
                                "topped_up": bool(topup_pre.get("topped_up")),
                                "transferred_usdc": float(topup_pre_amt),
                                "txhash": topup_pre_txhash,
                            },
                            "siphon": {
                                "attempted": bool(siphon.get("attempted")),
                                "requested_usdc": float(siphon.get("requested_usdc") or 0.0),
                                "transferred_usdc": float(siphon_amt),
                                "txhash": siphon_txhash,
                            },
                            "topup_post": {
                                "topped_up": bool(topup_post.get("topped_up")),
                                "transferred_usdc": float(topup_post_amt),
                                "txhash": topup_post_txhash,
                            },
                        },
                        "net_bank_flow_usdc": float(siphon_amt - topup_pre_amt - topup_post_amt),
                        "orders": {
                            "entry": {
                                "client_id": entry_meta.get("client_id"),
                                "txhash": entry_txhash,
                                "fills": ((entry_meta.get("fill") or {}).get("fills") or []),
                            },
                            "exit": {
                                "client_id": exit_meta.get("client_id"),
                                "txhash": exit_txhash,
                                "fills": ((exit_meta.get("fill") or {}).get("fills") or []),
                            },
                        },
                        "latency_s": {
                            "entry_fill_latency_s": _finite_or_none(entry_fill_latency_s),
                            "exit_fill_latency_s": _finite_or_none(exit_fill_latency_s),
                        },
                        "config": {
                            "profit_siphon_frac": float(self.cfg.profit_siphon_frac),
                            "max_leverage_cap": ("auto" if int(self.cfg.max_leverage) <= 0 else int(self.cfg.max_leverage)),
                        },
                    }

                    self.bank_log.write_trade_closed(trade_rec)
                    try:
                        self._bank_snapshot("after_trade_closed", {"trade_id": str(tr.get("trade_id") or "")})
                    except Exception:
                        pass
                except Exception as e:
                    print("Bank log write failed:", e)

            extra = dict(extra)
            extra.update(
                {
                    "exit_reason": str(exit_reason),
                    "exit_pred_gap_pct": exit_pred_gap_pct,
                    "exit_gap_tau": float(self.exit_gap_tau) if self.exit_gap is not None else None,
                    "exit_gap_min_exit_k": int(self.exit_gap_min_exit_k) if self.exit_gap is not None else None,
                }
            )

            self._audit_trade_closed(
                paper=False,
                entry_time_utc=_iso_utc(tr.get("entry_time")),
                exit_time_utc=_iso_utc(exit_time),
                entry_price=float(entry_fill_px),
                exit_price=float(exit_fill_px),
                exit_rel_min=int(exit_k),
                realized_ret_pct=float(realized_assumed_fee),
                predicted_ret_pct=pred,
                entry_score=_finite_or_none(tr.get("entry_score")),
                entry_threshold=_finite_or_none(tr.get("entry_threshold")),
                extra=extra,
            )

            self.records.append(
                {
                    "paper": False,
                    "trade_mode": "real",
                    "entry_time": pd.to_datetime(tr.get("entry_time"), utc=True).to_pydatetime(),
                    "entry_fill_time": pd.to_datetime(tr.get("entry_fill_time_utc") or tr.get("entry_time"), utc=True).to_pydatetime(),
                    "exit_time": pd.to_datetime(exit_time, utc=True).to_pydatetime(),
                    "exit_fill_time": pd.to_datetime(close_meta.get("fill_time_utc") or exit_time, utc=True).to_pydatetime(),
                    "entry_fill_latency_s": _finite_or_none(entry_fill_latency_s),
                    "exit_fill_latency_s": _finite_or_none(exit_fill_latency_s),
                    "exit_rel_min": int(exit_k),
                    "exit_reason": str(exit_reason),
                    "exit_pred_gap_pct": exit_pred_gap_pct,
                    "exit_gap_tau": float(self.exit_gap_tau) if self.exit_gap is not None else None,
                    "exit_gap_min_exit_k": int(self.exit_gap_min_exit_k) if self.exit_gap is not None else None,
                    "predicted_ret_pct": pred,
                    "realized_ret_pct": float(realized_assumed_fee),
                    "realized_ret_pct_fill": float(realized_fill),
                    "equity_before_usdc": float(bank_ops.get("equity_before_usdc") or 0.0),
                    "equity_after_usdc": float(bank_ops.get("equity_after_usdc") or 0.0),
                    "profit_usdc": float(bank_ops.get("profit_usdc") or 0.0),
                    "siphon_requested_usdc": float(((bank_ops.get("siphon") or {}).get("requested_usdc") or 0.0)),
                    "siphon_transferred_usdc": float(((bank_ops.get("siphon") or {}).get("transferred_usdc") or 0.0)),
                    "topup_pre_transferred_usdc": float(((tr.get("topup_pre") or {}).get("transferred_usdc") or 0.0)),
                    "topup_transferred_usdc": float(((bank_ops.get("topup") or {}).get("transferred_usdc") or 0.0)),
                    "net_bank_flow_usdc": float(
                        float(((bank_ops.get("siphon") or {}).get("transferred_usdc") or 0.0))
                        - float(((tr.get("topup_pre") or {}).get("transferred_usdc") or 0.0))
                        - float(((bank_ops.get("topup") or {}).get("transferred_usdc") or 0.0))
                    ),
                }
            )

            self.open_trade = None

    def flush_outputs(self) -> None:
        if not self.records:
            return
        trades = pd.DataFrame(self.records)
        trades.sort_values("entry_time", inplace=True)
        trades["date"] = pd.to_datetime(trades["entry_time"]).dt.date

        daily_src = trades.copy()

        # Ensure optional real-only columns exist even for paper-only runs.
        for c in [
            "profit_usdc",
            "siphon_transferred_usdc",
            "topup_pre_transferred_usdc",
            "topup_transferred_usdc",
            "net_bank_flow_usdc",
        ]:
            if c not in daily_src.columns:
                daily_src[c] = np.nan

        is_real = daily_src["trade_mode"].astype(str).str.lower().eq("real")
        daily_src["is_real"] = is_real.astype(int)

        daily_src["profit_usdc_real"] = daily_src["profit_usdc"].where(is_real)
        daily_src["net_bank_flow_usdc_real"] = daily_src["net_bank_flow_usdc"].where(is_real)
        daily_src["siphon_transferred_usdc_real"] = daily_src["siphon_transferred_usdc"].where(is_real)
        daily_src["topup_pre_transferred_usdc_real"] = daily_src["topup_pre_transferred_usdc"].where(is_real)
        daily_src["topup_transferred_usdc_real"] = daily_src["topup_transferred_usdc"].where(is_real)

        daily = daily_src.groupby("date", as_index=False).agg(
            # Counts
            n_trades=("realized_ret_pct", "size"),
            n_real_trades=("is_real", "sum"),
            # Money made (primary)
            sum_profit_usdc=("profit_usdc_real", "sum"),
            mean_profit_usdc=("profit_usdc_real", "mean"),
            sum_net_bank_flow_usdc=("net_bank_flow_usdc_real", "sum"),
            # Transfers
            sum_siphon_transferred_usdc=("siphon_transferred_usdc_real", "sum"),
            sum_topup_pre_transferred_usdc=("topup_pre_transferred_usdc_real", "sum"),
            sum_topup_transferred_usdc=("topup_transferred_usdc_real", "sum"),
            # Percent returns (secondary)
            mean_daily_pct=("realized_ret_pct", "mean"),
            sum_daily_pct=("realized_ret_pct", "sum"),
            median_daily_pct=("realized_ret_pct", "median"),
            top_day_pct=("realized_ret_pct", "max"),
            worst_day_pct=("realized_ret_pct", "min"),
        )

        ts = _now_ts()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        trades.to_csv(self.out_dir / f"trades_{ts}.csv", index=False)
        daily.to_csv(self.out_dir / f"daily_{ts}.csv", index=False)

        # Keep state/retention in sync with long-running operation.
        try:
            self._save_prior_scores()
        except Exception:
            pass
        try:
            self._cleanup_out_dir()
        except Exception:
            pass


def _run_ws_candles(
    *,
    ws_url: str,
    market: str,
    out_q: "queue.Queue[Dict[str, Any]]",
    stop_evt: threading.Event,
) -> None:
    # Reconnect loop for long-running operation.
    backoff = 1.0

    while not stop_evt.is_set():
        asm = CandleAssembler()

        def on_open(ws):
            ws.candles.subscribe(id=market, resolution=CandlesResolution.ONE_MINUTE, batched=False)

        def on_message(ws, msg):
            if stop_evt.is_set():
                ws.close()
                return

            t = msg.get("type")
            if t == "connected":
                return

            if t == "subscribed":
                candles = ((msg.get("contents") or {}).get("candles") or [])
                # snapshot comes newest-first; process oldest-first
                try:
                    candles = sorted(candles, key=lambda c: c.get("startedAt") or "")
                except Exception:
                    pass
                for c in candles:
                    closed = asm.ingest(c)
                    if closed is not None:
                        out_q.put(closed)
                return

            if t == "channel_data" and msg.get("channel") == "v4_candles":
                c = (msg.get("contents") or {})
                closed = asm.ingest(c)
                if closed is not None:
                    out_q.put(closed)
                return

        def on_error(ws, error):
            print("ws_error:", error)

        def on_close(ws, status, msg):
            print("ws_closed", status, msg)

        started = time.time()
        try:
            sock = IndexerSocket(url=ws_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
            sock.run_forever(ping_interval=15, ping_timeout=10)
        except Exception as e:
            print("ws_loop_error:", e)

        if stop_evt.is_set():
            break

        dur = float(time.time() - started)
        if dur >= 60.0:
            backoff = 1.0
        else:
            backoff = min(float(backoff) * 2.0, 60.0)

        print(f"ws_reconnect_sleep_s={backoff:.1f}")
        time.sleep(float(backoff))


async def _warm_backfill(clients, *, market: str, hours: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(hours=int(hours))

    # Fetch in <=1000 candle chunks.
    frames: List[pd.DataFrame] = []
    s = start
    while s < end:
        e = min(s + timedelta(minutes=999), end)
        resp = await clients.indexer.markets.get_perpetual_market_candles(
            market=market,
            resolution="1MIN",
            from_iso=s.isoformat().replace("+00:00", "Z"),
            to_iso=e.isoformat().replace("+00:00", "Z"),
            limit=1000,
        )
        candles = resp.get("candles") if isinstance(resp, dict) else None
        if isinstance(candles, list) and candles:
            df = pd.DataFrame(candles)
            # startedAt is the bar open time
            df["timestamp"] = pd.to_datetime(df["startedAt"], utc=True)
            df["open"] = df["open"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["close"] = df["close"].astype(float)
            df["volume"] = df["baseTokenVolume"].astype(float)
            frames.append(df[["timestamp", "open", "high", "low", "close", "volume"]])
        s = e
        await asyncio.sleep(0.05)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    bars = pd.concat(frames, ignore_index=True)
    bars = bars.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return bars


def main() -> None:
    load_dotenv(Path(".env"), override=False)

    ap = argparse.ArgumentParser(description="Live dYdX v4 BTC-USD runner (bank+trading subaccounts)")
    ap.add_argument(
        "--entry-model",
        default=str(REPO_ROOT / "data" / "pattern_entry_regressor" / "pattern_entry_regressor_v2_2026-01-02T23-51-17Z.joblib"),
        help="Entry model artifact. Supports both legacy (features) and pattern v2 (feature_cols).",
    )
    ap.add_argument(
        "--exit-model",
        default="",
        help="Optional legacy exit model artifact. Leave empty to disable (fixed-hold strategy).",
    )

    ap.add_argument("--market", default=os.getenv("DYDX_MARKET", "BTC-USD"))
    ap.add_argument("--target-frac", type=float, default=0.0001, help="Entry take fraction (0.0001=0.01%%, 0.001=0.1%%)")
    ap.add_argument("--hold-min", type=int, default=15)

    ap.add_argument("--subaccount-trading", type=int, default=int(os.getenv("DYDX_SUBACCOUNT_TRADING", "1")))
    ap.add_argument("--subaccount-bank", type=int, default=int(os.getenv("DYDX_SUBACCOUNT_BANK", "0")))

    ap.add_argument(
        "--exit-gap-model",
        default="",
        help="Optional oracle-gap regressor artifact (joblib dict with keys: model, feature_cols).",
    )
    ap.add_argument("--exit-gap-tau", type=float, default=0.10, help="Exit when pred_gap_pct <= this (pct points)")
    ap.add_argument("--exit-gap-min-exit-k", type=int, default=2, help="Do not exit before this minute-in-trade")

    ap.add_argument("--trade-mode", choices=["paper", "real"], default="paper")
    ap.add_argument("--yes-really", action="store_true", help="Required for --trade-mode real")

    ap.add_argument("--backfill-hours", type=int, default=168)

    ap.add_argument(
        "--seed-csv",
        default=str(REPO_ROOT / "data" / "seeds" / "dydx_BTC-USD_1MIN_seed_168h.csv"),
        help="Optional local CSV to seed bars if warm backfill fails. Use '' to disable.",
    )

    ap.add_argument(
        "--initial-topup-budget-usdc",
        type=float,
        default=50.0,
        help="Hard stop budget: once exceeded, the runner will refuse new entries (exits still allowed).",
    )

    # Offline replay mode: feed bars from CSV instead of WS/REST.
    ap.add_argument(
        "--replay-csv",
        default="",
        help="If set, replay closed 1m bars from this CSV (expects timestamp/open/high/low/close/volume). Only supported in --trade-mode paper.",
    )
    ap.add_argument("--replay-max-bars", type=int, default=0, help="If >0, only replay the first N bars (debug)")
    ap.add_argument("--replay-sleep-sec", type=float, default=0.0, help="If >0, sleep between replayed bars")
    ap.add_argument("--thresholds-csv", default="", help="Optional seed thresholds CSV (date, threshold)")

    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "live_dydx"))
    ap.add_argument(
        "--audit-log",
        default=str(REPO_ROOT / "data" / "live_dydx" / f"audit_{_now_ts()}.jsonl"),
        help="Append-only JSONL audit log (validated against contracts/v1). Use '' to disable.",
    )
    ap.add_argument(
        "--bank-log",
        default="AUTO",
        help=(
            "Append-only bank-proof JSONL log (hash-chained). "
            "'AUTO' => <out-dir>/bank_proof_{timestamp}.jsonl. Use '' to disable."
        ),
    )

    # Ops / long-running options
    ap.add_argument(
        "--metrics-port",
        type=int,
        default=0,
        help="If >0, start a minimal HTTP server on this port exposing /health and /metrics",
    )
    ap.add_argument(
        "--health-max-staleness-sec",
        type=float,
        default=120.0,
        help="/health fails if no closed bar was processed within this many seconds",
    )
    ap.add_argument(
        "--stall-seconds",
        type=int,
        default=300,
        help="If >0, exit the process if no closed 1m bars arrive for this long (lets Docker/systemd restart)",
    )
    ap.add_argument(
        "--thresholds-live-out",
        default="AUTO",
        help="Where to persist daily entry thresholds CSV. 'AUTO' => <out-dir>/entry_score_thresholds_live.csv. Use '' to disable.",
    )
    ap.add_argument(
        "--prior-scores-path",
        default="AUTO",
        help="Where to persist prior entry score history JSON. 'AUTO' => <out-dir>/prior_scores.json. Use '' to disable.",
    )
    ap.add_argument(
        "--max-prior-scores",
        type=int,
        default=200_000,
        help="Max number of prior entry scores to keep persisted (rolling window)",
    )
    ap.add_argument(
        "--min-prior-scores",
        type=int,
        default=2000,
        help="Require at least this many prior scores before enabling thresholds (else threshold=+inf)",
    )
    ap.add_argument(
        "--seed-days",
        type=int,
        default=2,
        help="On startup (if no prior_scores.json), seed prior score pool from this many backfill days",
    )
    ap.add_argument(
        "--retention-days",
        type=int,
        default=0,
        help="If >0, delete old output files in out-dir older than this many days (minutes bars, trades, audit, bank logs)",
    )

    args = ap.parse_args()

    if args.trade_mode == "real" and not bool(args.yes_really):
        raise SystemExit("Refusing to place real orders without --yes-really")

    cfg = DydxV4Config.from_env()
    cfg.market = str(args.market).upper()
    cfg.subaccount_trading = int(args.subaccount_trading)
    cfg.subaccount_bank = int(args.subaccount_bank)

    exit_path = Path(args.exit_model) if str(args.exit_model).strip() else None
    models = load_models(Path(args.entry_model), exit_path)

    exit_gap_path = Path(args.exit_gap_model) if str(getattr(args, "exit_gap_model", "")).strip() else None
    exit_gap = load_exit_gap_regressor(exit_gap_path)

    out_dir = Path(args.out_dir)

    audit_log = Path(args.audit_log) if str(args.audit_log).strip() else None

    raw_bank_log = str(args.bank_log).strip()
    if raw_bank_log == "":
        bank_log_path = None
    elif raw_bank_log.upper() == "AUTO":
        bank_log_path = out_dir / f"bank_proof_{_now_ts()}.jsonl"
    else:
        bank_log_path = Path(raw_bank_log)

    thresholds_csv = Path(args.thresholds_csv) if str(args.thresholds_csv).strip() else None

    raw_thr_out = str(getattr(args, "thresholds_live_out", "AUTO") or "").strip()
    if raw_thr_out == "":
        thresholds_live_out = None
    elif raw_thr_out.upper() == "AUTO":
        thresholds_live_out = out_dir / "entry_score_thresholds_live.csv"
    else:
        thresholds_live_out = Path(raw_thr_out)

    raw_scores = str(getattr(args, "prior_scores_path", "AUTO") or "").strip()
    if raw_scores == "":
        prior_scores_path = None
    elif raw_scores.upper() == "AUTO":
        prior_scores_path = out_dir / "prior_scores.json"
    else:
        prior_scores_path = Path(raw_scores)

    audit_cm = AuditLogger.create(audit_log) if audit_log else nullcontext()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # For --trade-mode paper we only need indexer REST + websocket; no mnemonic/node required.
    if str(args.trade_mode).lower() == "paper":
        clients = SimpleNamespace(
            network=SimpleNamespace(rest_indexer=str(cfg.indexer_rest).rstrip("/"), websocket_indexer=str(cfg.indexer_ws).rstrip("/")),
            indexer=IndexerClient(str(cfg.indexer_rest).rstrip("/")),
        )
    else:
        clients = loop.run_until_complete(connect_v4(cfg))

    with audit_cm as audit:
        # Share one run_id across audit + bank logs (when both enabled).
        run_id = str(audit.run_id) if audit is not None else uuid.uuid4().hex

        bank_cm = BankTradeLogger.create(bank_log_path, run_id=run_id) if bank_log_path else nullcontext()
        with bank_cm as bank_log:
            if audit is not None:
                audit.write(
                    {
                        "kind": KIND_RUN_META,
                        "mode": "live",
                        "symbol": str(cfg.market).upper(),
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
                            "exit_gap": {
                                "artifact": (exit_gap.artifact if exit_gap is not None else None),
                                "created_utc": (exit_gap.created_utc if exit_gap is not None else None),
                                "n_features": (len(exit_gap.feature_cols) if exit_gap is not None else 0),
                                "tau": (float(getattr(args, "exit_gap_tau", 0.10)) if exit_gap is not None else None),
                                "min_exit_k": (int(getattr(args, "exit_gap_min_exit_k", 2)) if exit_gap is not None else None),
                            },
                            "pre_min": int(models.pre_min),
                            "entry_pre_min": int(models.entry_pre_min),
                            "exit_pre_min": int(models.exit_pre_min),
                        },
                        "code_version": {
                            "script": Path(__file__).name,
                            "git_commit": _git_commit_hash(REPO_ROOT),
                        },
                        "exchange": "dydx_v4",
                        "network": str(cfg.env),
                        "subaccounts": {"trading": int(cfg.subaccount_trading), "bank": int(cfg.subaccount_bank)},
                    }
                )

            if bank_log is not None:
                bank_log.write_meta(
                    {
                        "mode": "live",
                        "symbol": str(cfg.market).upper(),
                        "exchange": "dydx_v4",
                        "network": str(cfg.env),
                        "address": str(cfg.address),
                        "subaccounts": {"trading": int(cfg.subaccount_trading), "bank": int(cfg.subaccount_bank)},
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
                            "exit_gap": {
                                "artifact": (exit_gap.artifact if exit_gap is not None else None),
                                "created_utc": (exit_gap.created_utc if exit_gap is not None else None),
                                "n_features": (len(exit_gap.feature_cols) if exit_gap is not None else 0),
                                "tau": (float(getattr(args, "exit_gap_tau", 0.10)) if exit_gap is not None else None),
                                "min_exit_k": (int(getattr(args, "exit_gap_min_exit_k", 2)) if exit_gap is not None else None),
                            },
                            "pre_min": int(models.pre_min),
                            "entry_pre_min": int(models.entry_pre_min),
                            "exit_pre_min": int(models.exit_pre_min),
                        },
                        "code_version": {"script": Path(__file__).name, "git_commit": _git_commit_hash(REPO_ROOT)},
                        "trade_floor_usdc": float(cfg.trade_floor_usdc),
                        "profit_siphon_frac": float(cfg.profit_siphon_frac),
                        "bank_threshold_usdc": float(getattr(cfg, "bank_threshold_usdc", float("nan"))),
                        "liquidation_recap_bank_frac": float(getattr(cfg, "liquidation_recap_bank_frac", float("nan"))),
                        "max_leverage_cap": ("auto" if int(cfg.max_leverage) <= 0 else int(cfg.max_leverage)),
                        "notes": (
                            "profit_usdc is computed as trading equity delta (after close, before siphon/topup). "
                            "siphon/topup are internal transfers between bank/trading subaccounts. "
                            "This log is hash-chained (prev_hash -> hash) for tamper-evidence."
                        ),
                    }
                )

            runner = DydxRunner(
                market=str(cfg.market),
                models=models,
                target_frac=float(args.target_frac),
                hold_min=int(args.hold_min),
                out_dir=Path(out_dir),
                trade_mode=str(args.trade_mode),
                backfill_hours=int(args.backfill_hours),
                thresholds_csv=thresholds_csv,
                audit=audit,
                bank_log=bank_log,
                cfg=cfg,
                clients=clients,
                loop=loop,
                exit_gap=exit_gap,
                exit_gap_tau=float(getattr(args, "exit_gap_tau", 0.10)),
                exit_gap_min_exit_k=int(getattr(args, "exit_gap_min_exit_k", 2)),
                thresholds_live_out=thresholds_live_out,
                prior_scores_path=prior_scores_path,
                max_prior_scores=int(args.max_prior_scores),
                min_prior_scores=int(getattr(args, "min_prior_scores", 2000)),
                seed_days=int(getattr(args, "seed_days", 2)),
                retention_days=int(args.retention_days),
                metrics_port=int(args.metrics_port),
                health_max_staleness_sec=float(args.health_max_staleness_sec),
                initial_topup_budget_usdc=float(getattr(args, "initial_topup_budget_usdc", 50.0)),
            )

            # Start health/metrics server (optional)
            if int(getattr(args, "metrics_port", 0) or 0) > 0:
                threading.Thread(target=runner._serve_http, daemon=True).start()
                runner._alert("http_server", "HTTP server started", port=int(args.metrics_port))

            # Long-run housekeeping
            try:
                runner._cleanup_out_dir()
            except Exception:
                pass

            # Safety: on startup in real mode, flatten any leftover position.
            if str(args.trade_mode).lower() == "real":
                runner.reconcile_startup()

            replay_csv = str(getattr(args, "replay_csv", "") or "").strip()
            if replay_csv:
                # Replay is meant for maximum parity testing without relying on WS.
                if str(args.trade_mode).lower() != "paper":
                    raise SystemExit("--replay-csv is only supported with --trade-mode paper")

                p = Path(replay_csv)
                if not p.exists():
                    raise SystemExit(f"replay csv not found: {p}")

                df = pd.read_csv(p)
                # Accept either 'timestamp' or dYdX-style 'startedAt'
                if "timestamp" not in df.columns and "startedAt" in df.columns:
                    df["timestamp"] = df["startedAt"]
                if "volume" not in df.columns and "baseTokenVolume" in df.columns:
                    df["volume"] = df["baseTokenVolume"]

                need = ["timestamp", "open", "high", "low", "close", "volume"]
                missing = [c for c in need if c not in df.columns]
                if missing:
                    raise SystemExit(f"replay csv missing columns: {missing}")

                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.sort_values("timestamp").reset_index(drop=True)

                max_n = int(getattr(args, "replay_max_bars", 0) or 0)
                if max_n > 0:
                    df = df.iloc[:max_n].copy().reset_index(drop=True)

                sleep_s = float(getattr(args, "replay_sleep_sec", 0.0) or 0.0)

                print("Replay CSV:", str(p))
                print("Replay bars:", int(len(df)))

                prefill_n = int(max(0, int(getattr(args, "backfill_hours", 0) or 0)) * 60)
                prefill_n = min(int(prefill_n), int(len(df)))

                # Match live startup behavior: set an initial backfill window all at once, then
                # replay subsequent bars through the normal on_closed_bar pipeline.
                if prefill_n > 0:
                    pre = df.iloc[:prefill_n][need].copy()
                    pre["timestamp"] = pd.to_datetime(pre["timestamp"], utc=True)
                    for c in ["open", "high", "low", "close", "volume"]:
                        pre[c] = pd.to_numeric(pre[c], errors="coerce").astype(float)

                    runner.bars = pre.reset_index(drop=True)

                    # Persist the backfill chunk by date (same as warm_backfill).
                    try:
                        for d, g in runner.bars.assign(date=lambda x: pd.to_datetime(x["timestamp"]).dt.date).groupby("date"):
                            p2 = Path(out_dir) / f"minute_bars_{d}.csv"
                            p2.parent.mkdir(parents=True, exist_ok=True)
                            g.drop(columns=["date"]).to_csv(p2, index=False)
                    except Exception:
                        pass

                for ts, o, h, l, c, v in df.iloc[prefill_n:][need].itertuples(index=False, name=None):
                    runner.on_closed_bar(
                        {
                            "timestamp": pd.to_datetime(ts, utc=True),
                            "open": float(o),
                            "high": float(h),
                            "low": float(l),
                            "close": float(c),
                            "volume": float(v),
                        },
                        allow_trade_logic=True,
                    )
                    if sleep_s > 0:
                        time.sleep(float(sleep_s))

                runner.flush_outputs()
                raise SystemExit(0)

            # Warm backfill
            bars = loop.run_until_complete(_warm_backfill(clients, market=str(cfg.market), hours=int(args.backfill_hours)))

            seed_csv = Path(args.seed_csv) if str(getattr(args, "seed_csv", "") or "").strip() else None
            if bars.empty and seed_csv is not None and seed_csv.exists():
                try:
                    df = pd.read_csv(seed_csv, comment="#")
                    # Accept either 'timestamp' or dYdX-style 'startedAt'
                    if "timestamp" not in df.columns and "startedAt" in df.columns:
                        df["timestamp"] = df["startedAt"]
                    if "volume" not in df.columns and "baseTokenVolume" in df.columns:
                        df["volume"] = df["baseTokenVolume"]

                    need = ["timestamp", "open", "high", "low", "close", "volume"]
                    df = df[need].copy()
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                    for c in ["open", "high", "low", "close", "volume"]:
                        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

                    # Keep last <backfill-hours> worth of bars
                    end_ts = df["timestamp"].max()
                    start_ts = end_ts - timedelta(hours=int(args.backfill_hours))
                    bars = df[df["timestamp"] >= start_ts].copy().sort_values("timestamp").reset_index(drop=True)

                    if not bars.empty:
                        print("Warm backfill failed; seeded from CSV:", str(seed_csv))
                except Exception as e:
                    print("Seed CSV load failed:", e)

            if not bars.empty:
                runner.bars = bars
                # Persist backfill by date
                for d, g in bars.assign(date=lambda x: pd.to_datetime(x["timestamp"]).dt.date).groupby("date"):
                    p = Path(out_dir) / f"minute_bars_{d}.csv"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    g.drop(columns=["date"]).to_csv(p, index=False)

            # WS candles → queue
            q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=10_000)
            stop_evt = threading.Event()

            ws_url = str(clients.network.websocket_indexer)
            t = threading.Thread(
                target=_run_ws_candles,
                kwargs={"ws_url": ws_url, "market": str(cfg.market), "out_q": q, "stop_evt": stop_evt},
                daemon=True,
            )
            t.start()

            print("dYdX WS:", ws_url)
            print("Market:", cfg.market)
            print("Trade mode:", args.trade_mode)
            print("Out dir:", args.out_dir)
            print("Trade floor USDC:", float(cfg.trade_floor_usdc))
            print("Profit siphon frac:", float(cfg.profit_siphon_frac))
            print("Bank threshold USDC:", float(getattr(cfg, "bank_threshold_usdc", float("nan"))))
            print("Liquidation recap bank frac:", float(getattr(cfg, "liquidation_recap_bank_frac", float("nan"))))
            print("Max leverage cap:", "auto" if int(cfg.max_leverage) <= 0 else int(cfg.max_leverage))
            if audit_log:
                print("Audit:", audit_log)
            if bank_log_path:
                print("Bank log:", bank_log_path)

            import signal

            # Avoid occasional stdlib queue.Condition edge-case on Ctrl-C where a KeyboardInterrupt
            # during q.get() can surface as: RuntimeError: release unlocked lock.
            # Instead of raising KeyboardInterrupt inside q.get(), we translate SIGINT into a
            # shutdown request and let the loop exit on the next timeout.
            prev_sigint = signal.getsignal(signal.SIGINT)

            def _handle_sigint(signum, frame):  # type: ignore[no-untyped-def]
                if not stop_evt.is_set():
                    print("\nSIGINT received; stopping...")
                stop_evt.set()

            signal.signal(signal.SIGINT, _handle_sigint)

            exit_code = 0
            stall_seconds = int(getattr(args, "stall_seconds", 0) or 0)

            try:
                while not stop_evt.is_set():
                    try:
                        bar = q.get(timeout=5.0)
                    except queue.Empty:
                        # Stall watchdog: exit so Docker/systemd can restart.
                        if stall_seconds > 0:
                            age = float(time.time() - float(getattr(runner, "last_closed_wall", 0.0)))
                            if age > float(stall_seconds):
                                print(f"Feed stalled for {age:.1f}s (> {stall_seconds}s); exiting for restart...")
                                try:
                                    runner._alert(
                                        "feed_stall_exit",
                                        "No closed 1m bars recently; exiting",
                                        seconds_since_last=round(age, 1),
                                        stall_seconds=int(stall_seconds),
                                    )
                                except Exception:
                                    pass
                                stop_evt.set()
                                exit_code = 2
                        continue

                    runner.on_closed_bar(bar)

                    # Periodic flush
                    if len(runner.records) and (len(runner.records) % 20 == 0):
                        runner.flush_outputs()

            finally:
                stop_evt.set()

                # Safety: if we're stopping in real mode and we still have an open trade tracked,
                # attempt to flatten it with a reduce-only market order.
                if str(args.trade_mode).lower() == "real":
                    try:
                        tr = runner.open_trade
                        if isinstance(tr, dict) and str(tr.get("trade_mode") or "").lower() == "real":
                            live_sz = runner._safe_get_live_position_size_btc()
                            size_btc = float(abs(float(live_sz))) if live_sz is not None else float(tr.get("size_btc") or 0.0)
                            if size_btc > 0.0:
                                print(f"Shutdown: closing open position reduce-only (size_btc={size_btc})...")
                                runner.loop.run_until_complete(runner._close_real_long(size_btc=size_btc))
                    except Exception as e:
                        print("Shutdown close failed:", e)

                try:
                    runner.flush_outputs()
                finally:
                    try:
                        signal.signal(signal.SIGINT, prev_sigint)
                    except Exception:
                        pass

            if int(exit_code) != 0:
                raise SystemExit(int(exit_code))


if __name__ == "__main__":
    main()
