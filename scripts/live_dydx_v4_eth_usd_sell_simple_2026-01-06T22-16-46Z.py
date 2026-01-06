#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-06T22:16:46Z
"""Simplified dYdX v4 ETH-USD perpetual runner (SELL-only).

Goals (keep it boring):
- Same 1m closed-bar feed style as other runners.
- Entry: entry model score vs *explicit* threshold (no daily quantiles by default).
- Exit: exit-gap model minute-by-minute (pred_gap<=tau for k>=min_exit_k) else force at hold_min.
- When action is enter -> attempt to open; when action is exit -> attempt to close.
- Bankroll: reuse the existing two-subaccount scheme (profit siphon + floor topups).
- Audit: write contract-validated JSONL events (market_bar_1m_closed, entry_decision, trade_closed).

WARNING: In --trade-mode real, this places real orders and transfers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.append(str(REPO_ROOT))

from contracts import AuditLogger
from contracts.v1 import (
    KIND_ENTRY_DECISION,
    KIND_MARKET_BAR_1M_CLOSED,
    KIND_RUN_META,
    KIND_TRADE_CLOSED,
)

from dydx_adapter import load_dotenv
from dydx_adapter.v4 import DydxV4Config, connect_v4, new_client_id, usdc_to_quantums

from dydx_v4_client import OrderFlags
from dydx_v4_client.indexer.candles_resolution import CandlesResolution
from dydx_v4_client.indexer.rest.constants import OrderType
from dydx_v4_client.indexer.rest.indexer_client import IndexerClient
from dydx_v4_client.indexer.socket.websocket import IndexerSocket
from dydx_v4_client.node.client import QueryNodeClient
from dydx_v4_client.node.market import Market
from dydx_v4_client.node.message import subaccount

from v4_proto.dydxprotocol.clob.order_pb2 import Order

from binance_adapter.crossref_oracle_vs_agent_patterns import compute_feature_frame


# Fees: per-side fee (0.0005 = 0.05% per side; 0.1% round trip)
DEFAULT_FEE_SIDE = 0.0005


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _iso_utc(ts_like: Any) -> str:
    return pd.to_datetime(ts_like, utc=True).to_pydatetime().isoformat().replace("+00:00", "Z")


def _finite_or_none(x: Any) -> Optional[float]:
    try:
        fx = float(x)
    except Exception:
        return None
    return fx if np.isfinite(fx) else None


def _pb_to_jsonable(x: Any) -> Any:
    try:
        return QueryNodeClient.transcode_response(x)
    except Exception:
        return str(x)


def net_mult_sell(entry_px: float, exit_px: float, fee_side: float) -> float:
    entry_px = max(1e-12, float(entry_px))
    exit_px = max(1e-12, float(exit_px))
    gross_mult = entry_px / exit_px
    return float(gross_mult) * (1.0 - float(fee_side)) / (1.0 + float(fee_side))


def net_ret_pct_sell(entry_px: float, exit_px: float, fee_side: float) -> float:
    return float((net_mult_sell(entry_px, exit_px, fee_side) - 1.0) * 100.0)


# ---- dYdX helpers (copied from the existing ETH sell runner for consistency) ----


def _imf_to_max_leverage(imf: float) -> int:
    if not np.isfinite(imf) or imf <= 0:
        return 1
    return max(1, int(1.0 / float(imf)))


async def _fetch_market_meta(clients, ticker: str) -> Dict[str, Any]:
    resp = await clients.indexer.markets.get_perpetual_markets(market=str(ticker))
    mkts = resp.get("markets") if isinstance(resp, dict) else None
    if isinstance(mkts, dict) and ticker in mkts:
        return dict(mkts[ticker])
    if isinstance(mkts, dict) and len(mkts) == 1:
        return dict(list(mkts.values())[0])
    raise RuntimeError(f"Failed to fetch market meta for {ticker}: keys={list((resp or {}).keys())}")


async def _get_subaccount_equity_usdc(clients, *, address: str, subaccount_number: int) -> float:
    res = await clients.indexer.account.get_subaccount(address, int(subaccount_number))
    if not isinstance(res, dict):
        raise RuntimeError(f"Unexpected subaccount response type: {type(res)}")

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


async def _get_open_position_size_base(clients, *, address: str, subaccount_number: int, market: str) -> float:
    res = await clients.indexer.account.get_subaccount(address, int(subaccount_number))
    sub = res.get("subaccount") if isinstance(res, dict) and isinstance(res.get("subaccount"), dict) else None
    opp = sub.get("openPerpetualPositions") if isinstance(sub, dict) else None
    if not isinstance(opp, dict) or not opp:
        return 0.0

    p = opp.get(str(market).upper())
    if not isinstance(p, dict):
        return 0.0

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
    min_size_base: float,
    timeout_s: float = 10.0,
    poll_s: float = 0.5,
) -> Dict[str, Any]:
    side_u = str(side).upper()
    want = max(0.0, float(min_size_base))
    deadline = time.time() + float(timeout_s)

    best: Dict[str, Any] = {
        "found": False,
        "side": side_u,
        "market": str(market).upper(),
        "since_time_utc": since_dt.isoformat(),
        "size_base": 0.0,
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
                "size_base": float(qty),
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
            self._cur = dict(row)
            return None

        closed = dict(self._cur or {})
        self._cur_started_at = ts
        self._cur = dict(row)
        return closed


def _run_ws_candles(
    *,
    ws_url: str,
    market: str,
    out_q: "queue.Queue[Dict[str, Any]]",
    stop_evt: threading.Event,
) -> None:
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
            if t in {"connected", "ping"}:
                return

            if t == "subscribed":
                candles = ((msg.get("contents") or {}).get("candles") or [])
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
        backoff = 1.0 if dur >= 60.0 else min(float(backoff) * 2.0, 60.0)
        print(f"ws_reconnect_sleep_s={backoff:.1f}")
        time.sleep(float(backoff))


async def _warm_backfill(clients, *, market: str, hours: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(hours=int(hours))

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
            df["timestamp"] = pd.to_datetime(df["startedAt"], utc=True)
            for c in ["open", "high", "low", "close"]:
                df[c] = df[c].astype(float)
            df["volume"] = df["baseTokenVolume"].astype(float)
            frames.append(df[["timestamp", "open", "high", "low", "close", "volume"]])
        s = e
        await asyncio.sleep(0.05)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    bars = pd.concat(frames, ignore_index=True)
    bars = bars.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return bars


# ---- Models ----


@dataclass(frozen=True)
class SellEntryModel:
    model: Any
    feature_cols: List[str]
    created_utc: Optional[str]
    pre_min: int
    artifact: str


@dataclass(frozen=True)
class SellExitGapModel:
    model: Any
    feature_cols: List[str]
    created_utc: Optional[str]
    artifact: str


def load_sell_entry_model(path: Path) -> SellEntryModel:
    p = Path(path)
    obj = joblib.load(p)
    if not isinstance(obj, dict) or "model" not in obj or "feature_cols" not in obj:
        raise SystemExit(f"Unexpected SELL entry model artifact format: {p}")
    pre_min = int(obj.get("pre_min") or 0)
    return SellEntryModel(
        model=obj["model"],
        feature_cols=[str(c) for c in list(obj.get("feature_cols") or [])],
        created_utc=obj.get("created_utc"),
        pre_min=pre_min,
        artifact=str(p.name),
    )


def load_sell_exit_gap_model(path: Path) -> SellExitGapModel:
    p = Path(path)
    obj = joblib.load(p)
    if not isinstance(obj, dict) or "model" not in obj or "feature_cols" not in obj:
        raise SystemExit(f"Unexpected SELL exit gap model artifact format: {p}")
    return SellExitGapModel(
        model=obj["model"],
        feature_cols=[str(c) for c in list(obj.get("feature_cols") or [])],
        created_utc=obj.get("created_utc"),
        artifact=str(p.name),
    )


@dataclass
class OpenTrade:
    trade_id: str
    trade_mode: str
    entry_time_utc: str
    entry_price: float
    entry_score: float
    entry_threshold: float
    size_base: float

    mins_in_trade: int = 0
    ret_hist: List[float] = field(default_factory=list)
    peak_ret: float = -1e30


class FeatureComputer:
    """Recompute base+ctx series and build precontext feature maps at index i."""

    def __init__(self, *, base_feats: List[str], ctx_windows: List[int]) -> None:
        self.base_feats = list(base_feats)
        self.ctx_windows = [int(w) for w in ctx_windows]

        self.bars: Optional[pd.DataFrame] = None
        self.base: Optional[pd.DataFrame] = None
        self.ctx: Optional[pd.DataFrame] = None

    def set_bars(self, bars: pd.DataFrame) -> None:
        self.bars = bars

    def recompute(self) -> None:
        if self.bars is None or self.bars.empty:
            self.base = None
            self.ctx = None
            return

        bars = self.bars.copy().sort_values("timestamp").reset_index(drop=True)
        bars["ts_min"] = pd.to_datetime(bars["timestamp"], utc=True)

        base_full = compute_feature_frame(bars)
        keep_base = [c for c in self.base_feats if c in base_full.columns]
        self.base = base_full[keep_base].copy().reset_index(drop=True)

        close_prev = pd.to_numeric(bars["close"], errors="coerce").shift(1)
        high_prev = pd.to_numeric(bars["high"], errors="coerce").shift(1)
        low_prev = pd.to_numeric(bars["low"], errors="coerce").shift(1)
        vol_prev = pd.to_numeric(bars["volume"], errors="coerce").shift(1)

        ctx_cols: Dict[str, pd.Series] = {}
        for w in self.ctx_windows:
            ctx_cols[f"mom_{w}m_pct"] = close_prev.pct_change(w) * 100.0
            ctx_cols[f"vol_std_{w}m"] = close_prev.rolling(w, min_periods=w).std(ddof=0)

            rng = (high_prev.rolling(w, min_periods=w).max() - low_prev.rolling(w, min_periods=w).min())
            ctx_cols[f"range_{w}m"] = rng
            ctx_cols[f"range_norm_{w}m"] = rng / close_prev.replace(0, np.nan)

            cmax = close_prev.rolling(w, min_periods=w).max()
            cmin = close_prev.rolling(w, min_periods=w).min()
            crng = cmax - cmin
            eps = 1e-9

            cp = close_prev
            ctx_cols[f"close_dd_from_{w}m_max_pct"] = (cmax / (cp.clip(lower=1e-9)) - 1.0) * 100.0
            ctx_cols[f"close_bounce_from_{w}m_min_pct"] = (cp / (cmin.clip(lower=1e-9)) - 1.0) * 100.0
            ctx_cols[f"close_pos_in_{w}m_range"] = (cp - cmin) / (crng + eps)

            v = pd.to_numeric(vol_prev, errors="coerce").fillna(0.0)
            cp2 = pd.to_numeric(cp, errors="coerce").fillna(0.0)
            sum_v = v.rolling(w, min_periods=w).sum()
            sum_pv = (v * cp2).rolling(w, min_periods=w).sum()
            vwap = sum_pv / (sum_v.clip(lower=1e-9))
            ctx_cols[f"vwap_dev_{w}m"] = np.where(sum_v > 0.0, ((cp2 - vwap) / (vwap.clip(lower=1e-9))) * 100.0, 0.0)

        self.ctx = pd.DataFrame(ctx_cols).reset_index(drop=True)

    @staticmethod
    def _agg5_from_window(win5: np.ndarray) -> Dict[str, float]:
        w = np.asarray(win5, dtype=np.float64)
        if w.shape != (5,):
            return {}

        def _safe_std(vals: np.ndarray) -> float:
            v = np.asarray(vals, dtype=np.float64)
            m = np.nanmean(v)
            if not np.isfinite(m):
                return float("nan")
            return float(np.nanmean((v - m) ** 2) ** 0.5)

        def _slope5_window(vals: np.ndarray) -> float:
            v = np.asarray(vals, dtype=np.float64)
            xc = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
            return float(np.nansum(v * xc) / 10.0)

        return {
            "last": float(w[-1]),
            "mean5": float(np.nanmean(w)),
            "std5": float(_safe_std(w)),
            "min5": float(np.nanmin(w)),
            "max5": float(np.nanmax(w)),
            "range5": float(np.nanmax(w) - np.nanmin(w)),
            "slope5": float(_slope5_window(w)),
        }

    def feature_map_at(self, i: int) -> Dict[str, float]:
        if self.bars is None or self.base is None or self.ctx is None:
            return {}

        i = int(i)
        if i < 5:
            return {}

        j0 = i - 5
        j1 = i

        close = pd.to_numeric(self.bars["close"], errors="coerce").to_numpy(np.float64)
        vol = pd.to_numeric(self.bars["volume"], errors="coerce").to_numpy(np.float64)

        close_prev = close[j0:j1]
        if close_prev.shape != (5,):
            return {}

        close_last = float(close_prev[-1])
        if not (np.isfinite(close_last) and close_last > 0.0):
            return {}

        close_norm = (close_prev / close_last - 1.0) * 100.0
        px = self._agg5_from_window(close_norm)

        px_ret5m = float(close_norm[-1] - close_norm[0])
        px_absret5m = float(abs(px_ret5m))

        vol_prev = vol[j0:j1]
        vol_log = np.log1p(np.maximum(0.0, vol_prev.astype(np.float64)))
        vv = self._agg5_from_window(vol_log)

        out: Dict[str, float] = {}

        for k, v in px.items():
            out[f"px_close_norm_pct__{k}"] = float(v)
        out["px_close_norm_pct__ret5m"] = float(px_ret5m)
        out["px_close_norm_pct__absret5m"] = float(px_absret5m)
        out["px_close_norm_pct__m5"] = float(close_norm[0])
        out["px_close_norm_pct__m4"] = float(close_norm[1])
        out["px_close_norm_pct__m3"] = float(close_norm[2])
        out["px_close_norm_pct__m2"] = float(close_norm[3])
        out["px_close_norm_pct__m1"] = float(close_norm[4])

        for k, v in vv.items():
            out[f"vol_log1p__{k}"] = float(v)

        for c in self.base_feats:
            if c not in self.base.columns:
                continue
            arr = pd.to_numeric(self.base[c], errors="coerce").to_numpy(np.float64)
            w = arr[j0:j1]
            agg = self._agg5_from_window(w)
            for k, v in agg.items():
                out[f"{c}__{k}"] = float(v)

        for c in self.ctx.columns:
            arr = pd.to_numeric(self.ctx[c], errors="coerce").to_numpy(np.float64)
            w = arr[j0:j1]
            agg = self._agg5_from_window(w)
            for k, v in agg.items():
                out[f"{c}__{k}"] = float(v)

        out["missing_close_n"] = float(np.sum(~np.isfinite(close_prev)))

        miss_any = bool(np.any(~np.isfinite(close_prev)))
        if not miss_any:
            for c in self.base_feats:
                if c in self.base.columns:
                    v = pd.to_numeric(self.base[c], errors="coerce").to_numpy(np.float64)[j0:j1]
                    if np.any(~np.isfinite(v)):
                        miss_any = True
                        break
        if not miss_any:
            for c in self.ctx.columns:
                v = pd.to_numeric(self.ctx[c], errors="coerce").to_numpy(np.float64)[j0:j1]
                if np.any(~np.isfinite(v)):
                    miss_any = True
                    break
        out["missing_any"] = float(1.0 if miss_any else 0.0)

        return out


class EthSellSimpleRunner:
    def __init__(
        self,
        *,
        market: str,
        cfg: DydxV4Config,
        clients,
        loop: asyncio.AbstractEventLoop,
        entry: SellEntryModel,
        exit_gap: SellExitGapModel,
        trade_mode: str,
        out_dir: Path,
        audit: AuditLogger,
        entry_threshold: float,
        hold_min: int,
        exit_gap_tau: float,
        exit_gap_min_exit_k: int,
        fee_side: float,
        log_every: int,
    ) -> None:
        self.market = str(market).upper()
        self.cfg = cfg
        self.clients = clients
        self.loop = loop

        self.entry = entry
        self.exit_gap = exit_gap

        self.trade_mode = str(trade_mode).lower()
        self.entry_threshold = float(entry_threshold)

        self.log_every = int(log_every)

        self.hold_min = int(hold_min)
        self.exit_gap_tau = float(exit_gap_tau)
        self.exit_gap_min_exit_k = int(exit_gap_min_exit_k)

        self.fee_side = float(fee_side)

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.audit = audit

        self.bars: Optional[pd.DataFrame] = None
        self.features = FeatureComputer(
            base_feats=[
                "ret_1m_pct",
                "mom_3m_pct",
                "mom_5m_pct",
                "vol_std_5m",
                "range_5m",
                "range_norm_5m",
                "macd",
                "vwap_dev_5m",
            ],
            ctx_windows=[30, 60, 120],
        )

        self.open_trade: Optional[OpenTrade] = None
        self._trade_seq = 0
        self.trades: List[Dict[str, Any]] = []

        # Budget state (reused naming convention from existing ETH sell runner)
        st = str(self.market).lower().replace("-", "_")
        self.budget_state_path = self.out_dir / f"topup_budget_state_{st}_sell_tr{int(self.cfg.subaccount_trading)}_bank{int(self.cfg.subaccount_bank)}.json"
        self.initial_topup_budget_usdc = float(getattr(self.cfg, "trade_floor_usdc", 10.0)) * 0.0  # informational only
        self.initial_topup_spent_usdc = 0.0
        self._load_budget_state()

        self.last_closed_wall = time.time()

    def _load_budget_state(self) -> None:
        try:
            if self.budget_state_path.exists():
                obj = json.loads(self.budget_state_path.read_text(encoding="utf-8"))
                self.initial_topup_spent_usdc = float(obj.get("initial_topup_spent_usdc") or obj.get("spent_usdc") or 0.0)
        except Exception:
            self.initial_topup_spent_usdc = 0.0

    def _save_budget_state(self) -> None:
        try:
            self.budget_state_path.write_text(
                json.dumps(
                    {
                        "initial_topup_budget_usdc": float(self.initial_topup_budget_usdc),
                        "initial_topup_spent_usdc": float(self.initial_topup_spent_usdc),
                        "updated_utc": _now_ts(),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    def _append_bar(self, bar: Mapping[str, Any]) -> int:
        row = {
            "timestamp": pd.to_datetime(bar["timestamp"], utc=True),
            "open": float(bar["open"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "close": float(bar["close"]),
            "volume": float(bar.get("volume") or 0.0),
        }

        if self.bars is None or self.bars.empty:
            self.bars = pd.DataFrame([row])
        else:
            self.bars = pd.concat([self.bars, pd.DataFrame([row])], ignore_index=True)

        self.bars["timestamp"] = pd.to_datetime(self.bars["timestamp"], utc=True)
        self.bars = self.bars.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)

        # Keep enough history for ctx120
        if len(self.bars) > 12000:
            self.bars = self.bars.iloc[-12000:].copy().reset_index(drop=True)

        self.features.set_bars(self.bars)
        self.features.recompute()

        return int(len(self.bars) - 1)

    def _entry_score_at(self, i: int) -> Optional[float]:
        feat = self.features.feature_map_at(i)
        if not feat:
            return None
        x = np.asarray([float(feat.get(c, float("nan"))) for c in self.entry.feature_cols], dtype=np.float32)
        try:
            s = float(self.entry.model.predict(np.asarray([x], dtype=np.float32))[0])
        except Exception:
            return None
        return s if np.isfinite(s) else None

    def _exit_gap_pred_at(
        self,
        *,
        i: int,
        mins_in_trade: int,
        cur_ret: float,
        r_prev1: float,
        r_prev2: float,
        r_prev3: float,
        peak_ret: float,
    ) -> Optional[float]:
        feat = self.features.feature_map_at(i)
        if not feat:
            return None

        full = dict(feat)
        full.update(
            {
                "mins_in_trade": float(mins_in_trade),
                "mins_remaining": float(max(0, int(self.hold_min) - int(mins_in_trade))),
                "delta_mark_pct": float(cur_ret),
                "delta_mark_prev1_pct": float(r_prev1),
                "delta_mark_prev2_pct": float(r_prev2),
                "delta_mark_change_1m": float(cur_ret - r_prev1) if np.isfinite(r_prev1) else float("nan"),
                "delta_mark_change_2m": float(cur_ret - r_prev2) if np.isfinite(r_prev2) else float("nan"),
                "delta_mark_change_3m": float(cur_ret - r_prev3) if np.isfinite(r_prev3) else float("nan"),
                "drawdown_from_peak_pct": float(cur_ret - peak_ret),
            }
        )

        x = np.asarray([float(full.get(c, float("nan"))) for c in self.exit_gap.feature_cols], dtype=np.float32)
        try:
            p = float(self.exit_gap.model.predict(np.asarray([x], dtype=np.float32))[0])
        except Exception:
            return None
        return p if np.isfinite(p) else None

    async def _ensure_trading_floor(self) -> Dict[str, Any]:
        try:
            trading_eq = float(await _get_subaccount_equity_usdc(self.clients, address=self.cfg.address, subaccount_number=self.cfg.subaccount_trading))
        except Exception:
            trading_eq = 0.0

        floor = float(self.cfg.trade_floor_usdc)
        if trading_eq >= floor:
            return {"topped_up": False, "needed_usdc": 0.0, "transferred_usdc": 0.0}

        needed = float(floor - trading_eq)
        try:
            bank_eq = float(await _get_subaccount_equity_usdc(self.clients, address=self.cfg.address, subaccount_number=self.cfg.subaccount_bank))
        except Exception:
            bank_eq = 0.0

        transfer = min(float(needed), float(bank_eq))
        if transfer <= 0.0:
            return {
                "topped_up": False,
                "needed_usdc": float(needed),
                "transferred_usdc": 0.0,
                "skipped": True,
                "skip_reason": "no_transfer_possible",
                "bank_equity_usdc": float(bank_eq),
                "trading_equity_usdc": float(trading_eq),
            }

        amt_q = usdc_to_quantums(float(transfer))
        tx = await self.clients.node.transfer(
            self.clients.wallet,
            subaccount(self.cfg.address, int(self.cfg.subaccount_bank)),
            subaccount(self.cfg.address, int(self.cfg.subaccount_trading)),
            int(self.clients.usdc_asset_id),
            int(amt_q),
        )

        self.initial_topup_spent_usdc += float(transfer)
        self._save_budget_state()

        return {
            "topped_up": True,
            "needed_usdc": float(needed),
            "transferred_usdc": float(transfer),
            "bank_equity_usdc": float(bank_eq),
            "tx": _pb_to_jsonable(tx),
        }

    async def _siphon_profit_and_topup(self, *, equity_before: float) -> Dict[str, Any]:
        equity_after = float(await _get_subaccount_equity_usdc(self.clients, address=self.cfg.address, subaccount_number=self.cfg.subaccount_trading))
        profit = float(equity_after - float(equity_before))

        siphon = {"attempted": False, "profit_usdc": float(profit), "requested_usdc": 0.0, "transferred_usdc": 0.0}
        if profit > 0:
            frac = float(self.cfg.profit_siphon_frac)
            requested = max(0.0, float(profit) * float(frac))
            requested = min(requested, float(equity_after))
            if requested > 0:
                bank_eq = float(await _get_subaccount_equity_usdc(self.clients, address=self.cfg.address, subaccount_number=self.cfg.subaccount_bank))
                amt_q = usdc_to_quantums(requested)
                tx = await self.clients.node.transfer(
                    self.clients.wallet,
                    subaccount(self.cfg.address, int(self.cfg.subaccount_trading)),
                    subaccount(self.cfg.address, int(self.cfg.subaccount_bank)),
                    int(self.clients.usdc_asset_id),
                    int(amt_q),
                )
                siphon = {
                    "attempted": True,
                    "profit_usdc": float(profit),
                    "requested_usdc": float(requested),
                    "transferred_usdc": float(requested),
                    "bank_equity_usdc": float(bank_eq),
                    "tx": _pb_to_jsonable(tx),
                }

        topup = await self._ensure_trading_floor()
        return {
            "equity_before_usdc": float(equity_before),
            "equity_after_usdc": float(equity_after),
            "profit_usdc": float(profit),
            "siphon": siphon,
            "topup": topup,
        }

    async def _open_real_short(self, *, equity_before: float) -> Dict[str, Any]:
        mk = await _fetch_market_meta(self.clients, self.market)
        imf = float(mk.get("initialMarginFraction") or 1.0)
        max_lev_imf = int(_imf_to_max_leverage(imf))
        user_cap = int(getattr(self.cfg, "max_leverage", 0) or 0)
        max_lev = int(max_lev_imf) if user_cap <= 0 else min(int(user_cap), int(max_lev_imf))

        px = float(mk.get("oraclePrice") or 0.0)
        if px <= 0:
            raise RuntimeError("oraclePrice not positive")

        try:
            slip = float(os.getenv("DYDX_ORDER_SLIPPAGE_FRAC", "0.002"))
        except Exception:
            slip = 0.002
        slip = max(0.0, min(0.05, slip))
        limit_px = float(px) * (1.0 - slip)

        margin_usdc = float(equity_before) * float(self.cfg.use_margin_frac)
        margin_usdc = max(0.0, float(margin_usdc))

        max_notional_imf = float(margin_usdc) / max(1e-12, float(imf))
        max_notional_user = float("inf")
        if int(user_cap) > 0:
            max_notional_user = float(margin_usdc) * float(max_lev)

        notional_usd = float(min(max_notional_imf, max_notional_user)) * float(self.cfg.leverage_safety_frac)
        if notional_usd <= 0:
            raise RuntimeError("Computed notional_usd <= 0")

        size_base = float(notional_usd) / float(px)

        mkt = Market(mk)
        coid = int(new_client_id())
        oid = mkt.order_id(
            address=self.cfg.address,
            subaccount_number=int(self.cfg.subaccount_trading),
            client_id=int(coid),
            order_flags=int(OrderFlags.SHORT_TERM),
        )

        quantums = int(mkt.calculate_quantums(float(size_base)))
        size_base_q = float(quantums) * (10 ** float(mk["atomicResolution"]))

        good_til_block = int(await self.clients.node.latest_block_height()) + 20

        order = mkt.order(
            order_id=oid,
            order_type=OrderType.MARKET,
            side=Order.SIDE_SELL,
            size=float(size_base_q),
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
            side="SELL",
            since_dt=sent_at - timedelta(seconds=5),
            min_size_base=float(size_base_q),
            timeout_s=10.0,
            poll_s=0.5,
        )

        fill_assumed = False
        filled_sz = float(fill.get("size_base") or 0.0)
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
                min_size_base=float(1e-12),
                timeout_s=30.0,
                poll_s=1.0,
            )
            if float(fill2.get("size_base") or 0.0) > 0.0 and float(fill2.get("vwap_price") or 0.0) > 0.0:
                fill = fill2
                filled_sz = float(fill.get("size_base") or 0.0)
                filled_px = float(fill.get("vwap_price") or 0.0)
                fee_usdc = float(fill.get("fee_usdc") or 0.0)

        if filled_sz <= 0.0 or filled_px <= 0.0:
            try:
                pos_sz = float(
                    await _get_open_position_size_base(
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
            "fill_price": float(filled_px) if filled_px > 0 else float(px),
            "fill_size_base": float(filled_sz) if filled_sz > 0 else float(size_base_q),
            "fee_usdc": float(fee_usdc),
            "fill_assumed": bool(fill_assumed),
            "imf": float(imf),
            "max_lev": int(max_lev),
            "margin_usdc": float(margin_usdc),
            "notional_usd": float(notional_usd),
            "size_base": float(size_base_q),
            "size_base_quantums": int(quantums),
        }

    async def _close_real_short(self, *, size_base: float) -> Dict[str, Any]:
        mk = await _fetch_market_meta(self.clients, self.market)
        px = float(mk.get("oraclePrice") or 0.0)

        try:
            slip = float(os.getenv("DYDX_ORDER_SLIPPAGE_FRAC", "0.002"))
        except Exception:
            slip = 0.002
        slip = max(0.0, min(0.05, slip))
        limit_px = float(px) * (1.0 + slip)

        mkt = Market(mk)

        coid = int(new_client_id())
        oid = mkt.order_id(
            address=self.cfg.address,
            subaccount_number=int(self.cfg.subaccount_trading),
            client_id=int(coid),
            order_flags=int(OrderFlags.SHORT_TERM),
        )

        good_til_block = int(await self.clients.node.latest_block_height()) + 20

        order = mkt.order(
            order_id=oid,
            order_type=OrderType.MARKET,
            side=Order.SIDE_BUY,
            size=float(size_base),
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
            side="BUY",
            since_dt=sent_at - timedelta(seconds=5),
            min_size_base=float(size_base),
            timeout_s=10.0,
            poll_s=0.5,
        )

        fill_assumed = False
        filled_sz = float(fill.get("size_base") or 0.0)
        filled_px = float(fill.get("vwap_price") or 0.0)
        fee_usdc = float(fill.get("fee_usdc") or 0.0)

        if filled_sz <= 0.0 or filled_px <= 0.0:
            fill2 = await _wait_for_fills(
                self.clients,
                address=self.cfg.address,
                subaccount_number=int(self.cfg.subaccount_trading),
                market=self.market,
                side="BUY",
                since_dt=sent_at - timedelta(seconds=5),
                min_size_base=float(1e-12),
                timeout_s=30.0,
                poll_s=1.0,
            )
            if float(fill2.get("size_base") or 0.0) > 0.0 and float(fill2.get("vwap_price") or 0.0) > 0.0:
                fill = fill2
                filled_sz = float(fill.get("size_base") or 0.0)
                filled_px = float(fill.get("vwap_price") or 0.0)
                fee_usdc = float(fill.get("fee_usdc") or 0.0)

        if filled_sz <= 0.0 or filled_px <= 0.0:
            try:
                pos_sz = float(
                    await _get_open_position_size_base(
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
                filled_sz = float(size_base)
                filled_px = float(px)

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
            "fill_price": float(filled_px) if filled_px > 0 else float(px),
            "fill_size_base": float(filled_sz) if filled_sz > 0 else float(size_base),
            "fee_usdc": float(fee_usdc),
            "fill_assumed": bool(fill_assumed),
        }

    def _audit_market_bar(self, bar: Mapping[str, Any]) -> None:
        t0 = pd.to_datetime(bar["timestamp"], utc=True)
        t1 = t0 + timedelta(minutes=1)
        self.audit.write(
            {
                "kind": KIND_MARKET_BAR_1M_CLOSED,
                "symbol": self.market,
                "source": "dydx_indexer_ws",
                "interval": "1m",
                "bar_open_time_utc": t0.to_pydatetime().isoformat(),
                "bar_close_time_utc": t1.to_pydatetime().isoformat(),
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": float(bar.get("volume") or 0.0),
            }
        )

    def _audit_entry_decision(self, *, bar: Mapping[str, Any], score: float, threshold: float, action: str, i: int) -> None:
        t0 = pd.to_datetime(bar["timestamp"], utc=True)
        t1 = t0 + timedelta(minutes=1)

        feat = self.features.feature_map_at(int(i))
        feature_names = list(self.entry.feature_cols)
        feature_values = [_finite_or_none(feat.get(c)) for c in feature_names]

        self.audit.write(
            {
                "kind": KIND_ENTRY_DECISION,
                "symbol": self.market,
                "decision_time_utc": t1.to_pydatetime().isoformat(),
                "bar_open_time_utc": t0.to_pydatetime().isoformat(),
                "bar_close_time_utc": t1.to_pydatetime().isoformat(),
                "score": float(score),
                "threshold": float(threshold),
                "action": str(action),
                "planned_entry_time_utc": t1.to_pydatetime().isoformat(),
                "policy": "fixed_threshold_next_open",
                "feature_names": feature_names,
                "feature_values": feature_values,
                "model": {
                    "role": "entry",
                    "artifact": str(self.entry.artifact),
                    "created_utc": str(self.entry.created_utc or "unknown"),
                    "features": feature_names,
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
        entry_score: Optional[float],
        entry_threshold: Optional[float],
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "kind": KIND_TRADE_CLOSED,
            "symbol": self.market,
            "paper": bool(paper),
            "entry_time_utc": str(entry_time_utc),
            "exit_time_utc": str(exit_time_utc),
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "fee_side": float(self.fee_side),
            "exit_rel_min": int(exit_rel_min),
            "realized_ret_pct": float(realized_ret_pct),
            "entry_score": _finite_or_none(entry_score),
            "entry_threshold": _finite_or_none(entry_threshold),
        }
        if extra:
            payload.update(dict(extra))
        self.audit.write(payload)

    def _flush_trades_csv(self) -> None:
        if not self.trades:
            return
        df = pd.DataFrame(self.trades)
        df.to_csv(self.out_dir / "eth_sell_simple_trades.csv", index=False)

    def on_closed_bar(self, bar: Mapping[str, Any]) -> None:
        self.last_closed_wall = float(time.time())

        try:
            self._audit_market_bar(bar)
        except Exception as e:
            # Audit should not kill the runner.
            print("audit_market_bar_failed:", e)

        i = self._append_bar(bar)
        if self.bars is None:
            return

        t0 = pd.to_datetime(self.bars.iloc[i]["timestamp"], utc=True)
        t1 = t0 + timedelta(minutes=1)

        # Exit/update if in trade
        if self.open_trade is not None:
            tr = self.open_trade

            # (Real-mode safety) If we think we're in a real trade but the exchange is flat, treat as forced close.
            if self.trade_mode == "real" and str(tr.trade_mode).lower() == "real":
                try:
                    live_sz = float(
                        self.loop.run_until_complete(
                            _get_open_position_size_base(
                                self.clients,
                                address=self.cfg.address,
                                subaccount_number=int(self.cfg.subaccount_trading),
                                market=self.market,
                            )
                        )
                    )
                except Exception:
                    live_sz = float("nan")

                dt_entry = pd.to_datetime(tr.entry_time_utc, utc=True, errors="coerce")
                age_s = float((t1 - dt_entry).total_seconds()) if pd.notna(dt_entry) else float("inf")
                if (np.isfinite(live_sz) and abs(float(live_sz)) <= 0.0) and age_s >= 60.0:
                    # Clear state; caller can decide whether to recap externally.
                    print("unexpected_flat_position: clearing open_trade")
                    self.open_trade = None
                    return

            tr.mins_in_trade = int(tr.mins_in_trade) + 1
            k = int(tr.mins_in_trade)

            exit_px = float(bar["close"])
            cur_ret = float(net_ret_pct_sell(tr.entry_price, exit_px, float(self.fee_side)))

            tr.ret_hist.append(float(cur_ret))
            tr.peak_ret = float(max(float(tr.peak_ret), float(cur_ret)))

            r_prev1 = float(tr.ret_hist[-2]) if len(tr.ret_hist) >= 2 else float("nan")
            r_prev2 = float(tr.ret_hist[-3]) if len(tr.ret_hist) >= 3 else float("nan")
            r_prev3 = float(tr.ret_hist[-4]) if len(tr.ret_hist) >= 4 else float("nan")

            pred_gap = self._exit_gap_pred_at(
                i=int(i),
                mins_in_trade=int(k),
                cur_ret=float(cur_ret),
                r_prev1=float(r_prev1),
                r_prev2=float(r_prev2),
                r_prev3=float(r_prev3),
                peak_ret=float(tr.peak_ret),
            )

            exit_reason = ""
            if int(k) >= int(self.hold_min):
                exit_reason = "hold_min"
            elif int(k) >= int(self.exit_gap_min_exit_k) and pred_gap is not None and float(pred_gap) <= float(self.exit_gap_tau):
                exit_reason = "pred_gap<=tau"

            if not exit_reason:
                return

            print(
                f"{_iso_utc(t1)} EXIT signal trade_id={tr.trade_id} k={int(k)} reason={exit_reason} cur_ret_1x_pct={float(cur_ret):.3f} pred_gap={(_finite_or_none(pred_gap))}"
            )

            # Exit now
            if self.trade_mode == "paper":
                self.trades.append(
                    {
                        "trade_id": tr.trade_id,
                        "trade_mode": "paper",
                        "symbol": self.market,
                        "direction": "SHORT",
                        "entry_time_utc": tr.entry_time_utc,
                        "exit_time_utc": _iso_utc(t1),
                        "entry_price": float(tr.entry_price),
                        "exit_price": float(exit_px),
                        "exit_rel_min": int(k),
                        "entry_score": float(tr.entry_score),
                        "entry_threshold": float(tr.entry_threshold),
                        "exit_pred_gap_pct": _finite_or_none(pred_gap),
                        "exit_gap_tau": float(self.exit_gap_tau),
                        "exit_gap_min_exit_k": int(self.exit_gap_min_exit_k),
                        "exit_reason": str(exit_reason),
                        "realized_ret_1x_pct": float(cur_ret),
                    }
                )

                try:
                    self._audit_trade_closed(
                        paper=True,
                        entry_time_utc=str(tr.entry_time_utc),
                        exit_time_utc=_iso_utc(t1),
                        entry_price=float(tr.entry_price),
                        exit_price=float(exit_px),
                        exit_rel_min=int(k),
                        realized_ret_pct=float(cur_ret),
                        entry_score=float(tr.entry_score),
                        entry_threshold=float(tr.entry_threshold),
                        extra={
                            "exit_pred_gap_pct": _finite_or_none(pred_gap),
                            "exit_gap_tau": float(self.exit_gap_tau),
                            "exit_gap_min_exit_k": int(self.exit_gap_min_exit_k),
                            "exit_reason": str(exit_reason),
                        },
                    )
                except Exception as e:
                    print("audit_trade_closed_failed:", e)

                self.open_trade = None
                self._flush_trades_csv()
                return

            # real mode exit
            try:
                equity_before = float(
                    self.loop.run_until_complete(
                        _get_subaccount_equity_usdc(self.clients, address=self.cfg.address, subaccount_number=int(self.cfg.subaccount_trading))
                    )
                )
            except Exception:
                equity_before = float("nan")

            try:
                live_sz = float(
                    self.loop.run_until_complete(
                        _get_open_position_size_base(
                            self.clients,
                            address=self.cfg.address,
                            subaccount_number=int(self.cfg.subaccount_trading),
                            market=self.market,
                        )
                    )
                )
            except Exception:
                live_sz = 0.0

            size_to_close = float(abs(live_sz)) if abs(float(live_sz)) > 0.0 else float(tr.size_base)

            close_meta = {}
            bank_ops = {}
            try:
                close_meta = self.loop.run_until_complete(self._close_real_short(size_base=float(size_to_close)))
            except Exception as e:
                print("close_real_short_failed:", e)

            try:
                if np.isfinite(float(equity_before)):
                    bank_ops = self.loop.run_until_complete(self._siphon_profit_and_topup(equity_before=float(equity_before)))
            except Exception as e:
                print("bank_ops_failed:", e)

            self.trades.append(
                {
                    "trade_id": tr.trade_id,
                    "trade_mode": "real",
                    "symbol": self.market,
                    "direction": "SHORT",
                    "entry_time_utc": tr.entry_time_utc,
                    "exit_time_utc": _iso_utc(t1),
                    "entry_price": float(tr.entry_price),
                    "exit_price": float(exit_px),
                    "exit_rel_min": int(k),
                    "entry_score": float(tr.entry_score),
                    "entry_threshold": float(tr.entry_threshold),
                    "exit_pred_gap_pct": _finite_or_none(pred_gap),
                    "exit_gap_tau": float(self.exit_gap_tau),
                    "exit_gap_min_exit_k": int(self.exit_gap_min_exit_k),
                    "exit_reason": str(exit_reason),
                    "realized_ret_1x_pct_assumed_fee": float(cur_ret),
                    "close": close_meta,
                    "bank_ops": bank_ops,
                }
            )

            try:
                self._audit_trade_closed(
                    paper=False,
                    entry_time_utc=str(tr.entry_time_utc),
                    exit_time_utc=_iso_utc(t1),
                    entry_price=float(tr.entry_price),
                    exit_price=float(exit_px),
                    exit_rel_min=int(k),
                    realized_ret_pct=float(cur_ret),
                    entry_score=float(tr.entry_score),
                    entry_threshold=float(tr.entry_threshold),
                    extra={
                        "exit_pred_gap_pct": _finite_or_none(pred_gap),
                        "exit_gap_tau": float(self.exit_gap_tau),
                        "exit_gap_min_exit_k": int(self.exit_gap_min_exit_k),
                        "exit_reason": str(exit_reason),
                        "close": close_meta,
                        "bank_ops": bank_ops,
                    },
                )
            except Exception as e:
                print("audit_trade_closed_failed:", e)

            self.open_trade = None
            self._flush_trades_csv()
            return

        # No open trade: decide entry
        score = self._entry_score_at(int(i))
        if score is None or not np.isfinite(float(score)):
            return

        thr = float(self.entry_threshold)
        action = "enter" if float(score) >= float(thr) else "hold"

        if int(getattr(self, "log_every", 0) or 0) > 0:
            if action == "enter" or (int(i) % int(self.log_every) == 0):
                print(f"{_iso_utc(t1)} entry_decision score={float(score):.6f} thr={float(thr):.6f} action={action}")

        try:
            self._audit_entry_decision(bar=bar, score=float(score), threshold=float(thr), action=str(action), i=int(i))
        except Exception as e:
            print("audit_entry_decision_failed:", e)

        if action != "enter":
            return

        if self.trade_mode == "paper":
            self._trade_seq += 1
            tr = OpenTrade(
                trade_id=f"paper_{self._trade_seq}",
                trade_mode="paper",
                entry_time_utc=_iso_utc(t1),
                entry_price=float(bar["close"]),
                entry_score=float(score),
                entry_threshold=float(thr),
                size_base=0.0,
            )
            self.open_trade = tr
            print(f"{_iso_utc(t1)} ENTER paper trade_id={tr.trade_id} entry_px={tr.entry_price:.2f}")
            return

        # real mode entry
        try:
            self.loop.run_until_complete(self._ensure_trading_floor())
        except Exception as e:
            print("pre_entry_topup_failed:", e)

        try:
            equity_before = float(
                self.loop.run_until_complete(
                    _get_subaccount_equity_usdc(self.clients, address=self.cfg.address, subaccount_number=int(self.cfg.subaccount_trading))
                )
            )
        except Exception:
            equity_before = 0.0

        entry_meta = {}
        try:
            entry_meta = self.loop.run_until_complete(self._open_real_short(equity_before=float(equity_before)))
        except Exception as e:
            print("open_real_short_failed:", e)
            return

        entry_px = float(entry_meta.get("fill_price") or float(bar["close"]))
        size_base = float(entry_meta.get("fill_size_base") or 0.0)
        entry_time_utc = str(entry_meta.get("fill_time_utc") or _iso_utc(t1))

        self._trade_seq += 1
        tr = OpenTrade(
            trade_id=f"real_{self._trade_seq}",
            trade_mode="real",
            entry_time_utc=str(entry_time_utc),
            entry_price=float(entry_px),
            entry_score=float(score),
            entry_threshold=float(thr),
            size_base=float(size_base),
        )
        self.open_trade = tr
        print(f"{entry_time_utc} ENTER real trade_id={tr.trade_id} entry_px={tr.entry_price:.2f} size_base={tr.size_base:.6f}")


def main() -> None:
    load_dotenv(Path(".env"), override=False)

    ap = argparse.ArgumentParser(description="Simplified live dYdX v4 ETH-USD SELL-only runner")

    ap.add_argument(
        "--entry-model",
        default=str(
            REPO_ROOT
            / "data"
            / "entry_regressor_oracle15m_sell_ctx120_weighted_oracleentry5m_2026-01-04T22-45-20Z"
            / "models"
            / "entry_regressor_sell_oracle15m_ctx120_weighted_oracleentry5m_2026-01-04T22-45-20Z.joblib"
        ),
    )
    ap.add_argument(
        "--exit-gap-model",
        default=str(REPO_ROOT / "data" / "exit_oracle_sell" / "exit_oracle_gap_regressor_sell_hold15_top10_2026-01-05T12-27-59Z.joblib"),
    )

    ap.add_argument("--market", default=os.getenv("DYDX_MARKET", "ETH-USD"))
    ap.add_argument("--trade-mode", choices=["paper", "real"], default="paper")
    ap.add_argument("--yes-really", action="store_true")

    ap.add_argument("--subaccount-trading", type=int, default=int(os.getenv("DYDX_SUBACCOUNT_TRADING", "3") or 3))
    ap.add_argument("--subaccount-bank", type=int, default=int(os.getenv("DYDX_SUBACCOUNT_BANK", "0") or 0))

    ap.add_argument("--backfill-hours", type=int, default=24)

    ap.add_argument("--entry-threshold", type=float, default=float(os.getenv("ETH_SELL_ENTRY_THRESHOLD", "0.39") or 0.39))

    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--exit-gap-tau", type=float, default=0.10)
    ap.add_argument("--exit-gap-min-exit-k", type=int, default=2)

    ap.add_argument("--fee-side", type=float, default=float(os.getenv("DYDX_FEE_SIDE", str(DEFAULT_FEE_SIDE)) or DEFAULT_FEE_SIDE))

    ap.add_argument("--log-every", type=int, default=60, help="Print an entry decision line every N bars (also prints immediately on enter/exit)")

    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "live_dydx" / "eth_sell"))
    ap.add_argument("--audit-path", default="")

    args = ap.parse_args()

    if str(args.trade_mode).lower() == "real" and not bool(args.yes_really):
        raise SystemExit("Refusing to place real orders without --yes-really")

    cfg = DydxV4Config.from_env()
    cfg.market = str(args.market).upper()
    cfg.subaccount_trading = int(args.subaccount_trading)
    cfg.subaccount_bank = int(args.subaccount_bank)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_path = Path(args.audit_path) if str(args.audit_path).strip() else (out_dir / f"audit_eth_sell_simple_{_now_ts()}.jsonl")

    entry = load_sell_entry_model(Path(args.entry_model))
    exit_gap = load_sell_exit_gap_model(Path(args.exit_gap_model))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if str(args.trade_mode).lower() == "paper":
        clients = SimpleNamespace(
            network=SimpleNamespace(rest_indexer=str(cfg.indexer_rest).rstrip("/"), websocket_indexer=str(cfg.indexer_ws).rstrip("/")),
            indexer=IndexerClient(str(cfg.indexer_rest).rstrip("/")),
        )
    else:
        clients = loop.run_until_complete(connect_v4(cfg))

    with AuditLogger.create(audit_path) as audit:
        audit.write(
            {
                "kind": KIND_RUN_META,
                "mode": "live",
                "symbol": str(cfg.market),
                "models": {
                    "entry": {"artifact": str(entry.artifact), "created_utc": str(entry.created_utc or "unknown")},
                    "exit_gap": {"artifact": str(exit_gap.artifact), "created_utc": str(exit_gap.created_utc or "unknown")},
                },
                "code_version": {
                    "script": Path(__file__).name,
                },
            }
        )

        runner = EthSellSimpleRunner(
            market=str(cfg.market),
            cfg=cfg,
            clients=clients,
            loop=loop,
            entry=entry,
            exit_gap=exit_gap,
            trade_mode=str(args.trade_mode),
            out_dir=out_dir,
            audit=audit,
            entry_threshold=float(args.entry_threshold),
            hold_min=int(args.hold_min),
            exit_gap_tau=float(args.exit_gap_tau),
            exit_gap_min_exit_k=int(args.exit_gap_min_exit_k),
            fee_side=float(args.fee_side),
            log_every=int(args.log_every),
        )

        # Warm backfill for stable rolling features.
        bars = loop.run_until_complete(_warm_backfill(clients, market=str(cfg.market), hours=int(args.backfill_hours)))
        if not bars.empty:
            runner.bars = bars
            runner.features.set_bars(bars)
            runner.features.recompute()

        q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=10_000)
        stop_evt = threading.Event()

        ws_url = str(getattr(clients, "network").websocket_indexer)
        t = threading.Thread(
            target=_run_ws_candles,
            kwargs={"ws_url": ws_url, "market": str(cfg.market), "out_q": q, "stop_evt": stop_evt},
            daemon=True,
        )
        t.start()

        print("dYdX WS:", ws_url)
        print("Market:", cfg.market)
        print("Trade mode:", args.trade_mode)
        print("Subaccounts: trading=", int(cfg.subaccount_trading), " bank=", int(cfg.subaccount_bank))
        print("Entry threshold:", float(args.entry_threshold))
        print("Exit policy: tau=", float(args.exit_gap_tau), " min_exit_k=", int(args.exit_gap_min_exit_k), " hold_min=", int(args.hold_min))
        print("Audit:", str(audit_path))
        print("Out dir:", str(out_dir))

        try:
            while not stop_evt.is_set():
                try:
                    bar = q.get(timeout=5.0)
                except queue.Empty:
                    continue

                try:
                    runner.on_closed_bar(bar)
                except Exception as e:
                    # Runner must not die on a single bad bar.
                    print("on_closed_bar_error:", e)
        finally:
            stop_evt.set()
            try:
                runner._flush_trades_csv()
            except Exception:
                pass


if __name__ == "__main__":
    main()
