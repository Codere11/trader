#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-05T19:10:28Z
"""Live/paper dYdX v4 runner for ETH-USD perpetual (SELL-only) with dedicated trading+bank subaccounts.

This implements a SELL agent using:
- Entry model: SELL oracle15m ctx120 weighted regressor (pre_min=5)
- Exit model: SELL oracle-gap regressor (hold_min=15)

Selection policy
- Causal per-day thresholding targeting a fixed fraction of minutes (default: 5%).
- Starts trading immediately by seeding the prior-score pool from the warm backfill window.

Exit policy
- At each minute-in-trade k:
  - compute current profitability (SELL net return)
  - compute profitability deltas (prev1/prev2/prev3, drawdown-from-peak)
  - exit when pred_gap_pct <= tau_gap for k>=min_exit_k; else force-exit at hold_min.

Bankroll / subaccounts
- Same two-subaccount scheme as the BTC runner:
  - trading subaccount: trades
  - bank subaccount: receives siphoned profits; funds floor topups/recaps
- This script allows overriding the bank threshold to 200 (instead of prior 150) per requirements.
- The "initial external topup budget" is treated as a cap on *bank->trading* floor topups while
  bank_equity < bank_threshold (since true external topups are manual).

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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.append(str(REPO_ROOT))

from dydx_adapter import load_dotenv
from dydx_adapter.v4 import DydxV4Config, connect_v4, new_client_id, usdc_to_quantums

from dydx_v4_client import OrderFlags
from dydx_v4_client.indexer.rest.constants import OrderType
from dydx_v4_client.indexer.rest.indexer_client import IndexerClient
from dydx_v4_client.indexer.socket.websocket import IndexerSocket
from dydx_v4_client.node.client import QueryNodeClient
from dydx_v4_client.node.market import Market
from dydx_v4_client.node.message import subaccount

from v4_proto.dydxprotocol.clob.order_pb2 import Order

from binance_adapter.crossref_oracle_vs_agent_patterns import FEATURES, compute_feature_frame


# ---- Fixed SELL agent settings (can be overridden by CLI where exposed) ----
HOLD_MIN = 15
EXIT_GAP_TAU = 0.10
EXIT_GAP_MIN_EXIT_K = 2

# Default selection coverage.
# NOTE: The full-dataset DO_NOT_TOUCH backtests for this SELL stack were run at multiple coverages
# (0.10, 0.05, 0.01, 0.001, 0.0005). The most used/most stable default in those runs is 0.01 (00100bps).
TARGET_FRAC = 0.01  # 1% of minutes/day (causal threshold from prior days)

# Fees: use per-side fee (0.0005 = 0.05% per side; 0.1% round trip)
FEE_SIDE_ASSUMED = 0.0005

# Entry model needs base features from compute_feature_frame
BASE_FEATS = [
    "ret_1m_pct",
    "mom_3m_pct",
    "mom_5m_pct",
    "vol_std_5m",
    "range_5m",
    "range_norm_5m",
    "macd",
    "vwap_dev_5m",
]

CTX_WINDOWS = [30, 60, 120]


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _iso_utc(ts_like: Any) -> str:
    return pd.to_datetime(ts_like, utc=True).to_pydatetime().isoformat().replace("+00:00", "Z")


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


def _slope5_window(vals: np.ndarray) -> float:
    v = np.asarray(vals, dtype=np.float64)
    if v.shape != (5,):
        return float("nan")
    xc = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    return float(np.nansum(v * xc) / 10.0)


def _safe_std(vals: np.ndarray) -> float:
    v = np.asarray(vals, dtype=np.float64)
    m = np.nanmean(v)
    if not np.isfinite(m):
        return float("nan")
    return float(np.nanmean((v - m) ** 2) ** 0.5)


def _agg5_from_window(win5: np.ndarray) -> Dict[str, float]:
    w = np.asarray(win5, dtype=np.float64)
    return {
        "last": float(w[-1]),
        "mean5": float(np.nanmean(w)),
        "std5": float(_safe_std(w)),
        "min5": float(np.nanmin(w)),
        "max5": float(np.nanmax(w)),
        "range5": float(np.nanmax(w) - np.nanmin(w)),
        "slope5": float(_slope5_window(w)),
    }


def _orderstat_threshold(prior: np.ndarray, frac: float) -> float:
    prior = np.asarray(prior, dtype=np.float64)
    prior = prior[np.isfinite(prior)]
    if prior.size == 0:
        return float("inf")
    # threshold is the (1-frac) quantile
    k = int(np.floor((1.0 - float(frac)) * float(prior.size - 1)))
    k = max(0, min(int(prior.size - 1), k))
    return float(np.partition(prior, k)[k])


@dataclass(frozen=True)
class SellEntryModel:
    model: Any
    feature_cols: List[str]
    created_utc: Optional[str]
    pre_min: int


@dataclass(frozen=True)
class SellExitGapModel:
    model: Any
    feature_cols: List[str]
    created_utc: Optional[str]


def load_sell_entry_model(path: Path) -> SellEntryModel:
    obj = joblib.load(Path(path))
    if not isinstance(obj, dict) or "model" not in obj or "feature_cols" not in obj:
        raise SystemExit(f"Unexpected SELL entry model artifact format: {path}")
    pre_min = int(obj.get("pre_min") or 0)
    if pre_min != 5:
        raise SystemExit(f"Expected SELL entry model pre_min=5, got pre_min={pre_min}")
    return SellEntryModel(
        model=obj["model"],
        feature_cols=[str(c) for c in list(obj.get("feature_cols") or [])],
        created_utc=obj.get("created_utc"),
        pre_min=pre_min,
    )


def load_sell_exit_gap_model(path: Path) -> SellExitGapModel:
    obj = joblib.load(Path(path))
    if not isinstance(obj, dict) or "model" not in obj or "feature_cols" not in obj:
        raise SystemExit(f"Unexpected SELL exit gap model artifact format: {path}")
    return SellExitGapModel(
        model=obj["model"],
        feature_cols=[str(c) for c in list(obj.get("feature_cols") or [])],
        created_utc=obj.get("created_utc"),
    )


# ---- dYdX helpers (adapted from the BTC runner) ----


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


class CandleAggregator:
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


def _run_ws_candles(*, ws_url: str, market: str, out_q: "queue.Queue[Dict[str, Any]]", stop_evt: threading.Event) -> None:
    agg = CandleAggregator()

    def on_open(sock):
        try:
            msg = {"type": "subscribe", "channel": "v4_candles", "id": str(market).upper(), "resolution": "1MIN"}
            sock.send(json.dumps(msg))
        except Exception:
            pass

    def on_message(sock, message: str):
        if stop_evt.is_set():
            try:
                sock.close()
            except Exception:
                pass
            return

        try:
            msg = json.loads(message)
        except Exception:
            return

        contents = msg.get("contents") if isinstance(msg, dict) else None
        candles = contents.get("candles") if isinstance(contents, dict) else None
        if not isinstance(candles, list):
            return

        for c in candles:
            if not isinstance(c, dict):
                continue
            closed = agg.ingest(c)
            if closed is None:
                continue
            try:
                out_q.put_nowait(closed)
            except queue.Full:
                pass

    def on_error(sock, error):
        if not stop_evt.is_set():
            print("ws_error:", error)

    def on_close(sock, close_status_code=None, close_msg=None):
        if not stop_evt.is_set():
            print("ws_closed")

    backoff = 1.0
    while not stop_evt.is_set():
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


@dataclass
class OpenTrade:
    trade_id: str
    trade_mode: str
    entry_time_utc: str
    entry_price: float
    entry_score: float
    entry_threshold: float
    size_base: float

    mins_in_trade: int
    ret_hist: List[float]
    peak_ret: float


class EthSellRunner:
    def __init__(
        self,
        *,
        market: str,
        cfg: DydxV4Config,
        clients,
        loop: asyncio.AbstractEventLoop,
        entry: EntryModel,
        exit_gap: ExitGapModel,
        out_dir: Path,
        trade_mode: str,
        backfill_hours: int,
        seed_csv: Optional[Path],
        bank_threshold_usdc: float,
        initial_topup_budget_usdc: float,
        target_frac: float,
        min_prior_scores: int,
        max_prior_scores: int,
        metrics_port: int,
        health_max_staleness_sec: float,
    ) -> None:
        self.market = str(market).upper()
        self.cfg = cfg
        self.clients = clients
        self.loop = loop

        self.entry = entry
        self.exit_gap = exit_gap

        self.trade_mode = str(trade_mode).lower()
        self.backfill_hours = int(backfill_hours)

        self.target_frac = float(target_frac)
        self.min_prior_scores = int(min_prior_scores)
        self.max_prior_scores = int(max_prior_scores)

        # Override threshold per requirements ("after 200" policy)
        self.cfg.bank_threshold_usdc = float(bank_threshold_usdc)

        self.initial_topup_budget_usdc = float(initial_topup_budget_usdc)
        self.initial_topup_spent_usdc = 0.0

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        st = str(self.market).lower().replace("-", "_")
        legacy_path = self.out_dir / "eth_sell_budget_state.json"
        self.budget_state_path = self.out_dir / f"topup_budget_state_{st}_sell_tr{int(self.cfg.subaccount_trading)}_bank{int(self.cfg.subaccount_bank)}.json"

        # Legacy migration: if the old file exists and the new one doesn't, import the spent value.
        if legacy_path.exists() and not self.budget_state_path.exists():
            try:
                obj = json.loads(legacy_path.read_text(encoding="utf-8"))
                self.initial_topup_spent_usdc = float(obj.get("initial_topup_spent_usdc") or 0.0)
            except Exception:
                pass

        self._load_budget_state()
        self._save_budget_state()

        self.bars: Optional[pd.DataFrame] = None
        self.base: Optional[pd.DataFrame] = None
        self.ctx: Optional[pd.DataFrame] = None

        self.open_trade: Optional[OpenTrade] = None
        self.pending_entry: Optional[Dict[str, Any]] = None

        # Threshold state
        self.cur_day = None
        self.scores_cur_day: List[float] = []
        self.prior_scores: List[float] = []
        self.thr_cur_day: float = float("inf")

        self.records: List[Dict[str, Any]] = []
        self._trade_seq = 0

        self.metrics_port = int(metrics_port)
        self.health_max_staleness_sec = float(health_max_staleness_sec)

        self.last_closed_wall = time.time()

    def _load_budget_state(self) -> None:
        try:
            if self.budget_state_path.exists():
                obj = json.loads(self.budget_state_path.read_text(encoding="utf-8"))
                self.initial_topup_spent_usdc = float(obj.get("initial_topup_spent_usdc") or 0.0)
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

    def _recompute_feature_sources(self) -> None:
        if self.bars is None or self.bars.empty:
            self.base = None
            self.ctx = None
            return

        bars = self.bars.copy().sort_values("timestamp").reset_index(drop=True)
        bars["ts_min"] = pd.to_datetime(bars["timestamp"], utc=True)

        base_full = compute_feature_frame(bars)
        # Keep only the base features we need
        keep_base = [c for c in BASE_FEATS if c in base_full.columns]
        base = base_full[keep_base].copy()

        # Context series (causal): computed from shifted OHLCV (t-1 and earlier)
        close_prev = pd.to_numeric(bars["close"], errors="coerce").shift(1)
        high_prev = pd.to_numeric(bars["high"], errors="coerce").shift(1)
        low_prev = pd.to_numeric(bars["low"], errors="coerce").shift(1)
        vol_prev = pd.to_numeric(bars["volume"], errors="coerce").shift(1)

        ctx_cols: Dict[str, pd.Series] = {}
        for w in CTX_WINDOWS:
            w = int(w)
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

        ctx = pd.DataFrame(ctx_cols)

        self.base = base.reset_index(drop=True)
        self.ctx = ctx.reset_index(drop=True)

    def _feature_map_at(self, i: int) -> Dict[str, float]:
        """Build the full precontext+ctx feature map for a decision at index i.

        Convention: decision at close[i], uses only i-5..i-1.
        """
        if self.bars is None or self.base is None or self.ctx is None:
            return {}

        i = int(i)
        if i < 5:
            return {}

        # window indices for precontext
        j0 = i - 5
        j1 = i

        # close normalization window
        close = pd.to_numeric(self.bars["close"], errors="coerce").to_numpy(np.float64)
        vol = pd.to_numeric(self.bars["volume"], errors="coerce").to_numpy(np.float64)

        close_prev = close[j0:j1]
        if close_prev.shape != (5,):
            return {}

        close_last = float(close_prev[-1])
        if not (np.isfinite(close_last) and close_last > 0.0):
            return {}

        close_norm = (close_prev / close_last - 1.0) * 100.0
        px = _agg5_from_window(close_norm)

        px_ret5m = float(close_norm[-1] - close_norm[0])
        px_absret5m = float(abs(px_ret5m))

        vol_prev = vol[j0:j1]
        vol_log = np.log1p(np.maximum(0.0, vol_prev.astype(np.float64)))
        vv = _agg5_from_window(vol_log)

        out: Dict[str, float] = {}

        # px_close_norm_pct
        for k, v in px.items():
            out[f"px_close_norm_pct__{k}"] = float(v)
        out["px_close_norm_pct__ret5m"] = float(px_ret5m)
        out["px_close_norm_pct__absret5m"] = float(px_absret5m)
        out["px_close_norm_pct__m5"] = float(close_norm[0])
        out["px_close_norm_pct__m4"] = float(close_norm[1])
        out["px_close_norm_pct__m3"] = float(close_norm[2])
        out["px_close_norm_pct__m2"] = float(close_norm[3])
        out["px_close_norm_pct__m1"] = float(close_norm[4])

        # vol_log1p
        for k, v in vv.items():
            out[f"vol_log1p__{k}"] = float(v)

        # base features (from compute_feature_frame)
        for c in BASE_FEATS:
            if c not in self.base.columns:
                continue
            arr = pd.to_numeric(self.base[c], errors="coerce").to_numpy(np.float64)
            w = arr[j0:j1]
            if w.shape != (5,):
                continue
            agg = _agg5_from_window(w)
            for k, v in agg.items():
                out[f"{c}__{k}"] = float(v)

        # ctx series
        for c in self.ctx.columns:
            arr = pd.to_numeric(self.ctx[c], errors="coerce").to_numpy(np.float64)
            w = arr[j0:j1]
            if w.shape != (5,):
                continue
            agg = _agg5_from_window(w)
            for k, v in agg.items():
                out[f"{c}__{k}"] = float(v)

        # missing indicators
        missing_close_n = float(np.sum(~np.isfinite(close_prev)))
        out["missing_close_n"] = float(missing_close_n)

        # missing_any over close + base + ctx in the window
        miss_any = False
        if np.any(~np.isfinite(close_prev)):
            miss_any = True
        for c in BASE_FEATS:
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

    def _entry_score_at(self, i: int) -> Optional[float]:
        feat = self._feature_map_at(i)
        if not feat:
            return None
        x = np.asarray([float(feat.get(c, float("nan"))) for c in self.entry.feature_cols], dtype=np.float32)
        try:
            s = float(self.entry.model.predict(np.asarray([x], dtype=np.float32))[0])
        except Exception:
            return None
        return s if np.isfinite(s) else None

    def _exit_gap_pred_at(self, *, i: int, mins_in_trade: int, cur_ret: float, r_prev1: float, r_prev2: float, r_prev3: float, peak_ret: float) -> Optional[float]:
        feat = self._feature_map_at(i)
        if not feat:
            return None

        full = dict(feat)
        full.update(
            {
                "mins_in_trade": float(mins_in_trade),
                "mins_remaining": float(max(0, int(HOLD_MIN) - int(mins_in_trade))),
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

    def _roll_day(self, new_day) -> None:
        if self.cur_day is None:
            self.cur_day = new_day
            self.scores_cur_day = []
            self.thr_cur_day = float("inf")
            return

        if new_day == self.cur_day:
            return

        # Move previous day scores into prior pool
        if self.scores_cur_day:
            self.prior_scores.extend([float(x) for x in self.scores_cur_day if np.isfinite(float(x))])
            if self.max_prior_scores > 0 and len(self.prior_scores) > int(self.max_prior_scores):
                self.prior_scores[:] = self.prior_scores[-int(self.max_prior_scores) :]

        self.cur_day = new_day
        self.scores_cur_day = []

        if len(self.prior_scores) < int(self.min_prior_scores):
            self.thr_cur_day = float("inf")
        else:
            self.thr_cur_day = _orderstat_threshold(np.asarray(self.prior_scores, dtype=np.float64), float(self.target_frac))

    def seed_thresholds_from_backfill(self) -> None:
        """Seed prior_scores from the current warm backfill bars so we can trade immediately."""
        if self.bars is None or self.bars.empty:
            return

        self._recompute_feature_sources()
        if self.base is None or self.ctx is None:
            return

        ts = pd.to_datetime(self.bars["timestamp"], utc=True)
        days = ts.dt.date.to_numpy()
        if len(days) == 0:
            return

        last_day = days[-1]
        # Score each minute and bucket by day
        by_day: Dict[Any, List[float]] = {}
        for i in range(5, len(self.bars) - 2):
            s = self._entry_score_at(i)
            if s is None:
                continue
            d = days[i]
            by_day.setdefault(d, []).append(float(s))

        prior: List[float] = []
        for d, vals in by_day.items():
            if d == last_day:
                continue
            prior.extend([float(x) for x in vals])

        self.prior_scores = prior[-int(self.max_prior_scores) :] if (self.max_prior_scores > 0) else prior
        self.cur_day = last_day
        self.scores_cur_day = list(by_day.get(last_day, []))

        if len(self.prior_scores) >= int(self.min_prior_scores) and len(self.prior_scores) > 0:
            self.thr_cur_day = _orderstat_threshold(np.asarray(self.prior_scores, dtype=np.float64), float(self.target_frac))
        else:
            self.thr_cur_day = float("inf")

        self.records.append(
            {
                "event": "seed_thresholds",
                "time_utc": _now_ts(),
                "target_frac": float(self.target_frac),
                "min_prior_scores": int(self.min_prior_scores),
                "max_prior_scores": int(self.max_prior_scores),
                "prior_scores": int(len(self.prior_scores)),
                "scores_cur_day": int(len(self.scores_cur_day)),
                "thr_cur_day": float(self.thr_cur_day) if np.isfinite(self.thr_cur_day) else None,
            }
        )

    async def _ensure_trading_floor_with_budget(self) -> Dict[str, Any]:
        """Top up trading to floor from bank (between trades), honoring initial topup budget."""
        trading_eq = float(await _get_subaccount_equity_usdc(self.clients, address=self.cfg.address, subaccount_number=self.cfg.subaccount_trading))
        floor = float(self.cfg.trade_floor_usdc)

        if trading_eq >= floor:
            return {"topped_up": False, "needed_usdc": 0.0, "transferred_usdc": 0.0}

        needed = float(floor - trading_eq)
        bank_eq = float(await _get_subaccount_equity_usdc(self.clients, address=self.cfg.address, subaccount_number=self.cfg.subaccount_bank))
        bank_threshold = float(getattr(self.cfg, "bank_threshold_usdc", float("inf")))

        # Only allow floor topups while bank < threshold (same policy as BTC runner).
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

        # Budget cap applies in this initial stage.
        remaining = max(0.0, float(self.initial_topup_budget_usdc) - float(self.initial_topup_spent_usdc))
        if remaining <= 0.0:
            return {
                "topped_up": False,
                "needed_usdc": float(needed),
                "transferred_usdc": 0.0,
                "skipped": True,
                "skip_reason": "initial_topup_budget_exhausted",
                "budget_usdc": float(self.initial_topup_budget_usdc),
                "spent_usdc": float(self.initial_topup_spent_usdc),
                "bank_equity_usdc": float(bank_eq),
                "trading_equity_usdc": float(trading_eq),
            }

        transfer = min(float(needed), float(bank_eq), float(remaining))
        if transfer <= 0.0:
            return {
                "topped_up": False,
                "needed_usdc": float(needed),
                "transferred_usdc": 0.0,
                "skipped": True,
                "skip_reason": "no_transfer_possible",
                "budget_remaining_usdc": float(remaining),
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
            "budget_usdc": float(self.initial_topup_budget_usdc),
            "spent_usdc": float(self.initial_topup_spent_usdc),
            "bank_equity_usdc": float(bank_eq),
            "bank_threshold_usdc": float(bank_threshold),
            "tx": _pb_to_jsonable(tx),
        }

    async def _refinance_after_liquidation(self) -> Dict[str, Any]:
        floor = float(self.cfg.trade_floor_usdc)
        bank_threshold = float(getattr(self.cfg, "bank_threshold_usdc", 200.0))
        recap_frac = float(getattr(self.cfg, "liquidation_recap_bank_frac", 0.20))

        trading_eq = float(await _get_subaccount_equity_usdc(self.clients, address=self.cfg.address, subaccount_number=self.cfg.subaccount_trading))
        bank_eq = float(await _get_subaccount_equity_usdc(self.clients, address=self.cfg.address, subaccount_number=self.cfg.subaccount_bank))

        action = ""
        requested = 0.0

        if float(bank_eq) < float(bank_threshold):
            action = "refill_to_floor"
            requested = max(0.0, float(floor) - float(trading_eq))
            requested = min(float(requested), float(bank_eq))

            # Budget applies only while bank < threshold.
            remaining = max(0.0, float(self.initial_topup_budget_usdc) - float(self.initial_topup_spent_usdc))
            if remaining <= 0.0:
                return {
                    "attempted": False,
                    "action": str(action),
                    "requested_usdc": float(requested),
                    "transferred_usdc": 0.0,
                    "budget_usdc": float(self.initial_topup_budget_usdc),
                    "spent_usdc": float(self.initial_topup_spent_usdc),
                    "skip_reason": "initial_topup_budget_exhausted",
                    "bank_equity_usdc": float(bank_eq),
                    "trading_equity_usdc": float(trading_eq),
                    "bank_threshold_usdc": float(bank_threshold),
                    "trade_floor_usdc": float(floor),
                    "liquidation_recap_bank_frac": float(recap_frac),
                }

            requested = min(float(requested), float(remaining))
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

        tx = await self.clients.node.transfer(
            self.clients.wallet,
            subaccount(self.cfg.address, int(self.cfg.subaccount_bank)),
            subaccount(self.cfg.address, int(self.cfg.subaccount_trading)),
            int(self.clients.usdc_asset_id),
            int(amt_q),
        )

        if str(action) == "refill_to_floor" and float(bank_eq) < float(bank_threshold):
            self.initial_topup_spent_usdc += float(requested)
            self._save_budget_state()

        return {
            "attempted": True,
            "action": str(action),
            "requested_usdc": float(requested),
            "transferred_usdc": float(requested),
            "budget_usdc": float(self.initial_topup_budget_usdc),
            "spent_usdc": float(self.initial_topup_spent_usdc),
            "bank_equity_usdc": float(bank_eq),
            "trading_equity_usdc": float(trading_eq),
            "bank_threshold_usdc": float(bank_threshold),
            "trade_floor_usdc": float(floor),
            "liquidation_recap_bank_frac": float(recap_frac),
            "tx": _pb_to_jsonable(tx),
        }

    def reconcile_startup(self) -> None:
        if str(self.trade_mode).lower() != "real":
            return

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

        if abs(float(live_sz)) <= 0.0:
            return

        print(f"Startup: detected existing open position; closing reduce-only (size_base={abs(float(live_sz))})...")
        try:
            self.loop.run_until_complete(self._close_real_short(size_base=float(abs(float(live_sz)))))
        except Exception as e:
            print("Startup close failed:", e)

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
        limit_px = float(px) * (1.0 - slip)  # sell: cross bid

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
            "fill_price": float(filled_px),
            "fill_size_base": float(filled_sz),
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
        limit_px = float(px) * (1.0 + slip)  # buy-to-close: cross ask

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
            "fill_price": float(filled_px),
            "fill_size_base": float(filled_sz),
            "fee_usdc": float(fee_usdc),
            "fill_assumed": bool(fill_assumed),
        }

    async def _siphon_profit_and_topup(self, *, equity_before: float) -> Dict[str, Any]:
        equity_after = float(await _get_subaccount_equity_usdc(self.clients, address=self.cfg.address, subaccount_number=self.cfg.subaccount_trading))
        profit = float(equity_after - float(equity_before))

        siphon = {"attempted": False, "profit_usdc": float(profit), "requested_usdc": 0.0, "transferred_usdc": 0.0}

        if profit > 0:
            frac = float(self.cfg.profit_siphon_frac)
            requested = float(profit) * float(frac)
            requested = max(0.0, requested)
            bank_eq = float(await _get_subaccount_equity_usdc(self.clients, address=self.cfg.address, subaccount_number=self.cfg.subaccount_bank))

            requested = min(requested, float(equity_after))
            if requested > 0:
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
                    "bank_threshold_usdc": float(getattr(self.cfg, "bank_threshold_usdc", float("inf"))),
                    "tx": _pb_to_jsonable(tx),
                }

        topup = await self._ensure_trading_floor_with_budget()
        return {
            "equity_before_usdc": float(equity_before),
            "equity_after_usdc": float(equity_after),
            "profit_usdc": float(profit),
            "siphon": siphon,
            "topup": topup,
        }

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

        # Keep enough history for ctx120 + safety
        if len(self.bars) > 12000:
            self.bars = self.bars.iloc[-12000:].copy().reset_index(drop=True)

        return int(len(self.bars) - 1)

    def on_closed_bar(self, bar: Mapping[str, Any]) -> None:
        self.last_closed_wall = float(time.time())

        i = self._append_bar(bar)
        self._recompute_feature_sources()

        if self.bars is None:
            return

        day = pd.to_datetime(self.bars.iloc[i]["timestamp"], utc=True).date()
        self._roll_day(day)

        # Paper pending entry executes at next bar open.
        if self.trade_mode == "paper" and self.pending_entry is not None and self.open_trade is None:
            p = dict(self.pending_entry)
            self.pending_entry = None

            entry_px = float(bar["open"])  # next-bar open
            self._trade_seq += 1
            tr = OpenTrade(
                trade_id=f"paper_{self._trade_seq}",
                trade_mode="paper",
                entry_time_utc=_iso_utc(pd.to_datetime(bar["timestamp"], utc=True)),
                entry_price=float(entry_px),
                entry_score=float(p["entry_score"]),
                entry_threshold=float(self.thr_cur_day),
                size_base=0.0,
                mins_in_trade=0,
                ret_hist=[],
                peak_ret=-1e30,
            )
            self.open_trade = tr

        # If open trade: update + maybe exit.
        if self.open_trade is not None:
            tr = self.open_trade

            # Live safety: if we were in a real trade but are now flat unexpectedly, treat as liquidation/forced close.
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
                dt_now = pd.to_datetime(bar["timestamp"], utc=True) + timedelta(minutes=1)
                age_s = float((dt_now - dt_entry).total_seconds()) if pd.notna(dt_entry) else float("inf")

                if (np.isfinite(live_sz) and abs(float(live_sz)) <= 0.0) and (age_s >= 60.0):
                    recap = self.loop.run_until_complete(self._refinance_after_liquidation())
                    self.records.append(
                        {
                            "event": "liquidation_detected",
                            "trade_id": tr.trade_id,
                            "trade_mode": "real",
                            "symbol": self.market,
                            "entry_time_utc": tr.entry_time_utc,
                            "detected_at_time_utc": _iso_utc(dt_now),
                            "entry_price": float(tr.entry_price),
                            "entry_score": float(tr.entry_score),
                            "entry_threshold": float(tr.entry_threshold),
                            "age_s": float(age_s),
                            "recap": recap,
                        }
                    )
                    self.open_trade = None
                    return

            tr.mins_in_trade = int(tr.mins_in_trade) + 1
            k_rel = int(tr.mins_in_trade)

            exit_px = float(bar["close"])
            cur_ret = float(net_ret_pct_sell(tr.entry_price, exit_px, float(FEE_SIDE_ASSUMED)))

            tr.ret_hist.append(float(cur_ret))
            tr.peak_ret = float(max(float(tr.peak_ret), float(cur_ret)))

            r_prev1 = float(tr.ret_hist[-2]) if len(tr.ret_hist) >= 2 else float("nan")
            r_prev2 = float(tr.ret_hist[-3]) if len(tr.ret_hist) >= 3 else float("nan")
            r_prev3 = float(tr.ret_hist[-4]) if len(tr.ret_hist) >= 4 else float("nan")

            pred_gap = self._exit_gap_pred_at(
                i=int(i),
                mins_in_trade=int(k_rel),
                cur_ret=float(cur_ret),
                r_prev1=float(r_prev1),
                r_prev2=float(r_prev2),
                r_prev3=float(r_prev3),
                peak_ret=float(tr.peak_ret),
            )

            exit_reason = ""
            if int(k_rel) >= int(HOLD_MIN):
                exit_reason = "hold_min"
            elif int(k_rel) >= int(EXIT_GAP_MIN_EXIT_K) and pred_gap is not None and float(pred_gap) <= float(EXIT_GAP_TAU):
                exit_reason = "pred_gap<=tau"

            if not exit_reason:
                return

            # Exit
            if self.trade_mode == "paper":
                self.records.append(
                    {
                        "trade_id": tr.trade_id,
                        "trade_mode": "paper",
                        "symbol": self.market,
                        "direction": "SHORT",
                        "entry_time_utc": tr.entry_time_utc,
                        "exit_time_utc": _iso_utc(pd.to_datetime(bar["timestamp"], utc=True) + timedelta(minutes=1)),
                        "entry_price": float(tr.entry_price),
                        "exit_price": float(exit_px),
                        "exit_rel_min": int(k_rel),
                        "entry_score": float(tr.entry_score),
                        "entry_threshold": float(tr.entry_threshold),
                        "exit_pred_gap_pct": float(pred_gap) if pred_gap is not None else float("nan"),
                        "exit_gap_tau": float(EXIT_GAP_TAU),
                        "exit_gap_min_exit_k": int(EXIT_GAP_MIN_EXIT_K),
                        "exit_reason": str(exit_reason),
                        "realized_ret_1x_pct": float(cur_ret),
                    }
                )
                self.open_trade = None
                return

            # real mode exit
            equity_before = float(
                self.loop.run_until_complete(
                    _get_subaccount_equity_usdc(self.clients, address=self.cfg.address, subaccount_number=int(self.cfg.subaccount_trading))
                )
            )

            live_sz = 0.0
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
            close_meta = self.loop.run_until_complete(self._close_real_short(size_base=float(size_to_close)))

            bank_ops = self.loop.run_until_complete(self._siphon_profit_and_topup(equity_before=float(equity_before)))

            self.records.append(
                {
                    "trade_id": tr.trade_id,
                    "trade_mode": "real",
                    "symbol": self.market,
                    "direction": "SHORT",
                    "entry_time_utc": tr.entry_time_utc,
                    "exit_time_utc": _iso_utc(pd.to_datetime(bar["timestamp"], utc=True) + timedelta(minutes=1)),
                    "entry_price": float(tr.entry_price),
                    "exit_price": float(exit_px),
                    "exit_rel_min": int(k_rel),
                    "entry_score": float(tr.entry_score),
                    "entry_threshold": float(tr.entry_threshold),
                    "exit_pred_gap_pct": float(pred_gap) if pred_gap is not None else float("nan"),
                    "exit_gap_tau": float(EXIT_GAP_TAU),
                    "exit_gap_min_exit_k": int(EXIT_GAP_MIN_EXIT_K),
                    "exit_reason": str(exit_reason),
                    "realized_ret_1x_pct_assumed_fee": float(cur_ret),
                    "close": close_meta,
                    "bank_ops": bank_ops,
                }
            )
            self.open_trade = None
            return

        # No open trade: consider opening.
        score = self._entry_score_at(int(i))
        if score is not None and np.isfinite(float(score)):
            self.scores_cur_day.append(float(score))

        if self.open_trade is not None:
            return

        if score is None or not np.isfinite(float(score)):
            return

        if not np.isfinite(float(self.thr_cur_day)):
            return

        if float(score) < float(self.thr_cur_day):
            return

        if self.trade_mode == "paper":
            self.pending_entry = {"entry_score": float(score)}
            return

        # Hard stop: do not open new trades if the initial topup budget is exhausted.
        if float(self.initial_topup_spent_usdc) >= float(self.initial_topup_budget_usdc):
            self.records.append(
                {
                    "event": "entry_skipped_budget_exhausted",
                    "time_utc": _iso_utc(pd.to_datetime(bar["timestamp"], utc=True) + timedelta(minutes=1)),
                    "symbol": self.market,
                    "direction": "SHORT",
                    "entry_score": float(score),
                    "entry_threshold": float(self.thr_cur_day),
                    "budget_usdc": float(self.initial_topup_budget_usdc),
                    "spent_usdc": float(self.initial_topup_spent_usdc),
                }
            )
            return

        # real mode: top up to floor (subject to budget), then open short.
        topup_pre = self.loop.run_until_complete(self._ensure_trading_floor_with_budget())
        equity_before = float(
            self.loop.run_until_complete(
                _get_subaccount_equity_usdc(self.clients, address=self.cfg.address, subaccount_number=int(self.cfg.subaccount_trading))
            )
        )

        # Hard floor: if we couldn't reach the floor (e.g., budget exhausted), skip opening.
        if float(equity_before) < float(self.cfg.trade_floor_usdc):
            self.records.append(
                {
                    "event": "entry_skipped_no_floor",
                    "time_utc": _iso_utc(pd.to_datetime(bar["timestamp"], utc=True) + timedelta(minutes=1)),
                    "symbol": self.market,
                    "direction": "SHORT",
                    "entry_score": float(score),
                    "entry_threshold": float(self.thr_cur_day),
                    "trading_equity_usdc": float(equity_before),
                    "trade_floor_usdc": float(self.cfg.trade_floor_usdc),
                    "topup_pre": topup_pre,
                }
            )
            return

        entry_meta = self.loop.run_until_complete(self._open_real_short(equity_before=float(equity_before)))

        entry_px = float(entry_meta.get("fill_price") or float(bar["close"]))
        size_base = float(entry_meta.get("fill_size_base") or 0.0)
        entry_time_utc = str(entry_meta.get("fill_time_utc") or _iso_utc(pd.to_datetime(bar["timestamp"], utc=True) + timedelta(minutes=1)))

        self._trade_seq += 1
        tr = OpenTrade(
            trade_id=f"real_{self._trade_seq}",
            trade_mode="real",
            entry_time_utc=str(entry_time_utc),
            entry_price=float(entry_px),
            entry_score=float(score),
            entry_threshold=float(self.thr_cur_day),
            size_base=float(size_base),
            mins_in_trade=0,
            ret_hist=[],
            peak_ret=-1e30,
        )
        self.open_trade = tr

        self.records.append(
            {
                "event": "entry_opened",
                "trade_id": tr.trade_id,
                "trade_mode": "real",
                "symbol": self.market,
                "direction": "SHORT",
                "time_utc": str(entry_time_utc),
                "entry_price": float(entry_px),
                "size_base": float(size_base),
                "entry_score": float(score),
                "entry_threshold": float(self.thr_cur_day),
                "topup_pre": topup_pre,
                "entry": entry_meta,
            }
        )

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
                            "max_staleness_seconds": float(max_age),
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
                        has_open_trade = 1 if getattr(runner, "open_trade", None) is not None else 0
                        records_total = int(len(getattr(runner, "records", []) or []))
                        lines = [
                            "# HELP autotrader_last_closed_bar_age_seconds Seconds since last closed 1m bar was received",
                            "# TYPE autotrader_last_closed_bar_age_seconds gauge",
                            f"autotrader_last_closed_bar_age_seconds {age}",
                            "# HELP autotrader_has_open_trade Whether the runner currently tracks an open trade",
                            "# TYPE autotrader_has_open_trade gauge",
                            f"autotrader_has_open_trade {has_open_trade}",
                            "# HELP autotrader_records_total Number of records buffered in memory",
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
                    return

            httpd = ThreadingHTTPServer((host, port), Handler)
            httpd.serve_forever()
        except Exception as e:
            print("http_server_error:", e)

    def flush_outputs(self) -> None:
        if not self.records:
            return
        ts = _now_ts()
        out = pd.DataFrame(self.records)
        out.to_csv(self.out_dir / f"eth_sell_live_events_{ts}.csv", index=False)


def main() -> None:
    load_dotenv(Path(".env"), override=False)

    ap = argparse.ArgumentParser(description="Live dYdX v4 ETH-USD SELL-only runner (ctx120 entry + SELL exit-gap)")

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
    ap.add_argument("--yes-really", action="store_true", help="Required for --trade-mode real")

    ap.add_argument("--subaccount-trading", type=int, default=int(os.getenv("DYDX_SUBACCOUNT_TRADING", "1")))
    ap.add_argument("--subaccount-bank", type=int, default=int(os.getenv("DYDX_SUBACCOUNT_BANK", "0")))

    ap.add_argument("--backfill-hours", type=int, default=168)

    ap.add_argument(
        "--seed-csv",
        default=str(REPO_ROOT / "data" / "dydx_ETH-USD_1MIN_full_2026-01-03T23-48-53Z.csv"),
        help="Optional local CSV to seed bars if warm backfill fails. Use '' to disable.",
    )

    ap.add_argument("--target-frac", type=float, default=float(TARGET_FRAC))
    ap.add_argument("--min-prior-scores", type=int, default=2000)
    ap.add_argument("--max-prior-scores", type=int, default=200_000)

    ap.add_argument("--bank-threshold-usdc", type=float, default=200.0)
    ap.add_argument("--initial-topup-budget-usdc", type=float, default=50.0)

    ap.add_argument("--metrics-port", type=int, default=0)
    ap.add_argument("--health-max-staleness-sec", type=float, default=120.0)
    ap.add_argument("--stall-seconds", type=int, default=0)

    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "live_dydx"))

    args = ap.parse_args()

    if str(args.trade_mode).lower() == "real" and not bool(args.yes_really):
        raise SystemExit("Refusing to place real orders without --yes-really")

    cfg = DydxV4Config.from_env()
    cfg.market = str(args.market).upper()
    cfg.subaccount_trading = int(args.subaccount_trading)
    cfg.subaccount_bank = int(args.subaccount_bank)

    # Override the bank threshold to 200 unless the user passed a different value.
    cfg.bank_threshold_usdc = float(args.bank_threshold_usdc)

    entry = load_sell_entry_model(Path(args.entry_model))
    exit_gap = load_sell_exit_gap_model(Path(args.exit_gap_model))

    out_dir = Path(args.out_dir)

    seed_csv = Path(args.seed_csv) if str(args.seed_csv).strip() else None

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if str(args.trade_mode).lower() == "paper":
        clients = SimpleNamespace(
            network=SimpleNamespace(rest_indexer=str(cfg.indexer_rest).rstrip("/"), websocket_indexer=str(cfg.indexer_ws).rstrip("/")),
            indexer=IndexerClient(str(cfg.indexer_rest).rstrip("/")),
        )
    else:
        clients = loop.run_until_complete(connect_v4(cfg))

    runner = EthSellRunner(
        market=str(cfg.market),
        cfg=cfg,
        clients=clients,
        loop=loop,
        entry=entry,
        exit_gap=exit_gap,
        out_dir=out_dir,
        trade_mode=str(args.trade_mode),
        backfill_hours=int(args.backfill_hours),
        seed_csv=seed_csv,
        bank_threshold_usdc=float(args.bank_threshold_usdc),
        initial_topup_budget_usdc=float(args.initial_topup_budget_usdc),
        target_frac=float(args.target_frac),
        min_prior_scores=int(args.min_prior_scores),
        max_prior_scores=int(args.max_prior_scores),
        metrics_port=int(args.metrics_port),
        health_max_staleness_sec=float(args.health_max_staleness_sec),
    )

    if str(args.trade_mode).lower() == "real":
        runner.reconcile_startup()

    # Warm backfill (for stable features + threshold seeding)
    bars = loop.run_until_complete(_warm_backfill(clients, market=str(cfg.market), hours=int(args.backfill_hours)))

    if bars.empty and seed_csv is not None and seed_csv.exists():
        try:
            df = pd.read_csv(seed_csv)
            if "timestamp" not in df.columns and "startedAt" in df.columns:
                df["timestamp"] = df["startedAt"]
            if "volume" not in df.columns and "baseTokenVolume" in df.columns:
                df["volume"] = df["baseTokenVolume"]
            need = ["timestamp", "open", "high", "low", "close", "volume"]
            df = df[need].copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

            end = df["timestamp"].max()
            start = end - timedelta(hours=int(args.backfill_hours))
            bars = df[df["timestamp"] >= start].copy().sort_values("timestamp").reset_index(drop=True)

            if not bars.empty:
                print("Warm backfill failed; seeded from CSV:", str(seed_csv))
        except Exception as e:
            print("Seed CSV load failed:", e)

    if not bars.empty:
        runner.bars = bars
        runner.seed_thresholds_from_backfill()

    q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=10_000)
    stop_evt = threading.Event()

    ws_url = str(getattr(clients, "network").websocket_indexer)
    t = threading.Thread(
        target=_run_ws_candles,
        kwargs={"ws_url": ws_url, "market": str(cfg.market), "out_q": q, "stop_evt": stop_evt},
        daemon=True,
    )
    t.start()

    # Optional health/metrics server
    if int(getattr(args, "metrics_port", 0) or 0) > 0:
        threading.Thread(target=runner._serve_http, daemon=True).start()

    print("dYdX WS:", ws_url)
    print("Market:", cfg.market)
    print("Trade mode:", args.trade_mode)
    print("Subaccounts: trading=", int(cfg.subaccount_trading), " bank=", int(cfg.subaccount_bank))
    print("SELL setup:")
    print("  target_frac:", float(runner.target_frac))
    print("  thr_cur_day:", (float(runner.thr_cur_day) if np.isfinite(runner.thr_cur_day) else None))
    print("  exit_gap: tau=", float(EXIT_GAP_TAU), " min_exit_k=", int(EXIT_GAP_MIN_EXIT_K), " hold_min=", int(HOLD_MIN))
    print("  bank_threshold_usdc:", float(cfg.bank_threshold_usdc))
    print("  initial_topup_budget_usdc:", float(runner.initial_topup_budget_usdc), " spent:", float(runner.initial_topup_spent_usdc))
    print("Out dir:", str(out_dir))

    try:
        while not stop_evt.is_set():
            try:
                bar = q.get(timeout=5.0)
            except queue.Empty:
                stall_seconds = int(getattr(args, "stall_seconds", 0) or 0)
                if stall_seconds > 0:
                    age = float(time.time() - float(getattr(runner, "last_closed_wall", 0.0)))
                    if age > float(stall_seconds):
                        print(f"Feed stalled for {age:.1f}s (> {stall_seconds}s); exiting for restart...")
                        stop_evt.set()
                        raise SystemExit(2)
                continue

            runner.on_closed_bar(bar)

            if len(runner.records) and (len(runner.records) % 50 == 0):
                runner.flush_outputs()

    finally:
        stop_evt.set()
        try:
            runner.flush_outputs()
        except Exception:
            pass


if __name__ == "__main__":
    main()
