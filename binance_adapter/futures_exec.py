# Timestamp (UTC): 2025-12-22T14:19:18Z
from __future__ import annotations

"""Binance USDT-M futures execution helpers.

This module intentionally avoids non-stdlib crypto deps; it signs requests with HMAC-SHA256.

Scope (minimal):
- Fetch exchange filters (stepSize, minQty, MIN_NOTIONAL, tickSize)
- Fetch leverage brackets
- Set margin type + leverage
- Place MARKET open/close orders (long-only)

Do NOT print secrets.
"""

from dataclasses import dataclass
import hashlib
import hmac
import math
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple
import urllib.parse

import requests


class BinanceAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class SymbolFilters:
    symbol: str
    min_qty: float
    step_size: float
    min_notional: float
    tick_size: float


def _floor_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return math.floor(x / step) * step


def _ceil_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return math.ceil(x / step) * step


def _ensure(cond: bool, msg: str) -> None:
    if not cond:
        raise BinanceAPIError(msg)


def fetch_spot_price(symbol: str, *, timeout: float = 10.0) -> float:
    """Public spot ticker price. Useful for EURUSDT conversion."""
    r = requests.get(
        "https://api.binance.com/api/v3/ticker/price",
        params={"symbol": symbol.upper()},
        timeout=timeout,
    )
    r.raise_for_status()
    obj = r.json()
    return float(obj["price"])


def fetch_futures_exchange_info(symbol: str, *, base_url: str = "https://fapi.binance.com", timeout: float = 10.0) -> SymbolFilters:
    r = requests.get(
        f"{base_url}/fapi/v1/exchangeInfo",
        params={"symbol": symbol.upper()},
        timeout=timeout,
    )
    r.raise_for_status()
    info = r.json()
    syms = info.get("symbols", [])
    _ensure(bool(syms), f"exchangeInfo returned no symbol data for {symbol}")
    s = syms[0]

    tick_size = None
    min_qty = None
    step_size = None
    min_notional = None

    for f in s.get("filters", []):
        ft = f.get("filterType")
        if ft == "PRICE_FILTER":
            tick_size = float(f.get("tickSize"))
        elif ft == "LOT_SIZE":
            min_qty = float(f.get("minQty"))
            step_size = float(f.get("stepSize"))
        elif ft == "MIN_NOTIONAL":
            min_notional = float(f.get("notional"))

    _ensure(tick_size is not None, "Missing PRICE_FILTER.tickSize")
    _ensure(min_qty is not None and step_size is not None, "Missing LOT_SIZE minQty/stepSize")
    _ensure(min_notional is not None, "Missing MIN_NOTIONAL.notional")

    return SymbolFilters(
        symbol=str(symbol).upper(),
        min_qty=float(min_qty),
        step_size=float(step_size),
        min_notional=float(min_notional),
        tick_size=float(tick_size),
    )


class BinanceUSDMFuturesClient:
    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        base_url: str = "https://fapi.binance.com",
        sapi_base_url: str = "https://api.binance.com",
        recv_window: int = 5000,
        timeout: float = 10.0,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.sapi_base_url = str(sapi_base_url).rstrip("/")
        self.api_key = str(api_key)
        self.api_secret = str(api_secret)
        self.recv_window = int(recv_window)
        self.timeout = float(timeout)

        self._session = requests.Session()
        self._session.headers.update({"X-MBX-APIKEY": self.api_key})

    def _sign(self, qs: str) -> str:
        sig = hmac.new(self.api_secret.encode("utf-8"), qs.encode("utf-8"), hashlib.sha256).hexdigest()
        return sig

    def _request_on(
        self,
        base_url: str,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        signed: bool = False,
    ) -> Any:
        p: Dict[str, Any] = dict(params or {})
        if signed:
            p.setdefault("timestamp", int(time.time() * 1000))
            p.setdefault("recvWindow", int(self.recv_window))

        qs = urllib.parse.urlencode(p, doseq=True)
        if signed:
            sig = self._sign(qs)
            qs = qs + ("&" if qs else "") + f"signature={sig}"

        base = str(base_url).rstrip("/")
        url = f"{base}{path}"
        if qs:
            url = url + "?" + qs

        r = self._session.request(method.upper(), url, timeout=self.timeout)
        # Binance uses JSON body on error with code/msg.
        try:
            obj = r.json()
        except Exception:
            r.raise_for_status()
            raise

        if r.status_code >= 400:
            code = obj.get("code") if isinstance(obj, dict) else None
            msg = obj.get("msg") if isinstance(obj, dict) else str(obj)
            raise BinanceAPIError(f"HTTP {r.status_code} {path} failed: {code} {msg}")

        return obj

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        signed: bool = False,
    ) -> Any:
        return self._request_on(self.base_url, method, path, params=params, signed=signed)

    # -------- signed endpoints --------
    def get_account(self) -> Any:
        return self._request("GET", "/fapi/v2/account", signed=True)

    def futures_available_balance(self, asset: str) -> Optional[float]:
        """Best-effort available balance for an asset in the USDT-M futures wallet."""
        try:
            obj = self.get_account()
            for a in obj.get("assets", []) if isinstance(obj, dict) else []:
                if not isinstance(a, Mapping):
                    continue
                if str(a.get("asset", "")).upper() != str(asset).upper():
                    continue
                v = a.get("availableBalance")
                if v is None:
                    return None
                return float(v)
        except Exception:
            return None
        return None

    def get_position_side_dual(self) -> Any:
        # returns {"dualSidePosition": "true"/"false"}
        return self._request("GET", "/fapi/v1/positionSide/dual", signed=True)

    def get_leverage_brackets(self, symbol: str) -> Any:
        return self._request("GET", "/fapi/v1/leverageBracket", params={"symbol": symbol.upper()}, signed=True)

    def set_margin_type(self, symbol: str, margin_type: str) -> Any:
        return self._request(
            "POST",
            "/fapi/v1/marginType",
            params={"symbol": symbol.upper(), "marginType": margin_type},
            signed=True,
        )

    def set_leverage(self, symbol: str, leverage: int) -> Any:
        return self._request(
            "POST",
            "/fapi/v1/leverage",
            params={"symbol": symbol.upper(), "leverage": int(leverage)},
            signed=True,
        )

    def new_order(self, **params: Any) -> Any:
        return self._request("POST", "/fapi/v1/order", params=params, signed=True)

    def futures_transfer(self, *, asset: str, amount: float, transfer_type: int) -> Any:
        """Transfer between Spot and USDT-M futures wallets via SAPI.

        Common transfer_type values:
        - 1: Spot -> USDT-M futures
        - 2: USDT-M futures -> Spot
        """
        amt = float(amount)
        _ensure(amt > 0.0, "transfer amount must be > 0")

        params = {
            "asset": str(asset).upper(),
            # Binance expects a decimal string.
            "amount": f"{amt:.8f}",
            "type": int(transfer_type),
        }
        return self._request_on(self.sapi_base_url, "POST", "/sapi/v1/futures/transfer", params=params, signed=True)

    # -------- public endpoints (no signature) --------
    def ticker_price(self, symbol: str) -> float:
        obj = self._request("GET", "/fapi/v1/ticker/price", params={"symbol": symbol.upper()}, signed=False)
        return float(obj["price"])

    def mark_price(self, symbol: str) -> float:
        obj = self._request("GET", "/fapi/v1/premiumIndex", params={"symbol": symbol.upper()}, signed=False)
        return float(obj["markPrice"])


def _extract_brackets(brackets_resp: Any) -> Optional[List[Mapping[str, Any]]]:
    try:
        arr = brackets_resp
        if isinstance(arr, list) and arr:
            brs = arr[0].get("brackets", [])
        elif isinstance(arr, dict) and "brackets" in arr:
            brs = arr.get("brackets", [])
        else:
            return None
        return [b for b in brs if isinstance(b, Mapping)]
    except Exception:
        return None


def max_leverage_for_notional(brackets_resp: Any, *, notional: float) -> Optional[int]:
    """Return max initial leverage for a given notional based on /leverageBracket."""
    try:
        brs = _extract_brackets(brackets_resp)
        if not brs:
            return None

        n = float(notional)
        for b in brs:
            cap = float(b.get("notionalCap", 0))
            floor = float(b.get("notionalFloor", 0))
            lev = int(b.get("initialLeverage"))
            if n >= floor and (cap == 0 or n <= cap):
                return lev
        return None
    except Exception:
        return None


def max_symbol_leverage(brackets_resp: Any) -> Optional[int]:
    """Return the maximum leverage supported for the symbol (highest initialLeverage)."""
    try:
        brs = _extract_brackets(brackets_resp)
        if not brs:
            return None
        return max(int(b.get("initialLeverage")) for b in brs)
    except Exception:
        return None


def best_leverage_for_margin(brackets_resp: Any, *, margin_usdt: float) -> int:
    """Choose the highest leverage L such that L is allowed for notional = margin_usdt * L."""
    m = float(margin_usdt)
    if m <= 0:
        return 1

    max_l = max_symbol_leverage(brackets_resp)
    if max_l is None:
        return 1

    for lev in range(int(max_l), 0, -1):
        n = m * float(lev)
        allowed = max_leverage_for_notional(brackets_resp, notional=n)
        if allowed is None:
            continue
        if int(lev) <= int(allowed):
            return int(lev)

    return 1


@dataclass
class MarketOrderResult:
    symbol: str
    side: str
    executed_qty: float
    avg_price: float
    order_id: Optional[int]
    client_order_id: Optional[str]
    raw: Any


class FuturesLongOnlyExecutor:
    def __init__(
        self,
        *,
        client: BinanceUSDMFuturesClient,
        filters: SymbolFilters,
        margin_type: str = "ISOLATED",
    ):
        self.client = client
        self.filters = filters
        self.margin_type = str(margin_type).upper()

    def ensure_margin_and_leverage(
        self,
        *,
        symbol: str,
        target_leverage: int,
        notional_estimate: float,
        allow_above_target: bool = True,
        margin_usdt: Optional[float] = None,
    ) -> int:
        """Ensure margin type is set and set leverage.

        If allow_above_target=True (default), this chooses the highest feasible leverage for the given
        margin (if provided), constrained by Binance leverage brackets.

        - margin_usdt should be the intended isolated margin amount in USDT.
        - notional_estimate is kept for fallback behavior and compatibility.
        """
        sym = symbol.upper()

        # Margin type: ignore "no need to change" errors.
        try:
            self.client.set_margin_type(sym, self.margin_type)
        except BinanceAPIError as e:
            if "No need to change margin type" not in str(e):
                raise

        br = self.client.get_leverage_brackets(sym)

        # Prefer a margin-based max leverage calculation (stable w.r.t. notional = margin * leverage).
        lev = None
        if allow_above_target and margin_usdt is not None and float(margin_usdt) > 0.0:
            lev = int(best_leverage_for_margin(br, margin_usdt=float(margin_usdt)))

        # Fallback to notional-estimate clamp.
        if lev is None:
            max_lev = max_leverage_for_notional(br, notional=float(notional_estimate))
            if max_lev is None:
                max_lev = int(target_leverage)

            desired = int(target_leverage)
            if allow_above_target:
                desired = max(desired, int(max_lev))

            lev = min(int(desired), int(max_lev))

        lev = max(1, int(lev))
        self.client.set_leverage(sym, lev)
        return int(lev)

    def qty_for_notional(self, *, notional_usdt: float, price_usdt: float) -> float:
        """Compute a valid futures quantity given notional and exchange filters."""
        px = float(price_usdt)
        _ensure(px > 0, "price must be > 0")

        raw_qty = float(notional_usdt) / px
        qty = _floor_to_step(raw_qty, self.filters.step_size)

        if qty < self.filters.min_qty:
            qty = float(self.filters.min_qty)

        # Ensure MIN_NOTIONAL (use *price* approximation; futures uses notional threshold).
        notional_final = qty * px
        if notional_final < float(self.filters.min_notional):
            qty = _ceil_to_step(float(self.filters.min_notional) / px, self.filters.step_size)

        # One more clamp.
        qty = max(qty, float(self.filters.min_qty))
        qty = round(qty, 12)
        return qty

    def open_long_market(self, *, symbol: str, qty: float, client_order_id: Optional[str] = None, position_side: Optional[str] = None) -> MarketOrderResult:
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": "BUY",
            "type": "MARKET",
            "quantity": float(qty),
            "newOrderRespType": "RESULT",
        }
        if client_order_id:
            params["newClientOrderId"] = str(client_order_id)
        if position_side:
            params["positionSide"] = str(position_side)

        obj = self.client.new_order(**params)
        executed_qty = float(obj.get("executedQty", 0.0))
        avg_price = float(obj.get("avgPrice", 0.0) or 0.0)
        order_id = obj.get("orderId")
        return MarketOrderResult(
            symbol=symbol.upper(),
            side="BUY",
            executed_qty=executed_qty,
            avg_price=avg_price,
            order_id=int(order_id) if order_id is not None else None,
            client_order_id=obj.get("clientOrderId"),
            raw=obj,
        )

    def close_long_market(self, *, symbol: str, qty: float, client_order_id: Optional[str] = None, position_side: Optional[str] = None) -> MarketOrderResult:
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": "SELL",
            "type": "MARKET",
            "quantity": float(qty),
            "reduceOnly": "true",
            "newOrderRespType": "RESULT",
        }
        if client_order_id:
            params["newClientOrderId"] = str(client_order_id)
        if position_side:
            params["positionSide"] = str(position_side)

        obj = self.client.new_order(**params)
        executed_qty = float(obj.get("executedQty", 0.0))
        avg_price = float(obj.get("avgPrice", 0.0) or 0.0)
        order_id = obj.get("orderId")
        return MarketOrderResult(
            symbol=symbol.upper(),
            side="SELL",
            executed_qty=executed_qty,
            avg_price=avg_price,
            order_id=int(order_id) if order_id is not None else None,
            client_order_id=obj.get("clientOrderId"),
            raw=obj,
        )
