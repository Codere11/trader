#!/usr/bin/env python3
# Timestamp (UTC): 2025-12-23T18:22:04Z
"""Minimal REAL trade + (optional) futures->spot USDT transfer smoke test.

Goal: prove the end-to-end plumbing works with the smallest reasonable footprint:
- set ISOLATED margin
- set leverage (default 1x)
- open a long MARKET order sized to the minimum notional
- wait a tiny amount of time
- close the position with a reduceOnly MARKET order
- transfer USDT from futures -> spot (default: 1 USDT; or use --transfer-trade-notional / --transfer-all)

WARNING: This places real orders.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from binance_adapter.futures_exec import (
    BinanceAPIError,
    BinanceUSDMFuturesClient,
    FuturesLongOnlyExecutor,
    fetch_futures_exchange_info,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal real trade + transfer smoke test (Binance USDT-M futures)")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--margin-type", type=str, default="ISOLATED", help="ISOLATED or CROSSED")

    ap.add_argument("--leverage", type=int, default=1, help="Target leverage (default: 1x for minimal risk)")
    ap.add_argument(
        "--notional-usdt",
        type=float,
        default=0.0,
        help="If >0, use this notional. If 0, uses exchange MIN_NOTIONAL (plus filter rounding).",
    )
    ap.add_argument("--hold-seconds", type=float, default=0.5, help="Seconds to wait between entry and exit")

    ap.add_argument(
        "--transfer-usdt",
        type=float,
        default=1.0,
        help="USDT amount to transfer futures->spot after closing (set 0 to disable)",
    )
    ap.add_argument(
        "--transfer-trade-notional",
        action="store_true",
        help="If set, transfer the trade's executed entry notional (USDT) futures->spot after closing",
    )
    ap.add_argument(
        "--transfer-all",
        action="store_true",
        help="If set, transfer the entire available USDT balance futures->spot after closing",
    )

    ap.add_argument("--binance-base-url", type=str, default="https://fapi.binance.com")
    ap.add_argument("--sapi-base-url", type=str, default="https://api.binance.com")

    args = ap.parse_args()

    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")
    if not api_key or not api_secret:
        raise SystemExit("BINANCE_API_KEY and BINANCE_API_SECRET env vars are required")

    symbol = str(args.symbol).upper()

    client = BinanceUSDMFuturesClient(
        api_key=api_key,
        api_secret=api_secret,
        base_url=str(args.binance_base_url),
        sapi_base_url=str(args.sapi_base_url),
    )

    # Detect position mode (one-way vs hedge).
    ps = client.get_position_side_dual()
    dual = ps.get("dualSidePosition")
    dual_b = str(dual).lower() == "true" if isinstance(dual, str) else bool(dual)
    position_side = "LONG" if dual_b else None

    filters = fetch_futures_exchange_info(symbol, base_url=str(args.binance_base_url))
    ex = FuturesLongOnlyExecutor(client=client, filters=filters, margin_type=str(args.margin_type))

    # Minimal notional.
    min_notional = float(filters.min_notional)
    notional = float(args.notional_usdt) if float(args.notional_usdt) > 0 else float(min_notional)

    # Leverage: keep minimal by default.
    lev = max(1, int(args.leverage))

    # Set margin + leverage (do not auto-increase leverage).
    _ = ex.ensure_margin_and_leverage(
        symbol=symbol,
        target_leverage=int(lev),
        notional_estimate=float(notional),
        allow_above_target=False,
        margin_usdt=None,
    )

    px = float(client.mark_price(symbol))
    qty = float(ex.qty_for_notional(notional_usdt=float(notional), price_usdt=float(px)))

    print(
        "Placing MINIMAL REAL smoke trade:",
        {
            "symbol": symbol,
            "mark_price": px,
            "target_leverage": int(lev),
            "notional_usdt_requested": notional,
            "qty": qty,
            "filters": asdict(filters),
            "position_side": position_side,
        },
    )

    coid_entry = f"smoke_min_entry_{int(time.time())}"
    entry = ex.open_long_market(symbol=symbol, qty=qty, client_order_id=coid_entry, position_side=position_side)
    entry_px = float(entry.avg_price) if float(entry.avg_price) > 0 else float(client.ticker_price(symbol))
    entry_qty = float(entry.executed_qty) if float(entry.executed_qty) > 0 else float(qty)

    print("ENTRY:", {"order": entry.order_id, "client": entry.client_order_id, "qty": entry_qty, "avg_price": entry_px})

    # Best-effort actual executed notional for the entry order.
    entry_notional_usdt = 0.0
    try:
        if isinstance(entry.raw, dict) and entry.raw.get("cumQuote") is not None:
            entry_notional_usdt = float(entry.raw.get("cumQuote"))
    except Exception:
        entry_notional_usdt = 0.0

    if float(entry_notional_usdt) <= 0.0:
        entry_notional_usdt = float(entry_px) * float(entry_qty)

    time.sleep(max(0.0, float(args.hold_seconds)))

    coid_exit = f"smoke_min_exit_{int(time.time())}"
    exit_res = ex.close_long_market(symbol=symbol, qty=entry_qty, client_order_id=coid_exit, position_side=position_side)
    exit_px = float(exit_res.avg_price) if float(exit_res.avg_price) > 0 else float(client.ticker_price(symbol))
    exit_qty = float(exit_res.executed_qty) if float(exit_res.executed_qty) > 0 else float(entry_qty)

    print("EXIT:", {"order": exit_res.order_id, "client": exit_res.client_order_id, "qty": exit_qty, "avg_price": exit_px})

    approx_pnl_usdt = (exit_px - entry_px) * float(exit_qty)
    print("Approx PnL (USDT, excl fees):", round(float(approx_pnl_usdt), 6))

    # Transfer to spot (optional)
    transfer_mode = "NONE"
    if bool(args.transfer_all):
        transfer_mode = "ALL"
    elif bool(args.transfer_trade_notional):
        transfer_mode = "TRADE_NOTIONAL"
    elif float(args.transfer_usdt) > 0.0:
        transfer_mode = "FIXED"

    if transfer_mode != "NONE":
        # Give Binance a moment to settle balance.
        time.sleep(0.5)

        avail = client.futures_available_balance("USDT")
        avail_f = float(avail) if avail is not None else 0.0

        if transfer_mode == "ALL":
            amt = float(avail_f)
        elif transfer_mode == "TRADE_NOTIONAL":
            amt = min(float(entry_notional_usdt), float(avail_f))
        else:
            amt = min(float(args.transfer_usdt), float(avail_f))

        if amt <= 0.0:
            print("TRANSFER: skipped (no available USDT)", {"mode": transfer_mode, "availableBalance": avail})
        else:
            tx = client.futures_transfer(asset="USDT", amount=float(amt), transfer_type=2)  # futures -> spot
            print(
                "TRANSFER futures->spot:",
                {"mode": transfer_mode, "amount_usdt": amt, "trade_notional_usdt": entry_notional_usdt, "availableBalance": avail, "tx": tx},
            )


if __name__ == "__main__":
    try:
        main()
    except BinanceAPIError as e:
        raise SystemExit(f"Binance API error: {e}")
