#!/usr/bin/env python3
# Timestamp (UTC): 2025-12-22T22:20:51Z
"""One-off real-trade smoke test on Binance USDT-M futures.

This script is NOT the strategy loop. It exists to quickly validate:
- signed API access works (IP whitelist + permissions)
- MARKET open/close works for BTCUSDT
- futures -> spot transfer works (stablecoin siphon plumbing)

By default it does NOT transfer any funds. Use --transfer-usdt to transfer a fixed amount.

WARNING: This places real orders.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from binance_adapter.futures_exec import (
    BinanceUSDMFuturesClient,
    FuturesLongOnlyExecutor,
    fetch_futures_exchange_info,
    fetch_spot_price,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Real trade + transfer smoke test (Binance USDT-M futures)")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--margin-eur", type=float, default=1.0, help="Intended margin amount in EUR")
    ap.add_argument("--hold-seconds", type=float, default=1.0, help="Seconds to wait between entry and exit")
    ap.add_argument("--margin-type", type=str, default="ISOLATED", help="ISOLATED or CROSSED")
    ap.add_argument("--binance-base-url", type=str, default="https://fapi.binance.com")
    ap.add_argument("--sapi-base-url", type=str, default="https://api.binance.com")
    ap.add_argument(
        "--transfer-usdt",
        type=float,
        default=0.0,
        help="If >0, transfer this USDT amount from futures -> spot after closing the trade.",
    )

    args = ap.parse_args()

    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")
    if not api_key or not api_secret:
        raise SystemExit("BINANCE_API_KEY and BINANCE_API_SECRET env vars are required")

    symbol = str(args.symbol).upper()

    eurusdt = float(fetch_spot_price("EURUSDT"))
    margin_eur = float(args.margin_eur)
    margin_usdt = float(margin_eur) * float(eurusdt)

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

    # Choose maximum feasible leverage for this margin.
    px = float(client.mark_price(symbol))
    lev = int(
        ex.ensure_margin_and_leverage(
            symbol=symbol,
            target_leverage=100,
            notional_estimate=float(margin_usdt) * 100.0,
            allow_above_target=True,
            margin_usdt=float(margin_usdt),
        )
    )

    notional_usdt = float(margin_usdt) * float(lev)
    qty = float(ex.qty_for_notional(notional_usdt=float(notional_usdt), price_usdt=float(px)))

    print(
        "Placing REAL smoke trade:",
        {
            "symbol": symbol,
            "margin_eur": margin_eur,
            "eurusdt": eurusdt,
            "margin_usdt": margin_usdt,
            "mark_price": px,
            "leverage": lev,
            "notional_usdt": notional_usdt,
            "qty": qty,
            "filters": asdict(filters),
            "position_side": position_side,
        },
    )

    coid_entry = f"smoke_entry_{int(time.time())}"
    entry = ex.open_long_market(symbol=symbol, qty=qty, client_order_id=coid_entry, position_side=position_side)
    entry_px = float(entry.avg_price) if float(entry.avg_price) > 0 else float(client.ticker_price(symbol))
    entry_qty = float(entry.executed_qty) if float(entry.executed_qty) > 0 else float(qty)

    print("ENTRY:", {"order": entry.order_id, "client": entry.client_order_id, "qty": entry_qty, "avg_price": entry_px})

    time.sleep(max(0.0, float(args.hold_seconds)))

    coid_exit = f"smoke_exit_{int(time.time())}"
    exit_res = ex.close_long_market(symbol=symbol, qty=entry_qty, client_order_id=coid_exit, position_side=position_side)
    exit_px = float(exit_res.avg_price) if float(exit_res.avg_price) > 0 else float(client.ticker_price(symbol))
    exit_qty = float(exit_res.executed_qty) if float(exit_res.executed_qty) > 0 else float(entry_qty)

    print("EXIT:", {"order": exit_res.order_id, "client": exit_res.client_order_id, "qty": exit_qty, "avg_price": exit_px})

    approx_pnl_usdt = (exit_px - entry_px) * float(exit_qty)
    print("Approx PnL (USDT, excl fees):", round(float(approx_pnl_usdt), 6))

    if float(args.transfer_usdt) > 0.0:
        amt = float(args.transfer_usdt)
        avail = client.futures_available_balance("USDT")
        if avail is not None:
            amt = min(float(amt), float(avail))
        if amt <= 0.0:
            raise SystemExit(f"No available USDT to transfer (availableBalance={avail})")

        tx = client.futures_transfer(asset="USDT", amount=float(amt), transfer_type=2)
        print("TRANSFER futures->spot:", tx)


if __name__ == "__main__":
    main()
