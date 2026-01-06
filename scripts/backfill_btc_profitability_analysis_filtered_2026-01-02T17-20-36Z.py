#!/usr/bin/env python3
"""Timestamp (UTC): 2026-01-02T17:20:36Z

Backfill `data/btc_profitability_analysis_filtered.csv` from its last timestamp up to now.

- Uses Binance spot 1m klines (api.binance.com /api/v3/klines) for BTCUSDT.
- Appends rows in the existing CSV schema.
- Leaves oracle-related columns blank (perfect_trade_action / perfect_trade_profit_pct).

This is intended to extend the market tape so other scripts can run on 2024->now.
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = REPO_ROOT / "data" / "btc_profitability_analysis_filtered.csv"
BINANCE_BASE = "https://api.binance.com"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LIMIT = 1000

OUT_COLS = [
    "unix_timestamp",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "price_change_pct",
    "perfect_trade_action",
    "perfect_trade_profit_pct",
    "date",
]


def _read_last_data_line(path: Path) -> str:
    # Read last non-empty line efficiently.
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        if end == 0:
            raise SystemExit(f"Empty file: {path}")
        # read backwards in chunks
        chunk = 4096
        buf = b""
        pos = end
        while pos > 0:
            step = min(chunk, pos)
            pos -= step
            f.seek(pos)
            buf = f.read(step) + buf
            if b"\n" in buf:
                lines = buf.splitlines()
                # last line could be empty
                for line in reversed(lines):
                    if line.strip():
                        return line.decode("utf-8", errors="replace")
        # fallback
        return buf.decode("utf-8", errors="replace").strip().splitlines()[-1]


def _parse_last_timestamp(last_line: str) -> datetime:
    # CSV columns: unix_timestamp,timestamp,... where timestamp contains a space.
    # Use csv.reader to parse safely.
    row = next(csv.reader([last_line]))
    if len(row) < 2:
        raise SystemExit(f"Unexpected last line format (need >=2 columns): {last_line[:200]}")
    ts_s = row[1]
    ts = pd.to_datetime(ts_s, utc=True, errors="raise")
    return ts.to_pydatetime()


def _fetch_klines(start_ms: int, end_ms: int) -> list[list]:
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": int(start_ms),
        "endTime": int(end_ms),
        "limit": LIMIT,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json() or []


def _kline_rows(kl: Iterable[list]) -> list[dict]:
    rows: list[dict] = []
    for k in kl:
        # Binance kline fields: [ openTime, open, high, low, close, volume, closeTime, ...]
        ts = pd.to_datetime(int(k[0]), unit="ms", utc=True)
        o = float(k[1]); h = float(k[2]); l = float(k[3]); c = float(k[4]); v = float(k[5])
        unix_ts = int(ts.timestamp())
        price_change_pct = (c / o - 1.0) * 100.0 if o != 0 else 0.0
        rows.append(
            {
                "unix_timestamp": unix_ts,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
                "price_change_pct": float(price_change_pct),
                "perfect_trade_action": "",
                "perfect_trade_profit_pct": "",
                "date": ts.strftime("%Y-%m-%d"),
            }
        )
    return rows


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing: {CSV_PATH}")

    last_line = _read_last_data_line(CSV_PATH)
    last_ts = _parse_last_timestamp(last_line)

    start = (last_ts.replace(tzinfo=timezone.utc) + timedelta(minutes=1)).replace(second=0, microsecond=0)
    # fetch up to the last fully-closed minute (floor to minute)
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    if start >= end:
        print(f"Already up to date. last_ts={last_ts.isoformat()} start={start.isoformat()} end={end.isoformat()}")
        return

    print(f"Backfilling {CSV_PATH}")
    print(f"  last_ts: {last_ts.isoformat()}")
    print(f"  start:   {start.isoformat()}")
    print(f"  end:     {end.isoformat()}")

    total_rows = 0
    cur = start

    # append mode; assume file already has header
    with CSV_PATH.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)

        while cur < end:
            # Binance allows up to 1000 bars. Request a window up to LIMIT minutes.
            nxt = min(end, cur + timedelta(minutes=LIMIT))
            start_ms = int(cur.timestamp() * 1000)
            end_ms = int(nxt.timestamp() * 1000) - 1

            try:
                kl = _fetch_klines(start_ms, end_ms)
            except Exception as e:
                print("WARN fetch failed", cur.isoformat(), "->", nxt.isoformat(), e)
                time.sleep(2.0)
                cur = nxt
                continue

            rows = _kline_rows(kl)
            # guard against duplicates
            for r in rows:
                ts = pd.to_datetime(r["timestamp"], utc=True)
                if ts.to_pydatetime().replace(tzinfo=timezone.utc) < start:
                    continue
                w.writerow(r)
                total_rows += 1

            cur = nxt
            time.sleep(0.25)

    print(f"Done. Appended {total_rows} rows.")


if __name__ == "__main__":
    main()
