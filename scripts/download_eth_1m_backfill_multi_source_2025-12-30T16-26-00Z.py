#!/usr/bin/env python3
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
import sys
import requests
import pandas as pd

# dYdX v4 indexer (post-2023)
DEFAULT_INDEXER = 'https://indexer.dydx.trade'
MARKET = 'ETH-USD'
COLS = ['timestamp','open','high','low','close','volume']

BINANCE_BASE = 'https://api.binance.com'
BINANCE_SYMBOL = 'ETHUSDT'


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00','Z')


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_daily(out_dir: Path, rows: list[dict]) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)
    df['date'] = df['timestamp'].dt.date
    n = 0
    for d, g in df.groupby('date'):
        p = out_dir / f"minute_bars_{d}.csv"
        g[COLS].to_csv(p, index=False)
        n += 1
    return n

# ---------- dYdX v4 ----------

def fetch_dydx_chunk(sess: requests.Session, indexer_base: str, start: datetime, end: datetime, limit: int = 1000) -> list[dict]:
    url = f"{indexer_base.rstrip('/')}/v4/candles/perpetualMarkets/{MARKET}"
    params = {
        'resolution': '1MIN',
        'fromISO': iso(start),
        'toISO': iso(end),
        'limit': str(limit),
    }
    r = sess.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json() or {}
    candles = data.get('candles') or []
    rows: list[dict] = []
    for c in candles:
        try:
            rows.append({
                'timestamp': pd.to_datetime(c['startedAt'], utc=True),
                'open': float(c['open']),
                'high': float(c['high']),
                'low': float(c['low']),
                'close': float(c['close']),
                'volume': float(c.get('baseTokenVolume') or 0.0),
            })
        except Exception:
            continue
    return rows


def autodetect_dydx_earliest(sess: requests.Session, indexer_base: str) -> datetime | None:
    probe = datetime(2020,1,1,tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    while probe < now:
        try:
            rows = fetch_dydx_chunk(sess, indexer_base, probe, probe + timedelta(days=2))
            if rows:
                return rows[0]['timestamp'].to_pydatetime().replace(tzinfo=timezone.utc)
        except Exception:
            pass
        probe += timedelta(days=31)
    return None

# ---------- Binance (pre-dYdX) ----------

def fetch_binance_klines(start_ms: int, end_ms: int, limit: int = 1000) -> list[list]:
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {
        'symbol': BINANCE_SYMBOL,
        'interval': '1m',
        'startTime': start_ms,
        'endTime': end_ms,
        'limit': limit,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json() or []


def backfill_binance(out_dir: Path, start: datetime, end: datetime) -> None:
    cur = start
    while cur < end:
        nxt = min(end, cur + timedelta(days=3))
        # Binance expects ms epoch
        start_ms = int(cur.timestamp() * 1000)
        end_ms = int(nxt.timestamp() * 1000) - 1
        try:
            kl = fetch_binance_klines(start_ms, end_ms, limit=1000)
        except Exception as e:
            print('WARN binance fetch failed', cur, '->', nxt, e)
            time.sleep(1.0)
            cur = nxt
            continue
        rows: list[dict] = []
        for k in kl:
            # kline fields per Binance docs
            ts = pd.to_datetime(k[0], unit='ms', utc=True)
            o, h, l, c, v = float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
            rows.append({'timestamp': ts, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})
        write_daily(out_dir, rows)
        cur = nxt
        time.sleep(0.2)


def main() -> None:
    ap = argparse.ArgumentParser(description='Backfill ETH 1m candles 2020->now using dYdX v4 when available, Binance otherwise')
    ap.add_argument('--indexer', default=DEFAULT_INDEXER)
    ap.add_argument('--out-dir', default='data/eth_usd')
    ap.add_argument('--start', default='2020-01-01')
    ap.add_argument('--end', default=None, help='YYYY-MM-DD (default: today)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) if args.end else datetime.now(timezone.utc)

    sess = requests.Session()
    earliest_dydx = autodetect_dydx_earliest(sess, args.indexer) or end_dt
    # 1) Pre-dYdX range via Binance
    if start_dt < earliest_dydx:
        print(f'Backfilling via Binance: {start_dt.date()} -> {(earliest_dydx - timedelta(days=1)).date()}')
        backfill_binance(out_dir, start_dt, earliest_dydx)
    else:
        print('No Binance backfill needed.')

    # 2) dYdX range (skip if already present files)
    cur = max(start_dt, earliest_dydx)
    print(f'Filling via dYdX: {cur.date()} -> {end_dt.date()}')
    batch: list[dict] = []
    while cur < end_dt:
        nxt = min(end_dt, cur + timedelta(hours=24))
        try:
            rows = fetch_dydx_chunk(sess, args.indexer, cur, nxt)
        except Exception as e:
            print('WARN dydx fetch failed', cur, '->', nxt, e)
            time.sleep(1.0)
            cur = nxt
            continue
        batch.extend(rows)
        if batch and (len(batch) >= 12*60 or (rows and pd.Timestamp(rows[-1]['timestamp']).hour in {0,12})):
            write_daily(out_dir, batch)
            batch = []
        cur = nxt
        time.sleep(0.15)
    if batch:
        write_daily(out_dir, batch)

    print('Done.')

if __name__ == '__main__':
    main()
