#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import requests
import pandas as pd

DEFAULT_INDEXER = os.getenv('DYDX_INDEXER_REST', 'https://indexer.dydx.trade').rstrip('/')
MARKET = 'ETH-USD'

COLS = ['timestamp','open','high','low','close','volume']


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00','Z')


def fetch_chunk(session: requests.Session, indexer_base: str, start: datetime, end: datetime, limit: int = 1000) -> list[dict]:
    url = f"{indexer_base}/v4/candles/perpetualMarkets/{MARKET}"
    params = {
        'resolution': '1MIN',
        'fromISO': iso(start),
        'toISO': iso(end),
        'limit': str(limit),
    }
    r = session.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
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


def write_daily(out_dir: Path, rows: list[dict]) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)
    df['date'] = df['timestamp'].dt.date
    n_files = 0
    for d, g in df.groupby('date'):
        p = out_dir / f"minute_bars_{d}.csv"
        header = not p.exists()
        g[COLS].to_csv(p, mode='a', header=header, index=False)
        n_files += 1
    return n_files


def main() -> None:
    ap = argparse.ArgumentParser(description='Download dYdX v4 ETH-USD 1m candles to daily CSVs')
    ap.add_argument('--indexer', default=DEFAULT_INDEXER)
    ap.add_argument('--start', default='auto', help='UTC date YYYY-MM-DD or "auto" to probe earliest available')
    ap.add_argument('--end', default=None, help='UTC date YYYY-MM-DD (default: today)')
    ap.add_argument('--out-dir', default='data/eth_usd')
    ap.add_argument('--chunk-hours', type=int, default=24, help='Fetch window size per request')
    ap.add_argument('--sleep-ms', type=int, default=200, help='Sleep between requests to be gentle')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Time range
    if args.start.strip().lower() == 'auto':
        # Probe with a far past date; if 404, increment by months until 200
        probe = datetime(2020,1,1,tzinfo=timezone.utc)
        sess = requests.Session()
        while True:
            try:
                rows = fetch_chunk(sess, args.indexer, probe, probe + timedelta(days=1))
                if rows:
                    start_dt = rows[0]['timestamp'].to_pydatetime().replace(tzinfo=timezone.utc)
                    break
            except Exception:
                pass
            probe += timedelta(days=31)
            if probe > datetime.now(timezone.utc):
                raise SystemExit('Failed to auto-detect earliest candles')
    else:
        try:
            start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        except Exception:
            start_dt = datetime(2020,1,1,tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) if args.end else datetime.now(timezone.utc)

    sess = requests.Session()
    cur = start_dt
    last_written = None
    rows_batch: list[dict] = []
    while cur < end_dt:
        nxt = min(end_dt, cur + timedelta(hours=int(args.chunk_hours)))
        try:
            rows = fetch_chunk(sess, args.indexer, cur, nxt)
        except Exception as e:
            print('WARN fetch failed at', cur.isoformat(), '->', nxt.isoformat(), 'err=', e)
            time.sleep(1.0)
            cur = nxt
            continue
        rows_batch.extend(rows)
        # Flush at day boundary or every ~12 hours of data
        if rows_batch and (len(rows_batch) >= 12*60 or (rows and pd.Timestamp(rows[-1]['timestamp']).to_pydatetime().hour in {0,12})):
            write_daily(out_dir, rows_batch)
            last_written = rows_batch[-1]['timestamp']
            rows_batch = []
        cur = nxt
        time.sleep(max(0.0, float(args.sleep_ms)/1000.0))

    if rows_batch:
        write_daily(out_dir, rows_batch)
        last_written = rows_batch[-1]['timestamp']

    print('Completed. Last candle:', last_written)

if __name__ == '__main__':
    main()
