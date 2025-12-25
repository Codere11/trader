#!/usr/bin/env python3
"""Bulk-download dYdX v4 (Chain) perpetual market candles and save as a CSV.

Design goals:
- Normal bulk download (chunked API pagination), not minute-by-minute manual work.
- Output schema compatible with existing BTCUSDT market CSVs.
- Fast enough to run locally: uses time-window queries and optional parallelism.

Notes:
- dYdX v4 indexer candles endpoint supports max limit=1000 per request.
- History length is market-specific; the script will download whatever exists in the
  requested window.

Example:
  python3 scripts/download_dydx_btc_usd_perp_1m_2025-12-24T11-15-16Z.py \
    --ticker BTC-USD \
    --from-iso 2018-01-01T00:00:00Z \
    --resolution 1MIN \
    --workers 6

Output:
- data/dydx_<TICKER>_<RES>_<ts>.csv
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

INDEXER_BASE = "https://indexer.dydx.trade/v4"


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _parse_iso(s: str) -> datetime:
    # Accepts Z or +00:00, returns tz-aware UTC datetime.
    s = str(s).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def _to_iso(dt: datetime) -> str:
    # Always emit Z form.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _get(session: requests.Session, url: str, params: dict, timeout_s: float, retries: int = 10) -> dict:
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout_s)

            # Handle rate limiting explicitly.
            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                reset_ms = resp.headers.get("ratelimit-reset")
                wait_s = 5.0
                try:
                    if retry_after is not None:
                        wait_s = float(retry_after)
                    elif reset_ms is not None:
                        wait_s = max(0.0, (float(reset_ms) / 1000.0) - time.time())
                except Exception:
                    wait_s = 5.0

                # Add a bit of jitter/backoff.
                wait_s = max(0.5, wait_s) + (0.25 * (2**min(attempt, 6)))
                time.sleep(wait_s)
                continue

            resp.raise_for_status()

            payload = resp.json()
            if isinstance(payload, dict) and payload.get("errors"):
                # Surface request validation errors early.
                raise RuntimeError(str(payload["errors"]))
            return payload
        except Exception:
            if attempt >= retries:
                raise
            # small backoff
            time.sleep(0.25 * (2**min(attempt, 6)))
    raise RuntimeError("unreachable")


def _fetch_candles(
    session: requests.Session,
    *,
    ticker: str,
    resolution: str,
    from_iso: str | None,
    to_iso: str,
    limit: int,
    timeout_s: float,
) -> list[dict]:
    url = f"{INDEXER_BASE}/candles/perpetualMarkets/{ticker}"
    params: dict[str, object] = {
        "resolution": resolution,
        "limit": int(limit),
        "toISO": to_iso,
    }
    if from_iso is not None:
        params["fromISO"] = from_iso

    data = _get(session, url, params, timeout_s)
    return list(data.get("candles", []))


def _has_any_candle_at_or_before(
    session: requests.Session,
    *,
    ticker: str,
    resolution: str,
    to_dt: datetime,
    timeout_s: float,
) -> bool:
    candles = _fetch_candles(
        session,
        ticker=ticker,
        resolution=resolution,
        from_iso=None,
        to_iso=_to_iso(to_dt),
        limit=1,
        timeout_s=timeout_s,
    )
    return bool(candles)


def find_first_available_minute(
    *,
    ticker: str,
    resolution: str,
    search_from: datetime,
    search_to: datetime,
    timeout_s: float,
) -> datetime | None:
    """Find the first minute where candles exist (market history start).

    If there is no data at all up to search_to, returns None.

    Implementation: binary-search for the smallest toISO that returns any candle.
    Then read the candle returned at that boundary; when the boundary is within <1 minute
    of the true start, that candle is the first available minute.
    """

    if search_from.tzinfo is None:
        search_from = search_from.replace(tzinfo=timezone.utc)
    if search_to.tzinfo is None:
        search_to = search_to.replace(tzinfo=timezone.utc)

    with requests.Session() as s:
        if not _has_any_candle_at_or_before(s, ticker=ticker, resolution=resolution, to_dt=search_to, timeout_s=timeout_s):
            return None

        if _has_any_candle_at_or_before(s, ticker=ticker, resolution=resolution, to_dt=search_from, timeout_s=timeout_s):
            # We only need a start >= search_from.
            return search_from.replace(second=0, microsecond=0)

        lo = search_from
        hi = search_to

        # Binary search in seconds; stop once window < 1 minute.
        while (hi - lo).total_seconds() > 60:
            mid = lo + (hi - lo) / 2
            if _has_any_candle_at_or_before(s, ticker=ticker, resolution=resolution, to_dt=mid, timeout_s=timeout_s):
                hi = mid
            else:
                lo = mid

        # Fetch the candle at the boundary.
        candles = _fetch_candles(
            s,
            ticker=ticker,
            resolution=resolution,
            from_iso=None,
            to_iso=_to_iso(hi),
            limit=1,
            timeout_s=timeout_s,
        )
        if not candles:
            return None

        started = _parse_iso(candles[0]["startedAt"]).replace(second=0, microsecond=0)

        # Sanity: ensure there's nothing strictly before this minute.
        probe = started - timedelta(seconds=1)
        if _has_any_candle_at_or_before(s, ticker=ticker, resolution=resolution, to_dt=probe, timeout_s=timeout_s):
            # If this ever triggers, it means the window wasn't tight enough.
            return started

        return started


def _window_pairs(from_dt: datetime, to_dt: datetime, *, max_points: int) -> list[tuple[datetime, datetime]]:
    """Create inclusive minute windows that return <= max_points candles.

    For 1MIN candles, a window with span (max_points-1) minutes contains at most max_points candles.
    """

    from_dt = from_dt.replace(second=0, microsecond=0)
    to_dt = to_dt.replace(second=0, microsecond=0)
    if to_dt < from_dt:
        return []

    win_minutes = max(1, int(max_points) - 1)
    cur = from_dt
    out: list[tuple[datetime, datetime]] = []
    while cur <= to_dt:
        end = min(to_dt, cur + timedelta(minutes=win_minutes))
        out.append((cur, end))
        cur = end + timedelta(minutes=1)
    return out


def download_candles_parallel(
    *,
    ticker: str,
    resolution: str,
    from_dt: datetime,
    to_dt: datetime,
    limit: int,
    workers: int,
    timeout_s: float,
) -> pd.DataFrame:
    """Download candles using independent time windows (parallelizable)."""

    windows = _window_pairs(from_dt, to_dt, max_points=limit)
    if not windows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    def fetch_one(w: tuple[datetime, datetime]) -> pd.DataFrame:
        w_from, w_to = w
        with requests.Session() as s:
            candles = _fetch_candles(
                s,
                ticker=ticker,
                resolution=resolution,
                from_iso=_to_iso(w_from),
                to_iso=_to_iso(w_to),
                limit=limit,
                timeout_s=timeout_s,
            )

        if not candles:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        rows: list[tuple[datetime, float, float, float, float, float]] = []
        for c in candles:
            ts = _parse_iso(c["startedAt"]).replace(second=0, microsecond=0)
            if ts < w_from or ts > w_to:
                continue
            rows.append(
                (
                    ts,
                    float(c["open"]),
                    float(c["high"]),
                    float(c["low"]),
                    float(c["close"]),
                    float(c.get("baseTokenVolume", 0.0)),
                )
            )

        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        return df

    dfs: list[pd.DataFrame] = []

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=len(windows), desc=f"Downloading {ticker} {resolution}", unit="req")

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = [ex.submit(fetch_one, w) for w in windows]
        try:
            for fut in as_completed(futs):
                df = fut.result()
                if not df.empty:
                    dfs.append(df)
                if pbar is not None:
                    pbar.update(1)
        finally:
            if pbar is not None:
                pbar.close()

    if not dfs:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    return out


def to_btcusdt_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Match the existing BTCUSDT market CSV schema (extra cols included).

    Existing columns:
      ['unix_timestamp', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
       'price_change_pct', 'perfect_trade_action', 'perfect_trade_profit_pct', 'date']
    """

    out = df.copy()
    out["unix_timestamp"] = (pd.to_datetime(out["timestamp"], utc=True).astype("int64") / 1e9).astype(float)

    # Match BTCUSDT file: timestamp string without timezone suffix.
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")

    out["price_change_pct"] = out["close"].pct_change() * 100.0
    out["perfect_trade_action"] = "hold"
    out["perfect_trade_profit_pct"] = np.nan
    out["date"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%d")

    cols = [
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
    return out[cols]


def main() -> None:
    ap = argparse.ArgumentParser(description="Download dYdX v4 perpetual candles to BTCUSDT-like CSV")
    ap.add_argument("--ticker", default="BTC-USD", help="Perpetual market ticker (e.g., BTC-USD)")
    ap.add_argument(
        "--resolution",
        default="1MIN",
        choices=["1MIN", "5MINS", "15MINS", "30MINS", "1HOUR", "4HOURS", "1DAY"],
    )
    ap.add_argument("--from-iso", default="2018-01-01T00:00:00Z", help="Lower bound (inclusive), ISO8601")
    ap.add_argument("--to-iso", default=None, help="Upper bound (inclusive), ISO8601 (default: now)")
    ap.add_argument("--limit", type=int, default=1000, help="Candles per request (max works at 1000)")
    ap.add_argument("--workers", type=int, default=2, help="Parallel workers for windowed downloads")
    ap.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout (seconds)")
    ap.add_argument("--out", default=None, help="Output CSV path")
    args = ap.parse_args()

    from_dt = _parse_iso(args.from_iso)
    to_dt = _parse_iso(args.to_iso) if args.to_iso else datetime.now(timezone.utc)

    ts = _now_ts()
    out_path = Path(args.out) if args.out else Path(f"data/dydx_{args.ticker}_{args.resolution}_{ts}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first_avail = find_first_available_minute(
        ticker=str(args.ticker),
        resolution=str(args.resolution),
        search_from=from_dt,
        search_to=to_dt,
        timeout_s=float(args.timeout),
    )
    if first_avail is None:
        raise SystemExit("No candles returned for requested range. (Market might not exist / no indexer history.)")

    effective_from = max(from_dt.replace(second=0, microsecond=0), first_avail)

    raw = download_candles_parallel(
        ticker=str(args.ticker),
        resolution=str(args.resolution),
        from_dt=effective_from,
        to_dt=to_dt,
        limit=int(args.limit),
        workers=int(args.workers),
        timeout_s=float(args.timeout),
    )

    if raw.empty:
        raise SystemExit("No candles returned for requested range after windowing.")

    out = to_btcusdt_schema(raw)
    out.to_csv(out_path, index=False)

    print("First available minute:", first_avail.strftime("%Y-%m-%d %H:%M:%S"))
    print("Wrote market CSV:", out_path)
    print("Rows:", len(out))
    print("First timestamp:", out["timestamp"].iloc[0])
    print("Last timestamp:", out["timestamp"].iloc[-1])


if __name__ == "__main__":
    main()
