#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-03T23:24:23Z
"""Download all available dYdX v4 ETH-USD 1MIN candles and build a ready-to-use dataset.

Outputs (by default, under data/):
- dydx_ETH-USD_1MIN_<ts>.csv
  Columns: timestamp, open, high, low, close, volume
- dydx_ETH-USD_1MIN_features_<ts>.csv
  Columns: timestamp, open, high, low, close, volume, <feature cols>
- dydx_ETH-USD_1MIN_features_<ts>.parquet (if pyarrow/fastparquet installed)

Notes
- Uses the public dYdX indexer REST endpoint:
    /v4/candles/perpetualMarkets/{TICKER}
- Auto-detects earliest available candle via a binary search.
- Default end time is the last fully-closed minute.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent.parent


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


def _get(session: requests.Session, url: str, params: dict[str, Any], timeout_s: float, retries: int = 10) -> dict:
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
                raise RuntimeError(str(payload["errors"]))
            if not isinstance(payload, dict):
                raise RuntimeError(f"Unexpected JSON type: {type(payload)}")
            return payload
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(0.25 * (2**min(attempt, 6)))

    raise RuntimeError("unreachable")


def _fetch_candles(
    session: requests.Session,
    *,
    indexer_v4: str,
    ticker: str,
    resolution: str,
    from_iso: str | None,
    to_iso: str,
    limit: int,
    timeout_s: float,
) -> list[dict]:
    url = f"{indexer_v4.rstrip('/')}/candles/perpetualMarkets/{ticker}"
    params: dict[str, Any] = {
        "resolution": str(resolution),
        "limit": int(limit),
        "toISO": str(to_iso),
    }
    if from_iso is not None:
        params["fromISO"] = str(from_iso)

    data = _get(session, url, params, timeout_s)
    return list(data.get("candles", []))


def _has_any_candle_at_or_before(
    session: requests.Session,
    *,
    indexer_v4: str,
    ticker: str,
    resolution: str,
    to_dt: datetime,
    timeout_s: float,
) -> bool:
    candles = _fetch_candles(
        session,
        indexer_v4=indexer_v4,
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
    indexer_v4: str,
    ticker: str,
    resolution: str,
    search_from: datetime,
    search_to: datetime,
    timeout_s: float,
) -> datetime | None:
    """Find the first minute where candles exist (market history start).

    If there is no data at all up to search_to, returns None.

    Implementation: binary-search for the smallest toISO that returns any candle.
    """

    if search_from.tzinfo is None:
        search_from = search_from.replace(tzinfo=timezone.utc)
    if search_to.tzinfo is None:
        search_to = search_to.replace(tzinfo=timezone.utc)

    with requests.Session() as s:
        if not _has_any_candle_at_or_before(
            s, indexer_v4=indexer_v4, ticker=ticker, resolution=resolution, to_dt=search_to, timeout_s=timeout_s
        ):
            return None

        if _has_any_candle_at_or_before(
            s, indexer_v4=indexer_v4, ticker=ticker, resolution=resolution, to_dt=search_from, timeout_s=timeout_s
        ):
            return search_from.replace(second=0, microsecond=0)

        lo = search_from
        hi = search_to

        # Binary search in seconds; stop once window < 1 minute.
        while (hi - lo).total_seconds() > 60:
            mid = lo + (hi - lo) / 2
            if _has_any_candle_at_or_before(
                s, indexer_v4=indexer_v4, ticker=ticker, resolution=resolution, to_dt=mid, timeout_s=timeout_s
            ):
                hi = mid
            else:
                lo = mid

        candles = _fetch_candles(
            s,
            indexer_v4=indexer_v4,
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
        if _has_any_candle_at_or_before(
            s, indexer_v4=indexer_v4, ticker=ticker, resolution=resolution, to_dt=probe, timeout_s=timeout_s
        ):
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
    indexer_v4: str,
    ticker: str,
    resolution: str,
    from_dt: datetime,
    to_dt: datetime,
    limit: int,
    workers: int,
    timeout_s: float,
) -> pd.DataFrame:
    windows = _window_pairs(from_dt, to_dt, max_points=limit)
    if not windows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    def fetch_one(w: tuple[datetime, datetime]) -> pd.DataFrame:
        w_from, w_to = w
        with requests.Session() as s:
            candles = _fetch_candles(
                s,
                indexer_v4=indexer_v4,
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
                    float(c.get("baseTokenVolume", 0.0) or 0.0),
                )
            )

        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])

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
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    return out


def add_standard_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the repo's standard 1m candle features (causal; uses only current/past)."""

    # Local import to keep script usable even if pathing differs.
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))

    from binance_adapter.crossref_oracle_vs_agent_patterns import FEATURES, compute_feature_frame  # type: ignore

    bars = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars["ts_min"] = bars["timestamp"]

    feats = compute_feature_frame(bars)
    feats = feats.rename(columns={"ts_min": "timestamp"})

    # compute_feature_frame emits timestamp as dtype datetime64[ns] (tz-naive);
    # normalize to tz-aware UTC to match `bars` for a safe merge.
    feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)

    # Keep only expected feature list (stable order) + timestamp.
    keep = ["timestamp"] + [c for c in FEATURES if c in feats.columns]
    feats = feats[keep].copy()

    merged = bars.drop(columns=["ts_min"]).merge(feats, on="timestamp", how="left")
    return merged


def _gap_summary(ts: pd.Series) -> tuple[int, int]:
    """Return (missing_minutes_estimate, max_gap_minutes)."""
    if ts.empty:
        return (0, 0)
    t = pd.to_datetime(ts, utc=True).sort_values()
    d = t.diff().dropna()
    gaps = (d / pd.Timedelta(minutes=1)).astype(float)
    max_gap = int(gaps.max()) if len(gaps) else 1
    # missing minutes are gaps > 1
    missing = int(gaps[gaps > 1].sub(1).sum()) if len(gaps) else 0
    return (missing, max_gap)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download dYdX ETH-USD 1MIN candles and build dataset + features")
    ap.add_argument("--ticker", default="ETH-USD", help="Perpetual market ticker (e.g., ETH-USD)")
    ap.add_argument(
        "--resolution",
        default="1MIN",
        choices=["1MIN", "5MINS", "15MINS", "30MINS", "1HOUR", "4HOURS", "1DAY"],
    )

    default_indexer = os.getenv("DYDX_INDEXER_REST", "https://indexer.dydx.trade").rstrip("/")
    ap.add_argument("--indexer", default=default_indexer, help="dYdX indexer base URL (no /v4)")

    ap.add_argument("--from-iso", default="auto", help='Lower bound ISO8601 (inclusive) or "auto"')
    ap.add_argument(
        "--to-iso",
        default=None,
        help="Upper bound ISO8601 (inclusive). Default: last fully-closed minute.",
    )

    ap.add_argument("--limit", type=int, default=1000, help="Candles per request (dYdX max is 1000)")
    ap.add_argument("--workers", type=int, default=2, help="Parallel workers for windowed downloads")
    ap.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout (seconds)")

    ap.add_argument("--out-dir", default="data", help="Output directory")
    ap.add_argument("--tag", default=None, help="Optional tag to include in output filenames")

    args = ap.parse_args()

    indexer_v4 = f"{str(args.indexer).rstrip('/')}/v4"

    # Default end time = last fully closed minute.
    if args.to_iso:
        to_dt = _parse_iso(args.to_iso).replace(second=0, microsecond=0)
    else:
        to_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=1)

    # Auto-detect start (earliest available candle).
    if str(args.from_iso).strip().lower() == "auto":
        search_from = datetime(2018, 1, 1, tzinfo=timezone.utc)
        first = find_first_available_minute(
            indexer_v4=indexer_v4,
            ticker=str(args.ticker),
            resolution=str(args.resolution),
            search_from=search_from,
            search_to=to_dt,
            timeout_s=float(args.timeout),
        )
        if first is None:
            raise SystemExit("No candles returned. (Market might not exist / no indexer history.)")
        from_dt = first
    else:
        from_dt = _parse_iso(args.from_iso).replace(second=0, microsecond=0)

    if from_dt > to_dt:
        raise SystemExit(f"from_dt ({from_dt}) is after to_dt ({to_dt})")

    print("Index: ", indexer_v4)
    print("Market:", args.ticker, args.resolution)
    print("Range: ", _to_iso(from_dt), "→", _to_iso(to_dt))

    raw = download_candles_parallel(
        indexer_v4=indexer_v4,
        ticker=str(args.ticker),
        resolution=str(args.resolution),
        from_dt=from_dt,
        to_dt=to_dt,
        limit=int(args.limit),
        workers=int(args.workers),
        timeout_s=float(args.timeout),
    )

    if raw.empty:
        raise SystemExit("No candles returned for requested range.")

    missing, max_gap = _gap_summary(raw["timestamp"])
    print(f"Rows: {len(raw):,}")
    print(f"First ts: {raw['timestamp'].iloc[0]}  Last ts: {raw['timestamp'].iloc[-1]}")
    print(f"Gaps: missing_minutes_est={missing:,}  max_gap_min={max_gap}")

    # Build outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = _now_ts()
    tag = f"_{args.tag}" if args.tag else ""

    safe_ticker = str(args.ticker).replace("/", "-")

    out_raw_csv = out_dir / f"dydx_{safe_ticker}_{args.resolution}{tag}_{ts}.csv"
    out_feat_csv = out_dir / f"dydx_{safe_ticker}_{args.resolution}_features{tag}_{ts}.csv"
    out_feat_parq = out_dir / f"dydx_{safe_ticker}_{args.resolution}_features{tag}_{ts}.parquet"

    # Write raw
    raw_out = raw.copy()
    # Keep timestamp ISO with timezone offset for clarity.
    raw_out["timestamp"] = pd.to_datetime(raw_out["timestamp"], utc=True)
    raw_out.to_csv(out_raw_csv, index=False)

    # Add features
    feat = add_standard_features(raw_out)
    feat.to_csv(out_feat_csv, index=False)

    # Optional parquet
    try:
        feat.to_parquet(out_feat_parq, index=False)
        wrote_parquet = True
    except Exception as e:
        wrote_parquet = False
        print(f"NOTE: Parquet write skipped ({type(e).__name__}: {e}).")

    print("Wrote raw CSV:", out_raw_csv)
    print("Wrote features CSV:", out_feat_csv)
    if wrote_parquet:
        print("Wrote features Parquet:", out_feat_parq)


if __name__ == "__main__":
    main()
