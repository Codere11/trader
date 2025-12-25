#!/usr/bin/env python3
# Timestamp (UTC): 2025-12-23T18:05:06Z
"""Healthcheck helper.

Used by docker-compose healthchecks to verify the live runner is receiving closed 1m bars.

Exits 0 only if GET /health returns HTTP 200.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request


def main() -> None:
    ap = argparse.ArgumentParser(description="Healthcheck: GET /health and exit 0 on HTTP 200")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--path", type=str, default="/health")
    ap.add_argument("--timeout", type=float, default=2.0)
    args = ap.parse_args()

    url = f"http://{args.host}:{int(args.port)}{args.path}"
    try:
        with urllib.request.urlopen(url, timeout=float(args.timeout)) as r:
            code = getattr(r, "status", None) or r.getcode()
            if int(code) != 200:
                raise RuntimeError(f"HTTP {code}")
    except Exception:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
