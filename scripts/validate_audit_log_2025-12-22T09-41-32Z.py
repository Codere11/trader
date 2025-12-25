#!/usr/bin/env python3
# Timestamp (UTC): 2025-12-22T09:41:32Z
"""Validate a JSONL audit log against contracts/v1.

Example:
  python3 scripts/validate_audit_log_2025-12-22T09-41-32Z.py \
    --audit data/live/audit_2025-12-22T09-45-00Z.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from contracts.v1 import ContractValidationError, validate_event_v1


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate JSONL audit log against contracts/v1")
    ap.add_argument("--audit", required=True, help="Path to JSONL audit log")
    ap.add_argument("--max-lines", type=int, default=0, help="Optional: stop after N lines (0 = no limit)")
    args = ap.parse_args()

    path = Path(args.audit)
    if not path.exists():
        raise SystemExit(f"Audit file not found: {path}")

    counts: Counter[str] = Counter()
    n = 0

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if args.max_lines and lineno > int(args.max_lines):
                break
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                raise SystemExit(f"Invalid JSON on line {lineno}: {e}") from e

            try:
                validate_event_v1(obj)
            except ContractValidationError as e:
                snippet = s[:400] + ("..." if len(s) > 400 else "")
                raise SystemExit(f"Contract validation failed on line {lineno}: {e}\nEvent: {snippet}") from e

            kind = str(obj.get("kind", "<missing>"))
            counts[kind] += 1
            n += 1

    print(f"OK: validated {n:,} events in {path}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
