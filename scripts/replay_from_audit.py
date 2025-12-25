#!/usr/bin/env python3
# Timestamp (UTC): 2025-12-23T18:06:31Z
"""Replay helper: reproduce a run using a saved audit log + persisted minute bars.

This script:
- reads a JSONL audit log
- infers a replay date range and (optionally) model artifact paths
- runs `live_paper_trade_2025-12-20T15-01-14Z.py --mode replay` using `--bars-dir` so it can run offline

It does NOT place real orders.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent


KIND_RUN_META = "run_meta"
KIND_TRADE_CLOSED = "trade_closed"


def _parse_dt(s: str) -> datetime:
    # Accept both "+00:00" and trailing "Z".
    s2 = s[:-1] + "+00:00" if s.endswith("Z") else s
    dt = datetime.fromisoformat(s2)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _scan_audit(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[datetime], Optional[datetime], int]:
    run_meta: Optional[Dict[str, Any]] = None
    min_t: Optional[datetime] = None
    max_t: Optional[datetime] = None
    n_trades = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue

            kind = obj.get("kind")
            if kind == KIND_RUN_META and run_meta is None:
                run_meta = obj if isinstance(obj, dict) else None

            if kind == KIND_TRADE_CLOSED and isinstance(obj, dict):
                n_trades += 1
                et = obj.get("entry_time_utc")
                if isinstance(et, str):
                    try:
                        dt = _parse_dt(et)
                    except Exception:
                        continue
                    if min_t is None or dt < min_t:
                        min_t = dt
                    if max_t is None or dt > max_t:
                        max_t = dt

    return run_meta, min_t, max_t, n_trades


def main() -> None:
    ap = argparse.ArgumentParser(description="Run replay using local minute_bars based on an audit log")
    ap.add_argument("--audit", required=True, help="Path to JSONL audit log")
    ap.add_argument("--bars-dir", default="", help="Directory containing minute_bars_YYYY-MM-DD.csv (default: audit dir)")
    ap.add_argument("--out-dir", default="data/replay_from_audit", help="Output directory for replay artifacts")

    ap.add_argument("--symbol", default="", help="Override symbol (default: inferred from run_meta)")
    ap.add_argument("--entry-model", default="", help="Override entry model artifact path")
    ap.add_argument("--exit-model", default="", help="Override exit model artifact path")
    ap.add_argument("--thresholds-csv", default="", help="Override thresholds CSV path")

    ap.add_argument("--strict", action="store_true", help="Fail if replay produces a different number of trades")

    args = ap.parse_args()

    audit_path = Path(args.audit)
    if not audit_path.exists():
        raise SystemExit(f"Audit file not found: {audit_path}")

    run_meta, min_t, max_t, n_trades = _scan_audit(audit_path)
    if min_t is None or max_t is None:
        raise SystemExit("Could not infer date range from audit (no trade_closed.entry_time_utc found)")

    start_date = min_t.date()
    end_date = (max_t.date() + timedelta(days=1))

    symbol = str(args.symbol).strip().upper()
    if not symbol and isinstance(run_meta, dict):
        symbol = str(run_meta.get("symbol", "")).strip().upper()
    if not symbol:
        symbol = "BTCUSDT"

    bars_dir = Path(args.bars_dir) if str(args.bars_dir).strip() else audit_path.parent

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer model artifacts from run_meta, but allow overrides.
    entry_model = str(args.entry_model).strip()
    exit_model = str(args.exit_model).strip()
    if isinstance(run_meta, dict) and (not entry_model or not exit_model):
        models = run_meta.get("models") if isinstance(run_meta.get("models"), dict) else {}
        if not entry_model and isinstance(models.get("entry"), dict):
            entry_model = str(models["entry"].get("artifact", "") or "").strip()
        if not exit_model and isinstance(models.get("exit"), dict):
            exit_model = str(models["exit"].get("artifact", "") or "").strip()

    thresholds_csv = str(args.thresholds_csv).strip()
    if not thresholds_csv:
        live_thr = bars_dir / "entry_score_thresholds_online_live.csv"
        if live_thr.exists():
            thresholds_csv = str(live_thr)

    audit_out = out_dir / f"audit_replay_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}.jsonl"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "live_paper_trade_2025-12-20T15-01-14Z.py"),
        "--mode",
        "replay",
        "--symbol",
        symbol,
        "--start",
        str(start_date),
        "--end",
        str(end_date),
        "--bars-dir",
        str(bars_dir),
        "--out-dir",
        str(out_dir),
        "--audit-log",
        str(audit_out),
    ]

    if entry_model:
        cmd += ["--entry-model", entry_model]
    if exit_model:
        cmd += ["--exit-model", exit_model]
    if thresholds_csv:
        cmd += ["--thresholds-csv", thresholds_csv]

    print("Running:")
    print(" ", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(res.returncode)

    # Optional sanity check: compare number of trades.
    if args.strict:
        # Find latest trades_*.csv in out_dir.
        trades = sorted(out_dir.glob("trades_*.csv"), key=lambda p: p.stat().st_mtime)
        if not trades:
            raise SystemExit("Strict mode: no trades_*.csv produced")
        import pandas as pd

        df = pd.read_csv(trades[-1])
        n_replay = int(len(df))
        if n_replay != int(n_trades):
            raise SystemExit(f"Strict mode: audit trades={n_trades} != replay trades={n_replay}")


if __name__ == "__main__":
    main()
