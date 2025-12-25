#!/usr/bin/env python3
"""
Evaluate the current best strategy configuration and save outputs with a timestamp.
This is a thin wrapper around scripts/eval_rule_ranker_strategy.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

# Repo root and imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Import the evaluator as a module
from scripts import eval_rule_ranker_strategy as eval_mod  # type: ignore


def main() -> None:
    # Use a stable timestamp for file outputs (UTC, filename-safe)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    # Paths with timestamped outputs (rule compliance: include timestamp in new files)
    out_dir = REPO_ROOT / "data" / "entry_ranked_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_trades = out_dir / f"rule_ranker_trades_{ts}.csv"
    out_daily = out_dir / f"rule_ranker_daily_{ts}.csv"
    out_metrics = out_dir / f"rule_ranker_metrics_{ts}.csv"

    # Default "current best" hyperparameters inferred from past sweeps/metrics
    # - entry_threshold 0.52 (strong performance region in sweeps)
    # - k_per_day 3, hold_min 8, min_profit_pct 0.10, thr_exit 0.50 (house defaults)
    # - end-date pinned to last processed date to avoid leakage into holdout
    argv = [
        "eval_rule_ranker_strategy.py",
        "--entry-threshold", "0.52",
        "--k-per-day", "3",
        "--hold-min", "8",
        "--min-profit-pct", "0.10",
        "--thr-exit", "0.50",
        "--end-date", "2025-12-18",
        "--out-trades", str(out_trades),
        "--out-daily", str(out_daily),
        "--metrics-out", str(out_metrics),
    ]

    # Run evaluator with the constructed argv
    old_argv = sys.argv
    try:
        sys.argv = argv
        eval_mod.main()
    finally:
        sys.argv = old_argv

    print("\n[OK] Evaluation complete.")
    print("Trades:", out_trades)
    print("Daily:", out_daily)
    print("Metrics:", out_metrics)


if __name__ == "__main__":
    main()
