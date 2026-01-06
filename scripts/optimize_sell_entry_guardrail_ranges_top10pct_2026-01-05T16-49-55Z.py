#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-05T16:49:55Z
"""Optimize simple volatility/volume guardrail *ranges* for SELL entries.

This script is intentionally non-ML: it searches one-feature (and small greedy multi-feature)
threshold ranges on entry precontext features to:
- keep a high fraction of already-selected trades (10% coverage trade list), while
- removing a disproportionate fraction of entry-doomed trades (oracle_best_ret_1x_pct <= 0)
  and worst losers.

It reuses the same precontext feature engineering as the SELL entry model.

Inputs
- A trades.csv produced by:
  scripts/analyze_sell_10pct_badtrades_entry_vs_exit_precontext_*.py

Outputs
- CSV of candidate single-feature ranges and metrics
- JSON with suggested multi-feature guardrail sets for several keep-rate targets
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


BASE_FEATS_DEFAULT = [
    "ret_1m_pct",
    "mom_3m_pct",
    "mom_5m_pct",
    "vol_std_5m",
    "range_5m",
    "range_norm_5m",
    "macd",
    "vwap_dev_5m",
]


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _rolling_prev_matrix(x: np.ndarray, L: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    mat = np.full((n, L), np.nan, dtype=np.float64)
    for k in range(1, L + 1):
        mat[k:, L - k] = x[: n - k]
    return mat


def _safe_nanstd(x: np.ndarray) -> np.ndarray:
    import warnings

    out = np.full(x.shape[0], np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m = np.nanmean(x, axis=1)
        ok = np.isfinite(m)
        if np.any(ok):
            out[ok] = np.nanmean((x[ok] - m[ok, None]) ** 2, axis=1) ** 0.5
    return out


def _slope_5(y: np.ndarray) -> np.ndarray:
    xc = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    return np.nansum(y * xc[None, :], axis=1) / 10.0


def _agg_5(x: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean5 = np.nanmean(x, axis=1)
        min5 = np.nanmin(x, axis=1)
        max5 = np.nanmax(x, axis=1)

    return {
        f"{prefix}__last": x[:, -1],
        f"{prefix}__mean5": mean5,
        f"{prefix}__std5": _safe_nanstd(x),
        f"{prefix}__min5": min5,
        f"{prefix}__max5": max5,
        f"{prefix}__range5": max5 - min5,
        f"{prefix}__slope5": _slope_5(x),
    }


def _rolling_sum_nan_prefix(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    cs = np.cumsum(x)
    out[w - 1 :] = cs[w - 1 :] - np.concatenate(([0.0], cs[: n - w]))
    return out


def _compute_ctx_series(df_bars: pd.DataFrame, windows: list[int]) -> dict[str, np.ndarray]:
    close_prev = pd.to_numeric(df_bars["close"], errors="coerce").shift(1)
    high_prev = pd.to_numeric(df_bars["high"], errors="coerce").shift(1)
    low_prev = pd.to_numeric(df_bars["low"], errors="coerce").shift(1)
    vol_prev = pd.to_numeric(df_bars["volume"], errors="coerce").shift(1)

    out: dict[str, np.ndarray] = {}

    for w in windows:
        w = int(w)
        out[f"mom_{w}m_pct"] = (close_prev.pct_change(w) * 100.0).to_numpy(np.float64)
        out[f"vol_std_{w}m"] = close_prev.rolling(w, min_periods=w).std(ddof=0).to_numpy(np.float64)

        rng = (high_prev.rolling(w, min_periods=w).max() - low_prev.rolling(w, min_periods=w).min()).to_numpy(np.float64)
        out[f"range_{w}m"] = rng
        out[f"range_norm_{w}m"] = rng / close_prev.replace(0, np.nan).to_numpy(np.float64)

        cmax = close_prev.rolling(w, min_periods=w).max().to_numpy(np.float64)
        cmin = close_prev.rolling(w, min_periods=w).min().to_numpy(np.float64)
        crng = cmax - cmin
        eps = 1e-9

        cp = close_prev.to_numpy(np.float64)
        out[f"close_dd_from_{w}m_max_pct"] = (cmax / np.maximum(1e-9, cp) - 1.0) * 100.0
        out[f"close_bounce_from_{w}m_min_pct"] = (cp / np.maximum(1e-9, cmin) - 1.0) * 100.0
        out[f"close_pos_in_{w}m_range"] = (cp - cmin) / (crng + eps)

        v = vol_prev.to_numpy(np.float64, copy=False)
        sum_v = _rolling_sum_nan_prefix(np.nan_to_num(v, nan=0.0), w)
        sum_pv = _rolling_sum_nan_prefix(np.nan_to_num(v, nan=0.0) * np.nan_to_num(cp, nan=0.0), w)
        vwap = sum_pv / np.maximum(1e-9, sum_v)
        out[f"vwap_dev_{w}m"] = np.where(sum_v > 0.0, ((cp - vwap) / np.maximum(1e-9, vwap)) * 100.0, 0.0)

    return out


def build_precontext_features(
    df: pd.DataFrame,
    *,
    L: int,
    base_feat_cols: list[str],
    ctx_series: dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, list[str]]:
    """Build the same precontext features used by SELL entry."""
    close = df["close"].to_numpy(np.float64, copy=False)
    vol = df["volume"].to_numpy(np.float64, copy=False)

    close_prev = _rolling_prev_matrix(close, L)
    close_last = close_prev[:, -1]
    close_last = np.where((np.isfinite(close_last)) & (close_last > 0.0), close_last, np.nan)
    close_norm = (close_prev / close_last[:, None] - 1.0) * 100.0

    out: dict[str, np.ndarray] = {}

    out.update(_agg_5(close_norm, "px_close_norm_pct"))
    out["px_close_norm_pct__ret5m"] = close_norm[:, -1] - close_norm[:, 0]
    out["px_close_norm_pct__absret5m"] = np.abs(out["px_close_norm_pct__ret5m"])
    out["px_close_norm_pct__m5"] = close_norm[:, 0]
    out["px_close_norm_pct__m4"] = close_norm[:, 1]
    out["px_close_norm_pct__m3"] = close_norm[:, 2]
    out["px_close_norm_pct__m2"] = close_norm[:, 3]
    out["px_close_norm_pct__m1"] = close_norm[:, 4]

    vol_prev = _rolling_prev_matrix(vol, L)
    vol_log = np.log1p(np.maximum(0.0, vol_prev))
    out.update(_agg_5(vol_log, "vol_log1p"))

    for c in base_feat_cols:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64, copy=False)
        x_prev = _rolling_prev_matrix(x, L)
        out.update(_agg_5(x_prev, c))

    for name, arr in ctx_series.items():
        arr_prev = _rolling_prev_matrix(arr, L)
        out.update(_agg_5(arr_prev, name))

    out["missing_close_n"] = np.sum(~np.isfinite(close_prev), axis=1).astype(np.float64)
    mats = [close_prev]
    for c in base_feat_cols:
        if c in df.columns:
            mats.append(_rolling_prev_matrix(pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64, copy=False), L))
    all_vals = np.column_stack(mats)
    out["missing_any"] = np.any(~np.isfinite(all_vals), axis=1).astype(np.float64)

    df_out = pd.DataFrame(out)
    return df_out, list(df_out.columns)


@dataclass(frozen=True)
class Rule:
    feature: str
    lo: Optional[float]
    hi: Optional[float]
    kind: str  # "min" | "max" | "band"
    q_lo: Optional[float] = None
    q_hi: Optional[float] = None


def _finite_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, float(q)))


def rule_mask(x: np.ndarray, rule: Rule) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.isfinite(x)
    if rule.lo is not None and np.isfinite(rule.lo):
        m &= x >= float(rule.lo)
    if rule.hi is not None and np.isfinite(rule.hi):
        m &= x <= float(rule.hi)
    return m


def eval_mask(
    mask: np.ndarray,
    *,
    bad_entry: np.ndarray,
    worst_bad: np.ndarray,
) -> Dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.size)
    keep = int(mask.sum())
    bad_total = int(bad_entry.sum())
    good_total = int((~bad_entry).sum())
    worst_total = int(worst_bad.sum())

    bad_kept = int((bad_entry & mask).sum())
    good_kept = int(((~bad_entry) & mask).sum())
    worst_kept = int((worst_bad & mask).sum())

    out = {
        "n": float(n),
        "keep": float(keep),
        "keep_rate": float(keep) / float(n) if n else float("nan"),
        "bad_entry_total": float(bad_total),
        "bad_entry_kept": float(bad_kept),
        "bad_entry_rate_kept": float(bad_kept) / float(keep) if keep else float("nan"),
        "bad_entry_removed_frac": 1.0 - (float(bad_kept) / float(bad_total) if bad_total else 0.0),
        "good_entry_kept_frac": float(good_kept) / float(good_total) if good_total else float("nan"),
        "worst_total": float(worst_total),
        "worst_kept": float(worst_kept),
        "worst_removed_frac": 1.0 - (float(worst_kept) / float(worst_total) if worst_total else 0.0),
    }
    return out


def greedy_select(
    *,
    masks_by_rule: List[np.ndarray],
    rules: List[Rule],
    keep_target: float,
    bad_entry: np.ndarray,
    worst_bad: np.ndarray,
    max_rules: int,
    beta_good: float,
) -> Tuple[List[int], Dict[str, float]]:
    """A conservative greedy selector (kept for reference)."""
    n = int(bad_entry.size)
    cur = np.ones(n, dtype=bool)
    chosen: List[int] = []

    base = eval_mask(cur, bad_entry=bad_entry, worst_bad=worst_bad)

    for _step in range(int(max_rules)):
        best_i: Optional[int] = None
        best_score: float = -1e18
        best_next: Optional[np.ndarray] = None
        best_metrics: Optional[Dict[str, float]] = None

        cur_bad = int((bad_entry & cur).sum())
        cur_good = int(((~bad_entry) & cur).sum())
        if cur_bad + cur_good == 0:
            break

        for i, m_rule in enumerate(masks_by_rule):
            if i in chosen:
                continue
            nxt = cur & m_rule
            keep_rate = float(nxt.mean()) if nxt.size else 0.0
            if keep_rate + 1e-12 < float(keep_target):
                continue

            nxt_bad = int((bad_entry & nxt).sum())
            nxt_good = int(((~bad_entry) & nxt).sum())

            # fractions removed within current kept-set
            bad_removed_frac = (float(cur_bad - nxt_bad) / float(cur_bad)) if cur_bad else 0.0
            good_removed_frac = (float(cur_good - nxt_good) / float(cur_good)) if cur_good else 0.0

            # objective: remove bad much faster than good
            score = bad_removed_frac - float(beta_good) * good_removed_frac

            # tie-breakers: prefer higher keep_rate, and higher worst removal
            metrics = eval_mask(nxt, bad_entry=bad_entry, worst_bad=worst_bad)
            score += 0.01 * float(metrics.get("keep_rate", 0.0))
            score += 0.02 * float(metrics.get("worst_removed_frac", 0.0))

            if score > best_score:
                best_score = float(score)
                best_i = int(i)
                best_next = nxt
                best_metrics = metrics

        if best_i is None or best_next is None or best_metrics is None:
            break

        chosen.append(int(best_i))
        cur = best_next

        # stop if no meaningful improvement
        if float(base.get("bad_entry_rate_kept", 1.0)) - float(best_metrics.get("bad_entry_rate_kept", 1.0)) < 1e-4:
            base = best_metrics
            break

        base = best_metrics

    return chosen, base


def bruteforce_best_combo(
    *,
    candidate_idxs: List[int],
    masks_by_rule: List[np.ndarray],
    rules: List[Rule],
    keep_target: float,
    bad_entry: np.ndarray,
    worst_bad: np.ndarray,
    max_rules: int,
    w_bad_removed: float,
    w_worst_removed: float,
    w_keep: float,
) -> Tuple[List[int], Dict[str, float]]:
    """Brute-force small combos over a pre-filtered candidate set.

    Objective (higher is better):
      w_bad_removed * bad_entry_removed_frac
    + w_worst_removed * worst_removed_frac
    + w_keep * (keep_rate - keep_target)

    We enforce keep_rate >= keep_target.
    """
    n = int(bad_entry.size)
    best_combo: List[int] = []
    best_metrics: Dict[str, float] = eval_mask(np.ones(n, dtype=bool), bad_entry=bad_entry, worst_bad=worst_bad)
    best_obj = -1e18

    for k in range(1, int(max_rules) + 1):
        for combo in combinations(candidate_idxs, k):
            # avoid multiple rules on the same feature (usually redundant/conflicting)
            feats = [rules[i].feature for i in combo]
            if len(set(feats)) != len(feats):
                continue

            m = np.ones(n, dtype=bool)
            for i in combo:
                m &= masks_by_rule[i]
            keep_rate = float(m.mean()) if m.size else 0.0
            if keep_rate + 1e-12 < float(keep_target):
                continue

            met = eval_mask(m, bad_entry=bad_entry, worst_bad=worst_bad)
            obj = (
                float(w_bad_removed) * float(met.get("bad_entry_removed_frac", 0.0))
                + float(w_worst_removed) * float(met.get("worst_removed_frac", 0.0))
                + float(w_keep) * (float(met.get("keep_rate", 0.0)) - float(keep_target))
            )

            # tie-breaker: prefer higher keep if objectives are extremely close
            if obj > best_obj + 1e-12 or (abs(obj - best_obj) <= 1e-12 and keep_rate > float(best_metrics.get("keep_rate", 0.0))):
                best_obj = float(obj)
                best_combo = list(combo)
                best_metrics = met

    return best_combo, best_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Optimize SELL entry guardrail ranges using 10% trade list")

    ap.add_argument(
        "--bars",
        default="data/dydx_ETH-USD_1MIN_features_full_2026-01-03T23-48-53Z.parquet",
    )
    ap.add_argument(
        "--entry-model",
        default="data/entry_regressor_oracle15m_sell_ctx120_weighted_oracleentry5m_2026-01-04T22-45-20Z/models/entry_regressor_sell_oracle15m_ctx120_weighted_oracleentry5m_2026-01-04T22-45-20Z.joblib",
    )
    ap.add_argument(
        "--trades-csv",
        default="data/backtests/analysis_sell_badtrades_top10pct_2026-01-05T16-31-07Z/trades.csv",
        help="Trades list from the 10% analysis script.",
    )
    ap.add_argument(
        "--analysis-dir",
        default="data/backtests/analysis_sell_badtrades_top10pct_2026-01-05T16-31-07Z",
        help="Directory containing entry_precontext_*.csv from the 10% analysis script.",
    )

    ap.add_argument("--keep-targets", default="0.90,0.80,0.70")
    ap.add_argument("--max-rules", type=int, default=4)
    ap.add_argument("--beta-good", type=float, default=2.0, help="Penalty multiplier for removing good entries in greedy search")

    ap.add_argument("--q-low", default="0.05,0.10,0.15,0.20,0.25,0.30")
    ap.add_argument("--q-high", default="0.95,0.90,0.85,0.80,0.75,0.70")

    ap.add_argument("--topk", type=int, default=30, help="How many top separators from each CSV to consider")
    ap.add_argument(
        "--feature-substr",
        default="range_norm_5m,range_5m,vol_std_5m,ret_1m_pct,px_close_norm_pct,vol_log1p,vwap_dev_5m",
        help="Comma-separated substrings to keep as candidate guardrail features.",
    )

    ap.add_argument("--slice-warmup-mins", type=int, default=2000, help="How much history to include before earliest trade for feature calc")

    ap.add_argument("--out-dir", default="data/backtests")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()
    ts = now_ts()

    keep_targets = [float(x) for x in str(args.keep_targets).split(",") if str(x).strip()]
    q_low = [float(x) for x in str(args.q_low).split(",") if str(x).strip()]
    q_high = [float(x) for x in str(args.q_high).split(",") if str(x).strip()]
    substr_keep = [s.strip() for s in str(args.feature_substr).split(",") if s.strip()]

    trades = pd.read_csv(Path(args.trades_csv))
    if trades.empty:
        raise SystemExit("Empty trades CSV")

    # Labels
    oracle_best = pd.to_numeric(trades["oracle_best_ret_1x_pct"], errors="coerce").to_numpy(np.float64)
    realized = pd.to_numeric(trades["realized_ret_1x_pct"], errors="coerce").to_numpy(np.float64)

    bad_entry = np.isfinite(oracle_best) & (oracle_best <= 0.0)  # entry-doomed

    bad_realized = np.isfinite(realized) & (realized <= 0.0)
    bad_vals = realized[bad_realized]
    thr_worst = float(np.quantile(bad_vals, 0.30)) if bad_vals.size else float("nan")
    worst_bad = bad_realized & (realized <= thr_worst)

    if args.verbose:
        print(f"n_trades: {len(trades)}")
        print(f"bad_entry_rate (oracle_best<=0): {bad_entry.mean():.4f}")
        print(f"bad_realized_rate (realized<=0): {bad_realized.mean():.4f}")
        print(f"worst_thr_realized_1x_pct (30% of bad): {thr_worst:.6f}")

    # Map trade entry timestamps to bar indices.
    # IMPORTANT: the dataset can have missing minutes; do NOT assume entry_time_open - 1 minute exists.
    # The analysis script opens at the *next row* (entry_idx = decision_i + 1), so we recover
    # decision_i as (entry_idx_global - 1) by index adjacency, not by wall-clock minute.
    bars = pd.read_parquet(Path(args.bars))
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    bars = bars.sort_values("timestamp").reset_index(drop=True)

    ts_bar = pd.to_datetime(bars["timestamp"], utc=True)
    # Build a fast lookup (timestamp -> index). Assumes unique bar timestamps.
    idx_by_ts = pd.Series(np.arange(len(bars), dtype=np.int64), index=ts_bar)

    entry_time_open = pd.to_datetime(trades["entry_time_open_utc"], utc=True, errors="coerce")
    entry_idx_f = idx_by_ts.reindex(entry_time_open).to_numpy(np.float64)

    if not np.isfinite(entry_idx_f).all():
        miss = int(np.sum(~np.isfinite(entry_idx_f)))
        bad_ts = entry_time_open[~np.isfinite(entry_idx_f)].iloc[:5].tolist()
        raise SystemExit(f"Failed to map {miss} entry_time_open timestamps to bars; examples={bad_ts}")

    entry_idx = entry_idx_f.astype(np.int64)
    if np.any(entry_idx <= 0):
        raise SystemExit("Some entry indices mapped to 0; cannot derive decision_idx = entry_idx - 1")

    decision_idx = entry_idx - 1

    min_i = int(np.min(decision_idx))
    max_i = int(np.max(decision_idx))

    start_i = max(0, min_i - int(args.slice_warmup_mins))
    end_i_excl = min(len(bars), max_i + 5)

    sub = bars.iloc[start_i:end_i_excl].copy()
    sub = sub.reset_index(drop=False).rename(columns={"index": "orig_i"})

    # Build precontext features on this slice
    ctx_series = _compute_ctx_series(sub, [30, 60, 120])
    base_cols = [c for c in BASE_FEATS_DEFAULT if c in sub.columns]
    pre_df, pre_cols = build_precontext_features(sub, L=5, base_feat_cols=base_cols, ctx_series=ctx_series)

    entry_art = joblib.load(Path(args.entry_model))
    entry_features = [str(c) for c in list(entry_art.get("feature_cols") or [])]

    # Extract trade feature rows
    rel_idx = decision_idx - int(start_i)
    if np.any(rel_idx < 0) or np.any(rel_idx >= len(sub)):
        raise SystemExit("Decision indices fell outside computed slice")

    X = pre_df.iloc[rel_idx][entry_features].copy()
    X.insert(0, "_trade_i", np.arange(len(trades), dtype=np.int64))

    # Candidate feature selection from separation CSVs
    analysis_dir = Path(args.analysis_dir)
    p_bad_good = analysis_dir / "entry_precontext_bad_vs_good.csv"
    p_worst_rest = analysis_dir / "entry_precontext_worst_vs_restbad.csv"

    sep_bad_good = pd.read_csv(p_bad_good) if p_bad_good.exists() else pd.DataFrame()
    sep_worst_rest = pd.read_csv(p_worst_rest) if p_worst_rest.exists() else pd.DataFrame()

    def _top(df: pd.DataFrame, k: int) -> pd.DataFrame:
        if df.empty:
            return df
        if "abs_robust_z" not in df.columns:
            return df
        return df.sort_values("abs_robust_z", ascending=False).head(int(k)).reset_index(drop=True)

    top_bg = _top(sep_bad_good, int(args.topk))
    top_wr = _top(sep_worst_rest, int(args.topk))

    # Decide whether each feature should get a min-cut, max-cut, or band-pass
    z_bg = {str(r.feature): float(r.robust_z) for r in top_bg.itertuples(index=False)} if not top_bg.empty else {}
    z_wr = {str(r.feature): float(r.robust_z) for r in top_wr.itertuples(index=False)} if not top_wr.empty else {}

    candidates: List[Tuple[str, bool, bool]] = []
    for f in entry_features:
        if not any(s in f for s in substr_keep):
            continue
        # use top separators as hint
        want_min = (f in z_bg) and (z_bg[f] < 0.0)
        want_max = (f in z_wr) and (z_wr[f] > 0.0)
        if want_min or want_max:
            candidates.append((f, bool(want_min), bool(want_max)))

    # If CSVs are missing or too restrictive, fall back to a small default set
    if not candidates:
        fallback = [
            "range_norm_5m__mean5",
            "range_norm_5m__max5",
            "ret_1m_pct__std5",
            "px_close_norm_pct__std5",
            "vol_log1p__last",
            "vwap_dev_5m__std5",
        ]
        for f in fallback:
            if f in entry_features and f in X.columns:
                candidates.append((f, True, True))

    if args.verbose:
        print(f"candidate features: {len(candidates)}")
        for f, mi, ma in candidates[:30]:
            print(f"  {f}  min={mi} max={ma}")

    # Build candidate rules
    rules: List[Rule] = []

    for f, want_min, want_max in candidates:
        if f not in X.columns:
            continue
        xv = pd.to_numeric(X[f], errors="coerce").to_numpy(np.float64)
        if not np.isfinite(xv).any():
            continue

        if want_min:
            for q in q_low:
                lo = _finite_quantile(xv, q)
                if np.isfinite(lo):
                    rules.append(Rule(feature=f, lo=float(lo), hi=None, kind="min", q_lo=float(q)))

        if want_max:
            for q in q_high:
                hi = _finite_quantile(xv, q)
                if np.isfinite(hi):
                    rules.append(Rule(feature=f, lo=None, hi=float(hi), kind="max", q_hi=float(q)))

        if want_min and want_max:
            for ql in q_low:
                for qh in q_high:
                    if float(ql) >= float(qh):
                        continue
                    lo = _finite_quantile(xv, ql)
                    hi = _finite_quantile(xv, qh)
                    if np.isfinite(lo) and np.isfinite(hi) and float(lo) < float(hi):
                        rules.append(Rule(feature=f, lo=float(lo), hi=float(hi), kind="band", q_lo=float(ql), q_hi=float(qh)))

    if not rules:
        raise SystemExit("No rules generated")

    # Precompute per-rule masks
    masks_by_rule: List[np.ndarray] = []
    for r in rules:
        xv = pd.to_numeric(X[r.feature], errors="coerce").to_numpy(np.float64)
        masks_by_rule.append(rule_mask(xv, r))

    # Evaluate all single rules
    base_metrics = eval_mask(np.ones(len(trades), dtype=bool), bad_entry=bad_entry, worst_bad=worst_bad)

    rows = []
    for r, m in zip(rules, masks_by_rule):
        met = eval_mask(m, bad_entry=bad_entry, worst_bad=worst_bad)
        rows.append(
            {
                "feature": r.feature,
                "kind": r.kind,
                "lo": r.lo,
                "hi": r.hi,
                "q_lo": r.q_lo,
                "q_hi": r.q_hi,
                **met,
            }
        )

    rules_df = pd.DataFrame(rows)

    # For each keep target, pick best single-rule candidates
    best_single: Dict[str, List[Dict[str, Any]]] = {}
    for kt in keep_targets:
        sub_df = rules_df[rules_df["keep_rate"] >= float(kt) - 1e-12].copy()
        if sub_df.empty:
            best_single[str(kt)] = []
            continue
        sub_df["score"] = (
            (base_metrics["bad_entry_rate_kept"] - sub_df["bad_entry_rate_kept"])  # reduce bad rate
            + 0.5 * sub_df["worst_removed_frac"]
            + 0.05 * sub_df["keep_rate"]
        )
        best = sub_df.sort_values("score", ascending=False).head(20)
        best_single[str(kt)] = best[["feature", "kind", "lo", "hi", "q_lo", "q_hi", "keep_rate", "bad_entry_rate_kept", "bad_entry_removed_frac", "good_entry_kept_frac", "worst_removed_frac"]].to_dict("records")

    # Multi-rule selection: brute-force small combos over a narrowed candidate pool
    best_multi: Dict[str, Dict[str, Any]] = {}
    for kt in keep_targets:
        kt = float(kt)
        sub_df = rules_df[rules_df["keep_rate"] >= kt - 1e-12].copy()
        if sub_df.empty:
            best_multi[str(kt)] = {"keep_target": kt, "chosen_rules": [], "metrics": {}}
            continue

        # Score single rules to preselect a small pool for combo search
        sub_df["pre_score"] = (
            (base_metrics["bad_entry_rate_kept"] - sub_df["bad_entry_rate_kept"])  # reduce bad rate
            + 0.8 * sub_df["worst_removed_frac"]
            + 0.05 * (sub_df["keep_rate"] - kt)
        )
        # Keep a modest pool to make brute force tractable
        pool = sub_df.sort_values("pre_score", ascending=False).head(30)
        cand_idxs = pool.index.to_list()  # these are row indices into rules_df

        chosen, met = bruteforce_best_combo(
            candidate_idxs=[int(i) for i in cand_idxs],
            masks_by_rule=masks_by_rule,
            rules=rules,
            keep_target=kt,
            bad_entry=bad_entry,
            worst_bad=worst_bad,
            max_rules=int(args.max_rules),
            w_bad_removed=1.0,
            w_worst_removed=0.8,
            w_keep=0.1,
        )

        best_multi[str(kt)] = {
            "keep_target": kt,
            "candidate_pool_size": int(len(cand_idxs)),
            "chosen_rules": [
                {
                    "feature": rules[i].feature,
                    "kind": rules[i].kind,
                    "lo": rules[i].lo,
                    "hi": rules[i].hi,
                    "q_lo": rules[i].q_lo,
                    "q_hi": rules[i].q_hi,
                }
                for i in chosen
            ],
            "metrics": met,
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"guardrail_range_search_top10pct_{ts}.csv"
    rules_df.to_csv(out_csv, index=False)

    out_json = out_dir / f"guardrail_range_suggestions_top10pct_{ts}.json"
    payload = {
        "created_utc": ts,
        "source_trades_csv": str(Path(args.trades_csv)),
        "source_analysis_dir": str(Path(args.analysis_dir)),
        "n_trades": int(len(trades)),
        "baseline": base_metrics,
        "worst_thr_realized_1x_pct": float(thr_worst),
        "candidate_feature_count": int(len(candidates)),
        "rule_count": int(len(rules)),
        "best_single_by_keep_target": best_single,
        "best_multi_by_keep_target": best_multi,
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Wrote:")
    print(" ", out_csv)
    print(" ", out_json)

    # Console summary
    print("Baseline:")
    print(f"  keep_rate: {base_metrics['keep_rate']:.4f}")
    print(f"  bad_entry_rate_kept: {base_metrics['bad_entry_rate_kept']:.4f}")
    print(f"  worst_removed_frac: {base_metrics['worst_removed_frac']:.4f}")

    for kt in keep_targets:
        info = best_multi.get(str(kt), {})
        if not info:
            continue
        met = info.get("metrics", {})
        print(f"\nBest greedy multi @ keep_target={kt:.2f}:")
        print(f"  keep_rate: {float(met.get('keep_rate', float('nan'))):.4f}")
        print(f"  bad_entry_rate_kept: {float(met.get('bad_entry_rate_kept', float('nan'))):.4f}")
        print(f"  bad_entry_removed_frac: {float(met.get('bad_entry_removed_frac', float('nan'))):.4f}")
        print(f"  good_entry_kept_frac: {float(met.get('good_entry_kept_frac', float('nan'))):.4f}")
        print(f"  worst_removed_frac: {float(met.get('worst_removed_frac', float('nan'))):.4f}")
        for r in info.get("chosen_rules", []):
            lo = r.get("lo")
            hi = r.get("hi")
            if lo is not None and hi is not None:
                print(f"   - {r['feature']} in [{lo:.6g}, {hi:.6g}] (q_lo={r.get('q_lo')}, q_hi={r.get('q_hi')})")
            elif lo is not None:
                print(f"   - {r['feature']} >= {lo:.6g} (q_lo={r.get('q_lo')})")
            elif hi is not None:
                print(f"   - {r['feature']} <= {hi:.6g} (q_hi={r.get('q_hi')})")


if __name__ == "__main__":
    main()
