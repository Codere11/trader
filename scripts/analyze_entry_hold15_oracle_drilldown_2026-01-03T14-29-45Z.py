#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-03T14:29:45Z
"""Drill-down: last-N trades taken by the live-style entry policy vs oracle exits within hold_min.

What this does
- Loads a 1m OHLCV CSV
- Builds pattern v2 features (same as live runner)
- Scores every minute with pattern_entry_regressor_v2
- Simulates the live runner entry policy:
  - decision at minute close i
  - enter at next minute open i+1
  - per-day causal threshold based on prior days' scores (target_frac)
  - fixed-hold non-overlap blocking (hold_min)
- For each trade, computes:
  - hold-15 net return (exit at close after hold_min)
  - oracle best net return within 1..hold_min (exit at close)
  - oracle_k and oracle_gap = oracle_ret - hold_ret
  - path stats (MFE/MAE using high/low)
- Selects worst/best cases from the last-N trades and saves plots.

Outputs
- out_dir/report.json (includes generated_utc and summaries)
- out_dir/last_trades.csv (trade rows; includes generated_utc column)
- out_dir/plots/*.png

Notes
- CSV timestamps are treated as UTC-aligned minutes (stored as naive in the file).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import FEATURES, compute_feature_frame


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _slope5(series: pd.Series) -> pd.Series:
    return ((-2.0 * series.shift(4)) + (-1.0 * series.shift(3)) + (1.0 * series.shift(1)) + (2.0 * series)) / 10.0


def _accel5(series: pd.Series) -> pd.Series:
    return (series - 2.0 * series.shift(2) + series.shift(4)) / 2.0


def _zscore_roll(s: pd.Series, win: int) -> pd.Series:
    mu = s.rolling(int(win), min_periods=int(win)).mean()
    sd = s.rolling(int(win), min_periods=int(win)).std()
    return (s - mu) / (sd + 1e-12)


def build_entry_pattern_frame_v2(bars: pd.DataFrame, base_features: List[str]) -> pd.DataFrame:
    """Must match the live runner's build_entry_pattern_frame_v2."""
    bars = bars.copy().sort_values("timestamp").reset_index(drop=True)
    bars["ts_min"] = pd.to_datetime(bars["timestamp"]).dt.floor("min")

    base = compute_feature_frame(bars)
    src = base[[c for c in FEATURES if c in base.columns]]

    missing = [f for f in base_features if f not in src.columns]
    if missing:
        raise ValueError(f"Missing required base features: {missing}")

    df = pd.DataFrame({"timestamp": pd.to_datetime(bars["timestamp"], utc=True)})

    # 5-minute pattern descriptors
    for f in base_features:
        s = pd.to_numeric(src[f], errors="coerce")
        df[f"{f}__last"] = s
        df[f"{f}__slope5"] = _slope5(s)
        df[f"{f}__accel5"] = _accel5(s)
        df[f"{f}__std5"] = s.rolling(5, min_periods=5).std()
        df[f"{f}__min5"] = s.rolling(5, min_periods=5).min()
        df[f"{f}__max5"] = s.rolling(5, min_periods=5).max()
        df[f"{f}__range5"] = df[f"{f}__max5"] - df[f"{f}__min5"]

    # Cross-feature correlations
    pairs = [
        ("macd", "ret_1m_pct"),
        ("vol_std_5m", "ret_1m_pct"),
        ("range_norm_5m", "ret_1m_pct"),
    ]
    for a, b in pairs:
        if a in base_features and b in base_features:
            df[f"corr5__{a}__{b}"] = pd.to_numeric(src[a], errors="coerce").rolling(5, min_periods=5).corr(
                pd.to_numeric(src[b], errors="coerce")
            )

    # Price context
    close = pd.to_numeric(bars["close"], errors="coerce")
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")

    df["px__ret1m_close"] = close.pct_change() * 100.0
    df["px__ret1m_abs"] = df["px__ret1m_close"].abs()
    df["px__range_norm1m"] = (high - low) / (close + 1e-12)
    df["px__range_norm1m_abs"] = df["px__range_norm1m"].abs()

    # New indicators: spike/pump/vol regime
    win = 1440
    df["z1d__px_ret1m"] = _zscore_roll(df["px__ret1m_close"], win)
    df["z1d__px_ret1m_abs"] = df["z1d__px_ret1m"].abs()
    df["z1d__px_range1m"] = _zscore_roll(df["px__range_norm1m"], win)
    df["z1d__px_range1m_abs"] = df["z1d__px_range1m"].abs()

    # Context baselines
    vol5 = df.get("vol_std_5m__last")
    rng5max = df.get("range_norm_5m__max5")
    if vol5 is not None:
        v = pd.to_numeric(vol5, errors="coerce")
        df["z1d__vol5"] = _zscore_roll(v, win)
        df["risk__ret1m_abs_over_vol5"] = df["px__ret1m_abs"] / (v.abs() + 1e-9)
        df["risk__range1m_over_vol5"] = df["px__range_norm1m"] / (v.abs() + 1e-9)

    if rng5max is not None:
        rm = pd.to_numeric(rng5max, errors="coerce")
        df["risk__range1m_over_range5max"] = df["px__range_norm1m"] / (rm.abs() + 1e-12)

    # Directional decomposition
    if "ret_1m_pct__last" in df.columns:
        df["ret_1m_pct__last_pos"] = df["ret_1m_pct__last"].clip(lower=0.0)
        df["ret_1m_pct__last_neg"] = (-df["ret_1m_pct__last"]).clip(lower=0.0)

    # Simple extreme flags
    df["flag__ret1m_abs_z_gt3"] = (df["z1d__px_ret1m_abs"] > 3.0).astype(np.float32)
    df["flag__range1m_abs_z_gt3"] = (df["z1d__px_range1m_abs"] > 3.0).astype(np.float32)

    return df


def net_return_pct(entry_px: float, exit_px: float, fee_side: float) -> float:
    gross_mult = float(exit_px) / max(1e-12, float(entry_px))
    net_mult = gross_mult * (1.0 - float(fee_side)) / (1.0 + float(fee_side))
    return float((net_mult - 1.0) * 100.0)


@dataclass(frozen=True)
class TradeRow:
    signal_i: int
    signal_time: str
    day: str
    score: float
    threshold: float
    entry_i: int
    entry_time: str
    entry_open: float
    hold_exit_i: int
    hold_exit_time: str
    hold_exit_close: float
    hold_ret_1x_pct_net: float
    oracle_k: int
    oracle_exit_i: int
    oracle_exit_time: str
    oracle_exit_close: float
    oracle_ret_1x_pct_net: float
    oracle_gap_pct_points: float
    mfe_gross_pct_hi: float
    mae_gross_pct_lo: float


def simulate_trades(
    *,
    bars: pd.DataFrame,
    scores: np.ndarray,
    target_frac: float,
    hold_min: int,
    fee_side: float,
    max_prior_scores: int,
) -> List[TradeRow]:
    ts = pd.to_datetime(bars["timestamp"])  # naive in CSV
    dates = ts.dt.date.to_numpy()

    prior_scores: List[float] = []
    cur_day_scores: List[float] = []
    blocked_until_signal_i = -10**18

    trades: List[TradeRow] = []

    cur_day = None
    thr = float("inf")

    for i in range(len(bars)):
        d = dates[i]
        if cur_day is None:
            cur_day = d
            thr = float(np.quantile(np.asarray(prior_scores, dtype=np.float64), 1.0 - target_frac)) if prior_scores else float("inf")

        if d != cur_day:
            if cur_day_scores:
                prior_scores.extend(cur_day_scores)
                if max_prior_scores > 0 and len(prior_scores) > max_prior_scores:
                    prior_scores = prior_scores[-max_prior_scores:]
            cur_day_scores = []
            cur_day = d
            thr = float(np.quantile(np.asarray(prior_scores, dtype=np.float64), 1.0 - target_frac)) if prior_scores else float("inf")

        s = scores[i]
        if np.isfinite(s):
            cur_day_scores.append(float(s))

        if i < int(blocked_until_signal_i):
            continue

        if not np.isfinite(s):
            continue

        if float(s) < float(thr):
            continue

        entry_i = i + 1
        exit_i = entry_i + int(hold_min)
        if exit_i >= len(bars):
            continue

        entry_open = float(bars.loc[entry_i, "open"])

        # oracle best exit within k=1..hold_min at close
        best_ret = -1e30
        best_k = 1
        best_exit_i = entry_i + 1
        best_exit_px = float(bars.loc[best_exit_i, "close"])
        for k in range(1, int(hold_min) + 1):
            j = entry_i + k
            px = float(bars.loc[j, "close"])
            r = net_return_pct(entry_open, px, fee_side)
            if r > best_ret:
                best_ret = float(r)
                best_k = int(k)
                best_exit_i = int(j)
                best_exit_px = float(px)

        hold_px = float(bars.loc[exit_i, "close"])
        hold_ret = net_return_pct(entry_open, hold_px, fee_side)

        win_slice = slice(entry_i, exit_i + 1)
        max_high = float(np.nanmax(pd.to_numeric(bars.loc[win_slice, "high"], errors="coerce").to_numpy(np.float64)))
        min_low = float(np.nanmin(pd.to_numeric(bars.loc[win_slice, "low"], errors="coerce").to_numpy(np.float64)))
        mfe_gross_pct = (max_high / entry_open - 1.0) * 100.0
        mae_gross_pct = (min_low / entry_open - 1.0) * 100.0

        trades.append(
            TradeRow(
                signal_i=int(i),
                signal_time=str(ts.iloc[i]),
                day=str(cur_day),
                score=float(s),
                threshold=float(thr),
                entry_i=int(entry_i),
                entry_time=str(ts.iloc[entry_i]),
                entry_open=float(entry_open),
                hold_exit_i=int(exit_i),
                hold_exit_time=str(ts.iloc[exit_i]),
                hold_exit_close=float(hold_px),
                hold_ret_1x_pct_net=float(hold_ret),
                oracle_k=int(best_k),
                oracle_exit_i=int(best_exit_i),
                oracle_exit_time=str(ts.iloc[best_exit_i]),
                oracle_exit_close=float(best_exit_px),
                oracle_ret_1x_pct_net=float(best_ret),
                oracle_gap_pct_points=float(best_ret - hold_ret),
                mfe_gross_pct_hi=float(mfe_gross_pct),
                mae_gross_pct_lo=float(mae_gross_pct),
            )
        )

        blocked_until_signal_i = int(exit_i)

    return trades


def _safe_feature_snapshot(feats: pd.DataFrame, i: int, cols: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if i < 0 or i >= len(feats):
        return out
    row = feats.iloc[int(i)]
    for c in cols:
        if c not in feats.columns:
            continue
        v = row.get(c)
        try:
            fv = float(v)
        except Exception:
            continue
        if not np.isfinite(fv):
            continue
        out[c] = fv
    return out


def plot_trade(
    *,
    bars: pd.DataFrame,
    trade: TradeRow,
    hold_min: int,
    fee_side: float,
    out_path: Path,
    feats: pd.DataFrame,
    feat_cols_for_box: List[str],
    pre_lookback_min: int,
    post_lookback_min: int,
) -> None:
    ts = pd.to_datetime(bars["timestamp"])  # naive in CSV

    signal_i = int(trade.signal_i)
    entry_i = int(trade.entry_i)
    hold_exit_i = int(trade.hold_exit_i)
    oracle_exit_i = int(trade.oracle_exit_i)

    w0 = max(0, signal_i - int(pre_lookback_min))
    w1 = min(len(bars) - 1, hold_exit_i + int(post_lookback_min))

    w = bars.loc[w0:w1, ["timestamp", "open", "high", "low", "close"]].copy()
    w_ts = pd.to_datetime(w["timestamp"])

    entry_open = float(trade.entry_open)

    # exit curve (1..hold_min)
    ks = np.arange(1, int(hold_min) + 1)
    rets = []
    for k in ks:
        j = entry_i + int(k)
        px = float(bars.loc[j, "close"])
        rets.append(net_return_pct(entry_open, px, fee_side))

    # feature snapshots
    snap_sig = _safe_feature_snapshot(feats, int(trade.signal_i), feat_cols_for_box)
    snap_entry = _safe_feature_snapshot(feats, int(trade.entry_i), feat_cols_for_box)

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.0, 1.2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2 = fig.add_subplot(gs[2, 0])

    # --- Price path (with more history) ---
    ax1.plot(w_ts, w["close"].to_numpy(np.float64), linewidth=1.2, color="black", label="close")
    ax1.fill_between(
        w_ts,
        w["low"].to_numpy(np.float64),
        w["high"].to_numpy(np.float64),
        color="#cccccc",
        alpha=0.35,
        linewidth=0.0,
        label="low-high",
    )

    def _vline(i: int, color: str, label: str, ax):
        ax.axvline(ts.iloc[i], color=color, linestyle="--", linewidth=1.2, label=label)

    _vline(signal_i, "#9467bd", "signal (decide)", ax1)
    _vline(entry_i, "#1f77b4", "entry", ax1)
    _vline(oracle_exit_i, "#2ca02c", f"oracle exit (k={trade.oracle_k})", ax1)
    _vline(hold_exit_i, "#d62728", f"hold exit (k={hold_min})", ax1)

    ax1.set_title(
        f"Entry {trade.entry_time} | score={trade.score:.4f} thr={trade.threshold:.4f} | "
        f"hold={trade.hold_ret_1x_pct_net:.3f}% oracle={trade.oracle_ret_1x_pct_net:.3f}% gap={trade.oracle_gap_pct_points:.3f}pp"
    )
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="upper left", fontsize=9)

    # --- Mid panel: what happened right before? (ret1m + range1m from engineered frame) ---
    ww = feats.loc[w0:w1, ["timestamp"]].copy()
    ww_ts = pd.to_datetime(ww["timestamp"], utc=True)

    if "px__ret1m_close" in feats.columns:
        ax_mid.plot(ww_ts, pd.to_numeric(feats.loc[w0:w1, "px__ret1m_close"], errors="coerce"), color="#ff7f0e", linewidth=1.0, label="ret1m_close (%)")
    if "px__range_norm1m" in feats.columns:
        ax_mid.plot(ww_ts, pd.to_numeric(feats.loc[w0:w1, "px__range_norm1m"], errors="coerce") * 100.0, color="#17becf", linewidth=1.0, label="range_norm1m (%)")

    _vline(signal_i, "#9467bd", "signal", ax_mid)
    _vline(entry_i, "#1f77b4", "entry", ax_mid)
    ax_mid.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax_mid.set_ylabel("micro-move")
    ax_mid.grid(True, alpha=0.2)
    ax_mid.legend(loc="upper left", fontsize=9)

    # --- Exit curve ---
    ax2.plot(ks, rets, marker="o", linewidth=1.2, color="#444444")
    ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax2.scatter([int(trade.oracle_k)], [float(trade.oracle_ret_1x_pct_net)], color="#2ca02c", s=80, zorder=5, label="oracle")
    ax2.scatter([int(hold_min)], [float(trade.hold_ret_1x_pct_net)], color="#d62728", s=80, zorder=5, label="hold")
    ax2.set_xlabel("k (minutes after entry, exit at close)")
    ax2.set_ylabel("net return %")
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc="upper left", fontsize=9)

    # Text box
    lines = []
    lines.append(f"MFE_hi_gross: {trade.mfe_gross_pct_hi:.3f}%")
    lines.append(f"MAE_lo_gross: {trade.mae_gross_pct_lo:.3f}%")
    lines.append("\nfeat@signal_i:")
    for k, v in sorted(snap_sig.items()):
        lines.append(f"  {k}: {v:.4g}")
    lines.append("\nfeat@entry_i:")
    for k, v in sorted(snap_entry.items()):
        lines.append(f"  {k}: {v:.4g}")

    fig.text(0.69, 0.06, "\n".join(lines[:44]), fontsize=8, family="monospace")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 0.66, 1])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def group_feature_summary(df: pd.DataFrame, feat_cols: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if len(df) == 0:
        return out
    for c in feat_cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        s = s[np.isfinite(s.to_numpy(np.float64))]
        if len(s) == 0:
            continue
        out[c] = {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "p10": float(s.quantile(0.10)),
            "p90": float(s.quantile(0.90)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Drill-down analysis + plotting (hold vs oracle)")
    ap.add_argument("--market-csv", default=str(Path("data/dydx_BTC-USD_1MIN_2026-01-02T18-50-42Z.csv")))
    ap.add_argument(
        "--entry-model",
        default=str(Path("data/pattern_entry_regressor/pattern_entry_regressor_v2_2026-01-02T23-51-17Z.joblib")),
    )
    ap.add_argument("--target-frac", type=float, default=0.001)
    ap.add_argument("--hold-min", type=int, default=15)
    ap.add_argument("--fee-side", type=float, default=0.001)
    ap.add_argument("--max-prior-scores", type=int, default=200_000)

    ap.add_argument("--last-n", type=int, default=100)
    ap.add_argument("--n-worst", type=int, default=10)
    ap.add_argument("--n-best", type=int, default=10)

    ap.add_argument("--pre-lookback-min", type=int, default=60, help="Minutes to show before the signal in plots")
    ap.add_argument("--post-lookback-min", type=int, default=10, help="Minutes to show after hold exit in plots")

    ap.add_argument("--gap-thr", type=float, default=0.75, help="Oracle gap (pct-points) threshold for harmful_gap bucket")
    ap.add_argument("--early-k", type=int, default=2, help="Max oracle_k for harmful_gap bucket")

    ap.add_argument("--out-dir", default=f"data/analysis_oracle_drilldown_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}")

    args = ap.parse_args()

    generated_utc = now_utc_iso()

    market_csv = Path(args.market_csv)
    model_path = Path(args.entry_model)
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data/model
    bars = pd.read_csv(market_csv, usecols=["timestamp", "open", "high", "low", "close", "volume"], parse_dates=["timestamp"])
    bars = bars.sort_values("timestamp").reset_index(drop=True)

    art = joblib.load(model_path)
    model = art["model"]
    feature_cols = list(art.get("feature_cols") or art.get("features") or [])
    base_features = list(art.get("base_features") or ["macd", "ret_1m_pct", "mom_5m_pct", "vol_std_5m", "range_norm_5m"])

    feats = build_entry_pattern_frame_v2(bars, base_features=base_features)

    X = feats[feature_cols]
    arr = X.to_numpy(dtype=np.float64)
    finite = np.isfinite(arr).all(axis=1)
    idx = np.arange(len(bars), dtype=np.int64)
    finite &= (idx >= 1440)

    scores = np.full(len(bars), np.nan, dtype=np.float64)
    if np.any(finite):
        scores[finite] = model.predict(X.iloc[np.where(finite)[0]].to_numpy(dtype=np.float32))

    trades = simulate_trades(
        bars=bars,
        scores=scores,
        target_frac=float(args.target_frac),
        hold_min=int(args.hold_min),
        fee_side=float(args.fee_side),
        max_prior_scores=int(args.max_prior_scores),
    )

    tr = pd.DataFrame([asdict(t) for t in trades])

    # Save last trades
    if len(tr) == 0:
        payload = {"generated_utc": generated_utc, "error": "no trades produced", "params": vars(args)}
        (out_dir / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    last_n = int(args.last_n)
    last = tr.tail(last_n).copy().reset_index(drop=True)

    # Attach some feature snapshots (signal/entry) to the last-N frame for group stats.
    feat_interest = [
        "macd__last",
        "ret_1m_pct__last",
        "mom_5m_pct__last",
        "vol_std_5m__last",
        "range_norm_5m__last",
        "corr5__macd__ret_1m_pct",
        "corr5__vol_std_5m__ret_1m_pct",
        "corr5__range_norm_5m__ret_1m_pct",
        "px__ret1m_close",
        "px__range_norm1m",
        "z1d__px_ret1m_abs",
        "z1d__px_range1m_abs",
        "z1d__vol5",
        "risk__ret1m_abs_over_vol5",
        "risk__range1m_over_vol5",
        "risk__range1m_over_range5max",
        "flag__ret1m_abs_z_gt3",
        "flag__range1m_abs_z_gt3",
    ]
    feat_interest = [c for c in feat_interest if c in feats.columns]

    def _attach(prefix: str, idx_col: str) -> None:
        cols = []
        for c in feat_interest:
            cols.append(f"{prefix}{c}")
        for c in cols:
            last[c] = np.nan
        for r_i, row in last.iterrows():
            ii = int(row[idx_col])
            snap = _safe_feature_snapshot(feats, ii, feat_interest)
            for c, v in snap.items():
                last.loc[r_i, f"{prefix}{c}"] = float(v)

    _attach("sig_", "signal_i")
    _attach("ent_", "entry_i")

    # Add raw pre-signal (lookback) metrics computed directly from OHLC.
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(np.float64)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(np.float64)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(np.float64)
    open_ = pd.to_numeric(bars["open"], errors="coerce").to_numpy(np.float64)

    ret1m = np.full(len(bars), np.nan, dtype=np.float64)
    ret1m[1:] = close[1:] / np.maximum(1e-12, close[:-1]) - 1.0

    def _pre_metrics(i: int) -> Dict[str, float]:
        out: Dict[str, float] = {}

        def _slice(n: int) -> slice:
            a = max(0, i - int(n) + 1)
            return slice(a, i + 1)

        for win in (5, 15, 60):
            if i - win >= 0:
                out[f"pre_ret_{win}m"] = float(close[i] / np.maximum(1e-12, close[i - win]) - 1.0)
            else:
                out[f"pre_ret_{win}m"] = float("nan")

        for win in (15, 60):
            sl = _slice(win)
            h = float(np.nanmax(high[sl]))
            l = float(np.nanmin(low[sl]))
            out[f"pre_range_{win}m"] = float(h / np.maximum(1e-12, l) - 1.0) if (np.isfinite(h) and np.isfinite(l)) else float("nan")

            rr = ret1m[sl]
            out[f"pre_vol_{win}m"] = float(np.nanstd(rr))
            out[f"pre_absret_max_{win}m"] = float(np.nanmax(np.abs(rr)))
            out[f"pre_absret_sum_{win}m"] = float(np.nansum(np.abs(rr)))
            out[f"pre_negret_sum_{win}m"] = float(np.nansum(np.clip(rr, None, 0.0)))
            out[f"pre_posret_sum_{win}m"] = float(np.nansum(np.clip(rr, 0.0, None)))
            out[f"pre_pos_frac_{win}m"] = float(np.nanmean((rr > 0.0).astype(np.float64)))

        # candle shape of the decision candle (index i)
        o = float(open_[i])
        c = float(close[i])
        hi = float(high[i])
        lo = float(low[i])
        rng = max(1e-12, hi - lo)
        out["candle_ret_1m"] = float(c / np.maximum(1e-12, o) - 1.0)
        out["candle_range_norm"] = float((hi - lo) / np.maximum(1e-12, c))
        out["candle_body_frac"] = float(abs(c - o) / rng)
        out["candle_up"] = float(1.0 if c > o else 0.0)

        return out

    # Attach pre-metrics to last-N
    for c in [
        "pre_ret_5m",
        "pre_ret_15m",
        "pre_ret_60m",
        "pre_range_15m",
        "pre_range_60m",
        "pre_vol_15m",
        "pre_absret_max_15m",
        "candle_range_norm",
        "candle_body_frac",
        "candle_up",
    ]:
        last[c] = np.nan

    for r_i, row in last.iterrows():
        ii = int(row["signal_i"])
        m = _pre_metrics(ii)
        for k, v in m.items():
            if k in last.columns:
                last.loc[r_i, k] = float(v)

    # Selection sets (refined)
    n_worst = int(args.n_worst)
    n_best = int(args.n_best)

    # A) worst by gap (overall)
    worst_gap = last.sort_values("oracle_gap_pct_points", ascending=False).head(n_worst).copy()

    # B) harmful gap among losers: big improvement is possible, but only with early exit
    harmful_gap = last[(last["hold_ret_1x_pct_net"] < 0.0) & (last["oracle_gap_pct_points"] >= float(args.gap_thr)) & (last["oracle_k"] <= int(args.early_k))]
    harmful_gap = harmful_gap.sort_values("oracle_gap_pct_points", ascending=False).head(n_worst).copy()

    # C) worst hold returns
    worst_hold = last.sort_values("hold_ret_1x_pct_net", ascending=True).head(n_worst).copy()

    # D) unrecoverable losers: even oracle<=0
    unrecoverable = last[(last["hold_ret_1x_pct_net"] < 0.0) & (last["oracle_ret_1x_pct_net"] <= 0.0)]
    unrecoverable = unrecoverable.sort_values("hold_ret_1x_pct_net", ascending=True).head(n_worst).copy()

    # E) best hold returns
    best_hold = last.sort_values("hold_ret_1x_pct_net", ascending=False).head(n_best).copy()

    # F) missed profit among winners
    missed_profit = last[last["hold_ret_1x_pct_net"] > 0.0].sort_values("oracle_gap_pct_points", ascending=False).head(n_best).copy()

    # Plotting
    feat_cols_for_box = [
        "ret_1m_pct__last",
        "mom_5m_pct__last",
        "vol_std_5m__last",
        "range_norm_5m__last",
        "px__ret1m_close",
        "px__range_norm1m",
        "z1d__px_ret1m_abs",
        "z1d__px_range1m_abs",
        "risk__ret1m_abs_over_vol5",
        "risk__range1m_over_vol5",
        "flag__ret1m_abs_z_gt3",
        "flag__range1m_abs_z_gt3",
    ]
    feat_cols_for_box = [c for c in feat_cols_for_box if c in feats.columns]

    def _plot_set(df: pd.DataFrame, tag: str) -> List[str]:
        out_files: List[str] = []
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            t = TradeRow(**{k: row[k] for k in TradeRow.__dataclass_fields__.keys()})
            name = f"{tag}_rank{rank:02d}_entry_{str(t.entry_time).replace(' ','_').replace(':','-')}_gen_{generated_utc.replace(':','-')}".replace("+00-00", "Z")
            out_path = plots_dir / f"{name}.png"
            plot_trade(
                bars=bars,
                trade=t,
                hold_min=int(args.hold_min),
                fee_side=float(args.fee_side),
                out_path=out_path,
                feats=feats,
                feat_cols_for_box=feat_cols_for_box,
                pre_lookback_min=int(args.pre_lookback_min),
                post_lookback_min=int(args.post_lookback_min),
            )
            out_files.append(str(out_path))
        return out_files

    plots = {
        "worst_gap": _plot_set(worst_gap, "worst_gap"),
        "harmful_gap": _plot_set(harmful_gap, "harmful_gap"),
        "worst_hold": _plot_set(worst_hold, "worst_hold"),
        "unrecoverable": _plot_set(unrecoverable, "unrecoverable"),
        "best_hold": _plot_set(best_hold, "best_hold"),
        "missed_profit": _plot_set(missed_profit, "missed_profit"),
    }

    # Summaries
    def _stats(df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) == 0:
            return {}
        hold = df["hold_ret_1x_pct_net"].to_numpy(np.float64)
        oracle = df["oracle_ret_1x_pct_net"].to_numpy(np.float64)
        gap = df["oracle_gap_pct_points"].to_numpy(np.float64)
        ks = df["oracle_k"].to_numpy(np.int64)
        return {
            "n": int(len(df)),
            "hold_mean_pct": float(np.mean(hold)),
            "hold_median_pct": float(np.median(hold)),
            "hold_win_rate": float(np.mean(hold > 0.0)),
            "oracle_mean_pct": float(np.mean(oracle)),
            "oracle_median_pct": float(np.median(oracle)),
            "oracle_win_rate": float(np.mean(oracle > 0.0)),
            "gap_mean_pct_points": float(np.mean(gap)),
            "gap_median_pct_points": float(np.median(gap)),
            "oracle_k_mean": float(np.mean(ks)),
            "oracle_k_counts": {str(int(k)): int(v) for k, v in pd.Series(ks).value_counts().sort_index().items()},
        }

    # Feature summaries for groups (signal-time)
    sig_cols = [f"sig_{c}" for c in feat_interest if f"sig_{c}" in last.columns]

    def _auc_rank(y: np.ndarray, x: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.int64)
        x = np.asarray(x, dtype=np.float64)
        m = np.isfinite(x)
        y = y[m]
        x = x[m]
        if len(np.unique(y)) < 2:
            return float("nan")
        order = np.argsort(x)
        xs = x[order]
        ys = y[order]
        ranks = np.empty_like(xs, dtype=np.float64)
        i = 0
        r = 1
        n = len(xs)
        while i < n:
            j = i
            while j + 1 < n and xs[j + 1] == xs[i]:
                j += 1
            avg_rank = (r + (r + (j - i))) / 2.0
            ranks[i : j + 1] = avg_rank
            r += (j - i + 1)
            i = j + 1
        pos = ys == 1
        n_pos = int(pos.sum())
        n_neg = int((~pos).sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        sum_r_pos = float(ranks[pos].sum())
        return float((sum_r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

    def _top_auc(df_in: pd.DataFrame, label_mask: np.ndarray, cols: List[str], top_k: int = 12) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        y = np.asarray(label_mask, dtype=np.int64)
        for c in cols:
            x = pd.to_numeric(df_in[c], errors="coerce").to_numpy(np.float64)
            auc = _auc_rank(y, x)
            if not np.isfinite(auc):
                continue
            out.append({"col": str(c), "auc": float(auc), "abs_auc_minus_0p5": float(abs(auc - 0.5)), "direction": ("higher->positive" if auc > 0.5 else "lower->positive")})
        out.sort(key=lambda d: float(d["abs_auc_minus_0p5"]), reverse=True)
        return out[: int(top_k)]

    # Predictor analysis on last-N
    sig_cols2 = [c for c in last.columns if c.startswith("sig_")]
    pre_cols2 = [c for c in last.columns if c.startswith("pre_") or c.startswith("candle_")]

    y_hold_loss = (pd.to_numeric(last["hold_ret_1x_pct_net"], errors="coerce").to_numpy(np.float64) < 0.0)
    y_catastrophic = (pd.to_numeric(last["hold_ret_1x_pct_net"], errors="coerce").to_numpy(np.float64) < -1.0)
    y_harmful_gap = (
        (pd.to_numeric(last["hold_ret_1x_pct_net"], errors="coerce").to_numpy(np.float64) < 0.0)
        & (pd.to_numeric(last["oracle_gap_pct_points"], errors="coerce").to_numpy(np.float64) >= float(args.gap_thr))
        & (pd.to_numeric(last["oracle_k"], errors="coerce").to_numpy(np.float64) <= float(args.early_k))
    )

    predictors = {
        "hold_loss": {
            "n_pos": int(np.sum(y_hold_loss)),
            "sig_top": _top_auc(last, y_hold_loss, sig_cols2, top_k=12),
            "pre_top": _top_auc(last, y_hold_loss, pre_cols2, top_k=12),
        },
        "catastrophic_hold": {
            "n_pos": int(np.sum(y_catastrophic)),
            "sig_top": _top_auc(last, y_catastrophic, sig_cols2, top_k=12),
            "pre_top": _top_auc(last, y_catastrophic, pre_cols2, top_k=12),
        },
        "harmful_gap": {
            "n_pos": int(np.sum(y_harmful_gap)),
            "sig_top": _top_auc(last, y_harmful_gap, sig_cols2, top_k=12),
            "pre_top": _top_auc(last, y_harmful_gap, pre_cols2, top_k=12),
        },
    }

    report = {
        "generated_utc": generated_utc,
        "params": vars(args),
        "dataset": {"path": str(market_csv), "n_rows": int(len(bars)), "min_ts": str(bars["timestamp"].min()), "max_ts": str(bars["timestamp"].max())},
        "model": {"path": str(model_path), "n_feature_cols": int(len(feature_cols)), "base_features": base_features},
        "trade_stream": {"total_trades": int(len(tr)), "last_n": int(last_n), "last_range_entry": [str(last["entry_time"].iloc[0]), str(last["entry_time"].iloc[-1])]},
        "sets": {
            "worst_gap": {"stats": _stats(worst_gap), "feature_summary_sig": group_feature_summary(worst_gap, sig_cols)},
            "harmful_gap": {"stats": _stats(harmful_gap), "feature_summary_sig": group_feature_summary(harmful_gap, sig_cols)},
            "worst_hold": {"stats": _stats(worst_hold), "feature_summary_sig": group_feature_summary(worst_hold, sig_cols)},
            "unrecoverable": {"stats": _stats(unrecoverable), "feature_summary_sig": group_feature_summary(unrecoverable, sig_cols)},
            "best_hold": {"stats": _stats(best_hold), "feature_summary_sig": group_feature_summary(best_hold, sig_cols)},
            "missed_profit": {"stats": _stats(missed_profit), "feature_summary_sig": group_feature_summary(missed_profit, sig_cols)},
        },
        "predictors": predictors,
        "plots": plots,
    }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Save last trades CSV (include generated_utc column)
    last_out = last.copy()
    last_out.insert(0, "generated_utc", generated_utc)
    last_out.to_csv(out_dir / "last_trades.csv", index=False)


if __name__ == "__main__":
    main()
