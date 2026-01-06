#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-04T01:24:37Z
"""Compounding-focused trade selection analysis (50x leverage + profit siphon).

Goal
- Prefer MANY decently-profitable trades (e.g. net_return_pct >= 0.2%) rather than only top-tail winners.
- Split by side (BUY vs SELL) to avoid mixing asymmetric dynamics.

What this does
- Loads side-split aggregated entry-context descriptor datasets:
    data/analysis_oracle_entry_outliers_by_side_*/BUY/entry_context_agg_scored.parquet
    data/analysis_oracle_entry_outliers_by_side_*/SELL/entry_context_agg_scored.parquet
- Builds a label y = (trade_net_return_pct >= r_min).
- Trains a LightGBM classifier (time split) to predict y from pre-entry descriptors.
- Evaluates precision/recall tradeoff and simulates equity growth with:
    leverage=50x, siphon=30% of profits each trade

IMPORTANT
- This dataset contains *oracle-chosen* trades, so returns are always > 0.
  This analysis is about selecting better trades among oracle opportunities,
  not about avoiding losing trades in the real market.

Outputs
- data/analysis_compounding_selector_50x_<ts>/BUY/...
- data/analysis_compounding_selector_50x_<ts>/SELL/...
Each side folder writes:
- metrics.json
- pr_curve.csv
- thresholds.csv (precision/recall, trades/day, simulated equity)
- plots/*.png
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping, log_evaluation
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _side_dir(root: Path, side: str) -> Path:
    d = root / str(side).upper()
    (d / "plots").mkdir(parents=True, exist_ok=True)
    return d


def _feature_columns(df: pd.DataFrame) -> list[str]:
    # Use numeric descriptor columns only; exclude metadata/leak-prone columns.
    drop = {
        "trade_row",
        "trade_net_return_pct",
        "outlier_score",
        "nn_dist",
        "missing_any",
        "missing_close_n",
    }
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if c.startswith("trade_"):
            continue
        if df[c].dtype.kind not in "fc":
            continue
        cols.append(c)

    # Missingness can be useful as a feature *if* your live data has gaps.
    # Here we train/eval on complete contexts anyway, so omit to keep it clean.
    return cols


def _simulate_compounding(
    *,
    df: pd.DataFrame,
    selected_mask: np.ndarray,
    leverage: float,
    siphon: float,
    start_equity: float = 1.0,
) -> tuple[float, float, float, int]:
    """Simulate equity + bank growth over a sequence of trades.

    Uses log-space tracking to avoid overflow.

    Returns: (log10_equity, log10_bank, log10_total, n_trades_taken)
    """

    r = pd.to_numeric(df["trade_net_return_pct"], errors="coerce").to_numpy(np.float64)
    sel = selected_mask.astype(bool)

    # log-space; start with bank=0 (log=-inf)
    if start_equity <= 0:
        raise ValueError("start_equity must be > 0")
    log_eq = float(np.log(float(start_equity)))
    log_bank = float(-np.inf)

    n = 0

    for ret in r[sel]:
        profit_frac = float(leverage) * float(ret) / 100.0
        if profit_frac <= -1.0:
            # Equity wiped.
            log_eq = float(-np.inf)
            break

        if profit_frac > 0.0:
            # withdraw = eq * siphon * profit_frac
            log_withdraw = log_eq + float(np.log(float(siphon))) + float(np.log(profit_frac))
            log_bank = float(np.logaddexp(log_bank, log_withdraw))

            # reinvest equity multiplier
            mult = 1.0 + (1.0 - float(siphon)) * profit_frac
            log_eq += float(np.log(mult))
        else:
            mult = 1.0 + profit_frac
            log_eq += float(np.log(mult))

        n += 1

    # total = eq + bank
    log_total = float(np.logaddexp(log_eq, log_bank))

    # convert to log10
    ln10 = float(np.log(10.0))
    log10_eq = float(log_eq / ln10) if np.isfinite(log_eq) else float(-np.inf)
    log10_bank = float(log_bank / ln10) if np.isfinite(log_bank) else float(-np.inf)
    log10_total = float(log_total / ln10) if np.isfinite(log_total) else float(-np.inf)

    return log10_eq, log10_bank, log10_total, int(n)


def _trades_per_day(df: pd.DataFrame, selected_mask: np.ndarray) -> dict[str, float]:
    t = pd.to_datetime(df["trade_entry_time"], utc=True, errors="coerce")
    sel = selected_mask.astype(bool)
    s = pd.Series(sel.astype(int), index=t.dt.date)
    by = s.groupby(level=0).sum()
    return {
        "median": float(by.median()) if len(by) else 0.0,
        "p90": float(by.quantile(0.9)) if len(by) else 0.0,
        "max": float(by.max()) if len(by) else 0.0,
        "mean": float(by.mean()) if len(by) else 0.0,
    }


def _analyze_side(
    *,
    df_all: pd.DataFrame,
    side: str,
    out_dir: Path,
    r_min: float,
    leverage: float,
    siphon: float,
    test_frac: float,
    lgb_log_period: int,
    early_stopping_rounds: int,
    threshold_grid_points: int,
    progress_every: int,
) -> None:
    side = str(side).upper()
    d = _side_dir(out_dir, side)
    plots = d / "plots"

    t_side0 = time.perf_counter()

    df = df_all[df_all["trade_side"].astype(str).str.upper() == side].copy()
    if df.empty:
        print(f"[{side}] No rows", flush=True)
        return

    n_raw = len(df)

    # Keep complete contexts only.
    if "missing_any" in df.columns:
        df = df[df["missing_any"].astype(float) == 0.0]

    n_complete = len(df)

    # Chronological order
    df["trade_entry_time"] = pd.to_datetime(df["trade_entry_time"], utc=True, errors="coerce")
    df = df.sort_values("trade_entry_time").reset_index(drop=True)

    print(
        f"[{side}] rows: raw={n_raw:,}  complete={n_complete:,}  label=ret>={float(r_min):.3f}%  L={float(leverage):.0f}x siphon={float(siphon):.2f}",
        flush=True,
    )

    y = (pd.to_numeric(df["trade_net_return_pct"], errors="coerce") >= float(r_min)).astype(int).to_numpy(np.int64)

    feat_cols = _feature_columns(df)
    X = df[feat_cols].to_numpy(np.float32)

    # NaN handling: replace NaNs with column medians.
    Xf = X.astype(np.float64, copy=True)
    med = np.nanmedian(Xf, axis=0)
    ii = np.where(~np.isfinite(Xf))
    if ii[0].size:
        Xf[ii] = med[ii[1]]
    X = Xf.astype(np.float32)

    n = len(df)
    split = int(n * (1.0 - float(test_frac)))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    print(
        f"[{side}] split: train={len(y_tr):,} test={len(y_te):,}  base_rate(train)={float(np.mean(y_tr)):.3f} base_rate(test)={float(np.mean(y_te)):.3f}",
        flush=True,
    )

    # Train classifier
    model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=128,
        max_depth=-1,
        min_child_samples=200,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    print(f"[{side}] Training LightGBM: n_features={len(feat_cols)} n_estimators={model.n_estimators}", flush=True)
    t_fit0 = time.perf_counter()

    callbacks = []
    if int(lgb_log_period) > 0:
        callbacks.append(log_evaluation(period=int(lgb_log_period)))
    if int(early_stopping_rounds) > 0:
        callbacks.append(early_stopping(stopping_rounds=int(early_stopping_rounds), verbose=True))

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_te, y_te)],
        eval_metric="auc",
        callbacks=callbacks,
    )

    t_fit = time.perf_counter() - t_fit0
    best_iter = getattr(model, "best_iteration_", None)
    best_score = getattr(model, "best_score_", None)
    print(f"[{side}] Training done in {t_fit:.1f}s  best_iteration={best_iter}  best_score={best_score}", flush=True)

    p_te = model.predict_proba(X_te)[:, 1]

    ap = float(average_precision_score(y_te, p_te))
    try:
        auc = float(roc_auc_score(y_te, p_te))
    except Exception:
        auc = float("nan")

    prec, rec, thr = precision_recall_curve(y_te, p_te)
    pr = pd.DataFrame({"precision": prec, "recall": rec})
    pr.to_csv(d / "pr_curve.csv", index=False)

    # Evaluate thresholds: precision/recall + compounding sim
    # Use a grid over percentiles of predicted score.
    n_grid = max(10, int(threshold_grid_points))
    grid = np.unique(np.quantile(p_te, np.linspace(0.0, 0.99, n_grid))).tolist()

    rows = []
    df_te = df.iloc[split:].reset_index(drop=True)

    best_log10_total = float(-np.inf)
    best_row = None

    t_loop0 = time.perf_counter()

    for i, tcut in enumerate(grid):
        sel = p_te >= float(tcut)

        # Metrics
        tp = int(((sel) & (y_te == 1)).sum())
        fp = int(((sel) & (y_te == 0)).sum())
        fn = int(((~sel) & (y_te == 1)).sum())
        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))

        log10_eq, log10_bank, log10_total, n_trades = _simulate_compounding(
            df=df_te,
            selected_mask=sel,
            leverage=leverage,
            siphon=siphon,
            start_equity=1.0,
        )
        tpd = _trades_per_day(df_te, sel)

        row = {
            "threshold": float(tcut),
            "n_trades": int(n_trades),
            "precision": float(precision),
            "recall": float(recall),
            "log10_equity_reinvest": float(log10_eq),
            "log10_bank_withdrawn": float(log10_bank),
            "log10_total": float(log10_total),
            "trades_per_day_median": float(tpd["median"]),
            "trades_per_day_p90": float(tpd["p90"]),
        }
        rows.append(row)

        if np.isfinite(log10_total) and log10_total > best_log10_total:
            best_log10_total = float(log10_total)
            best_row = dict(row)

        pe = max(1, int(progress_every))
        if (i % pe == 0) or (i == len(grid) - 1):
            elapsed = time.perf_counter() - t_loop0
            rate = elapsed / max(1, i + 1)
            eta = rate * max(0, len(grid) - (i + 1))
            print(
                f"[{side}] thr_grid {i+1:>3}/{len(grid)}  t={float(tcut):.4f}  trades={n_trades:>5}  tpd_med={tpd['median']:.1f}  prec={precision:.3f} rec={recall:.3f}  log10_total={log10_total:.2f}  ETA={eta:.1f}s",
                flush=True,
            )

    if best_row is not None:
        print(
            f"[{side}] best log10_total={best_log10_total:.2f} at threshold={best_row['threshold']:.4f}  trades={best_row['n_trades']}  tpd_med={best_row['trades_per_day_median']:.1f}  prec={best_row['precision']:.3f} rec={best_row['recall']:.3f}",
            flush=True,
        )

    thr_df = pd.DataFrame(rows).sort_values("threshold", ascending=True).reset_index(drop=True)
    thr_df.to_csv(d / "thresholds.csv", index=False)

    # Plots: precision vs trades/day; total vs threshold
    plt.figure(figsize=(10, 5))
    plt.plot(thr_df["trades_per_day_median"], thr_df["precision"], marker="o", linewidth=1.2)
    plt.title(f"{side}: precision vs trades/day (label: ret>={r_min}%)")
    plt.xlabel("median trades/day (test)")
    plt.ylabel("precision")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots / "precision_vs_trades_per_day.png", dpi=170)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(thr_df["threshold"], thr_df["log10_total"], marker="o", linewidth=1.2)
    plt.title(f"{side}: log10(total) vs score threshold (50x, siphon={siphon:.2f})")
    plt.xlabel("score threshold")
    plt.ylabel("log10(total)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(plots / "log_total_vs_threshold.png", dpi=170)
    plt.close()

    # Save metrics
    metrics = {
        "side": side,
        "n_rows": int(n),
        "test_frac": float(test_frac),
        "label_r_min_pct": float(r_min),
        "base_rate": float(y.mean()),
        "base_rate_test": float(y_te.mean()),
        "average_precision": ap,
        "roc_auc": auc,
        "feature_count": int(len(feat_cols)),
        "feature_cols": feat_cols,
    }

    (d / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    t_side = time.perf_counter() - t_side0
    print(f"[{side}] AP={ap:.4f} AUC={auc:.4f}  wrote {d}  elapsed={t_side:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compounding-focused trade selection analysis (50x, siphon)")
    ap.add_argument(
        "--input-root",
        default="data/analysis_oracle_entry_outliers_by_side_2026-01-04T00-55-44Z",
        help="Root folder with BUY/SELL entry_context_agg_scored.parquet",
    )
    ap.add_argument("--r-min", type=float, default=0.2, help="Minimum net_return_pct considered a 'decent' trade")
    ap.add_argument("--leverage", type=float, default=50.0)
    ap.add_argument("--siphon", type=float, default=0.30)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--side", default="both", choices=["both", "BUY", "SELL", "buy", "sell"])
    ap.add_argument("--out-dir", default="data")

    ap.add_argument("--lgb-log-period", type=int, default=50, help="LightGBM eval logging period (0 disables)")
    ap.add_argument("--early-stopping-rounds", type=int, default=100, help="LightGBM early stopping rounds (0 disables)")
    ap.add_argument("--threshold-grid-points", type=int, default=60, help="Number of thresholds to evaluate (percentile grid)")
    ap.add_argument("--progress-every", type=int, default=10, help="Print progress every N thresholds")

    args = ap.parse_args()

    t0 = time.perf_counter()

    root = Path(args.input_root)
    buy_p = root / "BUY" / "entry_context_agg_scored.parquet"
    sell_p = root / "SELL" / "entry_context_agg_scored.parquet"
    if not buy_p.exists() or not sell_p.exists():
        raise SystemExit(f"Expected BUY/SELL scored parquet under {root}")

    print("Config:", flush=True)
    print(f"  input_root={root}", flush=True)
    print(f"  r_min={float(args.r_min):.3f}%  leverage={float(args.leverage):.0f}x  siphon={float(args.siphon):.2f}  test_frac={float(args.test_frac):.2f}", flush=True)
    print(f"  lgb_log_period={int(args.lgb_log_period)}  early_stopping_rounds={int(args.early_stopping_rounds)}", flush=True)
    print(f"  threshold_grid_points={int(args.threshold_grid_points)}  progress_every={int(args.progress_every)}", flush=True)

    print("Loading BUY/SELL datasets...", flush=True)
    buy = pd.read_parquet(buy_p)
    sell = pd.read_parquet(sell_p)
    df_all = pd.concat([buy, sell], ignore_index=True)
    print(f"Loaded: BUY={len(buy):,}  SELL={len(sell):,}  total={len(df_all):,}", flush=True)

    out_root = Path(args.out_dir) / f"analysis_compounding_selector_50x_{_now_ts()}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_root}", flush=True)

    sides = ["BUY", "SELL"] if str(args.side).upper() == "BOTH" else [str(args.side).upper()]

    for s in sides:
        _analyze_side(
            df_all=df_all,
            side=s,
            out_dir=out_root,
            r_min=float(args.r_min),
            leverage=float(args.leverage),
            siphon=float(args.siphon),
            test_frac=float(args.test_frac),
            lgb_log_period=int(args.lgb_log_period),
            early_stopping_rounds=int(args.early_stopping_rounds),
            threshold_grid_points=int(args.threshold_grid_points),
            progress_every=int(args.progress_every),
        )

    elapsed = time.perf_counter() - t0
    print("Wrote:", out_root, flush=True)
    print(f"Done in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
