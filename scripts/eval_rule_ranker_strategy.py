#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover
    raise SystemExit("lightgbm is required. pip install lightgbm") from exc

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover
    raise SystemExit("xgboost is required. pip install xgboost") from exc

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import compute_feature_frame  # noqa: E402

ENTRY_FEATURES = [
    "ret_1m_pct",
    "mom_3m_pct",
    "mom_5m_pct",
    "vol_std_5m",
    "range_5m",
    "range_norm_5m",
    "macd",
    "vwap_dev_5m",
]

EXIT_FEATURE_COLS = [
    "elapsed_min",
    "since_entry_return_pct",
    "prob_enter_at_entry",
    "daily_pred_high_profit",
    "momentum_3m",
    "momentum_5m",
    "momentum_10m",
    "volatility_3m",
    "volatility_5m",
    "volatility_10m",
    "drawdown_from_max_pct",
    "since_entry_cummax",
    "mins_since_peak",
]


@dataclass
class StrategyResult:
    trades: list[dict]
    daily: list[dict]
    final_capital: float
    final_bank: float
    total_liquidations: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate rule-ranker strategy on BTCUSDT data")
    ap.add_argument("--market-csv", default="data/btc_profitability_analysis_filtered.csv")
    ap.add_argument("--rule-trades", default="data/pattern_runs/rule_best/rule_rule__rn0.0015__vw0.000__mn1__m30.050__vwmin0.020__sl5.0__rsi55.0.parquet")
    ap.add_argument("--entry-models-dir", default="models")
    ap.add_argument("--entry-stem", default="rule_best_ranker")
    ap.add_argument("--exit-model", default="models/exit_timing_model_weighted.joblib")
    ap.add_argument("--entry-threshold", type=float, default=0.52)
    ap.add_argument("--k-per-day", type=int, default=3)
    ap.add_argument("--hold-min", type=int, default=8)
    ap.add_argument("--min-profit-pct", type=float, default=0.10)
    ap.add_argument("--thr-exit", type=float, default=0.50)
    ap.add_argument("--fee-rate", type=float, default=0.001, help="Per-side fee (e.g., 0.001 = 0.1%)")
    ap.add_argument("--start-date", type=str, default="2022-01-01")
    ap.add_argument("--end-date", type=str, default="2025-12-18")
    ap.add_argument("--start-capital", type=float, default=30.0)
    # Entry context prefilters
    ap.add_argument("--min-vol-std-5m", type=float, default=None)
    ap.add_argument("--min-range-norm-5m", type=float, default=None)
    ap.add_argument("--rsi-min", type=float, default=None)
    ap.add_argument("--rsi-max", type=float, default=None)
    ap.add_argument("--max-vwap-dev-abs", type=float, default=None, help="Require |vwap_dev_5m| <= this threshold (in %)")
    ap.add_argument("--min-mom-3m-pct", type=float, default=None, help="Require mom_3m_pct >= this value")
    ap.add_argument("--allow-hours", type=str, default=None, help="Comma-separated UTC hours, e.g., 0,3,9,10,19")
    # Exit behavior tweaks
    ap.add_argument("--early-cut-at", type=int, default=7, help="If no profit hit, cut at this minute instead of hold_min")
    ap.add_argument("--enable-peak-stop", action="store_true")
    ap.add_argument("--peak-stop-dd", type=float, default=-0.20, help="Drawdown from max (in %) to trigger exit after hitting a min profit")
    ap.add_argument("--peak-trigger-min-profit", type=float, default=0.10, help="Min profit (in %) before enabling peak stop")
    # Outputs
    ap.add_argument("--out-trades", type=str, default="data/entry_ranked_outputs/rule_ranker_trades.csv")
    ap.add_argument("--out-daily", type=str, default="data/entry_ranked_outputs/rule_ranker_daily.csv")
    ap.add_argument("--metrics-out", type=str, default="data/entry_ranked_outputs/rule_ranker_metrics.csv")
    return ap.parse_args()


def load_market(csv_path: Path, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["ts_min"] = pd.to_datetime(df["timestamp"]).dt.floor("min")
    agg = (
        df.groupby("ts_min")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )
    if start is not None:
        agg = agg[agg["ts_min"] >= start]
    if end is not None:
        agg = agg[agg["ts_min"] <= end]
    agg = agg.sort_values("ts_min").reset_index(drop=True)
    return agg


def map_ts_to_index(ts_sorted: np.ndarray, query: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(ts_sorted, query)
    ok = (pos < ts_sorted.size) & (ts_sorted[pos] == query)
    return np.where(ok, pos, -1).astype(np.int64)


def attach_entry_features(trades: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    ts_sorted = feat["ts_min"].to_numpy(dtype="datetime64[ns]")
    idx = map_ts_to_index(ts_sorted, trades["entry_ts"].to_numpy(dtype="datetime64[ns]"))
    ok = idx >= 0
    trades = trades.loc[ok].reset_index(drop=True)
    idx = idx[ok]
    feat_cols = feat.iloc[idx][ENTRY_FEATURES].reset_index(drop=True)
    trades = pd.concat([trades.reset_index(drop=True), feat_cols], axis=1)
    trades["entry_date"] = trades["entry_ts"].dt.date
    return trades


def load_entry_models(models_dir: Path, stem: str) -> tuple[Optional[lgb.Booster], Optional[xgb.Booster]]:
    lgb_path = models_dir / f"{stem}__entry_ranker_lgb.txt"
    xgb_path = models_dir / f"{stem}__entry_ranker_xgb.json"
    lgb_model = lgb.Booster(model_file=str(lgb_path)) if lgb_path.exists() else None
    xgb_model: Optional[xgb.Booster] = None
    if xgb_path.exists():
        xgb_model = xgb.Booster()
        xgb_model.load_model(str(xgb_path))
    return lgb_model, xgb_model


def score_entries(df: pd.DataFrame, lgb_model, xgb_model) -> pd.DataFrame:
    feats = df[ENTRY_FEATURES].fillna(0.0).astype(np.float32)
    scores = []
    if lgb_model is not None:
        cols = list(lgb_model.feature_name())
        Xl = feats.reindex(columns=cols, fill_value=0.0).to_numpy(dtype=float)
        scores.append(lgb_model.predict(Xl))
    if xgb_model is not None:
        Xx = feats[ENTRY_FEATURES].to_numpy(dtype=float)
        dmat = xgb.DMatrix(Xx, feature_names=ENTRY_FEATURES)
        scores.append(xgb_model.predict(dmat))
    if not scores:
        raise SystemExit("No entry models found for scoring")
    df = df.copy()
    df["score"] = np.mean(np.vstack(scores), axis=0)
    return df


def select_entries(df: pd.DataFrame, thr: float, k_per_day: int, hold_min: int, allow_hours: Optional[set[int]], filters: dict) -> pd.DataFrame:
    rows = []
    hold_delta = pd.Timedelta(minutes=hold_min)
    # Apply entry prefilters
    gdf = df.copy()
    mv = filters.get("min_vol_std_5m"); mrn = filters.get("min_range_norm_5m"); rmin = filters.get("rsi_min"); rmax = filters.get("rsi_max"); mvd = filters.get("max_vwap_dev_abs"); mm3 = filters.get("min_mom_3m_pct")
    if mv is not None:
        gdf = gdf[gdf["vol_std_5m"] >= float(mv)]
    if mrn is not None:
        gdf = gdf[gdf["range_norm_5m"] >= float(mrn)]
    if rmin is not None:
        gdf = gdf[gdf["rsi_14"] >= float(rmin)]
    if rmax is not None:
        gdf = gdf[gdf["rsi_14"] <= float(rmax)]
    if mvd is not None:
        gdf = gdf[gdf["vwap_dev_5m"].abs() <= float(mvd)]
    if mm3 is not None:
        gdf = gdf[gdf["mom_3m_pct"] >= float(mm3)]

    for date, g in gdf.groupby("entry_date"):
        g = g[g["score"] >= thr].sort_values("score", ascending=False)
        last_exit_ts: Optional[pd.Timestamp] = None
        chosen = []
        for _, row in g.iterrows():
            if allow_hours is not None and int(pd.to_datetime(row["entry_ts"]).hour) not in allow_hours:
                continue
            if last_exit_ts is not None and row["entry_ts"] < last_exit_ts:
                continue
            chosen.append(row)
            last_exit_ts = row["entry_ts"] + hold_delta
            if len(chosen) >= k_per_day:
                break
        if chosen:
            rows.append(pd.DataFrame(chosen))
    if not rows:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(rows, ignore_index=True)


def build_exit_features(day_slice: pd.DataFrame, entry_idx: int, hold_min: int, fee_rate: float) -> pd.DataFrame:
    start_row = day_slice.iloc[entry_idx]
    entry_time = start_row["timestamp"]
    entry_price = float(start_row["close"])
    end_idx = min(len(day_slice) - 1, entry_idx + hold_min)
    seq = day_slice.iloc[entry_idx : end_idx + 1].copy().reset_index(drop=True)
    seq["elapsed_min"] = (seq["timestamp"] - entry_time).dt.total_seconds().div(60).astype(int)
    entry_cost = entry_price * (1 + fee_rate)
    exit_value = seq["close"] * (1 - fee_rate)
    seq["since_entry_return_pct"] = (exit_value / entry_cost - 1.0) * 100.0
    seq["prob_enter_at_entry"] = 0.5
    seq["daily_pred_high_profit"] = 1.0
    for w in [3, 5, 10]:
        seq[f"momentum_{w}m"] = seq["close"].pct_change(w) * 100.0
        seq[f"volatility_{w}m"] = seq["close"].rolling(w).std()
    seq["since_entry_cummax"] = seq["since_entry_return_pct"].cummax()
    seq["drawdown_from_max_pct"] = seq["since_entry_return_pct"] - seq["since_entry_cummax"]
    best = -1e9
    best_i = 0
    mins_since_peak = []
    for i, v in enumerate(seq["since_entry_return_pct"].tolist()):
        if v > best:
            best = v
            best_i = i
        mins_since_peak.append(i - best_i)
    seq["mins_since_peak"] = mins_since_peak
    return seq[["timestamp"] + EXIT_FEATURE_COLS].copy()


def leverage_for_equity(total_equity: float) -> float:
    # Adjusted tiers to regain upside while controlling drawdowns
    if total_equity < 50.0:
        return 20.0
    if total_equity < 100.0:
        return 30.0
    if total_equity < 500.0:
        return 60.0
    return 100.0


def simulate_strategy(entries: pd.DataFrame, day_lookup: dict, exit_model, hold_min: int, min_profit_pct: float, thr_exit: float, fee_rate: float, start_capital: float, early_cut_at: int, enable_peak_stop: bool, peak_stop_dd: float, peak_trigger_min_profit: float) -> StrategyResult:
    if entries.empty:
        return StrategyResult([], [], start_capital, 0.0, 0)

    capital = float(start_capital)
    bank = 0.0
    total_liq = 0
    trades: list[dict] = []
    daily_returns: dict[object, float] = {}

    entries = entries.sort_values("entry_ts").reset_index(drop=True)

    for row in tqdm(entries.itertuples(index=False), desc="Simulating trades"):
        day = day_lookup.get(row.entry_date)
        if day is None or day.empty:
            continue
        ts_arr = day["timestamp"].to_numpy(dtype="datetime64[ns]")
        entry_ts = np.datetime64(row.entry_ts)
        pos = int(np.searchsorted(ts_arr, entry_ts))
        if pos >= len(day):
            continue
        feats = build_exit_features(day, pos, hold_min, fee_rate)
        if feats.empty:
            continue
        # Prepare features for exit model; support payload dict with custom feature list (including lags)
        augmented = feats.copy()
        for lag in (1, 3):
            lagged = augmented[EXIT_FEATURE_COLS].shift(lag)
            lagged.columns = [f"{c}_m{lag}" for c in lagged.columns]
            augmented = pd.concat([augmented, lagged], axis=1)
        # Determine model and feature list
        model = exit_model
        feat_list = None
        if isinstance(exit_model, dict):
            model = exit_model.get("model", exit_model)
            feat_list = exit_model.get("features", None)
        if feat_list is not None:
            X = augmented.reindex(columns=feat_list, fill_value=0.0).to_numpy(dtype=float)
        else:
            X = feats[EXIT_FEATURE_COLS].fillna(0.0).to_numpy(dtype=float)
        probs = model.predict_proba(X)[:, 1]
        window = (feats["elapsed_min"] >= 3) & (feats["elapsed_min"] <= hold_min)
        ok = window & (feats["since_entry_return_pct"] >= min_profit_pct)
        idxs = np.where(ok.to_numpy())[0]
        # Base exit: first time the exit_model confirms within OK window; else fallback
        if idxs.size > 0:
            hits = np.where(probs[idxs] >= thr_exit)[0]
            exit_idx = int(idxs[hits[0]]) if hits.size else int(idxs[0])
        else:
            exit_idx = int(min(len(feats) - 1, early_cut_at if early_cut_at is not None else hold_min))

        # 1) Dynamic fallback using prob: from minute >=5, if since<=0 and prob<0.50, cut now (prefer minute 5 over 6)
        neg_mask = feats["since_entry_return_pct"] <= 0.0
        minutes = feats["elapsed_min"].to_numpy()
        p = probs
        bad = np.where((minutes >= 5) & neg_mask.to_numpy() & (p < 0.50))[0]
        if bad.size:
            # bias to 5 if available, else first such minute
            five = bad[np.where(minutes[bad] == 5)[0]]
            cand = int(five[0]) if five.size else int(bad[0])
            exit_idx = min(exit_idx, cand)

        # 2) Time-underwater stop: 4 consecutive negative minutes from minute >=3
        neg_arr = neg_mask.to_numpy()
        run = 0
        uw_idx = None
        for i, em in enumerate(feats["elapsed_min"].to_numpy()):
            if neg_arr[i] and em >= 3:
                run += 1
                if run >= 4:
                    uw_idx = i
                    break
            else:
                run = 0
        if uw_idx is not None:
            exit_idx = min(exit_idx, int(uw_idx))

        # 2b) If strong prob and small profit, allow 2-min run but exit on quick DD
        strong = np.where((p >= 0.60) & (feats["since_entry_return_pct"].to_numpy() >= 0.10))[0]
        if strong.size:
            k = int(strong[0])
            # lookahead within next 2 minutes for DD breach
            end = min(len(feats)-1, k+2)
            after = feats.iloc[k:end+1]
            dd_hit = (after["drawdown_from_max_pct"] <= -0.15).to_numpy()
            if dd_hit.any():
                rel = int(np.argmax(dd_hit))
                exit_idx = min(exit_idx, k + rel)

        # 2c) Adaptive trailing stop after profit (tiered by reached profit)
        if enable_peak_stop:
            s = feats.copy()
            cond = s["since_entry_return_pct"] >= peak_trigger_min_profit
            if cond.any():
                peak_i = int(np.argmax(cond.to_numpy()))
                after = s.iloc[peak_i:]
                # Profit-based tiers: >=0.50 -> -0.35, >=0.30 -> -0.25, >=0.10 -> -0.20, else fallback
                prof = after["since_entry_cummax"].to_numpy()
                dd_series = after["drawdown_from_max_pct"].to_numpy()
                dyn_dd = np.full_like(dd_series, fill_value=peak_stop_dd, dtype=float)
                dyn_dd = np.where(prof >= 0.50, -0.35, dyn_dd)
                dyn_dd = np.where((prof >= 0.30) & (prof < 0.50), -0.25, dyn_dd)
                dyn_dd = np.where((prof >= 0.10) & (prof < 0.30), -0.20, dyn_dd)
                dd_hit = dd_series <= dyn_dd
                if dd_hit.any():
                    rel = int(np.argmax(dd_hit))
                    exit_idx = min(exit_idx, peak_i + rel)

        # 2c) Profit-target release valve: take profit at 0.30% unless model prob is strong (>=0.55)
        pt_idx = np.where((feats["since_entry_return_pct"].to_numpy() >= 0.30) & (probs < 0.55))[0]
        if pt_idx.size:
            exit_idx = min(exit_idx, int(pt_idx[0]))

        exit_row = feats.iloc[exit_idx]
        since_entry_pct = float(exit_row["since_entry_return_pct"])
        total_equity = capital + bank
        lever = leverage_for_equity(total_equity)
        levered_ret = lever * (since_entry_pct / 100.0)
        cap_before = capital
        bank_before = bank
        if (1.0 + levered_ret) <= 0:
            capital = 0.0
            liq_reason = "hard"
        else:
            capital = capital * (1.0 + levered_ret)
            liq_reason = ""
        profit = capital - cap_before
        bank_delta = 0.0
        # Only siphon profit when equity is reasonably robust
        if profit > 0 and total_equity >= 75.0 and lever >= 50.0:
            bank_delta = profit * 0.5
            capital -= bank_delta
            bank += bank_delta
        liquidation_event = False
        restart_source = ""
        restart_amount = 0.0
        if capital < 5.0:
            liquidation_event = True
            total_liq += 1
            if bank < 150.0:
                restart_amount = 30.0
                capital = restart_amount
                restart_source = "external_refinance"
            else:
                restart_amount = bank * 0.20
                bank -= restart_amount
                capital = restart_amount
                restart_source = "bank_topup"
            if not liq_reason:
                liq_reason = "balance_floor"
        elif liq_reason:
            liquidation_event = True
            total_liq += 1
        daily_returns[row.entry_date] = daily_returns.get(row.entry_date, 0.0) + (levered_ret * 100.0)
        trades.append(
            dict(
                date=row.entry_date,
                entry_time=row.entry_ts,
                exit_time=exit_row["timestamp"],
                duration_min=int(exit_row["elapsed_min"]),
                raw_return_pct=since_entry_pct,
                leverage=lever,
                trade_return_pct=levered_ret * 100.0,
                capital_before=cap_before,
                capital_after=capital,
                bank_before=bank_before,
                bank_after=bank,
                bank_delta=bank_delta,
                profit=profit,
                liquidated=liquidation_event,
                liquidation_reason=liq_reason,
                restart_source=restart_source,
                restart_amount=restart_amount,
            )
        )

    daily = [dict(date=d, daily_return_pct=v) for d, v in sorted(daily_returns.items())]
    return StrategyResult(trades, daily, capital, bank, total_liq)


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start_date) if args.start_date else None
    end = pd.Timestamp(args.end_date) if args.end_date else None

    trades = pd.read_parquet(args.rule_trades)
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"])
    trades["exit_ts"] = pd.to_datetime(trades["exit_ts"])
    if start is not None:
        trades = trades[trades["entry_ts"] >= start]
    if end is not None:
        trades = trades[trades["entry_ts"] <= end]

    market = load_market(Path(args.market_csv), start, end)
    feat = compute_feature_frame(market)
    feat = feat.dropna(subset=ENTRY_FEATURES).reset_index(drop=True)

    trades = attach_entry_features(trades, feat)

    models_dir = Path(args.entry_models_dir)
    lgb_model, xgb_model = load_entry_models(models_dir, args.entry_stem)
    trades_scored = score_entries(trades, lgb_model, xgb_model)
    allow_hours: Optional[set[int]] = None
    if args.allow_hours:
        try:
            allow_hours = {int(h) for h in str(args.allow_hours).split(',') if h.strip()}
        except Exception:
            allow_hours = None
    filters = dict(
        min_vol_std_5m=args.min_vol_std_5m,
        min_range_norm_5m=args.min_range_norm_5m,
        rsi_min=args.rsi_min,
        rsi_max=args.rsi_max,
        max_vwap_dev_abs=args.max_vwap_dev_abs,
        min_mom_3m_pct=args.min_mom_3m_pct,
    )
    selected = select_entries(trades_scored, args.entry_threshold, args.k_per_day, args.hold_min, allow_hours, filters)

    exit_model = joblib.load(args.exit_model)

    market["timestamp"] = market["ts_min"]
    market["date"] = market["timestamp"].dt.date
    day_lookup = {d: g.sort_values("timestamp").reset_index(drop=True) for d, g in market.groupby("date")}

    result = simulate_strategy(
        selected,
        day_lookup,
        exit_model,
        args.hold_min,
        args.min_profit_pct,
        args.thr_exit,
        args.fee_rate,
        args.start_capital,
        args.early_cut_at,
        bool(args.enable_peak_stop),
        float(args.peak_stop_dd),
        float(args.peak_trigger_min_profit),
    )

    trades_df = pd.DataFrame(result.trades)
    daily_df = pd.DataFrame(result.daily)

    Path(args.out_trades).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_daily).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(args.out_trades, index=False)
    daily_df.to_csv(args.out_daily, index=False)

    final_total = result.final_capital + result.final_bank
    daily_mean = float(daily_df["daily_return_pct"].mean()) if not daily_df.empty else 0.0
    metrics = dict(
        start_date=args.start_date,
        end_date=args.end_date,
        entry_threshold=args.entry_threshold,
        k_per_day=args.k_per_day,
        hold_min=args.hold_min,
        min_profit_pct=args.min_profit_pct,
        thr_exit=args.thr_exit,
        fee_rate=args.fee_rate,
        final_capital=result.final_capital,
        final_bank=result.final_bank,
        final_total_equity=final_total,
        daily_mean_pct=daily_mean,
        total_liquidations=result.total_liquidations,
        num_trades=len(trades_df),
        trade_days=len(daily_df),
    )
    pd.DataFrame([metrics]).to_csv(args.metrics_out, index=False)

    print("Executed trades:", len(trades_df))
    print(f"Final capital: {result.final_capital:.2f}")
    print(f"Final bank: {result.final_bank:.2f}")
    print(f"Total equity: {final_total:.2f}")
    print(f"Average daily %: {daily_mean:.4f}")
    print(f"Liquidations: {result.total_liquidations}")


if __name__ == "__main__":
    main()