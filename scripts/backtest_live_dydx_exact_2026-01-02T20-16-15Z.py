#!/usr/bin/env python3
# Timestamp (UTC): 2026-01-02T20:16:15Z
"""
Offline backtest that EXACTLY replicates the live dYdX runner logic:
- Entry: causal per-day threshold from prior scores (quantile-based)
- Exit: deterioration-based exit (hold >=2min, exit if yhat < running_best, force exit at 10min)
- Bankroll: dYdX policy (10 USDC floor, 30% profit siphon, 10% bank recap after liquidation)
- Liquidations: wick-based (long position liquidated if candle low crosses threshold)

Usage:
  python3 scripts/backtest_live_dydx_exact_2026-01-02T20-16-15Z.py \\
    --market-csv data/dydx_BTC-USD_1MIN_2026-01-02T18-50-42Z.csv \\
    --entry-model models/entry_regressor_btcusdt_2025-12-20T13-47-55Z.joblib \\
    --exit-model models/exit_regressor_btcusdt_2025-12-20T14-41-31Z.joblib \\
    --out-dir data/backtests
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import compute_feature_frame


@dataclass
class BankrollState:
    """dYdX live bankroll policy state."""
    trading_equity: float
    bank_equity: float
    
    # Policy parameters (match dYdX config defaults)
    trade_floor_usdc: float = 10.0
    profit_siphon_frac: float = 0.30
    bank_threshold_usdc: float = 280.0
    liquidation_recap_bank_frac: float = 0.10
    fee_side: float = 0.001
    
    # Counters
    n_external_topups: int = 0
    n_bank_recaps: int = 0
    total_external_topup: float = 0.0
    total_bank_recap: float = 0.0


def net_return_pct(entry_px: float, exit_px: float, fee_side: float) -> float:
    """Net return % at 1x (accounts for entry and exit fees)."""
    gross_mult = float(exit_px) / max(1e-12, float(entry_px))
    net_mult = gross_mult * (1.0 - fee_side) / (1.0 + fee_side)
    return float((net_mult - 1.0) * 100.0)


def liquidation_price_long(entry_px: float, fee_side: float, leverage: float) -> float:
    """Liquidation price for a long position (simplified model: equity=0 when net_mult = 1 - 1/L)."""
    if leverage <= 1.0:
        return 0.0
    target_net_mult = 1.0 - 1.0 / float(leverage)
    liq_px = float(entry_px) * (1.0 + fee_side) * target_net_mult / max(1e-12, (1.0 - fee_side))
    return float(liq_px)


def check_wick_liquidation(entry_px: float, low_window: np.ndarray, fee_side: float, leverage: float) -> tuple[bool, int]:
    """Returns (liquidated, rel_min_at_breach). Checks if any candle low <= liq_price."""
    if leverage <= 1.0:
        return False, -1
    liq_px = liquidation_price_long(entry_px, fee_side, leverage)
    breach = np.where(low_window <= liq_px)[0]
    if breach.size == 0:
        return False, -1
    return True, int(breach[0])


def topup_to_floor(bankroll: BankrollState) -> float:
    """Top up trading to floor from external (when bank < threshold). Returns amount transferred."""
    if bankroll.trading_equity >= bankroll.trade_floor_usdc:
        return 0.0
    needed = bankroll.trade_floor_usdc - bankroll.trading_equity
    bankroll.trading_equity += needed
    bankroll.total_external_topup += needed
    bankroll.n_external_topups += 1
    return needed


def siphon_profit_to_bank(bankroll: BankrollState, profit: float) -> float:
    """Siphon a fraction of profit to bank. Returns amount transferred."""
    if profit <= 0.0:
        return 0.0
    amt = profit * bankroll.profit_siphon_frac
    bankroll.trading_equity -= amt
    bankroll.bank_equity += amt
    return amt


def refinance_after_liquidation(bankroll: BankrollState) -> tuple[float, str]:
    """Refinance trading after liquidation. Returns (amount_transferred, source: 'external' or 'bank')."""
    bankroll.trading_equity = 0.0  # liquidation => zero equity
    
    if bankroll.bank_equity < bankroll.bank_threshold_usdc:
        # External topup to floor
        amt = bankroll.trade_floor_usdc
        bankroll.trading_equity = amt
        bankroll.total_external_topup += amt
        bankroll.n_external_topups += 1
        return amt, "external"
    else:
        # Bank recap: transfer 10% of bank to trading
        amt = bankroll.bank_equity * bankroll.liquidation_recap_bank_frac
        amt = min(amt, bankroll.bank_equity)
        bankroll.bank_equity -= amt
        bankroll.trading_equity += amt
        bankroll.total_bank_recap += amt
        bankroll.n_bank_recaps += 1
        return amt, "bank"


def _parse_pre_min(payload: Any) -> int:
    ctx = payload.get("context") if isinstance(payload, dict) else None
    if not isinstance(ctx, dict):
        return 0
    pm = ctx.get("pre_min")
    if pm is None:
        return 0
    try:
        return max(0, int(pm))
    except Exception:
        return 0


def load_models(entry_path: Path, exit_path: Path) -> Dict[str, Any]:
    """Load entry and exit models + extract metadata (same as live runner)."""
    entry_art = joblib.load(entry_path)
    exit_art = joblib.load(exit_path)
    
    if not isinstance(entry_art, dict) or "model" not in entry_art or "features" not in entry_art:
        raise ValueError(f"Unexpected entry model format: {entry_path}")
    if not isinstance(exit_art, dict) or "model" not in exit_art or "features" not in exit_art:
        raise ValueError(f"Unexpected exit model format: {exit_path}")
    
    entry_model = entry_art["model"]
    exit_model = exit_art["model"]
    
    entry_features = list(entry_art.get("features", []))
    exit_features = list(exit_art.get("features", []))
    
    entry_pre_min = int(_parse_pre_min(entry_art))
    exit_pre_min = int(_parse_pre_min(exit_art))
    
    return {
        "entry_model": entry_model,
        "exit_model": exit_model,
        "entry_features": entry_features,
        "exit_features": exit_features,
        "entry_pre_min": entry_pre_min,
        "exit_pre_min": exit_pre_min,
        "entry_artifact": entry_path.name,
        "exit_artifact": exit_path.name,
    }


def simulate_live_logic(
    bars: pd.DataFrame,
    models: Dict[str, Any],
    target_frac: float = 0.001,
    trade_floor_usdc: float = 10.0,
    profit_siphon_frac: float = 0.30,
    bank_threshold_usdc: float = 280.0,
    liquidation_recap_bank_frac: float = 0.10,
    fee_side: float = 0.001,
    max_leverage: int = 0,  # 0 = no cap (live default)
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Simulate the exact live runner logic:
    - Causal per-day entry threshold (quantile of prior scores)
    - Deterioration-based exit (hold >=2min, exit if yhat < running_best, force at 10min)
    - dYdX bankroll policy
    - Wick liquidations at max leverage (50x for dYdX BTC-USD mainnet)
    """
    
    bars = bars.copy().sort_values("timestamp").reset_index(drop=True)
    
    entry_model = models["entry_model"]
    exit_model = models["exit_model"]
    entry_features = models["entry_features"]
    exit_features = models["exit_features"]
    entry_pre_min = models["entry_pre_min"]
    exit_pre_min = models["exit_pre_min"]
    
    # Build feature frame with rolling context (same as live runner)
    all_features = list(set(entry_features) | set(exit_features))
    all_pre_mins = [m for m in [entry_pre_min, exit_pre_min] if m > 0]
    
    # Compute features (requires ts_min column)
    bars_with_ts = bars.rename(columns={"timestamp": "ts_min"})
    base = compute_feature_frame(bars_with_ts)
    src = base[[c for c in compute_feature_frame.__globals__.get("FEATURES", []) if c in base.columns]]
    
    # Add rolling context means
    full = src.copy()
    for m in all_pre_mins:
        ctx_mean = src.rolling(int(m), min_periods=int(m)).mean().add_suffix(f"_mean_{int(m)}m")
        full = pd.concat([full, ctx_mean], axis=1)
    
    # Check for missing columns
    missing_entry = [c for c in entry_features if c not in full.columns]
    missing_exit = [c for c in exit_features if c not in full.columns]
    
    if missing_entry:
        raise ValueError(f"Missing entry features: {missing_entry}")
    if missing_exit:
        raise ValueError(f"Missing exit features: {missing_exit}")
    
    # Build final feature frame
    feats = pd.concat([base[["ts_min"]], full[all_features]], axis=1).rename(columns={"ts_min": "timestamp"})
    
    # Score all minutes (entry model only)
    entry_scores = entry_model.predict(feats[entry_features].to_numpy(dtype=np.float32))
    feats["entry_score"] = entry_scores
    feats["date"] = pd.to_datetime(feats["timestamp"]).dt.date
    
    # Pre-compute exit predictions for all minutes (for exit policy)
    exit_predictions = exit_model.predict(feats[exit_features].to_numpy(dtype=np.float32))
    feats["exit_yhat"] = exit_predictions
    
    # Initialize bankroll
    bankroll = BankrollState(
        trading_equity=trade_floor_usdc,
        bank_equity=0.0,
        trade_floor_usdc=trade_floor_usdc,
        profit_siphon_frac=profit_siphon_frac,
        bank_threshold_usdc=bank_threshold_usdc,
        liquidation_recap_bank_frac=liquidation_recap_bank_frac,
        fee_side=fee_side,
    )
    
    # Causal threshold computation: per day, use quantile of *prior* day scores
    dates = sorted(feats["date"].unique())
    prior_scores: List[float] = []
    thresholds: Dict[Any, float] = {}
    
    for d in dates:
        if len(prior_scores) > 0:
            thresholds[d] = float(np.quantile(prior_scores, 1.0 - target_frac))
        else:
            thresholds[d] = float("-inf")  # first day: accept all (or use a default high threshold)
        
        # Move today's scores to prior after computing threshold
        day_scores = feats[feats["date"] == d]["entry_score"].tolist()
        prior_scores.extend([float(x) for x in day_scores if np.isfinite(float(x))])
    
    feats["entry_threshold"] = feats["date"].map(thresholds)
    
    # Trading loop
    trades: List[Dict[str, Any]] = []
    open_trade: Optional[Dict[str, Any]] = None
    
    # dYdX mainnet BTC-USD has max leverage 50x (IMF ~0.02)
    market_max_lev = 50 if max_leverage <= 0 else max_leverage
    
    ts_arr = bars["timestamp"].to_numpy(dtype="datetime64[ns]")
    open_arr = bars["open"].to_numpy(np.float64)
    low_arr = bars["low"].to_numpy(np.float64)
    close_arr = bars["close"].to_numpy(np.float64)
    
    for i in range(len(feats)):
        ts = feats.iloc[i]["timestamp"]
        d = feats.iloc[i]["date"]
        
        score_i = feats.iloc[i]["entry_score"]
        thr_i = feats.iloc[i]["entry_threshold"]
        
        # Entry logic: if no open trade, check if score > threshold
        if open_trade is None:
            if np.isfinite(score_i) and np.isfinite(thr_i) and score_i > thr_i:
                # Top up to floor before entry
                topup_amt = topup_to_floor(bankroll)
                
                equity_before = bankroll.trading_equity
                
                # Entry at next minute open (i+1)
                if i + 1 < len(bars):
                    entry_idx = i + 1
                    entry_ts = bars.iloc[entry_idx]["timestamp"]
                    entry_px = bars.iloc[entry_idx]["open"]
                    
                    open_trade = {
                        "entry_index": entry_idx,
                        "entry_time": entry_ts,
                        "entry_price": entry_px,
                        "entry_score": score_i,
                        "entry_threshold": thr_i,
                        "equity_before": equity_before,
                        "topup_pre": topup_amt,
                        "running_best": -1e30,
                        "best_k": None,
                    }
            continue
        
        # Progress open trade
        tr = open_trade
        entry_idx = tr["entry_index"]
        k_rel = i - entry_idx
        
        if k_rel <= 0:
            continue
        if k_rel > 10:
            k_rel = 10
        
        # Exit policy: exit NOW if model says to exit (predicts negative return)
        # The exit model predicts expected return from this minute forward
        # If negative: exit now. If positive: hold (but force exit at 10 min)
        exit_signal = False
        yhat = feats.iloc[i]["exit_yhat"]
        pred_at_exit = None
        
        if np.isfinite(yhat):
            pred_at_exit = float(yhat)
            # Decision: should I exit NOW or stay?
            # If model predicts negative return, exit immediately
            if yhat < 0.0 and k_rel >= 1:
                exit_signal = True
            # Track best prediction seen (for fallback)
            if yhat > float(tr.get("running_best", -1e30)):
                tr["running_best"] = yhat
                tr["best_k"] = k_rel
        
        if exit_signal or k_rel >= 10:
            exit_k = k_rel
            exit_idx = i
            exit_ts = bars.iloc[exit_idx]["timestamp"]
            exit_px = bars.iloc[exit_idx]["close"]
            
            entry_px = tr["entry_price"]
            
            # Check for wick liquidation
            low_window = low_arr[entry_idx : exit_idx + 1]
            liq, liq_k = check_wick_liquidation(entry_px, low_window, fee_side, market_max_lev)
            
            liquidated = liq
            liq_source = ""
            topup_post = 0.0
            
            if liquidated:
                realized_ret_pct = -100.0
                profit_usdc = -bankroll.trading_equity
                
                # Refinance after liquidation
                recap_amt, recap_source = refinance_after_liquidation(bankroll)
                topup_post = recap_amt
                liq_source = recap_source
                
                siphon_amt = 0.0
            else:
                realized_ret_pct = net_return_pct(entry_px, exit_px, fee_side)
                profit_usdc = bankroll.trading_equity * (realized_ret_pct / 100.0)
                bankroll.trading_equity += profit_usdc
                
                # Siphon profit to bank
                siphon_amt = siphon_profit_to_bank(bankroll, profit_usdc)
                
                # Top up to floor after trade
                topup_post = topup_to_floor(bankroll)
            
            # Prefer prediction at exit moment
            if pred_at_exit is not None and np.isfinite(pred_at_exit):
                pred = pred_at_exit
            else:
                pred = float(tr["running_best"]) if tr["running_best"] > -1e29 else None
            
            trades.append({
                "entry_time": tr["entry_time"],
                "exit_time": exit_ts,
                "exit_rel_min": exit_k,
                "entry_price": entry_px,
                "exit_price": exit_px,
                "realized_ret_pct": realized_ret_pct,
                "predicted_ret_pct": pred,
                "entry_score": tr["entry_score"],
                "entry_threshold": tr["entry_threshold"],
                "equity_before": tr["equity_before"],
                "profit_usdc": profit_usdc,
                "siphon_usdc": siphon_amt,
                "topup_pre_usdc": tr["topup_pre"],
                "topup_post_usdc": topup_post,
                "trading_equity": bankroll.trading_equity,
                "bank_equity": bankroll.bank_equity,
                "liquidated": liquidated,
                "liquidation_source": liq_source if liquidated else "",
                "date": d,
            })
            
            open_trade = None
    
    if not trades:
        return pd.DataFrame(), pd.DataFrame(), {}
    
    df = pd.DataFrame(trades)
    
    daily = df.groupby("date", as_index=False).agg(
        n_trades=("realized_ret_pct", "size"),
        mean_ret_pct=("realized_ret_pct", "mean"),
        sum_ret_pct=("realized_ret_pct", "sum"),
        sum_profit_usdc=("profit_usdc", "sum"),
        end_trading_equity=("trading_equity", "last"),
        end_bank_equity=("bank_equity", "last"),
        n_liquidations=("liquidated", "sum"),
    )
    
    summary = {
        "n_trades": len(df),
        "n_liquidations": int(df["liquidated"].sum()),
        "n_external_topups": bankroll.n_external_topups,
        "n_bank_recaps": bankroll.n_bank_recaps,
        "total_external_topup_usdc": bankroll.total_external_topup,
        "total_bank_recap_usdc": bankroll.total_bank_recap,
        "final_trading_equity": bankroll.trading_equity,
        "final_bank_equity": bankroll.bank_equity,
        "final_total_equity": bankroll.trading_equity + bankroll.bank_equity,
        "mean_ret_pct": df["realized_ret_pct"].mean(),
        "sum_ret_pct": df["realized_ret_pct"].sum(),
        "sum_profit_usdc": df["profit_usdc"].sum(),
    }
    
    return df, daily, summary


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def main() -> None:
    ap = argparse.ArgumentParser(description="Exact offline backtest of live dYdX runner logic")
    ap.add_argument("--market-csv", required=True, help="CSV with timestamp, open, high, low, close, volume")
    ap.add_argument("--entry-model", default=str(REPO_ROOT / "models" / "entry_regressor_btcusdt_2025-12-20T13-47-55Z.joblib"))
    ap.add_argument("--exit-model", default=str(REPO_ROOT / "models" / "exit_regressor_btcusdt_2025-12-20T14-41-31Z.joblib"))
    ap.add_argument("--out-dir", default="data/backtests")
    
    # Live config defaults (match dYdX adapter)
    ap.add_argument("--target-frac", type=float, default=0.001, help="Entry threshold quantile target (default 0.001 = 0.1% of minutes)")
    ap.add_argument("--trade-floor-usdc", type=float, default=10.0)
    ap.add_argument("--profit-siphon-frac", type=float, default=0.30)
    ap.add_argument("--bank-threshold-usdc", type=float, default=280.0)
    ap.add_argument("--liquidation-recap-bank-frac", type=float, default=0.10)
    ap.add_argument("--fee-side", type=float, default=0.001)
    ap.add_argument("--max-leverage", type=int, default=0, help="Max leverage cap (0=auto/50x for dYdX BTC-USD)")
    
    args = ap.parse_args()
    
    bars = pd.read_csv(args.market_csv, parse_dates=["timestamp"])
    models = load_models(Path(args.entry_model), Path(args.exit_model))
    
    print("Running exact live logic backtest...")
    print(f"  Entry model: {models['entry_artifact']}")
    print(f"  Exit model: {models['exit_artifact']}")
    print(f"  Market data: {args.market_csv} ({len(bars)} bars)")
    print(f"  Date range: {bars['timestamp'].min()} → {bars['timestamp'].max()}")
    print(f"  Trade floor: {args.trade_floor_usdc} USDC")
    print(f"  Profit siphon: {args.profit_siphon_frac * 100}%")
    print(f"  Bank threshold: {args.bank_threshold_usdc} USDC")
    print(f"  Liquidation recap: {args.liquidation_recap_bank_frac * 100}%")
    print(f"  Fee per side: {args.fee_side * 100}%")
    print(f"  Max leverage: {'auto (50x)' if args.max_leverage <= 0 else args.max_leverage}")
    
    trades, daily, summary = simulate_live_logic(
        bars,
        models,
        target_frac=args.target_frac,
        trade_floor_usdc=args.trade_floor_usdc,
        profit_siphon_frac=args.profit_siphon_frac,
        bank_threshold_usdc=args.bank_threshold_usdc,
        liquidation_recap_bank_frac=args.liquidation_recap_bank_frac,
        fee_side=args.fee_side,
        max_leverage=args.max_leverage,
    )
    
    ts = now_ts()
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    trades_path = out_dir / f"backtest_live_exact_trades_{ts}.csv"
    daily_path = out_dir / f"backtest_live_exact_daily_{ts}.csv"
    
    trades.to_csv(trades_path, index=False)
    daily.to_csv(daily_path, index=False)
    
    print("\n=== BACKTEST SUMMARY ===")
    for k in [
        "n_trades",
        "n_liquidations",
        "n_external_topups",
        "n_bank_recaps",
        "total_external_topup_usdc",
        "total_bank_recap_usdc",
        "final_trading_equity",
        "final_bank_equity",
        "final_total_equity",
        "mean_ret_pct",
        "sum_ret_pct",
        "sum_profit_usdc",
    ]:
        print(f"  {k}: {summary.get(k)}")
    
    print(f"\nTrades: {trades_path}")
    print(f"Daily: {daily_path}")


if __name__ == "__main__":
    main()
