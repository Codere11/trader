#!/usr/bin/env python3
"""Find optimal threshold for max profit on 50x leverage for ETH SELL."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import compute_feature_frame
from score_eth_sell_locally import build_ctx_frame, feature_map_at

# SELL strategy parameters
HOLD_MIN = 15
FEE_SIDE = 0.0005  # 0.05% per side
LEVERAGE = 50  # 50x leverage


def net_mult_sell(entry_px, exit_px, fee_side):
    """SELL: profit when exit_px < entry_px (buy back lower)."""
    entry_px = max(1e-12, float(entry_px))
    exit_px = max(1e-12, float(exit_px))
    gross_mult = entry_px / exit_px
    return float(gross_mult) * (1.0 - float(fee_side)) / (1.0 + float(fee_side))


def net_ret_pct_sell(entry_px, exit_px, fee_side):
    return float((net_mult_sell(entry_px, exit_px, fee_side) - 1.0) * 100.0)


def backtest_threshold(bars, base, ctx, model, feature_cols, threshold):
    """Backtest with given threshold and return profit metrics."""
    entries = []
    
    # Find all entries >= threshold for today
    for idx in range(len(bars)):
        if bars.loc[idx, "ts_min"].date() != pd.Timestamp("2026-01-06").date():
            continue
            
        feat = feature_map_at(idx, bars, base, ctx)
        if not feat:
            continue
        
        x = np.asarray([float(feat.get(c, float("nan"))) for c in feature_cols], dtype=np.float32)
        try:
            s = float(model.predict(np.asarray([x], dtype=np.float32))[0])
            if np.isfinite(s) and s >= threshold:
                entries.append({
                    "entry_idx": idx,
                    "entry_time": bars.loc[idx, "ts_min"],
                    "entry_price": float(bars.loc[idx, "close"]),
                    "entry_score": s
                })
        except:
            pass
    
    if len(entries) == 0:
        return {
            "threshold": threshold,
            "n_trades": 0,
            "total_ret_pct": 0.0,
            "total_profit_usd": 0.0,
            "win_rate": 0.0,
            "avg_ret": 0.0
        }
    
    # Backtest each trade
    trades = []
    for entry in entries:
        entry_idx = entry["entry_idx"]
        exit_idx = entry_idx + HOLD_MIN
        
        if exit_idx >= len(bars):
            continue
        
        entry_price = entry["entry_price"]
        exit_price = float(bars.loc[exit_idx, "close"])
        
        ret_pct = net_ret_pct_sell(entry_price, exit_price, FEE_SIDE)
        trades.append(ret_pct)
    
    if len(trades) == 0:
        return {
            "threshold": threshold,
            "n_trades": 0,
            "total_ret_pct": 0.0,
            "total_profit_usd": 0.0,
            "win_rate": 0.0,
            "avg_ret": 0.0
        }
    
    trades_arr = np.array(trades)
    
    # COMPOUNDING with leverage!
    # Each trade: capital_new = capital_old * (1 + return_pct/100 * leverage)
    capital = 10.0  # Starting capital $10 USDC
    for ret_pct in trades_arr:
        leverage_return = ret_pct / 100.0 * LEVERAGE  # Leverage multiplies the return
        capital = capital * (1 + leverage_return)
    
    final_capital = capital
    total_profit_usd = final_capital - 10.0
    total_ret_pct = (final_capital / 10.0 - 1.0) * 100.0
    
    win_rate = (trades_arr > 0).sum() / len(trades_arr) * 100
    avg_ret = trades_arr.mean()
    
    return {
        "threshold": threshold,
        "n_trades": len(trades),
        "total_ret_pct": total_ret_pct,
        "total_profit_usd": total_profit_usd,
        "win_rate": win_rate,
        "avg_ret": avg_ret,
        "best": trades_arr.max(),
        "worst": trades_arr.min(),
        "median": np.median(trades_arr)
    }


def main():
    print("Loading data and model...")
    
    # Load data
    data_path = REPO_ROOT / "data" / "eth_combined_for_scoring.csv"
    bars = pd.read_csv(data_path)
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    bars["ts_min"] = pd.to_datetime(bars["timestamp"], utc=True)
    
    # Load model
    model_path = REPO_ROOT / "data" / "entry_regressor_sell_oracle15m_ctx120_weighted_oracleentry5m_2026-01-04T22-45-20Z.joblib"
    model_obj = joblib.load(model_path)
    model = model_obj["model"]
    feature_cols = model_obj["feature_cols"]
    
    # Build features
    base_full = compute_feature_frame(bars)
    base = base_full[["ret_1m_pct", "mom_3m_pct", "mom_5m_pct", "vol_std_5m", "range_5m", "range_norm_5m", "macd", "vwap_dev_5m"]].copy().reset_index(drop=True)
    ctx = build_ctx_frame(bars).reset_index(drop=True)
    
    print(f"Data loaded: {len(bars)} bars")
    print(f"Optimizing threshold for 50x leverage on $10 capital...\n")
    
    # Get all scores for today to determine range
    all_scores = []
    for idx in range(len(bars)):
        if bars.loc[idx, "ts_min"].date() != pd.Timestamp("2026-01-06").date():
            continue
            
        feat = feature_map_at(idx, bars, base, ctx)
        if not feat:
            continue
        
        x = np.asarray([float(feat.get(c, float("nan"))) for c in feature_cols], dtype=np.float32)
        try:
            s = float(model.predict(np.asarray([x], dtype=np.float32))[0])
            if np.isfinite(s):
                all_scores.append(s)
        except:
            pass
    
    all_scores = np.array(all_scores)
    print(f"Today's score range: [{all_scores.min():.6f}, {all_scores.max():.6f}]")
    print(f"Score statistics: mean={all_scores.mean():.6f}, median={np.median(all_scores):.6f}, std={all_scores.std():.6f}\n")
    
    # Test thresholds from 0.2 to 0.7 in increments
    thresholds = np.arange(0.20, 0.71, 0.01)
    
    results = []
    for thr in thresholds:
        result = backtest_threshold(bars, base, ctx, model, feature_cols, thr)
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # Find best by total profit
    best_idx = df["total_profit_usd"].idxmax()
    best = df.loc[best_idx]
    
    print(f"{'='*100}")
    print(f"OPTIMIZATION RESULTS (50x leverage, $10 capital, 15-min hold)")
    print(f"{'='*100}\n")
    
    print(f"BEST THRESHOLD: {best['threshold']:.4f}")
    print(f"  Total profit:      ${best['total_profit_usd']:.2f}")
    print(f"  Number of trades:  {int(best['n_trades'])}")
    print(f"  Total return:      {best['total_ret_pct']:.4f}%")
    print(f"  Win rate:          {best['win_rate']:.2f}%")
    print(f"  Avg return/trade:  {best['avg_ret']:.4f}%")
    print(f"  Best trade:        {best['best']:.4f}%")
    print(f"  Worst trade:       {best['worst']:.4f}%")
    print(f"  Median:            {best['median']:.4f}%")
    
    # Show top 10 thresholds by profit
    print(f"\n{'='*100}")
    print(f"TOP 10 THRESHOLDS BY PROFIT:")
    print(f"{'='*100}")
    print(f"{'Rank':<6} {'Threshold':<12} {'Profit':<12} {'Trades':<8} {'Total Ret%':<12} {'Win Rate%':<12} {'Avg Ret%':<12}")
    print(f"{'-'*100}")
    
    top10 = df.nlargest(10, "total_profit_usd")
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"{rank:<6} {row['threshold']:<12.4f} ${row['total_profit_usd']:<11.2f} {int(row['n_trades']):<8} "
              f"{row['total_ret_pct']:<12.4f} {row['win_rate']:<12.2f} {row['avg_ret']:<12.4f}")
    
    # Show profit vs threshold chart (subset)
    print(f"\n{'='*100}")
    print(f"PROFIT vs THRESHOLD (every 0.05):")
    print(f"{'='*100}")
    print(f"{'Threshold':<12} {'Profit':<12} {'Trades':<8} {'Win Rate%':<12} {'Avg Ret%':<12}")
    print(f"{'-'*100}")
    
    for thr in np.arange(0.20, 0.71, 0.05):
        row = df[df["threshold"] == round(thr, 2)].iloc[0] if len(df[df["threshold"] == round(thr, 2)]) > 0 else None
        if row is not None:
            print(f"{thr:<12.2f} ${row['total_profit_usd']:<11.2f} {int(row['n_trades']):<8} "
                  f"{row['win_rate']:<12.2f} {row['avg_ret']:<12.4f}")
    
    # Show current threshold performance
    current_thr = 0.344
    current = df[abs(df["threshold"] - current_thr) < 0.001].iloc[0] if len(df[abs(df["threshold"] - current_thr) < 0.001]) > 0 else None
    
    if current is not None:
        print(f"\n{'='*100}")
        print(f"CURRENT THRESHOLD (0.344) PERFORMANCE:")
        print(f"{'='*100}")
        print(f"  Total profit:      ${current['total_profit_usd']:.2f}")
        print(f"  Number of trades:  {int(current['n_trades'])}")
        print(f"  Total return:      {current['total_ret_pct']:.4f}%")
        print(f"  Win rate:          {current['win_rate']:.2f}%")
        print(f"  Avg return/trade:  {current['avg_ret']:.4f}%")
        
        improvement = best['total_profit_usd'] - current['total_profit_usd']
        print(f"\n  POTENTIAL IMPROVEMENT: ${improvement:.2f} (+{improvement/current['total_profit_usd']*100:.1f}%)")


if __name__ == "__main__":
    main()
