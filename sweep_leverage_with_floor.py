#!/usr/bin/env python3
"""Sweep leverage and threshold with $10 floor topup from bank."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import compute_feature_frame
from score_eth_sell_locally import build_ctx_frame, feature_map_at

HOLD_MIN = 15
FEE_SIDE = 0.0005
FLOOR = 10.0  # Floor topup amount


def net_mult_sell(entry_px, exit_px, fee_side):
    entry_px = max(1e-12, float(entry_px))
    exit_px = max(1e-12, float(exit_px))
    gross_mult = entry_px / exit_px
    return float(gross_mult) * (1.0 - float(fee_side)) / (1.0 + float(fee_side))


def net_ret_pct_sell(entry_px, exit_px, fee_side):
    return float((net_mult_sell(entry_px, exit_px, fee_side) - 1.0) * 100.0)


def backtest_with_floor(bars, base, ctx, model, feature_cols, threshold, leverage):
    """Backtest with $10 floor topups and compounding."""
    entries = []
    
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
                entries.append(idx)
        except:
            pass
    
    if len(entries) == 0:
        return None
    
    # Backtest with floor topups
    capital = 10.0
    bank_spent = 0.0  # Track how much bank spent on topups
    max_capital = capital
    trades = []
    
    for entry_idx in entries:
        exit_idx = entry_idx + HOLD_MIN
        
        if exit_idx >= len(bars):
            continue
        
        # Top up to floor if needed BEFORE trade
        if capital < FLOOR:
            topup = FLOOR - capital
            bank_spent += topup
            capital = FLOOR
        
        entry_price = float(bars.loc[entry_idx, "close"])
        exit_price = float(bars.loc[exit_idx, "close"])
        
        ret_pct = net_ret_pct_sell(entry_price, exit_price, FEE_SIDE)
        leveraged_ret = ret_pct * leverage / 100.0
        
        # Check for liquidation (lose more than 100%)
        if leveraged_ret <= -1.0:
            # LIQUIDATED - lose all capital
            capital = 0.0
            trades.append(ret_pct)
            break  # Stop trading after liquidation
        
        new_capital = capital * (1 + leveraged_ret)
        trades.append(ret_pct)
        capital = new_capital
        max_capital = max(max_capital, capital)
    
    if len(trades) == 0:
        return None
    
    # Final state
    liquidated = (capital == 0.0)
    final_capital = capital
    net_profit = final_capital - 10.0 - bank_spent  # Subtract initial $10 and bank topups
    
    trades_arr = np.array(trades)
    return {
        "liquidated": liquidated,
        "threshold": threshold,
        "leverage": leverage,
        "n_trades": len(trades),
        "final_capital": final_capital,
        "bank_spent": bank_spent,
        "net_profit": net_profit,  # Profit after subtracting bank topups
        "gross_profit": final_capital - 10.0,  # Profit ignoring topups
        "return_pct": (final_capital / (10.0 + bank_spent) - 1.0) * 100.0,
        "max_capital": max_capital,
        "win_rate": (trades_arr > 0).sum() / len(trades_arr) * 100,
        "avg_ret": trades_arr.mean(),
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
    
    print(f"Data loaded: {len(bars)} bars\n")
    
    # Sweep parameters
    leverages = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    thresholds = np.arange(0.30, 0.51, 0.01)
    
    print(f"Sweeping {len(leverages)} leverages × {len(thresholds)} thresholds = {len(leverages)*len(thresholds)} combinations...")
    print(f"With $10 floor topups from bank before each trade\n")
    
    results = []
    for leverage in leverages:
        for threshold in thresholds:
            result = backtest_with_floor(bars, base, ctx, model, feature_cols, threshold, leverage)
            if result is not None:
                results.append(result)
    
    df = pd.DataFrame(results)
    
    # Remove liquidations
    df_alive = df[~df["liquidated"]].copy()
    
    if len(df_alive) == 0:
        print("ALL COMBINATIONS RESULTED IN LIQUIDATION!")
        return
    
    # Find best by NET profit (after bank topups)
    best_idx = df_alive["net_profit"].idxmax()
    best = df_alive.loc[best_idx]
    
    print(f"{'='*100}")
    print(f"BEST COMBINATION (by net profit after bank topups):")
    print(f"{'='*100}")
    print(f"Leverage:        {int(best['leverage'])}x")
    print(f"Threshold:       {best['threshold']:.4f}")
    print(f"Final capital:   ${best['final_capital']:.2f}")
    print(f"Bank spent:      ${best['bank_spent']:.2f}")
    print(f"Gross profit:    ${best['gross_profit']:.2f}")
    print(f"Net profit:      ${best['net_profit']:.2f}")
    print(f"Return:          {best['return_pct']:.2f}%")
    print(f"Trades:          {int(best['n_trades'])}")
    print(f"Win rate:        {best['win_rate']:.2f}%")
    print(f"Max capital:     ${best['max_capital']:.2f}")
    
    # Top 20 by net profit
    print(f"\n{'='*100}")
    print(f"TOP 20 COMBINATIONS BY NET PROFIT:")
    print(f"{'='*100}")
    print(f"{'Rank':<6} {'Lev':<5} {'Thr':<8} {'NetProfit':<11} {'BankSpent':<11} {'Final$':<10} {'Trades':<8} {'WinRate%':<10}")
    print(f"{'-'*100}")
    
    top20 = df_alive.nlargest(20, "net_profit")
    for rank, (_, row) in enumerate(top20.iterrows(), 1):
        print(f"{rank:<6} {int(row['leverage']):<5} {row['threshold']:<8.3f} ${row['net_profit']:<10.2f} "
              f"${row['bank_spent']:<10.2f} ${row['final_capital']:<9.2f} {int(row['n_trades']):<8} {row['win_rate']:<10.2f}")
    
    # Best for each leverage
    print(f"\n{'='*100}")
    print(f"BEST THRESHOLD FOR EACH LEVERAGE:")
    print(f"{'='*100}")
    print(f"{'Lev':<6} {'Threshold':<11} {'NetProfit':<12} {'BankSpent':<12} {'Final$':<11} {'Trades':<9} {'WinRate%':<10}")
    print(f"{'-'*100}")
    
    for lev in sorted(df_alive["leverage"].unique()):
        lev_df = df_alive[df_alive["leverage"] == lev]
        if len(lev_df) == 0:
            continue
        best_lev = lev_df.loc[lev_df["net_profit"].idxmax()]
        print(f"{int(lev):<6} {best_lev['threshold']:<11.3f} ${best_lev['net_profit']:<11.2f} "
              f"${best_lev['bank_spent']:<11.2f} ${best_lev['final_capital']:<10.2f} {int(best_lev['n_trades']):<9} {best_lev['win_rate']:<10.2f}")
    
    # Liquidations
    n_liq = (df["liquidated"] == True).sum()
    print(f"\n{'='*100}")
    print(f"LIQUIDATIONS: {n_liq}/{len(df)} combinations ({n_liq/len(df)*100:.1f}%)")
    if n_liq > 0:
        print(f"Liquidated leverages: {sorted(df[df['liquidated']]['leverage'].unique())}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
