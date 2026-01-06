#!/usr/bin/env python3
"""Backtest ETH SELL trades that should have fired today based on scores >= 0.344."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

from binance_adapter.crossref_oracle_vs_agent_patterns import compute_feature_frame
from score_eth_sell_locally import build_ctx_frame, feature_map_at

# SELL strategy parameters (from live script)
HOLD_MIN = 15
FEE_SIDE = 0.0005  # 0.05% per side


def net_mult_sell(entry_px, exit_px, fee_side):
    """SELL: profit when exit_px < entry_px (buy back lower)."""
    entry_px = max(1e-12, float(entry_px))
    exit_px = max(1e-12, float(exit_px))
    gross_mult = entry_px / exit_px
    return float(gross_mult) * (1.0 - float(fee_side)) / (1.0 + float(fee_side))


def net_ret_pct_sell(entry_px, exit_px, fee_side):
    return float((net_mult_sell(entry_px, exit_px, fee_side) - 1.0) * 100.0)


def main():
    # Load data
    data_path = REPO_ROOT / "data" / "eth_combined_for_scoring.csv"
    bars = pd.read_csv(data_path)
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    bars["ts_min"] = pd.to_datetime(bars["timestamp"], utc=True)
    
    # Filter for today only
    today_bars = bars[bars["ts_min"].dt.date == pd.Timestamp("2026-01-06").date()].copy()
    today_start_idx = today_bars.index[0]
    
    print(f"Today data: {len(today_bars)} bars from {today_bars['ts_min'].min()} to {today_bars['ts_min'].max()}")
    
    # Load model
    model_path = REPO_ROOT / "data" / "entry_regressor_sell_oracle15m_ctx120_weighted_oracleentry5m_2026-01-04T22-45-20Z.joblib"
    model_obj = joblib.load(model_path)
    model = model_obj["model"]
    feature_cols = model_obj["feature_cols"]
    
    # Build features for full dataset
    base_full = compute_feature_frame(bars)
    base = base_full[["ret_1m_pct", "mom_3m_pct", "mom_5m_pct", "vol_std_5m", "range_5m", "range_norm_5m", "macd", "vwap_dev_5m"]].copy().reset_index(drop=True)
    ctx = build_ctx_frame(bars).reset_index(drop=True)
    
    # Score all bars and find entries >= 0.344
    threshold = 0.344
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
                entries.append({
                    "entry_idx": idx,
                    "entry_time": bars.loc[idx, "ts_min"],
                    "entry_price": float(bars.loc[idx, "close"]),
                    "entry_score": s
                })
        except:
            pass
    
    print(f"\n{'='*80}")
    print(f"Found {len(entries)} entry signals today (score >= {threshold})")
    print(f"{'='*80}\n")
    
    if len(entries) == 0:
        print("No trades to backtest.")
        return
    
    # Backtest each trade: SELL for 15 minutes
    trades = []
    for entry in entries:
        entry_idx = entry["entry_idx"]
        entry_time = entry["entry_time"]
        entry_price = entry["entry_price"]
        entry_score = entry["entry_score"]
        
        # Exit at bar[entry_idx + HOLD_MIN] (15 minutes later)
        exit_idx = entry_idx + HOLD_MIN
        
        if exit_idx >= len(bars):
            # Not enough data to hold for full duration
            continue
        
        exit_time = bars.loc[exit_idx, "ts_min"]
        exit_price = float(bars.loc[exit_idx, "close"])
        
        # Calculate SELL return: profit when price drops
        ret_pct = net_ret_pct_sell(entry_price, exit_price, FEE_SIDE)
        
        trades.append({
            "entry_time": entry_time,
            "entry_price": entry_price,
            "entry_score": entry_score,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "ret_pct": ret_pct,
            "hold_min": HOLD_MIN
        })
    
    if len(trades) == 0:
        print("No completed trades (not enough bars for 15-min hold).")
        return
    
    # Statistics
    df = pd.DataFrame(trades)
    total_trades = len(df)
    winners = df[df["ret_pct"] > 0]
    losers = df[df["ret_pct"] <= 0]
    
    win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
    avg_ret = df["ret_pct"].mean()
    median_ret = df["ret_pct"].median()
    total_ret = df["ret_pct"].sum()
    
    print(f"BACKTEST RESULTS (15-min hold, 0.05% fees per side)")
    print(f"{'='*80}")
    print(f"Total trades:     {total_trades}")
    print(f"Winners:          {len(winners)} ({len(winners)/total_trades*100:.1f}%)")
    print(f"Losers:           {len(losers)} ({len(losers)/total_trades*100:.1f}%)")
    print(f"Win rate:         {win_rate:.2f}%")
    print(f"Average return:   {avg_ret:.4f}%")
    print(f"Median return:    {median_ret:.4f}%")
    print(f"Total return:     {total_ret:.4f}%")
    print(f"Best trade:       {df['ret_pct'].max():.4f}%")
    print(f"Worst trade:      {df['ret_pct'].min():.4f}%")
    
    if len(winners) > 0:
        print(f"\nWinner stats:")
        print(f"  Avg win:        {winners['ret_pct'].mean():.4f}%")
        print(f"  Median win:     {winners['ret_pct'].median():.4f}%")
    
    if len(losers) > 0:
        print(f"\nLoser stats:")
        print(f"  Avg loss:       {losers['ret_pct'].mean():.4f}%")
        print(f"  Median loss:    {losers['ret_pct'].median():.4f}%")
    
    print(f"\n{'='*80}")
    print(f"ALL TRADES (chronological):")
    print(f"{'='*80}")
    for i, row in df.iterrows():
        pnl_str = f"+{row['ret_pct']:.4f}%" if row['ret_pct'] > 0 else f"{row['ret_pct']:.4f}%"
        print(f"{i+1:3d}. {row['entry_time']} | Score: {row['entry_score']:.6f} | "
              f"Entry: ${row['entry_price']:.2f} → Exit: ${row['exit_price']:.2f} | PnL: {pnl_str}")


if __name__ == "__main__":
    main()
