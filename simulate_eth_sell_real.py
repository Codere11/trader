#!/usr/bin/env python3
"""Simulate ETH SELL with real trading logic: floor topups + 30% profit siphon."""

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
LEVERAGE = 50
THRESHOLD = 0.39
FLOOR = 10.0
SIPHON_RATE = 0.30  # 30% of profit goes to bank


def net_mult_sell(entry_px, exit_px, fee_side):
    entry_px = max(1e-12, float(entry_px))
    exit_px = max(1e-12, float(exit_px))
    gross_mult = entry_px / exit_px
    return float(gross_mult) * (1.0 - float(fee_side)) / (1.0 + float(fee_side))


def net_ret_pct_sell(entry_px, exit_px, fee_side):
    return float((net_mult_sell(entry_px, exit_px, fee_side) - 1.0) * 100.0)


# Load data
bars = pd.read_csv(REPO_ROOT / "data" / "eth_combined_for_scoring.csv")
bars = bars.sort_values("timestamp").reset_index(drop=True)
bars["ts_min"] = pd.to_datetime(bars["timestamp"], utc=True)

# Load model
model_obj = joblib.load(REPO_ROOT / "data" / "entry_regressor_sell_oracle15m_ctx120_weighted_oracleentry5m_2026-01-04T22-45-20Z.joblib")
model = model_obj["model"]
feature_cols = model_obj["feature_cols"]

# Build features
base_full = compute_feature_frame(bars)
base = base_full[["ret_1m_pct", "mom_3m_pct", "mom_5m_pct", "vol_std_5m", "range_5m", "range_norm_5m", "macd", "vwap_dev_5m"]].copy().reset_index(drop=True)
ctx = build_ctx_frame(bars).reset_index(drop=True)

# Find entries at threshold
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
        if np.isfinite(s) and s >= THRESHOLD:
            entries.append(idx)
    except:
        pass

print(f"Found {len(entries)} entry signals at threshold {THRESHOLD}")
print(f"Simulating with:")
print(f"  - 50x leverage")
print(f"  - $10 floor topups from bank")
print(f"  - 30% profit siphon to bank after each trade")
print(f"  - Rest compounds in trading account")
print()

# Simulate
trading_capital = 10.0
bank_balance = 0.0
bank_topups_total = 0.0

print(f"{'Trade':<7} {'Entry$':<10} {'Ret%':<10} {'LevRet%':<10} {'Exit$':<10} {'Profit$':<10} {'Siphon$':<10} {'Trading$':<10} {'Bank$':<10}")
print("-" * 100)

for i, entry_idx in enumerate(entries, 1):
    exit_idx = entry_idx + HOLD_MIN
    if exit_idx >= len(bars):
        continue
    
    # Top up to floor if needed
    if trading_capital < FLOOR:
        topup = FLOOR - trading_capital
        bank_topups_total += topup
        trading_capital = FLOOR
    
    entry_capital = trading_capital
    entry_price = float(bars.loc[entry_idx, "close"])
    exit_price = float(bars.loc[exit_idx, "close"])
    
    ret_pct = net_ret_pct_sell(entry_price, exit_price, FEE_SIDE)
    leveraged_ret = ret_pct * LEVERAGE / 100.0
    
    # Check liquidation
    if leveraged_ret <= -1.0:
        print(f"{i:<7} ${entry_capital:<9.2f} {ret_pct:>+9.4f} {leveraged_ret*100:>+9.2f} LIQUIDATED!")
        trading_capital = 0.0
        break
    
    exit_capital = entry_capital * (1 + leveraged_ret)
    profit = exit_capital - entry_capital
    
    # Siphon 30% of profit to bank (only if profitable)
    if profit > 0:
        siphon = profit * SIPHON_RATE
        bank_balance += siphon
        trading_capital = exit_capital - siphon
    else:
        siphon = 0.0
        trading_capital = exit_capital
    
    print(f"{i:<7} ${entry_capital:<9.2f} {ret_pct:>+9.4f} {leveraged_ret*100:>+9.2f} ${exit_capital:<9.2f} ${profit:>+9.2f} ${siphon:>9.2f} ${trading_capital:<9.2f} ${bank_balance:<9.2f}")

print()
print("=" * 100)
print(f"FINAL STATE:")
print(f"  Trading account: ${trading_capital:.2f}")
print(f"  Bank balance:    ${bank_balance:.2f}")
print(f"  Bank topups:     ${bank_topups_total:.2f}")
print(f"  Total assets:    ${trading_capital + bank_balance:.2f}")
print(f"  Net profit:      ${trading_capital + bank_balance - 10.0 - bank_topups_total:.2f}")
print(f"  ROI:             {(trading_capital + bank_balance) / (10.0 + bank_topups_total) * 100 - 100:.2f}%")
print("=" * 100)
