# ETH SELL Topup & Profit Siphoning Fix
**Created: 2026-01-08T07:47:09Z**

## Problems Fixed

### 1. **Topup Budget Hardcoded to Zero**
**Location:** Line 754 (old script)
```python
self.initial_topup_budget_usdc = float(getattr(self.cfg, "trade_floor_usdc", 10.0)) * 0.0  # informational only
```
**Issue:** Topup budget was multiplied by 0.0, effectively disabling topups.

**Fix:** Changed to 5x the floor:
```python
self.initial_topup_budget_usdc = float(getattr(self.cfg, "trade_floor_usdc", 10.0)) * 5.0  # 5x floor = 50 USDC budget
```

---

### 2. **Quantization Rounding to Zero**
**Issue:** When `trading_equity` was ~9.999999 and `floor` was 10.0, the needed amount (~1e-6 USDC) rounded to zero quantums and got skipped.

**Fix:** Added `MIN_TRANSFER_USDC = 0.01` and round up logic:
```python
# Round up to MIN_TRANSFER_USDC to avoid quantization rounding to zero
if 0 < transfer < MIN_TRANSFER_USDC:
    transfer = MIN_TRANSFER_USDC
    print(f"_ensure_trading_floor: rounded transfer up to MIN_TRANSFER_USDC={MIN_TRANSFER_USDC}")
```

Same logic applied to profit siphoning:
```python
# Round up to MIN_TRANSFER_USDC to avoid quantization rounding to zero
if 0 < requested < MIN_TRANSFER_USDC:
    requested = MIN_TRANSFER_USDC
    print(f"_siphon_profit_and_topup: rounded siphon up to MIN_TRANSFER_USDC={MIN_TRANSFER_USDC}")
```

---

### 3. **No Budget Check in Topup Logic**
**Issue:** The function would attempt topups even if budget was exhausted.

**Fix:** Added budget tracking and enforcement:
```python
budget_remaining = float(self.initial_topup_budget_usdc) - float(self.initial_topup_spent_usdc)
if budget_remaining <= 0.0:
    print(f"_ensure_trading_floor: topup budget exhausted (spent={self.initial_topup_spent_usdc:.2f}/{self.initial_topup_budget_usdc:.2f})")
    return {
        "topped_up": False,
        "needed_usdc": float(needed),
        "transferred_usdc": 0.0,
        "skipped": True,
        "skip_reason": "budget_exhausted",
        ...
    }
```

---

### 4. **Silent Failures in Siphon/Topup**
**Issue:** Exceptions were caught but not logged, making debugging impossible.

**Fix:** Added explicit logging and tracebacks:
```python
except Exception as e:
    print(f"_ensure_trading_floor: failed to get trading equity: {e}")
```
```python
except Exception as e:
    print("bank_ops_failed:", e)
    import traceback
    traceback.print_exc()
```

---

### 5. **Zero-Quantum Transfer Detection**
**Issue:** Even after computing a transfer amount, if it rounded to zero quantums, the transfer would fail silently.

**Fix:** Added explicit check and logging:
```python
amt_q = usdc_to_quantums(float(transfer))
if amt_q <= 0:
    print(f"_ensure_trading_floor: transfer amount rounded to zero quantums (transfer={transfer:.6f})")
    return {
        "topped_up": False,
        "needed_usdc": float(needed),
        "transferred_usdc": 0.0,
        "skipped": True,
        "skip_reason": "transfer_amount_rounded_to_zero",
        ...
    }
```

---

## Changes Summary

### Files Modified
1. `scripts/live_dydx_v4_eth_usd_sell_simple_2026-01-08T07-47-09Z.py` (new timestamped copy with fixes)
2. `docker-compose.yml` (updated to use new script)

### Key Additions
- `MIN_TRANSFER_USDC = 0.01` constant
- Budget tracking and enforcement in `_ensure_trading_floor()`
- Rounding guards for tiny transfer amounts (topup and siphon)
- Zero-quantum detection before transfer attempts
- Comprehensive logging for all failure paths
- Exception logging with full tracebacks

### Logging Improvements
All topup/siphon operations now log:
- When transfers are rounded up to minimum
- When budget is exhausted
- When transfers round to zero quantums
- When equity fetch fails
- Success messages with amounts and budget status

---

## Testing Checklist

### Before Deploying to Cloud
- [x] Code changes applied and saved
- [x] Timestamped copy created
- [x] docker-compose.yml updated
- [ ] Commit changes to git
- [ ] Rebuild Docker image
- [ ] Deploy to cloud VM
- [ ] Monitor logs for topup/siphon messages

### After Deployment
- [ ] Verify topup budget = 50 USDC in logs
- [ ] Confirm topups execute when trading_equity < floor
- [ ] Confirm profit siphoning executes after profitable exits
- [ ] Check bank_proof.jsonl for transfer records (if implemented)
- [ ] Verify no "rounded to zero quantums" messages in logs

---

## Expected Behavior

### Topup Flow (Entry)
1. Before entry: `_ensure_trading_floor()` called
2. If `trading_eq < floor` and `budget_remaining > 0`:
   - Transfer `min(needed, bank_eq, budget_remaining)` from bank→trading
   - Round up to 0.01 USDC if needed
3. Log: `"_ensure_trading_floor: topped up X.XX USDC (spent=Y.YY/50.00)"`

### Siphon Flow (Exit)
1. After exit: `_siphon_profit_and_topup()` called
2. If `profit > 0`:
   - Calculate `requested = profit * 0.30` (30% siphon)
   - Round up to 0.01 USDC if needed
   - Transfer from trading→bank
3. Log: `"_siphon_profit_and_topup: siphoned X.XX USDC (profit=Y.YY, frac=0.3)"`
4. Then call `_ensure_trading_floor()` to top up if needed

---

## Deployment Commands

### On Local Machine (prepare)
```bash
cd /home/maksich/Documents/plus500-autotrader
git add scripts/live_dydx_v4_eth_usd_sell_simple_2026-01-08T07-47-09Z.py docker-compose.yml
git commit -m "Fix ETH sell topup budget (0.0→50 USDC) and quantization rounding"
git push
```

### On Cloud VM (deploy)
```bash
cd /path/to/plus500-autotrader
git pull
docker compose build --no-cache dydx_eth_sell
docker compose up -d dydx_eth_sell
docker compose logs -f dydx_eth_sell | grep -E "topup|siphon|bank_ops"
```

---

## Rollback Plan
If issues occur:
```bash
# Revert docker-compose.yml to old script
docker compose down dydx_eth_sell
# Edit docker-compose.yml: change script back to 2026-01-06T22-16-46Z.py
docker compose up -d dydx_eth_sell
```
