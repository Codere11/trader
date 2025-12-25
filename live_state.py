# Timestamp (UTC): 2025-12-22T14:18:15Z
from __future__ import annotations

"""Small persisted state helpers for live trading.

This module is intentionally dependency-light so it can be imported by the live runner
without pulling in exchange-specific code.

It provides:
- GateState: rolling last-N paper trade outcomes and sticky go-live enablement.
- StrategyLedger: a local EUR-denominated capital/bank ledger mirroring DO_NOT_TOUCH.txt.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------
# Go-live gate state
# -----------------


@dataclass
class GateState:
    window: int = 5
    enabled: bool = False
    enabled_utc: Optional[str] = None

    # Rolling window of outcomes for *paper* trades only.
    # True means profitable (realized_ret_pct > 0).
    outcomes: List[bool] = field(default_factory=list)
    trade_entry_times_utc: List[str] = field(default_factory=list)

    # Monotonic count of *completed* paper trades, persisted across restarts.
    paper_trades_total: int = 0

    def record_paper_trade(self, *, entry_time_utc: str, realized_ret_pct: float) -> None:
        win = bool(float(realized_ret_pct) > 0.0)
        self.outcomes.append(win)
        self.trade_entry_times_utc.append(str(entry_time_utc))
        self.paper_trades_total = int(self.paper_trades_total) + 1

        # Keep only the last `window` outcomes for quick inspection.
        if len(self.outcomes) > int(self.window):
            self.outcomes = self.outcomes[-int(self.window) :]
        if len(self.trade_entry_times_utc) > int(self.window):
            self.trade_entry_times_utc = self.trade_entry_times_utc[-int(self.window) :]

    def should_enable_real(self) -> bool:
        # NOTE: this is only evaluated at EOD/day-roll by the live runner.
        if self.enabled:
            return False
        return int(self.paper_trades_total) >= int(self.window)

    def enable_real(self) -> None:
        if not self.enabled:
            self.enabled = True
            self.enabled_utc = utc_now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GateState":
        outcomes = list(d.get("outcomes", []))
        trade_times = list(d.get("trade_entry_times_utc", []))

        paper_total = d.get("paper_trades_total")
        if paper_total is None:
            # Backward-compat: older state only tracked the last `window` trades.
            paper_total = max(len(outcomes), len(trade_times))

        return cls(
            window=int(d.get("window", 5)),
            enabled=bool(d.get("enabled", False)),
            enabled_utc=d.get("enabled_utc"),
            outcomes=outcomes,
            trade_entry_times_utc=trade_times,
            paper_trades_total=int(paper_total),
        )


def load_gate_state(path: Path, *, window: int = 5) -> GateState:
    path = Path(path)
    if not path.exists():
        return GateState(window=int(window))
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        st = GateState.from_dict(obj if isinstance(obj, dict) else {})
        if int(st.window) <= 0:
            st.window = int(window)
        return st
    except Exception:
        # If corrupted, do not crash the bot; start fresh.
        return GateState(window=int(window))


def save_gate_state(state: GateState, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


# -----------------
# Strategy ledger
# -----------------


@dataclass
class StrategyLedger:
    # EUR-denominated internal ledger.
    trading_capital_eur: float = 30.0
    bank_eur: float = 0.0

    # Accounting metadata.
    updated_utc: str = field(default_factory=utc_now_iso)
    n_trades: int = 0

    # Last trade details (for debugging/reporting).
    last_trade_pnl_eur: Optional[float] = None
    last_trade_realized_ret_pct: Optional[float] = None
    last_trade_leverage: Optional[int] = None

    def equity_eur(self) -> float:
        return float(self.trading_capital_eur) + float(self.bank_eur)

    @staticmethod
    def target_leverage_from_equity(equity_eur: float) -> int:
        # Mirrors DO_NOT_TOUCH.txt tier schedule.
        e = float(equity_eur)
        if e < 100_000:
            return 100
        if e < 200_000:
            return 50
        if e < 500_000:
            return 20
        if e < 1_000_000:
            return 10
        return 1

    @staticmethod
    def compute_pnl_eur(*, realized_ret_pct: float, leverage: int, margin_eur: float) -> float:
        lev = int(leverage)
        margin = float(margin_eur)
        ret = float(realized_ret_pct) / 100.0
        return (margin * float(lev)) * ret

    @staticmethod
    def default_bank_delta_eur(*, pnl_eur: float, leverage: int) -> float:
        lev = int(leverage)
        pnl = float(pnl_eur)
        # Profit-taking: siphon half of profits into the bank whenever leverage is > 10x.
        if pnl > 0.0 and lev > 10:
            return 0.5 * pnl
        return 0.0

    def apply_trade_closed(
        self,
        *,
        realized_ret_pct: float,
        leverage: int,
        margin_eur: float,
        bank_override_eur: Optional[float] = None,
    ) -> Dict[str, float]:
        """Apply bookkeeping.

        If bank_override_eur is provided, it is used as the actual banked amount (e.g. the amount
        successfully transferred to a stablecoin wallet), and is clamped to [0, max(pnl_eur, 0)].
        """

        lev = int(leverage)
        pnl_eur = float(self.compute_pnl_eur(realized_ret_pct=float(realized_ret_pct), leverage=int(lev), margin_eur=float(margin_eur)))

        bank_delta = float(self.default_bank_delta_eur(pnl_eur=float(pnl_eur), leverage=int(lev)))
        if bank_override_eur is not None:
            bo = float(bank_override_eur)
            bank_delta = max(0.0, min(bo, max(0.0, pnl_eur)))

        # Apply PnL to trading capital, then siphon bank.
        self.trading_capital_eur = float(self.trading_capital_eur) + pnl_eur - bank_delta
        self.bank_eur = float(self.bank_eur) + bank_delta

        self.n_trades += 1
        self.updated_utc = utc_now_iso()

        self.last_trade_pnl_eur = float(pnl_eur)
        self.last_trade_realized_ret_pct = float(realized_ret_pct)
        self.last_trade_leverage = int(lev)

        # Refinance rules.
        refinance_delta = 0.0
        if float(self.trading_capital_eur) < 5.0:
            if float(self.bank_eur) < 150.0:
                # External top-up to 30 EUR.
                refinance_delta = 30.0 - float(self.trading_capital_eur)
                self.trading_capital_eur = 30.0
            else:
                deposit = 0.2 * float(self.bank_eur)
                self.bank_eur = float(self.bank_eur) - deposit
                refinance_delta = deposit - float(self.trading_capital_eur)
                self.trading_capital_eur = float(deposit)

        return {
            "pnl_eur": float(pnl_eur),
            "bank_delta_eur": float(bank_delta),
            "refinance_delta_eur": float(refinance_delta),
            "capital_post_eur": float(self.trading_capital_eur),
            "bank_post_eur": float(self.bank_eur),
        }

    def on_trade_closed(
        self,
        *,
        realized_ret_pct: float,
        leverage: int,
        margin_eur: float,
    ) -> Dict[str, float]:
        return self.apply_trade_closed(
            realized_ret_pct=float(realized_ret_pct),
            leverage=int(leverage),
            margin_eur=float(margin_eur),
            bank_override_eur=None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyLedger":
        return cls(
            trading_capital_eur=float(d.get("trading_capital_eur", 30.0)),
            bank_eur=float(d.get("bank_eur", 0.0)),
            updated_utc=str(d.get("updated_utc", utc_now_iso())),
            n_trades=int(d.get("n_trades", 0)),
            last_trade_pnl_eur=d.get("last_trade_pnl_eur"),
            last_trade_realized_ret_pct=d.get("last_trade_realized_ret_pct"),
            last_trade_leverage=d.get("last_trade_leverage"),
        )


def load_strategy_ledger(path: Path) -> StrategyLedger:
    path = Path(path)
    if not path.exists():
        return StrategyLedger()
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return StrategyLedger.from_dict(obj if isinstance(obj, dict) else {})
    except Exception:
        return StrategyLedger()


def save_strategy_ledger(ledger: StrategyLedger, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
