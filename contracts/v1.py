# Timestamp (UTC): 2025-12-22T09:41:32Z
from __future__ import annotations

"""Data contracts (v1) + runtime validators.

Goals:
- Freeze the live/replay interface between data ingestion → features → model outputs → trade records.
- Make audit logs reproducible: every event includes a schema version and a run_id.
- Keep dependencies minimal (no pydantic/jsonschema needed).

Conventions (v1):
- All timestamps are ISO8601 strings with explicit UTC offset (e.g. "2025-12-22T09:41:32+00:00") or a trailing "Z".
- Prices are floats in quote currency per 1 unit of base (e.g. BTCUSDT price in USDT).
- Returns are percentages (pct), not decimals.
"""

from datetime import datetime, timedelta
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional


SCHEMA_VERSION_V1 = "v1"

KIND_RUN_META = "run_meta"
KIND_MARKET_BAR_1M_CLOSED = "market_bar_1m_closed"
KIND_ENTRY_DECISION = "entry_decision"
KIND_TRADE_CLOSED = "trade_closed"
KIND_TRADE_MODE_SWITCH = "trade_mode_switch"

ALL_KINDS_V1 = {
    KIND_RUN_META,
    KIND_MARKET_BAR_1M_CLOSED,
    KIND_ENTRY_DECISION,
    KIND_TRADE_CLOSED,
    KIND_TRADE_MODE_SWITCH,
}


class ContractValidationError(ValueError):
    pass


def _ensure(cond: bool, msg: str) -> None:
    if not cond:
        raise ContractValidationError(msg)


def _is_str(x: Any) -> bool:
    return isinstance(x, str)


def _is_bool(x: Any) -> bool:
    return isinstance(x, bool)


def _is_int(x: Any) -> bool:
    # bool is an int subclass; exclude it.
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return True
    if isinstance(x, float):
        return math.isfinite(x)
    return False


def _is_number_or_null(x: Any) -> bool:
    return x is None or _is_number(x)


def _require_keys(obj: Mapping[str, Any], keys: Iterable[str]) -> None:
    missing = [k for k in keys if k not in obj]
    _ensure(not missing, f"Missing required keys: {missing}")


def _parse_utc(ts: str) -> datetime:
    # Accept both "+00:00" and trailing "Z".
    ts2 = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
    try:
        dt = datetime.fromisoformat(ts2)
    except Exception as e:
        raise ContractValidationError(f"Invalid ISO8601 timestamp: {ts!r}") from e
    _ensure(dt.tzinfo is not None, f"Timestamp must include timezone offset: {ts!r}")
    _ensure(dt.utcoffset() == timedelta(0), f"Timestamp must be UTC: {ts!r}")
    return dt


def _validate_common(obj: Mapping[str, Any]) -> None:
    _require_keys(obj, ["schema_version", "kind", "run_id", "event_time_utc"])
    _ensure(obj["schema_version"] == SCHEMA_VERSION_V1, "schema_version must be 'v1'")
    _ensure(_is_str(obj["kind"]) and obj["kind"] in ALL_KINDS_V1, "Unknown/invalid kind")
    _ensure(_is_str(obj["run_id"]) and len(obj["run_id"]) >= 8, "run_id must be a non-empty string")
    _ensure(_is_str(obj["event_time_utc"]), "event_time_utc must be a string")
    _parse_utc(obj["event_time_utc"])


def validate_run_meta_v1(obj: Mapping[str, Any]) -> None:
    _validate_common(obj)
    _ensure(obj["kind"] == KIND_RUN_META, "kind mismatch")
    _require_keys(obj, ["mode", "symbol", "models", "code_version"])
    _ensure(obj["mode"] in {"replay", "live"}, "mode must be 'replay' or 'live'")
    _ensure(_is_str(obj["symbol"]) and len(obj["symbol"]) >= 3, "symbol must be a string")
    _ensure(isinstance(obj["models"], Mapping), "models must be an object")
    _ensure(isinstance(obj["code_version"], Mapping), "code_version must be an object")


def validate_market_bar_1m_closed_v1(obj: Mapping[str, Any]) -> None:
    _validate_common(obj)
    _ensure(obj["kind"] == KIND_MARKET_BAR_1M_CLOSED, "kind mismatch")
    _require_keys(
        obj,
        [
            "symbol",
            "source",
            "interval",
            "bar_open_time_utc",
            "bar_close_time_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
    )
    _ensure(_is_str(obj["symbol"]) and len(obj["symbol"]) >= 3, "symbol must be a string")
    _ensure(_is_str(obj["source"]) and len(obj["source"]) >= 3, "source must be a string")
    _ensure(obj["interval"] == "1m", "interval must be '1m'")
    _ensure(_is_str(obj["bar_open_time_utc"]), "bar_open_time_utc must be a string")
    _ensure(_is_str(obj["bar_close_time_utc"]), "bar_close_time_utc must be a string")
    t0 = _parse_utc(obj["bar_open_time_utc"])
    t1 = _parse_utc(obj["bar_close_time_utc"])
    _ensure(t1 > t0, "bar_close_time_utc must be after bar_open_time_utc")

    for k in ("open", "high", "low", "close", "volume"):
        _ensure(_is_number(obj[k]), f"{k} must be a finite number")

    o = float(obj["open"])
    h = float(obj["high"])
    l = float(obj["low"])
    c = float(obj["close"])
    v = float(obj["volume"])

    _ensure(v >= 0.0, "volume must be >= 0")
    _ensure(h >= max(o, c), "high must be >= max(open, close)")
    _ensure(l <= min(o, c), "low must be <= min(open, close)")
    _ensure(h >= l, "high must be >= low")


def validate_entry_decision_v1(obj: Mapping[str, Any]) -> None:
    _validate_common(obj)
    _ensure(obj["kind"] == KIND_ENTRY_DECISION, "kind mismatch")
    _require_keys(
        obj,
        [
            "symbol",
            "decision_time_utc",
            "bar_open_time_utc",
            "bar_close_time_utc",
            "score",
            "threshold",
            "action",
            "planned_entry_time_utc",
            "policy",
            "feature_names",
            "feature_values",
            "model",
        ],
    )

    _ensure(_is_str(obj["symbol"]) and len(obj["symbol"]) >= 3, "symbol must be a string")
    _ensure(_is_str(obj["decision_time_utc"]), "decision_time_utc must be a string")
    _parse_utc(obj["decision_time_utc"])

    _ensure(_is_str(obj["bar_open_time_utc"]), "bar_open_time_utc must be a string")
    _ensure(_is_str(obj["bar_close_time_utc"]), "bar_close_time_utc must be a string")
    _parse_utc(obj["bar_open_time_utc"])
    _parse_utc(obj["bar_close_time_utc"])

    # NOTE: JSON audit logs must not contain NaN/Inf; keep these finite.
    _ensure(_is_number(obj["score"]), "score must be a finite number")
    _ensure(_is_number(obj["threshold"]), "threshold must be a finite number")

    _ensure(obj["action"] in {"enter", "hold"}, "action must be 'enter' or 'hold'")

    _ensure(_is_str(obj["planned_entry_time_utc"]), "planned_entry_time_utc must be a string")
    _parse_utc(obj["planned_entry_time_utc"])

    _ensure(_is_str(obj["policy"]) and len(obj["policy"]) >= 3, "policy must be a string")

    _ensure(isinstance(obj["feature_names"], list) and obj["feature_names"], "feature_names must be a non-empty list")
    _ensure(isinstance(obj["feature_values"], list), "feature_values must be a list")
    _ensure(len(obj["feature_names"]) == len(obj["feature_values"]), "feature_names and feature_values length mismatch")
    _ensure(all(_is_str(x) for x in obj["feature_names"]), "feature_names items must be strings")
    _ensure(all(_is_number_or_null(x) for x in obj["feature_values"]), "feature_values items must be numbers or null")

    _ensure(isinstance(obj["model"], Mapping), "model must be an object")
    _require_keys(obj["model"], ["role", "artifact", "created_utc", "features"])
    _ensure(obj["model"]["role"] == "entry", "model.role must be 'entry'")
    _ensure(_is_str(obj["model"]["artifact"]) and obj["model"]["artifact"], "model.artifact must be a string")
    _ensure(_is_str(obj["model"]["created_utc"]) and obj["model"]["created_utc"], "model.created_utc must be a string")
    _ensure(isinstance(obj["model"]["features"], list) and obj["model"]["features"], "model.features must be a non-empty list")


def validate_trade_closed_v1(obj: Mapping[str, Any]) -> None:
    _validate_common(obj)
    _ensure(obj["kind"] == KIND_TRADE_CLOSED, "kind mismatch")
    _require_keys(
        obj,
        [
            "symbol",
            "paper",
            "entry_time_utc",
            "exit_time_utc",
            "entry_price",
            "exit_price",
            "fee_side",
            "exit_rel_min",
            "realized_ret_pct",
        ],
    )

    _ensure(_is_str(obj["symbol"]) and len(obj["symbol"]) >= 3, "symbol must be a string")
    _ensure(_is_bool(obj["paper"]), "paper must be boolean")

    _ensure(_is_str(obj["entry_time_utc"]), "entry_time_utc must be a string")
    _ensure(_is_str(obj["exit_time_utc"]), "exit_time_utc must be a string")
    t0 = _parse_utc(obj["entry_time_utc"])
    t1 = _parse_utc(obj["exit_time_utc"])
    _ensure(t1 >= t0, "exit_time_utc must be >= entry_time_utc")

    for k in ("entry_price", "exit_price", "fee_side", "realized_ret_pct"):
        _ensure(_is_number(obj[k]), f"{k} must be a finite number")

    _ensure(_is_int(obj["exit_rel_min"]) and obj["exit_rel_min"] >= 1, "exit_rel_min must be int >= 1")

    _ensure(float(obj["entry_price"]) > 0.0, "entry_price must be > 0")
    _ensure(float(obj["exit_price"]) > 0.0, "exit_price must be > 0")
    _ensure(float(obj["fee_side"]) >= 0.0, "fee_side must be >= 0")

    # Optional fields
    if "predicted_ret_pct" in obj:
        _ensure(_is_number(obj["predicted_ret_pct"]) or obj["predicted_ret_pct"] is None, "predicted_ret_pct must be number or null")
    if "entry_score" in obj:
        _ensure(_is_number(obj["entry_score"]) or obj["entry_score"] is None, "entry_score must be number or null")
    if "entry_threshold" in obj:
        _ensure(_is_number(obj["entry_threshold"]) or obj["entry_threshold"] is None, "entry_threshold must be number or null")


def validate_trade_mode_switch_v1(obj: Mapping[str, Any]) -> None:
    _validate_common(obj)
    _ensure(obj["kind"] == KIND_TRADE_MODE_SWITCH, "kind mismatch")
    _require_keys(
        obj,
        [
            "symbol",
            "from_trade_mode",
            "to_trade_mode",
            "reason",
        ],
    )
    _ensure(_is_str(obj["symbol"]) and len(obj["symbol"]) >= 3, "symbol must be a string")
    _ensure(obj["from_trade_mode"] in {"paper", "auto", "real"}, "from_trade_mode must be paper/auto/real")
    _ensure(obj["to_trade_mode"] in {"paper", "auto", "real"}, "to_trade_mode must be paper/auto/real")
    _ensure(_is_str(obj["reason"]) and len(obj["reason"]) >= 3, "reason must be a string")


def validate_event_v1(obj: Mapping[str, Any]) -> None:
    """Validate a single event dict.

    Raises ContractValidationError on failure.
    """
    _validate_common(obj)
    kind = obj["kind"]
    if kind == KIND_RUN_META:
        validate_run_meta_v1(obj)
    elif kind == KIND_MARKET_BAR_1M_CLOSED:
        validate_market_bar_1m_closed_v1(obj)
    elif kind == KIND_ENTRY_DECISION:
        validate_entry_decision_v1(obj)
    elif kind == KIND_TRADE_CLOSED:
        validate_trade_closed_v1(obj)
    elif kind == KIND_TRADE_MODE_SWITCH:
        validate_trade_mode_switch_v1(obj)
    else:
        raise ContractValidationError(f"Unsupported kind: {kind}")
