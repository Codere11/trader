# Timestamp (UTC): 2025-12-22T09:41:32Z
from __future__ import annotations

"""JSONL audit logging for live/replay runs.

- Writes one event per line (JSONL).
- Enforces strict JSON (no NaN/Inf) and validates events against v1 contracts.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
from typing import Any, Mapping, Optional

from .v1 import SCHEMA_VERSION_V1, validate_event_v1


def utc_now_iso() -> str:
    # ISO8601 with explicit UTC offset
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AuditLogger:
    path: Path
    run_id: str
    autoflush: bool = True

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    @classmethod
    def create(cls, path: Path, *, run_id: Optional[str] = None, autoflush: bool = True) -> "AuditLogger":
        rid = run_id or uuid.uuid4().hex
        return cls(path=Path(path), run_id=rid, autoflush=autoflush)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def write(self, event: Mapping[str, Any]) -> None:
        # Stamp required common fields if caller didn't.
        obj = dict(event)
        obj.setdefault("schema_version", SCHEMA_VERSION_V1)
        obj.setdefault("run_id", self.run_id)
        obj.setdefault("event_time_utc", utc_now_iso())

        # Validate before writing.
        validate_event_v1(obj)

        # Strict JSON: disallow NaN/Inf.
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        self._fh.write(line + "\n")
        if self.autoflush:
            self._fh.flush()

    def __enter__(self) -> "AuditLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
