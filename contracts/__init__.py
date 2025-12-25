# Timestamp (UTC): 2025-12-22T09:41:32Z
"""Versioned data contracts + audit logging utilities.

These contracts are meant to freeze the live/replay interface between:
- market data ingestion
- feature building
- model inference/decisioning
- (paper) execution + trade records

See `contracts/v1.py` for v1 event definitions + validators.
"""

from .audit import AuditLogger
from .v1 import SCHEMA_VERSION_V1, validate_event_v1

__all__ = [
    "AuditLogger",
    "SCHEMA_VERSION_V1",
    "validate_event_v1",
]
