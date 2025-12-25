# Timestamp (UTC): 2025-12-24T13:14:02Z
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def load_dotenv(path: Optional[Path] = None, *, override: bool = False) -> Optional[Path]:
    """Load KEY=VALUE pairs from a local .env into os.environ.

    - No external deps (no python-dotenv).
    - Comments/blank lines are ignored.
    - Values are not shell-parsed; surrounding single/double quotes are stripped.

    Returns the path that was loaded, or None if no file exists.
    """

    p = Path(path) if path is not None else Path(".env")
    if not p.exists() or not p.is_file():
        return None

    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue

        # Strip surrounding quotes.
        if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
            v = v[1:-1]

        if (not override) and (k in os.environ):
            continue
        os.environ[k] = v

    return p
