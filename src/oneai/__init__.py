"""Load the repo's .env (if any) so keys work however the app is started. Real env vars win."""

import os
from pathlib import Path

_env = Path(__file__).resolve().parents[2] / ".env"
if _env.is_file():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))
