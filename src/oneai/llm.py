"""One shared way to call the LLM. Uses OPENAI_API_KEY, or CURSOR_API_KEY via the Cursor agent CLI."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from typing import Any, Dict, Optional

_cursor_failed = False  # set after an auth failure so live replies stop waiting on a bad key
_CURSOR_BIN = shutil.which("cursor-agent") or os.path.expanduser("~/.local/bin/cursor-agent")


def provider() -> Optional[str]:
    # Only something shaped like an OpenAI key; other tokens would just get 401s.
    if os.getenv("OPENAI_API_KEY", "").startswith("sk-"):
        return "openai"
    if os.getenv("CURSOR_API_KEY") and not _cursor_failed and os.path.exists(_CURSOR_BIN):
        return "cursor"
    return None


def available() -> bool:
    return provider() is not None


def model() -> str:
    if provider() == "cursor":
        return os.getenv("CURSOR_MODEL", "cursor default")
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    return OpenAI(timeout=10)


@lru_cache(maxsize=1)
def _cursor_workspace() -> str:
    # An empty folder, so the agent never sees or touches the repo.
    return tempfile.mkdtemp(prefix="oneai-cursor-")


def _cursor(prompt: str) -> str:
    """One prompt through `cursor-agent` in read-only ask mode; raises on any failure."""
    global _cursor_failed
    cmd = [_CURSOR_BIN, "-p", "--mode", "ask", "--trust", "--workspace", _cursor_workspace()]
    if os.getenv("CURSOR_MODEL"):
        cmd += ["--model", os.environ["CURSOR_MODEL"]]
    out = subprocess.run(cmd + [prompt], capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL)
    if out.returncode != 0 or not out.stdout.strip():
        if "api key" in out.stderr.lower() or "not logged in" in out.stderr.lower():
            _cursor_failed = True
        raise RuntimeError(f"cursor-agent failed: {out.stderr.strip()[:200]}")
    return out.stdout.strip()


def chat_json(prompt: str) -> Dict[str, Any]:
    if provider() == "cursor":
        text = _cursor(prompt + "\n\nAnswer with one JSON object only, no other text.")
        match = re.search(r"\{.*\}", text, re.S)
        return json.loads(match.group(0) if match else "{}")
    resp = _client().chat.completions.create(
        model=model(),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content or "{}")


def chat_text(system: str, user: str) -> str:
    if provider() == "cursor":
        return _cursor(f"{system}\n\n{user}\n\nReply with only the agent's next line, nothing else.")
    resp = _client().chat.completions.create(
        model=model(),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
        max_tokens=120,
    )
    return (resp.choices[0].message.content or "").strip()
