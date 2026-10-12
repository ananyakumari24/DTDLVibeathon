"""One shared way to call the LLM. Uses ANTHROPIC_API_KEY (Claude), OPENAI_API_KEY, or CURSOR_API_KEY via the Cursor agent CLI."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")

_cursor_failed = False  # set after an auth failure so live replies stop waiting on a bad key
_CURSOR_BIN = shutil.which("cursor-agent") or os.path.expanduser("~/.local/bin/cursor-agent")
# Models that accept the server-side refusal fallback ("default" routing).
_CLAUDE_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
# How many LLM calls the offline pipeline runs at once.
WORKERS = int(os.getenv("LLM_WORKERS", "16"))


def provider() -> Optional[str]:
    # Only keys of the right shape; other tokens would just get 401s.
    if os.getenv("ANTHROPIC_API_KEY", "").startswith("sk-ant-"):
        return "anthropic"
    if has_openai_key():
        return "openai"
    if os.getenv("CURSOR_API_KEY") and not _cursor_failed and os.path.exists(_CURSOR_BIN):
        return "cursor"
    return None


def has_openai_key() -> bool:
    return os.getenv("OPENAI_API_KEY", "").startswith("sk-")


def available() -> bool:
    return provider() is not None


def model() -> str:
    if provider() == "anthropic":
        return os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
    if provider() == "cursor":
        return os.getenv("CURSOR_MODEL", "cursor default")
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def label_model() -> str:
    """Model for picking labels from a fixed list; ANTHROPIC_LABEL_MODEL can point it at a cheaper one."""
    if provider() == "anthropic":
        return os.getenv("ANTHROPIC_LABEL_MODEL") or model()
    return model()


def pmap(fn: Callable[[T], R], items: Iterable[T]) -> List[R]:
    """Map in parallel threads when LLM calls are in play, keeping input order."""
    items = list(items)
    if not available() or WORKERS <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(fn, items))


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    return OpenAI(timeout=10)


@lru_cache(maxsize=1)
def _anthropic_client():
    import anthropic

    # Keys not scoped to a workspace must name one on every request.
    workspace = os.getenv("ANTHROPIC_WORKSPACE_ID")
    headers = {"anthropic-workspace-id": workspace} if workspace else None
    return anthropic.Anthropic(timeout=10, default_headers=headers)


def _claude(prompt: str, system: str = "", model_name: Optional[str] = None) -> str:
    """One prompt to Claude; raises on refusal or an empty answer so callers fall back."""
    model_name = model_name or model()
    kwargs: Dict[str, Any] = {"system": system} if system else {}
    if model_name in _CLAUDE_FALLBACK_MODELS:
        kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    resp = _anthropic_client().beta.messages.create(
        model=model_name,
        max_tokens=4000,  # thinking counts toward this; low effort keeps both it and latency small
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if resp.stop_reason == "refusal" or not text:
        raise RuntimeError(f"Claude returned no answer (stop_reason={resp.stop_reason})")
    return text


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


def _first_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0) if match else "{}")


def chat_json(prompt: str, label: bool = False) -> Dict[str, Any]:
    """label=True uses label_model(), for simple labelling calls."""
    if provider() == "anthropic":
        return _first_json(
            _claude(prompt + "\n\nAnswer with one JSON object only, no other text.", model_name=label_model() if label else None)
        )
    if provider() == "cursor":
        return _first_json(_cursor(prompt + "\n\nAnswer with one JSON object only, no other text."))
    resp = _client().chat.completions.create(
        model=model(),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content or "{}")


def chat_text(system: str, user: str) -> str:
    if provider() == "anthropic":
        return _claude(f"{user}\n\nReply with only the agent's next line, nothing else.", system)
    if provider() == "cursor":
        return _cursor(f"{system}\n\n{user}\n\nReply with only the agent's next line, nothing else.")
    resp = _client().chat.completions.create(
        model=model(),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
        max_tokens=120,
    )
    return (resp.choices[0].message.content or "").strip()
