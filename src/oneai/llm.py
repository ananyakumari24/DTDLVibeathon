"""One shared way to call the LLM. Needs OPENAI_API_KEY."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict


def available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    return OpenAI(timeout=10)


def chat_json(prompt: str) -> Dict[str, Any]:
    resp = _client().chat.completions.create(
        model=model(),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content or "{}")


def chat_text(system: str, user: str) -> str:
    resp = _client().chat.completions.create(
        model=model(),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
        max_tokens=120,
    )
    return (resp.choices[0].message.content or "").strip()
