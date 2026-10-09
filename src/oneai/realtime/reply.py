"""Guided-reply generator, plus the plain baseline bot used in the A/B demo.

Uses the LLM when OPENAI_API_KEY is set and falls back to templates otherwise.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from oneai import llm
from oneai.offline.agent import REPEAT_WORDS
from oneai.taxonomy import PatternPack

REPLY_SYSTEM = """You are a voice support agent. Reply in one or two short spoken sentences.
Follow this Pattern Pack learned from past calls like this one:
- Style: {style}. {style_cue}
- Opener that works: {opener}
- Useful follow-up questions: {follow_ups}
- Watch out for: {friction}
Do not ask for anything the customer has already told you."""

PLAIN_SYSTEM = "You are a voice support agent. Reply in one or two short spoken sentences."

PLAIN_TEMPLATE = "Thanks for calling. Could you tell me more about the problem?"


def _history_text(history: Sequence[str]) -> str:
    return "\n".join(f"agent: {h}" for h in history)


def _template_reply(turn_text: str, pack: PatternPack, history: Sequence[str]) -> str:
    lowered = turn_text.lower()
    parts: List[str] = []
    if history and any(w in lowered for w in REPEAT_WORDS):
        parts.append("You're right, I have what you've told me, so you won't need to repeat it.")
    elif not history:
        parts.append(pack.recommended_opener)
    asked = " ".join(history)
    follow_up = next((q for q in pack.follow_ups if q not in asked), None)
    if follow_up:
        parts.append(follow_up)
    elif history:
        parts.append("I'm taking care of it now and I'll confirm the next step.")
    return " ".join(p for p in parts if p) or pack.recommended_opener


def suggest_reply(turn_text: str, pack: PatternPack, history: Optional[Sequence[str]] = None) -> str:
    """The agent's suggested reply to a live customer turn, guided by the Pattern Pack."""
    history = list(history or [])
    if llm.available():
        system = REPLY_SYSTEM.format(
            style=pack.style,
            style_cue=pack.style_cue,
            opener=pack.recommended_opener,
            follow_ups="; ".join(pack.follow_ups),
            friction="; ".join(pack.friction_alerts) or "nothing specific",
        )
        try:
            return llm.chat_text(system, f"{_history_text(history)}\ncustomer: {turn_text}".strip())
        except Exception:
            pass
    return _template_reply(turn_text, pack, history)


def plain_reply(turn_text: str, history: Optional[Sequence[str]] = None) -> str:
    """The baseline bot: same model, no Pattern Pack."""
    if llm.available():
        try:
            return llm.chat_text(PLAIN_SYSTEM, f"{_history_text(history or [])}\ncustomer: {turn_text}".strip())
        except Exception:
            pass
    return PLAIN_TEMPLATE
