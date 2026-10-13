"""Agent-style and friction labelling, in the Insight format.

Uses the LLM when OPENAI_API_KEY is set and falls back to rules otherwise.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from oneai import llm
from oneai.offline.embeddings import embed_many
from oneai.offline.extractors import extract_friction
from oneai.taxonomy import Conversation

STYLES = ("empathic", "instructional")
FRICTION_FLAGS = (
    "customer_already_explained",
    "customer_refuses_repeat",
    "repeat_contact",
    "negative_sentiment",
    "repeated_question",
)

EMPATHY_WORDS = ("sorry", "apolog", "i understand", "completely understand", "hear you", "frustrat", "stressful", "together", "oh no")
NEGATIVE_WORDS = ("frustrated", "annoying", "ridiculous", "angry", "upset", "unacceptable", "worst", "fed up")
REPEAT_WORDS = ("already", "again", "third time", "second time", "repeat")
REPEATED_QUESTION_SIMILARITY = 0.85

INSIGHT_PROMPT = """You label one customer support call.
Return JSON: {{"agent_style": one of {styles}, "friction_flags": a list drawn from {flags}}}
"empathic" agents acknowledge feelings; "instructional" agents give steps and ask for details.

Call:
{call}"""


def _first_agent_turn(conv: Conversation) -> str:
    return next((t.text for t in conv.turns if t.speaker == "agent"), "")


def _style_rules(conv: Conversation) -> str:
    opener = _first_agent_turn(conv).lower()
    return "empathic" if any(w in opener for w in EMPATHY_WORDS) else "instructional"


def _friction_rules(conv: Conversation) -> List[str]:
    flags = set(extract_friction(conv.full_text))
    customer = [t.text for t in conv.turns if t.speaker == "customer"]
    if any(w in " ".join(customer).lower() for w in NEGATIVE_WORDS):
        flags.add("negative_sentiment")
    if len(customer) > 1:
        vecs = embed_many(customer)
        sims = vecs @ vecs.T
        np.fill_diagonal(sims, 0)
        if float(sims.max()) > REPEATED_QUESTION_SIMILARITY:
            flags.add("repeated_question")
    return sorted(flags)


def _llm_insight(conv: Conversation) -> Dict:
    call = "\n".join(f"{t.speaker}: {t.text}" for t in conv.turns)
    data = llm.chat_json(INSIGHT_PROMPT.format(styles=list(STYLES), flags=list(FRICTION_FLAGS), call=call))
    flags = data.get("friction_flags", [])
    if data.get("agent_style") not in STYLES or not isinstance(flags, list):
        raise ValueError(f"LLM returned labels outside the taxonomy: {data}")
    return {
        "agent_style": data["agent_style"],
        "friction_flags": sorted(f for f in flags if f in FRICTION_FLAGS),
        "method": "llm",
    }


def label_agent(conv: Conversation, method: str = "auto") -> Dict:
    """Agent style and friction flags for one call."""
    if method != "rules" and llm.available():
        try:
            return _llm_insight(conv)
        except Exception:
            pass
    return {"agent_style": _style_rules(conv), "friction_flags": _friction_rules(conv), "method": "rules"}


def extract_agent_style(conv: Conversation, method: str = "auto") -> str:
    if method != "rules" and llm.available():
        return label_agent(conv, method)["agent_style"]
    return _style_rules(conv)
