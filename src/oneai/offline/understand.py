"""Issue and question-type labelling.

Methods: "llm" (needs OPENAI_API_KEY), "zeroshot" (embedding similarity to label descriptions)
and "rules" (Day 1 keywords). "auto" uses the LLM when a key is set, zero-shot otherwise, and
falls back to rules if either fails.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict

import numpy as np

from oneai import llm
from oneai.offline.embeddings import embed_many
from oneai.offline.extractors import extract_intent, extract_issue
from oneai.taxonomy import Conversation

ISSUE_LABELS: Dict[str, str] = {
    "billing": "billing problem: charges, duplicate charge, invoice, payment, refund, fees",
    "shipping": "shipping problem: package, delivery, tracking, carrier, order arrival",
    "account_access": "account access problem: login, password reset, locked account, two-factor",
}

QUESTION_TYPES: Dict[str, str] = {
    "troubleshoot": "something is broken or wrong and the customer wants it fixed",
    "info": "the customer is asking how, what, when or where to get information",
    "escalate": "the customer demands a supervisor, manager or escalation",
}

LLM_PROMPT = """Classify this customer support utterance.
Return JSON: {{"issue": one of {issues}, "question_type": one of {qtypes}}}

Utterance: {text}"""


def _rules(text: str) -> Dict[str, str]:
    return {"issue": extract_issue(text), "question_type": extract_intent(text), "method": "rules"}


@lru_cache(maxsize=1)
def _label_matrices():
    issue_keys = list(ISSUE_LABELS)
    qtype_keys = list(QUESTION_TYPES)
    issue_vecs = embed_many([ISSUE_LABELS[k] for k in issue_keys])
    qtype_vecs = embed_many([QUESTION_TYPES[k] for k in qtype_keys])
    return issue_keys, issue_vecs, qtype_keys, qtype_vecs


def _zeroshot(text: str) -> Dict[str, str]:
    issue_keys, issue_vecs, qtype_keys, qtype_vecs = _label_matrices()
    v = embed_many([text])[0]
    return {
        "issue": issue_keys[int(np.argmax(issue_vecs @ v))],
        "question_type": qtype_keys[int(np.argmax(qtype_vecs @ v))],
        "method": "zeroshot",
    }


def _llm(text: str) -> Dict[str, str]:
    prompt = LLM_PROMPT.format(issues=list(ISSUE_LABELS), qtypes=list(QUESTION_TYPES), text=text)
    data = llm.chat_json(prompt)
    if data.get("issue") not in ISSUE_LABELS or data.get("question_type") not in QUESTION_TYPES:
        raise ValueError(f"LLM returned labels outside the taxonomy: {data}")
    return {"issue": data["issue"], "question_type": data["question_type"], "method": "llm"}


def understand_text(text: str, method: str = "auto") -> Dict[str, str]:
    if method == "rules":
        return _rules(text)
    if method == "zeroshot":
        return _zeroshot(text)
    if method == "llm" and not llm.available():
        raise RuntimeError("No AI key set (OPENAI_API_KEY or CURSOR_API_KEY)")
    try:
        return _llm(text) if llm.available() else _zeroshot(text)
    except Exception:
        return _rules(text)


def understand(conv: Conversation, method: str = "auto") -> Dict[str, str]:
    return understand_text(conv.customer_text, method=method)
