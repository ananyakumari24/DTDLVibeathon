"""Rule-based extractors: issue, intent, style, friction (Day 1 Person A)."""

from __future__ import annotations

import re
from typing import Dict, List

from oneai.taxonomy import Conversation

ISSUE_KEYWORDS: Dict[str, List[str]] = {
    "billing": ["bill", "charge", "charged", "invoice", "payment", "refund", "fee", "duplicate"],
    "shipping": ["package", "shipping", "shipment", "tracking", "delivery", "delivered", "carrier", "order"],
    "account_access": ["login", "password", "locked", "account", "two-factor", "2fa", "sso", "credentials"],
}

INTENT_KEYWORDS: Dict[str, List[str]] = {
    "troubleshoot": ["fix", "wrong", "can't", "cannot", "broken", "failed", "stuck", "damaged", "never got"],
    "info": ["what", "how", "when", "where", "explain", "tell me", "understand", "mean"],
    "escalate": ["escalate", "supervisor", "manager", "higher up", "specialist", "urgent", "priority"],
}

STYLE_KEYWORDS: Dict[str, List[str]] = {
    "empathic": ["sorry", "frustrating", "hear you", "appreciate", "stressful", "together"],
    "instructional": ["step", "please share", "next", "confirm", "walk you", "exactly"],
}

FRICTION_PATTERNS = [
    (re.compile(r"already (said|explained|told)", re.I), "customer_already_explained"),
    (re.compile(r"don'?t make me repeat", re.I), "customer_refuses_repeat"),
    (re.compile(r"second time|three times|again", re.I), "repeat_contact"),
]


def _score(text: str, lexicon: Dict[str, List[str]]) -> Dict[str, int]:
    lowered = text.lower()
    return {label: sum(1 for kw in kws if kw in lowered) for label, kws in lexicon.items()}


def _best(scores: Dict[str, int], fallback: str) -> str:
    if not scores or max(scores.values()) == 0:
        return fallback
    return max(scores, key=scores.get)


def extract_issue(text: str) -> str:
    return _best(_score(text, ISSUE_KEYWORDS), "billing")


def extract_intent(text: str) -> str:
    return _best(_score(text, INTENT_KEYWORDS), "info")


def extract_style(agent_text: str) -> str:
    return _best(_score(agent_text, STYLE_KEYWORDS), "instructional")


def extract_friction(full_text: str) -> List[str]:
    flags: List[str] = []
    for pattern, name in FRICTION_PATTERNS:
        if pattern.search(full_text):
            flags.append(name)
    return flags


def annotate_conversation(conv: Conversation) -> Conversation:
    flags = extract_friction(conv.full_text)
    return conv.model_copy(update={"friction_flags": flags})


def evaluate_against_labels(conversations: List[Conversation]) -> Dict[str, float]:
    issue_ok = intent_ok = style_ok = 0
    for c in conversations:
        if extract_issue(c.customer_text) == c.issue:
            issue_ok += 1
        if extract_intent(c.customer_text + " " + c.full_text) == c.intent:
            intent_ok += 1
        if extract_style(c.agent_text) == c.style:
            style_ok += 1
    n = max(len(conversations), 1)
    return {
        "issue_accuracy": round(issue_ok / n, 3),
        "intent_accuracy": round(intent_ok / n, 3),
        "style_accuracy": round(style_ok / n, 3),
        "n": float(n),
    }
