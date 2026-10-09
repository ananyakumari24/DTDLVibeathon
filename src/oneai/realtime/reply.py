"""Guided-reply generator, plus the plain baseline bot used in the A/B demo.

Uses the LLM when OPENAI_API_KEY is set and falls back to templates otherwise.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

from oneai import llm
from oneai.offline.agent import REPEAT_WORDS
from oneai.realtime.faq import QUESTION, FaqTurn
from oneai.realtime.tone import closing_signal, customer_tone
from oneai.taxonomy import PatternPack

REPLY_SYSTEM = """You are a voice support agent. Reply in one or two short spoken sentences.
Follow this Pattern Pack learned from past calls like this one:
- Style: {style}. {style_cue}
- Opener that works: {opener}
- Useful follow-up questions: {follow_ups}
- Watch out for: {friction}
Do not ask for anything the customer has already told you.
{tone_rule}
{knowledge}"""

TONE_RULES = {
    "upset": "The customer sounds upset. Start with a sincere apology (\"I'm really sorry...\") before anything else.",
    "complaint": "The customer is reporting a problem. Start with a short apology (\"I'm sorry...\") before anything else.",
    "neutral": "",
}
# Said first whenever the customer is complaining, unless the reply already opens with an apology.
APOLOGIES = {
    "upset": "I'm really sorry about this, I can hear how frustrating it's been.",
    "complaint": "I'm sorry about that.",
}
APOLOGY_WORDS = re.compile(r"\b(sorry|apologi[sz]e|apologies)\b", re.I)
# A sentence that is only an apology, e.g. "I'm sorry about this." Dropped when an earlier one already apologised.
APOLOGY_ONLY = re.compile(r"^(i'?m|i am|we'?re|we are)?\s*(really |so |very |truly )?(sorry|apologi[sz]e)\b[^,;]{0,30}[.!]$", re.I)
# How the bot ends the call once the customer is done: (if it was solved, sign-off).
GOODBYES = {
    "empathic": ("I'm so glad that's sorted.", "Thank you for reaching out, I was glad to help. Take care!"),
    "instructional": ("Great, that's done.", "Thanks for reaching out, glad I could help. Goodbye!"),
}
ESCALATE_GOODBYE_LEAD = {
    "empathic": "You're all set, and the team will keep you updated.",
    "instructional": "That's all in hand, and you'll get updates by SMS.",
}

FAQ_KNOWLEDGE = """This call follows the company FAQ "{question}" ({intent}).
Say exactly this content now, reworded only to match the style above, and nothing beyond it: {say}
{pacing}
Do not invent policy, numbers or steps that are not in the content above."""

# How the bot moves between FAQ steps, per agent style: (after "done", after "didn't work").
TRANSITIONS = {
    "empathic": ("Great, thanks for bearing with me.", "Sorry that didn't do it, let's try the next thing."),
    "instructional": ("Good.", "Okay, next step."),
}
# What the bot says after a step, per (intent, style), so the customer knows to answer before we go on.
CHECK_INS = {
    ("troubleshoot", "empathic"): "Take your time, and tell me when you've done that.",
    ("troubleshoot", "instructional"): "Tell me once that's done and I'll give you the next step.",
    ("escalate", "empathic"): "I'm here with you, just let me know when you're ready to go on.",
    ("escalate", "instructional"): "Let me know and I'll move to the next step.",
}
LAST_STEP_CHECK = {
    "troubleshoot": "Did that sort it out?",
    "escalate": "Is there anything else you'd like me to add to the case?",
}
CLOSINGS = {
    "empathic": "I'm really glad that's sorted. Is there anything else I can help with today?",
    "instructional": "Great, that's done. Anything else I can help with?",
}
ESCALATE_CLOSINGS = {
    "empathic": "You're all set, and the team will keep you updated. Is there anything else I can help with today?",
    "instructional": "That's all in hand, and you'll get updates by SMS. Anything else?",
}
# Used instead of the pack's opener when the pack was mined for a different intent than the FAQ,
# e.g. an apologetic troubleshoot opener in front of a plain factual answer.
INTENT_OPENERS = {
    ("info", "empathic"): "Good question, happy to explain.",
    ("info", "instructional"): "Sure.",
    ("troubleshoot", "empathic"): "I'm sorry about the trouble, let's sort it out together.",
    ("troubleshoot", "instructional"): "I can help with that. Let's go step by step.",
    ("escalate", "empathic"): "I'm really sorry this has happened. I'll take care of it.",
    ("escalate", "instructional"): "Understood. I'll handle this for you.",
}

# When the customer wanders off mid-flow: a side question that pauses the current flow, an off-topic
# aside that holds the current step, and the line that brings them back afterwards.
DETOUR_OPENERS = {"empathic": "Sure, let's look at that first.", "instructional": "Sure, quick one first."}
ASIDE_LEADS = {
    "empathic": "Let's finish this step first, and then I'm happy to help with anything else.",
    "instructional": "Let's finish this step first.",
}
# The customer is stuck on a step (no email, expired link), so we skip to the FAQ's fallback.
BLOCKED_LEADS = {
    "empathic": "Thanks for telling me, let's not get stuck on that step.",
    "instructional": "Okay, we'll do it another way.",
}
# Pushback after the fallback was given: restate the hand-off instead of starting the FAQ over.
HANDED_OFF_LEADS = {
    "empathic": "I understand, and I won't send you through those steps again.",
    "instructional": "Understood, we won't repeat those steps.",
}
# A question about the handed-off case that the FAQ can't answer: note it for the team.
CASE_NOTES = {
    "empathic": "Good question. I'll add that to your case so the team covers it when they get back to you. Is there anything else I can help with today?",
    "instructional": "I'll add that to the case so the team covers it when they follow up. Anything else?",
}
DETOUR_DONE = {"empathic": "Great, glad that one's sorted.", "instructional": "Good, that's done."}
BACK_TO = {"empathic": "Now, back to where we were.", "instructional": "Back to where we were."}

PLAIN_SYSTEM = "You are a voice support agent. Reply in one or two short spoken sentences."

PLAIN_TEMPLATE = "Thanks for calling. Could you tell me more about the problem?"


def _history_text(history: Sequence[str]) -> str:
    return "\n".join(f"agent: {h}" for h in history)


def _style(pack: PatternPack) -> str:
    return pack.style if pack.style in TRANSITIONS else "empathic"


def _faq_pacing(faq: FaqTurn, style: str) -> str:
    """The line after the FAQ content: a check-in, a last-step check, or nothing."""
    faq = faq.resume or faq
    if faq.kind not in ("steps", "aside") or faq.entry["intent"] == "info" or faq.say[-1].rstrip().endswith("?"):
        return ""
    if faq.is_last_step:
        return LAST_STEP_CHECK.get(faq.entry["intent"], "")
    return CHECK_INS.get((faq.entry["intent"], style), "")


def _handed_off_content(turn_text: str, faq: FaqTurn, style: str) -> List[str]:
    """After the fallback: restate it only on pushback; otherwise answer what the customer said."""
    if faq.signal in ("failed", "blocked", "no") or customer_tone(turn_text) != "neutral":
        return [HANDED_OFF_LEADS[style], faq.entry["if_not_resolved"] or ESCALATE_CLOSINGS[style]]
    if faq.signal is None and QUESTION.search(turn_text.lower()):
        return [CASE_NOTES[style]]
    return [ESCALATE_CLOSINGS[style]]  # "ok", "yes please", "great"


def _faq_content(turn_text: str, faq: FaqTurn, style: str) -> List[str]:
    if faq.kind == "aside":
        content = [ASIDE_LEADS[style]] + faq.say
    elif faq.kind == "resolved":
        content = [DETOUR_DONE[style] if faq.resume else (ESCALATE_CLOSINGS if faq.entry["intent"] == "escalate" else CLOSINGS)[style]]
    elif faq.kind == "not_resolved":
        lead = BLOCKED_LEADS[style] if faq.signal == "blocked" else ""
        content = [lead, faq.entry["if_not_resolved"] or CLOSINGS[style]]
    elif faq.kind == "handed_off":
        content = _handed_off_content(turn_text, faq, style)
    else:
        content = [BACK_TO[style]] if faq.back else []
        if faq.signal == "done":
            content.append(TRANSITIONS[style][0])
        elif faq.signal == "failed":
            content.append(TRANSITIONS[style][1])
        content += faq.say
    if faq.resume:
        content += [BACK_TO[style]] + faq.resume.say
    return content


def _faq_reply(turn_text: str, pack: PatternPack, history: Sequence[str], faq: FaqTurn) -> str:
    style = _style(pack)
    parts: List[str] = []
    if faq.switched and faq.detour:
        parts.append(DETOUR_OPENERS[style])
    elif faq.switched:
        intent = faq.entry["intent"]
        # Mined openers lean apologetic, which reads oddly before a plain factual answer.
        same_intent = pack.intent == intent and intent != "info" and not pack.pack_id.startswith("unknown__")
        parts.append(pack.recommended_opener if same_intent and not history else INTENT_OPENERS[(intent, style)])
    elif any(w in turn_text.lower() for w in REPEAT_WORDS):
        parts.append("You're right, I have what you've told me, so you won't need to repeat it.")
    parts += _faq_content(turn_text, faq, style)
    parts.append(_faq_pacing(faq, style))
    return " ".join(p for p in parts if p)


def _template_reply(turn_text: str, pack: PatternPack, history: Sequence[str]) -> str:
    lowered = turn_text.lower()
    parts: List[str] = []
    if pack.issue == "unknown" or pack.pack_id.startswith("unknown__"):
        if not history:
            parts.append(pack.recommended_opener)
        asked = " ".join(history)
        follow_up = next((q for q in pack.follow_ups if q not in asked), None)
        if follow_up:
            parts.append(follow_up)
        elif history:
            parts.append("Thanks — once I know whether this is billing, shipping, or account access, I can pull the right playbook.")
        return " ".join(p for p in parts if p) or pack.recommended_opener
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


def with_apology(reply: str, tone: str) -> str:
    """Put an apology first for a complaining customer, and never say sorry twice in one reply."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", reply.strip()) if s]
    if not sentences:
        return reply
    if tone in APOLOGIES and not APOLOGY_WORDS.search(sentences[0]):
        sentences = [APOLOGIES[tone]] + [s for s in sentences if not APOLOGY_WORDS.search(s)]
    kept: List[str] = []
    for sentence in sentences:
        if not (APOLOGY_ONLY.match(sentence) and any(APOLOGY_WORDS.search(k) for k in kept)):
            kept.append(sentence)
    return " ".join(kept)


def call_ends(turn_text: str, faq: Optional[FaqTurn] = None, history: Sequence[str] = ()) -> bool:
    """True when the customer's query is resolved or they're done, so the bot should sign off."""
    if faq is not None and faq.kind == "handed_off" and faq.signal == "no":
        # "No" to "anything else?" is goodbye; "no" straight after the fallback is pushback.
        return bool(history) and "anything else" in history[-1].lower()
    if faq is not None and (faq.paused or faq.resume):
        # Only a side question was settled; the call ends only if the customer says goodbye.
        return closing_signal(turn_text, mid_steps=True) == "bye"
    if faq is not None and faq.kind == "resolved" and "?" not in turn_text:
        return True
    if faq is not None and faq.kind == "not_resolved":
        return False  # the fix didn't work, so give the fallback rather than "glad I could help"
    mid_steps = faq is not None and faq.kind == "steps" and not faq.switched
    return closing_signal(turn_text, mid_steps) is not None


def goodbye(turn_text: str, pack: PatternPack, faq: Optional[FaqTurn] = None) -> str:
    """Thank the customer and end the call; mention the fix or hand-off only when there was one."""
    style = _style(pack)
    solved_lead, sign_off = GOODBYES[style]
    handed_off = faq is not None and faq.kind == "handed_off"
    if handed_off and faq.signal == "no":
        return sign_off  # they just heard the hand-off and said there's nothing else
    if handed_off or (faq is not None and faq.kind == "resolved" and faq.entry["intent"] == "escalate"):
        return f"{ESCALATE_GOODBYE_LEAD[style]} {sign_off}"
    solved = (faq is not None and faq.kind == "resolved") or closing_signal(turn_text) == "solved"
    return f"{solved_lead} {sign_off}" if solved else sign_off


def suggest_reply(
    turn_text: str, pack: PatternPack, history: Optional[Sequence[str]] = None, faq: Optional[FaqTurn] = None
) -> str:
    """The agent's suggested reply to a live customer turn, guided by the Pattern Pack.

    The pack sets tone (style, opener, friction); faq, when one matched, supplies the facts for
    this turn only (one step, or the whole answer for an info question). A complaining customer
    always hears an apology first, and once they're done the bot thanks them and signs off.
    """
    tone = customer_tone(turn_text)
    if call_ends(turn_text, faq, history or ()):
        return with_apology(goodbye(turn_text, pack, faq), tone)
    return with_apology(_guided_reply(turn_text, pack, list(history or []), faq, tone), tone)


def _guided_reply(turn_text: str, pack: PatternPack, history: List[str], faq: Optional[FaqTurn], tone: str) -> str:
    if llm.available():
        system = REPLY_SYSTEM.format(
            style=pack.style,
            style_cue=pack.style_cue,
            opener=pack.recommended_opener,
            follow_ups="; ".join(pack.follow_ups),
            friction="; ".join(pack.friction_alerts) or "nothing specific",
            tone_rule=TONE_RULES[tone],
            knowledge=(
                FAQ_KNOWLEDGE.format(
                    question=faq.entry["question"],
                    intent=faq.entry["intent"],
                    say=" ".join(_faq_content(turn_text, faq, _style(pack))),
                    pacing=f"Then end with: {_faq_pacing(faq, _style(pack))}" if _faq_pacing(faq, _style(pack)) else "",
                )
                if faq
                else "If you don't know a company policy, ask a clarifying question instead of guessing."
            ),
        )
        try:
            return llm.chat_text(system, f"{_history_text(history)}\ncustomer: {turn_text}".strip())
        except Exception:
            pass
    return _faq_reply(turn_text, pack, history, faq) if faq else _template_reply(turn_text, pack, history)


def plain_reply(turn_text: str, history: Optional[Sequence[str]] = None) -> str:
    """The baseline bot: same model, no Pattern Pack."""
    if llm.available():
        try:
            return llm.chat_text(PLAIN_SYSTEM, f"{_history_text(history or [])}\ncustomer: {turn_text}".strip())
        except Exception:
            pass
    return PLAIN_TEMPLATE
