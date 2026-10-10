"""One customer turn, end to end: pick the pack, plan the FAQ step, write the reply.

Shared by the API's /reply and the LiveKit voice agent, so the dashboard and the phone line
behave the same. CallState is what has to carry over from one turn to the next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from oneai.realtime.classify import get_live_classifier
from oneai.realtime.faq import FaqTurn, plan_turn
from oneai.realtime.reply import call_ends, suggest_reply
from oneai.taxonomy import PatternPack


@dataclass
class CallState:
    said: List[str] = field(default_factory=list)  # earlier customer turns
    history: List[str] = field(default_factory=list)  # earlier bot replies
    pack_id: Optional[str] = None  # pack used by the previous reply
    faq_id: Optional[str] = None  # FAQ in progress
    faq_step: int = 0  # next FAQ step to give
    paused_faq_id: Optional[str] = None  # FAQ paused for a side question
    paused_faq_step: int = 0  # where the paused FAQ resumes

    def record(self, text: str, turn: "Turn") -> None:
        """Carry this turn's outcome over to the next one."""
        self.said.append(text)
        self.history.append(turn.reply)
        self.pack_id = turn.pack.pack_id
        self.faq_id, self.faq_step = turn.faq_id, turn.faq_step
        self.paused_faq_id, self.paused_faq_step = turn.paused_faq_id, turn.paused_faq_step


@dataclass
class Turn:
    pack: PatternPack
    similarity: float
    latency_ms: float
    pack_created: bool
    faq: Optional[FaqTurn]
    ended: bool
    reply: str
    faq_id: Optional[str]
    faq_step: int
    paused_faq_id: Optional[str]
    paused_faq_step: int


def take_turn(text: str, state: CallState) -> Turn:
    pack, score, ms, created = get_live_classifier().classify_or_create(
        " ".join(state.said + [text]), latest=text, current=state.pack_id
    )
    faq = plan_turn(text, state.faq_id, state.faq_step, state.paused_faq_id, state.paused_faq_step)
    ended = call_ends(text, faq, state.history)
    # After a side question the reply already went back to the paused flow, so that's what continues.
    current = (faq.resume or faq) if faq else None
    in_progress = current is not None and current.next_step is not None and not ended
    paused = faq.paused if faq is not None and not ended else None
    return Turn(
        pack=pack,
        similarity=score,
        latency_ms=ms,
        pack_created=created,
        faq=faq,
        ended=ended,
        reply=suggest_reply(text, pack, state.history, faq),
        faq_id=current.entry["id"] if in_progress else None,
        faq_step=current.next_step if in_progress else 0,
        paused_faq_id=paused[0] if paused else None,
        paused_faq_step=paused[1] if paused else 0,
    )


def describe_faq(faq: Optional[FaqTurn]) -> Optional[str]:
    if faq is None:
        return None
    if faq.kind == "steps":
        where = f"step {faq.step + 1}/{faq.total}"
    elif faq.kind == "aside":
        where = f"off-topic, holding step {faq.step + 1}/{faq.total}"
    else:
        where = faq.kind.replace("_", " ")
    text = f"{faq.entry['id']} · {faq.entry['intent']} · {where}"
    if faq.detour:
        text += " · side question"
    if faq.paused:
        text += f" · {faq.paused[0]} paused"
    if faq.resume:
        text += f" · back to {faq.resume.entry['id']} step {faq.resume.step + 1}/{faq.resume.total}"
    return text
