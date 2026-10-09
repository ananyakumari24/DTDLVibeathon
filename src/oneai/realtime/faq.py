"""Telecom FAQ: finds the entry for a customer turn and walks its steps one turn at a time.

Each entry carries an issue and an intent like the Pattern Packs do, plus ordered steps:
- info: a factual answer, said in one turn.
- troubleshoot: one step per turn; the customer says done / didn't work and we move on.
- escalate: one action per turn, ending in a hand-off if it isn't resolved.
The caller keeps (faq_id, next_step) between turns and passes it back, so step tracking does
not depend on how the reply was worded.

Customers wander off mid-flow, so:
- a side question that matches another FAQ pauses the current flow (paused_id, paused_step);
  once the side flow finishes, the same reply picks the paused flow back up at the step it was on.
- an off-topic remark or question that matches nothing is an "aside": the current step is held
  and repeated, not advanced as if the customer had said "done".

Customers also push back, so:
- "blocked" ("I didn't get the email", "the link expired") on a troubleshoot step means the self-serve
  route is stuck, so we go straight to the FAQ's fallback rather than pretending the step was done.
- once the fallback is given the call is "handed off": the FAQ stays active, so further complaints
  ("still not working", "no one has fixed it") get the hand-off restated, not the FAQ from step 1,
  and follow-ups about the case ("how long will that take?") aren't mistaken for new problems.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from oneai.offline.embeddings import embed_many

FAQ_PATH = Path("data/faq.json")
MIN_SIMILARITY = 0.45
# A blocked turn ("I can't find the button") or a follow-up after the hand-off ("what's my reference
# number?") usually still concerns the FAQ in progress (<= 0.56 to other FAQs); a real new problem
# ("I can't log in", "I also want to change my plan") scores 0.71+.
STRICT_SWITCH_SIM = 0.6

# Checked in this order, so "not working" is read as failed before "working" as resolved.
SIGNAL_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("failed", re.compile(
        r"\b(didn'?t|did not|doesn'?t|does not|not) (work|help|change)|not working|still (not|no|can'?t|won'?t|the same)|no luck"
        r"|nothing happened|same problem|failed\b|\b(nobody|no one)\b|\bstopped\b|no longer work"
        r"|\b(haven'?t|hasn'?t|have not|has not|wasn'?t|was not|isn'?t|is not|never|not) (been |yet |really |actually )?(fixed|sorted|solved|resolved|working|worked)"
    )),
    ("blocked", re.compile(
        r"\b(can'?t|cannot|couldn'?t|unable to) (find|see|get|open|access|receive|log|sign|click|tap)"
        r"|\b(didn'?t|did not|haven'?t|have not|hasn'?t|has not|never) (get|got|receive|received|arrive|arrived|come|came|see|seen|show)"
        r"|\bno (email|code|link|text|sms|message|button|option)\b|\bexpired\b|\binvalid\b|\berror\b"
        r"|\b(don'?t|do not) have (access|that|the|it|a|an)\b|won'?t let me"
    )),
    ("resolved", re.compile(r"\b(that|it) (worked|fixed it)|\bfixed\b|\bsorted\b|all good|working now|\bresolved\b")),
    ("no", re.compile(r"^\s*(no|nope|nah)\b")),
    ("done", re.compile(r"^\s*(ok(ay)?|yes|yep|yeah|done|sure|right|go ahead|got it|next)\b|\b(i'?ve )?done (that|it)\b|\bfinished\b|\bit says\b|\bit shows\b")),
]
# A turn that asks something. Mid-flow, one that matches no FAQ is an aside, not an answer to the step.
QUESTION = re.compile(r"\?|^\s*(by the way|btw|what|how|why|when|where|who|which|can|could|is|are|do|does|will|would)\b")


def progress_signal(text: str) -> Optional[str]:
    lowered = text.lower()
    for name, pattern in SIGNAL_PATTERNS:
        if pattern.search(lowered):
            return name
    return None


@dataclass
class FaqTurn:
    """What the bot should say from the FAQ on this turn."""

    entry: dict
    kind: str  # "steps" | "resolved" | "not_resolved" | "handed_off" | "aside" (off-topic; `say` repeats the held step)
    say: List[str] = field(default_factory=list)
    step: int = 0  # index of the first step in `say`
    next_step: Optional[int] = None  # None once this FAQ is finished
    signal: Optional[str] = None  # customer's reaction to the previous step
    switched: bool = False  # True when this turn started a new FAQ
    detour: bool = False  # True when this FAQ is a side question asked mid-way through another
    paused: Optional[Tuple[str, int]] = None  # (faq_id, next_step) of the flow waiting for us to come back
    resume: Optional["FaqTurn"] = None  # the paused flow's held step, said right after this turn's content
    back: bool = False  # True when the customer asked to go back to the paused flow

    @property
    def total(self) -> int:
        return len(self.entry["steps"])

    @property
    def handed_off(self) -> bool:
        return self.next_step is not None and self.next_step > self.total

    @property
    def is_last_step(self) -> bool:
        return self.kind in ("steps", "aside") and self.step + len(self.say) >= self.total


class FaqIndex:
    def __init__(self, path: Path = FAQ_PATH):
        self.entries: List[dict] = json.loads(path.read_text(encoding="utf-8"))
        self.by_id: Dict[str, dict] = {e["id"]: e for e in self.entries}
        texts, owners = [], []
        for i, e in enumerate(self.entries):
            for t in [e["question"], *e.get("variants", [])]:
                texts.append(t)
                owners.append(i)
        self.vecs = embed_many(texts)
        self.owners = np.asarray(owners)

    def lookup(self, text: str, min_similarity: float = MIN_SIMILARITY) -> Optional[Tuple[dict, float]]:
        sims = self.vecs @ embed_many([text])[0]
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score < min_similarity:
            return None
        return self.entries[int(self.owners[best])], score

    def _held_step(self, entry: dict, next_step: int, **kw) -> FaqTurn:
        """Repeat the step the customer was on (the one before next_step) without advancing."""
        idx = min(max(next_step - 1, 0), len(entry["steps"]) - 1)
        return FaqTurn(entry, kw.pop("kind", "steps"), [entry["steps"][idx]], idx, idx + 1, **kw)

    def plan_turn(
        self,
        text: str,
        active_id: Optional[str] = None,
        next_step: int = 0,
        paused_id: Optional[str] = None,
        paused_step: int = 0,
    ) -> Optional[FaqTurn]:
        active = self.by_id.get(active_id) if active_id else None
        paused = self.by_id.get(paused_id) if paused_id else None
        signal = progress_signal(text)

        # A short "done" / "didn't work" continues the FAQ in progress; anything else may be a new question,
        # including "that worked, but how do I ...?" and "ok, and how do I ...?".
        new_question = signal in ("resolved", "done") and "?" in text
        handed_off = active is not None and next_step > len(active["steps"])
        if active and (signal == "blocked" or (handed_off and signal is None)):
            hit = self.lookup(text, STRICT_SWITCH_SIM)
        else:
            hit = None if (active and signal and not new_question) else self.lookup(text)
        # A handed-off flow is finished from the customer's side, so a new question doesn't pause it.
        detour = active is not None and active.get("intent") != "info" and not handed_off
        if hit and paused and hit[0]["id"] == paused["id"]:
            # "Back to the package": pick the paused flow up where it stopped, not from step 1.
            return self._held_step(paused, paused_step, back=True)
        if hit and (active is None or hit[0]["id"] != active["id"]):
            if detour and paused is None:
                paused, paused_step = active, next_step
            entry, start, signal, switched = hit[0], 0, None, True
        elif active:
            if signal is None and QUESTION.search(text.lower()) and not handed_off:
                return self._held_step(
                    active, next_step, kind="aside", paused=(paused["id"], paused_step) if paused else None
                )
            entry, start, switched = active, next_step, False
        else:
            return None

        turn = self._advance(entry, start, signal, switched)
        turn.detour = switched and paused is not None
        if paused is not None:
            if turn.kind == "steps" and turn.next_step is not None:
                turn.paused = (paused["id"], paused_step)
            else:
                # The side flow is finished, so go straight back to the one we paused.
                turn.resume = self._held_step(paused, paused_step)
        return turn

    def _advance(self, entry: dict, start: int, signal: Optional[str], switched: bool) -> FaqTurn:
        steps = entry["steps"]
        # next_step past the end marks a handed-off call; keeping it means the FAQ stays active.
        handed_off = len(steps) + 1
        if signal == "resolved":
            return FaqTurn(entry, "resolved", step=start, signal=signal)
        if start >= handed_off:
            return FaqTurn(entry, "handed_off", step=start, next_step=handed_off, signal=signal)
        if signal == "blocked" and entry["intent"] == "troubleshoot" and not switched:
            return FaqTurn(entry, "not_resolved", step=start, next_step=handed_off, signal=signal)
        if start >= len(steps):
            # After the last step we asked "did that sort it?" (troubleshoot) or "anything to add?" (escalate),
            # so a plain "no" means not fixed for the first and "nothing more" for the second.
            unresolved = ("failed", "blocked", "no", None) if entry["intent"] == "troubleshoot" else ("failed", "blocked")
            if signal in unresolved:
                return FaqTurn(entry, "not_resolved", step=start, next_step=handed_off, signal=signal)
            return FaqTurn(entry, "resolved", step=start, signal=signal)
        if entry["intent"] == "info":
            return FaqTurn(entry, "steps", steps[start:], start, None, signal, switched)
        return FaqTurn(entry, "steps", [steps[start]], start, start + 1, signal, switched)


@lru_cache(maxsize=1)
def get_faq() -> Optional[FaqIndex]:
    return FaqIndex() if FAQ_PATH.exists() else None


def plan_turn(
    text: str,
    active_id: Optional[str] = None,
    next_step: int = 0,
    paused_id: Optional[str] = None,
    paused_step: int = 0,
) -> Optional[FaqTurn]:
    faq = get_faq()
    return faq.plan_turn(text, active_id, next_step, paused_id, paused_step) if faq else None
