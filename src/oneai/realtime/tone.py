"""Reads the customer's tone and spots when the call can end.

customer_tone: "upset" (angry, fed up, repeat contact), "complaint" (reports something wrong)
or "neutral". Any non-neutral turn gets an apology before anything else in the reply.
closing_signal: "solved" (it worked), "bye" (that's all / goodbye) or None, so the bot can thank
the customer and end the call instead of asking "anything else?" forever.
"""

from __future__ import annotations

import re
from typing import Optional

from oneai.realtime.faq import progress_signal

UPSET = re.compile(
    r"frustrat|annoy|ridiculous|\bangry\b|furious|\bupset\b|unacceptable|terrible|awful|horrible|\bworst\b"
    r"|useless|disgust|fed up|sick of|tired of|waste of|disappoint|outrag|pathetic|rubbish|nightmare|fuming"
    r"|\blivid\b|\bjoke\b|not happy|unhappy|third time|second time|(\d+|three|four|five|several|many|multiple) times"
    r"|how many times|already (said|explained|told|called|tried|did)|again and again|happened again"
    r"|nobody|no one (has|is|will)|still (waiting|haven'?t|hasn'?t)|for (days|weeks|ages|hours)"
    r"|\b(manager|supervisor)\b"
)
COMPLAINT = re.compile(
    r"not working|(isn'?t|aren'?t|doesn'?t|don'?t|stopped|won'?t|not) (work|working|connect|load|start|open|turn on|charge)"
    r"|(can'?t|cannot|unable to|couldn'?t) (log|sign|get|access|connect|make|use|pay|receive|send|find|see|open|call|hear)"
    r"|broken|damaged|faulty|\bwrong\b|incorrect|charged twice|double charged|overcharg|extra charge"
    r"|never (arrived|came|got|received|showed)|hasn'?t (arrived|come)|missing|\blost\b|\blate\b|delayed"
    r"|no (signal|service|network|internet|connection|data|bars)|dropp(ed|ing) calls?|keeps? (dropping|crashing|failing|freezing|disconnecting)"
    r"|crash|\bfail|declined|locked out|\berror\b|problem with|issue with|complain|\bpoor\b|\bslow\b"
    r"|still (not|no\b|nothing|the same|broken|down)|no luck|didn'?t (work|help)|did not (work|help)"
)
# Shouting reads as upset: two or more all-caps words of 4+ letters (so SIM, SMS, PIN don't count), or "!!".
SHOUT = re.compile(r"(\b[A-Z]{4,}\b.*){2,}|!{2,}")

BYE = re.compile(
    r"that'?s (all|everything)\b|nothing else\b(?! (works?|worked|helps?|helped|happen))|that'?ll be all|no(pe)?,? i'?m (good|fine|all set)"
    r"|i'?m (all )?(good|set|sorted) now|\b(bye|goodbye)\b|have a (good|nice|great|lovely) (day|one|evening|night)"
)
SOLVED = re.compile(
    r"that (worked|helped|fixed it|did it|did the trick|sorted it)|\bit (worked|works)\b|it'?s (working|fixed|sorted|resolved)"
    r"|working (now|again|fine)|(all|problem|issue|that'?s) (fixed|sorted|solved|resolved)|(fixed|sorted|solved|resolved) (it|now)"
    r"|you'?ve been (very |so |really )?helpful|thanks for (your|the|all the) help|appreciate (it|your help|the help)"
)
THANKS = re.compile(
    r"^\W*((ok(ay)?|great|perfect|cool|awesome|brilliant|lovely|got it|alright|right|wonderful)\W+)*"
    r"(thanks|thank you|cheers)(\W+(so|very) much)?\W*$"
)
# Words that turn "it worked" into "it hasn't worked" or add a new request.
CONTRARY = re.compile(r"n'?t\b|\b(not|never|still|but|however|also|another|one more|nobody|no one|stopped|no longer)\b")


def customer_tone(text: str) -> str:
    if SHOUT.search(text) or UPSET.search(text.lower()):
        return "upset"
    if COMPLAINT.search(text.lower()):
        return "complaint"
    return "neutral"


def closing_signal(text: str, mid_steps: bool = False) -> Optional[str]:
    """'solved' or 'bye' when the customer is done, else None.

    A bare "thanks" mid-way through FAQ steps just acknowledges the step, so it only ends the
    call when no steps are in progress.
    """
    lowered = text.lower()
    if "?" in lowered or progress_signal(lowered) == "failed":
        return None
    if BYE.search(lowered):
        return "bye"
    if CONTRARY.search(lowered):
        return None
    if SOLVED.search(lowered):
        return "solved"
    if not mid_steps and THANKS.search(lowered):
        return "bye"
    return None
