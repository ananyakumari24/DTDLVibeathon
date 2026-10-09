"""End-to-end scenario checks for the live reply flow: routing, live packs, tone, endings, detours.

Runs the FastAPI app in-process (no server needed) and writes live packs to a temp folder, so
data/ is never touched. Run from the repo root:

    PYTHONPATH=src .venv/bin/python scripts/scenario_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

os.environ.pop("OPENAI_API_KEY", None)  # template replies, so the checks are deterministic

from fastapi.testclient import TestClient  # noqa: E402

import oneai.realtime.classify as classify  # noqa: E402
from oneai.realtime import app as api  # noqa: E402

classify._live = classify.LiveClassifier(live_dir=Path(tempfile.mkdtemp()))
client = TestClient(api.app)

STATE = ("faq_id", "faq_step", "paused_faq_id", "paused_faq_step")
Check = Callable[[dict], Optional[str]]  # returns a failure message, or None if it passed


def call(turns: List[Tuple[str, List[Check]]]) -> List[str]:
    """Play one call turn by turn, carrying state like the dashboard does; return failures."""
    said: List[str] = []
    history: List[str] = []
    state: dict = {}
    pack_id = None
    failures = []
    for text, checks in turns:
        r = client.post(
            "/reply", json={"text": text, "said_before": said, "history": history, "pack_id": pack_id, **state}
        ).json()
        print(f"    C: {text}")
        print(f"       [{r['tone']}{' · ENDED' if r['call_ended'] else ''}] {r['pack_id']} · {r['faq_used']}")
        print(f"    B: {r['pack_reply']}")
        for check in checks:
            problem = check(r)
            if problem:
                failures.append(f"{text!r}: {problem}")
                print(f"       ✗ {problem}")
        said.append(text)
        history.append(r["pack_reply"])
        state = {k: r[k] for k in STATE}
        pack_id = r["pack_id"]
    return failures


# Small check builders.
def pack(prefix: str) -> Check:
    return lambda r: None if r["pack_id"].startswith(prefix) else f"pack {r['pack_id']} should start with {prefix}"


def created(expected: bool = True) -> Check:
    return lambda r: None if r["pack_created"] == expected else f"pack_created should be {expected}"


def tone(expected: str) -> Check:
    return lambda r: None if r["tone"] == expected else f"tone {r['tone']} should be {expected}"


def ended(expected: bool = True) -> Check:
    return lambda r: None if r["call_ended"] == expected else f"call_ended should be {expected}"


def starts(*words: str) -> Check:
    return lambda r: None if r["pack_reply"].lower().startswith(tuple(w.lower() for w in words)) else f"reply should start with {words}"


def says(text: str) -> Check:
    return lambda r: None if text.lower() in r["pack_reply"].lower() else f"reply should contain {text!r}"


def not_says(text: str) -> Check:
    return lambda r: None if text.lower() not in r["pack_reply"].lower() else f"reply should not contain {text!r}"


def faq(text: str) -> Check:
    return lambda r: None if text in (r["faq_used"] or "") else f"faq_used {r['faq_used']!r} should contain {text!r}"


def one_sorry(r: dict) -> Optional[str]:
    n = r["pack_reply"].lower().count("sorry")
    return None if n <= 1 else f"reply says sorry {n} times"


SCENARIOS = {
    # --- Routing and live packs ---
    "Known topic uses a mined pack": [
        ("My card was charged twice for the same order.", [pack("billing__"), created(False), faq("faq_07")]),
    ],
    "Vague opener clarifies, no pack made": [
        ("hi, I need some help", [pack("unknown__"), created(False)]),
    ],
    "New topic creates a live pack and the call stays on it": [
        ("my mobile network is not working", [pack("live__mobile_network"), created()]),
        ("yes I restarted the phone", [pack("live__mobile_network"), created(False)]),
        ("still nothing", [pack("live__mobile_network")]),
    ],
    "Same topic on a new call reuses the live pack": [
        ("I have no mobile network signal on my phone", [pack("live__mobile_network"), created(False)]),
    ],
    "A clear topic switch leaves the live pack": [
        ("my mobile network is not working", [pack("live__mobile_network")]),
        ("I cannot log into my account", [pack("account_access__")]),
    ],
    # --- Tone: apology first when complaining ---
    "Upset customer hears a strong apology first": [
        ("This is the third time I'm calling, I was charged twice and it's ridiculous!",
         [tone("upset"), starts("I'm really sorry"), one_sorry]),
    ],
    "Complaint gets an apology first": [
        ("my package never arrived", [tone("complaint"), starts("I'm", "sorry"), says("sorry"), one_sorry]),
    ],
    "Neutral question gets no apology": [
        ("When is my next billing date?", [tone("neutral"), not_says("sorry")]),
    ],
    # --- Ending the call ---
    "Resolved after the last step ends with thanks": [
        ("My card was charged twice for the same order.", []),
        ("done", [ended(False)]),
        ("yes that fixed it, thanks", [ended(), says("thank"), says("glad")]),
    ],
    "Thanks after an info answer ends the call": [
        ("When is my next billing date?", [ended(False)]),
        ("ok thank you so much", [ended(), says("thanks for reaching out")]),
    ],
    "Thanks mid-steps does not end the call": [
        ("My card was charged twice for the same order.", []),
        ("thanks", [ended(False)]),
    ],
    "Failed fix gives the fallback, not a goodbye": [
        ("My card was charged twice for the same order.", []),
        ("done", []),
        ("no, it didn't work", [ended(False), says("billing team"), not_says("glad")]),
    ],
    # --- Negative turns: never restart the FAQ or end the call ---
    "Negated fix doesn't end the call": [
        ("This is the third time I'm calling, I got charged twice and nobody has fixed it!!",
         [tone("upset"), ended(False), faq("faq_07 · troubleshoot · step 1/2"), not_says("glad")]),
    ],
    "'Worked then stopped' is a failure, not a goodbye": [
        ("I forgot my password", []),
        ("it worked for a day then stopped", [ended(False), not_says("glad")]),
    ],
    "Stuck on a step goes to the fallback, not the next step": [
        ("I forgot my password", [faq("step 1/3")]),
        ("I didn't receive the link", [faq("faq_02 · troubleshoot · not resolved"), says("verify you"),
                                       not_says("open the reset link"), ended(False)]),
    ],
    "Pushback after the fallback restates it instead of restarting": [
        ("I forgot my password", []),
        ("done", []),
        ("done", []),
        ("no", [faq("not resolved"), says("verify you")]),
        ("I still can't log in", [faq("faq_02 · troubleshoot · handed off"), says("won't send you through those steps"),
                                  not_says("tap forgot password"), ended(False)]),
        ("no one has fixed it", [faq("handed off"), ended(False), not_says("glad")]),
        ("ok thanks, bye", [ended(), says("team will keep you updated")]),
    ],
    "Follow-ups after the fallback get answers, not the fallback again": [
        ("My card was charged twice for the same order.", []),
        ("done", []),
        ("no, it didn't work", [says("billing team")]),
        ("how long will that take?", [faq("faq_07 · troubleshoot · handed off"), says("add that to"),
                                      not_says("won't repeat"), not_says("billing team")]),
        ("what's my reference number?", [faq("faq_07 · troubleshoot · handed off"), not_says("transfer code")]),
        ("ok great", [says("anything else"), not_says("won't repeat"), ended(False)]),
        ("no", [ended(), not_says("all set")]),
    ],
    "A new problem after the fallback still starts its own FAQ": [
        ("My card was charged twice for the same order.", []),
        ("done", []),
        ("no, it didn't work", [faq("not resolved")]),
        ("my payment was declined too", [faq("faq_09 · troubleshoot · step 1/3"), not_says("back to where we were")]),
    ],
    "Stuck mid-escalation moves the case on": [
        ("Order says delivered but I don't have it", [faq("faq_14 · escalate · step 1/3")]),
        ("no, the neighbours don't have it", [faq("faq_14 · escalate · step 2/3")]),
    ],
    # --- Going off-topic mid-call ---
    "Small talk holds the current step": [
        ("my mobile network is not working", [faq("step 1/3")]),
        ("ok done", [faq("step 2/3")]),
        ("by the way, how's your day going?", [faq("holding step 2/3"), says("finish this step first")]),
        ("ok done", [faq("step 3/3"), ended(False)]),
    ],
    "Unrelated question holds the current step": [
        ("My card was charged twice for the same order.", [faq("step 1/2")]),
        ("can I bring my dog on the train?", [faq("holding step 1/2")]),
        ("done", [faq("step 2/2"), ended(False)]),
    ],
    "Side question pauses the flow and comes back after": [
        ("I can't log into my account", [faq("faq_02 · troubleshoot · step 1/3")]),
        ("how do I update my email address?", [faq("faq_02 paused"), starts("Sure"), not_says("sorry about the trouble")]),
        ("done", [faq("faq_02 paused")]),
        ("done", [faq("faq_02 paused")]),
        ("yes", [faq("back to faq_02 step 1/3"), says("back to where we were"), ended(False)]),
        ("yes that worked", [ended(), says("thank you for reaching out")]),
    ],
    "Quick info question mid-flow answers and returns at once": [
        ("my package never arrived", [faq("faq_14 · escalate · step 1/3")]),
        ("also my bill looks wrong", [faq("back to faq_14 step 1/3"), says("back to where we were")]),
        ("ok done", [faq("faq_14 · escalate · step 2/3")]),
    ],
    "Customer asks to go back to the paused flow": [
        ("I can't log into my account", []),
        ("how do I update my email address?", [faq("faq_02 paused")]),
        ("done", []),
        ("actually, let's go back to resetting my password", [faq("faq_02 · troubleshoot · step 1/3"), says("back to where we were")]),
    ],
    "Goodbye during a side flow ends the call": [
        ("I can't log into my account", []),
        ("how do I update my email address?", []),
        ("that's all, bye", [ended()]),
    ],
    "'Ok, and how do I...?' starts the new question": [
        ("my mobile network is not working", [faq("faq_16")]),
        ("ok and how do I reset my password?", [faq("faq_02"), faq("faq_16 paused")]),
    ],
}


def main() -> int:
    failed = {}
    for name, turns in SCENARIOS.items():
        print(f"\n== {name}")
        problems = call(turns)
        if problems:
            failed[name] = problems
    live = client.get("/live-packs").json()
    network = next((p for p in live if p["pack_id"].startswith("live__mobile_network")), None)
    if network is None or network["call_count"] < 2:
        failed["Live pack learns from later calls"] = [f"mobile_network pack: {network}"]
    print(f"\nLive packs created: {[(p['pack_id'], p['call_count']) for p in live]}")
    print(f"\n{len(SCENARIOS) - len(failed) + ('Live pack learns from later calls' not in failed)}"
          f"/{len(SCENARIOS) + 1} scenarios passed")
    for name, problems in failed.items():
        print(f"  FAIL {name}")
        for p in problems:
            print(f"       {p}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
