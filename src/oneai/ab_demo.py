"""A/B demo: run the same calls through a plain bot and the Pattern-Pack bot, then score both.

Each reply is checked against four things a good support reply does. The engagement score is
the share of checks passed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from oneai import llm
from oneai.eval import HELDOUT_PATH, load_heldout
from oneai.offline.agent import EMPATHY_WORDS, REPEAT_WORDS
from oneai.offline.cluster import PREFERRED_STYLE
from oneai.realtime.classify import get_live_classifier
from oneai.realtime.reply import APOLOGY_WORDS, plain_reply, suggest_reply
from oneai.realtime.tone import customer_tone
from oneai.taxonomy import Conversation

AB_REPORT_PATH = Path("data/ab_report.json")

ISSUE_WORDS = {
    "billing": ("bill", "charge", "card", "refund", "payment", "month", "fee"),
    "shipping": ("order", "package", "delivery", "tracking", "box", "shipment"),
    "account_access": ("password", "email", "account", "log in", "login", "reset"),
}
REPEAT_ASKS = ("tell me more", "describe", "repeat", "explain the problem", "what seems to be")
ACTION_WORDS = ("i'll", "i will", "refund", "escalat", "replacement", "sent", "taking care", "fix")
CHECKS = ("right_tone", "on_topic", "no_repeat_burden", "moves_forward")


def score_reply(reply: str, conv: Conversation, turn_text: str) -> Dict[str, bool]:
    r = reply.lower()
    turn = turn_text.lower()
    wants_empathy = PREFERRED_STYLE[conv.intent] == "empathic"
    if customer_tone(turn_text) != "neutral":
        # A complaining customer should hear an apology before anything else.
        right_tone = bool(APOLOGY_WORDS.search(re.split(r"(?<=[.!?])\s+", reply.strip())[0]))
    elif wants_empathy:
        right_tone = any(w in r for w in EMPATHY_WORDS)
    else:
        right_tone = not r.startswith(("i'm sorry", "sorry"))
    return {
        "right_tone": right_tone,
        "on_topic": any(w in r for w in ISSUE_WORDS[conv.issue]),
        "no_repeat_burden": not (any(w in turn for w in REPEAT_WORDS) and any(a in r for a in REPEAT_ASKS)),
        "moves_forward": "?" in r or any(w in r for w in ACTION_WORDS),
    }


def run_call(conv: Conversation) -> Dict:
    live = get_live_classifier()
    said_so_far: List[str] = []
    plain_history: List[str] = []
    pack_history: List[str] = []
    turns = []
    for turn in conv.turns:
        if turn.speaker != "customer":
            continue
        said_so_far.append(turn.text)
        pack = live.classify(" ".join(said_so_far))
        plain = plain_reply(turn.text, plain_history)
        guided = suggest_reply(turn.text, pack, pack_history)
        plain_history.append(plain)
        pack_history.append(guided)
        turns.append(
            {
                "customer": turn.text,
                "pack_id": pack.pack_id,
                "plain_reply": plain,
                "pack_reply": guided,
                "plain_checks": score_reply(plain, conv, turn.text),
                "pack_checks": score_reply(guided, conv, turn.text),
            }
        )
    return {"call_id": conv.call_id, "turns": turns}


def _summary(calls: List[Dict], key: str) -> Dict:
    checks = [t[key] for c in calls for t in c["turns"]]
    rates = {name: round(sum(c[name] for c in checks) / len(checks), 3) for name in CHECKS}
    return {"engagement_score": round(sum(rates.values()) / len(CHECKS), 3), "checks": rates, "replies": len(checks)}


def run(path: Path = HELDOUT_PATH) -> Dict:
    calls = [run_call(r["conv"]) for r in load_heldout(path)]
    return {
        "reply_source": "llm" if llm.available() else "templates (no OPENAI_API_KEY)",
        "plain_bot": _summary(calls, "plain_checks"),
        "pattern_pack_bot": _summary(calls, "pack_checks"),
        "calls": calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plain bot vs Pattern-Pack bot on held-out calls")
    parser.add_argument("--in", dest="infile", type=Path, default=HELDOUT_PATH)
    parser.add_argument("--out", type=Path, default=AB_REPORT_PATH)
    args = parser.parse_args()
    report = run(args.infile)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Replies from: {report['reply_source']}")
    for bot in ("plain_bot", "pattern_pack_bot"):
        s = report[bot]
        print(f"{bot:18s} engagement {s['engagement_score']:.0%}  {s['checks']}")


if __name__ == "__main__":
    main()
