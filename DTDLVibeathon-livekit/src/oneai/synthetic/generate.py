"""Seeded synthetic Sprinklr-like voice support transcripts (Day 1 Person A)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

from oneai.taxonomy import ISSUES, INTENTS, STYLES, Transcript, Turn

SEED = 42

CUSTOMER_OPENERS: Dict[Tuple[str, str], List[str]] = {
    ("billing", "troubleshoot"): [
        "I was charged twice for last month and I'm pretty frustrated.",
        "There's a duplicate charge on my card.",
        "My bill looks wrong — I got billed twice.",
    ],
    ("billing", "info"): [
        "Can you explain what this line item on my bill means?",
        "When is my next billing date?",
    ],
    ("billing", "escalate"): [
        "I've called three times about this bill and need a supervisor.",
        "Escalate my duplicate charge case now.",
    ],
    ("shipping", "troubleshoot"): [
        "My package was marked delivered but I never got it.",
        "The shipment is stuck and hasn't moved in days.",
    ],
    ("shipping", "info"): [
        "Where is my package? Tracking hasn't updated.",
        "What's the estimated delivery date for my order?",
    ],
    ("shipping", "escalate"): [
        "I've been waiting two weeks — escalate this shipping failure.",
        "Escalate my missing package claim immediately.",
    ],
    ("account_access", "troubleshoot"): [
        "I can't log in even after resetting my password.",
        "My account is locked and the unlock email never arrives.",
    ],
    ("account_access", "info"): [
        "How do I reset my password from the mobile app?",
        "Where do I manage two-factor authentication?",
    ],
    ("account_access", "escalate"): [
        "I've reset my password three times — escalate this.",
        "I'm locked out and need a supervisor now.",
    ],
}

CUSTOMER_FOLLOW = {
    "billing": ["It shows two charges of $49.99.", "Both hits were on the same Visa."],
    "shipping": ["Order number is 882114.", "Tracking hasn't moved since Friday."],
    "account_access": ["Email is yes@company.com.", "I already tried reset twice."],
}

EMPATHIC_OPENERS = [
    "I'm really sorry about that — this sounds stressful. I can help sort it out.",
    "I'm sorry you've had to deal with this. Let's fix it together.",
]
INSTRUCTIONAL_OPENERS = [
    "I can help with that. Let's go step by step.",
    "Understood. I'll walk you through what we need to check.",
]


def generate_one(call_idx: int, issue: str, intent: str, style: str, rng: random.Random) -> Transcript:
    opener = rng.choice(CUSTOMER_OPENERS[(issue, intent)])
    follow = rng.choice(CUSTOMER_FOLLOW[issue])
    agent_opener = rng.choice(EMPATHIC_OPENERS if style == "empathic" else INSTRUCTIONAL_OPENERS)
    turns: List[Turn] = [
        Turn(speaker="customer", text=opener, ts=0.0),
        Turn(speaker="agent", text=agent_opener, ts=3.0),
        Turn(speaker="customer", text=follow, ts=8.0),
        Turn(speaker="agent", text="Thanks — I'll continue from those details.", ts=12.0),
    ]
    if rng.random() < 0.3:
        turns.append(
            Turn(
                speaker="customer",
                text=rng.choice(
                    [
                        "I already explained this once.",
                        "Please don't make me repeat the whole story.",
                    ]
                ),
                ts=16.0,
            )
        )
    turns.append(Turn(speaker="customer", text="Okay, thank you.", ts=20.0))
    turns.append(
        Turn(
            speaker="agent",
            text="I'll take care of the next steps and confirm by email.",
            ts=24.0,
        )
    )
    return Transcript(
        call_id=f"call_{call_idx:05d}",
        issue=issue,  # type: ignore[arg-type]
        intent=intent,  # type: ignore[arg-type]
        style=style,  # type: ignore[arg-type]
        turns=turns,
    )


def generate_all(n: int = 360, seed: int = SEED) -> List[Transcript]:
    rng = random.Random(seed)
    combos = [(i, intent, s) for i in ISSUES for intent in INTENTS for s in STYLES]
    out: List[Transcript] = []
    idx = 1
    while len(out) < n:
        issue, intent, style = combos[(idx - 1) % len(combos)]
        out.append(generate_one(idx, issue, intent, style, rng))
        idx += 1
    return out


def write_jsonl(path: Path, transcripts: List[Transcript]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in transcripts:
            f.write(t.model_dump_json() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic OneAI transcripts")
    parser.add_argument("--out", type=Path, default=Path("data/synthetic/transcripts.jsonl"))
    parser.add_argument("--n", type=int, default=360)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    transcripts = generate_all(n=args.n, seed=args.seed)
    write_jsonl(args.out, transcripts)
    print(f"Wrote {len(transcripts)} transcripts → {args.out}")


if __name__ == "__main__":
    main()
