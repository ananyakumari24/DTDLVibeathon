"""Backup demo path if LiveKit fails: replay a transcript turn by turn through the live chain."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from oneai.realtime.classify import get_live_classifier
from oneai.realtime.reply import suggest_reply


def replay(path: Path, call_id: str, delay: float) -> None:
    raw = next(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["call_id"] == call_id
    )
    live = get_live_classifier()
    said_so_far = []
    replies = []
    for turn in raw["turns"]:
        if turn["speaker"] != "customer":
            continue
        # Classify the whole customer side so far; a lone follow-up line rarely names the issue.
        said_so_far.append(turn["text"])
        pack, score, ms = live.classify_debug(" ".join(said_so_far))
        reply = suggest_reply(turn["text"], pack, replies)
        replies.append(reply)
        print(f"\nCUSTOMER: {turn['text']}")
        print(f"  pack: {pack.pack_id} (similarity {score:.2f}, {ms:.1f} ms{', placeholder' if pack.is_placeholder else ''})")
        print(f"  BOT:  {reply}")
        time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a transcript through classify -> suggest_reply")
    parser.add_argument("--in", dest="infile", type=Path, default=Path("data/heldout/labelled.jsonl"))
    parser.add_argument("--call-id", default="heldout_01")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()
    replay(args.infile, args.call_id, args.delay)


if __name__ == "__main__":
    main()
