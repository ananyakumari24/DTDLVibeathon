"""Day 1 Person A smoke test: load → normalize → extractors."""

from __future__ import annotations

import argparse
from pathlib import Path

from oneai.offline.extractors import annotate_conversation, evaluate_against_labels
from oneai.offline.normalize import load_conversations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in",
        dest="infile",
        type=Path,
        default=Path("data/synthetic/transcripts.jsonl"),
    )
    args = parser.parse_args()

    conversations = [annotate_conversation(c) for c in load_conversations(args.infile)]
    metrics = evaluate_against_labels(conversations)

    sample = conversations[0]
    print("=== Day 1 Person A report ===")
    print(f"Transcripts loaded: {int(metrics['n'])}")
    print(f"Extractor accuracy vs synthetic labels: {metrics}")
    print()
    print("=== Sample conversation ===")
    print(f"call_id: {sample.call_id}")
    print(f"labels: issue={sample.issue} intent={sample.intent} style={sample.style}")
    print(f"customer_text: {sample.customer_text[:120]}...")
    print(f"friction_flags: {sample.friction_flags}")
    friction_count = sum(1 for c in conversations if c.friction_flags)
    print(f"Calls with friction detected: {friction_count}")


if __name__ == "__main__":
    main()
