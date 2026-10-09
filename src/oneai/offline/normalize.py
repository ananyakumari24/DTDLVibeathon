"""Load JSONL transcripts and normalize to Conversation objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from oneai.taxonomy import Conversation, Transcript


def normalize_transcript(t: Transcript, friction_flags: List[str] | None = None) -> Conversation:
    customer_parts = [turn.text for turn in t.turns if turn.speaker == "customer"]
    agent_parts = [turn.text for turn in t.turns if turn.speaker == "agent"]
    full = " ".join(f"{turn.speaker}: {turn.text}" for turn in t.turns)
    return Conversation(
        call_id=t.call_id,
        issue=t.issue,
        intent=t.intent,
        style=t.style,
        turns=t.turns,
        customer_text=" ".join(customer_parts),
        agent_text=" ".join(agent_parts),
        full_text=full,
        friction_flags=friction_flags or [],
    )


def load_transcripts(path: Path) -> List[Transcript]:
    rows: List[Transcript] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(Transcript.model_validate_json(line))
    return rows


def load_conversations(path: Path) -> List[Conversation]:
    return [normalize_transcript(t) for t in load_transcripts(path)]
