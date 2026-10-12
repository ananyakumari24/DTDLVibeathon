"""Live voice call log: the LiveKit agent writes each call here, the dashboard reads it.

The voice agent runs as its own worker process, so the two sides share an append-only JSONL file
rather than memory. One line per event: a call starting, a customer turn with its pack and the
bot's reply, or the call closing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from oneai.realtime.tone import customer_tone
from oneai.realtime.turn import Turn, describe_faq

CALL_LOG_PATH = Path("data/voice_calls.jsonl")


def _write(event: dict, path: Path = CALL_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # One write per line keeps concurrent calls (separate job processes) from interleaving.
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), **event}) + "\n")


def log_call_start(call_id: str, room: str, greeting: str) -> None:
    _write({"event": "start", "call_id": call_id, "room": room, "greeting": greeting})


def log_turn(call_id: str, text: str, turn: Turn) -> None:
    _write(
        {
            "event": "turn",
            "call_id": call_id,
            "said": text,
            "pack_id": turn.pack.pack_id,
            "similarity": round(turn.similarity, 4),
            "latency_ms": round(turn.latency_ms, 2),
            "is_placeholder": turn.pack.is_placeholder,
            "pack_created": turn.pack_created,
            "tone": customer_tone(text),
            "faq_used": describe_faq(turn.faq),
            "call_ended": turn.ended,
            "reply": turn.reply,
        }
    )


def log_call_closed(call_id: str, reason: Optional[str] = None) -> None:
    _write({"event": "closed", "call_id": call_id, "reason": reason})


def recent_calls(limit: int = 10, path: Path = CALL_LOG_PATH) -> List[dict]:
    """Newest calls first, each with its turns in order."""
    if not path.exists():
        return []
    calls: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line still being written
        call = calls.setdefault(
            event["call_id"],
            {"call_id": event["call_id"], "room": None, "started": event["ts"], "updated": event["ts"],
             "greeting": None, "status": "live", "turns": []},
        )
        call["updated"] = event["ts"]
        kind = event.pop("event")
        if kind == "start":
            call.update(room=event["room"], started=event["ts"], greeting=event["greeting"])
        elif kind == "turn":
            del event["call_id"]
            call["turns"].append(event)
            if event["call_ended"]:
                call["status"] = "resolved"
        elif kind == "closed" and call["status"] == "live":
            call["status"] = "closed"
    return sorted(calls.values(), key=lambda c: c["updated"], reverse=True)[:limit]
