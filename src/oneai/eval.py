"""Accuracy, speed and unresolved-call scoring against held-out labelled calls."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List

from oneai import llm
from oneai.offline.agent import extract_agent_style
from oneai.offline.embeddings import backend
from oneai.offline.normalize import normalize_transcript
from oneai.offline.understand import understand_text
from oneai.realtime.classify import get_live_classifier
from oneai.taxonomy import Conversation, Transcript

HELDOUT_PATH = Path("data/heldout/labelled.jsonl")
REPORT_PATH = Path("data/eval_report.json")

RESOLUTION_WORDS = ["refund is issued", "you'll see it", "estimated delivery", "replacement has shipped", "resolved", "fixed"]
UNRESOLVED_WORDS = ["try again later", "escalated", "specialist", "call you back"]


def load_heldout(path: Path = HELDOUT_PATH) -> List[Dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw = json.loads(line)
            rows.append({"conv": normalize_transcript(Transcript.model_validate(raw)), "resolved": raw["resolved"]})
    return rows


def predict_unresolved(conv: Conversation) -> bool:
    last_agent = next((t.text.lower() for t in reversed(conv.turns) if t.speaker == "agent"), "")
    if any(w in last_agent for w in UNRESOLVED_WORDS):
        return True
    return not any(w in last_agent for w in RESOLUTION_WORDS)


def score_labelling(rows: List[Dict], method: str) -> Dict:
    issue_ok = intent_ok = 0
    for row in rows:
        conv = row["conv"]
        pred = understand_text(conv.customer_text, method=method)
        issue_ok += pred["issue"] == conv.issue
        intent_ok += pred["question_type"] == conv.intent
    n = len(rows)
    return {"issue_accuracy": round(issue_ok / n, 3), "intent_accuracy": round(intent_ok / n, 3)}


def score_live(rows: List[Dict], repeats: int = 20) -> Dict:
    live = get_live_classifier()
    routed_ok = 0
    latencies: List[float] = []
    for row in rows:
        conv = row["conv"]
        first_turn = conv.turns[0].text
        pack, _, _ = live.classify_debug(first_turn)
        # With K-means-derived clusters, several clusters can share an (issue, intent) pair, so
        # exact pack_id equality no longer makes sense. Check the routed pack's derived labels
        # instead: did we land on a pack with the right issue and intent.
        routed_ok += pack.issue == conv.issue and pack.intent == conv.intent
        for _ in range(repeats):
            latencies.append(live.classify_debug(first_turn)[2])
    latencies.sort()
    n = len(rows)
    return {
        "pack_routing_accuracy": round(routed_ok / n, 3),
        "misroute_rate": round(1 - routed_ok / n, 3),
        "latency_ms_p50": round(statistics.median(latencies), 2),
        "latency_ms_p95": round(latencies[int(0.95 * (len(latencies) - 1))], 2),
        "under_200ms": latencies[-1] < 200,
    }


def score_unresolved(rows: List[Dict]) -> Dict:
    correct = sum(predict_unresolved(r["conv"]) == (not r["resolved"]) for r in rows)
    actual = sum(not r["resolved"] for r in rows)
    predicted = sum(predict_unresolved(r["conv"]) for r in rows)
    n = len(rows)
    return {
        "unresolved_detection_accuracy": round(correct / n, 3),
        "actual_unresolved_rate": round(actual / n, 3),
        "predicted_unresolved_rate": round(predicted / n, 3),
    }


def run(path: Path = HELDOUT_PATH) -> Dict:
    rows = load_heldout(path)
    methods = ["rules", "zeroshot"] + (["llm"] if llm.available() else [])
    return {
        "heldout_calls": len(rows),
        "embedding_backend": backend(),
        "labelling": {m: score_labelling(rows, m) for m in methods},
        "agent_style_accuracy": round(sum(extract_agent_style(r["conv"]) == r["conv"].style for r in rows) / len(rows), 3),
        "live_classifier": score_live(rows),
        "unresolved": score_unresolved(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score labelling, routing and speed against held-out labelled calls")
    parser.add_argument("--in", dest="infile", type=Path, default=HELDOUT_PATH)
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    report = run(args.infile)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
