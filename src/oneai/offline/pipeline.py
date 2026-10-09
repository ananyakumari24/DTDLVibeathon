"""End-to-end pipeline: ingest calls, then build groups, Pattern Packs and scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from oneai import ab_demo
from oneai import eval as evaluation
from oneai.offline.agent import label_agent
from oneai.offline.cluster import build_clusters, save_clusters
from oneai.offline.normalize import load_conversations, normalize_transcript
from oneai.offline.packs import build_all, save_packs
from oneai.offline.understand import understand
from oneai.realtime.classify import reset_live_classifier
from oneai.taxonomy import Insight, Transcript


def ingest(transcripts: List[Transcript]) -> List[Insight]:
    insights = []
    for transcript in transcripts:
        conv = normalize_transcript(transcript)
        understood = understand(conv)
        agent = label_agent(conv)
        insights.append(
            Insight(
                call_id=conv.call_id,
                issue=understood["issue"],
                question_type=understood["question_type"],
                agent_style=agent["agent_style"],
                friction_flags=agent["friction_flags"],
                label_methods={"understand": understood["method"], "agent": agent["method"]},
            )
        )
    return insights


def run_all(transcripts_path: Path) -> None:
    conversations = load_conversations(transcripts_path)
    clusters = build_clusters(conversations)
    save_clusters(clusters)
    print(f"Groups: {len(clusters)} from {len(conversations)} calls")

    packs = build_all(clusters, conversations)
    save_packs(packs)
    print(f"Pattern Packs: {len(packs)} (built with {packs[0].built_with})")

    reset_live_classifier()
    report = evaluation.run()
    evaluation.REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    live = report["live_classifier"]
    print(f"Held-out: pack routing {live['pack_routing_accuracy']:.0%}, p95 {live['latency_ms_p95']} ms")

    ab = ab_demo.run()
    ab_demo.AB_REPORT_PATH.write_text(json.dumps(ab, indent=2) + "\n", encoding="utf-8")
    print(
        f"A/B engagement: plain {ab['plain_bot']['engagement_score']:.0%}, "
        f"Pattern-Pack {ab['pattern_pack_bot']['engagement_score']:.0%} ({ab['reply_source']})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build groups, Pattern Packs and scores end to end")
    parser.add_argument("--in", dest="infile", type=Path, default=Path("data/synthetic/transcripts.jsonl"))
    args = parser.parse_args()
    run_all(args.infile)


if __name__ == "__main__":
    main()
