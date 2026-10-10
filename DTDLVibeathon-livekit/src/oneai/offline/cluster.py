"""Embed conversations and group them along 3 axes: intent x issue x style."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from oneai.offline.agent import extract_agent_style
from oneai.offline.embeddings import backend, embed_many
from oneai.offline.normalize import load_conversations
from oneai.offline.understand import understand
from oneai.taxonomy import Cluster, Conversation

CLUSTERS_PATH = Path("data/clusters.json")
PACKS_DIR = Path("data/packs")

PREFERRED_STYLE = {"troubleshoot": "empathic", "escalate": "empathic", "info": "instructional"}


def cluster_id(issue: str, intent: str, style: str) -> str:
    return f"{issue}__{intent}__{style}"


def build_clusters(conversations: List[Conversation], method: str = "auto") -> List[Cluster]:
    vecs = embed_many([c.customer_text for c in conversations])
    groups: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for i, conv in enumerate(conversations):
        labels = understand(conv, method=method)
        groups[(labels["issue"], labels["question_type"], extract_agent_style(conv))].append(i)

    clusters: List[Cluster] = []
    for (issue, intent, style), idxs in sorted(groups.items()):
        centroid = vecs[idxs].mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-9)
        phrasings: List[str] = []
        for i in idxs:
            opener = conversations[i].turns[0].text
            if opener not in phrasings:
                phrasings.append(opener)
            if len(phrasings) == 5:
                break
        clusters.append(
            Cluster(
                id=cluster_id(issue, intent, style),
                intent=intent,
                issue=issue,
                style=style,
                member_call_ids=[conversations[i].call_id for i in idxs],
                centroid=centroid.round(5).tolist(),
                sample_phrasings=phrasings,
            )
        )
    return clusters


def save_clusters(clusters: List[Cluster], path: Path = CLUSTERS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"embedding_backend": backend(), "clusters": [c.model_dump() for c in clusters]}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def load_clusters(path: Path = CLUSTERS_PATH) -> List[Cluster]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("embedding_backend") != backend():
        raise RuntimeError(
            f"{path} was built with '{payload.get('embedding_backend')}' embeddings but this "
            f"process uses '{backend()}'. Rebuild with: python -m oneai.offline.pipeline"
        )
    return [Cluster.model_validate(c) for c in payload["clusters"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed conversations and group by intent x issue x style")
    parser.add_argument("--in", dest="infile", type=Path, default=Path("data/synthetic/transcripts.jsonl"))
    parser.add_argument("--out", type=Path, default=CLUSTERS_PATH)
    parser.add_argument("--method", default="auto", choices=["auto", "llm", "zeroshot", "rules"])
    args = parser.parse_args()

    conversations = load_conversations(args.infile)
    clusters = build_clusters(conversations, method=args.method)
    save_clusters(clusters, args.out)
    print(f"Built {len(clusters)} groups from {len(conversations)} calls ({backend()} embeddings) -> {args.out}")
    for c in clusters:
        print(f"  {c.id:45s} {len(c.member_call_ids):4d} calls")


if __name__ == "__main__":
    main()
