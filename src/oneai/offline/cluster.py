"""Embed conversations and group them along 3 axes: intent x issue x style."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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


def choose_k(vecs: np.ndarray, k_min: int = 2, k_max: int = 30) -> int:
    """Auto-pick the number of K-means clusters via silhouette score over a small range."""
    upper = min(k_max, len(vecs) - 1)
    if upper < k_min:
        return max(1, len(vecs))
    best_k, best_score = k_min, -1.0
    for k in range(k_min, upper + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(vecs)
        score = silhouette_score(vecs, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def build_clusters(conversations: List[Conversation], method: str = "auto", k: int | None = None) -> List[Cluster]:
    """Cluster conversations by embedding similarity (K-means).

    K-means only finds structure that is already present in the offline training set. If the
    transcripts only cover billing / shipping / account_access, you will not get a "mobile
    network" group — that has to appear in the data (or be rejected at live classify time).
    When several semantic sub-topics share the same majority issue x intent x style labels,
    they become distinct packs with ``__2``, ``__3``, … suffixes. Display labels still come
    from a majority vote so Pattern Pack building and the dashboard keep working.
    """
    vecs = embed_many([c.customer_text for c in conversations])
    n_clusters = k or choose_k(vecs)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(vecs)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        groups[int(label)].append(i)

    labels_cache: Dict[int, Dict[str, str]] = {}
    styles_cache: Dict[int, str] = {}

    def issue_intent(i: int) -> Dict[str, str]:
        if i not in labels_cache:
            labels_cache[i] = understand(conversations[i], method=method)
        return labels_cache[i]

    def style(i: int) -> str:
        if i not in styles_cache:
            styles_cache[i] = extract_agent_style(conversations[i])
        return styles_cache[i]

    clusters: List[Cluster] = []
    seen_ids: Counter = Counter()
    for cluster_idx, idxs in sorted(groups.items()):
        issue_votes = Counter(issue_intent(i)["issue"] for i in idxs)
        intent_votes = Counter(issue_intent(i)["question_type"] for i in idxs)
        style_votes = Counter(style(i) for i in idxs)
        issue = issue_votes.most_common(1)[0][0]
        intent = intent_votes.most_common(1)[0][0]
        group_style = style_votes.most_common(1)[0][0]

        base_id = cluster_id(issue, intent, group_style)
        seen_ids[base_id] += 1
        unique_id = base_id if seen_ids[base_id] == 1 else f"{base_id}__{seen_ids[base_id]}"

        centroid = np.asarray(kmeans.cluster_centers_[cluster_idx], dtype=np.float32)
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
                id=unique_id,
                intent=intent,
                issue=issue,
                style=group_style,
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
    parser = argparse.ArgumentParser(description="Embed conversations and cluster them with K-means")
    parser.add_argument("--in", dest="infile", type=Path, default=Path("data/synthetic/transcripts.jsonl"))
    parser.add_argument("--out", type=Path, default=CLUSTERS_PATH)
    parser.add_argument("--method", default="auto", choices=["auto", "llm", "zeroshot", "rules"])
    parser.add_argument("--k", type=int, default=None, help="Number of K-means clusters (default: auto via silhouette score)")
    args = parser.parse_args()

    conversations = load_conversations(args.infile)
    clusters = build_clusters(conversations, method=args.method, k=args.k)
    save_clusters(clusters, args.out)
    print(f"Built {len(clusters)} groups from {len(conversations)} calls ({backend()} embeddings) -> {args.out}")
    for c in clusters:
        print(f"  {c.id:45s} {len(c.member_call_ids):4d} calls")


if __name__ == "__main__":
    main()
