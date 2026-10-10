"""Live classifier: classify(turn_text) -> PatternPack, using cached group centres."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from oneai.offline.cluster import CLUSTERS_PATH, PACKS_DIR, PREFERRED_STYLE, cluster_id, load_clusters
from oneai.offline.embeddings import embed_many
from oneai.taxonomy import Cluster, PatternPack


def placeholder_pack(cluster: Cluster) -> PatternPack:
    return PatternPack(
        pack_id=cluster.id,
        intent=cluster.intent,
        issue=cluster.issue,
        style=cluster.style,
        top_customer_phrasings=cluster.sample_phrasings,
        recommended_opener="I can help with that.",
        call_count=len(cluster.member_call_ids),
        is_placeholder=True,
    )


class LiveClassifier:
    """Embeds a live turn, picks the nearest group and returns its Pattern Pack.

    Group centres and loaded packs are cached in memory so a turn costs one embedding call.
    """

    def __init__(self, clusters_path: Path = CLUSTERS_PATH, packs_dir: Path = PACKS_DIR):
        self.clusters = load_clusters(clusters_path)
        self.by_id = {c.id: c for c in self.clusters}
        self.centroids = np.asarray([c.centroid for c in self.clusters], dtype=np.float32)
        self.packs_dir = packs_dir
        self._packs: Dict[str, PatternPack] = {}
        embed_many(["warm up"])

    def _pack_for(self, cluster: Cluster) -> PatternPack:
        if cluster.id not in self._packs:
            path = self.packs_dir / f"{cluster.id}.json"
            if path.exists():
                self._packs[cluster.id] = PatternPack.model_validate_json(path.read_text(encoding="utf-8"))
            else:
                self._packs[cluster.id] = placeholder_pack(cluster)
        return self._packs[cluster.id]

    def classify_debug(self, turn_text: str) -> Tuple[PatternPack, float, float]:
        start = time.perf_counter()
        v = embed_many([turn_text])[0]
        sims = self.centroids @ v
        best = self.clusters[int(np.argmax(sims))]
        # Customer words don't tell agent styles apart, so pick the recommended style for the cell.
        preferred = self.by_id.get(cluster_id(best.issue, best.intent, PREFERRED_STYLE[best.intent]))
        chosen = preferred or best
        score = float(sims[self.clusters.index(chosen)])
        ms = (time.perf_counter() - start) * 1000
        return self._pack_for(chosen), score, ms

    def classify(self, turn_text: str) -> PatternPack:
        return self.classify_debug(turn_text)[0]


_live: Optional[LiveClassifier] = None


def get_live_classifier() -> LiveClassifier:
    global _live
    if _live is None:
        _live = LiveClassifier()
    return _live


def reset_live_classifier() -> None:
    global _live
    _live = None


def classify(turn_text: str) -> PatternPack:
    return get_live_classifier().classify(turn_text)
