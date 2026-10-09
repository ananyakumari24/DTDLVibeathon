"""Live classifier: classify(turn_text) -> PatternPack, using cached group centres."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from oneai.offline.cluster import CLUSTERS_PATH, PACKS_DIR, PREFERRED_STYLE, load_clusters
from oneai.offline.embeddings import embed_many
from oneai.offline.understand import understand_text
from oneai.realtime.live_packs import LIVE_PACKS_DIR, LivePackStore, is_live
from oneai.taxonomy import Cluster, PatternPack

# Cosine similarity to the nearest training centroid. Above CONFIDENT_SIM we trust the match.
# Between LOW_SIM and CONFIDENT_SIM we only accept if a cheap taxonomy check agrees on issue;
# below LOW_SIM (or disagreement in the soft band) we return the clarify pack so novel topics
# like "my mobile network is not working" are not forced into billing/shipping/account_access.
CONFIDENT_SIM = 0.35
LOW_SIM = 0.28
# Only swap to the preferred agent style when that pack is nearly as close as the nearest hit.
STYLE_MARGIN = 0.05
# A call on a live pack only moves to a mined pack when the latest turn alone matches one this
# well. Generic follow-ups ("I tried turning it off and on again") reach ~0.46 against mined
# troubleshoot packs; real topic switches ("where is my package?") score 0.57+.
SWITCH_SIM = 0.52

UNKNOWN_PACK_ID = "unknown__clarify__empathic"


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


def unknown_pack() -> PatternPack:
    """Returned when the live turn is too far from every mined Pattern Pack."""
    return PatternPack(
        pack_id=UNKNOWN_PACK_ID,
        intent="info",
        issue="unknown",
        style="empathic",
        top_customer_phrasings=[],
        recommended_opener=(
            "I want to make sure I help with the right thing — "
            "could you tell me a bit more about what's going wrong?"
        ),
        style_cue="Do not guess the issue. Ask one short clarifying question.",
        follow_ups=[
            "Is this about a bill, payment, or charge?",
            "Is this about a delivery or package?",
            "Is this about logging into your account?",
        ],
        friction_alerts=["Low confidence match — topic is outside mined Pattern Packs."],
        agent_patterns=[],
        typical_flow=["clarify topic", "route once the issue is known"],
        call_count=0,
        built_with="rules",
        is_placeholder=True,
    )


class LiveClassifier:
    """Embeds a live turn, picks the nearest group and returns its Pattern Pack.

    Group centres and loaded packs are cached in memory so a turn costs one embedding call.
    K-means only discovers structure in the offline training set, so a turn far from every mined
    group is checked against live packs (topics first seen on a live call); classify_or_create
    builds a new live pack when none of those match either. Vague turns get the clarify pack.
    """

    def __init__(
        self, clusters_path: Path = CLUSTERS_PATH, packs_dir: Path = PACKS_DIR, live_dir: Path = LIVE_PACKS_DIR
    ):
        self.clusters = load_clusters(clusters_path)
        self.by_id = {c.id: c for c in self.clusters}
        self.centroids = np.asarray([c.centroid for c in self.clusters], dtype=np.float32)
        self.packs_dir = packs_dir
        self._packs: Dict[str, PatternPack] = {}
        self._unknown = unknown_pack()
        self.live = LivePackStore(live_dir)
        embed_many(["warm up"])

    def _pack_for(self, cluster: Cluster) -> PatternPack:
        if cluster.id not in self._packs:
            path = self.packs_dir / f"{cluster.id}.json"
            if path.exists():
                self._packs[cluster.id] = PatternPack.model_validate_json(path.read_text(encoding="utf-8"))
            else:
                self._packs[cluster.id] = placeholder_pack(cluster)
        return self._packs[cluster.id]

    def _choose_style(self, best_idx: int, sims: np.ndarray) -> int:
        """Prefer the recommended agent style when it is almost as close as the nearest hit."""
        best = self.clusters[best_idx]
        preferred_style = PREFERRED_STYLE.get(best.intent)
        if preferred_style is None:
            return best_idx
        candidate_idxs = [
            i
            for i, c in enumerate(self.clusters)
            if c.issue == best.issue and c.intent == best.intent and c.style == preferred_style
        ]
        if not candidate_idxs:
            return best_idx
        pref_idx = max(candidate_idxs, key=lambda i: float(sims[i]))
        if float(sims[pref_idx]) >= float(sims[best_idx]) - STYLE_MARGIN:
            return pref_idx
        return best_idx

    def _is_confident(self, turn_text: str, nearest_issue: str, raw_score: float) -> bool:
        if raw_score >= CONFIDENT_SIM:
            return True
        if raw_score < LOW_SIM:
            return False
        # Soft band: accept weak centroid matches only when taxonomy labelling agrees on issue.
        try:
            guessed = understand_text(turn_text, method="zeroshot")
        except Exception:
            guessed = understand_text(turn_text, method="rules")
        return guessed.get("issue") == nearest_issue

    def _mined(self, turn_text: str, v: np.ndarray) -> Tuple[Optional[PatternPack], float]:
        """The confident mined pack and its score, or (None, best raw score)."""
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = self.centroids @ v
        best_idx = int(np.argmax(sims))
        raw_score = float(sims[best_idx])
        if not self._is_confident(turn_text, self.clusters[best_idx].issue, raw_score):
            return None, raw_score
        chosen_idx = self._choose_style(best_idx, sims)
        return self._pack_for(self.clusters[chosen_idx]), float(sims[chosen_idx])

    def _route(self, turn_text: str, v: np.ndarray) -> Tuple[PatternPack, float]:
        """Mined pack if confident, else a live pack if close enough, else the clarify pack."""
        pack, score = self._mined(turn_text, v)
        if pack is not None:
            return pack, score
        return self.live.match(v) or (self._unknown, score)

    def classify_debug(self, turn_text: str) -> Tuple[PatternPack, float, float]:
        start = time.perf_counter()
        pack, score = self._route(turn_text, embed_many([turn_text])[0])
        return pack, score, (time.perf_counter() - start) * 1000

    def classify_or_create(
        self, turn_text: str, latest: Optional[str] = None, current: Optional[str] = None
    ) -> Tuple[PatternPack, float, float, bool]:
        """Like classify_debug, but a clear turn on a new topic gets a new live pack.

        turn_text is what the customer has said so far (used for routing); latest is this turn
        alone, stored as the pack's phrasing; current is the pack the previous turn used. A call on
        a live pack stays there unless this turn alone strongly matches a mined pack, since the
        call so far keeps resembling its opener whatever follows. Returns (pack, score, ms, created).
        """
        start = time.perf_counter()
        latest = latest or turn_text
        v = embed_many([turn_text])[0]
        ms = lambda: (time.perf_counter() - start) * 1000  # noqa: E731

        live_now = self.live.get(current) if current else None
        if live_now is not None:
            switch, switch_score = self._mined(latest, embed_many([latest])[0])
            if switch is None or switch_score < SWITCH_SIM:
                return live_now, float(np.asarray(live_now.centroid, dtype=np.float32) @ v), ms(), False

        pack, score = self._route(turn_text, v)
        created = False
        if pack is self._unknown:
            made = self.live.create(turn_text, v, latest)
            if made:
                pack, created = made
        elif is_live(pack) and pack.pack_id != current:
            pack = self.live.learn(pack.pack_id, v, latest)
        return pack, score, ms(), created

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
