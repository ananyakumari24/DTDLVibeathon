"""Live Pattern Packs: created on the fly when a turn matches no mined pack.

Mined packs only cover topics seen in the offline training calls. When a live turn is clear
but far from all of them (e.g. "my mobile network is not working"), we build a starter pack
for that topic, save it under data/live_packs/ and route to it. Its centroid is the turn's
embedding, so later calls on the same topic land on it; each one nudges the centroid and adds
its phrasing. The offline pipeline never touches this folder, so live packs survive a rebuild.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from oneai import llm
from oneai.offline.cluster import PREFERRED_STYLE
from oneai.offline.extractors import extract_intent
from oneai.offline.packs import INTENT_CUES, STYLE_CUES
from oneai.taxonomy import INTENTS, STYLES, PatternPack

LIVE_PACKS_DIR = Path("data/live_packs")
LIVE_PREFIX = "live__"
# A live centroid is one utterance, not a mean over many calls, so paraphrases score higher
# against it than against mined centroids; this bar matches the FAQ lookup's.
LIVE_MATCH_SIM = 0.45
MAX_PHRASINGS = 10
# Fewer content words than this and we can't tell what the topic is, so we clarify instead.
MIN_CONTENT_WORDS = 2

NEW_PACK_PROMPT = """A customer on a voice support call said something that matches none of our existing playbooks.
Customer so far: {text}

If it is too vague to know the topic (a greeting, "I need help", "it's not working"), return {{"clear": false}}.
Otherwise return JSON:
{{"clear": true,
  "issue": short snake_case topic, e.g. "mobile_network" or "sim_activation",
  "intent": one of {intents},
  "style": one of {styles} (empathic if the customer is upset or something is broken, else instructional),
  "recommended_opener": one short spoken sentence to open the reply,
  "style_cue": one sentence of guidance for the agent,
  "follow_ups": up to 3 short questions that collect the details needed to help,
  "friction_alerts": up to 2 short things the agent should avoid}}"""

STOPWORDS = set(
    """a an the i im i'm me my mine we our you your it its it's this that these those is are was were be been
    am do does did doing have has had can can't cannot could would should will won't not no don't doesn't didn't
    isn't aren't wasn't and or but so if then to of in on at for from with about by as up down out over again
    just still very really there here what when where why how who which hi hello hey please thanks thank help
    need want get got getting keep keeps kept working work works worked problem problems issue issues trouble
    since today yesterday now anymore any some all also like know think try tried trying going go""".split()
)

FOLLOW_UPS = {
    "troubleshoot": ["When did this start?", "What have you tried so far?", "Which device or service is affected?"],
    "info": ["What exactly would you like to know?", "Is this for your own account?"],
    "escalate": ["What's happened so far, so I can add it to the case?", "What would a good outcome look like for you?"],
}
OPENERS = {
    "troubleshoot": "I'm sorry you're dealing with that, let's get it sorted.",
    "info": "Sure, I can help with that.",
    "escalate": "I understand, and I'll make sure this gets to the right team.",
}
NEW_TOPIC_ALERT = "New topic with no mined calls yet; don't quote policy, prices or timelines you haven't been given."


def _slug(text: str, max_len: int = 32) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len].rstrip("_")


def content_words(text: str) -> List[str]:
    words = re.findall(r"[a-z][a-z0-9'-]*", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _rules_draft(text: str) -> Optional[dict]:
    words = content_words(text)
    if len(words) < MIN_CONTENT_WORDS:
        return None
    intent = extract_intent(text)
    # "not working" reads as info to the keyword extractor, but it's a fault report.
    if intent == "info" and re.search(r"\b(not|isn'?t|won'?t|doesn'?t|can'?t|no) \w*\s?(work|connect|load|signal|start)", text.lower()):
        intent = "troubleshoot"
    style = PREFERRED_STYLE.get(intent, "empathic")
    return {
        "issue": _slug("_".join(dict.fromkeys(words[:2]))),
        "intent": intent,
        "style": style,
        "recommended_opener": OPENERS[intent],
        "style_cue": f"{STYLE_CUES[style]} {INTENT_CUES[intent]}",
        "follow_ups": FOLLOW_UPS[intent],
        "friction_alerts": [],
        "built_with": "live-rules",
    }


def _llm_draft(text: str) -> Optional[dict]:
    data = llm.chat_json(NEW_PACK_PROMPT.format(text=text, intents=list(INTENTS), styles=list(STYLES)))
    if not data.get("clear"):
        return None
    issue = _slug(str(data.get("issue") or ""))
    if not issue or not isinstance(data.get("recommended_opener"), str):
        raise ValueError(f"Unexpected LLM live-pack output: {data}")
    intent = data.get("intent") if data.get("intent") in INTENTS else "troubleshoot"
    style = data.get("style") if data.get("style") in STYLES else PREFERRED_STYLE[intent]
    return {
        "issue": issue,
        "intent": intent,
        "style": style,
        "recommended_opener": data["recommended_opener"],
        "style_cue": str(data.get("style_cue") or f"{STYLE_CUES[style]} {INTENT_CUES[intent]}"),
        "follow_ups": [str(q) for q in (data.get("follow_ups") or [])[:3]] or FOLLOW_UPS[intent],
        "friction_alerts": [str(f) for f in (data.get("friction_alerts") or [])[:2]],
        "built_with": "live-llm",
    }


def draft_pack(text: str) -> Optional[dict]:
    """Pack fields for a new topic, or None when the turn is too vague to name one."""
    if llm.available():
        try:
            return _llm_draft(text)
        except Exception:
            pass
    return _rules_draft(text)


def is_live(pack: PatternPack) -> bool:
    return pack.pack_id.startswith(LIVE_PREFIX)


class LivePackStore:
    """Live packs on disk plus an in-memory centroid matrix for routing."""

    def __init__(self, packs_dir: Path = LIVE_PACKS_DIR):
        self.dir = packs_dir
        self._lock = threading.RLock()
        self.packs: List[PatternPack] = [
            PatternPack.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(self.dir.glob("*.json"))
        ]
        self._rebuild()

    def _rebuild(self) -> None:
        self.centroids = (
            np.asarray([p.centroid for p in self.packs], dtype=np.float32) if self.packs else np.zeros((0, 0), np.float32)
        )

    def _save(self, pack: PatternPack) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"{pack.pack_id}.json").write_text(pack.model_dump_json(indent=2) + "\n", encoding="utf-8")

    def nearest(self, v: np.ndarray) -> Optional[Tuple[int, float]]:
        if not self.packs or self.centroids.shape[1] != v.shape[0]:
            return None
        sims = self.centroids @ v
        best = int(np.argmax(sims))
        return best, float(sims[best])

    def match(self, v: np.ndarray, min_similarity: float = LIVE_MATCH_SIM) -> Optional[Tuple[PatternPack, float]]:
        hit = self.nearest(v)
        if hit is None or hit[1] < min_similarity:
            return None
        return self.packs[hit[0]], hit[1]

    def learn(self, pack_id: str, v: np.ndarray, phrasing: str) -> PatternPack:
        """Fold one more call into a live pack: running-mean centroid, phrasing, call count."""
        with self._lock:
            i = next(i for i, p in enumerate(self.packs) if p.pack_id == pack_id)
            pack = self.packs[i]
            n = pack.call_count or 1
            centroid = _normalize((np.asarray(pack.centroid, dtype=np.float32) * n + v) / (n + 1))
            phrasings = pack.top_customer_phrasings
            if phrasing not in phrasings:
                phrasings = (phrasings + [phrasing])[-MAX_PHRASINGS:]
            pack = pack.model_copy(
                update={"centroid": centroid.tolist(), "top_customer_phrasings": phrasings, "call_count": n + 1}
            )
            self.packs[i] = pack
            self._rebuild()
            self._save(pack)
            return pack

    def create(self, text: str, v: np.ndarray, phrasing: str) -> Optional[Tuple[PatternPack, bool]]:
        """Build and save a pack for this turn's topic. Returns (pack, created), or None if too vague.

        If a live pack for the same issue/intent/style already exists, the turn is folded into it
        instead, so two phrasings of one topic don't become two packs.
        """
        draft = draft_pack(text)
        if draft is None:
            return None
        pack_id = f"{LIVE_PREFIX}{draft['issue']}__{draft['intent']}__{draft['style']}"
        pack = PatternPack(
            pack_id=pack_id,
            intent=draft["intent"],
            issue=draft["issue"],
            style=draft["style"],
            top_customer_phrasings=[phrasing],
            recommended_opener=draft["recommended_opener"],
            style_cue=draft["style_cue"],
            follow_ups=draft["follow_ups"],
            friction_alerts=draft["friction_alerts"] + [NEW_TOPIC_ALERT],
            typical_flow=["customer states the problem", "agent gathers details", "agent confirms next steps"],
            centroid=_normalize(v).tolist(),
            call_count=1,
            built_with=draft["built_with"],
        )
        with self._lock:
            if self.get(pack_id) is not None:
                return self.learn(pack_id, v, phrasing), False
            self.packs.append(pack)
            self._rebuild()
            self._save(pack)
        return pack, True

    def get(self, pack_id: str) -> Optional[PatternPack]:
        return next((p for p in self.packs if p.pack_id == pack_id), None)

    def summary(self) -> List[Dict]:
        return [
            p.model_dump(include={"pack_id", "issue", "intent", "style", "call_count", "built_with", "top_customer_phrasings"})
            for p in self.packs
        ]
