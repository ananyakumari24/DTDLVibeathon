"""Pattern Pack builder: one pack per group, learned from the group's calls."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from oneai import llm
from oneai.offline.agent import EMPATHY_WORDS, REPEAT_WORDS, label_agent
from oneai.offline.cluster import CLUSTERS_PATH, PACKS_DIR, load_clusters
from oneai.offline.normalize import load_conversations
from oneai.taxonomy import Cluster, Conversation, PatternPack, Turn

STYLE_CUES = {
    "empathic": "Acknowledge how the customer feels before anything else.",
    "instructional": "Be clear and stepwise, and ask for one detail at a time.",
}

INTENT_CUES = {
    "troubleshoot": "Get one key detail, then say what you'll fix and when.",
    "info": "Answer directly, then say where they can find it next time.",
    "escalate": "Confirm the escalation and give a reference number and a callback time.",
}

# Used only when the group's own agents never asked a question.
FOLLOW_UP_BANK = {
    "billing": {
        "troubleshoot": ["Which month shows the wrong charge?", "Was it the same card both times?"],
        "info": ["Which line on the bill would you like me to explain?"],
        "escalate": ["Do you have the case number from your last call?"],
    },
    "shipping": {
        "troubleshoot": ["What's your order number?", "When did tracking last update?"],
        "info": ["What's your order number?"],
        "escalate": ["What's your order number, so I can attach it to the escalation?"],
    },
    "account_access": {
        "troubleshoot": ["Which email is on the account?", "Did the reset email arrive?"],
        "info": ["Are you using the app or the website?"],
        "escalate": ["Which email is on the account, so the specialist can find it?"],
    },
}

FRICTION_ALERTS = {
    "customer_already_explained": "Customers often say they already explained; don't ask them to repeat",
    "customer_refuses_repeat": "Customers push back on repeating the story; summarise what you know instead",
    "repeat_contact": "Many of these are repeat contacts; acknowledge the earlier calls",
    "negative_sentiment": "Customers are often upset; acknowledge that before solving",
    "repeated_question": "Customers often ask the same thing twice; answer it clearly the first time",
}

REFINE_PROMPT = """These calls come from one group of support conversations
(issue: {issue}, question type: {intent}, agent style: {style}).

Customer openers: {openers}
Agent lines: {agent_lines}
Draft guidance: {draft}

Improve the guidance for a voice bot handling calls like these.
Return JSON with keys "recommended_opener" (string), "style_cue" (string) and
"follow_ups" (list of up to 3 short questions)."""

ACCEPT_STARTS = ("okay", "ok", "alright", "thanks", "thank you", "please proceed")


def _step(turn: Turn, customer_index: int, is_last_agent: bool) -> str:
    text = turn.text.lower()
    if turn.speaker == "customer":
        if customer_index == 0:
            return "customer states the problem"
        if any(w in text for w in REPEAT_WORDS) or "don't make me" in text:
            return "customer pushes back"
        if text.startswith(ACCEPT_STARTS):
            return "customer accepts"
        return "customer gives details"
    if is_last_agent:
        return "agent confirms next steps"
    if "?" in text:
        return "agent asks a clarifying question"
    if any(w in text for w in EMPATHY_WORDS):
        return "agent acknowledges feelings"
    return "agent continues"


def flow_of(conv: Conversation) -> Tuple[str, ...]:
    last_agent = max(i for i, t in enumerate(conv.turns) if t.speaker == "agent")
    steps: List[str] = []
    customer_index = 0
    for i, turn in enumerate(conv.turns):
        step = _step(turn, customer_index, i == last_agent)
        if turn.speaker == "customer":
            customer_index += 1
        if not steps or steps[-1] != step:
            steps.append(step)
    return tuple(steps)


def _pct(count: int, total: int) -> str:
    return f"{round(100 * count / total)}%"


def _rules_pack(cluster: Cluster, members: List[Conversation]) -> PatternPack:
    n = len(members)
    openers = Counter(c.turns[0].text for c in members)
    agent_first = Counter(next(t.text for t in c.turns if t.speaker == "agent") for c in members)
    agent_lines = Counter(t.text for c in members for t in c.turns if t.speaker == "agent")
    questions = Counter(t.text for c in members for t in c.turns if t.speaker == "agent" and "?" in t.text)
    flows = Counter(flow_of(c) for c in members)
    friction = Counter(f for c in members for f in label_agent(c, method="rules")["friction_flags"])

    follow_ups = [q for q, _ in questions.most_common(3)]
    for q in FOLLOW_UP_BANK[cluster.issue][cluster.intent]:
        if len(follow_ups) < 3 and q not in follow_ups:
            follow_ups.append(q)

    top_flow, flow_count = flows.most_common(1)[0]
    return PatternPack(
        pack_id=cluster.id,
        intent=cluster.intent,
        issue=cluster.issue,
        style=cluster.style,
        top_customer_phrasings=[p for p, _ in openers.most_common(5)],
        recommended_opener=agent_first.most_common(1)[0][0],
        style_cue=f"{STYLE_CUES[cluster.style]} {INTENT_CUES[cluster.intent]}",
        follow_ups=follow_ups,
        friction_alerts=[
            f"{FRICTION_ALERTS[f]} ({_pct(k, n)} of calls)." for f, k in friction.most_common() if f in FRICTION_ALERTS
        ],
        agent_patterns=[f"{line} ({_pct(k, n)} of calls)" for line, k in agent_lines.most_common(3)],
        typical_flow=list(top_flow) + [f"({_pct(flow_count, n)} of calls follow this flow)"],
        centroid=cluster.centroid,
        call_count=n,
        built_with="rules",
    )


def _refine_with_llm(pack: PatternPack, members: List[Conversation]) -> PatternPack:
    draft = {"recommended_opener": pack.recommended_opener, "style_cue": pack.style_cue, "follow_ups": pack.follow_ups}
    agent_lines = [t.text for c in members[:15] for t in c.turns if t.speaker == "agent"]
    data = llm.chat_json(
        REFINE_PROMPT.format(
            issue=pack.issue,
            intent=pack.intent,
            style=pack.style,
            openers=pack.top_customer_phrasings,
            agent_lines=agent_lines[:30],
            draft=draft,
        )
    )
    follow_ups = data.get("follow_ups")
    if not isinstance(data.get("recommended_opener"), str) or not isinstance(follow_ups, list):
        raise ValueError(f"Unexpected LLM pack output: {data}")
    return pack.model_copy(
        update={
            "recommended_opener": data["recommended_opener"],
            "style_cue": str(data.get("style_cue") or pack.style_cue),
            "follow_ups": [str(q) for q in follow_ups[:3]],
            "built_with": "llm",
        }
    )


def build_pack(cluster: Cluster, members: List[Conversation]) -> PatternPack:
    pack = _rules_pack(cluster, members)
    if llm.available():
        try:
            return _refine_with_llm(pack, members)
        except Exception:
            pass
    return pack


def build_all(clusters: List[Cluster], conversations: List[Conversation]) -> List[PatternPack]:
    by_id: Dict[str, Conversation] = {c.call_id: c for c in conversations}
    return [build_pack(cl, [by_id[i] for i in cl.member_call_ids if i in by_id]) for cl in clusters]


def save_packs(packs: List[PatternPack], packs_dir: Path = PACKS_DIR) -> None:
    packs_dir.mkdir(parents=True, exist_ok=True)
    ids = {p.pack_id for p in packs}
    for stale in packs_dir.glob("*.json"):
        if stale.stem not in ids:
            stale.unlink()
    for pack in packs:
        (packs_dir / f"{pack.pack_id}.json").write_text(pack.model_dump_json(indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one Pattern Pack per group")
    parser.add_argument("--in", dest="infile", type=Path, default=Path("data/synthetic/transcripts.jsonl"))
    parser.add_argument("--clusters", type=Path, default=CLUSTERS_PATH)
    parser.add_argument("--out-dir", type=Path, default=PACKS_DIR)
    args = parser.parse_args()
    packs = build_all(load_clusters(args.clusters), load_conversations(args.infile))
    save_packs(packs, args.out_dir)
    print(f"Built {len(packs)} Pattern Packs -> {args.out_dir}")


if __name__ == "__main__":
    main()
