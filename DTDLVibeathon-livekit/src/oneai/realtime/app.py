"""OneAI API: groups, Pattern Packs, live classify and reply, ingest, scores and the dashboard.

Run: PYTHONPATH=src uvicorn oneai.realtime.app:app --port 8000
"""

from __future__ import annotations

import html
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from oneai.ab_demo import AB_REPORT_PATH
from oneai.eval import REPORT_PATH
from oneai.offline.cluster import CLUSTERS_PATH, PACKS_DIR
from oneai.offline.pipeline import ingest
from oneai.realtime.classify import get_live_classifier
from oneai.realtime.reply import plain_reply, suggest_reply
from oneai.taxonomy import Insight, PatternPack, Transcript

STATIC_DIR = Path(__file__).parent / "static"

COVERAGE = [
    ("Cluster by intent, issue type and style", "offline/cluster.py", "done"),
    ("Customer phrasing and question types", "offline/understand.py", "done"),
    ("Agent response patterns and style", "offline/agent.py, offline/packs.py", "done"),
    ("Multi-turn flow and friction", "offline/agent.py, offline/packs.py", "done"),
    ("Pattern Packs per group", "offline/packs.py", "done"),
    ("Real-time lookup for the voice bot", "realtime/classify.py", "done"),
    ("Guided reply for the voice bot", "realtime/reply.py", "done"),
    ("LiveKit integration", "livekit_adapter/agent.py", "written, not run (needs LiveKit keys)"),
    ("Evaluation: intent accuracy, unresolved calls, engagement", "eval.py, ab_demo.py", "done"),
]


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the embedding model and group centres before the first live turn arrives.
    if CLUSTERS_PATH.exists():
        get_live_classifier()
    yield


app = FastAPI(title="OneAI", lifespan=lifespan)


class TextIn(BaseModel):
    text: str


class ClassifyOut(BaseModel):
    pack: PatternPack
    similarity: float
    latency_ms: float


class ReplyIn(BaseModel):
    text: str
    said_before: List[str] = Field(default_factory=list, description="Earlier customer turns in this call")
    history: List[str] = Field(default_factory=list, description="Earlier bot replies in this call")


class ReplyOut(BaseModel):
    pack_id: str
    similarity: float
    latency_ms: float
    is_placeholder: bool
    pack_reply: str
    plain_reply: str


class IngestIn(BaseModel):
    transcripts: List[Transcript]


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/clusters")
def list_clusters() -> List[dict]:
    return [c.model_dump(exclude={"centroid"}) for c in get_live_classifier().clusters]


@app.get("/clusters/{cluster_id}")
def get_cluster(cluster_id: str) -> dict:
    cluster = get_live_classifier().by_id.get(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail=f"No cluster {cluster_id}")
    return cluster.model_dump(exclude={"centroid"})


@app.get("/packs")
def list_packs() -> List[dict]:
    packs = [PatternPack.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(PACKS_DIR.glob("*.json"))]
    return [p.model_dump(include={"pack_id", "issue", "intent", "style", "call_count", "built_with"}) for p in packs]


@app.get("/packs/{pack_id}")
def get_pack(pack_id: str) -> dict:
    path = PACKS_DIR / f"{pack_id}.json"
    if path.parent != PACKS_DIR or not path.exists():
        raise HTTPException(status_code=404, detail=f"No pack {pack_id}")
    return PatternPack.model_validate_json(path.read_text(encoding="utf-8")).model_dump(exclude={"centroid"})


@app.post("/classify", response_model=ClassifyOut)
def classify_turn(body: TextIn) -> ClassifyOut:
    pack, score, ms = get_live_classifier().classify_debug(body.text)
    return ClassifyOut(pack=pack, similarity=round(score, 4), latency_ms=round(ms, 2))


@app.post("/reply", response_model=ReplyOut)
def reply(body: ReplyIn) -> ReplyOut:
    pack, score, ms = get_live_classifier().classify_debug(" ".join(body.said_before + [body.text]))
    return ReplyOut(
        pack_id=pack.pack_id,
        similarity=round(score, 4),
        latency_ms=round(ms, 2),
        is_placeholder=pack.is_placeholder,
        pack_reply=suggest_reply(body.text, pack, body.history),
        plain_reply=plain_reply(body.text, body.history),
    )


@app.post("/ingest", response_model=List[Insight])
def ingest_calls(body: IngestIn) -> List[Insight]:
    return ingest(body.transcripts)


@app.get("/ab")
def ab_results() -> dict:
    if not AB_REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="No A/B report yet. Run: python -m oneai.offline.pipeline")
    return json.loads(AB_REPORT_PATH.read_text(encoding="utf-8"))


@app.get("/metrics", response_class=HTMLResponse)
def metrics_page() -> str:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else None
    rows = "".join(
        f"<tr><td>{html.escape(a)}</td><td><code>{html.escape(b)}</code></td><td>{html.escape(c)}</td></tr>"
        for a, b, c in COVERAGE
    )
    if report is None:
        body = "<p>No report yet. Run <code>python -m oneai.offline.pipeline</code>.</p>"
    else:
        live = report["live_classifier"]
        unresolved = report["unresolved"]
        labelling = "".join(
            f"<tr><td>{m}</td><td>{s['issue_accuracy']:.0%}</td><td>{s['intent_accuracy']:.0%}</td></tr>"
            for m, s in report["labelling"].items()
        )
        body = f"""
<p>{report['heldout_calls']} held-out calls, {report['embedding_backend']} embeddings.</p>
<h2>Labelling accuracy</h2>
<table><tr><th>Method</th><th>Issue</th><th>Question type</th></tr>{labelling}</table>
<p>Agent style accuracy: {report['agent_style_accuracy']:.0%}</p>
<h2>Live classifier</h2>
<table>
<tr><td>Pack routing accuracy</td><td>{live['pack_routing_accuracy']:.0%}</td></tr>
<tr><td>Latency p50 / p95</td><td>{live['latency_ms_p50']} ms / {live['latency_ms_p95']} ms</td></tr>
<tr><td>Under 200 ms</td><td>{'yes' if live['under_200ms'] else 'no'}</td></tr>
</table>
<h2>Unresolved calls</h2>
<table>
<tr><td>Detection accuracy</td><td>{unresolved['unresolved_detection_accuracy']:.0%}</td></tr>
<tr><td>Actual / predicted unresolved rate</td><td>{unresolved['actual_unresolved_rate']:.0%} / {unresolved['predicted_unresolved_rate']:.0%}</td></tr>
</table>"""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>OneAI metrics</title>
<style>body{{font-family:system-ui;max-width:760px;margin:2rem auto}}table{{border-collapse:collapse;margin-bottom:1rem}}
td,th{{border:1px solid #ccc;padding:.4rem .7rem;text-align:left}}</style></head><body>
<h1>OneAI metrics</h1><p><a href="/">Back to dashboard</a></p>{body}
<h2>Problem-statement coverage</h2>
<table><tr><th>Requirement</th><th>Module</th><th>Status</th></tr>{rows}</table></body></html>"""
