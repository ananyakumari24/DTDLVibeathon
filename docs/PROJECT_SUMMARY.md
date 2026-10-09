# OneAI — Project Summary (Day 1–3, Person A + Person B)

## Person A — offline mining
- `synthetic/generate.py`: 360 seeded calls (3 issues × 3 intents × 2 styles).
- `offline/normalize.py`, `offline/extractors.py`: loading and Day 1 rule extractors.
- `offline/embeddings.py`: MiniLM sentence embeddings with a hashing fallback.
- `offline/understand.py`: issue + question-type labelling (LLM, zero-shot or rules).
- `offline/cluster.py`: groups calls into intent × issue × style cells with centroids → `data/clusters.json`.
- `eval.py`: held-out scoring of labelling, routing, latency and unresolved detection → `data/eval_report.json`.

## Person B — packs and real time
- `offline/agent.py`: agent style and friction flags (negative sentiment, repeated questions, repeat contact).
- `offline/packs.py`: Pattern Packs (phrasings, opener, style cue, agent patterns, flow, friction alerts, follow-ups) → `data/packs/*.json`.
- `realtime/classify.py`: nearest-centroid live classifier with in-memory cache.
- `realtime/reply.py`: pack-guided reply (LLM or template) and plain baseline.
- `realtime/app.py` + `static/index.html`: FastAPI API and dashboard.
- `realtime/mock_replay.py`, `livekit_adapter/agent.py`: turn-by-turn replay and LiveKit voice agent.

## Joint
- `offline/pipeline.py`: one command builds everything; `ab_demo.py` runs the A/B rubric.
- `scripts/run.sh`, README, `docs/PITCH.md`.

## Results (no OpenAI key; zero-shot labelling, template replies)
| Metric | Value |
|---|---|
| Groups / packs | 16 / 16 from 360 calls |
| Held-out pack routing | 100% (5 calls) |
| Live classify p95 | ~6 ms (warm `/reply` ~10 ms) |
| Issue / intent accuracy (zero-shot) | 100% / 100% |
| Agent style, unresolved detection | 100%, 100% |
| A/B engagement | plain 50% → Pattern-Pack 80% |

## Limitations
- Held-out set is only 5 hand-written calls; data is synthetic.
- LLM paths are implemented but untested without a key.
- LiveKit adapter is written but `livekit-agents` was not installed or run.
- The plain baseline is a fixed template, so the A/B result is illustrative.
- Packs are stored as JSON (no SQLite).
