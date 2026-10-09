# OneAI (Vibeathon)

Mine support-call transcripts offline into **Pattern Packs** (intent × issue × style), then use them to guide a voice bot's replies in real time.

## Setup

```bash
cd ~/oneai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: set OPENAI_API_KEY for LLM labelling and replies
```

## Run

```bash
scripts/run.sh generate    # 360 synthetic calls -> data/synthetic/transcripts.jsonl
scripts/run.sh pipeline    # groups, packs, eval report, A/B report
scripts/run.sh api         # dashboard at http://localhost:8000
scripts/run.sh mock --call-id heldout_04   # replay a held-out call turn by turn
scripts/run.sh eval        # labelling, routing, latency report
scripts/run.sh ab          # plain bot vs Pattern-Pack bot
scripts/run.sh livekit     # LiveKit voice agent (needs livekit-agents installed)
```

## Layout

```
src/oneai/
  taxonomy.py          shared models (Transcript, Insight, Cluster, PatternPack)
  llm.py               optional OpenAI helpers
  eval.py, ab_demo.py  evaluation and A/B demo
  synthetic/           synthetic transcript generator
  offline/             normalize, extract, embed, label, cluster, build packs, pipeline
  realtime/            live classifier, guided reply, FastAPI app + dashboard, mock replay
  livekit_adapter/     LiveKit agent that injects pack guidance each turn
data/                  transcripts, held-out set, clusters, packs, reports
docs/                  summaries and pitch
```

See `docs/PROJECT_SUMMARY.md` for what was built and `docs/PITCH.md` for the demo pitch.
