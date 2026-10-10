# OneAI — Hackathon prep (read this before demo day)

**Repo:** `~/oneai`  
**One-liner:** We mine past voice-support transcripts into **Pattern Packs**, then match each live customer turn to the right pack in milliseconds so the bot sounds like your best agents—not a generic script.

---

## 1. Story for judges (60–90 seconds)


| Beat                  | What to say                                                                                                                                                                                                                                                                                         |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Problem**           | Voice bots use one playbook. Humans learn from thousands of calls: empathy for billing anger, step-by-step for “how do I…”, and when *not* to ask the customer to repeat themselves.                                                                                                                |
| **Solution**          | **Offline:** label calls (issue, intent, agent style, friction) → group into cells → each cell becomes a **Pattern Pack** (real phrasings, opener, flow, follow-ups, friction alerts). **Live:** embed the customer’s words → nearest pack centroid → inject pack into the reply (template or LLM). |
| **Why it fits OneAI** | Same taxonomy the problem statement cares about (intent × issue × style). Fast enough for voice (~6 ms classify p95). Plugs into LiveKit via one turn hook.                                                                                                                                         |
| **Proof**             | 360 synthetic calls → **16 packs**. Held-out routing **100%** (5 calls). Classify **p95 ~6 ms** (under 200 ms bar). A/B rubric: engagement **50% → 80%** (plain vs pack-guided templates).                                                                                                          |
| **Next**              | Real transcripts, production repeat-call rate, LLM replies with `OPENAI_API_KEY`, full LiveKit voice demo.                                                                                                                                                                                          |


Full script: `docs/PITCH.md`.

---



## 2. Architecture (draw or point at screen)

```
OFFLINE (batch)                         LIVE (per customer turn)
─────────────────                       ────────────────────────
transcripts.jsonl                       customer text (+ history)
    → normalize / label                     → embed
    → embed per call                        → nearest centroid → Pattern Pack
    → group: intent×issue×style             → suggest_reply(pack) or plain_reply
    → build Pattern Pack JSON               → (optional) LiveKit TurnGuide
data/clusters.json
data/packs/*.json
```

**Three axes (memorize):**

- **Issue:** billing, connectivity, account_access  
- **Intent (question type):** troubleshoot, info, escalate  
- **Style:** empathic vs instructional (preferred per intent: troubleshoot/escalate → empathic, info → instructional)

**What’s inside a Pattern Pack:** `top_customer_phrasings`, `recommended_opener`, `style_cue`, `follow_ups`, `friction_alerts`, `agent_patterns`, `typical_flow`, `centroid`.

---



## 3. What you actually built (scope for Q&A)


| Area                      | Status                                               | Where                                                             |
| ------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------- |
| Synthetic data + labels   | Done                                                 | `synthetic/generate.py`, `data/synthetic/transcripts.jsonl`       |
| Offline pipeline          | Done                                                 | `offline/pipeline.py` via `scripts/run.sh pipeline`               |
| Embeddings                | Done (MiniLM cached; hashing fallback)               | `offline/embeddings.py`                                           |
| Labelling                 | Done: rules / zero-shot / LLM if key                 | `offline/understand.py`, `offline/agent.py`                       |
| Pattern Packs             | Done (16 JSON files)                                 | `offline/packs.py`, `data/packs/`                                 |
| Live classify + reply     | Done                                                 | `realtime/classify.py`, `realtime/reply.py`                       |
| Dashboard + API           | Done                                                 | `realtime/app.py`, [http://localhost:8000](http://localhost:8000) |
| Eval + A/B reports        | Done                                                 | `data/eval_report.json`, `data/ab_report.json`                    |
| Mock call replay          | Done                                                 | `realtime/mock_replay.py`                                         |
| LiveKit voice agent       | **Code only** — needs `livekit-agents` + credentials | `livekit_adapter/agent.py`                                        |
| Real customer transcripts | **Not in repo** — synthetic only                     |                                                                   |
| SQLite store              | **Not built** — JSON on disk                         |                                                                   |


---



## 4. Demo runbook (practice twice)



### Before you leave / open laptop

```bash
cd ~/oneai
source .venv/bin/activate
export PYTHONPATH=src
export PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1   # model already cached; fewer warnings
# Optional for richer replies + LLM labelling:
# export OPENAI_API_KEY=sk-...
```



### One-time or if data missing

```bash
scripts/run.sh generate    # if transcripts.jsonl missing
scripts/run.sh pipeline      # ~15s — clusters, packs, eval, A/B
```



### Live demo (recommended)

```bash
scripts/run.sh api
```

Open **[http://localhost:8000](http://localhost:8000)**

1. **Groups / packs** — show a pack (e.g. `billing__troubleshoot__empathic`): opener, friction alerts, phrasings.
2. **Live try** — type a customer line; show **plain vs Pattern-Pack** side by side.
3. **Metrics** — `/metrics` or dashboard coverage table: routing, latency, A/B.



### Backup (terminal, no browser)

```bash
scripts/run.sh mock --call-id heldout_01   # billing double charge — strong empathic opener
scripts/run.sh eval                        # prints summary from eval_report
scripts/run.sh ab                          # plain vs pack engagement
```



### Strong demo lines (type in dashboard)


| Customer says                                              | Expected pack flavor                                     |
| ---------------------------------------------------------- | -------------------------------------------------------- |
| "I was charged twice this month and I'm frustrated."       | billing · troubleshoot · empathic                        |
| "How do I reset my password in the app?"                   | account_access · info · instructional                    |
| "My internet keeps dropping after I restarted the router." | connectivity · troubleshoot                              |
| "Just get me someone who can fix it."                      | escalate · empathic (classify cumulative text in replay) |


**Tip:** Classifier uses **cumulative customer text** in mock replay and `/reply` with history—mention that follow-up turns stay routed correctly.

---



## 5. Numbers to cite (from `data/eval_report.json` / pipeline)


| Metric                            | Value             |
| --------------------------------- | ----------------- |
| Training calls                    | 360               |
| Groups / Pattern Packs            | 16 / 16           |
| Held-out calls (eval)             | 5                 |
| Pack routing accuracy             | 100%              |
| Classify latency p50 / p95        | ~5.3 ms / ~6.0 ms |
| Under 200 ms requirement          | Yes               |
| Zero-shot issue / intent accuracy | 100% / 100%       |
| Agent style labelling             | 100%              |
| Unresolved detection              | 100%              |
| A/B engagement (templates)        | 50% → 80%         |


Say clearly: data is **synthetic**; held-out is **small**; A/B uses a **fixed plain template** vs **pack templates** (not a full GPT baseline unless you set `OPENAI_API_KEY`).

---



## 6. Split for two people tomorrow


| Person                  | Own on stage                                                               | Files                                               |
| ----------------------- | -------------------------------------------------------------------------- | --------------------------------------------------- |
| **Person A (offline)**  | Pipeline, taxonomy, how packs are built, eval numbers                      | `offline/`*, `synthetic/`, `eval.py`, `taxonomy.py` |
| **Person B (realtime)** | Dashboard demo, classify/reply path, LiveKit hook (even if not live voice) | `realtime/`*, `livekit_adapter/`, `ab_demo.py`      |


**Handoff line:** “Person A mined the calls into packs; Person B wires those packs into the bot in real time under 10 ms.”

---



## 7. Likely judge questions + honest answers


| Question                         | Answer                                                                                                                                         |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Real transcripts?                | Hackathon used **seeded synthetic** calls with ground-truth labels; pipeline is the same for real JSONL.                                       |
| Why 16 packs?                    | 3 issues × 3 intents × 2 styles = 18 cells; some combinations have no calls in synthetic data, so 16 non-empty groups.                         |
| How do you label without labels? | **Zero-shot** embedding similarity to label descriptions; optional **LLM**; **rules** fallback.                                                |
| Cold start / first request slow? | First load embeds model (~7 s); API **lifespan** warms classifier—wait for “Application startup complete” before demo.                         |
| LiveKit?                         | `TurnGuide` + `on_user_turn_completed` injects pack instructions; install `livekit-agents` and run `scripts/run.sh livekit` for voice stretch. |
| Hallucination?                   | Pack text is **retrieved patterns**, not free-form invention; LLM path is constrained by system prompt + pack.                                 |
| Wrong pack?                      | Nearest centroid can misroute odd phrasing—show friction alerts and follow-ups as guardrails; future: confidence threshold + clarify.          |


---



## 8. Pre-demo checklist (night before + morning)

- [ ] `source .venv/bin/activate` works  
- [ ] `scripts/run.sh pipeline` exits 0 and prints 16 packs  
- [ ] `data/packs/` has 16 `.json` files  
- [ ] `scripts/run.sh api` — wait until server ready (~10 s after startup)  
- [ ] Dashboard loads; one `/reply` returns in < 200 ms after warm  
- [ ] Decide: demo **with** or **without** OpenAI key (templates are reliable offline)  
- [ ] Browser tab open: home + one pack + metrics  
- [ ] Terminal backup: `mock --call-id heldout_01`  
- [ ] Agree who speaks which section (90 s total)  

---



## 9. File map (quick reference)

```
src/oneai/taxonomy.py       # Transcript, Insight, Cluster, PatternPack
src/oneai/offline/pipeline.py   # one command: everything offline + reports
src/oneai/realtime/app.py       # FastAPI + dashboard
scripts/run.sh                  # generate | pipeline | api | mock | eval | ab | livekit
docs/PITCH.md                   # pitch script
docs/PROJECT_SUMMARY.md         # technical summary
data/eval_report.json           # numbers for slides
data/ab_report.json             # example plain vs pack replies
```

---



## 10. If something breaks on stage

1. **API won’t start** — run mock replay in terminal (`heldout_01`).
2. **Empty packs** — run `scripts/run.sh pipeline`.
3. **Embedding error** — `HF_HUB_OFFLINE=1` and ensure you ran pipeline once on this machine.
4. **Port in use** — `lsof -ti tcp:8000 | xargs kill` then restart api.

Good luck tomorrow—you have a full offline → live story, working UI, and defensible metrics. Lead with the **customer problem**, show **one pack**, then **one live turn** plain vs guided.