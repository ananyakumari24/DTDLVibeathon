# Day 1 — Person A Summary (Offline Intelligence)

**Project path:** `/Users/ananya.kumari/oneai`  
**Date completed:** Fresh build from empty repo (only `.git` retained).

---

## What was done

### 1. Project reset
- Cleared prior application code; repository contains only a fresh Day 1 Person A slice.

### 2. Synthetic transcript generator
- **File:** `src/oneai/synthetic/generate.py`
- Generates **360** multi-turn support calls (seed **42**).
- Each call has ground-truth labels: `issue` × `intent` × `style`.
- Output: `data/synthetic/transcripts.jsonl` (one JSON object per line).

### 3. Normalize loader
- **File:** `src/oneai/offline/normalize.py`
- Reads JSONL → `Transcript` → `Conversation` with:
  - `customer_text`, `agent_text`, `full_text`
- Shared types in `src/oneai/taxonomy.py`.

### 4. Rule-based extractors
- **File:** `src/oneai/offline/extractors.py`
- Keyword/rules for: `extract_issue`, `extract_intent`, `extract_style`, `extract_friction`
- `annotate_conversation` attaches friction flags to each call.
- `evaluate_against_labels` compares rules to synthetic ground truth (for demo/debug).

### 5. Day 1 verification script
- **File:** `src/oneai/offline/day1_report.py`
- **Run:** `PYTHONPATH=src python -m oneai.offline.day1_report`

---

## Commands used

```bash
cd ~/oneai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
python -m oneai.synthetic.generate
python -m oneai.offline.day1_report
```

Or:

```bash
./scripts/run_generate.sh
./scripts/run_day1_report.sh
```

---

## Results (last run)

| Metric | Value |
|--------|--------|
| Transcripts | 360 |
| Issue accuracy (rules vs labels) | 1.0 |
| Intent accuracy | ~0.62 |
| Style accuracy | ~0.50 |
| Calls with friction detected | 119 |

Intent/style accuracy is **expected to be imperfect** on Day 1 — rules are a stand-in until Day 2 clustering/packs and realtime nearest-pack lookup. Issue keywords match synthetic billing/shipping/login vocabulary well.

---

## Folder layout (Person A)

```
oneai/
  data/synthetic/transcripts.jsonl
  docs/DAY1_PERSON_A_SUMMARY.md
  requirements.txt
  README.md
  scripts/run_generate.sh
  scripts/run_day1_report.sh
  src/oneai/
    taxonomy.py
    synthetic/generate.py
    offline/normalize.py
    offline/extractors.py
    offline/day1_report.py
```

---

## Handoff to Person B (Day 1 parallel)

Person B can start with **3 mock Pattern Packs** in `data/packs/` (not created in this slice). Contract fields should match the plan: `pack_id`, `issue`, `intent`, `style`, `top_customer_phrasings`, `recommended_opener`, `style_cue`, `follow_ups`, `friction_alerts`, `centroid`.

---

## Day 2 (Person A — not done yet)

- Embeddings (`sentence-transformers`)
- k-means or label-grouping → **Pattern Packs** → `data/packs/*.json` + SQLite
