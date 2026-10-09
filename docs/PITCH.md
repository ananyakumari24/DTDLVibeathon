# OneAI — 90-second pitch

**Problem (15s).** Voice bots answer every caller the same way. Human agents learn from thousands of calls which words calm an angry customer, which question unblocks a fix, and which mistakes make people call back. Bots never get that knowledge.

**Idea (20s).** OneAI mines past transcripts offline and groups them by *what* the customer wants (intent), *what it's about* (issue) and *how* the best agents answered (style). Each group becomes a **Pattern Pack**: real customer phrasings, the opener that works, the usual call flow, follow-up questions, and friction alerts such as "don't make them repeat themselves".

**Live (20s).** During a call, every customer turn is matched to the nearest pack in under 10 ms. The pack is injected into the bot's prompt, so the reply has the right tone, stays on topic and moves the call forward. It plugs into LiveKit as a single hook.

**Proof (20s).** On 360 calls we built 16 packs. Held-out calls routed to the right pack 100% of the time with p95 latency around 6 ms. On our rubric (right tone, on topic, no repeated burden, moves forward), engagement rose from **50% to 80%** versus a plain bot.

**Ask (15s).** Next steps: run it on real transcripts, add an LLM for richer replies, and measure repeat-call rate in production.

---

## One-pager

- **What:** Pattern Packs turn past support calls into live guidance for voice bots.
- **How:** embed → label (intent, issue, style, friction) → group → build packs → nearest-centroid lookup each turn → guided reply.
- **Numbers:** 16 packs · 100% held-out routing · ~6 ms p95 · engagement 50% → 80%.
- **Demo:** `scripts/run.sh pipeline && scripts/run.sh api`, then open the dashboard and type a customer turn to see the plain vs Pattern-Pack replies side by side.
- **Caveats:** synthetic data, 5 held-out calls, template replies without an API key.
