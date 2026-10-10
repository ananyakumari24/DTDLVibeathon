"""LiveKit adapter: sits between speech-to-text and the agent's LLM reply.

On each finished customer turn it classifies the call so far, fetches the Pattern Pack and
injects it into the LLM context, so the voice bot replies with the learned opener, style and
follow-ups. The bot itself is unchanged.

Run (LiveKit keys in .env; OPENAI_API_KEY optional):
    pip install "livekit-agents[openai,silero]~=1.0"
    scripts/run.sh livekit
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from oneai.realtime.classify import get_live_classifier
from oneai.taxonomy import PatternPack

log = logging.getLogger("oneai.voice")

BASE_INSTRUCTIONS = "You are a voice support agent. Keep replies to one or two short spoken sentences."
GREETING = "Hi, thanks for calling support. How can I help you today?"


class TurnGuide:
    """Keeps the customer side of one call and returns the pack for the latest turn."""

    def __init__(self) -> None:
        self.said: List[str] = []

    def guide(self, turn_text: str) -> Tuple[PatternPack, float]:
        self.said.append(turn_text)
        pack, _, ms = get_live_classifier().classify_debug(" ".join(self.said))
        return pack, ms


def pack_instructions(pack: PatternPack) -> str:
    lines = [
        f"Guidance learned from past {pack.issue} calls ({pack.intent}, {pack.style} style):",
        f"- {pack.style_cue}",
        f"- If this is your first reply, open like: {pack.recommended_opener}",
    ]
    if pack.follow_ups:
        lines.append(f"- Useful follow-up questions: {'; '.join(pack.follow_ups)}")
    lines.extend(f"- {alert}" for alert in pack.friction_alerts)
    lines.append("- Never ask the customer for something they have already told you.")
    return "\n".join(lines)


def _build_agent():
    from livekit.agents import Agent, ChatContext, ChatMessage

    class OneAIAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions=BASE_INSTRUCTIONS)
            self.turn_guide = TurnGuide()

        async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
            text = new_message.text_content or ""
            pack, latency_ms = self.turn_guide.guide(text)
            log.info("customer said %r -> pack %s (%.1f ms)", text, pack.pack_id, latency_ms)
            turn_ctx.add_message(role="system", content=pack_instructions(pack))

    return OneAIAgent()


LIVEKIT_STT = "deepgram/nova-3"
LIVEKIT_LLM = "openai/gpt-4.1-mini"
LIVEKIT_TTS = "rime/mistv3"


def _voice_models():
    """OpenAI plugins with an OpenAI key, otherwise LiveKit Cloud's hosted models (LiveKit keys only)."""
    from oneai import llm

    if llm.available():
        from livekit.plugins import openai

        return openai.STT(), openai.LLM(), openai.TTS()
    return LIVEKIT_STT, LIVEKIT_LLM, LIVEKIT_TTS


def _log_session(session) -> None:
    session.on("agent_state_changed", lambda ev: log.info("agent %s -> %s", ev.old_state, ev.new_state))
    session.on("user_state_changed", lambda ev: log.info("caller %s -> %s", ev.old_state, ev.new_state))
    session.on(
        "user_input_transcribed",
        lambda ev: ev.is_final and log.info("heard: %r", ev.transcript),
    )
    session.on(
        "conversation_item_added",
        lambda ev: getattr(ev.item, "role", None) == "assistant"
        and log.info("bot said: %r", ev.item.text_content),
    )
    session.on("error", lambda ev: log.error("session error from %s: %s", type(ev.source).__name__, ev.error))


def prewarm(proc) -> None:
    """Load the VAD and the classifier's embedding model before a call arrives."""
    from livekit.plugins import silero

    proc.userdata["vad"] = silero.VAD.load()
    get_live_classifier().classify("warm up")


async def entrypoint(ctx) -> None:
    from livekit.agents import AgentSession
    from livekit.plugins import silero

    await ctx.connect()
    stt, llm_model, tts = _voice_models()
    log.info("voice models: stt=%s llm=%s tts=%s", stt, llm_model, tts)
    vad = ctx.proc.userdata.get("vad") or silero.VAD.load()
    session = AgentSession(stt=stt, llm=llm_model, tts=tts, vad=vad)
    _log_session(session)
    await session.start(room=ctx.room, agent=_build_agent())
    await session.say(GREETING, allow_interruptions=False)


if __name__ == "__main__":
    from livekit.agents import WorkerOptions, cli

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            num_idle_processes=1,
            # prewarm loads the embedding model (~9 s on a laptop); the default limit is 10 s
            initialize_process_timeout=60,
        )
    )
