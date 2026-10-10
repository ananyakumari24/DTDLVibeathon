"""LiveKit adapter: sits between speech-to-text and text-to-speech.

On each finished customer turn it runs the same turn as the API's /reply (realtime/turn.py):
pick the Pattern Pack, plan the FAQ step, write the guided reply. The bot speaks that reply, so
the phone line gives FAQ steps one per turn, holds them through side questions and signs off,
exactly like the dashboard. The session's LLM is not asked to reply on its own.

Run (LiveKit keys in .env; OPENAI_API_KEY optional):
    pip install "livekit-agents[openai,silero]~=1.0"
    scripts/run.sh livekit
"""

from __future__ import annotations

import asyncio
import logging

from oneai.realtime.classify import get_live_classifier
from oneai.realtime.faq import get_faq
from oneai.realtime.turn import CallState, describe_faq, take_turn

log = logging.getLogger("oneai.voice")

BASE_INSTRUCTIONS = "You are a voice support agent. Keep replies to one or two short spoken sentences."
GREETING = "Hi, thanks for calling support. How can I help you today?"


def _build_agent():
    from livekit.agents import Agent, ChatContext, ChatMessage, StopResponse

    class OneAIAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions=BASE_INSTRUCTIONS)
            self.call = CallState()

        async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
            text = (new_message.text_content or "").strip()
            if not text:
                raise StopResponse()
            # Off the event loop: the classifier embeds the turn, and the reply may call an LLM.
            turn = await asyncio.to_thread(take_turn, text, self.call)
            self.call.record(text, turn)
            log.info(
                "customer said %r -> pack %s (%.1f ms) · faq %s%s",
                text,
                turn.pack.pack_id,
                turn.latency_ms,
                describe_faq(turn.faq) or "none",
                " · call ended" if turn.ended else "",
            )
            self.session.say(turn.reply)
            raise StopResponse()  # the reply above is the whole turn; skip the LLM's own

    return OneAIAgent()


LIVEKIT_STT = "deepgram/nova-3"
LIVEKIT_LLM = "openai/gpt-4.1-mini"
LIVEKIT_TTS = "rime/mistv3"


def _voice_models():
    """OpenAI plugins with an OpenAI key, otherwise LiveKit Cloud's hosted models (LiveKit keys only)."""
    from oneai import llm

    # provider(), not available(): a Cursor key has no speech models, so it falls back to LiveKit's.
    if llm.provider() == "openai":
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
    """Load the VAD, the classifier's embedding model and the FAQ index before a call arrives."""
    from livekit.plugins import silero

    proc.userdata["vad"] = silero.VAD.load()
    get_live_classifier().classify("warm up")
    get_faq()


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
