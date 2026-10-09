"""LiveKit adapter: sits between speech-to-text and the agent's LLM reply.

On each finished customer turn it classifies the call so far, fetches the Pattern Pack and
injects it into the LLM context, so the voice bot replies with the learned opener, style and
follow-ups. The bot itself is unchanged.

Run (needs LiveKit and OpenAI keys):
    pip install "livekit-agents[openai,silero]~=1.0"
    export LIVEKIT_URL=... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=... OPENAI_API_KEY=...
    PYTHONPATH=src python -m oneai.livekit_adapter.agent dev
"""

from __future__ import annotations

from typing import List, Tuple

from oneai.realtime.classify import get_live_classifier
from oneai.taxonomy import PatternPack

BASE_INSTRUCTIONS = "You are a voice support agent. Keep replies to one or two short spoken sentences."


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
            pack, _ = self.turn_guide.guide(new_message.text_content or "")
            turn_ctx.add_message(role="system", content=pack_instructions(pack))

    return OneAIAgent()


async def entrypoint(ctx) -> None:
    from livekit.agents import AgentSession
    from livekit.plugins import openai, silero

    await ctx.connect()
    session = AgentSession(stt=openai.STT(), llm=openai.LLM(), tts=openai.TTS(), vad=silero.VAD.load())
    await session.start(room=ctx.room, agent=_build_agent())


if __name__ == "__main__":
    from livekit.agents import WorkerOptions, cli

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
