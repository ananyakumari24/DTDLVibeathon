"""Shared labels and data formats for OneAI. Only change these at the daily sync."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ISSUES = ("billing", "shipping", "account_access")
INTENTS = ("troubleshoot", "info", "escalate")
STYLES = ("empathic", "instructional")

Issue = Literal["billing", "shipping", "account_access"]
Intent = Literal["troubleshoot", "info", "escalate"]
Style = Literal["empathic", "instructional"]


class Turn(BaseModel):
    speaker: Literal["customer", "agent"]
    text: str
    ts: float = 0.0


class Transcript(BaseModel):
    call_id: str
    issue: Issue
    intent: Intent
    style: Style
    turns: List[Turn]


class Conversation(BaseModel):
    call_id: str
    issue: Issue
    intent: Intent
    style: Style
    turns: List[Turn]
    customer_text: str
    agent_text: str
    full_text: str
    friction_flags: List[str] = Field(default_factory=list)


class Insight(BaseModel):
    call_id: str
    issue: str
    question_type: str
    agent_style: str
    friction_flags: List[str] = Field(default_factory=list)
    label_methods: Dict[str, str] = Field(default_factory=dict)


class Cluster(BaseModel):
    id: str
    intent: str
    issue: str
    style: str
    member_call_ids: List[str]
    centroid: List[float] = Field(default_factory=list)
    sample_phrasings: List[str] = Field(default_factory=list)


class PatternPack(BaseModel):
    pack_id: str
    intent: str
    issue: str
    style: str
    top_customer_phrasings: List[str] = Field(default_factory=list)
    recommended_opener: str = ""
    style_cue: str = ""
    follow_ups: List[str] = Field(default_factory=list)
    friction_alerts: List[str] = Field(default_factory=list)
    agent_patterns: List[str] = Field(default_factory=list)
    typical_flow: List[str] = Field(default_factory=list)
    centroid: List[float] = Field(default_factory=list)
    call_count: Optional[int] = None
    built_with: str = "rules"
    is_placeholder: bool = False
