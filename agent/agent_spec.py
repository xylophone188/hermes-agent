from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AgentSpec:
    agent_id: str
    role: str
    persona: str
    capability: str
    tier: str
    depth: int = 0
    target_profile: str = "default"
    output_contract: str = "summary-v1"
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    routing_hints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoutingEnvelope:
    request_id: str
    agent_spec: AgentSpec
    task_kind: str
    summary: str
    priority: str = "normal"
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "agent_spec": self.agent_spec.to_dict(),
            "task_kind": self.task_kind,
            "summary": self.summary,
            "priority": self.priority,
            "constraints": self.constraints,
            "metadata": self.metadata,
        }
