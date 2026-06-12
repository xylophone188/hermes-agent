from __future__ import annotations

from typing import Any


def pick_lane(spec: dict[str, Any]) -> str:
    tier = spec.get("tier")
    capability = spec.get("capability")

    if capability:
        cap = str(capability).lower()
        if cap in {"workflow-design", "task-planning", "reasoning"}:
            return "reasoning"
        if cap in {"implementation", "code"}:
            return "code"
        if cap in {"review", "verification"}:
            return "strict"
        if cap in {"summary", "synthesis"}:
            return "context"
        if cap in {"research", "fact-finding"}:
            return "research"

    if tier == "consultant":
        return "reasoning"
    if tier == "executor":
        return "code"
    if tier == "verifier":
        return "strict"
    if tier == "summarizer":
        return "context"
    if tier == "investigator":
        return "research"

    return "default"


def pick_route(spec: dict[str, Any]) -> dict[str, Any]:
    lane = pick_lane(spec)
    return {
        "lane": lane,
        "agent_id": spec["agent_id"],
        "role": spec["role"],
        "output_contract": spec["output_contract"],
    }
