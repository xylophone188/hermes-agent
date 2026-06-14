from __future__ import annotations

from typing import Any

# Model names map to 9router combo names (localhost:20128/v1)
# consultant/verifier tiers get intelligence (top reasoning chain)
# all worker tiers get auto (cheap-first, escalates inside 9router)
_TIER_TO_MODEL: dict[str, str] = {
    "consultant": "intelligence",
    "verifier": "intelligence",
    "executor": "auto",
    "investigator": "auto",
    "summarizer": "auto",
    "default": "auto",
}

# Capability overrides: reasoning-heavy capabilities use intelligence
_CAP_TO_MODEL: dict[str, str] = {
    "workflow-design": "intelligence",
    "task-planning": "intelligence",
    "reasoning": "intelligence",
    "review": "intelligence",
    "verification": "intelligence",
    "implementation": "auto",
    "code": "auto",
    "summary": "auto",
    "synthesis": "auto",
    "research": "auto",
    "fact-finding": "auto",
}


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


def pick_model(spec: dict[str, Any]) -> str:
    """Return the 9router model name for this agent spec.

    Capability takes precedence over tier; tier over default.
    Both resolve to a 9router combo: 'intelligence' or 'auto'.
    """
    cap = str(spec.get("capability") or "").lower()
    if cap and cap in _CAP_TO_MODEL:
        return _CAP_TO_MODEL[cap]
    tier = str(spec.get("tier") or "").lower()
    return _TIER_TO_MODEL.get(tier, "auto")


def pick_route(spec: dict[str, Any]) -> dict[str, Any]:
    lane = pick_lane(spec)
    model = pick_model(spec)
    return {
        "lane": lane,
        "model": model,
        "agent_id": spec["agent_id"],
        "role": spec["role"],
        "output_contract": spec["output_contract"],
    }