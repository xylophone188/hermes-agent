from __future__ import annotations
import pytest
from agent.routing import pick_lane, pick_model, pick_route


# ── pick_model ──────────────────────────────────────────────
class TestPickModel:
    def _spec(self, tier=None, capability=None):
        return {
            "agent_id": "x", "role": "r", "tier": tier,
            "capability": capability, "output_contract": "summary-v1",
        }

    def test_consultant_tier_gives_intelligence(self):
        assert pick_model(self._spec(tier="consultant")) == "intelligence"

    def test_verifier_tier_gives_intelligence(self):
        assert pick_model(self._spec(tier="verifier")) == "intelligence"

    def test_executor_tier_gives_auto(self):
        assert pick_model(self._spec(tier="executor")) == "auto"

    def test_investigator_tier_gives_auto(self):
        assert pick_model(self._spec(tier="investigator")) == "auto"

    def test_summarizer_tier_gives_auto(self):
        assert pick_model(self._spec(tier="summarizer")) == "auto"

    def test_unknown_tier_gives_auto(self):
        assert pick_model(self._spec(tier="unknown")) == "auto"

    def test_no_tier_gives_auto(self):
        assert pick_model(self._spec()) == "auto"

    def test_capability_overrides_tier(self):
        # executor tier but reasoning capability → intelligence
        assert pick_model(self._spec(tier="executor", capability="reasoning")) == "intelligence"

    def test_review_capability_gives_intelligence(self):
        assert pick_model(self._spec(tier="executor", capability="review")) == "intelligence"

    def test_workflow_design_gives_intelligence(self):
        assert pick_model(self._spec(capability="workflow-design")) == "intelligence"

    def test_implementation_cap_gives_auto(self):
        assert pick_model(self._spec(tier="consultant", capability="implementation")) == "auto"


# ── pick_lane ──────────────────────────────────────────────
class TestPickLane:
    def _spec(self, tier=None, capability=None):
        return {"agent_id": "x", "role": "r", "tier": tier,
                "capability": capability, "output_contract": "summary-v1"}

    def test_consultant_lane(self):
        assert pick_lane(self._spec(tier="consultant")) == "reasoning"

    def test_executor_lane(self):
        assert pick_lane(self._spec(tier="executor")) == "code"

    def test_verifier_lane(self):
        assert pick_lane(self._spec(tier="verifier")) == "strict"

    def test_summarizer_lane(self):
        assert pick_lane(self._spec(tier="summarizer")) == "context"

    def test_investigator_lane(self):
        assert pick_lane(self._spec(tier="investigator")) == "research"

    def test_default_lane(self):
        assert pick_lane(self._spec()) == "default"

    def test_capability_overrides_tier_lane(self):
        # summarizer tier but workflow-design capability → reasoning
        assert pick_lane(self._spec(tier="summarizer", capability="workflow-design")) == "reasoning"


# ── pick_route ──────────────────────────────────────────────
class TestPickRoute:
    def _spec(self, tier="executor", capability=None):
        return {
            "agent_id": "builder", "role": "worker",
            "tier": tier, "capability": capability,
            "output_contract": "diff-pack-v1",
        }

    def test_route_has_required_keys(self):
        r = pick_route(self._spec())
        assert {"lane", "model", "agent_id", "role", "output_contract"} <= r.keys()

    def test_route_model_consultant(self):
        r = pick_route(self._spec(tier="consultant"))
        assert r["model"] == "intelligence"
        assert r["lane"] == "reasoning"

    def test_route_model_executor(self):
        r = pick_route(self._spec(tier="executor"))
        assert r["model"] == "auto"
        assert r["lane"] == "code"

    def test_route_agent_id_preserved(self):
        r = pick_route(self._spec())
        assert r["agent_id"] == "builder"
