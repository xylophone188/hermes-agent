"""Tests for WorkflowOrchestrator with on_node_start/on_node_done callbacks."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from agent.workflow_orchestrator import WorkflowOrchestrator, WorkflowNode, NodeResult, WorkflowError
from agent.agent_spec import AgentSpec


def _make_spec(agent_id: str, tier: str = "executor") -> dict:
    return AgentSpec(
        agent_id=agent_id,
        role=agent_id,
        persona=agent_id,
        capability=agent_id,
        tier=tier,
        depth=0,
        target_profile="default",
        output_contract="summary-v1",
    ).to_dict()


def test_workflow_callbacks_fired():
    """on_node_start and on_node_done should be called for each node."""
    starts = []
    dones = []

    def on_start(node_id, spec):
        starts.append(node_id)

    def on_done(node_id, result):
        # result is NodeResult object
        dones.append((node_id, getattr(result, "status", "ok")))

    def delegate_fn(goal, routing_metadata=None):
        return {"summary": f"done-{routing_metadata['agent_spec']['agent_id']}", "next_action": "none"}

    orch = WorkflowOrchestrator(
        delegate_fn=delegate_fn,
        persist_fn=lambda nid, res: None,
        load_fn=lambda nid: None,
        on_node_start=on_start,
        on_node_done=on_done,
    )

    dag = {
        "kind": "test",
        "nodes": [
            {
                "id": "a",
                "agent_spec": _make_spec("a"),
                "input_contract": "intent-v1",
                "output_contract": "summary-v1",
                "success_criteria": ["summary", "next_action"],
            },
            {
                "id": "b",
                "agent_spec": _make_spec("b"),
                "input_contract": "summary-v1",
                "output_contract": "summary-v1",
                "success_criteria": ["summary", "next_action"],
            },
        ],
    }

    result = orch.run(dag, {"task": "test"})
    assert result is not None
    assert starts == ["a", "b"]
    assert len(dones) == 2


def test_workflow_escalation_callback():
    """Consultant escalation should also fire callbacks."""
    starts = []
    dones = []

    def on_start(node_id, spec):
        starts.append(node_id)

    def on_done(node_id, result):
        dones.append(node_id)

    call_count = 0

    def delegate_fn(goal, routing_metadata=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call returns invalid JSON → parse fails → status=failed
            return "not valid json at all"
        return {"summary": "fixed", "next_action": "none"}

    orch = WorkflowOrchestrator(
        delegate_fn=delegate_fn,
        persist_fn=lambda nid, res: None,
        load_fn=lambda nid: None,
        on_node_start=on_start,
        on_node_done=on_done,
    )

    dag = {
        "kind": "test",
        "nodes": [
            {
                "id": "fail_then_escalate",
                "agent_spec": _make_spec("fail_then_escalate", tier="executor"),
                "input_contract": "intent-v1",
                "output_contract": "summary-v1",
                "success_criteria": ["summary", "next_action"],
                "max_retries": 0,  # 1 attempt only, fail immediately
            },
        ],
    }

    result = orch.run(dag, {"task": "test"})
    assert result is not None
    # Should have worker + consultant node callbacks
    assert "fail_then_escalate" in starts
    assert "fail_then_escalate__consultant" in starts
    assert len(dones) >= 1


def test_workflow_final_output_is_dict():
    """Final output should be a dict with summary-v1 contract."""
    def delegate_fn(goal, routing_metadata=None):
        return {"summary": "final", "next_action": "done", "artifacts": []}

    orch = WorkflowOrchestrator(
        delegate_fn=delegate_fn,
        persist_fn=lambda nid, res: None,
        load_fn=lambda nid: None,
    )

    dag = {
        "kind": "test",
        "nodes": [
            {
                "id": "synth",
                "agent_spec": _make_spec("synth", tier="summarizer"),
                "input_contract": "review-report-v1",
                "output_contract": "summary-v1",
                "success_criteria": ["summary", "next_action"],
            },
        ],
    }

    result = orch.run(dag, {"task": "test"})
    assert isinstance(result, dict)
    assert result.get("summary") == "final"
    assert result.get("next_action") == "done"
