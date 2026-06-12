from __future__ import annotations
import pytest
from unittest.mock import MagicMock, call
from agent.workflow_orchestrator import (
    WorkflowNode, NodeResult, WorkflowOrchestrator,
    CONSULTANT_NODE_SUFFIX, CONSULTANT_THRESHOLD_CONFIDENCE,
)


def _make_node(node_id="n1", threshold=0.7, consultant_tier="consultant"):
    return WorkflowNode(
        id=node_id,
        agent_spec={"agent_id": node_id, "role": "worker", "tier": "executor",
                    "capability": "implementation", "output_contract": "diff-pack-v1"},
        input_contract="work-pack-v1",
        output_contract="diff-pack-v1",
        success_criteria=["files_changed"],
        confidence_threshold=threshold,
        consultant_tier=consultant_tier,
    )


def _make_result(status="done", confidence=1.0, output=None):
    return NodeResult(
        node_id="n1",
        status=status,
        output=output or {"files_changed": ["x.py"], "commands_run": []},
        raw_summary="ok",
        confidence=confidence,
    )


def _make_orchestrator():
    delegate = MagicMock()
    persist = MagicMock()
    load = MagicMock()
    return WorkflowOrchestrator(delegate, persist, load), delegate, persist


class TestShouldEscalate:
    def test_failed_result_escalates(self):
        orc, _, _ = _make_orchestrator()
        node = _make_node(threshold=0.7)
        result = _make_result(status="failed", confidence=1.0)
        assert orc._should_escalate(result, node) is True

    def test_low_confidence_escalates(self):
        orc, _, _ = _make_orchestrator()
        node = _make_node(threshold=0.7)
        result = _make_result(status="done", confidence=0.5)
        assert orc._should_escalate(result, node) is True

    def test_boundary_confidence_escalates(self):
        orc, _, _ = _make_orchestrator()
        node = _make_node(threshold=0.7)
        result = _make_result(status="done", confidence=0.69)
        assert orc._should_escalate(result, node) is True

    def test_sufficient_confidence_no_escalation(self):
        orc, _, _ = _make_orchestrator()
        node = _make_node(threshold=0.7)
        result = _make_result(status="done", confidence=0.7)
        assert orc._should_escalate(result, node) is False

    def test_high_confidence_done_no_escalation(self):
        orc, _, _ = _make_orchestrator()
        node = _make_node(threshold=0.7)
        result = _make_result(status="done", confidence=1.0)
        assert orc._should_escalate(result, node) is False

    def test_zero_threshold_never_escalates_on_confidence(self):
        orc, _, _ = _make_orchestrator()
        node = _make_node(threshold=0.0)
        result = _make_result(status="done", confidence=0.0)
        assert orc._should_escalate(result, node) is False


class TestRunConsultant:
    def test_consultant_node_id_has_suffix(self):
        orc, _, _ = _make_orchestrator()
        node = _make_node("build")
        worker_result = _make_result(status="failed")

        # Patch _run_node to capture what node it receives
        captured = []
        def fake_run_node(n, ctx, kind, original_task):
            captured.append(n)
            return _make_result(status="done", confidence=1.0)
        orc._run_node = fake_run_node

        orc._run_consultant(node, {}, "test", "task", worker_result)
        assert captured[0].id == "build" + CONSULTANT_NODE_SUFFIX

    def test_consultant_spec_tier_overridden(self):
        orc, _, _ = _make_orchestrator()
        node = _make_node("build", consultant_tier="consultant")
        worker_result = _make_result(status="failed")

        captured = []
        def fake_run_node(n, ctx, kind, original_task):
            captured.append(n)
            return _make_result(status="done", confidence=1.0)
        orc._run_node = fake_run_node

        orc._run_consultant(node, {}, "test", "task", worker_result)
        assert captured[0].agent_spec["tier"] == "consultant"
        assert captured[0].agent_spec["_model_override"] == "intelligence"

    def test_worker_result_injected_into_context(self):
        orc, _, _ = _make_orchestrator()
        node = _make_node()
        worker_result = _make_result(status="failed", confidence=0.4)
        worker_result.error = "contract fail"

        captured_ctx = []
        def fake_run_node(n, ctx, kind, original_task):
            captured_ctx.append(ctx)
            return _make_result(status="done", confidence=1.0)
        orc._run_node = fake_run_node

        orc._run_consultant(node, {"task": "x"}, "test", "task", worker_result)
        assert "_worker_result" in captured_ctx[0]
        assert captured_ctx[0]["_worker_result"]["status"] == "failed"
        assert captured_ctx[0]["_worker_result"]["confidence"] == 0.4


class TestNodeResultDefaults:
    def test_confidence_default_is_1(self):
        r = NodeResult(node_id="x", status="done", output={}, raw_summary="")
        assert r.confidence == 1.0

    def test_confidence_settable(self):
        r = NodeResult(node_id="x", status="done", output={}, raw_summary="", confidence=0.3)
        assert r.confidence == 0.3


class TestWorkflowNodeDefaults:
    def test_confidence_threshold_default(self):
        n = _make_node()
        assert n.confidence_threshold == 0.7

    def test_consultant_tier_default(self):
        n = _make_node()
        assert n.consultant_tier == "consultant"
