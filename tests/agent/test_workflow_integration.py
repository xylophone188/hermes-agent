"""Integration tests for multi-agent workflow orchestration.

Tests the full pipeline:
  AgentSpec → RoutingEnvelope → pick_lane → WorkflowOrchestrator → output contract validation
"""

import json
import unittest
from agent import (
    AgentSpec,
    RoutingEnvelope,
    WorkflowOrchestrator,
    pick_lane,
    pick_route,
    validate_agent_spec,
    validate_output_contract,
    validate_routing_envelope,
)


class TestAgentSpecDataclass(unittest.TestCase):
    def test_to_dict_roundtrip(self):
        spec = AgentSpec(
            agent_id="architect",
            role="architect",
            persona="systems-architect",
            capability="workflow-design",
            tier="consultant",
            depth=0,
            output_contract="dag-spec-v1",
        )
        d = spec.to_dict()
        self.assertEqual(d["agent_id"], "architect")
        self.assertEqual(d["tier"], "consultant")
        self.assertEqual(d["output_contract"], "dag-spec-v1")

    def test_defaults(self):
        spec = AgentSpec(agent_id="x", role="y", persona="z", capability="w", tier="e")
        self.assertEqual(spec.depth, 0)
        self.assertEqual(spec.target_profile, "default")
        self.assertEqual(spec.output_contract, "summary-v1")
        self.assertEqual(spec.allowed_tools, [])
        self.assertEqual(spec.disallowed_tools, [])
        self.assertEqual(spec.routing_hints, {})


class TestRoutingEnvelopeDataclass(unittest.TestCase):
    def test_to_dict(self):
        spec = AgentSpec(
            agent_id="w1", role="worker", persona="p", capability="c", tier="executor"
        )
        env = RoutingEnvelope(
            request_id="req-1",
            agent_spec=spec,
            task_kind="implementation",
            summary="do the thing",
        )
        d = env.to_dict()
        self.assertEqual(d["request_id"], "req-1")
        self.assertEqual(d["agent_spec"]["agent_id"], "w1")
        self.assertEqual(d["task_kind"], "implementation")

    def test_to_dict_with_optional_fields(self):
        spec = AgentSpec(
            agent_id="w2", role="worker", persona="p", capability="c", tier="executor"
        )
        env = RoutingEnvelope(
            request_id="req-2",
            agent_spec=spec,
            task_kind="review",
            summary="review the code",
            priority="high",
            constraints={"max_depth": 2},
            metadata={"source": "kanban"},
        )
        d = env.to_dict()
        self.assertEqual(d["priority"], "high")
        self.assertEqual(d["constraints"]["max_depth"], 2)
        self.assertEqual(d["metadata"]["source"], "kanban")


class TestCapabilityBasedRouting(unittest.TestCase):
    def test_workflow_design_lane(self):
        spec = {"agent_id": "a1", "capability": "workflow-design", "tier": "consultant"}
        self.assertEqual(pick_lane(spec), "reasoning")

    def test_implementation_lane(self):
        spec = {"agent_id": "a2", "capability": "implementation", "tier": "executor"}
        self.assertEqual(pick_lane(spec), "code")

    def test_review_lane(self):
        spec = {"agent_id": "a3", "capability": "review", "tier": "verifier"}
        self.assertEqual(pick_lane(spec), "strict")

    def test_research_lane(self):
        spec = {"agent_id": "a4", "capability": "research", "tier": "investigator"}
        self.assertEqual(pick_lane(spec), "research")

    def test_summary_lane(self):
        spec = {"agent_id": "a5", "capability": "summary", "tier": "summarizer"}
        self.assertEqual(pick_lane(spec), "context")

    def test_fallback_default_lane(self):
        spec = {"agent_id": "a6", "tier": "unknown-tier"}
        self.assertEqual(pick_lane(spec), "default")

    def test_capability_overrides_tier(self):
        """Capability field takes precedence over tier for lane selection."""
        spec = {"agent_id": "a7", "capability": "research", "tier": "executor"}
        self.assertEqual(pick_lane(spec), "research")

    def test_pick_route_includes_all_fields(self):
        spec = {
            "agent_id": "exec-1",
            "role": "worker",
            "capability": "implementation",
            "tier": "executor",
            "output_contract": "diff-pack-v1",
        }
        route = pick_route(spec)
        self.assertEqual(route["lane"], "code")
        self.assertEqual(route["agent_id"], "exec-1")
        self.assertEqual(route["role"], "worker")
        self.assertEqual(route["output_contract"], "diff-pack-v1")


class TestValidationUnit(unittest.TestCase):
    def test_validate_agent_spec_ok(self):
        validate_agent_spec({
            "agent_id": "x", "role": "y", "tier": "z", "output_contract": "w"
        })

    def test_validate_agent_spec_missing_fields(self):
        with self.assertRaises(ValueError):
            validate_agent_spec({"agent_id": "x"})

    def test_validate_agent_spec_negative_depth(self):
        with self.assertRaises(ValueError):
            validate_agent_spec({
                "agent_id": "x", "role": "y", "tier": "z",
                "output_contract": "w", "depth": -1,
            })

    def test_validate_envelope_not_dict(self):
        with self.assertRaises(TypeError):
            validate_routing_envelope("not a dict")

    def test_validate_envelope_missing_spec(self):
        with self.assertRaises(ValueError):
            validate_routing_envelope({})


class TestOutputContracts(unittest.TestCase):
    def test_dag_spec_v1_valid(self):
        validate_output_contract("dag-spec-v1", {
            "nodes": [{"id": "n1"}], "edges": []
        })

    def test_dag_spec_v1_empty_nodes(self):
        with self.assertRaises(ValueError):
            validate_output_contract("dag-spec-v1", {"nodes": [], "edges": []})

    def test_work_pack_v1_valid(self):
        validate_output_contract("work-pack-v1", {"tasks": [{"id": "t1"}]})

    def test_review_report_v1_valid(self):
        validate_output_contract("review-report-v1", {
            "verdict": "pass", "evidence": []
        })

    def test_review_report_v1_invalid_verdict(self):
        with self.assertRaises(ValueError):
            validate_output_contract("review-report-v1", {
                "verdict": "maybe", "evidence": []
            })

    def test_summary_v1_valid(self):
        validate_output_contract("summary-v1", {
            "summary": "done", "next_action": "ship"
        })

    def test_summary_v1_missing_next_action(self):
        with self.assertRaises(ValueError):
            validate_output_contract("summary-v1", {"summary": "done"})

    def test_unknown_contract_raises(self):
        with self.assertRaises(ValueError):
            validate_output_contract("unknown-v99", {})

    def test_evidence_pack_v1_valid(self):
        validate_output_contract("evidence-pack-v1", {
            "sources": ["file:///a.py"], "claims": [{"claim": "c", "evidence": "e"}]
        })

    def test_diff_pack_v1_valid(self):
        validate_output_contract("diff-pack-v1", {
            "files_changed": ["a.py"], "commands_run": ["pytest"]
        })


class TestWorkflowOrchestratorIntegration(unittest.TestCase):
    """Integration: multi-node DAG with contract validation between nodes."""

    def test_two_node_pipeline(self):
        """architect (dag-spec-v1) → planner (summary-v1)."""
        call_log = []

        def fake_delegate(goal, context=None, routing_metadata=None, **kwargs):
            call_log.append({"goal": goal, "meta": routing_metadata})
            node_id = (routing_metadata or {}).get("agent_spec", {}).get("agent_id", goal)
            if node_id == "architect":
                return {"nodes": [{"id": "n1", "role": "planner", "inputs": ["intent-v1"], "outputs": ["summary-v1"]}], "edges": [{"from": "n1", "to": "n2"}], "gates": []}
            return {"summary": "done", "next_action": "ship", "artifacts": []}

        artifacts = {}

        orch = WorkflowOrchestrator(
            delegate_fn=fake_delegate,
            persist_fn=lambda nid, out: artifacts.__setitem__(nid, out),
            load_fn=lambda nid: artifacts.get(nid),
        )

        workflow = {
            "kind": "identity-routing",
            "nodes": [
                {
                    "id": "architect",
                    "agent_spec": {
                        "agent_id": "architect",
                        "role": "architect",
                        "tier": "consultant",
                        "output_contract": "dag-spec-v1",
                    },
                    "input_contract": "intent-v1",
                    "output_contract": "dag-spec-v1",
                    "success_criteria": ["nodes", "edges"],
                },
                {
                    "id": "planner",
                    "agent_spec": {
                        "agent_id": "planner",
                        "role": "planner",
                        "tier": "consultant",
                        "output_contract": "summary-v1",
                    },
                    "input_contract": "dag-spec-v1",
                    "output_contract": "summary-v1",
                    "success_criteria": ["summary", "next_action"],
                },
            ],
        }

        result = orch.run(workflow, {"task": "build feature X"})
        self.assertEqual(result["summary"], "done")
        self.assertEqual(len(call_log), 2)
        # Verify routing_metadata is passed through (use agent_id, not raw goal string)
        self.assertEqual(call_log[0]["meta"]["agent_spec"]["agent_id"], "architect")
        self.assertEqual(call_log[1]["meta"]["agent_spec"]["agent_id"], "planner")
        self.assertEqual(call_log[0]["meta"]["agent_spec"]["agent_id"], "architect")
        self.assertEqual(call_log[0]["meta"]["output_contract"], "dag-spec-v1")
        self.assertIn("ORIGINAL TASK", call_log[0]["goal"])
        self.assertIn("build feature X", call_log[0]["goal"])
        self.assertIn("PRIOR OUTPUT", call_log[1]["goal"])
        self.assertIn("nodes", call_log[1]["goal"])

    def test_contract_validation_blocks_bad_output(self):
        """If a node produces invalid output, ValueError is raised."""
        def bad_delegate(goal, context=None, routing_metadata=None, **kwargs):
            return {"wrong_key": "oops"}  # no 'summary' or 'next_action'

        orch = WorkflowOrchestrator(
            delegate_fn=bad_delegate,
            persist_fn=lambda *a: None,
            load_fn=lambda _: None,
        )
        workflow = {
            "kind": "test",
            "nodes": [
                {
                    "id": "n1",
                    "agent_spec": {
                        "agent_id": "n1", "role": "worker",
                        "tier": "summarizer", "output_contract": "summary-v1",
                    },
                    "input_contract": "intent-v1",
                    "output_contract": "summary-v1",
                    "success_criteria": ["summary"],
                    "failure_policy": "abort",
                    "max_retries": 0,
                },
            ],
        }
        from agent import WorkflowError
        with self.assertRaises(WorkflowError):
            orch.run(workflow, {"task": "fail fast"})

    def test_full_6_node_dag(self):
        """Full DAG: architect → planner → researcher → executor → reviewer → synthesizer."""
        step = 0
        call_order = []

        def fake_delegate(goal, context=None, routing_metadata=None, **kwargs):
            nonlocal step
            spec = routing_metadata["agent_spec"]
            node_id = spec["agent_id"]  # use agent_id, not goal string
            call_order.append(node_id)
            lane = pick_lane(spec)
            step += 1

            if node_id == "architect":
                return {"nodes": [{"id": "planner", "role": "planner", "inputs": ["dag-spec-v1"], "outputs": ["work-pack-v1"]}], "edges": [{"from": "architect", "to": "planner"}], "gates": ["validate_dag"]}
            elif node_id == "planner":
                return {"tasks": [{"id": "t1", "title": "Implement X", "owner": "executor", "deps": []}]}
            elif node_id == "researcher":
                return {"sources": ["file:///x.py"], "claims": [{"claim": "c", "evidence": "e"}]}
            elif node_id == "executor":
                return {"files_changed": ["x.py"], "commands_run": ["pytest"], "summary": "done"}
            elif node_id == "reviewer":
                return {"verdict": "pass", "evidence": ["tests pass"]}
            elif node_id == "synthesizer":
                return {"summary": "Feature X implemented and reviewed", "next_action": "merge", "artifacts": ["x.py"]}
            return {"summary": node_id, "next_action": "done", "artifacts": []}

        artifacts = {}
        orch = WorkflowOrchestrator(
            delegate_fn=fake_delegate,
            persist_fn=lambda nid, out: artifacts.__setitem__(nid, out),
            load_fn=lambda nid: artifacts.get(nid),
        )

        from agent.agent_spec import AgentSpec

        workflow = {
            "kind": "identity-routing",
            "nodes": [
                {
                    "id": "architect",
                    "agent_spec": AgentSpec(
                        agent_id="architect", role="architect", persona="sys",
                        capability="workflow-design", tier="consultant", output_contract="dag-spec-v1",
                    ).to_dict(),
                    "input_contract": "intent-v1",
                    "output_contract": "dag-spec-v1",
                    "success_criteria": ["nodes", "edges"],
                },
                {
                    "id": "planner",
                    "agent_spec": AgentSpec(
                        agent_id="planner", role="planner", persona="dec",
                        capability="task-planning", tier="consultant", output_contract="work-pack-v1",
                    ).to_dict(),
                    "input_contract": "dag-spec-v1",
                    "output_contract": "work-pack-v1",
                    "success_criteria": ["tasks"],
                },
                {
                    "id": "researcher",
                    "agent_spec": AgentSpec(
                        agent_id="researcher", role="worker", persona="find",
                        capability="research", tier="investigator", output_contract="evidence-pack-v1",
                    ).to_dict(),
                    "input_contract": "work-pack-v1",
                    "output_contract": "evidence-pack-v1",
                    "success_criteria": ["sources", "claims"],
                },
                {
                    "id": "executor",
                    "agent_spec": AgentSpec(
                        agent_id="executor", role="worker", persona="impl",
                        capability="implementation", tier="executor", output_contract="diff-pack-v1",
                    ).to_dict(),
                    "input_contract": "work-pack-v1",
                    "output_contract": "diff-pack-v1",
                    "success_criteria": ["files_changed", "commands_run"],
                },
                {
                    "id": "reviewer",
                    "agent_spec": AgentSpec(
                        agent_id="reviewer", role="verifier", persona="rev",
                        capability="review", tier="verifier", output_contract="review-report-v1",
                    ).to_dict(),
                    "input_contract": "diff-pack-v1",
                    "output_contract": "review-report-v1",
                    "success_criteria": ["verdict", "evidence"],
                },
                {
                    "id": "synthesizer",
                    "agent_spec": AgentSpec(
                        agent_id="synthesizer", role="synthesizer", persona="merge",
                        capability="summary", tier="summarizer", output_contract="summary-v1",
                    ).to_dict(),
                    "input_contract": "review-report-v1",
                    "output_contract": "summary-v1",
                    "success_criteria": ["summary", "next_action"],
                },
            ],
        }

        result = orch.run(workflow, {"task": "build feature X"})
        self.assertEqual(result["summary"], "Feature X implemented and reviewed")
        self.assertEqual(result["next_action"], "merge")
        self.assertEqual(call_order, [
            "architect", "planner", "researcher", "executor", "reviewer", "synthesizer"
        ])
        # All 6 artifacts persisted
        self.assertEqual(len(artifacts), 6)
        # Last artifact is synthesizer output
        self.assertIn("synthesizer", artifacts)


class TestWorkflowCommandParsing(unittest.TestCase):
    """Test that /workflow command would parse correctly (no CLI instantiation needed)."""

    def test_workflow_registry_exists(self):
        from hermes_cli.commands import COMMAND_REGISTRY
        names = [c.name for c in COMMAND_REGISTRY]
        self.assertIn("workflow", names)

    def test_workflow_command_def(self):
        from hermes_cli.commands import COMMAND_REGISTRY
        wf_cmd = next(c for c in COMMAND_REGISTRY if c.name == "workflow")
        self.assertEqual(wf_cmd.args_hint, "<task>")
        self.assertEqual(wf_cmd.category, "Tools & Skills")


if __name__ == "__main__":
    unittest.main()
