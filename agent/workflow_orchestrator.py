from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent.validation import validate_output_contract

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Escalation — cheap worker runs first; if confidence < threshold
# OR contract fails, re-run with consultant (intelligence) model.
# ──────────────────────────────────────────────────────────
CONSULTANT_THRESHOLD_CONFIDENCE: float = 0.7
CONSULTANT_NODE_SUFFIX: str = "__consultant"


# ──────────────────────────────────────────────────────────
# Contract prompts — appended to each node's goal so the
# model knows exactly what JSON shape to return.
# ──────────────────────────────────────────────────────────
_CONTRACT_PROMPTS: dict[str, str] = {
    "dag-spec-v1": (
        "\n\nYou MUST respond with ONLY a JSON object (no prose, no markdown fences) "
        "matching this schema:\n"
        '{"nodes": [{"id": str, "role": str, "inputs": [str], "outputs": [str]}], '
        '"edges": [{"from": str, "to": str}], '
        '"gates": [str]}'
    ),
    "work-pack-v1": (
        "\n\nYou MUST respond with ONLY a JSON object (no prose, no markdown fences) "
        "matching this schema:\n"
        '{"tasks": [{"id": str, "title": str, "owner": str, "deps": [str]}]}'
    ),
    "evidence-pack-v1": (
        "\n\nYou MUST respond with ONLY a JSON object (no prose, no markdown fences) "
        "matching this schema:\n"
        '{"sources": [str], "claims": [{"claim": str, "evidence": str}]}'
    ),
    "diff-pack-v1": (
        "\n\nYou MUST respond with ONLY a JSON object (no prose, no markdown fences) "
        "matching this schema:\n"
        '{"files_changed": [str], "commands_run": [str], "summary": str}'
    ),
    "review-report-v1": (
        "\n\nYou MUST respond with ONLY a JSON object (no prose, no markdown fences) "
        "matching this schema:\n"
        '{"verdict": "pass"|"fail", "evidence": [str], "notes": str}'
    ),
    "summary-v1": (
        "\n\nYou MUST respond with ONLY a JSON object (no prose, no markdown fences) "
        "matching this schema:\n"
        '{"summary": str, "next_action": str, "artifacts": [str]}'
    ),
}

# ──────────────────────────────────────────────────────────
# Role system prompts injected as context prefix
# ──────────────────────────────────────────────────────────
_ROLE_PROMPTS: dict[str, str] = {
    "architect": (
        "You are a systems architect. Your ONLY job is to design the workflow DAG "
        "for the given task. Do NOT implement anything. Output the DAG as JSON."
    ),
    "planner": (
        "You are a task decomposer. Your ONLY job is to break the DAG into concrete "
        "tasks with owners and dependencies. Do NOT implement anything. Output JSON."
    ),
    "researcher": (
        "You are a fact-finder. Your ONLY job is to research the task and gather "
        "evidence (files, docs, prior art). Output JSON with sources and claims."
    ),
    "executor": (
        "You are a code implementer. Your ONLY job is to implement the assigned tasks. "
        "Output JSON listing files changed and commands run."
    ),
    "reviewer": (
        "You are a strict code reviewer. Your ONLY job is to verify the implementation "
        "meets requirements. Output JSON with verdict (pass/fail) and evidence."
    ),
    "synthesizer": (
        "You are a final synthesizer. Your ONLY job is to merge all outputs into a "
        "human-readable summary with a clear next action. Output JSON."
    ),
}


def _extract_json_from_text(text: str) -> Optional[dict[str, Any]]:
    """Extract first valid JSON object from free-form text."""
    if not text:
        return None

    # Try direct parse first
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Try stripping markdown fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first {...} block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _build_node_goal(
    node_id: str,
    agent_spec: dict[str, Any],
    task_context: Any,
    output_contract: str,
    original_task: str = "",
) -> str:
    """Build the full goal string for a node, including role prompt + contract.

    Structure:
      1. Role-specific instruction (what this agent does / doesn't do)
      2. ORIGINAL TASK anchor (always present so model keeps focus)
      3. INPUT from prior node (empty on first node)
      4. OUTPUT CONTRACT schema (JSON-only requirement)
    """
    role = agent_spec.get("role", node_id)
    role_prompt = _ROLE_PROMPTS.get(node_id) or _ROLE_PROMPTS.get(role, "")
    contract_prompt = _CONTRACT_PROMPTS.get(output_contract, "")

    parts: list[str] = []

    if role_prompt:
        parts.append(role_prompt)

    # Always anchor to the original user task
    if original_task:
        parts.append(f"\n\nORIGINAL TASK:\n{original_task}")

    # Serialise upstream context (only if it adds info beyond the task itself)
    if task_context:
        if isinstance(task_context, dict):
            # If it's just the initial {"task": ...} wrapper, skip — already shown above
            if set(task_context.keys()) == {"task"}:
                ctx_str = None
            else:
                ctx_str = json.dumps(task_context, ensure_ascii=False, indent=2)
        else:
            ctx_str = str(task_context).strip() or None

        if ctx_str:
            parts.append(f"\n\nPRIOR OUTPUT (from upstream node):\n{ctx_str}")

    if contract_prompt:
        parts.append(contract_prompt)

    return "\n".join(parts)


@dataclass
class WorkflowNode:
    id: str
    agent_spec: dict[str, Any]
    input_contract: str
    output_contract: str
    success_criteria: list[str]
    failure_policy: str = "revise"
    max_retries: int = 2
    confidence_threshold: float = 0.7
    consultant_tier: str = "consultant"


@dataclass
class NodeResult:
    node_id: str
    status: str  # "ok" | "failed" | "retried"
    output: Optional[dict[str, Any]]
    raw_summary: str
    attempts: int = 1
    error: Optional[str] = None
    confidence: float = 1.0


class WorkflowOrchestrator:
    def __init__(
        self,
        delegate_fn: Callable,
        persist_fn: Callable,
        load_fn: Callable,
        on_node_start: Optional[Callable] = None,
        on_node_done: Optional[Callable] = None,
    ):
        self.delegate_fn = delegate_fn
        self.persist_fn = persist_fn
        self.load_fn = load_fn
        self.on_node_start = on_node_start
        self.on_node_done = on_node_done

    def _run_node(
        self,
        node: WorkflowNode,
        task_context: Any,
        workflow_kind: str,
        original_task: str = "",
    ) -> NodeResult:
        """Run a single node with retry. Returns NodeResult."""
        spec = node.agent_spec
        last_error = None

        for attempt in range(1, node.max_retries + 2):
            goal = _build_node_goal(
                node_id=node.id,
                agent_spec=spec,
                task_context=task_context,
                output_contract=node.output_contract,
                original_task=original_task,
            )

            # If retrying, add error feedback
            if attempt > 1 and last_error:
                goal += (
                    f"\n\nPREVIOUS ATTEMPT FAILED: {last_error}\n"
                    "Fix the output and try again. You MUST return valid JSON."
                )

            try:
                raw = self.delegate_fn(
                    goal=goal,
                    routing_metadata={
                        "agent_spec": spec,
                        "output_contract": node.output_contract,
                        "task_kind": workflow_kind,
                    },
                )
            except Exception as exc:
                last_error = f"delegate_fn raised: {exc}"
                logger.warning("Node %s attempt %d delegate error: %s", node.id, attempt, exc)
                continue

            # Try to parse output contract from summary
            extracted = self._extract_summary(raw)

            # If already a dict, use directly; otherwise parse JSON from text
            if isinstance(extracted, dict):
                parsed = extracted
                summary = json.dumps(extracted, ensure_ascii=False)
            else:
                summary = extracted
                parsed = _extract_json_from_text(summary)

            if parsed is None:
                last_error = f"no JSON found in output (len={len(summary)})"
                logger.warning("Node %s attempt %d: %s", node.id, attempt, last_error)
                continue

            # Validate contract
            try:
                validate_output_contract(node.output_contract, parsed)
            except (ValueError, TypeError) as exc:
                last_error = f"contract validation failed: {exc}"
                logger.warning("Node %s attempt %d: %s", node.id, attempt, last_error)
                continue

            status = "ok" if attempt == 1 else "retried"
            return NodeResult(
                node_id=node.id,
                status=status,
                output=parsed,
                raw_summary=summary,
                attempts=attempt,
            )

        # All attempts exhausted
        return NodeResult(
            node_id=node.id,
            status="failed",
            output=None,
            raw_summary="",
            attempts=node.max_retries + 1,
            error=last_error,
        )

    def _should_escalate(self, result: "NodeResult", node: "WorkflowNode") -> bool:
        """Return True if the worker result warrants consultant escalation.

        Triggers when:
        - result.status == 'failed'  (all retries exhausted / contract failed)
        - result.confidence < node.confidence_threshold
        """
        if result.status == "failed":
            return True
        if result.confidence < node.confidence_threshold:
            return True
        return False

    def _run_consultant(
        self,
        node: "WorkflowNode",
        task_context: Any,
        workflow_kind: str,
        original_task: str,
        worker_result: "NodeResult",
    ) -> "NodeResult":
        """Re-run node with consultant (intelligence) model via routing_metadata override."""
        consultant_spec = dict(node.agent_spec)
        consultant_spec["tier"] = node.consultant_tier
        # Force intelligence model for consultant lane
        consultant_spec["_model_override"] = "intelligence"

        consultant_node = WorkflowNode(
            id=node.id + CONSULTANT_NODE_SUFFIX,
            agent_spec=consultant_spec,
            input_contract=node.input_contract,
            output_contract=node.output_contract,
            success_criteria=node.success_criteria,
            failure_policy="abort",   # consultant failure → hard abort
            max_retries=1,
            confidence_threshold=0.0,  # no further escalation
            consultant_tier=node.consultant_tier,
        )

        # Enrich context with worker attempt info
        enriched_context: Any
        if isinstance(task_context, dict):
            enriched_context = dict(task_context)
            enriched_context["_worker_result"] = {
                "status": worker_result.status,
                "error": worker_result.error,
                "raw": worker_result.raw_summary[:500] if worker_result.raw_summary else "",
                "confidence": worker_result.confidence,
            }
        else:
            enriched_context = task_context

        logger.info(
            "Escalating node '%s' to consultant (worker confidence=%.2f, status=%s)",
            node.id, worker_result.confidence, worker_result.status,
        )
        return self._run_node(consultant_node, enriched_context, workflow_kind, original_task)

    @staticmethod
    def _extract_summary(raw: Any) -> str | dict:
        """Extract structured output from delegate_task return value.

        Returns dict directly if raw is already a valid structured object,
        otherwise returns a string for JSON extraction.
        """
        # Already a dict — could be a contract-shaped object from tests/mocks
        if isinstance(raw, dict):
            # If it looks like a delegate_task result envelope, unwrap it
            if "results" in raw:
                results = raw["results"]
                if results and isinstance(results, list):
                    first = results[0]
                    if isinstance(first, dict):
                        summary_val = first.get("summary") or first.get("error") or ""
                        return summary_val
            # Otherwise it IS the output already
            return raw

        if isinstance(raw, str):
            # Try to parse as JSON (delegate_task returns JSON string)
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return raw

            # {"results": [{"summary": "..."}]}
            if isinstance(data, dict):
                results = data.get("results") or []
                if results and isinstance(results, list):
                    first = results[0]
                    if isinstance(first, dict):
                        return first.get("summary") or first.get("error") or ""
                return data.get("summary") or data.get("error") or ""

        return str(raw) if raw else ""

    def run(self, workflow: dict[str, Any], initial_input: Any) -> dict[str, Any]:
        """
        Execute the DAG sequentially. Each node's output becomes the
        next node's input. Returns the final node's output.

        Raises WorkflowError if a node with failure_policy='abort' fails.
        """
        current_input = initial_input
        results: list[NodeResult] = []
        kind = workflow.get("kind", "workflow")

        # Extract original task string — carried through every node for context anchoring
        if isinstance(initial_input, dict):
            original_task = str(initial_input.get("task", ""))
        else:
            original_task = str(initial_input or "")

        for node_data in workflow["nodes"]:
            node = WorkflowNode(**node_data)

            if self.on_node_start:
                try:
                    self.on_node_start(node.id, node.agent_spec)
                except Exception:
                    pass

            result = self._run_node(node, current_input, kind, original_task=original_task)

            # Escalation gate: cheap worker first, consultant on low confidence or failure
            if self._should_escalate(result, node):
                result = self._run_consultant(
                    node, current_input, kind, original_task, worker_result=result
                )

            results.append(result)

            if self.on_node_done:
                try:
                    self.on_node_done(node.id, result)
                except Exception:
                    pass

            if result.status == "failed":
                if node.failure_policy == "abort":
                    raise WorkflowError(
                        f"Node '{node.id}' failed after {result.attempts} attempts: "
                        f"{result.error}"
                    )
                # failure_policy == "continue" or "revise" — use fallback
                logger.warning(
                    "Node %s failed (policy=%s), continuing with raw summary as context",
                    node.id,
                    node.failure_policy,
                )
                # Pass raw summary as context so next node has something to work with
                current_input = {"_failed_node": node.id, "_raw": result.raw_summary}
            else:
                self.persist_fn(node.id, result.output)
                current_input = result.output

        # Return last successful output (or last result's output)
        for r in reversed(results):
            if r.output is not None:
                return r.output

        return {"summary": "Workflow completed with no structured output", "next_action": "review"}


class WorkflowError(Exception):
    """Raised when a node fails and failure_policy is 'abort'."""
    pass
