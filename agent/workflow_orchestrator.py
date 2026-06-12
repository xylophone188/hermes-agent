from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.validation import validate_output_contract


@dataclass
class WorkflowNode:
    id: str
    agent_spec: dict[str, Any]
    input_contract: str
    output_contract: str
    success_criteria: list[str]
    failure_policy: str = "revise"
    max_retries: int = 2


class WorkflowOrchestrator:
    def __init__(self, delegate_fn, persist_fn, load_fn):
        self.delegate_fn = delegate_fn
        self.persist_fn = persist_fn
        self.load_fn = load_fn

    def run(self, workflow: dict[str, Any], initial_input: Any) -> Any:
        current_input = initial_input
        last_output = None

        for node_data in workflow["nodes"]:
            node = WorkflowNode(**node_data)

            result = self.delegate_fn(
                goal=node.id,
                context=current_input,
                routing_metadata={
                    "agent_spec": node.agent_spec,
                    "output_contract": node.output_contract,
                    "task_kind": workflow.get("kind", "workflow"),
                },
            )

            validate_output_contract(node.output_contract, result)
            self.persist_fn(node.id, result)

            last_output = result
            current_input = result

        return last_output
