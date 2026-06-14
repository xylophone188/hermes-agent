"""Agent internals -- extracted modules from run_agent.py.

These modules contain pure utility functions and self-contained classes
that were previously embedded in the 3,600-line run_agent.py. Extracting
them makes run_agent.py focused on the AIAgent orchestrator class.
"""

from . import jiter_preload as _jiter_preload  # noqa: F401

from .agent_spec import AgentSpec, RoutingEnvelope
from .routing import pick_lane, pick_route
from .validation import (
    validate_agent_spec,
    validate_output_contract,
    validate_routing_envelope,
)
from .workflow_orchestrator import WorkflowNode, WorkflowOrchestrator, WorkflowError
