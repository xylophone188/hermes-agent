from __future__ import annotations

from typing import Any


REQUIRED_AGENT_FIELDS = ("agent_id", "role", "tier", "output_contract")


def validate_agent_spec(spec: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_AGENT_FIELDS if not spec.get(k)]
    if missing:
        raise ValueError(f"agent spec missing required fields: {', '.join(missing)}")

    depth = spec.get("depth", 0)
    if not isinstance(depth, int) or depth < 0:
        raise ValueError("agent spec depth must be non-negative int")


def validate_routing_envelope(env: dict[str, Any]) -> None:
    if not isinstance(env, dict):
        raise TypeError("routing envelope must be dict")
    if "agent_spec" not in env:
        raise ValueError("routing envelope missing agent_spec")
    validate_agent_spec(env["agent_spec"])


def validate_output_contract(contract: str, output: Any) -> None:
    if contract == "dag-spec-v1":
        _validate_dag_spec(output)
    elif contract == "work-pack-v1":
        _validate_work_pack(output)
    elif contract == "evidence-pack-v1":
        _validate_evidence_pack(output)
    elif contract == "diff-pack-v1":
        _validate_diff_pack(output)
    elif contract == "review-report-v1":
        _validate_review_report(output)
    elif contract == "summary-v1":
        _validate_summary(output)
    else:
        raise ValueError(f"unknown output contract: {contract}")


def _require_keys(obj: Any, keys: list[str], name: str) -> None:
    if not isinstance(obj, dict):
        raise TypeError(f"{name} must be dict")
    missing = [k for k in keys if k not in obj]
    if missing:
        raise ValueError(f"{name} missing keys: {', '.join(missing)}")


def _validate_dag_spec(out: Any) -> None:
    _require_keys(out, ["nodes", "edges"], "dag-spec-v1")
    if not isinstance(out["nodes"], list) or not out["nodes"]:
        raise ValueError("dag-spec-v1.nodes must be non-empty list")
    if not isinstance(out["edges"], list):
        raise ValueError("dag-spec-v1.edges must be list")


def _validate_work_pack(out: Any) -> None:
    _require_keys(out, ["tasks"], "work-pack-v1")
    if not isinstance(out["tasks"], list):
        raise ValueError("work-pack-v1.tasks must be list")


def _validate_evidence_pack(out: Any) -> None:
    _require_keys(out, ["sources", "claims"], "evidence-pack-v1")


def _validate_diff_pack(out: Any) -> None:
    _require_keys(out, ["files_changed", "commands_run"], "diff-pack-v1")


def _validate_review_report(out: Any) -> None:
    _require_keys(out, ["verdict", "evidence"], "review-report-v1")
    if out["verdict"] not in {"pass", "fail"}:
        raise ValueError("review-report-v1.verdict must be pass|fail")


def _validate_summary(out: Any) -> None:
    _require_keys(out, ["summary", "next_action"], "summary-v1")
