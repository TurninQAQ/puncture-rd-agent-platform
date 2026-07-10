"""Deterministic result verification independent of the language model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .state import AgentState, TaskType, VerificationStatus


@dataclass(frozen=True)
class VerificationResult:
    status: str
    reasons: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)


def verify_agent_state(state: AgentState) -> VerificationResult:
    """Check cross-node invariants before a final report may be emitted."""

    if not state.case_id:
        return VerificationResult(
            VerificationStatus.MISSING_INPUT,
            ("case_id is required",),
        )

    subgraph_status = state.subgraph_result.get("status")
    if subgraph_status == "MISSING_INPUT":
        return VerificationResult(
            VerificationStatus.MISSING_INPUT,
            tuple(state.subgraph_result.get("reasons", ["required input missing"])),
        )
    if subgraph_status == "ERROR":
        last_tool_error = state.metadata.get("last_tool_error", {})
        if not last_tool_error.get("retryable", False):
            return VerificationResult(
                VerificationStatus.MANUAL_REVIEW,
                tuple(
                    state.subgraph_result.get(
                        "reasons", ["non-retryable tool error"]
                    )
                ),
            )
        if state.retry_count < state.max_retries:
            return VerificationResult(
                VerificationStatus.NEED_RETRY,
                tuple(state.subgraph_result.get("reasons", ["retryable tool error"])),
            )
        return VerificationResult(
            VerificationStatus.MANUAL_REVIEW,
            ("retry budget exhausted",),
        )
    if subgraph_status == "VALIDATION_FAILED":
        return VerificationResult(
            VerificationStatus.MANUAL_REVIEW,
            tuple(state.subgraph_result.get("reasons", ["validation failed"])),
        )
    if subgraph_status == "NO_FEASIBLE_PATH":
        return VerificationResult(
            VerificationStatus.NO_FEASIBLE_PATH,
            ("planner returned no feasible candidate",),
        )

    if state.task_type == TaskType.PLANNING_SAFETY:
        if not state.candidate_paths:
            return VerificationResult(
                VerificationStatus.NO_FEASIBLE_PATH,
                ("candidate path list is empty",),
            )
        accepted = state.safety_result.get("accepted_candidate_ids", [])
        if not accepted:
            return VerificationResult(
                VerificationStatus.NO_FEASIBLE_PATH,
                ("all candidate paths were rejected by safety checks",),
            )
        if state.risk_flags.get("contradictory"):
            return VerificationResult(
                VerificationStatus.MANUAL_REVIEW,
                ("risk flags are contradictory",),
            )
        return VerificationResult(
            VerificationStatus.PASS,
            evidence={
                "candidate_count": len(state.candidate_paths),
                "accepted_candidate_count": len(accepted),
            },
        )

    if state.task_type == TaskType.DATA_MODEL_VALIDATION:
        required_flags = (
            state.metadata.get("geometry_valid"),
            state.metadata.get("label_schema_valid"),
        )
        if not all(required_flags):
            return VerificationResult(
                VerificationStatus.MANUAL_REVIEW,
                ("geometry or label schema validation did not pass",),
            )
        if state.metadata.get("run_segmentation") and not state.metadata.get(
            "segmentation_valid"
        ):
            return VerificationResult(
                VerificationStatus.MANUAL_REVIEW,
                ("segmentation output validation did not pass",),
            )
        return VerificationResult(
            VerificationStatus.PASS,
            evidence={"validated_artifact_count": len(state.artifacts)},
        )

    return VerificationResult(
        VerificationStatus.MISSING_INPUT,
        ("unsupported or ambiguous task type",),
    )
