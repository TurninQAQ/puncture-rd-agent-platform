"""Deterministic result verification independent of the language model."""

from __future__ import annotations

from collections.abc import Mapping
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
    if state.metadata.get("partial_tool_results"):
        return VerificationResult(
            VerificationStatus.MANUAL_REVIEW,
            ("one or more tool results were partial",),
            evidence={
                "partial_tools": list(state.metadata.get("partial_tool_results", []))
            },
        )

    if state.task_type == TaskType.PLANNING_SAFETY:
        if not state.candidate_paths:
            return VerificationResult(
                VerificationStatus.NO_FEASIBLE_PATH,
                ("candidate path list is empty",),
            )
        candidate_ids = [candidate.get("candidate_id") for candidate in state.candidate_paths]
        if any(not isinstance(item, str) or not item for item in candidate_ids) or len(
            candidate_ids
        ) != len(set(candidate_ids)):
            return VerificationResult(
                VerificationStatus.MANUAL_REVIEW,
                ("candidate path identities are missing or duplicated",),
            )
        candidate_id_set = set(candidate_ids)
        accepted = state.safety_result.get("accepted_candidate_ids", [])
        if (
            not isinstance(accepted, (list, tuple))
            or any(not isinstance(item, str) or not item for item in accepted)
            or len(accepted) != len(set(accepted))
        ):
            return VerificationResult(
                VerificationStatus.MANUAL_REVIEW,
                ("accepted candidate identities are malformed",),
            )
        accepted_ids = set(accepted)

        if "rejected_candidate_ids" in state.safety_result:
            rejected = state.safety_result.get("rejected_candidate_ids")
        else:
            legacy_rejected = state.safety_result.get("rejected_candidates", [])
            if not isinstance(legacy_rejected, (list, tuple)):
                rejected = None
            else:
                rejected = [
                    item.get("candidate_id") if isinstance(item, Mapping) else None
                    for item in legacy_rejected
                ]
        if (
            not isinstance(rejected, (list, tuple))
            or any(not isinstance(item, str) or not item for item in rejected)
            or len(rejected) != len(set(rejected))
        ):
            return VerificationResult(
                VerificationStatus.MANUAL_REVIEW,
                ("rejected candidate identities are malformed",),
            )
        rejected_ids = set(rejected)

        overlap = sorted(accepted_ids & rejected_ids)
        if overlap:
            return VerificationResult(
                VerificationStatus.MANUAL_REVIEW,
                ("accepted and rejected candidate identities overlap",),
                evidence={"overlap_candidate_ids": overlap},
            )
        unknown_ids = sorted((accepted_ids | rejected_ids) - candidate_id_set)
        if unknown_ids:
            return VerificationResult(
                VerificationStatus.MANUAL_REVIEW,
                ("safety result references unknown candidate identities",),
                evidence={"unknown_candidate_ids": unknown_ids},
            )
        uncovered_ids = sorted(candidate_id_set - (accepted_ids | rejected_ids))
        if uncovered_ids:
            return VerificationResult(
                VerificationStatus.MANUAL_REVIEW,
                ("safety disposition does not cover every candidate",),
                evidence={"unclassified_candidate_ids": uncovered_ids},
            )

        safest = state.safety_result.get("safest_candidate_id")
        if safest is not None and safest not in accepted_ids:
            return VerificationResult(
                VerificationStatus.MANUAL_REVIEW,
                ("safest candidate is not present in the accepted set",),
            )
        assessments = state.safety_result.get("assessments")
        if assessments is not None:
            if not isinstance(assessments, (list, tuple)):
                return VerificationResult(
                    VerificationStatus.MANUAL_REVIEW,
                    ("path safety assessments are malformed",),
                )
            if any(not isinstance(item, Mapping) for item in assessments):
                return VerificationResult(
                    VerificationStatus.MANUAL_REVIEW,
                    ("path safety assessments are malformed",),
                )
            assessed_ids = [item.get("candidate_id") for item in assessments]
            if (
                any(not isinstance(item, str) or not item for item in assessed_ids)
                or len(assessed_ids) != len(set(assessed_ids))
                or set(assessed_ids) != candidate_id_set
            ):
                return VerificationResult(
                    VerificationStatus.MANUAL_REVIEW,
                    ("path safety assessments do not cover every candidate exactly once",),
                )
            disposition_by_id: dict[str, str] = {}
            for assessment in assessments:
                disposition = assessment.get("disposition")
                if hasattr(disposition, "value"):
                    disposition = disposition.value
                if disposition not in {
                    "ACCEPTED",
                    "ACCEPTED_WITH_WARNING",
                    "REJECTED",
                    "UNKNOWN",
                }:
                    return VerificationResult(
                        VerificationStatus.MANUAL_REVIEW,
                        ("path safety assessment disposition is not canonical",),
                    )
                disposition_by_id[assessment["candidate_id"]] = disposition

            unknown_dispositions = sorted(
                candidate_id
                for candidate_id, disposition in disposition_by_id.items()
                if disposition == "UNKNOWN"
            )
            if unknown_dispositions:
                return VerificationResult(
                    VerificationStatus.MANUAL_REVIEW,
                    ("path safety assessment disposition is unknown",),
                    evidence={"unknown_disposition_candidate_ids": unknown_dispositions},
                )

            mismatched_ids = sorted(
                candidate_id
                for candidate_id, disposition in disposition_by_id.items()
                if (
                    disposition in {"ACCEPTED", "ACCEPTED_WITH_WARNING"}
                    and candidate_id not in accepted_ids
                )
                or (disposition == "REJECTED" and candidate_id not in rejected_ids)
            )
            if mismatched_ids:
                return VerificationResult(
                    VerificationStatus.MANUAL_REVIEW,
                    ("path safety assessment disposition disagrees with summary sets",),
                    evidence={"mismatched_candidate_ids": mismatched_ids},
                )

        if not accepted:
            return VerificationResult(
                VerificationStatus.NO_FEASIBLE_PATH,
                ("all candidate paths were rejected by safety checks",),
            )
        if state.risk_flags.get("contradictory") or state.risk_flags.get(
            "requires_manual_review"
        ):
            return VerificationResult(
                VerificationStatus.MANUAL_REVIEW,
                ("risk result requires deterministic manual review",),
            )
        if not state.risk_flags:
            return VerificationResult(
                VerificationStatus.MANUAL_REVIEW,
                ("intraoperative risk evidence is missing",),
            )
        overall_level = state.risk_flags.get("overall_level")
        if overall_level is not None and overall_level not in {
            "SAFE",
            "WARNING",
            "STOP",
            "UNKNOWN",
        }:
            return VerificationResult(
                VerificationStatus.MANUAL_REVIEW,
                ("intraoperative risk level is not canonical",),
            )
        if overall_level in {"WARNING", "STOP", "UNKNOWN"} or any(
            value is True and (key.endswith("_warning") or key.endswith("_stop"))
            for key, value in state.risk_flags.items()
        ):
            return VerificationResult(
                VerificationStatus.MANUAL_REVIEW,
                ("intraoperative risk requires review or stop handling",),
            )
        selected_candidate_id = state.metadata.get("selected_candidate_id")
        if not isinstance(selected_candidate_id, str) or selected_candidate_id not in accepted:
            return VerificationResult(
                VerificationStatus.MANUAL_REVIEW,
                ("risk evidence is not bound to an accepted candidate",),
            )
        penetration = state.skin_penetration_result
        if not penetration:
            return VerificationResult(
                VerificationStatus.MANUAL_REVIEW,
                ("skin penetration evidence is missing",),
            )
        if "status" in penetration:
            if penetration.get("status") != "PENETRATED" or penetration.get(
                "crossed_skin"
            ) is not True:
                return VerificationResult(
                    VerificationStatus.MANUAL_REVIEW,
                    ("skin penetration is unknown, unsafe, or incomplete",),
                )
        elif (
            penetration.get("penetrated") is not True
            or penetration.get("slippage_suspected") is True
            or penetration.get("skin_not_punctured_suspected") is True
        ):
            return VerificationResult(
                VerificationStatus.MANUAL_REVIEW,
                ("skin penetration is unknown, unsafe, or incomplete",),
            )
        return VerificationResult(
            VerificationStatus.PASS,
            evidence={
                "candidate_count": len(state.candidate_paths),
                "accepted_candidate_count": len(accepted),
                "selected_candidate_id": selected_candidate_id,
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
