from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from puncture_agent.agent.state import AgentState, TaskType, VerificationStatus  # noqa: E402
from puncture_agent.agent.verifier import verify_agent_state  # noqa: E402


class VerifierTests(unittest.TestCase):
    def test_planning_requires_accepted_safe_candidate(self) -> None:
        state = AgentState(
            user_query="planning",
            case_id="Case-1",
            task_type=TaskType.PLANNING_SAFETY,
            candidate_paths=[{"candidate_id": "path-1"}],
            safety_result={
                "accepted_candidate_ids": [],
                "rejected_candidate_ids": ["path-1"],
                "assessments": [
                    {"candidate_id": "path-1", "disposition": "REJECTED"}
                ],
            },
            subgraph_result={"status": "SUCCESS"},
        )
        result = verify_agent_state(state)
        self.assertEqual(VerificationStatus.NO_FEASIBLE_PATH, result.status)

    def test_retryable_error_respects_retry_budget(self) -> None:
        state = AgentState(
            user_query="planning",
            case_id="Case-1",
            task_type=TaskType.PLANNING_SAFETY,
            subgraph_result={"status": "ERROR", "reasons": ["timeout"]},
            metadata={"last_tool_error": {"retryable": True}},
            retry_count=0,
            max_retries=1,
        )
        self.assertEqual(
            VerificationStatus.NEED_RETRY, verify_agent_state(state).status
        )
        state.retry_count = 1
        self.assertEqual(
            VerificationStatus.MANUAL_REVIEW, verify_agent_state(state).status
        )

    def test_contradictory_risk_flags_require_manual_review(self) -> None:
        state = AgentState(
            user_query="planning",
            case_id="Case-1",
            task_type=TaskType.PLANNING_SAFETY,
            candidate_paths=[{"candidate_id": "path-1"}],
            safety_result={"accepted_candidate_ids": ["path-1"]},
            risk_flags={"contradictory": True},
            subgraph_result={"status": "SUCCESS"},
        )
        self.assertEqual(
            VerificationStatus.MANUAL_REVIEW, verify_agent_state(state).status
        )

    def test_unknown_accepted_candidate_and_slip_fail_closed(self) -> None:
        state = AgentState(
            user_query="planning",
            case_id="Case-1",
            task_type=TaskType.PLANNING_SAFETY,
            candidate_paths=[{"candidate_id": "path-1"}],
            safety_result={"accepted_candidate_ids": ["ghost"]},
            risk_flags={"overall_level": "SAFE"},
            skin_penetration_result={"status": "PENETRATED", "crossed_skin": True},
            subgraph_result={"status": "SUCCESS"},
            metadata={"selected_candidate_id": "ghost"},
        )
        self.assertEqual(
            VerificationStatus.MANUAL_REVIEW, verify_agent_state(state).status
        )

        state.safety_result = {"accepted_candidate_ids": ["path-1"]}
        state.metadata["selected_candidate_id"] = "path-1"
        state.skin_penetration_result = {
            "status": "SUSPECTED_SLIP",
            "crossed_skin": True,
        }
        self.assertEqual(
            VerificationStatus.MANUAL_REVIEW, verify_agent_state(state).status
        )

    def test_complete_safe_evidence_passes(self) -> None:
        state = AgentState(
            user_query="planning",
            case_id="Case-1",
            task_type=TaskType.PLANNING_SAFETY,
            candidate_paths=[{"candidate_id": "path-1"}],
            safety_result={"accepted_candidate_ids": ["path-1"]},
            risk_flags={"overall_level": "SAFE"},
            skin_penetration_result={"status": "PENETRATED", "crossed_skin": True},
            subgraph_result={"status": "SUCCESS"},
            metadata={"selected_candidate_id": "path-1"},
        )
        self.assertEqual(VerificationStatus.PASS, verify_agent_state(state).status)

    def test_accepted_and_rejected_candidate_sets_must_be_disjoint(self) -> None:
        state = AgentState(
            user_query="planning",
            case_id="Case-1",
            task_type=TaskType.PLANNING_SAFETY,
            candidate_paths=[{"candidate_id": "path-1"}],
            safety_result={
                "accepted_candidate_ids": ["path-1"],
                "rejected_candidate_ids": ["path-1"],
            },
            subgraph_result={"status": "SUCCESS"},
        )

        result = verify_agent_state(state)

        self.assertEqual(VerificationStatus.MANUAL_REVIEW, result.status)
        self.assertEqual(["path-1"], result.evidence["overlap_candidate_ids"])

    def test_safety_summary_must_be_a_complete_candidate_partition(self) -> None:
        base = AgentState(
            user_query="planning",
            case_id="Case-1",
            task_type=TaskType.PLANNING_SAFETY,
            candidate_paths=[
                {"candidate_id": "path-1"},
                {"candidate_id": "path-2"},
            ],
            subgraph_result={"status": "SUCCESS"},
        )

        with self.subTest("unknown identity"):
            base.safety_result = {
                "accepted_candidate_ids": ["path-1"],
                "rejected_candidate_ids": ["ghost"],
            }
            result = verify_agent_state(base)
            self.assertEqual(VerificationStatus.MANUAL_REVIEW, result.status)
            self.assertEqual(["ghost"], result.evidence["unknown_candidate_ids"])

        with self.subTest("incomplete coverage"):
            base.safety_result = {
                "accepted_candidate_ids": ["path-1"],
                "rejected_candidate_ids": [],
            }
            result = verify_agent_state(base)
            self.assertEqual(VerificationStatus.MANUAL_REVIEW, result.status)
            self.assertEqual(
                ["path-2"], result.evidence["unclassified_candidate_ids"]
            )

    def test_assessment_disposition_must_match_summary_partition(self) -> None:
        state = AgentState(
            user_query="planning",
            case_id="Case-1",
            task_type=TaskType.PLANNING_SAFETY,
            candidate_paths=[
                {"candidate_id": "path-1"},
                {"candidate_id": "path-2"},
            ],
            safety_result={
                "accepted_candidate_ids": ["path-1"],
                "rejected_candidate_ids": ["path-2"],
                "assessments": [
                    {"candidate_id": "path-1", "disposition": "REJECTED"},
                    {"candidate_id": "path-2", "disposition": "ACCEPTED"},
                ],
            },
            subgraph_result={"status": "SUCCESS"},
        )

        result = verify_agent_state(state)

        self.assertEqual(VerificationStatus.MANUAL_REVIEW, result.status)
        self.assertEqual(
            ["path-1", "path-2"], result.evidence["mismatched_candidate_ids"]
        )

    def test_assessment_disposition_partition_can_pass(self) -> None:
        state = AgentState(
            user_query="planning",
            case_id="Case-1",
            task_type=TaskType.PLANNING_SAFETY,
            candidate_paths=[
                {"candidate_id": "path-1"},
                {"candidate_id": "path-2"},
            ],
            safety_result={
                "accepted_candidate_ids": ["path-1"],
                "rejected_candidate_ids": ["path-2"],
                "assessments": [
                    {
                        "candidate_id": "path-1",
                        "disposition": "ACCEPTED_WITH_WARNING",
                    },
                    {"candidate_id": "path-2", "disposition": "REJECTED"},
                ],
                "safest_candidate_id": "path-1",
            },
            risk_flags={"overall_level": "SAFE"},
            skin_penetration_result={"status": "PENETRATED", "crossed_skin": True},
            subgraph_result={"status": "SUCCESS"},
            metadata={"selected_candidate_id": "path-1"},
        )

        self.assertEqual(VerificationStatus.PASS, verify_agent_state(state).status)


if __name__ == "__main__":
    unittest.main()
