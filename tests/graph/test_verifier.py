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
            safety_result={"accepted_candidate_ids": []},
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


if __name__ == "__main__":
    unittest.main()
