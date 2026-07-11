from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
import sys
import unittest
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from puncture_agent.agent import (  # noqa: E402
    AgentState,
    AgentStatus,
    LangGraphRuntime,
    PostgresAdvisoryThreadExecutionLeaseManager,
    ThreadLeaseBusy,
    build_mock_handlers,
    langgraph_available,
    open_postgres_checkpointer,
)
from puncture_agent.agent.nodes import DeterministicMockToolExecutor  # noqa: E402


POSTGRES_DSN = os.environ.get("PUNCTURE_TEST_POSTGRES_DSN", "")


class CountingExecutor:
    def __init__(self) -> None:
        self.delegate = DeterministicMockToolExecutor()
        self._lock = Lock()
        self.counts: dict[str, int] = {}

    def execute(self, tool_name, request):
        with self._lock:
            self.counts[tool_name] = self.counts.get(tool_name, 0) + 1
        return self.delegate.execute(tool_name, request)


@unittest.skipUnless(
    POSTGRES_DSN and langgraph_available(),
    "PostgreSQL/LangGraph integration environment is not configured",
)
class LangGraphPostgresRestartTests(unittest.TestCase):
    def test_new_runtime_instance_resumes_without_duplicate_tool_execution(self) -> None:
        executor = CountingExecutor()
        handlers = build_mock_handlers(executor)
        session_id = f"postgres-restart-{uuid4().hex}"
        initial = AgentState(
            user_query="对 Case-990 做路径规划和安全评估",
            session_id=session_id,
        )

        with open_postgres_checkpointer(POSTGRES_DSN, setup=True) as first_saver:
            first_runtime = LangGraphRuntime(
                PROJECT_ROOT / "graph" / "main_graph.json",
                handlers,
                checkpointer=first_saver,
                execution_lease_manager=(
                    PostgresAdvisoryThreadExecutionLeaseManager(POSTGRES_DSN)
                ),
            )
            completed = first_runtime.run(initial)
            self.assertEqual(AgentStatus.SUCCEEDED, completed.status)
            counts_after_run = dict(executor.counts)

        with open_postgres_checkpointer(POSTGRES_DSN, setup=False) as second_saver:
            second_runtime = LangGraphRuntime(
                PROJECT_ROOT / "graph" / "main_graph.json",
                handlers,
                checkpointer=second_saver,
                execution_lease_manager=(
                    PostgresAdvisoryThreadExecutionLeaseManager(POSTGRES_DSN)
                ),
            )
            restored = second_runtime.checkpoint_state(thread_id=session_id)
            resumed = second_runtime.resume(thread_id=session_id)

        self.assertEqual(completed.to_dict(), restored.to_dict())
        self.assertEqual(completed.to_dict(), resumed.to_dict())
        self.assertEqual(counts_after_run, executor.counts)

    def test_dynamic_interrupt_resumes_child_without_replaying_completed_tools(
        self,
    ) -> None:
        from langgraph.types import interrupt

        executor = CountingExecutor()
        session_id = f"postgres-interrupt-{uuid4().hex}"

        def handlers_with_approval_gate():
            handlers = dict(build_mock_handlers(executor))
            original_router = handlers["candidate_router"]

            def approval_gate(state, context):
                decision = interrupt(
                    {
                        "kind": "candidate_review",
                        "prompt": "approve generated candidates",
                    }
                )
                state.metadata["candidate_review"] = decision
                return original_router(state, context)

            handlers["candidate_router"] = approval_gate
            return handlers

        initial = AgentState(
            user_query="对 Case-991 做路径规划和安全评估",
            session_id=session_id,
        )

        with open_postgres_checkpointer(POSTGRES_DSN, setup=True) as first_saver:
            first_runtime = LangGraphRuntime(
                PROJECT_ROOT / "graph" / "main_graph.json",
                handlers_with_approval_gate(),
                checkpointer=first_saver,
                execution_lease_manager=(
                    PostgresAdvisoryThreadExecutionLeaseManager(POSTGRES_DSN)
                ),
            )
            interrupted = first_runtime.run(initial)

        self.assertEqual(AgentStatus.AWAITING_INPUT, interrupted.status)
        self.assertEqual(
            "candidate_review",
            interrupted.metadata["pending_interrupts"][0]["value"]["kind"],
        )
        self.assertEqual({"generate_candidate_paths": 1}, executor.counts)

        with open_postgres_checkpointer(POSTGRES_DSN, setup=False) as second_saver:
            second_runtime = LangGraphRuntime(
                PROJECT_ROOT / "graph" / "main_graph.json",
                handlers_with_approval_gate(),
                checkpointer=second_saver,
                execution_lease_manager=(
                    PostgresAdvisoryThreadExecutionLeaseManager(POSTGRES_DSN)
                ),
            )
            restored = second_runtime.checkpoint_state(thread_id=session_id)
            self.assertEqual(AgentStatus.AWAITING_INPUT, restored.status)
            self.assertEqual(
                interrupted.metadata["pending_interrupts"],
                restored.metadata["pending_interrupts"],
            )
            self.assertEqual(
                ["generate_candidate_paths"],
                [call["tool_name"] for call in restored.tool_calls],
            )
            resumed = second_runtime.resume(
                thread_id=session_id,
                resume_value={"approved": True},
            )

        self.assertEqual(AgentStatus.SUCCEEDED, resumed.status)
        self.assertEqual({"approved": True}, resumed.metadata["candidate_review"])
        self.assertEqual(
            {
                "generate_candidate_paths": 1,
                "evaluate_path_safety": 1,
                "evaluate_intraoperative_risk": 1,
                "verify_skin_penetration": 1,
            },
            executor.counts,
        )
        self.assertEqual(
            1,
            resumed.visited_nodes.count(
                "planning_safety_subgraph.generate_candidate_paths"
            ),
        )

    def test_dedicated_advisory_connections_reject_same_thread(self) -> None:
        thread_id = f"postgres-lease-{uuid4().hex}"
        first_manager = PostgresAdvisoryThreadExecutionLeaseManager(POSTGRES_DSN)
        second_manager = PostgresAdvisoryThreadExecutionLeaseManager(POSTGRES_DSN)

        first = first_manager.acquire(thread_id, operation="run")
        try:
            with self.assertRaises(ThreadLeaseBusy):
                second_manager.acquire(thread_id, operation="resume")
        finally:
            first.release()

        replacement = second_manager.acquire(thread_id, operation="stream")
        replacement.assert_valid()
        replacement.release()


if __name__ == "__main__":
    unittest.main()
